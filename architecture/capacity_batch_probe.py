"""Remote A800 calibration: real finite-history prompts, fixed decode work.

No quality evaluation, new memory construction, native serving or prefix sharing.
Raw prompts, token IDs and logits stay on the remote host/in process.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def run(args, out, manifest):
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import torch
    import pynvml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from architecture.finite_prefix import cache_bytes, tensor_bytes
    from workload_accounting import decoder_shape
    torch.set_num_threads(1)
    torch.cuda.set_device(args.gpu)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    foreign = [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
               if p.pid != os.getpid()]
    if foreign:
        raise RuntimeError(f'GPU {args.gpu} has other compute processes: {foreign}')
    source_root = Path(args.source_root)
    info = json.loads((source_root / 'models.json').read_text())['Qwen/Qwen3-32B']
    tokenizer = AutoTokenizer.from_pretrained(info['path'], local_files_only=True)
    configs = json.loads((ROOT / 'evidence/phase2/workload_accounting.json').read_text())['models']
    assert configs['Qwen/Qwen3-32B']['revision'] == info['revision']
    kv_per_token = decoder_shape(configs['Qwen/Qwen3-32B']['config'])['kv_bytes_per_token']
    sources = [
        ('finite_prefix', source_root / 'outputs/finite_prefix_01'),
        ('representation_controls', source_root / 'outputs/representation_01')]
    groups = {s: [] for s in ('raw', 'raw_512', 'mem0')}
    receipts = []
    for name, directory in sources:
        frozen = json.loads((ROOT / f'evidence/architecture/{name}.json').read_text())
        for row in frozen['summary']['results']:
            i, strategy = row['index'], row['strategy']
            filename = f'{i:02d}_{strategy}_private.json' if name == 'finite_prefix' else f'{i:02d}_private.json'
            path = directory / filename
            digest = row.get('private_receipt_sha256') or row.get('private_sha256')
            assert digest and sha(path) == digest
            private = json.loads(path.read_text())
            ids = tokenizer.apply_chat_template(private['messages'], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            assert len(ids) == row['input_tokens']
            assert len(private['output_ids']) > args.steps
            groups[strategy].append(dict(index=i, ids=ids, outputs=private['output_ids']))
            receipts.append(dict(index=i, strategy=strategy, private_sha256=digest,
                                 input_tokens=len(ids)))
    for calls in groups.values():
        calls.sort(key=lambda c: c['index'])
        assert [c['index'] for c in calls] == [3, 28, 30, 31, 39, 48, 54]
    manifest.update(source_receipts=receipts, model_revision=info['revision'],
                    model_config_sha256=sha(Path(info['path']) / 'config.json'),
                    source_root=str(source_root), loading_started=time.time())
    write(out / 'manifest.json', manifest)
    print('Loading fixed 32B BF16 model on GPU', args.gpu, flush=True)
    device = f'cuda:{args.gpu}'
    model = AutoModelForCausalLM.from_pretrained(info['path'], torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation='sdpa', local_files_only=True).eval()
    weights = tensor_bytes(model.parameters())
    total = torch.cuda.get_device_properties(args.gpu).total_memory
    manifest.update(model_loaded=time.time(), weight_bytes=weights, gpu_memory_bytes=total,
                    torch=torch.__version__, transformers=__import__('transformers').__version__,
                    gpu=torch.cuda.get_device_name(args.gpu), kv_bytes_per_token=kv_per_token,
                    backend='HF eager BF16 SDPA, fixed padded length within strategy, teacher forced',
                    limits=['Distinct recorded requests; no new answer or quality claim.',
                            'No cross-request prefix sharing or optimized vLLM scheduling.',
                            'CUDA event forward times exclude tokenization and validation copies.',
                            'Memory includes padded KV; chunked prefill charged explicitly.',
                            'Single trajectory per case, 16 steps are sequential samples, not independent trials.',
                            'Shape calibration only; no production arrival, SLO or energy result.'])
    write(out / 'manifest.json', manifest)
    results, references = [], {}
    with torch.inference_mode():
        warm = torch.tensor([[tokenizer.eos_token_id]], device=device)
        response = model(input_ids=warm, use_cache=False, logits_to_keep=1)
        del response, warm
        torch.cuda.synchronize()

        def case(strategy, group, chunk, kind):
            batch = len(group)
            length = max(len(c['ids']) for c in groups[strategy])
            estimated_kv = batch * (length + args.steps) * kv_per_token
            row = dict(strategy=strategy, indices=[c['index'] for c in group], batch=batch,
                       padded_input_tokens=length, actual_input_tokens=[len(c['ids']) for c in group],
                       chunk_tokens=chunk, kind=kind, start=time.time(), status='running')
            results.append(row)
            def save():
                write(out / 'public_summary.json', dict(status='running', cases=results))
            if weights + estimated_kv + args.reserve_gib * 2**30 > total:
                row.update(status='capacity_skipped', projected_kv_bytes=estimated_kv,
                           reason='Weight + padded KV + declared workspace reserve exceeds device memory')
                save()
                print('SKIP', strategy, batch, 'capacity', flush=True)
                return
            gc.collect()
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()
            assert before < weights + 2**30, ('GPU tensors leaked between cases', before, weights)
            pad = tokenizer.pad_token_id
            assert pad is not None
            data = torch.tensor([[pad] * (length - len(c['ids'])) + c['ids'] for c in group], device=device)
            mask = torch.tensor([[0] * (length - len(c['ids'])) + [1] * (len(c['ids']) + args.steps)
                                 for c in group], device=device)
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            forced = torch.tensor([c['outputs'][:args.steps] for c in group], device=device)
            cache = None
            prefill, decode, validation = [], [], []
            torch.cuda.reset_peak_memory_stats()
            wall_start = time.perf_counter()

            def forward(ids, upto, pos):
                nonlocal cache
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                response = model(input_ids=ids, attention_mask=mask[:, :upto], position_ids=pos,
                                 past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = response.past_key_values
                end.record()
                end.synchronize()
                return response, start.elapsed_time(end) / 1000

            def check(response, step):
                logits = response.logits[:, -1, :].float().cpu()
                for lane, call in enumerate(group):
                    key = (strategy, call['index'], step)
                    vector = logits[lane]
                    if kind == 'single_reference':
                        references[key] = vector.clone()
                    else:
                        ref = references[key]
                        relative_l2 = (vector - ref).norm().item() / max(ref.norm().item(), 1e-12)
                        max_abs = (vector - ref).abs().max().item()
                        validation.append(dict(index=call['index'], step=step,
                            relative_logit_l2=relative_l2, max_absolute_logit_error=max_abs,
                            argmax_agreement=bool(vector.argmax() == ref.argmax())))
                        # BF16 tolerance is a numerical mapping check, not task quality.
                        if relative_l2 > 0.02 or max_abs > 0.5:
                            raise RuntimeError(f'Batch/chunk numerical check failed: {validation[-1]}')
                del logits

            for begin in range(0, length, chunk):
                end = min(begin + chunk, length)
                response, elapsed = forward(data[:, begin:end], end, positions[:, begin:end])
                prefill.append(elapsed)
                if end == length:
                    check(response, 0)
                del response
            prefill_kv = cache_bytes(cache)
            assert prefill_kv == batch * length * kv_per_token
            for step in range(1, args.steps + 1):
                upto = length + step
                response, elapsed = forward(forced[:, step-1:step], upto, positions[:, upto-1:upto])
                decode.append(elapsed)
                if step == args.steps:
                    check(response, step)
                del response
            final_kv = cache_bytes(cache)
            assert final_kv == estimated_kv
            row.update(status='complete', end=time.time(), prefill_chunks=len(prefill),
                       prefill_cuda_seconds=sum(prefill), decode_cuda_seconds=decode,
                       mean_decode_step_ms=1000 * statistics.mean(decode),
                       median_decode_step_ms=1000 * statistics.median(decode),
                       forward_decode_tokens_per_second=batch * args.steps / sum(decode),
                       instrumented_wall_seconds=time.perf_counter()-wall_start,
                       prefill_kv_bytes=prefill_kv, final_kv_bytes=final_kv,
                       allocated_bytes=torch.cuda.memory_allocated(),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       temperature_c=pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                       sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                       numerical_checks=validation)
            save()
            print('DONE', strategy, group[0]['index'], batch, kind,
                  round(row['mean_decode_step_ms'], 3), round(row['forward_decode_tokens_per_second'], 2), flush=True)
            del data, mask, positions, forced, cache

        for strategy, calls in groups.items():
            for call in calls:
                case(strategy, [call], args.prefill_chunk, 'single_reference')
            for batch in (2, 4, 7):
                case(strategy, calls[:batch], args.prefill_chunk, 'distinct_batch')
            case(strategy, calls[:1], max(len(c['ids']) for c in calls), 'unchunked_control')
    manifest.update(status='complete', end=time.time(), planned_cases=len(results),
                    completed_cases=sum(c['status']=='complete' for c in results),
                    numerical_checks=sum(len(c.get('numerical_checks', [])) for c in results))
    write(out / 'manifest.json', manifest)
    write(out / 'public_summary.json', dict(status='complete', manifest=manifest, cases=results))
    print('COMPLETE', manifest['completed_cases'], 'cases', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--source-root', default='/data/ziheng/agent-memory-architecture')
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--prefill-chunk', type=int, default=512)
    p.add_argument('--reserve-gib', type=float, default=6)
    args = p.parse_args()
    assert args.session.replace('_', '').isalnum() and args.steps > 0 and args.prefill_chunk > 0
    out = ROOT / 'outputs' / args.session
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', start=time.time(), pid=os.getpid(), args=vars(args),
                    runner_sha256=sha(__file__),
                    execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'manifest.json', manifest)
    try:
        run(args, out, manifest)
    except Exception:
        manifest.update(status='failed', end=time.time(), error=traceback.format_exc())
        write(out / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
