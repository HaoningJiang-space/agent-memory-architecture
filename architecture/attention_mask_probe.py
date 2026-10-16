"""Bounded single-layer control for the padding/GQA dispatch found on A800.

Synthetic tensors use Qwen3-32B heads and three actual prompt lengths. This is
not a full-model repair or a proposed accelerator. Run after the model probe.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    p.add_argument('--gpu', type=int, default=1)
    args = p.parse_args()
    out = ROOT / 'outputs' / args.session
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import pynvml
    from transformers.integrations.sdpa_attention import repeat_kv, use_gqa_in_sdpa
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(1)
    torch.manual_seed(20261017)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    assert not [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                if p.pid != os.getpid()], 'GPU must be free of other compute processes'
    device = f'cuda:{args.gpu}'
    cfg = json.loads((ROOT / 'evidence/phase2/workload_accounting.json').read_text())['models']['Qwen/Qwen3-32B']['config']
    heads, kv_heads, dim = cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim']
    assert (heads, kv_heads, dim) == (64, 8, 128)
    all_rows = []
    for name in ('finite_prefix', 'representation_controls'):
        all_rows += json.loads((ROOT / f'evidence/architecture/{name}.json').read_text())['summary']['results']
    result = dict(status='running', pid=os.getpid(), start=time.time(),
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(args.gpu), torch=torch.__version__,
        transformers=__import__('transformers').__version__,
        helper_sources={fn.__name__: inspect.getsource(fn) for fn in (repeat_kv, use_gqa_in_sdpa)},
        controls=dict(seed=20261017, batch=1, query_tokens=1, heads=heads, kv_heads=kv_heads,
                      head_dim=dim, dtype='BF16', warmup=2, timed_steps=16,
                      reference='FP32 grouped einsum, no repeated KV heads'), cases=[])
    path = out / 'public_summary.json'
    with torch.inference_mode():
        for strategy in ('raw', 'raw_512', 'mem0'):
            source = [r for r in all_rows if r['strategy'] == strategy]
            length = next(r['input_tokens'] for r in source if r['index'] == 3)
            padded = max(r['input_tokens'] for r in source)
            q = torch.randn(1, heads, 1, dim, device=device, dtype=torch.bfloat16)
            k = torch.randn(1, kv_heads, length, dim, device=device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            kp = torch.nn.functional.pad(k, (0, 0, padded-length, 0))
            vp = torch.nn.functional.pad(v, (0, 0, padded-length, 0))
            mask = torch.zeros(1, 1, 1, padded, device=device, dtype=torch.bfloat16)
            mask[..., :padded-length] = torch.finfo(torch.bfloat16).min
            q_grouped = q.float().reshape(1, kv_heads, heads // kv_heads, dim)
            scores = torch.einsum('bhgd,bhld->bhgl', q_grouped, k.float()) / dim**0.5
            reference = torch.einsum('bhgl,bhld->bhgd', scores.softmax(-1), v.float()).reshape(1, heads, 1, dim)
            del q_grouped, scores
            for route in ('native_no_padding', 'expanded_no_padding', 'hf_padded'):
                keys, values, attn_mask = (kp, vp, mask) if route == 'hf_padded' else (k, v, None)
                native = route != 'expanded_no_padding' and use_gqa_in_sdpa(attn_mask, keys)
                def forward():
                    kk, vv = (keys, values) if native else (
                        repeat_kv(keys, heads // kv_heads), repeat_kv(values, heads // kv_heads))
                    return torch.nn.functional.scaled_dot_product_attention(
                        q, kk, vv, attn_mask=attn_mask, dropout_p=0.0,
                        is_causal=False, enable_gqa=native)
                for _ in range(2):
                    y = forward()
                    del y
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                elapsed = []
                for _ in range(16):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    y = forward()
                    end.record()
                    end.synchronize()
                    elapsed.append(start.elapsed_time(end))
                    del y
                peak_increment = torch.cuda.max_memory_allocated() - baseline
                y = forward().float()
                error = y-reference
                relative = error.norm().item() / reference.norm().item()
                maximum = error.abs().max().item()
                del y, error
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    y = forward()
                    torch.cuda.synchronize()
                events = [dict(name=e.key, count=e.count, cuda_time_us=e.device_time_total)
                          for e in prof.key_averages()
                          if any(s in e.key for s in ('scaled_dot_product', 'clone', 'copy_', 'reshape'))]
                del y, prof
                result['cases'].append(dict(strategy=strategy, route=route, length=length,
                    padded_length=padded, native_gqa=native, forward_ms=elapsed,
                    peak_extra_bytes=peak_increment, relative_l2=relative, max_abs=maximum,
                    numerical_pass=relative <= 0.01 and maximum <= 0.005,
                    selected_profile_events=events))
                path.write_text(json.dumps(result, indent=2) + '\n')
                print(strategy, route, sum(elapsed)/len(elapsed), relative, maximum, flush=True)
            del q, k, v, kp, vp, mask, reference, keys, values, attn_mask
    result.update(status='complete', end=time.time())
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
