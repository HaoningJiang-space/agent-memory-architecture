"""Remote HF calibration on four distinct, independently ready native requests.

Teacher-force eight tokens from each recorded output; this is a kernel probe,
not a new completed answer or a native vLLM performance measurement.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.environ.update(HF_HUB_OFFLINE='1', HF_HOME=str(ROOT/'hf'), TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from architecture.finite_prefix import cache_bytes, tensor_bytes
    torch.set_num_threads(1)
    out = ROOT/'outputs/family_kernel_01'
    out.mkdir(exist_ok=False)
    source = ROOT/'outputs/family_02/simplemem/llm_private.jsonl'
    calls = sorted([json.loads(line) for line in source.read_text().splitlines()
                    if json.loads(line)['phase'] == 'construction'], key=lambda x:x['start'])
    assert len(calls) == 4 and all(c['validated'] for c in calls)
    info = json.loads((ROOT/'models.json').read_text())['Qwen/Qwen3-8B']
    tok = AutoTokenizer.from_pretrained(info['path'], local_files_only=True, padding_side='left')
    torch.cuda.set_device(1)
    model = AutoModelForCausalLM.from_pretrained(info['path'], torch_dtype=torch.bfloat16,
        device_map='cuda:1', attn_implementation='sdpa', local_files_only=True).eval()
    ids = [tok.apply_chat_template(c['request']['messages'], add_generation_prompt=True,
                                  enable_thinking=False, tokenize=True) for c in calls]
    outputs = [tok.encode(c['response']['choices'][0]['message']['content'], add_special_tokens=False) for c in calls]
    assert all(len(o) >= 8 for o in outputs)
    assert [len(x) for x in ids] == [c['response']['usage']['prompt_tokens'] for c in calls]
    probe_shapes = []
    stage = {}
    handles = []
    for name, module in [('layer0_q', model.model.layers[0].self_attn.q_proj),
                         ('layer0_gate', model.model.layers[0].mlp.gate_proj), ('lm_head', model.lm_head)]:
        def hook(module, args, _name=name):
            x = args[0]
            probe_shapes.append(dict(**stage, operation=_name, input_shape=list(x.shape),
                M=x.numel()//x.shape[-1], K=x.shape[-1], N=module.weight.shape[0]))
        handles.append(module.register_forward_pre_hook(hook))
    weights = tensor_bytes(model.parameters())
    results = []
    # Warm up the same primitive separately; no warmup enters reported samples.
    with torch.inference_mode():
        model(input_ids=torch.tensor([[tok.eos_token_id]], device='cuda:1'), use_cache=False)
        probe_shapes.clear()
        for group in [[0], [1], [2], [3], [0, 1, 2, 3]]:
            gc.collect(); torch.cuda.empty_cache()
            length = max(len(ids[i]) for i in group)
            data = torch.tensor([[tok.pad_token_id]*(length-len(ids[i]))+ids[i] for i in group], device='cuda:1')
            mask = torch.tensor([[0]*(length-len(ids[i]))+[1]*len(ids[i]) for i in group], device='cuda:1')
            cache = None
            for step in range(9):
                stage.clear(); stage.update(group=group, step=step, phase='prefill' if step==0 else 'decode')
                position = mask.long().cumsum(-1)-1
                position.masked_fill_(mask==0, 0)
                if step: position = position[:, -1:]
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                response = model(input_ids=data, attention_mask=mask, position_ids=position,
                                 past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = response.past_key_values
                end.record(); end.synchronize()
                results.append(dict(**stage, cuda_seconds=start.elapsed_time(end)/1000,
                    allocated_bytes=torch.cuda.memory_allocated(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    kv_bytes=cache_bytes(cache), padded_context_length=length+step))
                del response
                if step < 8:
                    data = torch.tensor([[outputs[i][step]] for i in group], device='cuda:1')
                    mask = torch.cat([mask, torch.ones((len(group),1), device='cuda:1', dtype=mask.dtype)], dim=1)
            del data,mask,cache
    for h in handles: h.remove()
    result = dict(status='complete', runner_sha256=sha(Path(__file__)), source_sha256=sha(source),
        source_ids=[c['id'] for c in calls], input_lengths=[len(x) for x in ids], model_revision=info['revision'],
        weight_bytes=weights, backend='A800 GPU1, HF eager BF16 SDPA, first-layer linear hooks, eight teacher-forced decode tokens',
        torch=torch.__version__, transformers=__import__('transformers').__version__,
        shapes=probe_shapes, forwards=results,
        limits=['Short kernel calibration; not complete native answers or vLLM SLA.',
                'Same four distinct native extraction inputs; no repeated/averaged fake batch.',
                'Left padding charges full B*max(L) prefill and allocated cache; packing not assumed.',
                'Linear hooks sample layer0 and lm_head; they are operator shapes, not GPU DRAM transaction counters.'])
    (out/'public_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('complete',result['input_lengths'],len(probe_shapes),flush=True)


if __name__ == '__main__':
    main()
