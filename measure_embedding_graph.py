"""Falsify a bandwidth diagnosis by removing host launch gaps on the same GPU."""
import json
import os
from pathlib import Path
import statistics
import time

import pynvml
import torch
from transformers import AutoModel

root = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.manual_seed(42)
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get('CALIB_GPU', '1')))
info = json.loads((root/'models.json').read_text())['Qwen/Qwen3-Embedding-0.6B']
model = AutoModel.from_pretrained(info['path'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
report = dict(model=info, torch=torch.__version__, input_kind='synthetic calibration',
              control='same weights, inputs, mask, GPU; eager versus CUDA Graph replay', cases=[])
output = root/'outputs/embedding_graph_calibration.json'

def measure(fn, repeats=20):
    times = []
    energy0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    wall0 = time.perf_counter()
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b)/1000)
    return dict(seconds=times, median_seconds=statistics.median(times),
                wall_seconds=time.perf_counter()-wall0,
                gross_joules_per_forward=(pynvml.nvmlDeviceGetTotalEnergyConsumption(h)-energy0)/1000/repeats)

with torch.inference_mode():
    for batch,length in [(1,32),(1,128),(1,512),(1,2048),(8,128),(32,128),(8,512),(32,512)]:
        ids = torch.randint(100,10000,(batch,length),device='cuda')
        mask = torch.ones_like(ids)
        captured = {}
        def remember_mask(module, positional, keyword):
            captured['mask'] = keyword.get('attention_mask')
        hook = model.layers[0].register_forward_pre_hook(remember_mask, with_kwargs=True)
        normal = model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state[:,-1].clone()
        hook.remove()
        # Reuse the exact mask created by native HF, without rebuilding it
        # inside graph capture or changing SDPA's dispatch.
        def forward():
            return model(input_ids=ids,attention_mask={'full_attention':captured['mask']},use_cache=False).last_hidden_state[:,-1]
        explicit = forward().clone()
        torch.testing.assert_close(explicit,normal,rtol=0.02,atol=0.02)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4): forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = forward()
        graph.replay(); torch.cuda.synchronize()
        torch.testing.assert_close(result,explicit,rtol=0.001,atol=0.001)
        eager = measure(forward)
        replay = measure(graph.replay)
        row = dict(batch=batch,length=length,eager=eager,graph=replay,
                   speedup=eager['median_seconds']/replay['median_seconds'],
                   max_abs_mask_difference=(normal-explicit).abs().max().item())
        if batch == 1 and length == 128:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                forward(); torch.cuda.synchronize()
            trace = root/'outputs/embedding_eager_profiler.json'
            prof.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            intervals=sorted((e['ts'],e['ts']+e['dur']) for e in events if e.get('cat')=='kernel' and e.get('ph')=='X')
            merged=[]
            for start,end in intervals:
                if merged and start<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],end)
                else: merged.append([start,end])
            row['profile'] = dict(kernel_count=len(intervals),
                kernel_union_seconds=sum(b-a for a,b in merged)/1e6,
                first_to_last_kernel_seconds=(intervals[-1][1]-intervals[0][0])/1e6)
        report['cases'].append(row)
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ['eager','graph']}),flush=True)
        del graph,result
