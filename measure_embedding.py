"""A800 shape calibration. Synthetic tokens here are calibration, not workload traces."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoModel
import pynvml

root = Path(__file__).resolve().parent
out = root / 'outputs' / 'embedding_calibration.json'
info = json.loads((root / 'models.json').read_text())['Qwen/Qwen3-Embedding-0.6B']
torch.manual_seed(42)
torch.set_num_threads(4)
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get('CALIB_GPU', '0')))
model = AutoModel.from_pretrained(info['path'], local_files_only=True,
                                torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
parameters = sum(p.numel() for p in model.parameters())
report = {'model': info, 'torch': torch.__version__, 'dtype': 'bfloat16',
          'parameters': parameters, 'weight_bytes': parameters * 2,
          'gpu': torch.cuda.get_device_name(), 'config': model.config.to_dict(),
          'input_kind': 'synthetic calibration tokens; not batched workload evidence', 'cases': []}

def energy():
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000
    except pynvml.NVMLError:
        return None

with torch.inference_mode():
    for length in [32, 128, 512, 2048]:
        for batch in [1, 8, 32]:
            ids = torch.randint(100, 10000, (batch, length), device='cuda')
            mask = torch.ones_like(ids)
            def forward():
                result = model(input_ids=ids, attention_mask=mask, use_cache=False)
                return torch.nn.functional.normalize(result.last_hidden_state[:, -1], dim=-1)
            for _ in range(3):
                forward()
            torch.cuda.synchronize()
            shapes = []
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    def capture(mod, inp, result, name=name):
                        shapes.append({'name': name, 'input': list(inp[0].shape),
                                       'weight': list(mod.weight.shape), 'output': list(result.shape)})
                    hooks.append(module.register_forward_hook(capture))
            forward()
            for hook in hooks:
                hook.remove()
            torch.cuda.synchronize()
            times = []
            e0 = energy()
            t0 = time.perf_counter()
            for _ in range(10):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                forward()
                b.record()
                b.synchronize()
                times.append(a.elapsed_time(b) / 1000)
            wall = time.perf_counter() - t0
            e1 = energy()
            row = {'batch': batch, 'length': length, 'seconds': times,
                   'median_seconds': statistics.median(times), 'wall_seconds': wall,
                   'gross_gpu_joules_per_forward': None if e0 is None else (e1-e0)/10,
                   'linear_shapes': shapes}
            report['cases'].append(row)
            out.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ['linear_shapes','seconds']}), flush=True)
