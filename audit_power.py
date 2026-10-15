"""Audit gross NVML readings; do not convert device counters to physical models."""
import json
from pathlib import Path
import statistics
import pynvml

root = Path(__file__).resolve().parent
pynvml.nvmlInit()
report = dict(scope='Gross device telemetry sanity checks; excludes CPU/DRAM, idle subtraction, spatial power and thermal calibration',runs=[])
for name in ['embed_full_01','bm25_full_02']:
    samples = [json.loads(s) for s in (root/'outputs'/name/'both_gpu.jsonl').read_text().splitlines()]
    for gpu in sorted({s['gpu'] for s in samples}):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)/1000
        rows = sorted([s for s in samples if s['gpu']==gpu],key=lambda s:s['time'])
        seconds = rows[-1]['time']-rows[0]['time']
        integral = sum((b['time']-a['time'])*(a['power_w']+b['power_w'])/2 for a,b in zip(rows,rows[1:]))
        counter = rows[-1]['energy_j']-rows[0]['energy_j']
        report['runs'].append(dict(run=name,gpu=gpu,configured_limit_w=limit,
            samples=len(rows),sample_interval_median_s=statistics.median(b['time']-a['time'] for a,b in zip(rows,rows[1:])),
            sample_interval_max_s=max(b['time']-a['time'] for a,b in zip(rows,rows[1:])),
            seconds=seconds,counter_gross_j=counter,sampled_integral_gross_j=integral,
            integral_vs_counter_relative_difference=abs(integral-counter)/counter,
            counter_average_w=counter/seconds,sampled_maximum_w=max(s['power_w'] for s in rows),
            samples_above_configured_limit=sum(s['power_w']>limit for s in rows)))
report['interpretation']='Above-limit instantaneous samples are retained and flagged; no assumption about their cause. These readings are not a validated per-operation energy model or a 3D power map.'
(root/'outputs/power_measurement_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
