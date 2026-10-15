"""Export small summaries only; do not export benchmark histories or responses."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

r=Path(__file__).resolve().parent
out=r/'outputs/summaries';out.mkdir(exist_ok=True)
raw=json.loads((r/'outputs/failed_update_replay.json').read_text())
raw['trials']=[{k:v for k,v in x.items() if k!='response'} for x in raw['trials']]
(out/'failed_update_replay_summary.json').write_text(json.dumps(raw,indent=2)+'\n')
batch=json.loads((r/'outputs/fact_batching_control.json').read_text())
g=batch['groups'];n=sum(x['native_fact_calls'] for x in g)
batch['aggregate']=dict(fact_count=n,group_count=len(g),median_group_speedup=statistics.median(x['speedup'] for x in g),
    minimum_cosine=min(x['min_cosine'] for x in g),
    weighted_top5_set_overlap=sum(x['top5_set_overlap']*x['native_fact_calls'] for x in g)/n,
    weighted_exact_top5_order=sum(x['exact_top5_order_fraction']*x['native_fact_calls'] for x in g)/n,
    total_serial_median_seconds=sum(statistics.median(x['serial_seconds']) for x in g),
    total_batch_median_seconds=sum(statistics.median(x['batch_seconds']) for x in g))
(out/'fact_batching_summary.json').write_text(json.dumps(batch,indent=2)+'\n')
api=[json.loads(s) for s in (r/'outputs/api_trace.jsonl').read_text().splitlines()]
rows=[x for x in api if x.get('context',{}).get('run')=='graph_full_03' and x['path']=='chat/completions']
reasons=collections.Counter()
bad=[]
for row in rows:
    reason=row.get('finish_reason')
    if reason is None:reason=row.get('response',{}).get('choices',[{}])[0].get('finish_reason')
    reasons[reason]+=1
    if reason=='length':bad.append(dict(trace_id=row['id'],usage=row.get('usage'),finish_reason=reason))
(out/'graph_failures.json').write_text(json.dumps(dict(calls=len(rows),finish_reasons=dict(reasons),
    failures=bad,scope='output truncation invalidates full construction; no QA run'),indent=2)+'\n')
files=['run_summary.json','calibration_analysis.json','model_capacity.json','environment_conditions.json',
       'spacy_batch_check.json','pipeline.json','requirements-runtime.lock.txt']
for name in files:
    p=r/'outputs'/name
    if p.exists():(out/name).write_bytes(p.read_bytes())
manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.is_file()}
(out/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(files=len(manifest),total_bytes=sum(x['bytes'] for x in manifest.values()))))
