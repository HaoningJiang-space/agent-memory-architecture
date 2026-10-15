"""Software control using real per-step Mem0 facts, preserving fact order."""
import collections
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from qdrant_client import QdrantClient

r=Path(__file__).resolve().parent
groups=collections.defaultdict(list)
rows=[json.loads(line) for line in (r/'outputs/api_trace.jsonl').read_text().splitlines()]
windows={}
for index in range(9):
    calls=sorted([x for x in rows if x.get('context',{}).get('run')=='mem0_full_01'
                 and x['context'].get('index')==index and x['path']=='chat/completions'],key=lambda x:x['start'])
    assert len(calls)==2
    windows[index]=(calls[0]['end'],calls[1]['start'])
for row in rows:
    c=row.get('context',{})
    if c.get('run')!='mem0_full_01' or c.get('index',100)>=9 or row['path']!='embeddings':
        continue
    begin,end=windows[c['index']]
    if not (row['start']>=begin and row['end']<=end):
        continue  # Post-update fallback embeddings have a different dependency.
    text=row['request']['input']
    if isinstance(text,str):text=[text]
    assert len(text)==1
    groups[c['index']].extend(text)

def embed(texts):
    req=urllib.request.Request('http://127.0.0.1:8033/v1/embeddings',
        data=json.dumps(dict(model='Qwen/Qwen3-Embedding-0.6B',input=texts)).encode(),
        headers={'Content-Type':'application/json'})
    return np.asarray([x['embedding'] for x in json.load(urllib.request.urlopen(req,timeout=120))['data']],dtype=np.float32)

db=QdrantClient(path=str(r/'outputs/mem0_full_01/state/mem0/qdrant'))
collections_available=[x.name for x in db.get_collections().collections]
assert 'mem0' in collections_available,collections_available
def top5(v):
    return [str(x.id) for x in db.query_points('mem0',query=v.tolist(),limit=5).points]

report=dict(scope='same actual texts; independent encoding control, not an end-to-end patched Mem0 run',
            selection='only embeddings between fact extraction and update LLM; excludes post-update writes',
            retrieval_scope='read-only frozen index after the failed run; numerical ranking check, not answer quality',
            source_run='mem0_full_01',groups=[])
embed(['warmup'])
for index,texts in sorted(groups.items()):
    serial_times=[];batch_times=[]
    for repeat in range(3):
        # Alternate order to reduce warmup/clock drift bias.
        for mode in (['serial','batch'] if repeat%2==0 else ['batch','serial']):
            begin=time.perf_counter()
            values=np.vstack([embed([s]) for s in texts]) if mode=='serial' else embed(texts)
            elapsed=time.perf_counter()-begin
            if mode=='serial':serial=values;serial_times.append(elapsed)
            else:batch=values;batch_times.append(elapsed)
    cosine=np.sum(serial*batch,axis=1)/(np.linalg.norm(serial,axis=1)*np.linalg.norm(batch,axis=1))
    s_ids=[top5(v) for v in serial];b_ids=[top5(v) for v in batch]
    row=dict(index=index,native_fact_calls=len(texts),serial_seconds=serial_times,batch_seconds=batch_times,
             speedup=float(np.median(serial_times)/np.median(batch_times)),min_cosine=float(cosine.min()),
             exact_top5_order_fraction=float(np.mean([a==b for a,b in zip(s_ids,b_ids)])),
             top5_set_overlap=float(np.mean([len(set(a)&set(b))/5 for a,b in zip(s_ids,b_ids)])))
    report['groups'].append(row)
    (r/'outputs/fact_batching_control.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(row),flush=True)
db.close()
