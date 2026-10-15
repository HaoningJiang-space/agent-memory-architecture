"""Run bounded remaining controls sequentially after the active graph run."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent
record=dict(start=time.time(),stages=[])
target=r/'outputs/graph_full_03/both_manifest.json'
deadline=time.monotonic()+7200
while time.monotonic()<deadline:
    m=json.loads(target.read_text())
    if m['status']!='running':break
    time.sleep(10)
else:raise RuntimeError('Graph completion deadline exceeded')
record['graph_status']=m['status']
commands=[
    ('fact_batching_control',['measure_fact_batching.py']),
    ('failed_update_replay',['replay_failed_update.py']),
    ('embed_full_01',['run_native.py','--algorithm','embedrag','--run','embed_full_01','--stage','both']),
    ('bm25_full_01',['run_native.py','--algorithm','bm25','--run','bm25_full_01','--stage','both']),
    ('summarize',['summarize_runs.py']),
    ('plot',['plot_motivation.py'])]
env=dict(os.environ,TIKTOKEN_CACHE_DIR=str(r/'tiktoken-cache'),NLTK_DATA=str(r/'nltk_data'),
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
for name,args in commands:
    start=time.time()
    with (r/'logs'/f'{name}.log').open('w') as log:
        completed=subprocess.run([sys.executable,'-u',str(r/args[0]),*args[1:]],cwd=r,
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    row=dict(name=name,start=start,end=time.time(),returncode=completed.returncode)
    record['stages'].append(row)
    (r/'outputs/pipeline.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(row),flush=True)
record['end']=time.time()
(r/'outputs/pipeline.json').write_text(json.dumps(record,indent=2)+'\n')
(r/'outputs/package_versions.json').write_text(json.dumps(
    {d.metadata['Name']:d.version for d in importlib.metadata.distributions()},indent=2)+'\n')
