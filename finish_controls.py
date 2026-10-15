"""Finish the bounded BM25 cap repair and stop on any validation failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
record = dict(start=time.time(), reason='BM25 answer 34 reached the original 256-token cap', stages=[])
commands = [
    ('bm25_full_02', ['run_native.py','--algorithm','bm25','--run','bm25_full_02',
                      '--stage','both','--answer-max-tokens','1024']),
    ('final_summary', ['summarize_runs.py']),
    ('validation', ['validate_evidence.py']),
    ('final_plot', ['plot_motivation.py']),
]
env = dict(os.environ, NLTK_DATA=str(root/'nltk_data'), TIKTOKEN_CACHE_DIR=str(root/'tiktoken-cache'),
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
for name, args in commands:
    begin = time.time()
    with (root/'logs'/f'{name}.log').open('w') as log:
        completed = subprocess.run([sys.executable,'-u',str(root/args[0]),*args[1:]], cwd=root,
                                   env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    stage = dict(name=name,start=begin,end=time.time(),returncode=completed.returncode)
    record['stages'].append(stage)
    record['status']='failed' if completed.returncode else 'running'
    (root/'outputs/completion_pipeline.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(stage),flush=True)
    if completed.returncode:
        sys.exit(completed.returncode)
record.update(status='complete',end=time.time())
(root/'outputs/completion_pipeline.json').write_text(json.dumps(record,indent=2)+'\n')
