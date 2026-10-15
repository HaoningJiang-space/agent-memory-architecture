"""Bounded same-request control for the observed BM25 answer truncation."""
import copy
import hashlib
import json
from pathlib import Path
import time
import urllib.request

root = Path(__file__).resolve().parent
rows = [json.loads(s) for s in (root/'outputs/api_trace.jsonl').read_text().splitlines()]
source = [x for x in rows if x['context'].get('run')=='bm25_full_01'
          and x['context'].get('index')==33 and x['path']=='chat/completions'][-1]
body = copy.deepcopy(source['request'])
body.update(stream=False, max_tokens=1024)
body.pop('stream_options',None)
start = time.time()
request = urllib.request.Request('http://127.0.0.1:8032/v1/chat/completions',
    data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
response = json.load(urllib.request.urlopen(request,timeout=600))
report = dict(source_trace_id=source['id'],original_cap=source['request']['max_tokens'],
              replay_cap=1024,request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),
              seconds=time.time()-start,usage=response['usage'],
              finish_reason=response['choices'][0]['finish_reason'],response=response)
(root/'outputs/capped_answer_replay.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='response'}))
assert report['finish_reason']=='stop','Larger cap did not repair truncation'
