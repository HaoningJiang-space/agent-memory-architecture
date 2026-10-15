"""Replay one observed bad Mem0 update; validate references, not just JSON syntax."""
import ast
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import urllib.request

r=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--model',choices=['Qwen/Qwen3-8B','Qwen/Qwen3-32B'])
parser.add_argument('--repeats',type=int,default=2)
args=parser.parse_args()
rows=[json.loads(s) for s in (r/'outputs/api_trace.jsonl').read_text().splitlines()]
source=[x for x in rows if x['context'].get('run')=='mem0_full_01' and
        x['context'].get('index')==9 and x['path']=='chat/completions'][-1]
text=source['request']['messages'][0]['content']
start=text.index('Below is the current content of my memory which I have collected till now.')
memory=ast.literal_eval(text[start:].split('```')[1].strip())
valid={str(x['id']) for x in memory}
def check(payload):
    operations=payload['memory']
    return [dict(id=x.get('id'),event=x.get('event')) for x in operations
            if x.get('event') in ['UPDATE','DELETE'] and str(x.get('id')) not in valid]
report=dict(source_trace_id=source['id'],input_sha256=hashlib.sha256(text.encode()).hexdigest(),
            existing_id_count=len(valid),original_invalid_references=check(json.loads(source['response_text'])),
            scope='single fixed failed request; reference validity only, not full memory quality',trials=[])
for model in ([args.model] if args.model else ['Qwen/Qwen3-8B','Qwen/Qwen3-32B']):
    for repeat in range(args.repeats):
        body=copy.deepcopy(source['request'])
        body.update(model=model,stream=False)
        body.pop('stream_options',None)
        t=time.time()
        port=8031 if model=='Qwen/Qwen3-8B' else 8032
        request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        response=json.load(urllib.request.urlopen(request,timeout=600))
        choice=response['choices'][0]
        result=json.loads(choice['message']['content'])
        trial=dict(model=model,repeat=repeat,seconds=time.time()-t,usage=response['usage'],
                   finish_reason=choice['finish_reason'],invalid_references=check(result),response=result)
        report['trials'].append(trial)
        suffix='_'+model.split('/')[-1] if args.model else ''
        (r/f'outputs/failed_update_replay{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in trial.items() if k!='response'}),flush=True)
