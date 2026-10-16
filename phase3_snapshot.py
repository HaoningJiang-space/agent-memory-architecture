"""Export bounded observations with hashes; never export raw prompts or answers."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--name',required=True)
args = parser.parse_args()
assert all(c.isalnum() or c in '_-' for c in args.name)
out = root/'outputs/phase3'
target = out/(args.name+'.json')
assert not target.exists(), 'Observations are immutable'
receipts = {}


def read_json(path):
    if not path.exists(): return {}
    raw = path.read_bytes()
    receipts[str(path.relative_to(root))] = dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw)


def rows(path):
    if not path.exists(): return []
    raw = path.read_bytes()
    raw = raw[:raw.rfind(b'\n')+1]
    receipts[str(path.relative_to(root))] = dict(prefix_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return list(map(json.loads,raw.splitlines()))


record = dict(created=time.time(), pipeline=read_json(out/'pipeline.json'),
              scope='Completed step prefixes only. Structural validation is not semantic quality. '
                    'API timing includes host/runtime overhead. All jobs use one real history. '
                    'No new-hardware performance or isolated energy claim.')
record['grammar'] = {}
for name in ('grammar_replay','grammar_factorial'):
    manifest = read_json(out/name/'manifest.json')
    manifest.pop('measurements',None)
    stop = read_json(out/name/'early_stop.json')
    record['grammar'][name] = dict(manifest=manifest,measurements=rows(out/name/'measurements.jsonl'),
                                  external_stop=stop,
                                  observed_status='early_stopped' if stop else manifest.get('status','not_started'))
trials = rows(out/'protocol_ablation/raw.jsonl')
grouped = defaultdict(list)
for r in trials:
    grouped[r['variant']].append(r)
record['protocol'] = {}
for variant, group in grouped.items():
    record['protocol'][variant] = dict(calls=len(group),
        seconds=sum(r['end']-r['start'] for r in group),
        ttft=sum(r['first_token']-r['start'] for r in group),
        output_tokens=sum(r['usage']['completion_tokens'] for r in group),
        input_tokens=sum(r['usage']['prompt_tokens'] for r in group),
        agreement=sum(bool(r.get('exact_mutation_match_original')) for r in group),
        invalid_reference_cases=sum(bool(r.get('invalid_references')) for r in group),
        parse_or_validation_errors=sum(bool(r.get('parse_error')) for r in group),
        truncated=sum(r.get('finish_reason')=='length' for r in group),
        noop_count=sum(r.get('noop_count',0) for r in group),
        samples=[{k:r.get(k) for k in ('chunk','start','first_token','end','usage',
                                      'finish_reason','exact_mutation_match_original','protocol_validation','parse_error')}
                 for r in group])
api = rows(out/'api_trace.jsonl')
record['runs'] = {}
for name in ('mem0_8b_delta','mem0_8b_snapshot','mem0_32b_delta'):
    run = out/name
    manifest = read_json(run/'both_manifest.json')
    if not manifest: continue
    steps = rows(run/'steps.jsonl')
    passed = [s for s in steps if s['validation']=='passed']
    contexts = {(s['phase'],s['index']) for s in passed}
    times = defaultdict(float)
    for step in passed: times[step['phase']] += step['end']-step['start']
    calls = [r for r in api if r.get('context',{}).get('run') == 'phase3/'+name
             and (r['context'].get('phase'),r['context'].get('index')) in contexts]
    roles = defaultdict(lambda:dict(calls=0,seconds=0,input_tokens=0,output_tokens=0,ttft=0,after_first=0,
                                   max_prompt_tokens=0,max_request_tokens=0,cached_input_tokens=0))
    for r in calls:
        if r.get('path')!='chat/completions' or not r.get('usage'): continue
        role = 'query' if r['context']['phase']=='query' else ('update' if len(r['request']['messages'])==1 else 'extraction')
        entry = roles[role]
        usage = r['usage']
        entry['calls'] += 1
        entry['seconds'] += r['end']-r['start']
        entry['input_tokens'] += usage['prompt_tokens']
        entry['output_tokens'] += usage['completion_tokens']
        if r.get('first_token'):
            entry['ttft'] += r['first_token']-r['start']
            entry['after_first'] += r['end']-r['first_token']
        entry['max_prompt_tokens'] = max(entry['max_prompt_tokens'],usage['prompt_tokens'])
        entry['max_request_tokens'] = max(entry['max_request_tokens'],usage['prompt_tokens']+usage['completion_tokens'])
        entry['cached_input_tokens'] += (usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
    index = [r for r in rows(run/'index.jsonl') if (r['context']['phase'],r['context']['index']) in contexts]
    index_time = sum(r['end']-r['start'] for r in index)
    validation = rows(run/'protocol_validation.jsonl')
    record['runs'][name] = dict(status=manifest['status'],validity=manifest['validity'],
        error=manifest.get('error'),failed_context=manifest.get('failed_context'),
        total_chunks=manifest.get('total_chunks'),total_queries=manifest.get('total_queries'),
        completed_chunks=sum(s['phase']=='construction' for s in passed),
        completed_queries=sum(s['phase']=='query' for s in passed),
        failed_steps=sum(s['validation']=='failed' for s in steps),
        phase_seconds=times,llm_roles=roles,index_seconds=index_time,
        validation_attempts=len(validation),rejected_attempts=sum(not r['accepted'] for r in validation),
        software_control=manifest.get('software_control'))
record['source_receipts'] = receipts
target.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k in ('created','runs')},indent=2))
