"""Frozen-state Mem0 update controls; operation agreement is not semantic accuracy."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import time
import urllib.request

root = Path(__file__).resolve().parent
out = root/'outputs/phase2/protocol_ablation'
out.mkdir(parents=True, exist_ok=True)
target = out/'raw.jsonl'
if target.exists():
    raise RuntimeError('Choose a new output before repeating the ablation')
raw = (root/'outputs/api_trace.jsonl').read_bytes()
receipt = json.loads((root/'outputs/raw_artifact_sha256.json').read_text())
assert hashlib.sha256(raw).hexdigest() == receipt['files']['outputs/api_trace.jsonl']['sha256']
sources = [row for row in map(json.loads, raw.splitlines())
           if row.get('context', {}).get('run') == 'mem0_full_01'
           and row.get('path') == 'chat/completions'
           and len(row['request']['messages']) == 1]
assert len(sources) == 10


def snapshot(source):
    prompt = source['request']['messages'][0]['content']
    position = prompt.index('Below is the current content of my memory which I have collected till now.')
    sections = prompt[position:].split('```')
    return ast.literal_eval(sections[1].strip()), ast.literal_eval(sections[3].strip())


def schema(valid, delta):
    common = dict(type='object', properties={
        'id':dict(type='string'), 'text':dict(type='string', minLength=1),
        'event':dict(type='string'), 'old_memory':dict(type='string')},
        required=['id', 'text', 'event'], additionalProperties=False)
    addition = copy.deepcopy(common)
    addition['properties']['event']['enum'] = ['ADD']
    branches = [addition]
    if valid:
        existing = copy.deepcopy(common)
        existing['properties']['id']['enum'] = valid
        existing['properties']['event']['enum'] = ['UPDATE','DELETE'] if delta else ['UPDATE','DELETE','NONE']
        branches.append(existing)
    return dict(type='object', properties={
        'memory':dict(type='array', items={'anyOf':branches})},
        required=['memory'], additionalProperties=False)


def normalized(result):
    return sorted((op['event'], op.get('id') if op['event'] != 'ADD' else None, op['text'])
                  for op in result['memory'] if op['event'] != 'NONE')


record = dict(start=time.time(), status='running', source_trace_sha256=hashlib.sha256(raw).hexdigest(),
              model='Qwen/Qwen3-8B', samples=len(sources),
              scope='Ten real frozen update inputs, no state writes. Native, legal-ID schema, '
                    'and legal-ID plus delta-output controls. Prompt/schema changes can change decisions. '
                    'Exact mutation agreement is a diagnostic, not an independent quality judgment. '
                    'GPU0 also hosts the embedding service; overlap is recorded, energy is not compared.')
(out/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
try:
    deadline = time.monotonic()+600
    while True:
        try:
            urllib.request.urlopen('http://127.0.0.1:8031/health',timeout=3).read()
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError('8B server readiness timeout')
            time.sleep(5)
    for source in sources:
        memory, facts = snapshot(source)
        valid = [str(item['id']) for item in memory]
        # Alternate order to expose, rather than always favor, cache warmness.
        variants = ['native','ids','delta']
        if source['context']['index'] % 2:
            variants.reverse()
        baseline = normalized(json.loads(source['response_text']))
        for variant in variants:
            body = copy.deepcopy(source['request'])
            body.update(model='Qwen/Qwen3-8B', stream=True, stream_options={'include_usage':True})
            if variant != 'native':
                body['guided_json'] = schema(valid, delta=variant == 'delta')
            if variant == 'delta':
                body['messages'][0]['content'] += (
                    '\nFor this response, return only actual ADD, UPDATE, and DELETE operations. '
                    'Do not serialize unchanged memories or NONE operations. An omitted existing '
                    'memory is retained without modification. An empty memory array means no changes. '
                    'This delta-only output rule overrides the snapshot and NONE examples above. '
                    'Existing IDs are references to the existing memory table, never output row numbers. '
                    'Use an existing ID for UPDATE or DELETE; ADD IDs are placeholders ignored by the store.')
            begin = time.time()
            row = dict(source_trace_id=source['id'], chunk=source['context']['index'],
                       variant=variant, existing_count=len(memory), new_fact_count=len(facts),
                       request=body, start=begin)
            request = urllib.request.Request('http://127.0.0.1:8031/v1/chat/completions',
                        data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
            content = []
            with urllib.request.urlopen(request, timeout=900) as response:
                for line in response:
                    if not line.startswith(b'data: ') or line[6:].strip() == b'[DONE]':
                        continue
                    payload = json.loads(line[6:])
                    if payload.get('usage'):
                        row['usage'] = payload['usage']
                    for choice in payload.get('choices', []):
                        text = choice.get('delta', {}).get('content')
                        if text:
                            row.setdefault('first_token', time.time())
                            content.append(text)
                        if choice.get('finish_reason'):
                            row['finish_reason'] = choice['finish_reason']
            row.update(end=time.time(), response_text=''.join(content))
            try:
                result = json.loads(row['response_text'])
                row['invalid_references'] = [op for op in result['memory']
                    if op['event'] in ('UPDATE','DELETE') and str(op.get('id')) not in valid]
                row['mutation_count'] = len(normalized(result))
                row['exact_mutation_match_original'] = normalized(result) == baseline
                row['noop_count'] = sum(op['event'] == 'NONE' for op in result['memory'])
            except (ValueError, KeyError, TypeError) as exc:
                row['parse_error'] = repr(exc)
            with target.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('request','response_text')}), flush=True)
    record['status'] = 'complete'
except Exception as exc:
    record.update(status='failed', error=repr(exc))
    raise
finally:
    record['end'] = time.time()
    (out/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
