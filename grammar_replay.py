"""CPU-only replay of identical valid JSON token streams through xgrammar.

This isolates grammar compilation/matching, not GPU masking or server latency.
Raw strings remain remote. JSON ADD placeholders are normalized explicitly so
the exact same stream is legal under every schema.
"""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

from protocol_controls import flat_schema, snapshot

root = Path(__file__).resolve().parent
out = root/'outputs/phase3/grammar_replay'
out.mkdir(parents=True, exist_ok=True)
assert not (out/'manifest.json').exists()
record = dict(start=time.time(), status='running', scope=__doc__)
(out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')


def branch_schema(valid):
    common = dict(type='object', properties={
        'id':dict(type='string'), 'text':dict(type='string',minLength=1),
        'event':dict(type='string'), 'old_memory':dict(type='string')},
        required=['id','text','event'],additionalProperties=False)
    add = copy.deepcopy(common)
    add['properties']['event']['enum'] = ['ADD']
    branches = [add]
    if valid:
        old = copy.deepcopy(common)
        old['properties']['id']['enum'] = valid
        old['properties']['event']['enum'] = ['UPDATE','DELETE','NONE']
        branches.append(old)
    return dict(type='object',properties={'memory':dict(type='array',items={'anyOf':branches})},
                required=['memory'],additionalProperties=False)


try:
    import torch
    import xgrammar as xgr
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    model = json.loads((root/'models.json').read_text())['Qwen/Qwen3-8B']['path']
    tokenizer = AutoTokenizer.from_pretrained(model,local_files_only=True)
    vocab_size = json.loads((Path(model)/'config.json').read_text())['vocab_size']
    info = xgr.TokenizerInfo.from_huggingface(tokenizer,vocab_size=vocab_size)
    raw = (root/'outputs/api_trace.jsonl').read_bytes()
    frozen = json.loads((root/'outputs/raw_artifact_sha256.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == frozen['files']['outputs/api_trace.jsonl']['sha256']
    sources = [r for r in map(json.loads,raw.splitlines()) if r.get('context',{}).get('run') == 'mem0_full_01'
               and r.get('path') == 'chat/completions' and len(r['request']['messages']) == 1
               and r['context']['index'] in (0,2,5,8)]
    rows = []
    for source in sources:
        old, _ = snapshot(source['request']['messages'][0]['content'])
        valid = [str(m['id']) for m in old]
        result = json.loads(source['response_text'])
        for op in result['memory']:
            if op['event'] == 'ADD':
                op['id'] = 'NEW'
            else:
                assert op['id'] in valid
        text = json.dumps(result,ensure_ascii=False)
        tokens = tokenizer.encode(text,add_special_tokens=False)
        schemas = dict(native=source['request']['guided_json'], branches=branch_schema(valid),flat=flat_schema(valid))
        for name, schema in schemas.items():
            compiler = xgr.GrammarCompiler(info,max_threads=8,cache_enabled=False)
            begin, cpu = time.perf_counter(),time.process_time()
            grammar = compiler.compile_json_schema(json.dumps(schema),any_whitespace=True)
            compile_wall, compile_cpu = time.perf_counter()-begin,time.process_time()-cpu
            for repeat in range(3):
                matcher = xgr.GrammarMatcher(grammar,max_rollback_tokens=0)
                bitmask = xgr.allocate_token_bitmask(1,vocab_size)
                begin, cpu = time.perf_counter(),time.process_time()
                durations = []
                for token in tokens:
                    start = time.perf_counter()
                    matcher.fill_next_token_bitmask(bitmask,0)
                    assert matcher.accept_token(token), (name,token)
                    durations.append(time.perf_counter()-start)
                assert matcher.accept_token(tokenizer.eos_token_id)
                row = dict(chunk=source['context']['index'],variant=name,repeat=repeat,
                           tokens=len(tokens),stream_sha256=hashlib.sha256(text.encode()).hexdigest(),
                           wall=time.perf_counter()-begin,cpu=time.process_time()-cpu,
                           compile_wall=compile_wall,compile_cpu=compile_cpu,
                           p50_token_seconds=statistics.median(durations),
                           p95_token_seconds=sorted(durations)[int(.95*(len(durations)-1))])
                rows.append(row)
                with (out/'measurements.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
    record.update(status='complete',xgrammar=importlib.metadata.version('xgrammar'),
                  source_sha256=hashlib.sha256(raw).hexdigest(),measurements=rows)
except Exception as exc:
    record.update(status='failed',error=repr(exc))
    raise
finally:
    record['end'] = time.time()
    (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
