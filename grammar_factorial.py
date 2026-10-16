"""Isolate anyOf, ID enum, required fields, and minLength in CPU grammar matching."""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

from protocol_controls import flat_schema, snapshot

root = Path(__file__).resolve().parent
out = root/'outputs/phase3/grammar_factorial'
out.mkdir(parents=True,exist_ok=True)
assert not (out/'manifest.json').exists()
record = dict(start=time.time(),status='running',
    scope='One identical serialized JSON stream from a real frozen request; CPU grammar only. '
          'Shared host and concurrent CPU grammar job; process CPU time reported separately. '
          'No inference or index operations. Not a GPU or new-hardware speedup.')
(out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
try:
    import torch
    import xgrammar as xgr
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    model = json.loads((root/'models.json').read_text())['Qwen/Qwen3-8B']['path']
    tokenizer = AutoTokenizer.from_pretrained(model,local_files_only=True)
    vocab = json.loads((Path(model)/'config.json').read_text())['vocab_size']
    info = xgr.TokenizerInfo.from_huggingface(tokenizer,vocab_size=vocab)
    raw = (root/'outputs/api_trace.jsonl').read_bytes()
    receipt = json.loads((root/'outputs/raw_artifact_sha256.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == receipt['files']['outputs/api_trace.jsonl']['sha256']
    source = next(r for r in map(json.loads,raw.splitlines()) if r.get('context',{}).get('run')=='mem0_full_01'
                  and r.get('context',{}).get('index')==2 and r.get('path')=='chat/completions'
                  and len(r['request']['messages'])==1)
    old, _ = snapshot(source['request']['messages'][0]['content'])
    valid = [str(m['id']) for m in old]
    result = json.loads(source['response_text'])
    for action in result['memory']:
        if action['event']=='ADD':
            action['id']='NEW'
        else:
            assert action['id'] in valid
    text = json.dumps(result,ensure_ascii=False)
    tokens = tokenizer.encode(text,add_special_tokens=False)
    # Cross minLength with schema shape while holding token stream fixed.
    schemas = {}
    for shape in ('native','flat','branches'):
        for minimum in (False,True):
            schema = copy.deepcopy(source['request']['guided_json']) if shape=='native' else flat_schema(valid)
            item = schema['properties']['memory']['items']
            if minimum:
                item['properties']['text']['minLength']=1
            if shape=='branches':
                add, existing = copy.deepcopy(item),copy.deepcopy(item)
                add['properties']['id']={'type':'string'}
                add['properties']['event']['enum']=['ADD']
                existing['properties']['id']['enum']=valid
                existing['properties']['event']['enum']=['UPDATE','DELETE','NONE']
                schema['properties']['memory']['items']={'anyOf':[add,existing]}
            schemas[f'{shape}_min{int(minimum)}']=schema
    rows=[]
    for repeat in range(2):
        order=list(schemas)
        if repeat: order.reverse()
        for name in order:
            compiler=xgr.GrammarCompiler(info,max_threads=8,cache_enabled=False)
            begin,cpu=time.perf_counter(),time.process_time()
            grammar=compiler.compile_json_schema(json.dumps(schemas[name]),any_whitespace=True)
            compile_wall,compile_cpu=time.perf_counter()-begin,time.process_time()-cpu
            matcher=xgr.GrammarMatcher(grammar,max_rollback_tokens=0)
            mask=xgr.allocate_token_bitmask(1,vocab)
            begin,cpu=time.perf_counter(),time.process_time()
            for token in tokens:
                matcher.fill_next_token_bitmask(mask,0)
                assert matcher.accept_token(token)
            assert matcher.accept_token(tokenizer.eos_token_id)
            row=dict(variant=name,repeat=repeat,tokens=len(tokens),wall=time.perf_counter()-begin,
                     cpu=time.process_time()-cpu,compile_wall=compile_wall,compile_cpu=compile_cpu)
            rows.append(row)
            with (out/'measurements.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    record.update(status='complete',source_trace_sha256=hashlib.sha256(raw).hexdigest(),
                  source_trace_id=source['id'],normalized_stream_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  schema_sha256={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in schemas.items()},
                  measurements=rows)
except BaseException as exc:
    record.update(status='failed',error=repr(exc))
    raise
finally:
    record['end']=time.time()
    (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
