"""Account for added model residency and saved KV using real paired QA requests.

Only a completed, structurally valid Mem0 run may enter this accounting. The
first-N queue is an explicit hypothetical admission scenario over distinct real
queries, not measured arrivals or synthetic replicated users. No speedup claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

from workload_accounting import decoder_shape

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--run',required=True)
parser.add_argument('--name',required=True)
args = parser.parse_args()
assert all(c.isalnum() or c in '_-' for c in args.name)
manifest = json.loads((root/'outputs'/args.run/'both_manifest.json').read_text())
trace_path = root/manifest['args']['trace_file']
out = trace_path.parent
target = out/(args.name+'.json')
assert not target.exists()
assert manifest['status']=='complete' and manifest['validity']=='valid', 'Full valid execution required'
models = json.loads((root/'models.json').read_text())
config = {name:json.loads((Path(v['path'])/'config.json').read_text()) for name,v in models.items()}
kv = decoder_shape(config['Qwen/Qwen3-32B'])['kv_bytes_per_token']
small_kv = decoder_shape(config[manifest['args']['construction_model']])['kv_bytes_per_token']
# Safetensors header data_offsets account for parameter bytes without loading weights.
def weight_bytes(path):
    total = 0
    for shard in Path(path).glob('*.safetensors'):
        with shard.open('rb') as f:
            size = int.from_bytes(f.read(8),'little')
            header = json.loads(f.read(size))
        total += sum(t['data_offsets'][1]-t['data_offsets'][0] for k,t in header.items() if k!='__metadata__')
    assert total>0
    return total

weights = {name:weight_bytes(v['path']) for name,v in models.items()}
old_raw = (root/'outputs/api_trace.jsonl').read_bytes()
receipt = json.loads((root/'outputs/raw_artifact_sha256.json').read_text())
assert hashlib.sha256(old_raw).hexdigest()==receipt['files']['outputs/api_trace.jsonl']['sha256']
new_raw = trace_path.read_bytes()
new_raw = new_raw[:new_raw.rfind(b'\n')+1]
def select(raw, run, phase):
    return [r for r in map(json.loads,raw.splitlines()) if r.get('context',{}).get('run')==run
            and r['context'].get('phase')==phase and r.get('path')=='chat/completions']
base = {r['context']['index']:r for r in select(old_raw,'embed_full_01','query')}
memory = {r['context']['index']:r for r in select(new_raw,args.run,'query')}
assert set(base)==set(memory)==set(range(60))
base_manifest = json.loads((root/'outputs/embed_full_01/both_manifest.json').read_text())
assert base_manifest['dataset_sha256']==manifest['dataset_sha256']
extra = weights['Qwen/Qwen3-8B'] if manifest['args']['construction_model']=='Qwen/Qwen3-8B' else 0
construction = select(new_raw,args.run,'construction')
construction_kv = max((r['usage']['prompt_tokens']+r['usage']['completion_tokens']-1)*small_kv
                      for r in construction)
paired = []
for index in sorted(base):
    bu, mu = base[index]['usage'],memory[index]['usage']
    # The first output uses prefill logits; the final sampled token need not enter KV.
    long_tokens = bu['prompt_tokens']+bu['completion_tokens']-1
    memory_tokens = mu['prompt_tokens']+mu['completion_tokens']-1
    paired.append(dict(index=index,long_prompt=bu['prompt_tokens'],memory_prompt=mu['prompt_tokens'],
                       long_peak_kv_tokens=long_tokens,memory_peak_kv_tokens=memory_tokens,
                       saved_kv_bytes=(long_tokens-memory_tokens)*kv))
cases = []
saved = 0
for pair in paired:
    saved += pair['saved_kv_bytes']
    cases.append(dict(distinct_queries=pair['index']+1,saved_query_kv_bytes=saved,
        net_added_bytes_no_construction_kv=extra-saved,
        net_added_bytes_with_one_peak_construction_request=extra+construction_kv-saved))
record = dict(scope=__doc__,run=args.run,source_dataset_sha256=manifest['dataset_sha256'],
              source_receipts=dict(baseline_sha256=hashlib.sha256(old_raw).hexdigest(),
                                   new_trace_prefix_bytes=len(new_raw),new_trace_sha256=hashlib.sha256(new_raw).hexdigest()),
              weight_bytes=weights,extra_construction_weight_bytes=extra,construction_peak_request_kv_bytes=construction_kv,
              reasoning_kv_bytes_per_token=kv,queries=paired,admission_scenarios=cases,
              limitations=['Semantic quality is not established by structural validity or this capacity calculation.',
                  'Logical peak KV for first N distinct queries, no measured simultaneous peak or arrival process.',
                  'No workspace/activation/fragmentation/prefix-sharing allowance; index remains on CPU.',
                  'Baseline disables prefix cache; new run enables it. No timing ranking is made.',
                  'Answer token caps differ (256 versus 1024); actual observed completion lengths are counted.'])
target.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(run=args.run,queries=len(paired),extra_weight_GiB=extra/2**30,
                     first_capacity_crossing=next((c for c in cases if c['net_added_bytes_no_construction_kv']<0),None)),indent=2))
