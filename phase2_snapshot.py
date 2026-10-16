"""Export a small, immutable overview observation while background runs continue."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
args = parser.parse_args()
if not all(c.isalnum() or c in '_-' for c in args.name):
    raise ValueError('Simple snapshot name required')
out = root/'outputs/phase2'/f'{args.name}.json'
if out.exists():
    raise RuntimeError('Snapshots are immutable')
receipts = {}


def read_rows(relative):
    path = root/relative
    if not path.exists():
        return []
    data = path.read_bytes()
    # An actively appended final line is not yet an observation.
    end = data.rfind(b'\n')+1
    data = data[:end]
    receipts[relative] = dict(prefix_bytes=end, sha256=hashlib.sha256(data).hexdigest())
    return [json.loads(line) for line in data.splitlines()]


run = root/'outputs/phase2/mem0_32b_native'
manifest = json.loads((run/'both_manifest.json').read_text())
steps = read_rows('outputs/phase2/mem0_32b_native/steps.jsonl')
construction = [s for s in steps if s['phase'] == 'construction']
recorded_construction = len(construction)
failed_context = manifest.get('failed_context')
validation_basis = 'Explicit step validation where available; legacy running snapshots are provisional.'
if manifest['status'] == 'failed' and steps and 'validation' not in steps[-1]:
    # Historical runner wrote its step before raising on captured upstream errors.
    # These frozen files are not rewritten; a new summary corrects that counting.
    if 'Upstream logged errors:' not in manifest.get('error','') or 'measured' not in manifest.get('traceback',''):
        raise RuntimeError('Cannot infer failed legacy step; audit before summarizing')
    failed_context = {k:steps[-1][k] for k in ['run','algorithm','phase','index']}
    construction = [s for s in construction if s is not steps[-1]]
    validation_basis = ('Legacy failed run: final recorded step excluded because it was written '
                        'before the upstream-error check raised. Prefix is structurally checked, not quality validated.')
construction = [s for s in construction if s.get('validation','passed') == 'passed']
cutoff = construction[-1]['end'] if construction else 0
last_index = construction[-1]['index'] if construction else -1
wall = sum(s['end']-s['start'] for s in construction)
index = [r for r in read_rows('outputs/phase2/mem0_32b_native/index.jsonl') if r['end'] <= cutoff]
operation = defaultdict(lambda:dict(calls=0,seconds=0))
for row in index:
    entry = operation[row['operation']]
    entry['calls'] += 1
    entry['seconds'] += row['end']-row['start']
index_seconds = sum(v['seconds'] for v in operation.values())
api = read_rows('outputs/phase2/api_trace.jsonl')
llm = [r for r in api if r.get('context',{}).get('run') == 'phase2/mem0_32b_native'
       and r['context'].get('phase') == 'construction' and r['context']['index'] <= last_index
       and r['path'] == 'chat/completions']
roles = defaultdict(lambda:dict(calls=0,input_tokens=0,output_tokens=0,seconds=0,ttft=0,after_first=0))
for row in llm:
    entry = roles['update' if len(row['request']['messages']) == 1 else 'extraction']
    entry['calls'] += 1
    entry['input_tokens'] += row['usage']['prompt_tokens']
    entry['output_tokens'] += row['usage']['completion_tokens']
    entry['seconds'] += row['end']-row['start']
    entry['ttft'] += row['first_token']-row['start']
    entry['after_first'] += row['end']-row['first_token']
protocol = read_rows('outputs/phase2/protocol_ablation/raw.jsonl')
control = []
for row in protocol:
    control.append({key:row.get(key) for key in [
        'source_trace_id','chunk','variant','existing_count','new_fact_count','start','first_token',
        'end','usage','finish_reason','invalid_references','mutation_count',
        'exact_mutation_match_original','noop_count','parse_error']})
record = dict(created=time.time(), native_status=manifest['status'],
              native_validity=manifest['validity'], completed_construction_chunks=len(construction),
              recorded_construction_steps=recorded_construction, failed_step_context=failed_context,
              validation_basis=validation_basis,
              total_construction_chunks=manifest['total_chunks'], completed_query_steps=sum(s['phase']=='query' for s in steps),
              construction_wall_seconds=wall, index_operations=operation, llm_roles=roles,
              index_fraction=index_seconds/wall if wall else None,
              zero_index_latency_serial_upper_bound=wall/(wall-index_seconds) if wall>index_seconds else None,
              protocol_trials=control, source_prefix_receipts=receipts,
              limitations=['Partial observations do not establish completed-run validity or semantic quality.',
                           'Index upper bound assumes the observed serial schedule and all other work unchanged.',
                           'Native uses GPU1 for 32B and GPU0 for embedding; protocol controls also use GPU0.',
                           'Do not use overlapping runs for uncontended end-to-end latency or GPU-energy comparisons.',
                           'Protocol operation agreement is measured against original output, not independent truth.'])
out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('protocol_trials','source_prefix_receipts')},indent=2))
