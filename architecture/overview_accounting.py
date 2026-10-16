"""Re-account existing algorithm records; no new hardware or latency model."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    names = ['evidence/architecture/hetero_projection_shapes.json',
             'evidence/architecture/batch_closure_traces.json.gz',
             'evidence/calibration/model_capacity.json']
    shapes = json.loads((root/names[0]).read_text())
    frozen = json.loads(gzip.decompress((root/names[1]).read_bytes()))
    capacity = json.loads((root/names[2]).read_text())
    weights = {m['model']:m['weight_bytes'] for m in capacity['models']}
    result = dict(scope='Existing finite calls and controlled legal groups, NOT DRAM counters or deployment mix',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
                  streams=[])
    for stream in shapes['streams']:
        if 'concurrent' not in stream['graph'] or stream['maintenance_cap'] is not None:
            continue
        totals = defaultdict(lambda:dict(one_pass_linear_weight_bytes=0, flops=0))
        for r in stream['rows']:
            b = 2*r['k']*r['n']*r['count']
            totals[r['model']]['one_pass_linear_weight_bytes'] += b
            totals[r['model']]['flops'] += b*r['m']
        total_b = sum(v['one_pass_linear_weight_bytes'] for v in totals.values())
        total_f = sum(v['flops'] for v in totals.values())
        for model,v in totals.items():
            v.update(resident_bf16_weight_bytes=weights[model],
                     weight_work_fraction=v['one_pass_linear_weight_bytes']/total_b,
                     flop_fraction=v['flops']/total_f)
        stages = defaultdict(lambda:dict(calls=0, input_tokens=0, output_tokens=0))
        graph = next(g for g in frozen['graphs'] if g['name'] == stream['graph'])
        for node in graph['nodes']:
            if node['kind'] != 'llm':
                continue
            stage = 'query' if node['stage'].startswith('query_') else node['stage']
            value = stages[(node['model'],stage)]
            value['calls'] += 1
            value['input_tokens'] += node['L']
            value['output_tokens'] += node['O']
        result['streams'].append(dict(graph=stream['graph'], maintenance_cap=None,
            models=dict(totals), stages=[dict(model=m,stage=s,**v) for (m,s),v in stages.items()],
            completed_main_query_calls=sum(v['calls'] for (m,s),v in stages.items() if m=='Qwen/Qwen3-32B')))
    (root/'evidence/architecture/overview_accounting.json').write_text(json.dumps(result,indent=2)+'\n')
    print([(s['graph'],s['completed_main_query_calls']) for s in result['streams']])


if __name__ == '__main__':
    main()
