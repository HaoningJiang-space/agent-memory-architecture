"""Recount fixed LightMem calls, without a latency or target-hardware model.

The byte count is one logical linear-weight pass per forward with independent
calls and no inter-forward cache reuse. It is neither measured DRAM traffic nor
the legal-batch accounting used by overview_accounting.py.
"""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from workload_accounting import decoder_shape


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = ['evidence/architecture/lightmem_paths.json.gz',
              'evidence/phase2/workload_accounting.json']
    traces = json.loads(gzip.decompress((root / inputs[0]).read_bytes()))
    models = json.loads((root / inputs[1]).read_text())['models']
    shapes = {m: decoder_shape(v['config']) for m, v in models.items()
              if v['shape'] is not None}
    result = dict(
        scope=__doc__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        decoder_shape_source_sha256=hashlib.sha256(
            (root / 'workload_accounting.py').read_bytes()).hexdigest(),
        input_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                      for p in inputs},
        exclusions=['attention', 'KV and activation traffic', 'prefix caching',
                    'batch reuse', 'queueing', 'quality and semantic validity',
                    'hardware time, energy and cost'],
        cases=[])
    for case in traces['cases']:
        assert case['status'] == 'complete'
        assert len({c['id'] for c in case['calls']}) == len(case['calls'])
        stages = defaultdict(lambda: dict(calls=0, input_tokens=0, output_tokens=0,
                                          linear_flops=0, one_pass_weight_bytes=0))
        for call in case['calls']:
            assert call['finish_reason'] == 'stop'
            assert call['L'] > 0 and call['O'] >= 1
            s = shapes[call['model']]
            v = stages[(call['phase'], call['model'])]
            v['calls'] += 1
            v['input_tokens'] += call['L']
            v['output_tokens'] += call['O']
            # Prefill gives the first output; each additional output is a forward.
            v['one_pass_weight_bytes'] += s['streamed_linear_weight_bytes'] * call['O']
            v['linear_flops'] += 2 * (s['projection_parameters'] *
                                     (call['L'] + call['O'] - 1) +
                                     s['head_parameters'] * call['O'])
        expected = {(s['phase'], s['model']): s for s in case['phase_stats']}
        assert set(expected) == set(stages)
        for key, v in stages.items():
            e = expected[key]
            assert (v['calls'], v['input_tokens'], v['output_tokens']) == (
                e['count'], e['total_input_tokens'], e['total_output_tokens'])
        write = sum(v['one_pass_weight_bytes'] for (p, m), v in stages.items()
                    if m == 'Qwen/Qwen3-8B' and p in ('construction', 'maintenance'))
        queries = [c for c in case['calls'] if c['phase'].startswith('query_')]
        assert len(queries) == 3 and all(c['model'] == 'Qwen/Qwen3-32B' for c in queries)
        assert all(p in ('construction', 'maintenance') or p.startswith('query_')
                   for p, m in stages)
        assert all(m == ('Qwen/Qwen3-32B' if p.startswith('query_') else 'Qwen/Qwen3-8B')
                   for p, m in stages)
        query = sum(v['one_pass_weight_bytes'] for (p, m), v in stages.items()
                    if p.startswith('query_'))
        result['cases'].append(dict(
            session=case['session'],
            stages=[dict(phase=p, model=m, **v) for (p, m), v in stages.items()],
            llm_calls=len(case['calls']),
            write_llm_one_pass_weight_bytes=write,
            three_query_llm_one_pass_weight_bytes=query,
            write_fraction_of_llm_proxy=write / (write + query),
            query_count_at_equal_llm_weight_proxy=3 * write / query,
            equality_scope='Continuous repetition of the observed three-query shape mix; '
                           'NOT valid semantic reuse or a lifecycle cost break-even.',
            embedding_sequences=sum(e['sequences'] for e in case['embedding_calls']),
            embedding_tokens=sum(e['tokens'] for e in case['embedding_calls']),
            encoder_linear_flops=sum(e['linear_flops'] for e in case['encoder_stages']),
            non_llm_scope='Embedding/encoder recorded separately, excluded from LLM-only equality.'))
    target = root / 'evidence/architecture/lifecycle_accounting.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    for case in result['cases']:
        print(case['session'], case['llm_calls'],
              round(case['write_fraction_of_llm_proxy'], 6),
              round(case['query_count_at_equal_llm_weight_proxy'], 6))


if __name__ == '__main__':
    main()
