"""Audit published evidence and comparison scope; no new hardware sweep.

Paper numbers are explicitly transcribed references, not reproduced results.
Only aggregate token counts, target-DAG ancestry and saved hardware parameters
are inspected. No private prompts or new inference are used.
"""
import gzip
import hashlib
import json
from pathlib import Path

from architecture.batch_closure import graph


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/architecture/paper_alignment.json'


def audit():
    sources = [
        'evidence/architecture/family_summary.json',
        'evidence/architecture/call_placement.json.gz',
        'architecture/family_workload.py',
        'architecture/batch_closure.py',
        'architecture/paper_alignment.py',
        'evidence/architecture/family_provenance.json',
    ]
    families = json.loads((ROOT / sources[0]).read_text())['algorithms']
    provenance = json.loads((ROOT / sources[-1]).read_text())
    light_config = provenance['runs']['family_01/lightmem']['phase_config']
    with gzip.open(ROOT / sources[1], 'rt') as handle:
        placement = json.load(handle)
    tokens, barriers = {}, {}
    for family in families:
        calls = [c for c in family['calls'] if c['phase'] == 'construction']
        embeddings = [e for e in family['embedding_calls'] if e['phase'] == 'construction']
        if not calls or not embeddings:
            raise ValueError('Required construction stage absent')
        prompt, output = sum(c['L'] for c in calls), sum(c['O'] for c in calls)
        embed = sum(e['tokens'] for e in embeddings)
        tokens[family['algorithm']] = dict(
            llm_prompt=prompt, llm_output=output, embedding_input=embed,
            output_share_llm_only=output / (prompt + output),
            output_share_all_construction_tokens=output / (prompt + output + embed),
            source_revision=family['native_revision'],
            scope='Observed counts, not elapsed time, FLOPs, or hardware traffic.')
        for parallel in (False, True):
            dag = graph(family, parallel)
            ancestors = {}
            construction = set()
            queries = []
            for node in dag['nodes']:
                ancestors[node['id']] = set(node['deps'])
                for parent in node['deps']:
                    ancestors[node['id']].update(ancestors[parent])
                if node['stage'].startswith('query_'):
                    queries.append(node['id'])
                elif node['kind'] in ('llm', 'embedding'):
                    construction.add(node['id'])
            if not queries or not construction:
                raise ValueError('Missing graph stage')
            # This is an audit of the frozen graph, not a desired scheduler rule.
            all_after = all(construction <= ancestors[q] for q in queries)
            assert all_after, 'Graph changed: revisit paper-reconciliation claim'
            key = family['algorithm'] + ('_concurrent_queries' if parallel else '_sequential_queries')
            barriers[key] = dict(query_nodes=len(queries), build_neural_nodes=len(construction),
                every_query_waits_for_all_build_neural_nodes=all_after)

    domains = {}
    for name, hw in placement['hetero_devices'].items():
        peak = 2 * hw['arrays'] * hw['rows'] * hw['cols'] * hw['frequency']
        bandwidth = hw['bandwidth'] * hw['bandwidth_efficiency']
        domains[name] = dict(peak_flops=peak, effective_dram_bytes_per_second=bandwidth,
            peak_balance_flop_per_byte=peak / bandwidth,
            bf16_weight_only_balance_M=peak / bandwidth,
            scope='Peak roofline reference only; excludes attention, tile reloads, '
                  'activation traffic, utilization and SRAM bottlenecks. Not a measured crossover.')

    examples = {}
    for name, stage in [('simplemem', 'construction'), ('lightmem', 'maintenance')]:
        run = next(r for r in placement['runs'] if r['graph'] == name + '_concurrent_queries'
                   and r['policy'] == 'joint' and r['attention_mode'] == 'frozen'
                   and not r['free_kv_time'])
        batch = next(b for b in run['batches'] if b['stage'] == stage and b['kind'] == 'llm')
        examples[name + '_' + stage] = dict(batch=batch['batch'],
            alternatives=batch['alternatives'], scope='Saved target-model services, not A800 timings.')

    return dict(
        scope='Comparison audit, not a paper reproduction or new architecture result.',
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        observed_construction_tokens=tokens,
        paper_reference=dict(url='https://arxiv.org/html/2606.06448v1', section='4.3, Figure 5',
            denominator='LLM prompt + LLM completion + embedding input tokens',
            across_system_median_output_share=0.046, simplemem_output_share=0.285,
            lightmem_in_this_paper_suite=False,
            provenance='Transcribed published values; different models and history, not reproduced.'),
        target_graph_barriers=barriers, target_domain_balance=domains,
        saved_service_examples=examples,
        lightmem_scope=dict(pre_compress=light_config['pre_compress'],
            topic_segment=light_config['topic_segment'],
            source='family_provenance.json: family_01/lightmem phase_config',
            consequence='Full LightMem paper pipeline and its work proportions are not represented.'))


if __name__ == '__main__':
    result = audit()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(OUT), tokens=result['observed_construction_tokens'],
                         graph_barriers=result['target_graph_barriers']), indent=2))
