"""Fixed-shape fluid service bounds, not a streaming Agent experiment.

Remove arrival barriers and queueing to ask whether overlap alone can rescue
the frozen private-domain configuration. No new hardware parameters are tuned.
The bounds do not predict tail latency, semantic freshness or energy.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from architecture import call_placement as placement, model
from architecture.family_replay import batch_capacity

ROOT = Path(__file__).resolve().parents[1]


def pooled_bound(rows):
    """U busy time <= sum_d a_d H_busy_time_d for ANY nonnegative job counts.

Each row is a fixed job type assigned to one H domain. This is a workload
capacity bound only: it assumes ideal preemption/time sharing, resident models
and no additional simultaneous KV requirement. It says nothing about deadlines.
"""
    alpha = {}
    for row in rows:
        if row['u_seconds'] <= 0 or row['h_seconds'] <= 0:
            raise ValueError('Positive service times required')
        domain = row['domain']
        alpha[domain] = max(alpha.get(domain, 0), row['u_seconds'] / row['h_seconds'])
    return dict(alpha=alpha, sum_alpha=sum(alpha.values()),
                pooled_capacity_covers_split=sum(alpha.values()) <= 1)


def bundle_frontier(rows):
    """Exact max of (a+xb)/max(c+xd,e+xf), x >= 0.

    One history build/maintenance bundle versus x copies of the recorded
    three-query bundle. Fractional x describes average offered work, not an
    executable partial query. Only endpoints and the domain-load crossing
    can maximize this ratio. Reuse changes neither outputs nor their shapes.
    """
    terms = {p: dict(U=0, C=0, N=0) for p in ['build_maintenance', 'three_queries']}
    for row in rows:
        p = 'three_queries' if row['stage'].startswith('query') else 'build_maintenance'
        terms[p]['U'] += row['u_seconds']
        terms[p][row['domain']] += row['h_seconds']
    a, b = terms.values()
    points = [dict(query_bundles_per_history=0, ratio=a['U']/max(a['C'], a['N'])),
              dict(query_bundles_per_history='infinite', ratio=b['U']/max(b['C'], b['N']))]
    if b['C'] != b['N']:
        x = (a['N']-a['C'])/(b['C']-b['N'])
        if x > 0:
            points.append(dict(query_bundles_per_history=x,
                ratio=(a['U']+x*b['U'])/max(a['C']+x*b['C'], a['N']+x*b['N'])))
    return dict(service_seconds=terms, extrema=points, max_ratio=max(p['ratio'] for p in points),
                scope='Ideal overlap resource ratio; not response time or sustainable SLO throughput.')


def main():
    source = ROOT / 'evidence/architecture'
    placement_path = source / 'call_placement.json.gz'
    graph_path = source / 'batch_closure_traces.json.gz'
    evidence = json.loads(gzip.decompress(placement_path.read_bytes()))
    frozen = json.loads(gzip.decompress(graph_path.read_bytes()))
    _, trace, _, capacity, _ = placement.load()
    weights = {row['model']: row['weight_bytes'] for row in capacity['models']}
    # Keep the published cheaper U reference, not a new per-workload winner.
    name = 'HBM3E4_a8_128x128_s32_k2_m8'
    u = model.Hardware(**evidence['unified_devices'][name])
    h = {d: model.Hardware(**cfg) for d, cfg in evidence['hetero_devices'].items()}
    assert u.cost_units < evidence['hetero_bill']['cost_units']
    residents = {'U': sum(weights.values()), 'N': weights[placement.M8],
                 'C': weights[placement.M32] + weights[placement.EMB]}
    summaries = []
    for graph in frozen['graphs']:
        if 'concurrent_queries' not in graph['name']:
            continue
        nodes = {n['id']: n for n in graph['nodes']}
        # Fixed legal grouping from the existing strong baseline. Both hardware
        # arms execute exactly these groups; never reuse device-specific timing.
        reference = next(r for r in frozen['runs'] if r['graph'] == graph['name'] and r['design'] == 'baseline')
        for attention_mode in ['frozen', 'free_attention']:
            rows, consumed = [], []
            for batch in reference['batches']:
                calls = [nodes[i] for i in batch['node_ids']]
                consumed.extend(batch['node_ids'])
                first = calls[0]
                domain = 'N' if first['model'] == placement.M8 else 'C'
                cfg = trace['models'][first['model']]['config']
                seconds, peaks = {}, {}
                for device, hw in [('U', u), (domain, h[domain])]:
                    if first['kind'] == 'llm':
                        cost = placement.phase_cost(hw, cfg, calls, attention_mode)
                        seconds[device] = cost['prefill']['seconds'] + cost['decode']['seconds']
                        peaks[device] = batch_capacity(cfg, calls, residents[device])
                    else:
                        assert len(calls) == 1
                        cost = placement.forward(hw, json.dumps(cfg, sort_keys=True),
                            first['L'], first['B'], False, True, attention_mode)
                        seconds[device] = cost['seconds']
                        peaks[device] = residents[device] + 2 * first['B'] * first['L'] * (
                            cfg['hidden_size'] + 2 * cfg['intermediate_size'])
                    if peaks[device] > hw.capacity_bytes:
                        raise ValueError('Batch does not fit fixed residency')
                stages = sorted({c['stage'] for c in calls})
                assert len(stages) == 1 or all(s.startswith('query') for s in stages)
                stage = stages[0] if len(stages) == 1 else 'query_mixed'
                rows.append(dict(nodes=batch['node_ids'], stage=stage, source_stages=stages, kind=first['kind'],
                    model=first['model'], batch=len(calls) if first['kind']=='llm' else first['B'],
                    domain=domain, u_seconds=seconds['U'], h_seconds=seconds[domain], peak_bytes=peaks))
            assert Counter(consumed) == Counter(n['id'] for n in graph['nodes'] if n['kind'] != 'host')
            totals = defaultdict(lambda: dict(u_seconds=0, C_seconds=0, N_seconds=0))
            for row in rows:
                total = totals[row['stage']]
                total['u_seconds'] += row['u_seconds']
                total[row['domain'] + '_seconds'] += row['h_seconds']
            bound = pooled_bound(rows)
            bound['witnesses'] = {d: max((r for r in rows if r['domain']==d),
                key=lambda r:r['u_seconds']/r['h_seconds']) for d in bound['alpha']}
            total_u = sum(r['u_seconds'] for r in rows)
            total_h = {d: sum(r['h_seconds'] for r in rows if r['domain']==d) for d in h}
            summaries.append(dict(graph=graph['name'], attention_mode=attention_mode,
                bound=bound, bundle_frontier=bundle_frontier(rows), stage_service_seconds=dict(totals), rows=rows,
                same_library_u_busy_seconds=total_u, same_library_h_busy_seconds=total_h,
                ideal_overlap_speed_ratio=total_u/max(total_h.values())))
    result = dict(scope='Fixed observed batch-shape library; fluid resource envelope, not application replay or SLO result.',
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [placement_path, graph_path, Path(__file__),
                                ROOT/'architecture/call_placement.py', ROOT/'architecture/spec.json']},
        unified_device=name, unified_cost=u.cost_units, split_cost=evidence['hetero_bill']['cost_units'],
        summaries=summaries,
        limits=['No fresh-state semantics, arrival distribution, tail latency or queueing prediction.',
                'No claim of dynamic batching optimum; bound only covers the fixed legal groups in this library.',
                'Both sides omit host, interface, cold load and switching; H overlaps domains freely.',
                'U is ideal pooled/time-shared; simultaneous KV residency and preemption cost are not established.',
                'Whole-call fixed 8B -> N, 32B/embedding -> C; no conclusion about arbitrary PD/shared banks.',
                'Old LightMem flat trace is deliberately retained as frozen control, not complete LightMem.'])
    (source/'service_envelope.json').write_text(json.dumps(result, indent=2)+'\n')
    for row in summaries:
        print(row['graph'], row['attention_mode'], row['bound']['alpha'],
              'sum', row['bound']['sum_alpha'], 'fixed-mix ratio', row['ideal_overlap_speed_ratio'], flush=True)
        print('max over query/build ratios:', row['bundle_frontier']['max_ratio'])


if __name__ == '__main__':
    main()
