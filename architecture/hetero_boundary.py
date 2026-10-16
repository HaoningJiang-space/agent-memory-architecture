"""Supply break-even conditions in a reduced matrix/near-memory model.

Uses recorded legal batches, not API timestamps. This is a projection-only,
optimistic resource screen, NOT a cycle simulator, equal-area result or speedup
prediction. Compression ratios are controlled scenarios, not extrapolated
whole-model measurements. No old architecture/evidence file is modified.
"""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/architecture'


def traffic(m, k, n, ratio=1.0, partitions=16):
    """N-sharded output, full K local: no cross-domain partial reduction.

    Conservative input unicast to each shard. Every operator returns BF16 Y
    to the central domain; no fusion, multicast or free activation residency.
    Counts one weight read, a lower bound independent of array mapping.
    """
    if min(m, k, n, partitions) < 1 or ratio <= 0:
        raise ValueError('Positive dimensions and ratio required; encoded blocks may expand')
    q = min(partitions, n)
    raw = 2 * k * n
    boundary = 2 * m * (q * k + n)
    return dict(flops=2*m*k*n, raw_weight_bytes=raw,
                encoded_weight_bytes=raw*ratio, activation_boundary_bytes=boundary,
                weight_only_intensity=m/ratio,
                traffic_crossover_m=ratio*k*n/(q*k+n), partitions=q)


def n_shard_reference(x, w, partitions):
    """Different output columns, identical full-K dot products per shard."""
    if partitions < 1:
        raise ValueError('partitions must be positive')
    return np.concatenate([x @ part for part in np.array_split(w, min(partitions, w.shape[1]), axis=1)], axis=1)


def projection_streams():
    frozen = json.loads(gzip.decompress((OUT/'batch_closure_traces.json.gz').read_bytes()))
    trace = json.loads((ROOT/'evidence/phase2/workload_accounting.json').read_text())
    streams = []
    for run in frozen['runs']:
        if run['design'] != 'baseline':
            continue
        graph = next(g for g in frozen['graphs'] if g['name'] == run['graph'])
        nodes = {n['id']: n for n in graph['nodes']}
        counts = Counter()
        seen = []
        decode_tokens = 0

        def forward(cfg, model, phase, length, batch, repeats, embedding=False):
            h = cfg['hidden_size']
            q = cfg['num_attention_heads'] * cfg['head_dim']
            kv = cfg['num_key_value_heads'] * cfg['head_dim']
            f = cfg['intermediate_size']
            m = batch if phase == 'decode' else batch * length
            for op, k, n in [('q', h, q), ('k', h, kv), ('v', h, kv), ('o', q, h),
                             ('gate', h, f), ('up', h, f), ('down', f, h)]:
                counts[(model, phase, op, m, k, n)] += repeats*cfg['num_hidden_layers']
            if not embedding:
                # Last prompt position generates the first output, as in the
                # frozen runner. LM head stays small M even during prefill.
                counts[(model, phase, 'lm_head', batch, h, cfg['vocab_size'])] += repeats

        for b in run['batches']:
            cfg = trace['models'][b['model']]['config']
            seen.extend(b['node_ids'])
            if b['kind'] == 'embedding':
                node = nodes[b['node_ids'][0]]
                forward(cfg, b['model'], 'embedding', node['L'], node['B'], 1, True)
            else:
                calls = [nodes[i] for i in b['node_ids']]
                forward(cfg, b['model'], 'prefill', max(c['L'] for c in calls), len(calls), 1)
                for seg in b['segments']:
                    forward(cfg, b['model'], 'decode', seg['padded_prompt'], seg['batch'], seg['count'])
                    decode_tokens += seg['batch']*seg['count']
        expected = [n['id'] for n in graph['nodes'] if n['kind'] != 'host']
        if Counter(seen) != Counter(expected):
            raise ValueError('Missing or duplicate recorded neural node')
        expected_tokens = sum(n['O']-1 for n in graph['nodes'] if n['kind'] == 'llm')
        if decode_tokens != expected_tokens:
            raise ValueError('O-1 token conservation failed')
        rows = [dict(model=key[0], phase=key[1], op=key[2], m=key[3], k=key[4], n=key[5], count=count)
                for key, count in sorted(counts.items())]
        streams.append(dict(graph=run['graph'], maintenance_cap=run['maintenance_cap'],
                            neural_nodes=len(seen), decode_tokens=decode_tokens, rows=rows))
    return frozen['designs'][0]['hardware'], streams


def arrays(rows, ratio, partitions):
    metrics = [traffic(r['m'], r['k'], r['n'], ratio, partitions) for r in rows]
    return {key: np.array([r[key] for r in metrics], dtype=float) for key in metrics[0]} | {
        'count': np.array([r['count'] for r in rows], dtype=float)}


def services(data, peak, bandwidth, fraction, bandwidth_ratio, sink_ratio,
             overhead=0.03, link_ratio=1.0, launch_seconds=100e-9):
    """A resource lower-bound comparison; gamma is a required, unpriced rate.

    U gets all MACs, shared bandwidth and identical compression/sink ability.
    H loses overhead MACs, then divides the rest. Operators serialize: no free
    concurrent memory access. Near execution pays input/output before/after
    its compute/read phase. SRAM tiling, attention, queueing and PPA are absent.
    """
    if not 0 < fraction < 1 or not 0 <= overhead < 1:
        raise ValueError('Invalid split/overhead')
    if min(peak, bandwidth, bandwidth_ratio, sink_ratio, link_ratio) <= 0:
        raise ValueError('Rates must be positive')
    work, raw, coded = (data[k] for k in ('flops', 'raw_weight_bytes', 'encoded_weight_bytes'))
    # A common unfolded-weight sink includes decode output, RF delivery and
    # local ports. Even raw weights need the physical sink; no hidden bypass.
    sink = raw/(sink_ratio*bandwidth)
    unified = np.maximum.reduce([work/peak, coded/bandwidth, sink])
    available = (1-overhead)*peak
    central = np.maximum.reduce([work/((1-fraction)*available), coded/bandwidth, sink])
    near = np.maximum.reduce([work/(fraction*available), coded/(bandwidth_ratio*bandwidth), sink])
    near += data['activation_boundary_bytes']/(link_ratio*bandwidth) + launch_seconds
    return unified, np.minimum(central, near), near < central


def required_bandwidth(data, peak, bandwidth, fraction, sink_ratio,
                       overhead=0.03, target_ratio=1.0, link_ratio=1.0):
    count = data['count']

    def evaluate(gamma):
        u, h, offload = services(data, peak, bandwidth, fraction, gamma,
                                 sink_ratio, overhead, link_ratio)
        return float(u @ count), float(h @ count), offload

    baseline, equal_bw, _ = evaluate(1)
    _, ideal_local, _ = evaluate(math.inf)
    target = baseline*target_ratio
    # Strict threshold; equality at infinite bandwidth is not a finite solution.
    if ideal_local >= target:
        gamma = None
    else:
        lo, hi = 1.0, 1.0
        while evaluate(hi)[1] >= target and hi < 2**20:
            hi *= 2
        if evaluate(hi)[1] >= target:
            gamma = None
        else:
            for _ in range(60):
                mid = (lo+hi)/2
                if evaluate(mid)[1] >= target:
                    lo = mid
                else:
                    hi = mid
            gamma = hi
    return dict(fraction=fraction, sink_ratio=sink_ratio, overhead=overhead,
                link_ratio=link_ratio, target_ratio=target_ratio,
                unified_projection_seconds=baseline,
                same_bandwidth_projection_seconds=equal_bw,
                infinite_local_bandwidth_projection_seconds=ideal_local,
                required_local_bandwidth_ratio=gamma,
                required_local_bytes_s=None if gamma is None else gamma*bandwidth,
                near_flops_s=fraction*(1-overhead)*peak,
                sink_unfolded_bytes_s=sink_ratio*bandwidth)


def main():
    hw, streams = projection_streams()
    peak = 2*hw['arrays']*hw['rows']*hw['cols']*hw['frequency']
    bandwidth = hw['bandwidth']*hw['bandwidth_efficiency']
    result = dict(scope=__doc__, parameters=dict(peak_flops_s=peak, bandwidth_bytes_s=bandwidth,
        partitions=16, ratios=[1.0, 0.8], fractions=[0.1, 0.25, 0.5], sink_ratios=[1.0, 2.0, 4.0],
        overhead_fraction=0.03, link_ratio=1.0, launch_seconds=100e-9,
        ratio_provenance='0.8 is a controlled whole-stream assumption, not a measured whole-model ratio',
        omitted='SRAM capacity/tiling and ports beyond sink bound, attention, vector ops, DAG rescheduling, energy, physical area/cost, bank timing; no whole-agent speedup'), streams=[])
    for stream in streams:
        row = {k: v for k, v in stream.items() if k != 'rows'}
        row['m_work'] = {}
        for label, lo, hi in [('1',1,1), ('2-8',2,8), ('9-32',9,32), ('33-128',33,128), ('129+',129,math.inf)]:
            selected = [r for r in stream['rows'] if lo <= r['m'] <= hi]
            row['m_work'][label] = dict(flops=sum(2*r['m']*r['k']*r['n']*r['count'] for r in selected),
                one_pass_weight_bytes=sum(2*r['k']*r['n']*r['count'] for r in selected))
        row['scenarios'] = []
        for ratio in (1.0, 0.8):
            data = arrays(stream['rows'], ratio, 16)
            scans = [required_bandwidth(data, peak, bandwidth, a, d)
                     for a in (0.1,0.25,0.5) for d in (1.0,2.0,4.0)]
            controls = [required_bandwidth(data, peak, bandwidth, .1, 2.0, overhead=o, link_ratio=l)
                        for o,l in [(0,1),(.1,1),(.03,.25)]]
            row['scenarios'].append(dict(ratio=ratio, requirements=scans, controls=controls))
        result['streams'].append(row)
    shapes = dict(scope='Recorded legal group shapes; no new target-hardware timeline', streams=streams)
    (OUT/'hetero_projection_shapes.json').write_text(json.dumps(shapes, indent=2)+'\n')
    inputs = ['architecture/hetero_boundary.py','evidence/architecture/batch_closure_traces.json.gz',
              'evidence/phase2/workload_accounting.json','evidence/architecture/weight_codec_screen.json']
    result['input_sha256'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs}
    (OUT/'hetero_boundary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for s in result['streams']:
        print(s['graph'], s['maintenance_cap'], [(v['ratio'],v['requirements'][1]['required_local_bandwidth_ratio']) for v in s['scenarios']])


if __name__ == '__main__':
    main()
