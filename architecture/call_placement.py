"""Bounded private-memory call/phase placement, NOT an H2-LLM reproduction.

Frozen finite DAG; complete model copies; same paid K2 and static ready batching.
The joint choice is clairvoyant greedy (observed O), not globally optimal and
not a new scheduler.
No changes to the frozen HB, bank, or shared-memory operator experiments.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import argparse
import gzip
import hashlib
import heapq
import json
from pathlib import Path

from architecture import model, mechanism, small_batch, supply_path
from architecture.explore import load
from architecture.family_replay import batch_capacity, segments

ROOT = Path(__file__).resolve().parents[1]
M8 = 'Qwen/Qwen3-8B'
M32 = 'Qwen/Qwen3-32B'
EMB = 'Qwen/Qwen3-Embedding-0.6B'


def hardware(spec, memory, packages, arrays, sram):
    config = deepcopy(spec)
    config.update(memory_options=[[memory, packages]], array_counts=[arrays],
                  array_shapes=[[128, 128]], sram_mib=[sram])
    raw, rejected = model.candidates(config)
    if not raw:
        raise ValueError(rejected)
    hw, _, bad = mechanism.provision(raw[0], mechanism.KShard(groups=2), spec)
    hw, _, more = small_batch.provision(hw, spec)
    if bad or more:
        raise ValueError(bad + more)
    return hw


def resource_bill(devices, spec, link_area=0, link_w=0, link_gbs=32):
    fields = dict(area_mm2='area_mm2', cost_units='cost_units',
                  power_w='peak_power_w', shoreline_mm='shoreline_mm')
    values = {k: sum(getattr(h, attr) for h in devices.values()) for k, attr in fields.items()}
    headroom = {k: spec['budget'][k] - v for k, v in values.items()}
    values['area_mm2'] += link_area
    values['cost_units'] += link_area * spec['logic_assumptions']['cost_per_mm2']
    # link_w is always-on power; separately reserve full transfer toggling.
    values['power_w'] += link_w + (link_gbs*1e9*8*5e-12 if len(devices)>1 else 0)
    return dict(**values, before_link_headroom=headroom,
                violations=[k for k, v in values.items() if v > spec['budget'][k]])


@lru_cache(maxsize=30000)
def forward(hw, config_json, length, batch, decode, embedding, attention_mode):
    cfg = json.loads(config_json)
    result = small_batch.forward(hw, cfg, length, batch, decode, 'baseline', embedding=embedding)
    if attention_mode == 'free_attention':
        # Deliberately optimistic common control; NOT an implemented GQA kernel.
        att = supply_path.scale(small_batch.attention(hw, length, cfg['num_attention_heads'],
            cfg['num_key_value_heads'], cfg['head_dim'], batch, decode), cfg['num_hidden_layers'])
        result = {k: max(0, v - att.get(k, 0)) for k, v in result.items()}
    elif attention_mode != 'frozen':
        raise ValueError(attention_mode)
    return result


def phase_cost(hw, cfg, calls, attention_mode='frozen'):
    config_json = json.dumps(cfg, sort_keys=True)
    pre = forward(hw, config_json, max(c['L'] for c in calls), len(calls),
                  False, False, attention_mode)
    dec = supply_path.add()
    ends = {c['id']: 0.0 for c in calls if c['O'] == 1}
    parts = segments(calls)
    for p in parts:
        row = forward(hw, config_json, p['padded_prompt'] + (p['start'] + p['end'] + 1) // 2,
                      p['batch'], True, False, attention_mode)
        dec = supply_path.add(dec, supply_path.scale(row, p['count']))
        ends.update({c['id']: dec['seconds'] for c in calls if c['O'] - 1 == p['end']})
    return dict(prefill=pre, decode=dec, decode_ends=ends, segments=parts)


def capacities(cfg, calls, resident_c, resident_n):
    """Whole-call and split peaks. Source and destination coexist during copy."""
    kv = model.decoder_shape(cfg)['kv_bytes_per_token']
    length, batch = max(c['L'] for c in calls), len(calls)
    activation = 2 * batch * length * (cfg['hidden_size'] + 2 * cfg['intermediate_size'])
    prompt_kv = kv * batch * length
    # Transfer only continuing rows, retaining the padded prompt layout.
    copied = kv * sum(c['O'] > 1 for c in calls) * length
    decode_peak = max([copied] + [kv * p['batch'] * (p['padded_prompt'] + p['end']) +
        2 * p['batch'] * (cfg['hidden_size'] + 2 * cfg['intermediate_size']) for p in segments(calls)])
    return dict(C=batch_capacity(cfg, calls, resident_c),
                N=batch_capacity(cfg, calls, resident_n),
                PD_C=resident_c + prompt_kv + activation,
                PD_N=resident_n + decode_peak, kv_transfer_bytes=copied)


def dynamic(row, hw):
    value = row['energy_j'] - hw.static_w * row['seconds']
    if value < -1e-7:
        raise AssertionError('Negative dynamic energy')
    return max(0, value)


def replay(dag, devices, residency, trace, weights, policy, *, link_gbs=32,
           attention_mode='frozen', free_kv_time=False, host_scale=1, link_w=8,
           active_only=False):
    """Conservative nonpreemptive resources; fixed complete model residency.

    Whole-call routes reserve one domain. PD reserves C until KV handoff and N
    from handoff through decode. Thus no uncharged DRAM/DMA overlap. An explicit
    zero-KV-time control favors PD; this is not Splitwise's implemented pipeline.
    CPU may progress on early row completion, accelerator batch cannot refill.
    """
    assert policy in ('unified', 'whole', 'pd', 'joint')
    assert not active_only or policy == 'unified'
    nodes = {n['id']: n for n in dag['nodes']}
    order = {n['id']: i for i, n in enumerate(dag['nodes'])}
    resident = {d: sum(weights[m] for m in ms) for d, ms in residency.items()}
    if active_only:
        resident = {'C': 0}
    for d, size in resident.items():
        if size > devices[d].capacity_bytes:
            raise ValueError('Resident weights exceed domain capacity')
    # All model copies are loaded once from a common host DRAM, serially. The
    # 32 GB/s host link, 5 pJ/bit and 20 W CPU are frozen common controls.
    initial_bytes = sum(resident.values())
    initial = sum(resident[d] / min(32e9, devices[d].bandwidth * devices[d].bandwidth_efficiency)
                  for d in devices)
    energy_dynamic = sum(resident[d] * 8 * (5 + devices[d].write_pj_bit) * 1e-12 for d in devices)
    free = {d: initial for d in devices}
    cpu_free = initial
    now = initial
    pending, done, starts, events = set(nodes), {}, {}, []
    batches, reservations = [], {d: [] for d in devices}
    peak = dict(resident)
    model_copies = Counter(m for ms in residency.values() for m in ms)
    cached_model = None

    def ready(kinds):
        return sorted((nodes[i] for i in pending if nodes[i]['kind'] in kinds and
                       all(d in done for d in nodes[i]['deps'])), key=lambda n: order[n['id']])

    def options(calls):
        name = calls[0]['model']
        cfg = trace['models'][name]['config']
        cap = capacities(cfg, calls, resident['C'], resident.get('N', 0))
        costs = {}
        routes = []
        for d, hw in devices.items():
            if name in residency[d] and cap[d] <= hw.capacity_bytes:
                costs[d] = phase_cost(hw, cfg, calls, attention_mode)
        permitted = ['C'] if policy == 'unified' else ['C', 'N']
        if policy == 'pd' and name == M8 and cap['kv_transfer_bytes']:
            permitted = []
        for d in permitted:
            if d not in costs:
                continue
            p = costs[d]
            begin = max(now, free[d])
            pre_end = begin + p['prefill']['seconds']
            end = pre_end + p['decode']['seconds']
            routes.append(dict(route=d, begin=begin, end=end,
                intervals={d: (begin, end)}, peaks={d: cap[d]},
                completion={c['id']: pre_end + p['decode_ends'][c['id']] for c in calls},
                prefill_seconds=p['prefill']['seconds'], decode_seconds=p['decode']['seconds'],
                kv_bytes=0, kv_seconds=0,
                dynamic_j=dynamic(p['prefill'], devices[d]) + dynamic(p['decode'], devices[d]),
                segment_batches=[(p['batch'], p['count']) for p in p['segments']]))
        if cap['kv_transfer_bytes'] and policy in ('pd', 'joint') and 'N' in devices and all(name in residency[d] for d in devices):
            if cap['PD_C'] <= devices['C'].capacity_bytes and cap['PD_N'] <= devices['N'].capacity_bytes:
                c, n = devices['C'], devices['N']
                pc = costs.get('C') or phase_cost(c, cfg, calls, attention_mode)
                pn = costs.get('N') or phase_cost(n, cfg, calls, attention_mode)
                begin = max(now, free['C'])
                pre_end = begin + pc['prefill']['seconds']
                byte_count = cap['kv_transfer_bytes']
                transfer = 0 if free_kv_time or not byte_count else byte_count / min(
                    link_gbs * 1e9, c.bandwidth * c.bandwidth_efficiency, n.bandwidth * n.bandwidth_efficiency) + 1e-6
                handoff = max(pre_end, free['N'])
                decode_begin = handoff + transfer
                end = decode_begin + pn['decode']['seconds']
                routes.append(dict(route='PD', begin=begin, end=end,
                    intervals={'C': (begin, decode_begin), 'N': (handoff, end)},
                    peaks={'C': cap['PD_C'], 'N': cap['PD_N']},
                    completion={c0['id']: pre_end if c0['O'] == 1 else decode_begin + pn['decode_ends'][c0['id']]
                                for c0 in calls},
                    prefill_seconds=pc['prefill']['seconds'], decode_seconds=pn['decode']['seconds'],
                    kv_bytes=byte_count, kv_seconds=transfer,
                    dynamic_j=dynamic(pc['prefill'], c) + dynamic(pn['decode'], n) +
                        byte_count * 8 * (c.read_pj_bit + n.write_pj_bit + 5) * 1e-12,
                    segment_batches=[(p['batch'], p['count']) for p in pn['segments']]))
        return routes

    while pending or events:
        while events and events[0][0] <= now + 1e-10:
            t, identity = heapq.heappop(events)
            done[identity] = t
        progressed = False
        if cpu_free <= now + 1e-10:
            rr = ready({'host'})
            if rr:
                n = rr[0]
                pending.remove(n['id'])
                starts[n['id']] = now
                cpu_free = now + n['seconds'] * host_scale
                heapq.heappush(events, (cpu_free, n['id']))
                progressed = True
                if cpu_free == now:
                    continue
        rr = ready({'llm', 'embedding'})
        inspected = set()
        for first in rr:
            group = (first['kind'], first['model'])
            if group in inspected:
                continue
            inspected.add(group)
            name = first['model']
            cfg = trace['models'][name]['config']
            if active_only:
                # A second strong unified control: evict the previous complete
                # checkpoint after its batch ends; every reload is charged.
                resident['C'] = weights[name]
            if first['kind'] == 'embedding':
                routes = []
                for d, hw in devices.items():
                    needed = resident[d] + 2 * first['B'] * first['L'] * (cfg['hidden_size'] + 2 * cfg['intermediate_size'])
                    if name not in residency[d] or needed > hw.capacity_bytes:
                        continue
                    r = forward(hw, json.dumps(cfg, sort_keys=True), first['L'], first['B'],
                                False, True, attention_mode)
                    begin = max(now, free[d])
                    end = begin + r['seconds']
                    routes.append(dict(route=d, begin=begin, end=end, intervals={d: (begin, end)},
                        peaks={d: needed}, completion={first['id']: end},
                        prefill_seconds=r['seconds'], decode_seconds=0, kv_bytes=0, kv_seconds=0,
                        dynamic_j=dynamic(r, hw), segment_batches=[]))
                selected = [first]
            else:
                compatible = [n for n in rr if n['kind'] == 'llm' and n['model'] == name]
                # Maximal ready prefix that any allowed route can hold. Never
                # cap M at eight to favor a special path; use fallback above M8.
                selected, routes = [], []
                for n in compatible:
                    trial = options(selected + [n])
                    if not trial:
                        break
                    selected.append(n)
                    routes = trial
            if not routes:
                raise ValueError(f'No capacity-feasible route for {first["id"]}')
            # Clairvoyant local choice; not a globally optimal scheduling bound.
            pick = min(routes, key=lambda r: (r['end'], r['dynamic_j']))
            if pick['begin'] <= now + 1e-10:
                pick['reload_bytes'] = 0
                pick['reload_seconds'] = 0
                if active_only and cached_model != name:
                    h = devices['C']
                    size = weights[name]
                    seconds = size / min(32e9, h.bandwidth * h.bandwidth_efficiency)
                    pick['reload_bytes'], pick['reload_seconds'] = size, seconds
                    pick['end'] += seconds
                    pick['completion'] = {i: t+seconds for i,t in pick['completion'].items()}
                    pick['intervals']['C'] = (pick['begin'], pick['end'])
                    pick['dynamic_j'] += size*8*(5+h.write_pj_bit)*1e-12
                    cached_model = name
                for d, (a, b) in pick['intervals'].items():
                    if b > a:
                        reservations[d].append((a, b))
                    free[d] = b
                    peak[d] = max(peak[d], pick['peaks'][d])
                for n in selected:
                    pending.remove(n['id'])
                    starts[n['id']] = pick['begin']
                    heapq.heappush(events, (pick['completion'][n['id']], n['id']))
                energy_dynamic += pick['dynamic_j']
                batches.append(dict(model=name, kind=first['kind'], stage=first['stage'],
                    nodes=[n['id'] for n in selected], batch=len(selected) if first['kind']=='llm' else first['B'],
                    alternatives=[{k: r[k] for k in ('route', 'end', 'prefill_seconds', 'decode_seconds', 'kv_bytes', 'kv_seconds')}
                                  for r in routes], **pick))
                progressed = True
                break
        if progressed:
            continue
        times = [t for t in [cpu_free, *free.values(), events[0][0] if events else float('inf')]
                 if t > now + 1e-10 and t < float('inf')]
        if not times and pending:
            raise RuntimeError('DAG deadlock')
        if times:
            now = min(times)
    wall = max([cpu_free, *free.values(), *done.values()])
    # Static power is charged exactly once, including idle/wait/cold loading.
    static = wall * (20 + sum(h.static_w for h in devices.values()) + (link_w if len(devices)>1 else 0))
    result = dict(seconds=wall, energy_j=static+energy_dynamic, dynamic_j=energy_dynamic,
        static_j=static, policy=policy, attention_mode=attention_mode, free_kv_time=free_kv_time,
        initial_load_seconds=initial, initial_load_bytes=initial_bytes, model_copies=dict(model_copies),
        initial_model_copies={} if active_only else dict(model_copies),
        residency=residency, active_only=active_only, peak_bytes=peak, reservations=reservations,
        reserved_seconds={d: sum(b-a for a,b in rows) for d,rows in reservations.items()},
        batches=batches, starts=starts, completions=done,
        query_completion_seconds=[done[n] for n in dag['query_ends']])
    validate(dag, result, devices)
    return result


def validate(dag, result, devices):
    nodes = {n['id']: n for n in dag['nodes']}
    assert set(result['completions']) == set(nodes) == set(result['starts'])
    used = [i for b in result['batches'] for i in b['nodes']]
    assert Counter(used) == Counter(n['id'] for n in dag['nodes'] if n['kind'] != 'host')
    for n in dag['nodes']:
        for dep in n['deps']:
            assert result['completions'][dep] <= result['starts'][n['id']] + 1e-8
    for d, rows in result['reservations'].items():
        rows = sorted(rows)
        assert all(a <= b for a,b in rows)
        assert all(b <= c+1e-8 for (_,b),(c,_) in zip(rows, rows[1:]))
        assert result['peak_bytes'][d] <= devices[d].capacity_bytes
    for b in result['batches']:
        if b['kind'] == 'llm':
            assert sum(m * steps for m,steps in b['segment_batches']) == sum(nodes[i]['O']-1 for i in b['nodes'])
            assert all(n in result['residency'][d] for d in b['intervals'] for n in [b['model']])


def compact_evidence(runs, unified, hetero_cost):
    """Keep all scalar results, detailed traces for controls and winning U points.

    Every run has already passed validate(). Omitted traces are reproducible
    and identified by a digest; avoid committing hundreds of redundant DAGs.
    """
    keep = set()
    for graph, mode in {(r['graph'], r['attention_mode']) for r in runs}:
        rows = [r for r in runs if r['graph']==graph and r['attention_mode']==mode and r['policy']=='unified']
        cheap = [r for r in rows if unified[r['design']]['C'].cost_units <= hetero_cost]
        for candidates, key in [(rows, 'seconds'), (rows, 'energy_j'), (cheap, 'seconds')]:
            keep.add(id(min(candidates, key=lambda r:r[key])))
    for r in runs:
        r['detail_retained'] = r['policy']!='unified' or id(r) in keep
        detail = {k:r[k] for k in ('batches','starts','completions','reservations')}
        r['detail_sha256'] = hashlib.sha256(json.dumps(detail,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        r['node_count'], r['batch_count'] = len(r['completions']), len(r['batches'])
        if not r['detail_retained']:
            for k in detail:
                del r[k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='evidence/architecture/call_placement.json.gz')
    args = parser.parse_args()
    spec, trace, _, capacity, _ = load()
    weights = {r['model']: r['weight_bytes'] for r in capacity['models']}
    frozen = json.loads(gzip.decompress((ROOT/'evidence/architecture/batch_closure_traces.json.gz').read_bytes()))
    unified = {}
    for raw in model.candidates(spec)[0]:
        hw, _, bad = mechanism.provision(raw, mechanism.KShard(groups=2), spec)
        hw, _, more = small_batch.provision(hw, spec)
        if not bad and not more and sum(weights.values()) < hw.capacity_bytes:
            unified[hw.name] = {'C': hw}
    hetero = dict(C=hardware(spec, 'LPDDR5X', 8, 16, 64), N=hardware(spec, 'HBM3E', 1, 8, 32))
    bill = resource_bill(hetero, spec, link_area=8, link_w=8)
    assert not bill['violations']
    runs = []
    for dag in frozen['graphs']:
        for label, devices in unified.items():
            for active_only in (False, True):
                r = replay(dag, devices, {'C': list(weights)}, trace, weights, 'unified', active_only=active_only)
                runs.append(dict(graph=dag['name'], design=label, **r))
        for attention_mode in ('frozen', 'free_attention'):
            if attention_mode == 'free_attention':
                # Re-evaluate the whole unified set; do not select its winner
                # under the weaker attention mapping and freeze that choice.
                for label, devices in unified.items():
                    for active_only in (False, True):
                        r = replay(dag, devices, {'C': list(weights)}, trace, weights, 'unified',
                                   attention_mode=attention_mode, active_only=active_only)
                        runs.append(dict(graph=dag['name'], design=label, **r))
            for policy in ('whole', 'pd', 'joint'):
                residency = {'C': [M32, EMB] + ([M8] if policy != 'whole' else []), 'N': [M8]}
                for free_kv in ([False, True] if policy in ('pd', 'joint') else [False]):
                    r = replay(dag, hetero, residency, trace, weights, policy,
                               attention_mode=attention_mode, free_kv_time=free_kv)
                    runs.append(dict(graph=dag['name'], design='private_LPDDR8_HBM1', **r))
        print(dag['name'], 'complete', flush=True)
    compact_evidence(runs, unified, bill['cost_units'])
    inputs = ['architecture/spec.json', 'evidence/architecture/batch_closure_traces.json.gz',
              'evidence/phase2/workload_accounting.json', 'evidence/calibration/model_capacity.json']
    code = ['architecture/'+n+'.py' for n in ('call_placement', 'model', 'mechanism', 'small_batch', 'supply_path', 'family_replay', 'explore')]
    result = dict(scope='Finite offline greedy private-domain placement; conditional model, no new hardware/scheduler claim.',
        input_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs+code},
        budget=spec['budget'], hetero_bill=bill,
        hetero_devices={d: asdict(h) for d,h in hetero.items()},
        unified_devices={k: asdict(v['C']) for k,v in unified.items()}, runs=runs,
        limits=['Observed output lengths used for route choice and capacity: offline oracle.',
                'Fixed full-weight residency plus unified active-only eviction control; no prefix reuse/continuous batch refill.',
                'PD serializes source and destination DMA; zero-copy-time control favors overlap.',
                'Attention retains frozen mapping; zero-attention common control is not a real kernel.',
                'Vector/softmax/sampling/runtime work absent from frozen tensor model.',
                'Link 32 GB/s, 8 mm2, 8 W, 5 pJ/bit are assumptions; no PHY/floorplan signoff.',
                'No shared-bank H2 execution baseline: distinct topology, existing operator study remains separate.',
                'LightMem compressor missing and stale-vector upstream behavior preserved.',
                'Host anchors from finite recorded trace; not production arrivals or agent quality.'])
    path = ROOT/args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(result, separators=(',', ':')).encode(), mtime=0))
    print('wrote', path, 'runs', len(runs), flush=True)


if __name__ == '__main__':
    main()
