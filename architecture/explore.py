"""Run the first physical-constraint exploration without application-success gates.

Real requests remain real local workloads; shortened query lengths and scaled
build/maintenance mixes are counterfactual scenarios, never complete Mem0 runs.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from architecture.model import GIB, add, candidates, decoder, hardware_dict
from workload_accounting import decoder_shape

ROOT = Path(__file__).resolve().parents[1]


def average(records):
    records = list(records)
    return {k:statistics.mean(r[k] for r in records) for k in records[0]}


def break_even(build, saved_per_query, maintenance_per_query=0):
    net = saved_per_query-maintenance_per_query
    return build/net if net>0 else None


def lifecycle(build, query, reuse, updates_per_query=0, update=0):
    return build + reuse*(query+updates_per_query*update)


def cold_load(extra_bytes):
    """Incremental construction weights, 32 GB/s host link, 5 pJ/bit scenario.

Main/embedding initial loads are common to both strategies and excluded.
Online models remain resident; offline 8B is unloaded after one build.
"""
    seconds = extra_bytes/32e9
    return dict(seconds=seconds,energy_j=extra_bytes*8*5e-12+30*seconds)


def load():
    paths = ['architecture/spec.json','evidence/phase2/workload_accounting.json',
             'evidence/architecture/embedding_shapes.json','evidence/calibration/model_capacity.json']
    data = [json.loads((ROOT/p).read_text()) for p in paths]
    return (*data, {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                    paths+['architecture/model.py','architecture/explore.py']})


def profile(hw, trace, embeddings):
    groups = {}
    for name, group in trace['groups'].items():
        rows = []
        for call in group['calls']:
            result = decoder(hw,trace['models'][call['model']]['config'],
                             call['prompt_tokens'],call['completion_tokens'])
            rows.append(dict(trace_id=call['trace_id'],prompt_tokens=call['prompt_tokens'],
                             completion_tokens=call['completion_tokens'],**result))
        groups[name] = dict(calls=rows,mean=average(r['total'] for r in rows),
                           mean_prefill=average(r['prefill'] for r in rows),
                           mean_decode=average(r['decode'] for r in rows))
    config = trace['models']['Qwen/Qwen3-Embedding-0.6B']['config']
    totals = defaultdict(list)
    for call in embeddings['calls']:
        context = call['context']
        name = context['run']+'_'+context['phase']
        parts = []
        for batch in call['batches']:
            parts.append(decoder(hw,config,batch['padded_length'],1,embedding=True,batch=batch['batch'])['total'])
        totals[name].append(add(*parts))
    for name, rows in totals.items():
        groups['embedding_'+name] = dict(calls=len(rows),total=add(*rows),mean=average(rows))
    return groups


def scenario_costs(hw, trace, groups, weights, length_cap, residency, cache_credit=0,
                   workspace_base_gib=4, workspace_chunk=None):
    """A shape-conditioned scenario; no quality-valid semantic point is implied.

The nine observed extraction/update pairs define one write-shape distribution.
We never say these nine chunks built the full history used by the 60 QA calls.
"""
    config = trace['models']['Qwen/Qwen3-32B']['config']
    original = trace['groups']['embed_full_01_query']['calls']
    if length_cap is None:
        query_results = groups['embed_full_01_query']['calls']
    else:
        query_results = [decoder(hw,config,min(c['prompt_tokens'],length_cap),c['completion_tokens'])
                         for c in original]
    # Optimistic software control: credit a fraction of ALL prefill service,
    # including attention, but do not reduce the retained KV capacity. This
    # deliberately favors reuse; it is not a prefix-cache implementation.
    query = average(add({k:v*(1-cache_credit) for k,v in r['prefill'].items()},r['decode'])
                    for r in query_results)
    peak_kv = max(r['kv_bytes'] for r in query_results)
    query_workspace = max(r['workspace_bytes'] for r in query_results)
    query_workspace += (workspace_base_gib-4)*GIB
    if workspace_chunk:
        h, ff = config['hidden_size'],config['intermediate_size']
        query_workspace = workspace_base_gib*GIB+2*workspace_chunk*(h+2*ff)
    write = add(groups['mem0_extract_prefix']['mean'],groups['mem0_update_prefix']['mean'])
    emb_write = groups.get('embedding_mem0_full_01_construction',{}).get('total',add())
    write = add(write,{k:v/9 for k,v in emb_write.items()})
    base_embedding = groups.get('embedding_embed_full_01_construction',{}).get('total',add())
    query_embedding = groups.get('embedding_embed_full_01_query',{}).get('mean',add())
    query = add(query,query_embedding)
    resident = weights['Qwen/Qwen3-32B']+weights['Qwen/Qwen3-Embedding-0.6B']
    construction_kv = max(r['kv_bytes'] for name in ('mem0_extract_prefix','mem0_update_prefix')
                          for r in groups[name]['calls'])
    extra = weights['Qwen/Qwen3-8B'] if length_cap is not None and residency=='online' else 0
    overlap_kv = construction_kv if extra else 0
    # Sequential execution shares a workspace. In online overlap the write
    # branch receives an additional peak workspace; no free coexistence.
    extra_workspace = max(r['workspace_bytes'] for name in ('mem0_extract_prefix','mem0_update_prefix')
                          for r in groups[name]['calls']) if extra else 0
    if extra:
        extra_workspace += (workspace_base_gib-4)*GIB
        if workspace_chunk:
            cfg8=trace['models']['Qwen/Qwen3-8B']['config']
            extra_workspace=workspace_base_gib*GIB+2*workspace_chunk*(cfg8['hidden_size']+2*cfg8['intermediate_size'])
    available = hw.capacity_bytes-resident-extra-query_workspace-overlap_kv-extra_workspace
    slots = max(0,math.floor(available/peak_kv))
    build_resident = weights['Qwen/Qwen3-8B']+weights['Qwen/Qwen3-Embedding-0.6B']
    build_peak = build_resident+construction_kv+max(r['workspace_bytes']
                    for name in ('mem0_extract_prefix','mem0_update_prefix') for r in groups[name]['calls'])
    build_fits = build_peak<=hw.capacity_bytes
    return dict(query=query,write_unit=write,base_embedding_build=base_embedding,
                raw_write_unit={k:v/88 for k,v in base_embedding.items()},
                peak_query_kv_bytes=peak_kv,query_workspace_bytes=query_workspace,
                extra_online_weight_bytes=extra,overlap_construction_kv_bytes=overlap_kv,
                extra_workspace_bytes=extra_workspace,max_active_slots=slots,
                build_fits=build_fits,resident_base_weights=resident)


def select(rows, metric):
    return min(rows,key=lambda r:(r[metric],r['hardware'],r['strategy'])) if rows else None


def run(output):
    spec, trace, embeddings, capacity, receipts = load()
    weights = {r['model']:r['weight_bytes'] for r in capacity['models']}
    hardware, rejected = candidates(spec)
    profiles, all_calls = {}, []
    start = time.time()
    for index, hw in enumerate(hardware):
        groups = profile(hw,trace,embeddings)
        profiles[hw.name] = groups
        for name, g in groups.items():
            all_calls.append(dict(hardware=hw.name,group=name,**{k:v for k,v in g.items() if k!='calls'}))
        print(f'profile {index+1}/{len(hardware)} {hw.name}',flush=True)
    # A finite controlled request-mix grid. Update and reuse axes are independent
    # of active KV slots. No replicated samples are called new observations.
    caps = [None,2048,4096,8192,16384]
    costs = {}
    for hw in hardware:
        for cap in caps:
            for residency in ('offline','online'):
                costs[hw.name,cap,residency] = scenario_costs(hw,trace,profiles[hw.name],weights,cap,residency)
    scenarios, crossovers = [], []
    for hw in hardware:
        for cap in caps[1:]:
            for residency in ('offline','online'):
                raw = costs[hw.name,None,residency]
                memory = costs[hw.name,cap,residency]
                for metric in ('seconds','energy_j'):
                    # An 88-write scenario scales the observed nine-write
                    # distribution; it is NOT observed full-build work.
                    build = (88*memory['write_unit'][metric]+cold_load(weights['Qwen/Qwen3-8B'])[metric]
                             -raw['base_embedding_build'][metric])
                    crossovers.append(dict(hardware=hw.name,length_cap=cap,residency=residency,metric=metric,
                        delta_build=build,query_saved=raw['query'][metric]-memory['query'][metric],
                        reuse_crossing=break_even(build,raw['query'][metric]-memory['query'][metric]),
                        raw_slots=raw['max_active_slots'],memory_slots=memory['max_active_slots']))
    ref = next(h.name for h in hardware if h.memory=='HBM3E' and h.arrays==16 and h.rows==128 and h.sram_mib==32)
    for reuse in (8,32,128,512,2048):
        for active in (1,4,8,16):
            for updates in (0,.01,.1):
                for residency in ('offline','online'):
                    if residency=='offline' and updates: continue
                    rows = []
                    for hw in hardware:
                        for cap in caps:
                            c = costs[hw.name,cap,residency]
                            if c['max_active_slots']<active or (cap and not c['build_fits']): continue
                            build = c['base_embedding_build'] if cap is None else {k:88*v for k,v in c['write_unit'].items()}
                            row = dict(hardware=hw.name,strategy='raw' if cap is None else f'cap_{cap}',
                                       reuse=reuse,active=active,updates_per_query=updates,residency=residency,
                                       slots=c['max_active_slots'],cost_units=hw.cost_units)
                            for metric in ('seconds','energy_j'):
                                setup = cold_load(weights['Qwen/Qwen3-8B'])[metric] if cap else 0
                                write_unit = c['write_unit'] if cap else c['raw_write_unit']
                                row[metric] = lifecycle(build[metric]+setup,c['query'][metric],reuse,
                                                       updates,write_unit[metric])
                            rows.append(row)
                    for metric in ('seconds','energy_j'):
                        joint = select(rows,metric)
                        fixed_h = select([r for r in rows if r['hardware']==ref],metric)
                        fixed_raw = select([r for r in rows if r['strategy']=='raw'],metric)
                        frozen_policy = select([r for r in rows if fixed_h and r['strategy']==fixed_h['strategy']],metric)
                        scenarios.append(dict(reuse=reuse,active=active,updates_per_query=updates,residency=residency,
                            metric=metric,joint=joint,fixed_hardware=fixed_h,fixed_raw_policy=fixed_raw,
                            hardware_after_policy_selection=frozen_policy,
                            frozen_policy_penalty=None if not frozen_policy or not joint else frozen_policy[metric]/joint[metric]))
    # Controls: never claim arbitrary pool splitting helps. Ideal P/D picks
    # the faster physical device per phase, before charging replication/link.
    # This is an *unbudgeted optimistic ceiling*, not a Splitwise reproduction.
    pd = []
    for group in trace['groups']:
        p = min(hardware,key=lambda h:profiles[h.name][group]['mean_prefill']['seconds'])
        d = min(hardware,key=lambda h:profiles[h.name][group]['mean_decode']['seconds'])
        best = min(hardware,key=lambda h:profiles[h.name][group]['mean']['seconds'])
        pd.append(dict(group=group,prefill_hardware=p.name,decode_hardware=d.name,
                       unbudgeted_phase_oracle_seconds=profiles[p.name][group]['mean_prefill']['seconds']+profiles[d.name][group]['mean_decode']['seconds'],
                       best_unified_seconds=profiles[best.name][group]['mean']['seconds'],best_unified=best.name,
                       qualification='Ideal two-device ceiling; replication and KV movement omitted. No budgeted P/D victory claimed.'))
    # Budgeted P/D local-request comparator. Both devices hold the same model;
    # transfer full prompt KV once, charge source read and destination write.
    # Serialized service, no synthetic queueing/overlap speedup.
    budgeted_pd = []
    for group, workload in trace['groups'].items():
        model = workload['calls'][0]['model']
        config = trace['models'][model]['config']
        kv_per_token = decoder_shape(config)['kv_bytes_per_token']
        mean_transfer = statistics.mean(c['prompt_tokens'] for c in workload['calls'])*kv_per_token
        options = []
        for p in hardware:
            for d in hardware:
                if (p.cost_units+d.cost_units>spec['budget']['cost_units'] or
                    p.area_mm2+d.area_mm2>spec['budget']['area_mm2'] or
                    p.peak_power_w+d.peak_power_w>spec['budget']['power_w']): continue
                def fits(h):
                    return all(weights[model]+r['kv_bytes']+r['workspace_bytes']<=h.capacity_bytes
                               for r in profiles[h.name][group]['calls'])
                if not fits(p) or not fits(d): continue
                transfer_time = mean_transfer/min(64e9,p.bandwidth*.8,d.bandwidth*.8)
                transfer_energy = mean_transfer*8*(p.read_pj_bit+d.write_pj_bit+5)*1e-12
                transfer_energy += (p.static_w+d.static_w)*transfer_time
                pre = profiles[p.name][group]['mean_prefill']
                dec = profiles[d.name][group]['mean_decode']
                # Idle peer remains powered during sequential phase service.
                joules = (pre['energy_j']+dec['energy_j']+transfer_energy+
                          d.static_w*pre['seconds']+p.static_w*dec['seconds'])
                options.append(dict(prefill=p.name,decode=d.name,cost_units=p.cost_units+d.cost_units,
                    seconds=pre['seconds']+dec['seconds']+transfer_time,energy_j=joules,
                    kv_transfer_bytes=mean_transfer,transfer_seconds=transfer_time,
                    replicated_weight_bytes=2*weights[model]))
        budgeted_pd.append(dict(group=group,feasible_pairs=len(options),
             fastest=min(options,key=lambda x:x['seconds']) if options else None,
             lowest_energy=min(options,key=lambda x:x['energy_j']) if options else None,
             scope='Per-role single-request P/D analytical baseline, not native Splitwise or a queued serving simulator.'))
    # Sensitivity changes the feasible set and hardware estimates, not measured
    # input shapes. Include cost, area and energy coefficients separately.
    sensitivity = []
    for label, cm, am, em in [('nominal',1,1,1),('memory_cost_half',.5,1,1),('memory_cost_double',2,1,1),
                              ('logic_area_half',1,.5,1),('logic_area_double',1,2,1),
                              ('mac_energy_half',1,1,.5),('mac_energy_double',1,1,2)]:
        hs, rs = candidates(spec,cm,am,em)
        for group in ('mem0_update_prefix','embed_full_01_query'):
            # Median input-length *actual request*, not the averaged fictitious
            # prompt. A limited sensitivity screen, not the full replay grid.
            calls = sorted(trace['groups'][group]['calls'],key=lambda c:c['prompt_tokens'])
            call = calls[len(calls)//2]
            scores = []
            for h in hs:
                r = decoder(h,trace['models'][call['model']]['config'],call['prompt_tokens'],call['completion_tokens'])
                scores.append(dict(hardware=h.name,seconds=r['total']['seconds'],energy_j=r['total']['energy_j']))
            sensitivity.append(dict(case=label,group=group,trace_id=call['trace_id'],feasible=len(hs),
                                    fastest=min(scores,key=lambda x:x['seconds']) if scores else None,
                                    lowest_energy=min(scores,key=lambda x:x['energy_j']) if scores else None))
    # Cached-prefill oracle: compare crossovers without attributing software
    # savings to a new engine. Both strategies receive the same fractional credit.
    cache = []
    for hw in hardware:
        for credit in (0,.5,.9,1.):
            raw = costs[hw.name,None,'online']; mem = costs[hw.name,4096,'online']
            p_raw = profiles[hw.name]['embed_full_01_query']['mean_prefill']
            shortened = [decoder(hw,trace['models'][c['model']]['config'],min(4096,c['prompt_tokens']),c['completion_tokens'])
                         for c in trace['groups']['embed_full_01_query']['calls']]
            p_mem = average(r['prefill'] for r in shortened)
            saved = raw['query']['seconds']-mem['query']['seconds']-credit*(p_raw['seconds']-p_mem['seconds'])
            cache.append(dict(hardware=hw.name,ideal_prefill_credit=credit,query_saved_seconds=saved,
                              reuse_crossing=break_even(88*mem['write_unit']['seconds']+
                                  cold_load(weights['Qwen/Qwen3-8B'])['seconds']-raw['base_embedding_build']['seconds'],saved)))
    # Mechanism control: expose the same amount of usable memory on every
    # candidate, retaining its native ports, costs and energy. This deliberately
    # wastes LPDDR capacity; it is an explanatory ablation, not the fair budget.
    equal_capacity = []
    common_capacity = min(h.capacity_bytes for h in hardware)
    eq_costs = {}
    for hw in hardware:
        for cap in caps:
            eq_costs[hw.name,cap] = scenario_costs(replace(hw,capacity_bytes=common_capacity),
                                trace,profiles[hw.name],weights,cap,'online')
    for reuse in (8,128,2048):
        for active in (1,4,8):
            rows = []
            for hw in hardware:
                for cap in caps:
                    c=eq_costs[hw.name,cap]
                    if c['max_active_slots']<active: continue
                    build = c['base_embedding_build']['seconds'] if cap is None else (
                        88*c['write_unit']['seconds']+cold_load(weights['Qwen/Qwen3-8B'])['seconds'])
                    rows.append(dict(hardware=hw.name,strategy='raw' if cap is None else f'cap_{cap}',
                                     seconds=build+reuse*c['query']['seconds']))
            joint=select(rows,'seconds')
            fixed=select([r for r in rows if r['hardware']==ref],'seconds')
            frozen=select([r for r in rows if fixed and r['strategy']==fixed['strategy']],'seconds')
            equal_capacity.append(dict(reuse=reuse,active=active,capacity_bytes=common_capacity,
                joint=joint,hardware_after_policy_selection=frozen,
                penalty=None if not joint or not frozen else frozen['seconds']/joint['seconds']))
    workspace_control=[]
    # Capacity-only optimistic control: no time penalty for chunked prefill.
    # It tests whether an unmeasured workspace allowance manufactures a cliff.
    for base in (.25,1,4,8):
        for chunk in (None,512):
            rows=[]
            for hw in hardware:
                for cap in caps:
                    c=scenario_costs(hw,trace,profiles[hw.name],weights,cap,'online',
                                     workspace_base_gib=base,workspace_chunk=chunk)
                    if c['max_active_slots']<4: continue
                    build=c['base_embedding_build']['seconds'] if cap is None else (
                        88*c['write_unit']['seconds']+cold_load(weights['Qwen/Qwen3-8B'])['seconds'])
                    rows.append(dict(hardware=hw.name,strategy='raw' if cap is None else f'cap_{cap}',
                                     seconds=build+8*c['query']['seconds']))
            joint=select(rows,'seconds'); fixed=select([r for r in rows if r['hardware']==ref],'seconds')
            frozen=select([r for r in rows if fixed and r['strategy']==fixed['strategy']],'seconds')
            workspace_control.append(dict(workspace_base_gib=base,chunk=chunk,reuse=8,active=4,
                joint=joint,frozen_policy=frozen,penalty=None if not joint or not frozen else frozen['seconds']/joint['seconds']))
    result = dict(scope=__doc__,elapsed_seconds=time.time()-start,input_sha256=receipts,
        hardware=[hardware_dict(h) for h in hardware],rejected=rejected,request_profiles=all_calls,
        crossovers=crossovers,scenarios=scenarios,pd_oracle=pd,budgeted_pd=budgeted_pd,
        sensitivity=sensitivity,cache_oracle=cache,equal_capacity_control=equal_capacity,
        workspace_control=workspace_control,
        assumptions=['All BF16, one device, no measured NPU timings or semantic quality-valid shortened inputs.',
          '88 writes scale first nine successful pairs of a failed run; no mixed sources presented as a real end-to-end run.',
          'Completion lengths are held fixed in counterfactual queries; they can change with memory representation.',
          'Active count is an admission-capacity requirement, not an observed batch or throughput.',
          'Request service is B=1; no queueing, time overlap, software protocol cost, or SLA prediction.',
          'Offline construction unloads 8B before serving; online charges extra weights, construction KV and workspace.',
          'Additional 8B cold load charged at 32 GB/s and 5 pJ/bit. P/D transfers use 64 GB/s and 5 pJ/bit; these are scenario parameters, not measured links.',
          'OS/WS GEMMs and causal GQA attention modeled. Softmax, normalization, vector kernels, host/index costs, bank timing and network excluded.',
          'Cost/logic-area/power assumptions are conditional; feasibility under them is not manufacturability or an iso-dollar result.',
          'All candidates are flexible unified pools. No new heterogeneous mechanism has been claimed.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'exploration.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(hardware=len(hardware),scenarios=len(scenarios),elapsed=time.time()-start)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/architecture')
    run(parser.parse_args().output)
