"""Bounded decision replay: dependency readiness, paid small-M supply, full paths.

Recorded timestamps identify source stages / host-duration anchors only. They
are never used as target-hardware arrival times. Three independent questions
are either released sequentially (native harness) or together (controlled).
"""
from collections import Counter
from dataclasses import asdict,replace
import hashlib
import heapq
import gzip
import json
from pathlib import Path

from architecture import model as base,supply_path as supply,small_batch as small
from architecture.collect_family import overlap
from architecture.explore import load
from architecture.family_replay import BASELINE,CANDIDATE,FRACTION,batch_capacity

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/architecture'


def union(rows):return overlap(rows)['api_interval_union_seconds'] if rows else 0.0


def subtract_union(outer,inner):
    if not outer:return 0.0
    points=sorted({r[k] for r in outer+inner for k in ('start','end')})
    return sum(b-a for a,b in zip(points,points[1:]) if
        any(r['start']<=(a+b)/2<r['end'] for r in outer) and
        not any(r['start']<=(a+b)/2<r['end'] for r in inner))


def graph(a,parallel_questions=False):
    nodes=[];used_calls=set();used_kernels=set();host_seconds=0.0
    def add(kind,deps=(),**values):
        identity=f'{a["algorithm"]}:{len(nodes):04d}'
        nodes.append(dict(id=identity,kind=kind,deps=list(deps),**values));return identity
    # CPU residual is a common measured-duration control, not scaled GPU API time.
    all_intervals=a['calls']+a['embedding_calls']+a['operations']
    residual=max(0,a['end']-a['start']-union(all_intervals))
    start=add('host',seconds=residual,stage='untraced_host_residual')
    host_seconds+=residual
    def tail(phase,lo,hi,deps,label):
        nonlocal host_seconds
        emb=[]
        for i,k in enumerate(a['embedding_kernels']):
            if k['phase']==phase and lo<=k['start']<hi:
                assert i not in used_kernels;used_kernels.add(i)
                emb.append(add('embedding',deps,model='Qwen/Qwen3-Embedding-0.6B',L=k['padded_length'],
                               B=k['batch'],stage=label,source_kernel=i))
        ops=[r for r in a['operations'] if r['phase']==phase and lo<=r['start']<hi]
        neural=[r for r in a['embedding_calls'] if r['phase']==phase]+[r for r in a['calls'] if r['phase']==phase]
        seconds=subtract_union(ops,neural);host_seconds+=seconds
        return add('host',emb or deps,seconds=seconds,stage=label)
    def llm(c,deps):
        assert c['id'] not in used_calls;used_calls.add(c['id'])
        return add('llm',deps,model=c['model'],L=c['L'],O=c['O'],source_id=c['id'],stage=c['phase'])
    construction=sorted([c for c in a['calls'] if c['phase']=='construction'],key=lambda c:c['start'])
    if a['algorithm']=='simplemem':
        ready=[llm(c,[start]) for c in construction]
        completed=tail('construction',float('-inf'),float('inf'),ready,'construction_finalize')
    else:
        # Source harness calls add_memory sequentially. Overlapping extraction
        # calls belong to one native buffer flush (the final group has two).
        groups=[]
        for c in construction:
            if not groups or c['start']>=max(x['end'] for x in groups[-1]):groups.append([])
            groups[-1].append(c)
        completed=start
        for i,g in enumerate(groups):
            ids=[llm(c,[completed]) for c in g]
            hi=groups[i+1][0]['start'] if i+1<len(groups) else float('inf')
            completed=tail('construction',g[0]['start'],hi,ids,f'construction_flush_{i}')
        maintenance=sorted([c for c in a['calls'] if c['phase']=='maintenance'],key=lambda c:c['start'])
        first=maintenance[0]['start']
        queue=tail('maintenance',float('-inf'),first,[completed],'maintenance_snapshot_queue')
        ids=[llm(c,[queue]) for c in maintenance]
        completed=tail('maintenance',first,float('inf'),ids,'maintenance_commit_barrier')
    build_end=completed;query_ends=[]
    for phase in sorted({c['phase'] for c in a['calls'] if c['phase'].startswith('query_')}):
        last=build_end if parallel_questions else completed
        calls=sorted([c for c in a['calls'] if c['phase']==phase],key=lambda c:c['start'])
        lo=float('-inf')
        for c in calls:
            between=tail(phase,lo,c['start'],[last],phase+'_retrieval')
            last=llm(c,[between]);lo=c['start']
        completed=tail(phase,lo,float('inf'),[last],phase+'_finalize');query_ends.append(completed)
    assert used_calls=={c['id'] for c in a['calls']}
    assert used_kernels==set(range(len(a['embedding_kernels'])))
    return dict(nodes=nodes,host_anchor_seconds=host_seconds,query_ends=query_ends,
        parallel_questions=parallel_questions,
        scope='Conservative native stage barriers; all 29 maintenance LLM decisions read one snapshot. '
              'CPU interval anchors exclude nested neural intervals; untraced host residual is a common initial control. '
              'No target arrival is copied from an API timestamp.')


def replay(dag,hw,mode,trace,weights,path=None,maintenance_cap=None,host_scale=1.0):
    nodes={n['id']:n for n in dag['nodes']};order={n['id']:i for i,n in enumerate(dag['nodes'])}
    pending=set(nodes);done={};starts={};events=[];now=0.;gpu_free=0.;cpu_free=0.;cache_model=None
    resident=sum(weights.values())  # all three checkpoints stay in backing
    if resident>hw.capacity_bytes:raise ValueError('Resident checkpoints exceed backing')
    # Identical initial full-checkpoint loading for every device; host link and
    # energy are an explicit common control, no free candidate model placement.
    initial=resident/32e9
    now=initial;gpu_free=initial;cpu_free=initial
    gpu_active=initial;energy=resident*8*5e-12+hw.static_w*initial
    batches=[];cold=[];host_active=0.;peak=resident
    def push(time,identity):heapq.heappush(events,(time,identity))
    def ready(kind):return sorted([nodes[i] for i in pending if nodes[i]['kind'] in kind and all(d in done for d in nodes[i]['deps'])],key=lambda n:order[n['id']])
    while pending or events:
        while events and events[0][0]<=now+1e-12:
            t,i=heapq.heappop(events);done[i]=t
        if not pending and not events:break
        progressed=False
        if cpu_free<=now+1e-12:
            rr=ready({'host'})
            if rr:
                n=rr[0];pending.remove(n['id']);duration=n['seconds']*host_scale
                starts[n['id']]=now
                cpu_free=now+duration;host_active+=duration;push(cpu_free,n['id']);progressed=True
                # Drain zero-duration dependency barriers before dispatching
                # GPU work, so simultaneous readiness is not serialized by
                # the Python order of otherwise instantaneous host nodes.
                if duration==0:continue
        if gpu_free<=now+1e-12:
            rr=ready({'llm','embedding'})
            if rr:
                n=rr[0];cfg=trace['models'][n['model']]['config'];fraction=0
                if n['kind']=='embedding':
                    selected=[n];charge=2*n['B']*n['L']*(cfg['hidden_size']+2*cfg['intermediate_size'])
                    if resident+charge>hw.capacity_bytes:raise ValueError('Embedding capacity overflow')
                    peak=max(peak,resident+charge)
                    result=small.forward(hw,cfg,n['L'],n['B'],False,mode,embedding=True)
                    finish={n['id']:result['seconds']};segs=[]
                else:
                    selected=[]
                    compatible=[x for x in rr if x['kind']=='llm' and x['model']==n['model']]
                    if maintenance_cap is not None and n['stage']=='maintenance':compatible=compatible[:maintenance_cap]
                    for x in compatible:
                        trial=selected+[x]
                        if batch_capacity(cfg,trial,resident)<=hw.capacity_bytes:selected=trial
                    if not selected:raise ValueError('No legal request fits with resident checkpoints')
                    fraction=min(FRACTION,path.capacity_bytes/weights[n['model']]) if path else 0
                    r=small.execute_batch(hw,cfg,selected,mode,path,fraction,resident)
                    peak=max(peak,r['peak_capacity_bytes']);result=r['total'];finish=r['completion_seconds'];segs=r['segments']
                fill=dict(bytes=0,seconds=0,energy_j=0)
                if path and n['kind']=='llm' and cache_model!=n['model']:
                    fill=supply.cold_fill(hw,path,weights[n['model']],fraction)
                    cold.append(dict(model=n['model'],start=now,**fill));cache_model=n['model']
                begin=now+fill['seconds'];gpu_free=begin+result['seconds']
                gpu_active+=fill['seconds']+result['seconds'];energy+=fill['energy_j']+result['energy_j']
                for x in selected:
                    pending.remove(x['id']);starts[x['id']]=begin;push(begin+finish[x['id']],x['id'])
                batches.append(dict(start=begin,end=gpu_free,model=n['model'],kind=n['kind'],
                    stage=n['stage'],node_ids=[x['id'] for x in selected],batch=len(selected) if n['kind']=='llm' else n['B'],
                    service=result,segments=[{**{k:v for k,v in p.items() if k!='service'},'seconds':p['service']['seconds']} for p in segs]))
                progressed=True
        if progressed:continue
        times=[t for t in (events[0][0] if events else float('inf'),gpu_free,cpu_free) if t>now+1e-12]
        if not times:raise RuntimeError('Dependency deadlock')
        now=min(times)
    wall=max([gpu_free,cpu_free]+list(done.values()))
    # Device stays powered while host barriers wait; both active and idle paid.
    idle=max(0,wall-gpu_active);energy+=hw.static_w*idle
    host_energy=20*wall  # common always-on CPU control, not measured package joules
    for n in nodes.values():
        for predecessor in n['deps']:assert done[predecessor]<=starts[n['id']]+1e-12
    return dict(seconds=wall,energy_j=energy+host_energy,accelerator_energy_j=energy,host_energy_j=host_energy,
        gpu_idle_seconds=idle,host_active_seconds=host_active,peak_backing_bytes=peak,resident_checkpoint_bytes=resident,
        initial_load_seconds=initial,cold_fills=cold,batches=batches,node_completions=done,node_starts=starts,
        query_completion_seconds=[done[i] for i in dag['query_ends']],
        maintenance_cap=maintenance_cap,host_scale=host_scale)


def main():
    spec,trace,_,capacity,_=load();weights={m['model']:m['weight_bytes'] for m in capacity['models']}
    previous=json.loads((OUT/'supply_exploration.json').read_text());designs={d['name']:d for d in previous['designs']}
    b0=base.Hardware(**designs[BASELINE]['hardware']);old=base.Hardware(**designs[CANDIDATE]['hardware'])
    path=supply.LocalPath(**designs[CANDIDATE]['path'])
    baseline,bcharge,bbad=small.provision(b0,spec);assert not bbad
    # Re-provision ONLY the same fixed candidate, never reselect parameters.
    old_backing=base.Hardware(**designs[CANDIDATE.split('_hb')[0]]['hardware'])
    hw_entries=[dict(name='baseline',mode='baseline',hardware=asdict(baseline),path=None,charges=bcharge,violations=[]),
                dict(name='old_m1',mode='old_m1',hardware=asdict(old),path=asdict(path),charges={},violations=[])]
    for tag,p in [('small_m',path),('bank_skew2',replace(path,bank_skew=2)),
                  ('half_interface',replace(path,interface_efficiency=.4)),
                  ('clock_loss5pct',replace(path,extra_clock_loss=.05)),
                  ('controller_area2x',replace(path,controller_area_scale=2))]:
        hw,ch,bad=supply.provision(old_backing,p,spec);hw,extra,more=small.provision(hw,spec)
        hw_entries.append(dict(name=tag,mode='small_m',hardware=asdict(hw),path=asdict(p),charges={**ch,'small_batch':extra},violations=sorted(set(bad+more))))
    source=json.loads((OUT/'family_summary.json').read_text());graphs=[];runs=[]
    for a in source['algorithms']:
        for parallel in (False,True):
            dag=graph(a,parallel);gid=a['algorithm']+('_concurrent_queries' if parallel else '_sequential_queries')
            graphs.append(dict(name=gid,**dag))
            for cap in ([5,None] if a['algorithm']=='lightmem' else [None]):
                for d in hw_entries:
                    if d['violations']:continue
                    hw=base.Hardware(**d['hardware']);p=supply.LocalPath(**d['path']) if d['path'] else None
                    result=replay(dag,hw,d['mode'],trace,weights,p,cap)
                    runs.append(dict(graph=gid,design=d['name'],**result))
                    print(gid,cap,d['name'],result['seconds'],result['energy_j'],flush=True)
    comparisons=[]
    for r in runs:
        b=next(x for x in runs if x['graph']==r['graph'] and x['design']=='baseline' and x['maintenance_cap']==r['maintenance_cap'])
        feasible=r['seconds']<=b['seconds']+1e-12
        hw=next(d['hardware'] for d in hw_entries if d['name']==r['design'])
        # Same deadline: faster designs stay powered until the baseline finish.
        matched=r['energy_j']+(b['seconds']-r['seconds'])*(base.Hardware(**hw).static_w+20) if feasible else None
        comparisons.append(dict(graph=r['graph'],design=r['design'],maintenance_cap=r['maintenance_cap'],
            speed_ratio=b['seconds']/r['seconds'],energy_ratio=b['energy_j']/r['energy_j'],
            baseline_deadline_feasible=feasible,energy_ratio_at_baseline_deadline=b['energy_j']/matched if matched else None))
    paths=['architecture/small_batch.py','architecture/batch_closure.py','architecture/supply_path.py',
        'architecture/family_replay.py','architecture/model.py','architecture/mechanism.py','architecture/spec.json',
        'architecture/collect_family.py','architecture/explore.py',
        'evidence/architecture/family_summary.json','evidence/architecture/supply_exploration.json',
        'evidence/phase2/workload_accounting.json','evidence/calibration/model_capacity.json']
    out=dict(scope=__doc__,designs=hw_entries,graphs=graphs,runs=runs,comparisons=comparisons,
        budget=spec['budget'],input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits=['Conditional DAG service model; not a full A800 rerun or new chip measurement.',
                'Only recorded finite histories/questions; no quality-equivalence claim; LightMem keeps its known stale-vector behavior.',
                'Static ready batches, no in-flight admission; all >8 ready requests allowed if capacity permits.',
                'No prefix reuse, exact NoC/bank timing or thermal calibration. Shared assumptions cannot establish positive novelty.',
                'Midpoint attention and free batch compaction retained; host durations are common measured-duration anchors, not new CPU predictions.',
                'All three weights resident in backing; one HB model copy at a time; switch refill and powered idle included.'])
    raw=json.dumps(out,separators=(',',':')).encode()
    packed=gzip.compress(raw,mtime=0)
    (OUT/'batch_closure_traces.json.gz').write_bytes(packed)
    concise=[]
    for r in runs:
        histogram=Counter()
        for b in r['batches']:
            for segment in b['segments']:histogram[segment['batch']]+=segment['count']
        concise.append({**{k:v for k,v in r.items() if k not in ('batches','node_completions','node_starts')},
                        'decode_step_M_histogram':dict(sorted(histogram.items())),
                        'llm_batches':sum(b['kind']=='llm' for b in r['batches']),
                        'embedding_batches':sum(b['kind']=='embedding' for b in r['batches'])})
    summary={**out,'runs':concise,'trace_archive_sha256':hashlib.sha256(packed).hexdigest()}
    (OUT/'batch_closure.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
