"""Crossed array/bandwidth controls, counted mapping, and paired L/O attribution."""
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path

from architecture.explore import load,average
from architecture.model import candidates,CYCLE_KEYS
from architecture.mechanism import KShard,provision,evaluate,linear,sharded,TRAFFIC

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/architecture'


def workloads(trace):
    calls=trace['groups']['mem0_update_prefix']['calls']
    result={'update':[dict(model=c['model'],L=c['prompt_tokens'],O=c['completion_tokens'],id=c['trace_id']) for c in calls]}
    rows=json.loads((OUT/'finite_prefix.json').read_text())['summary']['results']
    for a,b in [('raw','raw'),('mem0','raw'),('raw','mem0'),('mem0','mem0')]:
        label=a+'_'+b;items=[]
        for index in sorted({r['index'] for r in rows}):
            l=next(r for r in rows if r['index']==index and r['strategy']==a)
            o=next(r for r in rows if r['index']==index and r['strategy']==b)
            items.append(dict(model='Qwen/Qwen3-32B',L=l['input_tokens'],O=o['output_tokens'],id=index))
        result[label]=items
    return result


def profile(hw,calls,trace,scheme=None,keep=False):
    results=[dict(id=c['id'],L=c['L'],O=c['O'],**evaluate(hw,trace['models'][c['model']]['config'],c['L'],c['O'],scheme)) for c in calls]
    mean=average(r['total'] for r in results)
    for r in results:
        for component in r['components'].values():
            assert abs(sum(component['array_'+key+'_seconds'] for key in CYCLE_KEYS)-component['compute_seconds'])<1e-7
            assert abs(sum(component[k] for k in TRAFFIC[:4])-component['read_bytes'])<1
            assert abs(sum(component[k] for k in TRAFFIC[4:])-component['write_bytes'])<1
            assert abs(sum(component.get('limited_by_'+k+'_seconds',0) for k in ('compute','memory','sram','reduction'))-component['seconds'])<1e-7
    result=dict(mean=mean,prefill=average(r['prefill'] for r in results),decode=average(r['decode'] for r in results),
        dominant=max(('compute','memory','sram','reduction'),key=lambda k:mean.get('limited_by_'+k+'_seconds',0)))
    if keep:
        result['components']={k:average(r['components'][k] for r in results) for k in results[0]['components']}
        result['requests']=[dict(id=r['id'],L=r['L'],O=r['O'],total=r['total']) for r in results]
    return result


def main():
    spec,trace,_,_,receipts=load();hardware,_=candidates(spec)
    base=next(h for h in hardware if h.name=='HBM3E4_a16_128x128_s128')
    work=workloads(trace);cross=[]
    for mode in ('scheduled','continuous_k','ideal'):
        for factor in (1,2,4):
            for mapping in ('frozen','remap'):
                hw=replace(base,bandwidth=base.bandwidth*factor,cycle_mode='ideal' if mode=='ideal' else 'scheduled',
                    fill_mode='continuous_k' if mode=='continuous_k' else 'per_k_tile',
                    freeze_mapping_bandwidth=base.bandwidth if mapping=='frozen' else 0)
                for group in ('update','raw_raw','mem0_mem0'):
                    cross.append(dict(mode=mode,bandwidth_factor=factor,mapping=mapping,group=group,
                                      **profile(hw,work[group],trace,keep=True)))
    designs=[];rejected=[];design_profiles=[]
    for original in hardware:
        hw=replace(original,fill_mode='continuous_k')
        designs.append(dict(name=hw.name,kind='unified_continuous_k',hardware=asdict(hw),charges=None))
        for group,calls in work.items():
            design_profiles.append(dict(design=hw.name,group=group,**profile(hw,calls,trace)))
        if hw.rows!=128: continue
        for groups in (2,4,16,64):
            scheme=KShard(groups=groups);modified,cost,violations=provision(hw,scheme,spec)
            if violations:
                rejected.append(dict(name=modified.name,charges=cost,violations=violations));continue
            designs.append(dict(name=modified.name,kind='counted_k_shard',hardware=asdict(modified),scheme=asdict(scheme),charges=cost))
            for group,calls in work.items():
                design_profiles.append(dict(design=modified.name,group=group,**profile(modified,calls,trace,scheme)))
        print('profiled',original.name,flush=True)
    kinds={d['name']:d['kind'] for d in designs}
    best=[]
    for group in work:
        for kind in ('unified_continuous_k','counted_k_shard'):
            rows=[p for p in design_profiles if p['group']==group and kinds[p['design']]==kind]
            for metric in ('seconds','energy_j'):
                winner=min(rows,key=lambda p:p['mean'][metric])
                best.append(dict(group=group,kind=kind,metric=metric,design=winner['design'],value=winner['mean'][metric]))
    controls=[]
    for groups in (2,4,16,64):
        for tag,scheme in [('nominal',KShard(groups=groups)),('bank_skew2',KShard(groups=groups,bank_skew=2)),
            ('bank_skew4',KShard(groups=groups,bank_skew=4)),('overhead4',KShard(groups=groups,overhead_multiplier=4)),
            ('uncoalesced_refills',KShard(groups=groups,transfer_granularity='buffer_tile')),
            ('clock_drop_5pct_per_level',KShard(groups=groups,clock_drop_per_level=.05))]:
            modified,cost,violations=provision(base,scheme,spec)
            for factor in ((1,2,4) if tag=='nominal' else (1,)):
                hw=replace(modified,bandwidth=modified.bandwidth*factor)
                for group in ('update','raw_raw','mem0_mem0'):
                    controls.append(dict(group=group,tag=tag,k_groups=groups,bandwidth_factor=factor,
                        diagnostic_only=True,resource_violations=violations,charges=cost,**profile(hw,work[group],trace,scheme)))
    attribution=[]
    for tag,hw,scheme in [('original',base,None),('continuous_k',replace(base,fill_mode='continuous_k'),None),
                          ('counted_k2',provision(base,KShard(groups=2),spec)[0],KShard(groups=2))]:
        points={g:profile(hw,calls,trace,scheme)['mean']['seconds'] for g,calls in work.items() if g!='update'}
        input_credit=((points['raw_raw']-points['mem0_raw'])+(points['raw_mem0']-points['mem0_mem0']))/2
        output_credit=((points['raw_raw']-points['raw_mem0'])+(points['mem0_raw']-points['mem0_mem0']))/2
        assert abs(input_credit+output_credit-(points['raw_raw']-points['mem0_mem0']))<1e-9
        attribution.append(dict(mapping=tag,points=points,input_credit_seconds=input_credit,output_credit_seconds=output_credit,
            input_fraction=input_credit/(input_credit+output_credit),method='Symmetric average of both orders; middle points are counterfactual.'))
    paths=['architecture/mechanism.py','architecture/joint_diagnostics.py','evidence/architecture/finite_prefix.json']
    samples=[]
    for model in ('Qwen/Qwen3-8B','Qwen/Qwen3-32B'):
        cfg=trace['models'][model]['config'];h=cfg['hidden_size'];ff=cfg['intermediate_size']
        for groups in (2,4,16,64):
            scheme=KShard(groups=groups);hw,_,_=provision(base,scheme,spec)
            for op,k,n in [('Q',h,cfg['num_attention_heads']*cfg['head_dim']),('up',h,ff),('down',ff,h)]:
                samples.append(dict(model=model,op=op,shape=[1,k,n],k_groups=groups,estimate=sharded(hw,scheme,1,k,n)))
    output=dict(scope=__doc__,input_sha256={**receipts,**{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}},
        crossed_controls=cross,designs=designs,rejected=rejected,design_profiles=design_profiles,best=best,
        mapping_controls=controls,workloads=work,attribution=attribution,mapping_operator_samples=samples,
        assumptions=['2x/4x bandwidth and ideal cycles are diagnostic, absent from physical rankings.',
          'Physical comparisons use the same 850 cost, 450 mm2, 500 W and 50 mm budgets and unchanged DRAM technology candidates.',
          'Stronger unified baseline allows continuous K and chooses its best OS/WS tile.',
          'K sharding changes M=1 linear projections only; attention remains unchanged.',
          'Balanced bank layout and aggregate port service are modeled; skew controls are not a bank-timing simulation.',
          'Reduction, broadcast, buffers, muxes, clock loss and controller costs are charged but not synthesized.',
          'raw_raw and mem0_mem0 are observed pairs; other L/O combinations are counterfactuals.',
          'All physical winners are conditional estimates; no HB-on-Logic domain or measured hardware speedup.'])
    (OUT/'joint_diagnostics.json').write_text(json.dumps(output,indent=2)+'\n')
    for x in cross:
        if x['group']=='update' and x['mapping']=='frozen': print(x['mode'],x['bandwidth_factor'],x['mean']['seconds'],x['dominant'])
    print('best',json.dumps([b for b in best if b['metric']=='seconds']))


if __name__=='__main__': main()
