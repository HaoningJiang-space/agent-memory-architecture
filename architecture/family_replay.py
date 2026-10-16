"""Fixed-design, same-request family replay plus a legal batching control.

Independent B=1 sums are work accounting, never a native execution timeline.
The batch control uses four distinct, simultaneously ready SimpleMem windows;
it preserves every L/O and explicitly charges padding and the shrinking tail.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from architecture import model as base
from architecture import mechanism as mech
from architecture import supply_path as supply
from architecture.explore import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/architecture'
BASELINE = 'HBM3E4_a16_128x128_s128_k2'
CANDIDATE = 'HBM3E4_a16_128x128_s32_k2_hb32b51.2p4_price100'
FRACTION = 0.3052229000351929  # Frozen on old Mem0 update, not these algorithms.


def segments(calls):
    """Each original completion contributes exactly O-1 decode positions."""
    assert calls and len({c['id'] for c in calls}) == len(calls)
    assert all(c['L'] > 0 and c['O'] >= 1 for c in calls)
    previous = 0
    rows = []
    for end in sorted({c['O'] - 1 for c in calls if c['O'] > 1}):
        active = [c for c in calls if c['O'] - 1 > previous]
        rows.append(dict(start=previous, end=end, count=end-previous,
                         ids=[c['id'] for c in active], batch=len(active),
                         padded_prompt=max(c['L'] for c in active)))
        previous = end
    assert sum(r['count']*r['batch'] for r in rows) == sum(c['O']-1 for c in calls)
    return rows


def forward(hw, config, length, batch, decode, path=None, fraction=0, embedding=False):
    h, d = config['hidden_size'], config['head_dim']
    q, kv = config['num_attention_heads']*d, config['num_key_value_heads']*d
    ff, layers = config['intermediate_size'], config['num_hidden_layers']
    shapes = [(h,q), (h,kv), (h,kv), (q,h), (h,ff), (h,ff), (ff,h)]
    m = batch if decode else batch*length
    dense = supply.scale(supply.add(*(supply.linear(hw,m,k,n,True,path,fraction) for k,n in shapes)),layers)
    if not embedding:
        dense = supply.add(dense,supply.linear(hw,batch,h,config['vocab_size'],True,path,fraction))
    attn = mech.attention(supply.tile_hardware(hw),length,config['num_attention_heads'],
                          config['num_key_value_heads'],d,batch=batch,decode=decode)
    return supply.add(dense,supply.scale(supply.score(hw,attn),layers))


def batched(hw, config, calls, path=None, fraction=0):
    """Static ready group, padded prefill, compact only at true completion ends.

    Sequential attention time is approximated at each segment's midpoint, as
    in the frozen model. Compaction itself is an optimistic free control; no
    result is interpreted as an implemented runtime or whole-agent speedup.
    """
    parts = segments(calls)
    pre = forward(hw,config,max(c['L'] for c in calls),len(calls),False,path,fraction)
    dec = supply.add(*(supply.scale(forward(hw,config,
        p['padded_prompt']+(p['start']+p['end']+1)//2,p['batch'],True,path,fraction),p['count']) for p in parts))
    return dict(prefill=pre,decode=dec,total=supply.add(pre,dec),segments=parts,
                actual_input_tokens=sum(c['L'] for c in calls),
                padded_input_tokens=len(calls)*max(c['L'] for c in calls),
                actual_decode_tokens=sum(c['O']-1 for c in calls))


def totals(rows):
    return {phase:supply.add(*(r[phase] for r in rows)) for phase in ('prefill','decode','total')}


def main():
    _, trace, _, capacity, _ = load()
    weights = {m['model']:m['weight_bytes'] for m in capacity['models']}
    exploration = json.loads((OUT/'supply_exploration.json').read_text())
    designs = {d['name']:d for d in exploration['designs']}
    baseline = base.Hardware(**designs[BASELINE]['hardware'])
    candidate = base.Hardware(**designs[CANDIDATE]['hardware'])
    path = supply.LocalPath(**designs[CANDIDATE]['path'])
    family = json.loads((OUT/'family_summary.json').read_text())
    probe = json.loads((OUT/'family_kernel_probe.json').read_text())
    output = dict(scope=__doc__, baseline=BASELINE,candidate=CANDIDATE,
                  fraction_rule='min(0.3052229000351929, 10e9/model_weight_bytes)',
                  tuning='Both hardware and fraction rule frozen on earlier Mem0 update before family replay.',
                  algorithms=[],batch_control=[],kernel_probe={},source_sha256={})
    for algorithm in family['algorithms']:
        phase_rows = []
        for phase in sorted({c['phase'] for c in algorithm['calls']}):
            for model in sorted({c['model'] for c in algorithm['calls'] if c['phase']==phase}):
                calls = [c for c in algorithm['calls'] if c['phase']==phase and c['model']==model]
                cfg = trace['models'][model]['config']
                fraction = min(FRACTION,path.capacity_bytes/weights[model])
                a = totals([supply.evaluate(baseline,cfg,c['L'],c['O'],True) for c in calls])
                b = totals([supply.evaluate(candidate,cfg,c['L'],c['O'],True,path,fraction) for c in calls])
                phase_rows.append(dict(phase=phase,model=model,count=len(calls),fraction=fraction,
                    baseline=a,candidate=b,speed_ratio=a['total']['seconds']/b['total']['seconds'],
                    energy_ratio=a['total']['energy_j']/b['total']['energy_j'],
                    decode_share=a['decode']['seconds']/a['total']['seconds'],
                    decode_weight_read_share=a['decode']['weight_read_bytes']/(
                        a['decode']['backing_read_bytes'] or 1)))
        embedding_rows=[]
        cfg=trace['models']['Qwen/Qwen3-Embedding-0.6B']['config']
        for k in algorithm['embedding_kernels']:
            embedding_rows.append(dict(phase=k['phase'],batch=k['batch'],padded_length=k['padded_length'],
                baseline=forward(baseline,cfg,k['padded_length'],k['batch'],False,embedding=True),
                candidate=forward(candidate,cfg,k['padded_length'],k['batch'],False,embedding=True)))
        aggregate={name:supply.add(*(r[name]['total'] for r in phase_rows),
                                  *(r[name] for r in embedding_rows)) for name in ('baseline','candidate')}
        cold=[]
        for model in sorted({c['model'] for c in algorithm['calls']}):
            f=min(FRACTION,path.capacity_bytes/weights[model])
            cold.append(dict(model=model,**supply.cold_fill(candidate,path,weights[model],f)))
        output['algorithms'].append(dict(algorithm=algorithm['algorithm'],phases=phase_rows,embedding=embedding_rows,
            independent_service_sum=aggregate,
            independent_speed_ratio=aggregate['baseline']['seconds']/aggregate['candidate']['seconds'],
            independent_energy_ratio=aggregate['baseline']['energy_j']/aggregate['candidate']['energy_j'],
            cold_fill_per_model=cold,
            residency='Per-model separate profiling. 8B and 32B each have their own HB placement. Not simultaneously resident in one 10 GB HB cache; a mixed-model schedule must refill and account for evictions.',
            native_wall_seconds=algorithm['end']-algorithm['start']))
    simple=next(a for a in family['algorithms'] if a['algorithm']=='simplemem')
    calls=sorted([c for c in simple['calls'] if c['phase']=='construction'],key=lambda c:c['start'])
    assert [c['id'] for c in calls]==probe['source_ids']
    assert [c['L'] for c in calls]==probe['input_lengths']
    cfg=trace['models'][calls[0]['model']]['config']
    for tag,subset in [('eight_decode_steps',[dict(c,O=9) for c in calls]),('full_observed_outputs',calls)]:
        before=totals([supply.evaluate(baseline,cfg,c['L'],c['O'],True) for c in subset])
        after=batched(baseline,cfg,subset)
        local=batched(candidate,cfg,subset,path,FRACTION)
        output['batch_control'].append(dict(tag=tag,requests=[{k:c[k] for k in ('id','L','O')} for c in subset],
            independent_baseline=before,batched_baseline=after,batched_candidate=local,
            software_batch_speed_ratio=before['total']['seconds']/after['total']['seconds'],
            hardware_after_batch_ratio=after['total']['seconds']/local['total']['seconds'],
            limits='Controlled padded batch; no prefix reuse; completions only set work amount, no new answer quality assertion. Compact at real completion boundaries for free (optimistic software control).'))
    forwards=probe['forwards']
    for phase in ('prefill','decode'):
        solo=[r for r in forwards if len(r['group'])==1 and r['phase']==phase]
        batch=[r for r in forwards if len(r['group'])==4 and r['phase']==phase]
        output['kernel_probe'][phase]=dict(solo_sum_cuda_seconds=sum(r['cuda_seconds'] for r in solo),
            batch_sum_cuda_seconds=sum(r['cuda_seconds'] for r in batch),
            ratio=sum(r['cuda_seconds'] for r in solo)/sum(r['cuda_seconds'] for r in batch),
            solo_largest_peak_bytes=max(r['peak_allocated_bytes'] for r in solo),
            batch_peak_bytes=max(r['peak_allocated_bytes'] for r in batch),
            solo_largest_kv_bytes=max(r['kv_bytes'] for r in solo),batch_kv_bytes=max(r['kv_bytes'] for r in batch))
    for p in ['architecture/family_replay.py','architecture/supply_path.py','architecture/mechanism.py',
              'architecture/model.py','architecture/spec.json','evidence/architecture/family_summary.json',
              'evidence/architecture/family_kernel_probe.json','evidence/architecture/supply_exploration.json']:
        output['source_sha256'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    (OUT/'family_replay.json').write_text(json.dumps(output,indent=2)+'\n')
    print([(a['algorithm'],a['independent_speed_ratio'],a['independent_energy_ratio']) for a in output['algorithms']])
    print([(b['tag'],b['software_batch_speed_ratio'],b['hardware_after_batch_ratio']) for b in output['batch_control']])


if __name__=='__main__':main()
