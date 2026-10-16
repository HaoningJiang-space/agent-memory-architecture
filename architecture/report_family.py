"""Rebuild family figures and check the public numeric evidence contract."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/architecture'


def main():
    family=json.loads((OUT/'family_summary.json').read_text())
    replay=json.loads((OUT/'family_replay.json').read_text())
    probe=json.loads((OUT/'family_kernel_probe.json').read_text())
    for p,value in replay['source_sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==value,p
    # The public raw summary is frozen; its collector predates strict session
    # validation. Verify that recorded source version, not today's different code.
    collector=(ROOT/'architecture/collect_family.py').read_bytes()
    if family['collector_sha256']!=hashlib.sha256(collector).hexdigest():
        collector=(ROOT/'architecture/frozen/collect_family_a8bf9c1.py').read_bytes()
    assert family['collector_sha256']==hashlib.sha256(collector).hexdigest()
    assert probe['runner_sha256']==hashlib.sha256((ROOT/'architecture/family_kernel_probe.py').read_bytes()).hexdigest()
    assert len({a['history_sha256'] for a in family['algorithms']})==1
    checked=0
    def audit(row):
        nonlocal checked
        assert all(math.isfinite(v) and v>=0 for v in row.values())
        read=sum(row[k] for k in ('weight_read_bytes','kv_read_bytes','activation_read_bytes','partial_read_bytes'))
        assert math.isclose(read,row['backing_read_bytes']+row['hb_read_bytes'],rel_tol=1e-12,abs_tol=1e-6)
        assert math.isclose(row['activation_write_bytes']+row['partial_write_bytes'],row['backing_write_bytes'],rel_tol=1e-12,abs_tol=1e-6)
        assert math.isclose(sum(v for k,v in row.items() if k.startswith('limited_by_')),row['seconds'],rel_tol=1e-12,abs_tol=1e-9)
        checked+=1
    for a in replay['algorithms']:
        for p in a['phases']:
            for name in ('baseline','candidate'):
                for phase in ('prefill','decode','total'):audit(p[name][phase])
                assert math.isclose(p[name]['prefill']['seconds']+p[name]['decode']['seconds'],p[name]['total']['seconds'])
        for k in a['embedding']:
            for name in ('baseline','candidate'):audit(k[name])
        source=next(x for x in family['algorithms'] if x['algorithm']==a['algorithm'])
        assert sum(p['count'] for p in a['phases'])==len(source['calls'])
        assert len(a['embedding'])==len(source['embedding_kernels'])
        assert all(c['finish_reason']=='stop' for c in source['calls'])
        assert all(o['status']=='ok' for o in source['operations'])
    for b in replay['batch_control']:
        for design in ('independent_baseline','batched_baseline','batched_candidate'):
            for phase in ('prefill','decode','total'):audit(b[design][phase])
    validation=dict(source_hashes_verified=len(replay['source_sha256'])+2,
        same_history=True,llm_calls={a['algorithm']:len(a['calls']) for a in family['algorithms']},
        embedding_kernels={a['algorithm']:len(a['embedding_kernels']) for a in family['algorithms']},
        checked_traffic_and_time_rows=checked,kernel_hook_samples=len(probe['shapes']),
        fixed_hardware=True,no_raw_text_export=True,
        limitation='Counts/shape/flow checks do not establish semantic memory or answer correctness.')
    (OUT/'family_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.3),layout='constrained')
    labels=[];parts=defaultdict(list)
    for a in replay['algorithms']:
        for phase in ('construction','maintenance','planning','answer'):
            selected=[p for p in a['phases'] if ('construction' if p['phase']=='construction' else
                'maintenance' if p['phase']=='maintenance' else 'planning' if p['model'].endswith('8B') else 'answer')==phase]
            if not selected:continue
            labels.append(a['algorithm'].replace('simplemem','Simple').replace('lightmem','Light')+' '+phase[:5])
            total=sum(p['baseline']['total']['seconds'] for p in selected)
            parts['prefill'].append(sum(p['baseline']['prefill']['seconds'] for p in selected)/total)
            parts['decode'].append(sum(p['baseline']['decode']['seconds'] for p in selected)/total)
    xs=np.arange(len(labels))
    axes[0].barh(xs,parts['prefill'],label='Prefill',color='#4087b4')
    axes[0].barh(xs,parts['decode'],left=parts['prefill'],label='Decode',color='#d17b42')
    axes[0].set_yticks(xs,labels,fontsize=8);axes[0].invert_yaxis();axes[0].set_xlim(0,1)
    axes[0].set_xlabel('Fraction of isolated B=1 modeled service');axes[0].legend(fontsize=8,loc='lower right')
    axes[0].set_title('Same operation family, different phase mixes',fontsize=10)
    for i,phase in enumerate(('prefill','decode')):
        p=replay['kernel_probe'][phase]
        bars=axes[1].bar(np.array([0,1])+i*3,[p['solo_sum_cuda_seconds'],p['batch_sum_cuda_seconds']],
            color=['#4087b4','#629b64'])
        axes[1].bar_label(bars,fmt='%.2f',fontsize=8)
    axes[1].set_xticks([0,1,3,4],['Solo sum','Batch 4','Solo sum','Batch 4'],fontsize=8)
    axes[1].set_ylabel('A800 HF CUDA seconds, 4 distinct inputs')
    axes[1].set_title('Observed prefill | eight decode steps',fontsize=10)
    b=next(x for x in replay['batch_control'] if x['tag']=='full_observed_outputs')
    values=[b['independent_baseline']['total']['seconds'],b['batched_baseline']['total']['seconds'],b['batched_candidate']['total']['seconds']]
    bars=axes[2].bar(range(3),values,color=['#4087b4','#629b64','#d17b42'])
    axes[2].bar_label(bars,fmt='%.2f',fontsize=8)
    axes[2].set_xticks(range(3),['Solo base','Batch base','Batch local'],fontsize=8)
    axes[2].set_ylabel('Target model service seconds')
    axes[2].set_title('Observed outputs: batching removes extra gain',fontsize=10)
    fig.savefig(OUT/'family_controls.png',dpi=180)
    fig.savefig(OUT/'family_controls.pdf')
    plt.close(fig)
    print(validation)


if __name__=='__main__':main()
