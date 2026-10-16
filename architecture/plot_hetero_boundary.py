"""Export the recorded-work mix and conditional bandwidth threshold separately."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/architecture'


def main():
    data = json.loads((OUT/'hetero_boundary.json').read_text())
    selected = [s for s in data['streams'] if s['graph'].endswith('concurrent_queries')
                and s['maintenance_cap'] is None]
    fig, axes = plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    for column, stream in enumerate(selected):
        work = stream['m_work']
        labels = list(work)
        total_w = sum(v['one_pass_weight_bytes'] for v in work.values())
        total_f = sum(v['flops'] for v in work.values())
        x = np.arange(len(labels))
        axes[0,column].bar(x-.19,[100*v['one_pass_weight_bytes']/total_w for v in work.values()],.38,label='One-pass weight bytes')
        axes[0,column].bar(x+.19,[100*v['flops']/total_f for v in work.values()],.38,label='Projection FLOPs')
        axes[0,column].set(xticks=x,xticklabels=labels,ylabel='Share (%)',xlabel='GEMM M from legal replay',
                           title=stream['graph'].split('_')[0].capitalize()+' | recorded shapes, legal grouping')
        axes[0,column].legend(fontsize=8)
        for scenario in stream['scenarios']:
            points=[r for r in scenario['requirements'] if r['sink_ratio']==2]
            axes[1,column].plot([100*r['fraction'] for r in points],
                [r['required_local_bandwidth_ratio'] for r in points],marker='o',label=f"Encoded/raw = {scenario['ratio']:.1f}")
        axes[1,column].axhline(1,color='grey',linestyle=':')
        axes[1,column].set(xlabel='Near-memory share of remaining MAC capability (%)',
            ylabel='Required local / external bandwidth',title='Conditional break-even; bandwidth cost not modeled')
        axes[1,column].legend(fontsize=8)
        axes[1,column].grid(alpha=.2)
    fig.suptitle('Heterogeneous resource allocation: workload evidence and reduced-model requirements',fontsize=12)
    fig.supxlabel('External = 3.2 TB/s; unfolded sink = 6.4 TB/s; 3% extra MAC tax; 16 output shards.\n'
                  'Projection service only. No bank timing, SRAM tiling, physical area/energy or whole-agent speed claim.',fontsize=9)
    fig.savefig(OUT/'hetero_boundary.png',dpi=180)
    fig.savefig(OUT/'hetero_boundary.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
