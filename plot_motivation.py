"""Export measured falsification results; no simulated architecture ranking."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path(__file__).resolve().parent
summary=json.loads((r/'outputs/run_summary.json').read_text())
mem=next(x for x in summary['runs'] if x['run']=='mem0_full_01')['phases']['construction']
cal=json.loads((r/'outputs/calibration_analysis.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'savefig.bbox':'tight'})
fig,axes=plt.subplots(1,2,figsize=(11,3.6),gridspec_kw={'width_ratios':[1,1.35]})
ax=axes[0]
output=mem['output_token_fraction']
ax.barh(1,1-output,color='#9ecae1',label='Input tokens')
ax.barh(1,output,left=1-output,color='#e78a72',label='Output tokens')
total=mem['step_seconds']
parts=[mem['first_token_wait_sum_seconds']/total,mem['after_first_token_sum_seconds']/total,
       mem['embedding_interval_union_seconds']/total]
parts.append(1-sum(parts))
left=0
for value,color,label in zip(parts,['#9ecae1','#e78a72','#74c6a5','#d9d9d9'],['First-token wait','After first token','Embedding','Other']):
    ax.barh(0,value,left=left,color=color)
    if value>0.1:ax.text(left+value/2,0,f'{100*value:.1f}% after first token',ha='center',va='center',fontsize=9)
    left+=value
ax.text((1-output)/2,1,f'{100*(1-output):.1f}% input',ha='center',va='center')
ax.text(1-output/2,1,f'{100*output:.1f}%',ha='center',va='center',fontsize=8)
ax.set_yticks([0,1],['Construction wall time','LLM token count'])
ax.set_xlim(0,1);ax.set_xlabel('Fraction (different denominators)')
ax.set_title('Mem0: 9 chunks before the failed update')
ax=axes[1]
cases=[x for x in cal['cases'] if x['batch']==1 or (x['batch']==32 and x['length']==512)]
x=np.arange(len(cases));w=.36
ax.bar(x-w/2,[c['eager_ms'] for c in cases],w,color='#bdbdbd',label='HF eager')
ax.bar(x+w/2,[c['graph_ms'] for c in cases],w,color='#3182bd',label='CUDA Graph')
ax.set_xticks(x,[f"B{c['batch']} / L{c['length']}" for c in cases],rotation=20)
ax.set_ylabel('Median forward time (ms)');ax.set_title('Same A800 and weights; synthetic shape calibration')
ax.legend(frameon=False)
for i,c in enumerate(cases):ax.text(i,c['eager_ms']+4,f"{c['speedup']:.1f}×",ha='center',fontsize=9)
fig.tight_layout(rect=[0,0.12,1,1])
fig.text(.16,.015,'First-token wait includes queue / network. After-first-token time is not a GPU kernel counter.',fontsize=8)
out=r/'outputs/figures';out.mkdir(exist_ok=True)
fig.savefig(out/'motivation_falsification.pdf');fig.savefig(out/'motivation_falsification.png',dpi=180)
