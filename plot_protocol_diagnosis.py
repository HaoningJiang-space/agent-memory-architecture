"""Render measured protocol controls without treating host overhead as GPU work."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
r = json.loads((root/'evidence/phase3/observation_03.json').read_text())
p2 = json.loads((root/'evidence/phase2/completed/corrected_completed_observation.json').read_text())
measurements = r['grammar']['grammar_factorial']['measurements']
assert len(measurements)==12
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes = plt.subplots(1,3,figsize=(15,4.9))
positions = np.arange(3)
shapes = ['native','flat','branches']
for offset,minimum,color in [(-.18,0,'#377f91'),(.18,1,'#b76446')]:
    values = [statistics.median(x['cpu'] for x in measurements if x['variant']==f'{s}_min{minimum}') for s in shapes]
    axes[0].bar(positions+offset,values,width=.36,color=color,label='minLength: 1' if minimum else 'No minLength')
    for x,y in zip(positions+offset,values):
        axes[0].text(x,y*1.3,f'{y:.3f}' if y<1 else f'{y:.1f}',ha='center',fontsize=8)
axes[0].set_yscale('log')
axes[0].set_ylim(.005,300)
axes[0].set_xticks(positions,['Native','Flat IDs','anyOf IDs'])
axes[0].set_ylabel('CPU seconds per identical JSON stream (log)')
axes[0].set_title('(a) One schema rule dominates host work')
axes[0].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1),ncol=2)
axes[0].text(.5,-.18,'584 identical tokens; 2 repeats; no GPU inference',transform=axes[0].transAxes,ha='center',fontsize=8)

wall = p2['construction_wall_seconds']
extract = p2['llm_roles']['extraction']['seconds']
update = p2['llm_roles']['update']['seconds']
index = sum(v['seconds'] for v in p2['index_operations'].values())
times = [extract,update,index,wall-extract-update-index]
labels = ['Extract API','Update API','Index','Other']
colors = ['#377f91','#b76446','#588665','#aaa49a']
axes[1].barh(labels,[100*x/wall for x in times],color=colors)
for i,value in enumerate(times):
    axes[1].text(100*value/wall+1,i,f'{100*value/wall:.2f}%',va='center',fontsize=9)
axes[1].set_xlim(0,100)
axes[1].set_xlabel('Observed construction wall time (%)')
axes[1].set_title('(b) Index is not the measured bottleneck')
axes[1].invert_yaxis()
axes[1].text(.5,-.25,'32B: 81 checked chunks before failure; no quality claim\nZero-index-delay serial bound: 1.00179x',transform=axes[1].transAxes,ha='center',fontsize=8)

protocol = r['protocol']
names = ['native','flat_snapshot','flat_delta']
assert all(protocol[n]['calls']==10 for n in names)
for offset,key,color,label in [(-.18,'seconds','#377f91','API time'),(.18,'output_tokens','#b76446','Output tokens')]:
    values = [protocol[n][key]/protocol['native'][key] for n in names]
    axes[2].bar(positions+offset,values,width=.36,color=color,label=label)
    for x,y in zip(positions+offset,values): axes[2].text(x,y+.035,f'{y:.2f}x',ha='center',fontsize=8)
axes[2].set_ylim(0,1.75)
axes[2].set_xticks(positions,['Native','Flat snapshot','Flat delta'],rotation=13)
axes[2].set_ylabel('Relative to native on 10 frozen inputs')
axes[2].set_title('(c) A shorter protocol need not generate less')
axes[2].legend(frameon=False,fontsize=8,loc='upper left')
axes[2].text(.5,-.25,'Snapshot/delta: 1 truncated response each\nMutation agreement: 7/10 and 4/10; not semantic accuracy',transform=axes[2].transAxes,ha='center',fontsize=8)
fig.tight_layout(rect=(0,.16,1,1))
fig.text(.03,.025,'Pinned xgrammar 0.1.23 / Qwen3; host CPU is shared. '
         'No new-hardware speedup, same-quality ranking, or thermal conclusion.',fontsize=9)
out = root/'evidence/phase3'
fig.savefig(out/'protocol_diagnosis.png',dpi=180,bbox_inches='tight')
fig.savefig(out/'protocol_diagnosis.pdf',bbox_inches='tight')
