"""Plot measured service fractions and a separately labelled capacity upper bound."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
work = json.loads((root/'evidence/phase2/workload_accounting.json').read_text())
capacity = json.loads((root/'evidence/final/model_capacity.json').read_text())
weights = {row['model']: row['weight_bytes'] for row in capacity['models']}
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'pdf.fonttype':42})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
keys = ['mem0_extract_prefix', 'mem0_update_prefix', 'embed_full_01_query', 'bm25_full_02_query']
labels = ['Mem0 extract*', 'Mem0 update*', 'embedRAG QA', 'BM25 QA']
position = np.arange(len(keys))
axes[0].bar(position-.18, [100*work['groups'][k]['totals']['output_token_share'] for k in keys],
            width=.36, label='Output / LLM tokens', color='#4695bb')
axes[0].bar(position+.18, [100*work['groups'][k]['totals']['after_first_time_share'] for k in keys],
            width=.36, label='After-first-token / request time', color='#cc795b')
axes[0].set_xticks(position, labels, rotation=13)
axes[0].set_ylabel('Share (%) — different denominators')
axes[0].set_ylim(0, 108)
axes[0].set_title('Token shares do not identify runtime bottlenecks')
axes[0].legend(frameon=False, fontsize=8, loc='upper right', bbox_to_anchor=(1, .82))
for i, key in enumerate(keys):
    value = work['groups'][key]['totals']
    for offset, field in [(-.18,'output_token_share'),(.18,'after_first_time_share')]:
        axes[0].text(i+offset,100*value[field]+1.7,f'{100*value[field]:.1f}',ha='center',fontsize=8)

embedding = weights['Qwen/Qwen3-Embedding-0.6B']
large = weights['Qwen/Qwen3-32B']
small = weights['Qwen/Qwen3-8B']
capacities = np.linspace(64, 160, 385)
kv_per_token = 262144
assumed_prompt = 20*1024
report = dict(scope='Capacity upper bound only, not a latency or iso-cost comparison. '
                   'Assumed 20,480-token 32B requests; no construction KV, workspace or fragmentation.',
              assumed_prompt_tokens=assumed_prompt, kv_bytes_per_token=kv_per_token, cases=[])
for name, size, color in [('Shared 32B + embed', large+embedding,'#387c6b'),
                          ('8B + 32B + embed',large+small+embedding,'#a34c3c')]:
    counts = np.maximum(0,np.floor((capacities*2**30-size)/(kv_per_token*assumed_prompt)))
    axes[1].step(capacities,counts,where='post',label=name,color=color,lw=2)
    for cap in [80,96,128,160]:
        free = cap*2**30-size
        report['cases'].append(dict(configuration=name, capacity_GiB=cap,
            weight_GiB=size/2**30, free_GiB=free/2**30,
            max_32b_kv_tokens=max(0,free//kv_per_token),
            max_assumed_requests=max(0,free//(kv_per_token*assumed_prompt))))
axes[1].axvline(80,color='gray',ls=':',lw=1)
axes[1].set_xlim(64,160)
axes[1].set_xlabel('Total memory capacity (GiB)')
axes[1].set_ylabel('Concurrent 20K-token 32B requests: upper bound')
axes[1].set_title('An extra small model consumes concurrency capacity')
axes[1].legend(frameon=False,fontsize=8,loc='upper left')
axes[1].grid(axis='y',alpha=.2)
fig.tight_layout(rect=(0,.15,1,1))
fig.text(.06,.065,'* Mem0: first 9 structurally completed chunks of a failed run; no complete quality claim.\n'
         'Service times include software/queue overhead. Right: BF16 weight and KV accounting; no measured speedup.',fontsize=9)
out = root/'evidence/phase2'
(out/'capacity_bounds.json').write_text(json.dumps(report,indent=2)+'\n')
fig.savefig(out/'first_principles_overview.png',dpi=180,bbox_inches='tight')
fig.savefig(out/'first_principles_overview.pdf',bbox_inches='tight')
