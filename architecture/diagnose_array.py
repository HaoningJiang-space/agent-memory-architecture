"""Attribute the existing cycle model before proposing hardware changes.

Bandwidth changes and ideal cycles are diagnostic counterfactuals, not free
physical upgrades. Continuous K streaming is an alternative scheduling bound,
not a claim that the original per-K-tile drain is inherent to systolic arrays.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from architecture.explore import average, load
from architecture.model import candidates, decoder, gemm, CYCLE_KEYS

ROOT=Path(__file__).resolve().parents[1]


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows={r['case']:r for r in output['results'] if r['group']=='mem0_update_prefix'}
    base=rows['baseline']
    selected=['baseline','bw_2_frozen','bw_8_frozen','ideal_array_frozen',
              'continuous_k_frozen','reshape_64x256_per_k_tile']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    left=0
    for key in CYCLE_KEYS:
        value=base['decode']['array_'+key+'_seconds']
        axes[0].barh(['Update decode'],[value],left=left,label=key)
        left+=value
    axes[0].set_xlabel('Attributed array service seconds')
    axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2)
    axes[1].barh(selected,[rows[k]['mean']['seconds'] for k in selected])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Update modeled service seconds')
    fig.suptitle('Diagnostic controls: same real requests; counterfactual hardware/mapping bounds')
    fig.savefig(ROOT/'evidence/architecture/array_diagnosis.png',dpi=180)
    fig.savefig(ROOT/'evidence/architecture/array_diagnosis.pdf')
    plt.close(fig)
    parts='\n'.join(f"| {key} | {base['decode']['array_'+key+'_seconds']:.6f} | "
                    f"{100*base['decode']['array_'+key+'_seconds']/base['decode']['compute_seconds']:.3f}% |"
                    for key in CYCLE_KEYS)
    controls='\n'.join(f"| `{key}` | {r['mean']['seconds']:.6f} | "
                       f"{base['mean']['seconds']/r['mean']['seconds']:.3f}× |" for key,r in rows.items())
    doc=f'''# P1：更新 Decode 的阵列周期来自哪里

固定 `HBM3E4_a16_128x128_s128` 和原有 9 次更新请求。下列是条件模型诊断，**不是 A800 利用率测量或可实现新引擎的收益**。脚本：`python3 -m architecture.diagnose_array`。

![周期和敏感性](evidence/architecture/array_diagnosis.png)

更新平均总服务时间 {base['mean']['seconds']:.6f} s，decode 为 {base['decode']['seconds']:.6f} s。Decode 的有效 FLOPs / 服务时间为 {base['decode']['flops']/base['decode']['seconds']/1e12:.3f} TFLOP/s，相对 524.288 TFLOP/s 理论峰值为 {100*base['decode_useful_peak_fraction']:.3f}%。该比例不能解释为晶体管活动率。

## 周期守恒分解

| 项 | decode 阵列服务秒 | 占阵列周期 |
|---|---:|---:|
{parts}

GEMM 的原公式是 `waves × (K + ceil(K/tk) × (R+C−2))`。依次计必要 MAC、M 维 padding、N 维 padding、软件 tile 对齐、最后一波未用阵列、每段 K 的填排空。Attention 另计掩码/尾块无效 MAC，并分别记录 QK/PV 波次。顺序决定 padding 的记账分配；总量严格守恒。分解是机会成本，不是这些 MAC 实际切换或消耗同等能量。

低 M 行闲置占约 59%，填排空占约 38%。**这里的硬限制来自当前映射与周期假设，不能直接推断实际硬件也按此方式排空。** `continuous_k` 允许 GEMM 跨 K tile 连续流动，仅每个输出波次填排空一次；attention 的独立 QK/PV 仍保留填排空。它是替代调度界，不是已经证明可在给定端口实现。

## 带宽、理想周期与重新映射

| 控制 | 更新服务秒 | 相对原映射 |
|---|---:|---:|
{controls}

`frozen` 保持每个 GEMM 的原 tile、OS/WS 与流量；`remap` 重新搜索原 tile 集，仍按 M 映射行、N 映射列，**没有穷举 K 维空间并行、归约网络或专门 GEMV 映射**。Attention tile 由 SRAM 决定，在带宽控制中不变。`ideal_array` 仅以有效 MAC / 全部 MAC 替换阵列周期，DRAM/SRAM 服务仍计入。2×–8× 带宽控制不计新增 PHY、功率或成本，因此仅用于瓶颈定位。矩形阵列保持总 MAC、SRAM 和名义端口预算；布线、时钟与真实面积尚未验证，不能加入物理可行方案排名。

最有信息量的结果是：**加带宽无收益；理想阵列也只有约 {base['mean']['seconds']/rows['ideal_array_frozen']['mean']['seconds']:.2f}× 的更新服务加速界**，随后访存成为限制。不能把 0.46% 的倒数当成加速空间。64×256 的模型改善有限，继续变窄反而恶化，说明填排空、列 padding 与 attention 映射共同约束结果。

当前应检验 B=1 投影和 attention 的映射/连续流动能力，而非凭“decode”标签增添 HB 带宽。单请求的低 M 问题也属于一般 LLM decode；尚未证明 Agent Memory 独有的新机制。现有 A800 微基准只能辅助检查量级，不能校准这些未实现阵列的逐周期行为。

## 验证与搜索边界

全部 baseline 标量与 P0 的已提交探索值逐项一致；所有组件满足周期分项之和等于阵列时间，逐算子限制归属之和等于总服务时间。测试另含手算 2×2 阵列样例、冻结映射流量不变及内存服务下界。

P/D 最小混合组合成本 608.768 + 277.568 = 886.336 > 850；可行集合仅含 LPDDR 对。没有 HB-on-Logic 多域模型。本轮阴性结论仅适用于这个受限空间。

机器可读结果：[array_diagnosis.json](evidence/architecture/array_diagnosis.json)。
'''
    (ROOT/'ARRAY_DIAGNOSIS.md').write_text(doc)


def run():
    spec,trace,_,_,receipts=load()
    hardware,_=candidates(spec)
    base=next(h for h in hardware if h.name=='HBM3E4_a16_128x128_s128')
    cases=[('baseline',base)]
    for factor in (.5,2,4,8):
        for mapping in ('frozen','remap'):
            cases.append((f'bw_{factor:g}_{mapping}',replace(base,bandwidth=base.bandwidth*factor,
                          freeze_mapping_bandwidth=base.bandwidth if mapping=='frozen' else 0)))
    for mode in ('ideal_array','continuous_k'):
        for mapping in ('frozen','remap'):
            cases.append((f'{mode}_{mapping}',replace(base,
                cycle_mode='ideal' if mode=='ideal_array' else 'scheduled',
                fill_mode='continuous_k' if mode=='continuous_k' else 'per_k_tile',
                freeze_mapping_bandwidth=base.bandwidth if mapping=='frozen' else 0)))
    for rows in (64,32,16):
        for mode in ('per_k_tile','continuous_k'):
            cases.append((f'reshape_{rows}x{16384//rows}_{mode}',replace(base,rows=rows,cols=16384//rows,fill_mode=mode)))
    results=[]
    samples=[]
    for group,workload in trace['groups'].items():
        for label,hw in cases:
            runs=[decoder(hw,trace['models'][c['model']]['config'],c['prompt_tokens'],c['completion_tokens'])
                  for c in workload['calls']]
            components={k:average(r['components'][k] for r in runs) for k in runs[0]['components']}
            record=dict(group=group,case=label,mean=average(r['total'] for r in runs),
                        prefill=average(r['prefill'] for r in runs),decode=average(r['decode'] for r in runs),
                        components=components,peak_tflops=2*hw.macs*hw.frequency/1e12)
            record['decode_useful_peak_fraction']=record['decode']['flops']/(2*hw.macs*hw.frequency*record['decode']['seconds'])
            results.append(record)
        median=sorted(workload['calls'],key=lambda c:c['prompt_tokens'])[len(workload['calls'])//2]
        cfg=trace['models'][median['model']]['config']
        h,q,kv,ff=cfg['hidden_size'],cfg['num_attention_heads']*cfg['head_dim'],cfg['num_key_value_heads']*cfg['head_dim'],cfg['intermediate_size']
        for name,k,n in [('Q',h,q),('K',h,kv),('V',h,kv),('O',q,h),('gate',h,ff),('up',h,ff),('down',ff,h),('head',h,cfg['vocab_size'])]:
            samples.append(dict(group=group,trace_id=median['trace_id'],op=name,shape=[1,k,n],**gemm(base,1,k,n)))
    old=json.loads((ROOT/'evidence/architecture/exploration.json').read_text())
    for r in results:
        if r['case']!='baseline': continue
        prior=next(x for x in old['request_profiles'] if x['hardware']==base.name and x['group']==r['group'])
        for key in prior['mean']:
            assert abs(r['mean'][key]-prior['mean'][key])<=max(1e-9,abs(prior['mean'][key])*1e-12),(r['group'],key)
    for r in results:
        for part in r['components'].values():
            assert abs(sum(part[f'array_{k}_seconds'] for k in CYCLE_KEYS)-part['compute_seconds'])<1e-8
            assert abs(sum(part[f'limited_by_{k}_seconds'] for k in ('compute','memory','sram'))-part['seconds'])<1e-8
    min_hbm=min(h.cost_units for h in hardware if h.memory=='HBM3E')
    min_lp=min(h.cost_units for h in hardware if h.memory=='LPDDR5X')
    output=dict(scope=__doc__,hardware=base.name,input_sha256=receipts,results=results,decode_operator_samples=samples,
        pd_search_bound=dict(min_hbm_cost=min_hbm,min_lpddr_cost=min_lp,min_mixed_pair_cost=min_hbm+min_lp,
                             budget=spec['budget']['cost_units'],hbm_pairs_possible=min_hbm+min_lp<=spec['budget']['cost_units']),
        limitations=['Array elapsed time is not fully utilized useful compute.',
          'Cycle parts are additive opportunity-cost bookkeeping, not transistor activity.',
          'Traffic and SRAM service remain in ideal-array controls; ideal arithmetic cannot remove their lower bounds.',
          'Reshaping keeps the nominal MAC, buffer and port budget; changed wires/clock/area require synthesis.',
          'No HB-on-Logic tier or placement path is modeled.'])
    output['diagnostic_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target=ROOT/'evidence/architecture/array_diagnosis.json'
    target.write_text(json.dumps(output,indent=2)+'\n')
    report(output)
    for r in results:
        if r['group']=='mem0_update_prefix':
            print(r['case'],round(r['mean']['seconds'],6),round(r['decode_useful_peak_fraction']*100,3))


if __name__=='__main__': run()
