"""Validate the bounded replay ledger and regenerate its decision report/figure."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

from architecture.model import Hardware

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/architecture'


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7),(a,b)


def main():
    summary=json.loads((OUT/'batch_closure.json').read_text())
    packed=(OUT/'batch_closure_traces.json.gz').read_bytes()
    assert hashlib.sha256(packed).hexdigest()==summary['trace_archive_sha256']
    data=json.loads(gzip.decompress(packed))
    for p,digest in data['input_sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    for key in ('graphs','designs','comparisons','input_sha256','budget','limits'):
        assert data[key]==summary[key],key
    designs={d['name']:d for d in data['designs']}
    graphs={g['name']:g for g in data['graphs']}
    family=json.loads((OUT/'family_summary.json').read_text())
    sources={a['algorithm']:a for a in family['algorithms']}
    for d in designs.values():
        hw=d['hardware']
        actual={k for k,v in data['budget'].items() if hw['peak_power_w' if k=='power_w' else k]>v}
        assert actual==set(d['violations'])
    flow_rows=0
    for r in data['runs']:
        d=designs[r['design']];assert not d['violations']
        g=graphs[r['graph']];nodes={n['id']:n for n in g['nodes']}
        a=sources[r['graph'].split('_')[0]]
        assert set(r['node_starts'])==set(r['node_completions'])==set(nodes)
        assert {n['source_id'] for n in nodes.values() if n['kind']=='llm'}=={c['id'] for c in a['calls']}
        assert {n['source_kernel'] for n in nodes.values() if n['kind']=='embedding'}==set(range(len(a['embedding_kernels'])))
        assert all('start' not in n and 'end' not in n for n in nodes.values())
        for n in nodes.values():
            assert r['node_completions'][n['id']]+1e-10>=r['node_starts'][n['id']]
            assert all(r['node_starts'][n['id']]+1e-10>=r['node_completions'][p] for p in n['deps'])
        consumed=[];busy=[];hist=Counter()
        for b in r['batches']:
            consumed.extend(b['node_ids']);busy.append((b['start'],b['end']))
            row=b['service'];assert all(math.isfinite(v) and v>=0 for v in row.values())
            close(row['seconds'],b['end']-b['start'])
            close(sum(row[k] for k in ('weight_read_bytes','kv_read_bytes','activation_read_bytes','partial_read_bytes')),
                  row['backing_read_bytes']+row['hb_read_bytes'])
            close(row['activation_write_bytes']+row['partial_write_bytes'],row['backing_write_bytes'])
            close(sum(v for k,v in row.items() if k.startswith('limited_by_')),row['seconds'])
            close(row['shared_compute_bytes']+row['dma_sram_bytes'],row['shared_sram_bytes'])
            for p in b['segments']:hist[p['batch']]+=p['count']
            if b['kind']=='llm':
                close(sum(p['batch']*p['count'] for p in b['segments']),sum(nodes[i]['O']-1 for i in b['node_ids']))
            flow_rows+=1
        assert Counter(consumed)==Counter(n['id'] for n in nodes.values() if n['kind']!='host')
        for f in r['cold_fills']:
            assert f['bytes']<=d['path']['capacity_bytes']+1e-5
            busy.append((f['start'],f['start']+f['seconds']))
        busy.sort()
        assert all(x[1]<=y[0]+1e-9 for x,y in zip(busy,busy[1:]))
        close(r['initial_load_seconds']+sum(b-a for a,b in busy)+r['gpu_idle_seconds'],r['seconds'])
        close(r['host_active_seconds'],g['host_anchor_seconds']*r['host_scale'])
        assert r['resident_checkpoint_bytes']<=r['peak_backing_bytes']<=d['hardware']['capacity_bytes']
        hw=Hardware(**d['hardware'])
        load_energy=r['resident_checkpoint_bytes']*8*5e-12+hw.static_w*r['initial_load_seconds']
        accel=load_energy+sum(b['service']['energy_j'] for b in r['batches'])+sum(f['energy_j'] for f in r['cold_fills'])+hw.static_w*r['gpu_idle_seconds']
        close(accel,r['accelerator_energy_j']);close(20*r['seconds'],r['host_energy_j'])
        close(accel+r['host_energy_j'],r['energy_j'])
        compact=next(x for x in summary['runs'] if (x['graph'],x['design'],x['maintenance_cap'])==(r['graph'],r['design'],r['maintenance_cap']))
        for key in compact:
            if key not in ('decode_step_M_histogram','llm_batches','embedding_batches'):assert compact[key]==r[key],key
        assert compact['decode_step_M_histogram']=={str(k):v for k,v in sorted(hist.items())}
    keyed={(r['graph'],r['maintenance_cap'],r['design']):r for r in data['runs']}
    assert len(keyed)==len(data['runs'])==len(data['comparisons'])
    for c in data['comparisons']:
        r=keyed[c['graph'],c['maintenance_cap'],c['design']]
        b=keyed[c['graph'],c['maintenance_cap'],'baseline']
        close(c['speed_ratio'],b['seconds']/r['seconds'])
        close(c['energy_ratio'],b['energy_j']/r['energy_j'])
        feasible=r['seconds']<=b['seconds']+1e-12
        assert c['baseline_deadline_feasible']==feasible
        if feasible:
            idle_power=Hardware(**designs[r['design']]['hardware']).static_w+20
            close(c['energy_ratio_at_baseline_deadline'],b['energy_j']/(r['energy_j']+(b['seconds']-r['seconds'])*idle_power))
        else:assert c['energy_ratio_at_baseline_deadline'] is None
    checks=dict(input_hashes=len(data['input_sha256']),graphs=len(graphs),runs=len(data['runs']),
        checked_service_rows=flow_rows,all_neural_nodes_consumed_once=True,dependency_readiness_verified=True,
        nonoverlapping_gpu_and_fill=True,traffic_time_energy_conserved=True,capacity_and_budget_checked=True,
        trace_archive_sha256=summary['trace_archive_sha256'],
        scope='Internal consistency only; not physical calibration or semantic quality validation.')
    (OUT/'batch_closure_validation.json').write_text(json.dumps(checks,indent=2)+'\n')

    scenarios=[(g,c) for g in graphs for c in ([5,None] if g.startswith('lightmem') else [None])]
    def run(g,c,d):return next(r for r in data['runs'] if (r['graph'],r['maintenance_cap'],r['design'])==(g,c,d))
    def comp(g,c,d):return next(r for r in data['comparisons'] if (r['graph'],r['maintenance_cap'],r['design'])==(g,c,d))
    def label(g,c):return ('SimpleMem' if g.startswith('simple') else 'LightMem')+(' / 并发问题' if 'concurrent' in g else ' / 顺序问题')+(f' / 维护上限 {c}' if c else ' / 维护不设上限' if g.startswith('light') else '')
    lines=['# 小批量供数候选：闭环判定', '',
        '**决定：结束当前固定 10 GB HB 权重副本候选作为主方案的推进，冻结为负面对照。** '
        '支持 M=2–8 后确实恢复了部分收益，但完整已记录路径的优势对既有 bank/interface 不确定性不稳健。'
        '不继续通过低 M 特例、新算法、更大 HB、更多 K 分片或热管理延长这个候选。共同工作负载与核算模型保留。', '',
        '本轮只重放 a8bf9c1 已有 SimpleMem/LightMem 数值记录；没有新增 A800 实验。下述时间和焦耳全部是条件目标模型结果，'
        '不是实测新芯片性能，也不代表完整 Agent 任务质量已验证。', '',
        '## 一、同一物理组织扩展到 M=1–8', '',
        '固定 K=2、MAC 数、HB 容量、bank、接口与旧放置比例。两个 K 分区分别形成 M×N 部分结果，再用原归约通道逐行合并。'
        '每块权重只从 DRAM 读取一次并服务 M 行；X 分发、权重行间转发、部分和和 SRAM 服务均计费。'
        'M>8 合法请求保持大 batch，通过原映射回退，不拆成 8 来帮助候选。', '',
        '128×128 阵列分成两个 64×128 区域；一块 tile 的输入/权重双缓冲与部分和须同时放入既有 SRAM。'
        '部分和占 `2×4×2×M×128` 字节/阵列；HB 权重 tile 必须能放入局部 FIFO 的一半，另半用于双缓冲。'
        '归约只有原有 lanes，M 行依次占用；没有免费增加端口或归约宽度。', '',
        '强统一基线同样获得 M=1–8 映射。两边各增加 32 个行控制器、0.064 mm²，频率乘 1/1.005；'
        '控制器 0.32 W 加面积漏电按整个运行期保守计入。上述控制成本是显式未校准假设。'
        '独立数值检查覆盖 M=1/2/4/8、非整除 K/N，与 FP64 矩阵乘比较；这验证运算关系，不等于 RTL/PPA。', '',
        '| 设计 | 成本单位 | 逻辑面积 mm² | 峰值功率包络 W | 预算 |',
        '|---|---:|---:|---:|---|']
    for name in ('baseline','old_m1','small_m','controller_area2x'):
        d=designs[name];h=d['hardware'];lines.append(f"| {name} | {h['cost_units']:.3f} | {h['area_mm2']:.3f} | {h['peak_power_w']:.3f} | {'排除：成本超 850' if d['violations'] else '满足'} |")
    lines+=['', '共同上限为 850 cost units、450 mm²、500 W 和 50 mm shoreline；这是同一预算约束内的比较，'
        '不是实际花费相同。没有为新候选增加预算，也没有重新选择 HB 价格、容量或权重放置。', '',
        '## 二、依赖与完整账本', '',
        '- SimpleMem 四个构建窗口同时就绪，保留真实 4→3→2→1 输出结束边界；每个问题内部的规划/检索/回答链顺序执行。',
        '- LightMem 保留原生逐次 add_memory 的构建 flush；29 个维护决策读同一快照。分别重放原线程池上限 5 和不设人为上限的合法 29 请求控制。',
        '- 三个问题按原 harness 顺序执行；另做三问题在构建完成后同时就绪的受控场景。后继只随目标模型前驱完成而就绪，没有将 A800 API 到达时间冻结为新芯片时间。',
        '- 28/56 次 LLM 调用和 31/261 个 Embedding batch 全部消费一次。主机服务用已记录操作区间减去嵌套神经调用，残余 host 时间作为共同初始控制；不把 API 时间按加速比缩放。',
        '- 三个 checkpoint 全驻留 backing；HB 同时仅保存一个模型的副本。包含共同初始权重加载、模型切换重填、设备空闲功耗和全程 20 W CPU 控制。Embedding 使用 backing，不驱逐已有 HB 副本。', '',
        '调度器采用已就绪静态 batch，批次中不插入新请求。容量约束可拆批，8 并非调度上限。'
        '这是一组可复查的保守阶段 DAG；时间区间仅帮助识别原生阶段及 host 时长，不被解释为唯一可能的最优调度。', '',
        '| 场景 | 强基线 s | 旧 M=1 s | 小 batch s | 基线/候选时间 | 基线/候选能量 |',
        '|---|---:|---:|---:|---:|---:|']
    for g,c in scenarios:
        b,o,n=(run(g,c,d) for d in ('baseline','old_m1','small_m'));v=comp(g,c,'small_m')
        lines.append(f"| {label(g,c)} | {b['seconds']:.3f} | {o['seconds']:.3f} | {n['seconds']:.3f} | {v['speed_ratio']:.3f}× | {v['energy_ratio']:.3f}× |")
    lines+=['', '能量比小于 1 表示候选更耗能。SimpleMem 名义能量增加约 2.0–4.3%；LightMem 名义能量减少约 7.9%。'
        'LightMem 新小 batch 映射相对旧 M=1 候选的增量很小，其名义收益主要仍来自原生顺序构建；不将它重新包装成不能合批的目标特例。', '',
        'SimpleMem 四窗口构建本身（不含其他阶段与 cold fill）：', '',
        '| 设计 | 服务秒 | 模型能量 J |', '|---|---:|---:|']
    for d in ('baseline','old_m1','small_m','bank_skew2'):
        row=run('simplemem_sequential_queries',None,d)['batches'][0]['service']
        lines.append(f"| {d} | {row['seconds']:.3f} | {row['energy_j']:.2f} |")
    lines+=['', '旧报告的强基线为 37.895 s；本轮强基线同样获得已付费小 M 映射后为 37.752 s。'
        '因此不能把候选的映射改进全部算作 HB 独有收益。原独立执行 57.822 s 只是历史工作量对照，不作为新硬件加速分母。', '',
        '## 三、固定设计的反例与判定', '',
        '| 控制 | 完整路径时间比范围 | 完整路径能量比范围 | 判定 |', '|---|---:|---:|---|']
    for d in ('small_m','bank_skew2','half_interface','clock_loss5pct'):
        v=[comp(g,c,d) for g,c in scenarios]
        lines.append(f"| {d} | {min(x['speed_ratio'] for x in v):.4f}–{max(x['speed_ratio'] for x in v):.4f}× | {min(x['energy_ratio'] for x in v):.4f}–{max(x['energy_ratio'] for x in v):.4f}× | {'名义结果' if d=='small_m' else '既有不确定性控制；不调参'} |")
    lines+=['', 'bank skew 从 1 变 2，或接口效率从 0.8 变 0.4，都会使新增路径优势消失，并增加完整路径能量。'
        '这两个控制受同一供数上限限制，不能当作两份独立实验证据。控制器面积加倍则使成本超预算。', '',
        '另以强基线完成时刻为共同截止时间：较快设备保持供电直到该时刻，候选没有能量优势。'
        '这是明确的 always-on 策略控制，不证明采用 power-gating/DVFS 后也无能量空间；不过名义能量优势本身已在 bank/interface 控制下消失，'
        '因此本轮不转向能量叙事来保留当前候选。', '',
        '**结论是当前候选缺少值得继续物理校准的稳健净收益，不是所有 HB 架构无价值。** '
        '10 GB 均匀副本仅覆盖约 15.3% 的 32B 权重，本轮没有改变这一容量边界。'
        '共同执行形状、合法批处理记录和模型可用于以后独立提出的新机制；当前方案到此收尾。', '',
        '![固定设计的批处理闭环](evidence/architecture/batch_closure.png)', '',
        '## 四、证据边界与复现', '',
        '以下限制约束结果解释，并不转化为延长当前候选的任务清单：有限历史/三个问题；LightMem 未覆盖过滤器分支且保留已知旧向量行为；'
        '不证明跨算法质量等价；未建模 continuous batching、prefix reuse、精确 NoC/bank 时序或热；保留 midpoint attention 和免费 batch compaction；'
        'CPU 为共同实测时长锚点与功率控制，非新 CPU 预测。物理参数尚未校准，当前模型也不是击败 H²-LLM/MemExplorer 的证明。', '',
        f"自动对账覆盖 {checks['runs']} 次重放、{checks['checked_service_rows']} 个服务账本；检查依赖、节点一次性消费、非重叠设备服务、容量、预算、流量、能量和输入 SHA。",
        '完整数值事件保存在压缩 JSON 中，没有原始历史、prompt、答案或数据库。代码错误及影响见 [CODE_AUDIT.md](CODE_AUDIT.md)。', '',
        '```bash', 'python3 -m unittest batch_closure_unittest family_unittest supply_path_unittest mechanism_unittest architecture_unittest accounting_unittest protocol_controls_unittest',
        'python3 -m architecture.batch_closure', 'python3 -m architecture.report_batch_closure', '```', '',
        '- [摘要与来源哈希](evidence/architecture/batch_closure.json)',
        '- [完整数值事件压缩包](evidence/architecture/batch_closure_traces.json.gz)',
        '- [自动对账回执](evidence/architecture/batch_closure_validation.json)', '']
    (ROOT/'BATCH_CLOSURE_RESULTS.md').write_text('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    x=np.arange(len(scenarios));labels=['S seq','S conc','L seq 5','L seq 29','L conc 5','L conc 29']
    for shift,d,color,title in [(-.2,'small_m','#286e9b','Paid small M'),(.2,'bank_skew2','#cc784b','2x bank skew')]:
        axes[0].bar(x+shift,[comp(g,c,d)['speed_ratio'] for g,c in scenarios],.38,label=title,color=color)
        axes[1].bar(x+shift,[comp(g,c,d)['energy_ratio'] for g,c in scenarios],.38,color=color)
    for ax in axes[:2]:
        ax.axhline(1,color='black',lw=1);ax.set_xticks(x,labels,rotation=35,ha='right',fontsize=8)
        ax.set_ylim(.85,1.26);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Baseline / candidate (>1 favors candidate)')
    axes[0].set_title('Whole recorded path: time');axes[0].legend(fontsize=8)
    axes[1].set_title('Whole recorded path: energy')
    for i,(g,c) in enumerate([('simplemem_concurrent_queries',None),('lightmem_concurrent_queries',None)]):
        r=next(r for r in summary['runs'] if (r['graph'],r['maintenance_cap'],r['design'])==(g,c,'baseline'))
        h={int(k):v for k,v in r['decode_step_M_histogram'].items()}
        values=[h.get(1,0),sum(v for k,v in h.items() if 2<=k<=8),sum(v for k,v in h.items() if k>8)]
        axes[2].bar(np.arange(3)+(i-.5)*.35,values,.35,label=['SimpleMem','LightMem'][i],color=['#286e9b','#629b64'][i])
    axes[2].set_xticks(range(3),['M=1','M=2–8','M>8']);axes[2].set_yscale('log')
    axes[2].set_title('Legal target decode batches');axes[2].set_ylabel('Steps (log scale)');axes[2].legend(fontsize=8)
    fig.suptitle('Fixed 10 GB copy: nominal gains fail existing supply uncertainty controls',fontsize=12)
    fig.savefig(OUT/'batch_closure.png',dpi=180)
    fig.savefig(OUT/'batch_closure.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
