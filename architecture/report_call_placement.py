"""Render bounded placement evidence, retaining the physical-model limits."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT/'evidence/architecture/call_placement.json.gz'
    data = json.loads(gzip.decompress(path.read_bytes()))
    for name, sha in data['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha, name
    lines = ['# 完整调用、P/D 与驻留：第一组有界物理对照', '',
        '2026-10-16。本轮运行 CPU 条件模型，没有重新运行 A800 推理。旧 10 GB HB 副本继续关闭；不增加算法、量化、codec 或热参数。', '',
        '**结论：完整调用是值得保留的映射粒度，但不能仅靠改变映射粒度得到异构硬件收益。此次常规 LPDDR 容量域＋小型 HBM 域的实物参数情景，被更便宜的统一配置同时压低时间与能量。它是反例控制，不应直接成为下一项主架构。**', '',
        '这一结果只排除下述一个配置及当前执行模型，不否定完整模型异构，更不否定共享存储、不同封装或不同单位成本的供数组织。', '',
        '## 1. 比较的是什么', '',
        '复用被冻结的 SimpleMem / LightMem 四条 DAG：每个算法有顺序与并发三问两种场景。每条 DAG 的所有 LLM、Embedding 与 host 节点恰好消费一次。就绪时间来自目标模型中前驱完成，不来自 A800 API 到达时间。', '',
        '所有配置使用相同的连续 K、付费 K2 / M≤8 映射；更大合法 batch 走普通映射。实际输出边界控制 batch 收缩；无跨 prompt 拼接或未经证实的 KV 共享。', '',
        '| 对照 | 权重驻留与决策 |', '|---|---|',
        '| 统一 | 原 spec 中全部预算可行配置；常驻三个 checkpoint，另给仅驻留当前模型、逐次加载的控制 |',
        '| 完整调用 | C 放 32B＋Embedding，N 完整放 8B；不是部分权重副本 |',
        '| P/D | 8B 在 C、N 各有一份，C prefill、N decode；32B 留 C |',
        '| 完整调用或 P/D 的局部选择 | 同样复制 8B，在每个当前就绪 batch 上选预计完成最早的位置 |', '',
        '**最后一项使用已观察的输出长度，是有预知信息的局部贪心，既非在线算法，也非全局最优上界。** 为避免把更大的 batch 拆小来帮助 N，先选任一路径能容纳的最大就绪前缀，再选位置。它可能因改变后续就绪与合批而比固定映射差。', '',
        '**H²-LLM 共享 bank 的算子映射没有被伪装成此表的第四种调度策略。** 它拥有不同的权重归属和存储拓扑；已有 `hetero_boundary.py` 仅做投影筛查，缺少全链 attention、容量与竞争，不能放进本表冒充完整系统结果。', '',
        '## 2. 物理配置与预算', '',
        'C：16×128×128 阵列、64 MiB SRAM、8 组 LPDDR（128 GB）。N：8×128×128 阵列、32 MiB SRAM、1 组 HBM（24 GB）。两侧都计入原 spec 的固定逻辑和控制器，不默认共享。HBM 在这里是常规 HBM，**不是 HB-on-Logic**。', '',
        '跨域接口情景为 32 GB/s、8 mm²、8 W 常开功耗、5 pJ/bit；另计源 DRAM 读与目的 DRAM 写。满速传输再预留 1.28 W。接口数字未经综合/布线，仍属需校准参数。', '']
    bill = data['hetero_bill']
    lines += ['| 指标 | 双域情景 | 共同上限 |', '|---|---:|---:|']
    for key in ('area_mm2', 'cost_units', 'power_w', 'shoreline_mm'):
        lines.append(f'| {key} | {bill[key]:.3f} | {data["budget"][key]} |')
    lines += ['', f'两个域在接口之前只剩 **{bill["before_link_headroom"]["area_mm2"]:.3f} mm²** 面积余量；若接口占 16 mm²，此配置超限。成本为归一化单位，功率为条件包络；不是芯片签核结果。', '',
        '权重严格使用已有 storage 核对：8B 15.256 GiB、32B 61.024 GiB、Embedding 1.110 GiB。P/D 的额外 8B 副本为 15.256 GiB，32 GB/s 初始加载约多 0.512 s，另有内存写入和传输能量。24 GB 域不能执行完整 32B。', '',
        '这是“固定完整驻留集合”的初筛，尚未实现通用 KV 页迁移或动态模型缓存策略。统一侧另有单模型驻留控制，避免强迫它永远保留所有 checkpoint。', '',
        '## 3. 完整有限任务链结果', '',
        '下表时间单位 s，能量单位 kJ，均为条件模型数值。不是 A800 时延、生产吞吐或质量等价收益。统一最短时间配置在原候选集合内选择；还单列总成本不高于双域的最快统一点。', '',
        '| DAG | 最快统一 | 更便宜统一 | 完整调用双域 | P/D 双域 | 局部联合选择 |', '|---|---:|---:|---:|---:|---:|']
    summary = []
    for graph in sorted({r['graph'] for r in data['runs']}):
        rr = [r for r in data['runs'] if r['graph']==graph and r['attention_mode']=='frozen' and not r['free_kv_time']]
        unified = [r for r in rr if r['policy']=='unified']
        best = min(unified, key=lambda r:r['seconds'])
        cheap = min((r for r in unified if data['unified_devices'][r['design']]['cost_units'] <= bill['cost_units']), key=lambda r:r['seconds'])
        others = [next(r for r in rr if r['policy']==p) for p in ('whole','pd','joint')]
        selected = [best,cheap,*others]
        lines.append('| '+graph+' | '+' | '.join(f'{r["seconds"]:.3f} / {r["energy_j"]/1000:.3f}' for r in selected)+' |')
        summary.append(dict(graph=graph, entries=[dict(policy=r['policy'],design=r['design'],active_only=r['active_only'],
            seconds=r['seconds'],energy_j=r['energy_j']) for r in selected]))
    lines += ['', '最快统一点：4 HBM、16 阵列、128 MiB SRAM，成本 739.374、面积 339.374 mm²。更便宜统一点：4 HBM、8 阵列、32 MiB SRAM，成本 **609.687**、面积 209.687 mm²；双域成本 **628.861**。所列统一胜出点都选择常驻，单模型卸载控制未改善该记录的最优值。', '',
        '**所以差异不是统一方案买了更多成本预算。** 更便宜的统一点已经同时更快、更省能、面积更小。这不等于原候选集合是成熟架构的全局最优集合。', '',
        '## 4. 从第一性原理解释，而不是按角色分配硬件', '',
        '对于同一调用，整次 C/N 与 C→N 拆分的无排队服务量分别为：', '',
        r'\[T_C=P_C+D_C,\quad T_N=P_N+D_N,\quad T_{PD}=P_C+T_{KV}+D_N.\]', '',
        '因此 P/D 比整次 N 更快，至少需要 prefill 节省超过 KV 代价；还要偿还额外权重驻留/加载。这里 D 是整段实际 decode 时间，已包含真实 batch 收缩，不能再乘输出长度。', '',
        r'\[P_N-P_C>T_{KV}+\text{摊销的复制/加载代价}.\]', '',
        '如下是联合选择回放中首次出现的代表 batch；不是算法所有调用的均值：', '',
        '| 实际组 | B | C prefill | N prefill | C decode | N decode | KV 传输 |', '|---|---:|---:|---:|---:|---:|---:|']
    for graph, stage in [('simplemem_concurrent_queries','construction'),('lightmem_concurrent_queries','maintenance')]:
        r = next(r for r in data['runs'] if r['graph']==graph and r['policy']=='joint' and r['attention_mode']=='frozen' and not r['free_kv_time'])
        b = next(b for b in r['batches'] if b['kind']=='llm' and b['stage']==stage)
        opts = {x['route']:x for x in b['alternatives']}
        c,n,p = opts['C'],opts['N'],opts['PD']
        lines.append(f'| {graph.split("_")[0]} {stage} | {b["batch"]} | {c["prefill_seconds"]:.3f} | {n["prefill_seconds"]:.3f} | {c["decode_seconds"]:.3f} | {n["decode_seconds"]:.3f} | {p["kv_seconds"]:.3f} |')
    lines += ['', '**C 的 MAC 更多，却在这些输入处理上没有更快。** 真实吞吐还受有限 SRAM、权重/激活重读与 LPDDR 服务限制。“matrix side”只是名字，不能把峰值算力当成有效 prefill 性能。P/D 在这些组上连计算侧的正收益都没有，继续提高互联速率不能改变这个符号。', '',
        '**完整模型驻留解决了副本覆盖不足，却没有消除逐步读取权重。** N 与更便宜统一点都有 8 个阵列，但只有后者四分之一的名义 DRAM 带宽；它们使用同样的计算映射。把 8B 完整放下，并不意味着它能比统一设备更快执行。', '',
        '**粗粒度模型边界减少跨域激活/KV 传输，不会自动减少每个域内部的权重流量。** 如果它只增加一个资源池、复制固定逻辑，却没有改善单位成本的有效供数或计算，那么调度很难补偿这个物理差距。', '',
        '**容量需求大与适合低带宽并不等价。** 32B 在当前记忆算法下依然逐步生成，放 LPDDR 后这部分执行变慢；没有 HybridSpec 中 target 批量验证的算法转换。', '',
        '## 5. 控制、限制与论文边界', '',
        '- 每个图、每个配置都检查依赖、完整节点消费、O−1 token 守恒、容量和资源区间不重叠。接口拷贝时源/目的 KV 同时占空间；同一域不免费并行计算与 DMA。',
        '- P/D 同时提供零 KV 传输时间控制，字节和能量仍计入；SimpleMem 并发场景只减少约 0.287 s，LightMem 约 0.234 s。它有利于重叠传输，但不替代 Splitwise 的逐层管线实现。',
        '- 对所有统一点和双域策略同时将 attention 时间/能量置零，重新选择统一胜出点，当前排序仍未反转。这个极乐观控制用于识别旧 attention 映射影响，不证明任意真实 GQA kernel 的结果，也不补足未建模的 softmax/vector。',
        '- 无连续 batch refill、抢占、精确前缀缓存或生产到达分布；KV compaction 沿用免费控制。host 时间是共用实测残差锚点。小文本调用边界的序列化/运行时成本尚未逐项建模，不声称完整系统能量。',
        '- 主表是四条有限历史路径的完成时间，不是未来 Agent 的时延分位数。LightMem 压缩器分支未覆盖，旧向量行为未改；没有跨算法质量结论。', '',
        '[Splitwise §IV-C](https://arxiv.org/html/2311.18677v2) 已设计 P/D 分离及按层 KV 重叠传输。这里的保守资源预留不是其复现，不能据此宣称超过它。', '',
        '[Autellix §4](https://arxiv.org/html/2502.13965v1) 已研究程序依赖、已获得服务量及 KV 局部性。本文的离线局部选择没有它的非预知调度能力，不能把调用链调度本身列为新增贡献。', '',
        '[H²-LLM §4–5](https://chenzhangsjtu.github.io/files/2025-H2_LLM_ISCA25.pdf) 已联合考虑普通/近存通道、算子和数据映射。共享 bank 方案必须在相同权重归属、容量及竞争模型中比较；与两个私有存储域不是同一拓扑。', '',
        '[HybridSpec](https://doi.org/10.1109/ISCA66397.2026.00054) 的价值在于算法改变需求后才做硬件分工。我们当前没有证明 memory processing 对 32B decode 做了同类转换；不能照抄 draft/target 的存储分配。', '',
        '本地 PDF 阅读范围：HybridSpec pp.4/11；H²-LLM pp.6–7/10；Splitwise §IV-C。沿用 `hetero_literature.json` 的来源与哈希。', '',
        '## 6. 本轮决定', '',
        '**保留“完整调用/阶段边界＋驻留”作为架构评估层，不把它提前敲定为新的主机制。** 上一轮的算子局部选择确实需要这层约束，但新增调度器本身不能解释新的物理收益。', '',
        '下一项物理提案必须明确回答：相比允许卸载/批处理的统一设计，它以什么代价让被反复读取的完整模型权重获得更多有效供数，同时保存大模型容量？如果共享前端或控制能节省固定面积，要拆出可共享的实际模块，不能直接从每域 100 mm² 黑箱里免费扣掉一份。若采用共享存储，应回到 H²-LLM 的已覆盖机制作对照。', '',
        '本轮不为常规双设备换一组有利参数宣布成功；也不把一个负配置外推成异构无价值。当前尚未获得相对成熟架构的新物理机制，研究问题比“再做一个调用调度器”更窄且更具体。', '',
        '## 复现', '', '```bash', 'python3 -m architecture.call_placement_unittest',
        'python3 -m architecture.call_placement', 'python3 -m architecture.report_call_placement', '```', '',
        f'共 {len(data["runs"])} 条配置/调度控制记录；原始输出长度只用于冻结负载和明确标注的离线选择。',
        f'全部记录在生成时通过守恒检查；公开保留 {sum(r["detail_retained"] for r in data["runs"])} 条胜出统一点/双域控制的完整时间线，其余保留指标、节点数量与详细时间线 SHA256，可由相同命令重建。',
        '输入与依赖代码 SHA256、保留组的逐 batch 备选成本、节点完成时间、资源预留与容量记录保存在 `evidence/architecture/call_placement.json.gz`。', '']
    (ROOT/'CALL_PLACEMENT_RESULTS.md').write_text('\n'.join(lines))
    (ROOT/'evidence/architecture/call_placement_summary.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),graphs=summary),indent=2)+'\n')


if __name__ == '__main__':
    main()
