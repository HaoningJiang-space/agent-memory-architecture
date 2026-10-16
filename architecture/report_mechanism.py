"""Render crossed limits, the counted candidate, and bounded representation controls."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/architecture'


def main():
    data=json.loads((OUT/'joint_diagnostics.json').read_text())
    controls=json.loads((OUT/'representation_controls.json').read_text())
    paired=json.loads((OUT/'finite_prefix.json').read_text())
    reviews=json.loads((OUT/'two_case_review.json').read_text())
    assert controls['manifest']['original_manifest_sha256']==paired['summary']['manifest_sha256']
    for review in reviews['cases']:
        for strategy in ('raw','mem0'):
            row=next(x for x in paired['summary']['results'] if x['index']==review['index'] and x['strategy']==strategy)
            assert review[strategy+'_private_sha256']==row['private_receipt_sha256']
        row=next(x for x in controls['summary']['results'] if x['index']==review['index'])
        assert review['fine_private_sha256']==row['private_sha256']
    cross=data['crossed_controls']
    groups=('update','raw_raw','mem0_mem0')
    labels={'update':'8B update','raw_raw':'32B raw QA','mem0_mem0':'32B Mem0 QA'}
    modes=('scheduled','continuous_k','ideal')
    def point(group,mode,bw):
        return next(r for r in cross if r['group']==group and r['mode']==mode and r['bandwidth_factor']==bw and r['mapping']=='frozen')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,group in zip(axes,groups):
        matrix=np.array([[point(group,mode,bw)['mean']['seconds'] for bw in (1,2,4)] for mode in modes])
        speed=matrix[0,0]/matrix
        ax.imshow(speed,cmap='YlGnBu',vmin=1,vmax=6)
        for i,mode in enumerate(modes):
            for j,bw in enumerate((1,2,4)):
                r=point(group,mode,bw)
                ax.text(j,i,f"{matrix[i,j]:.3f}s\n{speed[i,j]:.2f}x {r['dominant']}",ha='center',va='center',fontsize=8,
                        color='white' if speed[i,j]>3.5 else 'black')
        ax.set_xticks([0,1,2],['1x BW','2x BW','4x BW'])
        ax.set_yticks([0,1,2],['Original','Continuous K','Ideal MACs']);ax.set_title(labels[group])
    fig.suptitle('Fixed mappings, unchanged SRAM: diagnostic limits, not budgeted hardware speedups')
    fig.savefig(OUT/'joint_limits.png',dpi=180);fig.savefig(OUT/'joint_limits.pdf');plt.close(fig)

    def best(group,kind,metric='seconds'):
        return next(r for r in data['best'] if r['group']==group and r['kind']==kind and r['metric']==metric)
    hardware_table=[]
    for g in groups:
        a=best(g,'unified_continuous_k');b=best(g,'counted_k_shard')
        hardware_table.append(f"| {labels[g]} | {a['value']:.6f} | {b['value']:.6f} | {a['value']/b['value']:.4f}× | `{b['design']}` |")
    bw_table=[]
    for g in (2,4,16,64):
        rows=[next(x for x in data['mapping_controls'] if x['tag']=='nominal' and x['group']=='mem0_mem0' and x['k_groups']==g and x['bandwidth_factor']==b) for b in (1,2,4)]
        bw_table.append(f"| {g} | "+' | '.join(f"{r['mean']['seconds']:.6f} ({r['dominant']})" for r in rows)+' |')
    traffic=[]
    for g in groups:
        r=point(g,'scheduled',1)['decode']
        traffic.append(f"| {labels[g]} | "+' | '.join(f"{r[k]/2**30:.3f}" for k in ('weight_read_bytes','kv_read_bytes','activation_read_bytes','partial_read_bytes','partial_write_bytes'))+' |')
    attribution=data['attribution'][0]
    attr_table='\n'.join(f"| {g} | {'实测形状' if g in ('raw_raw','mem0_mem0') else '反事实组合'} | {v:.6f} |"
                         for g,v in attribution['points'].items())
    sensitivity=[]
    for tag in ('nominal','bank_skew2','bank_skew4','overhead4','uncoalesced_refills','clock_drop_5pct_per_level'):
        r=next(x for x in data['mapping_controls'] if x['tag']==tag and x['group']=='mem0_mem0' and x['k_groups']==2 and x['bandwidth_factor']==1)
        sensitivity.append(f"| {tag} | {r['mean']['seconds']:.6f} | {r['mean']['energy_j']:.2f} | {r['charges']['cost_units']:.3f} |")
    doc=f'''# 算存联合机制：交叉诊断、计费映射与表示归因

本轮完成三项交付：阵列×带宽的 3×3 控制（固定/重新映射各一组）；有供数、归约、缓冲和接口代价的 K 分片候选；输入/输出四组合及 7 个细粒度原文查询。**单独改善阵列的 1.38× 不是算存联合改善的上界。** 但当前外存预算内，初版 K 分片相对连续 K 的强统一基线只有小幅增益。

旧 `7cd9029` 的模型、账本、配对数据均保持冻结。新代码在 [mechanism.py](architecture/mechanism.py) 中扩展，新结果与旧基线逐项对齐。没有续开 Mem0 构建或修改原始数据库。

## 1. 阵列改善后，高带宽开始有用

![交叉控制](evidence/architecture/joint_limits.png)

图中列是 DRAM 带宽倍数，行是原映射、连续 K、理想有效 MAC 周期。SRAM 服务始终保留；标签是逐算子限制归属的累计时间最多者，不能把整条路径视为单一算子的 max。

对于 update，固定映射下：

| 阵列控制 | 原带宽 | 2× 带宽 | 4× 带宽 |
|---|---:|---:|---:|
'''+ '\n'.join('| '+mode+' | '+' | '.join(f"{point('update',mode,b)['mean']['seconds']:.6f} s ({point('update',mode,b)['dominant']})" for b in (1,2,4))+' |' for mode in modes)+f'''

理想阵列＋4× 带宽相对原 5.108 s 的诊断空间约 **{point('update','scheduled',1)['mean']['seconds']/point('update','ideal',4)['mean']['seconds']:.2f}×**；SRAM 成为下一项限制。连续 K＋2× 带宽也有改善，随后再次受阵列限制。对应全部 9 次 update、14 次真实问答的逐请求数值及重新选 tile 的结果都保留在 JSON。

理想计算和 2×/4× 带宽没有计新增硬件成本，因此**只用于确定联合方向**。它们未进入下节的预算内方案排名，也不代表新 NPU 实测速度。

## 2. 从流量看：权重重要性上升，不能混同驻留和读取

以下是原映射下，每条请求整个 decode 阶段的平均 **DRAM 逻辑流量 GiB**；不是存储容量，也不是 A800 硬件计数器。权重/KV/激活/部分和读写分别核算，并逐算子对账到总 read/write bytes。

| 路径 | 权重读 | KV 读 | 激活读 | 部分和读 | 部分和写 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(traffic)}

短记忆的 KV 下降，但 32B 投影仍每步消费权重。模型中每 decode 步的投影权重读约 59.57 GiB；实际常驻权重为 61.024 GiB，两者不同，因为查表 embedding 等常驻参数不等于每步遍历的投影矩阵。长 prefill 还存在 OS 权重重读或 WS 部分和溢写，完整分类见 JSON。线性 K/V 投影生成的写入归入 activation_write，没有另外重复收取一次 KV 写入；KV_read 指 attention 读历史缓存。这里不能用常驻 61 GiB 直接宣称实测每步读取了 61 GiB。

## 3. 一个带代价的 K 分片候选

设计与公式见 [K_SHARD_DESIGN.md](K_SHARD_DESIGN.md)。固定一份权重，将物理阵列行分为 g 个 K 分区，每区输出 FP32 部分和；局部树合并后写回。仅 M=1 投影使用该映射，attention 保留原实现；若原映射更快则回退。两种模式共享 MAC 和 SRAM，新增复用器、地址发生器、归约通路及全局频率损失照常计入。

代码包含可执行的分片/二叉归约参考实现，覆盖 K/N 尾部；这证明运算语义及计数，不等于 RTL、浮点位精确或布局布线验证。

在原有 **850 成本、450 mm²、500 W、50 mm 接口**的条件预算内，枚举 {len(data['designs'])} 个配置，含 32 个强统一基线和 64 个带代价候选。此处不免费增加 DRAM 带宽。强基线允许连续 K 并独立选择最快 OS/WS tile。

| 真实工作负载 | 最佳强统一基线 s | 最佳 K 分片 s | 基线 / 候选 | K 分片配置 |
|---|---:|---:|---:|---|
{chr(10).join(hardware_table)}

因此，**现有外存条件下，不能用初版 K 分片宣称明显的架构突破**。原文路径甚至因新增控制/时钟成本略慢。需要保留的现象是：改变映射后额外带宽开始有效，而且最佳 K 分片数并非越大越好。

下表固定 HBM 参照与短记忆真实请求；2×/4× 列仍是未计带宽成本的诊断：

| K 分区数 | 原带宽 s | 2× 带宽 s | 4× 带宽 s |
|---|---:|---:|---:|
{chr(10).join(bw_table)}

过多 K 分区增加 FP32 部分和与归约传输，也无法绕开固定 SRAM 端口。所有候选都有明确额外硬件代价，但面积/布线/频率系数尚未综合校准；下面的敏感性用于暴露这一局限。均为参照配置、g=2、短记忆路径。

| 控制 | 服务秒 | 模型能量 J | 成本单位 |
|---|---:|---:|---:|
{chr(10).join(sensitivity)}

面积系数变化可能只影响成本/能量而不影响该点时间，不能以时间稳定证明 PPA 稳健。搜索空间仍只有既有 HBM/LPDDR 配置；没有 HB-on-Logic 域、层间通信或新封装，不能把 4× 带宽解释成已设计好的 HB Stack。

## 4. 输入缩短与输出改变分别贡献多少

在同一原 HBM 映射上，对每个问题组成四个 L/O 组合，实际两端保留其真实输出；中间两点只用于核算归因。

| L 来源_O 来源 | 证据类型 | 模型服务秒 |
|---|---|---:|
{attr_table}

平均两种改变顺序的边际差额，输入贡献 {attribution['input_credit_seconds']:.3f} s（{100*attribution['input_fraction']:.1f}%），输出贡献 {attribution['output_credit_seconds']:.3f} s（{100*(1-attribution['input_fraction']):.1f}%）。这是四点模型分解，不是实测因果干预，也不是质量等价的加速。连续 K 与 K 分片下的分解亦保存，结论量级相近。

在本轮资源网格中，四种形状组合都仍选择 HBM 系列；没有发现内存技术排序反转。跨表示的强统一基线与候选的相对差异也很小。**输入变化改变工作量和瓶颈，不自动证明硬件最优点发生了显著反转。**

## 5. 质量和普通检索控制

[独立报告](REPRESENTATION_CONTROLS.md) 完成两例检查及 512-token 原文检索。原文平均输入从 21151 降到约 2570 token，说明 2.1 万输入不是原文路径不可避免的成本；但时间锚点和多条支持事实可能丢失。这些质量问题限制跨策略收益，**不阻止在同一真实请求上比较硬件映射**。

H²-LLM §4.2 已讨论 K/N 划分的部分和合并与输入复制取舍，不能将 K 分片本身作为新贡献；TETRIS §3.3–4 的存储/累加和资源取舍也是相关先例。这里的候选只是验证工具，尚未复现两者的完整硬件。文献与差异见设计说明。

模型仍未覆盖 softmax/normalization 等 vector kernels、实际 bank/row-buffer 时序及并发到达队列；面积、网络和频率没有综合校准。因此约 1–2% 的预测增益尤其不能当作稳健 PPA 结果。部分和重排也尚未通过完整 LLM 的数值/质量验证。

可保留的机制假设是：**记忆表示减少阅读历史和 KV，却保留大模型投影；映射、外存供给、SRAM 及归约必须联合匹配。** 下一步若要主张新设计，应把有利的联合诊断点替换成真正满足预算的供数路径，并证明相对强统一映射的增量；当前小增益不能替代这一步。

复现：`python3 mechanism_unittest.py`，`python3 -m architecture.joint_diagnostics`，`python3 -m architecture.report_mechanism`。机器结果：[joint_diagnostics.json](evidence/architecture/joint_diagnostics.json)。
'''
    (ROOT/'MECHANISM_RESULTS.md').write_text(doc)

    costs=[]
    for g in (2,4,16,64):
        d=next(x for x in data['designs'] if x['name']==f'HBM3E4_a16_128x128_s128_k{g}')
        c=d['charges']
        costs.append(f"| {g} | {c['adders']} | {c['address_generators']} | {c['partial_buffer_bytes']/1024:.0f} | "
                     f"{c['added_logic_area_mm2']:.3f} | {c['clock_factor']:.4f} | {c['power_envelope_w']:.2f} |")
    design='''# K 分片与局部归约：可执行映射及资源合同

这是一项**有资源计费的分析候选**，不是已完成的 RTL/PPA 实现，也不声称 K 分片是新机制。实现：[mechanism.py](architecture/mechanism.py)，独立参考运算和计数检查：[mechanism_unittest.py](mechanism_unittest.py)。

1. 对 M=1 的 `y=xW`，每个 R×C 阵列分为 g 个 `(R/g)×C` 子阵列。每组读取连续 K 分片，保持 FP32 部分和。不同阵列处理不同 N tile。K/N 尾部补齐计入周期。
2. 权重预排布为 `[N tile, K group, K offset, column]`，等价于对 K 连续分组的列块布局；普通模式通过地址发生器读取同一份数据。没有复制一份额外权重。一次性权重重排不计入本次请求服务模型，不能据此声称全生命周期无代价。
3. K 分区使用双缓冲持续流动。一个 N tile/K 分区对应一条连续 DMA stream，buffer refill 不强制重启外部传输；额外提供逐 refill 开新传输的保守控制。供数不是零时间：沿用有限 DRAM 带宽、时延、32 个 outstanding groups 以及 SRAM 服务限制。
4. 所有组完成后，将局部 FP32 部分和送入二叉树。每阵列 16 个归约输出 lane，FP32 adder 假设 3 周期流水，8 周期同步；链路总宽每阵列 64 B/cycle。计算/供数阶段与最终归约阶段串行，不默认完美重叠。
5. 只在原映射和新映射都计入上述代价后择快。长 M 和 attention 保持普通映射；整颗装有扩展的设备仍承担新增逻辑、静态功耗和降频，不能只给受益算子收费。

对于单个 M=1 投影，记 `q=ceil(N/C)`、`waves=ceil(q/arrays)`、`Kp=ceil(K/g)`：

```text
array_cycles = waves * (Kp + R/g + C - 2)
DRAM weights = 2*K*N bytes
DRAM input   = 2*K*q bytes
DRAM output  = 2*N bytes
FP32 extra adds = (g-1)*N
reduction link traffic = 4*N*(2*g-2) bytes
tree_cycles = waves * (ceil(C/16) + 3*ceil(log2(g)) + 8)
T = max(array, DRAM, SRAM during MACs)
  + max(tree, reduction link, partial-buffer SRAM)
```

部分和写入、读出与最终结果写入均计局部 SRAM 字节及能量；不假设免费片上归约，也不把它错误计成 DRAM 流量。每阵列已有 SRAM 端口按 32 banks×16 B/cycle 的聚合上限解释，保留 2×/4× bank skew 控制。bank 平衡布局尚未经过逐周期 DRAM/SRAM 验证。g 增大时峰值供数需求 `2*g*C*f` 可以超过端口能力，此时服务时间强制停顿。

双缓冲权重、输入与部分和都必须装入现有 SRAM；部分和预留 `arrays*2*4*g*C` bytes，不能扩充容量。MAC 数量、DRAM 容量和外部带宽保持候选配置原值。原 HBM 参照下的额外资源如下：

| g | FP32 adders | 地址发生器 | 部分和预留 KiB | 新增逻辑 mm²估计 | 时钟比例 | 功率包络 W |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(costs)+'''

面积按 adder 数、两路 BF16 operand 复用器 bit 数、地址发生器计费；系数为公开待校准假设：adder=0.5×原 MAC 面积，mux=2e-7 mm²/bit，地址发生器=0.002 mm²。局部互连包含在端口/mux 的集总假设内，**没有独立布线面积或线长提取**。时钟比例 `1/(1+0.005*log2(g))` 是敏感性参数，不是 STA 结果。额外 FP32 累加、输入广播、树传输及复用器动态能量，以及附加逻辑泄漏均收费；完整参数可从 `KShard` 读取。功率用保留原峰值加新增活动的保守包络。

预算判断来自这些明确但未校准的系数，因此“通过预算”只表示分析模型内通过。4× 开销、较大时钟惩罚、bank skew 和不合并 refill 的控制都保留。未验证的布线可行性、数值重排对完整 LLM 的影响及能耗系数不能被小幅预测收益覆盖。

参考运算采用 FP32 分片和树合并，随机尾部/小矩阵与独立 NumPy matvec 比较。它不等于 BF16 输入/FP32 FMA 的逐位等价，也没有在新硬件上重跑问答。

文献边界：

- [H²-LLM，ISCA 2025](https://leesou.github.io/publications/2025-4-h2llm)：本地 PDF §4.2/Eq.1 已同时考虑 K 和 N 划分，权衡结果合并和输入复制；其 §4.1/4.2 还包含 bank/PE 绑定、全局输入 buffer 和局部权重/输出 buffer。本候选是统一阵列内的行分组，未复现它的 HB 域或跨通道 DSE，因此不能将胜过原低 M 映射称为胜过 H²-LLM。
- TETRIS，ASPLOS 2017，本地 PDF §3.3–4：累加位置、SRAM/PE 面积和数据移动共同决定取舍。其 DRAM 累加器的工艺与位宽不同，没有把论文的面积数字直接移植成本模型。

两份本地 PDF 的路径和 SHA256 已在 `evidence/phase2/overview_provenance.json` 冻结。此次核查关注上述具体章节，不据一般“近存”标签预设 HB 优势。
'''
    (ROOT/'K_SHARD_DESIGN.md').write_text(design)

    source_rows=paired['summary']['results'];fine=controls['summary']['results']
    stats=[]
    for name,rows in [('原文大块',[r for r in source_rows if r['strategy']=='raw']),('原文512',fine),('Mem0',[r for r in source_rows if r['strategy']=='mem0'])]:
        stats.append(f"| {name} | {statistics.mean(r['input_tokens'] for r in rows):.1f} | "
                     f"{statistics.mean(r['output_tokens'] for r in rows):.1f} | "
                     f"{statistics.mean(r['final_live']['kv_bytes'] for r in rows)/2**30:.3f} | "
                     f"{sum(bool(r['metrics']['substring_exact_match']) for r in rows)}/7 |")
    pairs=[]
    for f in fine:
        raw=next(x for x in source_rows if x['index']==f['index'] and x['strategy']=='raw')
        pairs.append(f"| {f['index']} | {raw['input_tokens']} | {f['input_tokens']} | {f['output_tokens']} | "
                     f"{int(raw['metrics']['substring_exact_match'])} / {int(f['metrics']['substring_exact_match'])} |")
    review='''# 表示控制：小块原文与两例质量定位

没有再运行 Mem0 construction。保留上一轮 19 段历史、同一组 7 个按标注支撑选出的问题、同一 Qwen3-32B、共同 prompt、日期、BF16、SDPA、greedy 和 256-token 上限。所有 7 次新回答自然 EOS，无截断。

普通细粒度控制：用 Qwen tokenizer 将每个原文片段切为 512-token 窗口、64-token overlap，共 189 段；Qwen3-Embedding-0.6B 编码，FAISS top-5。没有摘要生成或 Mem0 提取，未给每个窗口额外附加原会话日期。原始片段和位置信息只留远端。

| 表示 | 平均输入 | 平均输出 | 最终活跃 KV GiB | substring match |
|---|---:|---:|---:|---:|
'''+ '\n'.join(stats)+'''

普通小块检索已消除约 88% 的长输入。**不能把 2.1 万 token 当作原文检索不可避免的成本**；反过来，词面匹配和下面的支持信息遗漏也不支持把小块视作无损替代。这些只是 7 个有限范围问题，不能排序三种算法的总体质量。

| 问题 | 大块输入 | 小块输入 | 小块输出 | 大块/小块词面命中 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(pairs)+'''

## 两例人工检查

检查了全部 114 条持久事实、Mem0 实际 top-10、原文回答与参考；关键词筛选只用于定位，**不以关键词未匹配代替语义审查**。

- **问题 30**：所问事件及其时间不在 114 条事实中，实际 top-10 也没有，Mem0 明确回答信息不足。这不是仅扩大 top-k 就能补回的遗漏。原文大块的最终答案与参考时间长度不符；词面匹配却因中间日期/数字包含单字符参考“4”而返回 true。因此原先 4/7 中这一项是相对于参考答案的误命中，不能称为答对。小块检回了事件描述，却未携带事件日期，仍回答信息不足。
- **问题 39**：回答需要合计两次休息的时长；两个时长都未保存在 114 条事实中，Mem0 top-10 只有相关的使用习惯和未来计划，回答信息不足。大块原文具有两项支持信息并正确合计；小块只检回其中一次，输出单次时长，属于支持覆盖不足。

这两例没有显示“答案只是换了一种表述”的情况。可定位到 **持久事实中缺失必要信息**；此次没有进一步追查 extraction/update 哪一步丢失，也没有开启协议修复。细粒度原文同时暴露了时间锚点和跨片段覆盖问题；本轮控制到此结束。

人工结论不是官方判分或新自动 evaluator，未检查其余五题的语义正确性，不能简单把 4/7 改成另一个“正确率”。判断与冻结输入哈希一一对应，见 [two_case_review.json](evidence/architecture/two_case_review.json)。两例原始输入、答案、完整检查资料和哈希留在远端，新公开 JSON 只有计数、测量和哈希。

构建原文索引本身不免费：189 个向量、重叠后的输入及索引构建实测时间均保存在 manifest/远端 embedding trace；这次没有将旧 88-write 构建费与 7 个有限查询拼接成生命周期交点。没有下载新模型或数据，也没有向外部 LLM 发送问题。

运行：[representation_controls.py](architecture/representation_controls.py)。证据：[representation_controls.json](evidence/architecture/representation_controls.json)。同一请求的硬件比较不以跨策略质量等价为前提；跨策略总体收益则仍需质量约束。
'''
    (ROOT/'REPRESENTATION_CONTROLS.md').write_text(review)


if __name__=='__main__': main()
