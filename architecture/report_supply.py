"""Render the costed, capacity-aware local-supply experiment."""
import hashlib
import json
from pathlib import Path

from architecture.supply_path import LocalPath, backing_route
from architecture.model import Hardware

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/architecture'
LABELS={'update':'8B 更新','raw_raw':'32B 原文问答','mem0_mem0':'32B 短记忆问答'}


def main():
    data=json.loads((OUT/'supply_exploration.json').read_text())
    for path,value in data['input_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
    groups=tuple(data['workloads'])
    def best(g,kind='strong_baseline',price=None,metric='seconds'):
        return next(x for x in data['best'] if x['group']==g and x['kind']==kind and x['price']==price and x['metric']==metric)
    def profile(name,g):
        return next(x for x in data['profiles'] if x['name']==name and x['group']==g)
    def design(name):return next(d for d in data['designs'] if d['name']==name)
    rows=[];path_rows=[];traffic=[];capacity=[];cold=[]
    for g in groups:
        base=best(g);b=profile(base['name'],g)
        for price in data['prices']:
            winner=best(g,'hb_local',price);p=profile(winner['name'],g)
            rows.append(f"| {LABELS[g]} | {price:.0f} | {base['value']:.4f} | {winner['value']:.4f} | {winner['relative']:.3f}× | {b['mean']['total']['energy_j']:.2f} → {p['mean']['total']['energy_j']:.2f} |")
        winner=best(g,'hb_local',100);p=profile(winner['name'],g);d=design(winner['name']);path=LocalPath(**d['path']);hw=Hardware(**d['hardware']);charges=d['charges']
        routes=path.route().capacities();bank=routes['bank']/1e12;port=path.port_bytes_s(hw)/1e12
        path_rows.append(f"| {LABELS[g]} | {path.banks} × {path.gb_s_per_bank:g} | {path.ports_per_bank} × 16 B/cycle | {bank:.3f} / {port:.3f} | {charges['controller_mm2']:.2f} / {charges['local_sram_mm2']:.2f} | {charges['cost_units']:.2f} / {charges['area_mm2']:.2f} / {charges['power_w']:.2f} | {charges['max_incremental_memory_cost']:.2f} |")
        t=p['mean']['total']
        traffic.append(f"| {LABELS[g]} | {t['backing_read_bytes']/2**30:.2f} | {t['hb_read_bytes']/2**30:.2f} | {t['shared_sram_bytes']/2**30:.2f} | {t['local_sram_bytes']/2**30:.2f} | {p['dominant']} |")
        cap=p['capacity_fraction_max'];fraction=p['fraction']
        capacity.append(f"| {LABELS[g]} | {p['model_weight_bytes']/2**30:.3f} GiB | {cap:.2%} | {fraction:.2%} | {p['placement_weight_bytes']/1e9:.3f} GB | {1/(1-cap):.3f}× |")
        c=p['cold_fill'];saved=base['value']-winner['value']
        count=f"{c['seconds']/saved:.4f}" if saved>0 else '无正时间回收点'
        cold.append(f"| {LABELS[g]} | {c['seconds']:.5f} | {c['energy_j']:.4f} | {count} |")
    demand=[]
    for sample in data['operator_samples']:
        demand.append(f"| {LABELS[sample['group']]} {sample['operator']} | {sample['shape']} | {sample['required_hb_bytes_per_bank_compute_cycle']:.2f} | {sample['actual_bank_bytes_per_compute_cycle']:.2f} | {sample['actual_local_read_port_bytes_per_cycle']:.0f} | {sample['result']['limiting']} |")
    controls=[]
    for tag in ('nominal','bank_skew2','bank_skew4','half_interface','single_port','payload256','clock_loss5pct','controller_area2x','no_resident_weights'):
        cells=[]
        for g in groups:
            r=next(x for x in data['controls'] if x['group']==g and x['tag']==tag)
            status='；超预算' if r['violations'] else ''
            cells.append(f"{r['mean']['total']['seconds']:.4f} ({r['dominant']}{status})")
        controls.append('| '+tag+' | '+' | '.join(cells)+' |')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    english={'update':'8B update','raw_raw':'32B raw QA','mem0_mem0':'32B short QA'}
    xs=np.arange(3);width=.24
    for i,g in enumerate(groups):
        values=[best(g,'hb_local',price)['relative'] for price in data['prices']]
        bars=axes[0].bar(xs+(i-1)*width,values,width,label=english[g])
        axes[0].bar_label(bars,fmt='%.2f',fontsize=8)
    axes[0].axhline(1,color='black',ls='--',lw=1)
    axes[0].set_xticks(xs,[str(int(p)) for p in data['prices']]);axes[0].set_xlabel('Unknown 10 GB HB memory/assembly price (cost units)')
    axes[0].set_ylabel('Service speedup vs best unified / frozen K2');axes[0].legend(fontsize=8)
    axes[0].set_ylim(0,1.5);axes[0].set_title('Conditional budget slices; no calibrated HB price')
    tags=('nominal','bank_skew2','single_port','payload256','no_resident_weights')
    for g in groups:
        values=[best(g)['value']/next(x for x in data['controls'] if x['group']==g and x['tag']==tag)['mean']['total']['seconds'] for tag in tags]
        axes[1].plot(range(len(tags)),values,'o-',label=english[g])
    axes[1].axhline(1,color='black',ls='--',lw=1);axes[1].set_xticks(range(len(tags)),['Nominal','2x skew','1 port','256 B','No copy'],rotation=20)
    axes[1].set_ylabel('Speedup, same candidate / placement; bypass allowed')
    axes[1].set_ylim(.8,1.4);axes[1].legend(fontsize=8);axes[1].set_title('Bank, port and in-flight-window sensitivity')
    fig.savefig(OUT/'supply_limits.png',dpi=180);fig.savefig(OUT/'supply_limits.pdf');plt.close(fig)
    count={k:sum(d['kind']==k for d in data['designs']) for k in ('unified_continuous_k','frozen_k2','hb_local')}
    doc=fr'''# 有代价的局部供数路径：容量、bank、接口与 SRAM

本轮把“免费提高 DRAM 带宽”替换为可逐项检查的供数模型。**当前条件下，10 GB HB 副本可使 8B 更新约 1.24×、32B 短问答约 1.15×；长原文路径没有稳健收益。控制器面积、HB 成本或 bank 分布变化会消除优势，尚不能宣称新架构已获得物理收益。**

对象仍是冻结的 9 条更新和 14 条配对问答。同一条请求在所有硬件上使用相同模型、精度、L/O 和算子；没有用语义质量下降、答案缩短或理想 MAC 周期制造加速。只对已经观察到的独立请求比较服务量，不宣称跨算法覆盖或完整 Agent 生命周期改善。

## 1. 共同预算与强基线

保持 [spec.json](architecture/spec.json) 的 **850 normalized cost、450 mm² logic、500 W、50 mm shoreline**。新增代码位于 [supply_path.py](architecture/supply_path.py)，旧 model/spec/mechanism/joint_diagnostics 和历史结果均不修改。

- {count['unified_continuous_k']} 个统一配置：连续 K；重新枚举全部 OS/WS tile，包含原有 HBM 和 LPDDR 容量方案。
- {count['frozen_k2']} 个已计价 K2 配置：直接复用上一轮 KShard，不增加 K 分片数量、不改变归约机制。
- {count['hb_local']} 个满足当前预算切片的局部供数点，另有 {len(data['rejected'])} 个被拒绝。每点的带宽、缓冲、控制器和未知 HB 价格同时计费。

每组的“强基线”取统一与 K2 中更快者。HB 端只复制 M=1 投影需要的部分权重，原容量层保留完整模型与 KV。**每个算子均可绕过 HB，读原有副本**；没有把不利局部路径强制给候选使用。映射和驻留比例以时延优先、能量次优选择，下面能量列也是该映射下的能量。

本轮对所有方案共同加入了 DMA ingress/egress 的共享 SRAM 访问，并从现有 SRAM 中预留有限在途响应缓冲；所以秒数与上一轮仅计消费端访问的数值略有变化。旧报告保留原模型口径，不应混用两轮分母。

## 2. 数据到底走哪条路径

驻留权重：`HB DRAM bank → HB interface → 双缓冲局部 SRAM → 原有 K2/统一计算块`。

未驻留权重、KV 与其他访问：`原 HBM/LPDDR → 原共享 SRAM → 原计算块`。

已知长输入矩阵仍走原始路径，保持其 OS/WS 权重复用；本轮不假定低 M 的局部映射同时最适合长输入。两个权重域按静态 K/N 切片并行供数，最后保留冻结 K2 的串行归约。没有额外 MAC。局部端口是成对单端口 SRAM bank 的 ping-pong，生产和消费都必须得到供数。

对于一条域内流，模型用

\[
B_{{\rm eff}}=\min(B_{{\rm bank}}/s,\ \eta B_{{\rm interface}},\ n_{{\rm bank}}n_{{\rm outstanding}}P/\tau),
\quad T_{{\rm stream}}=\tau+D/B_{{\rm eff}}.
\]

`s` 是最忙 bank 相对平均 bank 的负载系数；`P` 是每个在途传输的 payload；`τ` 是端到端请求延迟。在途窗口形成吞吐上限，同时保留一次启动时间，不把每个缓冲 refill 都强制排空。局部 SRAM 另受端口数×16 B/cycle×频率限制；共享 SRAM 合并计算端、DMA 填入和写回读取的访问。

\[
T_{{\rm op}}=\max(T_{{\rm compute}},T_{{\rm backing}},T_{{\rm HB\ bank}},T_{{\rm HB\ interface}},T_{{\rm window}},T_{{\rm local\ SRAM}},T_{{\rm shared\ SRAM}})+T_{{\rm reduction}}.
\]

这是乐观重叠的解析模型，不是 bank/NoC 时序仿真。backing 默认 16 个服务分区/package、每分区一个 16 KiB 在途请求，缓冲从已有 SRAM 扣除；这是公开的控制器情景，不是 HBM 厂商 bank 数。HB 默认每 bank 两个 4 KiB 在途请求、100 ns 延迟。均有明确带宽、窗口、bank skew 和端口控制。

## 3. 哪些是文献锚点，哪些仍是假设

[H²-LLM](https://doi.org/10.1145/3695053.3731008) §7.1/Table 6 给出每 bank HB 接口 6.4–51.2 GB/s；6.76 mm² PE 中控制器面积占比为 4.6%、10.7%、19.7%、40.2%；其 40 nm SRAM 密度为 2.72 mm²/MB，HB I/O 能量为 0.88 pJ/bit。这里直接保留控制器面积和 SRAM 密度锚点，没有悄悄缩放到现代节点。**混合既有逻辑模型与 40 nm 锚点仍不是已校准 PPA。**

本地 HybridSpec §VI.A 使用 10 GB、4 TB/s、408 mm² 的 HB 栈。本轮只用 **10 GB 作为容量情景**，并没有把它的 4 TB/s 免费加到候选上，也没有复制其完整芯片。局部 bank 数与接口速率、缓冲面积按本轮设计逐项计算。450 mm² 是现有逻辑面积预算，**不是总封装面积预算**；408 mm² 栈 footprint 不能因此被宣布可与基线拼接放下。布局、制造节点与封装都仍待验证。

HB 存储单元与组装价格缺少可比锚点，因此扫描 **40/100/200 cost units**，不是报价或确定成本。控制器、FIFO、地址生成器在此之外另收费。新区域静态功耗、1% 额外时钟损失、SRAM 0.1 pJ/bit 和局部线能量也是显式未校准参数。供数峰值功耗保守计入完整 HB I/O、局部 SRAM 读写及局部线活动，不只计算 MAC。

100-unit 切片的最快候选如下，Bbank 与 Bport 均为聚合有效 TB/s：

| 路径 | bank × HB GB/s | 每 bank 消费端口 | Bbank / Bport | 控制器 / FIFO mm² | cost / logic mm² / peak W | 允许的 HB 价格上限 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(path_rows)+f'''

M=1 代表算子的逐 bank 消费需求如下。需求列使用该映射的阵列计算周期作为分母；需求超过供数说明阵列必须等待，不能用计算峰值继续推算。

| 算子 | M,K,N | HB 需求 B/compute-cycle/bank | bank 可供 B/cycle | 本地消费端口 B/cycle | 实际主要限制 |
|---|---|---:|---:|---:|---|
'''+ '\n'.join(demand)+'''

“允许价格上限”是扣掉既有硬件、控制器与 FIFO 后的成本余量；没有把不知价格的器件当免费资源。更高价格导致配置退出预算，不能简单在获胜点上保持原带宽不变。

## 4. 有限 HB 容量是首要约束

| 路径 | 全模型驻留权重 | 10 GB 最大覆盖 | 选择的静态覆盖 | 复制到 HB | 仅绕过权重读的理想上界 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(capacity)+'''

最后一列只是在本均匀静态放置约束下，假设所有时间都来自原容量层权重读取、HB 无限快且没有其他成本时的 `1/(1−fmax)`，不是整个系统加速预测，也不是所有可能热权重放置策略的通用上界。对 32B，仅复制约 15.3% 权重就无法消除另外约 84.7% 权重的 backing 流量。真实算子还要读 KV、执行计算并归约，上界不会自动变成实测收益。

8B 在名义最优点没有用满 10 GB：静态复制约一半容量后，HB 与 backing 服务较为平衡；继续把更多权重放在有限 bank 供数的局部域会拖长该域时间。算法没有变，改变的是资源分配和数据放置。

这里的覆盖是均匀静态权重切片，不是“所有权重等概率缓存命中”的 LRU 假设。模型以连续比例近似已对齐切片，尚未实现物理 allocator。原完整模型和当前请求的逻辑 KV/激活都检查原容量层能否容纳；CUDA 预留和多模型同时驻留不在本轮范围。

## 5. 条件收益与负面区域

![供数与成本限制](evidence/architecture/supply_limits.png)

表内是目标架构模型服务秒和焦耳，均不是 A800 延迟或功率。

| 路径 | HB 未知价格 | 最快强基线 s | 最快局部候选 s | 速度比 | 模型能量 J，基线 → 候选 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

每个价格切片允许重新搜索预算内硬件，因此表是探索边界，不是已在未见算法上冻结的通用设计。**100-unit 切片中，更新仍未达到建议的 1.3× 路径筛选目标，短问答只有约 1.15×；200-unit 切片中该候选族应被拒绝，保留强基线更好。** 不需要为了保留 HB 强行让它获胜。

100-unit 候选平均每个完整请求的累计逻辑流量如下，不是容量、A800 硬件计数器或每 token 流量：

| 路径 | backing 读 GiB | HB 读 GiB | 共享 SRAM GiB | 新局部 SRAM GiB | 累计主要限制 |
|---|---:|---:|---:|---:|---|
'''+ '\n'.join(traffic)+'''

新增 SRAM 流量包含 resident 权重的填入与消费，未驻留权重仍完整计入 backing 和共享 SRAM。HB 不改变实际 L/O，不消除全部权重流量。

## 6. 允许反证：同候选、同静态放置的控制

下面固定 100-unit 最快候选及其静态驻留比例，不在每个敏感性点重新选择硬件或驻留；算子仍可合法绕过 HB。`no_resident_weights` 保留付费硬件及其静态功耗，只令全部读走 backing。

| 控制 | 8B 更新 s | 32B 原文 s | 32B 短记忆 s |
|---|---:|---:|---:|
'''+ '\n'.join(controls)+'''

bank skew、端口不足、在途 payload 过小都会使局部路径失去收益。HB 可 bypass，因此有些控制回到付出额外时钟/静态成本的基线附近，而不是被强迫无限变慢。控制器面积翻倍不必改变模型服务秒，却可能让名义胜出点超预算；不能继续把这种点放进正式排名。

## 7. 冷填充与能量边界

HB 是额外持久副本。下表单独列出一次模型分片填充的 backing 读、HB 写和静态开销。假定副本后续常驻；重载、抢占或频繁切换模型需要重复支付，不能只用 warm 数字。时间回收以相同组平均请求对比，不能当应用质量一致的生命周期结论。

| 路径 | 一次冷填充 s | 一次冷填充 J | 仅填充时间需要的平均请求数 |
|---|---:|---:|---:|
'''+ '\n'.join(cold)+'''

服务能量包含新增控制器/DRAM 静态功耗、HB 读、两次局部 SRAM 访问和局部互联。闲置驻留时间未知，因此没有自行虚构 idle duty cycle；完整系统能量仍需实际到达轨迹。

## 8. 下一步判断与复现

本轮提出的机制是**将部分 M=1 权重读分散到有成本的局部供数域，并按有限容量和速率共同决定放置**。它不是新 K 分片算法，也不是简单把带宽乘四。H²-LLM 已有局部 HB 计算及数据流探索，故这里仅是可检查的候选，尚未证明相对其完整机制有新颖性。

当前最有信息量的结果是容量与供数必须共同平衡；负面结果是收益不足够大、且对价格/面积/bank 服务敏感。继续投入前，应先校准控制器/局部 SRAM 端口、总 footprint 和 bank 行为，再用其他真实记忆算法检验固定设计。若有代价参数下依然低于筛选目标，停止扩展这一候选；不要增加 HBF、SSD 或热调度来补一个未成立的供数收益。

```bash
python3 supply_path_unittest.py
python3 -m architecture.supply_path
python3 -m architecture.report_supply
```

数值证据：[supply_exploration.json](evidence/architecture/supply_exploration.json)。模型每个配置、阶段、输入输出、驻留、冷填充、拒绝原因均可追溯；文件包含输入与冻结源码 SHA256。有限窗口/流量守恒/物理计费/容量/路由 fallback 的测试独立检查基本不变量。未增加 GPU 运行、模型下载、热推断或对外部索引的假定瓶颈。
'''
    (ROOT/'SUPPLY_PATH_RESULTS.md').write_text(doc)
    validation=dict(input_hashes_verified=len(data['input_sha256']),accepted_designs=len(data['designs']),
                    rejected_designs=len(data['rejected']),profiles=len(data['profiles']),requests={g:len(data['workloads'][g]) for g in groups},
                    budget=data['budget'],all_finite=True,traffic_and_time_conservation=True)
    for d in data['designs']:
        hw=d['hardware']
        for field,budget in [('cost_units','cost_units'),('area_mm2','area_mm2'),('peak_power_w','power_w'),('shoreline_mm','shoreline_mm')]:
            assert hw[field] <= data['budget'][budget]
    import math
    for p in data['profiles']:
        assert p['capacity_requirement']['total_bytes']<=p['backing_capacity_bytes']
        assert p['placement_weight_bytes']<=10e9+1e-5
        for phase in ('prefill','decode','total'):
            r=p['mean'][phase]
            assert all(math.isfinite(v) and v>=0 for v in r.values())
            assert abs(sum(v for k,v in r.items() if k.startswith('limited_by_'))-r['seconds'])<1e-7
            assert abs(r['shared_compute_bytes']+r['dma_sram_bytes']-r['shared_sram_bytes'])<1e-2
            assert abs(r['local_sram_bytes']-2*r['hb_read_bytes'])<1e-2
            reads=sum(r[k] for k in ('weight_read_bytes','kv_read_bytes','activation_read_bytes','partial_read_bytes'))
            writes=r['activation_write_bytes']+r['partial_write_bytes']
            assert abs(reads-r['backing_read_bytes']-r['hb_read_bytes'])<1e-2
            assert abs(writes-r['backing_write_bytes'])<1e-2

    validation['generator_sha256']=hashlib.sha256((ROOT/'architecture/report_supply.py').read_bytes()).hexdigest()
    (OUT/'supply_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation))


if __name__=='__main__':main()
