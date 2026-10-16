"""Use measured paired lengths without promoting the pilot to equal quality."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from architecture.explore import average,load
from architecture.model import candidates,decoder,GIB

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/architecture'


def main():
    source=OUT/'finite_prefix.json'
    evidence=json.loads(source.read_text())
    assert evidence['manifest']['status']=='complete' and evidence['summary']['status']=='complete'
    rows=evidence['summary']['results']
    assert len(rows)==14 and len({(r['index'],r['strategy']) for r in rows})==14
    assert not any(r['capped'] for r in rows)
    spec,trace,_,_,receipts=load()
    hardware,_=candidates(spec)
    names=['HBM3E4_a16_128x128_s128','LPDDR5X8_a16_128x128_s128','LPDDR5X12_a16_128x128_s128']
    config=trace['models']['Qwen/Qwen3-32B']['config']
    profiles=[]
    for name in names:
        hw=next(h for h in hardware if h.name==name)
        for strategy in ('raw','mem0'):
            cases=[]
            for r in rows:
                if r['strategy']!=strategy:
                    continue
                result=decoder(hw,config,r['input_tokens'],r['output_tokens'])
                assert result['kv_bytes']==r['final_live']['kv_bytes']
                cases.append(dict(index=r['index'],**result))
            profiles.append(dict(hardware=name,strategy=strategy,
                mean=average(c['total'] for c in cases),prefill=average(c['prefill'] for c in cases),
                decode=average(c['decode'] for c in cases),cases=cases))
    write=dict(input_sha256={**receipts,'evidence/architecture/finite_prefix.json':hashlib.sha256(source.read_bytes()).hexdigest()},
               scope='Real query lengths and outputs; structurally valid guarded Mem0 prefix, not quality-equivalent algorithms.',profiles=profiles)
    (OUT/'finite_query_model.json').write_text(json.dumps(write,indent=2)+'\n')
    data={k:[r for r in rows if r['strategy']==k] for k in ('raw','mem0')}
    def mean(k,field):
        return statistics.mean(r[field] for r in data[k])
    def kv(k):
        return statistics.mean(r['final_live']['kv_bytes'] for r in data[k])
    capacity=json.loads((ROOT/'evidence/calibration/model_capacity.json').read_text())
    w8=next(r['weight_bytes'] for r in capacity['models'] if r['model']=='Qwen/Qwen3-8B')
    difference=kv('raw')-kv('mem0')
    quality={k:sum(bool(r['metrics']['substring_exact_match']) for r in data[k]) for k in data}
    paired=[]
    for index in sorted({r['index'] for r in rows}):
        a=next(r for r in data['raw'] if r['index']==index)
        b=next(r for r in data['mem0'] if r['index']==index)
        paired.append(f"| {index} | {a['input_tokens']} / {a['output_tokens']} | {b['input_tokens']} / {b['output_tokens']} | "
                      f"{a['final_live']['kv_bytes']/GIB:.3f} / {b['final_live']['kv_bytes']/GIB:.3f} | "
                      f"{int(a['metrics']['substring_exact_match'])} / {int(b['metrics']['substring_exact_match'])} |")
    residency=[]
    for k in data:
        def span(values):
            return f'{min(values)/GIB:.3f}–{max(values)/GIB:.3f}'
        residency.append(f"| {k} | {span([r['prefill_live']['allocated_bytes'] for r in data[k]])} | "
                         f"{span([r['peak_interval_allocated_bytes'] for r in data[k]])} | "
                         f"{span([r['prefill_live']['reserved_bytes'] for r in data[k]])} | "
                         f"{span([r['prefill_live']['nvml_outside_reserved_estimate'] for r in data[k]])} |")
    modelrows=[]
    for p in profiles:
        modelrows.append(f"| `{p['hardware']}` | {p['strategy']} | {p['prefill']['seconds']:.3f} | "
                         f"{p['decode']['seconds']:.3f} | {p['mean']['seconds']:.3f} |")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for k in data:
        axes[0].scatter([r['index'] for r in data[k]],[r['input_tokens'] for r in data[k]],label=k)
        axes[1].scatter([r['index'] for r in data[k]],[r['final_live']['kv_bytes']/GIB for r in data[k]],label=k)
    axes[0].set_ylabel('Actual prompt tokens (log scale)');axes[0].set_yscale('log')
    axes[1].set_ylabel('Measured final live KV (GiB)')
    for ax in axes:
        ax.set_xlabel('Question index: annotated support in first 19 chunks');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Finite-prefix pilot: all 7 eligible pairs; quality equivalence unverified')
    fig.savefig(OUT/'finite_prefix.png',dpi=180);fig.savefig(OUT/'finite_prefix.pdf');plt.close(fig)
    doc=f'''# P2：真实有限历史配对与驻留

已完成 **7 对、14 次真实 Qwen3-32B 问答**。平均输入从原文 {mean('raw','input_tokens'):.1f} 降到 Mem0 {mean('mem0','input_tokens'):.1f} token；平均输出分别为 {mean('raw','output_tokens'):.1f} / {mean('mem0','output_tokens'):.1f} token，均自然 EOS 结束。这里使用真实输出长度，未沿用原文答案长度。

**尚未证明质量等价**：原仓库 substring match 为 {quality['raw']}/7 对 {quality['mem0']}/7。它是词面指标，不是 LongMemEval 官方语义判分；完整 exact/F1/ROUGE 均保存在 JSON。小样本差异也不能证明总体劣势。不能将压缩或服务量下降写成同质量收益，也不能据此给 Mem0 做总体排名。

![配对输入和 KV](evidence/architecture/finite_prefix.png)

## 真实状态与问题范围

- 从 `phase4/mem0_8b_guarded` 的前 19 个结构检查通过的片段恢复 {evidence['manifest']['audit']['accepted_state_records']} 条事实。这是已有 guarded protocol 产生的状态，**不是原生协议完整成功，也不等于语义正确性已经通过**。
- 失败的第 20 段仍执行一次 ADD；仅在副本中删除这次 ADD，索引中每个 UUID/文本与截止前历史重放逐条相等。一个无时间戳 DELETE 由前后 SQLite 提交行的时间夹定在有效范围。原始文件哈希结束时全部不变。
- 从原来的 60 个问题中预先选择**所有**标注答案支撑 turn 在前 19 段出现的问题，共 7 个；只标准化空白，不按答案结果筛选。未标注的其他 turn 可能在范围外，因此这是按标注支撑定义的有限范围，而非完整数据集。
- 原文侧对同一 19 段构建 FAISS、top-5；Mem0 对恢复的 Qdrant 状态 top-10。沿用 Qwen3-Embedding-0.6B 和原生 query instruction。
- 两侧共同使用受控回答 prompt、数据集 question date、BF16 Qwen3-32B、禁用 thinking、greedy、最多 256 token。这是检索表示的配对控制，不冒充未经修改的端到端原生 QA。

| 问题序号 | 原文输入 / 输出 | Mem0 输入 / 输出 | 原文 / Mem0 最终 KV GiB | 词面匹配 原文 / Mem0 |
|---|---:|---:|---:|---:|
{chr(10).join(paired)}

## 同时驻留与瞬时峰值

GPU 1 上 HF eager + SDPA 串行执行；GPU 0 仅做 embedding。32B 权重实际唯一 storage 为 {evidence['manifest']['weight_bytes']/GIB:.3f} GiB，查询时 **8B 未驻留**。没有把两卡构造成新异构系统。

每次 forward 后同步 CUDA，记录同一采样点的权重 storage、实际 cache tensor storage、其他存活 PyTorch 分配、allocator reserved/slack 和 NVML 进程占用。最后 KV 严格等于 `(input + output − 1) × 262144` bytes。初始 prefill 已产生首 token，最后输出尚未再次进入 cache。

| 策略 | prefill 后 allocated GiB | 单次 forward 内 allocated 峰值 GiB | prefill 后 reserved GiB | NVML 减 reserved GiB |
|---|---:|---:|---:|---:|
{chr(10).join(residency)}

`allocated = weights + actual KV + other_live` 在同一采样点核对；other_live 包含模型 buffer、输入、logits 及其余活跃张量。reserved−allocated 是分配器保留空间，不能算活跃 KV。NVML 差值包含 CUDA/库和分配器外内存，采样粒度也可能产生偏差，不能精确归因于某个 runtime 模块。

峰值使用每次 forward 的 `max_memory_allocated`，**没有把各项独立峰值相加**。但内部峰值时刻的 KV/临时张量未逐算子采样，因此不能用“峰值减最后 KV”声称测出精确 workspace。这里没有 CUDA Graph、vLLM KV 预分配、chunked prefill 或多请求调度；同步采样时间也不是优化 serving SLA。新 NPU 的 workspace 仍应由其 buffer/数据流推导，不能直接用这些 HF 数值替换一个统一 4 GiB 常数。

## 重新检验容量与计算边界

实际平均每对查询节省 {difference/GIB:.3f} GiB 最终 KV。只计算额外 8B 权重与此项 KV，代数交点为 {w8/difference:.3f} 个平均活跃请求，取整 {math.ceil(w8/difference)} 个。**这是容量情景，不是测出的并发数**：仍需计构建 KV、workspace、状态时序及质量；本次离线查询根本不需常驻 8B。查询复用次数与同时活跃数仍分开。

将每个真实输入和输出送入现有条件模型，得到：

| 配置 | 表示 | Prefill 模型服务秒 | Decode 模型服务秒 | 总模型服务秒 |
|---|---|---:|---:|---:|
{chr(10).join(modelrows)}

这批真实短输入可以检验“语义记忆改变主模型的执行形状”，但所有时间仍是未校准 NPU 模型，且当前映射对 B=1 有 [已解释的阵列效率限制](ARRAY_DIAGNOSIS.md)。完整生命周期回收点没有拿这 7 个问题与原先 88 次构建情景直接拼接。

在 HBM 参照配置中，原文 prefill 占约 {100*profiles[0]['prefill']['seconds']/profiles[0]['mean']['seconds']:.1f}%，Mem0 查询 decode 则占约 {100*profiles[1]['decode']['seconds']/profiles[1]['mean']['seconds']:.1f}%。这给出了比固定“角色 → 硬件”更具体的线索：**记忆表示本身会改变主模型的执行阶段占比**。阶段占比来自模型，形状来自实测；尚不表示最优物理配置已反转。

原来 6.327× 的容量差异仍未恢复成可信 speedup：现在有真实长度和单请求驻留，但缺少同质量在线并发及目标 NPU 的 buffer 生命周期。不能把本次单请求可运行、或 512-token 乐观控制的 1×，外推成实际多请求 chunked prefill 的结论。

证据：[finite_prefix.json](evidence/architecture/finite_prefix.json)、[真实长度的条件模型](evidence/architecture/finite_query_model.json)。原文、答案、向量、数据库与逐步原始记录仅留在 moe-server；公开文件只含数值和哈希。
'''
    (ROOT/'FINITE_PREFIX_RESULTS.md').write_text(doc)


if __name__=='__main__':
    main()
