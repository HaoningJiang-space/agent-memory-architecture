# LightMem 路径补测：算法分支会改变硬件工作量

2026-10-16。**本轮实际运行了三个有边界的 A800 对照；没有复现整篇论文，也没有评价完整任务准确率。** 原始历史、prompt、响应与数据库留在远端。公开数据为 [lightmem_paths.json.gz](evidence/architecture/lightmem_paths.json.gz)。

结论：此前 flat 路径不能代表包含压缩、主题分段和缓冲的 LightMem。在同一小段历史上，恢复这些分支显著减少了生成式抽取/整合工作，却增加了文本编码工作。硬件资源配比必须依据有效算法路径重新判断；不能把关闭软件机制后产生的大量 Decode 当成不可消除的需求。

## 1. 实验只改变什么

固定同一历史的前 48 条按时间排列的消息、相同三个检索探针、Qwen3-8B 构建、Qwen3-32B 回答、Qwen3-Embedding-0.6B 编码。LLM 为 BF16、temperature=0；三个探针不保证能从这段历史回答。LightMem 源码固定 `8449d574df6bae1bdf3314a1564da65e2f37e046`。

| 对照 | 预压缩 | 主题分段 | STM 阈值 |
|---|---|---|---:|
| flat512 | 关闭 | 关闭 | 512 |
| full512 | 开启，保留比例 0.6 | 开启，共享 Encoder，attention layers 8–11 | 512 |
| full768 | 与 full512 相同 | 与 full512 相同 | 768 |

这里的 full 指**恢复本次所检验的两个分支**，不是完整论文配置：仍采用本仓库的 dense 8B/32B、flat extraction、hybrid messages 和有限历史，没有换成论文中的 MoE 模型、完整数据集或评价器。512 与 768 只是局部缓冲控制，不声称与论文全部超参一致。

Encoder 为 `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`，固定 revision `5f0c82792b7ea14c6484e015b6a072009496b7f2`；实际 177,341,186 参数、709,364,744 参数字节、FP32。配置/词表走 HF 镜像，权重走 ModelScope 镜像，按固定 HF revision 的 LFS SHA256 验证相同字节。下载回执随公共数据保留，模型不入 Git。

每个对照执行一次，顺序运行，间隔请求清理两个独立 vLLM 服务的 prefix cache；内部仍允许前缀复用。CUDA event 和 Linear hook 用来记录 Encoder 路径，具有测量开销。模型服务预先启动，初始化时间和之后的墙钟分开保存。A800 是测量平台，不是待证明的异构芯片。

## 2. 实测工作量

| 记录 | flat512 | full512 | full768 |
|---|---:|---:|---:|
| 抽取 LLM 调用数 | 24 | **4** | **4** |
| 抽取输入 token | 30,408 | 12,069 | 12,069 |
| 抽取输出 token | 10,268 | 1,689 | 1,679 |
| 维护 LLM 调用数 | 37 | **0** | **0** |
| 维护输入/输出 token | 23,942 / 1,564 | 0 / 0 | 0 / 0 |
| 回答 LLM 调用数 | 3 | 3 | 3 |
| 回答输入/输出 token 总和 | 3,129 / 105 | 3,035 / 104 | 3,040 / 104 |
| Embedding API/模型 batch 数 | 264 | 72 | 72 |
| Embedding 输入 token 总和 | 5,986 | **11,787** | **11,777** |
| 最终记忆记录数 | 261 | 38 | 38 |
| 初始化之后的完整记录墙钟，s | 161.608 | 30.866 | 30.742 |

64 → 7 是全部 LLM 调用数，而不是只数抽取。两个 full 对照仍执行了维护的检索和元数据更新；“维护 LLM 为零”不等于整个维护过程没有成本，更不代表长期运行永远不需要整合。

**Encoder 新增工作不能漏计。** full512 包含 49 次压缩前向、4 次分段前向，CUDA event 时间分别为 0.715 和 0.052 s。压缩实际张量为 B=1/2/3、L=512；分段为 B=1、L=289/449/461/480。对应 Linear 计数约 6.088 / 0.285 TFLOPs；这是实际形状的逻辑运算核算，未含 attention、softmax、非线性，也不是硬件计数器。

API batch 变少但 Embedding token 增加，原因是恢复的主题处理也会调用文本编码器。它不能由记忆条目数或 API 次数推算。Encoder 的实际 M 为 B×L；B=1 不等于逐 token 生成的 M=1。

两个 full 对照的四个抽取 prompt 哈希逐一相同，因此它们**没有在这个样本上产生不同的抽取分组**，不能作为两份独立的推广证据。原生 STM 按完整 segment 放入缓冲，不强制切碎超过阈值的 segment。单次生成有少量输出差异，未进行统计显著性或质量等价判断。

## 3. 验证与修复范围

原生强制收尾路径有两个输入守恒问题：强制 sensory 收尾会覆盖此前已经发出的 segment；没有新 segment 时提前返回，会让已有 STM 残留无法强制抽取。本次只通过运行时适配修复这两处，两组路径使用相同修复；原始 external 源码和旧实验均保留。

`lightmem_flush_unittest.py` 用原生方法的实际前半段建立三个独立检查：尾段不能覆盖已发出的段；空新增段仍须清空已存在的 STM；未知源码拒绝补丁。修复前后源码 SHA256 在每个 manifest 中保存。

每个完整对照均通过：48 条消息进入/离开缓冲的多重集合一致、所有待处理缓冲为空、无压缩错误回退、LLM finish_reason=stop、全部被记录操作成功；Embedding API 序列数/token 与实际 Encoder batch 逐项对账。三轮共 **78 次 LLM 调用、408 个 Embedding batch**。

守恒发生在**压缩后的消息进入抽取阶段**的边界，保证没有额外丢批或重批，**不保证压缩保留了全部事实，也不保证生成的 38 条记忆与 261 条质量相同**。旧版维护文本/向量不一致行为没有作为本轮支线修复；full 中未触发维护 LLM 也不能证明该缺陷不存在。

初始后台控制器曾在实验开始前因 shell 引号失败，后续 cache-reset 端点又因 vLLM 未开启开发端点返回 404；均未产生完成的工作负载记录。已改为 Git 管理的控制器，并仅在回环地址的自有服务启用该端点。成功三轮的 manifest 才进入此表。修复服务生命周期时校验 PID、完整命令和独立进程组；结束后已关闭本轮四项服务，确认显存释放。

## 4. 对架构研究的具体影响

1. **旧 LightMem 的调用/资源比例不能继续冒充论文算法的共同需求。** 旧记录仍是有效的 flat 消融案例，已有负面硬件结果不改写；新记录单独存放。
2. **减少生成与新增并行前向是同一算法变化的两面。** 不能只看构建输出占比，或只挑新增 Encoder 来凑一个 compute-rich 设备。此例 Encoder 的记录执行量较小，尚未支持给它增加独立设备。
3. **本例主要改变写入路径，三个最终回答的输入总量变化很小。** 因而不能照搬 HybridSpec，声称 32B 主模型由此获得更高权重复用、适合低带宽内存。
4. **墙钟下降来自算法路径与工作量变化，不是新硬件收益。** 数据不能证明同等质量的生命周期加速；也没有测量能量、在线新鲜度、P95/P99 或写读竞争。

这就是局部补测的用途：找出会改变硬件分析的漏项，而不要求重做论文全部实验。下一项物理比较应固定这些真实操作，或明确标注算法间的质量条件。持续写入的研究问题和资源筛查见 [STREAMING_RESOURCE_OVERVIEW.md](STREAMING_RESOURCE_OVERVIEW.md)。

## 5. 入口

```bash
python3 -m architecture.lightmem_flush_unittest
# moe-server isolated checkout, local model services already configured:
.venv/bin/python operations/run_lightmem_paths.py --prefix lightpath2
.venv/bin/python -m architecture.collect_lightmem_paths
```

运行目录 `/data/ziheng/agent-memory-alignment-01`，原始资料路径 `outputs/lightpath2_{flat512,full512,full768}/lightmem/`。`effective_config` 才是适配后参数；原 family runner 留下的 `config` 是未注入配置，不能当实际开关使用。公开回执仅导出有效开关和模型元数据，原文不入 Git。
