# 论文与本仓库为什么不同：口径、算法路径与物理机制对齐

2026-10-16。复核对象为 `main@9316681` 及其冻结证据；本轮重算聚合 token 和执行图祖先关系，没有新增 A800 实验、算法或硬件扫描。阅读了 `/home/abc/jhn` 下 HybridSpec、H²-LLM、Splitwise 等 PDF，以及下文标注版本的原论文。

**主要修正：SimpleMem 的 token 结构与 characterization 并没有此前暗示的巨大矛盾；当前 LightMem 缺少论文关键路径；最新私有双域实验没有重现论文的存储拓扑，也没有测试构建与服务争用。其负结果继续保留，但不能用于关闭 Construction–Serving 异构方向。**

此前把有限实例的生成成本，与跨系统 token 中位数比较，再向架构方向判断推进，是比较口径和证据外推的问题。不是再加几轮参数扫描就能修正的问题。

## 1. SimpleMem：先把分母与对象对齐

[Agent Memory Characterization v1 §4.3 / Figure 5](https://arxiv.org/html/2606.06448v1) 的 4.6% 是**跨系统中位数**，SimpleMem 自身为 **28.5%**。分母含 LLM 输入、LLM 输出和 Embedding 输入。LightMem 不在该论文的十系统集合中。

本仓库四个实际 SimpleMem 构建请求：

| 原始计数 / 重新核算 | 数值 |
|---|---:|
| LLM 输入 | 19,683 tokens |
| LLM 输出 | 7,587 tokens |
| Embedding 输入 | 2,473 tokens |
| 只在 LLM 内计算输出比例 | 27.8218% |
| 对齐 Figure 5 分母后的输出比例 | **25.5085%** |
| 论文 SimpleMem | **28.5%** |

差约 2.99 个百分点，而不是“论文 4.6%，我们 27.8%”所暗示的巨大偏离。这不是精确复现：模型、窗口和历史范围仍不同。**它足以撤回以这组数字作为直接冲突的论证。**

而且，token 比例不等于时间比例。Embedding 与 LLM 的参数规模不同；prefill 可跨 token 复用权重，decode 的合法 batch 与重读次数又不同。一次 embedding token 与一次 32B decode token 没有相同硬件成本。论文的 token 分解不能单独证明某芯片上 prefill 占主导；我们的目标模型 decode 占比也不能当作 A800 阶段计时反驳它。

现有 A800 四请求校准覆盖一次 prefill 和八步 teacher-forced decode。它验证有限形状的运行差异，未测得整段实际生成的完整阶段占比。实际 L/O 记录、A800 局部时间、目标 NPU 条件时间必须分列。

可复核计数与来源哈希见 [paper_alignment.json](evidence/architecture/paper_alignment.json)，命令为 `python3 -m architecture.paper_alignment`。

## 2. 当前运行不是这些论文的完整复现

| 对齐维度 | 论文 | 本仓库冻结实例 | 对硬件判断的影响 |
|---|---|---|---|
| Characterization 本地平台 | H100 80 GB；32B/14B FP8，较小模型 BF16 | 2× A800 测量平台；8B 构建、32B 回答，BF16 | 参数量、权重流量、容量及执行效率均不同 |
| Characterization Table 3 | Qwen3-32B；约 180 万历史 token、300 问答 | family 记录只有同一历史的前 48 条消息、3 个检索探针 | 不能比较完整生命周期比例或算法排名 |
| SimpleMem 效率表 | LoCoMo-10、GPT-4.1-mini；论文窗口 W=20 turns | profiling 仓库固定版本，每条消息转成一个 Dialogue；窗口 16 / overlap 2，8B，4 个构建窗口 | 切分单位和范围也需对齐，不能只比较窗口数字；调用数、并行度和时间不同 |
| LightMem 核心路径 | LLMLingua-2 预压缩、主题分段、缓冲、生成及离线维护 | **`pre_compress=False`, `topic_segment=False`**；flat extraction | 缺失并行文本处理，并改变后续生成数量和形状 |
| LightMem 模型 | 包括 Qwen3-30B-A3B-Instruct-2507；索引用 all-MiniLM-L6-v2 | Qwen3-8B dense、统一 32B 回答、Qwen3-Embedding-0.6B | MoE 激活计算、总驻留、Embedding 工作量都不相同 |

前两行依据 [characterization §3.2/4.2](https://arxiv.org/html/2606.06448v1)；SimpleMem 行依据 [SimpleMem v3 §3.1/3.3](https://arxiv.org/html/2601.02553v3)；LightMem 行依据 [LightMem v4 §5.1 / Table 5](https://arxiv.org/html/2510.18866v4)。本地来源为 `family_workload.py`、`family_summary.json` 和 `family_provenance.json`。

LightMem 论文明确在后续实验中启用预压缩；本地关闭它是工作负载覆盖限制，不能只列为无关紧要的实现差异。关闭过滤和缓冲组织后，真实运行得到的 flat 请求仍可研究，但不能声称它代表完整 LightMem 的高 M / 低 M 配比。当前旧向量更新行为也继续标为限制，不把它的工作量视为修复后算法成本。

SimpleMem 的 embedding 适配器还丢弃 query instruction；这影响质量与检索可比性。三问没有完整历史支持的质量验证。当前公共记录只能支撑有限实例的硬件形状与依赖分析。

还要分清收益来源：SimpleMem / LightMem 论文的速度比主要比较**不同记忆算法产生的工作量与执行组织**；本仓库固定同一批 L/O 比较硬件，刻意固定了这些变化。两类速度比不是同一个实验问题。

## 3. 与硬件论文相比，缺少的是哪些因果条件

### HybridSpec：算法先改变 target 的执行方式

本地 PDF pp.4/11 显示，SD 让 target 批量验证草稿，把原本逐步生成的一部分需求改为更高复用的计算；小 draft 完整驻留 HB。容量受限时，额外容量还会改变可接纳的并发。粗粒度交换只是其中一个条件。[HybridSpec](https://doi.org/10.1109/ISCA66397.2026.00054)

本仓库语义记忆缩短了输入，却**没有**把 32B autoregressive decode 改成批量验证。完整模型放在两个域、边界只传少量文本，并不会减少各域内部反复读取权重。因此不能直接照抄“大模型 LPDDR、小模型高带宽”的分工。

### H²-LLM：改变真实的数据路径和资源代价

本地 PDF pp.6–7/10 描述共享模型数据、中央/近存模式、channel/bank 分配、局部计算和数据流；Table 6 同面积下接口与 MAC 相互挤占。[H²-LLM](https://chenzhangsjtu.github.io/files/2025-H2_LLM_ISCA25.pdf)

最新 `call_placement.py` 使用两个**私有存储域**，且两侧沿用同一组阵列映射和成本假设。N 是常规 HBM，不是 HB-on-Logic。这个测试没有实现 H²-LLM 的共享 bank 与局部消费机制；此前 `hetero_boundary.py` 又只是投影级筛查，不能补成完整复现。

### MemExplorer：容量能否变成 batch，是决定性条件

[MemExplorer v2 §IV-C/V-A](https://arxiv.org/html/2604.16007v2) 以可用容量推导最大 decode batch，目标 NPU 采用 MX-INT4（包括块 scale），OSWorld 使用观测到的最长 90K 输入 / 8K 输出，系统搜索预算 700 W。GPU 验证另有自己的精度，不能把那一栏与 NPU 设置混用。

我们使用 BF16、有限就绪请求，比较阶段链完成时间，预算 500 W。统一 96 GB 配置的模型容量峰值约 87–88 GB，已经能装下当前负载。因此，论文中“更多容量→更大 batch→更多权重复用→更高吞吐”的路径在当前实例中没有被触发。两者不是互相否定。

精度差异也不只是一个固定速度系数：量化同时改变权重/KV 容量、实际流量和每字节计算量；块 scale、解码和相应算子成本仍须计入。不能把 BF16 与 MX-INT4 的差异归因于异构组织。

也不能简单把本地 batch 改成容量最大值来追论文：依赖尚未解除的请求不能一起执行。容量上限回答能装多少，实际就绪和时限决定能用多少。

## 4. 最新双域负例还暴露了两个实验边界

**第一，四条 DAG 全部先完成构建，再进入问答。** 自动祖先检查确认，每个 query 节点都依赖所有构建/维护神经节点。`concurrent_queries` 只是构建结束后同时释放三问，不是构建与服务并发。

所以，本实验没有测到线上查询挤占、后台写入排队或 freshness 约束。它能判定该有限阶段链与该私有双域配置的代价，不能判定 Construction–Serving 专用化是否改善争用下的效率。

**第二，所谓“矩阵侧”没有在实际 prefill 上取得优势。** 保存结果中：

| 请求组 | 合法 B | C prefill / s | N prefill / s |
|---|---:|---:|---:|
| SimpleMem 构建 | 4 | 3.271 | 2.430 |
| LightMem 维护 | 29 | 2.396 | 1.520 |

这些是目标模型服务量。C 有更多 MAC，但有限 SRAM/分块/数据重读及 LPDDR 供数仍限制有效执行。仅在这种配置上测试 P/D，连 prefill 的正收益都不存在；改善 KV 传输不能弥补这个符号。

用 BF16 投影的权重一次读取近似：

\[
I_W=\frac{2MKN}{2KN}=M,\qquad M_*\simeq F_{peak}/BW_{effective}.
\]

当前 C/N 的峰值平衡点约为 **1056 / 324 FLOP/Byte**。这不是实际 crossover：低 M 利用率、activation/KV、SRAM 和重读会改变结果。但它足以解释，B=4 或 29 不能仅因“可以离线合批”就被叫作 compute-bound。要讨论真实数据流下的 `FLOPs / transferred bytes`，而不是只看原始 prompt 长度或设备标签。

最新负例应缩限为：**这一组私有 LPDDR/HBM 资源配比，在有限、先构建后问答的 BF16 记录和当前映射模型下不划算。** 不等于完整调用异构已被否定，也没有复现上述论文后推翻它们。固定 10 GB HB 副本的旧关闭决定不变。

## 5. 对两条方向的取舍：先定义实际物理问题

### A. 构建与服务的异构资源配置继续优先，但去掉预设角色

把问题改为：**在真实依赖、记忆可见性和服务时限下，可以形成怎样的复用与状态驻留？哪种算力/带宽/容量配比用更低成本满足它？**

对于共享同一 checkpoint 的独立生成，合法 batch 至少同时受四项限制：

\[
B\le\min(B_{ready},B_{compatible},B_{capacity},B_{deadline}).
\]

兼容不仅包括模型，还包括推理模式；合批不是将不同内容拼接成一个新 prompt。构建还需满足：

\[
T_{wait}(B)+T_{build}(B)+T_{commit}\le D_{fresh}.
\]

若允许等待仍凑不出足够复用，构建侧也可能需要高效权重供数；若需求已被统一设备的合法 batching 处理，专用域可能失去价值。构建、规划、回答的角色不决定结果。

异构收益必须来自单位面积/能量的计算、有效供数、容量或搬运代价的改变。仅切两个名字不同的资源池，统一可分区资源原则上可以模仿。完整调用与 P/D 选择应作为物理比较所需的执行策略，不先包装成新机制。

### B. Semantic/KV 不是可任意替换的两种存储格式

语义事实是经算法选择和变换的信息；KV 是给定模型、精度、位置与前文下的执行状态。稳定事实不保证 KV 有效：检索集合、排列或先前上下文变化，均可能破坏复用。相反，在模型与上下文完全一致时，原文 RAG 也可以获得精确 prefix reuse。

[KEEP](https://arxiv.org/html/2602.23592v1) 已处理静/动态记忆分组、部分重算与层级加载；[CacheBlend](https://arxiv.org/abs/2405.16444) 已处理检索块缓存的跨块融合；[TokenCake §4.2](https://arxiv.org/html/2510.18586v3) 用工具暂停窗口和可接纳工作决定是否卸载。一般的“保留/卸载/重算”不是新的架构机制。

对相同语义与质量条件下、允许正确复用的执行状态，才可以比较：

\[
n_{valid}\big(C_{recompute}-C_{load}-C_{repair}\big)
>C_{initial-extra}+C_{residency}+C_{invalidation}.
\]

这里 C 必须统一为能量或设备服务量；端到端时延另按依赖与重叠计算。`n_valid` 是失效前真实可用的复用次数；`C_initial-extra` 只计相对重算基线额外的创建/保存费用，不能把本来就需要的首次 prefill 重复计费。

因此 B 暂作必须比较的状态复用基线与候选问题，不因 A 某个配置失败就自动宣布它成为新主线；两者可以共存。若要把 B 提升为主机制，需要有效 KV 复用量、失效和容量机会成本的实际证据。

## 6. 下一次有边界的交付应是什么

不追加 HB bank、codec 或热参数。也不先要求重跑所有论文。

1. **对齐一个完整算法路径。** SimpleMem 保留本地实例与论文版本差异；LightMem 若用于代表完整算法，必须补齐原论文预压缩/主题/缓冲路径并独立报告。现有 flat 路径保留为已知范围的样本，不覆盖或改名为论文复现。
2. **对齐一个物理机制。** 使用论文给定精度、映射、容量和供数关系复查一个代表点，再只替换工作负载；或者先在本地 BF16 上做相同资源/不同数据路径的对照。不能同时换算法、精度、存储拓扑和评价目标后将差异归因于 Agent memory。
3. **在真正共享时间轴上比较 A。** 同一可见性合同下，让构建与查询存在重叠；外部到达与 freshness 若缺实测，标为受控参数。统一设备同样允许 batching、chunked prefill、优先级与可延期维护；P/D 同样计模型副本与状态迁移。交付合法 batch 分布、服务时限/新鲜度违约、时间/能量与容量，而非再次只给累计 GPU 服务秒。

这三项用于确定区别究竟来自工作量、执行条件还是物理路径；没有预设必须得到正收益。若匹配条件后优势完全由已有 P/D、缓存或调度解释，应明确该机制已被覆盖；若条件不适用，应说明适用范围，而不是宣称异构方向普遍无效。

## 7. 复核回执

- `python3 -m architecture.paper_alignment`：检查两类构建 token 分母、四条目标 DAG 的阶段屏障、保存的 C/N 平衡点与代表 prefill；输入与脚本 SHA256 写入公开 JSON。
- 新增完整调用回放在远端隔离 checkout `7baaea7` 通过 8 项检查、重放 552 条记录。跨 Python 结果的结构/决策相同，浮点最大缩放差约 `1.53e-15`，在 `1e-12` 容差内；gzip 头和浮点末位导致文件及 detail hash 不同，**不声称字节一致**。见 [远端核对回执](evidence/architecture/call_placement_remote_validation.json)。这是 CPU 程序一致性检查，不是 A800 或物理校准。
- 本地 PDF 路径与哈希沿用 [文献清单](evidence/architecture/hetero_literature.json)。未下载新模型、运行新 GPU 任务或更改冻结实验的算法输出。

**当前结论：已有结果支持异构需求的调查，但尚没有与论文严格对齐后证明或否定的完整新架构。首先修正实验对象和比较口径，再让真实复用、数据路径和资源代价决定设计。**
