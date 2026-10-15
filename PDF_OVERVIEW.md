# 从本地 PDF 重新审视 Agent Memory 架构动机

这份记录服务于体系结构研究：找到优化后的现有设计仍然无法消除的资源失配，再决定硬件机制。当前结果尚未证明需要 Hybrid Bonding、PIM、HBF 或专用 Memory XPU。

**本次实际阅读的本地材料**

页码均为 PDF 文件页码。没有把目录中的实验曲线、海报和重复图片 PDF 当成新的参考论文，也没有声称读完目录内全部 417 个 PDF。

| 本地 PDF | 已重点阅读的位置 | 对本研究的实际约束 |
|---|---|---|
| [HybridSpec](/home/abc/jhn/HybridSpec_Exploiting_Hybrid-Bonding_Memory_to_Accelerate_LLM_Serving_Through_Heterogeneous_Architecture_and_Speculative_Decoding.pdf) | pp. 1–2、4–5、7–13，特别是 Fig. 4、Fig. 19 和硬件配置 | 算法改变执行形状与容量需求；容量、带宽、通信、动态调度共同决定映射。仅有“两个模型不同”不够。 |
| [Splitwise](/home/abc/jhn/thermal/data/literature/r22/Splitwise-ISCA24.pdf) | characterization、batching、P/D 传输及成本/功率比较 | P/D 异构和部署成本已有直接基线。真实请求长度分布、优化后的混合 batching、KV 传输都必须计入。 |
| [H²-LLM](/home/abc/jhn/asy/asymmetric_llm_3d/cooling/deep_review/papers/architecture/h2_llm_isca2025.pdf) | pp. 1–5、9，roofline、HB 控制器开销和 DSE | 高带宽需要控制器面积，挤占计算资源；低 batch、GQA/MQA 也会改变近存计算所需算力。固定 operator 映射不能替代 dataflow 探索。 |
| [Sarathi-Serve](/home/abc/jhn/thermal/data/literature/r22/Sarathi-OSDI24.pdf) | pp. 2–5 | Chunked prefill、混合 batching、减少 generation stalls 是成熟软件控制。不能拿未经优化的串行服务证明新硬件必要。 |
| [FGDRAM](/home/abc/jhn/thermal/data/literature/dram-boundaries-20261003/FGDRAM-MICRO2017.pdf) | pp. 1–3、5 | 物理机制必须具体到 bank 并行性、共享总线、激活粒度和数据移动距离；必须比较等带宽、等容量及增强后的基线。 |
| [TETRIS](/home/abc/jhn/thermal/data/literature/r23/TETRIS-ASPLOS17.pdf) | pp. 1–2、4 | 3D 内存的价值不只在峰值带宽，而在改变 SRAM、PE、数据复用之间的最优面积分配；需要硬件和数据流一起改变。 |

**HybridSpec 真正给我们的研究范式**

它不是先拿一个小模型放 HB，再测是否快。Speculative decoding 将 target 的串行生成变成并行 verification，同时 draft 保留低并行度生成。算法因此改变了两侧的算术强度和活跃数据规模。硬件配置利用这种变化，调度再处理在线负载、KV 增长、两侧利用率和 prefill/verification 冲突。

Fig. 19 的反转来自容量限制并发，进而影响排队与 batching。它不是“低带宽永远胜过高带宽”，也不是直接适用于任何模型的性能结论。其不同内存配置包括 512 GB LPDDR5X、40 GB HBM2 和 80 GB HBM2e；这张图本身不能替代我们自己的同预算控制。

其 HB 原型配置是 10 GB、4 TB/s、400 MHz，逻辑侧有明确的 MAC 和 SRAM 配置。我们的 Qwen3-8B BF16 权重约 15.26 GiB，不能原样映射进去。扩大 HB、量化或分片会改变面积、质量、通信与时延，必须重新评估。

**为什么之前的方向显得奇怪**

第一，误把 token 比例当成了执行时间比例。对于 BF16、权重流量主导的低 batch decode，以及计算充分利用的 prefill，忽略 attention 和首 token 修正，有

\[
T_P\approx \frac{2PS}{C},\quad
T_D\approx \frac{2PD}{B\,W},\quad
\frac{T_D}{T_P}\approx \frac{D}{S}\frac{C/W}{B}.
\]

这里 P 是参与线性计算的参数量，S、D 是输入和输出长度，C 是有效 FLOP/s，W 是有效 byte/s，B 是同一模型可共同复用权重的 decode batch。若输出占 LLM token 的 4.6%、C/W=100 FLOP/byte、B=1，则这个简化模型的 decode/prefill 时间比约为 4.82。**该算式是反例与条件分析，不是 A800 性能预测。** 如果 token 总数还混入 0.6B embedding 与 32B LLM 的输入，连每 token 的计算成本都不相同，更不能直接按比例判断瓶颈。

真实请求提供了相同方向的证据：旧 Mem0 失败运行中，前 9 个完成块的 extraction 输出只占其 LLM token 的 2.12%，但首 token 之后的时间占请求时间 84.19%。Update 对应比例为 28.13% 和 98.88%。反过来，完整 embedRAG 查询的首 token 时间占 LLM 请求时间约 86.84%。这些是 API 边界时间，包含软件开销，**不是 GPU kernel 占比，也不能把失败运行前缀当作质量有效的完整 Mem0 结果**。

第二，混淆了三类状态。

| 状态 | 当前测得或计算出的尺度 | 生命周期与共享性 | 对架构的含义 |
|---|---|---|---|
| 语义记忆、向量/词法索引 | 已完成的 embedRAG/BM25 状态约 1.99/3.31 MB | 持久、用户相关、可变；当前只测一个历史 | 不能据此主张索引是 HB、SSD 或大容量瓶颈。必须实测规模增长、访存和查询服务时间。 |
| Qwen3 KV | 8B 每 token 144 KiB；32B 每 token 256 KiB，BF16 | 请求相关；可释放、缓存、驱逐或重算，取决于运行时 | 必须跟踪真实 residency；不能假设 Agent 的整个生命周期都钉住 KV。 |
| 模型权重 | 8B 15.26 GiB；32B 61.02 GiB；embedding 1.11 GiB | 通常不可变，多用户共享 | 额外构建模型增加常驻容量；用户数增加不会按用户复制一份权重。 |

第三，把算法的串行调用写法当作不可消除的数据依赖。Pinned Mem0 在一个 chunk 中，事实提取依赖输入文本；各事实 embedding 不依赖此前那条事实的 embedding；检索读取旧状态；update LLM 决策依赖这些检索结果；真正修改状态之后，下一次有状态检索才必须看到新版本。因此可以测试批量 embedding、提前提取、多个独立用户 co-scheduling。只有 read-modify-write 依赖和质量约束排除这些控制之后，剩余串行性才可能支持硬件动机。

第四，当前 8B update 输出中存在大量 NONE 重述和错误引用。加速这些输出可能只是在加速协议冗余或失败工作。必须先测合法 ID、delta 输出等软件控制的收益及决策变化。省掉字数不自动等于语义等价。

**现在最值得保留的候选：模型选择、常驻容量和状态提交的耦合**

把构建模型从 32B 换成 8B，单次计算会变少，但当 reasoning 仍用 32B 时，系统多了一份 8B 权重。若构建与 reasoning 复用同一个 32B 模型，则无需第二份独立 8B 权重。在 80 GiB 总容量中，只计算 BF16 权重：

| 配置 | 权重总量 | 剩余容量上界 | 若全部用于 32B KV，可存 token 上界 |
|---|---:|---:|---:|
| 共用 32B + embedding | 62.13 GiB | 17.87 GiB | 73,179 |
| 8B construction + 32B reasoning + embedding | 77.39 GiB | 2.61 GiB | 10,689 |

这些是**排除 workspace、activation、allocator 和构建 KV 的容量上界**。一条约 20K-token 的 32B prompt 就需要约 5 GiB KV；双模型配置在这个容量点连该下界都无法满足。它证明原始部署设想存在容量问题，尚未证明新的异构架构优于复用模型、量化、offload 或普通容量扩展。两张 A800 的总预算是 160 GiB，不能只给新方案两张设备、却给统一基线一张。

这产生一个值得测的反直觉现象：**更小的构建模型可能使系统更慢，因为额外常驻权重降低并发、增加换入换出，或使记忆提交赶不上后续读取。** 目前只证明了容量前提，时延反转和质量条件仍待验证。

这个现象本身也可能属于通用多模型 serving。要成为 Agent Memory 的架构贡献，必须进一步证明：语义记忆写入/读取依赖如何改变可行 batch、权重/KV residency 和资源配置，并且这些变化不能被通用 model multiplexing 或 KV 管理解释。

**把 motivation 做成可被否定的实验，而不是四条预设结论**

| 图/实验 | 要回答的问题 | 必须加入的控制 | 何时否定候选 |
|---|---|---|---|
| A：真实 DAG 上的资源与临界路径 | 热点究竟是 prefill、decode、embedding、索引、CPU launch 还是等待提交？ | 合法且完成的工作；软件优化；按模型/操作分开计数；真实 kernel 与服务时间分开 | 加速的是无效输出，或软件优化消除了主要热点 |
| B：模型选择与容量反转 | 8B+32B 是否会输给共用 32B？ | 质量、量化、workspace、同容量/同成本；80–512 GiB 扫描；同一 arrival trace | 只有人为选取的容量点成立，或普通容量扩展已足够 |
| C：状态依赖的消融 | 相同算子与数据量，保留/移除真实跨会话依赖是否改变最佳设计？ | 真实多会话任务；独立用户 batching；受控 arrival 与实测 arrival 分开；无依赖版本只作诊断 | 移除 Agent Memory 依赖后收益不变，说明收益主要来自通用 LLM serving |
| D：物理预算与成熟基线 | 同面积/功耗/成本约束下，什么机制改变了可达的性能边界？ | 优化统一架构、P/D、普通 DRAM 扩容、KV/offload、MemExplorer 适用设计；互联与复制成本 | 优势仅来自免费增加带宽、容量或算力，或忽略搬运成本 |

这四张图是证据链的组织方式，不保证存在四条独立的创新。只要 B/C/D 没有跨过，不能把 A 的 characterization 写成已完成的架构贡献。

**近期工作的取舍**

后台运行原生 32B Mem0 完整构建与查询，校验 8B 的失败是否主要是能力/输出协议问题。同时后台重放 10 个真实冻结 update 输入，比较原生、合法 ID schema、合法 ID + delta 三种控制；只用于引用合法性、输出开销和操作一致性分析，不声称独立语义评测。

后续在线工作负载采用真实跨会话数据，避免把一个离线历史复制成许多假用户。MemoryArena 的代码已通过镜像取得并固定版本，尚未完成运行；其模型判断器、环境行为与到达过程都必须明确。已有 characterization 的 freshness 图使用受控到达时间回放，不能转述成测得生产系统自然每 5 秒到达一次。

参数化架构模型先做守恒和上界检查：没有物理效率差异时，允许灵活调度的理想统一资源池可以模拟资源分区，静态异构划分不会凭空增加资源。新硬件必须通过可论证的单位面积带宽、能耗、容量或数据流效率改变约束。HB/3D/thermal 只有在这个环节出现实际因果作用时才进入主线。

进一步阅读的当前版本：[Agent Memory characterization v2](https://arxiv.org/html/2606.06448v2)、[MemExplorer](https://arxiv.org/html/2604.16007v1)、[KEEP](https://arxiv.org/abs/2602.23592)、[TokenCake](https://arxiv.org/abs/2510.18586)。版本不可混用：characterization v2 已增加 multi-tenant co-scheduling 论述，旧版的 embedding 请求规模也不能直接当作当前 artifact 的实测值。
