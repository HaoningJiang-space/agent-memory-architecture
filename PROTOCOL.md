# Agent Memory motivation: bounded Phase 0

本项目检验真实 Agent Memory 工作负载是否支持新的异构架构动机。负结果同样作为结果保留；暂不优化热管理。

固定源码：Tambe-Lab/agent-memory-profiling，commit `30a157b7b49f732ee4b961cb19d7f877cf475f81`。其 GraphRAG 是仓库内 NetworkX/LangChain 实现，不是 Microsoft GraphRAG。上游缺少原始 fork SHA，不能把当前 Mem0 main 当作论文版本。

用户追加的硬约束：从真实算法出发；不预设 Hybrid Bonding、PIM 或 HBF 有收益；刻画结果不是架构贡献；A800 是校准平台；成熟软件优化必须作为强基线；所有假设必须可被推翻。最新修正：早期建模不等待完整应用通过，研究目标是改善约束下的可达边界，不是证明软件绝对解决不了。热评价需要功率—温度—性能/可行性/冷却代价的因果证据，不要求一定改变排名。

基线覆盖必须包括 [MemExplorer](https://arxiv.org/abs/2604.16007)、[KEEP](https://arxiv.org/abs/2602.23592) 与 [TokenCake](https://arxiv.org/abs/2510.18586)。KEEP 的动态 memory 更新、KV 失效与重算已构成直接重叠；TokenCake 的关键路径缓存保护和工具等待期间 offload/prefetch 已覆盖部分生命周期调度问题。不能把这些已知机制改名后算作新贡献。原生复现、等价机制控制、理想上界三类证据必须分开标注；不能用自写简化版本冒充完整基线。

固定模型及镜像下载记录见 `operations/download_models.py`。A800 仅作实测和校准。BF16 权重，非原论文 H100 FP8 配置，不直接复现其绝对耗时。禁用 thinking 并记录完整服务配置。不同精度或软件栈不能混用校准结果。

数据：MemoryAgentBench `7ea066982b140a19337e17e60d45d4076e042faf` 的 Accurate_Retrieval split。正式 pilot 使用第一个完整 `longmemeval_s*` record；其中包含 60 个问题。tiny fixture 仅用于连通性检查，不作为准确率证据。任何截短、子集或失败均独立标注。

采集 Mem0、EmbedRAG、GraphRAG 的构建、检索、更新与回答路径，保留原始调用起止时间、输入/输出 token、embedding 请求长度、模型版本、并发依赖和存储规模。API 请求 batch 与内核实际 batch 分开记录。token 比例不等于耗时或能耗比例；TTFT 也包含排队、网络与首 token 的开销，不直接等同于 prefill kernel 时间。

模型校准使用独立的输入形状微基准，记录 CUDA 时间、真实 Linear 张量形状和 NVML GPU 能量。FLOPs、权重/KV compulsory traffic 是从模型配置推导的量；没有硬件计数器时不称为实测 DRAM 流量。CPU 索引检索单独计时，不把 API 网络时间当作 DRAM 延迟。

架构模型采用任务 DAG 与参数化 compute/SRAM/DRAM/interconnect。共享预算必须明确列出算力、容量、接口/shoreline 和功率假设。优先报告固定资源控制，再报告使用 MemExplorer Table 1 技术参数的条件性 DSE；未获得物理面积/成本模型时，不宣称等硅面积或等美元成本。

最终比较统一资源、容量扩展、普通 prefill/decode 分离、按 memory 生命周期部署；允许相同的软件优化和模型选择。固定主要预算并声明附加约束，容量/带宽可以不同；等容量是另一个解释机制的控制。保留构建 decode 成本、模型驻留/复制、KV 转移、状态更新和查询读写依赖。禁止用额外硬件或不收费的模型副本制造优势。

扫描 construction/query ratio、active sessions/状态规模、更新到达率与 freshness deadline。开环队列不可稳定时报告不可行；查询使用未完成更新的记忆时不能算作满足新鲜度的任务。基准没有真实到达时间时，到达率扫描明确标为情景分析。

最终贡献验收（不作为开始 DSE 的前提）：

1. 原生算法确实运行，源数据、调用、状态和输出完整，没有截断伪成功。
2. 校准在独立形状/调用上给出误差，且模型能解释实际瓶颈。
3. 资源失配或排序变化在预算和敏感性控制下仍存在。
4. 优势不能被普通 P/D 分离、额外容量、小模型或调度单独解释。

前三项不自动满足第四项。未跨过最后一项时可以报告条件模型、负结果和架构探索，不写成已证实的新 Agent Memory 架构贡献。早期探索允许真实失败前缀与显式长度假设，但二者必须分别标注，不能拼接成已验证完整应用。协议调查冻结为支线，主线入口为 `architecture/`。
