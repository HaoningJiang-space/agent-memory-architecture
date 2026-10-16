# Agent Memory architecture motivation: Phase 0

本仓库从真实算法检验架构动机。A800 用于运行与校准；没有预设 Hybrid Bonding、PIM、HBF 或热优化的优势。

当前主线：面向神经网络加工的 Agent 语义记忆，联合研究计算、DRAM 供数和 SRAM 缓冲；Mem0 是样本之一。范围固定为已知文本的并行处理与低批量生成，不预设两类必须使用两颗芯片。

**推进重心已纠正：** [第一性原理 overview](FIRST_PRINCIPLES_OVERVIEW.md) 回到算法生命周期、容量与访问次数、合法复用和物理切分粒度。当前有限记录中，8B 贡献约 86%/97% 的投影权重工作量，32B 的单份容量更大；只有三个主问答，不能外推长期部署。局部 codec/解码器结果不再驱动主线，也尚未敲定新的物理机制。

**当前主线：异构 architecture。** [本轮第一性原理与 PDF 对照](HETEROGENEOUS_ARCHITECTURE.md) 将候选明确为共享权重的矩阵/近存两种消费域，压缩和共享解码属于供数子机制。已有六种合法分组的计算/字节统计与统一/异构 × 原始/压缩门槛筛查；没有将未定价的内部带宽当成加速结果。固定 10 GB HB 副本仍关闭，新候选的净收益和新颖性尚待验证。

**最新分析：** [bank 供数、MAC 与解码的联合约束](BANK_SUPPLY_ANALYSIS.md) 用 H²-LLM 的同面积接口/FPU 取舍替代任意拼接峰值资源。两个已有模型共新检查 5,767,168 个权重、22,528 个连续块，逐位重构通过；有限信用回放包含 704 个控制条件。共享解码的局部正例不足以支持其成为主机制，现保留为对照；没有新的同成本或端到端加速结论。

此前 [压缩供数初筛](NEXT_CANDIDATE.md) 检查了真实 BF16 权重样本，含块头、对齐及访问表的大小约为原始的 78.3% / 76.9%；这不是全模型压缩率。压缩同样给优化统一架构和已有异构基线。

**旧候选的决定保持：结束固定 10 GB HB 权重副本候选作为主方案的推进，冻结为负面对照。** [小批量供数闭环](BATCH_CLOSURE_RESULTS.md) 已为基线与候选同时加入有代价的 M=1–8 映射，并按真实依赖重放已有完整阶段链；允许更大合法 batch，计入 Embedding、冷填充、切换与空闲功耗。SimpleMem / LightMem 名义条件速度比分别约 1.08–1.09× / 1.19–1.20×，但既有 bank/interface 不确定性控制使优势消失。共同工作负载主线与模型保留；本候选不再通过低 M 特例、增大 HB 或热管理续推。

[代码审计](CODE_AUDIT.md) 修复必需 trace 缺失被记零、服务 session 写死、合批缺少容量检查；保留并标注上游 LightMem 更新文本却使用旧向量的行为。44 项单元检查及 36 次闭环重放对账通过。本轮没有新增 A800 运行。

此前 [跨算法与合法批处理结果](WORKLOAD_FAMILY_RESULTS.md) 已补 SimpleMem 28 次、LightMem 56 次原生 LLM 调用、292 个 Embedding 模型 batch，以及四条独立请求的 A800 张量形状校准。四条 SimpleMem 构建请求合法合批后，旧强基线的条件模型速度提高 1.526×，旧 M=1 局部供数候选的额外速度比仅 0.997×。这些历史结果保留，不将独立 B=1 服务量当作完整算法加速。

[有代价的供数模型](SUPPLY_PATH_RESULTS.md) 加入有限 HB 容量、bank、接口、在途窗口、SRAM 端口、控制器面积及冷填充；168 个预算内点中，100-unit HB 成本情景下旧更新/短问答约 1.24×/1.15×，但价格、bank skew 和控制器面积控制可使优势消失。尚未得到稳健物理收益，不进入热优化。

此前 [算存联合机制检验](MECHANISM_RESULTS.md) 保留交叉阵列/带宽控制、有资源代价的 K 分片、96 个条件预算配置，以及输入/输出四组合。更新路径的联合诊断空间达到 3.28×，但初版 K 分片相对连续 K 的强统一基线仅约 1–2%；该映射现冻结为对照，没有把诊断上界当成可实现加速。

[映射与资源合同](K_SHARD_DESIGN.md) 公开归约、广播、buffer、地址发生器、复用器和频率代价。[表示控制](REPRESENTATION_CONTROLS.md) 已完成 7 次小块原文问答及问题 30/39 的审查：普通小块检索也显著减少输入，但时间锚点/支持覆盖存在损失；原有词面命中包含误判，不能当正确率。

此前 [第一次有物理约束的架构对照](ARCHITECTURE_RESULTS.md) 保留真实请求驱动的阵列/SRAM/DRAM 模型、32 个配置及记忆策略反事实；探索不等待完整 Mem0 成功。完整应用质量与最终硬件结论仍单独验收。

复核后的 [阵列瓶颈诊断](ARRAY_DIAGNOSIS.md) 分开有效 MAC、低 M padding、波次和填排空，并比较固定映射带宽、理想阵列与重新映射。生命周期漏计已修复；这些诊断上界不作为可实现新硬件的收益。

[有限历史的真实配对与驻留](FINITE_PREFIX_RESULTS.md) 已完成 7 对 / 14 次 A800 问答：同一 19 段历史，原文和 Mem0 平均输入为 21151 / 240 token，输出为 84 / 50 token。实际 KV、同时分配和峰值均已记录；词面匹配为 4/7 与 2/7，尚未证明质量等价。

另已完成 [18 项 A800 算子微基准](evidence/architecture/operator_calibration.json)：真实请求对应的 GEMM 形状、合成矩阵、独立 CUDA 时间。它们是现有 GPU 的校准点，未外推成新 NPU 性能。

协议调查已冻结为支线，已排队的运行自然结束并释放服务，见 [收尾状态](evidence/architecture/protocol_branch_status.json)。历史结果保留在 [第二、三轮结果与原因](PHASE23_RESULTS.md)，研究方向见 [第一性原理 overview](OVERVIEW_NEXT.md)。

[本地 PDF 阅读](PDF_OVERVIEW.md) 对照 HybridSpec、Splitwise、H²-LLM、FGDRAM、TETRIS 等梳理证据门槛。[早期第一性原理图](evidence/phase2/first_principles_overview.pdf) 的 20K 上下文是明确的容量假设，不能代替实际 Mem0 查询长度；新分析同时核算语义压缩节省的 KV。

第一轮实测与反证见 [RESULTS.md](RESULTS.md)，进一步的失败与性能原因分析见 [CAUSES.md](CAUSES.md)。实验口径与否决条件见 [PROTOCOL.md](PROTOCOL.md)，成熟基线覆盖见 [BASELINES.md](BASELINES.md)，版本适配与测量限制见 [COMPATIBILITY.md](COMPATIBILITY.md)。最终结果由原始 API 响应与执行记录汇总，原生 profiler 的计数另行核对。

公开代码与汇总发布于 [HaoningJiang-space/agent-memory-architecture](https://github.com/HaoningJiang-space/agent-memory-architecture)。实验工作目录位于 `moe-server:/data/ziheng/agent-memory-architecture`。原始历史、请求、响应、数据库和 GPU 采样只保留在远端。本地 `evidence/` 只保存清理回执、汇总和图。Git 里不保存模型、环境或数据集；服务器上的独立 bare repository 用于同步实验代码。

主要入口：

| 文件 | 用途 |
|---|---|
| `architecture/hetero_boundary.py` / `hetero_boundary_unittest.py` | 合法投影分组、双域条件门槛与相同资源反例；不预测完整芯片性能 |
| `architecture/weight_codec_screen.py` / `weight_codec_unittest.py` | 现有 BF16 checkpoint 的 CPU 抽样编码、访问表核算和逐位重构检查 |
| `architecture/model.py` / `architecture/spec.json` | 显式分块、阵列利用率、SRAM/DRAM 服务与物理参数假设 |
| `architecture/explore.py` / `architecture/report.py` | 条件性架构比较、生命周期与容量相图、P/D/缓存/等容量控制 |
| `architecture/diagnose_array.py` | 阵列周期守恒分解、固定映射/重新映射的带宽与周期控制 |
| `architecture/finite_prefix.py` / `architecture/report_finite_prefix.py` | 远端有限历史配对、同时驻留采样及真实形状的模型对照 |
| `architecture/mechanism.py` / `mechanism_unittest.py` | K 分片、局部归约与资源计费；独立运算/计数验证 |
| `architecture/joint_diagnostics.py` / `architecture/report_mechanism.py` | 交叉控制、预算内比较、L/O 归因及报告 |
| `architecture/representation_controls.py` | 远端 512-token 原文检索对照和两例审查回执 |
| `architecture/validate_mechanism.py` | 冻结基线、模型输入哈希、预算及新 GPU 控制对账 |
| `architecture/family_workload.py` / `collect_family.py` | 远端有界原生 SimpleMem/LightMem 执行与纯数值导出 |
| `architecture/family_kernel_probe.py` | 四条不同真实输入的 B=1/B=4 张量与 CUDA 校准 |
| `architecture/supply_path.py` / `report_supply.py` | 同预算有限容量/bank/接口/SRAM 供数与敏感性 |
| `architecture/family_replay.py` / `report_family.py` | 固定硬件跨算法请求核算、合法合批控制和证据对账 |
| `architecture/small_batch.py` / `batch_closure_unittest.py` | 固定 K2 的有代价 M=1–8 映射与独立参考/边界检查 |
| `architecture/batch_closure.py` / `report_batch_closure.py` | 依赖就绪、合法大 batch、完整账本与候选收尾判定 |
| `operations/download_models.py` | 从 HF 镜像下载固定 revision 的三个模型 |
| `operations/prepare_dataset.py` | 从镜像重建固定 MemoryAgentBench 输入；已有文件不覆盖 |
| `launch_services.py` | 固定模型服务参数和 PID，隔离缓存 |
| `run_native.py` | 调用固定 artifact 的 Mem0 / GraphRAG / EmbedRAG / BM25；吞掉的错误和截断也判失败 |
| `trace_proxy.py` / `watch_gpu.py` | 独立 API 原始记录与 GPU 粗粒度功率、能量记录 |
| `measure_embedding_graph.py` | 同模型、同硬件、同输入的执行方式对照 |
| `measure_fact_batching.py` | 只批处理抽取与更新 LLM 之间的独立 fact 编码；检查检索差异 |
| `replay_failed_update.py` | 同一真实失败输入的 8B / 32B 重放；检查 ID 引用 |
| `summarize_runs.py` / `collect_summary_evidence.py` | 区分有效、失败、不完整运行；导出小型汇总 |
| `plot_motivation.py` | 从实测汇总生成独立 PNG / PDF 图 |
| `workload_accounting.py` / `accounting_unittest.py` | 从冻结真实请求计算逻辑 FLOPs、指定数据流下的字节数；检查 GQA 和首 token 口径 |
| `protocol_ablation.py` | 冻结状态上的原生、合法 ID、delta 输出对照；不修改数据库 |
| `operations/start_phase2.py` | 等待隔离服务就绪并执行完整原生 32B 对照，保留后台运行状态 |
| `phase2_snapshot.py` / `operations/finish_phase2.py` | 固定运行中的观察前缀；两个后台实验结束后汇总、冻结并释放准确匹配的服务进程 |
| `protocol_controls.py` | 明确标记的协议控制；写库前引用校验、有限重试和完整 API 回执检查 |
| `grammar_replay.py` / `grammar_factorial.py` | 相同 JSON token 流的 CPU 约束匹配对照，隔离 schema 规则开销 |
| `operations/start_phase3.py --recover` | 已审计的采集器修复后，从新状态顺序执行三个完整控制并收尾 |
| `phase3_snapshot.py` / `capacity_tradeoff.py` | 导出观察前缀；仅在完整合法运行后计算真实成对查询的 KV 容量账 |
| `operations/start_guarded.py` | 等当前实验释放服务后，运行保留原生提示词的引用约束控制；8B 失败时才回退 32B |

在远端已经准备好的环境中，可用新 run 名复查：

```bash
cd /data/ziheng/agent-memory-architecture
.venv/bin/python launch_services.py --session recheck --prefix-cache proxy embedding construction reasoning
# Wait until each local /health endpoint is ready before running workloads.
.venv/bin/python run_native.py --algorithm mem0 --run recheck/mem0_01 --stage construction --trace-file outputs/recheck/api_trace.jsonl
.venv/bin/python run_native.py --algorithm embedrag --run recheck/embed_01 --stage both --trace-file outputs/recheck/api_trace.jsonl
```

正式 pilot 使用第一个完整 LongMemEval record，包含 88 个构建片段、60 个问题；不把 tiny fixture 当质量证据。`--precompute-ner` 是显式的软件调度控制，批量执行原有 spaCy 管线后再并发调用概念抽取 LLM。初始服务关闭 prefix cache 用于刻画；不能因此声称优于启用成熟优化的服务基线。

没有通过正确性、校准、预算控制和强基线排除之前，不报告新架构获益或热设计结论。

上述要求约束最终贡献，不阻止真实请求层的早期建模。纯 CPU 探索可直接运行：

```bash
python3 architecture_unittest.py
python3 -m architecture.explore
python3 -m architecture.report
python3 -m architecture.diagnose_array
python3 -m architecture.report_finite_prefix
python3 mechanism_unittest.py
python3 -m architecture.joint_diagnostics
python3 -m architecture.report_mechanism
python3 -m architecture.validate_mechanism
python3 -m unittest family_unittest supply_path_unittest
python3 -m architecture.supply_path
python3 -m architecture.report_supply
python3 -m architecture.family_replay
python3 -m architecture.report_family
python3 -m unittest batch_closure_unittest
python3 -m architecture.batch_closure
python3 -m architecture.report_batch_closure
```

第一轮收尾核验入口为 `validate_evidence.py`；`audit_power.py` 检查粗粒度设备能量读数；`freeze_receipts.py` 固定远端原始证据哈希。`operations/stop_services.py` 是第一轮专用收尾脚本，不用于 phase2。新运行使用不同 session、trace 路径和证据回执，避免修改旧的冻结文件。
