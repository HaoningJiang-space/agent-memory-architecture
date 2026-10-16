# Agent Memory architecture motivation: Phase 0

本仓库从真实算法检验架构动机。A800 用于运行与校准；没有预设 Hybrid Bonding、PIM、HBF 或热优化的优势。

最新结果见 [第一次有物理约束的架构对照](ARCHITECTURE_RESULTS.md)：真实请求驱动的阵列/SRAM/DRAM 模型、32 个配置及记忆策略反事实；探索不再等待完整 Mem0 成功。完整应用质量与最终硬件结论仍单独验收。

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
| `architecture/model.py` / `architecture/spec.json` | 显式分块、阵列利用率、SRAM/DRAM 服务与物理参数假设 |
| `architecture/explore.py` / `architecture/report.py` | 条件性架构比较、生命周期与容量相图、P/D/缓存/等容量控制 |
| `architecture/diagnose_array.py` | 阵列周期守恒分解、固定映射/重新映射的带宽与周期控制 |
| `architecture/finite_prefix.py` / `architecture/report_finite_prefix.py` | 远端有限历史配对、同时驻留采样及真实形状的模型对照 |
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
```

第一轮收尾核验入口为 `validate_evidence.py`；`audit_power.py` 检查粗粒度设备能量读数；`freeze_receipts.py` 固定远端原始证据哈希。`operations/stop_services.py` 是第一轮专用收尾脚本，不用于 phase2。新运行使用不同 session、trace 路径和证据回执，避免修改旧的冻结文件。
