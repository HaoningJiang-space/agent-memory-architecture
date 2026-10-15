# Agent Memory architecture motivation: Phase 0

本仓库从真实算法检验架构动机。A800 用于运行与校准；没有预设 Hybrid Bonding、PIM、HBF 或热优化的优势。

当前 follow-up 见 [本地 PDF overview 与修订后的动机检验](PDF_OVERVIEW.md)。第一轮证据已冻结；新的原生 32B Mem0 与输出协议对照在远端 `outputs/phase2/` 独立后台运行，启用 prefix caching。[第一性原理图](evidence/phase2/first_principles_overview.pdf) 分别展示实测请求时间和权重/KV 容量上界，不代表已经得到新架构加速比。

第一轮实测与反证见 [RESULTS.md](RESULTS.md)，进一步的失败与性能原因分析见 [CAUSES.md](CAUSES.md)。实验口径与否决条件见 [PROTOCOL.md](PROTOCOL.md)，成熟基线覆盖见 [BASELINES.md](BASELINES.md)，版本适配与测量限制见 [COMPATIBILITY.md](COMPATIBILITY.md)。最终结果由原始 API 响应与执行记录汇总，原生 profiler 的计数另行核对。

代码与原始实验位于 `moe-server:/data/ziheng/agent-memory-architecture`。原始历史、请求、响应、数据库和 GPU 采样只保留在远端。本地 `evidence/` 只保存清理回执、汇总和图。Git 里不保存模型、环境或数据集；当前推送目标是该服务器上的独立 bare repository。

主要入口：

| 文件 | 用途 |
|---|---|
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

第一轮收尾核验入口为 `validate_evidence.py`；`audit_power.py` 检查粗粒度设备能量读数；`freeze_receipts.py` 固定远端原始证据哈希。`operations/stop_services.py` 是第一轮专用收尾脚本，不用于 phase2。新运行使用不同 session、trace 路径和证据回执，避免修改旧的冻结文件。
