# Prior coverage and falsification controls

The purpose of this matrix is to prevent a familiar software mechanism from being relabeled as a new architecture. Source inspection is complete for the entries below; native end-to-end reproduction of KEEP and TokenCake is **not** complete. A simplified control must never carry their names as if it reproduced the papers.

| Baseline | What it already covers | Required comparison / rejection condition |
|---|---|---|
| Homogeneous GPU/NPU | Shared compute and memory, continuous batching, chunked prefill, prefix caching | If an optimized shared device meets quality/freshness at the same resource budget, reject a claim that heterogeneity is necessary. |
| Optimized prefill/decode separation | Compute versus decode bandwidth asymmetry, interference and admission | Charge model replication, KV transfers, and interface capacity. A benefit explained entirely by ordinary P/D separation is not Agent Memory novelty. |
| Capacity / KV control | Additional DRAM, paged KV, prefix reuse and host offload | Match total capacity and resource budget; do not award a new engine an uncharged capacity increase. |
| [MemExplorer](https://arxiv.org/abs/2604.16007) | Joint compute/SRAM/memory-technology DSE for prefill and decode in agentic inference | Use its applicable design assumptions as a competing design space, not merely a weak HBM-only baseline. Its projected memory parameters are assumptions, not A800 calibration. |
| [KEEP](https://arxiv.org/abs/2602.23592) | Update-frequency grouping, dynamic-memory KV invalidation, selective multi-hop recomputation, layer-balanced loading | Test whether residual costs survive reuse/recomputation controls. Its embodied planner and selective recomputation semantics are not automatically equivalent to Mem0 fact extraction. |
| [TokenCake](https://arxiv.org/abs/2510.18586) | Agent-priority KV capacity allocation, lifecycle notifications, preservation/offload/prefetch during tool stalls | Require genuine DAG dependencies and stall intervals. A static offline corpus has no measured tool-stall distribution; do not invent one and label it workload evidence. |

Pinned code:

- Characterization artifact: `Tambe-Lab/agent-memory-profiling@30a157b7b49f732ee4b961cb19d7f877cf475f81`. Its vendored Mem0 makes extraction and update LLM calls, with per-fact embedding/search between them. Current upstream Mem0 main implements a different path, so its name alone is not provenance.
- Current Mem0: `mem0ai/mem0@0724291bd864cf807e88ee8981c8226644e8fb42`, `mem0/memory/main.py:919–1006` already retrieves existing context before a single extraction call, then calls `embed_batch` for extracted facts. This is source evidence, not a measured end-to-end comparison. The old artifact's serial loop is not representative of every Mem0 release. In the artifact loop itself, all fact searches precede memory mutations, so independent fact encodings may be batched while preserving fact order; numerical/retrieval parity still needs checking.
- KEEP: `PKU-SEC-Lab/KEEP_Embodied_Memory@493388b11e9cbbe0b323a4eb770e2b9bd0a25ed7`. `KEEP_transformers/modeling_qwen2_14b.py` implements module recomputation selection. The repository provides separate accuracy and vLLM latency implementations.
- TokenCake: `pku-lemonade/TokenCake@1901ebf31fdf63590148e61d5c2f65e9fb7ec010`. `vllm/tokencake/scheduling.py`, `offload_policy.py`, `offloading.py`, `lifecycle.py` implement scheduling and preservation; `config.py` explicitly requires native KV offloading capacity for the offload arm. A scheduling-score-only test cannot reproduce transfer or end-to-end gains.

Experiment rejection gates:

1. An input-token majority is insufficient: measure elapsed compute, output generation, embedding, index and idle/launch gaps separately.
2. An eager framework's high latency is insufficient: compare an optimized launch path with identical weights, inputs and outputs before inferring a hardware bandwidth limit.
3. An API batch is insufficient: record actual kernel batch and sequence lengths; keep native serialization dependencies.
4. An offline build/query trace cannot prove online freshness. Arrival-rate replay is a declared scenario, with stability and read-after-write constraints, not a measurement of production arrivals.
5. A technology sweep without an independently checked area/cost/power budget cannot establish a physical architecture winner.
6. Until the residual mismatch survives the baselines above, HB/PIM/HBF and thermal optimization remain unselected candidates.
