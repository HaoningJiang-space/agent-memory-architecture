# a8bf9c1 代码审计与修复

结论：有可复现的代码错误。下列三个本仓库问题已修复；上游 LightMem 的旧向量问题保留在冻结样本中并明确标注。单元检查通过不等于代码无 bug，也不等于物理参数已校准。

| 问题 | 触发与影响 | 本轮处理 |
|---|---|---|
| 必需的 trace 缺失被当成空列表 | `collect_family.records()` 对不存在的文件返回 `[]`；Embedding API/kernel 同时为空时，`0 == 0` 检查通过，缺失测量被记成零工作量 | 必需文件缺失抛错；API/kernel 为空时拒绝收集 |
| 服务 session 写死 | `--simple` / `--light` 改到新实验，采集器仍读取 `outputs/family_01`；可能混入旧 session 或得到空区间。旧失败实验回执也被无条件要求 | 新增必填 `--service-session`；旧失败回执仅在原实验组合且文件存在时附加 |
| 合批模型未检查容量 | 将 backing 容量改成 1 byte，旧 `batched()` 仍输出有效秒数；可能让不可能驻留的 batch 参与架构比较 | 使用固定 checkpoint 的实测权重字节数，计入 padded KV、逐步收缩的 decode 与逻辑激活峰值；超容量报错。新 DAG 只因容量不足拆批 |

前两项不说明已发布 family_01/family_02 的 Embedding 为零。原冻结汇总保留了 SimpleMem 31、LightMem 261 个模型 batch，其 API 与 kernel 数量/输入 token 已重新对账。第三项也没有推翻已发布四请求控制：该 batch 确实能放入所选 backing。重新生成旧 `family_replay.json` 后，原服务时间与能量不变，只增加容量与新版来源元数据。

旧汇总由旧 collector 生成，不能把当前修改过的 collector 哈希冒充当时版本。`architecture/frozen/collect_family_a8bf9c1.py` 保存该版本原文；`report_family.py` 验证其哈希，原始历史、prompt 和数据库仍在远端。

新版远端收集命令示例（使用新输出文件，不覆盖冻结证据）：

```bash
python -m architecture.collect_family \
  --simple family_02/simplemem --light family_01/lightmem \
  --service-session family_01 --out outputs/family_01/public_summary_recheck.json
```

## 上游 LightMem：更新文本但继续使用旧向量

固定版本 `8449d574df6bae1bdf3314a1564da65e2f37e046` 的 `src/lightmem/memory/lightmem.py:618–623`，先设置 `new_payload["memory"]`，随后把 `entry.get("vector")` 原样传回数据库；没有重新编码新文本。这会使向量检索仍以旧语义表示新文本。

对远端冻结维护前后状态只读比对：17 条 memory 文本变化，17 条向量均在 1e-7 内不变；其中 9 条完全一致，其余最大差异约 1.49e-8，可由归一化精度解释。[数值回执与状态哈希](evidence/architecture/lightmem_vector_audit.json) 不包含文本或向量。

本轮不修改上游算法、不重建状态、不启动质量/协议支线。它仍是一个真实版本的执行记录，但不能据此声称正确维护了向量语义，也不能把此执行量当成修复后算法的成本。

## 新闭环的检查

新增小 batch 数值参考检查、有限 FIFO/部分和容量、权重跨行复用、M>8 回退、目标依赖就绪、终止事件、全部调用一次性消费。实现过程中还修正了零时长 barrier 导致的错误串行化，以及末事件恰好结束时的误报死锁；对应回归均已保留。

44 项单元检查通过；36 次闭环重放的 7,572 个服务账本通过独立守恒检查。旧 family 验证和冻结 mechanism 来源/预算验证也通过。测试与复现入口见 [BATCH_CLOSURE_RESULTS.md](BATCH_CLOSURE_RESULTS.md)。

M=1 限制是旧候选的明确适用范围，不能和上述软件错误混为一谈。bank/interface 效率、控制器面积、midpoint attention、免费 batch compaction 等是模型假设；本轮用它们限定结论，不把它们包装为已验证硬件事实。
