# K 分片与局部归约：可执行映射及资源合同

这是一项**有资源计费的分析候选**，不是已完成的 RTL/PPA 实现，也不声称 K 分片是新机制。实现：[mechanism.py](architecture/mechanism.py)，独立参考运算和计数检查：[mechanism_unittest.py](mechanism_unittest.py)。

1. 对 M=1 的 `y=xW`，每个 R×C 阵列分为 g 个 `(R/g)×C` 子阵列。每组读取连续 K 分片，保持 FP32 部分和。不同阵列处理不同 N tile。K/N 尾部补齐计入周期。
2. 权重预排布为 `[N tile, K group, K offset, column]`，等价于对 K 连续分组的列块布局；普通模式通过地址发生器读取同一份数据。没有复制一份额外权重。一次性权重重排不计入本次请求服务模型，不能据此声称全生命周期无代价。
3. K 分区使用双缓冲持续流动。一个 N tile/K 分区对应一条连续 DMA stream，buffer refill 不强制重启外部传输；额外提供逐 refill 开新传输的保守控制。供数不是零时间：沿用有限 DRAM 带宽、时延、32 个 outstanding groups 以及 SRAM 服务限制。
4. 所有组完成后，将局部 FP32 部分和送入二叉树。每阵列 16 个归约输出 lane，FP32 adder 假设 3 周期流水，8 周期同步；链路总宽每阵列 64 B/cycle。计算/供数阶段与最终归约阶段串行，不默认完美重叠。
5. 只在原映射和新映射都计入上述代价后择快。长 M 和 attention 保持普通映射；整颗装有扩展的设备仍承担新增逻辑、静态功耗和降频，不能只给受益算子收费。

对于单个 M=1 投影，记 `q=ceil(N/C)`、`waves=ceil(q/arrays)`、`Kp=ceil(K/g)`：

```text
array_cycles = waves * (Kp + R/g + C - 2)
DRAM weights = 2*K*N bytes
DRAM input   = 2*K*q bytes
DRAM output  = 2*N bytes
FP32 extra adds = (g-1)*N
reduction link traffic = 4*N*(2*g-2) bytes
tree_cycles = waves * (ceil(C/16) + 3*ceil(log2(g)) + 8)
T = max(array, DRAM, SRAM during MACs)
  + max(tree, reduction link, partial-buffer SRAM)
```

部分和写入、读出与最终结果写入均计局部 SRAM 字节及能量；不假设免费片上归约，也不把它错误计成 DRAM 流量。每阵列已有 SRAM 端口按 32 banks×16 B/cycle 的聚合上限解释，保留 2×/4× bank skew 控制。bank 平衡布局尚未经过逐周期 DRAM/SRAM 验证。g 增大时峰值供数需求 `2*g*C*f` 可以超过端口能力，此时服务时间强制停顿。

双缓冲权重、输入与部分和都必须装入现有 SRAM；部分和预留 `arrays*2*4*g*C` bytes，不能扩充容量。MAC 数量、DRAM 容量和外部带宽保持候选配置原值。原 HBM 参照下的额外资源如下：

| g | FP32 adders | 地址发生器 | 部分和预留 KiB | 新增逻辑 mm²估计 | 时钟比例 | 功率包络 W |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 256 | 32 | 32 | 1.774 | 0.9950 | 362.35 |
| 4 | 768 | 64 | 64 | 1.902 | 0.9901 | 362.54 |
| 16 | 3840 | 256 | 256 | 2.670 | 0.9804 | 363.66 |
| 64 | 16128 | 1024 | 1024 | 5.742 | 0.9709 | 368.15 |

面积按 adder 数、两路 BF16 operand 复用器 bit 数、地址发生器计费；系数为公开待校准假设：adder=0.5×原 MAC 面积，mux=2e-7 mm²/bit，地址发生器=0.002 mm²。局部互连包含在端口/mux 的集总假设内，**没有独立布线面积或线长提取**。时钟比例 `1/(1+0.005*log2(g))` 是敏感性参数，不是 STA 结果。额外 FP32 累加、输入广播、树传输及复用器动态能量，以及附加逻辑泄漏均收费；完整参数可从 `KShard` 读取。功率用保留原峰值加新增活动的保守包络。

预算判断来自这些明确但未校准的系数，因此“通过预算”只表示分析模型内通过。4× 开销、较大时钟惩罚、bank skew 和不合并 refill 的控制都保留。未验证的布线可行性、数值重排对完整 LLM 的影响及能耗系数不能被小幅预测收益覆盖。

参考运算采用 FP32 分片和树合并，随机尾部/小矩阵与独立 NumPy matvec 比较。它不等于 BF16 输入/FP32 FMA 的逐位等价，也没有在新硬件上重跑问答。

文献边界：

- [H²-LLM，ISCA 2025](https://leesou.github.io/publications/2025-4-h2llm)：本地 PDF §4.2/Eq.1 已同时考虑 K 和 N 划分，权衡结果合并和输入复制；其 §4.1/4.2 还包含 bank/PE 绑定、全局输入 buffer 和局部权重/输出 buffer。本候选是统一阵列内的行分组，未复现它的 HB 域或跨通道 DSE，因此不能将胜过原低 M 映射称为胜过 H²-LLM。
- TETRIS，ASPLOS 2017，本地 PDF §3.3–4：累加位置、SRAM/PE 面积和数据移动共同决定取舍。其 DRAM 累加器的工艺与位宽不同，没有把论文的面积数字直接移植成本模型。

两份本地 PDF 的路径和 SHA256 已在 `evidence/phase2/overview_provenance.json` 冻结。此次核查关注上述具体章节，不据一般“近存”标签预设 HB 优势。
