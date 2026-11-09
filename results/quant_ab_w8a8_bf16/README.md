# 量化 A/B：W8A8 vs BF16（Qwen3.8-27B，910B4 TP=4）

同一节点、同一套 serving 参数下并跑 BF16 与 W8A8 两个 vLLM 实例（各占 4 卡）的量化对比。
服务参数两侧完全一致，唯一差异为 `--quantization=ascend`：
`max-num-seqs=32 / max-num-batched-tokens=16384 / max-model-len=262144 / TP=4 / MTP=3 / FULL_DECODE_ONLY / prefix-caching on`。

- 硬件：8 × Ascend 910B4-1（64 GB HBM）；BF16 占 4 卡、W8A8 占另 4 卡
- 软件：vLLM 0.23.0 + vllm-ascend（modelslim W8A8）、CANN 8.3.RC2、driver 26.0.rc1
- 方法：`vllm bench serve`，random 数据集定长、`seed 0`、`temperature 0`、`request-rate inf`、`ignore-eos`、
  `num_prompts = 5 × concurrency`；每 cell 预热、3 次取中位数、按轮次 ABBA 交错、
  负载门控（对端空闲 + host load 上限）+ Prometheus 前后采样
- 判定（跑前冻结）：效应量 ≥5% 且 3/3 同向；guardrail（P99 TTFT、mean TPOT）回退 >3% 不推荐

## T1 主实验（in∈{1024,4096} × c∈{1,8,32} × out=512 × 3 次重复 = 36 次运行，全部有效）

逐 cell 中位数见 [`t1_summary_median.csv`](t1_summary_median.csv)，A/B 判定见 [`t1_comparison.md`](t1_comparison.md)。

核心结论（W8A8 相对 BF16）：

| 场景 | 输出吞吐 | TPOT | 结论 |
|---|---|---|---|
| c=1（单流） | −18.1% / −9.9%（in=1024/4096），均 3/3 同向 | +23.6% / +12.3% | BF16 明确更优 |
| c=8 | −3.7% / +6.2%（≤3/3 同向） | +3.5% / +6.2% | 无有效差异 |
| c=32 | +1.2% / −3.7% | −0.7% / +1.9% | 等价 |

例外：`4096/c8` 的 P99 TTFT −60.9%（3287 vs 8406 ms，3/3 同向）、P99 E2EL −23% —— W8A8 显著压缩长输入中并发尾延迟。

## T3 kernel 级归因（c=1，torch profiler，rank0）

`step_trace`（µs）：BF16 Computing 3.413M / 非重叠通信 3.681M；W8A8 2.693M / 5.528M。

| 层面 | BF16 | W8A8 | 差 |
|---|---:|---:|---|
| kernel 总时间 | 3,413,051 | 2,692,889 | −21% |
| MatMul 路径合计 | 2,126,979（MatMulV2 ×45220） | 1,503,272（QuantBatchMatmulV3 + MatMulV2 + DynamicQuant） | −29% |
| 非重叠通信 | 3,681,436 | 5,528,458 | +50% |

结论：**量化算子本身更快（DynamicQuant 仅占 4.5%），单流劣势来自 quant→QMM→dequant 依赖链
（`fuse_attn_quant=False` 等）破坏计算/通信重叠，TP4 allreduce 被串行暴露**；
高并发流水打满后差距消失，与 T1 自洽。逐算子证据见 [`t3_op_statistic_bf16.csv`](t3_op_statistic_bf16.csv) /
[`t3_op_statistic_w8a8.csv`](t3_op_statistic_w8a8.csv)、[`t3_step_trace_*.csv`](t3_step_trace_bf16.csv)。

## 边界

仅限上述矩阵、TP=4、两模型各占 4 卡同机并行、突发负载（`request-rate inf`）；
不外推低 QPS、262144 长上下文、c∈{16,24}、out=1024 或其他 TP 配置。
贪心输出一致性（W8A8 vs BF16 token 级）不在本实验内，见相关仓库 vllm-ascend-quant-eval。
