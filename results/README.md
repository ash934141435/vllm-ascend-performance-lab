# results 目录索引

| 目录 | 实验 | 内容 |
|---|---|---|
| `stage1_serving/` | 实验一：Qwen3.6-27B Serving 参数调优（阶段一） | `summary_median.csv`（90 行）、`valid_results.csv`（270 条 canonical）、`figures/` |
| `execution_ablation/` | 实验二：执行机制消融（阶段二） | `summary.csv` / `raw.csv` / `correctness.csv` / prompts |
| `profiling/` | 实验二：CPU/NPU Profiler 汇总 | `exported_metrics.csv`、`metadata.md` |
| `quant_ab/` | 实验三：Qwen3.8-27B W8A8 vs BF16 | `summary.csv`（含 `_vs_base_pct`）、`raw.csv`（36 attempt）、T3 `op_statistic_*` / `step_trace_*`、`profiling_metadata.md` |

各实验环境与结论见根 README 对应小节；大型原始产物（trace、逐次 JSON、profiler 全量）不入库。
