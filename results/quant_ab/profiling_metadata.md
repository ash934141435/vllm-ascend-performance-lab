# T3 profiling metadata（kernel 级归因采集说明）

- 目的：解释 T1 中单流（c=1）W8A8 相对 BF16 的 TPOT 劣势（+23.6%/+12.3%）成因。
- 方式：两实例启动参数自带 torch profiler（`profiler=torch`）；分别
  `POST /start_profile` → 顺序 6 条 `≈8 tok prompt / max_tokens=64 / c=1` 请求 → `POST /stop_profile`。
- 解析：daemon 内不停机出报告（服务日志提示），离线
  `torch_npu.profiler.profiler.analyse(<rank_dir>)`；本目录取 rank0 的
  `op_statistic.csv` 与 `step_trace_time.csv`（重命名加 quant 后缀）。
- 采集窗口锚点：两边各自第 1500 次 `FusedInferAttentionScore` 前后（请求 4 decode 稳态）；
  全窗口聚合值见根 README T3 小节。
- 完整产物（`kernel_details.csv` ≈100 MB、`trace_view.json` ≈840 MB、rank1–3、host/device 原始数据）
  留存于采集节点，未入库；需要可按 sha256 清单另发。
- 局限：单 rank、窗口级聚合、6 条短请求；CPU launch 与 NPU idle 未在桶级区分。
