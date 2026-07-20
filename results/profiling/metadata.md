# Profiler metadata

- 对照：`BASE`（async on、MTP=3、`FULL_DECODE_ONLY`）与 `EAGER`（仅 `cudagraph_mode=NONE`）。
- 正式负载：decode-heavy，64 请求，concurrency=32，input≈1024，output=1024，temperature=0，ignore-eos。
- 采集：容器内 `msprof --application`，启动后延迟 600 秒，持续 180 秒；延迟期完成模型加载、首次编译和独立预热。
- 工具参数：`ascendcl=on runtime-api=on task-time=on hccl=on ai-core=on aic-metrics=PipeUtilization`。
- 原始报告：远端保留；BASE 4.3 GiB，EAGER 4.7 GiB。仓库只提交脱敏统计 CSV、聚合 CSV、图和复合 SHA256。
- 复合 SHA256 定义：对报告内每个文件生成排序的 `sha256sum` manifest，再计算该 manifest 的 SHA256。
- 限制：导出只覆盖一个 TP rank/device；`op_statistic` 无法把共享 MatMul 精确拆成 attention/FFN/MTP verify，故不伪造细分比例。

正式 benchmark：BASE mean TPOT 42.68 ms、MTP acceptance 64.23%；EAGER mean TPOT 112.60 ms、acceptance 63.84%。
