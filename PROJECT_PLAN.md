# vLLM-Ascend 框架、模型与异构性能优化计划

## 1. 目标

本计划沿现有 Serving 参数调优继续推进，不另建独立项目：

```text
固定 workload 和环境
→ 找到 max-num-seqs 引起的排队瓶颈
→ 得到 32/16384 候选配置
→ 分析框架执行机制和模型算子
→ 使用 CPU/NPU Profiler 定位瓶颈
→ 完成一次单变量优化和 A/B 回归
```

最终证明三项能力：

1. 实际使用并调优大模型推理框架；
2. 理解混合解码器中的 Gated DeltaNet、Gated Attention、FFN、MTP 和张量并行执行路径，并完成推理模型调试；
3. 能分析服务器端 CPU/NPU 异构瓶颈并验证优化效果。

本项目不证明训练经验，也不声称实现了自定义算子、Tensor Parallel runtime 或 HCCL 通信内核。

## 2. 已完成基线

现有仓库已经完成：

- TP=4 的 Qwen3.6-27B vLLM-Ascend 部署；
- 9 组服务配置、90 个测试场景和 270 条 canonical 结果；
- 对吞吐、TTFT、TPOT、排队、KV Cache 和异常结果的采集分析；
- 定位 `max-num-seqs=16` 导致的高并发排队；
- 得到 `max-num-seqs=32`、`max-num-batched-tokens=16384` 候选配置；
- 验证吞吐和 TTFT 改善，以及 TPOT 上升的代价。

新实验以候选配置为基线，不重跑原有完整矩阵。

## 3. 固定环境

所有远程实验统一在以下位置操作：

```text
服务器：60服务器
本机 SSH 别名：codex-60
远程工作目录：/mnt/nvme0/qwen3.5-vllm
目标容器：qwen3.6-vllm
目标 NPU：0,1,2,3
```

连接和进入工作目录：

```bash
ssh codex-60
cd /mnt/nvme0/qwen3.5-vllm
```

`qwen3.5-vllm` 是历史目录名，不代表本轮实验的模型版本。实际模型、tokenizer 和架构以容器挂载路径、模型 `config.json` 及每轮 `manifest.json` 为准。不得在其他服务器或目录生成可与本计划结果合并的数据。

```text
模型：Qwen3.6-27B W8A8
硬件：4 × Ascend 910B4-1
并行度：Tensor Parallel = 4
vLLM：0.21.0
vLLM-Ascend：0.21.0rc1
CANN：9.0.0
torch / torch_npu：2.10.0
triton-ascend：3.2.1
max-num-seqs：32
max-num-batched-tokens：16384
```

当前基线执行机制：

```text
async-scheduling：开启
MTP：开启，num_speculative_tokens=3
图模式：FULL_DECODE_ONLY
CPU binding：开启
```

首次执行先生成冻结的 `expected_manifest.json`。每轮实验生成 `results/<run_id>/manifest.json`，至少保存：

- 镜像 digest、vLLM commit 和 vLLM-Ascend commit；
- 模型与 tokenizer revision、`config.json` SHA256；
- CANN、driver、firmware、torch、torch_npu、triton-ascend 和 kernel 版本；
- CPU governor、NUMA/CPU binding、环境变量、MTP 和完整启动参数；
- 服务启动日志、测试前后 `/metrics`、原始 benchmark 输出和退出码；
- CPU load 和目标 NPU 0–3 的利用率、显存、健康状态及进程列表。

manifest 与冻结基线不一致立即终止，不把不同软件栈的结果放入同一 A/B。只允许 `run_id`、时间戳和显式实验变量不同。

服务器上的其他服务可以保持运行，但测试期间必须无业务负载。若发现目标设备外的异常 Host 负载或目标设备被其他进程占用，本轮结果作废。

## 4. Workload

### 4.1 历史基线复核

先使用现有随机定长输入复跑候选配置，确认新一轮环境没有超出历史波动：

```text
输入：1024
输出：1024
并发：32
重复：3 次
```

漂移阈值由第6节的机器规则判定。超阈值时先排查服务参数、后台负载、软件版本和预热，不进入消融实验。

### 4.2 新实验 workload

新实验使用固定、可公开的真实文本 prompt，所有配置复用完全相同的请求顺序和生成参数：

| 场景 | Prompt token 范围 | 固定输出 | 并发 | 目的 |
|---|---:|---:|---:|---|
| Prefill-heavy | 约 4096 | 128 | 1 | DeltaNet、Attention、FFN 和首 token 路径 |
| Decode-heavy | 约 1024 | 1024 | 32 | 调度、MTP、图模式和 decode |

要求：

- 保存 prompt 数据集名称、版本、筛选规则和 SHA256；
- 记录每条请求的真实 token 数，不把 token 范围写成精确长度；
- 固定请求顺序、seed、`temperature=0` 和输出上限；
- 不使用随机 token 结果解释 MTP 接受率；
- 不把公开数据集包装成真实生产流量。

性能 workload 统一固定：

```yaml
num_prompts: 64
request_mode: closed_loop
request_rate: null
timeout_seconds: 1800
retry_count: 0
connection_reuse: true
client_host: <recorded-host>
network_path: localhost
ignore_eos: true
```

若客户端改为远程运行，必须将 `network_path` 记为 `remote`，不得与 localhost 结果混合。性能测试使用 `ignore_eos=true` 保证等量工作；正确性测试单独使用自然 EOS 和较短输出，不进入性能 CSV。

## 5. 消融实验

只测试基线和三个单变量对照：

| ID | 与基线相比的唯一变化 | 主要问题 |
|---|---|---|
| BASE | 无 | 当前完整配置表现如何 |
| ASYNC_OFF | 关闭 `async-scheduling` | 异步调度是否改善 Host/NPU overlap |
| MTP_OFF | 关闭 MTP | speculative decoding 是否真正带来收益 |
| EAGER | 主模型切换为 eager | decode 图模式是否减少下发开销 |

固定矩阵：

```text
4 个配置 × 2 个 workload × 3 次独立重复 = 24 个结果
```

每个 workload 内轮换配置顺序：

```text
重复1：BASE → ASYNC_OFF → MTP_OFF → EAGER
重复2：MTP_OFF → EAGER → BASE → ASYNC_OFF
重复3：EAGER → BASE → ASYNC_OFF → MTP_OFF
```

这不是随机化证明，只用于降低固定顺序带来的编译缓存、温度、内存碎片和长时间运行漂移。

每次配置切换必须：

1. 从同一份基线启动脚本生成配置；
2. 保存可审查的参数 diff；
3. 重启服务；
4. 检查 `/v1/models` 和实际进程参数；
5. 完成独立预热；
6. 确认指标计数器的采样起止点；
7. 运行正式测试；
8. 检查错误、抢占、服务重启和后台负载。

### 5.1 通用指标

- Output throughput；
- Request throughput；
- TTFT P50/P95/P99；
- Mean/P99 TPOT；
- Mean/P99 ITL；
- E2E latency；
- queue time；
- running/waiting requests；
- KV Cache 峰值；
- preemption、失败请求和服务重启。

结果以三次独立运行的中位数为主，同时保留每次原始值和波动。

### 5.2 MTP 指标与正确性

从 `/metrics` 的测试前后差值计算：

- draft 数；
- draft token 数；
- accepted token 数；
- 总接受率；
- 平均接受长度；
- 每个 draft position 的接受率；

Prometheus counter 只用于计数和接受率，不声称能直接得到验证耗时：

```text
acceptance_rate = accepted_tokens / draft_tokens
mean_accepted_length = accepted_tokens / drafts
verify_overhead_per_output_token
  = profiler 中 target verify 总时间 / 正式窗口实际输出 token 数
verify_cost_per_accepted_token
  = profiler 中 target verify 总时间 / accepted token 数
```

后两个指标必须来自 profiler 与正式窗口 token 计数的联合计算，并标注数据来源。

使用固定 prompt、`temperature=0` 比较 MTP 开关后的输出 token IDs。出现不一致时先定位原因，不将该配置直接推荐为优化结果。

## 6. 机器判定规则

新增 `configs/analysis_rules.yaml`。阈值先由现有270条结果中相同 cell 的重复数据计算，再冻结到本轮实验配置；不能跑完后按结果改阈值。

最小规则：

```yaml
baseline_drift:
  method: historical_median_mad
  max_relative_deviation: "max(0.05, 3 * MAD / median)"

invalid_run:
  foreign_process_on_target_npu: true
  failed_requests_gt: 0
  service_restart_gt: 0
  missing_metrics_snapshot: true
  benchmark_exit_code_ne: 0
  manifest_mismatch: true
  host_load_threshold_source: preflight_idle_window

performance_decision:
  minimum_effect_size_pct: 5
  maximum_guardrail_regression_pct: 3
  require_same_direction_runs: 3
  aggregation: median

correctness:
  compare_by_request_id: true
  exact_token_ids: true
  capture_first_divergence: true

profiler_bottleneck:
  minimum_category_share_pct: 30
  minimum_lead_over_second_pct_points: 5
  otherwise: mixed
```

`5%` 只是最低效应量下限；若历史 `3*MAD/median` 更大，使用更大的阈值。Host load 数值在正式实验前通过无请求空闲窗口采样并冻结。每项实验必须在配置中指定一个 primary metric 和 guardrail metrics，例如：

- async：decode throughput 为 primary，TTFT/TPOT 为 guardrail；
- MTP：decode TPOT 为 primary，吞吐和正确性为 guardrail；
- 图模式：decode TPOT 为 primary，吞吐和 TTFT 为 guardrail。

只有效应超过阈值、三次配对结果方向一致、且 guardrail 回归不超过阈值时，才判定为收益。否则输出无显著收益或存在取舍。

机制负对照必须提前写入预期：

- `ASYNC_OFF × Prefill C1`：预期变化较小；
- `MTP_OFF × Prefill`：只影响短暂 decode 尾部；
- `EAGER × Prefill`：BASE 为 `FULL_DECODE_ONLY`，prefill 通常不应获得图模式收益。

负对照超出阈值时先排查混杂因素，不能直接解释成机制收益。

## 7. 框架和模型执行链

基于服务器实际安装的 vLLM 和 editable vLLM-Ascend 源码，形成：

```text
docs/framework_operator_path.md
```

服务器模型配置已经确认：服务名为 `Qwen3.6-27B`，实际 artifact 的 `architectures` 为 `Qwen3_5ForConditionalGeneration`；分析以该 artifact 的 `config.json` 和已安装源码为准。其64层按 `3 × linear_attention + 1 × full_attention` 重复16次，`linear_attention` 对应 `Qwen3_5GatedDeltaNet`，并包含原生 MTP 层。文档按两条真实层路径梳理：

```text
Request
→ Engine / Scheduler
→ Model Runner
→ Layer Dispatcher

Gated DeltaNet Layer（48层）：
  RMSNorm
  → Input Projection
  → Causal Conv1D / State Update
  → Gated DeltaNet
      prefill: chunk gated delta rule
      decode: recurrent gated delta rule
  → Gated Norm / Output Projection
  → Residual
  → RMSNorm
  → FFN / SwiGLU
  → Residual

Gated Attention Layer（16层）：
  RMSNorm
  → QKV Projection
  → Partial RoPE
  → GQA Attention
  → Output Projection
  → Residual
  → RMSNorm
  → FFN / SwiGLU
  → Residual

→ TP Collective
→ MTP Draft
→ Target Verify
→ Sampling
```

每个节点记录：

- 固定版本中的源码位置；
- 主要输入输出 shape；
- 对应的 profiler 算子或任务；
- prefill 和 decode 下的执行差异；
- TP collective 的实际边界；
- 当前项目能够验证和不能验证的内容。

不复制大段框架源码，不根据类名猜测调用链。

## 8. CPU/NPU Profiling

不对全部24个结果采集 msProf。

先完成普通 benchmark，再按第6节的效应量排序选择一组单变量对照。至少覆盖：

- BASE；
- 一个对应的对照配置。

如果 async 和图模式都超过效应量阈值，可各增加一组短时间采集；总采集配置不超过3个。

### 8.1 采集前验证

1. 记录 `msprof` 版本和实际支持的参数；
2. 做一次短时间探针；
3. 确认能够导出 Host/NPU 时间线和算子汇总；
4. 确认采集窗口不包含模型加载和首次编译；
5. 确认 Profiling 开销，不用 profiler 延迟替代普通 benchmark 延迟。

### 8.2 分析内容

- Scheduler、Host 下发和 NPU 执行的时间关系；
- Host launch gap 和 NPU 空闲区间；
- prefill/decode 的 Top N 耗时算子；
- Gated DeltaNet、Gated Attention、FFN、Norm、MTP verify 的时间占比；
- 图模式与 eager 的任务数量和间隙差异；
- 各 TP rank 的执行时间差异；
- HCCL/collective 时间——仅在工具能可靠导出时；
- 按第6节分类规则判定 CPU、NPU、调度、算子、通信或 `mixed`。

大型原始 profiler 报告不直接提交 Git。仓库保存脱敏后的 CSV、关键截图、采集命令、工具版本和原始报告 SHA256。

高并发下 MTP 没有收益不是失败。Speculative decoding 的典型受益区间是中低QPS、memory-bound decode；并发32下若验证成本或批处理效率抵消收益，应如实记录为适用边界。

## 9. 优化闭环

消融和 Profiling 完成后，只允许追加一个由证据决定的调优实验：

| 证据 | 后续动作示例 |
|---|---|
| async scheduling 达到效应阈值且 guardrail 通过 | 保留并说明适用并发范围 |
| async scheduling 未达到阈值或产生回归 | 关闭并完成回归验证 |
| MTP 后段接受率低 | 根据分位接受率测试一个更小的 speculative token 数 |
| MTP 验证成本高于收益 | 对该 workload 关闭 MTP |
| 图模式达到效应阈值并减少 decode 下发间隙 | 保留 `FULL_DECODE_ONLY` |
| eager 达到效应阈值且 guardrail 通过 | 记录原因并调整执行模式 |

最终结论必须形成：

```text
现象
→ 性能假设
→ Profiler/指标证据
→ 单变量修改
→ 正确性回归
→ 性能回归
→ 收益和适用边界
```

如果证据表明当前基线已经合理，消融实验本身可以证明配置选择；不得为了制造“优化”而修改无关参数。

## 10. 仓库结构与 README 整合

### 10.1 整合原则

本轮工作是原有并发测试的第二阶段，不建立平行项目：

```text
阶段一：容量和调度参数测试
  max-num-seqs / max-num-batched-tokens / 输入长度 / 并发
  → 定位 max-num-seqs 引起的排队
  → 选出 32/16384 候选配置

阶段二：候选配置下的执行机制测试
  Async / MTP / 图模式
  → 区分调度、模型算法和 Host/NPU 执行收益
  → Profiler 解释差异
  → 形成最终配置建议
```

两个阶段共享相同的环境采集、服务重启、预热、指标采样、异常判定和结果聚合逻辑。阶段二必须在 README 中明确引用阶段一产生的 `32/16384`，不得把它写成无来源的新 baseline。

### 10.2 最终目录

保留现有文件路径，避免破坏 README、历史链接和结果追溯；只为新实验增加必要文件：

```text
vllm-ascend-performance-lab/
├── README.md
├── PROJECT_PLAN.md
├── configs/
│   ├── experiment.yaml                 # 阶段一：并发/容量矩阵
│   ├── execution_ablation.yaml         # 阶段二：Async/MTP/图模式矩阵
│   ├── analysis_rules.yaml             # 两阶段统一判定规则
│   └── expected_manifest.json          # 阶段二冻结环境
├── scripts/
│   ├── bench_matrix.sh                 # 已有：公共运行流程和阶段一入口
│   ├── bench_report.py                 # 已有：公共解析、判定和聚合
│   ├── plot_results.py                 # 已有：阶段一图表
│   └── bench_execution_ablation.sh     # 阶段二薄入口，复用已有流程
├── tests/
│   ├── test_bench_matrix.sh
│   ├── test_bench_report.py
│   └── test_execution_ablation.py      # 只测新配置、配对和规则解析
├── results/
│   ├── valid_results.csv               # 阶段一：270条 canonical 结果
│   ├── summary_median.csv              # 阶段一：90行中位数
│   ├── figures/                        # 阶段一图表，路径保持不变
│   ├── execution_ablation/             # 阶段二公开结果
│   │   ├── raw.csv
│   │   ├── summary.csv
│   │   ├── correctness.csv
│   │   └── figures/
│   └── profiling/                      # 阶段二脱敏 Profiler 证据
│       ├── metadata.md
│       ├── exported_metrics.csv
│       └── figures/
└── docs/
    ├── architecture.drawio             # 更新为两阶段实验架构
    ├── architecture.svg
    ├── framework_operator_path.md      # 混合解码器真实执行链
    └── execution_optimization.md       # 阶段二 A/B、Profiler 和结论
```

远程原始运行目录使用：

```text
/mnt/nvme0/qwen3.5-vllm/bench_runs/<run_id>/
├── manifest.json
├── configs/
├── logs/
├── metrics/
├── results_raw/
└── profiler/
```

原始日志和大型 Profiler 报告留在服务器；脱敏汇总、关键图和 SHA256 进入仓库。公开仓库不创建单独的 `<run_id>` 顶层目录，避免目录膨胀。

`bench_execution_ablation.sh` 只负责选择阶段二配置和轮换顺序；服务检查、预热、采样和报告逻辑优先调用已有脚本/函数，不复制一套 benchmark 框架。

### 10.3 最终 README 结构

README 标题从“高并发参数调优”扩展为：

```text
Qwen3.6-27B vLLM-Ascend Serving 与执行机制性能优化
```

最终按以下顺序编排：

1. **项目摘要**
   - 一句话说明在 TP=4 Ascend 服务上，从容量参数下钻到执行机制和 CPU/NPU Profiling；
   - 列出阶段一候选配置和阶段二最终建议，但数字必须链接到结果文件。
2. **问题与推进路线**
   - 展示统一链路：容量测试 → 排队瓶颈 → 候选配置 → 执行机制消融 → Profiler → 优化闭环；
   - 明确阶段二为什么从 `32/16384` 开始。
3. **实验环境与模型架构**
   - 保留现有硬件/软件表；
   - 补充实际 artifact 是混合 Gated DeltaNet/Gated Attention 解码器；
   - 说明服务名与 artifact 架构名的区别。
4. **实验方法和可信度控制**
   - 公共预热、重复、中位数、manifest、历史 MAD、无效运行和正确性规则；
   - 区分固定输出性能测试与自然 EOS 正确性测试。
5. **阶段一：并发和容量参数调优**
   - 保留现有测试矩阵、五张图、关键发现和 baseline/candidate 表；
   - 结尾固定输出阶段一决策：`32/16384` 进入阶段二。
6. **阶段二：执行机制消融**
   - 展示 BASE、ASYNC_OFF、MTP_OFF、EAGER；
   - 分别报告 prefill/decode 结果、MTP 接受率和 guardrail；
   - 负对照与正式结论分开写。
7. **CPU/NPU Profiling 与模型算子解释**
   - 将端到端差异映射到 Scheduler、Gated DeltaNet、Gated Attention、FFN、MTP verify、图执行和 TP collective；
   - 只展示可由导出数据支持的结论。
8. **从生产 Baseline 到最终建议**
   - 用一张表串联：原配置 `16/8192` → 阶段一 `32/16384` → 阶段二执行机制建议；
   - 同时列出吞吐、TTFT、TPOT、正确性、代价和适用负载。
9. **如何复现**
   - 分成阶段一命令、阶段二命令和 Profiler 命令；
   - 标明远程真实路径只用于内部执行，公开示例继续使用环境变量。
10. **能力边界与结果索引**
    - 明确没有做训练、自定义算子、TP runtime 或 HCCL 内核；
    - 链接阶段一 CSV、阶段二 CSV、Profiler 汇总和两份深入文档。

README 不复制 `framework_operator_path.md` 和 `execution_optimization.md` 的全部内容，只保留面试官能在3–5分钟内读完的核心证据和链接。

### 10.4 README 更新时机

README 不提前填写阶段二数字，按证据逐步更新：

| 阶段 | README 修改 |
|---|---|
| P1 | 更新标题、项目路线、目录导航；阶段二标记为“进行中” |
| P2 | 加入消融矩阵、结果表和MTP正确性；数字来自 `summary.csv` |
| P3 | 加入模型路径与Profiler图；结论链接导出CSV |
| P4 | 加入“Baseline → Candidate → Final”总表和能力边界，移除“进行中” |

README 中每个性能数字都必须能追溯到一个 CSV 行；每张阶段二图必须由提交的 CSV 生成。历史阶段一数字继续以现有 `summary_median.csv` 为唯一数据源。

## 11. 阶段与验收

### P1：环境和 workload 固定

- 候选配置复跑通过；
- 真实 prompt 数据集固定并记录 hash；
- expected manifest 和机器规则已冻结；
- BASE 输出正确；
- 后台负载检查可用；
- README 已改为两阶段项目结构，阶段二明确标记为进行中；
- 架构图包含阶段一到阶段二的输入输出关系。

### P2：24组消融实验

- 四种配置均可启动并验证实际生效；
- 两个 workload 各完成三次重复；
- 原始结果、指标和参数 diff 完整；
- MTP token 一致性和接受率可计算；
- README 阶段二结果表仅从 `results/execution_ablation/summary.csv` 生成。

### P3：源码与 Profiler 证据

- 完成一条真实框架/模型调用链；
- profiler 报告避开加载和首次编译；
- 至少完成 BASE 与一个对照的 Host/NPU 分析；
- profiler 中的主要算子能够映射回模型阶段；
- README 的 Profiler 结论均链接到导出数据或深入文档。

### P4：优化闭环

- 根据证据选择且只选择一个后续调整；
- 完成正确性和性能 A/B；
- 明确收益、代价、适用 workload 和能力边界；
- README 完成“16/8192 → 32/16384 → 最终执行机制”的总表；
- README 中所有数字可追溯到提交的 CSV 或 profiler 导出；
- README、架构图、两份深入文档和结果索引互相一致。

## 12. 提交顺序

```text
P1: add execution-ablation workflow and integrate two-stage README structure
P2: add async, MTP and graph-mode ablation results and README evidence
P3: document hybrid-model path and CPU-NPU profiling evidence
P4: add final configuration decision and complete README narrative
```

工作流代码与真实实验结果分开提交。没有在服务器执行的阶段标记为未验证，不生成示例性能数据。

## 13. 时间和边界

| 阶段 | 预计时间 |
|---|---:|
| P1 环境、真实 prompt 和基线复核 | 2–4 小时 |
| P2 脚本和24组消融实验 | 6–10 小时 |
| P3 源码映射和 msProf | 5–9 小时 |
| P4 单变量调优、回归和文档 | 4–7 小时 |
| 合计 | 17–30 小时 |

明确不做：

- 训练或微调实验；
- 自定义 Ascend 算子；
- 重写 Scheduler、MTP 或 TP runtime；
- HCCL 算法或通信内核开发；
- TP=2/4/8 大矩阵；
- 与岗位目标无关的额外框架 patch。
