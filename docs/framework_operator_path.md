# Qwen3.6-27B 混合解码器执行链

本文只描述服务器 `60`、容器 `qwen3.6-vllm` 内实际安装源码：

- vLLM `0.21.0`，commit `ad7125a431e176d4161099480a66f0169609a690`
- vLLM-Ascend `0.21.0rc1`，commit `80610e4438dba05011b05f89fc45d91e96992671`
- artifact `architectures=Qwen3_5ForConditionalGeneration`，`config.json` SHA256 `69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c`

源码为容器内 editable checkout：`/vllm-workspace/vllm` 与 `/vllm-workspace/vllm-ascend`。下列行号固定于上述 commit。

## 1. 请求到模型

| 阶段 | 固定源码位置 | 作用 | 可观测证据 |
|---|---|---|---|
| Engine/Scheduler | `vllm/v1/core/sched/scheduler.py:310` `Scheduler.schedule` | 在 token budget、KV/SSM cache 与等待队列约束下形成 `SchedulerOutput` | queue time、running/waiting、preemption |
| Ascend Model Runner | `vllm_ascend/worker/model_runner_v1.py:1904` `NPUModelRunner.execute_model` | 更新 request state、准备输入与 metadata、选择 eager/图执行、调用模型并采样 | Host timeline、launch gap、图任务数 |
| 模型 forward | `vllm/model_executor/models/qwen3_5.py:622` `Qwen3_5ForConditionalGeneration.forward` | 合并文本/多模态 embedding 后进入语言模型 | model-forward range、输入 token 数 |
| 层分派 | `vllm/model_executor/models/qwen3_5.py:118,227` | 按 `config.layer_types[layer_idx]` 构造 linear/full attention 层 | 层序号与 profiler stack |

入口张量按本轮纯文本请求为：`input_ids [T]`、`positions [T]`、hidden states `[T,5120]`。`T` 是本次 scheduler 实际调度 token 总数，不等于固定 batch size；prefill 可大于 1，稳定 decode 通常接近运行 request 数，MTP verify 时还包含 draft token。

## 2. artifact 的真实层结构

`config.json` 给出：64 层、hidden size 5120、FFN intermediate size 17408、24 个 attention query heads、4 个 KV heads、head dim 256、1 个原生 MTP 层。`layer_types` 为 `3 × linear_attention + 1 × full_attention` 重复 16 次，因此共有 48 个 Gated DeltaNet 层和 16 个 full-attention 层。

服务名 `Qwen3.6-27B` 只是 `--served-model-name`；运行时 architecture 仍是 `Qwen3_5ForConditionalGeneration`。本文不据服务名推断代码路径。

### 2.1 Gated DeltaNet 层（48 层）

真实分派点是 `qwen3_5.py:133` 创建的 `GatedDeltaNetAttention`，实现位于 `vllm/model_executor/layers/mamba/gdn_linear_attn.py:239`，不是独立的 `Qwen3_5GatedDeltaNet` 类。

| 节点 | 源码位置 | 主要 shape / TP 边界 | prefill 与 decode |
|---|---|---|---|
| RMSNorm | `qwen3_5.py:170` | `[T,5120]` | 相同算子，T 不同 |
| QKVZ projection | `gdn_linear_attn.py:409,658` | 输入 `[T,5120]`；linear Q/K/V heads 为 16/16/48，K/V head dim 128；列并行 projection 在 TP rank 切分输出 | prefill/decode 共用投影 |
| Conv1D/state input | `gdn_linear_attn.py:510,622` | conv kernel 4；维护 convolution state 与 recurrent state | prefill 写入一段状态；decode 每步更新状态 |
| Gated Delta rule | `gdn_linear_attn.py:1047` | rank-local projected heads 与本 rank state | prefill 走 chunk rule；decode 走 recurrent fast/non-spec 分支（`:1312,:1378`） |
| Ascend patch | `vllm_ascend/patch/worker/patch_qwen3_5.py` | patch 上游 `_forward_core`，使用支持 float32 SSM state 的 Triton 路径 | profiler 中应映射到 chunk/recurrent gated-delta kernel，不能只按 Python 类名归类 |
| output projection | `gdn_linear_attn.py:665` | 输出回到 `[T,5120]`；row-parallel projection 的 reduce 是该 attention block 的 TP collective 边界 | 两阶段相同，T 不同 |
| residual + RMSNorm + SwiGLU FFN | `qwen3_5.py:118` 继承 `Qwen3NextDecoderLayer.forward`；MLP 在 `qwen3_5.py:156` 创建 | `[T,5120] -> [T,17408]` gate/up，激活后 down 回 `[T,5120]`；row-parallel down projection reduce | prefill 更偏大矩阵；decode 更偏小 batch/带宽与下发开销 |

### 2.2 Gated full-attention 层（16 层）

`qwen3_5.py:140` 创建 `Qwen3NextAttention`。路径为 input RMSNorm、QKV projection、partial RoPE、GQA attention、output projection、residual、post-attention RMSNorm、SwiGLU FFN、residual。

主要 shape：hidden `[T,5120]`；Q 为 24 heads，K/V 为 4 heads，head dim 256。TP=4 时 projection 按 head/输出维切分；attention 在 rank-local heads 上执行；attention output 的 row-parallel projection 与 FFN down projection产生 collective。prefill 处理 causal token block 并写 KV cache；decode 从 paged KV cache 读取历史，仅为新 token 计算 Q/K/V。

partial RoPE 与 attention backend 的精确 profiler kernel 名必须由本轮导出的 operator CSV 确认；源码路径只能证明 dispatch，不能证明耗时占比。

## 3. MTP draft、target verify 与 sampling

本轮 BASE 使用原生 MTP：`num_speculative_tokens=3`。Ascend runner 中：

- `model_runner_v1.py:1542` 构造 speculative metadata；
- `:1642` 产生 draft token IDs；
- `:1904` 的 `execute_model` 运行 target forward/verify；
- `:2338` sampling，`:2553` speculative sampling；
- scheduler `scheduler.py:1601` 接收 draft token IDs 并安排下一步。

`/metrics` counter 只用于 draft/accepted token 计数和接受率。target verify 时间必须来自 profiler range/operator 汇总，再除以正式窗口实际 output/accepted token；不能由 counter 推断耗时。

正确性按 request ID 比较固定 prompt、`temperature=0` 下 MTP on/off 的完整 token IDs，并记录首个 divergence。

## 4. 图模式与 eager

Ascend runner 的模型调用入口为 `model_runner_v1.py:2756`，图分派在 `:2846`，图 capture 在 `:4820`。BASE 是 `FULL_DECODE_ONLY`：prefill 通常不从 decode graph 获益；EAGER 将 `cudagraph_mode` 单变量改为 `NONE`。比较应看 decode task 数、Host launch gap、NPU idle gap 与 TPOT，不能把 prefill 差异直接解释为图收益。

## 5. TP collective 的可验证边界

本项目能验证：各 rank 的投影/FFN 执行时间、row-parallel output 的 collective、runner 的 hidden-state gather（`model_runner_v1.py:2708`）以及工具确实导出的 HCCL 时间。不能验证或声称实现了 TP runtime、collective 算法或 HCCL 内核。

若 msProf 无法可靠导出 HCCL 分类，只报告 rank 间时间差和 collective 邻接空洞，不把未知时间强行归为通信。

## 6. profiler 映射与证据边界

| profiler 类别 | 回映代码节点 | 判定限制 |
|---|---|---|
| Scheduler/Host | `Scheduler.schedule`、`NPUModelRunner.execute_model` | 需 Host timeline；仅有端到端延迟不够 |
| Gated DeltaNet | QKVZ/Conv/chunk or recurrent delta/output projection | 需 operator 名或调用栈支持 |
| Attention | QKV/RoPE/GQA/output projection | 需区分 prefill/decode 窗口 |
| FFN/Norm | RMSNorm、gate/up、SwiGLU、down | 融合算子按实际导出名归类 |
| MTP verify | target forward/verify range | 与正式 token counter 联合计算 |
| 图执行 | runner graph dispatch/captured task | 比较 BASE/EAGER task 与 gap |
| TP/HCCL | row-parallel reduce、runner gather | 只在工具可靠导出时报告通信时间 |

瓶颈分类遵循 `configs/analysis_rules.yaml`：第一类别至少占 30%，且领先第二名至少 5 个百分点；否则为 `mixed`。本文证明真实调用位置与可观测边界，不预填 profiler 结论。
