# 执行机制优化闭环

## 消融结论

24/24 次正式运行有效，每个配置、工作负载均有 3 次轮换重复。数字见 [`summary.csv`](../results/execution_ablation/summary.csv)。

- `ASYNC_OFF`：decode 吞吐仅 +2.12%，低于冻结的 5% 最小效应量；prefill 吞吐 -1.35%。无显著净收益。
- `MTP_OFF`：decode TPOT +21.78%、吞吐 -18.64%；prefill 吞吐 -30.05%。保留 MTP。
- `EAGER`：decode TPOT +175.08%、吞吐 -63.77%；prefill TPOT +298.22%、吞吐 -67.76%。保留 `FULL_DECODE_ONLY`。

MTP 正确性门禁没有通过：固定 prompt、temperature=0、自然 EOS 下仅 3/8 请求 token IDs 完全一致，首个分歧见 [`correctness.csv`](../results/execution_ablation/correctness.csv)。因此 `MTP_OFF` 不作为优化候选；该结果也说明不能用语义相近替代计划要求的精确 token 门禁。

## Profiler 解释

BASE 与最强对照 EAGER 使用相同 decode-heavy 正式负载。聚合数据见 [`exported_metrics.csv`](../results/profiling/exported_metrics.csv)，图见 [`operator_category_share.svg`](../results/profiling/figures/operator_category_share.svg)。

- BASE 的 NPU 最大类别为 data movement（34.86%），比第二类 other（17.90%）高 16.95 个百分点，满足冻结的瓶颈判定阈值。
- EAGER 的 HCCL 汇总时间为 23.08 s，BASE 为 1.31 s，约 17.6 倍；Host `node:launch` 次数从 290,855 增至 698,165，约 2.40 倍。
- 正式窗口 TPOT 从 42.68 ms 增至 112.60 ms，而 MTP 接受率 64.23% 与 63.84% 基本不变。退化主要对应图执行取消后的 launch/collective 开销，不是接受率变化。

## Final

证据没有支持新的安全配置。最终配置保持阶段二 BASE：`max-num-seqs=32`、`max-num-batched-tokens=16384`、async on、MTP=3、`FULL_DECODE_ONLY`。这不是“无优化”：单变量 A/B 排除了三个看似可调开关，并由 Host/NPU 数据解释最强回归。未追加第二个探索旋钮，避免在正确性门禁失败后制造不可推荐结果。

适用边界：结论针对 TP=4、NPU 0–3、突发 concurrency=32、固定输出长度性能测试；不外推到低 QPS、远程客户端、其他模型或训练。未修改自定义算子、TP runtime 或 HCCL 内核。
