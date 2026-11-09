#!/usr/bin/env python3
"""汇总 bench_ab_quant.sh 的结果：每 cell 3 次中位数、A/B 配对差、冻结判定规则。

用法: python3 summarize_ab.py /root/perf_ab_qwen38_YYYYmmdd_HHMM
产出: summary_median.csv / comparison.md / 标准输出
规则（跑之前冻结，别事后调）:
  - 有效运行: failed=0 且 json 存在（脚本侧已卡 rc 与对端空闲）
  - 有效差异: 3 次重复中位数差异 >= 5%，且 3/3 次同向
  - 负面保护: guardrail 指标回退 > 3% -> 不推荐
"""
import glob
import json
import os
import re
import sys

import pandas as pd

EFFECT_MIN = 5.0     # % 主指标最小有效效应量
GUARDRAIL_MAX = 3.0  # % guardrail 最大允许回退

METRICS = [
    "output_throughput", "request_throughput", "total_token_throughput",
    "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "p99_e2el_ms",
    "spec_decode_acceptance_rate", "queue_mean_ms", "kv_peak_pct", "waiting_peak",
]
PRIMARY = "output_throughput"
GUARDRAILS = ["p99_ttft_ms", "mean_tpot_ms"]
LOWER_IS_BETTER = {"p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "mean_tpot_ms",
                   "p99_tpot_ms", "mean_itl_ms", "p99_e2el_ms", "queue_mean_ms"}
PAT = re.compile(r"^(bf16|w8a8)_in(\d+)_out(\d+)_c(\d+)_r(\d+)\.json$")


def prom_metric(text, name):
    """取 `<name>{...} value` 的数值；后出现者覆盖（多实例场景够用）。"""
    val = None
    for line in text.splitlines():
        if line.startswith(name + "{"):
            try:
                val = float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return val


def prom_peak(text, name):
    vals = []
    for line in text.splitlines():
        if line.startswith(name + "{"):
            try:
                vals.append(float(line.rsplit(" ", 1)[1]))
            except (ValueError, IndexError):
                pass
    return max(vals) if vals else float("nan")


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def queue_mean_ms(scrape_dir, cid):
    before = read(os.path.join(scrape_dir, f"{cid}_before.prom"))
    after = read(os.path.join(scrape_dir, f"{cid}_after.prom"))
    base = "vllm:request_queue_time_seconds"
    bs, as_ = prom_metric(before, base + "_sum"), prom_metric(after, base + "_sum")
    bc, ac = prom_metric(before, base + "_count"), prom_metric(after, base + "_count")
    if None in (bs, as_, bc, ac) or (ac - bc) <= 0:
        return float("nan")
    return (as_ - bs) / (ac - bc) * 1000.0


def collect(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "results_raw", "*.json"))):
        m = PAT.match(os.path.basename(path))
        if not m:
            continue
        quant, inlen, outlen, conc, rep = m.group(1), *map(int, m.groups()[1:])
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("failed", 1) != 0:
            continue
        cid = os.path.basename(path)[:-5]
        samples = read(os.path.join(out_dir, "metrics", f"{cid}_samples.prom"))
        row = {
            "quant": quant, "in": inlen, "out": outlen, "c": conc, "rep": rep,
            "queue_mean_ms": queue_mean_ms(os.path.join(out_dir, "metrics"), cid),
            "kv_peak_pct": prom_peak(samples, "vllm:kv_cache_usage_perc"),
            "waiting_peak": prom_peak(samples, "vllm:num_requests_waiting"),
        }
        for k in METRICS:
            row.setdefault(k, d.get(k))
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(v):
    return "—" if pd.isna(v) else f"{v:.4g}"


def main(out_dir):
    df = collect(out_dir)
    if df.empty:
        print(f"no valid results under {out_dir}/results_raw")
        return 1

    keys = ["in", "out", "c"]
    num = [k for k in METRICS if k in df.columns]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    med = df.groupby(["quant"] + keys, dropna=False)[num].median().reset_index()
    med.to_csv(os.path.join(out_dir, "summary_median.csv"), index=False)

    lines = ["# W8A8 vs BF16 — 量化 A/B 性能对比", "",
             f"- 逐次结果: `{os.path.join(out_dir, 'results_raw')}`",
             f"- 有效运行: {len(df)} 次",
             f"- 规则: 效应量 >= {EFFECT_MIN}% 且 3/3 同向；guardrail 回退 <= {GUARDRAIL_MAX}%",
             "", "## 中位数汇总（每 cell 3 次取中位数）", ""]
    cols = ["quant"] + keys + ["output_throughput", "p50_ttft_ms", "p99_ttft_ms",
                             "mean_tpot_ms", "p99_e2el_ms", "queue_mean_ms", "kv_peak_pct"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in med.sort_values(["out", "in", "c", "quant"]).iterrows():
        lines.append("| " + " | ".join(str(r[c]) if c in keys + ["quant"] else fmt(r[c])
                                       for c in cols) + " |")

    # 配对比较：按 (in,out,c) 与 rep 对齐
    lines += ["", "## A/B 差值（W8A8 相对 BF16）", "",
              "| in | out | c | 指标 | BF16 | W8A8 | 差值% | 同向 | 判定 |",
              "|---|---|---|---|---:|---:|---:|---|---|"]
    for (inl, outl, conc), g in df.groupby(keys):
        b = g[g.quant == "bf16"].sort_values("rep")
        w = g[g.quant == "w8a8"].sort_values("rep")
        if b.empty or w.empty:
            continue
        for k in ["output_throughput"] + GUARDRAILS:
            if k not in df.columns:
                continue
            bv, wv = b[k].median(), w[k].median()
            if pd.isna(bv) or pd.isna(wv) or bv == 0:
                continue
            diff = (wv - bv) / bv * 100.0
            pairs = list(zip(b[k].tolist(), w[k].tolist()))
            same = sum(1 for x, y in pairs if (y - x) * (diff) > 0) if diff else 0
            lower_better = k in LOWER_IS_BETTER
            gain = (-diff if lower_better else diff)
            if abs(diff) < EFFECT_MIN or same < min(3, len(pairs)):
                verdict = "无差异"
            elif k == PRIMARY and gain > 0:
                verdict = "W8A8 更优"
            elif k == PRIMARY:
                verdict = "BF16 更优"
            elif gain < -GUARDRAIL_MAX:
                verdict = "guardrail 回退"
            else:
                verdict = "guardrail 通过"
            lines.append(f"| {inl} | {outl} | {conc} | {k} | {bv:.4g} | {wv:.4g} | "
                         f"{diff:+.1f}% | {same}/{len(pairs)} | {verdict} |")

    lines += ["", "## 结论边界", "",
              "- 仅限上表 in×out×c 组合、TP=4、bf16/W8A8 各占 4 卡的突发负载；不外推到低 QPS 或 262144 长上下文。",
              "- 正确性问题（贪心输出是否逐 token 一致）不在本脚本内，需单独用固定 prompt 比对。"]
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"\nwrote {out_dir}/summary_median.csv and comparison.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
