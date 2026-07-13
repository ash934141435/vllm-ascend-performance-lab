#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summary_median.csv"
FIGURES = SUMMARY.parent / "figures"
COLORS = {16: "#4C78A8", 24: "#F58518", 32: "#54A24B"}


def load_rows():
    with SUMMARY.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 90:
        raise ValueError(f"expected 90 summary rows, found {len(rows)}")
    for row in rows:
        for key, value in row.items():
            row[key] = float(value)
    return rows


def select(rows, **filters):
    return sorted(
        (row for row in rows if all(row[key] == value for key, value in filters.items())),
        key=lambda row: row["concurrency"],
    )


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_by_seq(rows, metric, ylabel, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, input_len in zip(axes, (1024, 4096)):
        for seq in (16, 24, 32):
            data = select(
                rows,
                input_len=input_len,
                max_num_seqs=seq,
                max_num_batched_tokens=12288,
            )
            ax.plot(
                [row["concurrency"] for row in data],
                [row[metric] for row in data],
                marker="o",
                color=COLORS[seq],
                label=f"max-num-seqs={seq}",
            )
        ax.set_title(f"输入 {input_len} tokens")
        ax.set_xlabel("并发")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False)
    fig.suptitle(f"{title}（max-num-batched-tokens=12288）")
    finish(fig, filename)


def plot_latency(rows, metrics, labels, ylabel, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, input_len in zip(axes, (1024, 4096)):
        data = select(
            rows,
            input_len=input_len,
            max_num_seqs=32,
            max_num_batched_tokens=12288,
        )
        for metric, label in zip(metrics, labels):
            ax.plot(
                [row["concurrency"] for row in data],
                [row[metric] / 1000 for row in data],
                marker="o",
                label=label,
            )
        ax.set_title(f"输入 {input_len} tokens")
        ax.set_xlabel("并发")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False)
    fig.suptitle(f"{title}（max-num-seqs=32，max-num-batched-tokens=12288）")
    finish(fig, filename)


def plot_queue(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.23
    for ax, input_len in zip(axes, (1024, 4096)):
        data = [
            select(
                rows,
                input_len=input_len,
                concurrency=32,
                max_num_seqs=seq,
                max_num_batched_tokens=12288,
            )[0]
            for seq in (16, 24, 32)
        ]
        x = range(3)
        ax.bar(
            [value - width / 2 for value in x],
            [row["queue_mean_ms"] / 1000 for row in data],
            width,
            label="Queue mean (s)",
            color="#4C78A8",
        )
        right = ax.twinx()
        right.bar(
            [value + width / 2 for value in x],
            [row["waiting_peak"] for row in data],
            width,
            label="Waiting peak",
            color="#F58518",
        )
        ax.set_xticks(list(x), (16, 24, 32))
        ax.set_xlabel("max-num-seqs")
        ax.set_ylabel("平均排队时间（秒）")
        right.set_ylabel("等待请求峰值")
        ax.set_title(f"输入 {input_len} tokens")
        ax.grid(axis="y", alpha=0.25)
        if input_len == 4096:
            handles = [ax.patches[0], right.patches[0]]
            fig.legend(
                handles,
                ["Queue mean", "Waiting peak"],
                loc="lower center",
                ncol=2,
                frameon=False,
            )
    fig.suptitle("32 并发下的排队行为（max-num-batched-tokens=12288）")
    finish(fig, "queue_and_waiting.png")


def pareto_frontier(rows):
    ordered = sorted(rows, key=lambda row: row["p99_ttft_ms"])
    frontier, best = [], -1.0
    for row in ordered:
        if row["output_throughput_tok_s"] > best:
            frontier.append(row)
            best = row["output_throughput_tok_s"]
    return frontier


def plot_pareto(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, input_len in zip(axes, (1024, 4096)):
        data = [row for row in rows if row["input_len"] == input_len]
        frontier = pareto_frontier(data)
        ax.scatter(
            [row["p99_ttft_ms"] / 1000 for row in data],
            [row["output_throughput_tok_s"] for row in data],
            color="#BAB0AC",
            alpha=0.55,
            label="全部配置",
        )
        ax.plot(
            [row["p99_ttft_ms"] / 1000 for row in frontier],
            [row["output_throughput_tok_s"] for row in frontier],
            marker="o",
            color="#E45756",
            label="Pareto frontier",
        )
        ax.set_title(f"输入 {input_len} tokens")
        ax.set_xlabel("P99 TTFT（秒，越低越好）")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("输出吞吐（tok/s，越高越好）")
    axes[1].legend(frameon=False)
    fig.suptitle("吞吐与尾延迟的权衡")
    finish(fig, "throughput_p99_pareto.png")


def main():
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
        }
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    plot_by_seq(
        rows,
        "output_throughput_tok_s",
        "输出吞吐（tok/s）",
        "吞吐随并发变化",
        "throughput_by_concurrency.png",
    )
    plot_latency(
        rows,
        ("p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms"),
        ("P50", "P95", "P99"),
        "TTFT（秒）",
        "首 Token 延迟随并发变化",
        "ttft_by_concurrency.png",
    )
    plot_latency(
        rows,
        ("mean_tpot_ms", "p99_tpot_ms"),
        ("Mean", "P99"),
        "TPOT（秒/token）",
        "单 Token 延迟随并发变化",
        "tpot_by_concurrency.png",
    )
    plot_queue(rows)
    plot_pareto(rows)
    print(f"generated 5 figures in {FIGURES}")


if __name__ == "__main__":
    main()
