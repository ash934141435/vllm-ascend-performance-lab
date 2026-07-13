#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import gzip
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

EXIT_VALID = 0
EXIT_RETRYABLE = 10
EXIT_WARNING = 11
EXIT_HARD = 20

CASE_RE = re.compile(
    r"seq(?P<max_num_seqs>\d+)_"
    r"bt(?P<max_num_batched_tokens>\d+)_"
    r"in(?P<input_len>\d+)_"
    r"out(?P<output_len>\d+)_"
    r"c(?P<concurrency>\d+)_"
    r"r(?P<repeat_id>\d+)"
)

METRIC_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)

VALID_SEQS = {16, 24, 32}
VALID_BT = {8192, 12288, 16384}
VALID_INPUT_LENS = (1024, 4096)
VALID_CONCURRENCIES = (1, 5, 10, 16, 32)
REPEATS = (1, 2, 3)
EXPECTED_GROUP_RESULTS = len(VALID_INPUT_LENS) * len(VALID_CONCURRENCIES) * len(REPEATS)
EXPECTED_VALID_RESULTS = len(VALID_SEQS) * len(VALID_BT) * EXPECTED_GROUP_RESULTS
EXPECTED_MEDIAN_ROWS = (
    len(VALID_SEQS) * len(VALID_BT) * len(VALID_INPUT_LENS) * len(VALID_CONCURRENCIES)
)
SEQ_RE = re.compile(r"(--max-num-seqs\s+)(\d+)")
BT_RE = re.compile(r"(--max-num-batched-tokens\s+)(\d+)")

RAW_FIELDS = [
    "max_num_seqs",
    "max_num_batched_tokens",
    "input_len",
    "output_len",
    "concurrency",
    "repeat_id",
    "num_prompts",
    "output_throughput_tok_s",
    "request_throughput_req_s",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p99_e2el_ms",
    "queue_mean_ms",
    "queue_p99_ms",
    "queue_p99_upper_bound_ms",
    "prefill_mean_ms",
    "decode_mean_ms",
    "kv_peak_pct",
    "waiting_peak",
    "running_peak",
    "preemptions",
    "failed_requests",
    "error_count",
    "case_id",
    "attempt_id",
    "attempt_status",
    "attempt_number",
    "anomaly_reasons",
    "warning_reasons",
    "result_file",
    "metrics_file",
    "serve_log_file",
]

GROUP_KEYS = [
    "max_num_seqs",
    "max_num_batched_tokens",
    "input_len",
    "output_len",
    "concurrency",
]

DISPERSION_METRICS = (
    "output_throughput_tok_s",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "mean_e2el_ms",
)

OOM_RE = re.compile(
    r"out of memory|\boom\b|oomkilled|killed process",
    re.IGNORECASE,
)


@dataclass
class Snapshot:
    scalars: dict[str, float] = field(default_factory=dict)
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)
    buckets: dict[str, dict[float, float]] = field(default_factory=dict)


@dataclass
class HistogramDelta:
    count: int
    total: float
    mean_ms: Optional[float]
    p99_upper_bound_ms: Optional[float]


@dataclass
class Sample:
    timestamp_ms: int
    running: float
    waiting: float
    kv_usage: float
    generation_tokens: float


@dataclass
class AttemptContext:
    case_id: str
    result: dict
    before: Snapshot
    after: Snapshot
    samples: list[Sample]
    bench_rc: int
    restart_count_before: int
    restart_count_after: int
    started_at_before: str
    started_at_after: str
    oom_killed: bool
    serve_log: str


def parse_case_id(case_id: str) -> dict[str, int]:
    match = CASE_RE.fullmatch(case_id)
    if not match:
        raise ValueError(f"invalid case id: {case_id}")
    return {key: int(value) for key, value in match.groupdict().items()}


def parse_labels(raw: Optional[str]) -> dict[str, str]:
    if not raw:
        return {}
    labels: dict[str, str] = {}
    for part in raw.split(","):
        key, value = part.split("=", 1)
        labels[key] = value.strip('"')
    return labels


def metric_key(name: str, labels: dict[str, str]) -> str:
    ignored = {"engine", "model_name", "finished_reason"}
    kept = sorted((key, value) for key, value in labels.items() if key not in ignored)
    suffix = ",".join(f"{key}={value}" for key, value in kept)
    return f"{name}|{suffix}" if suffix else name


def parse_prometheus(text: str) -> Snapshot:
    snapshot = Snapshot()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        labels = parse_labels(match.group("labels"))
        value = float(match.group("value"))
        if name.endswith("_bucket"):
            base = name[:-7]
            bound_raw = labels.get("le")
            if bound_raw is None:
                continue
            bound = math.inf if bound_raw == "+Inf" else float(bound_raw)
            snapshot.buckets.setdefault(base, {})
            snapshot.buckets[base][bound] = snapshot.buckets[base].get(bound, 0.0) + value
        elif name.endswith("_sum"):
            base = name[:-4]
            snapshot.sums[base] = snapshot.sums.get(base, 0.0) + value
        elif name.endswith("_count"):
            base = name[:-6]
            snapshot.counts[base] = snapshot.counts.get(base, 0.0) + value
        else:
            key = metric_key(name, labels)
            snapshot.scalars[key] = snapshot.scalars.get(key, 0.0) + value
    return snapshot


def histogram_delta(before: Snapshot, after: Snapshot, metric: str) -> HistogramDelta:
    count = int(after.counts.get(metric, 0.0) - before.counts.get(metric, 0.0))
    total = after.sums.get(metric, 0.0) - before.sums.get(metric, 0.0)
    mean_ms = total / count * 1000.0 if count > 0 else None
    if count <= 0:
        return HistogramDelta(count, total, mean_ms, None)
    target = count * 0.99
    p99 = None
    before_buckets = before.buckets.get(metric, {})
    after_buckets = after.buckets.get(metric, {})
    for bound in sorted(set(before_buckets) | set(after_buckets)):
        cumulative = after_buckets.get(bound, 0.0) - before_buckets.get(bound, 0.0)
        if cumulative >= target:
            p99 = None if math.isinf(bound) else bound * 1000.0
            break
    return HistogramDelta(count, total, mean_ms, p99)


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as handle:
            return handle.read()
    return path.read_text(errors="replace")


def scalar(snapshot: Snapshot, name: str) -> float:
    return sum(
        value
        for key, value in snapshot.scalars.items()
        if key == name or key.startswith(f"{name}|")
    )


def parse_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    chunks = read_text(path).split("# sample_time_unix_ms ")[1:]
    for chunk in chunks:
        first, *rest = chunk.splitlines()
        snapshot = parse_prometheus("\n".join(rest))
        samples.append(
            Sample(
                timestamp_ms=int(first.strip()),
                running=scalar(snapshot, "vllm:num_requests_running"),
                waiting=scalar(snapshot, "vllm:num_requests_waiting"),
                kv_usage=scalar(snapshot, "vllm:kv_cache_usage_perc"),
                generation_tokens=scalar(snapshot, "vllm:generation_tokens_total"),
            )
        )
    return samples


def replace_once(pattern: re.Pattern[str], text: str, value: int) -> str:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"{pattern.pattern} must appear exactly once")
    return pattern.sub(lambda match: f"{match.group(1)}{value}", text, count=1)


def normalize_start_script(text: str) -> str:
    text = SEQ_RE.sub(r"\1<SEQ>", text)
    return BT_RE.sub(r"\1<BT>", text)


def render_start_script(source: str, seq: int, bt: int) -> tuple[str, str]:
    if seq not in VALID_SEQS or bt not in VALID_BT:
        raise ValueError(f"unsupported matrix values: seq={seq} bt={bt}")
    rendered = replace_once(SEQ_RE, source, seq)
    rendered = replace_once(BT_RE, rendered, bt)
    if normalize_start_script(source) != normalize_start_script(rendered):
        raise ValueError("rendered script changed outside allowed parameters")
    diff_text = "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile="start-vllm.original.sh",
            tofile=f"start_seq{seq}_bt{bt}.sh",
        )
    )
    return rendered, diff_text


def counter_delta(before: Snapshot, after: Snapshot, name: str) -> float:
    return scalar(after, name) - scalar(before, name)


def sample_gap(samples: list[Sample]) -> bool:
    return any(
        right.timestamp_ms - left.timestamp_ms > 3000
        for left, right in zip(samples, samples[1:])
    )


def stalled_waiting(samples: list[Sample]) -> bool:
    if len(samples) < 10:
        return False
    for start in range(len(samples) - 9):
        window = samples[start : start + 10]
        waiting = [sample.waiting for sample in window]
        tokens = [sample.generation_tokens for sample in window]
        if (
            all(value > 0 for value in waiting)
            and waiting == sorted(waiting)
            and tokens[0] == tokens[-1]
        ):
            return True
    return False


def evaluate_attempt(context: AttemptContext) -> tuple[dict[str, object], str, list[str]]:
    case = parse_case_id(context.case_id)
    result = context.result
    expected_prompts = case["concurrency"] * 5
    expected_input_tokens = case["input_len"] * expected_prompts
    expected_output_tokens = case["output_len"] * expected_prompts

    queue = histogram_delta(context.before, context.after, "vllm:request_queue_time_seconds")
    prefill = histogram_delta(context.before, context.after, "vllm:request_prefill_time_seconds")
    decode = histogram_delta(context.before, context.after, "vllm:request_decode_time_seconds")
    request_delta = counter_delta(context.before, context.after, "vllm:request_success_total")
    prompt_delta = counter_delta(context.before, context.after, "vllm:prompt_tokens_total")
    generation_delta = counter_delta(context.before, context.after, "vllm:generation_tokens_total")
    preemptions = counter_delta(context.before, context.after, "vllm:num_preemptions_total")

    hard: list[str] = []
    retryable: list[str] = []
    warnings: list[str] = []

    if context.bench_rc != 0:
        hard.append("benchmark_exit")
    if result.get("failed", 0) > 0:
        hard.append("failed_requests")
    if any(result.get("errors", [])):
        hard.append("request_errors")
    if result.get("completed") != expected_prompts:
        hard.append("completed_count")
    if result.get("input_lens") != [case["input_len"]] * expected_prompts:
        hard.append("input_length")
    if result.get("output_lens") != [case["output_len"]] * expected_prompts:
        hard.append("output_length")
    if context.restart_count_after != context.restart_count_before:
        hard.append("container_restart")
    if context.started_at_after != context.started_at_before:
        hard.append("container_started_at")
    if context.oom_killed or OOM_RE.search(context.serve_log):
        hard.append("oom")
    if preemptions > 0:
        hard.append("preemption")
    if context.samples and max(sample.kv_usage for sample in context.samples) > 0.85:
        hard.append("kv_over_85_pct")
    if stalled_waiting(context.samples):
        hard.append("waiting_stall")
    if any(value < 0 for value in (request_delta, prompt_delta, generation_delta)):
        hard.append("counter_reset")

    if request_delta != expected_prompts:
        retryable.append("request_counter_pollution")
    if prompt_delta != expected_input_tokens:
        retryable.append("prompt_token_pollution")
    if generation_delta != expected_output_tokens:
        retryable.append("generation_token_pollution")
    if not context.samples or sample_gap(context.samples):
        retryable.append("metrics_sampling_gap")

    ttfts = result.get("ttfts", [])
    if len(ttfts) >= 3:
        later_median = statistics.median(ttfts[1:])
        if later_median > 0 and ttfts[0] > later_median * 5:
            retryable.append("warmup_miss")
    mean_ttft = float(result.get("mean_ttft_ms", 0.0))
    p99_ttft = float(result.get("p99_ttft_ms", 0.0))
    if mean_ttft > 0 and p99_ttft > mean_ttft * 3:
        warnings.append("tail_latency_warning")

    status = "hard" if hard else "retryable" if retryable else "valid"
    reasons = hard if hard else sorted(set(retryable))

    row: dict[str, object] = {
        **case,
        "num_prompts": expected_prompts,
        "output_throughput_tok_s": result.get("output_throughput"),
        "request_throughput_req_s": result.get("request_throughput"),
        "mean_ttft_ms": result.get("mean_ttft_ms"),
        "p50_ttft_ms": result.get("p50_ttft_ms"),
        "p90_ttft_ms": result.get("p90_ttft_ms"),
        "p95_ttft_ms": result.get("p95_ttft_ms"),
        "p99_ttft_ms": result.get("p99_ttft_ms"),
        "mean_tpot_ms": result.get("mean_tpot_ms"),
        "p99_tpot_ms": result.get("p99_tpot_ms"),
        "mean_itl_ms": result.get("mean_itl_ms"),
        "p99_itl_ms": result.get("p99_itl_ms"),
        "mean_e2el_ms": result.get("mean_e2el_ms"),
        "p99_e2el_ms": result.get("p99_e2el_ms"),
        "queue_mean_ms": queue.mean_ms if queue.mean_ms is not None else "N/A",
        "queue_p99_ms": "N/A",
        "queue_p99_upper_bound_ms": queue.p99_upper_bound_ms if queue.p99_upper_bound_ms is not None else "N/A",
        "prefill_mean_ms": prefill.mean_ms if prefill.mean_ms is not None else "N/A",
        "decode_mean_ms": decode.mean_ms if decode.mean_ms is not None else "N/A",
        "kv_peak_pct": max((sample.kv_usage for sample in context.samples), default=0.0) * 100.0,
        "waiting_peak": max((sample.waiting for sample in context.samples), default=0.0),
        "running_peak": max((sample.running for sample in context.samples), default=0.0),
        "preemptions": preemptions,
        "failed_requests": result.get("failed", 0),
        "error_count": sum(bool(error) for error in result.get("errors", [])),
        "attempt_status": status,
        "anomaly_reasons": reasons,
        "warning_reasons": sorted(set(warnings)),
    }
    return row, status, reasons


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def inspect_value(data: dict, path: Iterable[str], default=None):
    if isinstance(data, list):
        if not data:
            return default
        data = data[0]
    value = data
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def comparable_rows_for_tail(run_dir: Path, seq: int, bt: int, row: dict[str, object]) -> list[dict]:
    target = (
        int(row["input_len"]),
        int(row["output_len"]),
        int(row["concurrency"]),
    )
    rows = []
    for path in sorted((run_dir / "summary" / "attempts").glob("*.json")):
        candidate = load_json(path)
        if candidate.get("attempt_status") != "valid":
            continue
        if int(candidate["max_num_seqs"]) != seq or int(candidate["max_num_batched_tokens"]) != bt:
            continue
        key = (
            int(candidate["input_len"]),
            int(candidate["output_len"]),
            int(candidate["concurrency"]),
        )
        if key == target:
            rows.append(candidate)
    return rows


def degraded_vs_baseline(
    current: object, baseline_values: list[float], ratio_threshold: float = 1.3
) -> bool:
    if not isinstance(current, (int, float)) or not baseline_values:
        return False
    baseline = statistics.median(baseline_values)
    return baseline > 0 and float(current) >= baseline * ratio_threshold


def tail_latency_is_systemic(row: dict[str, object], comparable_rows: list[dict]) -> bool:
    warnings = row.get("warning_reasons") or []
    mean_ttft = numeric_or(row.get("mean_ttft_ms"))
    p95_ttft = numeric_or(row.get("p95_ttft_ms"))
    p99_ttft = numeric_or(row.get("p99_ttft_ms"))
    if "tail_latency_warning" not in warnings:
        return False
    if mean_ttft <= 0 or p99_ttft <= mean_ttft * 3 or p95_ttft <= mean_ttft * 2:
        return False
    if not comparable_rows:
        return False

    def baselines(field: str) -> list[float]:
        return [
            float(item[field])
            for item in comparable_rows
            if isinstance(item.get(field), (int, float))
        ]

    if not degraded_vs_baseline(row.get("mean_ttft_ms"), baselines("mean_ttft_ms")):
        return False
    if not degraded_vs_baseline(row.get("queue_mean_ms"), baselines("queue_mean_ms")) and not degraded_vs_baseline(
        row.get("queue_p99_upper_bound_ms"), baselines("queue_p99_upper_bound_ms")
    ):
        return False
    if not degraded_vs_baseline(row.get("prefill_mean_ms"), baselines("prefill_mean_ms")):
        return False

    ttft_band_hits = 0
    for field in ("p90_ttft_ms", "p95_ttft_ms", "p99_ttft_ms"):
        if degraded_vs_baseline(row.get(field), baselines(field), 1.15):
            ttft_band_hits += 1
    return ttft_band_hits >= 2


def ratio(values: list[float]) -> float:
    minimum = min(values)
    if minimum <= 0:
        return math.inf
    return max(values) / minimum


def group_anomalies(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    valid = [row for row in rows if row.get("attempt_status") == "valid"]
    anomalies: list[dict[str, object]] = []
    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for row in valid:
        key = (int(row["input_len"]), int(row["output_len"]), int(row["concurrency"]))
        grouped.setdefault(key, []).append(row)

    for key, repeats in grouped.items():
        if len(repeats) != 3:
            continue
        for metric in DISPERSION_METRICS:
            values = [float(row[metric]) for row in repeats]
            value_ratio = ratio(values)
            if value_ratio > 1.3:
                anomalies.append(
                    {
                        "rule": "repeat_dispersion",
                        "key": key,
                        "metric": metric,
                        "ratio": value_ratio,
                        "values": values,
                    }
                )

    input_lengths = sorted({int(row["input_len"]) for row in valid})
    for input_len in input_lengths:
        by_concurrency: dict[int, list[dict[str, object]]] = {}
        for row in valid:
            if int(row["input_len"]) == input_len:
                by_concurrency.setdefault(int(row["concurrency"]), []).append(row)
        if len(by_concurrency.get(1, [])) == 3 and len(by_concurrency.get(10, [])) == 3:
            c1_ttft = statistics.median(float(row["mean_ttft_ms"]) for row in by_concurrency[1])
            c10_ttft = statistics.median(float(row["mean_ttft_ms"]) for row in by_concurrency[10])
            if c1_ttft > c10_ttft * 1.5:
                anomalies.append(
                    {
                        "rule": "c1_ttft_vs_c10",
                        "input_len": input_len,
                        "c1": c1_ttft,
                        "c10": c10_ttft,
                    }
                )
            c1_throughput = statistics.median(
                float(row["output_throughput_tok_s"]) for row in by_concurrency[1]
            )
            c10_throughput = statistics.median(
                float(row["output_throughput_tok_s"]) for row in by_concurrency[10]
            )
            if c1_throughput > c10_throughput * 1.2:
                anomalies.append(
                    {
                        "rule": "c1_throughput_vs_c10",
                        "input_len": input_len,
                        "c1": c1_throughput,
                        "c10": c10_throughput,
                    }
                )
    return anomalies


def csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict], fields: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or RAW_FIELDS
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fields})


def median_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("attempt_status") != "valid":
            continue
        key = tuple(row[field] for field in GROUP_KEYS)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, repeats in sorted(grouped.items()):
        summary = dict(zip(GROUP_KEYS, key))
        for field_name in RAW_FIELDS:
            values = [row.get(field_name) for row in repeats if isinstance(row.get(field_name), (int, float))]
            if values and field_name not in GROUP_KEYS + ["repeat_id"]:
                summary[field_name] = statistics.median(values)
        p99_values = [
            float(row["p99_ttft_ms"])
            for row in repeats
            if isinstance(row.get("p99_ttft_ms"), (int, float))
        ]
        if p99_values:
            mean_p99 = statistics.mean(p99_values)
            summary["p99_ttft_cv"] = statistics.pstdev(p99_values) / mean_p99 if mean_p99 > 0 else math.inf
        summary["repeat_count"] = len(repeats)
        output.append(summary)
    return output


def normalized_score(value: float, best: float, higher_is_better: bool) -> float:
    if value <= 0 or best <= 0:
        return 0.0
    return value / best if higher_is_better else best / value


def numeric_or(value, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def recommendation_table(summary: list[dict]) -> list[dict]:
    score_fields = (
        "output_throughput_tok_s",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
    )
    eligible = [
        row
        for row in summary
        if row.get("repeat_count") == 3
        and numeric_or(row.get("preemptions")) == 0
        and numeric_or(row.get("kv_peak_pct")) <= 85
        and numeric_or(row.get("failed_requests")) == 0
        and all(isinstance(row.get(field), (int, float)) for field in score_fields)
    ]
    cells: dict[tuple[int, int], list[dict]] = {}
    for row in eligible:
        cells.setdefault((int(row["input_len"]), int(row["concurrency"])), []).append(row)

    scores: dict[tuple[int, int], list[float]] = {}
    for cell_rows in cells.values():
        best_throughput = max(float(row["output_throughput_tok_s"]) for row in cell_rows)
        best_ttft = min(float(row["mean_ttft_ms"]) for row in cell_rows)
        best_p99 = min(float(row["p99_ttft_ms"]) for row in cell_rows)
        best_tpot = min(float(row["mean_tpot_ms"]) for row in cell_rows)
        for row in cell_rows:
            key = (int(row["max_num_seqs"]), int(row["max_num_batched_tokens"]))
            score = statistics.mean(
                [
                    normalized_score(float(row["output_throughput_tok_s"]), best_throughput, True),
                    normalized_score(float(row["mean_ttft_ms"]), best_ttft, False),
                    normalized_score(float(row["p99_ttft_ms"]), best_p99, False),
                    normalized_score(float(row["mean_tpot_ms"]), best_tpot, False),
                ]
            )
            scores.setdefault(key, []).append(score)

    output = []
    for (seq, bt), values in scores.items():
        output.append(
            {
                "max_num_seqs": seq,
                "max_num_batched_tokens": bt,
                "composite_score": statistics.mean(values),
                "covered_cells": len(values),
            }
        )
    return sorted(output, key=lambda row: row["composite_score"], reverse=True)


def write_xlsx(
    path: Path,
    all_attempts: list[dict],
    valid: list[dict],
    excluded: list[dict],
    median: list[dict],
    recommendations: list[dict],
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    workbook = Workbook()
    default = workbook.active
    workbook.remove(default)
    for title, rows in (
        ("All Attempts", all_attempts),
        ("Valid Results", valid),
        ("Excluded Attempts", excluded),
        ("Median Summary", median),
        ("Recommendations", recommendations),
    ):
        sheet = workbook.create_sheet(title)
        fields = list(rows[0]) if rows else ["status"]
        sheet.append(fields)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for row in rows:
            sheet.append([csv_value(row.get(field, "")) for field in fields])
        sheet.freeze_panes = "A2"
        for column in sheet.columns:
            width = min(60, max(len(str(cell.value or "")) for cell in column) + 2)
            sheet.column_dimensions[column[0].column_letter].width = width
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def choose_best(summary: list[dict], field: str, reverse: bool) -> Optional[dict]:
    candidates = [row for row in summary if row.get("repeat_count") == 3]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: float(row[field]), reverse=reverse)[0]


def choose_stable_p99(summary: list[dict]) -> Optional[dict]:
    candidates = [row for row in summary if row.get("repeat_count") == 3]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            float(row.get("p99_ttft_cv", math.inf)),
            float(row["p99_ttft_ms"]),
        ),
    )


def write_report(path: Path, all_attempts: list[dict], median: list[dict], recommendations: list[dict]) -> None:
    excluded = [row for row in all_attempts if row.get("attempt_status") != "valid"]
    warnings = [row for row in all_attempts if row.get("warning_reasons")]
    throughput = choose_best(median, "output_throughput_tok_s", True)
    ttft = choose_best(median, "mean_ttft_ms", False)
    p99 = choose_stable_p99(median)
    by_input = {}
    for input_len in (1024, 4096):
        subset = [row for row in median if int(row["input_len"]) == input_len]
        by_input[input_len] = recommendation_table(subset)
    lines = [
        "# Qwen3.6 27B vLLM-Ascend Benchmark Report",
        "",
        f"- All attempts: {len(all_attempts)}",
        f"- Warning attempts: {len(warnings)}",
        f"- Excluded/anomalous attempts: {len(excluded)}",
        f"- Median combinations: {len(median)}",
        "",
        "## Ranking method",
        "",
        "For each input/concurrency cell, throughput, mean TTFT, P99 TTFT, and mean TPOT are normalized against the best eligible value and combined with equal 25% weight. Scores are averaged across cells.",
        "",
        "## Best measured points",
        "",
        f"- Throughput best: `{json.dumps(throughput, ensure_ascii=False)}`",
        f"- TTFT best: `{json.dumps(ttft, ensure_ascii=False)}`",
        f"- P99 TTFT most stable: `{json.dumps(p99, ensure_ascii=False)}`",
        "",
        "## Recommendations",
        "",
        f"- 1K/1K: `{json.dumps(by_input[1024][:3], ensure_ascii=False)}`",
        f"- 4K/1K: `{json.dumps(by_input[4096][:3], ensure_ascii=False)}`",
        f"- Overall: `{json.dumps(recommendations[:3], ensure_ascii=False)}`",
        "",
        "## Queue, KV, preemption, prefill, and decode checks",
        "",
    ]
    for label, field_name in (
        ("Highest queue mean", "queue_mean_ms"),
        ("Highest waiting peak", "waiting_peak"),
        ("Highest KV peak", "kv_peak_pct"),
        ("Highest prefill mean", "prefill_mean_ms"),
        ("Highest decode mean", "decode_mean_ms"),
    ):
        ranked = sorted(median, key=lambda row: numeric_or(row.get(field_name)), reverse=True)
        lines.append(f"- {label}: `{json.dumps(ranked[:5], ensure_ascii=False)}`")
    preempted = [row for row in median if numeric_or(row.get("preemptions")) > 0]
    lines.append(f"- Preemptions: `{json.dumps(preempted or 'none', ensure_ascii=False)}`")
    lines.extend(["", "## Not recommended", ""])
    bottom = list(reversed(recommendations[-3:]))
    if bottom:
        lines.extend(
            "- Low composite score: " f"`{json.dumps(row, ensure_ascii=False)}`"
            for row in bottom
        )
    else:
        lines.append("- N/A")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for row in warnings:
            lines.append(
                "- "
                f"{row.get('case_id', 'unknown')} "
                f"{row.get('attempt_id', 'unknown')} "
                f"{json.dumps(row.get('warning_reasons', []), ensure_ascii=False)} "
                f"`{row.get('result_file', '')}` "
                f"`{row.get('metrics_file', '')}` "
                f"`{row.get('serve_log_file', '')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Excluded / Anomalous Attempts", ""])
    if excluded:
        for row in excluded:
            lines.append(
                "- "
                f"{row.get('case_id', 'unknown')} "
                f"{row.get('attempt_id', 'unknown')} "
                f"{row.get('attempt_status', 'unknown')} "
                f"{json.dumps(row.get('anomaly_reasons', []), ensure_ascii=False)} "
                f"`{row.get('result_file', '')}` "
                f"`{row.get('metrics_file', '')}` "
                f"`{row.get('serve_log_file', '')}`"
            )
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def load_attempt_rows(run_dir: Path, seq: int, bt: int) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "summary" / "attempts").glob("*.json")):
        row = load_json(path)
        if int(row["max_num_seqs"]) == seq and int(row["max_num_batched_tokens"]) == bt:
            rows.append(row)
    return rows


def canonical_rows(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "summary" / "attempts").glob("*.json")):
        row = load_json(path)
        if row.get("attempt_status") == "valid":
            rows.append(row)
    unique = {}
    for row in rows:
        case_id = (
            f"seq{row['max_num_seqs']}_bt{row['max_num_batched_tokens']}_"
            f"in{row['input_len']}_out{row['output_len']}_"
            f"c{row['concurrency']}_r{row['repeat_id']}"
        )
        current = unique.get(case_id)
        if current is None or int(row.get("attempt_number", 0)) > int(current.get("attempt_number", 0)):
            unique[case_id] = row
    return [unique[key] for key in sorted(unique)]


def all_attempt_rows(run_dir: Path) -> list[dict]:
    return [
        load_json(path)
        for path in sorted((run_dir / "summary" / "attempts").glob("*.json"))
    ]


def command_render_start(args: argparse.Namespace) -> int:
    source_path = Path(args.source)
    dest_path = Path(args.dest)
    diff_path = Path(args.diff_out)
    rendered, diff_text = render_start_script(source_path.read_text(), args.seq, args.bt)
    dest_path.write_text(rendered)
    dest_path.chmod(0o755)
    diff_path.write_text(diff_text)
    return EXIT_VALID


def command_attempt(args: argparse.Namespace) -> int:
    result_path = Path(args.result_file)
    before_path = Path(args.metrics_before)
    after_path = Path(args.metrics_after)
    samples_path = Path(args.metrics_samples)
    before_inspect = load_json(Path(args.inspect_before))
    after_inspect = load_json(Path(args.inspect_after))
    context = AttemptContext(
        case_id=args.case_id,
        result=load_json(result_path),
        before=parse_prometheus(read_text(before_path)),
        after=parse_prometheus(read_text(after_path)),
        samples=parse_samples(samples_path),
        bench_rc=args.bench_rc,
        restart_count_before=int(inspect_value(before_inspect, ["RestartCount"], 0)),
        restart_count_after=int(inspect_value(after_inspect, ["RestartCount"], 0)),
        started_at_before=str(inspect_value(before_inspect, ["State", "StartedAt"], "")),
        started_at_after=str(inspect_value(after_inspect, ["State", "StartedAt"], "")),
        oom_killed=bool(inspect_value(after_inspect, ["State", "OOMKilled"], False)),
        serve_log=read_text(Path(args.case_serve_log)),
    )
    row, status, reasons = evaluate_attempt(context)
    if getattr(args, "run_dir", None) and getattr(args, "seq", None) and getattr(args, "bt", None):
        comparable_rows = comparable_rows_for_tail(Path(args.run_dir), int(args.seq), int(args.bt), row)
        if tail_latency_is_systemic(row, comparable_rows):
            status = "hard"
            reasons = sorted(set([*reasons, "tail_latency_systemic_degradation"]))
            row["attempt_status"] = status
            row["anomaly_reasons"] = reasons
    row.update(
        {
            "case_id": args.case_id,
            "attempt_id": args.attempt_id,
            "attempt_number": args.attempt_number,
            "result_file": str(result_path),
            "metrics_file": str(samples_path),
            "serve_log_file": args.serve_log_file,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "reasons": reasons,
                "warning_reasons": row.get("warning_reasons", []),
            },
            ensure_ascii=False,
        )
    )
    if status == "retryable":
        return EXIT_RETRYABLE
    if status == "hard":
        return EXIT_HARD
    if row.get("warning_reasons"):
        return EXIT_WARNING
    return EXIT_VALID


def command_group_review(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    rows = load_attempt_rows(run_dir, args.seq, args.bt)
    excluded = [row for row in rows if row.get("attempt_status") != "valid"]
    warnings = [row for row in rows if row.get("warning_reasons")]
    valid_by_case = {}
    for row in rows:
        if row.get("attempt_status") != "valid":
            continue
        key = (
            int(row["input_len"]),
            int(row["output_len"]),
            int(row["concurrency"]),
            int(row["repeat_id"]),
        )
        current = valid_by_case.get(key)
        if current is None or int(row.get("attempt_number", 0)) > int(current.get("attempt_number", 0)):
            valid_by_case[key] = row
    valid = [valid_by_case[key] for key in sorted(valid_by_case)]
    anomalies = group_anomalies(valid)
    payload = {
        "max_num_seqs": args.seq,
        "max_num_batched_tokens": args.bt,
        "valid_result_count": len(valid),
        "expected_result_count": EXPECTED_GROUP_RESULTS,
        "complete": len(valid) == EXPECTED_GROUP_RESULTS,
        "anomalies": anomalies,
        "warning_attempt_count": len(warnings),
        "warning_attempts": warnings,
        "valid_results": valid,
        "excluded_attempts": excluded,
        "attempts": rows,
    }
    summary_dir = run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    stem = summary_dir / f"group_seq{args.seq}_bt{args.bt}"
    stem.with_name(stem.name + "_review.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    with stem.with_name(stem.name + "_review.md").open("w") as handle:
        handle.write(
            f"# Group seq={args.seq}, bt={args.bt}\n\n"
            f"- All attempts: {len(rows)}\n"
            f"- Valid results: {len(valid)}/{EXPECTED_GROUP_RESULTS}\n"
            f"- Warning attempts: {len(warnings)}\n"
            f"- Excluded/anomalous attempts: {len(excluded)}\n"
            f"- Complete: {payload['complete']}\n"
            f"- Anomalies: {len(anomalies)}\n\n"
        )
        for anomaly in anomalies:
            handle.write(f"- `{anomaly['rule']}`: `{json.dumps(anomaly)}`\n")
        handle.write("\n## Warnings\n\n")
        if warnings:
            for row in warnings:
                handle.write(
                    f"- {row.get('case_id', 'unknown')} "
                    f"{row.get('attempt_id', 'unknown')} "
                    f"{json.dumps(row.get('warning_reasons', []), ensure_ascii=False)} "
                    f"`{row.get('result_file', '')}`; "
                    f"`{row.get('metrics_file', '')}`; "
                    f"`{row.get('serve_log_file', '')}`\n"
                )
        else:
            handle.write("- none\n")
        handle.write("\n## Excluded / Anomalous Attempts\n\n")
        if excluded:
            for row in excluded:
                handle.write(
                    f"- {row.get('case_id', 'unknown')} "
                    f"{row.get('attempt_id', 'unknown')} "
                    f"{row.get('attempt_status', 'unknown')} "
                    f"{json.dumps(row.get('anomaly_reasons', []), ensure_ascii=False)} "
                    f"`{row.get('result_file', '')}`; "
                    f"`{row.get('metrics_file', '')}`; "
                    f"`{row.get('serve_log_file', '')}`\n"
                )
        else:
            handle.write("- none\n")
        handle.write("\n## All Attempt Paths\n\n")
        for row in rows:
            handle.write(
                f"- `{row['result_file']}`; `{row['metrics_file']}`; "
                f"`{row['serve_log_file']}`\n"
            )
    write_csv(stem.with_name(stem.name + "_all_attempts.csv"), rows)
    write_csv(stem.with_name(stem.name + "_valid_results.csv"), valid)
    write_csv(stem.with_name(stem.name + "_excluded_attempts.csv"), excluded)
    if anomalies:
        return EXIT_HARD
    if args.require_complete and len(valid) != EXPECTED_GROUP_RESULTS:
        return EXIT_HARD
    return EXIT_VALID


def command_summarize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    all_attempts = all_attempt_rows(run_dir)
    valid = canonical_rows(run_dir)
    excluded = [row for row in all_attempts if row.get("attempt_status") != "valid"]
    summary_dir = run_dir / "summary"
    write_csv(summary_dir / "all_attempts.csv", all_attempts, RAW_FIELDS)
    write_csv(summary_dir / "valid_results.csv", valid, RAW_FIELDS)
    write_csv(summary_dir / "excluded_attempts.csv", excluded, RAW_FIELDS)
    write_report(summary_dir / "report.md", all_attempts, [], [])
    if len(valid) != EXPECTED_VALID_RESULTS:
        raise ValueError(f"expected {EXPECTED_VALID_RESULTS} valid results, found {len(valid)}")
    median = median_rows(valid)
    if len(median) != EXPECTED_MEDIAN_ROWS:
        raise ValueError(f"expected {EXPECTED_MEDIAN_ROWS} median rows, found {len(median)}")
    recommendations = recommendation_table(median)
    median_fields = list(median[0])
    write_csv(summary_dir / "summary_median.csv", median, median_fields)
    write_xlsx(
        summary_dir / "summary.xlsx",
        all_attempts,
        valid,
        excluded,
        median,
        recommendations,
    )
    write_report(summary_dir / "report.md", all_attempts, median, recommendations)
    return EXIT_VALID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render-start")
    render.add_argument("--source", required=True)
    render.add_argument("--dest", required=True)
    render.add_argument("--diff-out", required=True)
    render.add_argument("--seq", required=True, type=int)
    render.add_argument("--bt", required=True, type=int)
    render.set_defaults(func=command_render_start)

    attempt = subparsers.add_parser("attempt")
    attempt.add_argument("--case-id", required=True)
    attempt.add_argument("--attempt-id", required=True)
    attempt.add_argument("--attempt-number", required=True, type=int)
    attempt.add_argument("--result-file", required=True)
    attempt.add_argument("--metrics-before", required=True)
    attempt.add_argument("--metrics-after", required=True)
    attempt.add_argument("--metrics-samples", required=True)
    attempt.add_argument("--inspect-before", required=True)
    attempt.add_argument("--inspect-after", required=True)
    attempt.add_argument("--case-serve-log", required=True)
    attempt.add_argument("--serve-log-file", required=True)
    attempt.add_argument("--bench-rc", required=True, type=int)
    attempt.add_argument("--output", required=True)
    attempt.add_argument("--run-dir")
    attempt.add_argument("--seq", type=int)
    attempt.add_argument("--bt", type=int)
    attempt.set_defaults(func=command_attempt)

    group = subparsers.add_parser("group-review")
    group.add_argument("--run-dir", required=True)
    group.add_argument("--seq", required=True, type=int)
    group.add_argument("--bt", required=True, type=int)
    group.add_argument("--require-complete", action="store_true")
    group.set_defaults(func=command_group_review)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--run-dir", required=True)
    summary.set_defaults(func=command_summarize)
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
