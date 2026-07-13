import argparse
import gzip
import json
import statistics
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.bench_report import (
    CASE_RE,
    EXPERIMENT,
    EXPERIMENT_CONFIG_SHA256,
    EXIT_HARD,
    EXIT_WARNING,
    RAW_FIELDS,
    AttemptContext,
    Sample,
    command_attempt,
    command_group_review,
    command_summarize,
    build_parser,
    group_anomalies,
    histogram_delta,
    load_json,
    median_rows,
    parse_case_id,
    parse_prometheus,
    parse_samples,
    render_start_script,
    write_csv,
    evaluate_attempt,
)


BEFORE = """\
vllm:request_queue_time_seconds_bucket{engine="0",le="0.3",model_name="Qwen3.6-27B"} 10
vllm:request_queue_time_seconds_bucket{engine="0",le="+Inf",model_name="Qwen3.6-27B"} 10
vllm:request_queue_time_seconds_count{engine="0",model_name="Qwen3.6-27B"} 10
vllm:request_queue_time_seconds_sum{engine="0",model_name="Qwen3.6-27B"} 0.010
vllm:prompt_tokens_total{engine="0",model_name="Qwen3.6-27B"} 1000
vllm:generation_tokens_total{engine="0",model_name="Qwen3.6-27B"} 2000
vllm:request_success_total{engine="0",finished_reason="length",model_name="Qwen3.6-27B"} 10
vllm:num_preemptions_total{engine="0",model_name="Qwen3.6-27B"} 0
"""

AFTER = """\
vllm:request_queue_time_seconds_bucket{engine="0",le="0.3",model_name="Qwen3.6-27B"} 15
vllm:request_queue_time_seconds_bucket{engine="0",le="+Inf",model_name="Qwen3.6-27B"} 15
vllm:request_queue_time_seconds_count{engine="0",model_name="Qwen3.6-27B"} 15
vllm:request_queue_time_seconds_sum{engine="0",model_name="Qwen3.6-27B"} 0.015
vllm:prompt_tokens_total{engine="0",model_name="Qwen3.6-27B"} 6120
vllm:generation_tokens_total{engine="0",model_name="Qwen3.6-27B"} 7120
vllm:request_success_total{engine="0",finished_reason="length",model_name="Qwen3.6-27B"} 15
vllm:num_preemptions_total{engine="0",model_name="Qwen3.6-27B"} 0
"""


def result_json(ttfts=None, failed=0, output_lens=None):
    ttfts = ttfts or [0.50, 0.51, 0.49, 0.50, 0.52]
    output_lens = output_lens or [1024] * 5
    return {
        "duration": 150.0,
        "completed": 5 - failed,
        "failed": failed,
        "total_input_tokens": 5120,
        "total_output_tokens": sum(output_lens),
        "request_throughput": 0.033,
        "output_throughput": 34.13,
        "input_lens": [1024] * 5,
        "output_lens": output_lens,
        "ttfts": ttfts,
        "itls": [[0.03, 0.03]] * 5,
        "errors": [""] * (5 - failed) + (["boom"] if failed else []),
        "mean_ttft_ms": statistics.mean(ttfts) * 1000,
        "median_ttft_ms": statistics.median(ttfts) * 1000,
        "p50_ttft_ms": 500.0,
        "p90_ttft_ms": 520.0,
        "p95_ttft_ms": 520.0,
        "p99_ttft_ms": 520.0,
        "mean_tpot_ms": 29.0,
        "p99_tpot_ms": 35.0,
        "mean_itl_ms": 90.0,
        "p99_itl_ms": 100.0,
        "mean_e2el_ms": 30000.0,
        "p99_e2el_ms": 32000.0,
    }


def attempt_row(
    case_id: str,
    *,
    attempt_number: int = 1,
    attempt_status: str = "valid",
    anomaly_reasons=None,
    warning_reasons=None,
    output_throughput_tok_s: float = 100.0,
    request_throughput_req_s: float = 1.0,
    mean_ttft_ms: float = 1000.0,
    p50_ttft_ms: float = 900.0,
    p90_ttft_ms: float = 1100.0,
    p95_ttft_ms: float = 1200.0,
    p99_ttft_ms: float = 1300.0,
    mean_tpot_ms: float = 29.0,
    p99_tpot_ms: float = 35.0,
    mean_itl_ms: float = 90.0,
    p99_itl_ms: float = 100.0,
    mean_e2el_ms: float = 30000.0,
    p99_e2el_ms: float = 32000.0,
    queue_mean_ms: float = 1.0,
    queue_p99_upper_bound_ms: float = 300.0,
    prefill_mean_ms: float = 5.0,
    decode_mean_ms: float = 6.0,
) -> dict:
    row = {
        "experiment_config_sha256": EXPERIMENT_CONFIG_SHA256,
        **parse_case_id(case_id),
        "num_prompts": 5,
        "output_throughput_tok_s": output_throughput_tok_s,
        "request_throughput_req_s": request_throughput_req_s,
        "mean_ttft_ms": mean_ttft_ms,
        "p50_ttft_ms": p50_ttft_ms,
        "p90_ttft_ms": p90_ttft_ms,
        "p95_ttft_ms": p95_ttft_ms,
        "p99_ttft_ms": p99_ttft_ms,
        "mean_tpot_ms": mean_tpot_ms,
        "p99_tpot_ms": p99_tpot_ms,
        "mean_itl_ms": mean_itl_ms,
        "p99_itl_ms": p99_itl_ms,
        "mean_e2el_ms": mean_e2el_ms,
        "p99_e2el_ms": p99_e2el_ms,
        "queue_mean_ms": queue_mean_ms,
        "queue_p99_ms": "N/A",
        "queue_p99_upper_bound_ms": queue_p99_upper_bound_ms,
        "prefill_mean_ms": prefill_mean_ms,
        "decode_mean_ms": decode_mean_ms,
        "kv_peak_pct": 1.0,
        "waiting_peak": 0,
        "running_peak": 1,
        "preemptions": 0,
        "failed_requests": 0,
        "error_count": 0,
        "case_id": case_id,
        "attempt_id": f"{case_id}_a{attempt_number}",
        "attempt_status": attempt_status,
        "attempt_number": attempt_number,
        "anomaly_reasons": anomaly_reasons or [],
        "warning_reasons": warning_reasons or [],
        "result_file": f"{case_id}_a{attempt_number}.json",
        "metrics_file": f"{case_id}_a{attempt_number}.prom.gz",
        "serve_log_file": f"{case_id}.serve.log",
    }
    return row


class ParseTests(unittest.TestCase):
    def test_experiment_config_is_the_matrix_source(self):
        self.assertEqual(EXPERIMENT["model"], "Qwen3.6-27B")
        self.assertEqual(EXPERIMENT["max_num_seqs"], [16, 24, 32])
        self.assertEqual(len(EXPERIMENT_CONFIG_SHA256), 64)

    def test_parse_case_id(self):
        case = parse_case_id("seq16_bt8192_in1024_out1024_c1_r3")
        self.assertEqual(
            case,
            {
                "max_num_seqs": 16,
                "max_num_batched_tokens": 8192,
                "input_len": 1024,
                "output_len": 1024,
                "concurrency": 1,
                "repeat_id": 3,
            },
        )

    def test_rejects_noncanonical_case_id(self):
        self.assertIsNone(CASE_RE.fullmatch("seq16-bt8192"))

    def test_histogram_delta_uses_bucket_upper_bound(self):
        before = parse_prometheus(BEFORE)
        after = parse_prometheus(AFTER)
        delta = histogram_delta(
            before, after, "vllm:request_queue_time_seconds"
        )
        self.assertEqual(delta.count, 5)
        self.assertAlmostEqual(delta.mean_ms, 1.0)
        self.assertEqual(delta.p99_upper_bound_ms, 300.0)

    def test_parse_gzipped_samples(self):
        text = """\
# sample_time_unix_ms 1000
vllm:num_requests_running{engine="0",model_name="Qwen3.6-27B"} 1
vllm:num_requests_waiting{engine="0",model_name="Qwen3.6-27B"} 0
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen3.6-27B"} 0.1
vllm:generation_tokens_total{engine="0",model_name="Qwen3.6-27B"} 10

# sample_time_unix_ms 2000
vllm:num_requests_running{engine="0",model_name="Qwen3.6-27B"} 0
vllm:num_requests_waiting{engine="0",model_name="Qwen3.6-27B"} 0
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen3.6-27B"} 0
vllm:generation_tokens_total{engine="0",model_name="Qwen3.6-27B"} 20
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.prom.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(text)
            samples = parse_samples(path)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].timestamp_ms, 1000)
        self.assertEqual(samples[0].running, 1)
        self.assertAlmostEqual(samples[0].kv_usage, 0.1)

    def test_render_start_changes_only_allowed_values(self):
        source = """\
#!/bin/bash
exec vllm serve /model \\
  --max-model-len 65535 \\
  --max-num-seqs 16 \\
  --max-num-batched-tokens 8192 \\
  --temperature 0
"""
        rendered, diff_text = render_start_script(source, seq=24, bt=12288)
        self.assertIn("--max-num-seqs 24", rendered)
        self.assertIn("--max-num-batched-tokens 12288", rendered)
        self.assertIn("--max-model-len 65535", rendered)
        self.assertIn("--temperature 0", rendered)
        self.assertIn("-  --max-num-seqs 16", diff_text)
        self.assertIn("+  --max-num-seqs 24", diff_text)

    def test_render_start_rejects_duplicate_parameter(self):
        source = """\
--max-num-seqs 16
--max-num-seqs 24
--max-num-batched-tokens 8192
"""
        with self.assertRaisesRegex(ValueError, "exactly once"):
            render_start_script(source, seq=24, bt=12288)

    def test_render_start_rejects_outside_matrix(self):
        source = "--max-num-seqs 16\n--max-num-batched-tokens 8192\n"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            render_start_script(source, seq=64, bt=12288)

    def test_classify_valid_attempt(self):
        context = AttemptContext(
            case_id="seq16_bt8192_in1024_out1024_c1_r1",
            result=result_json(),
            before=parse_prometheus(BEFORE),
            after=parse_prometheus(AFTER),
            samples=[
                Sample(1000, 1, 0, 0.008, 2000),
                Sample(2000, 0, 0, 0.000, 7120),
            ],
            bench_rc=0,
            restart_count_before=0,
            restart_count_after=0,
            started_at_before="same",
            started_at_after="same",
            oom_killed=False,
            serve_log="",
        )
        row, status, reasons = evaluate_attempt(context)
        self.assertEqual(status, "valid")
        self.assertEqual(reasons, [])
        self.assertEqual(row["num_prompts"], 5)
        self.assertEqual(row["kv_peak_pct"], 0.8)

    def test_tail_latency_warning_stays_valid(self):
        result = result_json()
        result["mean_ttft_ms"] = 1000.0
        result["p99_ttft_ms"] = 3500.0
        context = AttemptContext(
            case_id="seq16_bt8192_in1024_out1024_c1_r1",
            result=result,
            before=parse_prometheus(BEFORE),
            after=parse_prometheus(AFTER),
            samples=[Sample(1000, 1, 0, 0.008, 2000)],
            bench_rc=0,
            restart_count_before=0,
            restart_count_after=0,
            started_at_before="same",
            started_at_after="same",
            oom_killed=False,
            serve_log="aclgraph",
        )
        row, status, reasons = evaluate_attempt(context)
        self.assertEqual(status, "valid")
        self.assertEqual(reasons, [])
        self.assertEqual(row["warning_reasons"], ["tail_latency_warning"])

    def test_capacity_pressure_stays_valid_with_warnings(self):
        after = parse_prometheus(
            AFTER.replace(
                'vllm:num_preemptions_total{engine="0",model_name="Qwen3.6-27B"} 0',
                'vllm:num_preemptions_total{engine="0",model_name="Qwen3.6-27B"} 1',
            )
        )
        context = AttemptContext(
            case_id="seq16_bt8192_in1024_out1024_c1_r1",
            result=result_json(),
            before=parse_prometheus(BEFORE),
            after=after,
            samples=[Sample(1000, 1, 0, 0.9, 2000)],
            bench_rc=0,
            restart_count_before=0,
            restart_count_after=0,
            started_at_before="same",
            started_at_after="same",
            oom_killed=False,
            serve_log="",
        )
        row, status, reasons = evaluate_attempt(context)
        self.assertEqual(status, "valid")
        self.assertEqual(reasons, [])
        self.assertEqual(
            row["warning_reasons"], ["kv_over_85_pct", "preemption_warning"]
        )

    def test_warmup_miss_stays_retryable(self):
        result = result_json(ttfts=[13.541, 0.545, 0.482, 0.486, 0.474])
        context = AttemptContext(
            case_id="seq16_bt8192_in1024_out1024_c1_r1",
            result=result,
            before=parse_prometheus(BEFORE),
            after=parse_prometheus(AFTER),
            samples=[Sample(1000, 1, 0, 0.008, 2000)],
            bench_rc=0,
            restart_count_before=0,
            restart_count_after=0,
            started_at_before="same",
            started_at_after="same",
            oom_killed=False,
            serve_log="aclgraph",
        )
        _, status, reasons = evaluate_attempt(context)
        self.assertEqual(status, "retryable")
        self.assertIn("warmup_miss", reasons)

    def test_failed_request_is_hard_failure(self):
        context = AttemptContext(
            case_id="seq16_bt8192_in1024_out1024_c1_r1",
            result=result_json(failed=1),
            before=parse_prometheus(BEFORE),
            after=parse_prometheus(AFTER),
            samples=[Sample(1000, 1, 0, 0.008, 2000)],
            bench_rc=0,
            restart_count_before=0,
            restart_count_after=0,
            started_at_before="same",
            started_at_after="same",
            oom_killed=False,
            serve_log="",
        )
        _, status, reasons = evaluate_attempt(context)
        self.assertEqual(status, "hard")
        self.assertIn("failed_requests", reasons)

    def test_repeat_dispersion_stops_group(self):
        rows = []
        for repeat, throughput in enumerate((100.0, 105.0, 140.0), 1):
            row = {
                **parse_case_id(
                    f"seq16_bt8192_in1024_out1024_c5_r{repeat}"
                ),
                "attempt_status": "valid",
                "output_throughput_tok_s": throughput,
                "mean_ttft_ms": 1000.0,
                "mean_tpot_ms": 30.0,
                "mean_itl_ms": 90.0,
                "mean_e2el_ms": 30000.0,
            }
            rows.append(row)
        anomalies = group_anomalies(rows)
        self.assertTrue(
            any(item["rule"] == "repeat_dispersion" for item in anomalies)
        )

    def test_c1_vs_c10_ttft_rule(self):
        rows = []
        for concurrency, ttft in ((1, 1600.0), (10, 1000.0)):
            for repeat in (1, 2, 3):
                rows.append(
                    {
                        **parse_case_id(
                            "seq16_bt8192_in1024_out1024_"
                            f"c{concurrency}_r{repeat}"
                        ),
                        "attempt_status": "valid",
                        "output_throughput_tok_s": 100.0 * concurrency,
                        "mean_ttft_ms": ttft,
                        "mean_tpot_ms": 30.0,
                        "mean_itl_ms": 90.0,
                        "mean_e2el_ms": 30000.0,
                    }
                )
        anomalies = group_anomalies(rows)
        self.assertTrue(
            any(item["rule"] == "c1_ttft_vs_c10" for item in anomalies)
        )

    def test_write_csv_preserves_na(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            write_csv(path, [{"queue_p99_ms": "N/A", "repeat_id": 1}], RAW_FIELDS)
            text = path.read_text()
        self.assertIn("queue_p99_ms", text)
        self.assertIn("N/A", text)

    def test_cli_exposes_expected_subcommands_only(self):
        parser = build_parser()
        subcommands = parser._subparsers._group_actions[0].choices
        self.assertEqual(
            set(subcommands),
            {"config-shell", "render-start", "attempt", "group-review", "summarize"},
        )
        self.assertNotIn("restore", subcommands)

    def test_median_rows_groups_three_repeats(self):
        rows = []
        for repeat, throughput in enumerate((100.0, 120.0, 110.0), 1):
            rows.append(
                {
                    **parse_case_id(
                        f"seq16_bt8192_in1024_out1024_c1_r{repeat}"
                    ),
                    "attempt_status": "valid",
                    "output_throughput_tok_s": throughput,
                    "mean_ttft_ms": 500.0 + repeat,
                    "p99_ttft_ms": 550.0 + repeat,
                }
            )
        summary = median_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["output_throughput_tok_s"], 110.0)
        self.assertIn("p99_ttft_cv", summary[0])

    def test_command_attempt_returns_warning_exit_code(self):
        result = result_json()
        result["mean_ttft_ms"] = 1000.0
        result["p95_ttft_ms"] = 1500.0
        result["p99_ttft_ms"] = 3500.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result))
            before_path = root / "before.prom"
            before_path.write_text(BEFORE)
            after_path = root / "after.prom"
            after_path.write_text(AFTER)
            samples_path = root / "samples.prom.gz"
            with gzip.open(samples_path, "wt") as handle:
                handle.write(
                    "# sample_time_unix_ms 1000\n"
                    'vllm:num_requests_running{engine="0",model_name="Qwen3.6-27B"} 1\n'
                    'vllm:num_requests_waiting{engine="0",model_name="Qwen3.6-27B"} 0\n'
                    'vllm:kv_cache_usage_perc{engine="0",model_name="Qwen3.6-27B"} 0.1\n'
                    'vllm:generation_tokens_total{engine="0",model_name="Qwen3.6-27B"} 10\n'
                )
            inspect_path = root / "inspect.json"
            inspect_path.write_text(
                json.dumps({"RestartCount": 0, "State": {"StartedAt": "same", "OOMKilled": False}})
            )
            case_log = root / "case.log"
            case_log.write_text("")
            output_path = root / "attempt.json"
            rc = command_attempt(
                argparse.Namespace(
                    case_id="seq16_bt8192_in1024_out1024_c1_r1",
                    attempt_id="seq16_bt8192_in1024_out1024_c1_r1_a1",
                    attempt_number=1,
                    result_file=str(result_path),
                    metrics_before=str(before_path),
                    metrics_after=str(after_path),
                    metrics_samples=str(samples_path),
                    inspect_before=str(inspect_path),
                    inspect_after=str(inspect_path),
                    case_serve_log=str(case_log),
                    serve_log_file="serve.log",
                    bench_rc=0,
                    output=str(output_path),
                    run_dir=str(root),
                    seq=16,
                    bt=8192,
                )
            )
            payload = load_json(output_path)
        self.assertEqual(rc, EXIT_WARNING)
        self.assertEqual(payload["attempt_status"], "valid")
        self.assertEqual(payload["warning_reasons"], ["tail_latency_warning"])

    def test_group_review_tracks_warning_attempts_and_latest_valid_per_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            attempts_dir = run_dir / "summary" / "attempts"
            attempts_dir.mkdir(parents=True)
            warning_case = "seq16_bt8192_in1024_out1024_c1_r1"
            latest_case = "seq16_bt8192_in1024_out1024_c1_r2"
            rows = [
                attempt_row(
                    warning_case,
                    attempt_number=1,
                    warning_reasons=["tail_latency_warning"],
                ),
                attempt_row(
                    latest_case,
                    attempt_number=1,
                    output_throughput_tok_s=90.0,
                ),
                attempt_row(
                    latest_case,
                    attempt_number=2,
                    output_throughput_tok_s=120.0,
                ),
                attempt_row(
                    "seq16_bt8192_in1024_out1024_c5_r1",
                    attempt_number=1,
                    attempt_status="hard",
                    anomaly_reasons=["failed_requests"],
                ),
            ]
            for index, row in enumerate(rows, 1):
                (attempts_dir / f"row{index}.json").write_text(json.dumps(row))
            rc = command_group_review(
                argparse.Namespace(run_dir=str(run_dir), seq=16, bt=8192, require_complete=False)
            )
            review = load_json(run_dir / "summary" / "group_seq16_bt8192_review.json")
            review_md = (run_dir / "summary" / "group_seq16_bt8192_review.md").read_text()
            valid_csv = (run_dir / "summary" / "group_seq16_bt8192_valid_results.csv").read_text()
            excluded_csv = (run_dir / "summary" / "group_seq16_bt8192_excluded_attempts.csv").read_text()
        self.assertEqual(rc, 0)
        self.assertEqual(review["warning_attempt_count"], 1)
        self.assertEqual(len(review["warning_attempts"]), 1)
        self.assertEqual(review["valid_result_count"], 2)
        self.assertFalse(review["complete"])
        self.assertIn("Warning attempts: 1", review_md)
        self.assertIn("tail_latency_warning", review_md)
        self.assertIn("failed_requests", excluded_csv)
        self.assertIn(f"{latest_case}_a2", valid_csv)
        self.assertNotIn(f"{latest_case}_a1", valid_csv)

    def test_summarize_keeps_warning_attempt_out_of_excluded_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            attempts_dir = run_dir / "summary" / "attempts"
            attempts_dir.mkdir(parents=True)
            base_row = {
                **parse_case_id("seq16_bt8192_in1024_out1024_c1_r1"),
                "num_prompts": 5,
                "output_throughput_tok_s": 100.0,
                "request_throughput_req_s": 1.0,
                "mean_ttft_ms": 1000.0,
                "p50_ttft_ms": 900.0,
                "p90_ttft_ms": 1100.0,
                "p95_ttft_ms": 1200.0,
                "p99_ttft_ms": 3500.0,
                "mean_tpot_ms": 29.0,
                "p99_tpot_ms": 35.0,
                "mean_itl_ms": 90.0,
                "p99_itl_ms": 100.0,
                "mean_e2el_ms": 30000.0,
                "p99_e2el_ms": 32000.0,
                "queue_mean_ms": 1.0,
                "queue_p99_ms": "N/A",
                "queue_p99_upper_bound_ms": 300.0,
                "prefill_mean_ms": 5.0,
                "decode_mean_ms": 6.0,
                "kv_peak_pct": 1.0,
                "waiting_peak": 0,
                "running_peak": 1,
                "preemptions": 0,
                "failed_requests": 0,
                "error_count": 0,
                "result_file": "valid.json",
                "metrics_file": "valid.prom.gz",
                "serve_log_file": "serve.log",
            }
            warning_row = {
                **base_row,
                "attempt_id": "seq16_bt8192_in1024_out1024_c1_r1_a1",
                "attempt_number": 1,
                "case_id": "seq16_bt8192_in1024_out1024_c1_r1",
                "attempt_status": "valid",
                "anomaly_reasons": [],
                "warning_reasons": ["tail_latency_warning"],
            }
            hard_row = {
                **base_row,
                **parse_case_id("seq16_bt8192_in1024_out1024_c1_r2"),
                "attempt_id": "seq16_bt8192_in1024_out1024_c1_r2_a1",
                "attempt_number": 1,
                "case_id": "seq16_bt8192_in1024_out1024_c1_r2",
                "attempt_status": "hard",
                "anomaly_reasons": ["failed_requests"],
                "warning_reasons": [],
            }
            (attempts_dir / "warning.json").write_text(json.dumps(warning_row))
            (attempts_dir / "hard.json").write_text(json.dumps(hard_row))
            with self.assertRaisesRegex(ValueError, "expected 270 valid results"):
                command_summarize(argparse.Namespace(run_dir=str(run_dir)))
            all_attempts = (run_dir / "summary" / "all_attempts.csv").read_text()
            excluded = (run_dir / "summary" / "excluded_attempts.csv").read_text()
            report = (run_dir / "summary" / "report.md").read_text()
        self.assertIn("tail_latency_warning", all_attempts)
        self.assertNotIn("tail_latency_warning", excluded)
        self.assertIn("Warnings", report)
        self.assertIn("tail_latency_warning", report)

    def test_summarize_success_writes_full_final_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            attempts_dir = run_dir / "summary" / "attempts"
            attempts_dir.mkdir(parents=True)
            rows = []
            for seq in (16, 24, 32):
                for bt in (8192, 12288, 16384):
                    for input_len in (1024, 4096):
                        for concurrency in (1, 5, 10, 16, 32):
                            for repeat in (1, 2, 3):
                                case_id = (
                                    f"seq{seq}_bt{bt}_in{input_len}_out1024_"
                                    f"c{concurrency}_r{repeat}"
                                )
                                rows.append(
                                    attempt_row(
                                        case_id,
                                        output_throughput_tok_s=float(seq * 100 + bt / 256 + concurrency + repeat),
                                        request_throughput_req_s=float(concurrency) / 10.0,
                                        mean_ttft_ms=float(700 + input_len / 16 + repeat),
                                        p50_ttft_ms=float(650 + repeat),
                                        p90_ttft_ms=float(800 + concurrency * 5 + repeat),
                                        p95_ttft_ms=float(850 + concurrency * 5 + repeat),
                                        p99_ttft_ms=float(950 + concurrency * 10 + repeat),
                                        mean_tpot_ms=float(20 + bt / 4096 + repeat / 10.0),
                                        mean_itl_ms=float(60 + concurrency),
                                        mean_e2el_ms=float(20000 + input_len + concurrency * 10 + repeat),
                                        queue_mean_ms=float(1 + concurrency / 10.0),
                                        queue_p99_upper_bound_ms=float(100 + concurrency),
                                        prefill_mean_ms=float(3 + input_len / 2048.0),
                                        decode_mean_ms=float(4 + repeat / 10.0),
                                        warning_reasons=["tail_latency_warning"]
                                        if seq == 16 and bt == 8192 and input_len == 1024 and concurrency == 1 and repeat == 1
                                        else [],
                                    )
                                )
            duplicate_case = "seq16_bt8192_in1024_out1024_c1_r1"
            rows.append(
                attempt_row(
                    duplicate_case,
                    attempt_number=2,
                    output_throughput_tok_s=999.0,
                    mean_ttft_ms=701.0,
                    p99_ttft_ms=960.0,
                    warning_reasons=["tail_latency_warning"],
                )
            )
            for index, row in enumerate(rows, 1):
                (attempts_dir / f"row{index}.json").write_text(json.dumps(row))
            rc = command_summarize(argparse.Namespace(run_dir=str(run_dir)))
            summary_dir = run_dir / "summary"
            all_attempts = summary_dir / "all_attempts.csv"
            valid_results = summary_dir / "valid_results.csv"
            excluded = summary_dir / "excluded_attempts.csv"
            median = summary_dir / "summary_median.csv"
            report = summary_dir / "report.md"
            workbook_path = summary_dir / "summary.xlsx"
            with zipfile.ZipFile(workbook_path) as workbook_zip:
                workbook_xml = workbook_zip.read("xl/workbook.xml").decode("utf-8")
            self.assertEqual(rc, 0)
            self.assertEqual(len(all_attempts.read_text().splitlines()), 272)
            self.assertEqual(len(valid_results.read_text().splitlines()), 271)
            self.assertEqual(len(excluded.read_text().splitlines()), 1)
            self.assertEqual(len(median.read_text().splitlines()), 91)
            self.assertIn("Warning attempts: 2", report.read_text())
            self.assertIn("tail_latency_warning", report.read_text())
            for sheet_name in (
                "All Attempts",
                "Valid Results",
                "Excluded Attempts",
                "Median Summary",
                "Recommendations",
            ):
                self.assertIn(f'name="{sheet_name}"', workbook_xml)

    def test_group_review_complete_uses_extended_matrix_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            attempts_dir = run_dir / "summary" / "attempts"
            attempts_dir.mkdir(parents=True)
            rows = []
            for input_len in (1024, 4096):
                for concurrency in (1, 5, 10, 16, 32):
                    for repeat in (1, 2, 3):
                        rows.append(
                            attempt_row(
                                f"seq16_bt8192_in{input_len}_out1024_c{concurrency}_r{repeat}",
                                attempt_status="valid",
                            )
                        )
            for index, row in enumerate(rows, 1):
                (attempts_dir / f"row{index}.json").write_text(json.dumps(row))
            rc = command_group_review(
                argparse.Namespace(run_dir=str(run_dir), seq=16, bt=8192, require_complete=True)
            )
            payload = json.loads((run_dir / "summary" / "group_seq16_bt8192_review.json").read_text())
            self.assertEqual(rc, 0)
            self.assertEqual(payload["valid_result_count"], 30)
            self.assertEqual(payload["expected_result_count"], 30)
            self.assertTrue(payload["complete"])

    def test_group_review_old_matrix_is_incomplete_after_c32_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            attempts_dir = run_dir / "summary" / "attempts"
            attempts_dir.mkdir(parents=True)
            rows = []
            for input_len in (1024, 4096):
                for concurrency in (1, 5, 10, 16):
                    for repeat in (1, 2, 3):
                        rows.append(
                            attempt_row(
                                f"seq16_bt8192_in{input_len}_out1024_c{concurrency}_r{repeat}",
                                attempt_status="valid",
                            )
                        )
            for index, row in enumerate(rows, 1):
                (attempts_dir / f"row{index}.json").write_text(json.dumps(row))
            rc = command_group_review(
                argparse.Namespace(run_dir=str(run_dir), seq=16, bt=8192, require_complete=True)
            )
            self.assertEqual(rc, EXIT_HARD)

    def test_command_attempt_upgrades_systemic_tail_latency_to_hard(self):
        result = result_json()
        result["mean_ttft_ms"] = 2000.0
        result["p90_ttft_ms"] = 2600.0
        result["p95_ttft_ms"] = 4500.0
        result["p99_ttft_ms"] = 7000.0
        before_text = BEFORE + (
            'vllm:request_prefill_time_seconds_bucket{engine="0",le="0.3",model_name="Qwen3.6-27B"} 10\n'
            'vllm:request_prefill_time_seconds_bucket{engine="0",le="+Inf",model_name="Qwen3.6-27B"} 10\n'
            'vllm:request_prefill_time_seconds_count{engine="0",model_name="Qwen3.6-27B"} 10\n'
            'vllm:request_prefill_time_seconds_sum{engine="0",model_name="Qwen3.6-27B"} 0.010\n'
        )
        after_text = AFTER + (
            'vllm:request_queue_time_seconds_bucket{engine="0",le="3",model_name="Qwen3.6-27B"} 15\n'
            'vllm:request_prefill_time_seconds_bucket{engine="0",le="0.3",model_name="Qwen3.6-27B"} 15\n'
            'vllm:request_prefill_time_seconds_bucket{engine="0",le="+Inf",model_name="Qwen3.6-27B"} 15\n'
            'vllm:request_prefill_time_seconds_count{engine="0",model_name="Qwen3.6-27B"} 15\n'
            'vllm:request_prefill_time_seconds_sum{engine="0",model_name="Qwen3.6-27B"} 0.060\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempts_dir = root / "summary" / "attempts"
            attempts_dir.mkdir(parents=True)
            baseline = {
                **parse_case_id("seq16_bt8192_in1024_out1024_c1_r9"),
                "attempt_status": "valid",
                "mean_ttft_ms": 1000.0,
                "p90_ttft_ms": 1200.0,
                "p95_ttft_ms": 1400.0,
                "p99_ttft_ms": 1800.0,
                "queue_mean_ms": 0.5,
                "queue_p99_upper_bound_ms": 100.0,
                "prefill_mean_ms": 2.0,
                "warning_reasons": [],
            }
            (attempts_dir / "baseline.json").write_text(json.dumps(baseline))
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result))
            before_path = root / "before.prom"
            before_path.write_text(before_text)
            after_path = root / "after.prom"
            after_path.write_text(after_text)
            samples_path = root / "samples.prom.gz"
            with gzip.open(samples_path, "wt") as handle:
                handle.write(
                    "# sample_time_unix_ms 1000\n"
                    'vllm:num_requests_running{engine="0",model_name="Qwen3.6-27B"} 1\n'
                    'vllm:num_requests_waiting{engine="0",model_name="Qwen3.6-27B"} 0\n'
                    'vllm:kv_cache_usage_perc{engine="0",model_name="Qwen3.6-27B"} 0.1\n'
                    'vllm:generation_tokens_total{engine="0",model_name="Qwen3.6-27B"} 10\n'
                )
            inspect_path = root / "inspect.json"
            inspect_path.write_text(
                json.dumps({"RestartCount": 0, "State": {"StartedAt": "same", "OOMKilled": False}})
            )
            case_log = root / "case.log"
            case_log.write_text("")
            output_path = root / "attempt.json"
            rc = command_attempt(
                argparse.Namespace(
                    case_id="seq16_bt8192_in1024_out1024_c1_r1",
                    attempt_id="seq16_bt8192_in1024_out1024_c1_r1_a1",
                    attempt_number=1,
                    result_file=str(result_path),
                    metrics_before=str(before_path),
                    metrics_after=str(after_path),
                    metrics_samples=str(samples_path),
                    inspect_before=str(inspect_path),
                    inspect_after=str(inspect_path),
                    case_serve_log=str(case_log),
                    serve_log_file="serve.log",
                    bench_rc=0,
                    output=str(output_path),
                    run_dir=str(root),
                    seq=16,
                    bt=8192,
                )
            )
            payload = load_json(output_path)
        self.assertEqual(rc, 20)
        self.assertEqual(payload["attempt_status"], "hard")
        self.assertIn("tail_latency_systemic_degradation", payload["anomaly_reasons"])


if __name__ == "__main__":
    unittest.main()
