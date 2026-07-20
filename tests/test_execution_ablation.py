import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("EXPERIMENT_CONFIG", "configs/experiment.yaml")

from scripts.bench_report import (  # noqa: E402
    build_parser,
    command_config_ablation,
    command_summarize_ablation,
    render_start_script,
    summarize_ablation,
)
from scripts.check_correctness import compare  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
START_SAMPLE = """\
#!/bin/bash
set -e
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
exec vllm serve /model \\
    --max-num-seqs 16 \\
    --max-num-batched-tokens 8192 \\
    --async-scheduling \\
    --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \\
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
"""


def _load_yaml(path: Path):
    import yaml

    return yaml.safe_load(path.read_text())


class TestRenderAblationToggles(unittest.TestCase):
    def test_base_keeps_all(self):
        out, _ = render_start_script(START_SAMPLE, 32, 16384)
        self.assertIn("--async-scheduling", out)
        self.assertIn("speculative_config", out)
        self.assertIn("FULL_DECODE_ONLY", out)

    def test_async_off_drops_flag_keeps_mtp(self):
        out, _ = render_start_script(START_SAMPLE, 32, 16384, disable_async=True)
        self.assertNotIn("--async-scheduling", out)
        self.assertIn("speculative_config", out)
        self.assertIn("FULL_DECODE_ONLY", out)

    def test_mtp_off_drops_speculative(self):
        out, _ = render_start_script(START_SAMPLE, 32, 16384, disable_mtp=True)
        self.assertNotIn("speculative_config", out)
        self.assertIn("--async-scheduling", out)

    def test_eager_switches_graph_mode(self):
        out, _ = render_start_script(START_SAMPLE, 32, 16384, eager=True)
        self.assertIn("NONE", out)
        self.assertNotIn("FULL_DECODE_ONLY", out)

    def test_applies_baseline_seq_bt(self):
        out, _ = render_start_script(START_SAMPLE, 32, 16384)
        self.assertIn("--max-num-seqs 32", out)
        self.assertIn("--max-num-batched-tokens 16384", out)


class TestAblationConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = _load_yaml(REPO / "configs" / "execution_ablation.yaml")

    def test_baseline_is_32_16384(self):
        self.assertEqual(self.cfg["baseline"]["max_num_seqs"], 32)
        self.assertEqual(self.cfg["baseline"]["max_num_batched_tokens"], 16384)

    def test_four_configs_single_variable(self):
        for cid, spec in self.cfg["configs"].items():
            if cid == "BASE":
                self.assertEqual(spec["changes"], [])
            else:
                self.assertEqual(len(spec["changes"]), 1)

    def test_rotation_order_matches_plan(self):
        rot = self.cfg["rotation"]
        self.assertEqual(rot["repeat1"], ["BASE", "ASYNC_OFF", "MTP_OFF", "EAGER"])
        self.assertEqual(rot["repeat2"], ["MTP_OFF", "EAGER", "BASE", "ASYNC_OFF"])
        self.assertEqual(rot["repeat3"], ["EAGER", "BASE", "ASYNC_OFF", "MTP_OFF"])

    def test_workloads_defined(self):
        self.assertIn("prefill_heavy", self.cfg["workloads"])
        self.assertIn("decode_heavy", self.cfg["workloads"])
        self.assertEqual(self.cfg["workloads"]["prefill_heavy"]["concurrency"], 1)
        self.assertEqual(self.cfg["workloads"]["decode_heavy"]["concurrency"], 32)


class TestConfigAblationEmitter(unittest.TestCase):
    def test_emits_shell_vars(self):
        parser = build_parser()
        args = parser.parse_args(
            ["config-ablation", "--config", str(REPO / "configs" / "execution_ablation.yaml")]
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = command_config_ablation(args)
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("ABLATION_BASELINE_SEQ=32", text)
        self.assertIn("ABLATION_BASELINE_BT=16384", text)
        self.assertIn("ABLATION_CONFIG_IDS='BASE ASYNC_OFF MTP_OFF EAGER'", text)
        self.assertIn("ABLATION_ROTATION_JSON=", text)
        self.assertIn("ABLATION_TIMEOUT_SECONDS=1800", text)


class TestAnalysisRules(unittest.TestCase):
    def setUp(self):
        self.rules = _load_yaml(REPO / "configs" / "analysis_rules.yaml")

    def test_baseline_drift_method(self):
        self.assertEqual(self.rules["baseline_drift"]["method"], "historical_median_mad")

    def test_drift_threshold_floor(self):
        for metric, thr in self.rules["baseline_drift"]["max_relative_deviation"].items():
            self.assertGreaterEqual(thr, 0.05)

    def test_primary_guardrail_present(self):
        pg = self.rules["performance_decision"]["primary_guardrail"]
        self.assertEqual(set(pg), {"ASYNC_OFF", "MTP_OFF", "EAGER"})

    def test_negative_controls_present(self):
        self.assertIn("ASYNC_OFF_x_prefill_C1", self.rules["negative_controls"])


class TestAblationSummary(unittest.TestCase):
    def test_median_and_mtp_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            attempts = run_dir / "summary" / "attempts"
            raw = run_dir / "results_raw" / "attempts"
            attempts.mkdir(parents=True)
            raw.mkdir(parents=True)
            for config in ("BASE", "MTP_OFF"):
                for repeat, throughput in enumerate((100, 120, 110), 1):
                    raw_path = raw / f"{config}_{repeat}.json"
                    raw_path.write_text(json.dumps({"spec_decode_accepted_tokens": 30 if config == "BASE" else None}))
                    row = {
                        "config": config, "input_len": 1024, "output_len": 1024,
                        "concurrency": 32, "repeat_id": repeat, "attempt_number": 1,
                        "attempt_status": "valid", "output_throughput_tok_s": throughput,
                        "result_file": str(raw_path),
                    }
                    (attempts / f"abl_{config}_{repeat}.json").write_text(json.dumps(row))
            rows, summaries = summarize_ablation(run_dir)
            output_dir = run_dir / "public"
            command_summarize_ablation(SimpleNamespace(run_dir=str(run_dir), output_dir=str(output_dir)))
            raw_header = (output_dir / "raw.csv").read_text().splitlines()[0]
        self.assertEqual(len(rows), 6)
        base = next(row for row in summaries if row["config"] == "BASE")
        self.assertEqual(base["output_throughput_tok_s"], 110)
        self.assertTrue(base["complete"])
        self.assertEqual(base["spec_decode_accepted_tokens"], 30)
        self.assertNotIn("result_file", raw_header)


class TestCorrectnessCompare(unittest.TestCase):
    def test_captures_first_divergence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {"request_id": "000-abc", "finish_reason": "stop", "text_sha256": "x"}
            (root / "base.json").write_text(json.dumps([{**common, "config": "BASE", "token_ids": [1, 2, 3]}]))
            (root / "candidate.json").write_text(json.dumps([{**common, "config": "MTP_OFF", "token_ids": [1, 9, 3]}]))
            rows = compare(root / "base.json", root / "candidate.json", root / "correctness.csv")
        self.assertFalse(rows[0]["exact_token_ids"])
        self.assertEqual(rows[0]["first_divergence"], 1)

    def test_workflow_allows_natural_eos(self):
        script = (REPO / "scripts" / "bench_execution_ablation.sh").read_text()
        self.assertIn('--config "$cfg" --max-tokens 1024', script)

    def test_workflow_resumes_valid_capture(self):
        script = (REPO / "scripts" / "bench_execution_ablation.sh").read_text()
        self.assertIn("correctness_capture_complete() {", script)
        self.assertIn('correctness_capture_complete "$run_dir/results/correctness_${cfg}.json" "$cfg"', script)


class TestBaselineReplayWorkflow(unittest.TestCase):
    def test_warmup_precedes_replay_repeats(self):
        script = (REPO / "scripts" / "bench_execution_ablation.sh").read_text()
        body = script.split("run_baseline_replay() {", 1)[1].split("\n}\n", 1)[0]
        self.assertIn("run_warmup", body)
        self.assertLess(body.index("run_warmup"), body.index("for rep in 1 2 3"))

    def test_historical_csv_is_frozen_inside_run(self):
        script = (REPO / "scripts" / "bench_execution_ablation.sh").read_text()
        self.assertIn('cp "$HISTORICAL_RESULTS" "$run_dir/configs/historical_valid_results.csv"', script)
        self.assertIn('"$run_dir/configs/historical_valid_results.csv" "$EXEC_RULES"', script)


class TestManifestIntegrity(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(
            (REPO / "configs" / "expected_manifest.json").read_text()
        )

    def test_frozen_commits_present(self):
        self.assertIn("vllm_commit", self.manifest["software"])
        self.assertIn("vllm_ascend_commit", self.manifest["software"])
        self.assertEqual(len(self.manifest["software"]["vllm_commit"]), 40)

    def test_manifest_mismatch_detected(self):
        tampered = json.loads(json.dumps(self.manifest))
        tampered["software"]["vllm_commit"] = "0" * 40
        self.assertNotEqual(self.manifest, tampered)

    def test_running_service_flagged(self):
        self.assertEqual(
            self.manifest["running_service_at_freeze"]["max_num_seqs"], 16
        )
        self.assertEqual(
            self.manifest["running_service_at_freeze"]["action_required"],
            "restart_service_to_baseline_32_16384",
        )


if __name__ == "__main__":
    unittest.main()
