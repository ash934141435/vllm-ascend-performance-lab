#!/usr/bin/env bash
# Stage-2 execution-mechanism ablation entry (plan §10.2).
# Thin: selects configs + rotation order; reuses bench_matrix.sh service/warmup/
# sampler/verify logic and bench_report.py for validation. Does NOT reimplement
# a benchmark framework.
#
# UNVERIFIED: not executed against the live service yet. Requires the service to
# be restarted to the 32/16384 baseline first (see configs/expected_manifest.json).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bench_matrix.sh
source "$SCRIPT_DIR/bench_matrix.sh"

# Diagnose unexpected termination: if the driver is killed by a signal (e.g.
# session cleanup), record it so we can distinguish signal-death from a
# set -e code exit. cleanup() is defined by bench_matrix.sh.
trap 'cleanup 2>/dev/null; echo "[$(date -Iseconds)] SIGNAL TERM/INT/HUP pid=$$" >>/tmp/ablation_signal.log' TERM INT HUP
# checkpoint written before each case so a silent death is locatable
ablation_checkpoint() { echo "[$(date -Iseconds)] CK $*" >>/tmp/ablation_ckpt.log; }

EXEC_CONFIG="${EXEC_CONFIG:-$SCRIPT_DIR/../configs/execution_ablation.yaml}"
EXEC_RULES="${EXEC_RULES:-$SCRIPT_DIR/../configs/analysis_rules.yaml}"
PROMPT_DIR="${PROMPT_DIR:-$SCRIPT_DIR/../results/execution_ablation/prompts}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../results/execution_ablation}"
HISTORICAL_RESULTS="${HISTORICAL_RESULTS:-$SCRIPT_DIR/../results/valid_results.csv}"

load_experiment_config
eval "$("$PYTHON_BIN" "$REPORT_SCRIPT" config-ablation --config "$EXEC_CONFIG")"

# Per-config toggle env from the matrix JSON.
set_config_toggles() {
  local cfg="$1"
  unset DISABLE_ASYNC DISABLE_MTP EAGER
  local da dm ea
  da="$(printf '%s' "$ABLATION_MATRIX_JSON" | "$PYTHON_BIN" -c "import json,sys; print(json.load(sys.stdin)['$cfg'].get('disable_async', False))")"
  dm="$(printf '%s' "$ABLATION_MATRIX_JSON" | "$PYTHON_BIN" -c "import json,sys; print(json.load(sys.stdin)['$cfg'].get('disable_mtp', False))")"
  ea="$(printf '%s' "$ABLATION_MATRIX_JSON" | "$PYTHON_BIN" -c "import json,sys; print(json.load(sys.stdin)['$cfg'].get('eager', False))")"
  [[ "$da" == "True" ]] && DISABLE_ASYNC=1 || true
  [[ "$dm" == "True" ]] && DISABLE_MTP=1 || true
  [[ "$ea" == "True" ]] && EAGER=1 || true
  return 0
}

workload_params() {
  local w="$1" key="$2"
  printf '%s' "$ABLATION_WORKLOADS_JSON" | "$PYTHON_BIN" -c "import json,sys; print(json.load(sys.stdin)['$w']['$key'])"
}

config_complete() {
  local run_dir="$1" cfg="$2" repeat="$3" w in_len out_len conc summary
  for w in prefill_heavy decode_heavy; do
    in_len="$(workload_params "$w" prompt_tokens_approx)"
    out_len="$(workload_params "$w" output_tokens)"
    conc="$(workload_params "$w" concurrency)"
    summary="$run_dir/summary/attempts/abl_${cfg}_in${in_len}_out${out_len}_c${conc}_r${repeat}_a1.json"
    [[ -f "$summary" ]] || return 1
    "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(json.load(open(sys.argv[1], encoding="utf-8")).get("attempt_status") != "valid")' "$summary" || return 1
  done
}

run_baseline_replay() {
  local run_dir="$1" rep override
  if [[ -f "$run_dir/baseline_replay.json" ]] && \
    "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(not json.load(open(sys.argv[1])).get("verified"))' \
      "$run_dir/baseline_replay.json"; then
    log "skip verified baseline replay"
    return 0
  fi
  unset DISABLE_ASYNC DISABLE_MTP EAGER
  if [[ -n "${REUSE_RUNNING_BASE:-}" ]]; then
    verify_running_config "$ABLATION_BASELINE_SEQ" "$ABLATION_BASELINE_BT"
    verify_ablation
    verify_expected_manifest "$run_dir"
    host_idle_guard "$run_dir" freeze
  else
    override="$(prepare_group_files "$run_dir" "$ABLATION_BASELINE_SEQ" "$ABLATION_BASELINE_BT")"
    restart_group_service "$run_dir" "$ABLATION_BASELINE_SEQ" "$ABLATION_BASELINE_BT" "$override"
  fi
  run_warmup "$run_dir" "$ABLATION_BASELINE_SEQ" "$ABLATION_BASELINE_BT" 1024 32
  for rep in 1 2 3; do
    run_case "$run_dir" "$ABLATION_BASELINE_SEQ" "$ABLATION_BASELINE_BT" 1024 32 "$rep"
    [[ "$CASE_RC" -eq 0 ]] || die "baseline replay invalid at repeat $rep"
  done
  "$PYTHON_BIN" - "$run_dir" "$run_dir/configs/historical_valid_results.csv" "$EXEC_RULES" <<'PY'
import csv, json, statistics, sys
from pathlib import Path
import yaml

run_dir, historical_path, rules_path = map(Path, sys.argv[1:])
metrics = yaml.safe_load(rules_path.read_text(encoding="utf-8"))["baseline_drift"]["max_relative_deviation"]
historical = [r for r in csv.DictReader(historical_path.open(encoding="utf-8"))
              if r["max_num_seqs"] == "32" and r["max_num_batched_tokens"] == "16384"
              and r["input_len"] == "1024" and r["output_len"] == "1024" and r["concurrency"] == "32"]
replay = []
for repeat in (1, 2, 3):
    candidates = sorted((run_dir / "summary" / "attempts").glob(
        f"seq32_bt16384_in1024_out1024_c32_r{repeat}_a*.json"))
    valid = [json.loads(p.read_text(encoding="utf-8")) for p in candidates
             if json.loads(p.read_text(encoding="utf-8")).get("attempt_status") == "valid"]
    if not valid:
        raise SystemExit(f"missing valid baseline replay repeat {repeat}")
    replay.append(valid[-1])
checks = {}
for metric, threshold in metrics.items():
    old = statistics.median(float(r[metric]) for r in historical)
    new = statistics.median(float(r[metric]) for r in replay)
    deviation = abs(new / old - 1) if old else 0.0
    checks[metric] = {"historical_median": old, "replay_median": new,
                      "relative_deviation": deviation, "threshold": float(threshold),
                      "pass": deviation <= float(threshold)}
payload = {"historical_rows": len(historical), "replay_repeats": 3,
           "verified": all(v["pass"] for v in checks.values()), "checks": checks}
(run_dir / "baseline_replay.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
raise SystemExit(not payload["verified"])
PY
}

correctness_capture_complete() {
  local path="$1" cfg="$2"
  [[ -f "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" "$cfg" <<'PY'
import json, sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(
    not rows
    or any(
        row.get("config") != sys.argv[2] or row.get("finish_reason") != "stop"
        for row in rows
    )
)
PY
}

run_correctness() {
  local run_dir="$1" cfg override
  mkdir -p "$run_dir/results" "$RESULTS_DIR"
  cp "$SCRIPT_DIR/check_correctness.py" "$run_dir/scripts/"
  for cfg in BASE MTP_OFF; do
    if correctness_capture_complete "$run_dir/results/correctness_${cfg}.json" "$cfg"; then
      log "skip completed correctness capture cfg=$cfg"
      continue
    fi
    set_config_toggles "$cfg"
    override="$(prepare_group_files "$run_dir" "$ABLATION_BASELINE_SEQ" "$ABLATION_BASELINE_BT")"
    restart_group_service "$run_dir" "$ABLATION_BASELINE_SEQ" "$ABLATION_BASELINE_BT" "$override"
    docker exec "$CONTAINER_NAME" python3 /bench_run/scripts/check_correctness.py capture \
      --prompt-file /bench_run/prompts/correctness.jsonl \
      --output "/bench_run/results/correctness_${cfg}.json" --config "$cfg" --max-tokens 1024
  done
  "$PYTHON_BIN" "$SCRIPT_DIR/check_correctness.py" compare \
    --reference "$run_dir/results/correctness_BASE.json" \
    --candidate "$run_dir/results/correctness_MTP_OFF.json" \
    --output "$run_dir/results/correctness.csv"
  cp "$run_dir/results/correctness.csv" "$RESULTS_DIR/correctness.csv"
}

run_ablation_warmup() {
  local run_dir="$1" in_len="$2" out_len="$3" conc="$4" prompt_file="$5" tag="$6"
  local warmup_log="$run_dir/logs/${tag}_warmup.log"
  local rc=1 attempt
  for attempt in 1 2; do
    set +e
    docker exec "$CONTAINER_NAME" vllm bench serve \
      --backend openai --base-url "$BASE_URL" --endpoint /v1/completions \
      --model "$MODEL" --tokenizer "$TOKENIZER" \
      --dataset-name custom --dataset-path "$prompt_file" \
      --num-prompts "$conc" --max-concurrency "$conc" \
      --request-rate inf --ignore-eos --output-len "$out_len" --seed 0 --temperature 0 \
      --ready-check-timeout-sec 0 --disable-tqdm \
      >>"$warmup_log" 2>&1
    rc=$?
    set -e
    [[ "$rc" -eq 0 ]] && break
    [[ "$attempt" -eq 1 ]] && log "warmup failed rc=$rc for $tag; retrying once"
  done
  [[ "$rc" -eq 0 ]] || die "warmup failed twice for $tag"
  wait_idle "$run_dir"
}

run_ablation_case() {
  local run_dir="$1" cfg="$2" in_len="$3" out_len="$4" conc="$5" repeat="$6" prompt_file="$7"
  local num_prompts=64
  local case_id="abl_${cfg}_in${in_len}_out${out_len}_c${conc}_r${repeat}"
  local attempt_id="${case_id}_a1"
  local result="/bench_run/results_raw/attempts/${attempt_id}.json"
  local host_result="$run_dir/results_raw/attempts/${attempt_id}.json"
  local before="$run_dir/metrics/${attempt_id}_before.prom"
  local samples="$run_dir/metrics/${attempt_id}_samples.prom"
  local samples_gz="${samples}.gz"
  local after="$run_dir/metrics/${attempt_id}_after.prom"
  local stdout="$run_dir/logs/${attempt_id}.stdout.log"
  local stderr="$run_dir/logs/${attempt_id}.stderr.log"
  local command_file="$run_dir/logs/${attempt_id}.command.txt"
  local inspect_before="$run_dir/logs/${attempt_id}.container_before.json"
  local inspect_after="$run_dir/logs/${attempt_id}.container_after.json"
  local case_serve_log="$run_dir/logs/${attempt_id}.serve.log"
  local attempt_summary="$run_dir/summary/attempts/${attempt_id}.json"
  local serve_log="$run_dir/logs/serve_${cfg}.log"
  local case_start
  case_start="$(date --iso-8601=seconds)"

  wait_idle "$run_dir"
  curl -fsS "$BASE_URL/metrics" >"$before"
  docker inspect "$CONTAINER_NAME" >"$inspect_before"

  printf '%q ' timeout --signal=TERM --kill-after=30s "${ABLATION_TIMEOUT_SECONDS}s" \
    docker exec "$CONTAINER_NAME" vllm bench serve \
    --backend openai --base-url "$BASE_URL" --endpoint /v1/completions \
    --model "$MODEL" --tokenizer "$TOKENIZER" \
    --dataset-name custom --dataset-path "$prompt_file" \
    --num-prompts "$num_prompts" --max-concurrency "$conc" \
    --request-rate inf --ignore-eos --output-len "$out_len" --seed 0 --temperature 0 \
    --ready-check-timeout-sec 0 --disable-tqdm \
    --request-id-prefix "${case_id}-" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 --save-result --save-detailed \
    --result-dir /bench_run/results_raw/attempts \
    --result-filename "${attempt_id}.json" \
    >"$command_file"
  printf '\n' >>"$command_file"

  start_sampler "$samples"
  set +e
  timeout --signal=TERM --kill-after=30s "${ABLATION_TIMEOUT_SECONDS}s" \
    docker exec "$CONTAINER_NAME" vllm bench serve \
    --backend openai --base-url "$BASE_URL" --endpoint /v1/completions \
    --model "$MODEL" --tokenizer "$TOKENIZER" \
    --dataset-name custom --dataset-path "$prompt_file" \
    --num-prompts "$num_prompts" --max-concurrency "$conc" \
    --request-rate inf --ignore-eos --output-len "$out_len" --seed 0 --temperature 0 \
    --ready-check-timeout-sec 0 --disable-tqdm \
    --request-id-prefix "${case_id}-" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 --save-result --save-detailed \
    --result-dir /bench_run/results_raw/attempts \
    --result-filename "${attempt_id}.json" \
    >"$stdout" 2>"$stderr"
  local bench_rc=$?
  set -e
  stop_sampler
  [[ -f "$samples" ]] || { : >"$samples"; log "WARNING: samples missing for $attempt_id"; }
  # Stop the sampler fully (pkill -P) before compressing; otherwise the metrics
  # file may still be written and `gzip -f` fails (file size changed) under set -e.
  sleep 1
  if [[ -f "$samples" ]]; then
    gzip -c "$samples" >"$samples_gz" 2>/dev/null || cp "$samples" "$samples_gz" 2>/dev/null || true
  else
    : >"$samples_gz"
  fi
  curl -fsS "$BASE_URL/metrics" >"$after"
  docker logs --timestamps --since "$case_start" "$CONTAINER_NAME" >"$case_serve_log" 2>&1 || true
  docker inspect "$CONTAINER_NAME" >"$inspect_after"

  # attempt-ablation returns: 0 valid, 11 valid+warning, 10 retryable, 20 hard.
  # None of these should abort the whole run under set -e; only hard (20) is fatal.
  set +e
  "$PYTHON_BIN" "$REPORT_SCRIPT" attempt-ablation \
    --case-id "$case_id" --attempt-id "$attempt_id" --attempt-number 1 \
    --result-file "$host_result" --metrics-before "$before" --metrics-after "$after" \
    --metrics-samples "$samples_gz" --inspect-before "$inspect_before" \
    --inspect-after "$inspect_after" --case-serve-log "$case_serve_log" \
    --serve-log-file "$serve_log" --bench-rc "$bench_rc" \
    --config-sha "$ABLATION_CONFIG_SHA256" --output "$attempt_summary"
  local attempt_rc=$?
  set -e
  if [[ "$attempt_rc" -eq 20 ]]; then
    die "hard validation failure for $attempt_id (rc=$attempt_rc)"
  elif [[ "$attempt_rc" -ne 0 ]]; then
    log "attempt-ablation rc=$attempt_rc for $attempt_id (non-fatal, continuing)"
  fi
}

main_ablation() {
  [[ "$#" -eq 1 && "$1" == init ]] || { echo "usage: $0 init" >&2; exit 2; }
  require_cmd docker docker-compose curl gzip sha256sum npu-smi lscpu free timeout
  "$PYTHON_BIN" -c 'import yaml, openpyxl' || die "missing pyyaml/openpyxl in $PYTHON_BIN"

  local run_dir="${RESUME_RUN_DIR:-}"
  if [[ -n "$run_dir" ]]; then
    [[ -d "$run_dir/summary/attempts" ]] || die "invalid RESUME_RUN_DIR: $run_dir"
  else
    run_dir="$(init_run)"
  fi
  log "run_dir=$run_dir"

  mkdir -p "$run_dir/configs" "$run_dir/profiler"
  cp "$EXEC_CONFIG" "$run_dir/configs/execution_ablation.yaml"
  cp "$EXEC_RULES" "$run_dir/configs/analysis_rules.yaml"
  cp "$EXPECTED_MANIFEST" "$run_dir/configs/expected_manifest.json"
  cp "$HISTORICAL_RESULTS" "$run_dir/configs/historical_valid_results.csv"
  EXPECTED_MANIFEST="$run_dir/configs/expected_manifest.json"

  # Copy frozen prompt datasets into run_dir so they are reachable inside the
  # container via the /bench_run mount (PROMPT_DIR itself is not mounted).
  mkdir -p "$run_dir/prompts"
  cp "$PROMPT_DIR"/prefill_heavy.jsonl "$PROMPT_DIR"/decode_heavy.jsonl \
    "$PROMPT_DIR"/correctness.jsonl "$run_dir/prompts/" 2>/dev/null \
    || die "failed to copy prompt datasets from $PROMPT_DIR into $run_dir/prompts"
  ( cd "$run_dir/prompts" && sha256sum *.jsonl ) >"$run_dir/env/prompt_sha256.txt"

  run_baseline_replay "$run_dir"

  local rep cfg w in_len out_len conc prompt_host prompt_file
  for rep in $(seq 1 "$ABLATION_REPEATS"); do
    local order
    order="$(printf '%s' "$ABLATION_ROTATION_JSON" | "$PYTHON_BIN" -c "import json,sys; print(' '.join(json.load(sys.stdin)['repeat$rep']))")"
    for cfg in $order; do
      if config_complete "$run_dir" "$cfg" "$rep"; then
        log "skip completed rep=$rep cfg=$cfg"
        continue
      fi
      set_config_toggles "$cfg"
      ablation_checkpoint "rep=$rep cfg=$cfg restart-group"
      local override
      override="$(prepare_group_files "$run_dir" "$ABLATION_BASELINE_SEQ" "$ABLATION_BASELINE_BT")"
      restart_group_service "$run_dir" "$ABLATION_BASELINE_SEQ" "$ABLATION_BASELINE_BT" "$override"
      for w in prefill_heavy decode_heavy; do
        in_len="$(workload_params "$w" prompt_tokens_approx)"
        out_len="$(workload_params "$w" output_tokens)"
        conc="$(workload_params "$w" concurrency)"
        prompt_host="$run_dir/prompts/${w}.jsonl"
        prompt_file="/bench_run/prompts/${w}.jsonl"
        [[ -f "$prompt_host" ]] || die "missing prompt file: $prompt_host (run scripts/fetch_prompts.py)"
        local tag="${cfg}_${w}"
        ablation_checkpoint "rep=$rep cfg=$cfg w=$w warmup"
        run_ablation_warmup "$run_dir" "$in_len" "$out_len" "$conc" "$prompt_file" "$tag"
        ablation_checkpoint "rep=$rep cfg=$cfg w=$w case"
        run_ablation_case "$run_dir" "$cfg" "$in_len" "$out_len" "$conc" "$rep" "$prompt_file"
        ablation_checkpoint "rep=$rep cfg=$cfg w=$w done"
      done
    done
  done
  run_correctness "$run_dir"
  "$PYTHON_BIN" "$REPORT_SCRIPT" summarize-ablation \
    --run-dir "$run_dir" --output-dir "$RESULTS_DIR"
  log "ablation complete; run_dir=$run_dir"
  echo "$run_dir"
}

main_ablation "$@"
