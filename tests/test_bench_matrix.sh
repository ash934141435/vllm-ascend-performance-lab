#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash -n "$ROOT/scripts/bench_matrix.sh"
bash -n "$ROOT/scripts/bench_execution_ablation.sh"

source "$ROOT/scripts/bench_matrix.sh"
get_vllm_args() { printf '%s\n' '--max-num-seqs 32 --max-num-batched-tokens 16384 --async-scheduling --speculative_config mtp --compilation-config cudagraph_mode=NONE'; }
EAGER=1 verify_ablation
is_container_process "$$" "$$"
if is_container_process 99999999 "$$"; then
  exit 1
fi
sampler_output="$(mktemp)"
BASE_URL=http://127.0.0.1:1 start_sampler "$sampler_output"
sleep 0.1
stop_sampler
test -z "$SAMPLER_PID"
rm -f "$sampler_output"
idle_dir="$(mktemp -d)"
mkdir -p "$idle_dir/env"
printf '%s\n' '{"thresholds":{"load1":1000000,"cpu_busy_pct":1000000}}' >"$idle_dir/env/host-idle.json"
PYTHON_BIN=python3 host_idle_guard "$idle_dir" verify
printf '%s\n' '{"thresholds":{"load1":-1,"cpu_busy_pct":-1}}' >"$idle_dir/env/host-idle.json"
if (PYTHON_BIN=python3 host_idle_guard "$idle_dir" verify) 2>/dev/null; then
  exit 1
fi
rm -rf "$idle_dir"

host_idle_checks=0
host_idle_guard() {
  host_idle_checks=$((host_idle_checks + 1))
  (( host_idle_checks >= 3 ))
}
HOST_IDLE_WAIT_SECONDS=2 HOST_IDLE_POLL_SECONDS=0 wait_host_idle /tmp
test "$host_idle_checks" -eq 3

set +e
output="$("$ROOT/scripts/bench_matrix.sh" 2>&1)"
rc=$?
set -e

test "$rc" -eq 2
grep -F "usage:" <<<"$output"
grep -F "bench_matrix.sh init" <<<"$output"
grep -F "bench_matrix.sh group RUN_DIR MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS" <<<"$output"
grep -F "bench_matrix.sh summarize RUN_DIR" <<<"$output"

script="$ROOT/scripts/bench_matrix.sh"

grep -F 'mkdir -p "$run_dir"/{scripts,configs,logs,results_raw/attempts,metrics,summary/attempts,env,profiler}' "$script"
grep -F 'cp -p "$ORIGINAL_START" "$run_dir/scripts/start-vllm.original.sh"' "$script"
grep -F 'cp -p "$ORIGINAL_START" "$run_dir/env/start-vllm.original.sh"' "$script"
grep -F 'docker exec "$CONTAINER_NAME" ps -eo args' "$script"
grep -F "wait_idle timed out after 5 minutes" "$script"
grep -F "foreign process on target NPU" "$script"
grep -F "warmup failed twice" "$script"
grep -F "WARNING: samples file missing" "$script"
grep -F 'if [[ "$validation_rc" -eq 0 || "$validation_rc" -eq 11 ]]; then' "$script"
grep -F "11)" "$script"
grep -F "confirmation attempt" "$script"
grep -F 'EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-$SCRIPT_DIR/../configs/experiment.yaml}"' "$script"
grep -F 'config-shell --config "$EXPERIMENT_CONFIG"' "$script"
grep -F 'CONCURRENCIES="${BENCH_CONCURRENCIES:-$DEFAULT_CONCURRENCIES}"' "$script"
grep -F 'MODEL_ID="$MODEL"' "$script"
grep -F 'for repeat in $REPEAT_IDS' "$script"

if grep -Fq 'Qwen3.6-27B' "$script"; then
  exit 1
fi

if grep -Fq "docker top" "$script"; then
  exit 1
fi
if grep -Fq "restore)" "$script"; then
  exit 1
fi

printf 'bench_matrix shell checks passed\n'
