#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash -n "$ROOT/scripts/bench_matrix.sh"

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

grep -F 'mkdir -p "$run_dir"/{scripts,logs,results_raw/attempts,metrics,summary/attempts,env}' "$script"
grep -F 'cp -p "$ORIGINAL_START" "$run_dir/scripts/start-vllm.original.sh"' "$script"
grep -F 'cp -p "$ORIGINAL_START" "$run_dir/env/start-vllm.original.sh"' "$script"
grep -F "ps -eo args" "$script"
grep -F "wait_idle timed out after 5 minutes" "$script"
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
