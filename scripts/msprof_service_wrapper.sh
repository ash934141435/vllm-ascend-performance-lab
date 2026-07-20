#!/usr/bin/env bash
set -euo pipefail

PROFILE_CONFIG="$(tr -d '[:space:]' </bench_run/profiler/current_profile_config)"
case "$PROFILE_CONFIG" in
  BASE) application=/bench_run/scripts/start_seq32_bt16384.sh ;;
  EAGER) application=/bench_run/scripts/start_seq32_bt16384_eager.sh ;;
  *) echo "unsupported PROFILE_CONFIG: $PROFILE_CONFIG" >&2; exit 2 ;;
esac

output="/bench_run/profiler/raw_${PROFILE_CONFIG}"
mkdir -p "$output"

exec msprof \
  --application="$application" \
  --output="$output" \
  --delay="${PROFILE_DELAY_SECONDS:-600}" \
  --duration="${PROFILE_DURATION_SECONDS:-180}" \
  --ascendcl=on \
  --runtime-api=on \
  --task-time=on \
  --hccl=on \
  --ai-core=on \
  --aic-metrics=PipeUtilization
