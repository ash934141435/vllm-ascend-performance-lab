#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="${SERVICE_DIR:-/opt/vllm-service}"
COMPOSE_FILE="${COMPOSE_FILE:-$SERVICE_DIR/docker-compose.yaml}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm}"
COMPOSE_SERVICE="${COMPOSE_SERVICE:-vllm}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8010}"
TOKENIZER="${TOKENIZER:-/model}"
PYTHON_BIN="${BENCH_PYTHON:-$SERVICE_DIR/.bench-venv/bin/python3}"
REPORT_SCRIPT="${REPORT_SCRIPT:-$SCRIPT_DIR/bench_report.py}"
ORIGINAL_START="${ORIGINAL_START:-$SERVICE_DIR/start-vllm.sh}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-$SCRIPT_DIR/../configs/experiment.yaml}"

SAMPLER_PID=""
SERVE_LOG_PID=""
ATTEMPT_RC=20
CASE_RC=20

usage() {
  cat >&2 <<'EOF'
usage:
  bench_matrix.sh init
  bench_matrix.sh group RUN_DIR MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
  bench_matrix.sh summarize RUN_DIR
EOF
}

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

contains_word() {
  local needle="$1"
  local values="$2"
  [[ " $values " == *" $needle "* ]]
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

load_experiment_config() {
  [[ -x "$PYTHON_BIN" ]] || die "missing benchmark Python: $PYTHON_BIN"
  [[ -f "$EXPERIMENT_CONFIG" ]] || die "missing experiment config: $EXPERIMENT_CONFIG"
  export EXPERIMENT_CONFIG
  eval "$("$PYTHON_BIN" "$REPORT_SCRIPT" config-shell --config "$EXPERIMENT_CONFIG")"
  CONCURRENCIES="${BENCH_CONCURRENCIES:-$DEFAULT_CONCURRENCIES}"
}

cleanup() {
  if [[ -n "$SAMPLER_PID" ]]; then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=""
  fi
  if [[ -n "$SERVE_LOG_PID" ]]; then
    kill "$SERVE_LOG_PID" 2>/dev/null || true
    wait "$SERVE_LOG_PID" 2>/dev/null || true
    SERVE_LOG_PID=""
  fi
}

trap cleanup EXIT INT TERM

collect_command() {
  local output="$1"
  shift
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } >"$output" 2>&1 || true
}

init_run() {
  load_experiment_config
  require_cmd docker
  require_cmd docker-compose
  require_cmd curl
  require_cmd gzip
  require_cmd sha256sum
  require_cmd npu-smi
  require_cmd lscpu
  require_cmd free
  "$PYTHON_BIN" -c 'import openpyxl, yaml' || die "openpyxl or PyYAML missing in $PYTHON_BIN environment"
  [[ -f "$ORIGINAL_START" ]] || die "missing $ORIGINAL_START"
  [[ -f "$COMPOSE_FILE" ]] || die "missing $COMPOSE_FILE"
  docker inspect "$CONTAINER_NAME" >/dev/null || die "missing container $CONTAINER_NAME"

  local run_dir="$SERVICE_DIR/bench_runs/$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$run_dir"/{scripts,logs,results_raw/attempts,metrics,summary/attempts,env}

  cp -p "$ORIGINAL_START" "$run_dir/scripts/start-vllm.original.sh"
  cp -p "$ORIGINAL_START" "$run_dir/env/start-vllm.original.sh"
  sha256sum "$ORIGINAL_START" >"$run_dir/env/start-vllm.original.sha256"
  cp -p "$SCRIPT_DIR/bench_matrix.sh" "$run_dir/scripts/"
  cp -p "$REPORT_SCRIPT" "$run_dir/scripts/"
  cp -p "$EXPERIMENT_CONFIG" "$run_dir/env/experiment.yaml"
  sha256sum "$EXPERIMENT_CONFIG" >"$run_dir/env/experiment.sha256"

  date -Iseconds >"$run_dir/env/collected_at.txt"
  hostname >"$run_dir/env/hostname.txt"
  cp /etc/os-release "$run_dir/env/os-release.txt"
  uname -a >"$run_dir/env/kernel.txt"
  collect_command "$run_dir/env/cpu.txt" lscpu
  collect_command "$run_dir/env/memory.txt" free -h
  collect_command "$run_dir/env/disk.txt" df -h "$SERVICE_DIR"
  collect_command "$run_dir/env/npu-smi-info.txt" npu-smi info
  collect_command "$run_dir/env/npu-smi-info-m.txt" npu-smi info -m
  collect_command "$run_dir/env/docker-version.txt" docker version
  docker inspect "$CONTAINER_NAME" >"$run_dir/env/docker-inspect.json"
  docker image inspect "$(docker inspect "$CONTAINER_NAME" --format '{{.Image}}')" \
    >"$run_dir/env/docker-image-inspect.json"
  docker-compose -f "$COMPOSE_FILE" config >"$run_dir/env/docker-compose-config.yaml"
  {
    docker exec "$CONTAINER_NAME" python3 --version
    docker exec "$CONTAINER_NAME" vllm --version
    docker exec "$CONTAINER_NAME" pip show vllm
    docker exec "$CONTAINER_NAME" pip show vllm-ascend
    docker exec "$CONTAINER_NAME" bash -lc \
      'cat /usr/local/Ascend/driver/version.info 2>/dev/null; \
       cat /etc/ascend_install.info 2>/dev/null; \
       find /usr/local/Ascend -maxdepth 3 -name version.info -print -exec cat {} \; 2>/dev/null'
  } >"$run_dir/env/versions-container.txt" 2>&1
  curl -fsS "$BASE_URL/v1/models" >"$run_dir/env/models.json"

  printf '%s\n' "$run_dir"
}

prepare_group_files() {
  local run_dir="$1"
  local seq="$2"
  local bt="$3"
  local start_script="$run_dir/scripts/start_seq${seq}_bt${bt}.sh"
  local diff_file="$run_dir/scripts/start_seq${seq}_bt${bt}.diff"
  local override="$run_dir/scripts/compose_seq${seq}_bt${bt}.override.yaml"

  "$PYTHON_BIN" "$REPORT_SCRIPT" render-start \
    --source "$run_dir/scripts/start-vllm.original.sh" \
    --dest "$start_script" \
    --diff-out "$diff_file" \
    --seq "$seq" \
    --bt "$bt"

  cat >"$override" <<EOF
services:
  $COMPOSE_SERVICE:
    volumes:
      - $start_script:/workspace/start.sh:ro
      - $run_dir:/bench_run
EOF

  docker-compose -f "$COMPOSE_FILE" -f "$override" config \
    >"$run_dir/scripts/compose_seq${seq}_bt${bt}.resolved.yaml"
  grep -F "$start_script" "$run_dir/scripts/compose_seq${seq}_bt${bt}.resolved.yaml" >/dev/null \
    || die "resolved compose does not use generated start script"
  printf '%s\n' "$override"
}

wait_ready() {
  local deadline=$((SECONDS + 1200))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 "$BASE_URL/v1/models" | MODEL_ID="$MODEL" "$PYTHON_BIN" -c \
      'import json,os,sys; d=json.load(sys.stdin); target=os.environ["MODEL_ID"]; raise SystemExit(0 if any(x.get("id")==target for x in d.get("data",[])) else 1)'
    then
      return 0
    fi
    sleep 5
  done
  return 1
}

get_vllm_args() {
  docker exec "$CONTAINER_NAME" sh -lc \
    "ps -eo args 2>/dev/null | grep '[v]llm serve' | head -1"
}

verify_running_config() {
  local seq="$1"
  local bt="$2"
  local args
  args="$(get_vllm_args)"
  grep -F -- "--max-num-seqs $seq" <<<"$args" >/dev/null || die "running max-num-seqs mismatch"
  grep -F -- "--max-num-batched-tokens $bt" <<<"$args" >/dev/null \
    || die "running max-num-batched-tokens mismatch"
  local source
  source="$(docker inspect "$CONTAINER_NAME" \
    --format '{{range .Mounts}}{{if eq .Destination "/workspace/start.sh"}}{{.Source}}{{end}}{{end}}')"
  [[ "$source" == *"start_seq${seq}_bt${bt}.sh" ]] || die "container start-script mount mismatch: $source"
}

restart_group_service() {
  local run_dir="$1"
  local seq="$2"
  local bt="$3"
  local override="$4"
  local serve_log="$run_dir/logs/serve_seq${seq}_bt${bt}.log"

  docker-compose -f "$COMPOSE_FILE" -f "$override" stop "$COMPOSE_SERVICE"
  docker-compose -f "$COMPOSE_FILE" -f "$override" up -d --force-recreate "$COMPOSE_SERVICE"
  local rc_before
  local started_before
  rc_before="$(docker inspect "$CONTAINER_NAME" --format '{{.RestartCount}}')"
  started_before="$(docker inspect "$CONTAINER_NAME" --format '{{.State.StartedAt}}')"
  printf '\n### service_start ts=%s seq=%s bt=%s\n' "$(date -Iseconds)" "$seq" "$bt" >>"$serve_log"
  docker logs --timestamps -f "$CONTAINER_NAME" >>"$serve_log" 2>&1 &
  SERVE_LOG_PID=$!

  wait_ready || die "service not ready after 20 minutes"
  verify_running_config "$seq" "$bt"
  {
    printf '### seq=%s bt=%s ts=%s\n' "$seq" "$bt" "$(date -Iseconds)"
    get_vllm_args
  } >>"$run_dir/env/serve_commands.txt"

  local rc_after
  local started_after
  rc_after="$(docker inspect "$CONTAINER_NAME" --format '{{.RestartCount}}')"
  started_after="$(docker inspect "$CONTAINER_NAME" --format '{{.State.StartedAt}}')"
  [[ "$rc_after" == "$rc_before" ]] || die "container restarted during model startup"
  [[ "$started_after" == "$started_before" ]] || die "container StartedAt changed during model startup"
  [[ "$(docker inspect "$CONTAINER_NAME" --format '{{.State.OOMKilled}}')" == "false" ]] \
    || die "container was OOM-killed during model startup"
  docker inspect "$CONTAINER_NAME" \
    --format 'Status={{.State.Status}} RestartCount={{.RestartCount}} OOMKilled={{.State.OOMKilled}} StartedAt={{.State.StartedAt}}'
}

wait_idle() {
  local stable=0
  local deadline=$((SECONDS + 300))
  while (( stable < 3 && SECONDS < deadline )); do
    local values
    values="$(curl -fsS --max-time 5 "$BASE_URL/metrics" |
      awk '
        /^vllm:num_requests_running[{ ]/ {running += $NF}
        /^vllm:num_requests_waiting[{ ]/ {waiting += $NF}
        END {printf "%.0f %.0f\n", running, waiting}
      ')" || values="unknown"
    if [[ "$values" == "0 0" ]]; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    sleep 1
  done
  (( stable >= 3 )) || die "wait_idle timed out after 5 minutes"
}

run_warmup() {
  local run_dir="$1"
  local seq="$2"
  local bt="$3"
  local input_len="$4"
  local concurrency="$5"
  local tag="seq${seq}_bt${bt}_in${input_len}_out${OUTPUT_LEN}_c${concurrency}_warmup"
  local warmup_log="$run_dir/logs/${tag}.log"
  local rc=1
  local attempt

  for attempt in 1 2; do
    set +e
    docker exec "$CONTAINER_NAME" vllm bench serve \
      --backend openai \
      --base-url "$BASE_URL" \
      --endpoint /v1/completions \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --dataset-name random \
      --random-input-len "$input_len" \
      --random-output-len "$OUTPUT_LEN" \
      --random-range-ratio 0 \
      --num-prompts "$concurrency" \
      --max-concurrency "$concurrency" \
      --request-rate inf \
      --ignore-eos \
      --seed 0 \
      --temperature 0 \
      --ready-check-timeout-sec 0 \
      --disable-tqdm \
      >>"$warmup_log" 2>&1
    rc=$?
    set -e
    [[ "$rc" -eq 0 ]] && break
    [[ "$attempt" -eq 1 ]] && log "warmup failed rc=$rc for $tag; retrying once"
  done

  [[ "$rc" -eq 0 ]] || die "warmup failed twice for $tag"
  wait_idle
}

start_sampler() {
  local output="$1"
  (
    while true; do
      printf '# sample_time_unix_ms %s\n' "$(date +%s%3N)" >>"$output"
      curl -fsS --max-time 5 "$BASE_URL/metrics" >>"$output" || printf '# scrape_error\n' >>"$output"
      printf '\n' >>"$output"
      sleep 1
    done
  ) &
  SAMPLER_PID=$!
}

stop_sampler() {
  if [[ -n "$SAMPLER_PID" ]]; then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=""
  fi
}

run_attempt() {
  local run_dir="$1"
  local seq="$2"
  local bt="$3"
  local input_len="$4"
  local concurrency="$5"
  local repeat="$6"
  local attempt_number="$7"

  local case_id="seq${seq}_bt${bt}_in${input_len}_out${OUTPUT_LEN}_c${concurrency}_r${repeat}"
  local attempt_id="${case_id}_a${attempt_number}"
  local num_prompts=$((concurrency * 5))
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
  local serve_log="$run_dir/logs/serve_seq${seq}_bt${bt}.log"
  local case_start
  case_start="$(date --iso-8601=seconds)"

  wait_idle
  curl -fsS "$BASE_URL/metrics" >"$before"
  docker inspect "$CONTAINER_NAME" >"$inspect_before"

  printf '%q ' docker exec "$CONTAINER_NAME" vllm bench serve \
    --backend openai --base-url "$BASE_URL" --endpoint /v1/completions \
    --model "$MODEL" --tokenizer "$TOKENIZER" --dataset-name random \
    --random-input-len "$input_len" --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0 --num-prompts "$num_prompts" \
    --max-concurrency "$concurrency" --request-rate inf --ignore-eos \
    --seed 0 --temperature 0 --ready-check-timeout-sec 0 --disable-tqdm \
    --request-id-prefix "${case_id}-" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 --save-result --save-detailed \
    --result-dir /bench_run/results_raw/attempts \
    --result-filename "${attempt_id}.json" \
    >"$command_file"
  printf '\n' >>"$command_file"

  start_sampler "$samples"
  set +e
  docker exec "$CONTAINER_NAME" vllm bench serve \
    --backend openai \
    --base-url "$BASE_URL" \
    --endpoint /v1/completions \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0 \
    --num-prompts "$num_prompts" \
    --max-concurrency "$concurrency" \
    --request-rate inf \
    --ignore-eos \
    --seed 0 \
    --temperature 0 \
    --ready-check-timeout-sec 0 \
    --disable-tqdm \
    --request-id-prefix "${case_id}-" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --result-dir /bench_run/results_raw/attempts \
    --result-filename "${attempt_id}.json" \
    >"$stdout" 2>"$stderr"
  local bench_rc=$?
  set -e
  stop_sampler
  if [[ ! -f "$samples" ]]; then
    log "WARNING: samples file missing for $attempt_id"
    : >"$samples"
  fi
  gzip -f "$samples"
  curl -fsS "$BASE_URL/metrics" >"$after"
  docker logs --timestamps --since "$case_start" "$CONTAINER_NAME" >"$case_serve_log" 2>&1 || true
  docker inspect "$CONTAINER_NAME" >"$inspect_after"

  if [[ ! -f "$host_result" ]]; then
    log "missing result JSON for $attempt_id"
    "$PYTHON_BIN" - "$attempt_summary" "$case_id" "$attempt_id" \
      "$attempt_number" "$serve_log" "$samples_gz" "$seq" "$bt" \
      "$input_len" "$OUTPUT_LEN" "$concurrency" "$repeat" \
      "$EXPERIMENT_CONFIG_SHA256" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    case_id,
    attempt_id,
    attempt_number,
    serve_log,
    metrics_file,
    seq,
    bt,
    input_len,
    output_len,
    concurrency,
    repeat_id,
    config_sha256,
) = sys.argv[1:]
payload = {
    "experiment_config_sha256": config_sha256,
    "case_id": case_id,
    "attempt_id": attempt_id,
    "attempt_number": int(attempt_number),
    "max_num_seqs": int(seq),
    "max_num_batched_tokens": int(bt),
    "input_len": int(input_len),
    "output_len": int(output_len),
    "concurrency": int(concurrency),
    "repeat_id": int(repeat_id),
    "attempt_status": "hard",
    "anomaly_reasons": ["missing_result_json"],
    "result_file": "",
    "metrics_file": metrics_file,
    "serve_log_file": serve_log,
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n")
PY
    ATTEMPT_RC=20
    return 0
  fi

  set +e
  "$PYTHON_BIN" "$REPORT_SCRIPT" attempt \
    --case-id "$case_id" \
    --attempt-id "$attempt_id" \
    --attempt-number "$attempt_number" \
    --run-dir "$run_dir" \
    --seq "$seq" \
    --bt "$bt" \
    --result-file "$host_result" \
    --metrics-before "$before" \
    --metrics-after "$after" \
    --metrics-samples "$samples_gz" \
    --inspect-before "$inspect_before" \
    --inspect-after "$inspect_after" \
    --case-serve-log "$case_serve_log" \
    --serve-log-file "$serve_log" \
    --bench-rc "$bench_rc" \
    --output "$attempt_summary"
  local validation_rc=$?
  set -e

  if [[ "$validation_rc" -eq 0 || "$validation_rc" -eq 11 ]]; then
    cp "$host_result" "$run_dir/results_raw/${case_id}.json"
  fi
  ATTEMPT_RC="$validation_rc"
  return 0
}

case_is_valid() {
  local run_dir="$1"
  local case_id="$2"
  [[ -f "$run_dir/results_raw/${case_id}.json" ]] || return 1
  "$PYTHON_BIN" - "$run_dir/summary/attempts" "$case_id" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
case_id = sys.argv[2]
for path in root.glob(f"{case_id}_a*.json"):
    if json.loads(path.read_text()).get("attempt_status") == "valid":
        raise SystemExit(0)
raise SystemExit(1)
PY
}

next_attempt_number() {
  local run_dir="$1"
  local case_id="$2"
  local highest=0
  local path number
  shopt -s nullglob
  for path in "$run_dir/results_raw/attempts/${case_id}_a"*.json; do
    number="${path##*_a}"
    number="${number%.json}"
    if (( number > highest )); then
      highest="$number"
    fi
  done
  shopt -u nullglob
  printf '%s\n' "$((highest + 1))"
}

run_case() {
  local run_dir="$1"
  local seq="$2"
  local bt="$3"
  local input_len="$4"
  local concurrency="$5"
  local repeat="$6"
  local case_id="seq${seq}_bt${bt}_in${input_len}_out${OUTPUT_LEN}_c${concurrency}_r${repeat}"

  if case_is_valid "$run_dir" "$case_id"; then
    log "skip valid $case_id"
    CASE_RC=0
    return 0
  fi

  local first_attempt retry_index attempt warning_hits
  first_attempt="$(next_attempt_number "$run_dir" "$case_id")"
  warning_hits=0
  for retry_index in 0 1 2; do
    attempt=$((first_attempt + retry_index))
    log "run $case_id attempt=$attempt"
    run_attempt "$run_dir" "$seq" "$bt" "$input_len" "$concurrency" "$repeat" "$attempt"
    local rc="$ATTEMPT_RC"
    case "$rc" in
      0)
        CASE_RC=0
        return 0
        ;;
      10)
        if [[ "$retry_index" -eq 0 ]]; then
          log "retryable anomaly in $case_id; immediate hot rerun"
          continue
        fi
        log "persistent retryable anomaly in $case_id"
        CASE_RC=20
        return 0
        ;;
      11)
        warning_hits=$((warning_hits + 1))
        if [[ "$warning_hits" -eq 1 ]]; then
          log "performance or capacity warning in $case_id; immediate hot rerun"
          continue
        fi
        if [[ "$warning_hits" -eq 2 ]]; then
          log "repeated warning in $case_id; one confirmation attempt"
          continue
        fi
        log "warning persisted in $case_id; retain and continue group"
        CASE_RC=0
        return 0
        ;;
      *)
        CASE_RC=20
        return 0
        ;;
    esac
  done
}

run_group() {
  load_experiment_config
  local run_dir="$1"
  local seq="$2"
  local bt="$3"
  [[ -d "$run_dir" ]] || die "missing run directory: $run_dir"
  contains_word "$seq" "$VALID_SEQS" || die "invalid max-num-seqs: $seq"
  contains_word "$bt" "$VALID_BT" || die "invalid max-num-batched-tokens: $bt"

  local override
  override="$(prepare_group_files "$run_dir" "$seq" "$bt")"
  restart_group_service "$run_dir" "$seq" "$bt" "$override"

  local input_len concurrency repeat
  for input_len in $INPUT_LENGTHS; do
    for concurrency in $CONCURRENCIES; do
      run_warmup "$run_dir" "$seq" "$bt" "$input_len" "$concurrency"
      for repeat in $REPEAT_IDS; do
        run_case "$run_dir" "$seq" "$bt" "$input_len" "$concurrency" "$repeat"
        if [[ "$CASE_RC" -ne 0 ]]; then
          "$PYTHON_BIN" "$REPORT_SCRIPT" group-review --run-dir "$run_dir" --seq "$seq" --bt "$bt" || true
          return 20
        fi
      done
      "$PYTHON_BIN" "$REPORT_SCRIPT" group-review --run-dir "$run_dir" --seq "$seq" --bt "$bt"
    done
  done

  "$PYTHON_BIN" "$REPORT_SCRIPT" group-review \
    --run-dir "$run_dir" --seq "$seq" --bt "$bt" --require-complete
  log "group complete; stop here for gpt-5.5/xhigh anomaly review"
}

summarize_run() {
  load_experiment_config
  local run_dir="$1"
  [[ -d "$run_dir" ]] || die "missing run directory: $run_dir"
  "$PYTHON_BIN" "$REPORT_SCRIPT" summarize --run-dir "$run_dir"
}

main() {
  local command="${1:-}"
  case "$command" in
    init)
      [[ "$#" -eq 1 ]] || { usage; exit 2; }
      init_run
      ;;
    group)
      [[ "$#" -eq 4 ]] || { usage; exit 2; }
      run_group "$2" "$3" "$4"
      ;;
    summarize)
      [[ "$#" -eq 2 ]] || { usage; exit 2; }
      summarize_run "$2"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
