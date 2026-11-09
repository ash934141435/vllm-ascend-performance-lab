#!/usr/bin/env bash
# Qwen3.8-27B Ascend 910B4 x4 / 量化 A/B 性能测试：W8A8 vs BF16
# 口径来自 vllm-ascend-performance-lab：vllm bench serve + 预热 + 3 次重复取中位数 + Prometheus 采样。
# 用法: ./bench_ab_quant.sh [t0|t1|t2]      OUT=/path 可覆盖输出目录
set -uo pipefail

TIER=${1:-t1}
OUT=${OUT:-$PWD/perf_ab_$(date +%Y%m%d_%H%M)}
LOAD_MAX_RATIO=${LOAD_MAX_RATIO:-1.5}

# 双实例端点：容器名 / served-model 名 / 权重目录由环境注入（公开版不含主机名与内部路径）
: "${BF16_CONTAINER:?export BF16_CONTAINER=<bf16 serve container>}"
: "${W8A8_CONTAINER:?export W8A8_CONTAINER=<w8a8 serve container>}"
: "${BF16_NAME:?export BF16_NAME=<served model name, bf16>}"
: "${W8A8_NAME:?export W8A8_NAME=<served model name, w8a8>}"
: "${BF16_PATH:?export BF16_PATH=<bf16 weights dir>}"
: "${W8A8_PATH:?export W8A8_PATH=<w8a8 weights dir>}"
PORT_BF16=${PORT_BF16:-40000}
PORT_W8A8=${PORT_W8A8:-40001}

declare -A CT PORT NAME DIR
CT[bf16]=$BF16_CONTAINER
CT[w8a8]=$W8A8_CONTAINER
PORT[bf16]=$PORT_BF16
PORT[w8a8]=$PORT_W8A8
NAME[bf16]=$BF16_NAME
NAME[w8a8]=$W8A8_NAME
DIR[bf16]=$BF16_PATH
DIR[w8a8]=$W8A8_PATH

case "$TIER" in
  t0) IN_LIST=(1024);      OUT_LIST=(64);   C_LIST=(1);       REPS=1 ;;
  t1) IN_LIST=(1024 4096); OUT_LIST=(512);  C_LIST=(1 8 32);  REPS=3 ;;
  t2) IN_LIST=(1024 4096); OUT_LIST=(1024); C_LIST=(16 24);   REPS=3 ;;
  *)  echo "usage: $0 [t0|t1|t2]" >&2; exit 2 ;;
esac

mkdir -p "$OUT"/{results_raw,metrics,logs,env,npu}
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$OUT/progress.log"; }

# ---------- 环境快照 ----------
{
  echo "tier=$TIER reps=$REPS in=${IN_LIST[*]} out=${OUT_LIST[*]} c=${C_LIST[*]}"
  echo "start=$(date -Is) host=$(hostname) loadavg=$(cat /proc/loadavg)"
  head -2 /etc/os-release
  python3 -V
  for q in bf16 w8a8; do
    echo "--- $q ${CT[$q]} :${PORT[$q]} ${NAME[$q]}"
    docker inspect -f '{{.State.Status}} {{json .Config.Env}} {{json .Config.Cmd}}' "${CT[$q]}"
    docker exec "${CT[$q]}" vllm --version 2>/dev/null | tail -1
  done
  curl -fsS "http://127.0.0.1:${PORT[bf16]}/v1/models"; echo
  curl -fsS "http://127.0.0.1:${PORT[w8a8]}/v1/models"; echo
} > "$OUT/env/env.txt" 2>&1
npu-smi info > "$OUT/npu/before.txt" 2>&1 || true
LOAD_BASE=$(cut -d' ' -f1 /proc/loadavg)
log "OUT=$OUT load_base=$LOAD_BASE"

# ---------- 门控 ----------
peer_idle() {  # 对端服务必须完全空闲，否则本轮串扰
  local busy
  busy=$(curl -fsS --max-time 3 "http://127.0.0.1:${PORT[$1]}/metrics" 2>/dev/null \
    | awk '/^vllm:num_requests_(running|waiting)\{/{s+=$2} END{print s+0}')
  [[ "${busy:-1}" == "0" ]]
}
load_ok() {
  awk -v b="$LOAD_BASE" -v r="$LOAD_MAX_RATIO" -v c="$(cut -d' ' -f1 /proc/loadavg)" \
    'BEGIN{exit !(c <= b*r)}'
}
wait_gate() {  # $1=peer quant $2=case id；等对端空闲且 load 回落，最多 300s
  local peer=$1 cid=$2 t=0
  while (( t < 300 )); do
    if peer_idle "$peer" && load_ok; then return 0; fi
    log "WAIT $cid peer-or-load not ready (${t}s) load=$(cut -d' ' -f1 /proc/loadavg)"
    sleep 15; t=$((t + 15))
  done
  return 1
}

# ---------- 压测执行 ----------
sampler() {  # $1=输出文件 $2=端口，1s 抓一次 /metrics
  : > "$1"
  while true; do
    curl -fsS --max-time 2 "http://127.0.0.1:$2/metrics" >> "$1" 2>/dev/null || true
    printf '# ---\n' >> "$1"
    sleep 1
  done
}

run_bench() {  # $1=quant $2=case_id $3=num_prompts $4=in_len $5=out_len $6=concurrency $7=save
  local q=$1 cid=$2 np=$3 inlen=$4 outlen=$5 conc=$6 save=$7
  local -a extra=()
  [[ "$save" == 1 ]] && extra=(--save-result --save-detailed --result-dir /tmp --result-filename "$cid.json")
  docker exec "${CT[$q]}" vllm bench serve \
    --backend openai --base-url "http://127.0.0.1:${PORT[$q]}" --endpoint /v1/completions \
    --model "${NAME[$q]}" --tokenizer "${DIR[$q]}" \
    --dataset-name random --random-input-len "$inlen" --random-output-len "$outlen" \
    --random-range-ratio 0 \
    --num-prompts "$np" --max-concurrency "$conc" --request-rate inf --ignore-eos \
    --seed 0 --temperature 0 --ready-check-timeout-sec 0 --disable-tqdm \
    --request-id-prefix "${cid}-" --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 "${extra[@]}"
}

run_case() {  # $1=quant $2=in_len $3=out_len $4=concurrency $5=rep
  local q=$1 inlen=$2 outlen=$3 conc=$4 rep=$5
  local peer; [[ "$q" == bf16 ]] && peer=w8a8 || peer=bf16
  local cid=${q}_in${inlen}_out${outlen}_c${conc}_r${rep}
  local json="$OUT/results_raw/$cid.json"
  local before="$OUT/metrics/${cid}_before.prom" after="$OUT/metrics/${cid}_after.prom"

  wait_gate "$peer" "$cid" || { log "INVALID $cid gate-timeout"; return; }

  local attempt rc failed spid
  for attempt in 1 2; do
    rm -f "$json"
    curl -fsS --max-time 5 "http://127.0.0.1:${PORT[$q]}/metrics" > "$before" 2>/dev/null || true
    sampler "$OUT/metrics/${cid}_samples.prom" "${PORT[$q]}" &
    spid=$!
    run_bench "$q" "$cid" $((conc * 5)) "$inlen" "$outlen" "$conc" 1 > "$OUT/logs/$cid.log" 2>&1
    rc=$?
    kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
    curl -fsS --max-time 5 "http://127.0.0.1:${PORT[$q]}/metrics" > "$after" 2>/dev/null || true
    docker cp "${CT[$q]}:/tmp/$cid.json" "$json" 2>/dev/null || true
    failed=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["failed"])' "$json" 2>/dev/null || echo NA)
    if [[ $rc -eq 0 && "$failed" == "0" ]]; then
      mv -f "$OUT/logs/$cid.log" "$OUT/logs/$cid.a${attempt}.log"
      if [ "$attempt" = 2 ]; then rm -f "$OUT/logs/$cid.a1.log"; fi
      log "OK $cid (attempt $attempt)"
      return
    fi
    log "RETRY $cid rc=$rc failed=$failed (attempt $attempt)"
  done
  log "INVALID $cid"
}

# ---------- 预热：每个 cell 一次，两模型各一轮 ----------
for q in bf16 w8a8; do
  for inlen in "${IN_LIST[@]}"; do
    for outlen in "${OUT_LIST[@]}"; do
      for conc in "${C_LIST[@]}"; do
        wid=${q}_in${inlen}_out${outlen}_c${conc}_warmup
        wn=$(( conc < 4 ? conc : 4 ))
        log "warmup $wid"
        run_bench "$q" "$wid" "$wn" "$inlen" "$outlen" "$conc" 0 > "$OUT/logs/$wid.log" 2>&1 \
          || log "WARN warmup $wid rc=$?"
      done
    done
  done
done

# ---------- 主循环：按重复轮次交错 A/B（ABBA），抵消温度/负载漂移 ----------
for rep in $(seq 1 "$REPS"); do
  if (( rep % 2 )); then ORDER=(bf16 w8a8); else ORDER=(w8a8 bf16); fi
  log "=== repeat $rep order=${ORDER[*]}"
  for inlen in "${IN_LIST[@]}"; do
    for outlen in "${OUT_LIST[@]}"; do
      for conc in "${C_LIST[@]}"; do
        for q in "${ORDER[@]}"; do
          run_case "$q" "$inlen" "$outlen" "$conc" "$rep"
        done
      done
    done
  done
done

npu-smi info > "$OUT/npu/after.txt" 2>&1 || true
log "=== all runs done, summarizing"
python3 "$(dirname "$0")/summarize_ab.py" "$OUT" && log "summary written"
