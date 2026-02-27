#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${ROOT_DIR}/target/release/examples/bench_1gb"
OUT_DIR="${ROOT_DIR}/bench_results"

SIZE_MIB=4096
RUNS=5
ITERS=1
WARMUPS=0
CHUNK_MIB=4
GPU_SUBCHUNK_KIB=512
TOKEN_FINALIZE_SEGMENT_SIZE=4096
GPU_SLOTS=6
GPU_BATCH_CHUNKS=""
GPU_SUBMIT_CHUNKS=6
STREAM_PIPELINE_DEPTH=2
STREAM_BATCH_CHUNKS=32
STREAM_MAX_INFLIGHT_CHUNKS=256
STREAM_MAX_INFLIGHT_MIB=0
SCHEDULER="legacy"
GPU_FRACTION=1.0
GPU_TAIL_STOP_RATIO=1.0
MODES="ratio"
BUILD=1
KEEP_PROFILE_VARS=0
PROFILE_TIMING=0
PROFILE_TIMING_DETAIL=0
PROFILE_TIMING_DEEP=0

usage() {
  cat <<'EOF'
Usage:
  ./bench.sh [options]

Options:
  --size-mib <N>           Input size in MiB (default: 4096)
  --runs <N>               Number of process-restart runs per mode (default: 5)
  --iters <N>              Iterations per process (default: 1)
  --warmups <N>            Warmup iterations per process (default: 0)
  --chunk-mib <N>          Chunk size in MiB (default: 4)
  --gpu-subchunk-kib <N>   GPU subchunk size in KiB (default: 512)
  --token-finalize-segment-size <N>  Token finalize segment size (default: 4096)
  --gpu-slots <N>          GPU slot/batch count (default: 6)
  --gpu-batch-chunks <N>   GPU dequeue batch size (default: same as --gpu-slots)
  --gpu-submit-chunks <N>  GPU submit group size (default: 6)
  --stream-pipeline-depth <N>  Stream prepare pipeline depth (default: 2)
  --stream-batch-chunks <N>    Stream batch chunk count (default: 32, 0: batch off continuous)
  --stream-max-inflight-chunks <N>  Inflight chunk cap in continuous mode (default: 256, 0: unlimited)
  --stream-max-inflight-mib <N>  Inflight raw MiB cap in continuous mode (default: 0, disabled)
  --scheduler <S>          Scheduler: legacy|global-local (default: legacy)
  --gpu-fraction <R>       GPU fraction 0.0..1.0 (default: 1.0)
  --gpu-tail-stop-ratio <R>  Stop new GPU dequeues when progress reaches ratio (default: 1.0; disabled)
  --mode <M>               speed|balanced|ratio or comma list (default: ratio)
  --profile-timing         Enable GPU timing logs
  --profile-timing-detail  Enable detailed GPU timing logs
  --profile-timing-deep    Enable deep timing probe logs
  --no-build               Skip cargo build
  --keep-profile-vars      Keep COZIP_PROFILE_TIMING/DEEP from current shell
  -h, --help               Show this help

Example:
  ./bench.sh --mode ratio --size-mib 4096 --runs 5
  ./bench.sh --mode speed,balanced,ratio --runs 3
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing command: $1" >&2
    exit 1
  fi
}

is_truthy() {
  local v="${1:-}"
  local l
  l="$(echo "${v}" | tr '[:upper:]' '[:lower:]')"
  [[ -n "${l}" && "${l}" != "0" && "${l}" != "false" && "${l}" != "off" ]]
}

stats_line() {
  local label="$1"
  local values="$2"
  local count
  count="$(printf "%s\n" "$values" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [[ "${count}" == "0" ]]; then
    echo "${label}: n=0"
    return
  fi

  local mean
  mean="$(
    printf "%s\n" "$values" \
      | sed '/^$/d' \
      | awk '{s+=$1;n++} END {if(n>0) printf "%.3f", s/n; else print "nan"}'
  )"

  local median
  median="$(
    printf "%s\n" "$values" \
      | sed '/^$/d' \
      | sort -n \
      | awk '
          {a[NR]=$1}
          END {
            if (NR==0) {print "nan"; exit}
            if (NR%2==1) {printf "%.3f", a[(NR+1)/2]}
            else {printf "%.3f", (a[NR/2]+a[NR/2+1])/2}
          }'
  )"

  local min_v max_v
  min_v="$(printf "%s\n" "$values" | sed '/^$/d' | sort -n | head -n1)"
  max_v="$(printf "%s\n" "$values" | sed '/^$/d' | sort -n | tail -n1)"
  printf "%s: n=%s mean=%s median=%s min=%s max=%s\n" "${label}" "${count}" "${mean}" "${median}" "${min_v}" "${max_v}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size-mib)
      SIZE_MIB="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --warmups)
      WARMUPS="$2"
      shift 2
      ;;
    --chunk-mib)
      CHUNK_MIB="$2"
      shift 2
      ;;
    --gpu-subchunk-kib)
      GPU_SUBCHUNK_KIB="$2"
      shift 2
      ;;
    --token-finalize-segment-size)
      TOKEN_FINALIZE_SEGMENT_SIZE="$2"
      shift 2
      ;;
    --gpu-slots)
      GPU_SLOTS="$2"
      shift 2
      ;;
    --gpu-batch-chunks)
      GPU_BATCH_CHUNKS="$2"
      shift 2
      ;;
    --gpu-submit-chunks)
      GPU_SUBMIT_CHUNKS="$2"
      shift 2
      ;;
    --stream-pipeline-depth)
      STREAM_PIPELINE_DEPTH="$2"
      shift 2
      ;;
    --stream-batch-chunks)
      STREAM_BATCH_CHUNKS="$2"
      shift 2
      ;;
    --stream-max-inflight-chunks)
      STREAM_MAX_INFLIGHT_CHUNKS="$2"
      shift 2
      ;;
    --stream-max-inflight-mib)
      STREAM_MAX_INFLIGHT_MIB="$2"
      shift 2
      ;;
    --scheduler)
      SCHEDULER="$2"
      shift 2
      ;;
    --gpu-fraction)
      GPU_FRACTION="$2"
      shift 2
      ;;
    --gpu-tail-stop-ratio)
      GPU_TAIL_STOP_RATIO="$2"
      shift 2
      ;;
    --mode)
      MODES="$2"
      shift 2
      ;;
    --profile-timing)
      PROFILE_TIMING=1
      shift
      ;;
    --profile-timing-detail)
      PROFILE_TIMING=1
      PROFILE_TIMING_DETAIL=1
      shift
      ;;
    --profile-timing-deep)
      PROFILE_TIMING=1
      PROFILE_TIMING_DEEP=1
      shift
      ;;
    --no-build)
      BUILD=0
      shift
      ;;
    --keep-profile-vars)
      KEEP_PROFILE_VARS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${GPU_BATCH_CHUNKS}" ]]; then
  GPU_BATCH_CHUNKS="${GPU_SLOTS}"
fi

if [[ "${KEEP_PROFILE_VARS}" -eq 0 ]]; then
  if [[ -n "${COZIP_PROFILE_TIMING:-}" || -n "${COZIP_PROFILE_TIMING_DETAIL:-}" || -n "${COZIP_PROFILE_DEEP:-}" ]]; then
    echo "[bench] detected COZIP_PROFILE_* env vars; auto-enabling --keep-profile-vars"
    KEEP_PROFILE_VARS=1
  fi
fi

if [[ "${PROFILE_TIMING}" -eq 0 ]] && is_truthy "${COZIP_PROFILE_TIMING:-}"; then
  PROFILE_TIMING=1
fi
if [[ "${PROFILE_TIMING_DETAIL}" -eq 0 ]] && is_truthy "${COZIP_PROFILE_TIMING_DETAIL:-}"; then
  PROFILE_TIMING=1
  PROFILE_TIMING_DETAIL=1
fi
if [[ "${PROFILE_TIMING_DEEP}" -eq 0 ]] && is_truthy "${COZIP_PROFILE_DEEP:-}"; then
  PROFILE_TIMING=1
  PROFILE_TIMING_DEEP=1
fi

require_cmd cargo
require_cmd awk
require_cmd sed
require_cmd sort
require_cmd tee
require_cmd rg

mkdir -p "${OUT_DIR}"

if [[ "${BUILD}" -eq 1 ]]; then
  echo "[bench] building release benchmark binary..."
  cargo build --release -p cozip_deflate --example bench_1gb
fi

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "[bench] benchmark binary not found: ${BIN_PATH}" >&2
  exit 1
fi

IFS=',' read -r -a MODE_LIST <<< "${MODES}"
TS="$(date +%Y%m%d_%H%M%S)"

for mode in "${MODE_LIST[@]}"; do
  mode="$(echo "${mode}" | tr -d '[:space:]')"
  [[ -z "${mode}" ]] && continue

  LOG_FILE="${OUT_DIR}/bench_${mode}_${TS}.log"
  echo "[bench] mode=${mode} runs=${RUNS} -> ${LOG_FILE}"
  {
    echo "# cozip bench log"
    echo "timestamp=${TS}"
    echo "mode=${mode}"
    echo "size_mib=${SIZE_MIB} runs=${RUNS} iters=${ITERS} warmups=${WARMUPS} chunk_mib=${CHUNK_MIB} gpu_subchunk_kib=${GPU_SUBCHUNK_KIB} token_finalize_segment_size=${TOKEN_FINALIZE_SEGMENT_SIZE} gpu_slots=${GPU_SLOTS} gpu_batch_chunks=${GPU_BATCH_CHUNKS} gpu_submit_chunks=${GPU_SUBMIT_CHUNKS} stream_pipeline_depth=${STREAM_PIPELINE_DEPTH} stream_batch_chunks=${STREAM_BATCH_CHUNKS} stream_max_inflight_chunks=${STREAM_MAX_INFLIGHT_CHUNKS} stream_max_inflight_mib=${STREAM_MAX_INFLIGHT_MIB} scheduler=${SCHEDULER} gpu_fraction=${GPU_FRACTION} gpu_tail_stop_ratio=${GPU_TAIL_STOP_RATIO}"
    echo "profile_timing=${PROFILE_TIMING} profile_timing_detail=${PROFILE_TIMING_DETAIL} profile_timing_deep=${PROFILE_TIMING_DEEP}"
    echo
  } | tee "${LOG_FILE}"

  for run_idx in $(seq 1 "${RUNS}"); do
    echo "===== RUN ${run_idx}/${RUNS} mode=${mode} =====" | tee -a "${LOG_FILE}"
    cmd=(
      "${BIN_PATH}"
      --size-mib "${SIZE_MIB}"
      --iters "${ITERS}"
      --warmups "${WARMUPS}"
      --chunk-mib "${CHUNK_MIB}"
      --gpu-subchunk-kib "${GPU_SUBCHUNK_KIB}"
      --token-finalize-segment-size "${TOKEN_FINALIZE_SEGMENT_SIZE}"
      --gpu-slots "${GPU_SLOTS}"
      --gpu-batch-chunks "${GPU_BATCH_CHUNKS}"
      --gpu-submit-chunks "${GPU_SUBMIT_CHUNKS}"
      --stream-pipeline-depth "${STREAM_PIPELINE_DEPTH}"
      --stream-batch-chunks "${STREAM_BATCH_CHUNKS}"
      --stream-max-inflight-chunks "${STREAM_MAX_INFLIGHT_CHUNKS}"
      --stream-max-inflight-mib "${STREAM_MAX_INFLIGHT_MIB}"
      --scheduler "${SCHEDULER}"
      --mode "${mode}"
      --gpu-fraction "${GPU_FRACTION}"
      --gpu-tail-stop-ratio "${GPU_TAIL_STOP_RATIO}"
    )
    if [[ "${PROFILE_TIMING}" -eq 1 ]]; then
      cmd+=(--profile-timing)
    fi
    if [[ "${PROFILE_TIMING_DETAIL}" -eq 1 ]]; then
      cmd+=(--profile-timing-detail)
    fi
    if [[ "${PROFILE_TIMING_DEEP}" -eq 1 ]]; then
      cmd+=(--profile-timing-deep)
    fi

    if [[ "${KEEP_PROFILE_VARS}" -eq 1 ]]; then
      "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
    else
      env -u COZIP_PROFILE_TIMING -u COZIP_PROFILE_TIMING_DETAIL -u COZIP_PROFILE_DEEP \
      "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
    fi
    echo | tee -a "${LOG_FILE}"
  done

  comp_speedup_values="$(
    (rg '^speedup\(cpu/hybrid\):' "${LOG_FILE}" || true) \
      | sed -E 's/.* compress=([0-9.]+)x.*/\1/'
  )"
  cpu_comp_values="$(rg '^CPU_ONLY:' "${LOG_FILE}" | sed -E 's/.*avg_comp_ms=([0-9.]+).*/\1/')"
  hybrid_comp_values="$(rg '^CPU\+GPU :' "${LOG_FILE}" | sed -E 's/.*avg_comp_ms=([0-9.]+).*/\1/')"
  gpu_chunks_values="$(rg '^CPU\+GPU :' "${LOG_FILE}" | sed -E 's/.*gpu_chunks=([0-9]+).*/\1/')"

  echo "----- SUMMARY mode=${mode} -----" | tee -a "${LOG_FILE}"
  stats_line "speedup_compress_x" "${comp_speedup_values}" | tee -a "${LOG_FILE}"
  echo "speedup_decompress_x: n=0 (deprecated: decompress is CPU-only path)" | tee -a "${LOG_FILE}"
  stats_line "cpu_only_avg_comp_ms" "${cpu_comp_values}" | tee -a "${LOG_FILE}"
  stats_line "cpu_gpu_avg_comp_ms" "${hybrid_comp_values}" | tee -a "${LOG_FILE}"
  stats_line "gpu_chunks" "${gpu_chunks_values}" | tee -a "${LOG_FILE}"
  echo | tee -a "${LOG_FILE}"
done

echo "[bench] done. logs: ${OUT_DIR}"
