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
GPU_FRACTION=1.0
MODES="ratio"
BUILD=1
KEEP_PROFILE_VARS=0

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
  --gpu-fraction <R>       GPU fraction 0.0..1.0 (default: 1.0)
  --mode <M>               speed|balanced|ratio or comma list (default: ratio)
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
    --gpu-fraction)
      GPU_FRACTION="$2"
      shift 2
      ;;
    --mode)
      MODES="$2"
      shift 2
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
    echo "size_mib=${SIZE_MIB} runs=${RUNS} iters=${ITERS} warmups=${WARMUPS} chunk_mib=${CHUNK_MIB} gpu_subchunk_kib=${GPU_SUBCHUNK_KIB} gpu_fraction=${GPU_FRACTION}"
    echo
  } | tee "${LOG_FILE}"

  for run_idx in $(seq 1 "${RUNS}"); do
    echo "===== RUN ${run_idx}/${RUNS} mode=${mode} =====" | tee -a "${LOG_FILE}"
    if [[ "${KEEP_PROFILE_VARS}" -eq 1 ]]; then
      "${BIN_PATH}" \
      --size-mib "${SIZE_MIB}" \
      --iters "${ITERS}" \
      --warmups "${WARMUPS}" \
      --chunk-mib "${CHUNK_MIB}" \
      --gpu-subchunk-kib "${GPU_SUBCHUNK_KIB}" \
      --mode "${mode}" \
      --gpu-fraction "${GPU_FRACTION}" \
      2>&1 | tee -a "${LOG_FILE}"
    else
      env -u COZIP_PROFILE_TIMING -u COZIP_PROFILE_DEEP \
      "${BIN_PATH}" \
      --size-mib "${SIZE_MIB}" \
      --iters "${ITERS}" \
      --warmups "${WARMUPS}" \
      --chunk-mib "${CHUNK_MIB}" \
      --gpu-subchunk-kib "${GPU_SUBCHUNK_KIB}" \
      --mode "${mode}" \
      --gpu-fraction "${GPU_FRACTION}" \
      2>&1 | tee -a "${LOG_FILE}"
    fi
    echo | tee -a "${LOG_FILE}"
  done

  comp_speedup_values="$(
    rg '^speedup\(cpu/hybrid\):' "${LOG_FILE}" \
      | sed -E 's/.* compress=([0-9.]+)x decompress=.*/\1/'
  )"
  decomp_speedup_values="$(
    rg '^speedup\(cpu/hybrid\):' "${LOG_FILE}" \
      | sed -E 's/.* decompress=([0-9.]+)x.*/\1/'
  )"
  cpu_comp_values="$(rg '^CPU_ONLY:' "${LOG_FILE}" | sed -E 's/.*avg_comp_ms=([0-9.]+).*/\1/')"
  hybrid_comp_values="$(rg '^CPU\+GPU :' "${LOG_FILE}" | sed -E 's/.*avg_comp_ms=([0-9.]+).*/\1/')"
  gpu_chunks_values="$(rg '^CPU\+GPU :' "${LOG_FILE}" | sed -E 's/.*gpu_chunks=([0-9]+).*/\1/')"

  echo "----- SUMMARY mode=${mode} -----" | tee -a "${LOG_FILE}"
  stats_line "speedup_compress_x" "${comp_speedup_values}" | tee -a "${LOG_FILE}"
  stats_line "speedup_decompress_x" "${decomp_speedup_values}" | tee -a "${LOG_FILE}"
  stats_line "cpu_only_avg_comp_ms" "${cpu_comp_values}" | tee -a "${LOG_FILE}"
  stats_line "cpu_gpu_avg_comp_ms" "${hybrid_comp_values}" | tee -a "${LOG_FILE}"
  stats_line "gpu_chunks" "${gpu_chunks_values}" | tee -a "${LOG_FILE}"
  echo | tee -a "${LOG_FILE}"
done

echo "[bench] done. logs: ${OUT_DIR}"
