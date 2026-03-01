#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${ROOT_DIR}/target/release/examples/bench_gdeflate"
OUT_DIR="${ROOT_DIR}/bench_results"

SIZE_MIB=4096
RUNS=3
BENCH_RUNS=1
WARMUPS=0
CPU_WORKERS=0
GPU_COMPRESS=0
GPU_WORKERS=1
GPU_SUBMIT_TILES=8
GPU_DECOMPRESS=0
DECOMP_GPU_WORKERS=4
DECOMP_GPU_SUBMIT_TILES=64
DECOMP_GPU_SUPER_BATCH_FACTOR=2
COMPARE_HYBRID=0
MODES="tryall"
BUILD=1

usage() {
  cat <<'EOF'
Usage:
  ./gdeflate_bench.sh [options]

Options:
  --mode <M>          tryall|stored|static|dynamic or comma list (default: tryall; auto is alias)
  --size-mib <N>      Input size in MiB (default: 4096)
  --runs <N>          Number of process-restart runs per mode (default: 3)
  --bench-runs <N>    Measured runs per process (default: 1)
  --warmups <N>       Warmup runs per process (default: 0)
  --cpu-workers <N>   0=auto, fixed worker count otherwise (default: 0)
  --gpu-compress      enable GPU StoredOnly/StaticHuffman compression path in bench binary
  --gpu-workers <N>   max in-flight GPU batches handled by one GPU manager (default: 1)
  --gpu-submit-tiles <N>  micro-batch tiles per GPU submit (default: 8)
  --gpu-decompress    enable GPU static-Huffman decompression path in bench binary
  --decomp-gpu-workers <N>  max in-flight GPU decode batches handled by one GPU manager (default: 4)
  --decomp-gpu-submit-tiles <N>  micro-batch tiles per GPU decode submit (default: 64)
  --decomp-gpu-super-batch-factor <N>  multiplier for GPU decode super-batch submit (default: 2)
  --compare-hybrid    print CPU_ONLY vs HYBRID comparison
  --no-build          Skip cargo build
  -h, --help          Show this help

Examples:
  ./gdeflate_bench.sh --mode tryall --size-mib 4096 --runs 3
  ./gdeflate_bench.sh --mode stored --size-mib 1024 --cpu-workers 8 --gpu-compress --gpu-workers 4 --gpu-submit-tiles 8 --compare-hybrid
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing command: $1" >&2
    exit 1
  fi
}

is_valid_mode() {
  case "$1" in
    tryall|auto|stored|static|dynamic) return 0 ;;
    *) return 1 ;;
  esac
}

canonical_mode() {
  case "$1" in
    auto) echo "tryall" ;;
    *) echo "$1" ;;
  esac
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
    --mode)
      MODES="$2"
      shift 2
      ;;
    --size-mib)
      SIZE_MIB="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --bench-runs)
      BENCH_RUNS="$2"
      shift 2
      ;;
    --warmups)
      WARMUPS="$2"
      shift 2
      ;;
    --cpu-workers)
      CPU_WORKERS="$2"
      shift 2
      ;;
    --gpu-compress)
      GPU_COMPRESS=1
      shift
      ;;
    --gpu-workers)
      GPU_WORKERS="$2"
      shift 2
      ;;
    --gpu-submit-tiles)
      GPU_SUBMIT_TILES="$2"
      shift 2
      ;;
    --gpu-decompress)
      GPU_DECOMPRESS=1
      shift
      ;;
    --decomp-gpu-workers)
      DECOMP_GPU_WORKERS="$2"
      shift 2
      ;;
    --decomp-gpu-submit-tiles)
      DECOMP_GPU_SUBMIT_TILES="$2"
      shift 2
      ;;
    --decomp-gpu-super-batch-factor)
      DECOMP_GPU_SUPER_BATCH_FACTOR="$2"
      shift 2
      ;;
    --compare-hybrid)
      COMPARE_HYBRID=1
      shift
      ;;
    --no-build)
      BUILD=0
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

if [[ ( "${GPU_COMPRESS}" -eq 1 || "${GPU_DECOMPRESS}" -eq 1 ) && "${COMPARE_HYBRID}" -eq 0 ]]; then
  COMPARE_HYBRID=1
fi

require_cmd cargo
require_cmd awk
require_cmd sed
require_cmd sort
require_cmd tee
require_cmd rg

mkdir -p "${OUT_DIR}"

if [[ "${BUILD}" -eq 1 ]]; then
  echo "[gdeflate-bench] building release benchmark binary..."
  cargo build --release -p cozip_gdeflate --example bench_gdeflate
fi

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "[gdeflate-bench] benchmark binary not found: ${BIN_PATH}" >&2
  exit 1
fi

IFS=',' read -r -a MODE_LIST <<< "${MODES}"
TS="$(date +%Y%m%d_%H%M%S)"

for raw_mode in "${MODE_LIST[@]}"; do
  raw_mode="$(echo "${raw_mode}" | tr -d '[:space:]')"
  [[ -z "${raw_mode}" ]] && continue
  if ! is_valid_mode "${raw_mode}"; then
    echo "invalid mode: ${raw_mode} (expected tryall|stored|static|dynamic; auto is alias)" >&2
    exit 2
  fi
  mode="$(canonical_mode "${raw_mode}")"

  LOG_FILE="${OUT_DIR}/gdeflate_bench_${mode}_${TS}.log"
  echo "[gdeflate-bench] mode=${mode} runs=${RUNS} -> ${LOG_FILE}"
  {
    echo "# cozip_gdeflate bench log"
    echo "timestamp=${TS}"
    echo "mode=${mode}"
    echo "size_mib=${SIZE_MIB} runs=${RUNS} bench_runs=${BENCH_RUNS} warmups=${WARMUPS} cpu_workers=${CPU_WORKERS} gpu_compress=${GPU_COMPRESS} gpu_workers=${GPU_WORKERS} gpu_submit_tiles=${GPU_SUBMIT_TILES} gpu_decompress=${GPU_DECOMPRESS} decomp_gpu_workers=${DECOMP_GPU_WORKERS} decomp_gpu_submit_tiles=${DECOMP_GPU_SUBMIT_TILES} decomp_gpu_super_batch_factor=${DECOMP_GPU_SUPER_BATCH_FACTOR} compare_hybrid=${COMPARE_HYBRID}"
    echo
  } | tee "${LOG_FILE}"

  for run_idx in $(seq 1 "${RUNS}"); do
    echo "===== RUN ${run_idx}/${RUNS} mode=${mode} =====" | tee -a "${LOG_FILE}"
    cmd=( "${BIN_PATH}" \
      --mode "${mode}" \
      --size-mib "${SIZE_MIB}" \
      --runs "${BENCH_RUNS}" \
      --warmups "${WARMUPS}" \
      --cpu-workers "${CPU_WORKERS}" \
      --gpu-workers "${GPU_WORKERS}" \
      --gpu-submit-tiles "${GPU_SUBMIT_TILES}" \
      --decomp-gpu-workers "${DECOMP_GPU_WORKERS}" \
      --decomp-gpu-submit-tiles "${DECOMP_GPU_SUBMIT_TILES}" \
      --decomp-gpu-super-batch-factor "${DECOMP_GPU_SUPER_BATCH_FACTOR}" )
    if [[ "${GPU_COMPRESS}" -eq 1 ]]; then
      cmd+=( --gpu-compress )
    fi
    if [[ "${GPU_DECOMPRESS}" -eq 1 ]]; then
      cmd+=( --gpu-decompress )
    fi
    if [[ "${COMPARE_HYBRID}" -eq 1 ]]; then
      cmd+=( --compare-hybrid )
    fi
    "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
    echo | tee -a "${LOG_FILE}"
  done

  run_lines="$(rg '^run [0-9]+/[0-9]+: ' "${LOG_FILE}" || true)"
  comp_ms_values="$(printf '%s\n' "${run_lines}" | awk '{for(i=1;i<=NF;i++) if($i ~ /^comp_ms=/){sub(/^comp_ms=/,"",$i); print $i}}')"
  decomp_ms_values="$(printf '%s\n' "${run_lines}" | awk '{for(i=1;i<=NF;i++) if($i ~ /^decomp_ms=/){sub(/^decomp_ms=/,"",$i); print $i}}')"
  comp_mib_s_values="$(printf '%s\n' "${run_lines}" | awk '{for(i=1;i<=NF;i++) if($i ~ /^comp_mib_s=/){sub(/^comp_mib_s=/,"",$i); print $i}}')"
  decomp_mib_s_values="$(printf '%s\n' "${run_lines}" | awk '{for(i=1;i<=NF;i++) if($i ~ /^decomp_mib_s=/){sub(/^decomp_mib_s=/,"",$i); print $i}}')"
  ratio_values="$(printf '%s\n' "${run_lines}" | awk '{for(i=1;i<=NF;i++) if($i ~ /^ratio=/){sub(/^ratio=/,"",$i); print $i}}')"

  echo "----- SUMMARY mode=${mode} -----" | tee -a "${LOG_FILE}"
  stats_line "comp_ms" "${comp_ms_values}" | tee -a "${LOG_FILE}"
  stats_line "decomp_ms" "${decomp_ms_values}" | tee -a "${LOG_FILE}"
  stats_line "comp_mib_s" "${comp_mib_s_values}" | tee -a "${LOG_FILE}"
  stats_line "decomp_mib_s" "${decomp_mib_s_values}" | tee -a "${LOG_FILE}"
  stats_line "ratio" "${ratio_values}" | tee -a "${LOG_FILE}"
  echo | tee -a "${LOG_FILE}"
done

echo "[gdeflate-bench] done. logs: ${OUT_DIR}"
