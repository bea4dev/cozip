use std::time::Instant;

use cozip_pdeflate::{
    CoZipDeflate, CompressionMode, DeflateCpuStreamStats, HybridOptions, HybridSchedulerPolicy,
};

#[derive(Debug, Clone, Copy)]
enum DatasetKind {
    Bench,
    Legacy,
    Random,
}

impl DatasetKind {
    fn from_str(v: &str) -> Option<Self> {
        match v {
            "bench" => Some(Self::Bench),
            "legacy" => Some(Self::Legacy),
            "random" => Some(Self::Random),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Bench => "bench",
            Self::Legacy => "legacy",
            Self::Random => "random",
        }
    }
}

#[derive(Debug, Clone)]
struct BenchConfig {
    size_mib: usize,
    runs: usize,
    warmups: usize,
    chunk_mib: usize,
    sections: usize,
    dataset: DatasetKind,
    verify_bytes: bool,
    skip_decompress: bool,
    gpu_compress: bool,
    gpu_only: bool,
    compare_hybrid: bool,
    gpu_workers: usize,
    gpu_slot_count: usize,
    gpu_batch_chunks: usize,
    gpu_submit_chunks: usize,
    gpu_subchunk_kib: usize,
    token_finalize_segment_size: usize,
    stream_pipeline_depth: usize,
    stream_batch_chunks: usize,
    stream_max_inflight_chunks: usize,
    stream_max_inflight_mib: usize,
    gpu_fraction: f32,
    gpu_min_chunk_kib: usize,
    scheduler_policy: HybridSchedulerPolicy,
    gpu_tail_stop_ratio: f32,
    mode: CompressionMode,
    profile_timing: bool,
    profile_timing_detail: bool,
    profile_timing_deep: bool,
    profile_outer: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            size_mib: 4096,
            runs: 1,
            warmups: 0,
            chunk_mib: 4,
            sections: 128,
            dataset: DatasetKind::Bench,
            verify_bytes: true,
            skip_decompress: true,
            gpu_compress: false,
            gpu_only: false,
            compare_hybrid: false,
            gpu_workers: 1,
            gpu_slot_count: 6,
            gpu_batch_chunks: 6,
            gpu_submit_chunks: 3,
            gpu_subchunk_kib: 512,
            token_finalize_segment_size: 4096,
            stream_pipeline_depth: 3,
            stream_batch_chunks: 0,
            stream_max_inflight_chunks: 0,
            stream_max_inflight_mib: 0,
            gpu_fraction: 1.0,
            gpu_min_chunk_kib: 64,
            scheduler_policy: HybridSchedulerPolicy::GlobalQueueLocalBuffers,
            gpu_tail_stop_ratio: 1.0,
            mode: CompressionMode::Ratio,
            profile_timing: env_flag("COZIP_PROFILE_TIMING"),
            profile_timing_detail: env_flag("COZIP_PROFILE_TIMING_DETAIL"),
            profile_timing_deep: env_flag("COZIP_PROFILE_DEEP"),
            profile_outer: env_flag("COZIP_PDEFLATE_PROFILE_OUTER")
                || env_flag("COZIP_PDEFLATE_PROFILE"),
        }
    }
}

#[derive(Debug, Clone)]
struct RunResult {
    comp_ms: f64,
    decomp_ms: Option<f64>,
    verify_compare_ms: Option<f64>,
    verify_roundtrip_ms: Option<f64>,
    ratio: f64,
    comp_mib_s: f64,
    decomp_mib_s: Option<f64>,
    compress_stats: DeflateCpuStreamStats,
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            let lowered = value.trim().to_ascii_lowercase();
            !(lowered.is_empty() || lowered == "0" || lowered == "false" || lowered == "off")
        }
        Err(_) => false,
    }
}

fn parse_bool(v: &str) -> Option<bool> {
    match v {
        "1" | "true" | "TRUE" | "True" => Some(true),
        "0" | "false" | "FALSE" | "False" => Some(false),
        _ => None,
    }
}

fn parse_args() -> Result<BenchConfig, String> {
    let mut cfg = BenchConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--size-mib" => {
                i += 1;
                cfg.size_mib = args
                    .get(i)
                    .ok_or("--size-mib requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --size-mib: {e}"))?;
            }
            "--runs" => {
                i += 1;
                cfg.runs = args
                    .get(i)
                    .ok_or("--runs requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --runs: {e}"))?;
            }
            "--warmups" => {
                i += 1;
                cfg.warmups = args
                    .get(i)
                    .ok_or("--warmups requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --warmups: {e}"))?;
            }
            "--chunk-mib" => {
                i += 1;
                cfg.chunk_mib = args
                    .get(i)
                    .ok_or("--chunk-mib requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --chunk-mib: {e}"))?;
            }
            "--sections" => {
                i += 1;
                cfg.sections = args
                    .get(i)
                    .ok_or("--sections requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --sections: {e}"))?;
            }
            "--dataset" => {
                i += 1;
                let v = args.get(i).ok_or("--dataset requires value")?;
                cfg.dataset = DatasetKind::from_str(v.as_str()).ok_or_else(|| {
                    format!("invalid --dataset: {v} (expected bench|legacy|random)")
                })?;
            }
            "--gpu-compress" => cfg.gpu_compress = true,
            "--gpu-only" => cfg.gpu_only = true,
            "--gpu-only-enabled" => {
                i += 1;
                let v = args.get(i).ok_or("--gpu-only-enabled requires value")?;
                cfg.gpu_only =
                    parse_bool(v).ok_or_else(|| format!("invalid --gpu-only-enabled: {v}"))?;
            }
            "--gpu-compress-enabled" => {
                i += 1;
                let v = args.get(i).ok_or("--gpu-compress-enabled requires value")?;
                cfg.gpu_compress =
                    parse_bool(v).ok_or_else(|| format!("invalid --gpu-compress-enabled: {v}"))?;
            }
            "--compare-hybrid" => cfg.compare_hybrid = true,
            "--compare-hybrid-enabled" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--compare-hybrid-enabled requires value")?;
                cfg.compare_hybrid = parse_bool(v)
                    .ok_or_else(|| format!("invalid --compare-hybrid-enabled: {v}"))?;
            }
            "--gpu-workers" => {
                i += 1;
                cfg.gpu_workers = args
                    .get(i)
                    .ok_or("--gpu-workers requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-workers: {e}"))?;
            }
            "--gpu-batch-chunks" => {
                i += 1;
                cfg.gpu_batch_chunks = args
                    .get(i)
                    .ok_or("--gpu-batch-chunks requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-batch-chunks: {e}"))?;
            }
            "--gpu-submit-chunks" => {
                i += 1;
                cfg.gpu_submit_chunks = args
                    .get(i)
                    .ok_or("--gpu-submit-chunks requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-submit-chunks: {e}"))?;
            }
            "--gpu-slot-count" | "--gpu-slots" => {
                i += 1;
                cfg.gpu_slot_count = args
                    .get(i)
                    .ok_or("--gpu-slot-count/--gpu-slots requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-slot-count: {e}"))?;
            }
            "--gpu-pipelined-submit-chunks" => {
                i += 1;
                cfg.gpu_submit_chunks = args
                    .get(i)
                    .ok_or("--gpu-pipelined-submit-chunks requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-pipelined-submit-chunks: {e}"))?;
            }
            "--gpu-subchunk-kib" => {
                i += 1;
                cfg.gpu_subchunk_kib = args
                    .get(i)
                    .ok_or("--gpu-subchunk-kib requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-subchunk-kib: {e}"))?;
            }
            "--token-finalize-segment-size" => {
                i += 1;
                cfg.token_finalize_segment_size = args
                    .get(i)
                    .ok_or("--token-finalize-segment-size requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --token-finalize-segment-size: {e}"))?;
            }
            "--stream-pipeline-depth" => {
                i += 1;
                cfg.stream_pipeline_depth = args
                    .get(i)
                    .ok_or("--stream-pipeline-depth requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --stream-pipeline-depth: {e}"))?;
            }
            "--stream-batch-chunks" => {
                i += 1;
                cfg.stream_batch_chunks = args
                    .get(i)
                    .ok_or("--stream-batch-chunks requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --stream-batch-chunks: {e}"))?;
            }
            "--stream-max-inflight-chunks" => {
                i += 1;
                cfg.stream_max_inflight_chunks = args
                    .get(i)
                    .ok_or("--stream-max-inflight-chunks requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --stream-max-inflight-chunks: {e}"))?;
            }
            "--stream-max-inflight-mib" => {
                i += 1;
                cfg.stream_max_inflight_mib = args
                    .get(i)
                    .ok_or("--stream-max-inflight-mib requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --stream-max-inflight-mib: {e}"))?;
            }
            "--gpu-fraction" => {
                i += 1;
                cfg.gpu_fraction = args
                    .get(i)
                    .ok_or("--gpu-fraction requires value")?
                    .parse::<f32>()
                    .map_err(|e| format!("invalid --gpu-fraction: {e}"))?;
            }
            "--gpu-min-chunk-kib" => {
                i += 1;
                cfg.gpu_min_chunk_kib = args
                    .get(i)
                    .ok_or("--gpu-min-chunk-kib requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-min-chunk-kib: {e}"))?;
            }
            "--scheduler" => {
                i += 1;
                let v = args.get(i).ok_or("--scheduler requires value")?;
                cfg.scheduler_policy = match v.as_str() {
                    "global" | "global-local" => HybridSchedulerPolicy::GlobalQueueLocalBuffers,
                    _ => return Err(format!("invalid --scheduler: {v}")),
                };
            }
            "--gpu-tail-stop-ratio" => {
                i += 1;
                cfg.gpu_tail_stop_ratio = args
                    .get(i)
                    .ok_or("--gpu-tail-stop-ratio requires value")?
                    .parse::<f32>()
                    .map_err(|e| format!("invalid --gpu-tail-stop-ratio: {e}"))?;
            }
            "--mode" => {
                i += 1;
                let v = args.get(i).ok_or("--mode requires value")?;
                cfg.mode = match v.as_str() {
                    "speed" => CompressionMode::Speed,
                    "balanced" => CompressionMode::Balanced,
                    "ratio" => CompressionMode::Ratio,
                    _ => return Err("invalid --mode (expected speed|balanced|ratio)".to_string()),
                };
            }
            "--verify" => cfg.verify_bytes = true,
            "--no-verify" => cfg.verify_bytes = false,
            "--verify-bytes" => {
                i += 1;
                let v = args.get(i).ok_or("--verify-bytes requires value")?;
                cfg.verify_bytes =
                    parse_bool(v).ok_or_else(|| format!("invalid --verify-bytes: {v}"))?;
            }
            "--skip-decompress" => cfg.skip_decompress = true,
            "--no-skip-decompress" => cfg.skip_decompress = false,
            "--profile-outer" => cfg.profile_outer = true,
            "--no-profile-outer" => cfg.profile_outer = false,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            x => return Err(format!("unknown argument: {x}")),
        }
        i += 1;
    }

    if cfg.runs == 0 || cfg.size_mib == 0 || cfg.chunk_mib == 0 {
        return Err("--size-mib/--runs/--chunk-mib must be > 0".to_string());
    }
    if cfg.gpu_slot_count == 0
        || cfg.gpu_batch_chunks == 0
        || cfg.gpu_submit_chunks == 0
        || cfg.gpu_subchunk_kib == 0
        || cfg.token_finalize_segment_size == 0
        || cfg.stream_pipeline_depth == 0
    {
        return Err("gpu-slot-count/gpu-batch-chunks/gpu-submit-chunks/gpu-subchunk-kib/token-finalize-segment-size/stream-pipeline-depth must be > 0".to_string());
    }
    if cfg.stream_batch_chunks != 0 {
        return Err(
            "--stream-batch-chunks is fixed to 0 (legacy batch mode was removed)".to_string(),
        );
    }
    if !(0.0..=1.0).contains(&cfg.gpu_fraction) {
        return Err("--gpu-fraction must be in range 0.0..=1.0".to_string());
    }
    if !(0.0..=1.0).contains(&cfg.gpu_tail_stop_ratio) {
        return Err("--gpu-tail-stop-ratio must be in range 0.0..=1.0".to_string());
    }
    if cfg.gpu_only && !cfg.gpu_compress {
        return Err("--gpu-only requires --gpu-compress".to_string());
    }

    if cfg.profile_timing_detail || cfg.profile_timing_deep {
        cfg.profile_timing = true;
    }

    Ok(cfg)
}

fn print_help() {
    println!(
        "usage: cargo run --release -p cozip_pdeflate --example bench_pdeflate -- [options]\n\
options:\n\
  --size-mib <N>\n\
  --runs <N>\n\
  --warmups <N>\n\
  --chunk-mib <N>\n\
  --dataset <bench|legacy|random> (default: bench)\n\
  --sections <N>\n\
  --gpu-compress\n\
  --gpu-only\n\
  --compare-hybrid\n\
  --gpu-workers <N>\n\
  --gpu-slot-count/--gpu-slots <N>\n\
  --gpu-batch-chunks <N>\n\
  --gpu-submit-chunks <N>\n\
  --gpu-pipelined-submit-chunks <N> (legacy alias of --gpu-submit-chunks)\n\
  --gpu-subchunk-kib <N>\n\
  --token-finalize-segment-size <N>\n\
  --stream-pipeline-depth <N>\n\
  --stream-batch-chunks <N> (fixed to 0)\n\
  --stream-max-inflight-chunks <N>\n\
  --stream-max-inflight-mib <N>\n\
  --gpu-fraction <R>\n\
  --gpu-min-chunk-kib <N>\n\
  --scheduler <global|global-local>\n\
  --gpu-tail-stop-ratio <R>\n\
  --mode <speed|balanced|ratio> (default: ratio)\n\
  --skip-decompress / --no-skip-decompress\n\
  --verify / --no-verify / --verify-bytes <0|1>\n\
  --profile-outer / --no-profile-outer"
    );
}

fn build_dataset_bench(size_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size_bytes);
    let mut state: u32 = 0x1234_5678;
    while out.len() < size_bytes {
        let zone = (out.len() / 4096) % 3;
        match zone {
            0 => out.extend_from_slice(b"cozip-cpu-gpu-deflate-"),
            1 => out.extend_from_slice(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            _ => {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                out.push((state >> 24) as u8);
            }
        }
    }
    out.truncate(size_bytes);
    out
}

fn build_dataset_legacy(size_bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; size_bytes];
    let text = b"ABABABABCCABCCD--cozip-pdeflate-bench--";
    let mut rng = 0x1234_5678_u32;
    for (i, b) in out.iter_mut().enumerate() {
        *b = match (i / 8192) % 6 {
            0 => text[i % text.len()],
            1 => b'A' + ((i / 11) % 8) as u8,
            2 => {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                (rng >> 24) as u8
            }
            3 => (i as u8).wrapping_mul(17).wrapping_add(31),
            4 => {
                if (i / 64) % 2 == 0 {
                    0
                } else {
                    255
                }
            }
            _ => (i % 251) as u8,
        };
    }
    out
}

fn build_dataset_random(size_bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; size_bytes];
    let mut state = 0x5a17_3c9d_u32;
    for b in &mut out {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *b = (state >> 24) as u8;
    }
    out
}

fn generate_input(size_bytes: usize, dataset: DatasetKind) -> Vec<u8> {
    match dataset {
        DatasetKind::Bench => build_dataset_bench(size_bytes),
        DatasetKind::Legacy => build_dataset_legacy(size_bytes),
        DatasetKind::Random => build_dataset_random(size_bytes),
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

fn min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn run_once(
    input: &[u8],
    cozip: &CoZipDeflate,
    size_mib: usize,
    skip_decompress: bool,
    verify_bytes: bool,
) -> Result<RunResult, String> {
    let c0 = Instant::now();
    let mut src = std::io::Cursor::new(input);
    let mut compressed = Vec::new();
    let compress = cozip
        .deflate_compress_stream_zip_compatible_with_index(&mut src, &mut compressed)
        .map_err(|e| e.to_string())?;
    let comp_ms = c0.elapsed().as_secs_f64() * 1000.0;

    let mut decomp_ms = None;
    let mut decomp_mib_s = None;
    let mut verify_compare_ms = None;
    let mut verify_roundtrip_ms = None;
    if !skip_decompress {
        let mut restored = Vec::with_capacity(input.len());
        let d0 = Instant::now();
        if let Err(primary_err) = cozip.pdeflate_decompress_bytes(&compressed, &mut restored) {
            if let Some(index) = compress.index.as_ref() {
                restored.clear();
                let mut reader = std::io::Cursor::new(&compressed);
                cozip
                    .deflate_decompress_stream_zip_compatible_with_index(
                        &mut reader,
                        &mut restored,
                        index,
                    )
                    .map_err(|fallback_err| {
                        format!(
                            "pdeflate_decompress_bytes failed: {primary_err}; indexed fallback failed: {fallback_err}"
                        )
                    })?;
            } else {
                return Err(primary_err.to_string());
            }
        }
        let d_ms = d0.elapsed().as_secs_f64() * 1000.0;
        if verify_bytes {
            let v0 = Instant::now();
            let ok = restored == input;
            let v_ms = v0.elapsed().as_secs_f64() * 1000.0;
            verify_compare_ms = Some(v_ms);
            if !ok {
                return Err("roundtrip mismatch".to_string());
            }
        }
        decomp_mib_s = Some(if d_ms > 0.0 {
            size_mib as f64 * 1000.0 / d_ms
        } else {
            0.0
        });
        decomp_ms = Some(d_ms);
    } else if verify_bytes {
        // Verification path for --skip-decompress:
        // run roundtrip check outside timing so throughput numbers stay compression-only.
        let mut restored = Vec::with_capacity(input.len());
        let v0 = Instant::now();
        if let Err(primary_err) = cozip.pdeflate_decompress_bytes(&compressed, &mut restored) {
            if let Some(index) = compress.index.as_ref() {
                restored.clear();
                let mut reader = std::io::Cursor::new(&compressed);
                cozip
                    .deflate_decompress_stream_zip_compatible_with_index(
                        &mut reader,
                        &mut restored,
                        index,
                    )
                    .map_err(|fallback_err| {
                        format!(
                            "pdeflate_decompress_bytes failed: {primary_err}; indexed fallback failed: {fallback_err}"
                        )
                    })?;
            } else {
                return Err(primary_err.to_string());
            }
        }
        let ok = restored == input;
        let v_ms = v0.elapsed().as_secs_f64() * 1000.0;
        verify_roundtrip_ms = Some(v_ms);
        if !ok {
            return Err("roundtrip mismatch".to_string());
        }
    }

    Ok(RunResult {
        comp_ms,
        decomp_ms,
        verify_compare_ms,
        verify_roundtrip_ms,
        ratio: compressed.len() as f64 / input.len() as f64,
        comp_mib_s: if comp_ms > 0.0 {
            size_mib as f64 * 1000.0 / comp_ms
        } else {
            0.0
        },
        decomp_mib_s,
        compress_stats: compress.stats,
    })
}

fn format_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.3}"))
        .unwrap_or_else(|| "SKIP".to_string())
}

fn format_opt_labeled(v: Option<f64>, label: &str) -> String {
    v.map(|x| format!(" {label}={x:.3}"))
        .unwrap_or_else(String::new)
}

fn cpu_parallelism_estimate(stats: &DeflateCpuStreamStats, comp_ms: f64) -> f64 {
    if comp_ms > 0.0 {
        stats.cpu_worker_busy_ms / comp_ms
    } else {
        0.0
    }
}

fn ensure_scheduler_equivalence(cpu: &HybridOptions, hybrid: &HybridOptions) -> Result<(), String> {
    let same = cpu.chunk_size == hybrid.chunk_size
        && cpu.gpu_subchunk_size == hybrid.gpu_subchunk_size
        && cpu.gpu_slot_count == hybrid.gpu_slot_count
        && cpu.stream_prepare_pipeline_depth == hybrid.stream_prepare_pipeline_depth
        && cpu.stream_batch_chunks == hybrid.stream_batch_chunks
        && cpu.stream_max_inflight_chunks == hybrid.stream_max_inflight_chunks
        && cpu.stream_max_inflight_bytes == hybrid.stream_max_inflight_bytes
        && cpu.gpu_batch_chunks == hybrid.gpu_batch_chunks
        && cpu.decode_gpu_batch_chunks == hybrid.decode_gpu_batch_chunks
        && cpu.gpu_pipelined_submit_chunks == hybrid.gpu_pipelined_submit_chunks
        && cpu.token_finalize_segment_size == hybrid.token_finalize_segment_size
        && cpu.compression_level == hybrid.compression_level
        && cpu.compression_mode == hybrid.compression_mode
        && cpu.gpu_tail_stop_ratio == hybrid.gpu_tail_stop_ratio
        && cpu.gpu_min_chunk_size == hybrid.gpu_min_chunk_size
        && cpu.scheduler_policy == hybrid.scheduler_policy
        && cpu.profile_timing == hybrid.profile_timing
        && cpu.profile_timing_detail == hybrid.profile_timing_detail
        && cpu.profile_timing_deep == hybrid.profile_timing_deep;
    if same {
        Ok(())
    } else {
        Err("CPU_ONLY と CPU+GPU で scheduler/queue 関連のオプションが一致していません".to_string())
    }
}

fn main() -> Result<(), String> {
    let cfg = parse_args()?;
    let size_bytes = cfg.size_mib * 1024 * 1024;
    let input = generate_input(size_bytes, cfg.dataset);

    let base_opts = HybridOptions {
        chunk_size: cfg.chunk_mib * 1024 * 1024,
        gpu_subchunk_size: cfg.gpu_subchunk_kib * 1024,
        gpu_slot_count: cfg.gpu_slot_count,
        gpu_batch_chunks: cfg.gpu_batch_chunks,
        gpu_pipelined_submit_chunks: cfg.gpu_submit_chunks,
        stream_prepare_pipeline_depth: cfg.stream_pipeline_depth,
        stream_batch_chunks: cfg.stream_batch_chunks,
        stream_max_inflight_chunks: cfg.stream_max_inflight_chunks,
        stream_max_inflight_bytes: cfg.stream_max_inflight_mib * 1024 * 1024,
        scheduler_policy: cfg.scheduler_policy,
        token_finalize_segment_size: cfg.token_finalize_segment_size,
        compression_level: 6,
        compression_mode: cfg.mode,
        // CPU_ONLY / Hybrid ともに同一スケジューラ条件を使う。
        // 差分は prefer_gpu/gpu_fraction/gpu_only のみ。
        prefer_gpu: false,
        gpu_only: false,
        gpu_fraction: 0.0,
        gpu_tail_stop_ratio: cfg.gpu_tail_stop_ratio,
        gpu_min_chunk_size: cfg.gpu_min_chunk_kib * 1024,
        profile_timing: cfg.profile_timing,
        profile_timing_detail: cfg.profile_timing_detail,
        profile_timing_deep: cfg.profile_timing_deep,
        ..HybridOptions::default()
    };
    let mut cpu_opts = base_opts.clone();
    cpu_opts.prefer_gpu = false;
    cpu_opts.gpu_only = false;
    cpu_opts.gpu_fraction = 0.0;

    let mut hybrid_opts = base_opts;
    hybrid_opts.prefer_gpu = cfg.gpu_compress;
    hybrid_opts.gpu_only = cfg.gpu_only && cfg.gpu_compress;
    hybrid_opts.gpu_fraction = if cfg.gpu_compress {
        cfg.gpu_fraction
    } else {
        0.0
    };

    ensure_scheduler_equivalence(&cpu_opts, &hybrid_opts)?;

    let cpu = CoZipDeflate::init(cpu_opts).map_err(|e| e.to_string())?;
    let hybrid = CoZipDeflate::init(hybrid_opts).map_err(|e| e.to_string())?;

    println!(
        "cozip_pdeflate benchmark\nsize_mib={} runs={} warmups={} chunk_mib={} sections={} dataset={} gpu_compress={} gpu_only={} gpu_workers={} gpu_slot_count={} gpu_batch_chunks={} gpu_submit_chunks={} gpu_subchunk_kib={} token_finalize_segment_size={} stream_pipeline_depth={} stream_batch_chunks={} stream_max_inflight_chunks={} stream_max_inflight_mib={} gpu_fraction={:.2} gpu_min_chunk_kib={} scheduler={:?} gpu_tail_stop_ratio={:.2} compare_hybrid={} verify_bytes={}",
        cfg.size_mib,
        cfg.runs,
        cfg.warmups,
        cfg.chunk_mib,
        cfg.sections,
        cfg.dataset.as_str(),
        cfg.gpu_compress,
        cfg.gpu_only,
        cfg.gpu_workers,
        cfg.gpu_slot_count,
        cfg.gpu_batch_chunks,
        cfg.gpu_submit_chunks,
        cfg.gpu_subchunk_kib,
        cfg.token_finalize_segment_size,
        cfg.stream_pipeline_depth,
        cfg.stream_batch_chunks,
        cfg.stream_max_inflight_chunks,
        cfg.stream_max_inflight_mib,
        cfg.gpu_fraction,
        cfg.gpu_min_chunk_kib,
        cfg.scheduler_policy,
        cfg.gpu_tail_stop_ratio,
        cfg.compare_hybrid,
        cfg.verify_bytes
    );
    println!("skip_decompress={}", cfg.skip_decompress);
    if cfg.skip_decompress && cfg.verify_bytes {
        println!("[bench] verify_bytes is enabled; roundtrip check runs outside timing");
    }
    let per_iteration_execs = if cfg.compare_hybrid { 2 } else { 1 };
    let total_timed_execs = cfg.runs.saturating_mul(per_iteration_execs);
    let total_warmup_execs = cfg.warmups.saturating_mul(per_iteration_execs);
    println!(
        "[bench] execution_plan iterations={} compare_hybrid={} execs_per_iteration={} timed_execs_total={} warmup_execs_total={}",
        cfg.runs, cfg.compare_hybrid, per_iteration_execs, total_timed_execs, total_warmup_execs
    );
    if cfg.compare_hybrid {
        println!("[bench] compare-hybrid runs CPU_ONLY then CPU+GPU in each iteration");
    }

    for _ in 0..cfg.warmups {
        if cfg.compare_hybrid {
            let _ = run_once(
                &input,
                &cpu,
                cfg.size_mib,
                cfg.skip_decompress,
                cfg.verify_bytes,
            )?;
        }
        let _ = run_once(
            &input,
            &hybrid,
            cfg.size_mib,
            cfg.skip_decompress,
            cfg.verify_bytes,
        )?;
    }

    let mut comp_ms = Vec::with_capacity(cfg.runs);
    let mut decomp_ms = Vec::with_capacity(cfg.runs);
    let mut comp_mib_s = Vec::with_capacity(cfg.runs);
    let mut decomp_mib_s = Vec::with_capacity(cfg.runs);
    let mut ratio = Vec::with_capacity(cfg.runs);
    let mut cpu_comp_ms = Vec::with_capacity(cfg.runs);
    let mut cpu_decomp_ms = Vec::with_capacity(cfg.runs);
    let mut speedup_comp = Vec::with_capacity(cfg.runs);
    let mut speedup_decomp = Vec::with_capacity(cfg.runs);
    let mut last_hybrid_stats = DeflateCpuStreamStats::default();
    let mut last_hybrid_comp_ms = 0.0_f64;
    let mut last_cpu_stats = DeflateCpuStreamStats::default();
    let mut last_cpu_comp_ms = 0.0_f64;

    for i in 0..cfg.runs {
        if cfg.compare_hybrid {
            let c = run_once(
                &input,
                &cpu,
                cfg.size_mib,
                cfg.skip_decompress,
                cfg.verify_bytes,
            )?;
            let h = run_once(
                &input,
                &hybrid,
                cfg.size_mib,
                cfg.skip_decompress,
                cfg.verify_bytes,
            )?;
            last_hybrid_stats = h.compress_stats;
            last_hybrid_comp_ms = h.comp_ms;
            last_cpu_stats = c.compress_stats;
            last_cpu_comp_ms = c.comp_ms;
            let spc = if h.comp_ms > 0.0 {
                c.comp_ms / h.comp_ms
            } else {
                0.0
            };
            let spd = match (c.decomp_ms, h.decomp_ms) {
                (Some(a), Some(b)) if b > 0.0 => a / b,
                _ => 0.0,
            };
            let hybrid_label = if cfg.gpu_only { "GPU_ONLY" } else { "CPU+GPU" };
            println!(
                "run {}/{}: CPU_ONLY comp_ms={:.3} decomp={}{}{} | {} comp_ms={:.3} decomp={}{}{} ratio={:.4} speedup_comp={:.3}x speedup_decomp={:.3}x",
                i + 1,
                cfg.runs,
                c.comp_ms,
                format_opt(c.decomp_ms),
                format_opt_labeled(c.verify_compare_ms, "verify_cmp_ms"),
                format_opt_labeled(c.verify_roundtrip_ms, "verify_roundtrip_ms"),
                hybrid_label,
                h.comp_ms,
                format_opt(h.decomp_ms),
                format_opt_labeled(h.verify_compare_ms, "verify_cmp_ms"),
                format_opt_labeled(h.verify_roundtrip_ms, "verify_roundtrip_ms"),
                h.ratio,
                spc,
                spd
            );
            if cfg.profile_outer {
                println!(
                    "[bench][outer] run={} mode=CPU_ONLY comp_ms={:.3} decomp_ms={}{}{}",
                    i + 1,
                    c.comp_ms,
                    format_opt(c.decomp_ms),
                    format_opt_labeled(c.verify_compare_ms, "verify_cmp_ms"),
                    format_opt_labeled(c.verify_roundtrip_ms, "verify_roundtrip_ms")
                );
                println!(
                    "[bench][outer] run={} mode={} comp_ms={:.3} decomp_ms={}{}{}",
                    i + 1,
                    hybrid_label,
                    h.comp_ms,
                    format_opt(h.decomp_ms),
                    format_opt_labeled(h.verify_compare_ms, "verify_cmp_ms"),
                    format_opt_labeled(h.verify_roundtrip_ms, "verify_roundtrip_ms")
                );
            }
            cpu_comp_ms.push(c.comp_ms);
            if let Some(v) = c.decomp_ms {
                cpu_decomp_ms.push(v);
            }
            speedup_comp.push(spc);
            if h.decomp_ms.is_some() {
                speedup_decomp.push(spd);
            }
            comp_ms.push(h.comp_ms);
            if let Some(v) = h.decomp_ms {
                decomp_ms.push(v);
            }
            comp_mib_s.push(h.comp_mib_s);
            if let Some(v) = h.decomp_mib_s {
                decomp_mib_s.push(v);
            }
            ratio.push(h.ratio);
        } else {
            let r = run_once(
                &input,
                &hybrid,
                cfg.size_mib,
                cfg.skip_decompress,
                cfg.verify_bytes,
            )?;
            last_hybrid_stats = r.compress_stats;
            last_hybrid_comp_ms = r.comp_ms;
            println!(
                "run {}/{}: comp_ms={:.3} decomp={}{}{} comp_mib_s={:.2} decomp_mib_s={} ratio={:.4}",
                i + 1,
                cfg.runs,
                r.comp_ms,
                format_opt(r.decomp_ms),
                format_opt_labeled(r.verify_compare_ms, "verify_cmp_ms"),
                format_opt_labeled(r.verify_roundtrip_ms, "verify_roundtrip_ms"),
                r.comp_mib_s,
                r.decomp_mib_s
                    .map(|x| format!("{x:.2}"))
                    .unwrap_or_else(|| "SKIP".to_string()),
                r.ratio
            );
            if cfg.profile_outer {
                println!(
                    "[bench][outer] run={} mode=SINGLE comp_ms={:.3} decomp_ms={}{}{}",
                    i + 1,
                    r.comp_ms,
                    format_opt(r.decomp_ms),
                    format_opt_labeled(r.verify_compare_ms, "verify_cmp_ms"),
                    format_opt_labeled(r.verify_roundtrip_ms, "verify_roundtrip_ms")
                );
            }
            comp_ms.push(r.comp_ms);
            if let Some(v) = r.decomp_ms {
                decomp_ms.push(v);
            }
            comp_mib_s.push(r.comp_mib_s);
            if let Some(v) = r.decomp_mib_s {
                decomp_mib_s.push(v);
            }
            ratio.push(r.ratio);
        }
    }

    println!("----- SUMMARY -----");
    println!(
        "comp_ms: n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
        comp_ms.len(),
        mean(&comp_ms),
        median(&comp_ms),
        min(&comp_ms),
        max(&comp_ms)
    );
    if decomp_ms.is_empty() {
        println!("decomp_ms: skipped");
    } else {
        println!(
            "decomp_ms: n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
            decomp_ms.len(),
            mean(&decomp_ms),
            median(&decomp_ms),
            min(&decomp_ms),
            max(&decomp_ms)
        );
    }
    println!(
        "comp_mib_s: n={} mean={:.2} median={:.2} min={:.2} max={:.2}",
        comp_mib_s.len(),
        mean(&comp_mib_s),
        median(&comp_mib_s),
        min(&comp_mib_s),
        max(&comp_mib_s)
    );
    if decomp_mib_s.is_empty() {
        println!("decomp_mib_s: skipped");
    } else {
        println!(
            "decomp_mib_s: n={} mean={:.2} median={:.2} min={:.2} max={:.2}",
            decomp_mib_s.len(),
            mean(&decomp_mib_s),
            median(&decomp_mib_s),
            min(&decomp_mib_s),
            max(&decomp_mib_s)
        );
    }
    println!(
        "ratio: n={} mean={:.4} median={:.4} min={:.4} max={:.4}",
        ratio.len(),
        mean(&ratio),
        median(&ratio),
        min(&ratio),
        max(&ratio)
    );
    if cfg.compare_hybrid {
        println!(
            "cpu_only_comp_ms: n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
            cpu_comp_ms.len(),
            mean(&cpu_comp_ms),
            median(&cpu_comp_ms),
            min(&cpu_comp_ms),
            max(&cpu_comp_ms)
        );
        if cpu_decomp_ms.is_empty() {
            println!("cpu_only_decomp_ms: skipped");
        } else {
            println!(
                "cpu_only_decomp_ms: n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
                cpu_decomp_ms.len(),
                mean(&cpu_decomp_ms),
                median(&cpu_decomp_ms),
                min(&cpu_decomp_ms),
                max(&cpu_decomp_ms)
            );
        }
        println!(
            "speedup_comp(cpu/hybrid): n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
            speedup_comp.len(),
            mean(&speedup_comp),
            median(&speedup_comp),
            min(&speedup_comp),
            max(&speedup_comp)
        );
        if speedup_decomp.is_empty() {
            println!("speedup_decomp(cpu/hybrid): skipped");
        } else {
            println!(
                "speedup_decomp(cpu/hybrid): n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
                speedup_decomp.len(),
                mean(&speedup_decomp),
                median(&speedup_decomp),
                min(&speedup_decomp),
                max(&speedup_decomp)
            );
        }
    }

    println!(
        "gpu_runtime_initialized={}",
        last_hybrid_stats.gpu_available
    );
    println!(
        "[cozip_pdeflate][timing][hybrid-summary] cpu_chunks={} gpu_assigned_chunks={} cpu_chunk_avg_ms={:.3} gpu_chunk_avg_ms_assigned={:.3}",
        last_hybrid_stats.cpu_chunks,
        last_hybrid_stats.gpu_chunks,
        if last_hybrid_stats.cpu_worker_chunks > 0 {
            last_hybrid_stats.cpu_worker_busy_ms / last_hybrid_stats.cpu_worker_chunks as f64
        } else {
            0.0
        },
        if last_hybrid_stats.gpu_worker_chunks > 0 {
            last_hybrid_stats.gpu_worker_busy_ms / last_hybrid_stats.gpu_worker_chunks as f64
        } else {
            0.0
        }
    );
    println!(
        "[cozip_pdeflate][timing][scheduler-probe] gpu_eligible_chunks={} cpu_claimed_gpu_eligible_chunks={} gpu_claimed_chunks={} gpu_encoded_chunks={} gpu_fallback_chunks={}",
        last_hybrid_stats.gpu_eligible_chunks,
        last_hybrid_stats.cpu_claimed_gpu_eligible_chunks,
        last_hybrid_stats.gpu_claimed_chunks,
        last_hybrid_stats.gpu_worker_chunks,
        last_hybrid_stats
            .gpu_claimed_chunks
            .saturating_sub(last_hybrid_stats.gpu_worker_chunks)
    );
    if cfg.compare_hybrid {
        println!(
            "[cozip_pdeflate][timing][cpu-only-probe] comp_ms={:.3} cpu_worker_busy_ms={:.3} cpu_parallelism={:.2} cpu_queue_lock_wait_ms={:.3} cpu_wait_for_task_ms={:.3} writer_wait_ms={:.3} writer_hol_wait_ms={:.3} write_stage_ms={:.3}",
            last_cpu_comp_ms,
            last_cpu_stats.cpu_worker_busy_ms,
            cpu_parallelism_estimate(&last_cpu_stats, last_cpu_comp_ms),
            last_cpu_stats.cpu_queue_lock_wait_ms,
            last_cpu_stats.cpu_wait_for_task_ms,
            last_cpu_stats.writer_wait_ms,
            last_cpu_stats.writer_hol_wait_ms,
            last_cpu_stats.write_stage_ms
        );
    }
    println!(
        "[cozip_pdeflate][timing][hybrid-probe] comp_ms={:.3} cpu_worker_busy_ms={:.3} cpu_parallelism={:.2} gpu_worker_busy_ms={:.3} cpu_queue_lock_wait_ms={:.3} gpu_queue_lock_wait_ms={:.3} cpu_wait_for_task_ms={:.3} gpu_wait_for_task_ms={:.3} writer_wait_ms={:.3} writer_hol_wait_ms={:.3} write_stage_ms={:.3}",
        last_hybrid_comp_ms,
        last_hybrid_stats.cpu_worker_busy_ms,
        cpu_parallelism_estimate(&last_hybrid_stats, last_hybrid_comp_ms),
        last_hybrid_stats.gpu_worker_busy_ms,
        last_hybrid_stats.cpu_queue_lock_wait_ms,
        last_hybrid_stats.gpu_queue_lock_wait_ms,
        last_hybrid_stats.cpu_wait_for_task_ms,
        last_hybrid_stats.gpu_wait_for_task_ms,
        last_hybrid_stats.writer_wait_ms,
        last_hybrid_stats.writer_hol_wait_ms,
        last_hybrid_stats.write_stage_ms
    );
    println!(
        "[cozip_pdeflate][timing][hybrid-gpu-stage] gpu_batch_count={} t_batch_call_ms={:.3} t_prepare_ms={:.3} t_prepare_table_build_ms={:.3} t_prepare_table_readback_ms={:.3} t_prepare_misc_ms={:.3} prepare_gpu_table_build_chunks={} prepare_gpu_table_readback_chunks={} t_profile_gpu_call_ms={:.3} t_finalize_ms={:.3} t_profile_gpu_total_ms={:.3} t_profile_gpu_wait_ms={:.3} t_profile_gpu_upload_ms={:.3} t_profile_gpu_map_copy_ms={:.3} t_match_total_ms={:.3} t_table_build_ms={:.3} t_section_encode_ms={:.3} t_header_pack_ms={:.3} t_fallback_cpu_ms={:.3} t_sparse_prepare_ms={:.3} t_sparse_total_ms={:.3} t_sparse_table_size_resolve_ms={:.3} t_sparse_prepare_misc_ms={:.3} t_sparse_upload_dispatch_ms={:.3} t_sparse_submit_ms={:.3} t_sparse_lens_wait_ms={:.3} t_sparse_lens_copy_ms={:.3} t_sparse_copy_ms={:.3} t_kernel_pack_inputs_ms={:.3} t_kernel_scratch_acquire_ms={:.3} t_kernel_match_dispatch_ms={:.3} t_kernel_match_submit_ms={:.3} t_kernel_section_dispatch_ms={:.3} t_kernel_section_submit_ms={:.3} t_kernel_section_wait_ms={:.3} t_batch_unaccounted_ms={:.3} t_profile_unaccounted_ms={:.3} gpu_profile_total_ms_per_assigned_chunk={:.3}",
        last_hybrid_stats.gpu_batch_count,
        last_hybrid_stats.legacy_gpu_batch_call_ms,
        last_hybrid_stats.legacy_gpu_prepare_ms,
        last_hybrid_stats.legacy_gpu_prepare_table_build_ms,
        last_hybrid_stats.legacy_gpu_prepare_table_readback_ms,
        last_hybrid_stats.legacy_gpu_prepare_misc_ms,
        last_hybrid_stats.legacy_gpu_prepare_gpu_table_build_chunks,
        last_hybrid_stats.legacy_gpu_prepare_gpu_table_readback_chunks,
        last_hybrid_stats.legacy_gpu_profile_call_ms,
        last_hybrid_stats.legacy_gpu_finalize_ms,
        last_hybrid_stats.legacy_gpu_total_ms,
        last_hybrid_stats.legacy_gpu_wait_ms,
        last_hybrid_stats.legacy_gpu_upload_ms,
        last_hybrid_stats.legacy_gpu_map_copy_ms,
        last_hybrid_stats.legacy_gpu_match_total_ms,
        last_hybrid_stats.legacy_gpu_table_build_ms,
        last_hybrid_stats.legacy_gpu_section_encode_ms,
        last_hybrid_stats.legacy_gpu_header_pack_ms,
        last_hybrid_stats.legacy_gpu_fallback_cpu_ms,
        last_hybrid_stats.legacy_gpu_sparse_prepare_ms,
        last_hybrid_stats.legacy_gpu_sparse_total_ms,
        last_hybrid_stats.legacy_gpu_sparse_table_size_resolve_ms,
        last_hybrid_stats.legacy_gpu_sparse_prepare_misc_ms,
        last_hybrid_stats.legacy_gpu_sparse_upload_dispatch_ms,
        last_hybrid_stats.legacy_gpu_sparse_submit_ms,
        last_hybrid_stats.legacy_gpu_sparse_lens_wait_ms,
        last_hybrid_stats.legacy_gpu_sparse_lens_copy_ms,
        last_hybrid_stats.legacy_gpu_sparse_copy_ms,
        last_hybrid_stats.legacy_gpu_kernel_pack_inputs_ms,
        last_hybrid_stats.legacy_gpu_kernel_scratch_acquire_ms,
        last_hybrid_stats.legacy_gpu_kernel_match_dispatch_ms,
        last_hybrid_stats.legacy_gpu_kernel_match_submit_ms,
        last_hybrid_stats.legacy_gpu_kernel_section_dispatch_ms,
        last_hybrid_stats.legacy_gpu_kernel_section_submit_ms,
        last_hybrid_stats.legacy_gpu_kernel_section_wait_ms,
        (last_hybrid_stats.legacy_gpu_batch_call_ms
            - (last_hybrid_stats.legacy_gpu_prepare_ms
                + last_hybrid_stats.legacy_gpu_profile_call_ms
                + last_hybrid_stats.legacy_gpu_finalize_ms))
            .max(0.0),
        (last_hybrid_stats.legacy_gpu_profile_call_ms - last_hybrid_stats.legacy_gpu_total_ms)
            .max(0.0),
        if last_hybrid_stats.gpu_chunks > 0 {
            last_hybrid_stats.legacy_gpu_total_ms / last_hybrid_stats.gpu_chunks as f64
        } else {
            0.0
        }
    );
    let k1_pack_inputs_ms = last_hybrid_stats.legacy_gpu_kernel_pack_inputs_ms;
    let k2_scratch_ms = last_hybrid_stats.legacy_gpu_kernel_scratch_acquire_ms;
    let k3_match_ms = last_hybrid_stats.legacy_gpu_kernel_match_dispatch_ms
        + last_hybrid_stats.legacy_gpu_kernel_match_submit_ms;
    let k4_section_ms = last_hybrid_stats.legacy_gpu_kernel_section_dispatch_ms
        + last_hybrid_stats.legacy_gpu_kernel_section_submit_ms
        + last_hybrid_stats.legacy_gpu_kernel_section_wait_ms;
    let k5_sparse_ms = last_hybrid_stats.legacy_gpu_sparse_total_ms;
    let call_known_ms =
        k1_pack_inputs_ms + k2_scratch_ms + k3_match_ms + k4_section_ms + k5_sparse_ms;
    let call_other_ms = (last_hybrid_stats.legacy_gpu_profile_call_ms - call_known_ms).max(0.0);
    let profile_call_ms = last_hybrid_stats.legacy_gpu_profile_call_ms.max(1e-9);
    let c1_match_profile_ms = last_hybrid_stats.legacy_gpu_match_total_ms;
    let c2_section_profile_ms = last_hybrid_stats.legacy_gpu_section_encode_ms;
    let c3_header_pack_ms = last_hybrid_stats.legacy_gpu_header_pack_ms;
    let c4_sparse_profile_ms = last_hybrid_stats.legacy_gpu_sparse_total_ms;
    let profile_stage_known_ms =
        c1_match_profile_ms + c2_section_profile_ms + c3_header_pack_ms + c4_sparse_profile_ms;
    let profile_stage_other_ms =
        (last_hybrid_stats.legacy_gpu_profile_call_ms - profile_stage_known_ms).max(0.0);
    let t1_upload_ms = last_hybrid_stats.legacy_gpu_upload_ms;
    let t2_wait_ms = last_hybrid_stats.legacy_gpu_wait_ms;
    let t3_map_copy_ms = last_hybrid_stats.legacy_gpu_map_copy_ms;
    let profile_transport_known_ms = t1_upload_ms + t2_wait_ms + t3_map_copy_ms;
    let profile_transport_other_ms =
        (last_hybrid_stats.legacy_gpu_total_ms - profile_transport_known_ms).max(0.0);
    let call_minus_profile_total_ms = (last_hybrid_stats.legacy_gpu_profile_call_ms
        - last_hybrid_stats.legacy_gpu_total_ms)
        .max(0.0);
    println!(
        "[cozip_pdeflate][timing][hybrid-gpu-call-breakdown] k1_pack_inputs_ms={:.3} k2_scratch_ms={:.3} k3_match_stage_ms={:.3} k4_section_stage_ms={:.3} k5_sparse_stage_ms={:.3} call_other_ms={:.3} k1_pct={:.1} k2_pct={:.1} k3_pct={:.1} k4_pct={:.1} k5_pct={:.1} other_pct={:.1}",
        k1_pack_inputs_ms,
        k2_scratch_ms,
        k3_match_ms,
        k4_section_ms,
        k5_sparse_ms,
        call_other_ms,
        100.0 * k1_pack_inputs_ms / profile_call_ms,
        100.0 * k2_scratch_ms / profile_call_ms,
        100.0 * k3_match_ms / profile_call_ms,
        100.0 * k4_section_ms / profile_call_ms,
        100.0 * k5_sparse_ms / profile_call_ms,
        100.0 * call_other_ms / profile_call_ms
    );
    println!(
        "[cozip_pdeflate][timing][hybrid-gpu-call-accounting] profile_call_ms={:.3} profile_total_ms={:.3} stage_match_ms={:.3} stage_section_ms={:.3} stage_header_pack_ms={:.3} stage_sparse_ms={:.3} stage_other_ms={:.3} upload_ms={:.3} wait_ms={:.3} map_copy_ms={:.3} transport_other_ms={:.3} call_minus_profile_total_ms={:.3}",
        last_hybrid_stats.legacy_gpu_profile_call_ms,
        last_hybrid_stats.legacy_gpu_total_ms,
        c1_match_profile_ms,
        c2_section_profile_ms,
        c3_header_pack_ms,
        c4_sparse_profile_ms,
        profile_stage_other_ms,
        t1_upload_ms,
        t2_wait_ms,
        t3_map_copy_ms,
        profile_transport_other_ms,
        call_minus_profile_total_ms
    );
    let post_total_pack_bytes_ms = last_hybrid_stats.legacy_gpu_post_total_pack_bytes_ms;
    let post_scale_profiles_ms = last_hybrid_stats.legacy_gpu_post_scale_profiles_ms;
    let post_chunk_materialize_ms = last_hybrid_stats.legacy_gpu_post_chunk_materialize_ms;
    let post_total_ms =
        post_total_pack_bytes_ms + post_scale_profiles_ms + post_chunk_materialize_ms;
    let residual_after_post_ms = (call_minus_profile_total_ms - post_total_ms).max(0.0);
    let stage_other_after_post_ms = (profile_stage_other_ms - post_total_ms).max(0.0);
    println!(
        "[cozip_pdeflate][timing][hybrid-gpu-call-post] post_total_pack_bytes_ms={:.3} post_scale_profiles_ms={:.3} post_chunk_materialize_ms={:.3} post_total_ms={:.3} residual_after_post_ms={:.3} stage_other_after_post_ms={:.3}",
        post_total_pack_bytes_ms,
        post_scale_profiles_ms,
        post_chunk_materialize_ms,
        post_total_ms,
        residual_after_post_ms,
        stage_other_after_post_ms
    );
    let unaccounted_pack_bind_group_ms = last_hybrid_stats.legacy_gpu_call_pack_bind_group_ms;
    let unaccounted_encoder_create_ms = last_hybrid_stats.legacy_gpu_call_encoder_create_ms;
    let unaccounted_table_stream_copy_ms = last_hybrid_stats.legacy_gpu_call_table_stream_copy_ms;
    let unaccounted_match_clear_ms = last_hybrid_stats.legacy_gpu_call_match_clear_ms;
    let unaccounted_sparse_prepare_dispatch_ms =
        last_hybrid_stats.legacy_gpu_call_sparse_prepare_dispatch_ms;
    let unaccounted_sparse_pack_dispatch_ms =
        last_hybrid_stats.legacy_gpu_call_sparse_pack_dispatch_ms;
    let unaccounted_result_readback_setup_ms =
        last_hybrid_stats.legacy_gpu_call_result_readback_setup_ms;
    let unaccounted_payload_readback_setup_ms =
        last_hybrid_stats.legacy_gpu_call_payload_readback_setup_ms;
    let unaccounted_known_ms = unaccounted_pack_bind_group_ms
        + unaccounted_encoder_create_ms
        + unaccounted_table_stream_copy_ms
        + unaccounted_match_clear_ms
        + unaccounted_sparse_prepare_dispatch_ms
        + unaccounted_sparse_pack_dispatch_ms
        + unaccounted_result_readback_setup_ms
        + unaccounted_payload_readback_setup_ms;
    let unaccounted_residual_ms = (call_minus_profile_total_ms - unaccounted_known_ms).max(0.0);
    println!(
        "[cozip_pdeflate][timing][hybrid-gpu-call-unaccounted] pack_bind_group_ms={:.3} encoder_create_ms={:.3} table_stream_copy_ms={:.3} match_clear_ms={:.3} sparse_prepare_dispatch_ms={:.3} sparse_pack_dispatch_ms={:.3} result_readback_setup_ms={:.3} payload_readback_setup_ms={:.3} known_ms={:.3} residual_ms={:.3}",
        unaccounted_pack_bind_group_ms,
        unaccounted_encoder_create_ms,
        unaccounted_table_stream_copy_ms,
        unaccounted_match_clear_ms,
        unaccounted_sparse_prepare_dispatch_ms,
        unaccounted_sparse_pack_dispatch_ms,
        unaccounted_result_readback_setup_ms,
        unaccounted_payload_readback_setup_ms,
        unaccounted_known_ms,
        unaccounted_residual_ms
    );
    println!(
        "[cozip_pdeflate][timing][hybrid-gpu-match-breakdown] table_copy_ms={:.3} prepare_dispatch_ms={:.3} kernel_dispatch_ms={:.3} submit_ms={:.3} subtotal_ms={:.3} match_total_ms={:.3} residual_ms={:.3}",
        last_hybrid_stats.legacy_gpu_match_table_copy_ms,
        last_hybrid_stats.legacy_gpu_match_prepare_dispatch_ms,
        last_hybrid_stats.legacy_gpu_match_kernel_dispatch_ms,
        last_hybrid_stats.legacy_gpu_match_submit_ms,
        last_hybrid_stats.legacy_gpu_match_table_copy_ms
            + last_hybrid_stats.legacy_gpu_match_prepare_dispatch_ms
            + last_hybrid_stats.legacy_gpu_match_kernel_dispatch_ms
            + last_hybrid_stats.legacy_gpu_match_submit_ms,
        last_hybrid_stats.legacy_gpu_match_total_ms,
        (last_hybrid_stats.legacy_gpu_match_total_ms
            - (last_hybrid_stats.legacy_gpu_match_table_copy_ms
                + last_hybrid_stats.legacy_gpu_match_prepare_dispatch_ms
                + last_hybrid_stats.legacy_gpu_match_kernel_dispatch_ms
                + last_hybrid_stats.legacy_gpu_match_submit_ms))
            .max(0.0)
    );
    println!(
        "[cozip_pdeflate][timing][hybrid-gpu-paths] gpu_assigned_chunks={} integrated_chunks={} fallback_match_chunks={} integrated_pct_assigned={:.1} fallback_pct_assigned={:.1}",
        last_hybrid_stats.gpu_chunks,
        last_hybrid_stats.legacy_gpu_integrated_chunks,
        last_hybrid_stats.legacy_gpu_fallback_match_chunks,
        if last_hybrid_stats.gpu_chunks > 0 {
            100.0 * (last_hybrid_stats.legacy_gpu_integrated_chunks as f64)
                / (last_hybrid_stats.gpu_chunks as f64)
        } else {
            0.0
        },
        if last_hybrid_stats.gpu_chunks > 0 {
            100.0 * (last_hybrid_stats.legacy_gpu_fallback_match_chunks as f64)
                / (last_hybrid_stats.gpu_chunks as f64)
        } else {
            0.0
        }
    );
    println!(
        "[cozip_pdeflate][timing][hybrid-gpu-fallbacks] integrated_error_chunks={} invalid_options_chunks={} invalid_stream_chunks={} numeric_overflow_chunks={} gpu_error_chunks={} other_error_chunks={} integrated_output_mismatch_chunks={} fallback_match_error_chunks={} fallback_match_output_mismatch_chunks={} fallback_match_packed_len_mismatch_chunks={}",
        last_hybrid_stats.legacy_gpu_fallback_integrated_error_chunks,
        last_hybrid_stats.legacy_gpu_fallback_integrated_invalid_options_chunks,
        last_hybrid_stats.legacy_gpu_fallback_integrated_invalid_stream_chunks,
        last_hybrid_stats.legacy_gpu_fallback_integrated_numeric_overflow_chunks,
        last_hybrid_stats.legacy_gpu_fallback_integrated_gpu_error_chunks,
        last_hybrid_stats.legacy_gpu_fallback_integrated_other_error_chunks,
        last_hybrid_stats.legacy_gpu_fallback_integrated_output_mismatch_chunks,
        last_hybrid_stats.legacy_gpu_fallback_match_error_chunks,
        last_hybrid_stats.legacy_gpu_fallback_match_output_mismatch_chunks,
        last_hybrid_stats.legacy_gpu_fallback_match_packed_len_mismatch_chunks
    );
    let pass_match_ms = last_hybrid_stats.legacy_gpu_match_total_ms.max(
        last_hybrid_stats.legacy_gpu_kernel_match_dispatch_ms
            + last_hybrid_stats.legacy_gpu_kernel_match_submit_ms,
    );
    let pass_section_ms = last_hybrid_stats.legacy_gpu_section_encode_ms.max(
        last_hybrid_stats.legacy_gpu_kernel_section_dispatch_ms
            + last_hybrid_stats.legacy_gpu_kernel_section_submit_ms
            + last_hybrid_stats.legacy_gpu_kernel_section_wait_ms,
    );
    let pass_header_pack_ms = last_hybrid_stats.legacy_gpu_header_pack_ms;
    let pass_sparse_ms = last_hybrid_stats.legacy_gpu_sparse_total_ms;
    let pass_transport_ms = last_hybrid_stats.legacy_gpu_total_ms;
    let pass_total_ms =
        pass_match_ms + pass_section_ms + pass_header_pack_ms + pass_sparse_ms + pass_transport_ms;
    let pass_residual_ms = (last_hybrid_stats.legacy_gpu_profile_call_ms - pass_total_ms).max(0.0);
    println!(
        "[cozip_pdeflate][timing][hybrid-gpu-call-pass-groups] match_ms={:.3} section_ms={:.3} header_pack_ms={:.3} sparse_ms={:.3} transport_ms={:.3} grouped_ms={:.3} residual_ms={:.3}",
        pass_match_ms,
        pass_section_ms,
        pass_header_pack_ms,
        pass_sparse_ms,
        pass_transport_ms,
        pass_total_ms,
        pass_residual_ms
    );
    let p1_alloc_ms = last_hybrid_stats.legacy_gpu_kernel_pack_alloc_setup_ms;
    let p2_resolve_ms = last_hybrid_stats.legacy_gpu_kernel_pack_resolve_sizes_ms;
    let p3_src_copy_ms = last_hybrid_stats.legacy_gpu_kernel_pack_src_copy_ms;
    let p4_metadata_ms = last_hybrid_stats.legacy_gpu_kernel_pack_metadata_loop_ms;
    let p5_host_copy_ms = last_hybrid_stats.legacy_gpu_kernel_pack_host_copy_ms;
    let p6_device_plan_ms = last_hybrid_stats.legacy_gpu_kernel_pack_device_copy_plan_ms;
    let p7_finalize_ms = last_hybrid_stats.legacy_gpu_kernel_pack_finalize_ms;
    let pack_known_ms = p1_alloc_ms
        + p2_resolve_ms
        + p3_src_copy_ms
        + p4_metadata_ms
        + p5_host_copy_ms
        + p6_device_plan_ms
        + p7_finalize_ms;
    let pack_other_ms = (k1_pack_inputs_ms - pack_known_ms).max(0.0);
    let pack_total_ms = k1_pack_inputs_ms.max(1e-9);
    println!(
        "[cozip_pdeflate][timing][hybrid-pack-breakdown] pack_inputs_ms={:.3} alloc_setup_ms={:.3} resolve_sizes_ms={:.3} src_copy_ms={:.3} metadata_loop_ms={:.3} host_copy_ms={:.3} device_copy_plan_ms={:.3} finalize_ms={:.3} other_ms={:.3} alloc_pct={:.1} resolve_pct={:.1} src_copy_pct={:.1} metadata_pct={:.1} host_copy_pct={:.1} device_plan_pct={:.1} finalize_pct={:.1} other_pct={:.1}",
        k1_pack_inputs_ms,
        p1_alloc_ms,
        p2_resolve_ms,
        p3_src_copy_ms,
        p4_metadata_ms,
        p5_host_copy_ms,
        p6_device_plan_ms,
        p7_finalize_ms,
        pack_other_ms,
        100.0 * p1_alloc_ms / pack_total_ms,
        100.0 * p2_resolve_ms / pack_total_ms,
        100.0 * p3_src_copy_ms / pack_total_ms,
        100.0 * p4_metadata_ms / pack_total_ms,
        100.0 * p5_host_copy_ms / pack_total_ms,
        100.0 * p6_device_plan_ms / pack_total_ms,
        100.0 * p7_finalize_ms / pack_total_ms,
        100.0 * pack_other_ms / pack_total_ms
    );
    let r1_scan_ms = last_hybrid_stats.legacy_gpu_kernel_pack_resolve_scan_ms;
    let r2_readback_setup_ms = last_hybrid_stats.legacy_gpu_kernel_pack_resolve_readback_setup_ms;
    let r3_submit_ms = last_hybrid_stats.legacy_gpu_kernel_pack_resolve_submit_ms;
    let r4_map_wait_ms = last_hybrid_stats.legacy_gpu_kernel_pack_resolve_map_wait_ms;
    let r5_parse_ms = last_hybrid_stats.legacy_gpu_kernel_pack_resolve_parse_ms;
    let resolve_known_ms =
        r1_scan_ms + r2_readback_setup_ms + r3_submit_ms + r4_map_wait_ms + r5_parse_ms;
    let resolve_other_ms = (p2_resolve_ms - resolve_known_ms).max(0.0);
    let resolve_total_ms = p2_resolve_ms.max(1e-9);
    println!(
        "[cozip_pdeflate][timing][hybrid-pack-resolve-breakdown] resolve_sizes_ms={:.3} scan_ms={:.3} readback_setup_ms={:.3} submit_ms={:.3} map_wait_ms={:.3} parse_ms={:.3} other_ms={:.3} scan_pct={:.1} readback_setup_pct={:.1} submit_pct={:.1} map_wait_pct={:.1} parse_pct={:.1} other_pct={:.1}",
        p2_resolve_ms,
        r1_scan_ms,
        r2_readback_setup_ms,
        r3_submit_ms,
        r4_map_wait_ms,
        r5_parse_ms,
        resolve_other_ms,
        100.0 * r1_scan_ms / resolve_total_ms,
        100.0 * r2_readback_setup_ms / resolve_total_ms,
        100.0 * r3_submit_ms / resolve_total_ms,
        100.0 * r4_map_wait_ms / resolve_total_ms,
        100.0 * r5_parse_ms / resolve_total_ms,
        100.0 * resolve_other_ms / resolve_total_ms
    );
    let c1_table_size_resolve_ms = last_hybrid_stats.legacy_gpu_sparse_table_size_resolve_ms;
    let c2_dispatch_submit_ms = last_hybrid_stats.legacy_gpu_sparse_upload_dispatch_ms
        + last_hybrid_stats.legacy_gpu_sparse_submit_ms;
    let c3_map_wait_ms = last_hybrid_stats.legacy_gpu_sparse_lens_wait_ms;
    let c4_cpu_copy_ms = last_hybrid_stats.legacy_gpu_sparse_lens_copy_ms
        + last_hybrid_stats.legacy_gpu_sparse_copy_ms;
    println!(
        "[cozip_pdeflate][timing][hybrid-gpu-culprit] c1_table_size_resolve_ms={:.3} c2_dispatch_submit_ms={:.3} c3_map_wait_ms={:.3} c4_cpu_copy_ms={:.3} c1_pct={:.1} c2_pct={:.1} c3_pct={:.1} c4_pct={:.1}",
        c1_table_size_resolve_ms,
        c2_dispatch_submit_ms,
        c3_map_wait_ms,
        c4_cpu_copy_ms,
        100.0 * c1_table_size_resolve_ms / profile_call_ms,
        100.0 * c2_dispatch_submit_ms / profile_call_ms,
        100.0 * c3_map_wait_ms / profile_call_ms,
        100.0 * c4_cpu_copy_ms / profile_call_ms
    );

    Ok(())
}
