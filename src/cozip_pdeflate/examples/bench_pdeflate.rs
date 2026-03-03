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
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RunResult {
    comp_ms: f64,
    decomp_ms: Option<f64>,
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
  --verify / --no-verify / --verify-bytes <0|1>"
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
    if !skip_decompress {
        let mut restored = Vec::with_capacity(input.len());
        let mut reader = std::io::Cursor::new(&compressed);
        let d0 = Instant::now();
        if let Some(index) = compress.index.as_ref() {
            cozip
                .deflate_decompress_stream_zip_compatible_with_index(
                    &mut reader,
                    &mut restored,
                    index,
                )
                .map_err(|e| e.to_string())?;
        } else {
            cozip
                .pdeflate_decompress_stream(&mut reader, &mut restored)
                .map_err(|e| e.to_string())?;
        }
        let d_ms = d0.elapsed().as_secs_f64() * 1000.0;
        if verify_bytes && restored != input {
            return Err("roundtrip mismatch".to_string());
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
        let mut reader = std::io::Cursor::new(&compressed);
        if let Some(index) = compress.index.as_ref() {
            cozip
                .deflate_decompress_stream_zip_compatible_with_index(
                    &mut reader,
                    &mut restored,
                    index,
                )
                .map_err(|e| e.to_string())?;
        } else {
            cozip
                .pdeflate_decompress_stream(&mut reader, &mut restored)
                .map_err(|e| e.to_string())?;
        }
        if restored != input {
            return Err("roundtrip mismatch".to_string());
        }
    }

    Ok(RunResult {
        comp_ms,
        decomp_ms,
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
                "run {}/{}: CPU_ONLY comp_ms={:.3} decomp={} | {} comp_ms={:.3} decomp={} ratio={:.4} speedup_comp={:.3}x speedup_decomp={:.3}x",
                i + 1,
                cfg.runs,
                c.comp_ms,
                format_opt(c.decomp_ms),
                hybrid_label,
                h.comp_ms,
                format_opt(h.decomp_ms),
                h.ratio,
                spc,
                spd
            );
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
                "run {}/{}: comp_ms={:.3} decomp={} comp_mib_s={:.2} decomp_mib_s={} ratio={:.4}",
                i + 1,
                cfg.runs,
                r.comp_ms,
                format_opt(r.decomp_ms),
                r.comp_mib_s,
                r.decomp_mib_s
                    .map(|x| format!("{x:.2}"))
                    .unwrap_or_else(|| "SKIP".to_string()),
                r.ratio
            );
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
        "[cozip_pdeflate][timing][hybrid-gpu-stage] gpu_batch_count={} t_batch_call_ms={:.3} t_prepare_ms={:.3} t_profile_gpu_call_ms={:.3} t_finalize_ms={:.3} t_profile_gpu_total_ms={:.3} t_profile_gpu_wait_ms={:.3} t_profile_gpu_upload_ms={:.3} t_profile_gpu_map_copy_ms={:.3} t_table_build_ms={:.3} t_section_encode_ms={:.3} t_header_pack_ms={:.3} t_fallback_cpu_ms={:.3} t_batch_unaccounted_ms={:.3} t_profile_unaccounted_ms={:.3} gpu_profile_total_ms_per_assigned_chunk={:.3}",
        last_hybrid_stats.gpu_batch_count,
        last_hybrid_stats.legacy_gpu_batch_call_ms,
        last_hybrid_stats.legacy_gpu_prepare_ms,
        last_hybrid_stats.legacy_gpu_profile_call_ms,
        last_hybrid_stats.legacy_gpu_finalize_ms,
        last_hybrid_stats.legacy_gpu_total_ms,
        last_hybrid_stats.legacy_gpu_wait_ms,
        last_hybrid_stats.legacy_gpu_upload_ms,
        last_hybrid_stats.legacy_gpu_map_copy_ms,
        last_hybrid_stats.legacy_gpu_table_build_ms,
        last_hybrid_stats.legacy_gpu_section_encode_ms,
        last_hybrid_stats.legacy_gpu_header_pack_ms,
        last_hybrid_stats.legacy_gpu_fallback_cpu_ms,
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

    Ok(())
}
