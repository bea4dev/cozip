use std::env;
use std::io::Cursor;
use std::time::{Duration, Instant};

use cozip_deflate::{
    CoZipDeflate, CompressionMode, DeflateCpuStreamStats, HybridOptions, HybridSchedulerPolicy,
    deflate_decompress_stream_on_cpu,
};

#[derive(Debug, Clone)]
struct BenchConfig {
    size_mib: usize,
    iters: usize,
    warmups: usize,
    chunk_mib: usize,
    gpu_subchunk_kib: usize,
    token_finalize_segment_size: usize,
    gpu_slots: usize,
    gpu_batch_chunks: usize,
    decomp_gpu_batch_chunks: usize,
    gpu_submit_chunks: usize,
    stream_pipeline_depth: usize,
    stream_batch_chunks: usize,
    stream_max_inflight_chunks: usize,
    stream_max_inflight_mib: usize,
    scheduler_policy: HybridSchedulerPolicy,
    gpu_fraction: f32,
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
            iters: 1,
            warmups: 0,
            chunk_mib: 4,
            gpu_subchunk_kib: 512,
            token_finalize_segment_size: 4096,
            gpu_slots: 6,
            gpu_batch_chunks: 6,
            decomp_gpu_batch_chunks: 0,
            gpu_submit_chunks: 3,
            stream_pipeline_depth: 3,
            stream_batch_chunks: 0,
            stream_max_inflight_chunks: 0,
            stream_max_inflight_mib: 0,
            scheduler_policy: HybridSchedulerPolicy::GlobalQueueLocalBuffers,
            gpu_fraction: 1.0,
            gpu_tail_stop_ratio: 1.0,
            mode: CompressionMode::Ratio,
            profile_timing: env_flag("COZIP_PROFILE_TIMING"),
            profile_timing_detail: env_flag("COZIP_PROFILE_TIMING_DETAIL"),
            profile_timing_deep: env_flag("COZIP_PROFILE_DEEP"),
        }
    }
}

fn env_flag(name: &str) -> bool {
    match env::var(name) {
        Ok(value) => {
            let lowered = value.trim().to_ascii_lowercase();
            !(lowered.is_empty() || lowered == "0" || lowered == "false" || lowered == "off")
        }
        Err(_) => false,
    }
}

impl BenchConfig {
    fn from_args() -> Result<Self, String> {
        let mut cfg = Self::default();
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            if arg == "--help" || arg == "-h" {
                return Err(help_text());
            }

            match arg.as_str() {
                "--profile-timing" => {
                    cfg.profile_timing = true;
                    continue;
                }
                "--profile-timing-detail" => {
                    cfg.profile_timing = true;
                    cfg.profile_timing_detail = true;
                    continue;
                }
                "--profile-timing-deep" => {
                    cfg.profile_timing = true;
                    cfg.profile_timing_deep = true;
                    continue;
                }
                _ => {}
            }

            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {}", arg))?;

            match arg.as_str() {
                "--size-mib" => {
                    cfg.size_mib = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --size-mib".to_string())?;
                }
                "--iters" => {
                    cfg.iters = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --iters".to_string())?;
                }
                "--warmups" => {
                    cfg.warmups = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --warmups".to_string())?;
                }
                "--chunk-mib" => {
                    cfg.chunk_mib = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --chunk-mib".to_string())?;
                }
                "--gpu-subchunk-kib" => {
                    cfg.gpu_subchunk_kib = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --gpu-subchunk-kib".to_string())?;
                }
                "--token-finalize-segment-size" => {
                    cfg.token_finalize_segment_size = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --token-finalize-segment-size".to_string())?;
                }
                "--gpu-slots" => {
                    cfg.gpu_slots = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --gpu-slots".to_string())?;
                }
                "--gpu-submit-chunks" => {
                    cfg.gpu_submit_chunks = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --gpu-submit-chunks".to_string())?;
                }
                "--gpu-batch-chunks" => {
                    cfg.gpu_batch_chunks = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --gpu-batch-chunks".to_string())?;
                }
                "--decomp-gpu-batch-chunks" => {
                    cfg.decomp_gpu_batch_chunks = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --decomp-gpu-batch-chunks".to_string())?;
                }
                "--stream-pipeline-depth" => {
                    cfg.stream_pipeline_depth = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --stream-pipeline-depth".to_string())?;
                }
                "--stream-batch-chunks" => {
                    cfg.stream_batch_chunks = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --stream-batch-chunks".to_string())?;
                    if cfg.stream_batch_chunks != 0 {
                        return Err(
                            "--stream-batch-chunks is fixed to 0 (legacy batch mode was removed)"
                                .to_string(),
                        );
                    }
                }
                "--stream-max-inflight-chunks" => {
                    cfg.stream_max_inflight_chunks = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --stream-max-inflight-chunks".to_string())?;
                }
                "--stream-max-inflight-mib" => {
                    cfg.stream_max_inflight_mib = value
                        .parse::<usize>()
                        .map_err(|_| "invalid --stream-max-inflight-mib".to_string())?;
                }
                "--scheduler" => {
                    cfg.scheduler_policy = match value.as_str() {
                        "global-local" => HybridSchedulerPolicy::GlobalQueueLocalBuffers,
                        _ => {
                            return Err("invalid --scheduler (expected: global-local)".to_string());
                        }
                    };
                }
                "--gpu-fraction" => {
                    cfg.gpu_fraction = value
                        .parse::<f32>()
                        .map_err(|_| "invalid --gpu-fraction".to_string())?;
                }
                "--gpu-tail-stop-ratio" => {
                    cfg.gpu_tail_stop_ratio = value
                        .parse::<f32>()
                        .map_err(|_| "invalid --gpu-tail-stop-ratio".to_string())?;
                }
                "--mode" => {
                    cfg.mode = match value.as_str() {
                        "speed" => CompressionMode::Speed,
                        "balanced" => CompressionMode::Balanced,
                        "ratio" => CompressionMode::Ratio,
                        _ => {
                            return Err(
                                "invalid --mode (expected: speed|balanced|ratio)".to_string()
                            );
                        }
                    };
                }
                _ => {
                    return Err(format!("unknown argument: {}\n{}", arg, help_text()));
                }
            }
        }

        if cfg.size_mib == 0
            || cfg.iters == 0
            || cfg.chunk_mib == 0
            || cfg.gpu_subchunk_kib == 0
            || cfg.token_finalize_segment_size == 0
            || cfg.gpu_slots == 0
            || cfg.gpu_batch_chunks == 0
            || cfg.gpu_submit_chunks == 0
            || cfg.stream_pipeline_depth == 0
        {
            return Err(
                "size/iters/chunk/subchunk/token-finalize-segment-size/gpu-batch-chunks/gpu-submit-chunks must be > 0".to_string(),
            );
        }
        if !(0.0..=1.0).contains(&cfg.gpu_fraction) {
            return Err("gpu-fraction must be in range 0.0..=1.0".to_string());
        }
        if !(0.0..=1.0).contains(&cfg.gpu_tail_stop_ratio) {
            return Err("gpu-tail-stop-ratio must be in range 0.0..=1.0".to_string());
        }
        if cfg.stream_batch_chunks != 0 {
            return Err(
                "--stream-batch-chunks is fixed to 0 (legacy batch mode was removed)".to_string(),
            );
        }
        if cfg.profile_timing_detail || cfg.profile_timing_deep {
            cfg.profile_timing = true;
        }

        Ok(cfg)
    }
}

fn help_text() -> String {
    let text = r#"usage: cargo run --release -p cozip_deflate --example bench_1gb -- [options]
  --size-mib <N>           input size in MiB (default: 4096)
  --iters <N>              measured iterations (default: 1)
  --warmups <N>            warmup iterations (default: 0)
  --chunk-mib <N>          host chunk size in MiB (default: 4)
  --gpu-subchunk-kib <N>   gpu subchunk size in KiB (default: 512)
  --token-finalize-segment-size <N> token finalize segment size (default: 4096)
  --gpu-slots <N>          gpu slot/batch count (default: 6)
  --gpu-batch-chunks <N>   gpu dequeue batch size (default: 6)
  --decomp-gpu-batch-chunks <N> decode-side gpu dequeue batch size (0: unlimited, default: 0)
  --gpu-submit-chunks <N>  gpu submit group size (default: 3)
  --stream-pipeline-depth <N>  stream prepare pipeline depth (default: 3)
  --stream-batch-chunks <N>    fixed to 0 (legacy batch mode removed)
  --stream-max-inflight-chunks <N> inflight chunk cap in continuous mode (default: 0, unlimited)
  --stream-max-inflight-mib <N> inflight raw MiB cap in continuous mode (default: 0, disabled)
  --scheduler <S>          global-local only
  --gpu-fraction <R>       gpu scheduling target ratio 0.0..=1.0 (default: 1.0)
  --gpu-tail-stop-ratio <R>  stop new GPU dequeues when progress reaches ratio (default: 1.0; disabled)
  --mode <M>               speed|balanced|ratio (default: ratio)
  --profile-timing         enable GPU timing logs
  --profile-timing-detail  enable detailed GPU timing logs
  --profile-timing-deep    enable one-shot deep GPU timing probe"#;
    text.to_string()
}

#[derive(Debug, Clone, Default)]
struct BenchAgg {
    compress_total: Duration,
    decompress_total: Duration,
    compressed_total: usize,
    input_total: usize,
    last_compress_stats: DeflateCpuStreamStats,
    last_decompress_stats: DeflateCpuStreamStats,
}

impl BenchAgg {
    fn add(
        &mut self,
        input_len: usize,
        compressed_len: usize,
        compress_elapsed: Duration,
        decompress_elapsed: Duration,
        compress_stats: DeflateCpuStreamStats,
        decompress_stats: DeflateCpuStreamStats,
    ) {
        self.compress_total += compress_elapsed;
        self.decompress_total += decompress_elapsed;
        self.compressed_total += compressed_len;
        self.input_total += input_len;
        self.last_compress_stats = compress_stats;
        self.last_decompress_stats = decompress_stats;
    }

    fn avg_compress_ms(&self, iters: usize) -> f64 {
        self.compress_total.as_secs_f64() * 1000.0 / iters as f64
    }

    fn avg_decompress_ms(&self, iters: usize) -> f64 {
        self.decompress_total.as_secs_f64() * 1000.0 / iters as f64
    }

    fn ratio(&self) -> f64 {
        if self.input_total == 0 {
            return 0.0;
        }
        self.compressed_total as f64 / self.input_total as f64
    }

    fn compress_mib_s(&self) -> f64 {
        if self.compress_total.is_zero() {
            return 0.0;
        }
        (self.input_total as f64 / (1024.0 * 1024.0)) / self.compress_total.as_secs_f64()
    }

    fn decompress_mib_s(&self) -> f64 {
        if self.decompress_total.is_zero() {
            return 0.0;
        }
        (self.input_total as f64 / (1024.0 * 1024.0)) / self.decompress_total.as_secs_f64()
    }
}

fn build_mixed_dataset(bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes);
    let mut state: u32 = 0x1234_5678;

    while out.len() < bytes {
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

    out.truncate(bytes);
    out
}

fn run_case(input: &[u8], cozip: &CoZipDeflate, iters: usize) -> BenchAgg {
    let mut agg = BenchAgg::default();
    for _ in 0..iters {
        let mut src = Cursor::new(input);
        let mut compressed = Vec::new();
        let t0 = Instant::now();
        let compress_result = cozip
            .deflate_compress_stream_zip_compatible_with_index(&mut src, &mut compressed)
            .expect("compress should succeed");
        let compress_stats = compress_result.stats;
        let compress_elapsed = t0.elapsed();

        let mut compressed_reader = Cursor::new(&compressed);
        let mut restored = Vec::with_capacity(input.len());
        let t1 = Instant::now();
        let decompress_stats = if let Some(index) = compress_result.index.as_ref() {
            cozip
                .deflate_decompress_stream_zip_compatible_with_index(
                    &mut compressed_reader,
                    &mut restored,
                    index,
                )
                .expect("indexed decompress should succeed")
        } else {
            deflate_decompress_stream_on_cpu(&mut compressed_reader, &mut restored)
                .expect("stream decompress should succeed")
        };
        let decompress_elapsed = t1.elapsed();

        assert_eq!(restored, input);

        agg.add(
            input.len(),
            compressed.len(),
            compress_elapsed,
            decompress_elapsed,
            compress_stats,
            decompress_stats,
        );
    }
    agg
}

fn avg_ms(total_ms: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total_ms / count as f64
    }
}

fn worker_parallelism_equiv(worker_busy_ms: f64, wall_ms: f64) -> f64 {
    if wall_ms <= 0.0 {
        0.0
    } else {
        worker_busy_ms / wall_ms
    }
}

fn main() {
    let cfg = match BenchConfig::from_args() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            std::process::exit(2);
        }
    };

    let size_bytes = cfg
        .size_mib
        .checked_mul(1024 * 1024)
        .expect("size is too large");
    let chunk_size = cfg.chunk_mib * 1024 * 1024;
    let gpu_subchunk_size = cfg.gpu_subchunk_kib * 1024;

    // This benchmark intentionally allocates large input to compare 1GiB-class behavior.
    let input = build_mixed_dataset(size_bytes);

    let cpu_only = HybridOptions {
        chunk_size,
        gpu_subchunk_size,
        gpu_slot_count: cfg.gpu_slots,
        stream_prepare_pipeline_depth: cfg.stream_pipeline_depth,
        stream_batch_chunks: cfg.stream_batch_chunks,
        stream_max_inflight_chunks: cfg.stream_max_inflight_chunks,
        stream_max_inflight_bytes: cfg.stream_max_inflight_mib * 1024 * 1024,
        scheduler_policy: cfg.scheduler_policy,
        gpu_batch_chunks: cfg.gpu_batch_chunks,
        decode_gpu_batch_chunks: cfg.decomp_gpu_batch_chunks,
        gpu_pipelined_submit_chunks: cfg.gpu_submit_chunks,
        token_finalize_segment_size: cfg.token_finalize_segment_size,
        compression_level: 6,
        compression_mode: cfg.mode,
        prefer_gpu: false,
        gpu_fraction: 0.0,
        gpu_tail_stop_ratio: 1.0,
        gpu_min_chunk_size: 64 * 1024,
        profile_timing: cfg.profile_timing,
        profile_timing_detail: cfg.profile_timing_detail,
        profile_timing_deep: cfg.profile_timing_deep,
        ..HybridOptions::default()
    };

    let cpu_gpu = HybridOptions {
        chunk_size,
        gpu_subchunk_size,
        gpu_slot_count: cfg.gpu_slots,
        stream_prepare_pipeline_depth: cfg.stream_pipeline_depth,
        stream_batch_chunks: cfg.stream_batch_chunks,
        stream_max_inflight_chunks: cfg.stream_max_inflight_chunks,
        stream_max_inflight_bytes: cfg.stream_max_inflight_mib * 1024 * 1024,
        scheduler_policy: cfg.scheduler_policy,
        gpu_batch_chunks: cfg.gpu_batch_chunks,
        decode_gpu_batch_chunks: cfg.decomp_gpu_batch_chunks,
        gpu_pipelined_submit_chunks: cfg.gpu_submit_chunks,
        token_finalize_segment_size: cfg.token_finalize_segment_size,
        compression_level: 6,
        compression_mode: cfg.mode,
        prefer_gpu: true,
        gpu_fraction: cfg.gpu_fraction,
        gpu_tail_stop_ratio: cfg.gpu_tail_stop_ratio,
        gpu_min_chunk_size: 64 * 1024,
        profile_timing: cfg.profile_timing,
        profile_timing_detail: cfg.profile_timing_detail,
        profile_timing_deep: cfg.profile_timing_deep,
        ..HybridOptions::default()
    };

    println!("cozip_deflate ZIP-compatible raw-deflate benchmark");
    println!(
        "size_mib={} iters={} warmups={} chunk_mib={} gpu_subchunk_kib={} mode={:?}",
        cfg.size_mib, cfg.iters, cfg.warmups, cfg.chunk_mib, cfg.gpu_subchunk_kib, cfg.mode
    );
    println!(
        "gpu_fraction={:.2} gpu_tail_stop_ratio={:.2} token_finalize_segment_size={} gpu_slots={} gpu_batch_chunks={} decomp_gpu_batch_chunks={} gpu_submit_chunks={} stream_pipeline_depth={} stream_batch_chunks={} stream_max_inflight_chunks={} stream_max_inflight_mib={} scheduler={:?} profile_timing={} profile_timing_detail={} profile_timing_deep={}",
        cfg.gpu_fraction,
        cfg.gpu_tail_stop_ratio,
        cfg.token_finalize_segment_size,
        cfg.gpu_slots,
        cfg.gpu_batch_chunks,
        cfg.decomp_gpu_batch_chunks,
        cfg.gpu_submit_chunks,
        cfg.stream_pipeline_depth,
        cfg.stream_batch_chunks,
        cfg.stream_max_inflight_chunks,
        cfg.stream_max_inflight_mib,
        cfg.scheduler_policy,
        cfg.profile_timing,
        cfg.profile_timing_detail,
        cfg.profile_timing_deep
    );

    let cpu_only_cozip = CoZipDeflate::init(cpu_only).expect("cpu-only init should succeed");
    let cpu_gpu_cozip = CoZipDeflate::init(cpu_gpu).expect("cpu+gpu init should succeed");

    if cfg.warmups > 0 {
        let _ = run_case(&input, &cpu_only_cozip, cfg.warmups);
        let _ = run_case(&input, &cpu_gpu_cozip, cfg.warmups);
    }

    let cpu = run_case(&input, &cpu_only_cozip, cfg.iters);
    let hybrid = run_case(&input, &cpu_gpu_cozip, cfg.iters);

    let comp_speedup = if hybrid.avg_compress_ms(cfg.iters) > 0.0 {
        cpu.avg_compress_ms(cfg.iters) / hybrid.avg_compress_ms(cfg.iters)
    } else {
        0.0
    };
    let cpu_cpu_parallelism = worker_parallelism_equiv(
        cpu.last_compress_stats.cpu_worker_busy_ms,
        cpu.avg_compress_ms(cfg.iters),
    );
    let cpu_gpu_parallelism = worker_parallelism_equiv(
        cpu.last_compress_stats.gpu_worker_busy_ms,
        cpu.avg_compress_ms(cfg.iters),
    );
    let hybrid_cpu_parallelism = worker_parallelism_equiv(
        hybrid.last_compress_stats.cpu_worker_busy_ms,
        hybrid.avg_compress_ms(cfg.iters),
    );
    let hybrid_gpu_parallelism = worker_parallelism_equiv(
        hybrid.last_compress_stats.gpu_worker_busy_ms,
        hybrid.avg_compress_ms(cfg.iters),
    );
    let cpu_cpu_chunk_ms = avg_ms(
        cpu.last_compress_stats.cpu_worker_busy_ms,
        cpu.last_compress_stats.cpu_worker_chunks,
    );
    let cpu_gpu_chunk_ms = avg_ms(
        cpu.last_compress_stats.gpu_worker_busy_ms,
        cpu.last_compress_stats.gpu_worker_chunks,
    );
    let cpu_gpu_batch_avg_ms = avg_ms(
        cpu.last_compress_stats.gpu_worker_busy_ms,
        cpu.last_compress_stats.gpu_batch_count,
    );
    let hybrid_cpu_chunk_ms = avg_ms(
        hybrid.last_compress_stats.cpu_worker_busy_ms,
        hybrid.last_compress_stats.cpu_worker_chunks,
    );
    let hybrid_gpu_chunk_ms = avg_ms(
        hybrid.last_compress_stats.gpu_worker_busy_ms,
        hybrid.last_compress_stats.gpu_worker_chunks,
    );
    let hybrid_gpu_batch_avg_ms = avg_ms(
        hybrid.last_compress_stats.gpu_worker_busy_ms,
        hybrid.last_compress_stats.gpu_batch_count,
    );
    let cpu_write_io_mib =
        cpu.last_compress_stats.write_io_bytes as f64 / (1024.0_f64 * 1024.0_f64);
    let hybrid_write_io_mib =
        hybrid.last_compress_stats.write_io_bytes as f64 / (1024.0_f64 * 1024.0_f64);
    let cpu_writer_hol_ready_avg = if cpu.last_compress_stats.writer_hol_wait_events > 0 {
        cpu.last_compress_stats.writer_hol_ready_sum as f64
            / cpu.last_compress_stats.writer_hol_wait_events as f64
    } else {
        0.0
    };
    let hybrid_writer_hol_ready_avg = if hybrid.last_compress_stats.writer_hol_wait_events > 0 {
        hybrid.last_compress_stats.writer_hol_ready_sum as f64
            / hybrid.last_compress_stats.writer_hol_wait_events as f64
    } else {
        0.0
    };

    println!(
        "CPU_ONLY: avg_comp_ms={:.3} avg_decomp_ms={:.3} comp_mib_s={:.2} decomp_mib_s={:.2} ratio={:.4} chunks={} cpu_chunks={} gpu_chunks={} gpu_available={} comp_stage_ms={:.1} layout_parse_ms={:.1} write_stage_ms={:.1} write_pack_ms={:.1} write_io_ms={:.1} write_io_calls={} write_io_mib={:.2} cpu_worker_busy_ms={:.1} gpu_worker_busy_ms={:.1} cpu_queue_lock_wait_ms={:.1} gpu_queue_lock_wait_ms={:.1} cpu_wait_for_task_ms={:.1} gpu_wait_for_task_ms={:.1} writer_wait_ms={:.1} writer_wait_events={} writer_hol_wait_ms={:.1} writer_hol_wait_events={} writer_hol_ready_avg={:.2} writer_hol_ready_max={} inflight_chunks_max={} ready_chunks_max={} cpu_no_task_events={} gpu_no_task_events={} cpu_yield_events={} gpu_yield_events={} cpu_worker_parallelism={:.2} gpu_worker_parallelism={:.2} cpu_worker_chunks={} gpu_worker_chunks={} cpu_chunk_avg_ms={:.2} gpu_chunk_avg_ms={:.2} cpu_steal_chunks={} gpu_batches={} gpu_batch_avg_ms={:.2} gpu_initial_queue_chunks={} gpu_steal_reserve_chunks={} gpu_runtime_disabled={} decomp_chunks={} decomp_cpu_chunks={} decomp_gpu_tasks={} decomp_gpu_available={} decomp_cpu_worker_busy_ms={:.1} decomp_gpu_worker_busy_ms={:.1} decomp_cpu_wait_for_task_ms={:.1} decomp_gpu_wait_for_task_ms={:.1} decomp_gpu_batches={} decomp_gpu_worker_chunks={} decomp_gpu_attempt_chunks={} decomp_gpu_fallback_chunks={} decomp_decode_prepare_ms={:.1} decomp_decode_gpu_call_ms={:.1} decomp_decode_gpu_fallback_cpu_ms={:.1}",
        cpu.avg_compress_ms(cfg.iters),
        cpu.avg_decompress_ms(cfg.iters),
        cpu.compress_mib_s(),
        cpu.decompress_mib_s(),
        cpu.ratio(),
        cpu.last_compress_stats.chunk_count,
        cpu.last_compress_stats.cpu_chunks,
        cpu.last_compress_stats.gpu_chunks,
        cpu.last_compress_stats.gpu_available,
        cpu.last_compress_stats.compress_stage_ms,
        cpu.last_compress_stats.layout_parse_ms,
        cpu.last_compress_stats.write_stage_ms,
        cpu.last_compress_stats.write_pack_ms,
        cpu.last_compress_stats.write_io_ms,
        cpu.last_compress_stats.write_io_calls,
        cpu_write_io_mib,
        cpu.last_compress_stats.cpu_worker_busy_ms,
        cpu.last_compress_stats.gpu_worker_busy_ms,
        cpu.last_compress_stats.cpu_queue_lock_wait_ms,
        cpu.last_compress_stats.gpu_queue_lock_wait_ms,
        cpu.last_compress_stats.cpu_wait_for_task_ms,
        cpu.last_compress_stats.gpu_wait_for_task_ms,
        cpu.last_compress_stats.writer_wait_ms,
        cpu.last_compress_stats.writer_wait_events,
        cpu.last_compress_stats.writer_hol_wait_ms,
        cpu.last_compress_stats.writer_hol_wait_events,
        cpu_writer_hol_ready_avg,
        cpu.last_compress_stats.writer_hol_ready_max,
        cpu.last_compress_stats.inflight_chunks_max,
        cpu.last_compress_stats.ready_chunks_max,
        cpu.last_compress_stats.cpu_no_task_events,
        cpu.last_compress_stats.gpu_no_task_events,
        cpu.last_compress_stats.cpu_yield_events,
        cpu.last_compress_stats.gpu_yield_events,
        cpu_cpu_parallelism,
        cpu_gpu_parallelism,
        cpu.last_compress_stats.cpu_worker_chunks,
        cpu.last_compress_stats.gpu_worker_chunks,
        cpu_cpu_chunk_ms,
        cpu_gpu_chunk_ms,
        cpu.last_compress_stats.cpu_steal_chunks,
        cpu.last_compress_stats.gpu_batch_count,
        cpu_gpu_batch_avg_ms,
        cpu.last_compress_stats.initial_gpu_queue_chunks,
        cpu.last_compress_stats.gpu_steal_reserve_chunks,
        cpu.last_compress_stats.gpu_runtime_disabled,
        cpu.last_decompress_stats.chunk_count,
        cpu.last_decompress_stats.cpu_chunks,
        cpu.last_decompress_stats.gpu_chunks,
        cpu.last_decompress_stats.gpu_available,
        cpu.last_decompress_stats.cpu_worker_busy_ms,
        cpu.last_decompress_stats.gpu_worker_busy_ms,
        cpu.last_decompress_stats.cpu_wait_for_task_ms,
        cpu.last_decompress_stats.gpu_wait_for_task_ms,
        cpu.last_decompress_stats.gpu_batch_count,
        cpu.last_decompress_stats.gpu_worker_chunks,
        cpu.last_decompress_stats.decode_gpu_attempt_chunks,
        cpu.last_decompress_stats.decode_gpu_fallback_chunks,
        cpu.last_decompress_stats.decode_prepare_ms,
        cpu.last_decompress_stats.decode_gpu_call_ms,
        cpu.last_decompress_stats.decode_gpu_fallback_cpu_ms,
    );
    println!(
        "CPU+GPU : avg_comp_ms={:.3} avg_decomp_ms={:.3} comp_mib_s={:.2} decomp_mib_s={:.2} ratio={:.4} chunks={} cpu_chunks={} gpu_chunks={} gpu_available={} comp_stage_ms={:.1} layout_parse_ms={:.1} write_stage_ms={:.1} write_pack_ms={:.1} write_io_ms={:.1} write_io_calls={} write_io_mib={:.2} cpu_worker_busy_ms={:.1} gpu_worker_busy_ms={:.1} cpu_queue_lock_wait_ms={:.1} gpu_queue_lock_wait_ms={:.1} cpu_wait_for_task_ms={:.1} gpu_wait_for_task_ms={:.1} writer_wait_ms={:.1} writer_wait_events={} writer_hol_wait_ms={:.1} writer_hol_wait_events={} writer_hol_ready_avg={:.2} writer_hol_ready_max={} inflight_chunks_max={} ready_chunks_max={} cpu_no_task_events={} gpu_no_task_events={} cpu_yield_events={} gpu_yield_events={} cpu_worker_parallelism={:.2} gpu_worker_parallelism={:.2} cpu_worker_chunks={} gpu_worker_chunks={} cpu_chunk_avg_ms={:.2} gpu_chunk_avg_ms={:.2} cpu_steal_chunks={} gpu_batches={} gpu_batch_avg_ms={:.2} gpu_initial_queue_chunks={} gpu_steal_reserve_chunks={} gpu_runtime_disabled={} decomp_chunks={} decomp_cpu_chunks={} decomp_gpu_tasks={} decomp_gpu_available={} decomp_cpu_worker_busy_ms={:.1} decomp_gpu_worker_busy_ms={:.1} decomp_cpu_wait_for_task_ms={:.1} decomp_gpu_wait_for_task_ms={:.1} decomp_gpu_batches={} decomp_gpu_worker_chunks={} decomp_gpu_attempt_chunks={} decomp_gpu_fallback_chunks={} decomp_decode_prepare_ms={:.1} decomp_decode_gpu_call_ms={:.1} decomp_decode_gpu_fallback_cpu_ms={:.1}",
        hybrid.avg_compress_ms(cfg.iters),
        hybrid.avg_decompress_ms(cfg.iters),
        hybrid.compress_mib_s(),
        hybrid.decompress_mib_s(),
        hybrid.ratio(),
        hybrid.last_compress_stats.chunk_count,
        hybrid.last_compress_stats.cpu_chunks,
        hybrid.last_compress_stats.gpu_chunks,
        hybrid.last_compress_stats.gpu_available,
        hybrid.last_compress_stats.compress_stage_ms,
        hybrid.last_compress_stats.layout_parse_ms,
        hybrid.last_compress_stats.write_stage_ms,
        hybrid.last_compress_stats.write_pack_ms,
        hybrid.last_compress_stats.write_io_ms,
        hybrid.last_compress_stats.write_io_calls,
        hybrid_write_io_mib,
        hybrid.last_compress_stats.cpu_worker_busy_ms,
        hybrid.last_compress_stats.gpu_worker_busy_ms,
        hybrid.last_compress_stats.cpu_queue_lock_wait_ms,
        hybrid.last_compress_stats.gpu_queue_lock_wait_ms,
        hybrid.last_compress_stats.cpu_wait_for_task_ms,
        hybrid.last_compress_stats.gpu_wait_for_task_ms,
        hybrid.last_compress_stats.writer_wait_ms,
        hybrid.last_compress_stats.writer_wait_events,
        hybrid.last_compress_stats.writer_hol_wait_ms,
        hybrid.last_compress_stats.writer_hol_wait_events,
        hybrid_writer_hol_ready_avg,
        hybrid.last_compress_stats.writer_hol_ready_max,
        hybrid.last_compress_stats.inflight_chunks_max,
        hybrid.last_compress_stats.ready_chunks_max,
        hybrid.last_compress_stats.cpu_no_task_events,
        hybrid.last_compress_stats.gpu_no_task_events,
        hybrid.last_compress_stats.cpu_yield_events,
        hybrid.last_compress_stats.gpu_yield_events,
        hybrid_cpu_parallelism,
        hybrid_gpu_parallelism,
        hybrid.last_compress_stats.cpu_worker_chunks,
        hybrid.last_compress_stats.gpu_worker_chunks,
        hybrid_cpu_chunk_ms,
        hybrid_gpu_chunk_ms,
        hybrid.last_compress_stats.cpu_steal_chunks,
        hybrid.last_compress_stats.gpu_batch_count,
        hybrid_gpu_batch_avg_ms,
        hybrid.last_compress_stats.initial_gpu_queue_chunks,
        hybrid.last_compress_stats.gpu_steal_reserve_chunks,
        hybrid.last_compress_stats.gpu_runtime_disabled,
        hybrid.last_decompress_stats.chunk_count,
        hybrid.last_decompress_stats.cpu_chunks,
        hybrid.last_decompress_stats.gpu_chunks,
        hybrid.last_decompress_stats.gpu_available,
        hybrid.last_decompress_stats.cpu_worker_busy_ms,
        hybrid.last_decompress_stats.gpu_worker_busy_ms,
        hybrid.last_decompress_stats.cpu_wait_for_task_ms,
        hybrid.last_decompress_stats.gpu_wait_for_task_ms,
        hybrid.last_decompress_stats.gpu_batch_count,
        hybrid.last_decompress_stats.gpu_worker_chunks,
        hybrid.last_decompress_stats.decode_gpu_attempt_chunks,
        hybrid.last_decompress_stats.decode_gpu_fallback_chunks,
        hybrid.last_decompress_stats.decode_prepare_ms,
        hybrid.last_decompress_stats.decode_gpu_call_ms,
        hybrid.last_decompress_stats.decode_gpu_fallback_cpu_ms,
    );
    println!(
        "speedup(cpu/hybrid): compress={:.3}x ( >1.0 means CPU+GPU faster )",
        comp_speedup
    );
}
