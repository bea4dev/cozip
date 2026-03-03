use std::time::Instant;

use cozip_pdeflate::{
    CoZipDeflate, CompressionMode, DeflateCpuStreamStats, HybridOptions, HybridSchedulerPolicy,
    deflate_decompress_stream_on_cpu,
};

#[derive(Debug, Clone)]
struct BenchConfig {
    size_mib: usize,
    runs: usize,
    warmups: usize,
    chunk_mib: usize,
    sections: usize,
    verify_bytes: bool,
    skip_decompress: bool,
    gpu_compress: bool,
    compare_hybrid: bool,
    gpu_workers: usize,
    gpu_slot_count: usize,
    gpu_submit_chunks: usize,
    gpu_pipelined_submit_chunks: usize,
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
            size_mib: 1024,
            runs: 3,
            warmups: 1,
            chunk_mib: 4,
            sections: 128,
            verify_bytes: true,
            skip_decompress: true,
            gpu_compress: false,
            compare_hybrid: false,
            gpu_workers: 1,
            gpu_slot_count: 16,
            gpu_submit_chunks: 4,
            gpu_pipelined_submit_chunks: 4,
            gpu_min_chunk_kib: 64,
            scheduler_policy: HybridSchedulerPolicy::GlobalQueueLocalBuffers,
            gpu_tail_stop_ratio: 1.0,
            mode: CompressionMode::Balanced,
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
            "--gpu-compress" => cfg.gpu_compress = true,
            "--gpu-compress-enabled" => {
                i += 1;
                let v = args.get(i).ok_or("--gpu-compress-enabled requires value")?;
                cfg.gpu_compress = parse_bool(v)
                    .ok_or_else(|| format!("invalid --gpu-compress-enabled: {v}"))?;
            }
            "--compare-hybrid" => cfg.compare_hybrid = true,
            "--compare-hybrid-enabled" => {
                i += 1;
                let v = args.get(i).ok_or("--compare-hybrid-enabled requires value")?;
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
            "--gpu-submit-chunks" => {
                i += 1;
                cfg.gpu_submit_chunks = args
                    .get(i)
                    .ok_or("--gpu-submit-chunks requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-submit-chunks: {e}"))?;
            }
            "--gpu-slot-count" => {
                i += 1;
                cfg.gpu_slot_count = args
                    .get(i)
                    .ok_or("--gpu-slot-count requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-slot-count: {e}"))?;
            }
            "--gpu-pipelined-submit-chunks" => {
                i += 1;
                cfg.gpu_pipelined_submit_chunks = args
                    .get(i)
                    .ok_or("--gpu-pipelined-submit-chunks requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-pipelined-submit-chunks: {e}"))?;
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
    if cfg.gpu_slot_count == 0 || cfg.gpu_submit_chunks == 0 || cfg.gpu_pipelined_submit_chunks == 0
    {
        return Err("gpu-slot-count/gpu-submit-chunks/gpu-pipelined-submit-chunks must be > 0"
            .to_string());
    }
    if !(0.0..=1.0).contains(&cfg.gpu_tail_stop_ratio) {
        return Err("--gpu-tail-stop-ratio must be in range 0.0..=1.0".to_string());
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
  --sections <N>\n\
  --gpu-compress\n\
  --compare-hybrid\n\
  --gpu-workers <N>\n\
  --gpu-slot-count <N>\n\
  --gpu-submit-chunks <N>\n\
  --gpu-pipelined-submit-chunks <N>\n\
  --gpu-min-chunk-kib <N>\n\
  --scheduler <global|global-local>\n\
  --gpu-tail-stop-ratio <R>\n\
  --mode <speed|balanced|ratio> (default: balanced)\n\
  --skip-decompress / --no-skip-decompress\n\
  --verify / --no-verify / --verify-bytes <0|1>"
    );
}

fn generate_input(size_bytes: usize) -> Vec<u8> {
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
                .deflate_decompress_stream_zip_compatible_with_index(&mut reader, &mut restored, index)
                .map_err(|e| e.to_string())?;
        } else {
            deflate_decompress_stream_on_cpu(&mut reader, &mut restored)
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
    v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "SKIP".to_string())
}

fn main() -> Result<(), String> {
    let cfg = parse_args()?;
    let size_bytes = cfg.size_mib * 1024 * 1024;
    let input = generate_input(size_bytes);

    let cpu_opts = HybridOptions {
        chunk_size: cfg.chunk_mib * 1024 * 1024,
        gpu_slot_count: cfg.gpu_slot_count,
        gpu_batch_chunks: cfg.gpu_submit_chunks,
        gpu_pipelined_submit_chunks: cfg.gpu_pipelined_submit_chunks,
        scheduler_policy: cfg.scheduler_policy,
        compression_mode: cfg.mode,
        prefer_gpu: false,
        gpu_fraction: 0.0,
        gpu_tail_stop_ratio: 1.0,
        gpu_min_chunk_size: cfg.gpu_min_chunk_kib * 1024,
        profile_timing: cfg.profile_timing,
        profile_timing_detail: cfg.profile_timing_detail,
        profile_timing_deep: cfg.profile_timing_deep,
        ..HybridOptions::default()
    };
    let hybrid_opts = HybridOptions {
        chunk_size: cfg.chunk_mib * 1024 * 1024,
        gpu_slot_count: cfg.gpu_slot_count,
        gpu_batch_chunks: cfg.gpu_submit_chunks,
        gpu_pipelined_submit_chunks: cfg.gpu_pipelined_submit_chunks,
        scheduler_policy: cfg.scheduler_policy,
        compression_mode: cfg.mode,
        prefer_gpu: cfg.gpu_compress,
        gpu_fraction: if cfg.gpu_compress { 1.0 } else { 0.0 },
        gpu_tail_stop_ratio: cfg.gpu_tail_stop_ratio,
        gpu_min_chunk_size: cfg.gpu_min_chunk_kib * 1024,
        profile_timing: cfg.profile_timing,
        profile_timing_detail: cfg.profile_timing_detail,
        profile_timing_deep: cfg.profile_timing_deep,
        ..HybridOptions::default()
    };

    let cpu = CoZipDeflate::init(cpu_opts).map_err(|e| e.to_string())?;
    let hybrid = CoZipDeflate::init(hybrid_opts).map_err(|e| e.to_string())?;

    println!(
        "cozip_pdeflate benchmark\nsize_mib={} runs={} warmups={} chunk_mib={} sections={} gpu_compress={} gpu_workers={} gpu_slot_count={} gpu_submit_chunks={} gpu_pipelined_submit_chunks={} gpu_min_chunk_kib={} scheduler={:?} gpu_tail_stop_ratio={:.2} compare_hybrid={} verify_bytes={}",
        cfg.size_mib,
        cfg.runs,
        cfg.warmups,
        cfg.chunk_mib,
        cfg.sections,
        cfg.gpu_compress,
        cfg.gpu_workers,
        cfg.gpu_slot_count,
        cfg.gpu_submit_chunks,
        cfg.gpu_pipelined_submit_chunks,
        cfg.gpu_min_chunk_kib,
        cfg.scheduler_policy,
        cfg.gpu_tail_stop_ratio,
        cfg.compare_hybrid,
        cfg.verify_bytes
    );
    println!("skip_decompress={}", cfg.skip_decompress);
    if cfg.skip_decompress && cfg.verify_bytes {
        println!("[bench] verify_bytes is ignored because decompression is skipped");
    }

    for _ in 0..cfg.warmups {
        if cfg.compare_hybrid {
            let _ = run_once(&input, &cpu, cfg.size_mib, cfg.skip_decompress, cfg.verify_bytes)?;
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

    for i in 0..cfg.runs {
        if cfg.compare_hybrid {
            let c = run_once(&input, &cpu, cfg.size_mib, cfg.skip_decompress, cfg.verify_bytes)?;
            let h = run_once(
                &input,
                &hybrid,
                cfg.size_mib,
                cfg.skip_decompress,
                cfg.verify_bytes,
            )?;
            last_hybrid_stats = h.compress_stats;
            let spc = if h.comp_ms > 0.0 { c.comp_ms / h.comp_ms } else { 0.0 };
            let spd = match (c.decomp_ms, h.decomp_ms) {
                (Some(a), Some(b)) if b > 0.0 => a / b,
                _ => 0.0,
            };
            println!(
                "run {}/{}: CPU_ONLY comp_ms={:.3} decomp={} | CPU+GPU comp_ms={:.3} decomp={} ratio={:.4} speedup_comp={:.3}x speedup_decomp={:.3}x",
                i + 1,
                cfg.runs,
                c.comp_ms,
                format_opt(c.decomp_ms),
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

    println!("gpu_runtime_initialized={}", last_hybrid_stats.gpu_available);
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

    Ok(())
}
