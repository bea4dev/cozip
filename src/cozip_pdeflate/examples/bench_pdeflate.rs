use std::time::Instant;

use cozip_pdeflate::{
    PDeflateOptions, pdeflate_compress_with_stats, pdeflate_decompress_with_stats,
    pdeflate_gpu_init,
};

#[derive(Debug, Clone)]
struct BenchConfig {
    size_mib: usize,
    runs: usize,
    warmups: usize,
    chunk_mib: usize,
    sections: usize,
    verify_bytes: bool,
    gpu_compress: bool,
    compare_hybrid: bool,
    gpu_workers: usize,
    gpu_submit_chunks: usize,
    gpu_min_chunk_kib: usize,
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
            gpu_compress: false,
            compare_hybrid: false,
            gpu_workers: 1,
            gpu_submit_chunks: 4,
            gpu_min_chunk_kib: 64,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RunResult {
    comp_ms: f64,
    decomp_ms: f64,
    ratio: f64,
    comp_mib_s: f64,
    decomp_mib_s: f64,
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
            "--gpu-compress" => {
                cfg.gpu_compress = true;
            }
            "--gpu-compress-enabled" => {
                i += 1;
                let v = args.get(i).ok_or("--gpu-compress-enabled requires value")?;
                cfg.gpu_compress =
                    parse_bool(v).ok_or_else(|| format!("invalid --gpu-compress-enabled: {v}"))?;
            }
            "--compare-hybrid" => {
                cfg.compare_hybrid = true;
            }
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
            "--gpu-submit-chunks" => {
                i += 1;
                cfg.gpu_submit_chunks = args
                    .get(i)
                    .ok_or("--gpu-submit-chunks requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-submit-chunks: {e}"))?;
            }
            "--gpu-min-chunk-kib" => {
                i += 1;
                cfg.gpu_min_chunk_kib = args
                    .get(i)
                    .ok_or("--gpu-min-chunk-kib requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-min-chunk-kib: {e}"))?;
            }
            "--verify" => {
                cfg.verify_bytes = true;
            }
            "--no-verify" => {
                cfg.verify_bytes = false;
            }
            "--verify-bytes" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--verify-bytes requires value (0|1|false|true)")?;
                cfg.verify_bytes =
                    parse_bool(v).ok_or_else(|| format!("invalid --verify-bytes: {v}"))?;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            x => return Err(format!("unknown argument: {x}")),
        }
        i += 1;
    }

    if cfg.size_mib == 0 {
        return Err("--size-mib must be > 0".to_string());
    }
    if cfg.runs == 0 {
        return Err("--runs must be > 0".to_string());
    }
    if cfg.chunk_mib == 0 {
        return Err("--chunk-mib must be > 0".to_string());
    }
    if cfg.sections == 0 {
        return Err("--sections must be > 0".to_string());
    }
    if cfg.gpu_workers == 0 {
        return Err("--gpu-workers must be > 0".to_string());
    }
    if cfg.gpu_submit_chunks == 0 {
        return Err("--gpu-submit-chunks must be > 0".to_string());
    }
    if cfg.gpu_min_chunk_kib == 0 {
        return Err("--gpu-min-chunk-kib must be > 0".to_string());
    }

    Ok(cfg)
}

fn print_help() {
    println!(
        "usage: cargo run --release -p cozip_pdeflate --example bench_pdeflate -- [options]\n\
options:\n\
  --size-mib <N>         input size in MiB (default: 1024)\n\
  --runs <N>             measured runs (default: 3)\n\
  --warmups <N>          warmup runs (default: 1)\n\
  --chunk-mib <N>        chunk size in MiB (default: 4)\n\
  --sections <N>         sections per chunk (default: 128)\n\
  --gpu-compress         enable GPU compress path\n\
  --gpu-compress-enabled <B>  GPU compress (0/1)\n\
  --compare-hybrid       compare CPU_ONLY vs CPU+GPU\n\
  --compare-hybrid-enabled <B> compare mode (0/1)\n\
  --gpu-workers <N>      GPU worker count (default: 2)\n\
  --gpu-submit-chunks <N> GPU batch chunk limit (default: 4)\n\
  --gpu-min-chunk-kib <N> GPU eligible chunk threshold in KiB (default: 64)\n\
  --verify               enable strict decoded-bytes check (default)\n\
  --no-verify            disable decoded-bytes check\n\
  --verify-bytes <B>     strict decoded-bytes check (0/1, default: 1)\n\
  -h, --help             show help"
    );
}

fn generate_input(size_bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; size_bytes];
    let text = b"ABABABABCCABCCD--cozip-pdeflate-bench--";
    let mut rng = 0x1234_5678_u32;
    for i in 0..size_bytes {
        out[i] = match (i / 8192) % 6 {
            0 => text[i % text.len()],
            1 => b'A' + ((i / 11) % 8) as u8,
            2 => {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                (rng >> 24) as u8
            }
            3 => (i as u8).wrapping_mul(17).wrapping_add(31),
            4 => {
                if (i / 64) % 2 == 0 {
                    0x00
                } else {
                    0xff
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
    opts: &PDeflateOptions,
    size_mib: usize,
    verify_bytes: bool,
) -> Result<RunResult, String> {
    let c0 = Instant::now();
    let (compressed, _) = pdeflate_compress_with_stats(input, opts).map_err(|e| e.to_string())?;
    let comp_ms = c0.elapsed().as_secs_f64() * 1000.0;

    let d0 = Instant::now();
    let (decoded, _) = pdeflate_decompress_with_stats(&compressed).map_err(|e| e.to_string())?;
    let decomp_ms = d0.elapsed().as_secs_f64() * 1000.0;

    if verify_bytes && decoded != input {
        return Err("roundtrip mismatch".to_string());
    }

    let ratio = compressed.len() as f64 / input.len() as f64;
    let size_mib_f = size_mib as f64;
    let comp_mib_s = if comp_ms > 0.0 {
        size_mib_f * 1000.0 / comp_ms
    } else {
        0.0
    };
    let decomp_mib_s = if decomp_ms > 0.0 {
        size_mib_f * 1000.0 / decomp_ms
    } else {
        0.0
    };

    Ok(RunResult {
        comp_ms,
        decomp_ms,
        ratio,
        comp_mib_s,
        decomp_mib_s,
    })
}

fn main() -> Result<(), String> {
    let cfg = parse_args()?;
    let size_bytes = cfg.size_mib * 1024 * 1024;
    let input = generate_input(size_bytes);

    let mut hybrid_opts = PDeflateOptions {
        chunk_size: cfg.chunk_mib * 1024 * 1024,
        section_count: cfg.sections,
        gpu_compress_enabled: cfg.gpu_compress,
        gpu_workers: cfg.gpu_workers,
        gpu_submit_chunks: cfg.gpu_submit_chunks,
        gpu_min_chunk_size: cfg.gpu_min_chunk_kib * 1024,
        ..PDeflateOptions::default()
    };
    let mut cpu_opts = hybrid_opts.clone();
    cpu_opts.gpu_compress_enabled = false;

    println!(
        "cozip_pdeflate benchmark\nsize_mib={} runs={} warmups={} chunk_mib={} sections={} gpu_compress={} gpu_workers={} gpu_submit_chunks={} gpu_min_chunk_kib={} compare_hybrid={} verify_bytes={}",
        cfg.size_mib,
        cfg.runs,
        cfg.warmups,
        cfg.chunk_mib,
        cfg.sections,
        cfg.gpu_compress,
        cfg.gpu_workers,
        cfg.gpu_submit_chunks,
        cfg.gpu_min_chunk_kib,
        cfg.compare_hybrid,
        cfg.verify_bytes
    );

    if cfg.gpu_compress {
        // Keep GPU runtime/device/pipeline init outside timed runs.
        let gpu_available = pdeflate_gpu_init();
        println!("gpu_runtime_initialized={}", gpu_available);
    }

    for _ in 0..cfg.warmups {
        if cfg.compare_hybrid {
            let _ = run_once(&input, &cpu_opts, cfg.size_mib, cfg.verify_bytes)?;
        }
        let _ = run_once(&input, &hybrid_opts, cfg.size_mib, cfg.verify_bytes)?;
    }

    let mut comp_ms = Vec::with_capacity(cfg.runs);
    let mut decomp_ms = Vec::with_capacity(cfg.runs);
    let mut ratio = Vec::with_capacity(cfg.runs);
    let mut comp_mib_s = Vec::with_capacity(cfg.runs);
    let mut decomp_mib_s = Vec::with_capacity(cfg.runs);
    let mut cpu_only_comp_ms = Vec::with_capacity(cfg.runs);
    let mut cpu_only_decomp_ms = Vec::with_capacity(cfg.runs);
    let mut speedup_comp = Vec::with_capacity(cfg.runs);
    let mut speedup_decomp = Vec::with_capacity(cfg.runs);

    for i in 0..cfg.runs {
        if cfg.compare_hybrid {
            let cpu = run_once(&input, &cpu_opts, cfg.size_mib, cfg.verify_bytes)?;
            let hyb = run_once(&input, &hybrid_opts, cfg.size_mib, cfg.verify_bytes)?;
            let sp_comp = if hyb.comp_ms > 0.0 {
                cpu.comp_ms / hyb.comp_ms
            } else {
                0.0
            };
            let sp_decomp = if hyb.decomp_ms > 0.0 {
                cpu.decomp_ms / hyb.decomp_ms
            } else {
                0.0
            };
            println!(
                "run {}/{}: CPU_ONLY comp_ms={:.3} decomp_ms={:.3} | CPU+GPU comp_ms={:.3} decomp_ms={:.3} ratio={:.4} speedup_comp={:.3}x speedup_decomp={:.3}x",
                i + 1,
                cfg.runs,
                cpu.comp_ms,
                cpu.decomp_ms,
                hyb.comp_ms,
                hyb.decomp_ms,
                hyb.ratio,
                sp_comp,
                sp_decomp
            );
            cpu_only_comp_ms.push(cpu.comp_ms);
            cpu_only_decomp_ms.push(cpu.decomp_ms);
            speedup_comp.push(sp_comp);
            speedup_decomp.push(sp_decomp);
            comp_ms.push(hyb.comp_ms);
            decomp_ms.push(hyb.decomp_ms);
            comp_mib_s.push(hyb.comp_mib_s);
            decomp_mib_s.push(hyb.decomp_mib_s);
            ratio.push(hyb.ratio);
        } else {
            let r = run_once(&input, &hybrid_opts, cfg.size_mib, cfg.verify_bytes)?;
            println!(
                "run {}/{}: comp_ms={:.3} decomp_ms={:.3} comp_mib_s={:.2} decomp_mib_s={:.2} ratio={:.4}",
                i + 1,
                cfg.runs,
                r.comp_ms,
                r.decomp_ms,
                r.comp_mib_s,
                r.decomp_mib_s,
                r.ratio,
            );
            comp_ms.push(r.comp_ms);
            decomp_ms.push(r.decomp_ms);
            comp_mib_s.push(r.comp_mib_s);
            decomp_mib_s.push(r.decomp_mib_s);
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
    println!(
        "decomp_ms: n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
        decomp_ms.len(),
        mean(&decomp_ms),
        median(&decomp_ms),
        min(&decomp_ms),
        max(&decomp_ms)
    );
    println!(
        "comp_mib_s: n={} mean={:.2} median={:.2} min={:.2} max={:.2}",
        comp_mib_s.len(),
        mean(&comp_mib_s),
        median(&comp_mib_s),
        min(&comp_mib_s),
        max(&comp_mib_s)
    );
    println!(
        "decomp_mib_s: n={} mean={:.2} median={:.2} min={:.2} max={:.2}",
        decomp_mib_s.len(),
        mean(&decomp_mib_s),
        median(&decomp_mib_s),
        min(&decomp_mib_s),
        max(&decomp_mib_s)
    );
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
            cpu_only_comp_ms.len(),
            mean(&cpu_only_comp_ms),
            median(&cpu_only_comp_ms),
            min(&cpu_only_comp_ms),
            max(&cpu_only_comp_ms)
        );
        println!(
            "cpu_only_decomp_ms: n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
            cpu_only_decomp_ms.len(),
            mean(&cpu_only_decomp_ms),
            median(&cpu_only_decomp_ms),
            min(&cpu_only_decomp_ms),
            max(&cpu_only_decomp_ms)
        );
        println!(
            "speedup_comp(cpu/hybrid): n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
            speedup_comp.len(),
            mean(&speedup_comp),
            median(&speedup_comp),
            min(&speedup_comp),
            max(&speedup_comp)
        );
        println!(
            "speedup_decomp(cpu/hybrid): n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
            speedup_decomp.len(),
            mean(&speedup_decomp),
            median(&speedup_decomp),
            min(&speedup_decomp),
            max(&speedup_decomp)
        );
    }

    // keep mutable use to satisfy clippy under feature toggles where opts may be tweaked later
    hybrid_opts.gpu_submit_chunks = hybrid_opts.gpu_submit_chunks.max(1);

    Ok(())
}
