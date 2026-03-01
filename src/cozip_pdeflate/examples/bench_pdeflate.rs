use std::time::Instant;

use cozip_pdeflate::{
    PDeflateOptions, pdeflate_compress_with_stats, pdeflate_decompress_with_stats,
};

#[derive(Debug, Clone)]
struct BenchConfig {
    size_mib: usize,
    runs: usize,
    warmups: usize,
    chunk_mib: usize,
    sections: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            size_mib: 1024,
            runs: 3,
            warmups: 1,
            chunk_mib: 4,
            sections: 128,
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

    Ok(cfg)
}

fn print_help() {
    println!(
        "usage: cargo run --release -p cozip_pdeflate --example bench_pdeflate -- [options]\n\
options:\n\
  --size-mib <N>   input size in MiB (default: 1024)\n\
  --runs <N>       measured runs (default: 3)\n\
  --warmups <N>    warmup runs (default: 1)\n\
  --chunk-mib <N>  chunk size in MiB (default: 4)\n\
  --sections <N>   sections per chunk (default: 128)\n\
  -h, --help       show help"
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

fn run_once(input: &[u8], opts: &PDeflateOptions, size_mib: usize) -> Result<RunResult, String> {
    let c0 = Instant::now();
    let (compressed, cstats) =
        pdeflate_compress_with_stats(input, opts).map_err(|e| e.to_string())?;
    let comp_ms = c0.elapsed().as_secs_f64() * 1000.0;

    let d0 = Instant::now();
    let (decoded, _dstats) =
        pdeflate_decompress_with_stats(&compressed).map_err(|e| e.to_string())?;
    let decomp_ms = d0.elapsed().as_secs_f64() * 1000.0;

    if decoded != input {
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

    let _ = cstats;
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

    let opts = PDeflateOptions {
        chunk_size: cfg.chunk_mib * 1024 * 1024,
        section_count: cfg.sections,
        ..PDeflateOptions::default()
    };

    println!(
        "cozip_pdeflate benchmark\nsize_mib={} runs={} warmups={} chunk_mib={} sections={}",
        cfg.size_mib, cfg.runs, cfg.warmups, cfg.chunk_mib, cfg.sections
    );

    for _ in 0..cfg.warmups {
        let _ = run_once(&input, &opts, cfg.size_mib)?;
    }

    let mut comp_ms = Vec::with_capacity(cfg.runs);
    let mut decomp_ms = Vec::with_capacity(cfg.runs);
    let mut ratio = Vec::with_capacity(cfg.runs);
    let mut comp_mib_s = Vec::with_capacity(cfg.runs);
    let mut decomp_mib_s = Vec::with_capacity(cfg.runs);

    for i in 0..cfg.runs {
        let r = run_once(&input, &opts, cfg.size_mib)?;
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

    Ok(())
}
