use std::time::Instant;

use cozip_gdeflate::{
    GDeflateCompressionMode, GDeflateOptions, gdeflate_compress, gdeflate_decompress,
};

#[derive(Debug, Clone)]
struct BenchConfig {
    mode: GDeflateCompressionMode,
    size_mib: usize,
    runs: usize,
    warmups: usize,
    cpu_workers: usize,
    gpu_compress: bool,
    gpu_workers: usize,
    gpu_submit_tiles: usize,
    gpu_decompress: bool,
    decomp_gpu_workers: usize,
    decomp_gpu_submit_tiles: usize,
    decomp_gpu_super_batch_factor: usize,
    compare_hybrid: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            mode: GDeflateCompressionMode::TryAll,
            size_mib: 1024,
            runs: 3,
            warmups: 1,
            cpu_workers: 0,
            gpu_compress: false,
            gpu_workers: 1,
            gpu_submit_tiles: 8,
            gpu_decompress: false,
            decomp_gpu_workers: 4,
            decomp_gpu_submit_tiles: 64,
            decomp_gpu_super_batch_factor: 2,
            compare_hybrid: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RunResult {
    comp_ms: f64,
    decomp_ms: f64,
    ratio: f64,
}

fn parse_mode(s: &str) -> Result<GDeflateCompressionMode, String> {
    match s {
        "tryall" | "auto" => Ok(GDeflateCompressionMode::TryAll),
        "stored" => Ok(GDeflateCompressionMode::StoredOnly),
        "static" => Ok(GDeflateCompressionMode::StaticHuffman),
        "dynamic" => Ok(GDeflateCompressionMode::DynamicHuffman),
        _ => Err(format!(
            "invalid --mode: {s} (expected tryall|stored|static|dynamic; auto is accepted as alias)"
        )),
    }
}

fn parse_args() -> Result<BenchConfig, String> {
    let mut cfg = BenchConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                let v = args.get(i).ok_or("--mode requires value")?;
                cfg.mode = parse_mode(v)?;
            }
            "--size-mib" => {
                i += 1;
                let v = args.get(i).ok_or("--size-mib requires value")?;
                cfg.size_mib = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --size-mib: {e}"))?;
            }
            "--runs" => {
                i += 1;
                let v = args.get(i).ok_or("--runs requires value")?;
                cfg.runs = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --runs: {e}"))?;
            }
            "--warmups" => {
                i += 1;
                let v = args.get(i).ok_or("--warmups requires value")?;
                cfg.warmups = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --warmups: {e}"))?;
            }
            "--cpu-workers" => {
                i += 1;
                let v = args.get(i).ok_or("--cpu-workers requires value")?;
                cfg.cpu_workers = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --cpu-workers: {e}"))?;
            }
            "--gpu-compress" => {
                cfg.gpu_compress = true;
            }
            "--gpu-workers" => {
                i += 1;
                let v = args.get(i).ok_or("--gpu-workers requires value")?;
                cfg.gpu_workers = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-workers: {e}"))?;
            }
            "--gpu-submit-tiles" => {
                i += 1;
                let v = args.get(i).ok_or("--gpu-submit-tiles requires value")?;
                cfg.gpu_submit_tiles = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --gpu-submit-tiles: {e}"))?;
            }
            "--gpu-decompress" => {
                cfg.gpu_decompress = true;
            }
            "--decomp-gpu-workers" => {
                i += 1;
                let v = args.get(i).ok_or("--decomp-gpu-workers requires value")?;
                cfg.decomp_gpu_workers = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --decomp-gpu-workers: {e}"))?;
            }
            "--decomp-gpu-submit-tiles" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--decomp-gpu-submit-tiles requires value")?;
                cfg.decomp_gpu_submit_tiles = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --decomp-gpu-submit-tiles: {e}"))?;
            }
            "--decomp-gpu-super-batch-factor" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--decomp-gpu-super-batch-factor requires value")?;
                cfg.decomp_gpu_super_batch_factor = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --decomp-gpu-super-batch-factor: {e}"))?;
            }
            "--compare-hybrid" => {
                cfg.compare_hybrid = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            x => {
                return Err(format!("unknown argument: {x}"));
            }
        }
        i += 1;
    }
    if cfg.size_mib == 0 {
        return Err("--size-mib must be > 0".to_string());
    }
    if cfg.runs == 0 {
        return Err("--runs must be > 0".to_string());
    }
    Ok(cfg)
}

fn print_help() {
    println!(
        "usage: cargo run --release -p cozip_gdeflate --example bench_gdeflate -- [options]\n\
options:\n\
  --mode <M>         tryall|stored|static|dynamic (default: tryall; auto is alias)\n\
  --size-mib <N>     input size in MiB (default: 1024)\n\
  --runs <N>         measured runs (default: 3)\n\
  --warmups <N>      warmup runs (default: 1)\n\
  --cpu-workers <N>  0=auto, else fixed worker count (default: 0)\n\
  --gpu-compress     enable GPU StoredOnly/StaticHuffman compression path\n\
  --gpu-workers <N>  max in-flight GPU batches handled by a single GPU manager (default: 1)\n\
  --gpu-submit-tiles <N> micro-batch tiles per GPU submit (default: 8)\n\
  --gpu-decompress   enable GPU static-Huffman decompression path\n\
  --decomp-gpu-workers <N> max in-flight GPU decode batches handled by one GPU manager (default: 4)\n\
  --decomp-gpu-submit-tiles <N> micro-batch tiles per GPU decode submit (default: 64)\n\
  --decomp-gpu-super-batch-factor <N> multiplier for decode super-batch submit (default: 2)\n\
  --compare-hybrid   run both CPU_ONLY and HYBRID and print speedup\n\
  -h, --help         show help"
    );
}

fn generate_input(size_bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; size_bytes];
    let text = b"The quick brown fox jumps over the lazy dog. ";
    let mut rng = 0x1234_5678_u32;
    for i in 0..size_bytes {
        let block = (i / 4096) % 5;
        out[i] = match block {
            0 => b'A',
            1 => text[i % text.len()],
            2 => (i as u8).wrapping_mul(17).wrapping_add(31),
            3 => {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                (rng >> 24) as u8
            }
            _ => {
                let base = (i % 256) as u8;
                if (i / 256) % 2 == 0 {
                    base
                } else {
                    base ^ 0x55
                }
            }
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

fn run_roundtrip(
    input: &[u8],
    size_mib: usize,
    opts: &GDeflateOptions,
) -> Result<RunResult, String> {
    let c0 = Instant::now();
    let compressed = gdeflate_compress(input, opts).map_err(|e| e.to_string())?;
    let c_ms = c0.elapsed().as_secs_f64() * 1000.0;

    let d0 = Instant::now();
    let decoded = gdeflate_decompress(&compressed, opts).map_err(|e| e.to_string())?;
    let d_ms = d0.elapsed().as_secs_f64() * 1000.0;
    if decoded != input {
        return Err("roundtrip mismatch".to_string());
    }
    let _ = size_mib;
    Ok(RunResult {
        comp_ms: c_ms,
        decomp_ms: d_ms,
        ratio: compressed.len() as f64 / input.len() as f64,
    })
}

fn main() -> Result<(), String> {
    let cfg = parse_args()?;
    let input_size = cfg
        .size_mib
        .checked_mul(1024 * 1024)
        .ok_or("size overflow")?;
    let input = generate_input(input_size);
    let opts = GDeflateOptions {
        compression_mode: cfg.mode,
        cpu_worker_count: cfg.cpu_workers,
        gpu_compress_enabled: cfg.gpu_compress,
        gpu_compress_workers: cfg.gpu_workers,
        gpu_compress_submit_tiles: cfg.gpu_submit_tiles,
        gpu_decompress_enabled: cfg.gpu_decompress,
        gpu_decompress_workers: cfg.decomp_gpu_workers,
        gpu_decompress_submit_tiles: cfg.decomp_gpu_submit_tiles,
        gpu_decompress_super_batch_factor: cfg.decomp_gpu_super_batch_factor,
        ..GDeflateOptions::default()
    };

    println!("cozip_gdeflate benchmark");
    println!(
        "mode={:?} size_mib={} runs={} warmups={} cpu_workers={} gpu_compress={} gpu_workers={} gpu_submit_tiles={} gpu_decompress={} decomp_gpu_workers={} decomp_gpu_submit_tiles={} decomp_gpu_super_batch_factor={} tile_kib=64",
        cfg.mode,
        cfg.size_mib,
        cfg.runs,
        cfg.warmups,
        cfg.cpu_workers,
        cfg.gpu_compress,
        cfg.gpu_workers,
        cfg.gpu_submit_tiles,
        cfg.gpu_decompress,
        cfg.decomp_gpu_workers,
        cfg.decomp_gpu_submit_tiles,
        cfg.decomp_gpu_super_batch_factor
    );

    let mut hybrid_comp_ms = Vec::with_capacity(cfg.runs);
    let mut hybrid_decomp_ms = Vec::with_capacity(cfg.runs);
    let mut hybrid_ratios = Vec::with_capacity(cfg.runs);
    let mut cpu_comp_ms = Vec::new();
    let mut cpu_decomp_ms = Vec::new();
    let mut speedup_comp = Vec::new();
    let mut speedup_decomp = Vec::new();

    let opts_cpu = GDeflateOptions {
        gpu_compress_enabled: false,
        gpu_decompress_enabled: false,
        ..opts.clone()
    };

    for warm in 0..cfg.warmups {
        let hybrid = run_roundtrip(&input, cfg.size_mib, &opts)?;
        if cfg.compare_hybrid {
            let cpu = run_roundtrip(&input, cfg.size_mib, &opts_cpu)?;
            println!(
                "warmup {}/{}: cpu_comp_ms={:.3} hybrid_comp_ms={:.3} cpu_decomp_ms={:.3} hybrid_decomp_ms={:.3}",
                warm + 1,
                cfg.warmups,
                cpu.comp_ms,
                hybrid.comp_ms,
                cpu.decomp_ms,
                hybrid.decomp_ms
            );
        } else {
            println!(
                "warmup {}/{}: comp_ms={:.3} decomp_ms={:.3} ratio={:.4}",
                warm + 1,
                cfg.warmups,
                hybrid.comp_ms,
                hybrid.decomp_ms,
                hybrid.ratio
            );
        }
    }

    for run in 0..cfg.runs {
        let hybrid = run_roundtrip(&input, cfg.size_mib, &opts)?;
        let comp_mib_s = (cfg.size_mib as f64) / (hybrid.comp_ms / 1000.0);
        let decomp_mib_s = (cfg.size_mib as f64) / (hybrid.decomp_ms / 1000.0);
        println!(
            "run {}/{}: comp_ms={:.3} decomp_ms={:.3} comp_mib_s={:.2} decomp_mib_s={:.2} ratio={:.4}",
            run + 1,
            cfg.runs,
            hybrid.comp_ms,
            hybrid.decomp_ms,
            comp_mib_s,
            decomp_mib_s,
            hybrid.ratio
        );
        hybrid_comp_ms.push(hybrid.comp_ms);
        hybrid_decomp_ms.push(hybrid.decomp_ms);
        hybrid_ratios.push(hybrid.ratio);

        if cfg.compare_hybrid {
            let cpu = run_roundtrip(&input, cfg.size_mib, &opts_cpu)?;
            let cpu_comp_mib_s = (cfg.size_mib as f64) / (cpu.comp_ms / 1000.0);
            let cpu_decomp_mib_s = (cfg.size_mib as f64) / (cpu.decomp_ms / 1000.0);
            println!(
                "CPU_ONLY: comp_ms={:.3} decomp_ms={:.3} comp_mib_s={:.2} decomp_mib_s={:.2} ratio={:.4}",
                cpu.comp_ms, cpu.decomp_ms, cpu_comp_mib_s, cpu_decomp_mib_s, cpu.ratio
            );
            println!(
                "HYBRID  : comp_ms={:.3} decomp_ms={:.3} comp_mib_s={:.2} decomp_mib_s={:.2} ratio={:.4}",
                hybrid.comp_ms, hybrid.decomp_ms, comp_mib_s, decomp_mib_s, hybrid.ratio
            );
            let s_comp = cpu.comp_ms / hybrid.comp_ms;
            let s_decomp = cpu.decomp_ms / hybrid.decomp_ms;
            println!(
                "speedup(cpu/hybrid): compress={:.3}x decompress={:.3}x ( >1.0 means HYBRID faster )",
                s_comp, s_decomp
            );
            cpu_comp_ms.push(cpu.comp_ms);
            cpu_decomp_ms.push(cpu.decomp_ms);
            speedup_comp.push(s_comp);
            speedup_decomp.push(s_decomp);
        }
    }

    println!("----- summary -----");
    println!(
        "comp_ms: mean={:.3} median={:.3} min={:.3} max={:.3}",
        mean(&hybrid_comp_ms),
        median(&hybrid_comp_ms),
        min(&hybrid_comp_ms),
        max(&hybrid_comp_ms)
    );
    println!(
        "decomp_ms: mean={:.3} median={:.3} min={:.3} max={:.3}",
        mean(&hybrid_decomp_ms),
        median(&hybrid_decomp_ms),
        min(&hybrid_decomp_ms),
        max(&hybrid_decomp_ms)
    );
    println!(
        "ratio: mean={:.4} median={:.4} min={:.4} max={:.4}",
        mean(&hybrid_ratios),
        median(&hybrid_ratios),
        min(&hybrid_ratios),
        max(&hybrid_ratios)
    );
    if cfg.compare_hybrid {
        println!(
            "cpu_only_comp_ms: mean={:.3} median={:.3} min={:.3} max={:.3}",
            mean(&cpu_comp_ms),
            median(&cpu_comp_ms),
            min(&cpu_comp_ms),
            max(&cpu_comp_ms)
        );
        println!(
            "cpu_only_decomp_ms: mean={:.3} median={:.3} min={:.3} max={:.3}",
            mean(&cpu_decomp_ms),
            median(&cpu_decomp_ms),
            min(&cpu_decomp_ms),
            max(&cpu_decomp_ms)
        );
        println!(
            "speedup_compress_x: mean={:.3} median={:.3} min={:.3} max={:.3}",
            mean(&speedup_comp),
            median(&speedup_comp),
            min(&speedup_comp),
            max(&speedup_comp)
        );
        println!(
            "speedup_decompress_x: mean={:.3} median={:.3} min={:.3} max={:.3}",
            mean(&speedup_decomp),
            median(&speedup_decomp),
            min(&speedup_decomp),
            max(&speedup_decomp)
        );
    }

    Ok(())
}
