use std::env;
use std::time::{Duration, Instant};

use cozip_deflate::{CoZip, CompressionMode, HybridOptions, HybridStats};

#[derive(Debug, Clone)]
struct BenchConfig {
    size_mib: usize,
    iters: usize,
    warmups: usize,
    chunk_mib: usize,
    gpu_subchunk_kib: usize,
    gpu_fraction: f32,
    mode: CompressionMode,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            size_mib: 1024,
            iters: 1,
            warmups: 0,
            chunk_mib: 4,
            gpu_subchunk_kib: 256,
            gpu_fraction: 1.0,
            mode: CompressionMode::Speed,
        }
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
                "--gpu-fraction" => {
                    cfg.gpu_fraction = value
                        .parse::<f32>()
                        .map_err(|_| "invalid --gpu-fraction".to_string())?;
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

        if cfg.size_mib == 0 || cfg.iters == 0 || cfg.chunk_mib == 0 || cfg.gpu_subchunk_kib == 0 {
            return Err("size/iters/chunk/subchunk must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&cfg.gpu_fraction) {
            return Err("gpu-fraction must be in range 0.0..=1.0".to_string());
        }

        Ok(cfg)
    }
}

fn help_text() -> String {
    let text = r#"usage: cargo run --release -p cozip_deflate --example bench_1gb -- [options]
  --size-mib <N>           input size in MiB (default: 1024)
  --iters <N>              measured iterations (default: 1)
  --warmups <N>            warmup iterations (default: 0)
  --chunk-mib <N>          host chunk size in MiB (default: 4)
  --gpu-subchunk-kib <N>   gpu subchunk size in KiB (default: 256)
  --gpu-fraction <R>       gpu scheduling target ratio 0.0..=1.0 (default: 1.0)
  --mode <M>               speed|balanced|ratio (default: speed)"#;
    text.to_string()
}

#[derive(Debug, Clone, Default)]
struct BenchAgg {
    compress_total: Duration,
    decompress_total: Duration,
    compressed_total: usize,
    input_total: usize,
    last_stats: HybridStats,
}

impl BenchAgg {
    fn add(
        &mut self,
        input_len: usize,
        compressed_len: usize,
        compress_elapsed: Duration,
        decompress_elapsed: Duration,
        stats: HybridStats,
    ) {
        self.compress_total += compress_elapsed;
        self.decompress_total += decompress_elapsed;
        self.compressed_total += compressed_len;
        self.input_total += input_len;
        self.last_stats = stats;
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

fn run_case(input: &[u8], cozip: &CoZip, iters: usize) -> BenchAgg {
    let mut agg = BenchAgg::default();
    for _ in 0..iters {
        let t0 = Instant::now();
        let compressed = cozip.compress(input).expect("compress should succeed");
        let compress_elapsed = t0.elapsed();

        let t1 = Instant::now();
        let decompressed = cozip
            .decompress(&compressed.bytes)
            .expect("decompress should succeed");
        let decompress_elapsed = t1.elapsed();

        assert_eq!(decompressed.bytes, input);

        agg.add(
            input.len(),
            compressed.bytes.len(),
            compress_elapsed,
            decompress_elapsed,
            compressed.stats,
        );
    }
    agg
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
        compression_level: 6,
        compression_mode: cfg.mode,
        prefer_gpu: false,
        gpu_fraction: 0.0,
        gpu_min_chunk_size: 64 * 1024,
        ..HybridOptions::default()
    };

    let cpu_gpu = HybridOptions {
        chunk_size,
        gpu_subchunk_size,
        compression_level: 6,
        compression_mode: cfg.mode,
        prefer_gpu: true,
        gpu_fraction: cfg.gpu_fraction,
        gpu_min_chunk_size: 64 * 1024,
        ..HybridOptions::default()
    };

    println!("cozip_deflate 1GB-class benchmark");
    println!(
        "size_mib={} iters={} warmups={} chunk_mib={} gpu_subchunk_kib={} mode={:?}",
        cfg.size_mib, cfg.iters, cfg.warmups, cfg.chunk_mib, cfg.gpu_subchunk_kib, cfg.mode
    );
    println!("gpu_fraction={:.2}", cfg.gpu_fraction);

    let cpu_only_cozip = CoZip::init(cpu_only).expect("cpu-only init should succeed");
    let cpu_gpu_cozip = CoZip::init(cpu_gpu).expect("cpu+gpu init should succeed");

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
    let decomp_speedup = if hybrid.avg_decompress_ms(cfg.iters) > 0.0 {
        cpu.avg_decompress_ms(cfg.iters) / hybrid.avg_decompress_ms(cfg.iters)
    } else {
        0.0
    };

    println!(
        "CPU_ONLY: avg_comp_ms={:.3} avg_decomp_ms={:.3} comp_mib_s={:.2} decomp_mib_s={:.2} ratio={:.4} chunks={} cpu_chunks={} gpu_chunks={} gpu_available={}",
        cpu.avg_compress_ms(cfg.iters),
        cpu.avg_decompress_ms(cfg.iters),
        cpu.compress_mib_s(),
        cpu.decompress_mib_s(),
        cpu.ratio(),
        cpu.last_stats.chunk_count,
        cpu.last_stats.cpu_chunks,
        cpu.last_stats.gpu_chunks,
        cpu.last_stats.gpu_available,
    );
    println!(
        "CPU+GPU : avg_comp_ms={:.3} avg_decomp_ms={:.3} comp_mib_s={:.2} decomp_mib_s={:.2} ratio={:.4} chunks={} cpu_chunks={} gpu_chunks={} gpu_available={}",
        hybrid.avg_compress_ms(cfg.iters),
        hybrid.avg_decompress_ms(cfg.iters),
        hybrid.compress_mib_s(),
        hybrid.decompress_mib_s(),
        hybrid.ratio(),
        hybrid.last_stats.chunk_count,
        hybrid.last_stats.cpu_chunks,
        hybrid.last_stats.gpu_chunks,
        hybrid.last_stats.gpu_available,
    );
    println!(
        "speedup(cpu/hybrid): compress={:.3}x decompress={:.3}x ( >1.0 means CPU+GPU faster )",
        comp_speedup, decomp_speedup
    );
}
