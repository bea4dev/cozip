use std::time::{Duration, Instant};

use cozip_deflate::{CompressionMode, HybridOptions, HybridStats, compress_hybrid, decompress_hybrid};

#[derive(Debug, Clone)]
struct BenchAgg {
    compress_total: Duration,
    decompress_total: Duration,
    compressed_total: usize,
    input_total: usize,
    last_stats: HybridStats,
}

impl Default for BenchAgg {
    fn default() -> Self {
        Self {
            compress_total: Duration::ZERO,
            decompress_total: Duration::ZERO,
            compressed_total: 0,
            input_total: 0,
            last_stats: HybridStats::default(),
        }
    }
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

fn run_case(input: &[u8], options: &HybridOptions, iters: usize) -> BenchAgg {
    let mut agg = BenchAgg::default();

    for _ in 0..iters {
        let start_c = Instant::now();
        let compressed = compress_hybrid(input, options).expect("compress_hybrid should succeed");
        let elapsed_c = start_c.elapsed();

        let start_d = Instant::now();
        let decompressed = decompress_hybrid(&compressed.bytes, options)
            .expect("decompress_hybrid should succeed");
        let elapsed_d = start_d.elapsed();

        assert_eq!(decompressed.bytes, input);
        agg.add(
            input.len(),
            compressed.bytes.len(),
            elapsed_c,
            elapsed_d,
            compressed.stats,
        );
    }

    agg
}

fn main() {
    let sizes = [4 * 1024 * 1024, 16 * 1024 * 1024];
    let iters = 5;

    println!("cozip_deflate benchmark (release)");
    println!("iters={}", iters);
    println!("sizes={:?}", sizes);

    for size in sizes {
        let input = build_mixed_dataset(size);
        let mib = size as f64 / (1024.0 * 1024.0);

        let cpu_only = HybridOptions {
            chunk_size: 4 * 1024 * 1024,
            gpu_subchunk_size: 256 * 1024,
            compression_level: 6,
            compression_mode: CompressionMode::Speed,
            prefer_gpu: false,
            gpu_fraction: 0.0,
            gpu_min_chunk_size: 64 * 1024,
        };

        let cpu_gpu = HybridOptions {
            chunk_size: 4 * 1024 * 1024,
            gpu_subchunk_size: 256 * 1024,
            compression_level: 6,
            compression_mode: CompressionMode::Speed,
            prefer_gpu: true,
            gpu_fraction: 0.5,
            gpu_min_chunk_size: 64 * 1024,
        };

        let _ = run_case(&input, &cpu_only, 1);
        let _ = run_case(&input, &cpu_gpu, 1);

        let cpu = run_case(&input, &cpu_only, iters);
        let hybrid = run_case(&input, &cpu_gpu, iters);

        println!();
        println!("size_bytes={} ({:.1} MiB)", size, mib);
        println!(
            "CPU_ONLY: avg_comp_ms={:.3} avg_decomp_ms={:.3} comp_mib_s={:.2} decomp_mib_s={:.2} ratio={:.4} chunks={} cpu_chunks={} gpu_chunks={} gpu_available={}",
            cpu.avg_compress_ms(iters),
            cpu.avg_decompress_ms(iters),
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
            hybrid.avg_compress_ms(iters),
            hybrid.avg_decompress_ms(iters),
            hybrid.compress_mib_s(),
            hybrid.decompress_mib_s(),
            hybrid.ratio(),
            hybrid.last_stats.chunk_count,
            hybrid.last_stats.cpu_chunks,
            hybrid.last_stats.gpu_chunks,
            hybrid.last_stats.gpu_available,
        );
    }
}
