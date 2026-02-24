use std::time::{Duration, Instant};

use cozip_deflate::{CompressionMode, HybridOptions, compress_hybrid, decompress_hybrid};

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

fn timed_roundtrip(
    input: &[u8],
    options: &HybridOptions,
) -> (
    cozip_deflate::CompressedFrame,
    cozip_deflate::DecompressedFrame,
    Duration,
    Duration,
) {
    let compress_start = Instant::now();
    let compressed = compress_hybrid(input, options).expect("compress_hybrid should succeed");
    let compress_elapsed = compress_start.elapsed();

    let decompress_start = Instant::now();
    let decompressed =
        decompress_hybrid(&compressed.bytes, options).expect("decompress_hybrid should succeed");
    let decompress_elapsed = decompress_start.elapsed();

    assert_eq!(decompressed.bytes, input);
    (
        compressed,
        decompressed,
        compress_elapsed,
        decompress_elapsed,
    )
}

#[test]
fn compare_cpu_only_vs_cpu_gpu_with_nocapture() {
    let input = build_mixed_dataset(6 * 1024 * 1024 + 321);

    let cpu_only = HybridOptions {
        chunk_size: 256 * 1024,
        gpu_subchunk_size: 128 * 1024,
        compression_level: 6,
        compression_mode: CompressionMode::Speed,
        prefer_gpu: false,
        gpu_fraction: 0.0,
        gpu_min_chunk_size: 64 * 1024,
    };

    let cpu_gpu = HybridOptions {
        chunk_size: 256 * 1024,
        gpu_subchunk_size: 128 * 1024,
        compression_level: 6,
        compression_mode: CompressionMode::Speed,
        prefer_gpu: true,
        gpu_fraction: 0.5,
        gpu_min_chunk_size: 64 * 1024,
    };

    let (cpu_cmp, cpu_dec, cpu_comp_t, cpu_decomp_t) = timed_roundtrip(&input, &cpu_only);
    let (hy_cmp, hy_dec, hy_comp_t, hy_decomp_t) = timed_roundtrip(&input, &cpu_gpu);

    let cpu_ratio = cpu_cmp.bytes.len() as f64 / input.len() as f64;
    let hy_ratio = hy_cmp.bytes.len() as f64 / input.len() as f64;

    println!("=== cozip_deflate integration compare ===");
    println!("input_bytes={}", input.len());
    println!(
        "[CPU_ONLY] compress_ms={:.3} decompress_ms={:.3} compressed_bytes={} ratio={:.4} chunks={} cpu_chunks={} gpu_chunks={} gpu_available={}",
        cpu_comp_t.as_secs_f64() * 1000.0,
        cpu_decomp_t.as_secs_f64() * 1000.0,
        cpu_cmp.bytes.len(),
        cpu_ratio,
        cpu_cmp.stats.chunk_count,
        cpu_cmp.stats.cpu_chunks,
        cpu_cmp.stats.gpu_chunks,
        cpu_cmp.stats.gpu_available,
    );
    println!(
        "[CPU+GPU ] compress_ms={:.3} decompress_ms={:.3} compressed_bytes={} ratio={:.4} chunks={} cpu_chunks={} gpu_chunks={} gpu_available={}",
        hy_comp_t.as_secs_f64() * 1000.0,
        hy_decomp_t.as_secs_f64() * 1000.0,
        hy_cmp.bytes.len(),
        hy_ratio,
        hy_cmp.stats.chunk_count,
        hy_cmp.stats.cpu_chunks,
        hy_cmp.stats.gpu_chunks,
        hy_cmp.stats.gpu_available,
    );

    assert_eq!(cpu_dec.bytes, hy_dec.bytes);

    if hy_cmp.stats.gpu_available {
        assert!(
            hy_cmp.stats.gpu_chunks > 0,
            "GPU is available, but no chunk was assigned to GPU"
        );
    } else {
        println!("GPU unavailable on this machine; CPU+GPU mode fell back to CPU only.");
    }
}

#[test]
fn hybrid_uses_both_cpu_and_gpu_when_gpu_is_available() {
    let input = build_mixed_dataset(8 * 1024 * 1024 + 111);
    let options = HybridOptions {
        chunk_size: 256 * 1024,
        gpu_subchunk_size: 128 * 1024,
        compression_level: 6,
        compression_mode: CompressionMode::Speed,
        prefer_gpu: true,
        gpu_fraction: 0.5,
        gpu_min_chunk_size: 64 * 1024,
    };

    let (compressed, _decompressed, _ct, _dt) = timed_roundtrip(&input, &options);
    println!(
        "hybrid stats: chunks={} cpu_chunks={} gpu_chunks={} gpu_available={}",
        compressed.stats.chunk_count,
        compressed.stats.cpu_chunks,
        compressed.stats.gpu_chunks,
        compressed.stats.gpu_available
    );

    if !compressed.stats.gpu_available {
        println!(
            "GPU unavailable on this machine; skipping strict CPU+GPU distribution assertion."
        );
        return;
    }

    assert!(
        compressed.stats.cpu_chunks > 0,
        "at least one CPU chunk is expected"
    );
    assert!(
        compressed.stats.gpu_chunks > 0,
        "at least one GPU chunk is expected"
    );
}
