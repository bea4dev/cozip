use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

use cozip_pdeflate::{PDeflateOptions, pdeflate_compress_with_stats, pdeflate_decompress_into};
use flate2::Compression;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;

#[derive(Debug, Clone)]
struct BenchConfig {
    size_mib: usize,
    runs: usize,
    warmups: usize,
    chunk_mib: usize,
    sections: usize,
    workers: usize,
    flate_level: u32,
    verify_bytes: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        let workers = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            size_mib: 1024,
            runs: 3,
            warmups: 1,
            chunk_mib: 4,
            sections: 128,
            workers,
            flate_level: 6,
            verify_bytes: true,
        }
    }
}

#[derive(Debug)]
struct CompressResult {
    ms: f64,
    mib_s: f64,
    ratio: f64,
    compressed_chunks: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Copy)]
struct DecompressResult {
    ms: f64,
    mib_s: f64,
    decode_ms: f64,
    verify_ms: f64,
}

#[derive(Debug, Clone, Copy)]
enum Algo {
    PDeflate,
    Flate2,
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
            "--workers" => {
                i += 1;
                cfg.workers = args
                    .get(i)
                    .ok_or("--workers requires value")?
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --workers: {e}"))?;
            }
            "--flate-level" => {
                i += 1;
                cfg.flate_level = args
                    .get(i)
                    .ok_or("--flate-level requires value")?
                    .parse::<u32>()
                    .map_err(|e| format!("invalid --flate-level: {e}"))?;
            }
            "--verify-bytes" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--verify-bytes requires value (0|1|false|true)")?;
                cfg.verify_bytes = match v.as_str() {
                    "1" | "true" | "TRUE" | "True" => true,
                    "0" | "false" | "FALSE" | "False" => false,
                    _ => return Err(format!("invalid --verify-bytes: {v}")),
                };
            }
            "--verify" => {
                cfg.verify_bytes = true;
            }
            "--no-verify" => {
                cfg.verify_bytes = false;
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
    if cfg.workers == 0 {
        return Err("--workers must be > 0".to_string());
    }
    if cfg.flate_level > 9 {
        return Err("--flate-level must be in 0..=9".to_string());
    }

    Ok(cfg)
}

fn print_help() {
    println!(
        "usage: cargo run --release -p cozip_pdeflate --example bench_tokenize_vs_flate2 -- [options]\n\
options:\n\
  --size-mib <N>      input size in MiB (default: 1024)\n\
  --runs <N>          measured runs (default: 3)\n\
  --warmups <N>       warmup runs (default: 1)\n\
  --chunk-mib <N>     chunk size in MiB (default: 4)\n\
  --sections <N>      pdeflate sections per chunk (default: 128)\n\
  --workers <N>       worker threads for both paths (default: hw threads)\n\
  --flate-level <N>   flate2 level 0..9 (default: 6)\n\
  --verify            enable strict decoded-bytes check (default)\n\
  --no-verify         disable decoded-bytes check\n\
  --verify-bytes <B>  strict decoded-bytes check (0/1, default: 1)\n\
  -h, --help          show help"
    );
}

fn generate_input(size_bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; size_bytes];
    let text = b"ABABABABCCABCCD--cozip-tokenize-vs-flate2--";
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

fn chunk_ranges(total: usize, chunk_size: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < total {
        let end = (pos + chunk_size).min(total);
        out.push((pos, end));
        pos = end;
    }
    out
}

fn run_parallel_compress(
    algo: Algo,
    input: &[u8],
    ranges: &[(usize, usize)],
    cfg: &BenchConfig,
) -> Result<CompressResult, String> {
    let total_in = input.len();
    let next = AtomicUsize::new(0);
    let t0 = Instant::now();
    let mut total_out = 0usize;
    let mut chunks_by_index: Vec<Option<Vec<u8>>> = (0..ranges.len()).map(|_| None).collect();

    thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::with_capacity(cfg.workers);
        for _ in 0..cfg.workers {
            handles.push(
                scope.spawn(|| -> Result<(usize, Vec<(usize, Vec<u8>)>), String> {
                    let mut local_out = 0usize;
                    let mut local_chunks = Vec::new();
                    loop {
                        let idx = next.fetch_add(1, Ordering::Relaxed);
                        if idx >= ranges.len() {
                            break;
                        }
                        let (start, end) = ranges[idx];
                        let chunk = &input[start..end];
                        let compressed = match algo {
                            Algo::PDeflate => {
                                let mut opts = PDeflateOptions::default();
                                opts.chunk_size = chunk.len().max(1);
                                opts.section_count = cfg.sections;
                                let (compressed, _) = pdeflate_compress_with_stats(chunk, &opts)
                                    .map_err(|e| format!("pdeflate chunk compress failed: {e}"))?;
                                compressed
                            }
                            Algo::Flate2 => {
                                let mut encoder = DeflateEncoder::new(
                                    Vec::new(),
                                    Compression::new(cfg.flate_level.clamp(0, 9)),
                                );
                                encoder
                                    .write_all(chunk)
                                    .map_err(|e| format!("flate2 write failed: {e}"))?;
                                encoder
                                    .finish()
                                    .map_err(|e| format!("flate2 finish failed: {e}"))?
                            }
                        };
                        local_out = local_out.saturating_add(compressed.len());
                        local_chunks.push((idx, compressed));
                    }
                    Ok((local_out, local_chunks))
                }),
            );
        }

        for handle in handles {
            let (local_out, local_chunks) = handle
                .join()
                .map_err(|_| "compression worker panicked".to_string())??;
            total_out = total_out.saturating_add(local_out);
            for (idx, chunk) in local_chunks {
                chunks_by_index[idx] = Some(chunk);
            }
        }
        Ok(())
    })?;

    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let size_mib = total_in as f64 / (1024.0 * 1024.0);
    let mib_s = if ms > 0.0 {
        size_mib * 1000.0 / ms
    } else {
        0.0
    };
    let ratio = if total_in > 0 {
        total_out as f64 / total_in as f64
    } else {
        0.0
    };
    let compressed_chunks = chunks_by_index
        .into_iter()
        .enumerate()
        .map(|(idx, c)| c.ok_or_else(|| format!("missing compressed chunk idx={idx}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CompressResult {
        ms,
        mib_s,
        ratio,
        compressed_chunks,
    })
}

fn run_parallel_decompress(
    algo: Algo,
    compressed_chunks: &[Vec<u8>],
    input: &[u8],
    ranges: &[(usize, usize)],
    cfg: &BenchConfig,
) -> Result<DecompressResult, String> {
    if compressed_chunks.len() != ranges.len() {
        return Err("decompress input/chunk-range length mismatch".to_string());
    }
    let total_out_expected: usize = ranges.iter().map(|(s, e)| e.saturating_sub(*s)).sum();
    let next = AtomicUsize::new(0);
    let t0 = Instant::now();
    let mut total_out = 0usize;
    let mut total_decode_ms = 0.0_f64;
    let mut total_verify_ms = 0.0_f64;

    thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::with_capacity(cfg.workers);
        for _ in 0..cfg.workers {
            handles.push(scope.spawn(|| -> Result<(usize, f64, f64), String> {
                let mut local_out = 0usize;
                let mut local_decode_ms = 0.0_f64;
                let mut local_verify_ms = 0.0_f64;
                let mut decode_buf = Vec::new();
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= compressed_chunks.len() {
                        break;
                    }
                    let expected_len = ranges[idx].1.saturating_sub(ranges[idx].0);
                    let decode_t0 = Instant::now();
                    match algo {
                        Algo::PDeflate => {
                            pdeflate_decompress_into(&compressed_chunks[idx], &mut decode_buf)
                                .map_err(|e| format!("pdeflate chunk decompress failed: {e}"))?
                        }
                        Algo::Flate2 => {
                            decode_buf.clear();
                            let mut decoder = DeflateDecoder::new(&compressed_chunks[idx][..]);
                            decoder
                                .read_to_end(&mut decode_buf)
                                .map_err(|e| format!("flate2 read_to_end failed: {e}"))?;
                        }
                    }
                    local_decode_ms += decode_t0.elapsed().as_secs_f64() * 1000.0;
                    let decoded = &decode_buf;
                    if decoded.len() != expected_len {
                        return Err(format!(
                            "decoded length mismatch at chunk {}: got {}, expected {}",
                            idx,
                            decoded.len(),
                            expected_len
                        ));
                    }
                    if cfg.verify_bytes {
                        let verify_t0 = Instant::now();
                        let (s, e) = ranges[idx];
                        let expected = &input[s..e];
                        if decoded.as_slice() != expected {
                            let mismatch = decoded
                                .iter()
                                .zip(expected.iter())
                                .position(|(a, b)| a != b)
                                .unwrap_or(0);
                            return Err(format!(
                                "decoded content mismatch at chunk {} (offset {} in chunk)",
                                idx, mismatch
                            ));
                        }
                        local_verify_ms += verify_t0.elapsed().as_secs_f64() * 1000.0;
                    }
                    local_out = local_out.saturating_add(decoded.len());
                }
                Ok((local_out, local_decode_ms, local_verify_ms))
            }));
        }

        for handle in handles {
            let (local_out, local_decode_ms, local_verify_ms) = handle
                .join()
                .map_err(|_| "decompression worker panicked".to_string())??;
            total_out = total_out.saturating_add(local_out);
            total_decode_ms += local_decode_ms;
            total_verify_ms += local_verify_ms;
        }
        Ok(())
    })?;

    if total_out != total_out_expected {
        return Err(format!(
            "decoded total mismatch: got {}, expected {}",
            total_out, total_out_expected
        ));
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let size_mib = total_out as f64 / (1024.0 * 1024.0);
    let mib_s = if ms > 0.0 {
        size_mib * 1000.0 / ms
    } else {
        0.0
    };
    Ok(DecompressResult {
        ms,
        mib_s,
        decode_ms: total_decode_ms,
        verify_ms: total_verify_ms,
    })
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

fn print_summary(name: &str, vals_ms: &[f64], vals_mib_s: &[f64], vals_ratio: &[f64]) {
    println!(
        "{} ms: n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
        name,
        vals_ms.len(),
        mean(vals_ms),
        median(vals_ms),
        min(vals_ms),
        max(vals_ms)
    );
    println!(
        "{} mib_s: n={} mean={:.2} median={:.2} min={:.2} max={:.2}",
        name,
        vals_mib_s.len(),
        mean(vals_mib_s),
        median(vals_mib_s),
        min(vals_mib_s),
        max(vals_mib_s)
    );
    println!(
        "{} ratio: n={} mean={:.4} median={:.4} min={:.4} max={:.4}",
        name,
        vals_ratio.len(),
        mean(vals_ratio),
        median(vals_ratio),
        min(vals_ratio),
        max(vals_ratio)
    );
}

fn print_summary_no_ratio(name: &str, vals_ms: &[f64], vals_mib_s: &[f64]) {
    println!(
        "{} ms: n={} mean={:.3} median={:.3} min={:.3} max={:.3}",
        name,
        vals_ms.len(),
        mean(vals_ms),
        median(vals_ms),
        min(vals_ms),
        max(vals_ms)
    );
    println!(
        "{} mib_s: n={} mean={:.2} median={:.2} min={:.2} max={:.2}",
        name,
        vals_mib_s.len(),
        mean(vals_mib_s),
        median(vals_mib_s),
        min(vals_mib_s),
        max(vals_mib_s)
    );
}

fn main() -> Result<(), String> {
    let cfg = parse_args()?;
    let input = generate_input(cfg.size_mib * 1024 * 1024);
    let ranges = chunk_ranges(input.len(), cfg.chunk_mib * 1024 * 1024);

    println!(
        "cozip_pdeflate vs flate2 (chunk-parallel)\nsize_mib={} runs={} warmups={} chunk_mib={} sections={} workers={} flate_level={}",
        cfg.size_mib,
        cfg.runs,
        cfg.warmups,
        cfg.chunk_mib,
        cfg.sections,
        cfg.workers,
        cfg.flate_level
    );
    println!("verify_bytes={}", cfg.verify_bytes);
    println!(
        "chunks={} (same chunk set / same worker count for both paths)",
        ranges.len()
    );

    for _ in 0..cfg.warmups {
        let p = run_parallel_compress(Algo::PDeflate, &input, &ranges, &cfg)?;
        let _ =
            run_parallel_decompress(Algo::PDeflate, &p.compressed_chunks, &input, &ranges, &cfg)?;
        let f = run_parallel_compress(Algo::Flate2, &input, &ranges, &cfg)?;
        let _ = run_parallel_decompress(Algo::Flate2, &f.compressed_chunks, &input, &ranges, &cfg)?;
    }

    let mut p_ms = Vec::with_capacity(cfg.runs);
    let mut p_mib_s = Vec::with_capacity(cfg.runs);
    let mut p_ratio = Vec::with_capacity(cfg.runs);
    let mut f_ms = Vec::with_capacity(cfg.runs);
    let mut f_mib_s = Vec::with_capacity(cfg.runs);
    let mut f_ratio = Vec::with_capacity(cfg.runs);
    let mut p_d_ms = Vec::with_capacity(cfg.runs);
    let mut p_d_mib_s = Vec::with_capacity(cfg.runs);
    let mut f_d_ms = Vec::with_capacity(cfg.runs);
    let mut f_d_mib_s = Vec::with_capacity(cfg.runs);

    for i in 0..cfg.runs {
        let p = run_parallel_compress(Algo::PDeflate, &input, &ranges, &cfg)?;
        let p_d =
            run_parallel_decompress(Algo::PDeflate, &p.compressed_chunks, &input, &ranges, &cfg)?;
        let f = run_parallel_compress(Algo::Flate2, &input, &ranges, &cfg)?;
        let f_d =
            run_parallel_decompress(Algo::Flate2, &f.compressed_chunks, &input, &ranges, &cfg)?;
        let speedup_comp = if p.ms > 0.0 { f.ms / p.ms } else { 0.0 };
        let speedup_decomp = if p_d.ms > 0.0 { f_d.ms / p_d.ms } else { 0.0 };
        println!(
            "run {}/{}: pdeflate_comp_ms={:.3} flate2_comp_ms={:.3} pdeflate_decomp_ms={:.3} flate2_decomp_ms={:.3} pdeflate_decomp_decode_ms={:.3} pdeflate_verify_ms={:.3} flate2_decomp_decode_ms={:.3} flate2_verify_ms={:.3} pdeflate_comp_mib_s={:.2} flate2_comp_mib_s={:.2} pdeflate_decomp_mib_s={:.2} flate2_decomp_mib_s={:.2} pdeflate_ratio={:.4} flate2_ratio={:.4} speedup_comp(flate2/pdeflate)={:.3}x speedup_decomp(flate2/pdeflate)={:.3}x",
            i + 1,
            cfg.runs,
            p.ms,
            f.ms,
            p_d.ms,
            f_d.ms,
            p_d.decode_ms,
            p_d.verify_ms,
            f_d.decode_ms,
            f_d.verify_ms,
            p.mib_s,
            f.mib_s,
            p_d.mib_s,
            f_d.mib_s,
            p.ratio,
            f.ratio,
            speedup_comp,
            speedup_decomp
        );
        p_ms.push(p.ms);
        p_mib_s.push(p.mib_s);
        p_ratio.push(p.ratio);
        f_ms.push(f.ms);
        f_mib_s.push(f.mib_s);
        f_ratio.push(f.ratio);
        p_d_ms.push(p_d.ms);
        p_d_mib_s.push(p_d.mib_s);
        f_d_ms.push(f_d.ms);
        f_d_mib_s.push(f_d.mib_s);
    }

    println!("----- SUMMARY -----");
    print_summary("pdeflate_comp", &p_ms, &p_mib_s, &p_ratio);
    print_summary("flate2_comp", &f_ms, &f_mib_s, &f_ratio);
    print_summary_no_ratio("pdeflate_decomp", &p_d_ms, &p_d_mib_s);
    print_summary_no_ratio("flate2_decomp", &f_d_ms, &f_d_mib_s);
    let mean_speedup_comp = if mean(&p_ms) > 0.0 {
        mean(&f_ms) / mean(&p_ms)
    } else {
        0.0
    };
    let mean_speedup_decomp = if mean(&p_d_ms) > 0.0 {
        mean(&f_d_ms) / mean(&p_d_ms)
    } else {
        0.0
    };
    println!(
        "speedup_comp(flate2/pdeflate): mean={:.3}x ( >1.0 means pdeflate faster )",
        mean_speedup_comp
    );
    println!(
        "speedup_decomp(flate2/pdeflate): mean={:.3}x ( >1.0 means pdeflate faster )",
        mean_speedup_decomp
    );
    println!(
        "note: this benchmark reports both full compression and full decompression under matched worker/chunk conditions."
    );

    Ok(())
}
