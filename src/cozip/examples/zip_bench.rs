use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cozip::{CoZip, CoZipError, CoZipOptions, CoZipStats, ZipDeflateMode, ZipOptions};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputMode {
    Auto,
    File,
    Directory,
}

#[derive(Debug, Clone)]
struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    level: u32,
    warmups: usize,
    runs: usize,
    mode: InputMode,
    engine: ZipDeflateMode,
    async_mode: bool,
}

fn print_usage() {
    println!(
        "usage: cargo run -p cozip --example zip_bench -- \\\n  --input <path> [--output <zip>] [--mode auto|file|dir] [--engine hybrid|cpu] [--level <0..9>] \\\n  [--warmups <n>] [--runs <n>] [--async]"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut level: u32 = 6;
    let mut warmups: usize = 0;
    let mut runs: usize = 1;
    let mut mode = InputMode::Auto;
    let mut engine = ZipDeflateMode::Hybrid;
    let mut async_mode = false;

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--input" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--input requires a value".to_string())?;
                input = Some(PathBuf::from(v));
            }
            "--output" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--output requires a value".to_string())?;
                output = Some(PathBuf::from(v));
            }
            "--level" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--level requires a value".to_string())?;
                level = v
                    .parse::<u32>()
                    .map_err(|_| format!("invalid --level: {v}"))?;
                if level > 9 {
                    return Err("--level must be within 0..=9".to_string());
                }
            }
            "--warmups" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--warmups requires a value".to_string())?;
                warmups = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --warmups: {v}"))?;
            }
            "--runs" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--runs requires a value".to_string())?;
                runs = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --runs: {v}"))?;
                if runs == 0 {
                    return Err("--runs must be >= 1".to_string());
                }
            }
            "--mode" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--mode requires a value".to_string())?;
                mode = match v.as_str() {
                    "auto" => InputMode::Auto,
                    "file" => InputMode::File,
                    "dir" | "directory" => InputMode::Directory,
                    _ => return Err(format!("invalid --mode: {v}")),
                };
            }
            "--engine" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--engine requires a value".to_string())?;
                engine = match v.as_str() {
                    "hybrid" => ZipDeflateMode::Hybrid,
                    "cpu" => ZipDeflateMode::Cpu,
                    _ => return Err(format!("invalid --engine: {v}")),
                };
            }
            "--async" => {
                async_mode = true;
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {other}"));
            }
            positional => {
                if input.is_none() {
                    input = Some(PathBuf::from(positional));
                } else {
                    return Err(format!("unexpected positional argument: {positional}"));
                }
            }
        }
    }

    let input = input.ok_or_else(|| "--input <path> is required".to_string())?;
    Ok(Args {
        input,
        output,
        level,
        warmups,
        runs,
        mode,
        engine,
        async_mode,
    })
}

fn determine_mode(path: &Path, mode: InputMode) -> Result<InputMode, String> {
    match mode {
        InputMode::File => {
            if path.is_file() {
                Ok(InputMode::File)
            } else {
                Err("--mode file was specified, but input is not a file".to_string())
            }
        }
        InputMode::Directory => {
            if path.is_dir() {
                Ok(InputMode::Directory)
            } else {
                Err("--mode dir was specified, but input is not a directory".to_string())
            }
        }
        InputMode::Auto => {
            if path.is_file() {
                Ok(InputMode::File)
            } else if path.is_dir() {
                Ok(InputMode::Directory)
            } else {
                Err("input path is neither file nor directory".to_string())
            }
        }
    }
}

fn default_output_path(input: &Path, mode: InputMode) -> Result<PathBuf, String> {
    match mode {
        InputMode::File => Ok(input.with_extension("zip")),
        InputMode::Directory => {
            let parent = input.parent().unwrap_or_else(|| Path::new("."));
            let name = input
                .file_name()
                .ok_or_else(|| "directory has no file name".to_string())?;
            let mut out_name = OsString::from(name);
            out_name.push(".zip");
            Ok(parent.join(out_name))
        }
        InputMode::Auto => Err("internal error: unresolved mode".to_string()),
    }
}

fn input_size_bytes(input: &Path, mode: InputMode) -> Result<u64, CoZipError> {
    match mode {
        InputMode::File => Ok(fs::metadata(input)?.len()),
        InputMode::Directory => {
            let mut total = 0_u64;
            let mut queue = vec![input.to_path_buf()];
            while let Some(dir) = queue.pop() {
                for entry in fs::read_dir(&dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.is_dir() {
                        queue.push(path);
                    } else if path.is_file() {
                        total = total.saturating_add(fs::metadata(path)?.len());
                    }
                }
            }
            Ok(total)
        }
        InputMode::Auto => Ok(0),
    }
}

fn run_once_sync(
    cozip: &CoZip,
    mode: InputMode,
    input: &Path,
    output: &Path,
) -> Result<CoZipStats, CoZipError> {
    match mode {
        InputMode::File => cozip.compress_file_from_name(input, output),
        InputMode::Directory => cozip.compress_directory(input, output),
        InputMode::Auto => unreachable!(),
    }
}

async fn run_once_async(
    cozip: &CoZip,
    mode: InputMode,
    input: &Path,
    output: &Path,
) -> Result<CoZipStats, CoZipError> {
    match mode {
        InputMode::File => cozip.compress_file_from_name_async(input, output).await,
        InputMode::Directory => cozip.compress_directory_async(input, output).await,
        InputMode::Auto => unreachable!(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|msg| {
        eprintln!("error: {msg}");
        print_usage();
        std::io::Error::new(std::io::ErrorKind::InvalidInput, msg)
    })?;

    let mode = determine_mode(&args.input, args.mode).map_err(|msg| {
        eprintln!("error: {msg}");
        print_usage();
        std::io::Error::new(std::io::ErrorKind::InvalidInput, msg)
    })?;

    let output = match args.output {
        Some(path) => path,
        None => default_output_path(&args.input, mode).map_err(|msg| {
            eprintln!("error: {msg}");
            std::io::Error::new(std::io::ErrorKind::InvalidInput, msg)
        })?,
    };

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    if mode == InputMode::Directory && output.starts_with(&args.input) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "output zip must be outside the input directory",
        )
        .into());
    }

    let input_bytes = input_size_bytes(&args.input, mode)?;

    // NOTE: init is intentionally outside the measured section.
    let cozip = CoZip::init(CoZipOptions::Zip {
        options: ZipOptions {
            compression_level: args.level,
            deflate_mode: args.engine,
        },
    })?;

    println!(
        "cozip zip bench\ninput={} mode={:?} engine={:?} output={} level={} warmups={} runs={} async={}",
        args.input.display(),
        mode,
        args.engine,
        output.display(),
        args.level,
        args.warmups,
        args.runs,
        args.async_mode,
    );

    if args.async_mode {
        let rt = tokio::runtime::Builder::new_current_thread().build()?;

        for _ in 0..args.warmups {
            let _ = rt.block_on(run_once_async(&cozip, mode, &args.input, &output))?;
        }

        let mut run_ms = Vec::with_capacity(args.runs);
        let mut final_stats = CoZipStats::default();
        for i in 0..args.runs {
            let start = Instant::now();
            final_stats = rt.block_on(run_once_async(&cozip, mode, &args.input, &output))?;
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            run_ms.push(ms);
            println!("run {}/{}: {:.3} ms", i + 1, args.runs, ms);
        }

        report(&run_ms, input_bytes, &output, final_stats)?;
    } else {
        for _ in 0..args.warmups {
            let _ = run_once_sync(&cozip, mode, &args.input, &output)?;
        }

        let mut run_ms = Vec::with_capacity(args.runs);
        let mut final_stats = CoZipStats::default();
        for i in 0..args.runs {
            let start = Instant::now();
            final_stats = run_once_sync(&cozip, mode, &args.input, &output)?;
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            run_ms.push(ms);
            println!("run {}/{}: {:.3} ms", i + 1, args.runs, ms);
        }

        report(&run_ms, input_bytes, &output, final_stats)?;
    }

    Ok(())
}

fn report(
    run_ms: &[f64],
    input_bytes: u64,
    output_path: &Path,
    stats: CoZipStats,
) -> Result<(), CoZipError> {
    let mut sorted = run_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let sum_ms: f64 = run_ms.iter().sum();
    let mean_ms = if run_ms.is_empty() {
        0.0
    } else {
        sum_ms / (run_ms.len() as f64)
    };
    let median_ms = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };

    let min_ms = sorted.first().copied().unwrap_or(0.0);
    let max_ms = sorted.last().copied().unwrap_or(0.0);

    let output_bytes = fs::metadata(output_path)?.len();
    let ratio = if input_bytes == 0 {
        0.0
    } else {
        (output_bytes as f64) / (input_bytes as f64)
    };
    let throughput_mib_s = if mean_ms <= 0.0 {
        0.0
    } else {
        (input_bytes as f64 / (1024.0 * 1024.0)) / (mean_ms / 1000.0)
    };

    println!(
        "summary: mean_ms={:.3} median_ms={:.3} min_ms={:.3} max_ms={:.3}",
        mean_ms, median_ms, min_ms, max_ms
    );
    println!(
        "size: input_bytes={} output_bytes={} ratio={:.4} throughput_mib_s={:.2}",
        input_bytes, output_bytes, ratio, throughput_mib_s
    );
    println!(
        "zip_stats: entries={} input_bytes={} output_bytes={}",
        stats.entries, stats.input_bytes, stats.output_bytes
    );

    Ok(())
}
