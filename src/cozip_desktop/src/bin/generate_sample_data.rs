use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const DEFAULT_SIZE: &str = "1GiB";
const DEFAULT_NAME: &str = "cozip_desktop_sample.bin";
const CHUNK_SIZE: usize = 8 * 1024 * 1024;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse(env::args().skip(1))?;
    let output_dir = workspace_target_dir()?;
    fs::create_dir_all(&output_dir)
        .map_err(|error| format!("failed to create {}: {error}", output_dir.display()))?;

    let output_path = output_dir.join(&args.name);
    if output_path.exists() && !args.force {
        return Err(format!(
            "output already exists: {} (use --force to overwrite)",
            output_path.display()
        ));
    }

    let file = File::create(&output_path)
        .map_err(|error| format!("failed to create {}: {error}", output_path.display()))?;
    let mut writer = BufWriter::with_capacity(CHUNK_SIZE, file);

    let mut remaining = args.size_bytes;
    let mut chunk_index = 0_u64;
    while remaining > 0 {
        let chunk_len = remaining.min(CHUNK_SIZE as u64) as usize;
        let chunk = build_chunk(chunk_index, chunk_len);
        writer
            .write_all(&chunk)
            .map_err(|error| format!("failed to write {}: {error}", output_path.display()))?;
        remaining -= chunk_len as u64;
        chunk_index = chunk_index.saturating_add(1);
    }

    writer
        .flush()
        .map_err(|error| format!("failed to flush {}: {error}", output_path.display()))?;

    println!(
        "created {} ({})",
        output_path.display(),
        format_bytes(args.size_bytes)
    );
    Ok(())
}

fn workspace_target_dir() -> Result<PathBuf, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "failed to resolve workspace root".to_string())?;
    Ok(root.join("target"))
}

#[derive(Debug)]
struct Args {
    size_bytes: u64,
    name: String,
    force: bool,
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut size = DEFAULT_SIZE.to_string();
        let mut name = DEFAULT_NAME.to_string();
        let mut force = false;

        let mut iter = args.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--size" => {
                    size = iter
                        .next()
                        .ok_or_else(|| "--size requires a value".to_string())?;
                }
                "--name" => {
                    name = iter
                        .next()
                        .ok_or_else(|| "--name requires a value".to_string())?;
                }
                "--force" => force = true,
                "--help" | "-h" => return Err(usage()),
                other => return Err(format!("unsupported argument: {other}\n\n{}", usage())),
            }
        }

        Ok(Self {
            size_bytes: parse_size(&size)?,
            name,
            force,
        })
    }
}

fn usage() -> String {
    "usage: cargo run -p cozip_desktop --bin generate_sample_data -- --size 10GiB --name sample.bin [--force]"
        .to_string()
}

fn parse_size(value: &str) -> Result<u64, String> {
    let normalized = value.trim().replace(' ', "").to_ascii_uppercase();
    if normalized.is_empty() {
        return Err("size is empty".to_string());
    }

    let split_index = normalized
        .find(|ch: char| !ch.is_ascii_digit() && ch != '.')
        .unwrap_or(normalized.len());
    let (number_text, unit_text) = normalized.split_at(split_index);
    let number: f64 = number_text
        .parse()
        .map_err(|_| format!("invalid size value: {value}"))?;
    let multiplier = match unit_text {
        "" | "B" => 1_u64,
        "KB" => 1_000_u64,
        "MB" => 1_000_000_u64,
        "GB" => 1_000_000_000_u64,
        "TB" => 1_000_000_000_000_u64,
        "KIB" => 1_u64 << 10,
        "MIB" => 1_u64 << 20,
        "GIB" => 1_u64 << 30,
        "TIB" => 1_u64 << 40,
        _ => return Err(format!("unsupported size unit: {unit_text}")),
    };

    let size = (number * multiplier as f64).floor();
    if !size.is_finite() || size <= 0.0 {
        return Err(format!("size must be positive: {value}"));
    }

    Ok(size as u64)
}

fn build_chunk(chunk_index: u64, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut rng = Lcg::new(0xC0FFEE1234_u64 ^ chunk_index.wrapping_mul(0x9E37_79B9_7F4A_7C15));

    while out.len() < len {
        match (chunk_index as usize + out.len() / 4096) % 5 {
            0 => append_log_block(&mut out, &mut rng, chunk_index),
            1 => append_csv_block(&mut out, &mut rng, chunk_index),
            2 => append_jsonl_block(&mut out, &mut rng, chunk_index),
            3 => append_doc_block(&mut out, &mut rng, chunk_index),
            _ => append_binaryish_block(&mut out, &mut rng, chunk_index),
        }
    }

    out.truncate(len);
    out
}

fn append_log_block(out: &mut Vec<u8>, rng: &mut Lcg, seq: u64) {
    const LEVELS: &[&str] = &["INFO", "WARN", "ERROR", "DEBUG"];
    const SERVICES: &[&str] = &[
        "shell.indexer",
        "archive.scheduler",
        "desktop.preview",
        "gpu.bridge",
        "worker.fs",
    ];
    const ACTIONS: &[&str] = &["scan", "hydrate", "compress", "extract", "prefetch", "flush"];
    const TARGETS: &[&str] = &[
        r"C:\media\capture",
        r"D:\project\assets",
        r"E:\backup\day-03",
        r"C:\Users\demo\Downloads",
        r"D:\logs\runtime",
    ];

    for line_index in 0..48_u64 {
        let timestamp = format!(
            "2026-03-{day:02}T{hour:02}:{minute:02}:{second:02}Z",
            day = ((seq + line_index) % 28) + 1,
            hour = (line_index * 3) % 24,
            minute = (line_index * 7) % 60,
            second = (line_index * 11) % 60
        );
        let line = format!(
            "{timestamp} [{level}] {service} request_id={request:08x} user={user:05} op={action} target=\"{target}\" latency_ms={latency} bytes={bytes}\n",
            timestamp = timestamp,
            level = pick(LEVELS, rng),
            service = pick(SERVICES, rng),
            request = rng.next_u32(),
            user = rng.range_usize(10, 99_999),
            action = pick(ACTIONS, rng),
            target = pick(TARGETS, rng),
            latency = rng.range_usize(2, 4_800),
            bytes = rng.range_usize(4_096, 1_048_576),
        );
        out.extend_from_slice(line.as_bytes());
    }
}

fn append_csv_block(out: &mut Vec<u8>, rng: &mut Lcg, seq: u64) {
    const REGIONS: &[&str] = &["tokyo", "osaka", "nagoya", "fukuoka", "sapporo"];
    const PRODUCTS: &[&str] = &[
        "storage-box",
        "scan-proxy",
        "mobile-sync",
        "gpu-kit",
        "desktop-plus",
    ];
    const CHANNELS: &[&str] = &["retail", "partner", "online", "renewal"];

    out.extend_from_slice(b"date,region,product,channel,units,revenue,discount,returns\n");
    for row in 0..64_u64 {
        let line = format!(
            "2026-{month:02}-{day:02},{region},{product},{channel},{units},{revenue},{discount},{returns}\n",
            month = ((seq + row) % 12) + 1,
            day = (row % 28) + 1,
            region = pick(REGIONS, rng),
            product = pick(PRODUCTS, rng),
            channel = pick(CHANNELS, rng),
            units = rng.range_usize(1, 2_500),
            revenue = rng.range_usize(10_000, 950_000),
            discount = rng.range_usize(0, 35),
            returns = rng.range_usize(0, 25),
        );
        out.extend_from_slice(line.as_bytes());
    }
}

fn append_jsonl_block(out: &mut Vec<u8>, rng: &mut Lcg, seq: u64) {
    const PROFILES: &[&str] = &["fast", "balanced", "ratio"];
    const DEVICES: &[&str] = &[
        "desktop-a",
        "desktop-b",
        "notebook-12",
        "workstation-02",
        "render-node-7",
    ];
    const PATHS: &[&str] = &[
        "photos/2025",
        "docs/contracts",
        "assets/build",
        "capture/day02",
        "vm/export",
    ];

    for row in 0..48_u64 {
        let ratio = 0.35 + (rng.next_u32() as f64 / u32::MAX as f64) * 0.45;
        let line = format!(
            "{{\"ts\":\"2026-03-{day:02}T{hour:02}:{minute:02}:{second:02}Z\",\"device\":\"{device}\",\"profile\":\"{profile}\",\"source\":\"{source}\",\"chunks\":{chunks},\"gpu\":{gpu},\"ratio\":{ratio:.3},\"warnings\":[\"queue_depth\",\"retryable_io\"]}}\n",
            day = ((seq + row) % 28) + 1,
            hour = row % 24,
            minute = (row * 3) % 60,
            second = (row * 7) % 60,
            device = pick(DEVICES, rng),
            profile = pick(PROFILES, rng),
            source = pick(PATHS, rng),
            chunks = rng.range_usize(4, 512),
            gpu = if rng.bool(72, 100) { "true" } else { "false" },
            ratio = ratio,
        );
        out.extend_from_slice(line.as_bytes());
    }
}

fn append_doc_block(out: &mut Vec<u8>, rng: &mut Lcg, seq: u64) {
    const MODULES: &[&str] = &[
        "scheduler",
        "context_menu",
        "desktop_ui",
        "archive_reader",
        "gpu_runtime",
    ];
    const STATES: &[&str] = &["draft", "review", "approved"];
    const OWNERS: &[&str] = &["platform", "desktop", "runtime", "storage"];

    let module = pick(MODULES, rng);
    let state = pick(STATES, rng);
    let owner = pick(OWNERS, rng);
    let doc = format!(
        "# design-{module}-{seq}\nstate = \"{state}\"\nowner = \"{owner}\"\npriority = {priority}\n\n## notes\n- keep command line entrypoints aligned with explorer context menu\n- prefer deterministic output naming when multiple items are selected\n- surface current file, throughput, and result path in the desktop UI\n\n```rust\nfn run_{module}(job_id: u64, prefer_gpu: bool) -> Result<(), String> {{\n    if prefer_gpu {{\n        return Ok(());\n    }}\n    Err(format!(\"fallback path triggered for {{job_id}}\"))\n}}\n```\n\n",
        module = module,
        seq = seq,
        state = state,
        owner = owner,
        priority = rng.range_usize(1, 5),
    );
    out.extend_from_slice(doc.as_bytes());
}

fn append_binaryish_block(out: &mut Vec<u8>, rng: &mut Lcg, seq: u64) {
    const PALETTE: &[u8] = &[0x00, 0x00, 0x20, 0x20, 0x2F, 0x30, 0x41, 0x45, 0x61, 0x65, 0x7A, 0x7F, 0xFF];

    let header = format!(
        "COZIP-SAMPLE-BLOB\nsegment={seq}\ncodec=hybrid\nlayout=chunked\n\0",
        seq = seq
    );
    out.extend_from_slice(header.as_bytes());

    for _ in 0..64 {
        let run_len = rng.range_usize(64, 8_192);
        let value = PALETTE[rng.range_usize(0, PALETTE.len())];
        let start = out.len();
        out.resize(start + run_len, value);

        if run_len >= 128 {
            let marker = format!(
                "PAGE:{page:08}:asset_{asset:05}.bin;",
                page = seq.saturating_mul(97),
                asset = rng.range_usize(0, 50_000)
            );
            let bytes = marker.as_bytes();
            let copy_len = bytes.len().min(run_len);
            out[start..start + copy_len].copy_from_slice(&bytes[..copy_len]);
        }
    }
}

fn pick<'a>(items: &'a [&str], rng: &mut Lcg) -> &'a str {
    items[rng.range_usize(0, items.len())]
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }

    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

#[derive(Clone, Debug)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn range_usize(&mut self, start: usize, end: usize) -> usize {
        debug_assert!(start < end);
        start + (self.next_u32() as usize % (end - start))
    }

    fn bool(&mut self, numer: u32, denom: u32) -> bool {
        self.next_u32() % denom < numer
    }
}

#[allow(dead_code)]
fn _assert_send_sync() {
    fn assert_impl<T: Send + Sync>() {}
    assert_impl::<BufWriter<File>>();
}
