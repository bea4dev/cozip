use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use flate2::Compression;
use thiserror::Error;

const FRAME_MAGIC: [u8; 4] = *b"CZDF";
const FRAME_VERSION: u8 = 3;
const HEADER_LEN: usize = 22;
const CHUNK_META_LEN_V3: usize = 11;
const CHUNK_META_LEN_V2: usize = 10;
const CHUNK_META_LEN_V1: usize = 9;
const TRANSFORM_LANES: usize = 2;

const LITLEN_SYMBOL_COUNT: usize = 286;
const DIST_SYMBOL_COUNT: usize = 30;
const DYN_TABLE_U32_COUNT: usize = (LITLEN_SYMBOL_COUNT * 2) + (DIST_SYMBOL_COUNT * 2);
const GPU_BATCH_CHUNKS: usize = 16;
const GPU_PIPELINED_SUBMIT_CHUNKS: usize = 4;
const GPU_FREQ_MAX_WORKGROUPS: u32 = 4096;
const PREFIX_SCAN_BLOCK_SIZE: usize = 256;
const TOKEN_FINALIZE_SEGMENT_SIZE: usize = 4096;
const GPU_DEFLATE_MAX_BITS_PER_BYTE: usize = 12;
const MAX_DISPATCH_WORKGROUPS_PER_DIM: u32 = 65_535;

mod frame;
mod gpu;

use frame::{encode_frame, parse_frame};
use gpu::GpuAssist;

#[derive(Debug, Clone)]
pub struct HybridOptions {
    pub chunk_size: usize,
    pub gpu_subchunk_size: usize,
    pub compression_level: u32,
    pub compression_mode: CompressionMode,
    pub prefer_gpu: bool,
    pub gpu_fraction: f32,
    pub gpu_min_chunk_size: usize,
    pub gpu_validation_mode: GpuValidationMode,
    pub gpu_validation_sample_every: usize,
    pub gpu_dynamic_self_check: bool,
    pub gpu_dump_bad_chunk: bool,
    pub gpu_dump_bad_chunk_limit: usize,
    pub gpu_dump_bad_chunk_dir: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMode {
    Speed,
    Balanced,
    Ratio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuValidationMode {
    Always,
    Sample,
    Off,
}

impl Default for HybridOptions {
    fn default() -> Self {
        Self {
            chunk_size: 4 * 1024 * 1024,
            gpu_subchunk_size: 256 * 1024,
            compression_level: 6,
            compression_mode: CompressionMode::Speed,
            prefer_gpu: true,
            gpu_fraction: 1.0,
            gpu_min_chunk_size: 64 * 1024,
            gpu_validation_mode: GpuValidationMode::Off,
            gpu_validation_sample_every: 8,
            gpu_dynamic_self_check: false,
            gpu_dump_bad_chunk: false,
            gpu_dump_bad_chunk_limit: 8,
            gpu_dump_bad_chunk_dir: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HybridStats {
    pub chunk_count: usize,
    pub cpu_chunks: usize,
    pub gpu_chunks: usize,
    pub gpu_available: bool,
    pub cpu_bytes: usize,
    pub gpu_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct CompressedFrame {
    pub bytes: Vec<u8>,
    pub stats: HybridStats,
}

#[derive(Debug, Clone)]
pub struct DecompressedFrame {
    pub bytes: Vec<u8>,
    pub stats: HybridStats,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CoZipDeflateInitStats {
    pub gpu_context_init_ms: f64,
    pub gpu_available: bool,
}

#[derive(Debug, Clone)]
pub struct CoZipDeflate {
    options: HybridOptions,
    gpu_context: Option<Arc<GpuAssist>>,
    init_stats: CoZipDeflateInitStats,
}

impl CoZipDeflate {
    pub fn init(options: HybridOptions) -> Result<Self, CozipDeflateError> {
        validate_options(&options)?;
        let mut init_stats = CoZipDeflateInitStats::default();
        let gpu_context = if options.prefer_gpu && options.gpu_fraction > 0.0 {
            let start = Instant::now();
            let ctx = GpuAssist::new().ok().map(Arc::new);
            init_stats.gpu_context_init_ms = elapsed_ms(start);
            init_stats.gpu_available = ctx.is_some();
            ctx
        } else {
            None
        };
        Ok(Self {
            options,
            gpu_context,
            init_stats,
        })
    }

    pub fn init_stats(&self) -> CoZipDeflateInitStats {
        self.init_stats
    }

    pub fn gpu_context_init_ms(&self) -> f64 {
        self.init_stats.gpu_context_init_ms
    }

    pub fn compress(&self, input: &[u8]) -> Result<CompressedFrame, CozipDeflateError> {
        compress_hybrid_with_context(input, &self.options, self.gpu_context.clone())
    }

    pub fn decompress_on_cpu(&self, frame: &[u8]) -> Result<DecompressedFrame, CozipDeflateError> {
        decompress_on_cpu_with_context(frame, &self.options)
    }

    pub fn decompress(&self, frame: &[u8]) -> Result<DecompressedFrame, CozipDeflateError> {
        self.decompress_on_cpu(frame)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkBackend {
    Cpu,
    GpuAssisted,
}

impl ChunkBackend {
    fn to_u8(self) -> u8 {
        match self {
            Self::Cpu => 0,
            Self::GpuAssisted => 1,
        }
    }

    fn from_u8(value: u8) -> Result<Self, CozipDeflateError> {
        match value {
            0 => Ok(Self::Cpu),
            1 => Ok(Self::GpuAssisted),
            _ => Err(CozipDeflateError::InvalidFrame("unknown backend id")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkTransform {
    None,
    EvenOdd,
}

impl ChunkTransform {
    fn to_u8(self) -> u8 {
        match self {
            Self::None => 0,
            Self::EvenOdd => 1,
        }
    }

    fn from_u8(value: u8) -> Result<Self, CozipDeflateError> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::EvenOdd),
            _ => Err(CozipDeflateError::InvalidFrame("unknown transform id")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkCodec {
    DeflateCpu,
    DeflateGpuFast,
}

impl ChunkCodec {
    fn to_u8(self) -> u8 {
        match self {
            Self::DeflateCpu => 0,
            Self::DeflateGpuFast => 1,
        }
    }

    fn from_u8(value: u8) -> Result<Self, CozipDeflateError> {
        match value {
            0 => Ok(Self::DeflateCpu),
            1 => Ok(Self::DeflateGpuFast),
            _ => Err(CozipDeflateError::InvalidFrame("unknown codec id")),
        }
    }
}

#[derive(Debug, Error)]
pub enum CozipDeflateError {
    #[error("invalid options: {0}")]
    InvalidOptions(&'static str),
    #[error("invalid frame: {0}")]
    InvalidFrame(&'static str),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("data too large")]
    DataTooLarge,
    #[error("gpu unavailable: {0}")]
    GpuUnavailable(String),
    #[error("gpu execution failed: {0}")]
    GpuExecution(String),
    #[error("internal error: {0}")]
    Internal(&'static str),
}

fn timing_profile_enabled() -> bool {
    false
}

fn timing_profile_detail_enabled() -> bool {
    false
}

fn deep_timing_profile_enabled() -> bool {
    false
}

fn maybe_warn_deep_profile_enabled() {}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Debug, Clone)]
struct ChunkTask {
    index: usize,
    preferred_gpu: bool,
    raw: Vec<u8>,
}

#[derive(Debug, Clone)]
struct ChunkMember {
    index: usize,
    backend: ChunkBackend,
    transform: ChunkTransform,
    codec: ChunkCodec,
    raw_len: u32,
    compressed: Vec<u8>,
}

#[derive(Debug, Clone)]
struct DecodedChunk {
    index: usize,
    raw: Vec<u8>,
}

#[derive(Debug, Clone)]
struct ChunkDescriptor {
    index: usize,
    backend: ChunkBackend,
    transform: ChunkTransform,
    codec: ChunkCodec,
    raw_len: u32,
    compressed: Vec<u8>,
}

fn workgroup_count(items: usize, group_size: usize) -> Result<u32, CozipDeflateError> {
    if group_size == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "group_size must be greater than 0",
        ));
    }
    let count = items.div_ceil(group_size);
    u32::try_from(count).map_err(|_| CozipDeflateError::DataTooLarge)
}

fn dispatch_grid_for_groups(total_groups: u32) -> (u32, u32) {
    if total_groups <= MAX_DISPATCH_WORKGROUPS_PER_DIM {
        (total_groups, 1)
    } else {
        (
            MAX_DISPATCH_WORKGROUPS_PER_DIM,
            total_groups.div_ceil(MAX_DISPATCH_WORKGROUPS_PER_DIM),
        )
    }
}

fn dispatch_grid_for_items(
    items: usize,
    group_size: usize,
) -> Result<(u32, u32), CozipDeflateError> {
    let groups = workgroup_count(items, group_size)?;
    Ok(dispatch_grid_for_groups(groups))
}

fn dispatch_grid_for_items_capped(
    items: usize,
    group_size: usize,
    max_groups: u32,
) -> Result<(u32, u32), CozipDeflateError> {
    let groups = workgroup_count(items, group_size)?;
    let capped = groups.min(max_groups.max(1));
    Ok(dispatch_grid_for_groups(capped))
}

fn bytes_len<T>(items: usize) -> Result<u64, CozipDeflateError> {
    let bytes = items
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CozipDeflateError::DataTooLarge)?;
    u64::try_from(bytes).map_err(|_| CozipDeflateError::DataTooLarge)
}

fn lock<'a, T>(mutex: &'a Mutex<T>) -> Result<std::sync::MutexGuard<'a, T>, CozipDeflateError> {
    mutex
        .lock()
        .map_err(|_| CozipDeflateError::Internal("mutex poisoned"))
}

pub fn deflate_compress_cpu(input: &[u8], level: u32) -> Result<Vec<u8>, CozipDeflateError> {
    let mut encoder =
        flate2::write::DeflateEncoder::new(Vec::new(), Compression::new(level.clamp(0, 9)));
    encoder.write_all(input)?;
    Ok(encoder.finish()?)
}

pub fn deflate_decompress_on_cpu(input: &[u8]) -> Result<Vec<u8>, CozipDeflateError> {
    let mut decoder = flate2::write::DeflateDecoder::new(Vec::new());
    decoder.write_all(input)?;
    Ok(decoder.finish()?)
}

pub fn compress_hybrid(
    input: &[u8],
    options: &HybridOptions,
) -> Result<CompressedFrame, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip.compress(input)
}

fn compress_hybrid_with_context(
    input: &[u8],
    options: &HybridOptions,
    gpu_context: Option<Arc<GpuAssist>>,
) -> Result<CompressedFrame, CozipDeflateError> {
    maybe_warn_deep_profile_enabled();
    validate_options(options)?;

    if input.is_empty() {
        return Ok(CompressedFrame {
            bytes: encode_frame(0, &[])?,
            stats: HybridStats {
                chunk_count: 0,
                cpu_chunks: 0,
                gpu_chunks: 0,
                gpu_available: false,
                cpu_bytes: 0,
                gpu_bytes: 0,
            },
        });
    }

    let gpu_requested = options.prefer_gpu && options.gpu_fraction > 0.0;
    let gpu_available = gpu_context.is_some();
    let tasks = build_chunk_tasks(input, options, gpu_requested && gpu_available)?;
    let has_gpu_tasks =
        gpu_available && gpu_requested && tasks.iter().any(|task| task.preferred_gpu);
    if has_gpu_tasks {
        return compress_hybrid_work_stealing_scheduler(input.len(), tasks, options, gpu_context);
    }

    let chunk_count = tasks.len();
    let queue = Arc::new(Mutex::new(VecDeque::from(tasks)));
    let results = Arc::new(Mutex::new(vec![None; chunk_count]));
    let error = Arc::new(Mutex::new(None::<CozipDeflateError>));

    let cpu_workers = cpu_worker_count(false);
    let mut handles = Vec::new();

    for _ in 0..cpu_workers {
        let queue_ref = Arc::clone(&queue);
        let result_ref = Arc::clone(&results);
        let err_ref = Arc::clone(&error);
        let opts = options.clone();
        let gpu_enabled = false;

        handles.push(std::thread::spawn(move || {
            compress_cpu_worker(queue_ref, result_ref, err_ref, &opts, gpu_enabled)
        }));
    }

    for handle in handles {
        let _ = handle.join();
    }

    if let Some(err) = lock(&error)?.take() {
        return Err(err);
    }

    let mut chunks = Vec::new();
    for item in lock(&results)?.drain(..) {
        let member = item.ok_or(CozipDeflateError::Internal("missing compressed chunk"))?;
        chunks.push(member);
    }
    chunks.sort_by_key(|chunk| chunk.index);

    let stats = summarize_encoded_chunks(&chunks, false);
    let frame = encode_frame(input.len(), &chunks)?;

    Ok(CompressedFrame {
        bytes: frame,
        stats,
    })
}

fn compress_hybrid_work_stealing_scheduler(
    original_len: usize,
    tasks: Vec<ChunkTask>,
    options: &HybridOptions,
    gpu_context: Option<Arc<GpuAssist>>,
) -> Result<CompressedFrame, CozipDeflateError> {
    let chunk_count = tasks.len();
    let gpu_available = gpu_context.is_some();
    let queue = Arc::new(Mutex::new(VecDeque::from(tasks)));
    let results = Arc::new(Mutex::new(vec![None; chunk_count]));
    let error = Arc::new(Mutex::new(None::<CozipDeflateError>));

    let cpu_workers = cpu_worker_count(gpu_available);
    let mut handles = Vec::new();

    for _ in 0..cpu_workers {
        let queue_ref = Arc::clone(&queue);
        let result_ref = Arc::clone(&results);
        let err_ref = Arc::clone(&error);
        let opts = options.clone();
        handles.push(std::thread::spawn(move || {
            compress_cpu_worker(queue_ref, result_ref, err_ref, &opts, gpu_available)
        }));
    }

    if let Some(gpu) = gpu_context {
        let queue_ref = Arc::clone(&queue);
        let result_ref = Arc::clone(&results);
        let err_ref = Arc::clone(&error);
        let opts = options.clone();
        handles.push(std::thread::spawn(move || {
            compress_gpu_worker(queue_ref, result_ref, err_ref, &opts, gpu)
        }));
    }

    for handle in handles {
        let _ = handle.join();
    }

    if let Some(err) = lock(&error)?.take() {
        return Err(err);
    }

    let mut chunks = Vec::new();
    for item in lock(&results)?.drain(..) {
        let member = item.ok_or(CozipDeflateError::Internal("missing compressed chunk"))?;
        chunks.push(member);
    }
    chunks.sort_by_key(|chunk| chunk.index);

    let stats = summarize_encoded_chunks(&chunks, gpu_available);
    let frame = encode_frame(original_len, &chunks)?;

    Ok(CompressedFrame {
        bytes: frame,
        stats,
    })
}

pub fn decompress_on_cpu(
    frame: &[u8],
    options: &HybridOptions,
) -> Result<DecompressedFrame, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip.decompress_on_cpu(frame)
}

pub fn decompress_hybrid(
    frame: &[u8],
    options: &HybridOptions,
) -> Result<DecompressedFrame, CozipDeflateError> {
    decompress_on_cpu(frame, options)
}

fn decompress_on_cpu_with_context(
    frame: &[u8],
    options: &HybridOptions,
) -> Result<DecompressedFrame, CozipDeflateError> {
    validate_options(options)?;

    let (original_len, descriptors) = parse_frame(frame)?;
    if descriptors.is_empty() {
        return Ok(DecompressedFrame {
            bytes: Vec::new(),
            stats: HybridStats {
                chunk_count: 0,
                cpu_chunks: 0,
                gpu_chunks: 0,
                gpu_available: false,
                cpu_bytes: 0,
                gpu_bytes: 0,
            },
        });
    }

    let queue = Arc::new(Mutex::new(VecDeque::from(descriptors)));
    let chunk_count = lock(&queue)?.len();
    let results = Arc::new(Mutex::new(vec![None; chunk_count]));
    let error = Arc::new(Mutex::new(None::<CozipDeflateError>));

    let cpu_workers = cpu_worker_count(false);
    let mut handles = Vec::new();

    for _ in 0..cpu_workers {
        let queue_ref = Arc::clone(&queue);
        let result_ref = Arc::clone(&results);
        let err_ref = Arc::clone(&error);
        let opts = options.clone();

        handles.push(std::thread::spawn(move || {
            decompress_worker_on_cpu(queue_ref, result_ref, err_ref, &opts)
        }));
    }

    for handle in handles {
        let _ = handle.join();
    }

    if let Some(err) = lock(&error)?.take() {
        return Err(err);
    }

    let mut decoded = Vec::with_capacity(original_len);
    let mut cpu_bytes = 0usize;
    for item in lock(&results)?.drain(..) {
        let chunk = item.ok_or(CozipDeflateError::Internal("missing decompressed chunk"))?;
        cpu_bytes = cpu_bytes.saturating_add(chunk.raw.len());
        decoded.extend_from_slice(&chunk.raw);
    }

    if decoded.len() != original_len {
        return Err(CozipDeflateError::InvalidFrame(
            "decoded size does not match frame header",
        ));
    }

    Ok(DecompressedFrame {
        bytes: decoded,
        stats: HybridStats {
            chunk_count,
            cpu_chunks: chunk_count,
            gpu_chunks: 0,
            gpu_available: false,
            cpu_bytes,
            gpu_bytes: 0,
        },
    })
}

fn cpu_worker_count(has_gpu: bool) -> usize {
    let available = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);

    if has_gpu {
        available.saturating_sub(1).max(1)
    } else {
        available.max(1)
    }
}

fn validate_options(options: &HybridOptions) -> Result<(), CozipDeflateError> {
    if options.chunk_size == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "chunk_size must be greater than 0",
        ));
    }

    if options.gpu_subchunk_size == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_subchunk_size must be greater than 0",
        ));
    }

    if !(0.0..=1.0).contains(&options.gpu_fraction) {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_fraction must be in range 0.0..=1.0",
        ));
    }

    if options.gpu_validation_sample_every == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_validation_sample_every must be greater than 0",
        ));
    }

    if options.gpu_dump_bad_chunk_limit == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_dump_bad_chunk_limit must be greater than 0",
        ));
    }

    Ok(())
}

fn compression_mode_id(mode: CompressionMode) -> u32 {
    match mode {
        CompressionMode::Speed => 0,
        CompressionMode::Balanced => 1,
        CompressionMode::Ratio => 2,
    }
}

fn should_validate_gpu_chunk(options: &HybridOptions, chunk_index: usize) -> bool {
    if options.compression_mode == CompressionMode::Speed {
        return false;
    }
    if options.compression_mode == CompressionMode::Ratio && options.gpu_dynamic_self_check {
        // Ratio mode uses dynamic GPU path; that path performs its own per-chunk
        // roundtrip guard and CPU fallback before returning compressed bytes.
        return false;
    }

    match options.gpu_validation_mode {
        GpuValidationMode::Always => true,
        GpuValidationMode::Off => false,
        GpuValidationMode::Sample => chunk_index % options.gpu_validation_sample_every == 0,
    }
}

#[derive(Debug)]
enum GpuRoundtripIssue {
    DecodeFailed(String),
    LengthMismatch {
        decoded_len: usize,
        prefix_match_len: usize,
        expected_next: Option<u8>,
        actual_next: Option<u8>,
    },
    ContentMismatch {
        first_diff: usize,
        expected: u8,
        actual: u8,
    },
}

fn gpu_chunk_roundtrip_diagnose(raw: &[u8], compressed: &[u8]) -> Result<(), GpuRoundtripIssue> {
    let decoded = deflate_decompress_on_cpu(compressed)
        .map_err(|err| GpuRoundtripIssue::DecodeFailed(err.to_string()))?;
    if decoded.len() != raw.len() {
        let prefix_match_len = decoded
            .iter()
            .zip(raw.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(decoded.len().min(raw.len()));
        return Err(GpuRoundtripIssue::LengthMismatch {
            decoded_len: decoded.len(),
            prefix_match_len,
            expected_next: raw.get(prefix_match_len).copied(),
            actual_next: decoded.get(prefix_match_len).copied(),
        });
    }
    if decoded != raw {
        let first_diff = decoded
            .iter()
            .zip(raw.iter())
            .position(|(lhs, rhs)| lhs != rhs)
            .unwrap_or(0);
        return Err(GpuRoundtripIssue::ContentMismatch {
            first_diff,
            expected: raw[first_diff],
            actual: decoded[first_diff],
        });
    }
    Ok(())
}

fn gpu_chunk_roundtrip_matches(raw: &[u8], compressed: &[u8]) -> bool {
    gpu_chunk_roundtrip_diagnose(raw, compressed).is_ok()
}

fn gpu_dynamic_dump_bad_chunk_dir(options: &HybridOptions) -> String {
    options
        .gpu_dump_bad_chunk_dir
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("/tmp/cozip_gpu_bad_chunks")
        .to_string()
}

fn fnv1a64(data: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn dump_gpu_dynamic_bad_chunk(
    options: &HybridOptions,
    dump_seq: &AtomicUsize,
    call_id: u64,
    chunk_index: usize,
    raw: &[u8],
    gpu_compressed: &[u8],
    cpu_fallback: &[u8],
    issue: &GpuRoundtripIssue,
) {
    if !options.gpu_dump_bad_chunk {
        return;
    }
    let seq = dump_seq.fetch_add(1, Ordering::Relaxed);
    if seq >= options.gpu_dump_bad_chunk_limit.max(1) {
        return;
    }
    let dir = gpu_dynamic_dump_bad_chunk_dir(options);
    let dir_path = std::path::Path::new(&dir);
    if std::fs::create_dir_all(dir_path).is_err() {
        return;
    }
    let base = format!("call{:04}_chunk{:04}_seq{:03}", call_id, chunk_index, seq);
    let raw_path = dir_path.join(format!("{base}.raw.bin"));
    let gpu_path = dir_path.join(format!("{base}.gpu.bin"));
    let cpu_path = dir_path.join(format!("{base}.cpu_fallback.bin"));
    let meta_path = dir_path.join(format!("{base}.meta.txt"));
    if std::fs::write(&raw_path, raw).is_err()
        || std::fs::write(&gpu_path, gpu_compressed).is_err()
        || std::fs::write(&cpu_path, cpu_fallback).is_err()
    {
        return;
    }
    let issue_text = match issue {
        GpuRoundtripIssue::DecodeFailed(message) => format!("decode_failed: {message}"),
        GpuRoundtripIssue::LengthMismatch {
            decoded_len,
            prefix_match_len,
            expected_next,
            actual_next,
        } => format!(
            "length_mismatch decoded_len={decoded_len} prefix_match_len={prefix_match_len} expected_next={expected_next:?} actual_next={actual_next:?}"
        ),
        GpuRoundtripIssue::ContentMismatch {
            first_diff,
            expected,
            actual,
        } => {
            format!("content_mismatch first_diff={first_diff} expected={expected} actual={actual}")
        }
    };
    let mut meta = String::new();
    meta.push_str(&format!("call_id={call_id}\n"));
    meta.push_str(&format!("chunk_index={chunk_index}\n"));
    meta.push_str(&format!("issue={issue_text}\n"));
    meta.push_str(&format!("raw_len={}\n", raw.len()));
    meta.push_str(&format!("gpu_len={}\n", gpu_compressed.len()));
    meta.push_str(&format!("cpu_fallback_len={}\n", cpu_fallback.len()));
    meta.push_str(&format!("raw_fnv1a64={:016x}\n", fnv1a64(raw)));
    meta.push_str(&format!("gpu_fnv1a64={:016x}\n", fnv1a64(gpu_compressed)));
    meta.push_str(&format!(
        "cpu_fallback_fnv1a64={:016x}\n",
        fnv1a64(cpu_fallback)
    ));
    let _ = std::fs::write(meta_path, meta);
}

fn build_chunk_tasks(
    input: &[u8],
    options: &HybridOptions,
    gpu_available: bool,
) -> Result<Vec<ChunkTask>, CozipDeflateError> {
    let mut tasks: Vec<ChunkTask> = input
        .chunks(options.chunk_size)
        .enumerate()
        .map(|(index, raw)| ChunkTask {
            index,
            preferred_gpu: false,
            raw: raw.to_vec(),
        })
        .collect();

    if !gpu_available || options.gpu_fraction <= 0.0 {
        return Ok(tasks);
    }

    let eligible: Vec<usize> = tasks
        .iter()
        .enumerate()
        .filter_map(|(position, task)| {
            if task.raw.len() >= options.gpu_min_chunk_size {
                Some(position)
            } else {
                None
            }
        })
        .collect();

    if eligible.is_empty() {
        return Ok(tasks);
    }

    let target_gpu = ((eligible.len() as f32) * options.gpu_fraction)
        .round()
        .clamp(0.0, eligible.len() as f32) as usize;
    let target_gpu = target_gpu.max(1).min(eligible.len());

    for slot in 0..target_gpu {
        let pos = slot * eligible.len() / target_gpu;
        let task_index = eligible[pos];
        let task = tasks
            .get_mut(task_index)
            .ok_or(CozipDeflateError::Internal("gpu reservation out of range"))?;
        task.preferred_gpu = true;
    }

    Ok(tasks)
}

fn summarize_encoded_chunks(chunks: &[ChunkMember], gpu_available: bool) -> HybridStats {
    let mut stats = HybridStats {
        chunk_count: chunks.len(),
        cpu_chunks: 0,
        gpu_chunks: 0,
        gpu_available,
        cpu_bytes: 0,
        gpu_bytes: 0,
    };

    for chunk in chunks {
        match chunk.backend {
            ChunkBackend::Cpu => {
                stats.cpu_chunks += 1;
                stats.cpu_bytes += chunk.raw_len as usize;
            }
            ChunkBackend::GpuAssisted => {
                stats.gpu_chunks += 1;
                stats.gpu_bytes += chunk.raw_len as usize;
            }
        }
    }

    stats
}

fn compress_cpu_worker(
    queue: Arc<Mutex<VecDeque<ChunkTask>>>,
    results: Arc<Mutex<Vec<Option<ChunkMember>>>>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    options: &HybridOptions,
    gpu_enabled: bool,
) {
    loop {
        if has_error(&error) {
            break;
        }

        let task = {
            let mut guard = match lock(&queue) {
                Ok(value) => value,
                Err(err) => {
                    set_error(&error, err);
                    break;
                }
            };
            pop_cpu_task(&mut guard, gpu_enabled, options.gpu_min_chunk_size)
        };

        let Some(task) = task else {
            let queue_empty = lock(&queue).map(|guard| guard.is_empty()).unwrap_or(true);
            if queue_empty {
                break;
            }
            std::thread::yield_now();
            continue;
        };

        match compress_chunk_cpu(task, options.compression_level) {
            Ok(encoded) => {
                if let Err(err) = store_encoded_result(&results, encoded) {
                    set_error(&error, err);
                    break;
                }
            }
            Err(err) => {
                set_error(&error, err);
                break;
            }
        }
    }
}

fn compress_gpu_worker(
    queue: Arc<Mutex<VecDeque<ChunkTask>>>,
    results: Arc<Mutex<Vec<Option<ChunkMember>>>>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    options: &HybridOptions,
    gpu: Arc<GpuAssist>,
) {
    loop {
        if has_error(&error) {
            break;
        }

        let tasks = {
            let mut guard = match lock(&queue) {
                Ok(value) => value,
                Err(err) => {
                    set_error(&error, err);
                    break;
                }
            };
            pop_gpu_batch_tasks(&mut guard, options.gpu_min_chunk_size, GPU_BATCH_CHUNKS)
        };

        if tasks.is_empty() {
            break;
        }

        let mut gpu_tasks = Vec::new();
        let mut cpu_fallback_tasks = Vec::new();

        for task in tasks {
            if task.raw.len() >= options.gpu_min_chunk_size {
                gpu_tasks.push(task);
            } else {
                cpu_fallback_tasks.push(task);
            }
        }

        let mut encoded_batch = Vec::new();
        if !gpu_tasks.is_empty() {
            match compress_chunk_gpu_batch(&gpu_tasks, options, &gpu) {
                Ok(value) => {
                    encoded_batch.extend(value);
                }
                Err(_) => {
                    for task in gpu_tasks {
                        match compress_chunk_cpu(task, options.compression_level) {
                            Ok(value) => encoded_batch.push(value),
                            Err(err) => {
                                set_error(&error, err);
                                return;
                            }
                        }
                    }
                }
            }
        }

        for task in cpu_fallback_tasks {
            match compress_chunk_cpu(task, options.compression_level) {
                Ok(value) => encoded_batch.push(value),
                Err(err) => {
                    set_error(&error, err);
                    return;
                }
            }
        }

        for value in encoded_batch {
            if let Err(err) = store_encoded_result(&results, value) {
                set_error(&error, err);
                break;
            }
        }

        if has_error(&error) {
            break;
        }
    }
}

fn pop_gpu_batch_tasks(
    queue: &mut VecDeque<ChunkTask>,
    gpu_min_chunk_size: usize,
    max_batch_chunks: usize,
) -> Vec<ChunkTask> {
    let mut tasks = Vec::with_capacity(max_batch_chunks.max(1));
    let batch_limit = max_batch_chunks.max(1);

    while tasks.len() < batch_limit {
        let Some(task) = pop_gpu_task(queue, gpu_min_chunk_size) else {
            break;
        };
        tasks.push(task);
    }

    tasks
}

fn compress_chunk_gpu_batch(
    tasks: &[ChunkTask],
    options: &HybridOptions,
    gpu: &GpuAssist,
) -> Result<Vec<ChunkMember>, CozipDeflateError> {
    if tasks.is_empty() {
        return Ok(Vec::new());
    }

    let task_data: Vec<&[u8]> = tasks.iter().map(|task| task.raw.as_slice()).collect();
    let compressed_batch = gpu.deflate_fixed_literals_batch(&task_data, options)?;

    if compressed_batch.len() != tasks.len() {
        return Err(CozipDeflateError::Internal(
            "gpu batch returned mismatched compressed vectors",
        ));
    }

    let mut out = Vec::with_capacity(tasks.len());
    for (task, compressed) in tasks.iter().zip(compressed_batch.into_iter()) {
        let raw_len = u32::try_from(task.raw.len()).map_err(|_| CozipDeflateError::DataTooLarge)?;
        let member = if should_validate_gpu_chunk(options, task.index)
            && !gpu_chunk_roundtrip_matches(&task.raw, &compressed)
        {
            let cpu_compressed = deflate_compress_cpu(&task.raw, options.compression_level)?;
            ChunkMember {
                index: task.index,
                backend: ChunkBackend::Cpu,
                transform: ChunkTransform::None,
                codec: ChunkCodec::DeflateCpu,
                raw_len,
                compressed: cpu_compressed,
            }
        } else {
            ChunkMember {
                index: task.index,
                backend: ChunkBackend::GpuAssisted,
                transform: ChunkTransform::None,
                codec: ChunkCodec::DeflateGpuFast,
                raw_len,
                compressed,
            }
        };
        out.push(member);
    }

    Ok(out)
}

fn decompress_worker_on_cpu(
    queue: Arc<Mutex<VecDeque<ChunkDescriptor>>>,
    results: Arc<Mutex<Vec<Option<DecodedChunk>>>>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    options: &HybridOptions,
) {
    loop {
        if has_error(&error) {
            break;
        }

        let descriptor = {
            let mut guard = match lock(&queue) {
                Ok(value) => value,
                Err(err) => {
                    set_error(&error, err);
                    break;
                }
            };
            pop_cpu_descriptor(&mut guard)
        };

        let Some(descriptor) = descriptor else { break };

        match decode_descriptor_on_cpu(descriptor, options) {
            Ok(decoded) => {
                if let Err(err) = store_decoded_result(&results, decoded) {
                    set_error(&error, err);
                    break;
                }
            }
            Err(err) => {
                set_error(&error, err);
                break;
            }
        }
    }
}

fn has_error(error: &Mutex<Option<CozipDeflateError>>) -> bool {
    error.lock().map(|guard| guard.is_some()).unwrap_or(true)
}

fn set_error(error: &Mutex<Option<CozipDeflateError>>, value: CozipDeflateError) {
    if let Ok(mut guard) = error.lock()
        && guard.is_none()
    {
        *guard = Some(value);
    }
}

fn pop_cpu_task(
    queue: &mut VecDeque<ChunkTask>,
    gpu_enabled: bool,
    gpu_min_chunk_size: usize,
) -> Option<ChunkTask> {
    if gpu_enabled {
        if let Some(pos) = queue
            .iter()
            .position(|task| !task.preferred_gpu || task.raw.len() < gpu_min_chunk_size)
        {
            return queue.remove(pos);
        }
        // Work stealing: if CPU-friendly work is exhausted, steal from GPU-preferred tasks.
        return queue.pop_front();
    }
    queue.pop_front()
}

fn pop_gpu_task(queue: &mut VecDeque<ChunkTask>, gpu_min_chunk_size: usize) -> Option<ChunkTask> {
    if let Some(pos) = queue.iter().position(|task| task.preferred_gpu) {
        return queue.remove(pos);
    }

    if let Some(pos) = queue
        .iter()
        .position(|task| task.raw.len() >= gpu_min_chunk_size)
    {
        return queue.remove(pos);
    }

    queue.pop_front()
}

fn pop_cpu_descriptor(queue: &mut VecDeque<ChunkDescriptor>) -> Option<ChunkDescriptor> {
    queue.pop_front()
}

fn compress_chunk_cpu(
    task: ChunkTask,
    compression_level: u32,
) -> Result<ChunkMember, CozipDeflateError> {
    let compressed = deflate_compress_cpu(&task.raw, compression_level)?;
    Ok(ChunkMember {
        index: task.index,
        backend: ChunkBackend::Cpu,
        transform: ChunkTransform::None,
        codec: ChunkCodec::DeflateCpu,
        raw_len: u32::try_from(task.raw.len()).map_err(|_| CozipDeflateError::DataTooLarge)?,
        compressed,
    })
}

fn decode_deflate_by_codec(
    codec: ChunkCodec,
    compressed: &[u8],
) -> Result<Vec<u8>, CozipDeflateError> {
    match codec {
        ChunkCodec::DeflateCpu | ChunkCodec::DeflateGpuFast => {
            deflate_decompress_on_cpu(compressed)
        }
    }
}

fn decode_descriptor_on_cpu(
    descriptor: ChunkDescriptor,
    options: &HybridOptions,
) -> Result<DecodedChunk, CozipDeflateError> {
    let inflated = decode_deflate_by_codec(descriptor.codec, &descriptor.compressed)?;

    let raw = match descriptor.transform {
        ChunkTransform::None => inflated,
        ChunkTransform::EvenOdd => {
            even_odd_transform_cpu(&inflated, options.gpu_subchunk_size, true)
        }
    };

    if raw.len() != descriptor.raw_len as usize {
        eprintln!(
            "[cozip][error] cpu_decode_len_mismatch index={} backend={:?} codec={:?} raw_len={} decoded_len={} compressed_len={}",
            descriptor.index,
            descriptor.backend,
            descriptor.codec,
            descriptor.raw_len,
            raw.len(),
            descriptor.compressed.len(),
        );
        return Err(CozipDeflateError::InvalidFrame(
            "raw chunk length mismatch in cpu path",
        ));
    }

    Ok(DecodedChunk {
        index: descriptor.index,
        raw,
    })
}

fn even_odd_transform_cpu(data: &[u8], block_size: usize, inverse: bool) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut out = vec![0_u8; data.len()];
    let block = block_size.max(1);

    let mut start = 0;
    while start < data.len() {
        let len = (data.len() - start).min(block);
        let even_count = len.div_ceil(TRANSFORM_LANES);

        for local in 0..len {
            let src_local = if !inverse {
                if local < even_count {
                    local * TRANSFORM_LANES
                } else {
                    (local - even_count) * TRANSFORM_LANES + 1
                }
            } else if local % TRANSFORM_LANES == 0 {
                local / TRANSFORM_LANES
            } else {
                even_count + (local / TRANSFORM_LANES)
            };

            out[start + local] = data[start + src_local];
        }

        start += len;
    }

    out
}

struct BitWriter {
    out: Vec<u8>,
    bitbuf: u64,
    bitcount: u8,
    total_bits: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            bitbuf: 0,
            bitcount: 0,
            total_bits: 0,
        }
    }

    fn write_bits(&mut self, value: u32, bits: u8) {
        self.bitbuf |= (value as u64) << self.bitcount;
        self.bitcount += bits;
        self.total_bits += bits as usize;

        while self.bitcount >= 8 {
            self.out.push((self.bitbuf & 0xFF) as u8);
            self.bitbuf >>= 8;
            self.bitcount -= 8;
        }
    }

    fn bit_len(&self) -> usize {
        self.total_bits
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bitcount > 0 {
            self.out.push((self.bitbuf & 0xFF) as u8);
        }
        self.out
    }
}

fn reverse_bits(value: u16, bit_len: u8) -> u16 {
    let mut out = 0_u16;
    let mut i = 0;
    while i < bit_len {
        out = (out << 1) | ((value >> i) & 1);
        i += 1;
    }
    out
}

fn build_huffman_code_lengths(freq: &[u32], max_bits: u8) -> Option<Vec<u8>> {
    #[derive(Clone, Copy)]
    struct Node {
        left: Option<usize>,
        right: Option<usize>,
        symbol: Option<usize>,
    }

    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut heap: BinaryHeap<(Reverse<u32>, usize)> = BinaryHeap::new();
    let mut nodes = Vec::<Node>::new();

    for (symbol, &weight) in freq.iter().enumerate() {
        if weight == 0 {
            continue;
        }
        let idx = nodes.len();
        nodes.push(Node {
            left: None,
            right: None,
            symbol: Some(symbol),
        });
        heap.push((Reverse(weight), idx));
    }

    if heap.is_empty() {
        return Some(vec![0; freq.len()]);
    }

    if heap.len() == 1 {
        let mut lengths = vec![0_u8; freq.len()];
        if let Some((_, idx)) = heap.pop()
            && let Some(sym) = nodes[idx].symbol
        {
            lengths[sym] = 1;
            return Some(lengths);
        }
        return None;
    }

    while heap.len() > 1 {
        let (Reverse(a_w), a_i) = heap.pop()?;
        let (Reverse(b_w), b_i) = heap.pop()?;
        let parent_idx = nodes.len();
        nodes.push(Node {
            left: Some(a_i),
            right: Some(b_i),
            symbol: None,
        });
        heap.push((Reverse(a_w.saturating_add(b_w)), parent_idx));
    }

    let root_idx = heap.pop()?.1;
    let mut lengths = vec![0_u8; freq.len()];
    let mut stack = vec![(root_idx, 0_u8)];
    let mut max_depth = 0_u8;
    while let Some((idx, depth)) = stack.pop() {
        let node = nodes[idx];
        if let Some(sym) = node.symbol {
            let actual_depth = depth.max(1);
            lengths[sym] = actual_depth;
            max_depth = max_depth.max(actual_depth);
            continue;
        }
        if let Some(left) = node.left {
            stack.push((left, depth.saturating_add(1)));
        }
        if let Some(right) = node.right {
            stack.push((right, depth.saturating_add(1)));
        }
    }

    if max_depth > max_bits {
        return None;
    }

    Some(lengths)
}

fn build_canonical_codes(lengths: &[u8], max_bits: u8) -> Option<Vec<(u16, u8)>> {
    let mut bl_count = vec![0_u16; usize::from(max_bits) + 1];
    for &len in lengths {
        if len > max_bits {
            return None;
        }
        if len > 0 {
            bl_count[len as usize] = bl_count[len as usize].saturating_add(1);
        }
    }

    let mut next_code = vec![0_u16; usize::from(max_bits) + 1];
    let mut code = 0_u16;
    for bits in 1..=usize::from(max_bits) {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    let mut out = vec![(0_u16, 0_u8); lengths.len()];
    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let canonical = next_code[len as usize];
        next_code[len as usize] = next_code[len as usize].saturating_add(1);
        out[symbol] = (reverse_bits(canonical, len), len);
    }

    Some(out)
}

fn encode_code_lengths_rle(lengths: &[u8]) -> Vec<(u8, u8)> {
    // tuple: (symbol, extra_value)
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lengths.len() {
        let current = lengths[i];
        let mut run = 1usize;
        while i + run < lengths.len() && lengths[i + run] == current {
            run += 1;
        }

        if current == 0 {
            let mut remaining = run;
            while remaining > 0 {
                if remaining >= 11 {
                    let count = remaining.min(138);
                    out.push((18, (count - 11) as u8));
                    remaining -= count;
                } else if remaining >= 3 {
                    let count = remaining.min(10);
                    out.push((17, (count - 3) as u8));
                    remaining -= count;
                } else {
                    out.push((0, 0));
                    remaining -= 1;
                }
            }
        } else {
            out.push((current, 0));
            let mut remaining = run - 1;
            while remaining > 0 {
                if remaining >= 3 {
                    let count = remaining.min(6);
                    out.push((16, (count - 3) as u8));
                    remaining -= count;
                } else {
                    out.push((current, 0));
                    remaining -= 1;
                }
            }
        }

        i += run;
    }
    out
}

#[derive(Debug, Clone)]
struct DynamicHuffmanPlan {
    litlen_codes: Vec<u32>,
    litlen_bits: Vec<u32>,
    dist_codes: Vec<u32>,
    dist_bits: Vec<u32>,
    header_bytes: Vec<u8>,
    header_bits: u32,
    eob_code: u16,
    eob_bits: u8,
}

fn build_dynamic_huffman_plan(
    litlen_freq_in: &[u32],
    dist_freq_in: &[u32],
) -> Result<DynamicHuffmanPlan, CozipDeflateError> {
    const MAX_BITS: u8 = 15;
    const CODELEN_MAX_BITS: u8 = 7;
    const CODELEN_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    if litlen_freq_in.len() != LITLEN_SYMBOL_COUNT || dist_freq_in.len() != DIST_SYMBOL_COUNT {
        return Err(CozipDeflateError::Internal("invalid frequency table size"));
    }

    let mut litlen_freq = litlen_freq_in.to_vec();
    let mut dist_freq = dist_freq_in.to_vec();
    litlen_freq[256] = litlen_freq[256].saturating_add(1);
    if dist_freq.iter().all(|value| *value == 0) {
        dist_freq[0] = 1;
    }

    let litlen_lengths = build_huffman_code_lengths(&litlen_freq, MAX_BITS).ok_or(
        CozipDeflateError::Internal("failed to build litlen huffman lengths"),
    )?;
    let dist_lengths = build_huffman_code_lengths(&dist_freq, MAX_BITS).ok_or(
        CozipDeflateError::Internal("failed to build dist huffman lengths"),
    )?;

    let hlit_count = litlen_lengths
        .iter()
        .rposition(|len| *len != 0)
        .map(|index| (index + 1).max(257))
        .unwrap_or(257);
    let hdist_count = dist_lengths
        .iter()
        .rposition(|len| *len != 0)
        .map(|index| (index + 1).max(1))
        .unwrap_or(1);

    let mut header_lengths = Vec::with_capacity(hlit_count + hdist_count);
    header_lengths.extend_from_slice(&litlen_lengths[..hlit_count]);
    header_lengths.extend_from_slice(&dist_lengths[..hdist_count]);
    let cl_rle = encode_code_lengths_rle(&header_lengths);

    let mut cl_freq = vec![0_u32; 19];
    for (symbol, _) in &cl_rle {
        cl_freq[*symbol as usize] = cl_freq[*symbol as usize].saturating_add(1);
    }
    if cl_freq.iter().all(|value| *value == 0) {
        cl_freq[0] = 1;
    }

    let cl_lengths = build_huffman_code_lengths(&cl_freq, CODELEN_MAX_BITS).ok_or(
        CozipDeflateError::Internal("failed to build codelen huffman lengths"),
    )?;
    let hclen_count = CODELEN_ORDER
        .iter()
        .rposition(|&sym| cl_lengths[sym] != 0)
        .map(|index| (index + 1).max(4))
        .unwrap_or(4);

    let litlen_codes_raw = build_canonical_codes(&litlen_lengths, MAX_BITS)
        .ok_or(CozipDeflateError::Internal("failed to build litlen codes"))?;
    let dist_codes_raw = build_canonical_codes(&dist_lengths, MAX_BITS)
        .ok_or(CozipDeflateError::Internal("failed to build dist codes"))?;
    let cl_codes = build_canonical_codes(&cl_lengths, CODELEN_MAX_BITS)
        .ok_or(CozipDeflateError::Internal("failed to build codelen codes"))?;

    let mut writer = BitWriter::new();
    writer.write_bits(1, 1);
    writer.write_bits(0b10, 2);
    writer.write_bits((hlit_count - 257) as u32, 5);
    writer.write_bits((hdist_count - 1) as u32, 5);
    writer.write_bits((hclen_count - 4) as u32, 4);
    for &sym in CODELEN_ORDER.iter().take(hclen_count) {
        writer.write_bits(cl_lengths[sym] as u32, 3);
    }
    for (symbol, extra) in cl_rle {
        let (code, bits) = cl_codes[symbol as usize];
        if bits == 0 {
            return Err(CozipDeflateError::Internal("missing code-length code"));
        }
        writer.write_bits(code as u32, bits);
        match symbol {
            16 => writer.write_bits(extra as u32, 2),
            17 => writer.write_bits(extra as u32, 3),
            18 => writer.write_bits(extra as u32, 7),
            _ => {}
        }
    }

    let header_bits =
        u32::try_from(writer.bit_len()).map_err(|_| CozipDeflateError::DataTooLarge)?;
    let header_bytes = writer.finish();

    let mut litlen_codes = vec![0_u32; LITLEN_SYMBOL_COUNT];
    let mut litlen_bits = vec![0_u32; LITLEN_SYMBOL_COUNT];
    for (idx, (code, bits)) in litlen_codes_raw.into_iter().enumerate() {
        litlen_codes[idx] = code as u32;
        litlen_bits[idx] = bits as u32;
    }

    let mut dist_codes = vec![0_u32; DIST_SYMBOL_COUNT];
    let mut dist_bits = vec![0_u32; DIST_SYMBOL_COUNT];
    for (idx, (code, bits)) in dist_codes_raw.into_iter().enumerate() {
        dist_codes[idx] = code as u32;
        dist_bits[idx] = bits as u32;
    }

    let eob_code = litlen_codes[256] as u16;
    let eob_bits = litlen_bits[256] as u8;
    if eob_bits == 0 {
        return Err(CozipDeflateError::Internal("missing end-of-block code"));
    }

    Ok(DynamicHuffmanPlan {
        litlen_codes,
        litlen_bits,
        dist_codes,
        dist_bits,
        header_bytes,
        header_bits,
        eob_code,
        eob_bits,
    })
}

fn store_encoded_result(
    results: &Mutex<Vec<Option<ChunkMember>>>,
    encoded: ChunkMember,
) -> Result<(), CozipDeflateError> {
    let mut guard = lock(results)?;
    let index = encoded.index;
    let slot = guard
        .get_mut(index)
        .ok_or(CozipDeflateError::Internal("compressed index out of range"))?;

    if slot.is_some() {
        return Err(CozipDeflateError::Internal("duplicate compressed index"));
    }

    *slot = Some(encoded);
    Ok(())
}

fn store_decoded_result(
    results: &Mutex<Vec<Option<DecodedChunk>>>,
    decoded: DecodedChunk,
) -> Result<(), CozipDeflateError> {
    let mut guard = lock(results)?;
    let index = decoded.index;
    let slot = guard
        .get_mut(index)
        .ok_or(CozipDeflateError::Internal("decoded index out of range"))?;

    if slot.is_some() {
        return Err(CozipDeflateError::Internal("duplicate decoded index"));
    }

    *slot = Some(decoded);
    Ok(())
}

#[cfg(test)]
mod tests;
