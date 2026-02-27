use std::collections::{BTreeMap, VecDeque};
use std::fs::File as StdFile;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use flate2::Compression;
use thiserror::Error;

const FRAME_MAGIC: [u8; 4] = *b"CZDF";
const FRAME_VERSION: u8 = 3;
const STREAM_MAGIC: [u8; 4] = *b"CZDS";
const STREAM_VERSION: u8 = 1;
const STREAM_HEADER_LEN: usize = 8;
const HEADER_LEN: usize = 22;
const CHUNK_META_LEN_V3: usize = 11;
const CHUNK_META_LEN_V2: usize = 10;
const CHUNK_META_LEN_V1: usize = 9;
const TRANSFORM_LANES: usize = 2;

const LITLEN_SYMBOL_COUNT: usize = 286;
const DIST_SYMBOL_COUNT: usize = 30;
const DYN_TABLE_U32_COUNT: usize = (LITLEN_SYMBOL_COUNT * 2) + (DIST_SYMBOL_COUNT * 2);
const MAX_GPU_BATCH_CHUNKS: usize = 16;
const GPU_FREQ_MAX_WORKGROUPS: u32 = 4096;
const PREFIX_SCAN_BLOCK_SIZE: usize = 256;
const DEFAULT_TOKEN_FINALIZE_SEGMENT_SIZE: usize = 4096;
const DEFAULT_STREAM_BATCH_CHUNKS: usize = 0;
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
    pub gpu_slot_count: usize,
    pub stream_prepare_pipeline_depth: usize,
    pub stream_batch_chunks: usize,
    pub stream_max_inflight_chunks: usize,
    pub stream_max_inflight_bytes: usize,
    pub gpu_batch_chunks: usize,
    pub gpu_pipelined_submit_chunks: usize,
    pub token_finalize_segment_size: usize,
    pub compression_level: u32,
    pub compression_mode: CompressionMode,
    pub prefer_gpu: bool,
    pub gpu_fraction: f32,
    pub gpu_tail_stop_ratio: f32,
    pub gpu_min_chunk_size: usize,
    pub gpu_validation_mode: GpuValidationMode,
    pub gpu_validation_sample_every: usize,
    pub gpu_dynamic_self_check: bool,
    pub gpu_dump_bad_chunk: bool,
    pub gpu_dump_bad_chunk_limit: usize,
    pub gpu_dump_bad_chunk_dir: Option<String>,
    pub profile_timing: bool,
    pub profile_timing_detail: bool,
    pub profile_timing_deep: bool,
    pub scheduler_policy: HybridSchedulerPolicy,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridSchedulerPolicy {
    GlobalQueueLocalBuffers,
}

impl Default for HybridOptions {
    fn default() -> Self {
        Self {
            chunk_size: 4 * 1024 * 1024,
            gpu_subchunk_size: 256 * 1024,
            gpu_slot_count: 4,
            stream_prepare_pipeline_depth: 2,
            stream_batch_chunks: DEFAULT_STREAM_BATCH_CHUNKS,
            stream_max_inflight_chunks: 256,
            stream_max_inflight_bytes: 0,
            gpu_batch_chunks: 4,
            gpu_pipelined_submit_chunks: 4,
            token_finalize_segment_size: DEFAULT_TOKEN_FINALIZE_SEGMENT_SIZE,
            compression_level: 6,
            compression_mode: CompressionMode::Speed,
            prefer_gpu: true,
            gpu_fraction: 1.0,
            gpu_tail_stop_ratio: 1.0,
            gpu_min_chunk_size: 64 * 1024,
            gpu_validation_mode: GpuValidationMode::Off,
            gpu_validation_sample_every: 8,
            gpu_dynamic_self_check: false,
            gpu_dump_bad_chunk: false,
            gpu_dump_bad_chunk_limit: 8,
            gpu_dump_bad_chunk_dir: None,
            profile_timing: false,
            profile_timing_detail: false,
            profile_timing_deep: false,
            scheduler_policy: HybridSchedulerPolicy::GlobalQueueLocalBuffers,
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

#[derive(Debug, Clone, Copy)]
pub struct StreamOptions {
    pub frame_input_size: usize,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            frame_input_size: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StreamStats {
    pub frames: usize,
    pub input_bytes: usize,
    pub output_bytes: usize,
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
        let gpu_context = if is_gpu_requested(&options) {
            let start = Instant::now();
            let ctx = GpuAssist::new(&options).ok().map(Arc::new);
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

    pub fn deflate_compress_stream_zip_compatible<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
        deflate_compress_stream_hybrid_zip_compatible_with_context(
            reader,
            writer,
            &self.options,
            self.gpu_context.clone(),
        )
    }

    pub fn compress_stream<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
        stream_options: StreamOptions,
    ) -> Result<StreamStats, CozipDeflateError> {
        if stream_options.frame_input_size == 0 {
            return Err(CozipDeflateError::InvalidOptions(
                "stream frame_input_size must be greater than 0",
            ));
        }

        writer.write_all(&stream_header_bytes())?;
        let mut stats = StreamStats {
            frames: 0,
            input_bytes: 0,
            output_bytes: STREAM_HEADER_LEN,
        };

        let mut buffer = vec![0_u8; stream_options.frame_input_size];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            let compressed = self.compress(&buffer[..read])?;
            let frame_len = u64::try_from(compressed.bytes.len())
                .map_err(|_| CozipDeflateError::DataTooLarge)?;
            writer.write_all(&frame_len.to_le_bytes())?;
            writer.write_all(&compressed.bytes)?;

            stats.frames += 1;
            stats.input_bytes = stats.input_bytes.saturating_add(read);
            stats.output_bytes = stats
                .output_bytes
                .saturating_add(std::mem::size_of::<u64>())
                .saturating_add(compressed.bytes.len());
        }

        writer.write_all(&0_u64.to_le_bytes())?;
        writer.flush()?;
        stats.output_bytes = stats
            .output_bytes
            .saturating_add(std::mem::size_of::<u64>());
        Ok(stats)
    }

    pub fn decompress_stream<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<StreamStats, CozipDeflateError> {
        let mut header = [0_u8; STREAM_HEADER_LEN];
        read_exact_frame(reader, &mut header)?;
        validate_stream_header(&header)?;

        let mut stats = StreamStats {
            frames: 0,
            input_bytes: STREAM_HEADER_LEN,
            output_bytes: 0,
        };
        let mut len_buf = [0_u8; std::mem::size_of::<u64>()];

        loop {
            read_exact_frame(reader, &mut len_buf)?;
            stats.input_bytes = stats.input_bytes.saturating_add(std::mem::size_of::<u64>());

            let frame_len = u64::from_le_bytes(len_buf);
            if frame_len == 0 {
                break;
            }

            let frame_len_usize =
                usize::try_from(frame_len).map_err(|_| CozipDeflateError::DataTooLarge)?;
            let mut frame = vec![0_u8; frame_len_usize];
            read_exact_frame(reader, &mut frame)?;
            stats.input_bytes = stats.input_bytes.saturating_add(frame_len_usize);

            let decompressed = self.decompress_on_cpu(&frame)?;
            writer.write_all(&decompressed.bytes)?;
            stats.frames += 1;
            stats.output_bytes = stats.output_bytes.saturating_add(decompressed.bytes.len());
        }

        writer.flush()?;
        Ok(stats)
    }

    pub fn compress_file(
        &self,
        input_file: StdFile,
        output_file: StdFile,
        stream_options: StreamOptions,
    ) -> Result<StreamStats, CozipDeflateError> {
        let mut reader = std::io::BufReader::new(input_file);
        let mut writer = std::io::BufWriter::new(output_file);
        self.compress_stream(&mut reader, &mut writer, stream_options)
    }

    pub fn decompress_file(
        &self,
        input_file: StdFile,
        output_file: StdFile,
    ) -> Result<StreamStats, CozipDeflateError> {
        let mut reader = std::io::BufReader::new(input_file);
        let mut writer = std::io::BufWriter::new(output_file);
        self.decompress_stream(&mut reader, &mut writer)
    }

    pub fn compress_file_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
        stream_options: StreamOptions,
    ) -> Result<StreamStats, CozipDeflateError> {
        let input = StdFile::open(input_path)?;
        let output = StdFile::create(output_path)?;
        self.compress_file(input, output, stream_options)
    }

    pub fn decompress_file_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
    ) -> Result<StreamStats, CozipDeflateError> {
        let input = StdFile::open(input_path)?;
        let output = StdFile::create(output_path)?;
        self.decompress_file(input, output)
    }

    pub async fn compress_file_async(
        &self,
        mut input_file: tokio::fs::File,
        mut output_file: tokio::fs::File,
        stream_options: StreamOptions,
    ) -> Result<StreamStats, CozipDeflateError> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        if stream_options.frame_input_size == 0 {
            return Err(CozipDeflateError::InvalidOptions(
                "stream frame_input_size must be greater than 0",
            ));
        }

        output_file.write_all(&stream_header_bytes()).await?;
        let mut stats = StreamStats {
            frames: 0,
            input_bytes: 0,
            output_bytes: STREAM_HEADER_LEN,
        };

        let mut buffer = vec![0_u8; stream_options.frame_input_size];
        loop {
            let read = input_file.read(&mut buffer).await?;
            if read == 0 {
                break;
            }
            let compressed = self.compress(&buffer[..read])?;
            let frame_len = u64::try_from(compressed.bytes.len())
                .map_err(|_| CozipDeflateError::DataTooLarge)?;
            output_file.write_all(&frame_len.to_le_bytes()).await?;
            output_file.write_all(&compressed.bytes).await?;

            stats.frames += 1;
            stats.input_bytes = stats.input_bytes.saturating_add(read);
            stats.output_bytes = stats
                .output_bytes
                .saturating_add(std::mem::size_of::<u64>())
                .saturating_add(compressed.bytes.len());
        }

        output_file.write_all(&0_u64.to_le_bytes()).await?;
        output_file.flush().await?;
        stats.output_bytes = stats
            .output_bytes
            .saturating_add(std::mem::size_of::<u64>());
        Ok(stats)
    }

    pub async fn decompress_file_async(
        &self,
        mut input_file: tokio::fs::File,
        mut output_file: tokio::fs::File,
    ) -> Result<StreamStats, CozipDeflateError> {
        use tokio::io::AsyncWriteExt;

        let mut header = [0_u8; STREAM_HEADER_LEN];
        read_exact_frame_async(&mut input_file, &mut header).await?;
        validate_stream_header(&header)?;

        let mut stats = StreamStats {
            frames: 0,
            input_bytes: STREAM_HEADER_LEN,
            output_bytes: 0,
        };
        let mut len_buf = [0_u8; std::mem::size_of::<u64>()];

        loop {
            read_exact_frame_async(&mut input_file, &mut len_buf).await?;
            stats.input_bytes = stats.input_bytes.saturating_add(std::mem::size_of::<u64>());

            let frame_len = u64::from_le_bytes(len_buf);
            if frame_len == 0 {
                break;
            }

            let frame_len_usize =
                usize::try_from(frame_len).map_err(|_| CozipDeflateError::DataTooLarge)?;
            let mut frame = vec![0_u8; frame_len_usize];
            read_exact_frame_async(&mut input_file, &mut frame).await?;
            stats.input_bytes = stats.input_bytes.saturating_add(frame_len_usize);

            let decompressed = self.decompress_on_cpu(&frame)?;
            output_file.write_all(&decompressed.bytes).await?;
            stats.frames += 1;
            stats.output_bytes = stats.output_bytes.saturating_add(decompressed.bytes.len());
        }

        output_file.flush().await?;
        Ok(stats)
    }

    pub async fn compress_file_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
        stream_options: StreamOptions,
    ) -> Result<StreamStats, CozipDeflateError> {
        let input = tokio::fs::File::open(input_path).await?;
        let output = tokio::fs::File::create(output_path).await?;
        self.compress_file_async(input, output, stream_options)
            .await
    }

    pub async fn decompress_file_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
    ) -> Result<StreamStats, CozipDeflateError> {
        let input = tokio::fs::File::open(input_path).await?;
        let output = tokio::fs::File::create(output_path).await?;
        self.decompress_file_async(input, output).await
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

fn maybe_warn_deep_profile_enabled(options: &HybridOptions) {
    if options.profile_timing_deep && !options.profile_timing {
        eprintln!("[cozip][warn] profile_timing_deep=true ignored because profile_timing=false");
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn accumulate_write_stage_metrics(
    stats: &mut DeflateCpuStreamStats,
    elapsed_stage_ms: f64,
    io_before: (u64, usize, u64),
    io_after: (u64, usize, u64),
) {
    let io_ns = io_after.0.saturating_sub(io_before.0);
    let io_calls = io_after.1.saturating_sub(io_before.1);
    let io_bytes = io_after.2.saturating_sub(io_before.2);
    let io_ms = io_ns as f64 / 1_000_000.0;
    let pack_ms = (elapsed_stage_ms - io_ms).max(0.0);
    stats.write_stage_ms += elapsed_stage_ms;
    stats.write_io_ms += io_ms;
    stats.write_pack_ms += pack_ms;
    stats.write_io_calls = stats.write_io_calls.saturating_add(io_calls);
    stats.write_io_bytes = stats.write_io_bytes.saturating_add(io_bytes);
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
    layout: Option<DeflateStreamLayout>,
    compressed: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default)]
struct EncodeWorkStats {
    cpu_busy_ms: f64,
    gpu_busy_ms: f64,
    cpu_queue_lock_wait_ms: f64,
    gpu_queue_lock_wait_ms: f64,
    cpu_wait_for_task_ms: f64,
    gpu_wait_for_task_ms: f64,
    cpu_chunks_done: usize,
    gpu_chunks_done: usize,
    cpu_steal_chunks: usize,
    gpu_batches: usize,
    cpu_no_task_events: usize,
    gpu_no_task_events: usize,
    cpu_yield_events: usize,
    gpu_yield_events: usize,
    initial_gpu_queue_chunks: usize,
    gpu_steal_reserve_chunks: usize,
}

#[derive(Debug, Default)]
struct WorkerCounters {
    cpu_busy_ns: std::sync::atomic::AtomicU64,
    gpu_busy_ns: std::sync::atomic::AtomicU64,
    cpu_queue_lock_wait_ns: std::sync::atomic::AtomicU64,
    gpu_queue_lock_wait_ns: std::sync::atomic::AtomicU64,
    cpu_wait_for_task_ns: std::sync::atomic::AtomicU64,
    gpu_wait_for_task_ns: std::sync::atomic::AtomicU64,
    cpu_chunks: AtomicUsize,
    gpu_chunks: AtomicUsize,
    cpu_steal_chunks: AtomicUsize,
    gpu_batches: AtomicUsize,
    cpu_no_task_events: AtomicUsize,
    gpu_no_task_events: AtomicUsize,
    cpu_yield_events: AtomicUsize,
    gpu_yield_events: AtomicUsize,
}

impl WorkerCounters {
    fn snapshot(&self) -> EncodeWorkStats {
        EncodeWorkStats {
            cpu_busy_ms: self.cpu_busy_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            gpu_busy_ms: self.gpu_busy_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            cpu_queue_lock_wait_ms: self.cpu_queue_lock_wait_ns.load(Ordering::Relaxed) as f64
                / 1_000_000.0,
            gpu_queue_lock_wait_ms: self.gpu_queue_lock_wait_ns.load(Ordering::Relaxed) as f64
                / 1_000_000.0,
            cpu_wait_for_task_ms: self.cpu_wait_for_task_ns.load(Ordering::Relaxed) as f64
                / 1_000_000.0,
            gpu_wait_for_task_ms: self.gpu_wait_for_task_ns.load(Ordering::Relaxed) as f64
                / 1_000_000.0,
            cpu_chunks_done: self.cpu_chunks.load(Ordering::Relaxed),
            gpu_chunks_done: self.gpu_chunks.load(Ordering::Relaxed),
            cpu_steal_chunks: self.cpu_steal_chunks.load(Ordering::Relaxed),
            gpu_batches: self.gpu_batches.load(Ordering::Relaxed),
            cpu_no_task_events: self.cpu_no_task_events.load(Ordering::Relaxed),
            gpu_no_task_events: self.gpu_no_task_events.load(Ordering::Relaxed),
            cpu_yield_events: self.cpu_yield_events.load(Ordering::Relaxed),
            gpu_yield_events: self.gpu_yield_events.load(Ordering::Relaxed),
            initial_gpu_queue_chunks: 0,
            gpu_steal_reserve_chunks: 0,
        }
    }
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

fn wait_on_condvar<'a, T>(
    condvar: &Condvar,
    guard: std::sync::MutexGuard<'a, T>,
) -> Result<std::sync::MutexGuard<'a, T>, CozipDeflateError> {
    condvar
        .wait(guard)
        .map_err(|_| CozipDeflateError::Internal("mutex poisoned"))
}

fn wait_timeout_on_condvar<'a, T>(
    condvar: &Condvar,
    guard: std::sync::MutexGuard<'a, T>,
    timeout: Duration,
) -> Result<std::sync::MutexGuard<'a, T>, CozipDeflateError> {
    condvar
        .wait_timeout(guard, timeout)
        .map(|(guard, _)| guard)
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

#[derive(Debug, Clone, Copy, Default)]
pub struct DeflateCpuStreamStats {
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub input_crc32: u32,
    pub output_crc32: u32,
    pub chunk_count: usize,
    pub cpu_chunks: usize,
    pub gpu_chunks: usize,
    pub gpu_available: bool,
    pub compress_stage_ms: f64,
    pub layout_parse_ms: f64,
    pub write_stage_ms: f64,
    pub write_pack_ms: f64,
    pub write_io_ms: f64,
    pub write_io_calls: usize,
    pub write_io_bytes: u64,
    pub cpu_worker_busy_ms: f64,
    pub gpu_worker_busy_ms: f64,
    pub cpu_queue_lock_wait_ms: f64,
    pub gpu_queue_lock_wait_ms: f64,
    pub cpu_wait_for_task_ms: f64,
    pub gpu_wait_for_task_ms: f64,
    pub writer_wait_ms: f64,
    pub writer_wait_events: usize,
    pub writer_hol_wait_ms: f64,
    pub writer_hol_wait_events: usize,
    pub writer_hol_ready_sum: usize,
    pub writer_hol_ready_max: usize,
    pub inflight_chunks_max: usize,
    pub ready_chunks_max: usize,
    pub gpu_runtime_disabled: bool,
    pub cpu_worker_chunks: usize,
    pub gpu_worker_chunks: usize,
    pub cpu_steal_chunks: usize,
    pub gpu_batch_count: usize,
    pub cpu_no_task_events: usize,
    pub gpu_no_task_events: usize,
    pub cpu_yield_events: usize,
    pub gpu_yield_events: usize,
    pub initial_gpu_queue_chunks: usize,
    pub gpu_steal_reserve_chunks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeflateStreamMode {
    Cpu,
    Hybrid,
}

impl Default for DeflateStreamMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

pub fn deflate_compress_stream_on_cpu<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    level: u32,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    const STREAM_BUF_SIZE: usize = 256 * 1024;

    let mut stats = DeflateCpuStreamStats::default();
    let mut input_crc = crc32fast::Hasher::new();
    let mut output = HashingCountWriter::new(writer);
    {
        let mut encoder =
            flate2::write::DeflateEncoder::new(&mut output, Compression::new(level.clamp(0, 9)));
        let mut buf = vec![0_u8; STREAM_BUF_SIZE];
        loop {
            let read = reader.read(&mut buf)?;
            if read == 0 {
                break;
            }
            encoder.write_all(&buf[..read])?;
            input_crc.update(&buf[..read]);
            stats.input_bytes = stats
                .input_bytes
                .saturating_add(u64::try_from(read).unwrap_or(u64::MAX));
        }
        encoder.finish()?;
    }

    stats.input_crc32 = input_crc.finalize();
    stats.output_bytes = output.written;
    stats.output_crc32 = output.hasher.finalize();
    Ok(stats)
}

pub fn deflate_compress_stream<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    level: u32,
    mode: DeflateStreamMode,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    match mode {
        DeflateStreamMode::Cpu => deflate_compress_stream_on_cpu(reader, writer, level),
        DeflateStreamMode::Hybrid => {
            deflate_compress_stream_hybrid_zip_compatible(reader, writer, level)
        }
    }
}

pub fn deflate_decompress_stream_on_cpu<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    const STREAM_BUF_SIZE: usize = 256 * 1024;

    let mut stats = DeflateCpuStreamStats::default();
    let mut output = HashingCountWriter::new(writer);
    {
        let mut decoder = flate2::write::DeflateDecoder::new(&mut output);
        let mut input_crc = crc32fast::Hasher::new();
        let mut buf = vec![0_u8; STREAM_BUF_SIZE];
        loop {
            let read = reader.read(&mut buf)?;
            if read == 0 {
                break;
            }
            decoder.write_all(&buf[..read])?;
            input_crc.update(&buf[..read]);
            stats.input_bytes = stats
                .input_bytes
                .saturating_add(u64::try_from(read).unwrap_or(u64::MAX));
        }
        decoder.finish()?;
        stats.input_crc32 = input_crc.finalize();
    }

    stats.output_bytes = output.written;
    stats.output_crc32 = output.hasher.finalize();
    Ok(stats)
}

const DEFLATE_MAX_HUFF_BITS: usize = 15;
const LENGTH_EXTRA_BITS: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_EXTRA_BITS: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CODELEN_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

#[derive(Debug, Clone, Copy)]
struct DeflateStreamLayout {
    final_header_bit: usize,
    end_bit: usize,
}

#[derive(Debug, Clone)]
struct HuffmanDecoder {
    counts: [u16; DEFLATE_MAX_HUFF_BITS + 1],
    symbols: Vec<u16>,
    max_bits: usize,
}

struct DeflateBitCursor<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> DeflateBitCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    fn total_bits(&self) -> usize {
        self.data.len().saturating_mul(8)
    }

    fn ensure_bits(&self, bits: usize) -> Result<(), CozipDeflateError> {
        if self.bit_pos.saturating_add(bits) > self.total_bits() {
            return Err(CozipDeflateError::InvalidFrame(
                "truncated deflate bitstream in chunk",
            ));
        }
        Ok(())
    }

    fn read_bit(&mut self) -> Result<u8, CozipDeflateError> {
        self.ensure_bits(1)?;
        let bit = bit_at(self.data, self.bit_pos);
        self.bit_pos += 1;
        Ok(bit)
    }

    fn read_bits(&mut self, bits: usize) -> Result<u32, CozipDeflateError> {
        let mut value = 0_u32;
        for shift in 0..bits {
            value |= u32::from(self.read_bit()?) << shift;
        }
        Ok(value)
    }

    fn align_to_byte(&mut self) {
        self.bit_pos = (self.bit_pos + 7) & !7;
    }

    fn skip_bits(&mut self, bits: usize) -> Result<(), CozipDeflateError> {
        self.ensure_bits(bits)?;
        self.bit_pos += bits;
        Ok(())
    }
}

struct DeflateBitWriter<'a, W: Write> {
    inner: &'a mut W,
    byte: u8,
    used: u8,
    out_buf: Vec<u8>,
    io_ns: u64,
    io_calls: usize,
    io_bytes: u64,
}

#[derive(Default)]
struct StreamTaskQueueState {
    queue: VecDeque<ChunkTask>,
    queued_bytes: usize,
    closed: bool,
}

struct StreamReadyChunk {
    chunk: ChunkMember,
    layout: DeflateStreamLayout,
    prepared_non_final_end_bit: Option<usize>,
}

impl<'a, W: Write> DeflateBitWriter<'a, W> {
    fn new(inner: &'a mut W) -> Self {
        Self {
            inner,
            byte: 0,
            used: 0,
            out_buf: Vec::with_capacity(64 * 1024),
            io_ns: 0,
            io_calls: 0,
            io_bytes: 0,
        }
    }

    fn flush_out_buf(&mut self) -> Result<(), CozipDeflateError> {
        if !self.out_buf.is_empty() {
            let bytes = self.out_buf.len();
            let io_start = Instant::now();
            self.inner.write_all(&self.out_buf)?;
            self.io_ns = self
                .io_ns
                .saturating_add(io_start.elapsed().as_nanos() as u64);
            self.io_calls = self.io_calls.saturating_add(1);
            self.io_bytes = self
                .io_bytes
                .saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX));
            self.out_buf.clear();
        }
        Ok(())
    }

    fn emit_byte(&mut self, byte: u8) -> Result<(), CozipDeflateError> {
        self.out_buf.push(byte);
        if self.out_buf.len() >= 32 * 1024 {
            self.flush_out_buf()?;
        }
        Ok(())
    }

    fn emit_bytes(&mut self, bytes: &[u8]) -> Result<(), CozipDeflateError> {
        if bytes.is_empty() {
            return Ok(());
        }

        if self.out_buf.is_empty() && bytes.len() >= 16 * 1024 {
            let io_start = Instant::now();
            self.inner.write_all(bytes)?;
            self.io_ns = self
                .io_ns
                .saturating_add(io_start.elapsed().as_nanos() as u64);
            self.io_calls = self.io_calls.saturating_add(1);
            self.io_bytes = self
                .io_bytes
                .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
            return Ok(());
        }

        if self.out_buf.len() + bytes.len() > 64 * 1024 {
            self.flush_out_buf()?;
            if bytes.len() >= 16 * 1024 {
                let io_start = Instant::now();
                self.inner.write_all(bytes)?;
                self.io_ns = self
                    .io_ns
                    .saturating_add(io_start.elapsed().as_nanos() as u64);
                self.io_calls = self.io_calls.saturating_add(1);
                self.io_bytes = self
                    .io_bytes
                    .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
                return Ok(());
            }
        }

        self.out_buf.extend_from_slice(bytes);
        if self.out_buf.len() >= 32 * 1024 {
            self.flush_out_buf()?;
        }
        Ok(())
    }

    fn write_bit(&mut self, bit: u8) -> Result<(), CozipDeflateError> {
        self.byte |= (bit & 1) << self.used;
        self.used += 1;
        if self.used == 8 {
            self.emit_byte(self.byte)?;
            self.byte = 0;
            self.used = 0;
        }
        Ok(())
    }

    fn write_byte_bits(&mut self, value: u8) -> Result<(), CozipDeflateError> {
        if self.used == 0 {
            self.emit_byte(value)?;
            return Ok(());
        }

        let merged = self.byte | (value << self.used);
        self.emit_byte(merged)?;
        self.byte = value >> (8 - self.used);
        Ok(())
    }

    fn write_bits_from_slice(
        &mut self,
        src: &[u8],
        mut src_bit: usize,
        mut bit_len: usize,
    ) -> Result<(), CozipDeflateError> {
        if bit_len == 0 {
            return Ok(());
        }

        while bit_len > 0 {
            if self.used == 0 && src_bit % 8 == 0 && bit_len >= 8 {
                let byte_count = bit_len / 8;
                let start = src_bit / 8;
                let end = start + byte_count;
                self.emit_bytes(&src[start..end])?;
                src_bit += byte_count * 8;
                bit_len -= byte_count * 8;
                continue;
            }

            if bit_len >= 8 {
                let byte = read_unaligned_byte(src, src_bit);
                self.write_byte_bits(byte)?;
                src_bit += 8;
                bit_len -= 8;
                continue;
            }

            self.write_bit(bit_at(src, src_bit))?;
            src_bit += 1;
            bit_len -= 1;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), CozipDeflateError> {
        if self.used > 0 {
            self.emit_byte(self.byte)?;
            self.byte = 0;
            self.used = 0;
        }
        self.flush_out_buf()?;
        Ok(())
    }

    fn io_counters(&self) -> (u64, usize, u64) {
        (self.io_ns, self.io_calls, self.io_bytes)
    }
}

fn bit_at(bytes: &[u8], bit_pos: usize) -> u8 {
    let byte = bytes[bit_pos / 8];
    (byte >> (bit_pos % 8)) & 1
}

fn read_unaligned_byte(bytes: &[u8], bit_pos: usize) -> u8 {
    let byte_pos = bit_pos / 8;
    let shift = bit_pos % 8;
    if shift == 0 {
        return bytes[byte_pos];
    }

    let lo = bytes[byte_pos] >> shift;
    let hi = bytes[byte_pos + 1] << (8 - shift);
    lo | hi
}

fn build_huffman_decoder(lengths: &[u8]) -> Result<HuffmanDecoder, CozipDeflateError> {
    let mut counts = [0_u16; DEFLATE_MAX_HUFF_BITS + 1];
    for &len in lengths {
        let len_usize = usize::from(len);
        if len_usize > DEFLATE_MAX_HUFF_BITS {
            return Err(CozipDeflateError::InvalidFrame(
                "invalid huffman code length in deflate stream",
            ));
        }
        if len_usize > 0 {
            counts[len_usize] = counts[len_usize].saturating_add(1);
        }
    }

    let mut max_bits = 0_usize;
    let total_symbols = counts
        .iter()
        .skip(1)
        .fold(0_usize, |acc, &count| acc.saturating_add(count as usize));
    let mut symbols = vec![0_u16; total_symbols];
    let mut offsets = [0_usize; DEFLATE_MAX_HUFF_BITS + 1];
    for bits in 1..DEFLATE_MAX_HUFF_BITS {
        offsets[bits + 1] = offsets[bits].saturating_add(counts[bits] as usize);
    }

    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let len_usize = usize::from(len);
        max_bits = max_bits.max(len_usize);
        let slot = offsets[len_usize];
        if slot >= symbols.len() {
            return Err(CozipDeflateError::InvalidFrame(
                "huffman symbol table overflow",
            ));
        }
        symbols[slot] = u16::try_from(symbol).map_err(|_| CozipDeflateError::DataTooLarge)?;
        offsets[len_usize] = offsets[len_usize].saturating_add(1);
    }

    Ok(HuffmanDecoder {
        counts,
        symbols,
        max_bits,
    })
}

fn decode_huffman_symbol(
    cursor: &mut DeflateBitCursor<'_>,
    decoder: &HuffmanDecoder,
) -> Result<u16, CozipDeflateError> {
    if decoder.max_bits == 0 {
        return Err(CozipDeflateError::InvalidFrame("empty huffman table"));
    }

    // Canonical DEFLATE decode (LSB-first bit order), equivalent to zlib/puff style.
    let mut code = 0_u32;
    let mut first = 0_u32;
    let mut index = 0_usize;
    for len in 1..=decoder.max_bits {
        code |= u32::from(cursor.read_bit()?);
        let count = u32::from(decoder.counts[len]);
        if code >= first && code - first < count {
            let offset = index.saturating_add((code - first) as usize);
            let Some(&symbol) = decoder.symbols.get(offset) else {
                return Err(CozipDeflateError::InvalidFrame(
                    "huffman symbol lookup out of range",
                ));
            };
            return Ok(symbol);
        }
        index = index.saturating_add(count as usize);
        first = (first + count) << 1;
        code <<= 1;
    }

    Err(CozipDeflateError::InvalidFrame(
        "failed to decode huffman symbol",
    ))
}

fn fixed_huffman_trees() -> Result<(HuffmanDecoder, HuffmanDecoder), CozipDeflateError> {
    let mut litlen_lengths = vec![0_u8; 288];
    litlen_lengths[..=143].fill(8);
    litlen_lengths[144..=255].fill(9);
    litlen_lengths[256..=279].fill(7);
    litlen_lengths[280..=287].fill(8);
    let dist_lengths = vec![5_u8; 32];
    Ok((
        build_huffman_decoder(&litlen_lengths)?,
        build_huffman_decoder(&dist_lengths)?,
    ))
}

fn read_dynamic_huffman_trees(
    cursor: &mut DeflateBitCursor<'_>,
) -> Result<(HuffmanDecoder, HuffmanDecoder), CozipDeflateError> {
    let hlit =
        usize::try_from(cursor.read_bits(5)?).map_err(|_| CozipDeflateError::DataTooLarge)? + 257;
    let hdist =
        usize::try_from(cursor.read_bits(5)?).map_err(|_| CozipDeflateError::DataTooLarge)? + 1;
    let hclen =
        usize::try_from(cursor.read_bits(4)?).map_err(|_| CozipDeflateError::DataTooLarge)? + 4;

    let mut codelen_lengths = [0_u8; 19];
    for idx in 0..hclen {
        let symbol = CODELEN_ORDER[idx];
        codelen_lengths[symbol] =
            u8::try_from(cursor.read_bits(3)?).map_err(|_| CozipDeflateError::DataTooLarge)?;
    }

    let codelen_decoder = build_huffman_decoder(&codelen_lengths)?;
    let total = hlit
        .checked_add(hdist)
        .ok_or(CozipDeflateError::DataTooLarge)?;
    let mut lengths = Vec::with_capacity(total);

    while lengths.len() < total {
        let sym = decode_huffman_symbol(cursor, &codelen_decoder)?;
        match sym {
            0..=15 => lengths.push(sym as u8),
            16 => {
                let prev = *lengths.last().ok_or(CozipDeflateError::InvalidFrame(
                    "repeat code without previous length",
                ))?;
                let repeat = usize::try_from(cursor.read_bits(2)?)
                    .map_err(|_| CozipDeflateError::DataTooLarge)?
                    + 3;
                for _ in 0..repeat {
                    lengths.push(prev);
                }
            }
            17 => {
                let repeat = usize::try_from(cursor.read_bits(3)?)
                    .map_err(|_| CozipDeflateError::DataTooLarge)?
                    + 3;
                lengths.extend(std::iter::repeat(0_u8).take(repeat));
            }
            18 => {
                let repeat = usize::try_from(cursor.read_bits(7)?)
                    .map_err(|_| CozipDeflateError::DataTooLarge)?
                    + 11;
                lengths.extend(std::iter::repeat(0_u8).take(repeat));
            }
            _ => {
                return Err(CozipDeflateError::InvalidFrame(
                    "invalid code-length alphabet symbol",
                ));
            }
        }
        if lengths.len() > total {
            return Err(CozipDeflateError::InvalidFrame(
                "dynamic huffman length run overflow",
            ));
        }
    }

    let litlen_decoder = build_huffman_decoder(&lengths[..hlit])?;
    let dist_decoder = build_huffman_decoder(&lengths[hlit..])?;
    Ok((litlen_decoder, dist_decoder))
}

fn parse_huffman_block_payload(
    cursor: &mut DeflateBitCursor<'_>,
    litlen_decoder: &HuffmanDecoder,
    dist_decoder: &HuffmanDecoder,
) -> Result<(), CozipDeflateError> {
    loop {
        let sym = decode_huffman_symbol(cursor, litlen_decoder)?;
        if sym < 256 {
            continue;
        }
        if sym == 256 {
            return Ok(());
        }
        if !(257..=285).contains(&sym) {
            return Err(CozipDeflateError::InvalidFrame(
                "invalid literal/length symbol",
            ));
        }

        let len_index = usize::from(sym - 257);
        let extra_len = usize::from(LENGTH_EXTRA_BITS[len_index]);
        if extra_len > 0 {
            let _ = cursor.read_bits(extra_len)?;
        }

        let dist_sym = decode_huffman_symbol(cursor, dist_decoder)?;
        if dist_sym >= 30 {
            return Err(CozipDeflateError::InvalidFrame("invalid distance symbol"));
        }
        let extra_dist = usize::from(DIST_EXTRA_BITS[usize::from(dist_sym)]);
        if extra_dist > 0 {
            let _ = cursor.read_bits(extra_dist)?;
        }
    }
}

fn parse_deflate_stream_layout(stream: &[u8]) -> Result<DeflateStreamLayout, CozipDeflateError> {
    if stream.is_empty() {
        return Err(CozipDeflateError::InvalidFrame("empty deflate chunk"));
    }

    let mut cursor = DeflateBitCursor::new(stream);
    loop {
        let final_header_bit = cursor.bit_pos;
        let bfinal = cursor.read_bit()?;
        let btype = cursor.read_bits(2)? as u8;
        match btype {
            0 => {
                cursor.align_to_byte();
                let len = usize::try_from(cursor.read_bits(16)?)
                    .map_err(|_| CozipDeflateError::DataTooLarge)?;
                let nlen = usize::try_from(cursor.read_bits(16)?)
                    .map_err(|_| CozipDeflateError::DataTooLarge)?;
                if (len ^ 0xFFFF) != nlen {
                    return Err(CozipDeflateError::InvalidFrame(
                        "stored block LEN/NLEN mismatch",
                    ));
                }
                cursor.skip_bits(len.saturating_mul(8))?;
            }
            1 => {
                let (litlen_decoder, dist_decoder) = fixed_huffman_trees()?;
                parse_huffman_block_payload(&mut cursor, &litlen_decoder, &dist_decoder)?;
            }
            2 => {
                let (litlen_decoder, dist_decoder) = read_dynamic_huffman_trees(&mut cursor)?;
                parse_huffman_block_payload(&mut cursor, &litlen_decoder, &dist_decoder)?;
            }
            _ => {
                return Err(CozipDeflateError::InvalidFrame("invalid block type"));
            }
        }

        if bfinal == 1 {
            return Ok(DeflateStreamLayout {
                final_header_bit,
                end_bit: cursor.bit_pos,
            });
        }
    }
}

fn append_deflate_chunk_as_block_sequence<W: Write>(
    writer: &mut DeflateBitWriter<'_, W>,
    chunk: &[u8],
    is_final_chunk: bool,
) -> Result<(), CozipDeflateError> {
    let layout = parse_deflate_stream_layout(chunk)?;
    append_deflate_chunk_as_block_sequence_with_layout(writer, chunk, layout, is_final_chunk)
}

fn append_deflate_chunk_as_block_sequence_with_layout<W: Write>(
    writer: &mut DeflateBitWriter<'_, W>,
    chunk: &[u8],
    layout: DeflateStreamLayout,
    is_final_chunk: bool,
) -> Result<(), CozipDeflateError> {
    if layout.end_bit == 0 {
        return Ok(());
    }
    if layout.final_header_bit >= layout.end_bit {
        return Err(CozipDeflateError::InvalidFrame(
            "invalid deflate layout: final header bit out of range",
        ));
    }

    let desired_final = if is_final_chunk { 1_u8 } else { 0_u8 };
    write_chunk_bits_with_final_override(writer, chunk, layout, desired_final)
}

fn write_chunk_bits_with_final_override<W: Write>(
    writer: &mut DeflateBitWriter<'_, W>,
    chunk: &[u8],
    layout: DeflateStreamLayout,
    desired_final: u8,
) -> Result<(), CozipDeflateError> {
    if layout.end_bit == 0 {
        return Ok(());
    }
    if layout.final_header_bit >= layout.end_bit {
        return Err(CozipDeflateError::InvalidFrame(
            "invalid deflate layout: final header bit out of range",
        ));
    }
    let desired_final = desired_final & 1;
    let current_final = bit_at(chunk, layout.final_header_bit);
    if current_final == desired_final {
        writer.write_bits_from_slice(chunk, 0, layout.end_bit)?;
        return Ok(());
    }

    if layout.final_header_bit > 0 {
        writer.write_bits_from_slice(chunk, 0, layout.final_header_bit)?;
    }
    writer.write_bit(desired_final)?;
    let tail_start = layout.final_header_bit + 1;
    if tail_start < layout.end_bit {
        writer.write_bits_from_slice(chunk, tail_start, layout.end_bit - tail_start)?;
    }
    Ok(())
}

fn append_empty_stored_block_non_final<W: Write>(
    writer: &mut DeflateBitWriter<'_, W>,
) -> Result<(), CozipDeflateError> {
    // BFINAL=0, BTYPE=00 (stored), then byte-align and emit LEN=0, NLEN=0xffff.
    writer.write_bit(0)?;
    writer.write_bit(0)?;
    writer.write_bit(0)?;
    while writer.used != 0 {
        writer.write_bit(0)?;
    }
    writer.write_byte_bits(0x00)?;
    writer.write_byte_bits(0x00)?;
    writer.write_byte_bits(0xff)?;
    writer.write_byte_bits(0xff)?;
    Ok(())
}

fn append_bit_to_vec_lsb(bits: &mut Vec<u8>, bit_len: &mut usize, bit: u8) {
    let byte_index = *bit_len / 8;
    if byte_index == bits.len() {
        bits.push(0);
    }
    let bit_index = *bit_len % 8;
    if bit & 1 == 1 {
        bits[byte_index] |= 1_u8 << bit_index;
    }
    *bit_len += 1;
}

fn append_byte_to_vec_lsb(bits: &mut Vec<u8>, bit_len: &mut usize, value: u8) {
    for shift in 0..8 {
        append_bit_to_vec_lsb(bits, bit_len, (value >> shift) & 1);
    }
}

fn append_empty_stored_block_non_final_bits(bits: &mut Vec<u8>, bit_len: &mut usize) {
    append_bit_to_vec_lsb(bits, bit_len, 0);
    append_bit_to_vec_lsb(bits, bit_len, 0);
    append_bit_to_vec_lsb(bits, bit_len, 0);
    while *bit_len % 8 != 0 {
        append_bit_to_vec_lsb(bits, bit_len, 0);
    }
    append_byte_to_vec_lsb(bits, bit_len, 0x00);
    append_byte_to_vec_lsb(bits, bit_len, 0x00);
    append_byte_to_vec_lsb(bits, bit_len, 0xff);
    append_byte_to_vec_lsb(bits, bit_len, 0xff);
}

fn prepare_chunk_bits_for_non_final_stream(
    chunk: &mut Vec<u8>,
    layout: DeflateStreamLayout,
) -> Result<usize, CozipDeflateError> {
    if layout.end_bit == 0 {
        return Ok(0);
    }
    if layout.final_header_bit >= layout.end_bit {
        return Err(CozipDeflateError::InvalidFrame(
            "invalid deflate layout: final header bit out of range",
        ));
    }

    let required_bytes = layout.end_bit.div_ceil(8);
    if chunk.len() < required_bytes {
        return Err(CozipDeflateError::InvalidFrame(
            "chunk shorter than declared deflate layout",
        ));
    }
    chunk.truncate(required_bytes);

    let byte_index = layout.final_header_bit / 8;
    let bit_mask = 1_u8 << (layout.final_header_bit % 8);
    chunk[byte_index] &= !bit_mask;

    let mut end_bit = layout.end_bit;
    if end_bit % 8 != 0 {
        append_empty_stored_block_non_final_bits(chunk, &mut end_bit);
    }
    Ok(end_bit)
}

fn parse_deflate_stream_layouts_parallel(
    chunks: &[ChunkMember],
) -> Result<Vec<DeflateStreamLayout>, CozipDeflateError> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let mut layouts = vec![
        DeflateStreamLayout {
            final_header_bit: 0,
            end_bit: 0,
        };
        chunks.len()
    ];
    let mut missing = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        if let Some(layout) = chunk.layout {
            layouts[index] = layout;
        } else {
            missing.push(index);
        }
    }

    if missing.is_empty() {
        return Ok(layouts);
    }

    let workers = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
        .max(1)
        .min(missing.len());

    if workers <= 1 || missing.len() <= 1 {
        for index in missing {
            layouts[index] = parse_deflate_stream_layout(&chunks[index].compressed)?;
        }
        return Ok(layouts);
    }

    let span = missing.len().div_ceil(workers);
    let ranges: Vec<(usize, usize)> = (0..missing.len())
        .step_by(span)
        .map(|start| (start, (start + span).min(missing.len())))
        .collect();
    let part_count = ranges.len();
    let mut first_err: Option<CozipDeflateError> = None;

    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel::<
            Result<Vec<(usize, DeflateStreamLayout)>, CozipDeflateError>,
        >();

        for (start, end) in &ranges {
            let tx = tx.clone();
            let index_slice = &missing[*start..*end];
            scope.spawn(move || {
                let mut local = Vec::with_capacity(index_slice.len());
                for &index in index_slice {
                    match parse_deflate_stream_layout(&chunks[index].compressed) {
                        Ok(layout) => local.push((index, layout)),
                        Err(err) => {
                            let _ = tx.send(Err(err));
                            return;
                        }
                    }
                }
                let _ = tx.send(Ok(local));
            });
        }
        drop(tx);

        for _ in 0..part_count {
            let Ok(payload) = rx.recv() else {
                if first_err.is_none() {
                    first_err = Some(CozipDeflateError::Internal(
                        "layout worker channel unexpectedly closed",
                    ));
                }
                continue;
            };
            match payload {
                Ok(local) => {
                    for (index, layout) in local {
                        layouts[index] = layout;
                    }
                }
                Err(err) => {
                    if first_err.is_none() {
                        first_err = Some(err);
                    }
                }
            }
        }
    });

    if let Some(err) = first_err {
        return Err(err);
    }

    Ok(layouts)
}

fn read_chunk_from_stream<R: Read>(
    reader: &mut R,
    chunk_size: usize,
) -> Result<Option<Vec<u8>>, CozipDeflateError> {
    let mut buffer = vec![0_u8; chunk_size];
    let mut total = 0_usize;

    while total < chunk_size {
        let read = reader.read(&mut buffer[total..])?;
        if read == 0 {
            break;
        }
        total += read;
    }

    if total == 0 {
        return Ok(None);
    }

    buffer.truncate(total);
    Ok(Some(buffer))
}

fn compression_mode_from_level(level: u32) -> CompressionMode {
    match level.clamp(0, 9) {
        0..=3 => CompressionMode::Speed,
        4..=6 => CompressionMode::Balanced,
        _ => CompressionMode::Ratio,
    }
}

fn default_hybrid_options_for_stream(level: u32) -> HybridOptions {
    let mut options = HybridOptions::default();
    options.compression_level = level.clamp(0, 9);
    options.compression_mode = compression_mode_from_level(level);
    options
}

fn is_gpu_requested(options: &HybridOptions) -> bool {
    options.prefer_gpu
}

fn has_gpu_eligible_task(tasks: &[ChunkTask], options: &HybridOptions) -> bool {
    tasks
        .iter()
        .any(|task| task.raw.len() >= options.gpu_min_chunk_size)
}

pub fn deflate_compress_stream_hybrid_zip_compatible<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    level: u32,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    let options = default_hybrid_options_for_stream(level);
    let gpu_context = if is_gpu_requested(&options) {
        GpuAssist::new(&options).ok().map(Arc::new)
    } else {
        None
    };
    deflate_compress_stream_hybrid_zip_compatible_with_context(
        reader,
        writer,
        &options,
        gpu_context,
    )
}

fn deflate_compress_stream_hybrid_zip_compatible_with_context<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &HybridOptions,
    gpu_context: Option<Arc<GpuAssist>>,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    validate_options(options)?;
    deflate_compress_stream_hybrid_zip_compatible_continuous_with_context(
        reader,
        writer,
        options,
        gpu_context,
    )
}

fn deflate_compress_stream_hybrid_zip_compatible_continuous_with_context<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &HybridOptions,
    gpu_context: Option<Arc<GpuAssist>>,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    let total_start = Instant::now();
    let mut stats = DeflateCpuStreamStats::default();
    let mut input_crc = crc32fast::Hasher::new();
    let mut output = HashingCountWriter::new(writer);
    let mut bit_writer = DeflateBitWriter::new(&mut output);

    let queue_state = Arc::new((Mutex::new(StreamTaskQueueState::default()), Condvar::new()));
    let ready_state = Arc::new((
        Mutex::new(BTreeMap::<usize, StreamReadyChunk>::new()),
        Condvar::new(),
    ));
    let error = Arc::new(Mutex::new(None::<CozipDeflateError>));
    let counters = Arc::new(WorkerCounters::default());
    let total_tasks = Arc::new(AtomicUsize::new(0));

    let gpu_enabled = gpu_context.is_some() && is_gpu_requested(options);
    let cpu_workers = cpu_worker_count(gpu_enabled);
    let mut handles = Vec::new();

    for _ in 0..cpu_workers {
        let queue_ref = Arc::clone(&queue_state);
        let ready_ref = Arc::clone(&ready_state);
        let err_ref = Arc::clone(&error);
        let opts = options.clone();
        let counters_ref = Arc::clone(&counters);
        handles.push(std::thread::spawn(move || {
            compress_cpu_stream_worker_continuous(
                queue_ref,
                ready_ref,
                err_ref,
                &opts,
                Some(counters_ref),
            )
        }));
    }

    if let Some(gpu) = gpu_context.clone().filter(|_| gpu_enabled) {
        let queue_ref = Arc::clone(&queue_state);
        let ready_ref = Arc::clone(&ready_state);
        let err_ref = Arc::clone(&error);
        let opts = options.clone();
        let counters_ref = Arc::clone(&counters);
        let total_tasks_ref = Arc::clone(&total_tasks);
        handles.push(std::thread::spawn(move || {
            compress_gpu_stream_worker_continuous(
                queue_ref,
                ready_ref,
                err_ref,
                &opts,
                gpu,
                total_tasks_ref,
                Some(counters_ref),
            )
        }));
    }

    let mut reached_eof = false;
    let mut next_read_index = 0usize;
    let mut next_write_index = 0usize;
    let mut inflight_raw_bytes = 0usize;

    loop {
        if has_error(&error) {
            break;
        }

        let mut progressed = false;
        loop {
            if reached_eof {
                break;
            }
            let inflight_chunks = next_read_index.saturating_sub(next_write_index);
            stats.inflight_chunks_max = stats.inflight_chunks_max.max(inflight_chunks);
            if options.stream_max_inflight_chunks > 0
                && inflight_chunks >= options.stream_max_inflight_chunks
            {
                break;
            }
            if options.stream_max_inflight_bytes > 0
                && inflight_raw_bytes >= options.stream_max_inflight_bytes
            {
                break;
            }

            let Some(raw) = read_chunk_from_stream(reader, options.chunk_size)? else {
                reached_eof = true;
                let (queue_lock, queue_cv) = &*queue_state;
                let mut state = lock(queue_lock)?;
                state.closed = true;
                queue_cv.notify_all();
                break;
            };

            input_crc.update(&raw);
            stats.input_bytes = stats
                .input_bytes
                .saturating_add(u64::try_from(raw.len()).unwrap_or(u64::MAX));

            let task = ChunkTask {
                index: next_read_index,
                preferred_gpu: false,
                raw,
            };
            next_read_index = next_read_index.saturating_add(1);
            total_tasks.store(next_read_index, Ordering::Relaxed);
            inflight_raw_bytes = inflight_raw_bytes.saturating_add(task.raw.len());

            let (queue_lock, queue_cv) = &*queue_state;
            let mut state = lock(queue_lock)?;
            state.queued_bytes = state.queued_bytes.saturating_add(task.raw.len());
            state.queue.push_back(task);
            queue_cv.notify_one();
            progressed = true;
        }

        loop {
            let (next_ready, ready_len) = {
                let (ready_lock, _) = &*ready_state;
                let mut ready = lock(ready_lock)?;
                let ready_len = ready.len();
                (ready.remove(&next_write_index), ready_len)
            };
            stats.ready_chunks_max = stats.ready_chunks_max.max(ready_len);
            let Some(ready) = next_ready else {
                break;
            };
            progressed = true;

            let is_final_chunk = reached_eof && next_write_index + 1 == next_read_index;
            let io_before = bit_writer.io_counters();
            let write_start = Instant::now();
            if next_write_index > 0 && bit_writer.used != 0 {
                append_empty_stored_block_non_final(&mut bit_writer)?;
            }
            if let Some(non_final_end_bit) = ready.prepared_non_final_end_bit {
                if is_final_chunk {
                    write_chunk_bits_with_final_override(
                        &mut bit_writer,
                        &ready.chunk.compressed,
                        ready.layout,
                        1,
                    )?;
                } else {
                    bit_writer.write_bits_from_slice(
                        &ready.chunk.compressed,
                        0,
                        non_final_end_bit,
                    )?;
                }
            } else {
                append_deflate_chunk_as_block_sequence_with_layout(
                    &mut bit_writer,
                    &ready.chunk.compressed,
                    ready.layout,
                    is_final_chunk,
                )?;
            }
            let write_elapsed = elapsed_ms(write_start);
            let io_after = bit_writer.io_counters();
            accumulate_write_stage_metrics(&mut stats, write_elapsed, io_before, io_after);

            stats.chunk_count = stats.chunk_count.saturating_add(1);
            match ready.chunk.backend {
                ChunkBackend::Cpu => stats.cpu_chunks = stats.cpu_chunks.saturating_add(1),
                ChunkBackend::GpuAssisted => stats.gpu_chunks = stats.gpu_chunks.saturating_add(1),
            }
            inflight_raw_bytes = inflight_raw_bytes.saturating_sub(ready.chunk.raw_len as usize);
            next_write_index = next_write_index.saturating_add(1);
        }

        if reached_eof && next_write_index == next_read_index {
            break;
        }

        if !progressed {
            let wait_start = Instant::now();
            let (ready_lock, ready_cv) = &*ready_state;
            let guard = lock(ready_lock)?;
            let ready_len = guard.len();
            stats.ready_chunks_max = stats.ready_chunks_max.max(ready_len);
            let hol_wait = ready_len > 0;
            drop(wait_timeout_on_condvar(
                ready_cv,
                guard,
                Duration::from_millis(2),
            )?);
            let wait_ms = elapsed_ms(wait_start);
            stats.writer_wait_ms += wait_ms;
            stats.writer_wait_events = stats.writer_wait_events.saturating_add(1);
            if hol_wait {
                stats.writer_hol_wait_ms += wait_ms;
                stats.writer_hol_wait_events = stats.writer_hol_wait_events.saturating_add(1);
                stats.writer_hol_ready_sum = stats.writer_hol_ready_sum.saturating_add(ready_len);
                stats.writer_hol_ready_max = stats.writer_hol_ready_max.max(ready_len);
            }
        }
    }

    {
        let (queue_lock, queue_cv) = &*queue_state;
        let mut state = lock(queue_lock)?;
        state.closed = true;
        queue_cv.notify_all();
    }

    for handle in handles {
        let _ = handle.join();
    }

    if let Some(err) = lock(&error)?.take() {
        return Err(err);
    }

    if stats.chunk_count == 0 {
        let empty = deflate_compress_cpu(&[], options.compression_level)?;
        let io_before = bit_writer.io_counters();
        let write_start = Instant::now();
        append_deflate_chunk_as_block_sequence(&mut bit_writer, &empty, true)?;
        let write_elapsed = elapsed_ms(write_start);
        let io_after = bit_writer.io_counters();
        accumulate_write_stage_metrics(&mut stats, write_elapsed, io_before, io_after);
    }

    let counters = counters.snapshot();
    stats.cpu_worker_busy_ms += counters.cpu_busy_ms;
    stats.gpu_worker_busy_ms += counters.gpu_busy_ms;
    stats.cpu_queue_lock_wait_ms += counters.cpu_queue_lock_wait_ms;
    stats.gpu_queue_lock_wait_ms += counters.gpu_queue_lock_wait_ms;
    stats.cpu_wait_for_task_ms += counters.cpu_wait_for_task_ms;
    stats.gpu_wait_for_task_ms += counters.gpu_wait_for_task_ms;
    stats.cpu_worker_chunks += counters.cpu_chunks_done;
    stats.gpu_worker_chunks += counters.gpu_chunks_done;
    stats.cpu_steal_chunks += counters.cpu_steal_chunks;
    stats.gpu_batch_count += counters.gpu_batches;
    stats.cpu_no_task_events += counters.cpu_no_task_events;
    stats.gpu_no_task_events += counters.gpu_no_task_events;
    stats.cpu_yield_events += counters.cpu_yield_events;
    stats.gpu_yield_events += counters.gpu_yield_events;
    stats.initial_gpu_queue_chunks += counters.initial_gpu_queue_chunks;
    stats.gpu_steal_reserve_chunks += counters.gpu_steal_reserve_chunks;

    bit_writer.finish()?;
    stats.input_crc32 = input_crc.finalize();
    stats.output_bytes = output.written;
    stats.output_crc32 = output.hasher.finalize();
    stats.gpu_available = gpu_enabled;
    stats.compress_stage_ms = (elapsed_ms(total_start) - stats.write_stage_ms).max(0.0);
    Ok(stats)
}

fn stream_task_is_gpu_eligible(task: &ChunkTask, options: &HybridOptions) -> bool {
    task.raw.len() >= options.gpu_min_chunk_size
}

fn compress_cpu_stream_worker_continuous(
    queue_state: Arc<(Mutex<StreamTaskQueueState>, Condvar)>,
    ready_state: Arc<(Mutex<BTreeMap<usize, StreamReadyChunk>>, Condvar)>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    options: &HybridOptions,
    counters: Option<Arc<WorkerCounters>>,
) {
    loop {
        if has_error(&error) {
            break;
        }
        let task = {
            let (queue_lock, queue_cv) = &*queue_state;
            let mut state = match lock(queue_lock) {
                Ok(guard) => guard,
                Err(err) => {
                    set_error(&error, err);
                    break;
                }
            };
            loop {
                if let Some(task) = state.queue.pop_front() {
                    state.queued_bytes = state.queued_bytes.saturating_sub(task.raw.len());
                    break Some(task);
                }
                if state.closed {
                    break None;
                }
                if let Some(counters) = &counters {
                    counters.cpu_yield_events.fetch_add(1, Ordering::Relaxed);
                }
                let wait_start = Instant::now();
                state = match wait_on_condvar(queue_cv, state) {
                    Ok(guard) => guard,
                    Err(err) => {
                        set_error(&error, err);
                        return;
                    }
                };
                if let Some(counters) = &counters {
                    counters
                        .cpu_wait_for_task_ns
                        .fetch_add(wait_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            }
        };

        let Some(task) = task else {
            if let Some(counters) = &counters {
                counters.cpu_no_task_events.fetch_add(1, Ordering::Relaxed);
            }
            break;
        };

        let start = Instant::now();
        let encoded = match compress_chunk_cpu(task, options.compression_level) {
            Ok(value) => value,
            Err(err) => {
                set_error(&error, err);
                break;
            }
        };
        let Some(layout) = encoded.layout else {
            set_error(
                &error,
                CozipDeflateError::Internal("cpu chunk missing layout metadata"),
            );
            break;
        };

        if let Some(counters) = &counters {
            counters
                .cpu_busy_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            counters.cpu_chunks.fetch_add(1, Ordering::Relaxed);
        }

        let (ready_lock, ready_cv) = &*ready_state;
        let mut ready = match lock(ready_lock) {
            Ok(guard) => guard,
            Err(err) => {
                set_error(&error, err);
                break;
            }
        };
        if ready
            .insert(
                encoded.index,
                StreamReadyChunk {
                    chunk: encoded,
                    layout,
                    prepared_non_final_end_bit: None,
                },
            )
            .is_some()
        {
            set_error(
                &error,
                CozipDeflateError::Internal("duplicate compressed index"),
            );
            break;
        }
        ready_cv.notify_all();
    }
}

fn compress_gpu_stream_worker_continuous(
    queue_state: Arc<(Mutex<StreamTaskQueueState>, Condvar)>,
    ready_state: Arc<(Mutex<BTreeMap<usize, StreamReadyChunk>>, Condvar)>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    options: &HybridOptions,
    gpu: Arc<GpuAssist>,
    total_tasks: Arc<AtomicUsize>,
    counters: Option<Arc<WorkerCounters>>,
) {
    let batch_limit = options.gpu_batch_chunks.clamp(1, MAX_GPU_BATCH_CHUNKS);
    loop {
        if has_error(&error) {
            break;
        }
        let tasks = {
            let (queue_lock, queue_cv) = &*queue_state;
            let mut state = match lock(queue_lock) {
                Ok(guard) => guard,
                Err(err) => {
                    set_error(&error, err);
                    break;
                }
            };
            loop {
                let total = total_tasks.load(Ordering::Relaxed);
                if state.closed
                    && should_stop_gpu_tail_pop(
                        total,
                        state.queue.len(),
                        options.gpu_tail_stop_ratio,
                    )
                {
                    break Vec::new();
                }

                let mut tasks = Vec::with_capacity(batch_limit);
                while tasks.len() < batch_limit {
                    let pos = state
                        .queue
                        .iter()
                        .position(|task| stream_task_is_gpu_eligible(task, options));
                    if let Some(pos) = pos {
                        if let Some(task) = state.queue.remove(pos) {
                            state.queued_bytes = state.queued_bytes.saturating_sub(task.raw.len());
                            tasks.push(task);
                        }
                    } else {
                        break;
                    }
                }
                if !tasks.is_empty() {
                    break tasks;
                }
                if state.closed {
                    break Vec::new();
                }
                if let Some(counters) = &counters {
                    counters.gpu_yield_events.fetch_add(1, Ordering::Relaxed);
                }
                let wait_start = Instant::now();
                state = match wait_on_condvar(queue_cv, state) {
                    Ok(guard) => guard,
                    Err(err) => {
                        set_error(&error, err);
                        return;
                    }
                };
                if let Some(counters) = &counters {
                    counters
                        .gpu_wait_for_task_ns
                        .fetch_add(wait_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            }
        };

        if tasks.is_empty() {
            if let Some(counters) = &counters {
                counters.gpu_no_task_events.fetch_add(1, Ordering::Relaxed);
            }
            break;
        }

        let batch_start = Instant::now();
        let encoded_batch = match compress_chunk_gpu_batch(&tasks, options, &gpu) {
            Ok(values) => {
                if values.len() != tasks.len() {
                    set_error(
                        &error,
                        CozipDeflateError::Internal("gpu worker produced mismatched batch"),
                    );
                    break;
                }
                values
            }
            Err(_) => {
                let mut fallback = Vec::with_capacity(tasks.len());
                for task in tasks {
                    match compress_chunk_cpu(task, options.compression_level) {
                        Ok(value) => fallback.push(value),
                        Err(err) => {
                            set_error(&error, err);
                            return;
                        }
                    }
                }
                fallback
            }
        };

        if let Some(counters) = &counters {
            counters
                .gpu_busy_ns
                .fetch_add(batch_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            counters.gpu_batches.fetch_add(1, Ordering::Relaxed);
            let gpu_chunks = encoded_batch
                .iter()
                .filter(|chunk| chunk.backend == ChunkBackend::GpuAssisted)
                .count();
            counters.gpu_chunks.fetch_add(gpu_chunks, Ordering::Relaxed);
            let cpu_chunks = encoded_batch.len().saturating_sub(gpu_chunks);
            counters.cpu_chunks.fetch_add(cpu_chunks, Ordering::Relaxed);
        }

        let layouts = match parse_deflate_stream_layouts_parallel(&encoded_batch) {
            Ok(value) => value,
            Err(err) => {
                set_error(&error, err);
                return;
            }
        };
        let (ready_lock, ready_cv) = &*ready_state;
        let mut ready = match lock(ready_lock) {
            Ok(guard) => guard,
            Err(err) => {
                set_error(&error, err);
                break;
            }
        };
        for (mut encoded, layout) in encoded_batch.into_iter().zip(layouts.into_iter()) {
            let prepared_non_final_end_bit = if encoded.backend == ChunkBackend::GpuAssisted {
                match prepare_chunk_bits_for_non_final_stream(&mut encoded.compressed, layout) {
                    Ok(value) => Some(value),
                    Err(err) => {
                        set_error(&error, err);
                        return;
                    }
                }
            } else {
                None
            };
            if ready
                .insert(
                    encoded.index,
                    StreamReadyChunk {
                        chunk: encoded,
                        layout,
                        prepared_non_final_end_bit,
                    },
                )
                .is_some()
            {
                set_error(
                    &error,
                    CozipDeflateError::Internal("duplicate compressed index"),
                );
                return;
            }
        }
        ready_cv.notify_all();
    }
}

struct HashingCountWriter<'a, W: Write> {
    inner: &'a mut W,
    hasher: crc32fast::Hasher,
    written: u64,
}

impl<'a, W: Write> HashingCountWriter<'a, W> {
    fn new(inner: &'a mut W) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
            written: 0,
        }
    }
}

impl<W: Write> Write for HashingCountWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf)?;
        if written > 0 {
            self.hasher.update(&buf[..written]);
            self.written = self
                .written
                .saturating_add(u64::try_from(written).unwrap_or(u64::MAX));
        }
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
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
    maybe_warn_deep_profile_enabled(options);
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

    let gpu_requested = is_gpu_requested(options);
    let gpu_available = gpu_context.is_some();
    let tasks = build_chunk_tasks(input, options, gpu_requested && gpu_available)?;
    let has_gpu_tasks = gpu_available && gpu_requested && has_gpu_eligible_task(&tasks, options);
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

        handles.push(std::thread::spawn(move || {
            compress_cpu_worker(queue_ref, result_ref, err_ref, &opts, None)
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

    let total_task_count = chunk_count;
    for _ in 0..cpu_workers {
        let queue_ref = Arc::clone(&queue);
        let result_ref = Arc::clone(&results);
        let err_ref = Arc::clone(&error);
        let opts = options.clone();
        handles.push(std::thread::spawn(move || {
            compress_cpu_worker_global_local(queue_ref, result_ref, err_ref, &opts, None)
        }));
    }

    if let Some(gpu) = gpu_context {
        let queue_ref = Arc::clone(&queue);
        let result_ref = Arc::clone(&results);
        let err_ref = Arc::clone(&error);
        let opts = options.clone();
        handles.push(std::thread::spawn(move || {
            compress_gpu_worker_global_local(
                queue_ref,
                result_ref,
                err_ref,
                &opts,
                gpu,
                total_task_count,
                None,
            )
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

pub fn compress_stream<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &HybridOptions,
    stream_options: StreamOptions,
) -> Result<StreamStats, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip.compress_stream(reader, writer, stream_options)
}

pub fn decompress_stream<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &HybridOptions,
) -> Result<StreamStats, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip.decompress_stream(reader, writer)
}

pub fn compress_file(
    input_file: StdFile,
    output_file: StdFile,
    options: &HybridOptions,
    stream_options: StreamOptions,
) -> Result<StreamStats, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip.compress_file(input_file, output_file, stream_options)
}

pub fn decompress_file(
    input_file: StdFile,
    output_file: StdFile,
    options: &HybridOptions,
) -> Result<StreamStats, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip.decompress_file(input_file, output_file)
}

pub fn compress_file_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
    input_path: PIn,
    output_path: POut,
    options: &HybridOptions,
    stream_options: StreamOptions,
) -> Result<StreamStats, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip.compress_file_from_name(input_path, output_path, stream_options)
}

pub fn decompress_file_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
    input_path: PIn,
    output_path: POut,
    options: &HybridOptions,
) -> Result<StreamStats, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip.decompress_file_from_name(input_path, output_path)
}

pub async fn compress_file_async(
    input_file: tokio::fs::File,
    output_file: tokio::fs::File,
    options: &HybridOptions,
    stream_options: StreamOptions,
) -> Result<StreamStats, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip
        .compress_file_async(input_file, output_file, stream_options)
        .await
}

pub async fn decompress_file_async(
    input_file: tokio::fs::File,
    output_file: tokio::fs::File,
    options: &HybridOptions,
) -> Result<StreamStats, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip.decompress_file_async(input_file, output_file).await
}

pub async fn compress_file_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
    input_path: PIn,
    output_path: POut,
    options: &HybridOptions,
    stream_options: StreamOptions,
) -> Result<StreamStats, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip
        .compress_file_from_name_async(input_path, output_path, stream_options)
        .await
}

pub async fn decompress_file_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
    input_path: PIn,
    output_path: POut,
    options: &HybridOptions,
) -> Result<StreamStats, CozipDeflateError> {
    let cozip = CoZipDeflate::init(options.clone())?;
    cozip
        .decompress_file_from_name_async(input_path, output_path)
        .await
}

fn stream_header_bytes() -> [u8; STREAM_HEADER_LEN] {
    let mut header = [0_u8; STREAM_HEADER_LEN];
    header[0..4].copy_from_slice(&STREAM_MAGIC);
    header[4] = STREAM_VERSION;
    header
}

fn validate_stream_header(header: &[u8; STREAM_HEADER_LEN]) -> Result<(), CozipDeflateError> {
    if &header[0..4] != STREAM_MAGIC.as_slice() {
        return Err(CozipDeflateError::InvalidFrame("invalid stream magic"));
    }
    if header[4] != STREAM_VERSION {
        return Err(CozipDeflateError::InvalidFrame(
            "unsupported stream version",
        ));
    }
    Ok(())
}

fn read_exact_frame<R: Read>(reader: &mut R, out: &mut [u8]) -> Result<(), CozipDeflateError> {
    let mut offset = 0usize;
    while offset < out.len() {
        match reader.read(&mut out[offset..]) {
            Ok(0) => return Err(CozipDeflateError::InvalidFrame("truncated stream")),
            Ok(read) => {
                offset += read;
            }
            Err(err) => return Err(CozipDeflateError::Io(err)),
        }
    }
    Ok(())
}

async fn read_exact_frame_async<R>(reader: &mut R, out: &mut [u8]) -> Result<(), CozipDeflateError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;

    let mut offset = 0usize;
    while offset < out.len() {
        match reader.read(&mut out[offset..]).await {
            Ok(0) => return Err(CozipDeflateError::InvalidFrame("truncated stream")),
            Ok(read) => {
                offset += read;
            }
            Err(err) => return Err(CozipDeflateError::Io(err)),
        }
    }
    Ok(())
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

    // Keep CPU worker parallelism unchanged even in hybrid mode.
    // The GPU worker mostly blocks on GPU progress, and reducing CPU workers by one
    // can underutilize CPU on CPU-heavy workloads (especially when GPU share is low).
    let _ = has_gpu;
    available.max(1)
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
    if options.gpu_slot_count == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_slot_count must be greater than 0",
        ));
    }
    if options.gpu_batch_chunks == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_batch_chunks must be greater than 0",
        ));
    }
    if options.gpu_pipelined_submit_chunks == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_pipelined_submit_chunks must be greater than 0",
        ));
    }
    if options.token_finalize_segment_size == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "token_finalize_segment_size must be greater than 0",
        ));
    }
    if options.stream_prepare_pipeline_depth == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "stream_prepare_pipeline_depth must be greater than 0",
        ));
    }
    if options.stream_batch_chunks != 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "stream_batch_chunks must be 0 (legacy batch mode was removed)",
        ));
    }

    if !(0.0..=1.0).contains(&options.gpu_fraction) {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_fraction must be in range 0.0..=1.0",
        ));
    }
    if !(0.0..=1.0).contains(&options.gpu_tail_stop_ratio) {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_tail_stop_ratio must be in range 0.0..=1.0",
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

    mark_gpu_preference(&mut tasks, options, gpu_available)?;
    Ok(tasks)
}

fn mark_gpu_preference(
    tasks: &mut [ChunkTask],
    _options: &HybridOptions,
    _gpu_available: bool,
) -> Result<(), CozipDeflateError> {
    for task in tasks.iter_mut() {
        task.preferred_gpu = false;
    }

    Ok(())
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

fn compress_cpu_worker_global_local(
    queue: Arc<Mutex<VecDeque<ChunkTask>>>,
    results: Arc<Mutex<Vec<Option<ChunkMember>>>>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    options: &HybridOptions,
    counters: Option<Arc<WorkerCounters>>,
) {
    loop {
        if has_error(&error) {
            break;
        }

        let task = (|| -> Result<Option<ChunkTask>, CozipDeflateError> {
            let lock_start = Instant::now();
            let mut guard = lock(&queue)?;
            if let Some(counters) = &counters {
                counters
                    .cpu_queue_lock_wait_ns
                    .fetch_add(lock_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            Ok(guard.pop_front())
        })();
        let Some(task) = (match task {
            Ok(value) => value,
            Err(err) => {
                set_error(&error, err);
                break;
            }
        }) else {
            if let Some(counters) = &counters {
                counters.cpu_no_task_events.fetch_add(1, Ordering::Relaxed);
            }
            break;
        };

        let start = Instant::now();
        match compress_chunk_cpu(task, options.compression_level) {
            Ok(encoded) => {
                if let Some(counters) = &counters {
                    let nanos = start.elapsed().as_nanos() as u64;
                    counters.cpu_busy_ns.fetch_add(nanos, Ordering::Relaxed);
                    counters.cpu_chunks.fetch_add(1, Ordering::Relaxed);
                }
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

fn compress_gpu_worker_global_local(
    queue: Arc<Mutex<VecDeque<ChunkTask>>>,
    results: Arc<Mutex<Vec<Option<ChunkMember>>>>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    options: &HybridOptions,
    gpu: Arc<GpuAssist>,
    total_task_count: usize,
    counters: Option<Arc<WorkerCounters>>,
) {
    let batch_limit = options.gpu_batch_chunks.clamp(1, MAX_GPU_BATCH_CHUNKS);
    let mut local_batch = Vec::with_capacity(batch_limit);

    loop {
        if has_error(&error) {
            break;
        }

        if local_batch.is_empty() {
            let refill = (|| -> Result<(), CozipDeflateError> {
                let lock_start = Instant::now();
                let mut guard = lock(&queue)?;
                if let Some(counters) = &counters {
                    counters
                        .gpu_queue_lock_wait_ns
                        .fetch_add(lock_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                if should_stop_gpu_tail_pop(
                    total_task_count,
                    guard.len(),
                    options.gpu_tail_stop_ratio,
                ) {
                    return Ok(());
                }
                local_batch = pop_front_batch_tasks(&mut guard, batch_limit);
                Ok(())
            })();
            if let Err(err) = refill {
                set_error(&error, err);
                break;
            }
        }

        if local_batch.is_empty() {
            if let Some(counters) = &counters {
                counters.gpu_no_task_events.fetch_add(1, Ordering::Relaxed);
            }
            break;
        }

        let tasks = std::mem::take(&mut local_batch);
        let batch_start = Instant::now();
        let encoded_batch = match compress_chunk_gpu_batch(&tasks, options, &gpu) {
            Ok(value) => value,
            Err(_) => {
                let mut fallback = Vec::with_capacity(tasks.len());
                for task in tasks {
                    match compress_chunk_cpu(task, options.compression_level) {
                        Ok(value) => fallback.push(value),
                        Err(err) => {
                            set_error(&error, err);
                            return;
                        }
                    }
                }
                fallback
            }
        };

        if let Some(counters) = &counters {
            let nanos = batch_start.elapsed().as_nanos() as u64;
            counters.gpu_busy_ns.fetch_add(nanos, Ordering::Relaxed);
            counters.gpu_batches.fetch_add(1, Ordering::Relaxed);
            let gpu_chunks = encoded_batch
                .iter()
                .filter(|chunk| chunk.backend == ChunkBackend::GpuAssisted)
                .count();
            counters.gpu_chunks.fetch_add(gpu_chunks, Ordering::Relaxed);
            let cpu_chunks = encoded_batch.len().saturating_sub(gpu_chunks);
            counters.cpu_chunks.fetch_add(cpu_chunks, Ordering::Relaxed);
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

fn should_stop_gpu_tail_pop(total_task_count: usize, queue_remaining: usize, ratio: f32) -> bool {
    if ratio >= 1.0 || total_task_count == 0 {
        return false;
    }
    let completed = total_task_count.saturating_sub(queue_remaining);
    (completed as f32 / total_task_count as f32) >= ratio
}

fn compress_cpu_worker(
    queue: Arc<Mutex<VecDeque<ChunkTask>>>,
    results: Arc<Mutex<Vec<Option<ChunkMember>>>>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    options: &HybridOptions,
    counters: Option<Arc<WorkerCounters>>,
) {
    loop {
        if has_error(&error) {
            break;
        }

        let task = {
            let lock_start = Instant::now();
            let mut guard = match lock(&queue) {
                Ok(value) => value,
                Err(err) => {
                    set_error(&error, err);
                    break;
                }
            };
            if let Some(counters) = &counters {
                counters
                    .cpu_queue_lock_wait_ns
                    .fetch_add(lock_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            guard.pop_front()
        };

        let Some(task) = task else {
            if let Some(counters) = &counters {
                counters.cpu_no_task_events.fetch_add(1, Ordering::Relaxed);
            }
            break;
        };

        let start = Instant::now();
        match compress_chunk_cpu(task, options.compression_level) {
            Ok(encoded) => {
                if let Some(counters) = &counters {
                    let nanos = start.elapsed().as_nanos() as u64;
                    counters.cpu_busy_ns.fetch_add(nanos, Ordering::Relaxed);
                    counters.cpu_chunks.fetch_add(1, Ordering::Relaxed);
                }
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

fn pop_front_batch_tasks(
    queue: &mut VecDeque<ChunkTask>,
    max_batch_chunks: usize,
) -> Vec<ChunkTask> {
    let mut tasks = Vec::with_capacity(max_batch_chunks.max(1));
    let batch_limit = max_batch_chunks.max(1);
    while tasks.len() < batch_limit {
        let Some(task) = queue.pop_front() else { break };
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
    for (task, encoded) in tasks.iter().zip(compressed_batch.into_iter()) {
        let raw_len = u32::try_from(task.raw.len()).map_err(|_| CozipDeflateError::DataTooLarge)?;
        let mut backend = if encoded.used_gpu {
            ChunkBackend::GpuAssisted
        } else {
            ChunkBackend::Cpu
        };
        let mut codec = if encoded.used_gpu {
            ChunkCodec::DeflateGpuFast
        } else {
            ChunkCodec::DeflateCpu
        };
        let mut compressed = encoded.compressed;
        let mut layout = encoded.end_bit.map(|end_bit| DeflateStreamLayout {
            final_header_bit: 0,
            end_bit,
        });

        if backend == ChunkBackend::GpuAssisted
            && should_validate_gpu_chunk(options, task.index)
            && !gpu_chunk_roundtrip_matches(&task.raw, &compressed)
        {
            compressed = deflate_compress_cpu(&task.raw, options.compression_level)?;
            backend = ChunkBackend::Cpu;
            codec = ChunkCodec::DeflateCpu;
            layout = Some(parse_deflate_stream_layout(&compressed)?);
        }

        let member = ChunkMember {
            index: task.index,
            backend,
            transform: ChunkTransform::None,
            codec,
            raw_len,
            layout,
            compressed,
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

fn pop_cpu_descriptor(queue: &mut VecDeque<ChunkDescriptor>) -> Option<ChunkDescriptor> {
    queue.pop_front()
}

fn compress_chunk_cpu(
    task: ChunkTask,
    compression_level: u32,
) -> Result<ChunkMember, CozipDeflateError> {
    let compressed = deflate_compress_cpu(&task.raw, compression_level)?;
    let layout = parse_deflate_stream_layout(&compressed)?;
    Ok(ChunkMember {
        index: task.index,
        backend: ChunkBackend::Cpu,
        transform: ChunkTransform::None,
        codec: ChunkCodec::DeflateCpu,
        raw_len: u32::try_from(task.raw.len()).map_err(|_| CozipDeflateError::DataTooLarge)?,
        layout: Some(layout),
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
    dyn_table: Vec<u32>,
    header_bytes_padded: Vec<u8>,
    header_copy_size: u64,
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
    let header_words = header_bytes.len().div_ceil(std::mem::size_of::<u32>());
    let header_copy_size = bytes_len::<u32>(header_words)?;
    let header_copy_size_usize =
        usize::try_from(header_copy_size).map_err(|_| CozipDeflateError::DataTooLarge)?;
    let mut header_bytes_padded = vec![0_u8; header_copy_size_usize];
    header_bytes_padded[..header_bytes.len()].copy_from_slice(&header_bytes);

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
    let mut dyn_table = Vec::with_capacity(DYN_TABLE_U32_COUNT);
    dyn_table.extend_from_slice(&litlen_codes);
    dyn_table.extend_from_slice(&litlen_bits);
    dyn_table.extend_from_slice(&dist_codes);
    dyn_table.extend_from_slice(&dist_bits);

    let eob_code = litlen_codes[256] as u16;
    let eob_bits = litlen_bits[256] as u8;
    if eob_bits == 0 {
        return Err(CozipDeflateError::Internal("missing end-of-block code"));
    }

    Ok(DynamicHuffmanPlan {
        dyn_table,
        header_bytes_padded,
        header_copy_size,
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
