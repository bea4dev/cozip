use std::collections::VecDeque;
use std::io::{Read, Write};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use thiserror::Error;

mod gpu;

const GDEFLATE_CODEC_ID: u8 = 4;
const GDEFLATE_TILE_SIZE: usize = 64 * 1024;
const GDEFLATE_MAX_TILES: usize = (1 << 16) - 1;
const GDEFLATE_NUM_STREAMS: usize = 32;
const GDEFLATE_STREAM_WORD_BYTES: usize = 4;
const GDEFLATE_TRAILING_PAD_BYTES: usize = GDEFLATE_NUM_STREAMS * GDEFLATE_STREAM_WORD_BYTES;
const TILE_STREAM_HEADER_SIZE: usize = 8;
const STATIC_LITERAL_MAX_BITS: usize = 9;
const DEFLATE_WINDOW_SIZE: usize = 32 * 1024;
const LZ_MIN_MATCH: usize = 3;
const LZ_MAX_MATCH: usize = 258;
const LZ_HASH_BITS: usize = 15;
const LZ_HASH_SIZE: usize = 1 << LZ_HASH_BITS;
const LZ_MAX_CHAIN: usize = 64;

const LENGTH_BASE: [usize; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [usize; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CODELEN_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GDeflateCompressionMode {
    TryAll,
    StoredOnly,
    StaticHuffman,
    DynamicHuffman,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GDeflateSchedulerPolicy {
    GlobalQueueLocalBuffers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GDeflateTileStreamHeader {
    pub id: u8,
    pub magic: u8,
    pub num_tiles: u16,
    pub tile_size_idx: u8,
    pub last_tile_size: u32,
    pub reserved: u16,
}

impl GDeflateTileStreamHeader {
    pub fn is_valid(&self) -> bool {
        self.id == (self.magic ^ 0xff) && self.tile_size_idx == 1
    }

    pub fn uncompressed_size(&self) -> usize {
        let tiles = usize::from(self.num_tiles);
        if tiles == 0 {
            return 0;
        }
        let base = tiles.saturating_mul(GDEFLATE_TILE_SIZE);
        if self.last_tile_size == 0 {
            base
        } else {
            base.saturating_sub(GDEFLATE_TILE_SIZE.saturating_sub(self.last_tile_size as usize))
        }
    }
}

#[derive(Debug, Clone)]
pub struct GDeflateOptions {
    pub tile_size: usize,
    pub compression_mode: GDeflateCompressionMode,
    // 0 => auto (available_parallelism()).
    pub cpu_worker_count: usize,
    // Keep single policy for now; shaped for future CPU+GPU shared queue workers.
    pub scheduler_policy: GDeflateSchedulerPolicy,
    // Enable GPU compression path (current stage supports StoredOnly and StaticHuffman modes).
    pub gpu_compress_enabled: bool,
    // Max in-flight GPU batches managed by a single GPU manager worker.
    pub gpu_compress_workers: usize,
    // Micro-batch submit size for GPU compression.
    pub gpu_compress_submit_tiles: usize,
    // Enable GPU static-Huffman decompression path.
    pub gpu_decompress_enabled: bool,
    // Max in-flight GPU decode batches managed by a single GPU manager worker.
    pub gpu_decompress_workers: usize,
    // Micro-batch submit size for GPU decode.
    pub gpu_decompress_submit_tiles: usize,
    // Super-batch multiplier for GPU decode submit to reduce submit frequency.
    pub gpu_decompress_super_batch_factor: usize,
}

impl Default for GDeflateOptions {
    fn default() -> Self {
        Self {
            tile_size: GDEFLATE_TILE_SIZE,
            compression_mode: GDeflateCompressionMode::TryAll,
            cpu_worker_count: 0,
            scheduler_policy: GDeflateSchedulerPolicy::GlobalQueueLocalBuffers,
            gpu_compress_enabled: false,
            gpu_compress_workers: 1,
            gpu_compress_submit_tiles: 8,
            gpu_decompress_enabled: false,
            gpu_decompress_workers: 4,
            gpu_decompress_submit_tiles: 64,
            gpu_decompress_super_batch_factor: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GDeflateStats {
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub tile_count: usize,
}

#[derive(Debug, Error)]
pub enum GDeflateError {
    #[error("invalid options: {0}")]
    InvalidOptions(&'static str),
    #[error("invalid stream: {0}")]
    InvalidStream(&'static str),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("data too large")]
    DataTooLarge,
    #[error("gpu error: {0}")]
    Gpu(String),
}

#[derive(Debug, Default, Clone)]
struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}

#[derive(Default)]
struct TaskQueueState {
    queue: VecDeque<usize>,
    closed: bool,
}

#[derive(Debug, Default, Clone, Copy)]
struct EncodeHybridStats {
    cpu_tiles: usize,
    gpu_tiles: usize,
    gpu_batches: usize,
    gpu_inflight_observed_max: usize,
    gpu_poll_ready_events: usize,
    gpu_poll_blocking_waits: usize,
    cpu_encode_ms: f64,
    gpu_upload_ms: f64,
    gpu_submit_wait_ms: f64,
    gpu_map_copy_ms: f64,
    gpu_repack_ms: f64,
    gpu_total_ms: f64,
    gpu_static_profiled_batches: usize,
    gpu_static_hash_reset_ms: f64,
    gpu_static_hash_build_ms: f64,
    gpu_static_match_ms: f64,
    gpu_static_scatter_ms: f64,
    gpu_static_serialize_ms: f64,
    gpu_static_copy_ms: f64,
    gpu_batch_submit_wait_min_ms: f64,
    gpu_batch_submit_wait_max_ms: f64,
    gpu_batch_map_copy_min_ms: f64,
    gpu_batch_map_copy_max_ms: f64,
    gpu_batch_repack_min_ms: f64,
    gpu_batch_repack_max_ms: f64,
    gpu_batch_total_min_ms: f64,
    gpu_batch_total_max_ms: f64,
    gpu_batch_tiles_min: usize,
    gpu_batch_tiles_max: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct DecodeHybridStats {
    cpu_tiles: usize,
    gpu_tiles: usize,
    gpu_batches: usize,
    gpu_fallback_tiles: usize,
    gpu_inflight_observed_max: usize,
    gpu_poll_ready_events: usize,
    gpu_poll_blocking_waits: usize,
    cpu_decode_ms: f64,
    gpu_upload_ms: f64,
    gpu_submit_wait_ms: f64,
    gpu_map_copy_ms: f64,
    gpu_total_ms: f64,
    gpu_profiled_batches: usize,
    gpu_decode_kernel_ms: f64,
    gpu_decode_copy_ms: f64,
    gpu_exec_ms: f64,
    gpu_submit_overhead_ms: f64,
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

impl BitWriter {
    fn push_bits(&mut self, mut value: u32, bits: usize) {
        for _ in 0..bits {
            let bit = (value & 1) as u8;
            value >>= 1;
            let byte_idx = self.bit_pos / 8;
            let bit_idx = self.bit_pos % 8;
            if byte_idx == self.bytes.len() {
                self.bytes.push(0);
            }
            self.bytes[byte_idx] |= bit << bit_idx;
            self.bit_pos += 1;
        }
    }

    fn align_byte(&mut self) {
        let rem = self.bit_pos % 8;
        if rem != 0 {
            self.push_bits(0, 8 - rem);
        }
    }

    fn push_u8(&mut self, value: u8) {
        self.push_bits(u32::from(value), 8);
    }

    fn push_u16_le(&mut self, value: u16) {
        self.push_u8((value & 0xff) as u8);
        self.push_u8((value >> 8) as u8);
    }
}

fn validate_options(options: &GDeflateOptions) -> Result<(), GDeflateError> {
    if options.tile_size != GDEFLATE_TILE_SIZE {
        return Err(GDeflateError::InvalidOptions(
            "tile_size must be 65536 for GDeflate v1 tile stream",
        ));
    }
    if options.gpu_compress_submit_tiles == 0 {
        return Err(GDeflateError::InvalidOptions(
            "gpu_compress_submit_tiles must be greater than 0",
        ));
    }
    if options.gpu_compress_workers == 0 {
        return Err(GDeflateError::InvalidOptions(
            "gpu_compress_workers must be greater than 0",
        ));
    }
    if options.gpu_decompress_submit_tiles == 0 {
        return Err(GDeflateError::InvalidOptions(
            "gpu_decompress_submit_tiles must be greater than 0",
        ));
    }
    if options.gpu_decompress_super_batch_factor == 0 {
        return Err(GDeflateError::InvalidOptions(
            "gpu_decompress_super_batch_factor must be greater than 0",
        ));
    }
    if options.gpu_decompress_workers == 0 {
        return Err(GDeflateError::InvalidOptions(
            "gpu_decompress_workers must be greater than 0",
        ));
    }
    Ok(())
}

fn compute_cpu_worker_count(configured: usize, task_count: usize) -> usize {
    if task_count == 0 {
        return 1;
    }
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let base = if configured == 0 {
        available
    } else {
        configured
    };
    base.max(1).min(task_count)
}

fn pop_global_task(
    queue_state: &Arc<(Mutex<TaskQueueState>, Condvar)>,
) -> Result<Option<usize>, GDeflateError> {
    let (queue_lock, queue_cv) = &**queue_state;
    let mut state = queue_lock
        .lock()
        .map_err(|_| GDeflateError::InvalidStream("task queue lock poisoned"))?;
    loop {
        if let Some(task) = state.queue.pop_front() {
            return Ok(Some(task));
        }
        if state.closed {
            return Ok(None);
        }
        state = queue_cv
            .wait(state)
            .map_err(|_| GDeflateError::InvalidStream("task queue wait poisoned"))?;
    }
}

fn pop_global_tasks_batch(
    queue_state: &Arc<(Mutex<TaskQueueState>, Condvar)>,
    max_items: usize,
) -> Result<Vec<usize>, GDeflateError> {
    let (queue_lock, _) = &**queue_state;
    let mut state = queue_lock
        .lock()
        .map_err(|_| GDeflateError::InvalidStream("task queue lock poisoned"))?;
    let mut out = Vec::with_capacity(max_items);
    for _ in 0..max_items {
        let Some(task) = state.queue.pop_front() else {
            break;
        };
        out.push(task);
    }
    Ok(out)
}

fn encode_tile_by_mode(
    tile: &[u8],
    mode: GDeflateCompressionMode,
) -> Result<Vec<u8>, GDeflateError> {
    match mode {
        GDeflateCompressionMode::StoredOnly => Ok(encode_tile_stored(tile)),
        GDeflateCompressionMode::StaticHuffman => Ok(encode_tile_static_huffman(tile)),
        GDeflateCompressionMode::DynamicHuffman => encode_tile_dynamic_huffman(tile),
        GDeflateCompressionMode::TryAll => {
            let stored = encode_tile_stored(tile);
            let static_page = encode_tile_static_huffman(tile);
            let dynamic_page = encode_tile_dynamic_huffman(tile)?;
            let mut best = stored;
            if static_page.len() < best.len() {
                best = static_page;
            }
            if dynamic_page.len() < best.len() {
                best = dynamic_page;
            }
            Ok(best)
        }
    }
}

pub fn gdeflate_compress(
    input: &[u8],
    options: &GDeflateOptions,
) -> Result<Vec<u8>, GDeflateError> {
    // Hybrid execution (CPU + optional GPU worker) is handled in encode_tiles_parallel().
    gdeflate_compress_cpu(input, options)
}

pub fn gdeflate_compress_cpu(
    input: &[u8],
    options: &GDeflateOptions,
) -> Result<Vec<u8>, GDeflateError> {
    validate_options(options)?;
    let total_start = Instant::now();
    let tiles = split_tiles(input, options.tile_size)?;
    let num_tiles = tiles.len();
    if num_tiles > GDEFLATE_MAX_TILES {
        return Err(GDeflateError::DataTooLarge);
    }

    let tiles_start = Instant::now();
    let pages = encode_tiles_parallel(&tiles, options)?;
    let t_tiles_ms = elapsed_ms(tiles_start);

    let build_start = Instant::now();
    let out = build_stream_from_pages(input.len(), pages)?;
    let t_build_ms = elapsed_ms(build_start);

    if options.gpu_compress_enabled {
        println!(
            "[cozip_gdeflate][timing][compress-phases] mode={:?} tasks={} t_tiles_ms={:.3} t_build_stream_ms={:.3} t_total_ms={:.3}",
            options.compression_mode,
            num_tiles,
            t_tiles_ms,
            t_build_ms,
            elapsed_ms(total_start),
        );
    }
    Ok(out)
}

fn build_stream_from_pages(
    input_len: usize,
    pages: Vec<Vec<u8>>,
) -> Result<Vec<u8>, GDeflateError> {
    let num_tiles = pages.len();
    let mut offsets = vec![0_u32; num_tiles];
    let mut payload_size = 0usize;
    for (i, page) in pages.iter().enumerate() {
        if i > 0 {
            offsets[i] = u32::try_from(payload_size).map_err(|_| GDeflateError::DataTooLarge)?;
        }
        payload_size = payload_size
            .checked_add(page.len())
            .ok_or(GDeflateError::DataTooLarge)?;
    }
    if let Some(last) = pages.last() {
        offsets[0] = u32::try_from(last.len()).map_err(|_| GDeflateError::DataTooLarge)?;
    }

    let mut out = Vec::with_capacity(
        TILE_STREAM_HEADER_SIZE
            .saturating_add(offsets.len().saturating_mul(4))
            .saturating_add(payload_size),
    );
    let header = build_header(input_len, num_tiles)?;
    encode_header(&mut out, &header);
    for off in &offsets {
        out.extend_from_slice(&off.to_le_bytes());
    }
    for page in pages {
        out.extend_from_slice(&page);
    }
    Ok(out)
}

fn encode_tiles_parallel(
    tiles: &[&[u8]],
    options: &GDeflateOptions,
) -> Result<Vec<Vec<u8>>, GDeflateError> {
    let total_start = Instant::now();
    let task_count = tiles.len();
    if task_count == 0 {
        return Ok(Vec::new());
    }
    let cpu_worker_count = compute_cpu_worker_count(options.cpu_worker_count, task_count);
    let use_gpu_worker = options.gpu_compress_enabled
        && matches!(
            options.compression_mode,
            GDeflateCompressionMode::StoredOnly | GDeflateCompressionMode::StaticHuffman
        );
    let gpu_worker_count = if use_gpu_worker { 1 } else { 0 };
    let gpu_inflight_batches = if use_gpu_worker {
        options.gpu_compress_workers.max(1)
    } else {
        0
    };
    if cpu_worker_count <= 1 && !use_gpu_worker {
        let mut pages = Vec::with_capacity(task_count);
        for &tile in tiles {
            pages.push(encode_tile_by_mode(tile, options.compression_mode)?);
        }
        return Ok(pages);
    }

    match options.scheduler_policy {
        GDeflateSchedulerPolicy::GlobalQueueLocalBuffers => {}
    }

    let queue_state = Arc::new((Mutex::new(TaskQueueState::default()), Condvar::new()));
    {
        let (queue_lock, _) = &*queue_state;
        let mut state = queue_lock
            .lock()
            .map_err(|_| GDeflateError::InvalidStream("task queue lock poisoned"))?;
        state.queue.reserve(task_count);
        for index in 0..task_count {
            state.queue.push_back(index);
        }
        state.closed = true;
    }

    let results: Arc<Vec<Mutex<Option<Result<Vec<u8>, GDeflateError>>>>> =
        Arc::new((0..task_count).map(|_| Mutex::new(None)).collect());
    let stats = Arc::new(Mutex::new(EncodeHybridStats::default()));
    let cpu_last_done_ms = Arc::new(Mutex::new(0.0f64));
    let gpu_last_done_ms = Arc::new(Mutex::new(0.0f64));
    let mode = options.compression_mode;
    let gpu_submit_tiles = options.gpu_compress_submit_tiles.max(1);
    let gpu_mode_for_cap = if use_gpu_worker {
        Some(match mode {
            GDeflateCompressionMode::StoredOnly => gpu::GpuEncodeMode::Stored,
            GDeflateCompressionMode::StaticHuffman => gpu::GpuEncodeMode::Static,
            _ => {
                return Err(GDeflateError::InvalidStream(
                    "gpu worker mode is not supported in this stage",
                ));
            }
        })
    } else {
        None
    };
    let gpu_submit_tiles_cap = if let Some(m) = gpu_mode_for_cap {
        gpu::max_submit_tiles_for_mode(m)?
    } else {
        0
    };
    let gpu_submit_tiles_effective = if use_gpu_worker {
        gpu_submit_tiles.min(gpu_submit_tiles_cap).max(1)
    } else {
        0
    };
    std::thread::scope(|scope| -> Result<(), GDeflateError> {
        let mut handles = Vec::with_capacity(cpu_worker_count + gpu_worker_count);
        for _ in 0..cpu_worker_count {
            let queue_ref = Arc::clone(&queue_state);
            let results_ref = Arc::clone(&results);
            let stats_ref = Arc::clone(&stats);
            let cpu_done_ref = Arc::clone(&cpu_last_done_ms);
            handles.push(scope.spawn(move || -> Result<(), GDeflateError> {
                let mut local_tiles = 0usize;
                let mut local_ms = 0.0f64;
                loop {
                    let Some(task_index) = pop_global_task(&queue_ref)? else {
                        break;
                    };
                    let t0 = Instant::now();
                    let page_res = encode_tile_by_mode(tiles[task_index], mode);
                    local_ms += elapsed_ms(t0);
                    local_tiles += 1;
                    let mut slot = results_ref[task_index]
                        .lock()
                        .map_err(|_| GDeflateError::InvalidStream("result slot lock poisoned"))?;
                    *slot = Some(page_res);
                }
                let mut s = stats_ref
                    .lock()
                    .map_err(|_| GDeflateError::InvalidStream("hybrid stats lock poisoned"))?;
                s.cpu_tiles += local_tiles;
                s.cpu_encode_ms += local_ms;
                let done_ms = elapsed_ms(total_start);
                let mut cpu_done = cpu_done_ref
                    .lock()
                    .map_err(|_| GDeflateError::InvalidStream("cpu done lock poisoned"))?;
                if done_ms > *cpu_done {
                    *cpu_done = done_ms;
                }
                Ok(())
            }));
        }
        for _ in 0..gpu_worker_count {
            let queue_ref = Arc::clone(&queue_state);
            let results_ref = Arc::clone(&results);
            let stats_ref = Arc::clone(&stats);
            let gpu_done_ref = Arc::clone(&gpu_last_done_ms);
            handles.push(scope.spawn(move || -> Result<(), GDeflateError> {
                struct InflightBatch {
                    task_indices: Vec<usize>,
                    pending: gpu::PendingGpuEncodeBatch,
                }
                let mut inflight: VecDeque<InflightBatch> = VecDeque::new();
                let mut queue_drained = false;
                let mut local_inflight_observed_max = 0usize;
                let mut local_poll_ready_events = 0usize;
                let mut local_poll_blocking_waits = 0usize;
                loop {
                    while !queue_drained && inflight.len() < gpu_inflight_batches {
                        let batch = pop_global_tasks_batch(&queue_ref, gpu_submit_tiles_effective)?;
                        if batch.is_empty() {
                            queue_drained = true;
                            break;
                        }
                        let batch_tiles: Vec<&[u8]> = batch.iter().map(|&i| tiles[i]).collect();
                        let gpu_mode = match mode {
                            GDeflateCompressionMode::StoredOnly => gpu::GpuEncodeMode::Stored,
                            GDeflateCompressionMode::StaticHuffman => gpu::GpuEncodeMode::Static,
                            _ => {
                                return Err(GDeflateError::InvalidStream(
                                    "gpu worker mode is not supported in this stage",
                                ));
                            }
                        };
                        let pending = gpu::submit_encode_tiles_gpu(gpu_mode, &batch_tiles)?;
                        inflight.push_back(InflightBatch {
                            task_indices: batch,
                            pending,
                        });
                        local_inflight_observed_max =
                            local_inflight_observed_max.max(inflight.len());
                    }

                    if inflight.is_empty() {
                        if queue_drained {
                            break;
                        }
                        continue;
                    }

                    let mut done_idx: Option<usize> = None;
                    let mut done_pages: Vec<Vec<u8>> = Vec::new();
                    let mut done_stats = gpu::GpuBatchStats::default();
                    for idx in 0..inflight.len() {
                        let maybe_done = {
                            let entry = inflight.get_mut(idx).ok_or(
                                GDeflateError::InvalidStream("inflight gpu entry missing"),
                            )?;
                            gpu::poll_encode_tiles_gpu(&mut entry.pending, false)?
                        };
                        if let Some((pages, stats)) = maybe_done {
                            local_poll_ready_events = local_poll_ready_events.saturating_add(1);
                            done_idx = Some(idx);
                            done_pages = pages;
                            done_stats = stats;
                            break;
                        }
                    }
                    if done_idx.is_none() {
                        local_poll_blocking_waits = local_poll_blocking_waits.saturating_add(1);
                        let maybe_done = {
                            let entry = inflight.front_mut().ok_or(
                                GDeflateError::InvalidStream("inflight gpu entry missing"),
                            )?;
                            gpu::poll_encode_tiles_gpu(&mut entry.pending, true)?
                        };
                        let (pages, stats) = maybe_done.ok_or(GDeflateError::InvalidStream(
                            "blocking gpu poll returned no result",
                        ))?;
                        done_idx = Some(0);
                        done_pages = pages;
                        done_stats = stats;
                    }

                    let idx = done_idx.ok_or(GDeflateError::InvalidStream(
                        "missing completed inflight gpu batch",
                    ))?;
                    let done = inflight.remove(idx).ok_or(GDeflateError::InvalidStream(
                        "failed to remove completed inflight gpu batch",
                    ))?;
                    if done_pages.len() != done.task_indices.len() {
                        return Err(GDeflateError::Gpu(
                            "gpu batch result count mismatch".to_string(),
                        ));
                    }
                    {
                        let mut s = stats_ref.lock().map_err(|_| {
                            GDeflateError::InvalidStream("hybrid stats lock poisoned")
                        })?;
                        if s.gpu_batches == 0 {
                            s.gpu_batch_submit_wait_min_ms = done_stats.submit_wait_ms;
                            s.gpu_batch_submit_wait_max_ms = done_stats.submit_wait_ms;
                            s.gpu_batch_map_copy_min_ms = done_stats.map_copy_ms;
                            s.gpu_batch_map_copy_max_ms = done_stats.map_copy_ms;
                            s.gpu_batch_repack_min_ms = done_stats.repack_ms;
                            s.gpu_batch_repack_max_ms = done_stats.repack_ms;
                            s.gpu_batch_total_min_ms = done_stats.total_ms;
                            s.gpu_batch_total_max_ms = done_stats.total_ms;
                            s.gpu_batch_tiles_min = done_stats.tiles;
                            s.gpu_batch_tiles_max = done_stats.tiles;
                        } else {
                            s.gpu_batch_submit_wait_min_ms = s
                                .gpu_batch_submit_wait_min_ms
                                .min(done_stats.submit_wait_ms);
                            s.gpu_batch_submit_wait_max_ms = s
                                .gpu_batch_submit_wait_max_ms
                                .max(done_stats.submit_wait_ms);
                            s.gpu_batch_map_copy_min_ms =
                                s.gpu_batch_map_copy_min_ms.min(done_stats.map_copy_ms);
                            s.gpu_batch_map_copy_max_ms =
                                s.gpu_batch_map_copy_max_ms.max(done_stats.map_copy_ms);
                            s.gpu_batch_repack_min_ms =
                                s.gpu_batch_repack_min_ms.min(done_stats.repack_ms);
                            s.gpu_batch_repack_max_ms =
                                s.gpu_batch_repack_max_ms.max(done_stats.repack_ms);
                            s.gpu_batch_total_min_ms =
                                s.gpu_batch_total_min_ms.min(done_stats.total_ms);
                            s.gpu_batch_total_max_ms =
                                s.gpu_batch_total_max_ms.max(done_stats.total_ms);
                            s.gpu_batch_tiles_min = s.gpu_batch_tiles_min.min(done_stats.tiles);
                            s.gpu_batch_tiles_max = s.gpu_batch_tiles_max.max(done_stats.tiles);
                        }
                        s.gpu_batches += 1;
                        s.gpu_tiles += done_stats.tiles;
                        s.gpu_upload_ms += done_stats.upload_ms;
                        s.gpu_submit_wait_ms += done_stats.submit_wait_ms;
                        s.gpu_map_copy_ms += done_stats.map_copy_ms;
                        s.gpu_repack_ms += done_stats.repack_ms;
                        s.gpu_total_ms += done_stats.total_ms;
                        if done_stats.static_profiled {
                            s.gpu_static_profiled_batches =
                                s.gpu_static_profiled_batches.saturating_add(1);
                            s.gpu_static_hash_reset_ms += done_stats.static_hash_reset_ms;
                            s.gpu_static_hash_build_ms += done_stats.static_hash_build_ms;
                            s.gpu_static_match_ms += done_stats.static_match_ms;
                            s.gpu_static_scatter_ms += done_stats.static_scatter_ms;
                            s.gpu_static_serialize_ms += done_stats.static_serialize_ms;
                            s.gpu_static_copy_ms += done_stats.static_copy_ms;
                        }
                    }
                    for (task_index, page) in
                        done.task_indices.into_iter().zip(done_pages.into_iter())
                    {
                        let mut slot = results_ref[task_index].lock().map_err(|_| {
                            GDeflateError::InvalidStream("result slot lock poisoned")
                        })?;
                        *slot = Some(Ok(page));
                    }
                    if queue_drained && inflight.is_empty() {
                        break;
                    }
                }
                {
                    let mut s = stats_ref
                        .lock()
                        .map_err(|_| GDeflateError::InvalidStream("hybrid stats lock poisoned"))?;
                    s.gpu_inflight_observed_max =
                        s.gpu_inflight_observed_max.max(local_inflight_observed_max);
                    s.gpu_poll_ready_events = s
                        .gpu_poll_ready_events
                        .saturating_add(local_poll_ready_events);
                    s.gpu_poll_blocking_waits = s
                        .gpu_poll_blocking_waits
                        .saturating_add(local_poll_blocking_waits);
                }
                let done_ms = elapsed_ms(total_start);
                let mut gpu_done = gpu_done_ref
                    .lock()
                    .map_err(|_| GDeflateError::InvalidStream("gpu done lock poisoned"))?;
                if done_ms > *gpu_done {
                    *gpu_done = done_ms;
                }
                Ok(())
            }));
        }
        for handle in handles {
            let worker_res = handle
                .join()
                .map_err(|_| GDeflateError::InvalidStream("compression worker panicked"))?;
            worker_res?;
        }
        Ok(())
    })?;
    let t_workers_ms = elapsed_ms(total_start);

    let collect_start = Instant::now();
    let mut pages = Vec::with_capacity(task_count);
    for i in 0..task_count {
        let mut slot = results[i]
            .lock()
            .map_err(|_| GDeflateError::InvalidStream("result slot lock poisoned"))?;
        let page = slot.take().ok_or(GDeflateError::InvalidStream(
            "missing compression worker result",
        ))??;
        pages.push(page);
    }
    let t_collect_ms = elapsed_ms(collect_start);
    if options.gpu_compress_enabled {
        let s = stats
            .lock()
            .map_err(|_| GDeflateError::InvalidStream("hybrid stats lock poisoned"))?;
        let cpu_done_ms = *cpu_last_done_ms
            .lock()
            .map_err(|_| GDeflateError::InvalidStream("cpu done lock poisoned"))?;
        let gpu_done_ms = *gpu_last_done_ms
            .lock()
            .map_err(|_| GDeflateError::InvalidStream("gpu done lock poisoned"))?;
        let tail_wait_ms = if gpu_done_ms > cpu_done_ms {
            gpu_done_ms - cpu_done_ms
        } else {
            0.0
        };
        let cpu_rate_tiles_per_ms = if cpu_done_ms > 0.0 {
            s.cpu_tiles as f64 / cpu_done_ms
        } else {
            0.0
        };
        let gpu_rate_tiles_per_ms = if gpu_done_ms > 0.0 {
            s.gpu_tiles as f64 / gpu_done_ms
        } else {
            0.0
        };
        let total_rate = cpu_rate_tiles_per_ms + gpu_rate_tiles_per_ms;
        let gpu_share_pct = if total_rate > 0.0 {
            (gpu_rate_tiles_per_ms / total_rate) * 100.0
        } else {
            0.0
        };
        let gpu_batches_f = s.gpu_batches as f64;
        let gpu_batch_tiles_avg = if s.gpu_batches > 0 {
            s.gpu_tiles as f64 / gpu_batches_f
        } else {
            0.0
        };
        let gpu_batch_submit_wait_avg_ms = if s.gpu_batches > 0 {
            s.gpu_submit_wait_ms / gpu_batches_f
        } else {
            0.0
        };
        let gpu_batch_map_copy_avg_ms = if s.gpu_batches > 0 {
            s.gpu_map_copy_ms / gpu_batches_f
        } else {
            0.0
        };
        let gpu_batch_repack_avg_ms = if s.gpu_batches > 0 {
            s.gpu_repack_ms / gpu_batches_f
        } else {
            0.0
        };
        let gpu_batch_total_avg_ms = if s.gpu_batches > 0 {
            s.gpu_total_ms / gpu_batches_f
        } else {
            0.0
        };
        println!(
            "[cozip_gdeflate][timing][hybrid-encode] mode={:?} tasks={} cpu_workers={} gpu_worker={} gpu_inflight_max={} gpu_inflight_obs_max={} gpu_poll_ready_events={} gpu_poll_blocking_waits={} gpu_submit_tiles_req={} gpu_submit_tiles_cap={} gpu_submit_tiles_eff={} cpu_tiles={} gpu_tiles={} gpu_batches={} gpu_batch_tiles_avg={:.2} gpu_batch_tiles_min={} gpu_batch_tiles_max={} t_cpu_encode_ms={:.3} t_gpu_upload_ms={:.3} t_gpu_submit_wait_ms={:.3} t_gpu_map_copy_ms={:.3} t_gpu_repack_ms={:.3} t_gpu_total_ms={:.3} t_gpu_batch_submit_wait_avg_ms={:.3} t_gpu_batch_submit_wait_min_ms={:.3} t_gpu_batch_submit_wait_max_ms={:.3} t_gpu_batch_map_copy_avg_ms={:.3} t_gpu_batch_map_copy_min_ms={:.3} t_gpu_batch_map_copy_max_ms={:.3} t_gpu_batch_repack_avg_ms={:.3} t_gpu_batch_repack_min_ms={:.3} t_gpu_batch_repack_max_ms={:.3} t_gpu_batch_total_avg_ms={:.3} t_gpu_batch_total_min_ms={:.3} t_gpu_batch_total_max_ms={:.3} t_cpu_done_ms={:.3} t_gpu_done_ms={:.3} t_gpu_tail_wait_ms={:.3} r_cpu_tiles_per_ms={:.3} r_gpu_tiles_per_ms={:.3} gpu_share_pct={:.2} gpu_static_profiled_batches={} t_gpu_static_hash_reset_ms={:.3} t_gpu_static_hash_build_ms={:.3} t_gpu_static_match_ms={:.3} t_gpu_static_scatter_ms={:.3} t_gpu_static_serialize_ms={:.3} t_gpu_static_copy_ms={:.3} t_workers_ms={:.3} t_collect_ms={:.3} t_total_ms={:.3}",
            options.compression_mode,
            task_count,
            cpu_worker_count,
            gpu_worker_count,
            gpu_inflight_batches,
            s.gpu_inflight_observed_max,
            s.gpu_poll_ready_events,
            s.gpu_poll_blocking_waits,
            gpu_submit_tiles,
            gpu_submit_tiles_cap,
            gpu_submit_tiles_effective,
            s.cpu_tiles,
            s.gpu_tiles,
            s.gpu_batches,
            gpu_batch_tiles_avg,
            s.gpu_batch_tiles_min,
            s.gpu_batch_tiles_max,
            s.cpu_encode_ms,
            s.gpu_upload_ms,
            s.gpu_submit_wait_ms,
            s.gpu_map_copy_ms,
            s.gpu_repack_ms,
            s.gpu_total_ms,
            gpu_batch_submit_wait_avg_ms,
            s.gpu_batch_submit_wait_min_ms,
            s.gpu_batch_submit_wait_max_ms,
            gpu_batch_map_copy_avg_ms,
            s.gpu_batch_map_copy_min_ms,
            s.gpu_batch_map_copy_max_ms,
            gpu_batch_repack_avg_ms,
            s.gpu_batch_repack_min_ms,
            s.gpu_batch_repack_max_ms,
            gpu_batch_total_avg_ms,
            s.gpu_batch_total_min_ms,
            s.gpu_batch_total_max_ms,
            cpu_done_ms,
            gpu_done_ms,
            tail_wait_ms,
            cpu_rate_tiles_per_ms,
            gpu_rate_tiles_per_ms,
            gpu_share_pct,
            s.gpu_static_profiled_batches,
            s.gpu_static_hash_reset_ms,
            s.gpu_static_hash_build_ms,
            s.gpu_static_match_ms,
            s.gpu_static_scatter_ms,
            s.gpu_static_serialize_ms,
            s.gpu_static_copy_ms,
            t_workers_ms,
            t_collect_ms,
            elapsed_ms(total_start),
        );
    }
    Ok(pages)
}

#[derive(Debug, Clone, Copy)]
struct DecodeTileTask {
    page_off: usize,
    page_len: usize,
    expected_len: usize,
}

pub fn gdeflate_decompress(
    stream: &[u8],
    options: &GDeflateOptions,
) -> Result<Vec<u8>, GDeflateError> {
    validate_options(options)?;
    let (header, offsets, payload) = parse_stream_header_and_offsets(stream)?;
    let tile_count = usize::from(header.num_tiles);
    let mut tasks = Vec::with_capacity(tile_count);
    for tile_index in 0..tile_count {
        let page_off = if tile_index == 0 {
            0usize
        } else {
            usize::try_from(offsets[tile_index]).map_err(|_| GDeflateError::DataTooLarge)?
        };
        let page_len = if tile_index + 1 < tile_count {
            let next = usize::try_from(offsets[tile_index + 1])
                .map_err(|_| GDeflateError::DataTooLarge)?;
            next.checked_sub(page_off)
                .ok_or(GDeflateError::InvalidStream("non-monotonic tile offsets"))?
        } else {
            usize::try_from(offsets[0]).map_err(|_| GDeflateError::DataTooLarge)?
        };
        tasks.push(DecodeTileTask {
            page_off,
            page_len,
            expected_len: expected_tile_output_size(&header, tile_index),
        });
    }

    let tiles = decode_tiles_parallel(payload, &tasks, options)?;
    let mut out = Vec::with_capacity(header.uncompressed_size());
    for tile in tiles {
        out.extend_from_slice(&tile);
    }
    if out.len() != header.uncompressed_size() {
        return Err(GDeflateError::InvalidStream(
            "decoded size does not match tile stream header",
        ));
    }
    Ok(out)
}

pub fn gdeflate_decompress_cpu(stream: &[u8]) -> Result<Vec<u8>, GDeflateError> {
    gdeflate_decompress_cpu_with_options(stream, &GDeflateOptions::default())
}

pub fn gdeflate_decompress_cpu_with_options(
    stream: &[u8],
    options: &GDeflateOptions,
) -> Result<Vec<u8>, GDeflateError> {
    let mut cpu_only = options.clone();
    cpu_only.gpu_decompress_enabled = false;
    gdeflate_decompress(stream, &cpu_only)
}

fn decode_tiles_parallel(
    payload: &[u8],
    tasks: &[DecodeTileTask],
    options: &GDeflateOptions,
) -> Result<Vec<Vec<u8>>, GDeflateError> {
    let task_count = tasks.len();
    if task_count == 0 {
        return Ok(Vec::new());
    }
    let cpu_worker_count = compute_cpu_worker_count(options.cpu_worker_count, task_count);
    let use_gpu_worker = options.gpu_decompress_enabled;
    let gpu_worker_count = if use_gpu_worker { 1 } else { 0 };
    let gpu_inflight_batches = if use_gpu_worker {
        options.gpu_decompress_workers.max(1)
    } else {
        0
    };
    let gpu_submit_tiles = options.gpu_decompress_submit_tiles.max(1);
    let gpu_submit_tiles_cap = if use_gpu_worker {
        gpu::max_submit_tiles_for_decode_mode(gpu::GpuDecodeMode::Static)?
    } else {
        0
    };
    let gpu_submit_tiles_effective = if use_gpu_worker {
        gpu_submit_tiles.min(gpu_submit_tiles_cap).max(1)
    } else {
        0
    };
    let gpu_submit_tiles_super = if use_gpu_worker {
        gpu_submit_tiles_effective
            .saturating_mul(options.gpu_decompress_super_batch_factor.max(1))
            .min(gpu_submit_tiles_cap)
            .max(1)
    } else {
        0
    };

    if cpu_worker_count <= 1 && !use_gpu_worker {
        let mut tiles = Vec::with_capacity(task_count);
        for task in tasks {
            let page_end = task
                .page_off
                .checked_add(task.page_len)
                .ok_or(GDeflateError::DataTooLarge)?;
            let page = payload
                .get(task.page_off..page_end)
                .ok_or(GDeflateError::InvalidStream("tile page out of bounds"))?;
            let mut tile_out = Vec::with_capacity(task.expected_len);
            decode_tile(page, task.expected_len, &mut tile_out)?;
            tiles.push(tile_out);
        }
        return Ok(tiles);
    }

    match options.scheduler_policy {
        GDeflateSchedulerPolicy::GlobalQueueLocalBuffers => {}
    }

    let queue_state = Arc::new((Mutex::new(TaskQueueState::default()), Condvar::new()));
    {
        let (queue_lock, _) = &*queue_state;
        let mut state = queue_lock
            .lock()
            .map_err(|_| GDeflateError::InvalidStream("task queue lock poisoned"))?;
        state.queue.reserve(task_count);
        for index in 0..task_count {
            state.queue.push_back(index);
        }
        state.closed = true;
    }

    let results: Arc<Vec<Mutex<Option<Result<Vec<u8>, GDeflateError>>>>> =
        Arc::new((0..task_count).map(|_| Mutex::new(None)).collect());
    let stats = Arc::new(Mutex::new(DecodeHybridStats::default()));
    let cpu_last_done_ms = Arc::new(Mutex::new(0.0f64));
    let gpu_last_done_ms = Arc::new(Mutex::new(0.0f64));
    let total_start = Instant::now();

    std::thread::scope(|scope| -> Result<(), GDeflateError> {
        let mut handles = Vec::with_capacity(cpu_worker_count + gpu_worker_count);
        for _ in 0..cpu_worker_count {
            let queue_ref = Arc::clone(&queue_state);
            let results_ref = Arc::clone(&results);
            let stats_ref = Arc::clone(&stats);
            let cpu_done_ref = Arc::clone(&cpu_last_done_ms);
            handles.push(scope.spawn(move || -> Result<(), GDeflateError> {
                let mut local_tiles = 0usize;
                let mut local_ms = 0.0f64;
                loop {
                    let Some(task_index) = pop_global_task(&queue_ref)? else {
                        break;
                    };
                    let task = tasks[task_index];
                    let page_res = (|| -> Result<Vec<u8>, GDeflateError> {
                        let t0 = Instant::now();
                        let page_end = task
                            .page_off
                            .checked_add(task.page_len)
                            .ok_or(GDeflateError::DataTooLarge)?;
                        let page = payload
                            .get(task.page_off..page_end)
                            .ok_or(GDeflateError::InvalidStream("tile page out of bounds"))?;
                        let mut tile_out = Vec::with_capacity(task.expected_len);
                        decode_tile(page, task.expected_len, &mut tile_out)?;
                        local_ms += elapsed_ms(t0);
                        local_tiles += 1;
                        Ok(tile_out)
                    })();
                    let mut slot = results_ref[task_index]
                        .lock()
                        .map_err(|_| GDeflateError::InvalidStream("result slot lock poisoned"))?;
                    *slot = Some(page_res);
                }
                let mut s = stats_ref
                    .lock()
                    .map_err(|_| GDeflateError::InvalidStream("decode stats lock poisoned"))?;
                s.cpu_tiles += local_tiles;
                s.cpu_decode_ms += local_ms;
                let done_ms = elapsed_ms(total_start);
                let mut cpu_done = cpu_done_ref
                    .lock()
                    .map_err(|_| GDeflateError::InvalidStream("decode cpu done lock poisoned"))?;
                if done_ms > *cpu_done {
                    *cpu_done = done_ms;
                }
                Ok(())
            }));
        }
        for _ in 0..gpu_worker_count {
            let queue_ref = Arc::clone(&queue_state);
            let results_ref = Arc::clone(&results);
            let stats_ref = Arc::clone(&stats);
            let gpu_done_ref = Arc::clone(&gpu_last_done_ms);
            handles.push(scope.spawn(move || -> Result<(), GDeflateError> {
                struct InflightBatch {
                    task_indices: Vec<usize>,
                    pending: gpu::PendingGpuDecodeBatch,
                }
                let decode_debug = std::env::var("COZIP_GDEFLATE_GPU_DECODE_DEBUG")
                    .ok()
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let mut local_logged_gpu_errs = 0usize;
                let mut inflight: VecDeque<InflightBatch> = VecDeque::new();
                let mut queue_drained = false;
                let mut local_inflight_observed_max = 0usize;
                let mut local_poll_ready_events = 0usize;
                let mut local_poll_blocking_waits = 0usize;
                let mut local_cpu_fallback_tiles = 0usize;
                let mut local_cpu_fallback_ms = 0.0f64;

                loop {
                    while !queue_drained && inflight.len() < gpu_inflight_batches {
                        let batch = pop_global_tasks_batch(&queue_ref, gpu_submit_tiles_super)?;
                        if batch.is_empty() {
                            queue_drained = true;
                            break;
                        }

                        let mut gpu_task_indices = Vec::new();
                        let mut gpu_pages: Vec<&[u8]> = Vec::new();
                        let mut gpu_expected_lens = Vec::new();
                        for &task_index in &batch {
                            let task = tasks[task_index];
                            let page_end = task
                                .page_off
                                .checked_add(task.page_len)
                                .ok_or(GDeflateError::DataTooLarge)?;
                            let page = payload
                                .get(task.page_off..page_end)
                                .ok_or(GDeflateError::InvalidStream("tile page out of bounds"))?;
                            let first = *page.first().unwrap_or(&0);
                            let btype = (first >> 1) & 0b11;
                            if btype == 0b01 {
                                gpu_task_indices.push(task_index);
                                gpu_pages.push(page);
                                gpu_expected_lens.push(task.expected_len);
                            } else {
                                let t0 = Instant::now();
                                let mut tile_out = Vec::with_capacity(task.expected_len);
                                decode_tile(page, task.expected_len, &mut tile_out)?;
                                local_cpu_fallback_ms += elapsed_ms(t0);
                                local_cpu_fallback_tiles += 1;
                                let mut slot = results_ref[task_index].lock().map_err(|_| {
                                    GDeflateError::InvalidStream("result slot lock poisoned")
                                })?;
                                *slot = Some(Ok(tile_out));
                            }
                        }
                        if gpu_task_indices.is_empty() {
                            continue;
                        }
                        let pending = match gpu::submit_decode_tiles_gpu(
                            gpu::GpuDecodeMode::Static,
                            &gpu_pages,
                            &gpu_expected_lens,
                        ) {
                            Ok(p) => p,
                            Err(e) => {
                                if decode_debug && local_logged_gpu_errs < 4 {
                                    eprintln!(
                                        "[cozip_gdeflate][decode-debug] gpu submit failed: {}",
                                        e
                                    );
                                    local_logged_gpu_errs += 1;
                                }
                                for &task_index in &gpu_task_indices {
                                    let task = tasks[task_index];
                                    let page_end = task
                                        .page_off
                                        .checked_add(task.page_len)
                                        .ok_or(GDeflateError::DataTooLarge)?;
                                    let page = payload.get(task.page_off..page_end).ok_or(
                                        GDeflateError::InvalidStream("tile page out of bounds"),
                                    )?;
                                    let t0 = Instant::now();
                                    let mut tile_out = Vec::with_capacity(task.expected_len);
                                    decode_tile(page, task.expected_len, &mut tile_out)?;
                                    local_cpu_fallback_ms += elapsed_ms(t0);
                                    local_cpu_fallback_tiles += 1;
                                    let mut slot =
                                        results_ref[task_index].lock().map_err(|_| {
                                            GDeflateError::InvalidStream(
                                                "result slot lock poisoned",
                                            )
                                        })?;
                                    *slot = Some(Ok(tile_out));
                                }
                                continue;
                            }
                        };
                        inflight.push_back(InflightBatch {
                            task_indices: gpu_task_indices,
                            pending,
                        });
                        local_inflight_observed_max =
                            local_inflight_observed_max.max(inflight.len());
                    }

                    if inflight.is_empty() {
                        if queue_drained {
                            break;
                        }
                        continue;
                    }

                    let mut completed: Vec<(
                        usize,
                        Result<(Vec<Vec<u8>>, gpu::GpuDecodeBatchStats), GDeflateError>,
                    )> = Vec::new();

                    gpu::poll_runtime_device(false)?;
                    for idx in 0..inflight.len() {
                        let polled = {
                            let entry = inflight.get_mut(idx).ok_or(
                                GDeflateError::InvalidStream("inflight gpu decode entry missing"),
                            )?;
                            gpu::poll_decode_tiles_gpu_no_poll(&mut entry.pending, false)
                        };
                        match polled {
                            Ok(Some(done)) => {
                                local_poll_ready_events = local_poll_ready_events.saturating_add(1);
                                completed.push((idx, Ok(done)));
                            }
                            Ok(None) => {}
                            Err(e) => {
                                local_poll_ready_events = local_poll_ready_events.saturating_add(1);
                                completed.push((idx, Err(e)));
                            }
                        }
                    }

                    if completed.is_empty() {
                        local_poll_blocking_waits = local_poll_blocking_waits.saturating_add(1);
                        gpu::poll_runtime_device(true)?;
                        for idx in 0..inflight.len() {
                            let polled = {
                                let entry = inflight.get_mut(idx).ok_or(
                                    GDeflateError::InvalidStream(
                                        "inflight gpu decode entry missing",
                                    ),
                                )?;
                                gpu::poll_decode_tiles_gpu_no_poll(&mut entry.pending, false)
                            };
                            match polled {
                                Ok(Some(done)) => {
                                    local_poll_ready_events =
                                        local_poll_ready_events.saturating_add(1);
                                    completed.push((idx, Ok(done)));
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    local_poll_ready_events =
                                        local_poll_ready_events.saturating_add(1);
                                    completed.push((idx, Err(e)));
                                }
                            }
                        }
                    }

                    if completed.is_empty() {
                        let blocking_res = {
                            let entry = inflight.front_mut().ok_or(
                                GDeflateError::InvalidStream("inflight gpu decode entry missing"),
                            )?;
                            gpu::poll_decode_tiles_gpu_no_poll(&mut entry.pending, true)
                        };
                        match blocking_res {
                            Ok(Some(done)) => completed.push((0, Ok(done))),
                            Ok(None) => {
                                return Err(GDeflateError::InvalidStream(
                                    "blocking decode gpu poll returned no result",
                                ));
                            }
                            Err(e) => completed.push((0, Err(e))),
                        }
                    }

                    completed.sort_by_key(|(idx, _)| *idx);
                    for (idx, done_result) in completed.into_iter().rev() {
                        let done = inflight.remove(idx).ok_or(GDeflateError::InvalidStream(
                            "failed to remove completed inflight gpu decode batch",
                        ))?;

                        let (done_tiles, done_stats, done_err) = match done_result {
                            Ok((tiles, stats)) => (tiles, stats, None),
                            Err(e) => (Vec::new(), gpu::GpuDecodeBatchStats::default(), Some(e)),
                        };

                        if done_err.is_none() && done_tiles.len() == done.task_indices.len() {
                        {
                            let mut s = stats_ref.lock().map_err(|_| {
                                GDeflateError::InvalidStream("decode stats lock poisoned")
                            })?;
                            s.gpu_batches += 1;
                            s.gpu_tiles += done_stats.tiles;
                            s.gpu_upload_ms += done_stats.upload_ms;
                            s.gpu_submit_wait_ms += done_stats.submit_wait_ms;
                            s.gpu_map_copy_ms += done_stats.map_copy_ms;
                            s.gpu_total_ms += done_stats.total_ms;
                            if done_stats.static_profiled {
                                s.gpu_profiled_batches =
                                    s.gpu_profiled_batches.saturating_add(1);
                                s.gpu_decode_kernel_ms += done_stats.static_decode_ms;
                                s.gpu_decode_copy_ms += done_stats.static_copy_ms;
                                s.gpu_exec_ms += done_stats.gpu_exec_ms;
                                s.gpu_submit_overhead_ms += done_stats.submit_overhead_ms;
                            }
                        }
                        for (task_index, tile) in done
                            .task_indices
                            .iter()
                            .copied()
                            .zip(done_tiles.into_iter())
                        {
                            let mut final_tile = tile;
                            if decode_debug && local_logged_gpu_errs < 4 {
                                let task = tasks[task_index];
                                let page_end = task
                                    .page_off
                                    .checked_add(task.page_len)
                                    .ok_or(GDeflateError::DataTooLarge)?;
                                let page = payload.get(task.page_off..page_end).ok_or(
                                    GDeflateError::InvalidStream("tile page out of bounds"),
                                )?;
                                let mut cpu_tile = Vec::with_capacity(task.expected_len);
                                decode_tile(page, task.expected_len, &mut cpu_tile)?;
                                if cpu_tile != final_tile {
                                    eprintln!(
                                        "[cozip_gdeflate][decode-debug] gpu tile mismatch: task={} expected_len={}",
                                        task_index,
                                        task.expected_len
                                    );
                                    local_logged_gpu_errs += 1;
                                }
                            }
                            let mut slot = results_ref[task_index]
                                .lock()
                                .map_err(|_| GDeflateError::InvalidStream("result slot lock poisoned"))?;
                            *slot = Some(Ok(std::mem::take(&mut final_tile)));
                        }
                        } else {
                            if decode_debug && local_logged_gpu_errs < 4 {
                                if let Some(ref e) = done_err {
                                    eprintln!(
                                        "[cozip_gdeflate][decode-debug] gpu poll failed: {}",
                                        e
                                    );
                                    local_logged_gpu_errs += 1;
                                }
                            }
                            for task_index in done.task_indices {
                                let task = tasks[task_index];
                                let page_end = task
                                    .page_off
                                    .checked_add(task.page_len)
                                    .ok_or(GDeflateError::DataTooLarge)?;
                                let page = payload.get(task.page_off..page_end).ok_or(
                                    GDeflateError::InvalidStream("tile page out of bounds"),
                                )?;
                                let t0 = Instant::now();
                                let mut tile_out = Vec::with_capacity(task.expected_len);
                                decode_tile(page, task.expected_len, &mut tile_out)?;
                                local_cpu_fallback_ms += elapsed_ms(t0);
                                local_cpu_fallback_tiles += 1;
                                let mut slot = results_ref[task_index].lock().map_err(|_| {
                                    GDeflateError::InvalidStream("result slot lock poisoned")
                                })?;
                                *slot = Some(Ok(tile_out));
                            }
                        }
                    }

                    if queue_drained && inflight.is_empty() {
                        break;
                    }
                }

                {
                    let mut s = stats_ref
                        .lock()
                        .map_err(|_| GDeflateError::InvalidStream("decode stats lock poisoned"))?;
                    s.gpu_fallback_tiles += local_cpu_fallback_tiles;
                    s.cpu_tiles += local_cpu_fallback_tiles;
                    s.cpu_decode_ms += local_cpu_fallback_ms;
                    s.gpu_inflight_observed_max =
                        s.gpu_inflight_observed_max.max(local_inflight_observed_max);
                    s.gpu_poll_ready_events = s
                        .gpu_poll_ready_events
                        .saturating_add(local_poll_ready_events);
                    s.gpu_poll_blocking_waits = s
                        .gpu_poll_blocking_waits
                        .saturating_add(local_poll_blocking_waits);
                }
                let done_ms = elapsed_ms(total_start);
                let mut gpu_done = gpu_done_ref
                    .lock()
                    .map_err(|_| GDeflateError::InvalidStream("decode gpu done lock poisoned"))?;
                if done_ms > *gpu_done {
                    *gpu_done = done_ms;
                }
                Ok(())
            }));
        }
        for handle in handles {
            let worker_res = handle
                .join()
                .map_err(|_| GDeflateError::InvalidStream("decompression worker panicked"))?;
            worker_res?;
        }
        Ok(())
    })?;

    let mut tiles = Vec::with_capacity(task_count);
    for i in 0..task_count {
        let mut slot = results[i]
            .lock()
            .map_err(|_| GDeflateError::InvalidStream("result slot lock poisoned"))?;
        let tile = slot.take().ok_or(GDeflateError::InvalidStream(
            "missing decompression worker result",
        ))??;
        tiles.push(tile);
    }
    if options.gpu_decompress_enabled {
        let s = stats
            .lock()
            .map_err(|_| GDeflateError::InvalidStream("decode stats lock poisoned"))?;
        let cpu_done_ms = *cpu_last_done_ms
            .lock()
            .map_err(|_| GDeflateError::InvalidStream("decode cpu done lock poisoned"))?;
        let gpu_done_ms = *gpu_last_done_ms
            .lock()
            .map_err(|_| GDeflateError::InvalidStream("decode gpu done lock poisoned"))?;
        let tail_wait_ms = if gpu_done_ms > cpu_done_ms {
            gpu_done_ms - cpu_done_ms
        } else {
            0.0
        };
        let cpu_rate_tiles_per_ms = if cpu_done_ms > 0.0 {
            s.cpu_tiles as f64 / cpu_done_ms
        } else {
            0.0
        };
        let gpu_rate_tiles_per_ms = if gpu_done_ms > 0.0 {
            s.gpu_tiles as f64 / gpu_done_ms
        } else {
            0.0
        };
        let total_rate = cpu_rate_tiles_per_ms + gpu_rate_tiles_per_ms;
        let gpu_share_pct = if total_rate > 0.0 {
            (gpu_rate_tiles_per_ms / total_rate) * 100.0
        } else {
            0.0
        };
        let gpu_batches_f = s.gpu_batches as f64;
        let gpu_batch_submit_wait_avg_ms = if s.gpu_batches > 0 {
            s.gpu_submit_wait_ms / gpu_batches_f
        } else {
            0.0
        };
        let gpu_batch_total_avg_ms = if s.gpu_batches > 0 {
            s.gpu_total_ms / gpu_batches_f
        } else {
            0.0
        };
        let gpu_profiled_batches_f = s.gpu_profiled_batches as f64;
        let gpu_exec_avg_ms = if s.gpu_profiled_batches > 0 {
            s.gpu_exec_ms / gpu_profiled_batches_f
        } else {
            0.0
        };
        let gpu_submit_overhead_avg_ms = if s.gpu_profiled_batches > 0 {
            s.gpu_submit_overhead_ms / gpu_profiled_batches_f
        } else {
            0.0
        };
        println!(
            "[cozip_gdeflate][timing][hybrid-decode] tasks={} cpu_workers={} gpu_worker={} gpu_inflight_max={} gpu_inflight_obs_max={} gpu_poll_ready_events={} gpu_poll_blocking_waits={} gpu_submit_tiles_req={} gpu_submit_tiles_cap={} gpu_submit_tiles_eff={} gpu_submit_tiles_super_factor={} gpu_submit_tiles_super_eff={} cpu_tiles={} gpu_tiles={} gpu_batches={} gpu_fallback_tiles={} t_cpu_decode_ms={:.3} t_gpu_upload_ms={:.3} t_gpu_submit_wait_ms={:.3} t_gpu_map_copy_ms={:.3} t_gpu_total_ms={:.3} t_gpu_batch_submit_wait_avg_ms={:.3} t_gpu_batch_total_avg_ms={:.3} gpu_profiled_batches={} t_gpu_decode_kernel_ms={:.3} t_gpu_decode_copy_ms={:.3} t_gpu_exec_ms={:.3} t_gpu_submit_overhead_ms={:.3} t_gpu_exec_avg_ms={:.3} t_gpu_submit_overhead_avg_ms={:.3} t_cpu_done_ms={:.3} t_gpu_done_ms={:.3} t_gpu_tail_wait_ms={:.3} r_cpu_tiles_per_ms={:.3} r_gpu_tiles_per_ms={:.3} gpu_share_pct={:.2}",
            task_count,
            cpu_worker_count,
            gpu_worker_count,
            gpu_inflight_batches,
            s.gpu_inflight_observed_max,
            s.gpu_poll_ready_events,
            s.gpu_poll_blocking_waits,
            gpu_submit_tiles,
            gpu_submit_tiles_cap,
            gpu_submit_tiles_effective,
            options.gpu_decompress_super_batch_factor,
            gpu_submit_tiles_super,
            s.cpu_tiles,
            s.gpu_tiles,
            s.gpu_batches,
            s.gpu_fallback_tiles,
            s.cpu_decode_ms,
            s.gpu_upload_ms,
            s.gpu_submit_wait_ms,
            s.gpu_map_copy_ms,
            s.gpu_total_ms,
            gpu_batch_submit_wait_avg_ms,
            gpu_batch_total_avg_ms,
            s.gpu_profiled_batches,
            s.gpu_decode_kernel_ms,
            s.gpu_decode_copy_ms,
            s.gpu_exec_ms,
            s.gpu_submit_overhead_ms,
            gpu_exec_avg_ms,
            gpu_submit_overhead_avg_ms,
            cpu_done_ms,
            gpu_done_ms,
            tail_wait_ms,
            cpu_rate_tiles_per_ms,
            gpu_rate_tiles_per_ms,
            gpu_share_pct,
        );
    }
    Ok(tiles)
}

pub fn gdeflate_compress_cpu_stream<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &GDeflateOptions,
) -> Result<GDeflateStats, GDeflateError> {
    let mut input = Vec::new();
    reader.read_to_end(&mut input)?;
    let output = gdeflate_compress_cpu(&input, options)?;
    writer.write_all(&output)?;
    Ok(GDeflateStats {
        input_bytes: u64::try_from(input.len()).unwrap_or(u64::MAX),
        output_bytes: u64::try_from(output.len()).unwrap_or(u64::MAX),
        tile_count: input.len().div_ceil(options.tile_size),
    })
}

pub fn gdeflate_decompress_cpu_stream<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
) -> Result<GDeflateStats, GDeflateError> {
    let mut stream = Vec::new();
    reader.read_to_end(&mut stream)?;
    let output = gdeflate_decompress_cpu(&stream)?;
    writer.write_all(&output)?;
    let tile_count = parse_tile_stream_header(&stream)
        .map(|header| usize::from(header.num_tiles))
        .unwrap_or(0);
    Ok(GDeflateStats {
        input_bytes: u64::try_from(stream.len()).unwrap_or(u64::MAX),
        output_bytes: u64::try_from(output.len()).unwrap_or(u64::MAX),
        tile_count,
    })
}

pub fn parse_tile_stream_header(stream: &[u8]) -> Result<GDeflateTileStreamHeader, GDeflateError> {
    if stream.len() < TILE_STREAM_HEADER_SIZE {
        return Err(GDeflateError::InvalidStream("tile stream header truncated"));
    }
    let id = stream[0];
    let magic = stream[1];
    let num_tiles = u16::from_le_bytes(
        stream[2..4]
            .try_into()
            .map_err(|_| GDeflateError::InvalidStream("num_tiles parse failed"))?,
    );
    let meta = u32::from_le_bytes(
        stream[4..8]
            .try_into()
            .map_err(|_| GDeflateError::InvalidStream("meta parse failed"))?,
    );
    let tile_size_idx = (meta & 0b11) as u8;
    let last_tile_size = (meta >> 2) & ((1 << 18) - 1);
    let reserved = (meta >> 20) as u16;
    let header = GDeflateTileStreamHeader {
        id,
        magic,
        num_tiles,
        tile_size_idx,
        last_tile_size,
        reserved,
    };
    if !header.is_valid() {
        return Err(GDeflateError::InvalidStream("invalid tile stream header"));
    }
    Ok(header)
}

pub fn parse_tile_offsets(stream: &[u8]) -> Result<Vec<u32>, GDeflateError> {
    let (header, offsets, _) = parse_stream_header_and_offsets(stream)?;
    let tile_count = usize::from(header.num_tiles);
    Ok(offsets[..tile_count].to_vec())
}

fn build_header(
    uncompressed_size: usize,
    num_tiles: usize,
) -> Result<GDeflateTileStreamHeader, GDeflateError> {
    let num_tiles_u16 = u16::try_from(num_tiles).map_err(|_| GDeflateError::DataTooLarge)?;
    let last_tile_size = if num_tiles == 0 {
        0_u32
    } else {
        let rem = uncompressed_size % GDEFLATE_TILE_SIZE;
        if rem == 0 {
            0
        } else {
            u32::try_from(rem).map_err(|_| GDeflateError::DataTooLarge)?
        }
    };
    Ok(GDeflateTileStreamHeader {
        id: GDEFLATE_CODEC_ID,
        magic: GDEFLATE_CODEC_ID ^ 0xff,
        num_tiles: num_tiles_u16,
        tile_size_idx: 1,
        last_tile_size,
        reserved: 0,
    })
}

fn encode_header(out: &mut Vec<u8>, header: &GDeflateTileStreamHeader) {
    out.push(header.id);
    out.push(header.magic);
    out.extend_from_slice(&header.num_tiles.to_le_bytes());
    let mut meta = 0_u32;
    meta |= u32::from(header.tile_size_idx) & 0b11;
    meta |= (header.last_tile_size & ((1 << 18) - 1)) << 2;
    meta |= u32::from(header.reserved) << 20;
    out.extend_from_slice(&meta.to_le_bytes());
}

fn parse_stream_header_and_offsets(
    stream: &[u8],
) -> Result<(GDeflateTileStreamHeader, Vec<u32>, &[u8]), GDeflateError> {
    let header = parse_tile_stream_header(stream)?;
    let tile_count = usize::from(header.num_tiles);
    let offsets_bytes = tile_count
        .checked_mul(4)
        .ok_or(GDeflateError::DataTooLarge)?;
    let offsets_end = TILE_STREAM_HEADER_SIZE
        .checked_add(offsets_bytes)
        .ok_or(GDeflateError::DataTooLarge)?;
    let offsets_slice = stream
        .get(TILE_STREAM_HEADER_SIZE..offsets_end)
        .ok_or(GDeflateError::InvalidStream("tile offset table truncated"))?;
    let mut offsets = vec![0_u32; tile_count];
    for (i, chunk) in offsets_slice.chunks_exact(4).enumerate() {
        offsets[i] = u32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| GDeflateError::InvalidStream("tile offset parse failed"))?,
        );
    }
    let payload = stream
        .get(offsets_end..)
        .ok_or(GDeflateError::InvalidStream("tile payload missing"))?;
    Ok((header, offsets, payload))
}

fn split_tiles(input: &[u8], tile_size: usize) -> Result<Vec<&[u8]>, GDeflateError> {
    if tile_size == 0 {
        return Err(GDeflateError::InvalidOptions(
            "tile_size must be greater than 0",
        ));
    }
    let mut tiles = Vec::new();
    let mut offset = 0usize;
    while offset < input.len() {
        let end = offset.saturating_add(tile_size).min(input.len());
        tiles.push(&input[offset..end]);
        offset = end;
    }
    Ok(tiles)
}

fn expected_tile_output_size(header: &GDeflateTileStreamHeader, tile_index: usize) -> usize {
    let total = usize::from(header.num_tiles);
    if total == 0 {
        return 0;
    }
    if tile_index + 1 < total {
        GDEFLATE_TILE_SIZE
    } else if header.last_tile_size == 0 {
        GDEFLATE_TILE_SIZE
    } else {
        header.last_tile_size as usize
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct LaneState {
    bitbuf: u64,
    bits: u8,
}

fn lane_read_bits(lane: &mut LaneState, bits: u8) -> Result<u32, GDeflateError> {
    if lane.bits < bits {
        return Err(GDeflateError::InvalidStream(
            "sub-stream did not have enough bits for decode step",
        ));
    }
    let mask = if bits == 32 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    };
    let value = (lane.bitbuf & mask) as u32;
    lane.bitbuf >>= bits;
    lane.bits -= bits;
    Ok(value)
}

fn lane_align_byte(lane: &mut LaneState) {
    let rem = lane.bits % 8;
    if rem != 0 {
        lane.bitbuf >>= rem;
        lane.bits -= rem;
    }
}

fn deserialize_lanes_words(page: &[u8]) -> Result<[Vec<u32>; GDEFLATE_NUM_STREAMS], GDeflateError> {
    if page.len() < GDEFLATE_TRAILING_PAD_BYTES {
        return Err(GDeflateError::InvalidStream("tile page too short"));
    }
    let payload_len = page.len() - GDEFLATE_TRAILING_PAD_BYTES;
    let round_bytes = GDEFLATE_NUM_STREAMS * GDEFLATE_STREAM_WORD_BYTES;
    if payload_len % round_bytes != 0 {
        return Err(GDeflateError::InvalidStream(
            "tile page payload is not aligned to lane-word rounds",
        ));
    }
    let words_per_lane = payload_len / round_bytes;
    let mut lanes: [Vec<u32>; GDEFLATE_NUM_STREAMS] =
        std::array::from_fn(|_| Vec::with_capacity(words_per_lane));
    for word_idx in 0..words_per_lane {
        for (lane_idx, lane_words) in lanes.iter_mut().enumerate() {
            let base = (word_idx * GDEFLATE_NUM_STREAMS + lane_idx) * GDEFLATE_STREAM_WORD_BYTES;
            let chunk = page
                .get(base..base + GDEFLATE_STREAM_WORD_BYTES)
                .ok_or(GDeflateError::InvalidStream("tile page word out of bounds"))?;
            let word = u32::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| GDeflateError::InvalidStream("failed to parse lane word"))?,
            );
            lane_words.push(word);
        }
    }
    Ok(lanes)
}

fn init_lane_states(
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
) -> (
    [LaneState; GDEFLATE_NUM_STREAMS],
    [usize; GDEFLATE_NUM_STREAMS],
) {
    let mut lanes: [LaneState; GDEFLATE_NUM_STREAMS] =
        std::array::from_fn(|_| LaneState::default());
    let mut cursors = [0_usize; GDEFLATE_NUM_STREAMS];
    for lane_idx in 0..GDEFLATE_NUM_STREAMS {
        if let Some(&word0) = lane_words[lane_idx].first() {
            lanes[lane_idx].bitbuf = u64::from(word0);
            lanes[lane_idx].bits = 32;
            cursors[lane_idx] = 1;
        }
    }
    (lanes, cursors)
}

fn lane_refill_if_needed(
    lane_idx: usize,
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
) -> bool {
    if lanes[lane_idx].bits > 32 {
        return true;
    }
    let Some(&word) = lane_words[lane_idx].get(cursors[lane_idx]) else {
        return false;
    };
    cursors[lane_idx] = cursors[lane_idx].saturating_add(1);
    lanes[lane_idx].bitbuf |= u64::from(word) << lanes[lane_idx].bits;
    lanes[lane_idx].bits = lanes[lane_idx].bits.saturating_add(32);
    true
}

fn lane_ensure_bits(
    lane_idx: usize,
    need_bits: u8,
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
) -> Result<(), GDeflateError> {
    while lanes[lane_idx].bits < need_bits {
        if !lane_refill_if_needed(lane_idx, lanes, lane_words, cursors) {
            return Err(GDeflateError::InvalidStream(
                "sub-stream did not have enough bits for decode step",
            ));
        }
    }
    Ok(())
}

fn lane_read_bits_checked(
    lane_idx: usize,
    bits: u8,
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
) -> Result<u32, GDeflateError> {
    lane_ensure_bits(lane_idx, bits, lanes, lane_words, cursors)?;
    lane_read_bits(&mut lanes[lane_idx], bits)
}

fn reverse_bits_u16(mut code: u16, len: u8) -> u16 {
    let mut out = 0_u16;
    for _ in 0..len {
        out = (out << 1) | (code & 1);
        code >>= 1;
    }
    out
}

fn static_ll_code_msb(symbol: u16) -> Option<(u16, u8)> {
    match symbol {
        0..=143 => Some((0x30 + symbol, 8)),
        144..=255 => Some((0x190 + (symbol - 144), 9)),
        256..=279 => Some((symbol - 256, 7)),
        280..=287 => Some((0xC0 + (symbol - 280), 8)),
        _ => None,
    }
}

fn static_ll_code_lsb(symbol: u16) -> Option<(u16, u8)> {
    let (msb, len) = static_ll_code_msb(symbol)?;
    Some((reverse_bits_u16(msb, len), len))
}

fn encode_static_ll_symbol(writer: &mut BitWriter, symbol: u16) -> Result<u8, GDeflateError> {
    let (code, len) = static_ll_code_lsb(symbol).ok_or(GDeflateError::InvalidStream(
        "invalid static literal/length symbol",
    ))?;
    writer.push_bits(u32::from(code), usize::from(len));
    Ok(len)
}

fn decode_static_ll_symbol_checked(
    lane_idx: usize,
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
) -> Result<u16, GDeflateError> {
    let mut code_lsb = 0_u16;
    for len in 1..=STATIC_LITERAL_MAX_BITS {
        code_lsb |=
            (lane_read_bits_checked(lane_idx, 1, lanes, lane_words, cursors)? as u16) << (len - 1);
        let msb = reverse_bits_u16(code_lsb, len as u8);
        match len {
            7 if msb <= 23 => return Ok(256 + msb),
            8 if (48..=191).contains(&msb) => return Ok(msb - 48),
            8 if (192..=199).contains(&msb) => return Ok(280 + (msb - 192)),
            9 if (400..=511).contains(&msb) => return Ok(144 + (msb - 400)),
            _ => {}
        }
    }
    Err(GDeflateError::InvalidStream(
        "failed to decode static literal/length symbol",
    ))
}

#[derive(Debug, Clone)]
struct HuffmanCodec {
    max_bits: u8,
    decode: Vec<Vec<u16>>,
    encode: Vec<Option<(u16, u8)>>,
}

fn build_huffman_codec(code_lens: &[u8]) -> Result<HuffmanCodec, GDeflateError> {
    let max_bits = code_lens.iter().copied().max().unwrap_or(0);
    if max_bits == 0 {
        return Err(GDeflateError::InvalidStream("huffman tree has no symbols"));
    }
    if max_bits > 15 {
        return Err(GDeflateError::InvalidStream(
            "huffman code length exceeds 15 bits",
        ));
    }

    let mut bl_count = [0_u16; 16];
    for &len in code_lens {
        if len > 0 {
            bl_count[usize::from(len)] = bl_count[usize::from(len)].saturating_add(1);
        }
    }

    let mut next_code = [0_u16; 16];
    let mut code = 0_u32;
    for bits in 1..=usize::from(max_bits) {
        code = (code + u32::from(bl_count[bits - 1])) << 1;
        if code + u32::from(bl_count[bits]) > (1_u32 << bits) {
            return Err(GDeflateError::InvalidStream("oversubscribed huffman tree"));
        }
        next_code[bits] = code as u16;
    }

    let mut decode = vec![Vec::<u16>::new(); usize::from(max_bits) + 1];
    for bits in 1..=usize::from(max_bits) {
        decode[bits] = vec![u16::MAX; 1usize << bits];
    }
    let mut encode = vec![None; code_lens.len()];

    for (sym, &len) in code_lens.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let bits = usize::from(len);
        let code_msb = next_code[bits];
        next_code[bits] = next_code[bits].saturating_add(1);
        let code_lsb = reverse_bits_u16(code_msb, len);
        let slot = &mut decode[bits][usize::from(code_lsb)];
        if *slot != u16::MAX {
            return Err(GDeflateError::InvalidStream("duplicate huffman code"));
        }
        *slot = u16::try_from(sym).map_err(|_| GDeflateError::DataTooLarge)?;
        encode[sym] = Some((code_lsb, len));
    }

    Ok(HuffmanCodec {
        max_bits,
        decode,
        encode,
    })
}

fn encode_with_huffman(
    writer: &mut BitWriter,
    codec: &HuffmanCodec,
    symbol: usize,
) -> Result<(), GDeflateError> {
    let (code, len) =
        codec
            .encode
            .get(symbol)
            .and_then(|x| *x)
            .ok_or(GDeflateError::InvalidStream(
                "symbol not present in huffman tree",
            ))?;
    writer.push_bits(u32::from(code), usize::from(len));
    Ok(())
}

fn decode_with_huffman(
    lane_idx: usize,
    codec: &HuffmanCodec,
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
) -> Result<u16, GDeflateError> {
    let mut code_lsb = 0_u16;
    for len in 1..=codec.max_bits {
        let bit = lane_read_bits_checked(lane_idx, 1, lanes, lane_words, cursors)? as u16;
        code_lsb |= bit << (len - 1);
        let table = &codec.decode[usize::from(len)];
        if !table.is_empty() {
            let sym = table[usize::from(code_lsb)];
            if sym != u16::MAX {
                return Ok(sym);
            }
        }
    }
    Err(GDeflateError::InvalidStream(
        "failed to decode huffman symbol",
    ))
}

#[derive(Debug, Clone, Copy)]
enum LzToken {
    Literal(u8),
    Match { len: usize, dist: usize },
}

fn hash3(bytes: &[u8]) -> usize {
    let v = u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16);
    ((v.wrapping_mul(2654435761) >> (32 - LZ_HASH_BITS)) as usize) & (LZ_HASH_SIZE - 1)
}

fn insert_hash_pos(tile: &[u8], pos: usize, head: &mut [usize], prev: &mut [usize]) {
    if pos + (LZ_MIN_MATCH - 1) >= tile.len() {
        return;
    }
    let h = hash3(&tile[pos..pos + LZ_MIN_MATCH]);
    prev[pos] = head[h];
    head[h] = pos;
}

fn find_best_match(
    tile: &[u8],
    pos: usize,
    head: &[usize],
    prev: &[usize],
) -> Option<(usize, usize)> {
    if pos + LZ_MIN_MATCH > tile.len() {
        return None;
    }
    let h = hash3(&tile[pos..pos + LZ_MIN_MATCH]);
    let mut best_len = 0usize;
    let mut best_dist = 0usize;
    let mut candidate = head[h];
    let mut chain = 0usize;
    while candidate != usize::MAX && chain < LZ_MAX_CHAIN {
        if candidate >= pos {
            break;
        }
        let dist = pos - candidate;
        if dist > DEFLATE_WINDOW_SIZE {
            break;
        }
        let max_len = LZ_MAX_MATCH.min(tile.len() - pos);
        let mut l = 0usize;
        while l < max_len && tile[candidate + l] == tile[pos + l] {
            l += 1;
        }
        if l >= LZ_MIN_MATCH && l > best_len {
            best_len = l;
            best_dist = dist;
            if best_len == LZ_MAX_MATCH {
                break;
            }
        }
        candidate = prev[candidate];
        chain += 1;
    }
    if best_len >= LZ_MIN_MATCH {
        Some((best_len, best_dist))
    } else {
        None
    }
}

fn build_lz77_tokens(tile: &[u8]) -> Vec<LzToken> {
    let mut tokens = Vec::new();
    if tile.is_empty() {
        return tokens;
    }
    let mut head = vec![usize::MAX; LZ_HASH_SIZE];
    let mut prev = vec![usize::MAX; tile.len()];
    let mut pos = 0usize;
    while pos < tile.len() {
        if let Some((best_len, best_dist)) = find_best_match(tile, pos, &head, &prev) {
            tokens.push(LzToken::Match {
                len: best_len,
                dist: best_dist,
            });
            for p in pos..(pos + best_len) {
                insert_hash_pos(tile, p, &mut head, &mut prev);
            }
            pos += best_len;
        } else {
            tokens.push(LzToken::Literal(tile[pos]));
            insert_hash_pos(tile, pos, &mut head, &mut prev);
            pos += 1;
        }
    }
    tokens
}

fn encode_length_symbol(len: usize) -> Result<(u16, u16, u8), GDeflateError> {
    if !(LZ_MIN_MATCH..=LZ_MAX_MATCH).contains(&len) {
        return Err(GDeflateError::InvalidStream("invalid LZ match length"));
    }
    if len == LZ_MAX_MATCH {
        return Ok((285, 0, 0));
    }
    for (i, &base) in LENGTH_BASE.iter().take(28).enumerate() {
        let extra_bits = LENGTH_EXTRA[i];
        let span = 1usize << extra_bits;
        if (base..base + span).contains(&len) {
            let symbol = 257_u16 + u16::try_from(i).unwrap_or(0);
            let extra = u16::try_from(len - base).unwrap_or(0);
            return Ok((symbol, extra, extra_bits));
        }
    }
    Err(GDeflateError::InvalidStream(
        "failed to map LZ match length to deflate symbol",
    ))
}

fn encode_distance_symbol(dist: usize) -> Result<(u16, u16, u8), GDeflateError> {
    if !(1..=DEFLATE_WINDOW_SIZE).contains(&dist) {
        return Err(GDeflateError::InvalidStream("invalid LZ distance"));
    }
    for (i, &base) in DIST_BASE.iter().enumerate() {
        let extra_bits = DIST_EXTRA[i];
        let span = 1usize << extra_bits;
        if (base..base + span).contains(&dist) {
            let symbol = u16::try_from(i).unwrap_or(0);
            let extra = u16::try_from(dist - base).unwrap_or(0);
            return Ok((symbol, extra, extra_bits));
        }
    }
    Err(GDeflateError::InvalidStream(
        "failed to map LZ distance to deflate symbol",
    ))
}

fn emit_static_match(writer: &mut BitWriter, len: usize, dist: usize) -> Result<(), GDeflateError> {
    let (len_sym, len_extra, len_extra_bits) = encode_length_symbol(len)?;
    let _ = encode_static_ll_symbol(writer, len_sym)?;
    if len_extra_bits > 0 {
        writer.push_bits(u32::from(len_extra), usize::from(len_extra_bits));
    }
    let (dist_sym, dist_extra, dist_extra_bits) = encode_distance_symbol(dist)?;
    let dist_code_lsb = reverse_bits_u16(dist_sym, 5);
    writer.push_bits(u32::from(dist_code_lsb), 5);
    if dist_extra_bits > 0 {
        writer.push_bits(u32::from(dist_extra), usize::from(dist_extra_bits));
    }
    Ok(())
}

fn build_unbounded_huffman_lengths(freqs: &[u32]) -> Vec<u8> {
    #[derive(Clone, Copy)]
    struct Node {
        freq: u32,
        parent: usize,
        left: usize,
        right: usize,
    }

    let mut lengths = vec![0_u8; freqs.len()];
    let active: Vec<usize> = freqs
        .iter()
        .enumerate()
        .filter_map(|(i, &f)| if f > 0 { Some(i) } else { None })
        .collect();
    if active.is_empty() {
        return lengths;
    }
    if active.len() == 1 {
        lengths[active[0]] = 1;
        return lengths;
    }

    let mut nodes: Vec<Node> = Vec::with_capacity(active.len() * 2);
    for &sym in &active {
        nodes.push(Node {
            freq: freqs[sym],
            parent: usize::MAX,
            left: usize::MAX,
            right: usize::MAX,
        });
    }

    let mut heap: std::collections::BinaryHeap<(
        std::cmp::Reverse<u32>,
        std::cmp::Reverse<usize>,
        usize,
    )> = std::collections::BinaryHeap::new();
    for (idx, node) in nodes.iter().enumerate() {
        heap.push((std::cmp::Reverse(node.freq), std::cmp::Reverse(idx), idx));
    }

    while heap.len() > 1 {
        let (_, _, a) = heap.pop().expect("heap pop a");
        let (_, _, b) = heap.pop().expect("heap pop b");
        let parent = nodes.len();
        let freq_sum = nodes[a].freq.saturating_add(nodes[b].freq);
        nodes[a].parent = parent;
        nodes[b].parent = parent;
        nodes.push(Node {
            freq: freq_sum,
            parent: usize::MAX,
            left: a,
            right: b,
        });
        heap.push((
            std::cmp::Reverse(freq_sum),
            std::cmp::Reverse(parent),
            parent,
        ));
    }

    let root = heap.pop().map(|x| x.2).unwrap_or(0);
    let mut stack = vec![(root, 0_u8)];
    while let Some((idx, depth)) = stack.pop() {
        let node = nodes[idx];
        if node.left == usize::MAX && node.right == usize::MAX {
            let sym = active[idx];
            lengths[sym] = depth.max(1);
        } else {
            if node.left != usize::MAX {
                stack.push((node.left, depth.saturating_add(1)));
            }
            if node.right != usize::MAX {
                stack.push((node.right, depth.saturating_add(1)));
            }
        }
    }
    lengths
}

fn build_length_limited_code_lengths(freqs: &[u32], max_bits: u8) -> Vec<u8> {
    let mut active = false;
    for &f in freqs {
        if f > 0 {
            active = true;
            break;
        }
    }
    if !active {
        return vec![0_u8; freqs.len()];
    }

    let mut shift = 0_u32;
    while shift < 24 {
        let adjusted: Vec<u32> = freqs
            .iter()
            .map(|&f| if f == 0 { 0 } else { (f >> shift).max(1) })
            .collect();
        let lens = build_unbounded_huffman_lengths(&adjusted);
        if lens.iter().copied().max().unwrap_or(0) <= max_bits {
            return lens;
        }
        shift += 1;
    }

    let mut lens = vec![0_u8; freqs.len()];
    for (i, &f) in freqs.iter().enumerate() {
        if f > 0 {
            lens[i] = max_bits;
        }
    }
    lens
}

#[derive(Debug, Clone, Copy)]
struct ClToken {
    sym: u8,
    extra: u16,
    extra_bits: u8,
}

fn rle_code_lengths(lengths: &[u8]) -> Vec<ClToken> {
    let mut out = Vec::<ClToken>::new();
    let mut i = 0usize;
    while i < lengths.len() {
        let v = lengths[i];
        if v == 0 {
            let mut run = 1usize;
            while i + run < lengths.len() && lengths[i + run] == 0 {
                run += 1;
            }
            let mut left = run;
            while left > 0 {
                if left >= 11 {
                    let rep = left.min(138);
                    out.push(ClToken {
                        sym: 18,
                        extra: u16::try_from(rep - 11).unwrap_or(0),
                        extra_bits: 7,
                    });
                    left -= rep;
                } else if left >= 3 {
                    let rep = left.min(10);
                    out.push(ClToken {
                        sym: 17,
                        extra: u16::try_from(rep - 3).unwrap_or(0),
                        extra_bits: 3,
                    });
                    left -= rep;
                } else {
                    out.push(ClToken {
                        sym: 0,
                        extra: 0,
                        extra_bits: 0,
                    });
                    left -= 1;
                }
            }
            i += run;
        } else {
            let mut run = 1usize;
            while i + run < lengths.len() && lengths[i + run] == v {
                run += 1;
            }
            out.push(ClToken {
                sym: v,
                extra: 0,
                extra_bits: 0,
            });
            let mut left = run - 1;
            while left > 0 {
                if left >= 3 {
                    let rep = left.min(6);
                    out.push(ClToken {
                        sym: 16,
                        extra: u16::try_from(rep - 3).unwrap_or(0),
                        extra_bits: 2,
                    });
                    left -= rep;
                } else {
                    out.push(ClToken {
                        sym: v,
                        extra: 0,
                        extra_bits: 0,
                    });
                    left -= 1;
                }
            }
            i += run;
        }
    }
    out
}

fn encode_dynamic_code_lengths(
    writer: &mut BitWriter,
    cl_codec: &HuffmanCodec,
    tokens: &[ClToken],
) -> Result<(), GDeflateError> {
    for token in tokens {
        encode_with_huffman(writer, cl_codec, usize::from(token.sym))?;
        if token.extra_bits > 0 {
            writer.push_bits(u32::from(token.extra), usize::from(token.extra_bits));
        }
    }
    Ok(())
}

fn decode_dynamic_code_lengths(
    lane_idx: usize,
    cl_codec: &HuffmanCodec,
    total: usize,
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
) -> Result<Vec<u8>, GDeflateError> {
    let mut out = Vec::with_capacity(total);
    while out.len() < total {
        let sym = decode_with_huffman(lane_idx, cl_codec, lanes, lane_words, cursors)?;
        match sym {
            0..=15 => out.push(sym as u8),
            16 => {
                if out.is_empty() {
                    return Err(GDeflateError::InvalidStream(
                        "repeat code 16 without previous length",
                    ));
                }
                let repeat =
                    3 + lane_read_bits_checked(lane_idx, 2, lanes, lane_words, cursors)? as usize;
                let prev = *out.last().unwrap_or(&0);
                if out.len() + repeat > total {
                    return Err(GDeflateError::InvalidStream(
                        "dynamic code length repeat overflow",
                    ));
                }
                out.resize(out.len() + repeat, prev);
            }
            17 => {
                let repeat =
                    3 + lane_read_bits_checked(lane_idx, 3, lanes, lane_words, cursors)? as usize;
                if out.len() + repeat > total {
                    return Err(GDeflateError::InvalidStream(
                        "dynamic code length repeat overflow",
                    ));
                }
                out.resize(out.len() + repeat, 0);
            }
            18 => {
                let repeat =
                    11 + lane_read_bits_checked(lane_idx, 7, lanes, lane_words, cursors)? as usize;
                if out.len() + repeat > total {
                    return Err(GDeflateError::InvalidStream(
                        "dynamic code length repeat overflow",
                    ));
                }
                out.resize(out.len() + repeat, 0);
            }
            _ => {
                return Err(GDeflateError::InvalidStream(
                    "invalid code-length alphabet symbol",
                ));
            }
        }
    }
    Ok(out)
}

fn encode_tile_dynamic_huffman(tile: &[u8]) -> Result<Vec<u8>, GDeflateError> {
    let mut lanes: [BitWriter; GDEFLATE_NUM_STREAMS] =
        std::array::from_fn(|_| BitWriter::default());
    lanes[0].push_bits(1, 1); // BFINAL
    lanes[0].push_bits(0b10, 2); // BTYPE=10 dynamic

    let tokens = build_lz77_tokens(tile);
    let mut lit_freq = vec![0_u32; 286];
    let mut dist_freq = vec![0_u32; 30];
    for token in &tokens {
        match *token {
            LzToken::Literal(b) => {
                lit_freq[usize::from(b)] = lit_freq[usize::from(b)].saturating_add(1);
            }
            LzToken::Match { len, dist } => {
                let (len_sym, _, _) = encode_length_symbol(len)?;
                lit_freq[usize::from(len_sym)] = lit_freq[usize::from(len_sym)].saturating_add(1);
                let (dist_sym, _, _) = encode_distance_symbol(dist)?;
                dist_freq[usize::from(dist_sym)] =
                    dist_freq[usize::from(dist_sym)].saturating_add(1);
            }
        }
    }
    lit_freq[256] = lit_freq[256].saturating_add(1); // EOB
    if dist_freq.iter().all(|&f| f == 0) {
        dist_freq[0] = 1;
    }

    let lit_lens_full = build_length_limited_code_lengths(&lit_freq, 15);
    let dist_lens_full = build_length_limited_code_lengths(&dist_freq, 15);
    let lit_last = lit_lens_full
        .iter()
        .rposition(|&x| x != 0)
        .unwrap_or(256)
        .max(256);
    let dist_last = dist_lens_full.iter().rposition(|&x| x != 0).unwrap_or(0);
    let hlit = lit_last + 1;
    let hdist = dist_last + 1;
    let lit_lens = lit_lens_full[..hlit].to_vec();
    let dist_lens = dist_lens_full[..hdist].to_vec();
    let lit_codec = build_huffman_codec(&lit_lens)?;
    let dist_codec = build_huffman_codec(&dist_lens)?;

    lanes[0].push_bits(u32::try_from(hlit - 257).unwrap_or(0), 5);
    lanes[0].push_bits(u32::try_from(hdist - 1).unwrap_or(0), 5);

    let mut all_lens = Vec::with_capacity(hlit + hdist);
    all_lens.extend_from_slice(&lit_lens);
    all_lens.extend_from_slice(&dist_lens);
    let cl_tokens = rle_code_lengths(&all_lens);
    let mut cl_freq = vec![0_u32; 19];
    for t in &cl_tokens {
        cl_freq[usize::from(t.sym)] = cl_freq[usize::from(t.sym)].saturating_add(1);
    }
    let cl_lens = build_length_limited_code_lengths(&cl_freq, 7);
    let mut hclen = 4usize;
    if let Some(last) = CODELEN_ORDER.iter().rposition(|&sym| cl_lens[sym] != 0) {
        hclen = (last + 1).max(4);
    }
    lanes[0].push_bits(u32::try_from(hclen - 4).unwrap_or(0), 4);
    for &sym in CODELEN_ORDER.iter().take(hclen) {
        lanes[0].push_bits(u32::from(cl_lens[sym]), 3);
    }
    let cl_codec = build_huffman_codec(&cl_lens)?;
    encode_dynamic_code_lengths(&mut lanes[0], &cl_codec, &cl_tokens)?;

    for (i, token) in tokens.iter().enumerate() {
        let lane = i % GDEFLATE_NUM_STREAMS;
        match *token {
            LzToken::Literal(b) => {
                encode_with_huffman(&mut lanes[lane], &lit_codec, usize::from(b))?;
            }
            LzToken::Match { len, dist } => {
                let (len_sym, len_extra, len_extra_bits) = encode_length_symbol(len)?;
                encode_with_huffman(&mut lanes[lane], &lit_codec, usize::from(len_sym))?;
                if len_extra_bits > 0 {
                    lanes[lane].push_bits(u32::from(len_extra), usize::from(len_extra_bits));
                }
                let (dist_sym, dist_extra, dist_extra_bits) = encode_distance_symbol(dist)?;
                encode_with_huffman(&mut lanes[lane], &dist_codec, usize::from(dist_sym))?;
                if dist_extra_bits > 0 {
                    lanes[lane].push_bits(u32::from(dist_extra), usize::from(dist_extra_bits));
                }
            }
        }
    }
    let eob_lane = tokens.len() % GDEFLATE_NUM_STREAMS;
    encode_with_huffman(&mut lanes[eob_lane], &lit_codec, 256)?;
    Ok(serialize_lanes_words(&lanes))
}

fn encode_tile_stored(tile: &[u8]) -> Vec<u8> {
    let mut lanes: [BitWriter; GDEFLATE_NUM_STREAMS] =
        std::array::from_fn(|_| BitWriter::default());
    let mut offset = 0usize;
    while offset < tile.len() {
        let seg_len = (tile.len() - offset).min(u16::MAX as usize);
        let is_final = offset + seg_len == tile.len();
        lanes[0].push_bits(if is_final { 1 } else { 0 }, 1); // BFINAL
        lanes[0].push_bits(0, 2); // BTYPE=00 (stored)
        lanes[0].align_byte();
        let len = seg_len as u16;
        lanes[0].push_u16_le(len);
        lanes[0].push_u16_le(!len);
        for (i, byte) in tile[offset..offset + seg_len].iter().copied().enumerate() {
            let lane = i % GDEFLATE_NUM_STREAMS;
            lanes[lane].push_u8(byte);
        }
        offset += seg_len;
    }
    serialize_lanes_words(&lanes)
}

fn encode_tile_static_huffman(tile: &[u8]) -> Vec<u8> {
    let mut lanes: [BitWriter; GDEFLATE_NUM_STREAMS] =
        std::array::from_fn(|_| BitWriter::default());
    lanes[0].push_bits(1, 1); // BFINAL
    lanes[0].push_bits(0b01, 2); // BTYPE=01 static Huffman

    let tokens = build_lz77_tokens(tile);
    for (i, token) in tokens.iter().enumerate() {
        let lane = i % GDEFLATE_NUM_STREAMS;
        match *token {
            LzToken::Literal(b) => {
                let _ = encode_static_ll_symbol(&mut lanes[lane], u16::from(b));
            }
            LzToken::Match { len, dist } => {
                let _ = emit_static_match(&mut lanes[lane], len, dist);
            }
        }
    }
    let eob_lane = tokens.len() % GDEFLATE_NUM_STREAMS;
    let _ = encode_static_ll_symbol(&mut lanes[eob_lane], 256);
    serialize_lanes_words(&lanes)
}

fn decode_tile(page: &[u8], expected_len: usize, out: &mut Vec<u8>) -> Result<(), GDeflateError> {
    let lane_words = deserialize_lanes_words(page)?;
    let (mut lanes, mut cursors) = init_lane_states(&lane_words);
    let bfinal = lane_read_bits_checked(0, 1, &mut lanes, &lane_words, &mut cursors)?;
    let btype = lane_read_bits_checked(0, 2, &mut lanes, &lane_words, &mut cursors)?;
    match btype {
        0 => decode_tile_stored_blocks(
            &mut lanes,
            &lane_words,
            &mut cursors,
            bfinal,
            expected_len,
            out,
        ),
        1 => decode_tile_static_huffman_blocks(
            &mut lanes,
            &lane_words,
            &mut cursors,
            bfinal,
            expected_len,
            out,
        ),
        2 => {
            if bfinal != 1 {
                return Err(GDeflateError::InvalidStream(
                    "non-final dynamic block in tile is not supported",
                ));
            }
            decode_tile_dynamic_huffman_single_block(
                &mut lanes,
                &lane_words,
                &mut cursors,
                expected_len,
                out,
            )
        }
        _ => Err(GDeflateError::InvalidStream(
            "unsupported BTYPE in current CPU GDeflate implementation",
        )),
    }
}

fn decode_tile_stored_blocks(
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
    mut bfinal: u32,
    expected_len: usize,
    out: &mut Vec<u8>,
) -> Result<(), GDeflateError> {
    let mut produced = 0usize;
    loop {
        lane_align_byte(&mut lanes[0]);
        let len = (lane_read_bits_checked(0, 8, lanes, lane_words, cursors)?
            | (lane_read_bits_checked(0, 8, lanes, lane_words, cursors)? << 8))
            as usize;
        let nlen = u16::from(lane_read_bits_checked(0, 8, lanes, lane_words, cursors)? as u8)
            | (u16::from(lane_read_bits_checked(0, 8, lanes, lane_words, cursors)? as u8) << 8);
        if nlen != !(len as u16) {
            return Err(GDeflateError::InvalidStream(
                "stored block LEN/NLEN mismatch",
            ));
        }
        for i in 0..len {
            let lane_idx = i % GDEFLATE_NUM_STREAMS;
            let byte = lane_read_bits_checked(lane_idx, 8, lanes, lane_words, cursors)? as u8;
            out.push(byte);
            produced = produced.saturating_add(1);
            if produced > expected_len {
                return Err(GDeflateError::InvalidStream(
                    "decoded more bytes than tile expected size",
                ));
            }
        }
        if bfinal == 1 {
            break;
        }
        bfinal = lane_read_bits_checked(0, 1, lanes, lane_words, cursors)?;
        let btype = lane_read_bits_checked(0, 2, lanes, lane_words, cursors)?;
        if btype != 0 {
            return Err(GDeflateError::InvalidStream(
                "mixed block types in tile are not supported for stored decode",
            ));
        }
    }
    if produced != expected_len {
        return Err(GDeflateError::InvalidStream(
            "stored block length does not match tile expected length",
        ));
    }
    Ok(())
}

fn decode_length_from_symbol(
    sym: u16,
    lane_idx: usize,
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
) -> Result<usize, GDeflateError> {
    if sym == 285 {
        return Ok(258);
    }
    if !(257..=284).contains(&sym) {
        return Err(GDeflateError::InvalidStream(
            "invalid static length symbol in stream",
        ));
    }
    let idx = usize::from(sym - 257);
    let base = LENGTH_BASE[idx];
    let extra_bits = LENGTH_EXTRA[idx];
    let extra = if extra_bits == 0 {
        0usize
    } else {
        lane_read_bits_checked(lane_idx, extra_bits, lanes, lane_words, cursors)? as usize
    };
    Ok(base + extra)
}

fn decode_distance_from_symbol(
    sym: u16,
    lane_idx: usize,
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
) -> Result<usize, GDeflateError> {
    if sym > 29 {
        return Err(GDeflateError::InvalidStream(
            "invalid distance symbol in stream",
        ));
    }
    let idx = usize::from(sym);
    let base = DIST_BASE[idx];
    let extra_bits = DIST_EXTRA[idx];
    let extra = if extra_bits == 0 {
        0usize
    } else {
        lane_read_bits_checked(lane_idx, extra_bits, lanes, lane_words, cursors)? as usize
    };
    Ok(base + extra)
}

fn decode_static_distance_symbol(
    lane_idx: usize,
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
) -> Result<usize, GDeflateError> {
    let code_lsb = lane_read_bits_checked(lane_idx, 5, lanes, lane_words, cursors)? as u16;
    let sym = reverse_bits_u16(code_lsb, 5);
    decode_distance_from_symbol(sym, lane_idx, lanes, lane_words, cursors)
}

fn decode_tile_static_huffman_single_block(
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
    expected_len: usize,
    produced_before: usize,
    tile_start: usize,
    out: &mut Vec<u8>,
) -> Result<usize, GDeflateError> {
    let mut produced_in_block = 0usize;
    loop {
        for lane_idx in 0..GDEFLATE_NUM_STREAMS {
            let sym = decode_static_ll_symbol_checked(lane_idx, lanes, lane_words, cursors)?;
            if sym < 256 {
                out.push(sym as u8);
                produced_in_block = produced_in_block.saturating_add(1);
                if produced_before + produced_in_block > expected_len {
                    return Err(GDeflateError::InvalidStream(
                        "decoded more bytes than tile expected size",
                    ));
                }
            } else if sym == 256 {
                return Ok(produced_in_block);
            } else if sym <= 285 {
                let len = decode_length_from_symbol(sym, lane_idx, lanes, lane_words, cursors)?;
                let dist = decode_static_distance_symbol(lane_idx, lanes, lane_words, cursors)?;
                let produced_total = produced_before + produced_in_block;
                if dist == 0 || dist > produced_total {
                    return Err(GDeflateError::InvalidStream(
                        "invalid static match distance",
                    ));
                }
                if produced_total + len > expected_len {
                    return Err(GDeflateError::InvalidStream(
                        "decoded more bytes than tile expected size",
                    ));
                }
                for _ in 0..len {
                    let src = tile_start + produced_before + produced_in_block - dist;
                    let b = out[src];
                    out.push(b);
                    produced_in_block += 1;
                }
            } else {
                return Err(GDeflateError::InvalidStream(
                    "reserved static length symbol",
                ));
            }
        }
    }
}

fn decode_tile_static_huffman_blocks(
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
    mut bfinal: u32,
    expected_len: usize,
    out: &mut Vec<u8>,
) -> Result<(), GDeflateError> {
    let tile_start = out.len();
    let mut produced_total = 0usize;
    loop {
        let produced_block = decode_tile_static_huffman_single_block(
            lanes,
            lane_words,
            cursors,
            expected_len,
            produced_total,
            tile_start,
            out,
        )?;
        produced_total = produced_total.saturating_add(produced_block);
        if produced_total > expected_len {
            return Err(GDeflateError::InvalidStream(
                "decoded more bytes than tile expected size",
            ));
        }

        if bfinal == 1 {
            if produced_total != expected_len {
                return Err(GDeflateError::InvalidStream(
                    "static block ended before expected tile size",
                ));
            }
            return Ok(());
        }

        for lane in lanes.iter_mut() {
            lane_align_byte(lane);
        }
        bfinal = lane_read_bits_checked(0, 1, lanes, lane_words, cursors)?;
        let btype = lane_read_bits_checked(0, 2, lanes, lane_words, cursors)?;
        if btype != 1 {
            return Err(GDeflateError::InvalidStream(
                "mixed block types in tile are not supported for static decode",
            ));
        }
    }
}

fn decode_tile_dynamic_huffman_single_block(
    lanes: &mut [LaneState; GDEFLATE_NUM_STREAMS],
    lane_words: &[Vec<u32>; GDEFLATE_NUM_STREAMS],
    cursors: &mut [usize; GDEFLATE_NUM_STREAMS],
    expected_len: usize,
    out: &mut Vec<u8>,
) -> Result<(), GDeflateError> {
    let hlit = 257 + lane_read_bits_checked(0, 5, lanes, lane_words, cursors)? as usize;
    let hdist = 1 + lane_read_bits_checked(0, 5, lanes, lane_words, cursors)? as usize;
    let hclen = 4 + lane_read_bits_checked(0, 4, lanes, lane_words, cursors)? as usize;
    if hlit > 286 || hdist > 32 || hclen > 19 {
        return Err(GDeflateError::InvalidStream(
            "dynamic header fields out of range",
        ));
    }

    let mut cl_lens = vec![0_u8; 19];
    for i in 0..hclen {
        let sym = CODELEN_ORDER[i];
        cl_lens[sym] = lane_read_bits_checked(0, 3, lanes, lane_words, cursors)? as u8;
    }
    let cl_codec = build_huffman_codec(&cl_lens)?;
    let total = hlit + hdist;
    let lens = decode_dynamic_code_lengths(0, &cl_codec, total, lanes, lane_words, cursors)?;
    let lit_lens = lens[..hlit].to_vec();
    let dist_lens = lens[hlit..].to_vec();

    let lit_codec = build_huffman_codec(&lit_lens)?;
    let dist_codec = build_huffman_codec(&dist_lens)?;
    let tile_start = out.len();
    let mut produced = 0usize;
    loop {
        for lane_idx in 0..GDEFLATE_NUM_STREAMS {
            let sym = decode_with_huffman(lane_idx, &lit_codec, lanes, lane_words, cursors)?;
            match sym {
                0..=255 => {
                    out.push(sym as u8);
                    produced += 1;
                    if produced > expected_len {
                        return Err(GDeflateError::InvalidStream(
                            "decoded more bytes than tile expected size",
                        ));
                    }
                }
                256 => {
                    if produced != expected_len {
                        return Err(GDeflateError::InvalidStream(
                            "dynamic block ended before expected tile size",
                        ));
                    }
                    return Ok(());
                }
                257..=285 => {
                    let len = decode_length_from_symbol(sym, lane_idx, lanes, lane_words, cursors)?;
                    let dist_sym =
                        decode_with_huffman(lane_idx, &dist_codec, lanes, lane_words, cursors)?;
                    let dist = decode_distance_from_symbol(
                        dist_sym, lane_idx, lanes, lane_words, cursors,
                    )?;
                    if dist == 0 || dist > produced {
                        return Err(GDeflateError::InvalidStream(
                            "invalid dynamic match distance",
                        ));
                    }
                    if produced + len > expected_len {
                        return Err(GDeflateError::InvalidStream(
                            "decoded more bytes than tile expected size",
                        ));
                    }
                    for _ in 0..len {
                        let src = tile_start + produced - dist;
                        let b = out[src];
                        out.push(b);
                        produced += 1;
                    }
                }
                _ => {
                    return Err(GDeflateError::InvalidStream(
                        "reserved dynamic literal/length symbol",
                    ));
                }
            }
        }
    }
}

fn serialize_lanes_words(lanes: &[BitWriter; GDEFLATE_NUM_STREAMS]) -> Vec<u8> {
    let lane_bytes: [Vec<u8>; GDEFLATE_NUM_STREAMS] =
        std::array::from_fn(|i| lanes[i].bytes.clone());
    let lane_words: [usize; GDEFLATE_NUM_STREAMS] =
        std::array::from_fn(|i| lane_bytes[i].len().div_ceil(GDEFLATE_STREAM_WORD_BYTES));
    let max_words = lane_words.iter().copied().max().unwrap_or(0);

    let mut out = Vec::with_capacity(
        max_words
            .saturating_mul(GDEFLATE_NUM_STREAMS)
            .saturating_mul(GDEFLATE_STREAM_WORD_BYTES)
            .saturating_add(GDEFLATE_TRAILING_PAD_BYTES),
    );
    for w in 0..max_words {
        let base = w.saturating_mul(GDEFLATE_STREAM_WORD_BYTES);
        for bytes in lane_bytes.iter().take(GDEFLATE_NUM_STREAMS) {
            for k in 0..GDEFLATE_STREAM_WORD_BYTES {
                out.push(*bytes.get(base + k).unwrap_or(&0));
            }
        }
    }
    out.resize(out.len().saturating_add(GDEFLATE_TRAILING_PAD_BYTES), 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data(size: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(size);
        for i in 0..size {
            let v = ((i as u32)
                .wrapping_mul(1664525)
                .wrapping_add(1013904223)
                .rotate_left(5)
                & 0xff) as u8;
            out.push(v);
        }
        out
    }

    #[test]
    fn roundtrip_empty() {
        let options = GDeflateOptions::default();
        let stream = gdeflate_compress_cpu(&[], &options).expect("compress should succeed");
        let header = parse_tile_stream_header(&stream).expect("header should parse");
        assert_eq!(header.num_tiles, 0);
        let decoded = gdeflate_decompress_cpu(&stream).expect("decompress should succeed");
        assert!(decoded.is_empty());
    }

    #[test]
    fn roundtrip_single_tile() {
        let input = sample_data(30_000);
        let stream = gdeflate_compress_cpu(&input, &GDeflateOptions::default())
            .expect("compress should succeed");
        let decoded = gdeflate_decompress_cpu(&stream).expect("decompress should succeed");
        assert_eq!(decoded, input);
        let header = parse_tile_stream_header(&stream).expect("header should parse");
        assert_eq!(header.num_tiles, 1);
    }

    #[test]
    fn roundtrip_multi_tile() {
        let input = sample_data(200_000);
        let stream = gdeflate_compress_cpu(&input, &GDeflateOptions::default())
            .expect("compress should succeed");
        let decoded = gdeflate_decompress_cpu(&stream).expect("decompress should succeed");
        assert_eq!(decoded, input);
        let offsets = parse_tile_offsets(&stream).expect("offsets should parse");
        assert_eq!(offsets.len(), 4);
    }

    #[test]
    fn roundtrip_single_tile_stored_multiblock_boundary() {
        let input = sample_data(GDEFLATE_TILE_SIZE);
        let options = GDeflateOptions {
            tile_size: GDEFLATE_TILE_SIZE,
            compression_mode: GDeflateCompressionMode::StoredOnly,
            ..GDeflateOptions::default()
        };
        let stream = gdeflate_compress_cpu(&input, &options).expect("compress should succeed");
        let decoded = gdeflate_decompress_cpu(&stream).expect("decompress should succeed");
        assert_eq!(decoded, input);
    }

    #[test]
    fn roundtrip_static_huffman_mode() {
        let input = sample_data(70_000);
        let options = GDeflateOptions {
            tile_size: GDEFLATE_TILE_SIZE,
            compression_mode: GDeflateCompressionMode::StaticHuffman,
            ..GDeflateOptions::default()
        };
        let stream = gdeflate_compress_cpu(&input, &options).expect("compress should succeed");
        let decoded = gdeflate_decompress_cpu(&stream).expect("decompress should succeed");
        assert_eq!(decoded, input);
    }

    #[test]
    fn static_mode_uses_lz_matches_on_repetitive_input() {
        let input = vec![b'A'; 128 * 1024];
        let static_stream = gdeflate_compress_cpu(
            &input,
            &GDeflateOptions {
                tile_size: GDEFLATE_TILE_SIZE,
                compression_mode: GDeflateCompressionMode::StaticHuffman,
                ..GDeflateOptions::default()
            },
        )
        .expect("static compress should succeed");
        let stored_stream = gdeflate_compress_cpu(
            &input,
            &GDeflateOptions {
                tile_size: GDEFLATE_TILE_SIZE,
                compression_mode: GDeflateCompressionMode::StoredOnly,
                ..GDeflateOptions::default()
            },
        )
        .expect("stored compress should succeed");
        assert!(
            static_stream.len() < stored_stream.len(),
            "static stream should be smaller than stored stream for repetitive data"
        );
        let decoded = gdeflate_decompress_cpu(&static_stream).expect("decompress should succeed");
        assert_eq!(decoded, input);
    }

    #[test]
    fn roundtrip_tryall_mode() {
        let input = sample_data(90_000);
        let options = GDeflateOptions {
            tile_size: GDEFLATE_TILE_SIZE,
            compression_mode: GDeflateCompressionMode::TryAll,
            ..GDeflateOptions::default()
        };
        let stream = gdeflate_compress_cpu(&input, &options).expect("compress should succeed");
        let decoded = gdeflate_decompress_cpu(&stream).expect("decompress should succeed");
        assert_eq!(decoded, input);
    }

    #[test]
    fn roundtrip_dynamic_huffman_mode() {
        let input = sample_data(120_000);
        let options = GDeflateOptions {
            tile_size: GDEFLATE_TILE_SIZE,
            compression_mode: GDeflateCompressionMode::DynamicHuffman,
            ..GDeflateOptions::default()
        };
        let stream = gdeflate_compress_cpu(&input, &options).expect("compress should succeed");
        let decoded = gdeflate_decompress_cpu(&stream).expect("decompress should succeed");
        assert_eq!(decoded, input);
    }

    #[test]
    fn invalid_magic_rejected() {
        let input = sample_data(1024);
        let mut stream = gdeflate_compress_cpu(&input, &GDeflateOptions::default())
            .expect("compress should succeed");
        stream[1] ^= 1;
        let err = gdeflate_decompress_cpu(&stream).expect_err("must fail");
        assert!(matches!(err, GDeflateError::InvalidStream(_)));
    }

    #[test]
    fn roundtrip_with_explicit_worker_settings() {
        let input = sample_data(300_000);
        let single = GDeflateOptions {
            tile_size: GDEFLATE_TILE_SIZE,
            compression_mode: GDeflateCompressionMode::TryAll,
            cpu_worker_count: 1,
            ..GDeflateOptions::default()
        };
        let multi = GDeflateOptions {
            tile_size: GDEFLATE_TILE_SIZE,
            compression_mode: GDeflateCompressionMode::TryAll,
            cpu_worker_count: 4,
            ..GDeflateOptions::default()
        };

        let stream_single =
            gdeflate_compress_cpu(&input, &single).expect("single compress should succeed");
        let stream_multi =
            gdeflate_compress_cpu(&input, &multi).expect("multi compress should succeed");
        let dec_single = gdeflate_decompress_cpu_with_options(&stream_single, &single)
            .expect("single decompress should succeed");
        let dec_multi = gdeflate_decompress_cpu_with_options(&stream_multi, &multi)
            .expect("multi decompress should succeed");
        assert_eq!(dec_single, input);
        assert_eq!(dec_multi, input);
    }

    #[test]
    fn roundtrip_tryall_many_tiles_many_workers() {
        let input = sample_data(4 * 1024 * 1024);
        let opts = GDeflateOptions {
            tile_size: GDEFLATE_TILE_SIZE,
            compression_mode: GDeflateCompressionMode::TryAll,
            cpu_worker_count: 20,
            ..GDeflateOptions::default()
        };
        let stream = gdeflate_compress_cpu(&input, &opts).expect("compress should succeed");
        let decoded = gdeflate_decompress_cpu_with_options(&stream, &opts)
            .expect("decompress should succeed");
        assert_eq!(decoded, input);
    }
}
