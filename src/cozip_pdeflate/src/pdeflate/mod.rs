use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::io::{Read, Seek, SeekFrom, Write};
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Instant;

use thiserror::Error;

mod gpu;
mod shaders;
mod stream;

const STREAM_MAGIC: [u8; 4] = *b"PDS0";
const CHUNK_MAGIC: [u8; 4] = *b"PDF0";
const STREAM_VERSION: u16 = 1;
const STREAM_FLAG_UNCOMPRESSED_SIZE: u16 = 1 << 0;
const STREAM_FLAG_FILE_NAME_PRESENT: u16 = 1 << 1;
const CHUNK_VERSION: u16 = 0;
const CHUNK_FLAG_HUFFMAN: u16 = 1 << 0;
const CHUNK_FLAG_FINAL_STREAM: u16 = 1 << 1;
const LITERAL_TAG: u16 = 0x0fff;
const MAX_TABLE_ID: usize = (LITERAL_TAG as usize) - 1;
const CHUNK_HEADER_SIZE: usize = 36;
const STREAM_HEADER_SIZE: usize = 12;
const STREAM_HEADER_UNCOMPRESSED_SIZE_BYTES: usize = 8;
const MAX_INLINE_LEN: usize = 14;
const EXT_LEN_BASE: usize = 15;
const EXT_LEN_MAX: usize = 255;
const MAX_CMD_LEN: usize = EXT_LEN_BASE + EXT_LEN_MAX;
const EMPTY_CANDIDATES: [usize; 0] = [];
const PREFIX_BUCKET_COUNT: usize = 256 * 256;
static PROFILE_DETAIL_GPU_PACK_SEQ: AtomicU64 = AtomicU64::new(0);
static PROFILE_DETAIL_GPU_BATCH_SEQ: AtomicU64 = AtomicU64::new(0);
static GPU_INTEGRATED_ERROR_SAMPLE_LOGGED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Default)]
struct EncodeScratch {
    repeat_table: Vec<Vec<u8>>,
    by_prefix2: Vec<Vec<usize>>,
    has_prefix2: Vec<u8>,
    prefix3_masks: Vec<[u64; 4]>,
    touched_prefix2: Vec<usize>,
}

impl EncodeScratch {
    fn new() -> Self {
        let mut by_prefix2 = Vec::with_capacity(PREFIX_BUCKET_COUNT);
        by_prefix2.resize_with(PREFIX_BUCKET_COUNT, Vec::new);
        Self {
            repeat_table: Vec::new(),
            by_prefix2,
            has_prefix2: vec![0u8; PREFIX_BUCKET_COUNT],
            prefix3_masks: vec![[0u64; 4]; PREFIX_BUCKET_COUNT],
            touched_prefix2: Vec::new(),
        }
    }

    fn rebuild_repeat_table(&mut self, table: &[Vec<u8>], max_ref_len: usize) {
        if self.repeat_table.len() < table.len() {
            self.repeat_table
                .resize_with(table.len(), || vec![0u8; max_ref_len.max(1)]);
        } else {
            self.repeat_table.truncate(table.len());
        }

        for (idx, entry) in table.iter().enumerate() {
            let rep = &mut self.repeat_table[idx];
            if rep.len() != max_ref_len.max(1) {
                rep.resize(max_ref_len.max(1), 0);
            }
            if entry.is_empty() {
                rep.fill(0);
                continue;
            }
            for i in 0..rep.len() {
                rep[i] = entry[i % entry.len()];
            }
        }
    }

    fn rebuild_prefix_index(&mut self, table: &[Vec<u8>]) {
        for &k2 in &self.touched_prefix2 {
            self.by_prefix2[k2].clear();
            self.has_prefix2[k2] = 0;
            self.prefix3_masks[k2] = [0u64; 4];
        }
        self.touched_prefix2.clear();

        for (id, entry) in table.iter().enumerate() {
            let k2 = if entry.len() >= 2 {
                ((entry[0] as usize) << 8) | (entry[1] as usize)
            } else {
                ((entry[0] as usize) << 8) | (entry[0] as usize)
            };
            if self.has_prefix2[k2] == 0 {
                self.has_prefix2[k2] = 1;
                self.touched_prefix2.push(k2);
            }
            self.by_prefix2[k2].push(id);
            let b2 = if entry.len() >= 3 {
                entry[2]
            } else if entry.len() == 2 {
                entry[0]
            } else {
                entry[0]
            } as usize;
            self.prefix3_masks[k2][b2 >> 6] |= 1u64 << (b2 & 63);
        }

        for &k2 in &self.touched_prefix2 {
            let candidates = &mut self.by_prefix2[k2];
            candidates
                .sort_by(|&a, &b| table[b].len().cmp(&table[a].len()).then_with(|| a.cmp(&b)));
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SeedSlot {
    fingerprint: u64,
    pos: u32,
    len: u16,
    used: u16,
    score: u64,
}

#[derive(Debug, Default)]
struct BuildScratch {
    seed_slots: Vec<SeedSlot>,
    used_seed_slot_indices: Vec<usize>,
    byte_scored: Vec<(u8, u64)>,
    scored_slot_indices: Vec<(usize, u64)>,
    head: Vec<u32>,
    touched_buckets: Vec<usize>,
    prev: Vec<u32>,
    keys: Vec<u32>,
    positions: Vec<u32>,
}

#[derive(Debug, Default)]
struct DecodeScratch {
    table_offsets: Vec<usize>,
    table_repeat: Vec<[u8; MAX_CMD_LEN]>,
}

#[derive(Debug, Default)]
struct FinalizeScratch {
    table_index: Vec<u8>,
    table_data: Vec<u8>,
    section_index: Vec<u8>,
    section_bitstream: Vec<u8>,
}

#[derive(Debug, Default)]
struct CompressTaskQueueState {
    queue: VecDeque<usize>,
    cpu_queue: VecDeque<usize>,
    gpu_queue: VecDeque<usize>,
}

thread_local! {
    static ENCODE_SCRATCH: RefCell<EncodeScratch> = RefCell::new(EncodeScratch::new());
    static BUILD_SCRATCH: RefCell<BuildScratch> = RefCell::new(BuildScratch::default());
    static DECODE_SCRATCH: RefCell<DecodeScratch> = RefCell::new(DecodeScratch::default());
    static FINALIZE_SCRATCH: RefCell<FinalizeScratch> = RefCell::new(FinalizeScratch::default());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PDeflateHybridSchedulerPolicy {
    GlobalQueue,
    GpuLedSplitQueue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PDeflateCompressionMode {
    Speed,
    Balanced,
    Ratio,
}

#[derive(Debug, Clone)]
pub struct PDeflateOptions {
    pub chunk_size: usize,
    pub section_count: usize,
    pub max_table_entries: usize,
    pub max_table_entry_len: usize,
    pub max_ref_len: usize,
    pub min_ref_len: usize,
    pub match_probe_limit: usize,
    pub hash_history_limit: usize,
    pub table_sample_stride: usize,
    pub gpu_table_sample_stride: usize,
    pub gpu_compress_enabled: bool,
    pub gpu_decompress_enabled: bool,
    pub gpu_decompress_force_gpu: bool,
    pub gpu_workers: usize,
    pub gpu_slot_count: usize,
    pub gpu_submit_chunks: usize,
    pub gpu_pipelined_submit_chunks: usize,
    pub gpu_min_chunk_size: usize,
    pub gpu_tail_stop_ratio: f32,
    pub parallel_read_threads: usize,
    pub parallel_write_threads: usize,
    pub huffman_encode_enabled: bool,
    pub compression_mode: PDeflateCompressionMode,
    pub hybrid_scheduler_policy: PDeflateHybridSchedulerPolicy,
}

impl Default for PDeflateOptions {
    fn default() -> Self {
        let parallel_threads = thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
            .max(1);
        Self {
            chunk_size: 4 * 1024 * 1024,
            section_count: 128,
            max_table_entries: 256,
            max_table_entry_len: 32,
            max_ref_len: MAX_CMD_LEN,
            min_ref_len: 3,
            match_probe_limit: 16,
            hash_history_limit: 16,
            table_sample_stride: 4,
            gpu_table_sample_stride: 16,
            gpu_compress_enabled: true,
            gpu_decompress_enabled: true,
            gpu_decompress_force_gpu: false,
            gpu_workers: 1,
            gpu_slot_count: 16,
            gpu_submit_chunks: 4,
            gpu_pipelined_submit_chunks: 4,
            gpu_min_chunk_size: 64 * 1024,
            gpu_tail_stop_ratio: 1.0,
            parallel_read_threads: parallel_threads,
            parallel_write_threads: parallel_threads,
            huffman_encode_enabled: false,
            compression_mode: PDeflateCompressionMode::Speed,
            hybrid_scheduler_policy: PDeflateHybridSchedulerPolicy::GlobalQueue,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PDeflateStats {
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub chunk_count: usize,
    pub table_entries_total: usize,
    pub section_count_total: usize,
}

#[derive(Debug, Error)]
pub enum PDeflateError {
    #[error("invalid options: {0}")]
    InvalidOptions(&'static str),
    #[error("invalid stream: {0}")]
    InvalidStream(&'static str),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("numeric overflow")]
    NumericOverflow,
    #[error("gpu error: {0}")]
    Gpu(String),
}

#[derive(Debug)]
struct ChunkCompressed {
    payload: Vec<u8>,
    table_entries: usize,
    section_count: usize,
    profile: ChunkEncodeProfile,
}

#[derive(Debug)]
struct ChunkDecoded {
    table_entries: usize,
    section_count: usize,
    profile: ChunkDecodeProfile,
}

#[derive(Debug, Clone, Copy)]
struct DecodeChunkTask {
    payload_offset: usize,
    payload_len: usize,
    out_offset: usize,
    out_len: usize,
}

#[derive(Debug)]
struct ChunkDecodePreprocess {
    table_count: usize,
    chunk_uncompressed_len: usize,
    section_meta: Vec<gpu::GpuDecodeSectionMeta>,
}

#[derive(Debug, Clone)]
struct StreamHeader {
    flags: u16,
    chunk_size: usize,
    uncompressed_size: Option<u64>,
    file_name: Option<String>,
    encoded_len: usize,
}

struct GpuPreparedChunk<'a> {
    index: usize,
    chunk: &'a [u8],
    table: Option<Vec<Vec<u8>>>,
    gpu_table: Option<gpu::GpuPackedTableDevice>,
    table_count: usize,
    table_index: Vec<u8>,
    table_data: Vec<u8>,
    table_profile: BuildTableProfile,
    table_build_ms: f64,
    max_ref_len: usize,
    gpu_eligible: bool,
}

struct BuiltTableDispatch {
    table: Option<Vec<Vec<u8>>>,
    gpu_table: Option<gpu::GpuPackedTableDevice>,
    table_count: usize,
    table_index: Vec<u8>,
    table_data: Vec<u8>,
    profile: BuildTableProfile,
    build_ms: f64,
}

#[derive(Debug, Clone, Default)]
struct GpuBatchCompressProfile {
    chunks_total: usize,
    gpu_job_chunks: usize,
    gpu_used_chunks: usize,
    gpu_integrated_chunks: usize,
    gpu_fallback_match_chunks: usize,
    fallback_integrated_error_chunks: usize,
    fallback_integrated_invalid_options_chunks: usize,
    fallback_integrated_invalid_stream_chunks: usize,
    fallback_integrated_numeric_overflow_chunks: usize,
    fallback_integrated_gpu_error_chunks: usize,
    fallback_integrated_other_error_chunks: usize,
    fallback_integrated_output_mismatch_chunks: usize,
    fallback_match_error_chunks: usize,
    fallback_match_output_mismatch_chunks: usize,
    fallback_match_packed_len_mismatch_chunks: usize,
    fallback_integrated_error_example: Option<String>,
    prepare_ms: f64,
    prepare_table_build_ms: f64,
    prepare_table_readback_ms: f64,
    prepare_misc_ms: f64,
    prepare_gpu_table_build_chunks: usize,
    prepare_gpu_table_readback_chunks: usize,
    gpu_call_ms: f64,
    gpu_match_total_ms: f64,
    gpu_match_upload_ms: f64,
    gpu_match_wait_ms: f64,
    gpu_match_map_copy_ms: f64,
    gpu_sparse_pack_upload_ms: f64,
    gpu_sparse_pack_wait_ms: f64,
    gpu_sparse_pack_map_copy_ms: f64,
    gpu_upload_ms: f64,
    gpu_wait_ms: f64,
    gpu_map_copy_ms: f64,
    gpu_section_encode_upload_ms: f64,
    gpu_section_encode_wait_ms: f64,
    gpu_section_encode_map_copy_ms: f64,
    gpu_section_encode_total_ms: f64,
    gpu_pack_inputs_ms: f64,
    gpu_pack_alloc_setup_ms: f64,
    gpu_pack_resolve_sizes_ms: f64,
    gpu_pack_resolve_scan_ms: f64,
    gpu_pack_resolve_readback_setup_ms: f64,
    gpu_pack_resolve_submit_ms: f64,
    gpu_pack_resolve_map_wait_ms: f64,
    gpu_pack_resolve_parse_ms: f64,
    gpu_pack_src_copy_ms: f64,
    gpu_pack_metadata_loop_ms: f64,
    gpu_pack_host_copy_ms: f64,
    gpu_pack_device_copy_plan_ms: f64,
    gpu_pack_finalize_ms: f64,
    gpu_scratch_acquire_ms: f64,
    gpu_match_table_copy_ms: f64,
    gpu_match_prepare_dispatch_ms: f64,
    gpu_match_kernel_dispatch_ms: f64,
    gpu_match_submit_ms: f64,
    gpu_section_setup_ms: f64,
    gpu_section_pass_dispatch_ms: f64,
    gpu_section_tokenize_dispatch_ms: f64,
    gpu_section_prefix_dispatch_ms: f64,
    gpu_section_scatter_dispatch_ms: f64,
    gpu_section_pack_dispatch_ms: f64,
    gpu_section_meta_dispatch_ms: f64,
    gpu_section_copy_dispatch_ms: f64,
    gpu_section_submit_ms: f64,
    gpu_section_tokenize_wait_ms: f64,
    gpu_section_prefix_wait_ms: f64,
    gpu_section_scatter_wait_ms: f64,
    gpu_section_pack_wait_ms: f64,
    gpu_section_meta_wait_ms: f64,
    gpu_section_wrap_ms: f64,
    gpu_sparse_lens_submit_ms: f64,
    gpu_sparse_lens_submit_done_wait_ms: f64,
    gpu_sparse_lens_map_after_done_ms: f64,
    gpu_sparse_lens_poll_calls: u64,
    gpu_sparse_lens_yield_calls: u64,
    gpu_sparse_lens_wait_ms: f64,
    gpu_sparse_lens_copy_ms: f64,
    gpu_sparse_prepare_ms: f64,
    gpu_sparse_table_size_resolve_ms: f64,
    gpu_sparse_prepare_misc_ms: f64,
    gpu_sparse_scratch_acquire_ms: f64,
    gpu_sparse_upload_dispatch_ms: f64,
    gpu_sparse_submit_ms: f64,
    gpu_sparse_wait_ms: f64,
    gpu_sparse_copy_ms: f64,
    gpu_sparse_total_ms: f64,
    gpu_call_pack_bind_group_ms: f64,
    gpu_call_encoder_create_ms: f64,
    gpu_call_table_stream_copy_ms: f64,
    gpu_call_match_clear_ms: f64,
    gpu_call_sparse_prepare_dispatch_ms: f64,
    gpu_call_sparse_pack_dispatch_ms: f64,
    gpu_call_result_readback_setup_ms: f64,
    gpu_call_payload_readback_setup_ms: f64,
    gpu_post_total_pack_bytes_ms: f64,
    gpu_post_scale_profiles_ms: f64,
    gpu_post_chunk_materialize_ms: f64,
    finalize_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Clone, Default)]
struct HybridCompressSchedulerProfile {
    cpu_chunks: usize,
    gpu_batches: usize,
    gpu_chunks_claimed: usize,
    gpu_chunks_encoded: usize,
    gpu_job_chunks: usize,
    gpu_used_chunks: usize,
    gpu_integrated_chunks: usize,
    gpu_fallback_match_chunks: usize,
    fallback_integrated_error_chunks: usize,
    fallback_integrated_invalid_options_chunks: usize,
    fallback_integrated_invalid_stream_chunks: usize,
    fallback_integrated_numeric_overflow_chunks: usize,
    fallback_integrated_gpu_error_chunks: usize,
    fallback_integrated_other_error_chunks: usize,
    fallback_integrated_output_mismatch_chunks: usize,
    fallback_match_error_chunks: usize,
    fallback_match_output_mismatch_chunks: usize,
    fallback_match_packed_len_mismatch_chunks: usize,
    fallback_integrated_error_example: Option<String>,
    cpu_lock_wait_ms: f64,
    gpu_lock_wait_ms: f64,
    cpu_encode_ms: f64,
    gpu_worker_busy_ms: f64,
    cpu_result_lock_wait_ms: f64,
    gpu_result_lock_wait_ms: f64,
    cpu_no_task_events: usize,
    gpu_no_task_events: usize,
    gpu_prepare_ms: f64,
    gpu_call_ms: f64,
    gpu_match_total_ms: f64,
    gpu_match_upload_ms: f64,
    gpu_match_wait_ms: f64,
    gpu_match_map_copy_ms: f64,
    gpu_sparse_pack_upload_ms: f64,
    gpu_sparse_pack_wait_ms: f64,
    gpu_sparse_pack_map_copy_ms: f64,
    gpu_upload_ms: f64,
    gpu_wait_ms: f64,
    gpu_map_copy_ms: f64,
    gpu_section_encode_upload_ms: f64,
    gpu_section_encode_wait_ms: f64,
    gpu_section_encode_map_copy_ms: f64,
    gpu_section_encode_total_ms: f64,
    gpu_pack_inputs_ms: f64,
    gpu_pack_alloc_setup_ms: f64,
    gpu_pack_resolve_sizes_ms: f64,
    gpu_pack_resolve_scan_ms: f64,
    gpu_pack_resolve_readback_setup_ms: f64,
    gpu_pack_resolve_submit_ms: f64,
    gpu_pack_resolve_map_wait_ms: f64,
    gpu_pack_resolve_parse_ms: f64,
    gpu_pack_src_copy_ms: f64,
    gpu_pack_metadata_loop_ms: f64,
    gpu_pack_host_copy_ms: f64,
    gpu_pack_device_copy_plan_ms: f64,
    gpu_pack_finalize_ms: f64,
    gpu_scratch_acquire_ms: f64,
    gpu_match_table_copy_ms: f64,
    gpu_match_prepare_dispatch_ms: f64,
    gpu_match_kernel_dispatch_ms: f64,
    gpu_match_submit_ms: f64,
    gpu_section_setup_ms: f64,
    gpu_section_pass_dispatch_ms: f64,
    gpu_section_tokenize_dispatch_ms: f64,
    gpu_section_prefix_dispatch_ms: f64,
    gpu_section_scatter_dispatch_ms: f64,
    gpu_section_pack_dispatch_ms: f64,
    gpu_section_meta_dispatch_ms: f64,
    gpu_section_copy_dispatch_ms: f64,
    gpu_section_submit_ms: f64,
    gpu_section_tokenize_wait_ms: f64,
    gpu_section_prefix_wait_ms: f64,
    gpu_section_scatter_wait_ms: f64,
    gpu_section_pack_wait_ms: f64,
    gpu_section_meta_wait_ms: f64,
    gpu_section_wrap_ms: f64,
    gpu_sparse_lens_submit_ms: f64,
    gpu_sparse_lens_submit_done_wait_ms: f64,
    gpu_sparse_lens_map_after_done_ms: f64,
    gpu_sparse_lens_poll_calls: u64,
    gpu_sparse_lens_yield_calls: u64,
    gpu_sparse_lens_wait_ms: f64,
    gpu_sparse_lens_copy_ms: f64,
    gpu_sparse_prepare_ms: f64,
    gpu_sparse_table_size_resolve_ms: f64,
    gpu_sparse_prepare_misc_ms: f64,
    gpu_sparse_scratch_acquire_ms: f64,
    gpu_sparse_upload_dispatch_ms: f64,
    gpu_sparse_submit_ms: f64,
    gpu_sparse_wait_ms: f64,
    gpu_sparse_copy_ms: f64,
    gpu_sparse_total_ms: f64,
    gpu_call_pack_bind_group_ms: f64,
    gpu_call_encoder_create_ms: f64,
    gpu_call_table_stream_copy_ms: f64,
    gpu_call_match_clear_ms: f64,
    gpu_call_sparse_prepare_dispatch_ms: f64,
    gpu_call_sparse_pack_dispatch_ms: f64,
    gpu_call_result_readback_setup_ms: f64,
    gpu_call_payload_readback_setup_ms: f64,
    gpu_post_total_pack_bytes_ms: f64,
    gpu_post_scale_profiles_ms: f64,
    gpu_post_chunk_materialize_ms: f64,
    gpu_finalize_ms: f64,
    gpu_batch_total_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ChunkEncodeProfile {
    table_build_ms: f64,
    table_freq_ms: f64,
    table_probe_ms: f64,
    table_materialize_ms: f64,
    table_sort_ms: f64,
    table_positions_scanned: u64,
    table_probe_pairs: u64,
    table_match_len_calls: u64,
    table_hash_key_mismatch: u64,
    anchored_short_hits: u64,
    section_encode_ms: f64,
    section_match_search_ms: f64,
    section_best_search_ms: f64,
    section_lookahead_search_ms: f64,
    section_emit_ref_ms: f64,
    section_emit_lit_ms: f64,
    section_find_calls: u64,
    section_best_calls: u64,
    section_lookahead_calls: u64,
    section_candidate_checks: u64,
    section_best_candidate_checks: u64,
    section_lookahead_candidate_checks: u64,
    section_compare_steps: u64,
    section_best_compare_steps: u64,
    section_lookahead_compare_steps: u64,
    section_prefilter_rejects: u64,
    section_no_prefix_fast_skips: u64,
    section_best_tail_rejects: u64,
    section_ref_cmds: u64,
    section_lit_cmds: u64,
    section_ref_bytes: u64,
    section_lit_bytes: u64,
    gpu_used: bool,
    gpu_upload_ms: f64,
    gpu_wait_ms: f64,
    gpu_map_copy_ms: f64,
    gpu_total_ms: f64,
    header_pack_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct BuildTableProfile {
    freq_ms: f64,
    probe_ms: f64,
    materialize_ms: f64,
    sort_ms: f64,
    positions_scanned: u64,
    probe_pairs: u64,
    match_len_calls: u64,
    hash_key_mismatch: u64,
    anchored_short_hits: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct SectionEncodeProfile {
    match_search_ms: f64,
    best_search_ms: f64,
    lookahead_search_ms: f64,
    emit_ref_ms: f64,
    emit_lit_ms: f64,
    find_calls: u64,
    best_calls: u64,
    lookahead_calls: u64,
    candidate_checks: u64,
    best_candidate_checks: u64,
    lookahead_candidate_checks: u64,
    compare_steps: u64,
    best_compare_steps: u64,
    lookahead_compare_steps: u64,
    prefilter_rejects: u64,
    no_prefix_fast_skips: u64,
    best_tail_rejects: u64,
    ref_cmds: u64,
    lit_cmds: u64,
    ref_bytes: u64,
    lit_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ChunkDecodeProfile {
    table_prepare_ms: f64,
    section_decode_ms: f64,
    cmd_count: u64,
    ref_cmds: u64,
    lit_cmds: u64,
    ref_bytes: u64,
    lit_bytes: u64,
    ref_copy_ms: f64,
    lit_copy_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct FindMetrics {
    candidate_checks: u64,
    compare_steps: u64,
    prefilter_rejects: u64,
    no_prefix_fast_skips: u64,
    tail_rejects: u64,
}

pub fn pdeflate_compress(
    input: &[u8],
    options: &PDeflateOptions,
) -> Result<Vec<u8>, PDeflateError> {
    pdeflate_compress_with_stats(input, options).map(|(out, _)| out)
}

pub(crate) fn pdeflate_compress_reader_with_stats<R: Read + Send, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &PDeflateOptions,
    uncompressed_size_hint: Option<u64>,
    file_name_hint: Option<&str>,
    target_buffered_bytes: usize,
    low_watermark_bytes: usize,
) -> Result<PDeflateStats, PDeflateError> {
    stream::compress_reader_with_stats(
        reader,
        writer,
        options,
        uncompressed_size_hint,
        file_name_hint,
        target_buffered_bytes,
        low_watermark_bytes,
    )
}

fn compress_stream_chunk_batch(
    chunks: &[Vec<u8>],
    options: &PDeflateOptions,
) -> Result<Vec<ChunkCompressed>, PDeflateError> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_size = options.chunk_size.max(1);
    let mut flat_input =
        Vec::with_capacity(chunks.iter().map(|chunk| chunk.len()).sum::<usize>().max(chunk_size));
    for (idx, chunk) in chunks.iter().enumerate() {
        if idx + 1 != chunks.len() && chunk.len() != chunk_size {
            return Err(PDeflateError::InvalidStream(
                "non-final stream chunk shorter than chunk size",
            ));
        }
        flat_input.extend_from_slice(chunk);
    }

    compress_chunks_hybrid(&flat_input, chunk_size, chunks.len(), options)
}

pub(crate) fn pdeflate_decompress_reader_with_stats<R: Read + Send, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &PDeflateOptions,
    decode_backlog_reporter: Option<crate::DecodeBacklogReporter>,
    output_write_reporter: Option<crate::OutputWriteReporter>,
) -> Result<PDeflateStats, PDeflateError> {
    stream::decompress_reader_with_stats(
        reader,
        writer,
        options,
        decode_backlog_reporter,
        output_write_reporter,
    )
}

pub(crate) fn pdeflate_decompress_file_parallel_write_with_stats(
    input_file: std::fs::File,
    output_file: std::fs::File,
    options: &PDeflateOptions,
    decode_backlog_reporter: Option<crate::DecodeBacklogReporter>,
    output_write_reporter: Option<crate::OutputWriteReporter>,
) -> Result<PDeflateStats, PDeflateError> {
    stream::decompress_file_parallel_write_with_stats(
        input_file,
        output_file,
        options,
        decode_backlog_reporter,
        output_write_reporter,
    )
}

pub fn pdeflate_decompress(stream: &[u8]) -> Result<Vec<u8>, PDeflateError> {
    pdeflate_decompress_with_stats(stream).map(|(out, _)| out)
}

fn encode_stream_header(
    chunk_size: usize,
    uncompressed_size: Option<u64>,
    file_name: Option<&str>,
) -> Result<Vec<u8>, PDeflateError> {
    let mut flags = 0u16;
    let file_name_bytes = file_name.map(str::as_bytes);
    let mut header = Vec::with_capacity(
        STREAM_HEADER_SIZE
            + if uncompressed_size.is_some() {
                STREAM_HEADER_UNCOMPRESSED_SIZE_BYTES
            } else {
                0
            }
            + file_name_bytes.map(|bytes| 2 + bytes.len()).unwrap_or(0),
    );
    header.extend_from_slice(&STREAM_MAGIC);
    write_u16_le(&mut header, STREAM_VERSION);
    if uncompressed_size.is_some() {
        flags |= STREAM_FLAG_UNCOMPRESSED_SIZE;
    }
    if file_name_bytes.is_some() {
        flags |= STREAM_FLAG_FILE_NAME_PRESENT;
    }
    write_u16_le(&mut header, flags);
    write_u32_le(
        &mut header,
        u32::try_from(chunk_size).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    if let Some(uncompressed_size) = uncompressed_size {
        header.extend_from_slice(&uncompressed_size.to_le_bytes());
    }
    if let Some(file_name_bytes) = file_name_bytes {
        write_u16_le(
            &mut header,
            u16::try_from(file_name_bytes.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        );
        header.extend_from_slice(file_name_bytes);
    }
    Ok(header)
}

fn read_stream_header<R: Read>(reader: &mut R) -> Result<StreamHeader, PDeflateError> {
    let mut stream_header = [0u8; STREAM_HEADER_SIZE];
    reader.read_exact(&mut stream_header)?;
    let mut cursor = 0usize;
    let magic = read_exact(&stream_header, &mut cursor, 4)?;
    if magic != STREAM_MAGIC {
        return Err(PDeflateError::InvalidStream("bad stream magic"));
    }

    let version = read_u16_le(&stream_header, &mut cursor)?;
    if version != STREAM_VERSION {
        return Err(PDeflateError::InvalidStream("unsupported stream version"));
    }
    let flags = read_u16_le(&stream_header, &mut cursor)?;
    let chunk_size = read_u32_le(&stream_header, &mut cursor)? as usize;

    let uncompressed_size = if (flags & STREAM_FLAG_UNCOMPRESSED_SIZE) != 0 {
        let mut size_bytes = [0u8; STREAM_HEADER_UNCOMPRESSED_SIZE_BYTES];
        reader.read_exact(&mut size_bytes)?;
        Some(u64::from_le_bytes(size_bytes))
    } else {
        None
    };
    let file_name = if (flags & STREAM_FLAG_FILE_NAME_PRESENT) != 0 {
        let mut len_bytes = [0u8; 2];
        reader.read_exact(&mut len_bytes)?;
        let len = u16::from_le_bytes(len_bytes) as usize;
        let mut name_bytes = vec![0u8; len];
        reader.read_exact(&mut name_bytes)?;
        Some(
            String::from_utf8(name_bytes)
                .map_err(|_| PDeflateError::InvalidStream("stream filename is not utf-8"))?,
        )
    } else {
        None
    };
    let file_name_encoded_len = file_name
        .as_ref()
        .map(|name| 2 + name.len())
        .unwrap_or(0);

    Ok(StreamHeader {
        flags,
        chunk_size,
        uncompressed_size,
        file_name,
        encoded_len: STREAM_HEADER_SIZE
            + if uncompressed_size.is_some() {
                STREAM_HEADER_UNCOMPRESSED_SIZE_BYTES
            } else {
                0
            }
            + file_name_encoded_len,
    })
}

pub fn pdeflate_stream_uncompressed_size<R: Read + Seek>(
    reader: &mut R,
) -> Result<Option<u64>, PDeflateError> {
    let start = reader.stream_position()?;
    let result = read_stream_header(reader).map(|header| header.uncompressed_size);
    reader.seek(SeekFrom::Start(start))?;
    result
}

pub fn pdeflate_stream_suggested_name<R: Read + Seek>(
    reader: &mut R,
) -> Result<Option<String>, PDeflateError> {
    let start = reader.stream_position()?;
    let result = read_stream_header(reader).map(|header| header.file_name);
    reader.seek(SeekFrom::Start(start))?;
    result
}

pub fn pdeflate_decompress_into(stream: &[u8], output: &mut Vec<u8>) -> Result<(), PDeflateError> {
    pdeflate_decompress_into_with_stats(stream, output).map(|_| ())
}

pub fn pdeflate_gpu_init() -> bool {
    gpu::is_runtime_available()
}

fn classify_integrated_batch_error(err: &PDeflateError) -> &'static str {
    match err {
        PDeflateError::InvalidOptions(_) => "invalid_options",
        PDeflateError::InvalidStream(_) => "invalid_stream",
        PDeflateError::NumericOverflow => "numeric_overflow",
        PDeflateError::Io(_) => "io_error",
        PDeflateError::Gpu(_) => "gpu_error",
    }
}

pub fn pdeflate_compress_with_stats(
    input: &[u8],
    options: &PDeflateOptions,
) -> Result<(Vec<u8>, PDeflateStats), PDeflateError> {
    validate_options(options)?;

    let chunk_size = options.chunk_size;
    let chunk_count = if input.is_empty() {
        0
    } else {
        input.len().div_ceil(chunk_size)
    };

    let mut out = Vec::with_capacity(input.len() / 2);
    out.extend_from_slice(&STREAM_MAGIC);
    write_u16_le(&mut out, STREAM_VERSION);
    write_u16_le(&mut out, 0);
    write_u32_le(
        &mut out,
        u32::try_from(chunk_size).map_err(|_| PDeflateError::NumericOverflow)?,
    );

    let mut stats = PDeflateStats {
        input_bytes: u64::try_from(input.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        chunk_count,
        ..PDeflateStats::default()
    };
    let profile = profile_enabled();
    let profile_detail = profile_detail_enabled();
    let mut prof_table_build_ms = 0.0_f64;
    let mut prof_table_freq_ms = 0.0_f64;
    let mut prof_table_probe_ms = 0.0_f64;
    let mut prof_table_materialize_ms = 0.0_f64;
    let mut prof_table_sort_ms = 0.0_f64;
    let mut prof_table_positions_scanned = 0_u64;
    let mut prof_table_probe_pairs = 0_u64;
    let mut prof_table_match_len_calls = 0_u64;
    let mut prof_table_hash_key_mismatch = 0_u64;
    let mut prof_table_anchored_short_hits = 0_u64;
    let mut prof_section_encode_ms = 0.0_f64;
    let mut prof_section_match_search_ms = 0.0_f64;
    let mut prof_section_best_search_ms = 0.0_f64;
    let mut prof_section_lookahead_search_ms = 0.0_f64;
    let mut prof_section_emit_ref_ms = 0.0_f64;
    let mut prof_section_emit_lit_ms = 0.0_f64;
    let mut prof_section_find_calls = 0_u64;
    let mut prof_section_best_calls = 0_u64;
    let mut prof_section_lookahead_calls = 0_u64;
    let mut prof_section_candidate_checks = 0_u64;
    let mut prof_section_best_candidate_checks = 0_u64;
    let mut prof_section_lookahead_candidate_checks = 0_u64;
    let mut prof_section_compare_steps = 0_u64;
    let mut prof_section_best_compare_steps = 0_u64;
    let mut prof_section_lookahead_compare_steps = 0_u64;
    let mut prof_section_prefilter_rejects = 0_u64;
    let mut prof_section_no_prefix_fast_skips = 0_u64;
    let mut prof_section_best_tail_rejects = 0_u64;
    let mut prof_section_ref_cmds = 0_u64;
    let mut prof_section_lit_cmds = 0_u64;
    let mut prof_section_ref_bytes = 0_u64;
    let mut prof_section_lit_bytes = 0_u64;
    let mut prof_gpu_chunks = 0usize;
    let mut prof_gpu_upload_ms = 0.0_f64;
    let mut prof_gpu_wait_ms = 0.0_f64;
    let mut prof_gpu_map_copy_ms = 0.0_f64;
    let mut prof_gpu_total_ms = 0.0_f64;
    let mut prof_header_pack_ms = 0.0_f64;
    let mut prof_total_ms = 0.0_f64;
    let mut prof_chunk_encode_suppressed = 0usize;

    // Always use the global-queue worker path so CPU_ONLY also scales with
    // available_parallelism. When GPU is disabled this becomes CPU workers only.
    let compressed_chunks = compress_chunks_hybrid(input, chunk_size, chunk_count, options)?;

    for (chunk_idx, mut compressed) in compressed_chunks.into_iter().enumerate() {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(input.len());
        let chunk_len = end.saturating_sub(start);
        if chunk_idx + 1 == chunk_count {
            set_chunk_final_stream_flag(&mut compressed.payload, true)?;
        }

        write_stream_chunk_frame(&mut out, &compressed.payload)?;

        stats.table_entries_total = stats
            .table_entries_total
            .saturating_add(compressed.table_entries);
        stats.section_count_total = stats
            .section_count_total
            .saturating_add(compressed.section_count);
        prof_table_build_ms += compressed.profile.table_build_ms;
        prof_table_freq_ms += compressed.profile.table_freq_ms;
        prof_table_probe_ms += compressed.profile.table_probe_ms;
        prof_table_materialize_ms += compressed.profile.table_materialize_ms;
        prof_table_sort_ms += compressed.profile.table_sort_ms;
        prof_table_positions_scanned =
            prof_table_positions_scanned.saturating_add(compressed.profile.table_positions_scanned);
        prof_table_probe_pairs =
            prof_table_probe_pairs.saturating_add(compressed.profile.table_probe_pairs);
        prof_table_match_len_calls =
            prof_table_match_len_calls.saturating_add(compressed.profile.table_match_len_calls);
        prof_table_hash_key_mismatch =
            prof_table_hash_key_mismatch.saturating_add(compressed.profile.table_hash_key_mismatch);
        prof_table_anchored_short_hits =
            prof_table_anchored_short_hits.saturating_add(compressed.profile.anchored_short_hits);
        prof_section_encode_ms += compressed.profile.section_encode_ms;
        prof_section_match_search_ms += compressed.profile.section_match_search_ms;
        prof_section_best_search_ms += compressed.profile.section_best_search_ms;
        prof_section_lookahead_search_ms += compressed.profile.section_lookahead_search_ms;
        prof_section_emit_ref_ms += compressed.profile.section_emit_ref_ms;
        prof_section_emit_lit_ms += compressed.profile.section_emit_lit_ms;
        prof_section_find_calls =
            prof_section_find_calls.saturating_add(compressed.profile.section_find_calls);
        prof_section_best_calls =
            prof_section_best_calls.saturating_add(compressed.profile.section_best_calls);
        prof_section_lookahead_calls =
            prof_section_lookahead_calls.saturating_add(compressed.profile.section_lookahead_calls);
        prof_section_candidate_checks = prof_section_candidate_checks
            .saturating_add(compressed.profile.section_candidate_checks);
        prof_section_best_candidate_checks = prof_section_best_candidate_checks
            .saturating_add(compressed.profile.section_best_candidate_checks);
        prof_section_lookahead_candidate_checks = prof_section_lookahead_candidate_checks
            .saturating_add(compressed.profile.section_lookahead_candidate_checks);
        prof_section_compare_steps =
            prof_section_compare_steps.saturating_add(compressed.profile.section_compare_steps);
        prof_section_best_compare_steps = prof_section_best_compare_steps
            .saturating_add(compressed.profile.section_best_compare_steps);
        prof_section_lookahead_compare_steps = prof_section_lookahead_compare_steps
            .saturating_add(compressed.profile.section_lookahead_compare_steps);
        prof_section_prefilter_rejects = prof_section_prefilter_rejects
            .saturating_add(compressed.profile.section_prefilter_rejects);
        prof_section_no_prefix_fast_skips = prof_section_no_prefix_fast_skips
            .saturating_add(compressed.profile.section_no_prefix_fast_skips);
        prof_section_best_tail_rejects = prof_section_best_tail_rejects
            .saturating_add(compressed.profile.section_best_tail_rejects);
        prof_section_ref_cmds =
            prof_section_ref_cmds.saturating_add(compressed.profile.section_ref_cmds);
        prof_section_lit_cmds =
            prof_section_lit_cmds.saturating_add(compressed.profile.section_lit_cmds);
        prof_section_ref_bytes =
            prof_section_ref_bytes.saturating_add(compressed.profile.section_ref_bytes);
        prof_section_lit_bytes =
            prof_section_lit_bytes.saturating_add(compressed.profile.section_lit_bytes);
        if compressed.profile.gpu_used {
            prof_gpu_chunks = prof_gpu_chunks.saturating_add(1);
        }
        prof_gpu_upload_ms += compressed.profile.gpu_upload_ms;
        prof_gpu_wait_ms += compressed.profile.gpu_wait_ms;
        prof_gpu_map_copy_ms += compressed.profile.gpu_map_copy_ms;
        prof_gpu_total_ms += compressed.profile.gpu_total_ms;
        prof_header_pack_ms += compressed.profile.header_pack_ms;
        prof_total_ms += compressed.profile.total_ms;

        let emit_chunk_detail =
            profile_detail && profile_detail_should_log_indexed(chunk_idx, chunk_count);
        if profile_detail && !emit_chunk_detail {
            prof_chunk_encode_suppressed = prof_chunk_encode_suppressed.saturating_add(1);
        }
        if emit_chunk_detail {
            eprintln!(
                "[cozip_pdeflate][timing][chunk-encode] idx={} in_kib={:.2} out_kib={:.2} sections={} table_entries={} gpu_used={} t_table_build_ms={:.3} t_section_encode_ms={:.3} t_gpu_total_ms={:.3} t_header_pack_ms={:.3} t_total_ms={:.3}",
                chunk_idx,
                (chunk_len as f64) / 1024.0,
                (compressed.payload.len() as f64) / 1024.0,
                compressed.section_count,
                compressed.table_entries,
                compressed.profile.gpu_used,
                compressed.profile.table_build_ms,
                compressed.profile.section_encode_ms,
                compressed.profile.gpu_total_ms,
                compressed.profile.header_pack_ms,
                compressed.profile.total_ms
            );
        }
        if emit_chunk_detail {
            eprintln!(
                "[cozip_pdeflate][timing][chunk-encode-detail] idx={} t_table_freq_ms={:.3} t_table_probe_ms={:.3} t_table_materialize_ms={:.3} t_table_sort_ms={:.3} table_positions={} table_probe_pairs={} table_match_len_calls={} table_hash_key_mismatch={} table_anchored_short_hits={} t_match_search_ms={:.3} t_best_search_ms={:.3} t_lookahead_search_ms={:.3} t_emit_ref_ms={:.3} t_emit_lit_ms={:.3} find_calls={} best_calls={} lookahead_calls={} candidate_checks={} best_candidate_checks={} lookahead_candidate_checks={} compare_steps={} best_compare_steps={} lookahead_compare_steps={} prefilter_rejects={} no_prefix_fast_skips={} best_tail_rejects={} ref_cmds={} lit_cmds={} ref_bytes={} lit_bytes={}",
                chunk_idx,
                compressed.profile.table_freq_ms,
                compressed.profile.table_probe_ms,
                compressed.profile.table_materialize_ms,
                compressed.profile.table_sort_ms,
                compressed.profile.table_positions_scanned,
                compressed.profile.table_probe_pairs,
                compressed.profile.table_match_len_calls,
                compressed.profile.table_hash_key_mismatch,
                compressed.profile.anchored_short_hits,
                compressed.profile.section_match_search_ms,
                compressed.profile.section_best_search_ms,
                compressed.profile.section_lookahead_search_ms,
                compressed.profile.section_emit_ref_ms,
                compressed.profile.section_emit_lit_ms,
                compressed.profile.section_find_calls,
                compressed.profile.section_best_calls,
                compressed.profile.section_lookahead_calls,
                compressed.profile.section_candidate_checks,
                compressed.profile.section_best_candidate_checks,
                compressed.profile.section_lookahead_candidate_checks,
                compressed.profile.section_compare_steps,
                compressed.profile.section_best_compare_steps,
                compressed.profile.section_lookahead_compare_steps,
                compressed.profile.section_prefilter_rejects,
                compressed.profile.section_no_prefix_fast_skips,
                compressed.profile.section_best_tail_rejects,
                compressed.profile.section_ref_cmds,
                compressed.profile.section_lit_cmds,
                compressed.profile.section_ref_bytes,
                compressed.profile.section_lit_bytes
            );
        }
    }

    if profile_detail && prof_chunk_encode_suppressed > 0 {
        eprintln!(
            "[cozip_pdeflate][timing][detail-sampling] suppressed_chunk_encode_logs={} total_chunks={}",
            prof_chunk_encode_suppressed, chunk_count
        );
    }

    stats.output_bytes = u64::try_from(out.len()).map_err(|_| PDeflateError::NumericOverflow)?;
    if profile {
        eprintln!(
            "[cozip_pdeflate][timing][compress-total] chunks={} in_mib={:.2} out_mib={:.2} ratio={:.4} gpu_chunks={} t_table_build_ms={:.3} t_section_encode_ms={:.3} t_gpu_upload_ms={:.3} t_gpu_wait_ms={:.3} t_gpu_map_copy_ms={:.3} t_gpu_total_ms={:.3} t_header_pack_ms={:.3} t_total_ms={:.3}",
            chunk_count,
            (input.len() as f64) / (1024.0 * 1024.0),
            (out.len() as f64) / (1024.0 * 1024.0),
            if input.is_empty() {
                0.0
            } else {
                (out.len() as f64) / (input.len() as f64)
            },
            prof_gpu_chunks,
            prof_table_build_ms,
            prof_section_encode_ms,
            prof_gpu_upload_ms,
            prof_gpu_wait_ms,
            prof_gpu_map_copy_ms,
            prof_gpu_total_ms,
            prof_header_pack_ms,
            prof_total_ms
        );
        if profile_detail {
            eprintln!(
                "[cozip_pdeflate][timing][compress-total-detail] t_table_freq_ms={:.3} t_table_probe_ms={:.3} t_table_materialize_ms={:.3} t_table_sort_ms={:.3} table_positions={} table_probe_pairs={} table_match_len_calls={} table_hash_key_mismatch={} table_anchored_short_hits={} t_match_search_ms={:.3} t_best_search_ms={:.3} t_lookahead_search_ms={:.3} t_emit_ref_ms={:.3} t_emit_lit_ms={:.3} find_calls={} best_calls={} lookahead_calls={} candidate_checks={} best_candidate_checks={} lookahead_candidate_checks={} compare_steps={} best_compare_steps={} lookahead_compare_steps={} prefilter_rejects={} no_prefix_fast_skips={} best_tail_rejects={} ref_cmds={} lit_cmds={} ref_bytes={} lit_bytes={}",
                prof_table_freq_ms,
                prof_table_probe_ms,
                prof_table_materialize_ms,
                prof_table_sort_ms,
                prof_table_positions_scanned,
                prof_table_probe_pairs,
                prof_table_match_len_calls,
                prof_table_hash_key_mismatch,
                prof_table_anchored_short_hits,
                prof_section_match_search_ms,
                prof_section_best_search_ms,
                prof_section_lookahead_search_ms,
                prof_section_emit_ref_ms,
                prof_section_emit_lit_ms,
                prof_section_find_calls,
                prof_section_best_calls,
                prof_section_lookahead_calls,
                prof_section_candidate_checks,
                prof_section_best_candidate_checks,
                prof_section_lookahead_candidate_checks,
                prof_section_compare_steps,
                prof_section_best_compare_steps,
                prof_section_lookahead_compare_steps,
                prof_section_prefilter_rejects,
                prof_section_no_prefix_fast_skips,
                prof_section_best_tail_rejects,
                prof_section_ref_cmds,
                prof_section_lit_cmds,
                prof_section_ref_bytes,
                prof_section_lit_bytes
            );
        }
    }
    Ok((out, stats))
}

pub fn pdeflate_decompress_with_stats(
    stream: &[u8],
) -> Result<(Vec<u8>, PDeflateStats), PDeflateError> {
    let mut output = Vec::new();
    let options = PDeflateOptions::default();
    let stats = pdeflate_decompress_into_with_stats_with_options(stream, &mut output, &options)?;
    Ok((output, stats))
}

pub fn pdeflate_decompress_into_with_stats(
    stream: &[u8],
    output: &mut Vec<u8>,
) -> Result<PDeflateStats, PDeflateError> {
    let options = PDeflateOptions::default();
    pdeflate_decompress_into_with_stats_with_options(stream, output, &options)
}

pub fn pdeflate_decompress_into_with_stats_with_options(
    stream: &[u8],
    output: &mut Vec<u8>,
    options: &PDeflateOptions,
) -> Result<PDeflateStats, PDeflateError> {
    let mut cursor = 0usize;
    let magic = read_exact(stream, &mut cursor, 4)?;
    if magic != STREAM_MAGIC {
        return Err(PDeflateError::InvalidStream("bad stream magic"));
    }

    let version = read_u16_le(stream, &mut cursor)?;
    if version != STREAM_VERSION {
        return Err(PDeflateError::InvalidStream("unsupported stream version"));
    }
    let _flags = read_u16_le(stream, &mut cursor)?;
    let _chunk_size = read_u32_le(stream, &mut cursor)?;

    let mut total_output_len = 0usize;
    let profile = profile_enabled();
    let profile_detail = profile_detail_enabled();
    let mut prof_table_prepare_ms = 0.0_f64;
    let mut prof_section_decode_ms = 0.0_f64;
    let mut prof_ref_copy_ms = 0.0_f64;
    let mut prof_lit_copy_ms = 0.0_f64;
    let mut prof_cmd_count = 0_u64;
    let mut prof_ref_cmds = 0_u64;
    let mut prof_lit_cmds = 0_u64;
    let mut prof_ref_bytes = 0_u64;
    let mut prof_lit_bytes = 0_u64;
    let mut prof_cpu_decode_chunks = 0_usize;
    let mut prof_gpu_decode_chunks = 0_usize;
    let prof_gpu_decompress_enabled =
        options.gpu_decompress_enabled || options.gpu_decompress_force_gpu;
    let mut prof_chunk_decode_suppressed = 0usize;
    let prof_parse_tasks_ms;
    let prof_decode_workers_ms;
    let prof_aggregate_stats_ms;
    let mut out_cursor = 0usize;
    let mut chunk_tasks = Vec::new();
    let mut gpu_decode_jobs = Vec::new();
    let parse_tasks_t0 = Instant::now();
    let mut chunk_index = 0usize;
    let mut saw_final_chunk = false;
    while cursor < stream.len() {
        let chunk_len = read_u32_le(stream, &mut cursor)? as usize;
        let payload_offset = cursor;
        let chunk_payload = read_exact(stream, &mut cursor, chunk_len)?;
        let preprocess = preprocess_chunk_for_gpu_decode(chunk_payload)?;
        let chunk_out_len = preprocess.chunk_uncompressed_len;
        let chunk_out_end = out_cursor
            .checked_add(chunk_out_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        chunk_tasks.push(DecodeChunkTask {
            payload_offset,
            payload_len: chunk_len,
            out_offset: out_cursor,
            out_len: chunk_out_len,
        });
        gpu_decode_jobs.push(gpu::GpuDecodeJob {
            chunk_index,
            payload: chunk_payload,
            table_count: preprocess.table_count,
            chunk_uncompressed_len: chunk_out_len,
            out_offset: out_cursor,
            out_len: chunk_out_len,
            section_meta: preprocess.section_meta,
            preferred_slot: None,
        });
        out_cursor = chunk_out_end;
        total_output_len = chunk_out_end;
        if chunk_final_stream_enabled(read_chunk_flags(chunk_payload)?) {
            saw_final_chunk = true;
            break;
        }
        chunk_index = chunk_index.saturating_add(1);
    }
    if cursor != stream.len() {
        return Err(PDeflateError::InvalidStream("trailing bytes in stream"));
    }
    if !chunk_tasks.is_empty() && !saw_final_chunk {
        return Err(PDeflateError::InvalidStream("missing final chunk"));
    }
    let chunk_count = chunk_tasks.len();

    output.clear();
    if output.capacity() < total_output_len {
        output.reserve_exact(total_output_len - output.capacity());
    }
    unsafe {
        output.set_len(total_output_len);
    }
    let mut stats = PDeflateStats {
        input_bytes: u64::try_from(stream.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        chunk_count,
        ..PDeflateStats::default()
    };
    prof_parse_tasks_ms = elapsed_ms(parse_tasks_t0);

    let mut decoded_chunks: Vec<Option<ChunkDecoded>> = (0..chunk_count).map(|_| None).collect();
    let mut gpu_decoded_flags = vec![false; chunk_count];
    let decode_workers_t0 = Instant::now();
    let output_base = output.as_mut_ptr() as usize;
    let gpu_enabled = prof_gpu_decompress_enabled && gpu::is_runtime_available();
    let cpu_worker_count = if options.gpu_decompress_force_gpu {
        0
    } else {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(chunk_count.max(1))
            .max(1)
    };
    let prof_cpu_decode_workers = cpu_worker_count;
    let gpu_batch_limit = gpu_decode_batch_limit(options, chunk_count);
    let queue_state = Arc::new(Mutex::new((0..chunk_count).collect::<VecDeque<usize>>()));
    let err_slot = Arc::new(Mutex::new(None::<PDeflateError>));
    let err_flag = Arc::new(AtomicBool::new(false));
    let mut cpu_worker_profiles: Vec<(usize, usize, f64)> = Vec::with_capacity(cpu_worker_count);
    let mut cpu_worker_outputs: Vec<Vec<(usize, Result<ChunkDecoded, PDeflateError>)>> =
        Vec::with_capacity(cpu_worker_count);
    let mut gpu_worker_output: Vec<(usize, Result<ChunkDecoded, PDeflateError>, bool)> = Vec::new();
    let mut gpu_worker_busy_ms = 0.0_f64;
    let mut gpu_claimed_chunks = 0usize;
    let mut worker_panicked = false;

    thread::scope(|scope| {
        let mut cpu_handles = Vec::with_capacity(cpu_worker_count);
        for worker_id in 0..cpu_worker_count {
            let queue_state = Arc::clone(&queue_state);
            let err_slot = Arc::clone(&err_slot);
            let err_flag = Arc::clone(&err_flag);
            let tasks = &chunk_tasks;
            cpu_handles.push(scope.spawn(move || {
                let mut local = Vec::new();
                let mut chunk_count_local = 0usize;
                let t_worker = Instant::now();
                loop {
                    if err_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let idx = {
                        let mut queue = match queue_state.lock() {
                            Ok(v) => v,
                            Err(_) => {
                                set_worker_error_once(
                                    &err_slot,
                                    &err_flag,
                                    PDeflateError::Gpu("decode queue mutex poisoned".to_string()),
                                );
                                break;
                            }
                        };
                        queue.pop_front()
                    };
                    let Some(idx) = idx else {
                        break;
                    };
                    chunk_count_local = chunk_count_local.saturating_add(1);
                    let task = tasks[idx];
                    let payload =
                        &stream[task.payload_offset..task.payload_offset + task.payload_len];
                    let out_ptr = (output_base + task.out_offset) as *mut u8;
                    let out_slice =
                        unsafe { std::slice::from_raw_parts_mut(out_ptr, task.out_len) };
                    local.push((idx, decompress_chunk_into(payload, out_slice)));
                }
                (local, worker_id, chunk_count_local, elapsed_ms(t_worker))
            }));
        }

        let gpu_handle = if gpu_enabled {
            let queue_state = Arc::clone(&queue_state);
            let err_slot = Arc::clone(&err_slot);
            let err_flag = Arc::clone(&err_flag);
            let tasks = &chunk_tasks;
            let gpu_jobs = &gpu_decode_jobs;
            Some(scope.spawn(move || {
                let mut local = Vec::new();
                let t_gpu_worker = Instant::now();
                loop {
                    if err_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let batch_indices = {
                        let mut queue = match queue_state.lock() {
                            Ok(v) => v,
                            Err(_) => {
                                set_worker_error_once(
                                    &err_slot,
                                    &err_flag,
                                    PDeflateError::Gpu("decode queue mutex poisoned".to_string()),
                                );
                                break;
                            }
                        };
                        let mut picked = Vec::with_capacity(gpu_batch_limit);
                        while picked.len() < gpu_batch_limit {
                            let idx = if options.gpu_decompress_force_gpu {
                                queue.pop_front()
                            } else {
                                // Keep scheduler policy intact while avoiding "front ineligible blocks GPU".
                                // Pick the next GPU-eligible chunk from the global queue.
                                let pos = queue.iter().position(|&idx| {
                                    tasks
                                        .get(idx)
                                        .map(|task| task.out_len >= options.gpu_min_chunk_size)
                                        .unwrap_or(false)
                                });
                                pos.and_then(|p| queue.remove(p))
                            };
                            let Some(idx) = idx else {
                                break;
                            };
                            picked.push(idx);
                        }
                        picked
                    };
                    if batch_indices.is_empty() {
                        break;
                    }
                    let batch_jobs: Vec<_> = batch_indices
                        .iter()
                        .map(|&idx| gpu_jobs[idx].clone())
                        .collect();
                    let gpu_results = match gpu::decode_chunks_gpu_v2(&batch_jobs, options) {
                        Ok(v) => v,
                        Err(err) => {
                            set_worker_error_once(&err_slot, &err_flag, err);
                            break;
                        }
                    };
                    if gpu_results.len() != batch_jobs.len() {
                        set_worker_error_once(
                            &err_slot,
                            &err_flag,
                            PDeflateError::Gpu(
                                "gpu decode v2 returned mismatched result length".to_string(),
                            ),
                        );
                        break;
                    }
                    for result in gpu_results {
                        let chunk_idx = result.chunk_index;
                        let task = match tasks.get(chunk_idx).copied() {
                            Some(v) => v,
                            None => {
                                set_worker_error_once(
                                    &err_slot,
                                    &err_flag,
                                    PDeflateError::InvalidStream(
                                        "gpu decode result chunk index out of bounds",
                                    ),
                                );
                                break;
                            }
                        };
                        match result.disposition {
                            gpu::GpuDecodeDisposition::SubmittedGpu => {
                                if let Some(decoded_chunk) = result.decoded_chunk {
                                    if decoded_chunk.len() == task.out_len {
                                        let out_ptr = (output_base + task.out_offset) as *mut u8;
                                        let out_slice = unsafe {
                                            std::slice::from_raw_parts_mut(out_ptr, task.out_len)
                                        };
                                        out_slice.copy_from_slice(&decoded_chunk);
                                        local.push((
                                            chunk_idx,
                                            Ok(ChunkDecoded {
                                                table_entries: gpu_jobs[chunk_idx].table_count,
                                                section_count: gpu_jobs[chunk_idx]
                                                    .section_meta
                                                    .len(),
                                                profile: ChunkDecodeProfile::default(),
                                            }),
                                            true,
                                        ));
                                        continue;
                                    }
                                }
                                let payload = &stream
                                    [task.payload_offset..task.payload_offset + task.payload_len];
                                let out_ptr = (output_base + task.out_offset) as *mut u8;
                                let out_slice = unsafe {
                                    std::slice::from_raw_parts_mut(out_ptr, task.out_len)
                                };
                                local.push((
                                    chunk_idx,
                                    decompress_chunk_into(payload, out_slice),
                                    false,
                                ));
                            }
                            gpu::GpuDecodeDisposition::CpuFallback => {
                                let payload = &stream
                                    [task.payload_offset..task.payload_offset + task.payload_len];
                                let out_ptr = (output_base + task.out_offset) as *mut u8;
                                let out_slice = unsafe {
                                    std::slice::from_raw_parts_mut(out_ptr, task.out_len)
                                };
                                local.push((
                                    chunk_idx,
                                    decompress_chunk_into(payload, out_slice),
                                    false,
                                ));
                            }
                        }
                    }
                }
                let busy_ms = elapsed_ms(t_gpu_worker);
                let claimed = local.len();
                (local, busy_ms, claimed)
            }))
        } else {
            None
        };

        for handle in cpu_handles {
            match handle.join() {
                Ok((local, worker_id, chunk_count_local, busy_ms)) => {
                    cpu_worker_profiles.push((worker_id, chunk_count_local, busy_ms));
                    cpu_worker_outputs.push(local);
                }
                Err(_) => worker_panicked = true,
            }
        }

        if let Some(handle) = gpu_handle {
            match handle.join() {
                Ok((local, busy_ms, claimed)) => {
                    gpu_worker_output = local;
                    gpu_worker_busy_ms = busy_ms;
                    gpu_claimed_chunks = claimed;
                }
                Err(_) => worker_panicked = true,
            }
        }
    });

    if worker_panicked {
        return Err(PDeflateError::InvalidStream("decode worker panicked"));
    }
    take_worker_error(&err_slot)?;

    for local in cpu_worker_outputs {
        for (idx, result) in local {
            if idx >= decoded_chunks.len() {
                return Err(PDeflateError::InvalidStream(
                    "cpu decode result chunk index out of bounds",
                ));
            }
            if decoded_chunks[idx].is_some() {
                return Err(PDeflateError::InvalidStream("duplicate cpu decode result"));
            }
            decoded_chunks[idx] = Some(result?);
        }
    }
    for (idx, result, from_gpu) in gpu_worker_output {
        if idx >= decoded_chunks.len() {
            return Err(PDeflateError::InvalidStream(
                "gpu decode result chunk index out of bounds",
            ));
        }
        if decoded_chunks[idx].is_some() {
            return Err(PDeflateError::InvalidStream("duplicate gpu decode result"));
        }
        decoded_chunks[idx] = Some(result?);
        gpu_decoded_flags[idx] = from_gpu;
        if from_gpu {
            prof_gpu_decode_chunks = prof_gpu_decode_chunks.saturating_add(1);
        }
    }
    let cpu_claimed_chunks: usize = cpu_worker_profiles
        .iter()
        .map(|(_, chunks, _)| *chunks)
        .sum();
    if profile {
        eprintln!(
            "[cozip_pdeflate][timing][decompress-scheduler-claims] cpu_workers={} cpu_claimed_chunks={} gpu_claimed_chunks={} gpu_submitted_chunks={} gpu_fallback_chunks={} gpu_busy_ms={:.3}",
            cpu_worker_count,
            cpu_claimed_chunks,
            gpu_claimed_chunks,
            prof_gpu_decode_chunks,
            gpu_claimed_chunks.saturating_sub(prof_gpu_decode_chunks),
            gpu_worker_busy_ms
        );
    }
    if profile && !cpu_worker_profiles.is_empty() {
        let mut min_chunks = usize::MAX;
        let mut max_chunks = 0usize;
        let mut min_busy_ms = f64::INFINITY;
        let mut max_busy_ms = 0.0_f64;
        let mut sum_busy_ms = 0.0_f64;
        let mut sum_chunks = 0usize;
        for (_, chunks, busy_ms) in &cpu_worker_profiles {
            min_chunks = min_chunks.min(*chunks);
            max_chunks = max_chunks.max(*chunks);
            min_busy_ms = min_busy_ms.min(*busy_ms);
            max_busy_ms = max_busy_ms.max(*busy_ms);
            sum_busy_ms += *busy_ms;
            sum_chunks = sum_chunks.saturating_add(*chunks);
        }
        let avg_busy_ms = sum_busy_ms / (cpu_worker_profiles.len() as f64);
        let busy_gap_ms = (max_busy_ms - min_busy_ms).max(0.0);
        eprintln!(
            "[cozip_pdeflate][timing][decompress-worker-balance] cpu_workers={} cpu_chunks={} chunk_min={} chunk_max={} busy_sum_ms={:.3} busy_avg_ms={:.3} busy_min_ms={:.3} busy_max_ms={:.3} busy_gap_ms={:.3}",
            cpu_worker_profiles.len(),
            sum_chunks,
            if min_chunks == usize::MAX {
                0
            } else {
                min_chunks
            },
            max_chunks,
            sum_busy_ms,
            avg_busy_ms,
            if min_busy_ms.is_finite() {
                min_busy_ms
            } else {
                0.0
            },
            max_busy_ms,
            busy_gap_ms
        );
        if profile_detail {
            cpu_worker_profiles.sort_by_key(|(worker_id, _, _)| *worker_id);
            for (worker_id, chunks, busy_ms) in cpu_worker_profiles {
                eprintln!(
                    "[cozip_pdeflate][timing][decompress-worker] id={} chunks={} busy_ms={:.3}",
                    worker_id, chunks, busy_ms
                );
            }
        }
    }
    prof_decode_workers_ms = elapsed_ms(decode_workers_t0);

    let aggregate_stats_t0 = Instant::now();
    for (chunk_idx, task) in chunk_tasks.iter().copied().enumerate() {
        let decoded = decoded_chunks[chunk_idx]
            .take()
            .ok_or(PDeflateError::InvalidStream("chunk decode result missing"))?;

        stats.table_entries_total = stats
            .table_entries_total
            .saturating_add(decoded.table_entries);
        stats.section_count_total = stats
            .section_count_total
            .saturating_add(decoded.section_count);
        prof_table_prepare_ms += decoded.profile.table_prepare_ms;
        prof_section_decode_ms += decoded.profile.section_decode_ms;
        prof_ref_copy_ms += decoded.profile.ref_copy_ms;
        prof_lit_copy_ms += decoded.profile.lit_copy_ms;
        prof_cmd_count = prof_cmd_count.saturating_add(decoded.profile.cmd_count);
        prof_ref_cmds = prof_ref_cmds.saturating_add(decoded.profile.ref_cmds);
        prof_lit_cmds = prof_lit_cmds.saturating_add(decoded.profile.lit_cmds);
        prof_ref_bytes = prof_ref_bytes.saturating_add(decoded.profile.ref_bytes);
        prof_lit_bytes = prof_lit_bytes.saturating_add(decoded.profile.lit_bytes);
        if !gpu_decoded_flags[chunk_idx] {
            prof_cpu_decode_chunks = prof_cpu_decode_chunks.saturating_add(1);
        }

        let emit_chunk_detail =
            profile_detail && profile_detail_should_log_indexed(chunk_idx, chunk_count);
        if profile_detail && !emit_chunk_detail {
            prof_chunk_decode_suppressed = prof_chunk_decode_suppressed.saturating_add(1);
        }
        if emit_chunk_detail {
            let backend = if gpu_decoded_flags[chunk_idx] {
                "GPU"
            } else {
                "CPU"
            };
            eprintln!(
                "[cozip_pdeflate][timing][chunk-decode] idx={} backend={} in_kib={:.2} out_kib={:.2} sections={} table_entries={} t_table_prepare_ms={:.3} t_section_decode_ms={:.3} cmds={} ref_cmds={} lit_cmds={} ref_bytes={} lit_bytes={} t_ref_copy_ms={:.3} t_lit_copy_ms={:.3}",
                chunk_idx,
                backend,
                (task.payload_len as f64) / 1024.0,
                (task.out_len as f64) / 1024.0,
                decoded.section_count,
                decoded.table_entries,
                decoded.profile.table_prepare_ms,
                decoded.profile.section_decode_ms,
                decoded.profile.cmd_count,
                decoded.profile.ref_cmds,
                decoded.profile.lit_cmds,
                decoded.profile.ref_bytes,
                decoded.profile.lit_bytes,
                decoded.profile.ref_copy_ms,
                decoded.profile.lit_copy_ms
            );
        }
    }
    prof_aggregate_stats_ms = elapsed_ms(aggregate_stats_t0);

    if profile_detail && prof_chunk_decode_suppressed > 0 {
        eprintln!(
            "[cozip_pdeflate][timing][detail-sampling] suppressed_chunk_decode_logs={} total_chunks={}",
            prof_chunk_decode_suppressed, chunk_count
        );
    }

    stats.output_bytes = u64::try_from(output.len()).map_err(|_| PDeflateError::NumericOverflow)?;
    if profile {
        eprintln!(
            "[cozip_pdeflate][timing][decompress-phases] chunks={} parse_tasks_ms={:.3} decode_workers_ms={:.3} aggregate_stats_ms={:.3}",
            chunk_count, prof_parse_tasks_ms, prof_decode_workers_ms, prof_aggregate_stats_ms
        );
        eprintln!(
            "[cozip_pdeflate][timing][decompress-total] chunks={} cpu_decode_workers={} cpu_decode_chunks={} gpu_decode_chunks={} gpu_decompress_enabled={} gpu_decompress_force_gpu={} in_mib={:.2} out_mib={:.2} t_table_prepare_ms={:.3} t_section_decode_ms={:.3} cmds={} ref_cmds={} lit_cmds={} ref_bytes={} lit_bytes={} t_ref_copy_ms={:.3} t_lit_copy_ms={:.3}",
            chunk_count,
            prof_cpu_decode_workers,
            prof_cpu_decode_chunks,
            prof_gpu_decode_chunks,
            prof_gpu_decompress_enabled,
            options.gpu_decompress_force_gpu,
            (stream.len() as f64) / (1024.0 * 1024.0),
            (output.len() as f64) / (1024.0 * 1024.0),
            prof_table_prepare_ms,
            prof_section_decode_ms,
            prof_cmd_count,
            prof_ref_cmds,
            prof_lit_cmds,
            prof_ref_bytes,
            prof_lit_bytes,
            prof_ref_copy_ms,
            prof_lit_copy_ms
        );
    }
    Ok(stats)
}

fn validate_options(options: &PDeflateOptions) -> Result<(), PDeflateError> {
    if options.chunk_size == 0 {
        return Err(PDeflateError::InvalidOptions("chunk_size must be > 0"));
    }
    if options.section_count == 0 {
        return Err(PDeflateError::InvalidOptions("section_count must be > 0"));
    }
    if options.max_table_entries == 0 {
        return Err(PDeflateError::InvalidOptions(
            "max_table_entries must be > 0",
        ));
    }
    if options.max_table_entry_len == 0 || options.max_table_entry_len > 254 {
        return Err(PDeflateError::InvalidOptions(
            "max_table_entry_len must be in 1..=254",
        ));
    }
    if options.min_ref_len < 3 || options.min_ref_len > MAX_CMD_LEN {
        return Err(PDeflateError::InvalidOptions(
            "min_ref_len must be in 3..=270",
        ));
    }
    if options.max_ref_len < options.min_ref_len || options.max_ref_len > MAX_CMD_LEN {
        return Err(PDeflateError::InvalidOptions(
            "max_ref_len must be in min_ref_len..=270",
        ));
    }
    if options.match_probe_limit == 0 {
        return Err(PDeflateError::InvalidOptions(
            "match_probe_limit must be > 0",
        ));
    }
    if options.hash_history_limit == 0 {
        return Err(PDeflateError::InvalidOptions(
            "hash_history_limit must be > 0",
        ));
    }
    if options.table_sample_stride == 0 {
        return Err(PDeflateError::InvalidOptions(
            "table_sample_stride must be > 0",
        ));
    }
    if options.gpu_table_sample_stride == 0 {
        return Err(PDeflateError::InvalidOptions(
            "gpu_table_sample_stride must be > 0",
        ));
    }
    if options.parallel_read_threads == 0 {
        return Err(PDeflateError::InvalidOptions(
            "parallel_read_threads must be > 0",
        ));
    }

    if options.parallel_write_threads == 0 {
        return Err(PDeflateError::InvalidOptions(
            "parallel_write_threads must be > 0",
        ));
    }
    if options.gpu_submit_chunks == 0 {
        return Err(PDeflateError::InvalidOptions(
            "gpu_submit_chunks must be > 0",
        ));
    }
    if options.gpu_slot_count == 0 {
        return Err(PDeflateError::InvalidOptions("gpu_slot_count must be > 0"));
    }
    if options.gpu_pipelined_submit_chunks == 0 {
        return Err(PDeflateError::InvalidOptions(
            "gpu_pipelined_submit_chunks must be > 0",
        ));
    }
    if options.gpu_workers == 0 {
        return Err(PDeflateError::InvalidOptions("gpu_workers must be > 0"));
    }
    if options.gpu_min_chunk_size == 0 {
        return Err(PDeflateError::InvalidOptions(
            "gpu_min_chunk_size must be > 0",
        ));
    }
    if !(0.0..=1.0).contains(&options.gpu_tail_stop_ratio) {
        return Err(PDeflateError::InvalidOptions(
            "gpu_tail_stop_ratio must be in range 0.0..=1.0",
        ));
    }
    Ok(())
}

fn should_stop_gpu_tail_pop(total_task_count: usize, queue_remaining: usize, ratio: f32) -> bool {
    if ratio >= 1.0 || total_task_count == 0 {
        return false;
    }
    let completed = total_task_count.saturating_sub(queue_remaining);
    (completed as f32 / total_task_count as f32) >= ratio
}

fn cpu_worker_count() -> usize {
    let available = thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1);
    // Keep CPU worker parallelism unchanged in both CPU_ONLY and Hybrid.
    // GPU worker mostly blocks on device progress, so reserving a CPU core here
    // can underutilize CPU on CPU-heavy inputs.
    available.max(1)
}

fn gpu_decode_batch_chunks_override() -> Option<usize> {
    static VALUE: OnceLock<Option<usize>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("COZIP_PDEFLATE_GPU_DECODE_BATCH_CHUNKS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
    })
}

fn gpu_decode_batch_limit(options: &PDeflateOptions, chunk_count: usize) -> usize {
    let base = options
        .gpu_pipelined_submit_chunks
        .max(options.gpu_submit_chunks)
        .max(options.gpu_slot_count)
        .max(1);
    let requested = gpu_decode_batch_chunks_override().unwrap_or(base);
    requested.min(chunk_count.max(1))
}

fn set_worker_error_once(
    err_slot: &Arc<Mutex<Option<PDeflateError>>>,
    err_flag: &AtomicBool,
    err: PDeflateError,
) {
    if let Ok(mut guard) = err_slot.lock() {
        if guard.is_none() {
            *guard = Some(err);
            err_flag.store(true, Ordering::Relaxed);
        }
    }
}

fn take_worker_error(err_slot: &Arc<Mutex<Option<PDeflateError>>>) -> Result<(), PDeflateError> {
    let mut guard = err_slot
        .lock()
        .map_err(|_| PDeflateError::Gpu("worker error mutex poisoned".to_string()))?;
    if let Some(err) = guard.take() {
        return Err(err);
    }
    Ok(())
}

fn compress_chunk_cpu_only(
    chunk: &[u8],
    options: &PDeflateOptions,
) -> Result<ChunkCompressed, PDeflateError> {
    let mut cpu_opts = options.clone();
    cpu_opts.gpu_compress_enabled = false;
    compress_chunk(chunk, &cpu_opts)
}

fn scale_gpu_profile(profile: gpu::GpuMatchProfile, scale: f64) -> gpu::GpuMatchProfile {
    gpu::GpuMatchProfile {
        upload_ms: profile.upload_ms * scale,
        wait_ms: profile.wait_ms * scale,
        map_copy_ms: profile.map_copy_ms * scale,
        total_ms: profile.total_ms * scale,
    }
}

fn scale_gpu_section_profile(
    profile: gpu::GpuSectionEncodeProfile,
    scale: f64,
) -> gpu::GpuSectionEncodeProfile {
    gpu::GpuSectionEncodeProfile {
        upload_ms: profile.upload_ms * scale,
        wait_ms: profile.wait_ms * scale,
        map_copy_ms: profile.map_copy_ms * scale,
        total_ms: profile.total_ms * scale,
    }
}

fn accumulate_gpu_kernel_profile(
    dst: &mut gpu::GpuBatchKernelProfile,
    src: &gpu::GpuBatchKernelProfile,
) {
    dst.pack_inputs_ms += src.pack_inputs_ms;
    dst.pack_alloc_setup_ms += src.pack_alloc_setup_ms;
    dst.pack_resolve_sizes_ms += src.pack_resolve_sizes_ms;
    dst.pack_resolve_scan_ms += src.pack_resolve_scan_ms;
    dst.pack_resolve_readback_setup_ms += src.pack_resolve_readback_setup_ms;
    dst.pack_resolve_submit_ms += src.pack_resolve_submit_ms;
    dst.pack_resolve_map_wait_ms += src.pack_resolve_map_wait_ms;
    dst.pack_resolve_parse_ms += src.pack_resolve_parse_ms;
    dst.pack_src_copy_ms += src.pack_src_copy_ms;
    dst.pack_metadata_loop_ms += src.pack_metadata_loop_ms;
    dst.pack_host_copy_ms += src.pack_host_copy_ms;
    dst.pack_device_copy_plan_ms += src.pack_device_copy_plan_ms;
    dst.pack_finalize_ms += src.pack_finalize_ms;
    dst.scratch_acquire_ms += src.scratch_acquire_ms;
    dst.match_table_copy_ms += src.match_table_copy_ms;
    dst.match_prepare_dispatch_ms += src.match_prepare_dispatch_ms;
    dst.match_kernel_dispatch_ms += src.match_kernel_dispatch_ms;
    dst.match_submit_ms += src.match_submit_ms;
    dst.section_setup_ms += src.section_setup_ms;
    dst.section_pass_dispatch_ms += src.section_pass_dispatch_ms;
    dst.section_tokenize_dispatch_ms += src.section_tokenize_dispatch_ms;
    dst.section_prefix_dispatch_ms += src.section_prefix_dispatch_ms;
    dst.section_scatter_dispatch_ms += src.section_scatter_dispatch_ms;
    dst.section_pack_dispatch_ms += src.section_pack_dispatch_ms;
    dst.section_meta_dispatch_ms += src.section_meta_dispatch_ms;
    dst.section_copy_dispatch_ms += src.section_copy_dispatch_ms;
    dst.section_submit_ms += src.section_submit_ms;
    dst.section_tokenize_wait_ms += src.section_tokenize_wait_ms;
    dst.section_prefix_wait_ms += src.section_prefix_wait_ms;
    dst.section_scatter_wait_ms += src.section_scatter_wait_ms;
    dst.section_pack_wait_ms += src.section_pack_wait_ms;
    dst.section_meta_wait_ms += src.section_meta_wait_ms;
    dst.section_wrap_ms += src.section_wrap_ms;
    dst.sparse_lens_submit_ms += src.sparse_lens_submit_ms;
    dst.sparse_lens_submit_done_wait_ms += src.sparse_lens_submit_done_wait_ms;
    dst.sparse_lens_map_after_done_ms += src.sparse_lens_map_after_done_ms;
    dst.sparse_lens_poll_calls = dst
        .sparse_lens_poll_calls
        .saturating_add(src.sparse_lens_poll_calls);
    dst.sparse_lens_yield_calls = dst
        .sparse_lens_yield_calls
        .saturating_add(src.sparse_lens_yield_calls);
    dst.sparse_lens_wait_ms += src.sparse_lens_wait_ms;
    dst.sparse_lens_copy_ms += src.sparse_lens_copy_ms;
    dst.sparse_prepare_ms += src.sparse_prepare_ms;
    dst.sparse_scratch_acquire_ms += src.sparse_scratch_acquire_ms;
    dst.sparse_upload_dispatch_ms += src.sparse_upload_dispatch_ms;
    dst.sparse_submit_ms += src.sparse_submit_ms;
    dst.sparse_wait_ms += src.sparse_wait_ms;
    dst.sparse_copy_ms += src.sparse_copy_ms;
    dst.sparse_total_ms += src.sparse_total_ms;
    dst.call_pack_bind_group_ms += src.call_pack_bind_group_ms;
    dst.call_encoder_create_ms += src.call_encoder_create_ms;
    dst.call_table_stream_copy_ms += src.call_table_stream_copy_ms;
    dst.call_match_clear_ms += src.call_match_clear_ms;
    dst.call_sparse_prepare_dispatch_ms += src.call_sparse_prepare_dispatch_ms;
    dst.call_sparse_pack_dispatch_ms += src.call_sparse_pack_dispatch_ms;
    dst.call_result_readback_setup_ms += src.call_result_readback_setup_ms;
    dst.call_payload_readback_setup_ms += src.call_payload_readback_setup_ms;
}

fn serialize_table_entries(table: &[Vec<u8>]) -> Result<(Vec<u8>, Vec<u8>), PDeflateError> {
    let mut table_index = Vec::with_capacity(table.len());
    let table_data_len: usize = table.iter().map(Vec::len).sum();
    let mut table_data = Vec::with_capacity(table_data_len);
    for entry in table {
        table_index.push(u8::try_from(entry.len()).map_err(|_| PDeflateError::NumericOverflow)?);
        table_data.extend_from_slice(entry);
    }
    Ok((table_index, table_data))
}

fn materialize_table_entries(
    table_index: &[u8],
    table_data: &[u8],
) -> Result<Vec<Vec<u8>>, PDeflateError> {
    let mut table = Vec::with_capacity(table_index.len());
    let mut cursor = 0usize;
    for &entry_len_u8 in table_index {
        let entry_len = usize::from(entry_len_u8);
        if entry_len == 0 {
            continue;
        }
        let end = cursor
            .checked_add(entry_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        if end > table_data.len() {
            return Err(PDeflateError::InvalidStream(
                "table_index/table_data length mismatch",
            ));
        }
        let mut entry = vec![0u8; entry_len];
        entry.copy_from_slice(&table_data[cursor..end]);
        table.push(entry);
        cursor = end;
    }
    if cursor != table_data.len() {
        return Err(PDeflateError::InvalidStream(
            "table_index/table_data trailing bytes",
        ));
    }
    Ok(table)
}

fn pack_chunk_payload_cpu(
    chunk_len: usize,
    flags: u16,
    table_count: usize,
    section_count: usize,
    table_index: &[u8],
    table_data: &[u8],
    huff_lut: &[u8],
    section_index: &[u8],
    section_cmd: &[u8],
) -> Result<Vec<u8>, PDeflateError> {
    let table_index_offset = CHUNK_HEADER_SIZE;
    let table_data_offset = table_index_offset
        .checked_add(table_index.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let huff_lut_offset = table_data_offset
        .checked_add(table_data.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let section_index_offset = huff_lut_offset
        .checked_add(huff_lut.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let section_bitstream_offset = section_index_offset
        .checked_add(section_index.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let total_len = section_bitstream_offset
        .checked_add(section_cmd.len())
        .ok_or(PDeflateError::NumericOverflow)?;

    let mut payload = Vec::with_capacity(total_len);
    payload.extend_from_slice(&CHUNK_MAGIC);
    write_u16_le(&mut payload, CHUNK_VERSION);
    write_u16_le(&mut payload, flags);
    write_u32_le(
        &mut payload,
        u32::try_from(chunk_len).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u16_le(
        &mut payload,
        u16::try_from(table_count).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u16_le(
        &mut payload,
        u16::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u32_le(
        &mut payload,
        u32::try_from(table_index_offset).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u32_le(
        &mut payload,
        u32::try_from(table_data_offset).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u32_le(
        &mut payload,
        u32::try_from(huff_lut_offset).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u32_le(
        &mut payload,
        u32::try_from(section_index_offset).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u32_le(
        &mut payload,
        u32::try_from(section_bitstream_offset).map_err(|_| PDeflateError::NumericOverflow)?,
    );

    if payload.len() != CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream("internal header sizing bug"));
    }

    payload.extend_from_slice(table_index);
    payload.extend_from_slice(table_data);
    payload.extend_from_slice(huff_lut);
    payload.extend_from_slice(section_index);
    payload.extend_from_slice(section_cmd);
    Ok(payload)
}

fn pack_chunk_payload(
    chunk_len: usize,
    flags: u16,
    table_count: usize,
    section_count: usize,
    table_index: &[u8],
    table_data: &[u8],
    huff_lut: &[u8],
    section_index: &[u8],
    section_cmd: &[u8],
    _prefer_gpu: bool,
) -> Result<(Vec<u8>, f64), PDeflateError> {
    let t0 = Instant::now();
    let payload = pack_chunk_payload_cpu(
        chunk_len,
        flags,
        table_count,
        section_count,
        table_index,
        table_data,
        huff_lut,
        section_index,
        section_cmd,
    )?;
    Ok((payload, elapsed_ms(t0)))
}

fn build_identity_huffman_lut_block() -> Result<Vec<u8>, PDeflateError> {
    let root = (0u16..=255u16)
        .map(|symbol| HuffmanLutEntry::Symbol { symbol, bit_len: 8 })
        .collect::<Vec<_>>();
    let lut = HuffmanLut {
        symbol_count: 256,
        root_bits: 8,
        max_code_bits: 8,
        root,
        subtables: Vec::new(),
    };
    serialize_huffman_lut(&lut)
}

#[inline]
fn chunk_huffman_enabled(flags: u16) -> bool {
    (flags & CHUNK_FLAG_HUFFMAN) != 0
}

#[inline]
fn chunk_final_stream_enabled(flags: u16) -> bool {
    (flags & CHUNK_FLAG_FINAL_STREAM) != 0
}

fn read_chunk_flags(payload: &[u8]) -> Result<u16, PDeflateError> {
    if payload.len() < 8 {
        return Err(PDeflateError::InvalidStream("chunk header truncated"));
    }
    Ok(u16::from_le_bytes([payload[6], payload[7]]))
}

fn set_chunk_final_stream_flag(payload: &mut [u8], enabled: bool) -> Result<(), PDeflateError> {
    if payload.len() < CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream("chunk header truncated"));
    }
    if &payload[..4] != CHUNK_MAGIC.as_slice() {
        return Err(PDeflateError::InvalidStream("bad chunk magic"));
    }
    let version = u16::from_le_bytes([payload[4], payload[5]]);
    if version != CHUNK_VERSION {
        return Err(PDeflateError::InvalidStream("unsupported chunk version"));
    }
    let mut flags = read_chunk_flags(payload)?;
    if enabled {
        flags |= CHUNK_FLAG_FINAL_STREAM;
    } else {
        flags &= !CHUNK_FLAG_FINAL_STREAM;
    }
    payload[6..8].copy_from_slice(&flags.to_le_bytes());
    Ok(())
}

fn write_stream_chunk_frame<W: Write>(writer: &mut W, payload: &[u8]) -> Result<(), PDeflateError> {
    writer.write_all(
        &u32::try_from(payload.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .to_le_bytes(),
    )?;
    writer.write_all(payload)?;
    Ok(())
}

fn finalize_chunk_from_table(
    chunk: &[u8],
    options: &PDeflateOptions,
    table: Vec<Vec<u8>>,
    table_count_override: Option<usize>,
    table_index_override: Option<&[u8]>,
    table_data_override: Option<&[u8]>,
    table_profile: BuildTableProfile,
    table_build_ms: f64,
    packed_matches: Option<&[u32]>,
    gpu_profile: Option<gpu::GpuMatchProfile>,
    gpu_used: bool,
) -> Result<ChunkCompressed, PDeflateError> {
    let chunk_t0 = Instant::now();
    let section_count = options.section_count;
    let table_count = table_count_override.unwrap_or(table.len());
    let max_ref_len = options.max_ref_len.min(MAX_CMD_LEN).min(64);

    let section_t0 = Instant::now();
    let mut section_profile = SectionEncodeProfile::default();
    let mut section_cmd_lens = Vec::with_capacity(section_count);
    let mut section_cmd = Vec::with_capacity(chunk.len() / 2);
    ENCODE_SCRATCH.with(|scratch| -> Result<(), PDeflateError> {
        let mut scratch = scratch.borrow_mut();
        scratch.rebuild_repeat_table(&table, max_ref_len);
        scratch.rebuild_prefix_index(&table);

        for sec in 0..section_count {
            let s0 = section_start(sec, section_count, chunk.len());
            let s1 = section_start(sec + 1, section_count, chunk.len());
            let (cmd_len, sec_prof) = if let Some(packed) = packed_matches {
                encode_section_with_packed_matches(
                    &mut section_cmd,
                    &chunk[s0..s1],
                    packed,
                    s0,
                    options,
                    max_ref_len,
                )?
            } else {
                encode_section_into(
                    &mut section_cmd,
                    &chunk[s0..s1],
                    &table,
                    &scratch.repeat_table,
                    &scratch.by_prefix2,
                    &scratch.has_prefix2,
                    &scratch.prefix3_masks,
                    options,
                )?
            };
            section_profile.match_search_ms += sec_prof.match_search_ms;
            section_profile.best_search_ms += sec_prof.best_search_ms;
            section_profile.lookahead_search_ms += sec_prof.lookahead_search_ms;
            section_profile.emit_ref_ms += sec_prof.emit_ref_ms;
            section_profile.emit_lit_ms += sec_prof.emit_lit_ms;
            section_profile.find_calls = section_profile
                .find_calls
                .saturating_add(sec_prof.find_calls);
            section_profile.best_calls = section_profile
                .best_calls
                .saturating_add(sec_prof.best_calls);
            section_profile.lookahead_calls = section_profile
                .lookahead_calls
                .saturating_add(sec_prof.lookahead_calls);
            section_profile.candidate_checks = section_profile
                .candidate_checks
                .saturating_add(sec_prof.candidate_checks);
            section_profile.best_candidate_checks = section_profile
                .best_candidate_checks
                .saturating_add(sec_prof.best_candidate_checks);
            section_profile.lookahead_candidate_checks = section_profile
                .lookahead_candidate_checks
                .saturating_add(sec_prof.lookahead_candidate_checks);
            section_profile.compare_steps = section_profile
                .compare_steps
                .saturating_add(sec_prof.compare_steps);
            section_profile.best_compare_steps = section_profile
                .best_compare_steps
                .saturating_add(sec_prof.best_compare_steps);
            section_profile.lookahead_compare_steps = section_profile
                .lookahead_compare_steps
                .saturating_add(sec_prof.lookahead_compare_steps);
            section_profile.prefilter_rejects = section_profile
                .prefilter_rejects
                .saturating_add(sec_prof.prefilter_rejects);
            section_profile.no_prefix_fast_skips = section_profile
                .no_prefix_fast_skips
                .saturating_add(sec_prof.no_prefix_fast_skips);
            section_profile.best_tail_rejects = section_profile
                .best_tail_rejects
                .saturating_add(sec_prof.best_tail_rejects);
            section_profile.ref_cmds = section_profile.ref_cmds.saturating_add(sec_prof.ref_cmds);
            section_profile.lit_cmds = section_profile.lit_cmds.saturating_add(sec_prof.lit_cmds);
            section_profile.ref_bytes =
                section_profile.ref_bytes.saturating_add(sec_prof.ref_bytes);
            section_profile.lit_bytes =
                section_profile.lit_bytes.saturating_add(sec_prof.lit_bytes);
            section_cmd_lens
                .push(u32::try_from(cmd_len).map_err(|_| PDeflateError::NumericOverflow)?);
        }
        Ok(())
    })?;
    let section_encode_ms = elapsed_ms(section_t0);

    let (payload, header_pack_ms) =
        FINALIZE_SCRATCH.with(|scratch| -> Result<(Vec<u8>, f64), PDeflateError> {
            let mut scratch = scratch.borrow_mut();
            scratch.section_index.clear();
            scratch.section_bitstream.clear();
            let huffman_enabled = options.huffman_encode_enabled;
            let mut section_cmd_cursor = 0usize;
            for &len_u32 in &section_cmd_lens {
                let sec_cmd_len =
                    usize::try_from(len_u32).map_err(|_| PDeflateError::NumericOverflow)?;
                let sec_cmd_end = section_cmd_cursor
                    .checked_add(sec_cmd_len)
                    .ok_or(PDeflateError::NumericOverflow)?;
                let sec_cmd = section_cmd.get(section_cmd_cursor..sec_cmd_end).ok_or(
                    PDeflateError::InvalidStream(
                        "section command range out of bounds during bitstream encode",
                    ),
                )?;
                let bit_len = if huffman_enabled {
                    let symbols = logical_commands_to_huffman_symbols(sec_cmd);
                    encode_huffman_symbols_to_bitstream(symbols, &mut scratch.section_bitstream)?
                } else {
                    scratch.section_bitstream.extend_from_slice(sec_cmd);
                    u32::try_from(
                        sec_cmd_len
                            .checked_mul(8)
                            .ok_or(PDeflateError::NumericOverflow)?,
                    )
                    .map_err(|_| PDeflateError::NumericOverflow)?
                };
                write_varint_u32(&mut scratch.section_index, bit_len);
                section_cmd_cursor = sec_cmd_end;
            }
            if section_cmd_cursor != section_cmd.len() {
                return Err(PDeflateError::InvalidStream(
                    "sum(section command len) != section command stream size",
                ));
            }

            let (table_index, table_data): (&[u8], &[u8]) = if let (Some(idx), Some(data)) =
                (table_index_override, table_data_override)
            {
                (idx, data)
            } else {
                scratch.table_index.clear();
                scratch.table_data.clear();
                scratch.table_index.reserve(table_count);
                let table_data_len: usize = table.iter().map(Vec::len).sum();
                scratch.table_data.reserve(table_data_len);
                for entry in &table {
                    scratch.table_index.push(
                        u8::try_from(entry.len()).map_err(|_| PDeflateError::NumericOverflow)?,
                    );
                    scratch.table_data.extend_from_slice(entry);
                }
                (&scratch.table_index, &scratch.table_data)
            };
            let huff_lut_storage = if huffman_enabled {
                Some(build_identity_huffman_lut_block()?)
            } else {
                None
            };
            let huff_lut = huff_lut_storage.as_deref().unwrap_or(&[]);
            let chunk_flags = if huffman_enabled {
                CHUNK_FLAG_HUFFMAN
            } else {
                0
            };
            pack_chunk_payload(
                chunk.len(),
                chunk_flags,
                table_index.len(),
                section_count,
                table_index,
                table_data,
                huff_lut,
                &scratch.section_index,
                &scratch.section_bitstream,
                gpu_used,
            )
        })?;
    let total_ms = elapsed_ms(chunk_t0);

    let gp = gpu_profile.unwrap_or_default();
    Ok(ChunkCompressed {
        payload,
        table_entries: table_count,
        section_count,
        profile: ChunkEncodeProfile {
            table_build_ms,
            table_freq_ms: table_profile.freq_ms,
            table_probe_ms: table_profile.probe_ms,
            table_materialize_ms: table_profile.materialize_ms,
            table_sort_ms: table_profile.sort_ms,
            table_positions_scanned: table_profile.positions_scanned,
            table_probe_pairs: table_profile.probe_pairs,
            table_match_len_calls: table_profile.match_len_calls,
            table_hash_key_mismatch: table_profile.hash_key_mismatch,
            anchored_short_hits: table_profile.anchored_short_hits,
            section_encode_ms,
            section_match_search_ms: section_profile.match_search_ms,
            section_best_search_ms: section_profile.best_search_ms,
            section_lookahead_search_ms: section_profile.lookahead_search_ms,
            section_emit_ref_ms: section_profile.emit_ref_ms,
            section_emit_lit_ms: section_profile.emit_lit_ms,
            section_find_calls: section_profile.find_calls,
            section_best_calls: section_profile.best_calls,
            section_lookahead_calls: section_profile.lookahead_calls,
            section_candidate_checks: section_profile.candidate_checks,
            section_best_candidate_checks: section_profile.best_candidate_checks,
            section_lookahead_candidate_checks: section_profile.lookahead_candidate_checks,
            section_compare_steps: section_profile.compare_steps,
            section_best_compare_steps: section_profile.best_compare_steps,
            section_lookahead_compare_steps: section_profile.lookahead_compare_steps,
            section_prefilter_rejects: section_profile.prefilter_rejects,
            section_no_prefix_fast_skips: section_profile.no_prefix_fast_skips,
            section_best_tail_rejects: section_profile.best_tail_rejects,
            section_ref_cmds: section_profile.ref_cmds,
            section_lit_cmds: section_profile.lit_cmds,
            section_ref_bytes: section_profile.ref_bytes,
            section_lit_bytes: section_profile.lit_bytes,
            gpu_used,
            gpu_upload_ms: gp.upload_ms,
            gpu_wait_ms: gp.wait_ms,
            gpu_map_copy_ms: gp.map_copy_ms,
            gpu_total_ms: gp.total_ms,
            header_pack_ms,
            total_ms,
        },
    })
}

fn compress_chunk_gpu_batch(
    input: &[u8],
    chunk_size: usize,
    indices: &[usize],
    options: &PDeflateOptions,
) -> Result<(Vec<(usize, ChunkCompressed)>, GpuBatchCompressProfile), PDeflateError> {
    if indices.is_empty() {
        return Ok((Vec::new(), GpuBatchCompressProfile::default()));
    }

    let t_total = Instant::now();
    let profile_detail = profile_detail_enabled();
    let t_prepare = Instant::now();
    let mut prepare_table_build_ms = 0.0_f64;
    let prepare_table_readback_ms = 0.0_f64;
    let mut prepare_gpu_table_build_chunks = 0usize;
    let prepare_gpu_table_readback_chunks = 0usize;
    let mut prepared = Vec::<Option<GpuPreparedChunk<'_>>>::with_capacity(indices.len());
    prepared.resize_with(indices.len(), || None);
    let mut gpu_table_batch_positions = Vec::<usize>::new();
    let mut gpu_table_batch_chunks = Vec::<&[u8]>::new();
    let mut gpu_table_build_requested_chunks = 0usize;
    for (pos, &chunk_idx) in indices.iter().enumerate() {
        let start = chunk_idx
            .checked_mul(chunk_size)
            .ok_or(PDeflateError::NumericOverflow)?;
        let end = (start + chunk_size).min(input.len());
        let chunk = &input[start..end];
        let max_ref_len = options.max_ref_len.min(MAX_CMD_LEN).min(64);
        if should_use_gpu_table_build(options, chunk.len()) {
            gpu_table_build_requested_chunks = gpu_table_build_requested_chunks.saturating_add(1);
            gpu_table_batch_positions.push(pos);
            gpu_table_batch_chunks.push(chunk);
            prepared[pos] = Some(GpuPreparedChunk {
                index: chunk_idx,
                chunk,
                table: None,
                gpu_table: None,
                table_count: 0,
                table_index: Vec::new(),
                table_data: Vec::new(),
                table_profile: BuildTableProfile::default(),
                table_build_ms: 0.0,
                max_ref_len,
                gpu_eligible: false,
            });
            continue;
        }
        let built = build_table_dispatch(chunk, options, false)?;
        prepare_table_build_ms += built.build_ms;
        let gpu_eligible = max_ref_len >= options.min_ref_len
            && (built.table_count > 0 || built.gpu_table.is_some());
        prepared[pos] = Some(GpuPreparedChunk {
            index: chunk_idx,
            chunk,
            table: built.table,
            gpu_table: built.gpu_table,
            table_count: built.table_count,
            table_index: built.table_index,
            table_data: built.table_data,
            table_profile: built.profile,
            table_build_ms: built.build_ms,
            max_ref_len,
            gpu_eligible,
        });
    }
    if !gpu_table_batch_chunks.is_empty() {
        let (gpu_table_sample_stride, gpu_match_probe_limit, gpu_hash_history_limit) =
            gpu_table_build_tuning(options);
        let batch_result = gpu::build_table_gpu_device_batch(
            &gpu_table_batch_chunks,
            options.max_table_entries.min(MAX_TABLE_ID),
            options.max_table_entry_len,
            options.min_ref_len,
            gpu_match_probe_limit,
            gpu_hash_history_limit,
            gpu_table_sample_stride,
        );
        match batch_result {
            Ok(entries) => {
                for ((pos, chunk), (gpu_table, gpu_prof)) in gpu_table_batch_positions
                    .iter()
                    .copied()
                    .zip(gpu_table_batch_chunks.iter().copied())
                    .zip(entries.into_iter())
                {
                    let built_profile = BuildTableProfile {
                        freq_ms: gpu_prof.freq_kernel_ms,
                        probe_ms: gpu_prof.candidate_kernel_ms,
                        materialize_ms: gpu_prof.materialize_ms,
                        sort_ms: gpu_prof.sort_ms,
                        ..BuildTableProfile::default()
                    };
                    let build_ms = gpu_table.build_breakdown.total_ms;
                    prepare_table_build_ms += build_ms;
                    prepare_gpu_table_build_chunks =
                        prepare_gpu_table_build_chunks.saturating_add(1);
                    let max_ref_len = options.max_ref_len.min(MAX_CMD_LEN).min(64);
                    prepared[pos] = Some(GpuPreparedChunk {
                        index: indices[pos],
                        chunk,
                        table: None,
                        gpu_table: Some(gpu_table),
                        table_count: 0,
                        table_index: Vec::new(),
                        table_data: Vec::new(),
                        table_profile: built_profile,
                        table_build_ms: build_ms,
                        max_ref_len,
                        gpu_eligible: max_ref_len >= options.min_ref_len,
                    });
                }
            }
            Err(err) => {
                for pos in gpu_table_batch_positions.iter().copied() {
                    let chunk_idx = indices[pos];
                    let start = chunk_idx
                        .checked_mul(chunk_size)
                        .ok_or(PDeflateError::NumericOverflow)?;
                    let end = (start + chunk_size).min(input.len());
                    let chunk = &input[start..end];
                    let built = build_table_dispatch(
                        chunk,
                        options,
                        should_use_gpu_table_build(options, chunk.len()),
                    )?;
                    prepare_table_build_ms += built.build_ms;
                    if built.gpu_table.is_some() {
                        prepare_gpu_table_build_chunks =
                            prepare_gpu_table_build_chunks.saturating_add(1);
                    }
                    let max_ref_len = options.max_ref_len.min(MAX_CMD_LEN).min(64);
                    let gpu_eligible = max_ref_len >= options.min_ref_len
                        && (built.table_count > 0 || built.gpu_table.is_some());
                    prepared[pos] = Some(GpuPreparedChunk {
                        index: chunk_idx,
                        chunk,
                        table: built.table,
                        gpu_table: built.gpu_table,
                        table_count: built.table_count,
                        table_index: built.table_index,
                        table_data: built.table_data,
                        table_profile: built.profile,
                        table_build_ms: built.build_ms,
                        max_ref_len,
                        gpu_eligible,
                    });
                }
            }
        }
    }
    let prepared = prepared
        .into_iter()
        .map(|v| v.expect("prepared chunk missing"))
        .collect::<Vec<_>>();
    let prepare_ms = elapsed_ms(t_prepare);
    let prepare_misc_ms =
        (prepare_ms - prepare_table_build_ms - prepare_table_readback_ms).max(0.0);

    let mut prof_by_chunk = vec![gpu::GpuMatchProfile::default(); prepared.len()];
    let mut packed_matches_by_chunk = std::iter::repeat_with(|| None::<Vec<u32>>)
        .take(prepared.len())
        .collect::<Vec<_>>();
    let mut gpu_encoded_chunks = std::iter::repeat_with(|| None::<ChunkCompressed>)
        .take(prepared.len())
        .collect::<Vec<_>>();
    let mut gpu_used_chunks = 0usize;
    let mut gpu_integrated_chunks = 0usize;
    let mut gpu_fallback_match_chunks = 0usize;
    let mut fallback_integrated_error_chunks = 0usize;
    let mut fallback_integrated_invalid_options_chunks = 0usize;
    let mut fallback_integrated_invalid_stream_chunks = 0usize;
    let mut fallback_integrated_numeric_overflow_chunks = 0usize;
    let mut fallback_integrated_gpu_error_chunks = 0usize;
    let mut fallback_integrated_other_error_chunks = 0usize;
    let mut fallback_integrated_output_mismatch_chunks = 0usize;
    let mut fallback_match_error_chunks = 0usize;
    let mut fallback_match_output_mismatch_chunks = 0usize;
    let mut fallback_match_packed_len_mismatch_chunks = 0usize;
    let mut fallback_integrated_error_example: Option<String> = None;
    let gpu_job_indices: Vec<usize> = prepared
        .iter()
        .enumerate()
        .filter_map(|(idx, chunk)| chunk.gpu_eligible.then_some(idx))
        .collect();
    let mut fallback_match_job_indices = Vec::<usize>::new();
    let mut gpu_match_upload_ms = 0.0_f64;
    let mut gpu_match_wait_ms = 0.0_f64;
    let mut gpu_match_map_copy_ms = 0.0_f64;
    let mut gpu_match_total_ms = 0.0_f64;
    let mut gpu_section_encode_upload_ms = 0.0_f64;
    let mut gpu_section_encode_wait_ms = 0.0_f64;
    let mut gpu_section_encode_map_copy_ms = 0.0_f64;
    let mut gpu_section_encode_total_ms = 0.0_f64;
    let mut gpu_sparse_pack_upload_ms = 0.0_f64;
    let mut gpu_sparse_pack_wait_ms = 0.0_f64;
    let mut gpu_sparse_pack_map_copy_ms = 0.0_f64;
    let mut gpu_sparse_lens_submit_ms = 0.0_f64;
    let mut gpu_sparse_lens_submit_done_wait_ms = 0.0_f64;
    let mut gpu_sparse_lens_map_after_done_ms = 0.0_f64;
    let mut gpu_sparse_lens_poll_calls = 0u64;
    let mut gpu_sparse_lens_yield_calls = 0u64;
    let mut gpu_sparse_lens_wait_ms = 0.0_f64;
    let mut gpu_sparse_lens_copy_ms = 0.0_f64;
    let mut gpu_sparse_prepare_ms = 0.0_f64;
    let mut gpu_sparse_table_size_resolve_ms = 0.0_f64;
    let mut gpu_sparse_prepare_misc_ms = 0.0_f64;
    let mut gpu_sparse_scratch_acquire_ms = 0.0_f64;
    let mut gpu_sparse_upload_dispatch_ms = 0.0_f64;
    let mut gpu_sparse_submit_ms = 0.0_f64;
    let mut gpu_sparse_wait_ms = 0.0_f64;
    let mut gpu_sparse_copy_ms = 0.0_f64;
    let mut gpu_sparse_total_ms = 0.0_f64;
    let mut gpu_post_total_pack_bytes_ms = 0.0_f64;
    let mut gpu_post_scale_profiles_ms = 0.0_f64;
    let mut gpu_post_chunk_materialize_ms = 0.0_f64;
    let mut kernel_profile = gpu::GpuBatchKernelProfile::default();

    let t_gpu_call = Instant::now();
    if !gpu_job_indices.is_empty() {
        let gpu_max_cmd_len = options.max_ref_len.min(MAX_CMD_LEN).min(64);
        let requested_submit_group = options.gpu_submit_chunks.max(1);
        let safe_submit_group = gpu::max_safe_match_batch_chunks(chunk_size);
        let submit_group = match safe_submit_group {
            Ok(limit) => requested_submit_group.min(limit.max(1)),
            Err(_) => requested_submit_group,
        }
        .min(gpu_job_indices.len())
        .max(1);
        if profile_detail && submit_group < requested_submit_group {
            let max_binding_mib = gpu::max_storage_binding_size_bytes()
                .ok()
                .map(|bytes| bytes as f64 / (1024.0 * 1024.0))
                .unwrap_or(0.0);
            let per_chunk_out_mib =
                (chunk_size.saturating_mul(4) as f64) / (1024.0 * 1024.0);
            let safe_limit = safe_submit_group.unwrap_or(submit_group);
            eprintln!(
                "[cozip_pdeflate][timing][gpu-submit-group-cap] requested={} effective={} safe_limit={} chunk_mib={:.2} per_chunk_match_out_mib={:.2} max_storage_binding_mib={:.2}",
                requested_submit_group,
                submit_group,
                safe_limit,
                chunk_size as f64 / (1024.0 * 1024.0),
                per_chunk_out_mib,
                max_binding_mib,
            );
        }
        let target_inflight = gpu_sparse_target_inflight()
            .max(1)
            .min(gpu_job_indices.len().max(1));
        let wait_high_watermark = gpu_sparse_wait_high_watermark()
            .max(target_inflight)
            .min(gpu_job_indices.len().max(1));
        let mut handle_integrated_batch =
            |prep_indices: Vec<usize>, batch_res: Result<gpu::GpuBatchSparsePackOutput, PDeflateError>| {
                match batch_res {
                    Ok(batch) => {
                        if batch.chunks.len() != prep_indices.len() {
                            fallback_integrated_output_mismatch_chunks =
                                fallback_integrated_output_mismatch_chunks
                                    .saturating_add(prep_indices.len());
                            if profile_detail {
                                eprintln!(
                                    "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason=output_count_mismatch out={} jobs={}",
                                    indices.len(),
                                    prep_indices.len(),
                                    batch.chunks.len(),
                                    prep_indices.len()
                                );
                            }
                            fallback_match_job_indices.extend(prep_indices);
                            return;
                        }

                        gpu_match_upload_ms += batch.match_profile.upload_ms;
                        gpu_match_wait_ms += batch.match_profile.wait_ms;
                        gpu_match_map_copy_ms += batch.match_profile.map_copy_ms;
                        gpu_match_total_ms += batch.match_profile.total_ms;
                        gpu_section_encode_upload_ms += batch.section_profile.upload_ms;
                        gpu_section_encode_wait_ms += batch.section_profile.wait_ms;
                        gpu_section_encode_map_copy_ms += batch.section_profile.map_copy_ms;
                        gpu_section_encode_total_ms += batch.section_profile.total_ms;
                        gpu_sparse_pack_upload_ms += batch.sparse_profile.upload_ms;
                        gpu_sparse_pack_wait_ms += batch.sparse_profile.wait_ms;
                        gpu_sparse_pack_map_copy_ms += batch.sparse_profile.map_copy_ms;
                        gpu_sparse_lens_submit_ms += batch.sparse_batch_profile.lens_submit_ms;
                        gpu_sparse_lens_submit_done_wait_ms +=
                            batch.sparse_batch_profile.lens_submit_done_wait_ms;
                        gpu_sparse_lens_map_after_done_ms +=
                            batch.sparse_batch_profile.lens_map_after_done_ms;
                        gpu_sparse_lens_poll_calls = gpu_sparse_lens_poll_calls
                            .saturating_add(batch.sparse_batch_profile.lens_poll_calls);
                        gpu_sparse_lens_yield_calls = gpu_sparse_lens_yield_calls
                            .saturating_add(batch.sparse_batch_profile.lens_yield_calls);
                        gpu_sparse_lens_wait_ms += batch.sparse_batch_profile.lens_wait_ms;
                        gpu_sparse_lens_copy_ms += batch.sparse_batch_profile.lens_copy_ms;
                        gpu_sparse_prepare_ms += batch.sparse_batch_profile.prepare_ms;
                        gpu_sparse_table_size_resolve_ms +=
                            batch.sparse_batch_profile.table_size_resolve_ms;
                        gpu_sparse_prepare_misc_ms += batch.sparse_batch_profile.prepare_misc_ms;
                        gpu_sparse_scratch_acquire_ms +=
                            batch.sparse_batch_profile.scratch_acquire_ms;
                        gpu_sparse_upload_dispatch_ms +=
                            batch.sparse_batch_profile.upload_dispatch_ms;
                        gpu_sparse_submit_ms += batch.sparse_batch_profile.submit_ms;
                        gpu_sparse_wait_ms += batch.sparse_batch_profile.wait_ms;
                        gpu_sparse_copy_ms += batch.sparse_batch_profile.copy_ms;
                        gpu_sparse_total_ms += batch.sparse_batch_profile.total_ms;
                        gpu_integrated_chunks =
                            gpu_integrated_chunks.saturating_add(prep_indices.len());
                        accumulate_gpu_kernel_profile(&mut kernel_profile, &batch.kernel_profile);
                        kernel_profile.sparse_lens_submit_ms +=
                            batch.sparse_batch_profile.lens_submit_ms;
                        kernel_profile.sparse_lens_submit_done_wait_ms +=
                            batch.sparse_batch_profile.lens_submit_done_wait_ms;
                        kernel_profile.sparse_lens_map_after_done_ms +=
                            batch.sparse_batch_profile.lens_map_after_done_ms;
                        kernel_profile.sparse_lens_poll_calls = kernel_profile
                            .sparse_lens_poll_calls
                            .saturating_add(batch.sparse_batch_profile.lens_poll_calls);
                        kernel_profile.sparse_lens_yield_calls = kernel_profile
                            .sparse_lens_yield_calls
                            .saturating_add(batch.sparse_batch_profile.lens_yield_calls);
                        kernel_profile.sparse_lens_wait_ms +=
                            batch.sparse_batch_profile.lens_wait_ms;
                        kernel_profile.sparse_lens_copy_ms +=
                            batch.sparse_batch_profile.lens_copy_ms;
                        kernel_profile.sparse_prepare_ms += batch.sparse_batch_profile.prepare_ms;
                        kernel_profile.sparse_scratch_acquire_ms +=
                            batch.sparse_batch_profile.scratch_acquire_ms;
                        kernel_profile.sparse_upload_dispatch_ms +=
                            batch.sparse_batch_profile.upload_dispatch_ms;
                        kernel_profile.sparse_submit_ms += batch.sparse_batch_profile.submit_ms;
                        kernel_profile.sparse_wait_ms += batch.sparse_batch_profile.wait_ms;
                        kernel_profile.sparse_copy_ms += batch.sparse_batch_profile.copy_ms;
                        kernel_profile.sparse_total_ms += batch.sparse_batch_profile.total_ms;

                        let emit_pack_batch_detail = if profile_detail {
                            let seq =
                                PROFILE_DETAIL_GPU_PACK_SEQ.fetch_add(1, Ordering::Relaxed);
                            profile_detail_should_log_stream(seq)
                        } else {
                            false
                        };
                        if emit_pack_batch_detail {
                            let pack_detail = batch.sparse_batch_profile;
                            let chunks_f = (pack_detail.chunks as f64).max(1.0);
                            eprintln!(
                                "[cozip_pdeflate][timing][gpu-pack-batch] chunks={} lens_kib={:.2} out_cmd_mib={:.2} predicted_payload_avg_kib={:.1} actual_payload_avg_kib={:.1} cap_payload_avg_kib={:.1} predicted_under_chunks={} predicted_under_mib={:.3} predicted_over_mib={:.3} t_lens_submit_ms={:.3} t_lens_submit_done_wait_ms={:.3} t_lens_map_after_done_ms={:.3} lens_poll_calls={} lens_yield_calls={} t_lens_wait_ms={:.3} t_lens_copy_ms={:.3} t_sparse_prepare_ms={:.3} t_sparse_table_size_resolve_ms={:.3} t_sparse_prepare_misc_ms={:.3} t_sparse_upload_dispatch_ms={:.3} t_sparse_submit_ms={:.3} t_sparse_wait_ms={:.3} t_sparse_copy_ms={:.3} t_sparse_total_ms={:.3}",
                                pack_detail.chunks,
                                (pack_detail.lens_bytes_total as f64) / 1024.0,
                                (pack_detail.out_cmd_bytes_total as f64) / (1024.0 * 1024.0),
                                (pack_detail.predicted_payload_bytes_total as f64) / chunks_f / 1024.0,
                                (pack_detail.actual_payload_bytes_total as f64) / chunks_f / 1024.0,
                                (pack_detail.cap_payload_bytes_total as f64) / chunks_f / 1024.0,
                                pack_detail.predicted_underestimate_chunks,
                                (pack_detail.predicted_underestimate_bytes as f64) / (1024.0 * 1024.0),
                                (pack_detail.predicted_overestimate_bytes as f64) / (1024.0 * 1024.0),
                                pack_detail.lens_submit_ms,
                                pack_detail.lens_submit_done_wait_ms,
                                pack_detail.lens_map_after_done_ms,
                                pack_detail.lens_poll_calls,
                                pack_detail.lens_yield_calls,
                                pack_detail.lens_wait_ms,
                                pack_detail.lens_copy_ms,
                                pack_detail.prepare_ms,
                                pack_detail.table_size_resolve_ms,
                                pack_detail.prepare_misc_ms,
                                pack_detail.upload_dispatch_ms,
                                pack_detail.submit_ms,
                                pack_detail.wait_ms,
                                pack_detail.copy_ms,
                                pack_detail.total_ms
                            );
                        }

                        let t_post_total_pack_bytes = Instant::now();
                        let total_pack_bytes: usize = prep_indices
                            .iter()
                            .map(|&idx| prepared[idx].chunk.len())
                            .sum();
                        gpu_post_total_pack_bytes_ms += elapsed_ms(t_post_total_pack_bytes);
                        for (&prep_idx, packed_chunk) in
                            prep_indices.iter().zip(batch.chunks.into_iter())
                        {
                            let t_post_scale = Instant::now();
                            let pack_scale = if total_pack_bytes == 0 {
                                0.0
                            } else {
                                (prepared[prep_idx].chunk.len() as f64)
                                    / (total_pack_bytes as f64)
                            };
                            let match_scaled = scale_gpu_profile(batch.match_profile, pack_scale);
                            let section_scaled =
                                scale_gpu_section_profile(batch.section_profile, pack_scale);
                            let pack_scaled = scale_gpu_profile(batch.sparse_profile, pack_scale);
                            let gpu_profile = gpu::GpuMatchProfile {
                                upload_ms: match_scaled.upload_ms
                                    + section_scaled.upload_ms
                                    + pack_scaled.upload_ms,
                                wait_ms: match_scaled.wait_ms
                                    + section_scaled.wait_ms
                                    + pack_scaled.wait_ms,
                                map_copy_ms: match_scaled.map_copy_ms
                                    + section_scaled.map_copy_ms
                                    + pack_scaled.map_copy_ms,
                                total_ms: match_scaled.total_ms
                                    + section_scaled.total_ms
                                    + pack_scaled.total_ms,
                            };
                            gpu_post_scale_profiles_ms += elapsed_ms(t_post_scale);
                            let t_post_materialize = Instant::now();
                            gpu_encoded_chunks[prep_idx] = Some(ChunkCompressed {
                                payload: packed_chunk.payload,
                                table_entries: packed_chunk.table_count,
                                section_count: options.section_count,
                                profile: ChunkEncodeProfile {
                                    table_build_ms: prepared[prep_idx].table_build_ms,
                                    table_freq_ms: prepared[prep_idx].table_profile.freq_ms,
                                    table_probe_ms: prepared[prep_idx].table_profile.probe_ms,
                                    table_materialize_ms: prepared[prep_idx]
                                        .table_profile
                                        .materialize_ms,
                                    table_sort_ms: prepared[prep_idx].table_profile.sort_ms,
                                    table_positions_scanned: prepared[prep_idx]
                                        .table_profile
                                        .positions_scanned,
                                    table_probe_pairs: prepared[prep_idx]
                                        .table_profile
                                        .probe_pairs,
                                    table_match_len_calls: prepared[prep_idx]
                                        .table_profile
                                        .match_len_calls,
                                    table_hash_key_mismatch: prepared[prep_idx]
                                        .table_profile
                                        .hash_key_mismatch,
                                    anchored_short_hits: prepared[prep_idx]
                                        .table_profile
                                        .anchored_short_hits,
                                    section_encode_ms: section_scaled.total_ms,
                                    section_match_search_ms: 0.0,
                                    section_best_search_ms: 0.0,
                                    section_lookahead_search_ms: 0.0,
                                    section_emit_ref_ms: 0.0,
                                    section_emit_lit_ms: 0.0,
                                    section_find_calls: 0,
                                    section_best_calls: 0,
                                    section_lookahead_calls: 0,
                                    section_candidate_checks: 0,
                                    section_best_candidate_checks: 0,
                                    section_lookahead_candidate_checks: 0,
                                    section_compare_steps: 0,
                                    section_best_compare_steps: 0,
                                    section_lookahead_compare_steps: 0,
                                    section_prefilter_rejects: 0,
                                    section_no_prefix_fast_skips: 0,
                                    section_best_tail_rejects: 0,
                                    section_ref_cmds: 0,
                                    section_lit_cmds: 0,
                                    section_ref_bytes: 0,
                                    section_lit_bytes: 0,
                                    gpu_used: true,
                                    gpu_upload_ms: gpu_profile.upload_ms,
                                    gpu_wait_ms: gpu_profile.wait_ms,
                                    gpu_map_copy_ms: gpu_profile.map_copy_ms,
                                    gpu_total_ms: gpu_profile.total_ms,
                                    header_pack_ms: pack_scaled.total_ms,
                                    total_ms: prepared[prep_idx].table_build_ms
                                        + gpu_profile.total_ms,
                                },
                            });
                            gpu_post_chunk_materialize_ms += elapsed_ms(t_post_materialize);
                            gpu_used_chunks = gpu_used_chunks.saturating_add(1);
                        }
                    }
                    Err(err) => {
                        fallback_integrated_error_chunks =
                            fallback_integrated_error_chunks
                                .saturating_add(prep_indices.len());
                        let reason_class = classify_integrated_batch_error(&err);
                        match reason_class {
                            "invalid_options" => {
                                fallback_integrated_invalid_options_chunks =
                                    fallback_integrated_invalid_options_chunks
                                        .saturating_add(prep_indices.len());
                            }
                            "invalid_stream" => {
                                fallback_integrated_invalid_stream_chunks =
                                    fallback_integrated_invalid_stream_chunks
                                        .saturating_add(prep_indices.len());
                            }
                            "numeric_overflow" => {
                                fallback_integrated_numeric_overflow_chunks =
                                    fallback_integrated_numeric_overflow_chunks
                                        .saturating_add(prep_indices.len());
                            }
                            "gpu_error" => {
                                fallback_integrated_gpu_error_chunks =
                                    fallback_integrated_gpu_error_chunks
                                        .saturating_add(prep_indices.len());
                            }
                            _ => {
                                fallback_integrated_other_error_chunks =
                                    fallback_integrated_other_error_chunks
                                        .saturating_add(prep_indices.len());
                            }
                        }
                        if fallback_integrated_error_example.is_none() {
                            fallback_integrated_error_example = Some(err.to_string());
                        }
                        if profile_enabled()
                            && GPU_INTEGRATED_ERROR_SAMPLE_LOGGED
                                .compare_exchange(
                                    false,
                                    true,
                                    Ordering::Relaxed,
                                    Ordering::Relaxed,
                                )
                                .is_ok()
                        {
                            eprintln!(
                                "[cozip_pdeflate][timing][gpu-integrated-error-sample] gpu_jobs={} class={} error={}",
                                prep_indices.len(),
                                reason_class,
                                err
                            );
                        }
                        if profile_detail {
                            eprintln!(
                                "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason={} ",
                                indices.len(),
                                prep_indices.len(),
                                err
                            );
                        }
                        fallback_match_job_indices.extend(prep_indices);
                    }
                }
            };

        let mut pending_batches = VecDeque::<(
            Vec<usize>,
            gpu::PendingGpuSparsePackBatch,
        )>::new();
        let drain_ready = |pending_batches: &mut VecDeque<(
            Vec<usize>,
            gpu::PendingGpuSparsePackBatch,
        )>|
         -> Result<Vec<(
            Vec<usize>,
            Result<gpu::GpuBatchSparsePackOutput, PDeflateError>,
        )>, PDeflateError> {
            let mut ready = Vec::new();
            let mut idx = 0usize;
            while idx < pending_batches.len() {
                let collect_res = {
                    let (_, pending) = pending_batches
                        .get_mut(idx)
                        .ok_or_else(|| {
                            PDeflateError::Gpu(
                                "gpu sparse pending batch queue corrupted".to_string(),
                            )
                        })?;
                    gpu::try_collect_matches_encode_and_pack_sparse_batch(pending)
                };
                match collect_res {
                    Ok(Some(batch)) => {
                        let (prep_indices, _) = pending_batches.remove(idx).ok_or_else(|| {
                            PDeflateError::Gpu(
                                "gpu sparse pending batch removal failed".to_string(),
                            )
                        })?;
                        ready.push((prep_indices, Ok(batch)));
                    }
                    Ok(None) => {
                        idx = idx.saturating_add(1);
                    }
                    Err(err) => {
                        let (prep_indices, _) = pending_batches.remove(idx).ok_or_else(|| {
                            PDeflateError::Gpu(
                                "gpu sparse pending batch removal failed".to_string(),
                            )
                        })?;
                        ready.push((prep_indices, Err(err)));
                    }
                }
            }
            Ok(ready)
        };

        for job_group in gpu_job_indices.chunks(submit_group) {
            for (prep_indices, batch_res) in drain_ready(&mut pending_batches)? {
                handle_integrated_batch(prep_indices, batch_res);
            }
            while pending_batches.len() >= target_inflight {
                let ready = drain_ready(&mut pending_batches)?;
                let drained = ready.len();
                for (prep_indices, batch_res) in ready {
                    handle_integrated_batch(prep_indices, batch_res);
                }
                if drained > 0 && pending_batches.len() < target_inflight {
                    break;
                }
                if pending_batches.len() >= wait_high_watermark || drained == 0 {
                    let Some((prep_indices, pending)) = pending_batches.pop_front() else {
                        break;
                    };
                    let batch_res =
                        gpu::collect_matches_encode_and_pack_sparse_batch(pending);
                    handle_integrated_batch(prep_indices, batch_res);
                }
            }

            let prep_indices = job_group.to_vec();
            let gpu_inputs: Vec<gpu::GpuMatchInput<'_>> = prep_indices
                .iter()
                .map(|&idx| {
                    let p = &prepared[idx];
                    gpu::GpuMatchInput {
                        src: p.chunk,
                        table: p.table.as_deref().unwrap_or(&[]),
                        table_gpu: p.gpu_table.as_ref(),
                        table_index: Some(&p.table_index),
                        table_data: Some(&p.table_data),
                        max_ref_len: p.max_ref_len,
                        min_ref_len: options.min_ref_len,
                        section_count: options.section_count,
                        huffman_encode_enabled: options.huffman_encode_enabled,
                        compression_mode: options.compression_mode,
                    }
                })
                .collect();
            match gpu::submit_matches_encode_and_pack_sparse_batch(
                &gpu_inputs,
                options.section_count,
                gpu_max_cmd_len,
            ) {
                Ok(pending) => {
                    pending_batches.push_back((prep_indices, pending));
                }
                Err(err) => {
                    handle_integrated_batch(prep_indices, Err(err));
                }
            }
            for (prep_indices, batch_res) in drain_ready(&mut pending_batches)? {
                handle_integrated_batch(prep_indices, batch_res);
            }
            while pending_batches.len() >= wait_high_watermark {
                let Some((prep_indices, pending)) = pending_batches.pop_front() else {
                    break;
                };
                let batch_res = gpu::collect_matches_encode_and_pack_sparse_batch(pending);
                handle_integrated_batch(prep_indices, batch_res);
                for (prep_indices, batch_res) in drain_ready(&mut pending_batches)? {
                    handle_integrated_batch(prep_indices, batch_res);
                }
            }
        }
        while !pending_batches.is_empty() {
            let ready = drain_ready(&mut pending_batches)?;
            if !ready.is_empty() {
                for (prep_indices, batch_res) in ready {
                    handle_integrated_batch(prep_indices, batch_res);
                }
                continue;
            }
            let Some((prep_indices, pending)) = pending_batches.pop_front() else {
                break;
            };
            let batch_res = gpu::collect_matches_encode_and_pack_sparse_batch(pending);
            handle_integrated_batch(prep_indices, batch_res);
        }

        if !fallback_match_job_indices.is_empty() {
            fallback_match_job_indices.sort_unstable();
            fallback_match_job_indices.dedup();
            gpu_fallback_match_chunks =
                gpu_fallback_match_chunks.saturating_add(fallback_match_job_indices.len());
            let fallback_inputs: Vec<gpu::GpuMatchInput<'_>> = fallback_match_job_indices
                .iter()
                .map(|&idx| {
                    let p = &prepared[idx];
                    gpu::GpuMatchInput {
                        src: p.chunk,
                        table: p.table.as_deref().unwrap_or(&[]),
                        table_gpu: p.gpu_table.as_ref(),
                        table_index: Some(&p.table_index),
                        table_data: Some(&p.table_data),
                        max_ref_len: p.max_ref_len,
                        min_ref_len: options.min_ref_len,
                        section_count: options.section_count,
                        huffman_encode_enabled: options.huffman_encode_enabled,
                        compression_mode: options.compression_mode,
                    }
                })
                .collect();
            match gpu::compute_matches_batch(&fallback_inputs) {
                Ok(batch) => {
                    gpu_match_upload_ms += batch.profile.upload_ms;
                    gpu_match_wait_ms += batch.profile.wait_ms;
                    gpu_match_map_copy_ms += batch.profile.map_copy_ms;
                    gpu_match_total_ms += batch.profile.total_ms;
                    if batch.chunks.len() == fallback_match_job_indices.len() {
                        let total_gpu_bytes: usize = fallback_match_job_indices
                            .iter()
                            .map(|&idx| prepared[idx].chunk.len())
                            .sum();
                        for (out_idx, chunk_out) in batch.chunks.into_iter().enumerate() {
                            let prep_idx = fallback_match_job_indices[out_idx];
                            if chunk_out.packed_matches.len() != prepared[prep_idx].chunk.len() {
                                fallback_match_packed_len_mismatch_chunks =
                                    fallback_match_packed_len_mismatch_chunks.saturating_add(1);
                                if profile_detail {
                                    eprintln!(
                                        "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason=packed_match_len_mismatch out_len={} src_len={}",
                                        indices.len(),
                                        fallback_match_job_indices.len(),
                                        chunk_out.packed_matches.len(),
                                        prepared[prep_idx].chunk.len()
                                    );
                                }
                                continue;
                            }
                            packed_matches_by_chunk[prep_idx] = Some(chunk_out.packed_matches);
                            let scale = if total_gpu_bytes == 0 {
                                0.0
                            } else {
                                (prepared[prep_idx].chunk.len() as f64) / (total_gpu_bytes as f64)
                            };
                            prof_by_chunk[prep_idx] = scale_gpu_profile(batch.profile, scale);
                            gpu_used_chunks = gpu_used_chunks.saturating_add(1);
                        }
                    } else if profile_detail {
                        fallback_match_output_mismatch_chunks =
                            fallback_match_output_mismatch_chunks
                                .saturating_add(fallback_match_job_indices.len());
                        eprintln!(
                            "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason=output_count_mismatch out={} jobs={}",
                            indices.len(),
                            fallback_match_job_indices.len(),
                            batch.chunks.len(),
                            fallback_match_job_indices.len()
                        );
                    }
                }
                Err(err) => {
                    fallback_match_error_chunks = fallback_match_error_chunks
                        .saturating_add(fallback_match_job_indices.len());
                    if profile_detail {
                        eprintln!(
                            "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason={}",
                            indices.len(),
                            fallback_match_job_indices.len(),
                            err
                        );
                    }
                }
            }
        }
    }
    let gpu_call_ms = elapsed_ms(t_gpu_call);

    let t_finalize = Instant::now();
    let mut out = Vec::with_capacity(prepared.len());
    for (idx, mut p) in prepared.into_iter().enumerate() {
        if let Some(chunk) = gpu_encoded_chunks[idx].take() {
            out.push((p.index, chunk));
            continue;
        }
        let packed_matches = packed_matches_by_chunk[idx].as_deref();

        // When match/section were computed against device-resident table entries,
        // serialize the exact same table to keep table-id space consistent.
        if packed_matches.is_some() && p.table_index.is_empty() {
            if let Some(table_gpu) = p.gpu_table.as_ref() {
                match gpu::readback_packed_table_device(table_gpu) {
                    Ok(packed_table) => {
                        p.table_count = packed_table.table_count;
                        p.table_index = packed_table.table_index;
                        p.table_data = packed_table.table_data;
                    }
                    Err(err) => {
                        return Err(PDeflateError::Gpu(format!(
                            "gpu table readback failed during finalize: {err}"
                        )));
                    }
                }
            }
        }

        if p.table.is_none() {
            if p.table_index.is_empty() {
                let (cpu_table, cpu_index, cpu_data) =
                    rebuild_table_cpu_for_fallback(p.chunk, options)?;
                p.table_count = cpu_table.len();
                p.table_index = cpu_index;
                p.table_data = cpu_data;
                p.table = Some(cpu_table);
            } else if p.table_count > 0 && packed_matches.is_none() {
                p.table = Some(materialize_table_entries(&p.table_index, &p.table_data)?);
            }
        }
        let chunk = finalize_chunk_from_table(
            p.chunk,
            options,
            p.table.unwrap_or_default(),
            Some(p.table_count),
            Some(&p.table_index),
            Some(&p.table_data),
            p.table_profile,
            p.table_build_ms,
            packed_matches,
            if packed_matches.is_some() {
                Some(prof_by_chunk[idx])
            } else {
                None
            },
            packed_matches.is_some(),
        )?;
        out.push((p.index, chunk));
    }
    let finalize_ms = elapsed_ms(t_finalize);
    let emit_gpu_batch_detail = if profile_enabled() {
        if !profile_detail {
            true
        } else {
        let seq = PROFILE_DETAIL_GPU_BATCH_SEQ.fetch_add(1, Ordering::Relaxed);
        profile_detail_should_log_stream(seq)
        }
    } else {
        false
    };
    if emit_gpu_batch_detail {
        let chunks_f = (indices.len() as f64).max(1.0);
        eprintln!(
            "[cozip_pdeflate][timing][gpu-batch] batch_chunks={} gpu_jobs={} gpu_used={} table_build_requested_chunks={} compression_mode={:?} t_prepare_ms={:.3} t_prepare_table_build_ms={:.3} t_prepare_table_readback_ms={:.3} t_prepare_misc_ms={:.3} prepare_gpu_table_build_chunks={} prepare_gpu_table_readback_chunks={} t_gpu_call_ms={:.3} t_finalize_ms={:.3} t_pack_inputs_ms={:.3} t_section_setup_ms={:.3} t_section_tok_dispatch_ms={:.3} t_section_pre_dispatch_ms={:.3} t_section_scat_dispatch_ms={:.3} t_section_pack_dispatch_ms={:.3} t_section_meta_dispatch_ms={:.3} t_section_tok_wait_ms={:.3} t_section_pre_wait_ms={:.3} t_section_scat_wait_ms={:.3} t_section_pack_wait_ms={:.3} t_section_meta_wait_ms={:.3} t_sparse_lens_submit_done_wait_ms={:.3} t_sparse_lens_map_after_done_ms={:.3} sparse_lens_poll_calls={} sparse_lens_yield_calls={} t_sparse_lens_wait_ms={:.3} t_sparse_wait_ms={:.3} t_sparse_copy_ms={:.3} t_upload_ms={:.3} t_wait_ms={:.3} t_map_copy_ms={:.3} gpu_call_ms_per_chunk={:.3}",
            indices.len(),
            gpu_job_indices.len(),
            gpu_used_chunks,
            gpu_table_build_requested_chunks,
            options.compression_mode,
            prepare_ms,
            prepare_table_build_ms,
            prepare_table_readback_ms,
            prepare_misc_ms,
            prepare_gpu_table_build_chunks,
            prepare_gpu_table_readback_chunks,
            gpu_call_ms,
            finalize_ms,
            kernel_profile.pack_inputs_ms,
            kernel_profile.section_setup_ms,
            kernel_profile.section_tokenize_dispatch_ms,
            kernel_profile.section_prefix_dispatch_ms,
            kernel_profile.section_scatter_dispatch_ms,
            kernel_profile.section_pack_dispatch_ms,
            kernel_profile.section_meta_dispatch_ms,
            kernel_profile.section_tokenize_wait_ms,
            kernel_profile.section_prefix_wait_ms,
            kernel_profile.section_scatter_wait_ms,
            kernel_profile.section_pack_wait_ms,
            kernel_profile.section_meta_wait_ms,
            kernel_profile.sparse_lens_submit_done_wait_ms,
            kernel_profile.sparse_lens_map_after_done_ms,
            kernel_profile.sparse_lens_poll_calls,
            kernel_profile.sparse_lens_yield_calls,
            kernel_profile.sparse_lens_wait_ms,
            kernel_profile.sparse_wait_ms,
            kernel_profile.sparse_copy_ms,
            gpu_match_upload_ms + gpu_section_encode_upload_ms + gpu_sparse_pack_upload_ms,
            gpu_match_wait_ms + gpu_section_encode_wait_ms + gpu_sparse_pack_wait_ms,
            gpu_match_map_copy_ms + gpu_section_encode_map_copy_ms + gpu_sparse_pack_map_copy_ms,
            gpu_call_ms / chunks_f
        );
    }
    Ok((
        out,
        GpuBatchCompressProfile {
            chunks_total: indices.len(),
            gpu_job_chunks: gpu_job_indices.len(),
            gpu_used_chunks,
            gpu_integrated_chunks,
            gpu_fallback_match_chunks,
            fallback_integrated_error_chunks,
            fallback_integrated_invalid_options_chunks,
            fallback_integrated_invalid_stream_chunks,
            fallback_integrated_numeric_overflow_chunks,
            fallback_integrated_gpu_error_chunks,
            fallback_integrated_other_error_chunks,
            fallback_integrated_output_mismatch_chunks,
            fallback_match_error_chunks,
            fallback_match_output_mismatch_chunks,
            fallback_match_packed_len_mismatch_chunks,
            fallback_integrated_error_example,
            prepare_ms,
            prepare_table_build_ms,
            prepare_table_readback_ms,
            prepare_misc_ms,
            prepare_gpu_table_build_chunks,
            prepare_gpu_table_readback_chunks,
            gpu_call_ms,
            gpu_match_total_ms,
            gpu_match_upload_ms,
            gpu_match_wait_ms,
            gpu_match_map_copy_ms,
            gpu_sparse_pack_upload_ms,
            gpu_sparse_pack_wait_ms,
            gpu_sparse_pack_map_copy_ms,
            gpu_upload_ms: gpu_match_upload_ms
                + gpu_section_encode_upload_ms
                + gpu_sparse_pack_upload_ms,
            gpu_wait_ms: gpu_match_wait_ms + gpu_section_encode_wait_ms + gpu_sparse_pack_wait_ms,
            gpu_map_copy_ms: gpu_match_map_copy_ms
                + gpu_section_encode_map_copy_ms
                + gpu_sparse_pack_map_copy_ms,
            gpu_section_encode_upload_ms,
            gpu_section_encode_wait_ms,
            gpu_section_encode_map_copy_ms,
            gpu_section_encode_total_ms,
            gpu_pack_inputs_ms: kernel_profile.pack_inputs_ms,
            gpu_pack_alloc_setup_ms: kernel_profile.pack_alloc_setup_ms,
            gpu_pack_resolve_sizes_ms: kernel_profile.pack_resolve_sizes_ms,
            gpu_pack_resolve_scan_ms: kernel_profile.pack_resolve_scan_ms,
            gpu_pack_resolve_readback_setup_ms: kernel_profile.pack_resolve_readback_setup_ms,
            gpu_pack_resolve_submit_ms: kernel_profile.pack_resolve_submit_ms,
            gpu_pack_resolve_map_wait_ms: kernel_profile.pack_resolve_map_wait_ms,
            gpu_pack_resolve_parse_ms: kernel_profile.pack_resolve_parse_ms,
            gpu_pack_src_copy_ms: kernel_profile.pack_src_copy_ms,
            gpu_pack_metadata_loop_ms: kernel_profile.pack_metadata_loop_ms,
            gpu_pack_host_copy_ms: kernel_profile.pack_host_copy_ms,
            gpu_pack_device_copy_plan_ms: kernel_profile.pack_device_copy_plan_ms,
            gpu_pack_finalize_ms: kernel_profile.pack_finalize_ms,
            gpu_scratch_acquire_ms: kernel_profile.scratch_acquire_ms,
            gpu_match_table_copy_ms: kernel_profile.match_table_copy_ms,
            gpu_match_prepare_dispatch_ms: kernel_profile.match_prepare_dispatch_ms,
            gpu_match_kernel_dispatch_ms: kernel_profile.match_kernel_dispatch_ms,
            gpu_match_submit_ms: kernel_profile.match_submit_ms,
            gpu_section_setup_ms: kernel_profile.section_setup_ms,
            gpu_section_pass_dispatch_ms: kernel_profile.section_pass_dispatch_ms,
            gpu_section_tokenize_dispatch_ms: kernel_profile.section_tokenize_dispatch_ms,
            gpu_section_prefix_dispatch_ms: kernel_profile.section_prefix_dispatch_ms,
            gpu_section_scatter_dispatch_ms: kernel_profile.section_scatter_dispatch_ms,
            gpu_section_pack_dispatch_ms: kernel_profile.section_pack_dispatch_ms,
            gpu_section_meta_dispatch_ms: kernel_profile.section_meta_dispatch_ms,
            gpu_section_copy_dispatch_ms: kernel_profile.section_copy_dispatch_ms,
            gpu_section_submit_ms: kernel_profile.section_submit_ms,
            gpu_section_tokenize_wait_ms: kernel_profile.section_tokenize_wait_ms,
            gpu_section_prefix_wait_ms: kernel_profile.section_prefix_wait_ms,
            gpu_section_scatter_wait_ms: kernel_profile.section_scatter_wait_ms,
            gpu_section_pack_wait_ms: kernel_profile.section_pack_wait_ms,
            gpu_section_meta_wait_ms: kernel_profile.section_meta_wait_ms,
            gpu_section_wrap_ms: kernel_profile.section_wrap_ms,
            gpu_sparse_lens_submit_ms,
            gpu_sparse_lens_submit_done_wait_ms,
            gpu_sparse_lens_map_after_done_ms,
            gpu_sparse_lens_poll_calls,
            gpu_sparse_lens_yield_calls,
            gpu_sparse_lens_wait_ms,
            gpu_sparse_lens_copy_ms,
            gpu_sparse_prepare_ms,
            gpu_sparse_table_size_resolve_ms,
            gpu_sparse_prepare_misc_ms,
            gpu_sparse_scratch_acquire_ms,
            gpu_sparse_upload_dispatch_ms,
            gpu_sparse_submit_ms,
            gpu_sparse_wait_ms,
            gpu_sparse_copy_ms,
            gpu_sparse_total_ms,
            gpu_call_pack_bind_group_ms: kernel_profile.call_pack_bind_group_ms,
            gpu_call_encoder_create_ms: kernel_profile.call_encoder_create_ms,
            gpu_call_table_stream_copy_ms: kernel_profile.call_table_stream_copy_ms,
            gpu_call_match_clear_ms: kernel_profile.call_match_clear_ms,
            gpu_call_sparse_prepare_dispatch_ms: kernel_profile.call_sparse_prepare_dispatch_ms,
            gpu_call_sparse_pack_dispatch_ms: kernel_profile.call_sparse_pack_dispatch_ms,
            gpu_call_result_readback_setup_ms: kernel_profile.call_result_readback_setup_ms,
            gpu_call_payload_readback_setup_ms: kernel_profile.call_payload_readback_setup_ms,
            gpu_post_total_pack_bytes_ms,
            gpu_post_scale_profiles_ms,
            gpu_post_chunk_materialize_ms,
            finalize_ms,
            total_ms: elapsed_ms(t_total),
        },
    ))
}

fn compress_chunks_hybrid(
    input: &[u8],
    chunk_size: usize,
    chunk_count: usize,
    options: &PDeflateOptions,
) -> Result<Vec<ChunkCompressed>, PDeflateError> {
    let gpu_enabled = options.gpu_compress_enabled && gpu::is_runtime_available();
    let mut initial_queue = VecDeque::with_capacity(chunk_count);
    let mut initial_cpu_queue = VecDeque::with_capacity(chunk_count);
    let mut initial_gpu_queue = VecDeque::with_capacity(chunk_count);
    for i in 0..chunk_count {
        let start = i.saturating_mul(chunk_size);
        let end = (start + chunk_size).min(input.len());
        let gpu_eligible = gpu_enabled && end.saturating_sub(start) >= options.gpu_min_chunk_size;
        match options.hybrid_scheduler_policy {
            PDeflateHybridSchedulerPolicy::GlobalQueue => {
                initial_queue.push_back(i);
            }
            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                if gpu_eligible {
                    initial_gpu_queue.push_back(i);
                } else {
                    initial_cpu_queue.push_back(i);
                }
            }
        }
    }
    let queue_state = Arc::new(Mutex::new(CompressTaskQueueState {
        queue: initial_queue,
        cpu_queue: initial_cpu_queue,
        gpu_queue: initial_gpu_queue,
    }));
    let results = Arc::new(Mutex::new(
        (0..chunk_count)
            .map(|_| None::<ChunkCompressed>)
            .collect::<Vec<_>>(),
    ));
    let err_slot = Arc::new(Mutex::new(None::<PDeflateError>));
    let err_flag = Arc::new(AtomicBool::new(false));

    let gpu_workers = if gpu_enabled {
        options.gpu_workers.max(1).min(chunk_count.max(1))
    } else {
        0
    };
    let cpu_workers = cpu_worker_count().min(chunk_count.max(1));
    let gpu_batch_limit = options.gpu_slot_count.max(options.gpu_submit_chunks).max(1);
    let prof_enabled = profile_enabled();
    let sched_prof = Arc::new(Mutex::new(HybridCompressSchedulerProfile::default()));

    thread::scope(|scope| {
        for _ in 0..cpu_workers {
            let queue_state = Arc::clone(&queue_state);
            let results = Arc::clone(&results);
            let err_slot = Arc::clone(&err_slot);
            let err_flag = Arc::clone(&err_flag);
            let sched_prof = Arc::clone(&sched_prof);
            scope.spawn(move || {
                loop {
                    if err_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let t_lock = Instant::now();
                    let mut should_retry = false;
                    let chunk_idx = {
                        let mut queue = match queue_state.lock() {
                            Ok(v) => v,
                            Err(_) => {
                                set_worker_error_once(
                                    &err_slot,
                                    &err_flag,
                                    PDeflateError::Gpu(
                                        "compression queue mutex poisoned".to_string(),
                                    ),
                                );
                                return;
                            }
                        };
                        match options.hybrid_scheduler_policy {
                            PDeflateHybridSchedulerPolicy::GlobalQueue => queue.queue.pop_front(),
                            PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                                if let Some(idx) = queue.cpu_queue.pop_front() {
                                    Some(idx)
                                } else if queue.gpu_queue.is_empty() {
                                    None
                                } else if should_stop_gpu_tail_pop(
                                    chunk_count,
                                    queue.gpu_queue.len(),
                                    options.gpu_tail_stop_ratio,
                                ) {
                                    queue.gpu_queue.pop_front()
                                } else {
                                    should_retry = true;
                                    None
                                }
                            }
                        }
                    };
                    if prof_enabled {
                        if let Ok(mut prof) = sched_prof.lock() {
                            prof.cpu_lock_wait_ms += elapsed_ms(t_lock);
                        }
                    }
                    if should_retry {
                        thread::yield_now();
                        continue;
                    }
                    let Some(chunk_idx) = chunk_idx else {
                        if prof_enabled {
                            if let Ok(mut prof) = sched_prof.lock() {
                                prof.cpu_no_task_events = prof.cpu_no_task_events.saturating_add(1);
                            }
                        }
                        break;
                    };
                    let start = chunk_idx.saturating_mul(chunk_size);
                    let end = (start + chunk_size).min(input.len());
                    let t_cpu_encode = Instant::now();
                    match compress_chunk_cpu_only(&input[start..end], options) {
                        Ok(chunk) => {
                            if prof_enabled {
                                if let Ok(mut prof) = sched_prof.lock() {
                                    prof.cpu_chunks = prof.cpu_chunks.saturating_add(1);
                                    prof.cpu_encode_ms += elapsed_ms(t_cpu_encode);
                                }
                            }
                            let t_result_lock = Instant::now();
                            let mut slots = match results.lock() {
                                Ok(v) => v,
                                Err(_) => {
                                    set_worker_error_once(
                                        &err_slot,
                                        &err_flag,
                                        PDeflateError::Gpu(
                                            "compression result mutex poisoned".to_string(),
                                        ),
                                    );
                                    return;
                                }
                            };
                            if prof_enabled {
                                if let Ok(mut prof) = sched_prof.lock() {
                                    prof.cpu_result_lock_wait_ms += elapsed_ms(t_result_lock);
                                }
                            }
                            if slots[chunk_idx].is_some() {
                                set_worker_error_once(
                                    &err_slot,
                                    &err_flag,
                                    PDeflateError::Gpu(
                                        "duplicate compressed chunk index".to_string(),
                                    ),
                                );
                                return;
                            }
                            slots[chunk_idx] = Some(chunk);
                        }
                        Err(err) => {
                            set_worker_error_once(&err_slot, &err_flag, err);
                            return;
                        }
                    }
                }
            });
        }

        if gpu_enabled {
            for _ in 0..gpu_workers {
                let queue_state = Arc::clone(&queue_state);
                let results = Arc::clone(&results);
                let err_slot = Arc::clone(&err_slot);
                let err_flag = Arc::clone(&err_flag);
                let sched_prof = Arc::clone(&sched_prof);
                scope.spawn(move || {
                    loop {
                        if err_flag.load(Ordering::Relaxed) {
                            break;
                        }
                        let t_lock = Instant::now();
                        let batch_indices = {
                            let mut queue = match queue_state.lock() {
                                Ok(v) => v,
                                Err(_) => {
                                    set_worker_error_once(
                                        &err_slot,
                                        &err_flag,
                                        PDeflateError::Gpu(
                                            "compression queue mutex poisoned".to_string(),
                                        ),
                                    );
                                    return;
                                }
                            };
                            match options.hybrid_scheduler_policy {
                                PDeflateHybridSchedulerPolicy::GlobalQueue => {
                                    let mut picked = Vec::with_capacity(gpu_batch_limit);
                                    while picked.len() < gpu_batch_limit {
                                        let pos = queue.queue.iter().position(|&idx| {
                                            let start = idx.saturating_mul(chunk_size);
                                            let end = (start + chunk_size).min(input.len());
                                            end.saturating_sub(start) >= options.gpu_min_chunk_size
                                        });
                                        if let Some(pos) = pos {
                                            if let Some(idx) = queue.queue.remove(pos) {
                                                picked.push(idx);
                                            }
                                        } else {
                                            break;
                                        }
                                    }
                                    picked
                                }
                                PDeflateHybridSchedulerPolicy::GpuLedSplitQueue => {
                                    if should_stop_gpu_tail_pop(
                                        chunk_count,
                                        queue.gpu_queue.len(),
                                        options.gpu_tail_stop_ratio,
                                    ) {
                                        Vec::new()
                                    } else {
                                        let mut picked = Vec::with_capacity(gpu_batch_limit);
                                        while picked.len() < gpu_batch_limit {
                                            if let Some(idx) = queue.gpu_queue.pop_front() {
                                                picked.push(idx);
                                            } else {
                                                break;
                                            }
                                        }
                                        picked
                                    }
                                }
                            }
                        };
                        if prof_enabled {
                            if let Ok(mut prof) = sched_prof.lock() {
                                prof.gpu_lock_wait_ms += elapsed_ms(t_lock);
                            }
                        }
                        if batch_indices.is_empty() {
                            if prof_enabled {
                                if let Ok(mut prof) = sched_prof.lock() {
                                    prof.gpu_no_task_events =
                                        prof.gpu_no_task_events.saturating_add(1);
                                }
                            }
                            break;
                        }
                        let t_gpu_batch = Instant::now();
                        match compress_chunk_gpu_batch(input, chunk_size, &batch_indices, options) {
                            Ok((batch, batch_prof)) => {
                                if prof_enabled {
                                    if let Ok(mut prof) = sched_prof.lock() {
                                        prof.gpu_batches = prof.gpu_batches.saturating_add(1);
                                        prof.gpu_worker_busy_ms += elapsed_ms(t_gpu_batch);
                                        prof.gpu_chunks_claimed = prof
                                            .gpu_chunks_claimed
                                            .saturating_add(batch_prof.chunks_total);
                                        prof.gpu_chunks_encoded =
                                            prof.gpu_chunks_encoded.saturating_add(batch.len());
                                        prof.gpu_job_chunks = prof
                                            .gpu_job_chunks
                                            .saturating_add(batch_prof.gpu_job_chunks);
                                        prof.gpu_used_chunks = prof
                                            .gpu_used_chunks
                                            .saturating_add(batch_prof.gpu_used_chunks);
                                        prof.gpu_prepare_ms += batch_prof.prepare_ms;
                                        prof.gpu_call_ms += batch_prof.gpu_call_ms;
                                        prof.gpu_match_total_ms += batch_prof.gpu_match_total_ms;
                                        prof.gpu_match_upload_ms += batch_prof.gpu_match_upload_ms;
                                        prof.gpu_match_wait_ms += batch_prof.gpu_match_wait_ms;
                                        prof.gpu_match_map_copy_ms +=
                                            batch_prof.gpu_match_map_copy_ms;
                                        prof.gpu_sparse_pack_upload_ms +=
                                            batch_prof.gpu_sparse_pack_upload_ms;
                                        prof.gpu_sparse_pack_wait_ms +=
                                            batch_prof.gpu_sparse_pack_wait_ms;
                                        prof.gpu_sparse_pack_map_copy_ms +=
                                            batch_prof.gpu_sparse_pack_map_copy_ms;
                                        prof.gpu_upload_ms += batch_prof.gpu_upload_ms;
                                        prof.gpu_wait_ms += batch_prof.gpu_wait_ms;
                                        prof.gpu_map_copy_ms += batch_prof.gpu_map_copy_ms;
                                        prof.gpu_section_encode_upload_ms +=
                                            batch_prof.gpu_section_encode_upload_ms;
                                        prof.gpu_section_encode_wait_ms +=
                                            batch_prof.gpu_section_encode_wait_ms;
                                        prof.gpu_section_encode_map_copy_ms +=
                                            batch_prof.gpu_section_encode_map_copy_ms;
                                        prof.gpu_section_encode_total_ms +=
                                            batch_prof.gpu_section_encode_total_ms;
                                        prof.gpu_pack_inputs_ms += batch_prof.gpu_pack_inputs_ms;
                                        prof.gpu_pack_alloc_setup_ms +=
                                            batch_prof.gpu_pack_alloc_setup_ms;
                                        prof.gpu_pack_resolve_sizes_ms +=
                                            batch_prof.gpu_pack_resolve_sizes_ms;
                                        prof.gpu_pack_resolve_scan_ms +=
                                            batch_prof.gpu_pack_resolve_scan_ms;
                                        prof.gpu_pack_resolve_readback_setup_ms +=
                                            batch_prof.gpu_pack_resolve_readback_setup_ms;
                                        prof.gpu_pack_resolve_submit_ms +=
                                            batch_prof.gpu_pack_resolve_submit_ms;
                                        prof.gpu_pack_resolve_map_wait_ms +=
                                            batch_prof.gpu_pack_resolve_map_wait_ms;
                                        prof.gpu_pack_resolve_parse_ms +=
                                            batch_prof.gpu_pack_resolve_parse_ms;
                                        prof.gpu_pack_src_copy_ms +=
                                            batch_prof.gpu_pack_src_copy_ms;
                                        prof.gpu_pack_metadata_loop_ms +=
                                            batch_prof.gpu_pack_metadata_loop_ms;
                                        prof.gpu_pack_host_copy_ms +=
                                            batch_prof.gpu_pack_host_copy_ms;
                                        prof.gpu_pack_device_copy_plan_ms +=
                                            batch_prof.gpu_pack_device_copy_plan_ms;
                                        prof.gpu_pack_finalize_ms +=
                                            batch_prof.gpu_pack_finalize_ms;
                                        prof.gpu_scratch_acquire_ms +=
                                            batch_prof.gpu_scratch_acquire_ms;
                                        prof.gpu_match_table_copy_ms +=
                                            batch_prof.gpu_match_table_copy_ms;
                                        prof.gpu_match_prepare_dispatch_ms +=
                                            batch_prof.gpu_match_prepare_dispatch_ms;
                                        prof.gpu_match_kernel_dispatch_ms +=
                                            batch_prof.gpu_match_kernel_dispatch_ms;
                                        prof.gpu_match_submit_ms += batch_prof.gpu_match_submit_ms;
                                        prof.gpu_section_setup_ms +=
                                            batch_prof.gpu_section_setup_ms;
                                        prof.gpu_section_pass_dispatch_ms +=
                                            batch_prof.gpu_section_pass_dispatch_ms;
                                        prof.gpu_section_tokenize_dispatch_ms +=
                                            batch_prof.gpu_section_tokenize_dispatch_ms;
                                        prof.gpu_section_prefix_dispatch_ms +=
                                            batch_prof.gpu_section_prefix_dispatch_ms;
                                        prof.gpu_section_scatter_dispatch_ms +=
                                            batch_prof.gpu_section_scatter_dispatch_ms;
                                        prof.gpu_section_pack_dispatch_ms +=
                                            batch_prof.gpu_section_pack_dispatch_ms;
                                        prof.gpu_section_meta_dispatch_ms +=
                                            batch_prof.gpu_section_meta_dispatch_ms;
                                        prof.gpu_section_copy_dispatch_ms +=
                                            batch_prof.gpu_section_copy_dispatch_ms;
                                        prof.gpu_section_submit_ms +=
                                            batch_prof.gpu_section_submit_ms;
                                        prof.gpu_section_tokenize_wait_ms +=
                                            batch_prof.gpu_section_tokenize_wait_ms;
                                        prof.gpu_section_prefix_wait_ms +=
                                            batch_prof.gpu_section_prefix_wait_ms;
                                        prof.gpu_section_scatter_wait_ms +=
                                            batch_prof.gpu_section_scatter_wait_ms;
                                        prof.gpu_section_pack_wait_ms +=
                                            batch_prof.gpu_section_pack_wait_ms;
                                        prof.gpu_section_meta_wait_ms +=
                                            batch_prof.gpu_section_meta_wait_ms;
                                        prof.gpu_section_wrap_ms += batch_prof.gpu_section_wrap_ms;
                                        prof.gpu_sparse_lens_submit_ms +=
                                            batch_prof.gpu_sparse_lens_submit_ms;
                                        prof.gpu_sparse_lens_submit_done_wait_ms +=
                                            batch_prof.gpu_sparse_lens_submit_done_wait_ms;
                                        prof.gpu_sparse_lens_map_after_done_ms +=
                                            batch_prof.gpu_sparse_lens_map_after_done_ms;
                                        prof.gpu_sparse_lens_poll_calls = prof
                                            .gpu_sparse_lens_poll_calls
                                            .saturating_add(batch_prof.gpu_sparse_lens_poll_calls);
                                        prof.gpu_sparse_lens_yield_calls = prof
                                            .gpu_sparse_lens_yield_calls
                                            .saturating_add(batch_prof.gpu_sparse_lens_yield_calls);
                                        prof.gpu_sparse_lens_wait_ms +=
                                            batch_prof.gpu_sparse_lens_wait_ms;
                                        prof.gpu_sparse_lens_copy_ms +=
                                            batch_prof.gpu_sparse_lens_copy_ms;
                                        prof.gpu_sparse_prepare_ms +=
                                            batch_prof.gpu_sparse_prepare_ms;
                                        prof.gpu_sparse_table_size_resolve_ms +=
                                            batch_prof.gpu_sparse_table_size_resolve_ms;
                                        prof.gpu_sparse_prepare_misc_ms +=
                                            batch_prof.gpu_sparse_prepare_misc_ms;
                                        prof.gpu_sparse_scratch_acquire_ms +=
                                            batch_prof.gpu_sparse_scratch_acquire_ms;
                                        prof.gpu_sparse_upload_dispatch_ms +=
                                            batch_prof.gpu_sparse_upload_dispatch_ms;
                                        prof.gpu_sparse_submit_ms +=
                                            batch_prof.gpu_sparse_submit_ms;
                                        prof.gpu_sparse_wait_ms += batch_prof.gpu_sparse_wait_ms;
                                        prof.gpu_sparse_copy_ms += batch_prof.gpu_sparse_copy_ms;
                                        prof.gpu_sparse_total_ms += batch_prof.gpu_sparse_total_ms;
                                        prof.gpu_call_pack_bind_group_ms +=
                                            batch_prof.gpu_call_pack_bind_group_ms;
                                        prof.gpu_call_encoder_create_ms +=
                                            batch_prof.gpu_call_encoder_create_ms;
                                        prof.gpu_call_table_stream_copy_ms +=
                                            batch_prof.gpu_call_table_stream_copy_ms;
                                        prof.gpu_call_match_clear_ms +=
                                            batch_prof.gpu_call_match_clear_ms;
                                        prof.gpu_call_sparse_prepare_dispatch_ms +=
                                            batch_prof.gpu_call_sparse_prepare_dispatch_ms;
                                        prof.gpu_call_sparse_pack_dispatch_ms +=
                                            batch_prof.gpu_call_sparse_pack_dispatch_ms;
                                        prof.gpu_call_result_readback_setup_ms +=
                                            batch_prof.gpu_call_result_readback_setup_ms;
                                        prof.gpu_call_payload_readback_setup_ms +=
                                            batch_prof.gpu_call_payload_readback_setup_ms;
                                        prof.gpu_post_total_pack_bytes_ms +=
                                            batch_prof.gpu_post_total_pack_bytes_ms;
                                        prof.gpu_post_scale_profiles_ms +=
                                            batch_prof.gpu_post_scale_profiles_ms;
                                        prof.gpu_post_chunk_materialize_ms +=
                                            batch_prof.gpu_post_chunk_materialize_ms;
                                        prof.gpu_finalize_ms += batch_prof.finalize_ms;
                                        prof.gpu_batch_total_ms += batch_prof.total_ms;
                                        prof.gpu_integrated_chunks = prof
                                            .gpu_integrated_chunks
                                            .saturating_add(batch_prof.gpu_integrated_chunks);
                                        prof.gpu_fallback_match_chunks = prof
                                            .gpu_fallback_match_chunks
                                            .saturating_add(batch_prof.gpu_fallback_match_chunks);
                                        prof.fallback_integrated_error_chunks = prof
                                            .fallback_integrated_error_chunks
                                            .saturating_add(
                                                batch_prof.fallback_integrated_error_chunks,
                                            );
                                        prof.fallback_integrated_invalid_options_chunks = prof
                                            .fallback_integrated_invalid_options_chunks
                                            .saturating_add(
                                                batch_prof
                                                    .fallback_integrated_invalid_options_chunks,
                                            );
                                        prof.fallback_integrated_invalid_stream_chunks = prof
                                            .fallback_integrated_invalid_stream_chunks
                                            .saturating_add(
                                                batch_prof
                                                    .fallback_integrated_invalid_stream_chunks,
                                            );
                                        prof.fallback_integrated_numeric_overflow_chunks = prof
                                            .fallback_integrated_numeric_overflow_chunks
                                            .saturating_add(
                                                batch_prof
                                                    .fallback_integrated_numeric_overflow_chunks,
                                            );
                                        prof.fallback_integrated_gpu_error_chunks = prof
                                            .fallback_integrated_gpu_error_chunks
                                            .saturating_add(
                                                batch_prof.fallback_integrated_gpu_error_chunks,
                                            );
                                        prof.fallback_integrated_other_error_chunks = prof
                                            .fallback_integrated_other_error_chunks
                                            .saturating_add(
                                                batch_prof.fallback_integrated_other_error_chunks,
                                            );
                                        prof.fallback_integrated_output_mismatch_chunks = prof
                                            .fallback_integrated_output_mismatch_chunks
                                            .saturating_add(
                                                batch_prof
                                                    .fallback_integrated_output_mismatch_chunks,
                                            );
                                        prof.fallback_match_error_chunks = prof
                                            .fallback_match_error_chunks
                                            .saturating_add(batch_prof.fallback_match_error_chunks);
                                        prof.fallback_match_output_mismatch_chunks = prof
                                            .fallback_match_output_mismatch_chunks
                                            .saturating_add(
                                                batch_prof
                                                    .fallback_match_output_mismatch_chunks,
                                            );
                                        prof.fallback_match_packed_len_mismatch_chunks = prof
                                            .fallback_match_packed_len_mismatch_chunks
                                            .saturating_add(
                                                batch_prof
                                                    .fallback_match_packed_len_mismatch_chunks,
                                            );
                                        if prof.fallback_integrated_error_example.is_none() {
                                            prof.fallback_integrated_error_example = batch_prof
                                                .fallback_integrated_error_example
                                                .clone();
                                        }
                                    }
                                }
                                let t_result_lock = Instant::now();
                                let mut slots = match results.lock() {
                                    Ok(v) => v,
                                    Err(_) => {
                                        set_worker_error_once(
                                            &err_slot,
                                            &err_flag,
                                            PDeflateError::Gpu(
                                                "compression result mutex poisoned".to_string(),
                                            ),
                                        );
                                        return;
                                    }
                                };
                                if prof_enabled {
                                    if let Ok(mut prof) = sched_prof.lock() {
                                        prof.gpu_result_lock_wait_ms += elapsed_ms(t_result_lock);
                                    }
                                }
                                for (idx, chunk) in batch {
                                    if slots[idx].is_some() {
                                        set_worker_error_once(
                                            &err_slot,
                                            &err_flag,
                                            PDeflateError::Gpu(
                                                "duplicate compressed chunk index".to_string(),
                                            ),
                                        );
                                        return;
                                    }
                                    slots[idx] = Some(chunk);
                                }
                            }
                            Err(err) => {
                                set_worker_error_once(&err_slot, &err_flag, err);
                                return;
                            }
                        }
                    }
                });
            }
        }
    });

    take_worker_error(&err_slot)?;
    let mut slots = results
        .lock()
        .map_err(|_| PDeflateError::Gpu("compression result mutex poisoned".to_string()))?;
    let mut out = Vec::with_capacity(chunk_count);
    for idx in 0..chunk_count {
        let Some(chunk) = slots[idx].take() else {
            return Err(PDeflateError::Gpu(format!(
                "missing compressed chunk at index {}",
                idx
            )));
        };
        out.push(chunk);
    }
    if prof_enabled {
        if let Ok(prof) = sched_prof.lock() {
            let gpu_batch_avg_chunks = if prof.gpu_batches > 0 {
                (prof.gpu_chunks_claimed as f64) / (prof.gpu_batches as f64)
            } else {
                0.0
            };
            let cpu_chunk_avg_ms = if prof.cpu_chunks > 0 {
                prof.cpu_encode_ms / (prof.cpu_chunks as f64)
            } else {
                0.0
            };
            let gpu_chunk_avg_ms_assigned = if prof.gpu_chunks_claimed > 0 {
                prof.gpu_worker_busy_ms / (prof.gpu_chunks_claimed as f64)
            } else {
                0.0
            };
            let gpu_chunk_avg_ms_encoded = if prof.gpu_chunks_encoded > 0 {
                prof.gpu_worker_busy_ms / (prof.gpu_chunks_encoded as f64)
            } else {
                0.0
            };
            let pct = |v: f64, total: f64| -> f64 {
                if total > 0.0 {
                    (v / total) * 100.0
                } else {
                    0.0
                }
            };
            eprintln!(
                "[cozip_pdeflate][timing][hybrid-scheduler] chunks={} cpu_workers={} gpu_workers={} gpu_enabled={} gpu_batch_limit={} scheduler={:?} gpu_tail_stop_ratio={:.2} cpu_chunks={} cpu_encode_ms={:.3} cpu_chunk_avg_ms={:.3} cpu_result_lock_wait_ms={:.3} gpu_batches={} gpu_assigned_chunks={} gpu_chunks_encoded={} gpu_job_chunks={} gpu_used_chunks={} gpu_batch_avg_chunks={:.2} gpu_worker_busy_ms={:.3} gpu_chunk_avg_ms_assigned={:.3} gpu_chunk_avg_ms_encoded={:.3} gpu_prepare_ms={:.3} gpu_call_ms={:.3} gpu_finalize_ms={:.3} gpu_prepare_pct={:.1} gpu_call_pct={:.1} gpu_finalize_pct={:.1} gpu_upload_ms={:.3} gpu_wait_ms={:.3} gpu_map_copy_ms={:.3} gpu_match_upload_ms={:.3} gpu_match_wait_ms={:.3} gpu_match_map_copy_ms={:.3} gpu_sparse_pack_upload_ms={:.3} gpu_sparse_pack_wait_ms={:.3} gpu_sparse_pack_map_copy_ms={:.3} gpu_section_encode_upload_ms={:.3} gpu_section_encode_wait_ms={:.3} gpu_section_encode_map_copy_ms={:.3} gpu_section_encode_total_ms={:.3} gpu_batch_total_ms={:.3} gpu_result_lock_wait_ms={:.3} cpu_lock_wait_ms={:.3} gpu_lock_wait_ms={:.3} cpu_no_task_events={} gpu_no_task_events={}",
                chunk_count,
                cpu_workers,
                gpu_workers,
                gpu_enabled,
                gpu_batch_limit,
                options.hybrid_scheduler_policy,
                options.gpu_tail_stop_ratio,
                prof.cpu_chunks,
                prof.cpu_encode_ms,
                cpu_chunk_avg_ms,
                prof.cpu_result_lock_wait_ms,
                prof.gpu_batches,
                prof.gpu_chunks_claimed,
                prof.gpu_chunks_encoded,
                prof.gpu_job_chunks,
                prof.gpu_used_chunks,
                gpu_batch_avg_chunks,
                prof.gpu_worker_busy_ms,
                gpu_chunk_avg_ms_assigned,
                gpu_chunk_avg_ms_encoded,
                prof.gpu_prepare_ms,
                prof.gpu_call_ms,
                prof.gpu_finalize_ms,
                pct(prof.gpu_prepare_ms, prof.gpu_batch_total_ms),
                pct(prof.gpu_call_ms, prof.gpu_batch_total_ms),
                pct(prof.gpu_finalize_ms, prof.gpu_batch_total_ms),
                prof.gpu_upload_ms,
                prof.gpu_wait_ms,
                prof.gpu_map_copy_ms,
                prof.gpu_match_upload_ms,
                prof.gpu_match_wait_ms,
                prof.gpu_match_map_copy_ms,
                prof.gpu_sparse_pack_upload_ms,
                prof.gpu_sparse_pack_wait_ms,
                prof.gpu_sparse_pack_map_copy_ms,
                prof.gpu_section_encode_upload_ms,
                prof.gpu_section_encode_wait_ms,
                prof.gpu_section_encode_map_copy_ms,
                prof.gpu_section_encode_total_ms,
                prof.gpu_batch_total_ms,
                prof.gpu_result_lock_wait_ms,
                prof.cpu_lock_wait_ms,
                prof.gpu_lock_wait_ms,
                prof.cpu_no_task_events,
                prof.gpu_no_task_events
            );
            eprintln!(
                "[cozip_pdeflate][timing][hybrid-summary] cpu_chunks={} gpu_assigned_chunks={} cpu_chunk_avg_ms={:.3} gpu_chunk_avg_ms_assigned={:.3}",
                prof.cpu_chunks,
                prof.gpu_chunks_claimed,
                cpu_chunk_avg_ms,
                gpu_chunk_avg_ms_assigned
            );
            let gpu_chunks_f = (prof.gpu_chunks_claimed as f64).max(1.0);
            let gpu_batches_f = (prof.gpu_batches as f64).max(1.0);
            eprintln!(
                "[cozip_pdeflate][timing][hybrid-bottleneck] gpu_batches={} gpu_chunks={} lens_wait_ms_per_batch={:.3} lens_wait_ms_per_chunk={:.3} sparse_wait_ms_per_chunk={:.3} pack_inputs_ms_per_chunk={:.3} gpu_call_ms_per_chunk={:.3}",
                prof.gpu_batches,
                prof.gpu_chunks_claimed,
                prof.gpu_sparse_lens_wait_ms / gpu_batches_f,
                prof.gpu_sparse_lens_wait_ms / gpu_chunks_f,
                prof.gpu_sparse_wait_ms / gpu_chunks_f,
                prof.gpu_pack_inputs_ms / gpu_chunks_f,
                prof.gpu_call_ms / gpu_chunks_f
            );
            let match_stage_ms = prof.gpu_match_table_copy_ms
                + prof.gpu_match_prepare_dispatch_ms
                + prof.gpu_match_kernel_dispatch_ms
                + prof.gpu_match_submit_ms;
            let section_stage_ms = prof.gpu_section_setup_ms
                + prof.gpu_section_pass_dispatch_ms
                + prof.gpu_section_copy_dispatch_ms
                + prof.gpu_section_submit_ms
                + prof.gpu_section_tokenize_wait_ms
                + prof.gpu_section_prefix_wait_ms
                + prof.gpu_section_scatter_wait_ms
                + prof.gpu_section_pack_wait_ms
                + prof.gpu_section_meta_wait_ms
                + prof.gpu_section_wrap_ms;
            let sparse_stage_ms = prof.gpu_sparse_lens_submit_ms
                + prof.gpu_sparse_lens_wait_ms
                + prof.gpu_sparse_lens_copy_ms
                + prof.gpu_sparse_prepare_ms
                + prof.gpu_sparse_scratch_acquire_ms
                + prof.gpu_sparse_upload_dispatch_ms
                + prof.gpu_sparse_submit_ms
                + prof.gpu_sparse_wait_ms
                + prof.gpu_sparse_copy_ms;
            let kernel_accounted_ms = prof.gpu_pack_inputs_ms
                + prof.gpu_scratch_acquire_ms
                + match_stage_ms
                + section_stage_ms
                + sparse_stage_ms;
            let gpu_call_unaccounted_ms = prof.gpu_call_ms - kernel_accounted_ms;
            let gpu_batch_unaccounted_ms = prof.gpu_batch_total_ms
                - (prof.gpu_prepare_ms + prof.gpu_call_ms + prof.gpu_finalize_ms);
            let gpu_call_post_ms = prof.gpu_post_total_pack_bytes_ms
                + prof.gpu_post_scale_profiles_ms
                + prof.gpu_post_chunk_materialize_ms;
            eprintln!(
                "[cozip_pdeflate][timing][hybrid-scheduler-kernel] gpu_batch_total_ms={:.3} kernel_accounted_ms={:.3} gpu_call_unaccounted_ms={:.3} gpu_batch_unaccounted_ms={:.3} match_stage_ms={:.3} section_stage_ms={:.3} sparse_stage_ms={:.3} match_stage_pct={:.1} section_stage_pct={:.1} sparse_stage_pct={:.1} pack_inputs_ms={:.3} scratch_acquire_ms={:.3} match_table_copy_ms={:.3} match_prepare_dispatch_ms={:.3} match_kernel_dispatch_ms={:.3} match_submit_ms={:.3} section_setup_ms={:.3} section_pass_dispatch_ms={:.3} section_tok_dispatch_ms={:.3} section_pre_dispatch_ms={:.3} section_scat_dispatch_ms={:.3} section_pack_dispatch_ms={:.3} section_meta_dispatch_ms={:.3} section_copy_dispatch_ms={:.3} section_submit_ms={:.3} section_tok_wait_ms={:.3} section_pre_wait_ms={:.3} section_scat_wait_ms={:.3} section_pack_wait_ms={:.3} section_meta_wait_ms={:.3} section_wrap_ms={:.3} sparse_lens_submit_ms={:.3} sparse_lens_submit_done_wait_ms={:.3} sparse_lens_map_after_done_ms={:.3} sparse_lens_poll_calls={} sparse_lens_yield_calls={} sparse_lens_wait_ms={:.3} sparse_lens_copy_ms={:.3} sparse_prepare_ms={:.3} sparse_table_size_resolve_ms={:.3} sparse_prepare_misc_ms={:.3} sparse_scratch_acquire_ms={:.3} sparse_upload_dispatch_ms={:.3} sparse_submit_ms={:.3} sparse_wait_ms={:.3} sparse_copy_ms={:.3} sparse_total_ms={:.3}",
                prof.gpu_batch_total_ms,
                kernel_accounted_ms,
                gpu_call_unaccounted_ms,
                gpu_batch_unaccounted_ms,
                match_stage_ms,
                section_stage_ms,
                sparse_stage_ms,
                pct(match_stage_ms, kernel_accounted_ms),
                pct(section_stage_ms, kernel_accounted_ms),
                pct(sparse_stage_ms, kernel_accounted_ms),
                prof.gpu_pack_inputs_ms,
                prof.gpu_scratch_acquire_ms,
                prof.gpu_match_table_copy_ms,
                prof.gpu_match_prepare_dispatch_ms,
                prof.gpu_match_kernel_dispatch_ms,
                prof.gpu_match_submit_ms,
                prof.gpu_section_setup_ms,
                prof.gpu_section_pass_dispatch_ms,
                prof.gpu_section_tokenize_dispatch_ms,
                prof.gpu_section_prefix_dispatch_ms,
                prof.gpu_section_scatter_dispatch_ms,
                prof.gpu_section_pack_dispatch_ms,
                prof.gpu_section_meta_dispatch_ms,
                prof.gpu_section_copy_dispatch_ms,
                prof.gpu_section_submit_ms,
                prof.gpu_section_tokenize_wait_ms,
                prof.gpu_section_prefix_wait_ms,
                prof.gpu_section_scatter_wait_ms,
                prof.gpu_section_pack_wait_ms,
                prof.gpu_section_meta_wait_ms,
                prof.gpu_section_wrap_ms,
                prof.gpu_sparse_lens_submit_ms,
                prof.gpu_sparse_lens_submit_done_wait_ms,
                prof.gpu_sparse_lens_map_after_done_ms,
                prof.gpu_sparse_lens_poll_calls,
                prof.gpu_sparse_lens_yield_calls,
                prof.gpu_sparse_lens_wait_ms,
                prof.gpu_sparse_lens_copy_ms,
                prof.gpu_sparse_prepare_ms,
                prof.gpu_sparse_table_size_resolve_ms,
                prof.gpu_sparse_prepare_misc_ms,
                prof.gpu_sparse_scratch_acquire_ms,
                prof.gpu_sparse_upload_dispatch_ms,
                prof.gpu_sparse_submit_ms,
                prof.gpu_sparse_wait_ms,
                prof.gpu_sparse_copy_ms,
                prof.gpu_sparse_total_ms
            );
            eprintln!(
                "[cozip_pdeflate][timing][hybrid-gpu-call-post] gpu_match_total_ms={:.3} post_total_pack_bytes_ms={:.3} post_scale_profiles_ms={:.3} post_chunk_materialize_ms={:.3} post_total_ms={:.3} residual_after_post_ms={:.3}",
                prof.gpu_match_total_ms,
                prof.gpu_post_total_pack_bytes_ms,
                prof.gpu_post_scale_profiles_ms,
                prof.gpu_post_chunk_materialize_ms,
                gpu_call_post_ms,
                (gpu_call_unaccounted_ms - gpu_call_post_ms).max(0.0)
            );
            eprintln!(
                "[cozip_pdeflate][timing][hybrid-gpu-integrated-fallbacks] integrated_error_chunks={} invalid_options_chunks={} invalid_stream_chunks={} numeric_overflow_chunks={} gpu_error_chunks={} other_error_chunks={} integrated_output_mismatch_chunks={} fallback_match_error_chunks={} fallback_match_output_mismatch_chunks={} fallback_match_packed_len_mismatch_chunks={}",
                prof.fallback_integrated_error_chunks,
                prof.fallback_integrated_invalid_options_chunks,
                prof.fallback_integrated_invalid_stream_chunks,
                prof.fallback_integrated_numeric_overflow_chunks,
                prof.fallback_integrated_gpu_error_chunks,
                prof.fallback_integrated_other_error_chunks,
                prof.fallback_integrated_output_mismatch_chunks,
                prof.fallback_match_error_chunks,
                prof.fallback_match_output_mismatch_chunks,
                prof.fallback_match_packed_len_mismatch_chunks
            );
            if let Some(example) = prof.fallback_integrated_error_example.as_deref() {
                eprintln!(
                    "[cozip_pdeflate][timing][hybrid-gpu-integrated-fallback-sample] error={}",
                    example
                );
            }
            eprintln!(
                "[cozip_pdeflate][timing][hybrid-kernel-pack-breakdown] pack_inputs_ms={:.3} alloc_setup_ms={:.3} resolve_sizes_ms={:.3} resolve_scan_ms={:.3} resolve_readback_setup_ms={:.3} resolve_submit_ms={:.3} resolve_map_wait_ms={:.3} resolve_parse_ms={:.3} src_copy_ms={:.3} metadata_loop_ms={:.3} host_copy_ms={:.3} device_copy_plan_ms={:.3} finalize_ms={:.3}",
                prof.gpu_pack_inputs_ms,
                prof.gpu_pack_alloc_setup_ms,
                prof.gpu_pack_resolve_sizes_ms,
                prof.gpu_pack_resolve_scan_ms,
                prof.gpu_pack_resolve_readback_setup_ms,
                prof.gpu_pack_resolve_submit_ms,
                prof.gpu_pack_resolve_map_wait_ms,
                prof.gpu_pack_resolve_parse_ms,
                prof.gpu_pack_src_copy_ms,
                prof.gpu_pack_metadata_loop_ms,
                prof.gpu_pack_host_copy_ms,
                prof.gpu_pack_device_copy_plan_ms,
                prof.gpu_pack_finalize_ms
            );
        }
    }
    Ok(out)
}

fn compress_chunk(
    chunk: &[u8],
    options: &PDeflateOptions,
) -> Result<ChunkCompressed, PDeflateError> {
    let mut built = build_table_dispatch(
        chunk,
        options,
        should_use_gpu_table_build(options, chunk.len()),
    )?;
    if built.table_count == 0 && built.table_index.is_empty() {
        if let Some(table_gpu) = built.gpu_table.as_ref() {
            let packed_table = gpu::readback_packed_table_device(table_gpu).map_err(|err| {
                PDeflateError::Gpu(format!(
                    "gpu table readback failed during single-chunk prepare: {err}"
                ))
            })?;
            built.table_count = packed_table.table_count;
            built.table_index = packed_table.table_index;
            built.table_data = packed_table.table_data;
        }
    }
    let mut table_index = built.table_index;
    let mut table_data = built.table_data;
    let mut table_count = built.table_count;
    let table_profile = built.profile;
    let table_build_ms = built.build_ms;
    let mut table = built.table;
    let max_ref_len = options.max_ref_len.min(MAX_CMD_LEN).min(64);
    let section_count = options.section_count;

    if table.is_none() {
        if table_index.is_empty() {
            let (cpu_table, cpu_index, cpu_data) = rebuild_table_cpu_for_fallback(chunk, options)?;
            table_count = cpu_table.len();
            table_index = cpu_index;
            table_data = cpu_data;
            table = Some(cpu_table);
        } else if table_count > 0 {
            table = Some(materialize_table_entries(&table_index, &table_data)?);
        }
    }

    let mut packed_matches: Option<Vec<u32>> = None;
    let mut gpu_profile = None;
    if options.gpu_compress_enabled
        && chunk.len() >= options.gpu_min_chunk_size
        && table_count > 0
        && max_ref_len >= options.min_ref_len
    {
        let gpu_input = gpu::GpuMatchInput {
            src: chunk,
            table: table.as_deref().unwrap_or(&[]),
            table_gpu: None,
            table_index: Some(&table_index),
            table_data: Some(&table_data),
            max_ref_len,
            min_ref_len: options.min_ref_len,
            section_count,
            huffman_encode_enabled: options.huffman_encode_enabled,
            compression_mode: options.compression_mode,
        };
        match gpu::compute_matches(&gpu_input) {
            Ok(out) => {
                if out.packed_matches.len() == chunk.len() {
                    packed_matches = Some(out.packed_matches);
                    gpu_profile = Some(out.profile);
                } else if profile_enabled() {
                    eprintln!(
                        "[cozip_pdeflate][timing][gpu-fallback] reason=packed_match_len_mismatch out_len={} src_len={}",
                        out.packed_matches.len(),
                        chunk.len()
                    );
                }
            }
            Err(err) => {
                if profile_enabled() {
                    eprintln!("[cozip_pdeflate][timing][gpu-fallback] reason={}", err);
                }
            }
        }
    }

    finalize_chunk_from_table(
        chunk,
        options,
        table.unwrap_or_default(),
        Some(table_count),
        Some(&table_index),
        Some(&table_data),
        table_profile,
        table_build_ms,
        packed_matches.as_deref(),
        gpu_profile,
        packed_matches.is_some(),
    )
}

fn encode_section_with_packed_matches(
    dst: &mut Vec<u8>,
    src: &[u8],
    packed_matches: &[u32],
    section_base: usize,
    options: &PDeflateOptions,
    max_ref_len: usize,
) -> Result<(usize, SectionEncodeProfile), PDeflateError> {
    let cmd_start = dst.len();
    let mut prof = SectionEncodeProfile::default();
    let prof_enabled = profile_enabled();
    if src.is_empty() {
        return Ok((0, prof));
    }

    let mut pos = 0usize;
    while pos < src.len() {
        let packed = packed_matches[section_base + pos];
        if let Some((id, mlen)) =
            unpack_gpu_match(packed, src.len() - pos, options.min_ref_len, max_ref_len)
        {
            let emit_t0 = if prof_enabled {
                Some(Instant::now())
            } else {
                None
            };
            emit_ref_command(dst, id as u16, mlen)?;
            if let Some(t0) = emit_t0 {
                prof.emit_ref_ms += elapsed_ms(t0);
            }
            prof.ref_cmds = prof.ref_cmds.saturating_add(1);
            prof.ref_bytes = prof.ref_bytes.saturating_add(mlen as u64);
            pos += mlen;
            continue;
        }

        let lit_start = pos;
        pos += 1;
        while pos < src.len() && (pos - lit_start) < MAX_CMD_LEN {
            let p = packed_matches[section_base + pos];
            if unpack_gpu_match(p, src.len() - pos, options.min_ref_len, max_ref_len).is_some() {
                break;
            }
            pos += 1;
        }
        let lit_run = &src[lit_start..pos];
        let emit_t0 = if prof_enabled {
            Some(Instant::now())
        } else {
            None
        };
        emit_literal_command(dst, lit_run)?;
        if let Some(t0) = emit_t0 {
            prof.emit_lit_ms += elapsed_ms(t0);
        }
        prof.lit_cmds = prof.lit_cmds.saturating_add(1);
        prof.lit_bytes = prof.lit_bytes.saturating_add(lit_run.len() as u64);
    }

    Ok((dst.len().saturating_sub(cmd_start), prof))
}

#[inline]
fn unpack_gpu_match(
    packed: u32,
    remaining: usize,
    min_ref_len: usize,
    max_ref_len: usize,
) -> Option<(usize, usize)> {
    if packed == 0 {
        return None;
    }
    let id = (packed >> 16) as usize;
    let len = (packed & 0xffff) as usize;
    if len < min_ref_len || len > max_ref_len || len > remaining {
        return None;
    }
    Some((id, len))
}

fn preprocess_chunk_for_gpu_decode(payload: &[u8]) -> Result<ChunkDecodePreprocess, PDeflateError> {
    if payload.len() < 12 {
        return Err(PDeflateError::InvalidStream("chunk header truncated"));
    }
    if payload.len() < CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream("chunk header truncated"));
    }

    let mut cursor = 0usize;
    let magic = read_exact(payload, &mut cursor, 4)?;
    if magic != CHUNK_MAGIC {
        return Err(PDeflateError::InvalidStream("bad chunk magic"));
    }
    let version = read_u16_le(payload, &mut cursor)?;
    if version != CHUNK_VERSION {
        return Err(PDeflateError::InvalidStream("unsupported chunk version"));
    }

    let flags = read_u16_le(payload, &mut cursor)?;
    if (flags & !(CHUNK_FLAG_HUFFMAN | CHUNK_FLAG_FINAL_STREAM)) != 0 {
        return Err(PDeflateError::InvalidStream("unsupported chunk flags"));
    }
    let huffman_enabled = chunk_huffman_enabled(flags);
    let chunk_uncompressed_len = read_u32_le(payload, &mut cursor)? as usize;
    let table_count = read_u16_le(payload, &mut cursor)? as usize;
    let section_count = read_u16_le(payload, &mut cursor)? as usize;
    let table_index_offset = read_u32_le(payload, &mut cursor)? as usize;
    let table_data_offset = read_u32_le(payload, &mut cursor)? as usize;
    let huff_lut_offset = read_u32_le(payload, &mut cursor)? as usize;
    let section_index_offset = read_u32_le(payload, &mut cursor)? as usize;
    let section_bitstream_offset = read_u32_le(payload, &mut cursor)? as usize;

    if table_count > MAX_TABLE_ID {
        return Err(PDeflateError::InvalidStream("table_count too large"));
    }
    if section_count == 0 {
        return Err(PDeflateError::InvalidStream("section_count is zero"));
    }

    if !(table_index_offset <= table_data_offset
        && table_data_offset <= huff_lut_offset
        && huff_lut_offset <= section_index_offset
        && section_index_offset <= section_bitstream_offset
        && section_bitstream_offset <= payload.len())
    {
        return Err(PDeflateError::InvalidStream("invalid chunk offsets"));
    }

    if table_index_offset != CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream(
            "unexpected table_index_offset",
        ));
    }
    let table_index = &payload[table_index_offset..table_data_offset];
    if table_index.len() != table_count {
        return Err(PDeflateError::InvalidStream("table index length mismatch"));
    }
    let section_index = &payload[section_index_offset..section_bitstream_offset];
    if huffman_enabled {
        let _ = deserialize_huffman_lut(&payload[huff_lut_offset..section_index_offset])?;
    }
    let section_cmd = &payload[section_bitstream_offset..];

    let mut section_idx_cursor = 0usize;
    let mut cmd_cursor = 0usize;
    let mut section_meta = Vec::with_capacity(section_count);
    for sec in 0..section_count {
        let sec_bit_len_u32 = read_varint_u32(section_index, &mut section_idx_cursor)?;
        let sec_bit_len =
            usize::try_from(sec_bit_len_u32).map_err(|_| PDeflateError::NumericOverflow)?;
        let sec_cmd_len = section_bit_len_to_byte_len(sec_bit_len)?;
        let sec_cmd_end = cmd_cursor
            .checked_add(sec_cmd_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        if sec_cmd_end > section_cmd.len() {
            return Err(PDeflateError::InvalidStream(
                "section cmd range out of bounds",
            ));
        }

        let out_start = section_start(sec, section_count, chunk_uncompressed_len);
        let out_end = section_start(sec + 1, section_count, chunk_uncompressed_len);
        section_meta.push(gpu::GpuDecodeSectionMeta {
            cmd_offset: cmd_cursor,
            cmd_len: sec_cmd_len,
            out_offset: out_start,
            out_len: out_end.saturating_sub(out_start),
        });
        cmd_cursor = sec_cmd_end;
    }
    if section_idx_cursor != section_index.len() {
        return Err(PDeflateError::InvalidStream(
            "trailing bytes in section index",
        ));
    }
    if cmd_cursor != section_cmd.len() {
        return Err(PDeflateError::InvalidStream(
            "sum(section_bit_len) != section_bitstream size",
        ));
    }

    Ok(ChunkDecodePreprocess {
        table_count,
        chunk_uncompressed_len,
        section_meta,
    })
}

fn decompress_chunk_into(payload: &[u8], out: &mut [u8]) -> Result<ChunkDecoded, PDeflateError> {
    if payload.len() < CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream("chunk header truncated"));
    }

    let mut cursor = 0usize;
    let magic = read_exact(payload, &mut cursor, 4)?;
    if magic != CHUNK_MAGIC {
        return Err(PDeflateError::InvalidStream("bad chunk magic"));
    }
    let version = read_u16_le(payload, &mut cursor)?;
    if version != CHUNK_VERSION {
        return Err(PDeflateError::InvalidStream("unsupported chunk version"));
    }

    let flags = read_u16_le(payload, &mut cursor)?;
    if (flags & !(CHUNK_FLAG_HUFFMAN | CHUNK_FLAG_FINAL_STREAM)) != 0 {
        return Err(PDeflateError::InvalidStream("unsupported chunk flags"));
    }
    let huffman_enabled = chunk_huffman_enabled(flags);
    let chunk_uncompressed_len = read_u32_le(payload, &mut cursor)? as usize;
    let table_count = read_u16_le(payload, &mut cursor)? as usize;
    let section_count = read_u16_le(payload, &mut cursor)? as usize;
    let table_index_offset = read_u32_le(payload, &mut cursor)? as usize;
    let table_data_offset = read_u32_le(payload, &mut cursor)? as usize;
    let huff_lut_offset = read_u32_le(payload, &mut cursor)? as usize;
    let section_index_offset = read_u32_le(payload, &mut cursor)? as usize;
    let section_bitstream_offset = read_u32_le(payload, &mut cursor)? as usize;

    if table_count > MAX_TABLE_ID {
        return Err(PDeflateError::InvalidStream("table_count too large"));
    }
    if section_count == 0 {
        return Err(PDeflateError::InvalidStream("section_count is zero"));
    }

    if !(table_index_offset <= table_data_offset
        && table_data_offset <= huff_lut_offset
        && huff_lut_offset <= section_index_offset
        && section_index_offset <= section_bitstream_offset
        && section_bitstream_offset <= payload.len())
    {
        return Err(PDeflateError::InvalidStream("invalid chunk offsets"));
    }

    if table_index_offset != CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream(
            "unexpected table_index_offset",
        ));
    }

    let table_index = &payload[table_index_offset..table_data_offset];
    if table_index.len() != table_count {
        return Err(PDeflateError::InvalidStream("table index length mismatch"));
    }

    if out.len() != chunk_uncompressed_len {
        return Err(PDeflateError::InvalidStream("chunk output len mismatch"));
    }
    let profile = profile_enabled();
    let mut decode_profile = ChunkDecodeProfile::default();
    let table_t0 = Instant::now();
    let table_data = &payload[table_data_offset..huff_lut_offset];
    let huff_lut = if huffman_enabled {
        Some(deserialize_huffman_lut(
            &payload[huff_lut_offset..section_index_offset],
        )?)
    } else {
        None
    };
    let section_index = &payload[section_index_offset..section_bitstream_offset];
    let section_cmd = &payload[section_bitstream_offset..];
    DECODE_SCRATCH.with(|scratch| -> Result<(), PDeflateError> {
        let mut scratch = scratch.borrow_mut();
        scratch.table_offsets.clear();
        scratch.table_offsets.push(0usize);
        for &entry_len_u8 in table_index {
            if entry_len_u8 == 0 {
                return Err(PDeflateError::InvalidStream("table entry len is zero"));
            }
            let next = scratch
                .table_offsets
                .last()
                .copied()
                .ok_or(PDeflateError::NumericOverflow)?
                .checked_add(entry_len_u8 as usize)
                .ok_or(PDeflateError::NumericOverflow)?;
            scratch.table_offsets.push(next);
        }

        if *scratch.table_offsets.last().unwrap_or(&0) != table_data.len() {
            return Err(PDeflateError::InvalidStream(
                "sum(table_entry_len) != table_data size",
            ));
        }

        if scratch.table_repeat.len() < table_count {
            scratch.table_repeat.resize(table_count, [0u8; MAX_CMD_LEN]);
        } else {
            scratch.table_repeat.truncate(table_count);
        }
        for id in 0..table_count {
            let t0 = scratch.table_offsets[id];
            let t1 = scratch.table_offsets[id + 1];
            let entry = table_data
                .get(t0..t1)
                .ok_or(PDeflateError::InvalidStream("table entry range invalid"))?;
            if entry.is_empty() {
                return Err(PDeflateError::InvalidStream("empty table entry"));
            }
            let rep = &mut scratch.table_repeat[id];
            fill_repeat(rep, entry);
        }
        decode_profile.table_prepare_ms = elapsed_ms(table_t0);

        let mut section_idx_cursor = 0usize;
        let mut cmd_cursor = 0usize;
        let mut section_bit_lens = Vec::with_capacity(section_count);
        let mut section_cmd_offsets = Vec::with_capacity(section_count);
        let mut section_cmd_lens = Vec::with_capacity(section_count);
        let mut section_out_offsets = Vec::with_capacity(section_count);
        let mut section_out_lens = Vec::with_capacity(section_count);
        for sec in 0..section_count {
            let sec_bit_len_u32 = read_varint_u32(section_index, &mut section_idx_cursor)?;
            let sec_bit_len =
                usize::try_from(sec_bit_len_u32).map_err(|_| PDeflateError::NumericOverflow)?;
            let sec_cmd_len = section_bit_len_to_byte_len(sec_bit_len)?;
            let sec_cmd_end = cmd_cursor
                .checked_add(sec_cmd_len)
                .ok_or(PDeflateError::NumericOverflow)?;
            if sec_cmd_end > section_cmd.len() {
                return Err(PDeflateError::InvalidStream(
                    "section cmd range out of bounds",
                ));
            }
            let out_start = section_start(sec, section_count, chunk_uncompressed_len);
            let out_end = section_start(sec + 1, section_count, chunk_uncompressed_len);
            let out_len = out_end - out_start;
            section_bit_lens
                .push(u32::try_from(sec_bit_len).map_err(|_| PDeflateError::NumericOverflow)?);
            section_cmd_offsets
                .push(u32::try_from(cmd_cursor).map_err(|_| PDeflateError::NumericOverflow)?);
            section_cmd_lens
                .push(u32::try_from(sec_cmd_len).map_err(|_| PDeflateError::NumericOverflow)?);
            section_out_offsets
                .push(u32::try_from(out_start).map_err(|_| PDeflateError::NumericOverflow)?);
            section_out_lens
                .push(u32::try_from(out_len).map_err(|_| PDeflateError::NumericOverflow)?);
            cmd_cursor = sec_cmd_end;
        }
        if section_idx_cursor != section_index.len() {
            return Err(PDeflateError::InvalidStream(
                "trailing bytes in section index",
            ));
        }
        if cmd_cursor != section_cmd.len() {
            return Err(PDeflateError::InvalidStream(
                "sum(section_bit_len) != section_bitstream size",
            ));
        }

        let sections_t0 = Instant::now();
        for sec in 0..section_count {
            let sec_cmd_off = section_cmd_offsets[sec] as usize;
            let sec_cmd_len = section_cmd_lens[sec] as usize;
            let sec_cmd_end = sec_cmd_off
                .checked_add(sec_cmd_len)
                .ok_or(PDeflateError::NumericOverflow)?;
            let out_start = section_out_offsets[sec] as usize;
            let out_len = section_out_lens[sec] as usize;
            let out_end = out_start
                .checked_add(out_len)
                .ok_or(PDeflateError::NumericOverflow)?;
            let sec_cmd =
                section_cmd
                    .get(sec_cmd_off..sec_cmd_end)
                    .ok_or(PDeflateError::InvalidStream(
                        "section cmd range out of bounds",
                    ))?;
            let sec_logical_cmd = if let Some(huff_lut) = huff_lut.as_ref() {
                let sec_symbols = decode_section_bitstream_to_huffman_symbols_with_lut(
                    sec_cmd,
                    section_bit_lens[sec] as usize,
                    huff_lut,
                )?;
                huffman_symbols_to_logical_commands(sec_symbols)
            } else {
                let byte_len = section_bit_len_to_byte_len(section_bit_lens[sec] as usize)?;
                if byte_len != sec_cmd.len() {
                    return Err(PDeflateError::InvalidStream(
                        "section bitstream length mismatch",
                    ));
                }
                sec_cmd.to_vec()
            };
            decode_section(
                &sec_logical_cmd,
                out_len,
                &scratch.table_repeat,
                &mut out[out_start..out_end],
                &mut decode_profile,
                profile,
            )?;
        }
        decode_profile.section_decode_ms = elapsed_ms(sections_t0);
        Ok(())
    })?;

    Ok(ChunkDecoded {
        table_entries: table_count,
        section_count,
        profile: decode_profile,
    })
}

fn decode_section(
    cmd_bytes: &[u8],
    expected_out_len: usize,
    table_repeat: &[[u8; MAX_CMD_LEN]],
    out: &mut [u8],
    prof: &mut ChunkDecodeProfile,
    profile: bool,
) -> Result<(), PDeflateError> {
    if out.len() != expected_out_len {
        return Err(PDeflateError::InvalidStream(
            "internal section size mismatch",
        ));
    }

    let mut cmd_cursor = 0usize;
    let mut out_cursor = 0usize;

    let cmd_len = cmd_bytes.len();
    while out_cursor < expected_out_len {
        if cmd_cursor + 2 > cmd_bytes.len() {
            return Err(PDeflateError::InvalidStream("unexpected eof"));
        }
        let cmd = unsafe {
            u16::from_le_bytes([
                *cmd_bytes.get_unchecked(cmd_cursor),
                *cmd_bytes.get_unchecked(cmd_cursor + 1),
            ])
        };
        cmd_cursor += 2;
        let tag = (cmd & 0x0fff) as usize;
        let len4 = ((cmd >> 12) & 0x000f) as usize;
        let len = if len4 < 0x0f {
            len4
        } else {
            if cmd_cursor >= cmd_bytes.len() {
                return Err(PDeflateError::InvalidStream("missing ext len"));
            }
            let ext = unsafe { *cmd_bytes.get_unchecked(cmd_cursor) as usize };
            cmd_cursor += 1;
            EXT_LEN_BASE + ext
        };

        if len == 0 {
            return Err(PDeflateError::InvalidStream("zero-length command"));
        }
        if len > MAX_CMD_LEN {
            return Err(PDeflateError::InvalidStream("command len out of range"));
        }
        if out_cursor + len > expected_out_len {
            return Err(PDeflateError::InvalidStream("section output overflow"));
        }
        if profile {
            prof.cmd_count = prof.cmd_count.saturating_add(1);
        }

        if tag == LITERAL_TAG as usize {
            let lit_end = cmd_cursor
                .checked_add(len)
                .ok_or(PDeflateError::NumericOverflow)?;
            if lit_end > cmd_len {
                return Err(PDeflateError::InvalidStream("unexpected eof"));
            }
            let t0 = if profile { Some(Instant::now()) } else { None };
            unsafe {
                ptr::copy_nonoverlapping(
                    cmd_bytes.as_ptr().add(cmd_cursor),
                    out.as_mut_ptr().add(out_cursor),
                    len,
                );
            }
            if let Some(start) = t0 {
                prof.lit_copy_ms += elapsed_ms(start);
            }
            if profile {
                prof.lit_cmds = prof.lit_cmds.saturating_add(1);
                prof.lit_bytes = prof.lit_bytes.saturating_add(len as u64);
            }
            cmd_cursor = lit_end;
            out_cursor += len;
            continue;
        }

        if tag >= table_repeat.len() {
            return Err(PDeflateError::InvalidStream("table id out of range"));
        }
        if len < 3 {
            return Err(PDeflateError::InvalidStream(
                "reference command with len < 3",
            ));
        }

        let rep = &table_repeat[tag];
        let t0 = if profile { Some(Instant::now()) } else { None };
        unsafe {
            ptr::copy_nonoverlapping(rep.as_ptr(), out.as_mut_ptr().add(out_cursor), len);
        }
        if let Some(start) = t0 {
            prof.ref_copy_ms += elapsed_ms(start);
        }
        if profile {
            prof.ref_cmds = prof.ref_cmds.saturating_add(1);
            prof.ref_bytes = prof.ref_bytes.saturating_add(len as u64);
        }
        out_cursor += len;
    }

    if cmd_cursor != cmd_bytes.len() {
        return Err(PDeflateError::InvalidStream(
            "trailing bytes in section command stream",
        ));
    }

    Ok(())
}

fn fill_repeat(dst: &mut [u8], pattern: &[u8]) {
    if dst.is_empty() || pattern.is_empty() {
        return;
    }
    if pattern.len() == 1 {
        dst.fill(pattern[0]);
        return;
    }
    let first = pattern.len().min(dst.len());
    dst[..first].copy_from_slice(&pattern[..first]);
    let mut filled = first;
    while filled < dst.len() {
        let copy_len = filled.min(dst.len() - filled);
        let (left, right) = dst.split_at_mut(filled);
        right[..copy_len].copy_from_slice(&left[..copy_len]);
        filled += copy_len;
    }
}

fn build_table_dispatch(
    chunk: &[u8],
    options: &PDeflateOptions,
    allow_gpu: bool,
) -> Result<BuiltTableDispatch, PDeflateError> {
    let t0 = Instant::now();
    if allow_gpu && options.gpu_compress_enabled && gpu::is_runtime_available() {
        let max_entries = options.max_table_entries.min(MAX_TABLE_ID);
        let (gpu_table_sample_stride, gpu_match_probe_limit, gpu_hash_history_limit) =
            gpu_table_build_tuning(options);
        if let Ok((gpu_table, gpu_prof)) = gpu::build_table_gpu_device(
            chunk,
            max_entries,
            options.max_table_entry_len,
            options.min_ref_len,
            gpu_match_probe_limit,
            gpu_hash_history_limit,
            gpu_table_sample_stride,
        ) {
            return Ok(BuiltTableDispatch {
                table: None,
                gpu_table: Some(gpu_table),
                table_count: 0,
                table_index: Vec::new(),
                table_data: Vec::new(),
                profile: BuildTableProfile {
                    freq_ms: gpu_prof.freq_kernel_ms,
                    probe_ms: gpu_prof.candidate_kernel_ms,
                    materialize_ms: gpu_prof.materialize_ms,
                    sort_ms: gpu_prof.sort_ms,
                    ..BuildTableProfile::default()
                },
                build_ms: elapsed_ms(t0),
            });
        }
    }
    let (table, prof) = build_table(chunk, options);
    let table_count = table.len();
    let (table_index, table_data) = serialize_table_entries(&table)?;
    Ok(BuiltTableDispatch {
        table: Some(table),
        gpu_table: None,
        table_count,
        table_index,
        table_data,
        profile: prof,
        build_ms: elapsed_ms(t0),
    })
}

fn gpu_table_build_tuning(options: &PDeflateOptions) -> (usize, usize, usize) {
    match options.compression_mode {
        PDeflateCompressionMode::Speed => (
            options.gpu_table_sample_stride.saturating_mul(4),
            options.match_probe_limit.min(8),
            options.hash_history_limit.min(8),
        ),
        PDeflateCompressionMode::Balanced => (
            options.gpu_table_sample_stride.saturating_mul(2),
            options.match_probe_limit.min(12),
            options.hash_history_limit.min(12),
        ),
        PDeflateCompressionMode::Ratio => (
            options.gpu_table_sample_stride,
            options.match_probe_limit,
            options.hash_history_limit,
        ),
    }
}

fn cpu_table_build_tuning(options: &PDeflateOptions) -> (usize, usize, usize, usize, usize) {
    match options.compression_mode {
        PDeflateCompressionMode::Speed => (
            options.table_sample_stride.saturating_mul(4),
            options.match_probe_limit.min(8),
            options.hash_history_limit.min(8),
            options.max_table_entries.min(128),
            options.max_table_entry_len.min(24),
        ),
        PDeflateCompressionMode::Balanced => (
            options.table_sample_stride.saturating_mul(2),
            options.match_probe_limit.min(12),
            options.hash_history_limit.min(12),
            options.max_table_entries.min(192),
            options.max_table_entry_len.min(28),
        ),
        PDeflateCompressionMode::Ratio => (
            options.table_sample_stride,
            options.match_probe_limit,
            options.hash_history_limit,
            options.max_table_entries,
            options.max_table_entry_len,
        ),
    }
}

fn should_use_gpu_table_build(options: &PDeflateOptions, chunk_len: usize) -> bool {
    options.gpu_compress_enabled
        && chunk_len >= options.gpu_min_chunk_size
        && options.compression_mode != PDeflateCompressionMode::Speed
        && gpu_table_build_enabled()
}

fn rebuild_table_cpu_for_fallback(
    chunk: &[u8],
    options: &PDeflateOptions,
) -> Result<(Vec<Vec<u8>>, Vec<u8>, Vec<u8>), PDeflateError> {
    let (table, _profile) = build_table(chunk, options);
    let (table_index, table_data) = serialize_table_entries(&table)?;
    Ok((table, table_index, table_data))
}

fn build_table(chunk: &[u8], options: &PDeflateOptions) -> (Vec<Vec<u8>>, BuildTableProfile) {
    let mut prof = BuildTableProfile::default();
    let detail_profile = profile_enabled();
    if chunk.len() < 3 {
        return (Vec::new(), prof);
    }

    let (table_sample_stride, match_probe_limit, hash_history_limit, tuned_max_entries, tuned_max_entry_len) =
        cpu_table_build_tuning(options);
    let max_entry_len = tuned_max_entry_len.min(254);
    let max_entries = tuned_max_entries.min(MAX_TABLE_ID);
    let sample_stride = table_sample_stride.max(1).saturating_mul(8);
    let probe_limit = match_probe_limit.max(1);
    let min_seed_match_len = options.min_ref_len.max(6);

    // zlib-style hot-path: fixed slots instead of HashMap entry churn.
    let mut seed_slots = Vec::<SeedSlot>::new();
    let mut used_seed_slot_indices = Vec::<usize>::new();
    let mut byte_scored = Vec::<(u8, u64)>::new();
    let mut scored_slot_indices = Vec::<(usize, u64)>::new();
    let mut head = Vec::<u32>::new();
    let mut touched_buckets = Vec::<usize>::new();
    let mut prev = Vec::<u32>::new();
    let mut keys = Vec::<u32>::new();
    let mut positions = Vec::<u32>::new();
    BUILD_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        std::mem::swap(&mut seed_slots, &mut scratch.seed_slots);
        std::mem::swap(
            &mut used_seed_slot_indices,
            &mut scratch.used_seed_slot_indices,
        );
        std::mem::swap(&mut byte_scored, &mut scratch.byte_scored);
        std::mem::swap(&mut scored_slot_indices, &mut scratch.scored_slot_indices);
        std::mem::swap(&mut head, &mut scratch.head);
        std::mem::swap(&mut touched_buckets, &mut scratch.touched_buckets);
        std::mem::swap(&mut prev, &mut scratch.prev);
        std::mem::swap(&mut keys, &mut scratch.keys);
        std::mem::swap(&mut positions, &mut scratch.positions);
    });
    for &idx in &used_seed_slot_indices {
        if let Some(slot) = seed_slots.get_mut(idx) {
            *slot = SeedSlot::default();
        }
    }
    used_seed_slot_indices.clear();
    byte_scored.clear();
    scored_slot_indices.clear();
    prev.clear();
    keys.clear();
    positions.clear();

    let freq_t0 = Instant::now();
    let mut byte_freq = [0u32; 256];
    for &b in chunk {
        byte_freq[b as usize] = byte_freq[b as usize].saturating_add(1);
    }
    for (b, &freq) in byte_freq.iter().enumerate() {
        if freq >= 8 {
            byte_scored.push((b as u8, u64::from(freq)));
        }
    }
    prof.freq_ms = elapsed_ms(freq_t0);

    let probe_t0 = Instant::now();
    const NO_POS: u32 = u32::MAX;
    let history_cap = hash_history_limit.max(1);
    let sampled_count = (chunk.len() - 2).div_ceil(sample_stride);
    let seed_slot_target = sampled_count.clamp(1 << 14, 1 << 18);
    let mut seed_slot_count = 1usize;
    while seed_slot_count < seed_slot_target {
        seed_slot_count <<= 1;
    }
    if seed_slots.len() != seed_slot_count {
        seed_slots.clear();
        seed_slots.resize(seed_slot_count, SeedSlot::default());
    }
    used_seed_slot_indices.reserve(seed_slot_count / 4);

    let target_hash_size = sampled_count.saturating_mul(2).clamp(1 << 15, 1 << 20);
    let mut hash_size = 1usize;
    while hash_size < target_hash_size {
        hash_size <<= 1;
    }
    let hash_mask = hash_size - 1;
    if head.len() < hash_size {
        head.resize(hash_size, NO_POS);
    }
    // Clear only previously touched buckets instead of whole-head fill.
    for &bucket in &touched_buckets {
        if bucket < hash_size {
            head[bucket] = NO_POS;
        }
    }
    touched_buckets.clear();
    if prev.capacity() < sampled_count {
        prev.reserve(sampled_count - prev.capacity());
    }
    if keys.capacity() < sampled_count {
        keys.reserve(sampled_count - keys.capacity());
    }
    if positions.capacity() < sampled_count {
        positions.reserve(sampled_count - positions.capacity());
    }

    let mut i = 0usize;
    while i + 2 < chunk.len() {
        if detail_profile {
            prof.positions_scanned = prof.positions_scanned.saturating_add(1);
        }
        let sample_idx = keys.len();
        let key = key3(chunk, i);
        keys.push(key);
        let bucket = hash3_to_bucket(key, hash_mask);

        let mut chain = head[bucket];
        prev.push(chain);
        positions.push(i as u32);
        if chain == NO_POS {
            touched_buckets.push(bucket);
        }
        head[bucket] = sample_idx as u32;

        let mut traversed = 0usize;
        let mut best_mlen_here = 0usize;
        while chain != NO_POS && traversed < history_cap && traversed < probe_limit {
            let prev_idx = chain as usize;
            if keys[prev_idx] != key {
                if detail_profile {
                    prof.hash_key_mismatch = prof.hash_key_mismatch.saturating_add(1);
                }
                chain = prev[prev_idx];
                traversed += 1;
                continue;
            }
            let prev_pos = positions[prev_idx] as usize;
            if detail_profile {
                prof.probe_pairs = prof.probe_pairs.saturating_add(1);
            }
            if prev_pos >= i {
                chain = prev[prev_idx];
                traversed += 1;
                continue;
            }
            // Fast anchor: first 3 bytes are equal by key3; require the 4th too.
            // This drops many weak candidates before match_len.
            if i + 3 < chunk.len()
                && prev_pos + 3 < chunk.len()
                && chunk[prev_pos + 3] != chunk[i + 3]
            {
                chain = prev[prev_idx];
                traversed += 1;
                continue;
            }
            let mlen = if i + 4 < chunk.len() && prev_pos + 4 < chunk.len() {
                if chunk[prev_pos + 4] != chunk[i + 4] {
                    // First 4 bytes are known-equal here; avoid full match_len call.
                    if detail_profile {
                        prof.anchored_short_hits = prof.anchored_short_hits.saturating_add(1);
                    }
                    4usize
                } else {
                    // zlib longest_match-style: grow anchor cheaply before full compare.
                    let mut anchored = 5usize;
                    let anchor_cap = 12usize
                        .min(max_entry_len)
                        .min(chunk.len().saturating_sub(i))
                        .min(chunk.len().saturating_sub(prev_pos));
                    while anchored < anchor_cap && chunk[prev_pos + anchored] == chunk[i + anchored]
                    {
                        anchored += 1;
                    }
                    if anchored < anchor_cap {
                        if detail_profile {
                            prof.anchored_short_hits = prof.anchored_short_hits.saturating_add(1);
                        }
                        anchored
                    } else {
                        if detail_profile {
                            prof.match_len_calls = prof.match_len_calls.saturating_add(1);
                        }
                        match_len_from(chunk, prev_pos, i, anchored, max_entry_len)
                    }
                }
            } else {
                if detail_profile {
                    prof.match_len_calls = prof.match_len_calls.saturating_add(1);
                }
                match_len(chunk, prev_pos, i, max_entry_len)
            };
            if mlen < min_seed_match_len {
                chain = prev[prev_idx];
                traversed += 1;
                continue;
            }
            if mlen > best_mlen_here {
                best_mlen_here = mlen;
            }
            let cand_len = choose_entry_len(mlen, max_entry_len);
            let gain = u64::try_from(mlen.saturating_sub(2)).unwrap_or(0);
            let fp = fingerprint_window(chunk, i, cand_len);
            let slot_mask = seed_slots.len() - 1;
            let mut slot_idx = (fp as usize) & slot_mask;
            let mut replacement_idx = slot_idx;
            let mut replacement_score = u64::MAX;
            let mut placed = false;
            for _ in 0..4 {
                let slot = &mut seed_slots[slot_idx];
                if slot.used == 0 {
                    slot.fingerprint = fp;
                    slot.pos = i as u32;
                    slot.len = cand_len as u16;
                    slot.score = gain;
                    slot.used = 1;
                    used_seed_slot_indices.push(slot_idx);
                    placed = true;
                    break;
                }
                if slot.fingerprint == fp
                    && slot.len as usize == cand_len
                    && windows_equal(chunk, slot.pos as usize, i, cand_len)
                {
                    slot.score = slot.score.saturating_add(gain);
                    placed = true;
                    break;
                }
                if slot.score < replacement_score {
                    replacement_score = slot.score;
                    replacement_idx = slot_idx;
                }
                slot_idx = (slot_idx + 1) & slot_mask;
            }
            if !placed && gain > replacement_score {
                let slot = &mut seed_slots[replacement_idx];
                slot.fingerprint = fp;
                slot.pos = i as u32;
                slot.len = cand_len as u16;
                slot.score = gain;
                slot.used = 1;
            }
            if mlen >= max_entry_len {
                break;
            }
            chain = prev[prev_idx];
            traversed += 1;
        }

        // zlib-style idea: when we already observed a long local match, skip
        // some near positions to avoid redundant seed probes in repetitive runs.
        let mut step = sample_stride.max(1);
        if best_mlen_here >= min_seed_match_len.saturating_mul(2) {
            step = step.max(best_mlen_here);
        }
        i = i.saturating_add(step);
    }
    prof.probe_ms = elapsed_ms(probe_t0);

    // Materialize only final candidates.
    let materialize_t0 = Instant::now();
    scored_slot_indices.reserve(used_seed_slot_indices.len());
    for &slot_idx in &used_seed_slot_indices {
        let seed = seed_slots[slot_idx];
        if seed.used == 0 || seed.score == 0 {
            continue;
        }
        scored_slot_indices.push((slot_idx, seed.score));
    }
    prof.materialize_ms = elapsed_ms(materialize_t0);

    let sort_t0 = Instant::now();
    byte_scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored_slot_indices.sort_by(|a, b| {
        b.1.cmp(&a.1).then_with(|| {
            let alen = seed_slots[a.0].len;
            let blen = seed_slots[b.0].len;
            blen.cmp(&alen)
        })
    });
    let mut out = Vec::with_capacity(max_entries.min(scored_slot_indices.len() + 64));
    for &(b, _) in byte_scored.iter().take(64) {
        if out.len() >= max_entries {
            break;
        }
        out.push(vec![b]);
    }
    for &(slot_idx, _) in &scored_slot_indices {
        if out.len() == max_entries {
            break;
        }
        let slot = seed_slots[slot_idx];
        let pos = slot.pos as usize;
        let len = slot.len as usize;
        if len == 0 || len > max_entry_len || pos + len > chunk.len() {
            continue;
        }
        let entry = &chunk[pos..pos + len];
        if out.iter().any(|e| e.as_slice() == entry) {
            continue;
        }
        out.push(entry.to_vec());
    }
    prof.sort_ms = elapsed_ms(sort_t0);
    BUILD_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        std::mem::swap(&mut seed_slots, &mut scratch.seed_slots);
        std::mem::swap(
            &mut used_seed_slot_indices,
            &mut scratch.used_seed_slot_indices,
        );
        std::mem::swap(&mut byte_scored, &mut scratch.byte_scored);
        std::mem::swap(&mut scored_slot_indices, &mut scratch.scored_slot_indices);
        std::mem::swap(&mut head, &mut scratch.head);
        std::mem::swap(&mut touched_buckets, &mut scratch.touched_buckets);
        std::mem::swap(&mut prev, &mut scratch.prev);
        std::mem::swap(&mut keys, &mut scratch.keys);
        std::mem::swap(&mut positions, &mut scratch.positions);
    });
    (out, prof)
}

fn choose_entry_len(match_len: usize, max_entry_len: usize) -> usize {
    const BUCKETS: [usize; 10] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];
    let capped = match_len.min(max_entry_len);
    let mut best = 1usize;
    for &v in &BUCKETS {
        if v <= capped {
            best = v;
        }
    }
    best
}

fn encode_section_into(
    dst: &mut Vec<u8>,
    src: &[u8],
    table: &[Vec<u8>],
    table_repeat: &[Vec<u8>],
    by_prefix2: &[Vec<usize>],
    has_prefix2: &[u8],
    prefix3_masks: &[[u64; 4]],
    options: &PDeflateOptions,
) -> Result<(usize, SectionEncodeProfile), PDeflateError> {
    let cmd_start = dst.len();
    let mut prof = SectionEncodeProfile::default();
    let prof_enabled = profile_enabled();
    let detail_profile = prof_enabled;
    if src.is_empty() {
        return Ok((0, prof));
    }

    let max_ref_len = options.max_ref_len.min(MAX_CMD_LEN);
    let min_ref_len = options.min_ref_len;

    let mut pos = 0usize;
    while pos < src.len() {
        let mut find_metrics = FindMetrics::default();
        let search_t0 = if prof_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let best = find_best_match(
            src,
            pos,
            table,
            table_repeat,
            by_prefix2,
            has_prefix2,
            prefix3_masks,
            options,
            max_ref_len,
            if detail_profile {
                Some(&mut find_metrics)
            } else {
                None
            },
        );
        if let Some(t0) = search_t0 {
            let dt = elapsed_ms(t0);
            prof.match_search_ms += dt;
            prof.best_search_ms += dt;
        }
        if detail_profile {
            prof.find_calls = prof.find_calls.saturating_add(1);
            prof.best_calls = prof.best_calls.saturating_add(1);
            prof.candidate_checks = prof
                .candidate_checks
                .saturating_add(find_metrics.candidate_checks);
            prof.best_candidate_checks = prof
                .best_candidate_checks
                .saturating_add(find_metrics.candidate_checks);
            prof.compare_steps = prof
                .compare_steps
                .saturating_add(find_metrics.compare_steps);
            prof.best_compare_steps = prof
                .best_compare_steps
                .saturating_add(find_metrics.compare_steps);
            prof.prefilter_rejects = prof
                .prefilter_rejects
                .saturating_add(find_metrics.prefilter_rejects);
            prof.no_prefix_fast_skips = prof
                .no_prefix_fast_skips
                .saturating_add(find_metrics.no_prefix_fast_skips);
            prof.best_tail_rejects = prof
                .best_tail_rejects
                .saturating_add(find_metrics.tail_rejects);
        }
        if let Some((id, mlen)) = best {
            let emit_t0 = if prof_enabled {
                Some(Instant::now())
            } else {
                None
            };
            emit_ref_command(dst, id as u16, mlen)?;
            if let Some(t0) = emit_t0 {
                prof.emit_ref_ms += elapsed_ms(t0);
            }
            if detail_profile {
                prof.ref_cmds = prof.ref_cmds.saturating_add(1);
                prof.ref_bytes = prof
                    .ref_bytes
                    .saturating_add(u64::try_from(mlen).unwrap_or(0));
            }
            pos += mlen;
            continue;
        }

        let lit_start = pos;
        pos += 1;

        while pos < src.len() && (pos - lit_start) < MAX_CMD_LEN {
            if pos + 1 < src.len() {
                let k2 = ((src[pos] as usize) << 8) | (src[pos + 1] as usize);
                if has_prefix2[k2] == 0 {
                    if detail_profile {
                        prof.no_prefix_fast_skips = prof.no_prefix_fast_skips.saturating_add(1);
                    }
                    pos += 1;
                    continue;
                }
                if pos + 2 < src.len() && !prefix3_hit(prefix3_masks, k2, src[pos + 2]) {
                    if detail_profile {
                        prof.no_prefix_fast_skips = prof.no_prefix_fast_skips.saturating_add(1);
                    }
                    pos += 1;
                    continue;
                }
                // Stronger lazy-match throttling for throughput scaling:
                // progressively reduce lookahead probe density as literal
                // run grows to limit compare-heavy checks under high thread counts.
                let lit_span = pos - lit_start;
                let should_probe = if lit_span < 8 {
                    true
                } else if lit_span < 32 {
                    (lit_span & 0x7) == 0
                } else if lit_span < 128 {
                    (lit_span & 0x1F) == 0
                } else {
                    (lit_span & 0x3F) == 0
                };
                if !should_probe {
                    pos += 1;
                    continue;
                }
            } else {
                pos += 1;
                continue;
            }

            let mut lookahead_metrics = FindMetrics::default();
            let search_t0 = if prof_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let has_match = has_any_match(
                src,
                pos,
                table,
                table_repeat,
                by_prefix2,
                has_prefix2,
                prefix3_masks,
                options,
                max_ref_len,
                if detail_profile {
                    Some(&mut lookahead_metrics)
                } else {
                    None
                },
            );
            if let Some(t0) = search_t0 {
                let dt = elapsed_ms(t0);
                prof.match_search_ms += dt;
                prof.lookahead_search_ms += dt;
            }
            if detail_profile {
                prof.find_calls = prof.find_calls.saturating_add(1);
                prof.lookahead_calls = prof.lookahead_calls.saturating_add(1);
                prof.candidate_checks = prof
                    .candidate_checks
                    .saturating_add(lookahead_metrics.candidate_checks);
                prof.lookahead_candidate_checks = prof
                    .lookahead_candidate_checks
                    .saturating_add(lookahead_metrics.candidate_checks);
                prof.compare_steps = prof
                    .compare_steps
                    .saturating_add(lookahead_metrics.compare_steps);
                prof.lookahead_compare_steps = prof
                    .lookahead_compare_steps
                    .saturating_add(lookahead_metrics.compare_steps);
                prof.prefilter_rejects = prof
                    .prefilter_rejects
                    .saturating_add(lookahead_metrics.prefilter_rejects);
                prof.no_prefix_fast_skips = prof
                    .no_prefix_fast_skips
                    .saturating_add(lookahead_metrics.no_prefix_fast_skips);
            }
            if has_match {
                break;
            }
            pos += 1;
        }

        let lit_run = &src[lit_start..pos];
        if lit_run.len() >= min_ref_len {
            // Keep literal runs even when long; decoder has clear op separation.
        }
        let emit_t0 = if prof_enabled {
            Some(Instant::now())
        } else {
            None
        };
        emit_literal_command(dst, lit_run)?;
        if let Some(t0) = emit_t0 {
            prof.emit_lit_ms += elapsed_ms(t0);
        }
        if detail_profile {
            prof.lit_cmds = prof.lit_cmds.saturating_add(1);
            prof.lit_bytes = prof
                .lit_bytes
                .saturating_add(u64::try_from(lit_run.len()).unwrap_or(0));
        }
    }

    Ok((dst.len().saturating_sub(cmd_start), prof))
}

fn has_any_match(
    src: &[u8],
    pos: usize,
    table: &[Vec<u8>],
    table_repeat: &[Vec<u8>],
    by_prefix2: &[Vec<usize>],
    has_prefix2: &[u8],
    prefix3_masks: &[[u64; 4]],
    options: &PDeflateOptions,
    max_ref_len: usize,
    metrics: Option<&mut FindMetrics>,
) -> bool {
    if pos >= src.len() || src.len() - pos < options.min_ref_len {
        return false;
    }
    let max_len = (src.len() - pos).min(max_ref_len);
    if max_len < options.min_ref_len {
        return false;
    }

    let min_check = options.min_ref_len;
    if pos + 1 >= src.len() {
        return false;
    }
    let k2 = ((src[pos] as usize) << 8) | (src[pos + 1] as usize);
    let mut metrics = metrics;
    if has_prefix2[k2] == 0 {
        if let Some(m) = metrics.as_mut() {
            m.no_prefix_fast_skips = m.no_prefix_fast_skips.saturating_add(1);
        }
        return false;
    }
    if pos + 2 < src.len() && !prefix3_hit(prefix3_masks, k2, src[pos + 2]) {
        if let Some(m) = metrics.as_mut() {
            m.no_prefix_fast_skips = m.no_prefix_fast_skips.saturating_add(1);
        }
        return false;
    }

    let candidates = select_candidates(src, pos, by_prefix2);
    let probe_limit = options.match_probe_limit.max(1);

    for &id in candidates.iter().take(probe_limit) {
        if let Some(m) = metrics.as_mut() {
            m.candidate_checks = m.candidate_checks.saturating_add(1);
        }
        let entry = &table[id];
        let rep = &table_repeat[id];
        if entry.is_empty() {
            continue;
        }
        if entry.len() > 1 {
            if pos + 1 >= src.len() || rep[1] != src[pos + 1] {
                if let Some(m) = metrics.as_mut() {
                    m.prefilter_rejects = m.prefilter_rejects.saturating_add(1);
                }
                continue;
            }
        }
        if entry.len() > 2 {
            if pos + 2 >= src.len() || rep[2] != src[pos + 2] {
                if let Some(m) = metrics.as_mut() {
                    m.prefilter_rejects = m.prefilter_rejects.saturating_add(1);
                }
                continue;
            }
        }
        if entry.len() > 3 {
            if pos + 3 >= src.len() || rep[3] != src[pos + 3] {
                if let Some(m) = metrics.as_mut() {
                    m.prefilter_rejects = m.prefilter_rejects.saturating_add(1);
                }
                continue;
            }
        }

        let mut ok = true;
        for k in 0..min_check.min(max_ref_len) {
            if let Some(m) = metrics.as_mut() {
                m.compare_steps = m.compare_steps.saturating_add(1);
            }
            if src[pos + k] != rep[k] {
                ok = false;
                break;
            }
        }
        if ok {
            return true;
        }
    }
    false
}

fn find_best_match(
    src: &[u8],
    pos: usize,
    table: &[Vec<u8>],
    table_repeat: &[Vec<u8>],
    by_prefix2: &[Vec<usize>],
    has_prefix2: &[u8],
    prefix3_masks: &[[u64; 4]],
    options: &PDeflateOptions,
    max_ref_len: usize,
    metrics: Option<&mut FindMetrics>,
) -> Option<(usize, usize)> {
    if pos >= src.len() || src.len() - pos < options.min_ref_len {
        return None;
    }

    if pos + 1 >= src.len() {
        return None;
    }
    let k2 = ((src[pos] as usize) << 8) | (src[pos + 1] as usize);
    let mut metrics = metrics;
    if has_prefix2[k2] == 0 {
        if let Some(m) = metrics.as_mut() {
            m.no_prefix_fast_skips = m.no_prefix_fast_skips.saturating_add(1);
        }
        return None;
    }
    if pos + 2 < src.len() && !prefix3_hit(prefix3_masks, k2, src[pos + 2]) {
        if let Some(m) = metrics.as_mut() {
            m.no_prefix_fast_skips = m.no_prefix_fast_skips.saturating_add(1);
        }
        return None;
    }

    let mut best_id = usize::MAX;
    let mut best_len = 0usize;

    let candidates = select_candidates(src, pos, by_prefix2);
    let probe_limit = options.match_probe_limit.max(1);
    let mut checked = 0usize;
    for &id in candidates.iter() {
        if checked >= probe_limit {
            break;
        }
        checked += 1;
        if let Some(m) = metrics.as_mut() {
            m.candidate_checks = m.candidate_checks.saturating_add(1);
        }
        let entry = &table[id];
        let rep = &table_repeat[id];
        if entry.is_empty() {
            continue;
        }

        let max_len = (src.len() - pos).min(max_ref_len);
        if max_len < options.min_ref_len {
            break;
        }

        if entry.len() > 1 {
            if pos + 1 >= src.len() {
                continue;
            }
            if rep[1] != src[pos + 1] {
                if let Some(m) = metrics.as_mut() {
                    m.prefilter_rejects = m.prefilter_rejects.saturating_add(1);
                }
                continue;
            }
        }
        if entry.len() > 2 {
            if pos + 2 >= src.len() {
                continue;
            }
            if rep[2] != src[pos + 2] {
                if let Some(m) = metrics.as_mut() {
                    m.prefilter_rejects = m.prefilter_rejects.saturating_add(1);
                }
                continue;
            }
        }
        if entry.len() > 3 {
            if pos + 3 >= src.len() {
                continue;
            }
            if rep[3] != src[pos + 3] {
                if let Some(m) = metrics.as_mut() {
                    m.prefilter_rejects = m.prefilter_rejects.saturating_add(1);
                }
                continue;
            }
        }

        if best_len + 1 < max_len && rep[best_len] != src[pos + best_len] {
            if let Some(m) = metrics.as_mut() {
                m.prefilter_rejects = m.prefilter_rejects.saturating_add(1);
            }
            continue;
        }

        // zlib longest_match style early-out: when a decent best match exists,
        // reject candidates that don't match near the current end byte.
        if best_len >= options.min_ref_len {
            let end = best_len - 1;
            if rep[end] != src[pos + end] {
                if let Some(m) = metrics.as_mut() {
                    m.prefilter_rejects = m.prefilter_rejects.saturating_add(1);
                    m.tail_rejects = m.tail_rejects.saturating_add(1);
                }
                continue;
            }
            if end >= 1 && rep[end - 1] != src[pos + end - 1] {
                if let Some(m) = metrics.as_mut() {
                    m.prefilter_rejects = m.prefilter_rejects.saturating_add(1);
                    m.tail_rejects = m.tail_rejects.saturating_add(1);
                }
                continue;
            }
        }

        let m = match_prefix_len(src, pos, rep, max_len, metrics.as_deref_mut());

        if m >= options.min_ref_len
            && (m > best_len
                || (m == best_len && best_id != usize::MAX && entry.len() < table[best_id].len()))
        {
            best_id = id;
            best_len = m;
            if best_len == max_len {
                break;
            }
        }
        // zlib-like chain cut: once we already have a solid match, reduce
        // further candidate probes.
        if best_len >= 64 {
            break;
        }
        if best_len >= 32 && checked >= 2 {
            break;
        }
        if best_len >= 16 && checked >= 3 {
            break;
        }
    }

    if best_id == usize::MAX {
        None
    } else {
        Some((best_id, best_len))
    }
}

fn select_candidates<'a>(src: &'a [u8], pos: usize, by_prefix2: &'a [Vec<usize>]) -> &'a [usize] {
    // zlib-style idea: strengthen the key used for candidate lookup so that
    // expensive match checks run on a shorter chain.
    if pos + 1 >= src.len() {
        return &EMPTY_CANDIDATES;
    }
    let k2 = ((src[pos] as usize) << 8) | (src[pos + 1] as usize);
    &by_prefix2[k2]
}

#[inline]
fn prefix3_hit(prefix3_masks: &[[u64; 4]], k2: usize, b2: u8) -> bool {
    let bit = b2 as usize;
    (prefix3_masks[k2][bit >> 6] & (1u64 << (bit & 63))) != 0
}

fn match_prefix_len(
    src: &[u8],
    pos: usize,
    rep: &[u8],
    max_len: usize,
    mut metrics: Option<&mut FindMetrics>,
) -> usize {
    let m = fast_prefix_len(&src[pos..], rep, max_len);
    if let Some(mm) = metrics.as_mut() {
        mm.compare_steps = mm.compare_steps.saturating_add(m as u64);
    }
    m
}

fn emit_ref_command(dst: &mut Vec<u8>, id: u16, mut len: usize) -> Result<(), PDeflateError> {
    if id >= LITERAL_TAG {
        return Err(PDeflateError::InvalidOptions("table id overflow"));
    }

    while len > 0 {
        let part = len.min(MAX_CMD_LEN);
        if part < 3 {
            return Err(PDeflateError::InvalidOptions("reference len < 3"));
        }
        write_cmd_header(dst, id, part)?;
        len -= part;
    }
    Ok(())
}

fn emit_literal_command(dst: &mut Vec<u8>, bytes: &[u8]) -> Result<(), PDeflateError> {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let part = (bytes.len() - pos).min(MAX_CMD_LEN);
        write_cmd_header(dst, LITERAL_TAG, part)?;
        dst.extend_from_slice(&bytes[pos..pos + part]);
        pos += part;
    }
    Ok(())
}

fn write_cmd_header(dst: &mut Vec<u8>, tag: u16, len: usize) -> Result<(), PDeflateError> {
    if tag > LITERAL_TAG {
        return Err(PDeflateError::InvalidOptions("invalid command tag"));
    }
    if len == 0 || len > MAX_CMD_LEN {
        return Err(PDeflateError::InvalidOptions("invalid command length"));
    }

    if len <= MAX_INLINE_LEN {
        let len4 = u16::try_from(len).map_err(|_| PDeflateError::NumericOverflow)?;
        let cmd = (len4 << 12) | (tag & 0x0fff);
        write_u16_le(dst, cmd);
        return Ok(());
    }

    let ext = len - EXT_LEN_BASE;
    let ext_u8 = u8::try_from(ext).map_err(|_| PDeflateError::NumericOverflow)?;
    let cmd = (0x000f_u16 << 12) | (tag & 0x0fff);
    write_u16_le(dst, cmd);
    dst.push(ext_u8);
    Ok(())
}

fn match_len(buf: &[u8], a: usize, b: usize, max_len: usize) -> usize {
    let limit = max_len
        .min(buf.len().saturating_sub(a))
        .min(buf.len().saturating_sub(b));
    fast_prefix_len(&buf[a..], &buf[b..], limit)
}

fn match_len_from(buf: &[u8], a: usize, b: usize, start: usize, max_len: usize) -> usize {
    let limit = max_len
        .min(buf.len().saturating_sub(a))
        .min(buf.len().saturating_sub(b));
    start
        + fast_prefix_len(
            &buf[a + start..],
            &buf[b + start..],
            limit.saturating_sub(start),
        )
}

#[inline]
fn first_diff_byte_in_u64(diff: u64) -> usize {
    #[cfg(target_endian = "little")]
    {
        (diff.trailing_zeros() >> 3) as usize
    }
    #[cfg(target_endian = "big")]
    {
        (diff.leading_zeros() >> 3) as usize
    }
}

#[inline]
fn fast_prefix_len(a: &[u8], b: &[u8], max_len: usize) -> usize {
    let mut i = 0usize;
    let limit = max_len.min(a.len()).min(b.len());

    while i + 8 <= limit {
        // Safe due to bounds check above; unaligned loads are used for speed.
        let aw = unsafe { ptr::read_unaligned(a.as_ptr().add(i) as *const u64) };
        let bw = unsafe { ptr::read_unaligned(b.as_ptr().add(i) as *const u64) };
        let diff = aw ^ bw;
        if diff != 0 {
            return i + first_diff_byte_in_u64(diff);
        }
        i += 8;
    }

    while i < limit && a[i] == b[i] {
        i += 1;
    }
    i
}

fn key3(buf: &[u8], i: usize) -> u32 {
    (buf[i] as u32) | ((buf[i + 1] as u32) << 8) | ((buf[i + 2] as u32) << 16)
}

#[inline]
fn hash3_to_bucket(key: u32, mask: usize) -> usize {
    ((key.wrapping_mul(0x1e35_a7bd)) as usize) & mask
}

fn windows_equal(buf: &[u8], a: usize, b: usize, len: usize) -> bool {
    buf[a..a + len] == buf[b..b + len]
}

fn fingerprint_window(buf: &[u8], pos: usize, len: usize) -> u64 {
    let mut h = (len as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15_u64);
    let end = pos + len;
    let first = buf[pos];
    let last = buf[end - 1];
    h ^= u64::from(first) << 1;
    h ^= u64::from(last) << 33;

    // Sample inner bytes for better spread without full per-byte hashing.
    if len >= 4 {
        h ^= u64::from(buf[pos + (len / 3)]) << 17;
        h ^= u64::from(buf[pos + ((2 * len) / 3)]) << 49;
    }

    // Include first up-to-8 bytes as a cheap anchored signature.
    let mut sig = 0u64;
    let n = len.min(8);
    for i in 0..n {
        sig |= u64::from(buf[pos + i]) << (i * 8);
    }
    h ^ sig.rotate_left(7)
}

fn section_start(index: usize, section_count: usize, total_len: usize) -> usize {
    if section_count == 0 || total_len == 0 {
        return 0;
    }
    if index == 0 {
        return 0;
    }
    if index >= section_count {
        return total_len;
    }
    let num = (index as u128) * (total_len as u128);
    let den = section_count as u128;
    let raw = (num / den) as usize;
    raw & !3usize
}

fn read_exact<'a>(
    src: &'a [u8],
    cursor: &mut usize,
    len: usize,
) -> Result<&'a [u8], PDeflateError> {
    let end = cursor
        .checked_add(len)
        .ok_or(PDeflateError::NumericOverflow)?;
    let slice = src
        .get(*cursor..end)
        .ok_or(PDeflateError::InvalidStream("unexpected eof"))?;
    *cursor = end;
    Ok(slice)
}

fn write_u8(dst: &mut Vec<u8>, v: u8) {
    dst.push(v);
}

fn write_u16_le(dst: &mut Vec<u8>, v: u16) {
    dst.extend_from_slice(&v.to_le_bytes());
}

fn write_u32_le(dst: &mut Vec<u8>, v: u32) {
    dst.extend_from_slice(&v.to_le_bytes());
}

fn write_u64_le(dst: &mut Vec<u8>, v: u64) {
    dst.extend_from_slice(&v.to_le_bytes());
}

fn read_u16_le(src: &[u8], cursor: &mut usize) -> Result<u16, PDeflateError> {
    let bytes = read_exact(src, cursor, 2)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u8(src: &[u8], cursor: &mut usize) -> Result<u8, PDeflateError> {
    Ok(*read_exact(src, cursor, 1)?
        .first()
        .ok_or(PDeflateError::InvalidStream("unexpected eof"))?)
}

fn read_u32_le(src: &[u8], cursor: &mut usize) -> Result<u32, PDeflateError> {
    let bytes = read_exact(src, cursor, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64_le(src: &[u8], cursor: &mut usize) -> Result<u64, PDeflateError> {
    let bytes = read_exact(src, cursor, 8)?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn write_varint_u32(dst: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        dst.push(((v & 0x7f) as u8) | 0x80);
        v >>= 7;
    }
    dst.push((v & 0x7f) as u8);
}

fn read_varint_u32(src: &[u8], cursor: &mut usize) -> Result<u32, PDeflateError> {
    let mut shift = 0u32;
    let mut value = 0u32;
    loop {
        if shift >= 35 {
            return Err(PDeflateError::InvalidStream("varint overflow"));
        }
        let b = *read_exact(src, cursor, 1)?
            .first()
            .ok_or(PDeflateError::InvalidStream("eof in varint"))?;
        value |= u32::from(b & 0x7f) << shift;
        if (b & 0x80) == 0 {
            return Ok(value);
        }
        shift += 7;
    }
}

const HUFF_MAX_CODE_BITS: u8 = 15;
const HUFF_LUT_ROOT_BITS_DEFAULT: u8 = 9;
const HUFF_LUT_ENTRY_KIND_INVALID: u32 = 0;
const HUFF_LUT_ENTRY_KIND_SYMBOL: u32 = 1;
const HUFF_LUT_ENTRY_KIND_SUBTABLE: u32 = 2;
const HUFF_LUT_HEADER_SIZE: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HuffmanSymbolCode {
    code_lsb: u32,
    bit_len: u8,
}

#[derive(Debug, Clone)]
struct HuffmanCanonicalCodebook {
    symbol_count: usize,
    max_code_bits: u8,
    codes: Vec<Option<HuffmanSymbolCode>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HuffmanLutEntry {
    Invalid,
    Symbol { symbol: u16, bit_len: u8 },
    Subtable { offset: u32, sub_bits: u8 },
}

#[derive(Debug, Clone)]
struct HuffmanLut {
    symbol_count: usize,
    root_bits: u8,
    max_code_bits: u8,
    root: Vec<HuffmanLutEntry>,
    subtables: Vec<HuffmanLutEntry>,
}

#[derive(Debug, Clone)]
struct HuffmanDecodeTreeNode {
    left: Option<usize>,
    right: Option<usize>,
    symbol: Option<u16>,
}

impl Default for HuffmanDecodeTreeNode {
    fn default() -> Self {
        Self {
            left: None,
            right: None,
            symbol: None,
        }
    }
}

#[derive(Debug, Clone)]
struct BitReaderLsb<'a> {
    src: &'a [u8],
    bit_len: usize,
    bit_cursor: usize,
}

impl<'a> BitReaderLsb<'a> {
    fn new(src: &'a [u8], bit_len: usize) -> Result<Self, PDeflateError> {
        let max_bits = src
            .len()
            .checked_mul(8)
            .ok_or(PDeflateError::NumericOverflow)?;
        if bit_len > max_bits {
            return Err(PDeflateError::InvalidStream(
                "huffman bit_len exceeds input buffer",
            ));
        }
        Ok(Self {
            src,
            bit_len,
            bit_cursor: 0,
        })
    }

    fn remaining_bits(&self) -> usize {
        self.bit_len.saturating_sub(self.bit_cursor)
    }

    fn read_bit(&mut self) -> Result<u8, PDeflateError> {
        if self.bit_cursor >= self.bit_len {
            return Err(PDeflateError::InvalidStream(
                "unexpected eof in huffman bitstream",
            ));
        }
        let pos = self.bit_cursor;
        self.bit_cursor = self.bit_cursor.saturating_add(1);
        let byte = self.src[pos / 8];
        Ok((byte >> (pos & 7)) & 1)
    }

    fn consume(&mut self, bit_len: u8) -> Result<(), PDeflateError> {
        let n = bit_len as usize;
        if n > self.remaining_bits() {
            return Err(PDeflateError::InvalidStream(
                "huffman code exceeds remaining bits",
            ));
        }
        self.bit_cursor = self.bit_cursor.saturating_add(n);
        Ok(())
    }

    fn peek_bits_padded(&self, bit_len: u8) -> u32 {
        let mut out = 0u32;
        for i in 0..bit_len {
            let pos = self.bit_cursor + (i as usize);
            if pos >= self.bit_len {
                break;
            }
            let byte = self.src[pos / 8];
            let bit = (byte >> (pos & 7)) & 1;
            out |= u32::from(bit) << i;
        }
        out
    }
}

#[inline]
fn reverse_low_bits(mut value: u32, bit_len: u8) -> u32 {
    let mut out = 0u32;
    for _ in 0..bit_len {
        out = (out << 1) | (value & 1);
        value >>= 1;
    }
    out
}

fn build_huffman_code_lengths_from_frequencies(
    frequencies: &[u32],
    max_code_bits: u8,
) -> Result<Vec<u8>, PDeflateError> {
    if frequencies.is_empty() {
        return Err(PDeflateError::InvalidOptions(
            "huffman symbol alphabet must not be empty",
        ));
    }
    if max_code_bits == 0 || max_code_bits > 31 {
        return Err(PDeflateError::InvalidOptions(
            "huffman max_code_bits must be in 1..=31",
        ));
    }

    #[derive(Debug, Clone, Copy)]
    struct HuffNode {
        parent: Option<usize>,
        symbol: Option<usize>,
    }

    let mut nodes = Vec::<HuffNode>::new();
    let mut leaf_node_by_symbol = vec![None; frequencies.len()];
    let mut heap = BinaryHeap::<Reverse<(u64, usize, usize)>>::new();
    let mut seq = 0usize;

    for (symbol, &freq_u32) in frequencies.iter().enumerate() {
        if freq_u32 == 0 {
            continue;
        }
        let node_index = nodes.len();
        nodes.push(HuffNode {
            parent: None,
            symbol: Some(symbol),
        });
        leaf_node_by_symbol[symbol] = Some(node_index);
        heap.push(Reverse((u64::from(freq_u32), seq, node_index)));
        seq = seq.saturating_add(1);
    }

    if heap.is_empty() {
        let mut lengths = vec![0u8; frequencies.len()];
        lengths[0] = 1;
        return Ok(lengths);
    }

    if heap.len() == 1 {
        let mut lengths = vec![0u8; frequencies.len()];
        let only = heap.pop().ok_or(PDeflateError::NumericOverflow)?.0.2;
        let symbol = nodes[only].symbol.ok_or(PDeflateError::InvalidStream(
            "huffman leaf node missing symbol",
        ))?;
        lengths[symbol] = 1;
        return Ok(lengths);
    }

    while heap.len() > 1 {
        let left = heap.pop().ok_or(PDeflateError::NumericOverflow)?.0;
        let right = heap.pop().ok_or(PDeflateError::NumericOverflow)?.0;
        let parent_freq = left
            .0
            .checked_add(right.0)
            .ok_or(PDeflateError::NumericOverflow)?;
        let parent_index = nodes.len();
        nodes.push(HuffNode {
            parent: None,
            symbol: None,
        });
        nodes[left.2].parent = Some(parent_index);
        nodes[right.2].parent = Some(parent_index);
        heap.push(Reverse((parent_freq, seq, parent_index)));
        seq = seq.saturating_add(1);
    }

    let mut lengths = vec![0u8; frequencies.len()];
    for (symbol, &leaf_node) in leaf_node_by_symbol.iter().enumerate() {
        let Some(mut node_index) = leaf_node else {
            continue;
        };
        let mut depth = 0usize;
        while let Some(parent) = nodes[node_index].parent {
            depth = depth.saturating_add(1);
            node_index = parent;
        }
        if depth == 0 {
            depth = 1;
        }
        if depth > usize::from(max_code_bits) {
            return Err(PDeflateError::InvalidOptions(
                "huffman code length exceeds max_code_bits",
            ));
        }
        lengths[symbol] = u8::try_from(depth).map_err(|_| PDeflateError::NumericOverflow)?;
    }
    Ok(lengths)
}

fn build_canonical_huffman_codebook_from_lengths(
    code_lengths: &[u8],
    max_code_bits: u8,
) -> Result<HuffmanCanonicalCodebook, PDeflateError> {
    if code_lengths.is_empty() {
        return Err(PDeflateError::InvalidOptions(
            "huffman symbol alphabet must not be empty",
        ));
    }
    if max_code_bits == 0 || max_code_bits > 31 {
        return Err(PDeflateError::InvalidOptions(
            "huffman max_code_bits must be in 1..=31",
        ));
    }

    let mut max_len_seen = 0u8;
    let mut any_symbol = false;
    let mut bl_count = vec![0u32; usize::from(max_code_bits) + 1];
    for &len in code_lengths {
        if len == 0 {
            continue;
        }
        if len > max_code_bits {
            return Err(PDeflateError::InvalidOptions(
                "huffman code length exceeds max_code_bits",
            ));
        }
        any_symbol = true;
        max_len_seen = max_len_seen.max(len);
        bl_count[len as usize] = bl_count[len as usize].saturating_add(1);
    }
    if !any_symbol {
        return Err(PDeflateError::InvalidOptions(
            "huffman code lengths contain no symbols",
        ));
    }

    let mut next_code = vec![0u32; usize::from(max_code_bits) + 1];
    let mut code = 0u32;
    for bits in 1..=usize::from(max_code_bits) {
        code = (code
            .checked_add(bl_count[bits - 1])
            .ok_or(PDeflateError::NumericOverflow)?)
            << 1;
        next_code[bits] = code;
    }

    let mut codes = vec![None; code_lengths.len()];
    for (symbol, &len) in code_lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let len_idx = len as usize;
        let code_msb = next_code[len_idx];
        let limit = 1u32
            .checked_shl(u32::from(len))
            .ok_or(PDeflateError::NumericOverflow)?;
        if code_msb >= limit {
            return Err(PDeflateError::InvalidOptions(
                "invalid canonical huffman code lengths (oversubscribed)",
            ));
        }
        next_code[len_idx] = next_code[len_idx].saturating_add(1);
        let code_lsb = reverse_low_bits(code_msb, len);
        codes[symbol] = Some(HuffmanSymbolCode {
            code_lsb,
            bit_len: len,
        });
    }

    Ok(HuffmanCanonicalCodebook {
        symbol_count: code_lengths.len(),
        max_code_bits: max_len_seen,
        codes,
    })
}

fn build_canonical_huffman_codebook_from_frequencies(
    frequencies: &[u32],
    max_code_bits: u8,
) -> Result<HuffmanCanonicalCodebook, PDeflateError> {
    let lengths = build_huffman_code_lengths_from_frequencies(frequencies, max_code_bits)?;
    build_canonical_huffman_codebook_from_lengths(&lengths, max_code_bits)
}

fn build_huffman_lut(
    codebook: &HuffmanCanonicalCodebook,
    root_bits: u8,
) -> Result<HuffmanLut, PDeflateError> {
    if root_bits == 0 {
        return Err(PDeflateError::InvalidOptions(
            "huffman root_bits must be > 0",
        ));
    }
    if root_bits > codebook.max_code_bits {
        return Err(PDeflateError::InvalidOptions(
            "huffman root_bits exceeds max code bits",
        ));
    }
    if root_bits > 24 {
        return Err(PDeflateError::InvalidOptions(
            "huffman root_bits must be <= 24",
        ));
    }

    let root_size = 1usize
        .checked_shl(u32::from(root_bits))
        .ok_or(PDeflateError::NumericOverflow)?;
    let mut root = vec![HuffmanLutEntry::Invalid; root_size];
    let mut max_sub_bits_by_prefix = vec![0u8; root_size];
    let root_mask = (1u32 << u32::from(root_bits)) - 1;

    for (symbol, code_opt) in codebook.codes.iter().enumerate() {
        let Some(code) = *code_opt else {
            continue;
        };
        if code.bit_len <= root_bits {
            let repeat = 1usize
                .checked_shl(u32::from(root_bits - code.bit_len))
                .ok_or(PDeflateError::NumericOverflow)?;
            let sym_u16 = u16::try_from(symbol).map_err(|_| PDeflateError::NumericOverflow)?;
            for suffix in 0..repeat {
                let idx = ((suffix as u32) << u32::from(code.bit_len)) | code.code_lsb;
                let idx = usize::try_from(idx).map_err(|_| PDeflateError::NumericOverflow)?;
                let entry = HuffmanLutEntry::Symbol {
                    symbol: sym_u16,
                    bit_len: code.bit_len,
                };
                match root[idx] {
                    HuffmanLutEntry::Invalid => root[idx] = entry,
                    v if v == entry => {}
                    _ => {
                        return Err(PDeflateError::InvalidOptions(
                            "huffman root table collision",
                        ));
                    }
                }
            }
            continue;
        }
        let prefix = (code.code_lsb & root_mask) as usize;
        let sub_bits = code.bit_len - root_bits;
        max_sub_bits_by_prefix[prefix] = max_sub_bits_by_prefix[prefix].max(sub_bits);
    }

    let mut subtables = Vec::<HuffmanLutEntry>::new();
    let mut sub_offset_by_prefix = vec![None; root_size];
    for prefix in 0..root_size {
        let sub_bits = max_sub_bits_by_prefix[prefix];
        if sub_bits == 0 {
            continue;
        }
        let sub_size = 1usize
            .checked_shl(u32::from(sub_bits))
            .ok_or(PDeflateError::NumericOverflow)?;
        let offset = subtables.len();
        let offset_u32 = u32::try_from(offset).map_err(|_| PDeflateError::NumericOverflow)?;
        subtables.resize(offset + sub_size, HuffmanLutEntry::Invalid);
        sub_offset_by_prefix[prefix] = Some(offset_u32);
        root[prefix] = HuffmanLutEntry::Subtable {
            offset: offset_u32,
            sub_bits,
        };
    }

    for (symbol, code_opt) in codebook.codes.iter().enumerate() {
        let Some(code) = *code_opt else {
            continue;
        };
        if code.bit_len <= root_bits {
            continue;
        }
        let prefix = (code.code_lsb & root_mask) as usize;
        let sub_offset = sub_offset_by_prefix[prefix].ok_or(PDeflateError::InvalidOptions(
            "huffman subtable offset missing",
        ))? as usize;
        let sub_bits = max_sub_bits_by_prefix[prefix];
        let rem_len = code.bit_len - root_bits;
        let rem_code = code.code_lsb >> u32::from(root_bits);
        let repeat = 1usize
            .checked_shl(u32::from(sub_bits - rem_len))
            .ok_or(PDeflateError::NumericOverflow)?;
        let sym_u16 = u16::try_from(symbol).map_err(|_| PDeflateError::NumericOverflow)?;
        for suffix in 0..repeat {
            let sub_idx = ((suffix as u32) << u32::from(rem_len)) | rem_code;
            let lut_idx = sub_offset
                .checked_add(usize::try_from(sub_idx).map_err(|_| PDeflateError::NumericOverflow)?)
                .ok_or(PDeflateError::NumericOverflow)?;
            let entry = HuffmanLutEntry::Symbol {
                symbol: sym_u16,
                bit_len: code.bit_len,
            };
            match subtables[lut_idx] {
                HuffmanLutEntry::Invalid => subtables[lut_idx] = entry,
                v if v == entry => {}
                _ => {
                    return Err(PDeflateError::InvalidOptions("huffman subtable collision"));
                }
            }
        }
    }

    Ok(HuffmanLut {
        symbol_count: codebook.symbol_count,
        root_bits,
        max_code_bits: codebook.max_code_bits,
        root,
        subtables,
    })
}

fn pack_huffman_lut_entry(entry: HuffmanLutEntry) -> Result<u32, PDeflateError> {
    let packed = match entry {
        HuffmanLutEntry::Invalid => HUFF_LUT_ENTRY_KIND_INVALID,
        HuffmanLutEntry::Symbol { symbol, bit_len } => {
            HUFF_LUT_ENTRY_KIND_SYMBOL | (u32::from(bit_len) << 2) | (u32::from(symbol) << 10)
        }
        HuffmanLutEntry::Subtable { offset, sub_bits } => {
            HUFF_LUT_ENTRY_KIND_SUBTABLE | (u32::from(sub_bits) << 2) | (offset << 10)
        }
    };
    Ok(packed)
}

fn unpack_huffman_lut_entry(packed: u32) -> Result<HuffmanLutEntry, PDeflateError> {
    let kind = packed & 0x3;
    let bits = ((packed >> 2) & 0xff) as u8;
    match kind {
        HUFF_LUT_ENTRY_KIND_INVALID => Ok(HuffmanLutEntry::Invalid),
        HUFF_LUT_ENTRY_KIND_SYMBOL => {
            let symbol = ((packed >> 10) & 0xffff) as u16;
            Ok(HuffmanLutEntry::Symbol {
                symbol,
                bit_len: bits,
            })
        }
        HUFF_LUT_ENTRY_KIND_SUBTABLE => {
            let offset = packed >> 10;
            Ok(HuffmanLutEntry::Subtable {
                offset,
                sub_bits: bits,
            })
        }
        _ => Err(PDeflateError::InvalidStream(
            "huffman lut entry kind invalid",
        )),
    }
}

fn serialize_huffman_lut(lut: &HuffmanLut) -> Result<Vec<u8>, PDeflateError> {
    let mut out = Vec::with_capacity(
        HUFF_LUT_HEADER_SIZE
            + lut
                .root
                .len()
                .saturating_add(lut.subtables.len())
                .saturating_mul(4),
    );
    write_u16_le(
        &mut out,
        u16::try_from(lut.symbol_count).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u8(&mut out, lut.root_bits);
    write_u8(&mut out, lut.max_code_bits);
    write_u32_le(
        &mut out,
        u32::try_from(lut.root.len()).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u32_le(
        &mut out,
        u32::try_from(lut.subtables.len()).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    for &entry in &lut.root {
        write_u32_le(&mut out, pack_huffman_lut_entry(entry)?);
    }
    for &entry in &lut.subtables {
        write_u32_le(&mut out, pack_huffman_lut_entry(entry)?);
    }
    Ok(out)
}

fn deserialize_huffman_lut(src: &[u8]) -> Result<HuffmanLut, PDeflateError> {
    let mut cursor = 0usize;
    if src.len() < HUFF_LUT_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream("huffman lut header truncated"));
    }
    let symbol_count = read_u16_le(src, &mut cursor)? as usize;
    if symbol_count == 0 {
        return Err(PDeflateError::InvalidStream(
            "huffman lut symbol_count is zero",
        ));
    }
    let root_bits = read_u8(src, &mut cursor)?;
    let max_code_bits = read_u8(src, &mut cursor)?;
    if root_bits == 0 || root_bits > max_code_bits || max_code_bits > 31 {
        return Err(PDeflateError::InvalidStream(
            "huffman lut bit width is invalid",
        ));
    }
    let root_len = read_u32_le(src, &mut cursor)? as usize;
    let sub_len = read_u32_le(src, &mut cursor)? as usize;
    let expected_root_len = 1usize
        .checked_shl(u32::from(root_bits))
        .ok_or(PDeflateError::NumericOverflow)?;
    if root_len != expected_root_len {
        return Err(PDeflateError::InvalidStream(
            "huffman lut root length mismatch",
        ));
    }

    let total_entry_count = root_len
        .checked_add(sub_len)
        .ok_or(PDeflateError::NumericOverflow)?;
    let expected_bytes = HUFF_LUT_HEADER_SIZE
        .checked_add(
            total_entry_count
                .checked_mul(4)
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .ok_or(PDeflateError::NumericOverflow)?;
    if src.len() != expected_bytes {
        return Err(PDeflateError::InvalidStream(
            "huffman lut payload length mismatch",
        ));
    }

    let mut root = Vec::with_capacity(root_len);
    for _ in 0..root_len {
        root.push(unpack_huffman_lut_entry(read_u32_le(src, &mut cursor)?)?);
    }
    let mut subtables = Vec::with_capacity(sub_len);
    for _ in 0..sub_len {
        subtables.push(unpack_huffman_lut_entry(read_u32_le(src, &mut cursor)?)?);
    }
    if cursor != src.len() {
        return Err(PDeflateError::InvalidStream(
            "trailing bytes in huffman lut payload",
        ));
    }

    for &entry in &root {
        match entry {
            HuffmanLutEntry::Invalid => {}
            HuffmanLutEntry::Symbol { symbol, bit_len } => {
                if bit_len == 0 || bit_len > root_bits {
                    return Err(PDeflateError::InvalidStream(
                        "huffman lut root symbol bit_len invalid",
                    ));
                }
                if usize::from(symbol) >= symbol_count {
                    return Err(PDeflateError::InvalidStream(
                        "huffman lut root symbol out of range",
                    ));
                }
            }
            HuffmanLutEntry::Subtable { offset, sub_bits } => {
                if sub_bits == 0 {
                    return Err(PDeflateError::InvalidStream(
                        "huffman lut subtable bit width is zero",
                    ));
                }
                let total_bits = root_bits
                    .checked_add(sub_bits)
                    .ok_or(PDeflateError::NumericOverflow)?;
                if total_bits > max_code_bits {
                    return Err(PDeflateError::InvalidStream(
                        "huffman lut subtable exceeds max bits",
                    ));
                }
                let span = 1usize
                    .checked_shl(u32::from(sub_bits))
                    .ok_or(PDeflateError::NumericOverflow)?;
                let end = (offset as usize)
                    .checked_add(span)
                    .ok_or(PDeflateError::NumericOverflow)?;
                if end > sub_len {
                    return Err(PDeflateError::InvalidStream(
                        "huffman lut subtable range out of bounds",
                    ));
                }
            }
        }
    }

    for &entry in &subtables {
        match entry {
            HuffmanLutEntry::Symbol { symbol, bit_len } => {
                if bit_len <= root_bits || bit_len > max_code_bits {
                    return Err(PDeflateError::InvalidStream(
                        "huffman lut subtable symbol bit_len invalid",
                    ));
                }
                if usize::from(symbol) >= symbol_count {
                    return Err(PDeflateError::InvalidStream(
                        "huffman lut subtable symbol out of range",
                    ));
                }
            }
            HuffmanLutEntry::Invalid => {
                return Err(PDeflateError::InvalidStream(
                    "huffman lut subtable has invalid hole",
                ));
            }
            HuffmanLutEntry::Subtable { .. } => {
                return Err(PDeflateError::InvalidStream(
                    "huffman lut subtable entry is not terminal symbol",
                ));
            }
        }
    }

    Ok(HuffmanLut {
        symbol_count,
        root_bits,
        max_code_bits,
        root,
        subtables,
    })
}

fn write_bits_lsb(
    dst: &mut Vec<u8>,
    bit_cursor: &mut usize,
    mut bits: u32,
    mut bit_len: u8,
) -> Result<(), PDeflateError> {
    while bit_len > 0 {
        let byte_index = *bit_cursor / 8;
        let bit_index = (*bit_cursor & 7) as u8;
        if byte_index == dst.len() {
            dst.push(0);
        }
        let room = 8 - bit_index;
        let take = room.min(bit_len);
        let mask = (1u32 << u32::from(take)) - 1;
        let chunk = (bits & mask) as u8;
        dst[byte_index] |= chunk << bit_index;
        *bit_cursor = bit_cursor.saturating_add(take as usize);
        bits >>= take;
        bit_len -= take;
    }
    Ok(())
}

fn encode_symbols_with_huffman_codebook(
    symbols: &[u8],
    codebook: &HuffmanCanonicalCodebook,
) -> Result<(Vec<u8>, usize), PDeflateError> {
    let mut out = Vec::new();
    let mut bit_cursor = 0usize;
    for &symbol in symbols {
        let sym = symbol as usize;
        if sym >= codebook.codes.len() {
            return Err(PDeflateError::InvalidStream(
                "huffman symbol out of codebook range",
            ));
        }
        let code = codebook.codes[sym].ok_or(PDeflateError::InvalidStream(
            "huffman symbol has no assigned code",
        ))?;
        write_bits_lsb(&mut out, &mut bit_cursor, code.code_lsb, code.bit_len)?;
    }
    Ok((out, bit_cursor))
}

fn build_decode_tree_from_codebook(
    codebook: &HuffmanCanonicalCodebook,
) -> Result<Vec<HuffmanDecodeTreeNode>, PDeflateError> {
    let mut tree = vec![HuffmanDecodeTreeNode::default()];
    for (symbol, code_opt) in codebook.codes.iter().enumerate() {
        let Some(code) = *code_opt else {
            continue;
        };
        let sym_u16 = u16::try_from(symbol).map_err(|_| PDeflateError::NumericOverflow)?;
        let mut node = 0usize;
        for bit_index in 0..code.bit_len {
            if tree[node].symbol.is_some() {
                return Err(PDeflateError::InvalidOptions(
                    "huffman decode tree prefix conflict",
                ));
            }
            let is_last = bit_index + 1 == code.bit_len;
            let bit = ((code.code_lsb >> u32::from(bit_index)) & 1) as u8;
            let current_next = if bit == 0 {
                tree[node].left
            } else {
                tree[node].right
            };
            if is_last {
                if let Some(child) = current_next {
                    if tree[child].symbol != Some(sym_u16)
                        || tree[child].left.is_some()
                        || tree[child].right.is_some()
                    {
                        return Err(PDeflateError::InvalidOptions(
                            "huffman decode tree leaf conflict",
                        ));
                    }
                } else {
                    tree.push(HuffmanDecodeTreeNode {
                        left: None,
                        right: None,
                        symbol: Some(sym_u16),
                    });
                    let created = tree.len() - 1;
                    if bit == 0 {
                        tree[node].left = Some(created);
                    } else {
                        tree[node].right = Some(created);
                    }
                }
            } else {
                let child = if let Some(existing) = current_next {
                    existing
                } else {
                    tree.push(HuffmanDecodeTreeNode::default());
                    let created = tree.len() - 1;
                    if bit == 0 {
                        tree[node].left = Some(created);
                    } else {
                        tree[node].right = Some(created);
                    }
                    created
                };
                node = child;
            }
        }
    }
    Ok(tree)
}

fn decode_huffman_symbols_with_tree(
    bitstream: &[u8],
    bit_len: usize,
    codebook: &HuffmanCanonicalCodebook,
) -> Result<Vec<u8>, PDeflateError> {
    let tree = build_decode_tree_from_codebook(codebook)?;
    let mut reader = BitReaderLsb::new(bitstream, bit_len)?;
    let mut out = Vec::new();
    while reader.remaining_bits() > 0 {
        let mut node = 0usize;
        loop {
            if let Some(symbol) = tree[node].symbol {
                out.push(u8::try_from(symbol).map_err(|_| PDeflateError::NumericOverflow)?);
                break;
            }
            let bit = reader.read_bit()?;
            let next = if bit == 0 {
                tree[node].left
            } else {
                tree[node].right
            }
            .ok_or(PDeflateError::InvalidStream(
                "huffman tree decode reached invalid edge",
            ))?;
            node = next;
        }
    }
    Ok(out)
}

fn decode_huffman_symbols_with_lut(
    bitstream: &[u8],
    bit_len: usize,
    lut: &HuffmanLut,
) -> Result<Vec<u8>, PDeflateError> {
    let mut reader = BitReaderLsb::new(bitstream, bit_len)?;
    let root_mask = (1u32 << u32::from(lut.root_bits)) - 1;
    let mut out = Vec::new();
    while reader.remaining_bits() > 0 {
        let root_idx = (reader.peek_bits_padded(lut.root_bits) & root_mask) as usize;
        let root_entry = *lut.root.get(root_idx).ok_or(PDeflateError::InvalidStream(
            "huffman lut root index out of bounds",
        ))?;
        match root_entry {
            HuffmanLutEntry::Symbol { symbol, bit_len } => {
                reader.consume(bit_len)?;
                out.push(u8::try_from(symbol).map_err(|_| PDeflateError::NumericOverflow)?);
            }
            HuffmanLutEntry::Subtable { offset, sub_bits } => {
                let full = reader.peek_bits_padded(lut.root_bits + sub_bits);
                let sub_idx = ((full >> u32::from(lut.root_bits))
                    & ((1u32 << u32::from(sub_bits)) - 1)) as usize;
                let entry = *lut
                    .subtables
                    .get((offset as usize).saturating_add(sub_idx))
                    .ok_or(PDeflateError::InvalidStream(
                        "huffman lut subtable index out of bounds",
                    ))?;
                let HuffmanLutEntry::Symbol { symbol, bit_len } = entry else {
                    return Err(PDeflateError::InvalidStream(
                        "huffman lut subtable entry is not terminal symbol",
                    ));
                };
                reader.consume(bit_len)?;
                out.push(u8::try_from(symbol).map_err(|_| PDeflateError::NumericOverflow)?);
            }
            HuffmanLutEntry::Invalid => {
                return Err(PDeflateError::InvalidStream(
                    "huffman lut decode hit invalid root entry",
                ));
            }
        }
    }
    Ok(out)
}

#[inline]
fn logical_commands_to_huffman_symbols(section_cmd: &[u8]) -> &[u8] {
    // T02: introduce a dedicated symbol layer before Huffman.
    // At this stage, each command byte maps to one symbol (identity).
    section_cmd
}

fn encode_huffman_symbols_to_bitstream(
    symbols: &[u8],
    section_bitstream: &mut Vec<u8>,
) -> Result<u32, PDeflateError> {
    section_bitstream.extend_from_slice(symbols);
    let bit_len = symbols
        .len()
        .checked_mul(8)
        .ok_or(PDeflateError::NumericOverflow)?;
    u32::try_from(bit_len).map_err(|_| PDeflateError::NumericOverflow)
}

fn section_bit_len_to_byte_len(section_bit_len: usize) -> Result<usize, PDeflateError> {
    if (section_bit_len & 7) != 0 {
        eprintln!(
            "[cozip_pdeflate][error][section-bit-len] bit_len={} low3={} (expected byte aligned)",
            section_bit_len,
            section_bit_len & 7
        );
        return Err(PDeflateError::InvalidStream(
            "section bit_len is not byte-aligned",
        ));
    }
    Ok(section_bit_len / 8)
}

fn decode_section_bitstream_to_huffman_symbols(
    section_bits: &[u8],
    section_bit_len: usize,
) -> Result<Vec<u8>, PDeflateError> {
    let lut = &build_identity_huffman_lut_block()
        .and_then(|payload| deserialize_huffman_lut(&payload))?;
    decode_section_bitstream_to_huffman_symbols_with_lut(section_bits, section_bit_len, lut)
}

fn decode_section_bitstream_to_huffman_symbols_with_lut(
    section_bits: &[u8],
    section_bit_len: usize,
    huff_lut: &HuffmanLut,
) -> Result<Vec<u8>, PDeflateError> {
    let byte_len = section_bit_len_to_byte_len(section_bit_len)?;
    if byte_len != section_bits.len() {
        return Err(PDeflateError::InvalidStream(
            "section bitstream length mismatch",
        ));
    }
    decode_huffman_symbols_with_lut(section_bits, section_bit_len, huff_lut)
}

#[inline]
fn huffman_symbols_to_logical_commands(symbols: Vec<u8>) -> Vec<u8> {
    // T02: inverse of logical_commands_to_huffman_symbols (identity for now).
    symbols
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn profile_enabled() -> bool {
    static PROFILE: OnceLock<bool> = OnceLock::new();
    *PROFILE.get_or_init(|| match std::env::var("COZIP_PDEFLATE_PROFILE") {
        Ok(v) => {
            let s = v.trim();
            !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    })
}

fn gpu_table_build_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("COZIP_PDEFLATE_GPU_TABLE_BUILD") {
        Ok(v) => {
            let s = v.trim();
            !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
        }
        Err(_) => true,
    })
}

fn profile_detail_enabled() -> bool {
    static PROFILE_DETAIL: OnceLock<bool> = OnceLock::new();
    *PROFILE_DETAIL.get_or_init(|| {
        if !profile_enabled() {
            return false;
        }
        if let Ok(v) = std::env::var("COZIP_PDEFLATE_PROFILE_DETAIL") {
            let s = v.trim();
            return !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"));
        }
        match std::env::var("COZIP_PDEFLATE_PROFILE") {
            Ok(v) => {
                let s = v.trim();
                s == "2"
                    || s.eq_ignore_ascii_case("detail")
                    || s.eq_ignore_ascii_case("full")
                    || s.eq_ignore_ascii_case("verbose")
            }
            Err(_) => false,
        }
    })
}

fn profile_detail_all_enabled() -> bool {
    static PROFILE_DETAIL_ALL: OnceLock<bool> = OnceLock::new();
    *PROFILE_DETAIL_ALL.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_DETAIL_ALL") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => false,
        },
    )
}

fn parse_nonzero_usize_env(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    match s.parse::<usize>() {
        Ok(v) if v > 0 => Some(v),
        _ => None,
    }
}

fn gpu_sparse_target_inflight() -> usize {
    parse_nonzero_usize_env("COZIP_PDEFLATE_GPU_SPARSE_TARGET_INFLIGHT").unwrap_or(2)
}

fn gpu_sparse_wait_high_watermark() -> usize {
    parse_nonzero_usize_env("COZIP_PDEFLATE_GPU_SPARSE_WAIT_HIGH_WATERMARK").unwrap_or(4)
}

fn profile_detail_sample_head() -> usize {
    static PROFILE_DETAIL_HEAD: OnceLock<usize> = OnceLock::new();
    *PROFILE_DETAIL_HEAD
        .get_or_init(|| parse_nonzero_usize_env("COZIP_PDEFLATE_PROFILE_DETAIL_HEAD").unwrap_or(8))
}

fn profile_detail_sample_stride() -> usize {
    static PROFILE_DETAIL_STRIDE: OnceLock<usize> = OnceLock::new();
    *PROFILE_DETAIL_STRIDE.get_or_init(|| {
        parse_nonzero_usize_env("COZIP_PDEFLATE_PROFILE_DETAIL_STRIDE").unwrap_or(64)
    })
}

fn emit_profile_detail_sampling_notice_once() {
    static PROFILE_DETAIL_SAMPLING_NOTICE: OnceLock<()> = OnceLock::new();
    if PROFILE_DETAIL_SAMPLING_NOTICE.set(()).is_ok() {
        eprintln!(
            "[cozip_pdeflate][timing][detail-sampling] mode=sampled head={} stride={} (set COZIP_PDEFLATE_PROFILE_DETAIL_ALL=1 to disable sampling)",
            profile_detail_sample_head(),
            profile_detail_sample_stride()
        );
    }
}

fn profile_detail_should_log_stream(seq: u64) -> bool {
    if profile_detail_all_enabled() {
        return true;
    }
    emit_profile_detail_sampling_notice_once();
    let head = profile_detail_sample_head() as u64;
    if seq < head {
        return true;
    }
    let stride = profile_detail_sample_stride() as u64;
    ((seq - head) % stride) == 0
}

fn profile_detail_should_log_indexed(index: usize, total: usize) -> bool {
    if profile_detail_all_enabled() {
        return true;
    }
    emit_profile_detail_sampling_notice_once();
    let head = profile_detail_sample_head();
    if index < head || index.saturating_add(head) >= total {
        return true;
    }
    let stride = profile_detail_sample_stride();
    ((index - head) % stride) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};

    fn gen_data(size: usize) -> Vec<u8> {
        let mut out = vec![0u8; size];
        let text = b"ABABABABCCABCCD--cozip-pdeflate--";
        let mut x = 0x1234_5678u32;
        for i in 0..size {
            out[i] = match (i / 4096) % 4 {
                0 => text[i % text.len()],
                1 => {
                    x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                    (x >> 24) as u8
                }
                2 => (i as u8).wrapping_mul(31).wrapping_add(7),
                _ => b'A' + ((i / 17) % 5) as u8,
            };
        }
        out
    }

    fn gen_data_seeded(size: usize, seed: u32) -> Vec<u8> {
        let mut out = vec![0u8; size];
        let mut x = seed;
        for b in &mut out {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (x >> 24) as u8;
        }
        out
    }

    fn decompress_with_options(stream: &[u8], opts: &PDeflateOptions) -> Vec<u8> {
        let mut out = Vec::new();
        pdeflate_decompress_into_with_stats_with_options(stream, &mut out, opts)
            .expect("decompress with options");
        out
    }

    fn extract_chunk_payloads(stream: &[u8]) -> Vec<Vec<u8>> {
        let mut cursor = 0usize;
        let magic = read_exact(stream, &mut cursor, 4).expect("stream magic");
        assert_eq!(magic, STREAM_MAGIC);
        let version = read_u16_le(stream, &mut cursor).expect("version");
        assert_eq!(version, STREAM_VERSION);
        let _flags = read_u16_le(stream, &mut cursor).expect("flags");
        let _chunk_size = read_u32_le(stream, &mut cursor).expect("chunk_size");
        let mut payloads = Vec::new();
        while cursor < stream.len() {
            let chunk_len = read_u32_le(stream, &mut cursor).expect("chunk_len") as usize;
            let payload = read_exact(stream, &mut cursor, chunk_len).expect("chunk_payload");
            let flags = read_chunk_flags(payload).expect("chunk flags");
            payloads.push(payload.to_vec());
            if chunk_final_stream_enabled(flags) {
                break;
            }
        }
        assert_eq!(cursor, stream.len(), "no trailing bytes expected");
        payloads
    }

    fn compress_single_chunk_payload_with_huffman(
        input: &[u8],
        section_count: usize,
        huffman_encode_enabled: bool,
    ) -> Vec<u8> {
        let opts = PDeflateOptions {
            chunk_size: input.len().max(1),
            section_count,
            huffman_encode_enabled,
            ..PDeflateOptions::default()
        };
        let stream = pdeflate_compress(input, &opts).expect("compress single chunk");
        let payloads = extract_chunk_payloads(&stream);
        assert_eq!(payloads.len(), 1, "single chunk expected");
        payloads.into_iter().next().expect("single payload")
    }

    fn compress_single_chunk_payload(input: &[u8], section_count: usize) -> Vec<u8> {
        compress_single_chunk_payload_with_huffman(input, section_count, false)
    }

    fn make_gpu_decode_job<'a>(chunk_index: usize, payload: &'a [u8]) -> gpu::GpuDecodeJob<'a> {
        let preprocess = preprocess_chunk_for_gpu_decode(payload).expect("preprocess");
        gpu::GpuDecodeJob {
            chunk_index,
            payload,
            table_count: preprocess.table_count,
            chunk_uncompressed_len: preprocess.chunk_uncompressed_len,
            out_offset: 0,
            out_len: preprocess.chunk_uncompressed_len,
            section_meta: preprocess.section_meta,
            preferred_slot: None,
        }
    }

    fn decode_chunk_payload_cpu(payload: &[u8]) -> Vec<u8> {
        let preprocess = preprocess_chunk_for_gpu_decode(payload).expect("preprocess");
        let mut out = vec![0u8; preprocess.chunk_uncompressed_len];
        decompress_chunk_into(payload, &mut out).expect("cpu decode");
        out
    }

    fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("gpu test lock")
    }

    fn chunk_section_cmd_len(payload: &[u8]) -> usize {
        let mut cursor = 0usize;
        let _magic = read_exact(payload, &mut cursor, 4).expect("chunk magic");
        let _version = read_u16_le(payload, &mut cursor).expect("chunk version");
        let _flags = read_u16_le(payload, &mut cursor).expect("chunk flags");
        let _chunk_uncompressed_len = read_u32_le(payload, &mut cursor).expect("chunk out len");
        let _table_count = read_u16_le(payload, &mut cursor).expect("chunk table count");
        let _section_count = read_u16_le(payload, &mut cursor).expect("chunk section count");
        let _table_index_offset = read_u32_le(payload, &mut cursor).expect("table index off");
        let _table_data_offset = read_u32_le(payload, &mut cursor).expect("table data off");
        let _huff_lut_offset = read_u32_le(payload, &mut cursor).expect("huff lut off");
        let _section_index_offset = read_u32_le(payload, &mut cursor).expect("section index off");
        let section_bitstream_offset =
            read_u32_le(payload, &mut cursor).expect("section bitstream off");
        payload
            .len()
            .saturating_sub(section_bitstream_offset as usize)
    }

    fn chunk_offsets(payload: &[u8]) -> (usize, usize, usize, usize, usize) {
        let mut cursor = 0usize;
        let _magic = read_exact(payload, &mut cursor, 4).expect("chunk magic");
        let _version = read_u16_le(payload, &mut cursor).expect("chunk version");
        let _flags = read_u16_le(payload, &mut cursor).expect("chunk flags");
        let _chunk_uncompressed_len = read_u32_le(payload, &mut cursor).expect("chunk out len");
        let _table_count = read_u16_le(payload, &mut cursor).expect("chunk table count");
        let _section_count = read_u16_le(payload, &mut cursor).expect("chunk section count");
        let table_index_offset = read_u32_le(payload, &mut cursor).expect("table index off");
        let table_data_offset = read_u32_le(payload, &mut cursor).expect("table data off");
        let huff_lut_offset = read_u32_le(payload, &mut cursor).expect("huff lut off");
        let section_index_offset = read_u32_le(payload, &mut cursor).expect("section index off");
        let section_bitstream_offset =
            read_u32_le(payload, &mut cursor).expect("section bitstream off");
        (
            table_index_offset as usize,
            table_data_offset as usize,
            huff_lut_offset as usize,
            section_index_offset as usize,
            section_bitstream_offset as usize,
        )
    }

    fn write_u32_le_at(dst: &mut [u8], offset: usize, value: u32) {
        let bytes = value.to_le_bytes();
        dst[offset..offset + 4].copy_from_slice(&bytes);
    }

    #[test]
    fn roundtrip_small() {
        let input = b"ABABABABCCABCCD".to_vec();
        let opts = PDeflateOptions::default();
        let compressed = pdeflate_compress(&input, &opts).expect("compress");
        let decoded = pdeflate_decompress(&compressed).expect("decompress");
        assert_eq!(decoded, input);
    }

    #[test]
    fn roundtrip_large() {
        let input = gen_data(2 * 1024 * 1024);
        let opts = PDeflateOptions::default();
        let compressed = pdeflate_compress(&input, &opts).expect("compress");
        let decoded = pdeflate_decompress(&compressed).expect("decompress");
        assert_eq!(decoded, input);
    }

    #[test]
    fn roundtrip_gpu_flag_enabled() {
        let input = gen_data(128 * 1024);
        let opts = PDeflateOptions {
            gpu_compress_enabled: true,
            ..PDeflateOptions::default()
        };
        let compressed = pdeflate_compress(&input, &opts).expect("compress");
        let decoded = pdeflate_decompress(&compressed).expect("decompress");
        assert_eq!(decoded, input);
    }

    #[test]
    fn reject_corrupted_chunk_offsets_v0_header() {
        let input = b"AAABBBCCCDDDEEEFFF".to_vec();
        let mut payload = compress_single_chunk_payload(&input, 1);
        let bad_section_bitstream_offset = u32::try_from(payload.len() + 1).expect("fits u32");
        // v0 header: section_bitstream_offset is at byte offset 32.
        write_u32_le_at(&mut payload, 32, bad_section_bitstream_offset);

        let preprocess_err = preprocess_chunk_for_gpu_decode(&payload).expect_err("must fail");
        assert!(
            matches!(
                preprocess_err,
                PDeflateError::InvalidStream("invalid chunk offsets")
            ),
            "unexpected preprocess error: {preprocess_err}"
        );

        let mut out = vec![0u8; input.len()];
        let decode_err = decompress_chunk_into(&payload, &mut out).expect_err("must fail");
        assert!(
            matches!(
                decode_err,
                PDeflateError::InvalidStream("invalid chunk offsets")
            ),
            "unexpected decode error: {decode_err}"
        );
    }

    #[test]
    fn reject_non_byte_aligned_section_bit_len() {
        let input = b"ABABABABABABABAB".to_vec();
        let mut payload = compress_single_chunk_payload(&input, 1);
        let section_index_offset = u32::from_le_bytes(
            payload[28..32]
                .try_into()
                .expect("section_index_offset bytes"),
        ) as usize;
        payload[section_index_offset] = payload[section_index_offset].wrapping_add(1);

        let preprocess_err = preprocess_chunk_for_gpu_decode(&payload).expect_err("must fail");
        assert!(
            matches!(
                preprocess_err,
                PDeflateError::InvalidStream("section bit_len is not byte-aligned")
            ),
            "unexpected preprocess error: {preprocess_err}"
        );
    }

    #[test]
    fn chunk_contains_identity_huffman_lut_block() {
        let input = b"ABABABABABABABAB".to_vec();
        let payload = compress_single_chunk_payload_with_huffman(&input, 1, true);
        let (
            table_index_offset,
            table_data_offset,
            huff_lut_offset,
            section_index_offset,
            section_bitstream_offset,
        ) = chunk_offsets(&payload);
        assert_eq!(table_index_offset, CHUNK_HEADER_SIZE);
        assert!(table_index_offset <= table_data_offset);
        assert!(table_data_offset <= huff_lut_offset);
        assert!(huff_lut_offset < section_index_offset);
        assert!(section_index_offset <= section_bitstream_offset);

        let huff_lut = &payload[huff_lut_offset..section_index_offset];
        assert!(!huff_lut.is_empty());
        let lut = deserialize_huffman_lut(huff_lut).expect("deserialize chunk lut");
        assert_eq!(lut.symbol_count, 256);
        assert_eq!(lut.root_bits, 8);
        assert_eq!(lut.max_code_bits, 8);
        assert_eq!(lut.root.len(), 256);
        assert!(lut.subtables.is_empty());
        for (idx, entry) in lut.root.iter().copied().enumerate() {
            match entry {
                HuffmanLutEntry::Symbol { symbol, bit_len } => {
                    assert_eq!(usize::from(symbol), idx);
                    assert_eq!(bit_len, 8);
                }
                _ => panic!("identity lut root must contain direct symbols"),
            }
        }
    }

    #[test]
    fn reject_corrupted_chunk_huffman_lut_block() {
        let input = b"ABABABABABABABAB".to_vec();
        let mut payload = compress_single_chunk_payload_with_huffman(&input, 1, true);
        let (_table_index_offset, _table_data_offset, huff_lut_offset, _section_index_offset, _) =
            chunk_offsets(&payload);
        // huff lut header: [symbol_count:u16][root_bits:u8][max_code_bits:u8][root_len:u32]...
        write_u32_le_at(&mut payload, huff_lut_offset + 4, 1);

        let preprocess_err = preprocess_chunk_for_gpu_decode(&payload).expect_err("must fail");
        assert!(
            matches!(
                preprocess_err,
                PDeflateError::InvalidStream("huffman lut root length mismatch")
            ),
            "unexpected preprocess error: {preprocess_err}"
        );

        let mut out = vec![0u8; input.len()];
        let decode_err = decompress_chunk_into(&payload, &mut out).expect_err("must fail");
        assert!(
            matches!(
                decode_err,
                PDeflateError::InvalidStream("huffman lut root length mismatch")
            ),
            "unexpected decode error: {decode_err}"
        );
    }

    #[test]
    fn logical_command_bitstream_identity_roundtrip() {
        let section_cmd_lens = [3u32, 0u32, 7u32, 2u32];
        let logical_cmd_stream: Vec<u8> = vec![
            0x01, 0x02, 0x03, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0xaa, 0xbb,
        ];
        assert_eq!(
            logical_cmd_stream.len(),
            section_cmd_lens
                .iter()
                .fold(0usize, |acc, &v| acc.saturating_add(v as usize))
        );

        let mut section_index = Vec::new();
        let mut section_bitstream = Vec::new();
        let mut src_cursor = 0usize;
        for &len_u32 in &section_cmd_lens {
            let len = usize::try_from(len_u32).expect("len fits usize");
            let end = src_cursor.saturating_add(len);
            let sec_cmd = &logical_cmd_stream[src_cursor..end];
            let symbols = logical_commands_to_huffman_symbols(sec_cmd);
            let bit_len = encode_huffman_symbols_to_bitstream(symbols, &mut section_bitstream)
                .expect("encode symbols");
            write_varint_u32(&mut section_index, bit_len);
            src_cursor = end;
        }
        assert_eq!(src_cursor, logical_cmd_stream.len());

        let mut decoded_cmd_stream = Vec::new();
        let mut index_cursor = 0usize;
        let mut bit_cursor = 0usize;
        for _ in 0..section_cmd_lens.len() {
            let sec_bit_len_u32 =
                read_varint_u32(&section_index, &mut index_cursor).expect("read bit len");
            let sec_bit_len = usize::try_from(sec_bit_len_u32).expect("fits usize");
            let sec_byte_len = section_bit_len_to_byte_len(sec_bit_len).expect("byte aligned");
            let sec_end = bit_cursor.saturating_add(sec_byte_len);
            let sec_bits = &section_bitstream[bit_cursor..sec_end];
            let symbols = decode_section_bitstream_to_huffman_symbols(sec_bits, sec_bit_len)
                .expect("decode symbols");
            let logical = huffman_symbols_to_logical_commands(symbols);
            decoded_cmd_stream.extend_from_slice(&logical);
            bit_cursor = sec_end;
        }
        assert_eq!(index_cursor, section_index.len());
        assert_eq!(bit_cursor, section_bitstream.len());
        assert_eq!(decoded_cmd_stream, logical_cmd_stream);
    }

    #[test]
    fn reject_section_bitstream_length_mismatch() {
        let err = decode_section_bitstream_to_huffman_symbols(&[0x12u8], 16)
            .expect_err("must reject mismatched section bitstream length");
        assert!(
            matches!(
                err,
                PDeflateError::InvalidStream("section bitstream length mismatch")
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn huffman_lut_decode_matches_tree_decode() {
        let mut symbols = Vec::new();
        for i in 0..4096usize {
            symbols.push((i % 7) as u8);
            if (i % 3) == 0 {
                symbols.push(((i / 5) % 11) as u8);
            }
            if (i % 17) == 0 {
                symbols.push(31u8);
            }
        }

        let mut frequencies = vec![0u32; 256];
        for &s in &symbols {
            frequencies[s as usize] = frequencies[s as usize].saturating_add(1);
        }

        let codebook =
            build_canonical_huffman_codebook_from_frequencies(&frequencies, HUFF_MAX_CODE_BITS)
                .expect("build canonical codebook");
        let root_bits = HUFF_LUT_ROOT_BITS_DEFAULT
            .min(codebook.max_code_bits)
            .max(1);
        let lut = build_huffman_lut(&codebook, root_bits).expect("build huffman lut");
        let serialized = serialize_huffman_lut(&lut).expect("serialize huffman lut");
        let parsed_lut = deserialize_huffman_lut(&serialized).expect("deserialize huffman lut");
        let (bitstream, bit_len) =
            encode_symbols_with_huffman_codebook(&symbols, &codebook).expect("encode symbols");

        let decoded_tree =
            decode_huffman_symbols_with_tree(&bitstream, bit_len, &codebook).expect("tree decode");
        let decoded_lut =
            decode_huffman_symbols_with_lut(&bitstream, bit_len, &parsed_lut).expect("lut decode");

        assert_eq!(decoded_tree, symbols);
        assert_eq!(decoded_lut, symbols);
        assert_eq!(decoded_lut, decoded_tree);
    }

    #[test]
    fn reject_huffman_lut_subtable_out_of_range() {
        let lut = HuffmanLut {
            symbol_count: 4,
            root_bits: 1,
            max_code_bits: 2,
            root: vec![
                HuffmanLutEntry::Subtable {
                    offset: 0,
                    sub_bits: 1,
                },
                HuffmanLutEntry::Symbol {
                    symbol: 0,
                    bit_len: 1,
                },
            ],
            subtables: vec![
                HuffmanLutEntry::Symbol {
                    symbol: 1,
                    bit_len: 2,
                },
                HuffmanLutEntry::Symbol {
                    symbol: 2,
                    bit_len: 2,
                },
            ],
        };
        let mut payload = serialize_huffman_lut(&lut).expect("serialize huffman lut");
        let bad = pack_huffman_lut_entry(HuffmanLutEntry::Subtable {
            offset: 3,
            sub_bits: 1,
        })
        .expect("pack bad lut entry");
        write_u32_le_at(&mut payload, HUFF_LUT_HEADER_SIZE, bad);
        let err = deserialize_huffman_lut(&payload).expect_err("must reject out-of-range subtable");
        assert!(
            matches!(
                err,
                PDeflateError::InvalidStream("huffman lut subtable range out of bounds")
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reject_huffman_lut_subtable_entry_that_points_to_subtable() {
        let lut = HuffmanLut {
            symbol_count: 4,
            root_bits: 1,
            max_code_bits: 2,
            root: vec![
                HuffmanLutEntry::Subtable {
                    offset: 0,
                    sub_bits: 1,
                },
                HuffmanLutEntry::Symbol {
                    symbol: 0,
                    bit_len: 1,
                },
            ],
            subtables: vec![
                HuffmanLutEntry::Symbol {
                    symbol: 1,
                    bit_len: 2,
                },
                HuffmanLutEntry::Symbol {
                    symbol: 2,
                    bit_len: 2,
                },
            ],
        };
        let mut payload = serialize_huffman_lut(&lut).expect("serialize huffman lut");
        let subtable_offset = HUFF_LUT_HEADER_SIZE + lut.root.len().saturating_mul(4);
        let bad = pack_huffman_lut_entry(HuffmanLutEntry::Subtable {
            offset: 0,
            sub_bits: 1,
        })
        .expect("pack bad lut entry");
        write_u32_le_at(&mut payload, subtable_offset, bad);
        let err =
            deserialize_huffman_lut(&payload).expect_err("must reject subtable chain/cycle shape");
        assert!(
            matches!(
                err,
                PDeflateError::InvalidStream("huffman lut subtable entry is not terminal symbol")
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gpu_decode_v2_matches_cpu_fixed_seed() {
        let _guard = gpu_test_lock();
        if !gpu::is_runtime_available() {
            return;
        }
        gpu::reset_decode_slot_pool_for_test().expect("reset decode slot pool");
        let input = gen_data_seeded(512 * 1024, 0x1234_5678);
        let compressed = pdeflate_compress(&input, &PDeflateOptions::default()).expect("compress");
        let cpu_opts = PDeflateOptions {
            gpu_decompress_enabled: false,
            gpu_decompress_force_gpu: false,
            ..PDeflateOptions::default()
        };
        let gpu_opts = PDeflateOptions {
            gpu_decompress_enabled: true,
            gpu_decompress_force_gpu: true,
            ..PDeflateOptions::default()
        };
        let cpu_out = decompress_with_options(&compressed, &cpu_opts);
        let gpu_out = decompress_with_options(&compressed, &gpu_opts);
        assert_eq!(cpu_out, input);
        assert_eq!(gpu_out, input);
        assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn gpu_decode_v2_matches_cpu_random_inputs() {
        let _guard = gpu_test_lock();
        if !gpu::is_runtime_available() {
            return;
        }
        gpu::reset_decode_slot_pool_for_test().expect("reset decode slot pool");
        let cpu_opts = PDeflateOptions {
            gpu_decompress_enabled: false,
            gpu_decompress_force_gpu: false,
            ..PDeflateOptions::default()
        };
        let gpu_opts = PDeflateOptions {
            gpu_decompress_enabled: true,
            gpu_decompress_force_gpu: true,
            ..PDeflateOptions::default()
        };
        for (i, seed) in [0x1u32, 0x1020_3040, 0xdead_beef].into_iter().enumerate() {
            let input = gen_data_seeded(192 * 1024 + i * 37_777, seed);
            let compressed =
                pdeflate_compress(&input, &PDeflateOptions::default()).expect("compress random");
            let cpu_out = decompress_with_options(&compressed, &cpu_opts);
            let gpu_out = decompress_with_options(&compressed, &gpu_opts);
            assert_eq!(cpu_out, input);
            assert_eq!(gpu_out, input);
            assert_eq!(gpu_out, cpu_out);
        }
    }

    #[test]
    fn gpu_decode_v2_section_boundary_cases() {
        let cases = [
            (1usize, 65_537usize, 0x1111_0001u32),
            (257usize, 131_089usize, 0x2222_0002u32),
            (128usize, 1_000_003usize, 0x3333_0003u32),
        ];
        for (section_count, input_len, seed) in cases {
            let input = gen_data_seeded(input_len, seed);
            let payload = compress_single_chunk_payload(&input, section_count);
            let preprocess = preprocess_chunk_for_gpu_decode(&payload).expect("preprocess");
            assert_eq!(preprocess.chunk_uncompressed_len, input_len);
            assert_eq!(preprocess.section_meta.len(), section_count);
            let mut total_out = 0usize;
            let mut total_cmd = 0usize;
            let mut prev_out_end = 0usize;
            let mut out_len_min = usize::MAX;
            let mut out_len_max = 0usize;
            for sec in &preprocess.section_meta {
                assert_eq!(
                    sec.out_offset, prev_out_end,
                    "section out must be contiguous"
                );
                if sec.out_offset != 0 && sec.out_offset != input_len {
                    assert_eq!(
                        sec.out_offset & 3,
                        0,
                        "internal section boundary must be 4-byte aligned"
                    );
                }
                total_out = total_out.saturating_add(sec.out_len);
                total_cmd = total_cmd.saturating_add(sec.cmd_len);
                prev_out_end = sec.out_offset.saturating_add(sec.out_len);
                out_len_min = out_len_min.min(sec.out_len);
                out_len_max = out_len_max.max(sec.out_len);
            }
            assert_eq!(total_out, input_len);
            assert_eq!(prev_out_end, input_len);
            assert_eq!(total_cmd, chunk_section_cmd_len(&payload));
            assert!(
                out_len_max.saturating_sub(out_len_min) <= 8,
                "section out len balance must be <=8 with 4-byte-aligned section boundaries"
            );
            if section_count == 1 {
                assert_eq!(preprocess.section_meta[0].out_offset, 0);
                assert_eq!(preprocess.section_meta[0].out_len, input_len);
            }
        }
    }

    #[test]
    fn gpu_decode_v2_uses_normal_and_large_slots() {
        let _guard = gpu_test_lock();
        if !gpu::is_runtime_available() {
            return;
        }
        gpu::reset_decode_slot_pool_for_test().expect("reset decode slot pool");
        let small_input = gen_data_seeded(48 * 1024, 0x4444_0101);
        let large_input = gen_data_seeded(96 * 1024, 0x4444_0202);
        let small_payload = compress_single_chunk_payload(&small_input, 64);
        let large_payload = compress_single_chunk_payload(&large_input, 64);
        let jobs = vec![
            make_gpu_decode_job(0, &small_payload),
            make_gpu_decode_job(1, &large_payload),
        ];
        let decode_opts = PDeflateOptions {
            gpu_decompress_enabled: true,
            gpu_decompress_force_gpu: true,
            chunk_size: 64 * 1024,
            gpu_slot_count: 4,
            gpu_submit_chunks: 2,
            ..PDeflateOptions::default()
        };
        let results = gpu::decode_chunks_gpu_v2(&jobs, &decode_opts).expect("decode v2");
        assert_eq!(results.len(), 2);
        let small = results
            .iter()
            .find(|r| r.chunk_index == 0)
            .expect("small result");
        let large = results
            .iter()
            .find(|r| r.chunk_index == 1)
            .expect("large result");
        assert_eq!(small.disposition, gpu::GpuDecodeDisposition::SubmittedGpu);
        assert_eq!(large.disposition, gpu::GpuDecodeDisposition::SubmittedGpu);
        assert_eq!(small.decoded_chunk.as_deref(), Some(small_input.as_slice()));
        assert_eq!(large.decoded_chunk.as_deref(), Some(large_input.as_slice()));
        let small_slot = small.slot.expect("small slot");
        let large_slot = large.slot.expect("large slot");
        assert!(
            small_slot.slot_index < (1usize << 30),
            "small chunk should use normal slot, got {}",
            small_slot.slot_index
        );
        assert!(
            large_slot.slot_index >= (1usize << 30),
            "large chunk should use large slot, got {}",
            large_slot.slot_index
        );
    }

    #[test]
    fn gpu_decode_v2_local_cpu_fallback_on_oversized_chunk() {
        let _guard = gpu_test_lock();
        if !gpu::is_runtime_available() {
            return;
        }
        gpu::reset_decode_slot_pool_for_test().expect("reset decode slot pool");
        let small_input = gen_data_seeded(32 * 1024, 0x5555_0101);
        let huge_input = gen_data_seeded(300 * 1024 + 13, 0x5555_0202);
        let small_payload = compress_single_chunk_payload(&small_input, 64);
        let huge_payload = compress_single_chunk_payload(&huge_input, 64);
        let mut huge_job = make_gpu_decode_job(1, &huge_payload);
        if let Some(last) = huge_job.section_meta.last_mut() {
            // Force only this job to fail GPU-side upload validation:
            // out_offset + out_len > chunk_uncompressed_len.
            last.out_offset = huge_job.chunk_uncompressed_len;
            last.out_len = 1;
        }
        let jobs = vec![make_gpu_decode_job(0, &small_payload), huge_job];
        let decode_opts = PDeflateOptions {
            gpu_decompress_enabled: true,
            gpu_decompress_force_gpu: true,
            chunk_size: 64 * 1024,
            gpu_slot_count: 4,
            gpu_submit_chunks: 2,
            ..PDeflateOptions::default()
        };
        let results = gpu::decode_chunks_gpu_v2(&jobs, &decode_opts).expect("decode v2");
        assert_eq!(results.len(), 2);
        let small = results
            .iter()
            .find(|r| r.chunk_index == 0)
            .expect("small result");
        let huge = results
            .iter()
            .find(|r| r.chunk_index == 1)
            .expect("huge result");
        assert_eq!(small.disposition, gpu::GpuDecodeDisposition::SubmittedGpu);
        assert_eq!(small.decoded_chunk.as_deref(), Some(small_input.as_slice()));
        assert_eq!(huge.disposition, gpu::GpuDecodeDisposition::CpuFallback);
        assert!(huge.decoded_chunk.is_none());
        let huge_cpu = decode_chunk_payload_cpu(&huge_payload);
        assert_eq!(huge_cpu, huge_input);
    }

    #[test]
    fn gpu_decode_v2_result_order_is_sorted_by_chunk_index() {
        let _guard = gpu_test_lock();
        if !gpu::is_runtime_available() {
            return;
        }
        gpu::reset_decode_slot_pool_for_test().expect("reset decode slot pool");
        let decode_opts = PDeflateOptions {
            gpu_decompress_enabled: true,
            gpu_decompress_force_gpu: true,
            chunk_size: 64 * 1024,
            gpu_slot_count: 6,
            gpu_submit_chunks: 4,
            ..PDeflateOptions::default()
        };
        let cases = [
            (30usize, 48 * 1024 + 11, 0x6000_0001u32),
            (10usize, 96 * 1024 + 19, 0x6000_0002u32),
            (40usize, 44 * 1024 + 7, 0x6000_0003u32),
            (20usize, 72 * 1024 + 5, 0x6000_0004u32),
        ];
        let mut payload_by_index = BTreeMap::<usize, Vec<u8>>::new();
        let mut expected_by_index = BTreeMap::<usize, Vec<u8>>::new();
        for (idx, len, seed) in cases {
            let input = gen_data_seeded(len, seed);
            let payload = compress_single_chunk_payload(&input, 64);
            payload_by_index.insert(idx, payload);
            expected_by_index.insert(idx, input);
        }
        let submit_order = [40usize, 10, 30, 20];
        let jobs: Vec<_> = submit_order
            .iter()
            .map(|idx| {
                make_gpu_decode_job(
                    *idx,
                    payload_by_index
                        .get(idx)
                        .expect("payload by chunk index")
                        .as_slice(),
                )
            })
            .collect();
        let results = gpu::decode_chunks_gpu_v2(&jobs, &decode_opts).expect("decode v2");
        assert_eq!(results.len(), submit_order.len());
        for w in results.windows(2) {
            assert!(
                w[0].chunk_index < w[1].chunk_index,
                "results must be sorted by chunk_index"
            );
        }
        for result in results {
            let expected = expected_by_index
                .get(&result.chunk_index)
                .expect("expected by chunk index");
            let actual = if let Some(bytes) = result.decoded_chunk {
                bytes
            } else {
                let payload = payload_by_index
                    .get(&result.chunk_index)
                    .expect("payload by chunk index");
                decode_chunk_payload_cpu(payload)
            };
            assert_eq!(actual, *expected, "decoded bytes mismatch");
        }
    }
}
