use std::cell::RefCell;
use std::collections::VecDeque;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use thiserror::Error;

mod gpu;

const STREAM_MAGIC: [u8; 4] = *b"PDS0";
const CHUNK_MAGIC: [u8; 4] = *b"PDF0";
const PDEFLATE_VERSION: u16 = 0;
const LITERAL_TAG: u16 = 0x0fff;
const MAX_TABLE_ID: usize = (LITERAL_TAG as usize) - 1;
const CHUNK_HEADER_SIZE: usize = 32;
const MAX_INLINE_LEN: usize = 14;
const EXT_LEN_BASE: usize = 15;
const EXT_LEN_MAX: usize = 255;
const MAX_CMD_LEN: usize = EXT_LEN_BASE + EXT_LEN_MAX;
const EMPTY_CANDIDATES: [usize; 0] = [];
const PREFIX_BUCKET_COUNT: usize = 256 * 256;
static PROFILE_DETAIL_GPU_PACK_SEQ: AtomicU64 = AtomicU64::new(0);
static PROFILE_DETAIL_GPU_BATCH_SEQ: AtomicU64 = AtomicU64::new(0);

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
}

#[derive(Debug, Default)]
struct CompressTaskQueueState {
    queue: VecDeque<usize>,
}

thread_local! {
    static ENCODE_SCRATCH: RefCell<EncodeScratch> = RefCell::new(EncodeScratch::new());
    static BUILD_SCRATCH: RefCell<BuildScratch> = RefCell::new(BuildScratch::default());
    static DECODE_SCRATCH: RefCell<DecodeScratch> = RefCell::new(DecodeScratch::default());
    static FINALIZE_SCRATCH: RefCell<FinalizeScratch> = RefCell::new(FinalizeScratch::default());
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
    pub gpu_compress_enabled: bool,
    pub gpu_workers: usize,
    pub gpu_submit_chunks: usize,
    pub gpu_min_chunk_size: usize,
}

impl Default for PDeflateOptions {
    fn default() -> Self {
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
            gpu_compress_enabled: false,
            gpu_workers: 1,
            gpu_submit_chunks: 4,
            gpu_min_chunk_size: 64 * 1024,
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

#[derive(Debug, Clone, Copy, Default)]
struct GpuBatchCompressProfile {
    chunks_total: usize,
    gpu_job_chunks: usize,
    gpu_used_chunks: usize,
    prepare_ms: f64,
    gpu_call_ms: f64,
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
    gpu_scratch_acquire_ms: f64,
    gpu_match_table_copy_ms: f64,
    gpu_match_prepare_dispatch_ms: f64,
    gpu_match_kernel_dispatch_ms: f64,
    gpu_match_submit_ms: f64,
    gpu_section_setup_ms: f64,
    gpu_section_pass_dispatch_ms: f64,
    gpu_section_copy_dispatch_ms: f64,
    gpu_section_submit_ms: f64,
    gpu_section_wrap_ms: f64,
    gpu_sparse_lens_submit_ms: f64,
    gpu_sparse_lens_submit_done_wait_ms: f64,
    gpu_sparse_lens_map_after_done_ms: f64,
    gpu_sparse_lens_poll_calls: u64,
    gpu_sparse_lens_yield_calls: u64,
    gpu_sparse_lens_wait_ms: f64,
    gpu_sparse_lens_copy_ms: f64,
    gpu_sparse_prepare_ms: f64,
    gpu_sparse_scratch_acquire_ms: f64,
    gpu_sparse_upload_dispatch_ms: f64,
    gpu_sparse_submit_ms: f64,
    gpu_sparse_wait_ms: f64,
    gpu_sparse_copy_ms: f64,
    gpu_sparse_total_ms: f64,
    finalize_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct HybridCompressSchedulerProfile {
    cpu_chunks: usize,
    gpu_batches: usize,
    gpu_chunks_claimed: usize,
    gpu_chunks_encoded: usize,
    gpu_job_chunks: usize,
    gpu_used_chunks: usize,
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
    gpu_scratch_acquire_ms: f64,
    gpu_match_table_copy_ms: f64,
    gpu_match_prepare_dispatch_ms: f64,
    gpu_match_kernel_dispatch_ms: f64,
    gpu_match_submit_ms: f64,
    gpu_section_setup_ms: f64,
    gpu_section_pass_dispatch_ms: f64,
    gpu_section_copy_dispatch_ms: f64,
    gpu_section_submit_ms: f64,
    gpu_section_wrap_ms: f64,
    gpu_sparse_lens_submit_ms: f64,
    gpu_sparse_lens_submit_done_wait_ms: f64,
    gpu_sparse_lens_map_after_done_ms: f64,
    gpu_sparse_lens_poll_calls: u64,
    gpu_sparse_lens_yield_calls: u64,
    gpu_sparse_lens_wait_ms: f64,
    gpu_sparse_lens_copy_ms: f64,
    gpu_sparse_prepare_ms: f64,
    gpu_sparse_scratch_acquire_ms: f64,
    gpu_sparse_upload_dispatch_ms: f64,
    gpu_sparse_submit_ms: f64,
    gpu_sparse_wait_ms: f64,
    gpu_sparse_copy_ms: f64,
    gpu_sparse_total_ms: f64,
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

pub fn pdeflate_decompress(stream: &[u8]) -> Result<Vec<u8>, PDeflateError> {
    pdeflate_decompress_with_stats(stream).map(|(out, _)| out)
}

pub fn pdeflate_decompress_into(stream: &[u8], output: &mut Vec<u8>) -> Result<(), PDeflateError> {
    pdeflate_decompress_into_with_stats(stream, output).map(|_| ())
}

pub fn pdeflate_gpu_init() -> bool {
    gpu::is_runtime_available()
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
    write_u16_le(&mut out, PDEFLATE_VERSION);
    write_u16_le(&mut out, 0);
    write_u32_le(
        &mut out,
        u32::try_from(chunk_size).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u64_le(
        &mut out,
        u64::try_from(input.len()).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u32_le(
        &mut out,
        u32::try_from(chunk_count).map_err(|_| PDeflateError::NumericOverflow)?,
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

    for (chunk_idx, compressed) in compressed_chunks.into_iter().enumerate() {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(input.len());
        let chunk_len = end.saturating_sub(start);

        write_u32_le(
            &mut out,
            u32::try_from(compressed.payload.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        );
        out.extend_from_slice(&compressed.payload);

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
    let stats = pdeflate_decompress_into_with_stats(stream, &mut output)?;
    Ok((output, stats))
}

pub fn pdeflate_decompress_into_with_stats(
    stream: &[u8],
    output: &mut Vec<u8>,
) -> Result<PDeflateStats, PDeflateError> {
    let mut cursor = 0usize;
    let magic = read_exact(stream, &mut cursor, 4)?;
    if magic != STREAM_MAGIC {
        return Err(PDeflateError::InvalidStream("bad stream magic"));
    }

    let version = read_u16_le(stream, &mut cursor)?;
    if version != PDEFLATE_VERSION {
        return Err(PDeflateError::InvalidStream("unsupported stream version"));
    }
    let _flags = read_u16_le(stream, &mut cursor)?;
    let _chunk_size = read_u32_le(stream, &mut cursor)?;
    let original_len = read_u64_le(stream, &mut cursor)? as usize;
    let chunk_count = read_u32_le(stream, &mut cursor)? as usize;

    // Decoder writes the full output span deterministically; avoid eager zero-fill
    // to reduce allocator/memory-bandwidth overhead on multi-threaded decode.
    output.clear();
    if output.capacity() < original_len {
        output.reserve_exact(original_len - output.capacity());
    }
    unsafe {
        output.set_len(original_len);
    }
    let mut stats = PDeflateStats {
        input_bytes: u64::try_from(stream.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        chunk_count,
        ..PDeflateStats::default()
    };
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
    let mut prof_chunk_decode_suppressed = 0usize;
    let mut out_cursor = 0usize;

    for chunk_idx in 0..chunk_count {
        let chunk_len = read_u32_le(stream, &mut cursor)? as usize;
        let chunk_payload = read_exact(stream, &mut cursor, chunk_len)?;
        let chunk_out_len = chunk_uncompressed_len(chunk_payload)?;
        let chunk_out_end = out_cursor
            .checked_add(chunk_out_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        if chunk_out_end > output.len() {
            return Err(PDeflateError::InvalidStream("decoded size overflow"));
        }
        let decoded = decompress_chunk_into(chunk_payload, &mut output[out_cursor..chunk_out_end])?;
        out_cursor = chunk_out_end;

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

        let emit_chunk_detail =
            profile_detail && profile_detail_should_log_indexed(chunk_idx, chunk_count);
        if profile_detail && !emit_chunk_detail {
            prof_chunk_decode_suppressed = prof_chunk_decode_suppressed.saturating_add(1);
        }
        if emit_chunk_detail {
            eprintln!(
                "[cozip_pdeflate][timing][chunk-decode] idx={} in_kib={:.2} out_kib={:.2} sections={} table_entries={} t_table_prepare_ms={:.3} t_section_decode_ms={:.3} cmds={} ref_cmds={} lit_cmds={} ref_bytes={} lit_bytes={} t_ref_copy_ms={:.3} t_lit_copy_ms={:.3}",
                chunk_idx,
                (chunk_payload.len() as f64) / 1024.0,
                (chunk_out_len as f64) / 1024.0,
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

    if profile_detail && prof_chunk_decode_suppressed > 0 {
        eprintln!(
            "[cozip_pdeflate][timing][detail-sampling] suppressed_chunk_decode_logs={} total_chunks={}",
            prof_chunk_decode_suppressed, chunk_count
        );
    }

    if out_cursor != original_len {
        return Err(PDeflateError::InvalidStream("decoded size mismatch"));
    }
    if cursor != stream.len() {
        return Err(PDeflateError::InvalidStream("trailing bytes in stream"));
    }

    stats.output_bytes = u64::try_from(output.len()).map_err(|_| PDeflateError::NumericOverflow)?;
    if profile {
        eprintln!(
            "[cozip_pdeflate][timing][decompress-total] chunks={} in_mib={:.2} out_mib={:.2} t_table_prepare_ms={:.3} t_section_decode_ms={:.3} cmds={} ref_cmds={} lit_cmds={} ref_bytes={} lit_bytes={} t_ref_copy_ms={:.3} t_lit_copy_ms={:.3}",
            chunk_count,
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
    if options.gpu_submit_chunks == 0 {
        return Err(PDeflateError::InvalidOptions(
            "gpu_submit_chunks must be > 0",
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
    Ok(())
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
    table_count: usize,
    section_count: usize,
    table_index: &[u8],
    table_data: &[u8],
    section_index: &[u8],
    section_cmd: &[u8],
) -> Result<Vec<u8>, PDeflateError> {
    let table_index_offset = CHUNK_HEADER_SIZE;
    let table_data_offset = table_index_offset
        .checked_add(table_index.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let section_index_offset = table_data_offset
        .checked_add(table_data.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let section_cmd_offset = section_index_offset
        .checked_add(section_index.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let total_len = section_cmd_offset
        .checked_add(section_cmd.len())
        .ok_or(PDeflateError::NumericOverflow)?;

    let mut payload = Vec::with_capacity(total_len);
    payload.extend_from_slice(&CHUNK_MAGIC);
    write_u16_le(&mut payload, PDEFLATE_VERSION);
    write_u16_le(&mut payload, 0);
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
        u32::try_from(section_index_offset).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u32_le(
        &mut payload,
        u32::try_from(section_cmd_offset).map_err(|_| PDeflateError::NumericOverflow)?,
    );

    if payload.len() != CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream("internal header sizing bug"));
    }

    payload.extend_from_slice(table_index);
    payload.extend_from_slice(table_data);
    payload.extend_from_slice(section_index);
    payload.extend_from_slice(section_cmd);
    Ok(payload)
}

fn pack_chunk_payload(
    chunk_len: usize,
    table_count: usize,
    section_count: usize,
    table_index: &[u8],
    table_data: &[u8],
    section_index: &[u8],
    section_cmd: &[u8],
    _prefer_gpu: bool,
) -> Result<(Vec<u8>, f64), PDeflateError> {
    let t0 = Instant::now();
    let payload = pack_chunk_payload_cpu(
        chunk_len,
        table_count,
        section_count,
        table_index,
        table_data,
        section_index,
        section_cmd,
    )?;
    Ok((payload, elapsed_ms(t0)))
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
            for &len in &section_cmd_lens {
                write_varint_u32(&mut scratch.section_index, len);
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
            pack_chunk_payload(
                chunk.len(),
                table_index.len(),
                section_count,
                table_index,
                table_data,
                &scratch.section_index,
                &section_cmd,
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
    let mut prepared = Vec::<GpuPreparedChunk<'_>>::with_capacity(indices.len());
    for &chunk_idx in indices {
        let start = chunk_idx
            .checked_mul(chunk_size)
            .ok_or(PDeflateError::NumericOverflow)?;
        let end = (start + chunk_size).min(input.len());
        let chunk = &input[start..end];
        // Batch hybrid path currently performs better by keeping table build on CPU.
        // GPU table build has high fixed per-chunk overhead at 4MiB chunk sizes.
        let built = build_table_dispatch(chunk, options, false)?;
        let max_ref_len = options.max_ref_len.min(MAX_CMD_LEN).min(64);
        let gpu_eligible = max_ref_len >= options.min_ref_len
            && (built.table_count > 0 || built.gpu_table.is_some());
        prepared.push(GpuPreparedChunk {
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
    let prepare_ms = elapsed_ms(t_prepare);

    let mut prof_by_chunk = vec![gpu::GpuMatchProfile::default(); prepared.len()];
    let mut packed_matches_by_chunk = std::iter::repeat_with(|| None::<Vec<u32>>)
        .take(prepared.len())
        .collect::<Vec<_>>();
    let mut gpu_encoded_chunks = std::iter::repeat_with(|| None::<ChunkCompressed>)
        .take(prepared.len())
        .collect::<Vec<_>>();
    let mut gpu_used_chunks = 0usize;
    let gpu_job_indices: Vec<usize> = prepared
        .iter()
        .enumerate()
        .filter_map(|(idx, chunk)| chunk.gpu_eligible.then_some(idx))
        .collect();
    let mut fallback_match_job_indices = gpu_job_indices.clone();
    let mut gpu_match_upload_ms = 0.0_f64;
    let mut gpu_match_wait_ms = 0.0_f64;
    let mut gpu_match_map_copy_ms = 0.0_f64;
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
    let mut gpu_sparse_scratch_acquire_ms = 0.0_f64;
    let mut gpu_sparse_upload_dispatch_ms = 0.0_f64;
    let mut gpu_sparse_submit_ms = 0.0_f64;
    let mut gpu_sparse_wait_ms = 0.0_f64;
    let mut gpu_sparse_copy_ms = 0.0_f64;
    let mut gpu_sparse_total_ms = 0.0_f64;
    let mut kernel_profile = gpu::GpuBatchKernelProfile::default();

    let t_gpu_call = Instant::now();
    // Keep a single GPU submit stream (cozip_deflate-aligned behavior) to avoid
    // queue/poll contention when multiple hybrid GPU workers are active.
    let _gpu_submit_stream_guard = if !gpu_job_indices.is_empty() {
        Some(gpu::lock_submit_stream()?)
    } else {
        None
    };
    if !gpu_job_indices.is_empty() {
        let gpu_inputs: Vec<gpu::GpuMatchInput<'_>> = gpu_job_indices
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
                }
            })
            .collect();

        let gpu_max_cmd_len = options.max_ref_len.min(MAX_CMD_LEN).min(64);
        match gpu::compute_matches_and_encode_sections_batch(
            &gpu_inputs,
            options.section_count,
            gpu_max_cmd_len,
        ) {
            Ok(batch) => {
                gpu_match_upload_ms += batch.match_profile.upload_ms;
                gpu_match_wait_ms += batch.match_profile.wait_ms;
                gpu_match_map_copy_ms += batch.match_profile.map_copy_ms;
                gpu_section_encode_upload_ms += batch.section_profile.upload_ms;
                gpu_section_encode_wait_ms += batch.section_profile.wait_ms;
                gpu_section_encode_map_copy_ms += batch.section_profile.map_copy_ms;
                gpu_section_encode_total_ms += batch.section_profile.total_ms;
                kernel_profile = batch.kernel_profile;
                if batch.chunks.len() == gpu_job_indices.len() {
                    fallback_match_job_indices.clear();
                    let total_gpu_bytes: usize = gpu_job_indices
                        .iter()
                        .map(|&idx| prepared[idx].chunk.len())
                        .sum();
                    enum PendingTableRef {
                        Prepared,
                        Uploaded(usize),
                    }
                    struct PendingSparsePack {
                        prep_idx: usize,
                        section_device: gpu::GpuSectionCmdDeviceOutput,
                        table_ref: PendingTableRef,
                        section_scaled: gpu::GpuSectionEncodeProfile,
                        match_scaled: gpu::GpuMatchProfile,
                    }
                    let mut uploaded_tables = Vec::<gpu::GpuPackedTableDevice>::new();
                    let mut pending = Vec::<PendingSparsePack>::new();
                    for (out_idx, chunk_out) in batch.chunks.into_iter().enumerate() {
                        let prep_idx = gpu_job_indices[out_idx];
                        let Some(section_device) = chunk_out.section_cmd_device else {
                            if profile_detail {
                                eprintln!(
                                    "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason=missing_section_device_output",
                                    indices.len(),
                                    gpu_job_indices.len(),
                                );
                            }
                            fallback_match_job_indices.push(prep_idx);
                            continue;
                        };
                        let scale = if total_gpu_bytes == 0 {
                            0.0
                        } else {
                            (prepared[prep_idx].chunk.len() as f64) / (total_gpu_bytes as f64)
                        };
                        let section_scaled =
                            scale_gpu_section_profile(batch.section_profile, scale);
                        let match_scaled = scale_gpu_profile(batch.match_profile, scale);
                        let table_ref = if prepared[prep_idx].gpu_table.is_some() {
                            PendingTableRef::Prepared
                        } else {
                            match gpu::upload_packed_table_device(
                                prepared[prep_idx].table_count,
                                &prepared[prep_idx].table_index,
                                &prepared[prep_idx].table_data,
                            ) {
                                Ok(table_gpu) => {
                                    uploaded_tables.push(table_gpu);
                                    PendingTableRef::Uploaded(uploaded_tables.len() - 1)
                                }
                                Err(err) => {
                                    if profile_detail {
                                        eprintln!(
                                            "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason=table_upload_failed err={}",
                                            indices.len(),
                                            gpu_job_indices.len(),
                                            err
                                        );
                                    }
                                    fallback_match_job_indices.push(prep_idx);
                                    continue;
                                }
                            }
                        };
                        pending.push(PendingSparsePack {
                            prep_idx,
                            section_device,
                            table_ref,
                            section_scaled,
                            match_scaled,
                        });
                    }
                    if !pending.is_empty() {
                        let pack_inputs: Vec<gpu::GpuSparsePackInput<'_>> = pending
                            .iter()
                            .map(|p| {
                                let table_ref = match p.table_ref {
                                    PendingTableRef::Prepared => prepared[p.prep_idx]
                                        .gpu_table
                                        .as_ref()
                                        .expect("prepared table missing"),
                                    PendingTableRef::Uploaded(i) => &uploaded_tables[i],
                                };
                                gpu::GpuSparsePackInput {
                                    chunk_len: prepared[p.prep_idx].chunk.len(),
                                    section_count: options.section_count,
                                    table: table_ref,
                                    section: &p.section_device,
                                }
                            })
                            .collect();
                        match gpu::read_section_commands_from_device_batch(&pack_inputs) {
                            Ok((host_sections, pack_profile, pack_detail)) => {
                                let emit_pack_batch_detail = if profile_detail {
                                    let seq =
                                        PROFILE_DETAIL_GPU_PACK_SEQ.fetch_add(1, Ordering::Relaxed);
                                    profile_detail_should_log_stream(seq)
                                } else {
                                    false
                                };
                                if emit_pack_batch_detail {
                                    eprintln!(
                                        "[cozip_pdeflate][timing][gpu-pack-batch] chunks={} lens_kib={:.2} out_cmd_mib={:.2} t_lens_submit_ms={:.3} t_lens_submit_done_wait_ms={:.3} t_lens_map_after_done_ms={:.3} lens_poll_calls={} lens_yield_calls={} t_lens_wait_ms={:.3} t_lens_copy_ms={:.3} t_sparse_prepare_ms={:.3} t_sparse_upload_dispatch_ms={:.3} t_sparse_submit_ms={:.3} t_sparse_wait_ms={:.3} t_sparse_copy_ms={:.3} t_sparse_total_ms={:.3}",
                                        pack_detail.chunks,
                                        (pack_detail.lens_bytes_total as f64) / 1024.0,
                                        (pack_detail.out_cmd_bytes_total as f64)
                                            / (1024.0 * 1024.0),
                                        pack_detail.lens_submit_ms,
                                        pack_detail.lens_submit_done_wait_ms,
                                        pack_detail.lens_map_after_done_ms,
                                        pack_detail.lens_poll_calls,
                                        pack_detail.lens_yield_calls,
                                        pack_detail.lens_wait_ms,
                                        pack_detail.lens_copy_ms,
                                        pack_detail.prepare_ms,
                                        pack_detail.upload_dispatch_ms,
                                        pack_detail.submit_ms,
                                        pack_detail.wait_ms,
                                        pack_detail.copy_ms,
                                        pack_detail.total_ms
                                    );
                                }
                                if host_sections.len() != pending.len() {
                                    if profile_detail {
                                        eprintln!(
                                            "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason=sparse_readback_output_count_mismatch out={} jobs={}",
                                            indices.len(),
                                            gpu_job_indices.len(),
                                            host_sections.len(),
                                            pending.len()
                                        );
                                    }
                                    for p in pending {
                                        fallback_match_job_indices.push(p.prep_idx);
                                    }
                                } else {
                                    gpu_sparse_pack_upload_ms += pack_profile.upload_ms;
                                    gpu_sparse_pack_wait_ms += pack_profile.wait_ms;
                                    gpu_sparse_pack_map_copy_ms += pack_profile.map_copy_ms;
                                    gpu_sparse_lens_submit_ms += pack_detail.lens_submit_ms;
                                    gpu_sparse_lens_submit_done_wait_ms +=
                                        pack_detail.lens_submit_done_wait_ms;
                                    gpu_sparse_lens_map_after_done_ms +=
                                        pack_detail.lens_map_after_done_ms;
                                    gpu_sparse_lens_poll_calls = gpu_sparse_lens_poll_calls
                                        .saturating_add(pack_detail.lens_poll_calls);
                                    gpu_sparse_lens_yield_calls = gpu_sparse_lens_yield_calls
                                        .saturating_add(pack_detail.lens_yield_calls);
                                    gpu_sparse_lens_wait_ms += pack_detail.lens_wait_ms;
                                    gpu_sparse_lens_copy_ms += pack_detail.lens_copy_ms;
                                    gpu_sparse_prepare_ms += pack_detail.prepare_ms;
                                    gpu_sparse_scratch_acquire_ms += pack_detail.scratch_acquire_ms;
                                    gpu_sparse_upload_dispatch_ms += pack_detail.upload_dispatch_ms;
                                    gpu_sparse_submit_ms += pack_detail.submit_ms;
                                    gpu_sparse_wait_ms += pack_detail.wait_ms;
                                    gpu_sparse_copy_ms += pack_detail.copy_ms;
                                    gpu_sparse_total_ms += pack_detail.total_ms;
                                    kernel_profile.sparse_lens_submit_ms +=
                                        pack_detail.lens_submit_ms;
                                    kernel_profile.sparse_lens_submit_done_wait_ms +=
                                        pack_detail.lens_submit_done_wait_ms;
                                    kernel_profile.sparse_lens_map_after_done_ms +=
                                        pack_detail.lens_map_after_done_ms;
                                    kernel_profile.sparse_lens_poll_calls = kernel_profile
                                        .sparse_lens_poll_calls
                                        .saturating_add(pack_detail.lens_poll_calls);
                                    kernel_profile.sparse_lens_yield_calls = kernel_profile
                                        .sparse_lens_yield_calls
                                        .saturating_add(pack_detail.lens_yield_calls);
                                    kernel_profile.sparse_lens_wait_ms += pack_detail.lens_wait_ms;
                                    kernel_profile.sparse_lens_copy_ms += pack_detail.lens_copy_ms;
                                    kernel_profile.sparse_prepare_ms += pack_detail.prepare_ms;
                                    kernel_profile.sparse_scratch_acquire_ms +=
                                        pack_detail.scratch_acquire_ms;
                                    kernel_profile.sparse_upload_dispatch_ms +=
                                        pack_detail.upload_dispatch_ms;
                                    kernel_profile.sparse_submit_ms += pack_detail.submit_ms;
                                    kernel_profile.sparse_wait_ms += pack_detail.wait_ms;
                                    kernel_profile.sparse_copy_ms += pack_detail.copy_ms;
                                    kernel_profile.sparse_total_ms += pack_detail.total_ms;
                                    let total_pack_bytes: usize = pending
                                        .iter()
                                        .map(|p| prepared[p.prep_idx].chunk.len())
                                        .sum();
                                    for (p, host_chunk) in
                                        pending.into_iter().zip(host_sections.into_iter())
                                    {
                                        let pack_scale = if total_pack_bytes == 0 {
                                            0.0
                                        } else {
                                            (prepared[p.prep_idx].chunk.len() as f64)
                                                / (total_pack_bytes as f64)
                                        };
                                        let pack_scaled =
                                            scale_gpu_profile(pack_profile, pack_scale);
                                        let gpu_profile = gpu::GpuMatchProfile {
                                            upload_ms: p.match_scaled.upload_ms
                                                + p.section_scaled.upload_ms
                                                + pack_scaled.upload_ms,
                                            wait_ms: p.match_scaled.wait_ms
                                                + p.section_scaled.wait_ms
                                                + pack_scaled.wait_ms,
                                            map_copy_ms: p.match_scaled.map_copy_ms
                                                + p.section_scaled.map_copy_ms
                                                + pack_scaled.map_copy_ms,
                                            total_ms: p.match_scaled.total_ms
                                                + p.section_scaled.total_ms
                                                + pack_scaled.total_ms,
                                        };
                                        let t_pack_header = Instant::now();
                                        let mut section_index = Vec::with_capacity(
                                            host_chunk.section_cmd_lens.len().saturating_mul(2),
                                        );
                                        for &len in &host_chunk.section_cmd_lens {
                                            write_varint_u32(&mut section_index, len);
                                        }
                                        let payload = match pack_chunk_payload_cpu(
                                            prepared[p.prep_idx].chunk.len(),
                                            prepared[p.prep_idx].table_count,
                                            options.section_count,
                                            &prepared[p.prep_idx].table_index,
                                            &prepared[p.prep_idx].table_data,
                                            &section_index,
                                            &host_chunk.section_cmd,
                                        ) {
                                            Ok(payload) => payload,
                                            Err(err) => {
                                                if profile_detail {
                                                    eprintln!(
                                                        "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason=host_pack_failed err={}",
                                                        indices.len(),
                                                        gpu_job_indices.len(),
                                                        err
                                                    );
                                                }
                                                fallback_match_job_indices.push(p.prep_idx);
                                                continue;
                                            }
                                        };
                                        let header_pack_ms = elapsed_ms(t_pack_header);
                                        gpu_encoded_chunks[p.prep_idx] = Some(ChunkCompressed {
                                            payload,
                                            table_entries: prepared[p.prep_idx].table_count,
                                            section_count: options.section_count,
                                            profile: ChunkEncodeProfile {
                                                table_build_ms: prepared[p.prep_idx].table_build_ms,
                                                table_freq_ms: prepared[p.prep_idx]
                                                    .table_profile
                                                    .freq_ms,
                                                table_probe_ms: prepared[p.prep_idx]
                                                    .table_profile
                                                    .probe_ms,
                                                table_materialize_ms: prepared[p.prep_idx]
                                                    .table_profile
                                                    .materialize_ms,
                                                table_sort_ms: prepared[p.prep_idx]
                                                    .table_profile
                                                    .sort_ms,
                                                table_positions_scanned: prepared[p.prep_idx]
                                                    .table_profile
                                                    .positions_scanned,
                                                table_probe_pairs: prepared[p.prep_idx]
                                                    .table_profile
                                                    .probe_pairs,
                                                table_match_len_calls: prepared[p.prep_idx]
                                                    .table_profile
                                                    .match_len_calls,
                                                table_hash_key_mismatch: prepared[p.prep_idx]
                                                    .table_profile
                                                    .hash_key_mismatch,
                                                anchored_short_hits: prepared[p.prep_idx]
                                                    .table_profile
                                                    .anchored_short_hits,
                                                section_encode_ms: p.section_scaled.total_ms,
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
                                                header_pack_ms,
                                                total_ms: prepared[p.prep_idx].table_build_ms
                                                    + gpu_profile.total_ms
                                                    + header_pack_ms,
                                            },
                                        });
                                        gpu_used_chunks = gpu_used_chunks.saturating_add(1);
                                    }
                                }
                            }
                            Err(err) => {
                                if profile_detail {
                                    eprintln!(
                                        "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason=sparse_readback_failed err={}",
                                        indices.len(),
                                        gpu_job_indices.len(),
                                        err
                                    );
                                }
                                for p in pending {
                                    fallback_match_job_indices.push(p.prep_idx);
                                }
                            }
                        }
                    }
                } else {
                    if profile_detail {
                        eprintln!(
                            "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason=output_count_mismatch out={} jobs={}",
                            indices.len(),
                            gpu_job_indices.len(),
                            batch.chunks.len(),
                            gpu_job_indices.len()
                        );
                    }
                }
            }
            Err(err) => {
                if profile_detail {
                    eprintln!(
                        "[cozip_pdeflate][timing][gpu-batch-fallback] chunks={} gpu_jobs={} reason={}",
                        indices.len(),
                        gpu_job_indices.len(),
                        err
                    );
                }
            }
        }

        if !fallback_match_job_indices.is_empty() {
            fallback_match_job_indices.sort_unstable();
            fallback_match_job_indices.dedup();
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
                    }
                })
                .collect();
            match gpu::compute_matches_batch(&fallback_inputs) {
                Ok(batch) => {
                    gpu_match_upload_ms += batch.profile.upload_ms;
                    gpu_match_wait_ms += batch.profile.wait_ms;
                    gpu_match_map_copy_ms += batch.profile.map_copy_ms;
                    if batch.chunks.len() == fallback_match_job_indices.len() {
                        let total_gpu_bytes: usize = fallback_match_job_indices
                            .iter()
                            .map(|&idx| prepared[idx].chunk.len())
                            .sum();
                        for (out_idx, chunk_out) in batch.chunks.into_iter().enumerate() {
                            let prep_idx = fallback_match_job_indices[out_idx];
                            if chunk_out.packed_matches.len() != prepared[prep_idx].chunk.len() {
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
        if p.table.is_none() {
            if p.table_index.is_empty() {
                let (cpu_table, cpu_index, cpu_data) =
                    rebuild_table_cpu_for_fallback(p.chunk, options)?;
                p.table_count = cpu_table.len();
                p.table_index = cpu_index;
                p.table_data = cpu_data;
                p.table = Some(cpu_table);
            } else if p.table_count > 0 {
                p.table = Some(materialize_table_entries(&p.table_index, &p.table_data)?);
            }
        }
        let packed_matches = packed_matches_by_chunk[idx].as_deref();
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
    let emit_gpu_batch_detail = if profile_detail {
        let seq = PROFILE_DETAIL_GPU_BATCH_SEQ.fetch_add(1, Ordering::Relaxed);
        profile_detail_should_log_stream(seq)
    } else {
        false
    };
    if emit_gpu_batch_detail {
        let chunks_f = (indices.len() as f64).max(1.0);
        eprintln!(
            "[cozip_pdeflate][timing][gpu-batch] batch_chunks={} gpu_jobs={} gpu_used={} t_prepare_ms={:.3} t_gpu_call_ms={:.3} t_finalize_ms={:.3} t_pack_inputs_ms={:.3} t_section_setup_ms={:.3} t_sparse_lens_submit_done_wait_ms={:.3} t_sparse_lens_map_after_done_ms={:.3} sparse_lens_poll_calls={} sparse_lens_yield_calls={} t_sparse_lens_wait_ms={:.3} t_sparse_wait_ms={:.3} t_sparse_copy_ms={:.3} t_upload_ms={:.3} t_wait_ms={:.3} t_map_copy_ms={:.3} gpu_call_ms_per_chunk={:.3}",
            indices.len(),
            gpu_job_indices.len(),
            gpu_used_chunks,
            prepare_ms,
            gpu_call_ms,
            finalize_ms,
            kernel_profile.pack_inputs_ms,
            kernel_profile.section_setup_ms,
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
            prepare_ms,
            gpu_call_ms,
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
            gpu_scratch_acquire_ms: kernel_profile.scratch_acquire_ms,
            gpu_match_table_copy_ms: kernel_profile.match_table_copy_ms,
            gpu_match_prepare_dispatch_ms: kernel_profile.match_prepare_dispatch_ms,
            gpu_match_kernel_dispatch_ms: kernel_profile.match_kernel_dispatch_ms,
            gpu_match_submit_ms: kernel_profile.match_submit_ms,
            gpu_section_setup_ms: kernel_profile.section_setup_ms,
            gpu_section_pass_dispatch_ms: kernel_profile.section_pass_dispatch_ms,
            gpu_section_copy_dispatch_ms: kernel_profile.section_copy_dispatch_ms,
            gpu_section_submit_ms: kernel_profile.section_submit_ms,
            gpu_section_wrap_ms: kernel_profile.section_wrap_ms,
            gpu_sparse_lens_submit_ms,
            gpu_sparse_lens_submit_done_wait_ms,
            gpu_sparse_lens_map_after_done_ms,
            gpu_sparse_lens_poll_calls,
            gpu_sparse_lens_yield_calls,
            gpu_sparse_lens_wait_ms,
            gpu_sparse_lens_copy_ms,
            gpu_sparse_prepare_ms,
            gpu_sparse_scratch_acquire_ms,
            gpu_sparse_upload_dispatch_ms,
            gpu_sparse_submit_ms,
            gpu_sparse_wait_ms,
            gpu_sparse_copy_ms,
            gpu_sparse_total_ms,
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
    for i in 0..chunk_count {
        initial_queue.push_back(i);
    }
    let queue_state = Arc::new(Mutex::new(CompressTaskQueueState {
        queue: initial_queue,
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
    let gpu_batch_limit = if gpu_enabled {
        let requested = options.gpu_submit_chunks.max(1);
        match gpu::max_safe_match_batch_chunks(chunk_size) {
            Ok(limit) => requested.min(limit.max(1)),
            Err(_) => requested,
        }
    } else {
        options.gpu_submit_chunks.max(1)
    };
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
                        queue.queue.pop_front()
                    };
                    if prof_enabled {
                        if let Ok(mut prof) = sched_prof.lock() {
                            prof.cpu_lock_wait_ms += elapsed_ms(t_lock);
                        }
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
                                        prof.gpu_section_copy_dispatch_ms +=
                                            batch_prof.gpu_section_copy_dispatch_ms;
                                        prof.gpu_section_submit_ms +=
                                            batch_prof.gpu_section_submit_ms;
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
                                        prof.gpu_sparse_scratch_acquire_ms +=
                                            batch_prof.gpu_sparse_scratch_acquire_ms;
                                        prof.gpu_sparse_upload_dispatch_ms +=
                                            batch_prof.gpu_sparse_upload_dispatch_ms;
                                        prof.gpu_sparse_submit_ms +=
                                            batch_prof.gpu_sparse_submit_ms;
                                        prof.gpu_sparse_wait_ms += batch_prof.gpu_sparse_wait_ms;
                                        prof.gpu_sparse_copy_ms += batch_prof.gpu_sparse_copy_ms;
                                        prof.gpu_sparse_total_ms += batch_prof.gpu_sparse_total_ms;
                                        prof.gpu_finalize_ms += batch_prof.finalize_ms;
                                        prof.gpu_batch_total_ms += batch_prof.total_ms;
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
                "[cozip_pdeflate][timing][hybrid-scheduler] chunks={} cpu_workers={} gpu_workers={} gpu_enabled={} gpu_batch_limit={} cpu_chunks={} cpu_encode_ms={:.3} cpu_chunk_avg_ms={:.3} cpu_result_lock_wait_ms={:.3} gpu_batches={} gpu_assigned_chunks={} gpu_chunks_encoded={} gpu_job_chunks={} gpu_used_chunks={} gpu_batch_avg_chunks={:.2} gpu_worker_busy_ms={:.3} gpu_chunk_avg_ms_assigned={:.3} gpu_chunk_avg_ms_encoded={:.3} gpu_prepare_ms={:.3} gpu_call_ms={:.3} gpu_finalize_ms={:.3} gpu_prepare_pct={:.1} gpu_call_pct={:.1} gpu_finalize_pct={:.1} gpu_upload_ms={:.3} gpu_wait_ms={:.3} gpu_map_copy_ms={:.3} gpu_match_upload_ms={:.3} gpu_match_wait_ms={:.3} gpu_match_map_copy_ms={:.3} gpu_sparse_pack_upload_ms={:.3} gpu_sparse_pack_wait_ms={:.3} gpu_sparse_pack_map_copy_ms={:.3} gpu_section_encode_upload_ms={:.3} gpu_section_encode_wait_ms={:.3} gpu_section_encode_map_copy_ms={:.3} gpu_section_encode_total_ms={:.3} gpu_batch_total_ms={:.3} gpu_result_lock_wait_ms={:.3} cpu_lock_wait_ms={:.3} gpu_lock_wait_ms={:.3} cpu_no_task_events={} gpu_no_task_events={}",
                chunk_count,
                cpu_workers,
                gpu_workers,
                gpu_enabled,
                gpu_batch_limit,
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
            eprintln!(
                "[cozip_pdeflate][timing][hybrid-scheduler-kernel] gpu_batch_total_ms={:.3} kernel_accounted_ms={:.3} gpu_call_unaccounted_ms={:.3} gpu_batch_unaccounted_ms={:.3} match_stage_ms={:.3} section_stage_ms={:.3} sparse_stage_ms={:.3} match_stage_pct={:.1} section_stage_pct={:.1} sparse_stage_pct={:.1} pack_inputs_ms={:.3} scratch_acquire_ms={:.3} match_table_copy_ms={:.3} match_prepare_dispatch_ms={:.3} match_kernel_dispatch_ms={:.3} match_submit_ms={:.3} section_setup_ms={:.3} section_pass_dispatch_ms={:.3} section_copy_dispatch_ms={:.3} section_submit_ms={:.3} section_wrap_ms={:.3} sparse_lens_submit_ms={:.3} sparse_lens_submit_done_wait_ms={:.3} sparse_lens_map_after_done_ms={:.3} sparse_lens_poll_calls={} sparse_lens_yield_calls={} sparse_lens_wait_ms={:.3} sparse_lens_copy_ms={:.3} sparse_prepare_ms={:.3} sparse_scratch_acquire_ms={:.3} sparse_upload_dispatch_ms={:.3} sparse_submit_ms={:.3} sparse_wait_ms={:.3} sparse_copy_ms={:.3} sparse_total_ms={:.3}",
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
                prof.gpu_section_copy_dispatch_ms,
                prof.gpu_section_submit_ms,
                prof.gpu_section_wrap_ms,
                prof.gpu_sparse_lens_submit_ms,
                prof.gpu_sparse_lens_submit_done_wait_ms,
                prof.gpu_sparse_lens_map_after_done_ms,
                prof.gpu_sparse_lens_poll_calls,
                prof.gpu_sparse_lens_yield_calls,
                prof.gpu_sparse_lens_wait_ms,
                prof.gpu_sparse_lens_copy_ms,
                prof.gpu_sparse_prepare_ms,
                prof.gpu_sparse_scratch_acquire_ms,
                prof.gpu_sparse_upload_dispatch_ms,
                prof.gpu_sparse_submit_ms,
                prof.gpu_sparse_wait_ms,
                prof.gpu_sparse_copy_ms,
                prof.gpu_sparse_total_ms
            );
        }
    }
    Ok(out)
}

fn compress_chunk(
    chunk: &[u8],
    options: &PDeflateOptions,
) -> Result<ChunkCompressed, PDeflateError> {
    let built = build_table_dispatch(chunk, options, false)?;
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

fn chunk_uncompressed_len(payload: &[u8]) -> Result<usize, PDeflateError> {
    if payload.len() < 12 {
        return Err(PDeflateError::InvalidStream("chunk header truncated"));
    }
    if payload.get(0..4) != Some(&CHUNK_MAGIC) {
        return Err(PDeflateError::InvalidStream("bad chunk magic"));
    }
    let version = u16::from_le_bytes([payload[4], payload[5]]);
    if version != PDEFLATE_VERSION {
        return Err(PDeflateError::InvalidStream("unsupported chunk version"));
    }
    Ok(u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]) as usize)
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
    if version != PDEFLATE_VERSION {
        return Err(PDeflateError::InvalidStream("unsupported chunk version"));
    }

    let _flags = read_u16_le(payload, &mut cursor)?;
    let chunk_uncompressed_len = read_u32_le(payload, &mut cursor)? as usize;
    let table_count = read_u16_le(payload, &mut cursor)? as usize;
    let section_count = read_u16_le(payload, &mut cursor)? as usize;
    let table_index_offset = read_u32_le(payload, &mut cursor)? as usize;
    let table_data_offset = read_u32_le(payload, &mut cursor)? as usize;
    let section_index_offset = read_u32_le(payload, &mut cursor)? as usize;
    let section_cmd_offset = read_u32_le(payload, &mut cursor)? as usize;

    if table_count > MAX_TABLE_ID {
        return Err(PDeflateError::InvalidStream("table_count too large"));
    }
    if section_count == 0 {
        return Err(PDeflateError::InvalidStream("section_count is zero"));
    }

    if !(table_index_offset <= table_data_offset
        && table_data_offset <= section_index_offset
        && section_index_offset <= section_cmd_offset
        && section_cmd_offset <= payload.len())
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
    let table_data = &payload[table_data_offset..section_index_offset];
    let section_index = &payload[section_index_offset..section_cmd_offset];
    let section_cmd = &payload[section_cmd_offset..];
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

        let sections_t0 = Instant::now();
        let mut section_idx_cursor = 0usize;
        let mut cmd_cursor = 0usize;
        for sec in 0..section_count {
            let sec_cmd_len = read_varint_u32(section_index, &mut section_idx_cursor)? as usize;
            let sec_cmd_end = cmd_cursor
                .checked_add(sec_cmd_len)
                .ok_or(PDeflateError::NumericOverflow)?;
            let sec_cmd =
                section_cmd
                    .get(cmd_cursor..sec_cmd_end)
                    .ok_or(PDeflateError::InvalidStream(
                        "section cmd range out of bounds",
                    ))?;
            let out_start = section_start(sec, section_count, chunk_uncompressed_len);
            let out_end = section_start(sec + 1, section_count, chunk_uncompressed_len);
            let out_len = out_end - out_start;

            decode_section(
                sec_cmd,
                out_len,
                &scratch.table_repeat,
                &mut out[out_start..out_end],
                &mut decode_profile,
                profile,
            )?;
            cmd_cursor = sec_cmd_end;
        }
        if section_idx_cursor != section_index.len() {
            return Err(PDeflateError::InvalidStream(
                "trailing bytes in section index",
            ));
        }
        if cmd_cursor != section_cmd.len() {
            return Err(PDeflateError::InvalidStream(
                "sum(section_cmd_len) != section_cmd size",
            ));
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
        if let Ok((gpu_table, gpu_prof)) = gpu::build_table_gpu_device(
            chunk,
            max_entries,
            options.max_table_entry_len,
            options.min_ref_len,
            options.match_probe_limit,
            options.hash_history_limit,
            options.table_sample_stride,
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

    let max_entry_len = options.max_table_entry_len.min(254);
    let max_entries = options.max_table_entries.min(MAX_TABLE_ID);
    // Speed-biased sampling: keep external option semantics while reducing
    // probe volume for large chunks.
    let sample_stride = options.table_sample_stride.max(1).saturating_mul(8);
    // Table seeding is the dominant cost path. Keep probe fan-out tighter here.
    let probe_limit = options.match_probe_limit.min(1);
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
    let history_cap = options.hash_history_limit.max(1);
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
    let probe_limit = options.match_probe_limit.min(1);

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
    let probe_limit = options.match_probe_limit.min(1);
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
    let num = (index as u128) * (total_len as u128);
    let den = section_count as u128;
    (num / den) as usize
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
}
