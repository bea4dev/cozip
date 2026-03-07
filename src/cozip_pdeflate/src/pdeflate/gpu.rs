use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::shaders::*;
use super::{profile_enabled, PDeflateError, PDeflateOptions};

const WORKGROUP_SIZE: u32 = 128;
const PREFIX2_TABLE_SIZE: usize = 1 << 16;
const BUILD_TABLE_SORT_BUCKETS: usize = 1 << 16;
const PACK_CHUNK_HEADER_SIZE: usize = 36;
const PACK_CHUNK_MAGIC: [u8; 4] = *b"PDF0";
const PACK_CHUNK_VERSION: u16 = 0;
const BUILD_TABLE_STAGE_COUNT: usize = 7;
const BUILD_TABLE_STAGE_QUERY_COUNT: u32 = (BUILD_TABLE_STAGE_COUNT as u32) * 2;
const SPARSE_KERNEL_STAGE_COUNT: usize = 2;
const SPARSE_KERNEL_STAGE_QUERY_COUNT: u32 = (SPARSE_KERNEL_STAGE_COUNT as u32) * 2;
const GPU_DECODE_TABLE_REPEAT_STRIDE: usize = 270;
const GPU_DECODE_SECTION_META_HEADER_WORDS: usize = 6;
const GPU_DECODE_SECTION_META_WORDS: usize = 4;
const GPU_DECODE_MAX_TABLE_ID: usize = 0x0fff - 1;
const GPU_DECODE_V2_WORKGROUP_SIZE: u32 = 64;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuMatchProfile {
    pub(crate) upload_ms: f64,
    pub(crate) wait_ms: f64,
    pub(crate) map_copy_ms: f64,
    pub(crate) total_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuBatchKernelProfile {
    pub(crate) pack_inputs_ms: f64,
    pub(crate) pack_alloc_setup_ms: f64,
    pub(crate) pack_resolve_sizes_ms: f64,
    pub(crate) pack_resolve_scan_ms: f64,
    pub(crate) pack_resolve_readback_setup_ms: f64,
    pub(crate) pack_resolve_submit_ms: f64,
    pub(crate) pack_resolve_map_wait_ms: f64,
    pub(crate) pack_resolve_parse_ms: f64,
    pub(crate) pack_src_copy_ms: f64,
    pub(crate) pack_metadata_loop_ms: f64,
    pub(crate) pack_host_copy_ms: f64,
    pub(crate) pack_device_copy_plan_ms: f64,
    pub(crate) pack_finalize_ms: f64,
    pub(crate) scratch_acquire_ms: f64,
    pub(crate) match_table_copy_ms: f64,
    pub(crate) match_prepare_dispatch_ms: f64,
    pub(crate) match_kernel_dispatch_ms: f64,
    pub(crate) match_submit_ms: f64,
    pub(crate) section_setup_ms: f64,
    pub(crate) section_pass_dispatch_ms: f64,
    pub(crate) section_tokenize_dispatch_ms: f64,
    pub(crate) section_prefix_dispatch_ms: f64,
    pub(crate) section_scatter_dispatch_ms: f64,
    pub(crate) section_pack_dispatch_ms: f64,
    pub(crate) section_meta_dispatch_ms: f64,
    pub(crate) section_copy_dispatch_ms: f64,
    pub(crate) section_submit_ms: f64,
    pub(crate) section_tokenize_wait_ms: f64,
    pub(crate) section_prefix_wait_ms: f64,
    pub(crate) section_scatter_wait_ms: f64,
    pub(crate) section_pack_wait_ms: f64,
    pub(crate) section_meta_wait_ms: f64,
    pub(crate) section_wrap_ms: f64,
    pub(crate) sparse_lens_submit_ms: f64,
    pub(crate) sparse_lens_submit_done_wait_ms: f64,
    pub(crate) sparse_lens_map_after_done_ms: f64,
    pub(crate) sparse_lens_poll_calls: u64,
    pub(crate) sparse_lens_yield_calls: u64,
    pub(crate) sparse_lens_wait_ms: f64,
    pub(crate) sparse_lens_copy_ms: f64,
    pub(crate) sparse_prepare_ms: f64,
    pub(crate) sparse_scratch_acquire_ms: f64,
    pub(crate) sparse_upload_dispatch_ms: f64,
    pub(crate) sparse_submit_ms: f64,
    pub(crate) sparse_wait_ms: f64,
    pub(crate) sparse_copy_ms: f64,
    pub(crate) sparse_total_ms: f64,
    pub(crate) call_pack_bind_group_ms: f64,
    pub(crate) call_encoder_create_ms: f64,
    pub(crate) call_table_stream_copy_ms: f64,
    pub(crate) call_match_clear_ms: f64,
    pub(crate) call_sparse_prepare_dispatch_ms: f64,
    pub(crate) call_sparse_pack_dispatch_ms: f64,
    pub(crate) call_result_readback_setup_ms: f64,
    pub(crate) call_payload_readback_setup_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ResolvePackedTableSizesBatchProfile {
    unresolved_count: usize,
    scan_ms: f64,
    readback_setup_ms: f64,
    submit_ms: f64,
    submit_done_wait_ms: f64,
    map_callback_wait_ms: f64,
    map_wait_ms: f64,
    pre_wait_latest_build_submit_ms: f64,
    latest_build_submit_seq: u64,
    readback_submit_seq: u64,
    parse_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct PackMatchBatchInputsProfile {
    total_ms: f64,
    alloc_setup_ms: f64,
    resolve_sizes_ms: f64,
    resolve_scan_ms: f64,
    resolve_readback_setup_ms: f64,
    resolve_submit_ms: f64,
    resolve_map_wait_ms: f64,
    resolve_parse_ms: f64,
    src_copy_ms: f64,
    metadata_loop_ms: f64,
    host_copy_ms: f64,
    device_copy_plan_ms: f64,
    finalize_ms: f64,
}

pub(crate) struct GpuMatchInput<'a> {
    pub(crate) src: &'a [u8],
    pub(crate) table: &'a [Vec<u8>],
    pub(crate) table_gpu: Option<&'a GpuPackedTableDevice>,
    pub(crate) table_index: Option<&'a [u8]>,
    pub(crate) table_data: Option<&'a [u8]>,
    pub(crate) max_ref_len: usize,
    pub(crate) min_ref_len: usize,
    pub(crate) section_count: usize,
    pub(crate) huffman_encode_enabled: bool,
    pub(crate) compression_mode: super::PDeflateCompressionMode,
}

pub(crate) struct GpuMatchOutput {
    pub(crate) packed_matches: Vec<u32>,
    pub(crate) profile: GpuMatchProfile,
}

pub(crate) struct GpuBatchChunkOutput {
    pub(crate) packed_matches: Vec<u32>,
}

pub(crate) struct GpuBatchMatchOutput {
    pub(crate) chunks: Vec<GpuBatchChunkOutput>,
    pub(crate) profile: GpuMatchProfile,
}

pub(crate) struct GpuBatchSectionEncodeChunkOutput {
    pub(crate) section_cmd_lens: Vec<u32>,
    pub(crate) section_cmd: Vec<u8>,
    pub(crate) section_cmd_device: Option<GpuSectionCmdDeviceOutput>,
}

pub(crate) struct GpuBatchSectionEncodeOutput {
    pub(crate) chunks: Vec<GpuBatchSectionEncodeChunkOutput>,
    pub(crate) match_profile: GpuMatchProfile,
    pub(crate) section_profile: GpuSectionEncodeProfile,
    pub(crate) kernel_profile: GpuBatchKernelProfile,
    pub(crate) keepalive: Option<GpuBatchSectionKeepalive>,
}

pub(crate) struct GpuBatchSparsePackOutput {
    pub(crate) chunks: Vec<GpuSparsePackChunkOutput>,
    pub(crate) match_profile: GpuMatchProfile,
    pub(crate) section_profile: GpuSectionEncodeProfile,
    pub(crate) sparse_profile: GpuMatchProfile,
    pub(crate) sparse_batch_profile: GpuSparsePackBatchProfile,
    pub(crate) kernel_profile: GpuBatchKernelProfile,
}

struct IntegratedSparsePackPrepared {
    chunk_len: usize,
    huffman_encode_enabled: bool,
    section_count: usize,
    table_data_stream_offset: usize,
    section_index_stream_offset: usize,
    section_index_cap_len: usize,
    section_cmd_cap_len: usize,
    total_len_cap: usize,
    total_bytes_cap: u64,
    table_count: usize,
    table_index_len: usize,
    table_data_len: usize,
}

#[derive(Clone, Copy)]
struct IntegratedSparsePackHostJob {
    section_meta_words_off: usize,
    table_index_off: usize,
    table_data_off: usize,
    section_index_off: usize,
    out_lens_words_off: usize,
    section_prefix_words_off: usize,
    section_offsets_words_off: usize,
    out_cmd_off: usize,
    out_base_word: usize,
}

struct IntegratedDirectSectionState {
    chunk_len: usize,
    src_base_u32: u32,
    out_lens_bytes: u64,
    out_cmd_bytes: u64,
    max_tokens_per_section: u32,
    max_cmd_words_per_section: u32,
    predicted_payload_bytes: usize,
    predicted_payload_class: GpuSparsePayloadClass,
    slot_cap_bytes: u64,
    cap_payload_class: GpuSparsePayloadClass,
    prep: IntegratedSparsePackPrepared,
    host: IntegratedSparsePackHostJob,
    section_offsets_local: Vec<u32>,
    section_offsets_global: Vec<u32>,
    section_caps: Vec<u32>,
    section_token_offsets: Vec<u32>,
    section_token_caps: Vec<u32>,
}

pub(crate) struct PendingGpuSparsePackBatch {
    inputs_len: usize,
    compression_mode: super::PDeflateCompressionMode,
    direct_states: Vec<IntegratedDirectSectionState>,
    payload_readback_offsets: Vec<u64>,
    payload_readback_plan_bytes: Vec<u64>,
    result_readback_bytes: u64,
    payload_readback_bytes: u64,
    sparse_prebuild_build_id_min: u64,
    sparse_prebuild_build_id_max: u64,
    sparse_prebuild_submit_seq_min: u64,
    sparse_prebuild_submit_seq_max: u64,
    sparse_latest_build_submit_seq: u64,
    sparse_latest_build_id: u64,
    sparse_build_ids: Vec<u64>,
    result_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    payload_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    result_ready: bool,
    payload_ready: bool,
    submit_at: Instant,
    pack_alloc_setup_ms: f64,
    pack_resolve_sizes_ms: f64,
    pack_resolve_scan_ms: f64,
    pack_resolve_readback_setup_ms: f64,
    pack_resolve_submit_ms: f64,
    pack_resolve_map_wait_ms: f64,
    pack_resolve_parse_ms: f64,
    pack_src_copy_ms: f64,
    pack_metadata_loop_ms: f64,
    pack_host_copy_ms: f64,
    pack_device_copy_plan_ms: f64,
    pack_finalize_ms: f64,
    match_upload_ms: f64,
    match_table_copy_ms: f64,
    match_prepare_dispatch_ms: f64,
    match_kernel_dispatch_ms: f64,
    section_upload_ms: f64,
    section_setup_ms: f64,
    section_pass_dispatch_ms: f64,
    section_tokenize_dispatch_ms: f64,
    section_prefix_dispatch_ms: f64,
    section_scatter_dispatch_ms: f64,
    section_pack_dispatch_ms: f64,
    section_meta_dispatch_ms: f64,
    pack_inputs_ms: f64,
    batch_scratch_acquire_ms: f64,
    sparse_scratch_acquire_ms: f64,
    sparse_prepare_ms: f64,
    table_size_resolve_ms: f64,
    pack_bind_group_ms: f64,
    encoder_create_ms: f64,
    table_stream_copy_ms: f64,
    match_clear_ms: f64,
    sparse_prepare_dispatch_ms: f64,
    sparse_pack_dispatch_ms: f64,
    result_readback_setup_ms: f64,
    payload_readback_setup_ms: f64,
    submit_ms: f64,
    phase1_submit_done_wait_ms: f64,
    phase1_result_wait_calls: u64,
    phase1_payload_wait_calls: u64,
    phase1_result_ready_hits: u64,
    phase1_payload_ready_hits: u64,
    phase1_yield_calls: u64,
    batch_scratch: Option<GpuBatchScratch>,
    sparse_scratch: Option<GpuSparsePackScratch>,
}

pub(crate) struct GpuSectionCmdDeviceOutput {
    pub(crate) section_count: usize,
    pub(crate) section_offsets_buffer: Arc<wgpu::Buffer>,
    pub(crate) out_lens_buffer: Arc<wgpu::Buffer>,
    pub(crate) out_lens_readback_buffer: Arc<wgpu::Buffer>,
    pub(crate) section_prefix_buffer: Arc<wgpu::Buffer>,
    pub(crate) section_index_buffer: Arc<wgpu::Buffer>,
    pub(crate) section_meta_buffer: Arc<wgpu::Buffer>,
    pub(crate) section_meta_readback_buffer: Arc<wgpu::Buffer>,
    pub(crate) out_cmd_buffer: Arc<wgpu::Buffer>,
    pub(crate) out_lens_bytes: u64,
    pub(crate) out_cmd_bytes: u64,
    pub(crate) section_index_cap_bytes: u64,
}

pub(crate) struct GpuBatchSectionKeepalive {
    scratch: Option<GpuBatchScratch>,
}

impl GpuBatchSectionKeepalive {
    fn new(scratch: GpuBatchScratch) -> Self {
        Self {
            scratch: Some(scratch),
        }
    }
}

impl Drop for GpuBatchSectionKeepalive {
    fn drop(&mut self) {
        let Some(scratch) = self.scratch.take() else {
            return;
        };
        let Ok(r) = runtime() else {
            return;
        };
        release_batch_scratch(r, scratch);
    }
}

pub(crate) struct GpuSparsePackInput<'a> {
    pub(crate) chunk_len: usize,
    pub(crate) section_count: usize,
    pub(crate) huffman_encode_enabled: bool,
    pub(crate) table: &'a GpuPackedTableDevice,
    pub(crate) section: &'a GpuSectionCmdDeviceOutput,
}

pub(crate) struct GpuSparsePackChunkOutput {
    pub(crate) payload: Vec<u8>,
    pub(crate) table_count: usize,
}

pub(crate) struct GpuSparseSectionHostOutput {
    pub(crate) section_cmd_lens: Vec<u32>,
    pub(crate) section_cmd: Vec<u8>,
}

const GPU_SPARSE_PAYLOAD_CLASS_COUNT: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuSparsePayloadClass {
    Tiny,
    Small,
    Medium,
    Large,
    XLarge,
}

impl GpuSparsePayloadClass {
    fn idx(self) -> usize {
        match self {
            Self::Tiny => 0,
            Self::Small => 1,
            Self::Medium => 2,
            Self::Large => 3,
            Self::XLarge => 4,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
            Self::XLarge => "xlarge",
        }
    }
}

const GPU_SPARSE_PAYLOAD_CLASS_ORDER: [GpuSparsePayloadClass; GPU_SPARSE_PAYLOAD_CLASS_COUNT] = [
    GpuSparsePayloadClass::Tiny,
    GpuSparsePayloadClass::Small,
    GpuSparsePayloadClass::Medium,
    GpuSparsePayloadClass::Large,
    GpuSparsePayloadClass::XLarge,
];

fn classify_sparse_payload_bytes(bytes: usize) -> GpuSparsePayloadClass {
    if bytes <= 256 * 1024 {
        GpuSparsePayloadClass::Tiny
    } else if bytes <= 1024 * 1024 {
        GpuSparsePayloadClass::Small
    } else if bytes <= 2 * 1024 * 1024 {
        GpuSparsePayloadClass::Medium
    } else if bytes <= 4 * 1024 * 1024 {
        GpuSparsePayloadClass::Large
    } else {
        GpuSparsePayloadClass::XLarge
    }
}

fn format_sparse_payload_class_counts(counts: &[u32; GPU_SPARSE_PAYLOAD_CLASS_COUNT]) -> String {
    GPU_SPARSE_PAYLOAD_CLASS_ORDER
    .into_iter()
    .map(|class_id| format!("{}:{}", class_id.label(), counts[class_id.idx()]))
    .collect::<Vec<_>>()
    .join(",")
}

fn sparse_payload_class_capacity_bytes(
    class_id: GpuSparsePayloadClass,
    predicted_payload_bytes: usize,
) -> usize {
    match class_id {
        GpuSparsePayloadClass::Tiny => 256 * 1024,
        GpuSparsePayloadClass::Small => 1024 * 1024,
        GpuSparsePayloadClass::Medium => 2 * 1024 * 1024,
        GpuSparsePayloadClass::Large => 4 * 1024 * 1024,
        GpuSparsePayloadClass::XLarge => align_up4(predicted_payload_bytes.max(4)).div_ceil(1024 * 1024) * (1024 * 1024),
    }
}

fn estimate_sparse_payload_bytes(
    compression_mode: super::PDeflateCompressionMode,
    chunk_len: usize,
    section_count: usize,
    table_count: usize,
    table_index_len: usize,
    table_data_len: usize,
    section_index_cap_len: usize,
    section_cmd_cap_len: usize,
) -> usize {
    let base = PACK_CHUNK_HEADER_SIZE
        .saturating_add(table_index_len)
        .saturating_add(table_data_len)
        .saturating_add(section_index_cap_len);
    let section_meta_bias = section_count.saturating_mul(16);
    let table_bias = table_count.saturating_mul(8);
    let chunk_bias = match compression_mode {
        super::PDeflateCompressionMode::Speed => chunk_len / 2,
        super::PDeflateCompressionMode::Balanced => (chunk_len / 4).saturating_mul(3),
        super::PDeflateCompressionMode::Ratio => chunk_len,
    };
    let cmd_estimate = section_cmd_cap_len.min(
        chunk_bias
            .saturating_add(section_meta_bias)
            .saturating_add(table_bias),
    );
    align_up4(base.saturating_add(cmd_estimate))
}

fn planned_sparse_payload_readback_bytes(
    predicted_payload_bytes: usize,
    slot_cap_bytes: u64,
) -> u64 {
    let predicted = align_up4(predicted_payload_bytes.max(4));
    let guarded = predicted
        .saturating_add(predicted / 8)
        .saturating_add(16 * 1024);
    let planned = align_up4(guarded);
    u64::try_from(planned)
        .unwrap_or(u64::MAX)
        .min(slot_cap_bytes.max(4))
        .max(4)
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuSparsePackBatchProfile {
    pub(crate) chunks: usize,
    pub(crate) lens_bytes_total: u64,
    pub(crate) out_cmd_bytes_total: u64,
    pub(crate) predicted_payload_bytes_total: u64,
    pub(crate) cap_payload_bytes_total: u64,
    pub(crate) actual_payload_bytes_total: u64,
    pub(crate) predicted_class_counts: [u32; GPU_SPARSE_PAYLOAD_CLASS_COUNT],
    pub(crate) cap_class_counts: [u32; GPU_SPARSE_PAYLOAD_CLASS_COUNT],
    pub(crate) actual_class_counts: [u32; GPU_SPARSE_PAYLOAD_CLASS_COUNT],
    pub(crate) predicted_underestimate_chunks: u32,
    pub(crate) predicted_underestimate_bytes: u64,
    pub(crate) predicted_overestimate_bytes: u64,
    pub(crate) lens_submit_ms: f64,
    pub(crate) lens_submit_done_wait_ms: f64,
    pub(crate) lens_map_after_done_ms: f64,
    pub(crate) lens_poll_calls: u64,
    pub(crate) lens_yield_calls: u64,
    pub(crate) lens_wait_ms: f64,
    pub(crate) lens_copy_ms: f64,
    pub(crate) prepare_ms: f64,
    pub(crate) table_size_resolve_ms: f64,
    pub(crate) prepare_misc_ms: f64,
    pub(crate) scratch_acquire_ms: f64,
    pub(crate) upload_dispatch_ms: f64,
    pub(crate) submit_ms: f64,
    pub(crate) wait_ms: f64,
    pub(crate) copy_ms: f64,
    pub(crate) total_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuSectionEncodeProfile {
    pub(crate) upload_ms: f64,
    pub(crate) wait_ms: f64,
    pub(crate) map_copy_ms: f64,
    pub(crate) total_ms: f64,
}

#[allow(dead_code)]
pub(crate) struct GpuSectionEncodeOutput {
    pub(crate) section_cmd_lens: Vec<u32>,
    pub(crate) section_cmd: Vec<u8>,
    pub(crate) profile: GpuSectionEncodeProfile,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuBuildTableProfile {
    pub(crate) upload_ms: f64,
    pub(crate) freq_kernel_ms: f64,
    pub(crate) candidate_kernel_ms: f64,
    pub(crate) readback_ms: f64,
    pub(crate) materialize_ms: f64,
    pub(crate) sort_ms: f64,
    pub(crate) sample_count: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuTableBuildBreakdown {
    pub(crate) total_ms: f64,
    pub(crate) pre_resource_ms: f64,
    pub(crate) resource_setup_ms: f64,
    pub(crate) upload_ms: f64,
    pub(crate) encode_pass1_ms: f64,
    pub(crate) submit_pass1_ms: f64,
    pub(crate) encode_pass2_ms: f64,
    pub(crate) submit_pass2_ms: f64,
    pub(crate) stage_probe_submit_ms: f64,
    pub(crate) stage_probe_wait_ms: f64,
    pub(crate) stage_probe_parse_ms: f64,
    pub(crate) stage_gpu_sum_ms: f64,
    pub(crate) tail_ms: f64,
    pub(crate) other_ms: f64,
}

pub(crate) struct GpuPackedTable {
    pub(crate) table_index: Vec<u8>,
    pub(crate) table_data: Vec<u8>,
    pub(crate) table_count: usize,
}

pub(crate) struct GpuPackedTableDevice {
    pub(crate) table_index_buffer: Arc<wgpu::Buffer>,
    pub(crate) table_data_buffer: Arc<wgpu::Buffer>,
    pub(crate) table_meta_buffer: Arc<wgpu::Buffer>,
    pub(crate) table_index_offset: u64,
    pub(crate) table_data_offset: u64,
    pub(crate) table_meta_offset: u64,
    pub(crate) table_count: usize,
    pub(crate) table_index_len: usize,
    pub(crate) table_data_len: usize,
    pub(crate) sizes_known: bool,
    pub(crate) max_entries: usize,
    pub(crate) table_data_bytes_cap: usize,
    pub(crate) build_id: u64,
    pub(crate) build_submit_seq: u64,
    pub(crate) build_submit_index: Option<wgpu::SubmissionIndex>,
    pub(crate) build_breakdown: GpuTableBuildBreakdown,
}

struct GpuMatchRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    supports_timestamp_query: bool,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    pack_chunk_bind_group_layout: wgpu::BindGroupLayout,
    pack_chunk_pipeline: wgpu::ComputePipeline,
    pack_sparse_prepare_bind_group_layout: wgpu::BindGroupLayout,
    pack_sparse_prepare_pipeline: wgpu::ComputePipeline,
    pack_sparse_bind_group_layout: wgpu::BindGroupLayout,
    pack_sparse_pipeline: wgpu::ComputePipeline,
    pack_sparse_stats_zero_buffer: wgpu::Buffer,
    section_encode_bind_group_layout: wgpu::BindGroupLayout,
    section_encode_pipeline: wgpu::ComputePipeline,
    section_tokenize_bind_group_layout: wgpu::BindGroupLayout,
    section_tokenize_pipeline: wgpu::ComputePipeline,
    section_prefix_bind_group_layout: wgpu::BindGroupLayout,
    section_prefix_pipeline: wgpu::ComputePipeline,
    section_scatter_bind_group_layout: wgpu::BindGroupLayout,
    section_scatter_pipeline: wgpu::ComputePipeline,
    section_cmd_pack_bind_group_layout: wgpu::BindGroupLayout,
    section_cmd_pack_pipeline: wgpu::ComputePipeline,
    section_meta_bind_group_layout: wgpu::BindGroupLayout,
    section_meta_pipeline: wgpu::ComputePipeline,
    build_table_freq_bind_group_layout: wgpu::BindGroupLayout,
    build_table_freq_pipeline: wgpu::ComputePipeline,
    build_table_candidate_bind_group_layout: wgpu::BindGroupLayout,
    build_table_candidate_pipeline: wgpu::ComputePipeline,
    build_table_bucket_count_bind_group_layout: wgpu::BindGroupLayout,
    build_table_bucket_count_pipeline: wgpu::ComputePipeline,
    build_table_bucket_prefix_bind_group_layout: wgpu::BindGroupLayout,
    build_table_bucket_prefix_pipeline: wgpu::ComputePipeline,
    build_table_bucket_scatter_bind_group_layout: wgpu::BindGroupLayout,
    build_table_bucket_scatter_pipeline: wgpu::ComputePipeline,
    build_table_finalize_bind_group_layout: wgpu::BindGroupLayout,
    build_table_finalize_pipeline: wgpu::ComputePipeline,
    build_table_finalize_emit_desc_bind_group_layout: wgpu::BindGroupLayout,
    build_table_finalize_emit_desc_pipeline: wgpu::ComputePipeline,
    build_table_finalize_emit_bind_group_layout: wgpu::BindGroupLayout,
    build_table_finalize_emit_pipeline: wgpu::ComputePipeline,
    build_table_pack_index_bind_group_layout: wgpu::BindGroupLayout,
    build_table_pack_index_pipeline: wgpu::ComputePipeline,
    match_prepare_table_bind_group_layout: wgpu::BindGroupLayout,
    match_prepare_table_pipeline: wgpu::ComputePipeline,
    decode_v2_bind_group_layout: wgpu::BindGroupLayout,
    decode_v2_pipeline: wgpu::ComputePipeline,
    decode_v2_batch_bind_group_layout: wgpu::BindGroupLayout,
    decode_v2_batch_pipeline: wgpu::ComputePipeline,
    max_storage_binding_size: u64,
    scratch_pool: Mutex<Vec<GpuBatchScratch>>,
    scratch_hot: Mutex<Option<GpuBatchScratch>>,
    table_build_batch_pool: Mutex<Vec<GpuTableBuildBatchScratch>>,
    table_build_batch_hot: Mutex<Option<GpuTableBuildBatchScratch>>,
    sparse_pack_pool: Mutex<Vec<GpuSparsePackScratch>>,
    sparse_pack_hot: Mutex<Option<GpuSparsePackScratch>>,
    decode_v2_true_batch_pool: Mutex<Vec<GpuDecodeV2TrueBatchScratch>>,
    decode_slot_pool: Mutex<GpuDecodeSlotPool>,
}

struct GpuBatchScratch {
    src_cap_bytes: u64,
    chunk_starts_cap_bytes: u64,
    table_base_cap_bytes: u64,
    table_count_cap_bytes: u64,
    table_index_offsets_cap_bytes: u64,
    table_data_offsets_cap_bytes: u64,
    table_meta_cap_bytes: u64,
    table_index_cap_bytes: u64,
    prefix_first_cap_bytes: u64,
    table_lens_cap_bytes: u64,
    table_offsets_cap_bytes: u64,
    table_data_cap_bytes: u64,
    prep_params_cap_bytes: u64,
    params_cap_bytes: u64,
    out_cap_bytes: u64,
    src_buffer: wgpu::Buffer,
    chunk_starts_buffer: wgpu::Buffer,
    table_base_buffer: wgpu::Buffer,
    table_count_buffer: wgpu::Buffer,
    table_index_offsets_buffer: wgpu::Buffer,
    table_data_offsets_buffer: wgpu::Buffer,
    table_meta_buffer: wgpu::Buffer,
    table_index_buffer: wgpu::Buffer,
    prefix_first_buffer: wgpu::Buffer,
    table_lens_buffer: wgpu::Buffer,
    table_offsets_buffer: wgpu::Buffer,
    table_data_buffer: wgpu::Buffer,
    prep_params_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    prep_bind_group: wgpu::BindGroup,
    section_slots: Vec<GpuSectionEncodeSlot>,
}

#[derive(Clone, Copy)]
struct GpuTableBuildBatchScratchCaps {
    src_bytes: u64,
    freq_params_bytes: u64,
    cand_params_bytes: u64,
    bucket_params_bytes: u64,
    finalize_params_bytes: u64,
    freq_bytes: u64,
    cand_bytes: u64,
    bucket_bytes: u64,
    sorted_idx_bytes: u64,
    emit_desc_bytes: u64,
    emit_word_bytes: u64,
}

struct GpuTableBuildBatchScratch {
    caps: GpuTableBuildBatchScratchCaps,
    src_buffer: wgpu::Buffer,
    freq_params_buffer: wgpu::Buffer,
    cand_params_buffer: wgpu::Buffer,
    bucket_params_buffer: wgpu::Buffer,
    finalize_params_buffer: wgpu::Buffer,
    freq_buffer: wgpu::Buffer,
    cand_buffer: wgpu::Buffer,
    bucket_count_buffer: wgpu::Buffer,
    bucket_offset_buffer: wgpu::Buffer,
    bucket_cursor_buffer: wgpu::Buffer,
    sorted_idx_buffer: wgpu::Buffer,
    emit_desc_buffer: wgpu::Buffer,
    emit_word_buffer: wgpu::Buffer,
}

struct GpuSectionEncodeSlot {
    section_count_cap: usize,
    cmd_cap_bytes: u64,
    token_cap_count: u64,
    section_index_cap_bytes: u64,
    section_offsets_buffer: Arc<wgpu::Buffer>,
    section_caps_buffer: Arc<wgpu::Buffer>,
    section_token_offsets_buffer: Arc<wgpu::Buffer>,
    section_token_caps_buffer: Arc<wgpu::Buffer>,
    section_params_buffer: Arc<wgpu::Buffer>,
    out_lens_buffer: Arc<wgpu::Buffer>,
    out_lens_readback_buffer: Arc<wgpu::Buffer>,
    section_token_counts_buffer: Arc<wgpu::Buffer>,
    section_token_meta_buffer: Arc<wgpu::Buffer>,
    section_token_pos_buffer: Arc<wgpu::Buffer>,
    section_token_cmd_offsets_buffer: Arc<wgpu::Buffer>,
    out_cmd_byte_buffer: Arc<wgpu::Buffer>,
    section_prefix_buffer: Arc<wgpu::Buffer>,
    section_index_buffer: Arc<wgpu::Buffer>,
    section_meta_buffer: Arc<wgpu::Buffer>,
    out_cmd_buffer: Arc<wgpu::Buffer>,
    readback_meta_buffer: Arc<wgpu::Buffer>,
    bind_group: wgpu::BindGroup,
    tokenize_bind_group: wgpu::BindGroup,
    prefix_bind_group: wgpu::BindGroup,
    scatter_bind_group: wgpu::BindGroup,
    pack_bind_group: wgpu::BindGroup,
    meta_bind_group: wgpu::BindGroup,
}

struct GpuSparsePackScratch {
    caps: GpuSparsePackScratchCaps,
    desc_buffer: wgpu::Buffer,
    section_meta_buffer: wgpu::Buffer,
    table_index_buffer: wgpu::Buffer,
    table_data_buffer: wgpu::Buffer,
    section_index_buffer: wgpu::Buffer,
    out_lens_buffer: wgpu::Buffer,
    section_prefix_buffer: wgpu::Buffer,
    section_offsets_buffer: wgpu::Buffer,
    out_cmd_buffer: wgpu::Buffer,
    result_buffer: wgpu::Buffer,
    result_readback_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    payload_readback_buffer: wgpu::Buffer,
    prepare_bind_group: wgpu::BindGroup,
}

#[derive(Clone, Copy)]
struct GpuDecodeV2TrueBatchScratchCaps {
    desc_bytes: u64,
    section_meta_bytes: u64,
    cmd_bytes: u64,
    table_bytes: u64,
    out_bytes: u64,
    error_bytes: u64,
}

impl GpuDecodeV2TrueBatchScratchCaps {
    fn fits(&self, req: GpuDecodeV2TrueBatchScratchCaps) -> bool {
        self.desc_bytes >= req.desc_bytes
            && self.section_meta_bytes >= req.section_meta_bytes
            && self.cmd_bytes >= req.cmd_bytes
            && self.table_bytes >= req.table_bytes
            && self.out_bytes >= req.out_bytes
            && self.error_bytes >= req.error_bytes
    }

    fn covers(&self, other: &GpuDecodeV2TrueBatchScratchCaps) -> bool {
        self.fits(*other)
    }
}

struct GpuDecodeV2TrueBatchScratch {
    caps: GpuDecodeV2TrueBatchScratchCaps,
    desc_buffer: wgpu::Buffer,
    section_meta_buffer: wgpu::Buffer,
    cmd_buffer: wgpu::Buffer,
    table_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    error_buffer: wgpu::Buffer,
    out_readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    ts_query_set: Option<wgpu::QuerySet>,
    ts_resolve_buffer: Option<wgpu::Buffer>,
    ts_readback_buffer: Option<wgpu::Buffer>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuDecodeSlotClass {
    Normal,
    Large,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuDecodeSlotState {
    Free,
    Inflight,
    Ready,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct GpuDecodeSlotCaps {
    cmd_cap_bytes: u64,
    section_meta_cap: usize,
    table_cap_bytes: u64,
    out_cap_bytes: u64,
    error_cap_words: usize,
}

impl GpuDecodeSlotCaps {
    fn fits(&self, req: GpuDecodeSlotRequirement) -> bool {
        req.cmd_bytes <= self.cmd_cap_bytes
            && req.section_meta_count <= self.section_meta_cap
            && req.table_bytes <= self.table_cap_bytes
            && req.out_bytes <= self.out_cap_bytes
            && req.error_words <= self.error_cap_words
    }

    fn max_with(self, rhs: GpuDecodeSlotCaps) -> GpuDecodeSlotCaps {
        GpuDecodeSlotCaps {
            cmd_cap_bytes: self.cmd_cap_bytes.max(rhs.cmd_cap_bytes),
            section_meta_cap: self.section_meta_cap.max(rhs.section_meta_cap),
            table_cap_bytes: self.table_cap_bytes.max(rhs.table_cap_bytes),
            out_cap_bytes: self.out_cap_bytes.max(rhs.out_cap_bytes),
            error_cap_words: self.error_cap_words.max(rhs.error_cap_words),
        }
    }
}

struct GpuDecodeSlotBuffers {
    cmd_buffer: wgpu::Buffer,
    section_meta_buffer: wgpu::Buffer,
    table_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    out_readback_buffer: wgpu::Buffer,
    error_buffer: wgpu::Buffer,
}

struct GpuDecodeSlotEntry {
    slot_index: usize,
    generation: u64,
    state: GpuDecodeSlotState,
    caps: GpuDecodeSlotCaps,
    buffers: GpuDecodeSlotBuffers,
    decode_bind_group: wgpu::BindGroup,
}

#[derive(Default)]
struct GpuDecodeSlotGroup {
    caps: GpuDecodeSlotCaps,
    slots: Vec<GpuDecodeSlotEntry>,
}

#[derive(Default)]
struct GpuDecodeSlotPool {
    normal: GpuDecodeSlotGroup,
    large: GpuDecodeSlotGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct GpuDecodeSlotRequirement {
    cmd_bytes: u64,
    section_meta_count: usize,
    table_bytes: u64,
    out_bytes: u64,
    error_words: usize,
}

#[derive(Debug, Clone, Copy)]
struct GpuDecodePayloadLayout {
    table_count: usize,
    table_index_start: usize,
    table_data_start: usize,
    table_data_end: usize,
    huff_lut_start: usize,
    huff_lut_end: usize,
    huffman_enabled: bool,
    cmd_start: usize,
}

#[derive(Debug, Clone, Copy)]
struct PreparedGpuDecodeJob<'a> {
    job: &'a GpuDecodeJob<'a>,
    req: GpuDecodeSlotRequirement,
    layout: GpuDecodePayloadLayout,
}

#[derive(Clone, Copy)]
struct GpuDecodeBatchHostJob<'a> {
    prepared: PreparedGpuDecodeJob<'a>,
    out_base: usize,
    out_len: usize,
    error_base: usize,
    error_words: usize,
}

struct PendingGpuDecodeMap {
    out_data_copy_len: u64,
    out_total_copy_len: u64,
    out_len: usize,
    error_words: usize,
    submit_seq: u64,
    submit_ms: f64,
    kernel_timestamp_ms: Option<f64>,
    ts_readback_buffer: Option<wgpu::Buffer>,
    ts_map_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    submit_at: Instant,
    completion_gate: Arc<AtomicBool>,
    map_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

struct PendingGpuDecodeTrueBatch<'a> {
    host_jobs: Vec<GpuDecodeBatchHostJob<'a>>,
    out_copy_bytes: usize,
    out_total_copy_bytes: usize,
    scratch: GpuDecodeV2TrueBatchScratch,
    map_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ts_map_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    submit_at: Instant,
    map_requested_at: Instant,
    completion_gate: Arc<AtomicBool>,
    submission: wgpu::SubmissionIndex,
    map_done: Option<Result<(), PDeflateError>>,
    map_timed_out: bool,
    kernel_timestamp_ms: Option<f64>,
    submit_done_wait_ms: f64,
    map_callback_wait_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct GpuDecodeV2WaitProfile {
    submit_seq: u64,
    submit_ms: f64,
    submit_done_wait_ms: f64,
    map_callback_wait_ms: f64,
    map_copy_ms: f64,
    kernel_timestamp_ms: f64,
    queue_stall_est_ms: f64,
}

struct InflightGpuDecodeSubmission<'a> {
    prepared: PreparedGpuDecodeJob<'a>,
    class: GpuDecodeSlotClass,
    slot_idx: usize,
    slot_handle: GpuDecodeSlot,
    pending: PendingGpuDecodeMap,
    map_requested_at: Option<Instant>,
    map_done: Option<Result<(), PDeflateError>>,
    map_timed_out: bool,
    wait_profile: GpuDecodeV2WaitProfile,
}

#[derive(Clone, Copy, Default)]
struct GpuSparsePackScratchCaps {
    desc_bytes: u64,
    section_meta_bytes: u64,
    table_index_bytes: u64,
    table_data_bytes: u64,
    section_index_bytes: u64,
    out_lens_bytes: u64,
    section_prefix_bytes: u64,
    section_offsets_bytes: u64,
    out_cmd_bytes: u64,
    result_bytes: u64,
    out_bytes: u64,
}

impl GpuSparsePackScratchCaps {
    fn fits(&self, req: GpuSparsePackScratchCaps) -> bool {
        self.desc_bytes >= req.desc_bytes
            && self.section_meta_bytes >= req.section_meta_bytes
            && self.table_index_bytes >= req.table_index_bytes
            && self.table_data_bytes >= req.table_data_bytes
            && self.section_index_bytes >= req.section_index_bytes
            && self.out_lens_bytes >= req.out_lens_bytes
            && self.section_prefix_bytes >= req.section_prefix_bytes
            && self.section_offsets_bytes >= req.section_offsets_bytes
            && self.out_cmd_bytes >= req.out_cmd_bytes
            && self.result_bytes >= req.result_bytes
            && self.out_bytes >= req.out_bytes
    }
}

#[derive(Clone, Copy)]
struct GpuBatchScratchCaps {
    src_bytes: u64,
    chunk_starts_bytes: u64,
    table_base_bytes: u64,
    table_count_bytes: u64,
    table_index_offsets_bytes: u64,
    table_data_offsets_bytes: u64,
    table_meta_bytes: u64,
    table_index_bytes: u64,
    prefix_first_bytes: u64,
    table_lens_bytes: u64,
    table_offsets_bytes: u64,
    table_data_bytes: u64,
    prep_params_bytes: u64,
    params_bytes: u64,
    out_bytes: u64,
}

impl GpuTableBuildBatchScratchCaps {
    fn fits(&self, req: Self) -> bool {
        self.src_bytes >= req.src_bytes
            && self.freq_params_bytes >= req.freq_params_bytes
            && self.cand_params_bytes >= req.cand_params_bytes
            && self.bucket_params_bytes >= req.bucket_params_bytes
            && self.finalize_params_bytes >= req.finalize_params_bytes
            && self.freq_bytes >= req.freq_bytes
            && self.cand_bytes >= req.cand_bytes
            && self.bucket_bytes >= req.bucket_bytes
            && self.sorted_idx_bytes >= req.sorted_idx_bytes
            && self.emit_desc_bytes >= req.emit_desc_bytes
            && self.emit_word_bytes >= req.emit_word_bytes
    }

    fn covers(&self, other: Self) -> bool {
        self.fits(other)
    }
}

const GPU_SCRATCH_POOL_LIMIT: usize = 4;
const GPU_TABLE_BUILD_BATCH_POOL_LIMIT: usize = 4;
const GPU_SPARSE_PACK_POOL_LIMIT: usize = 4;
const GPU_DECODE_V2_TRUE_BATCH_POOL_LIMIT: usize = 16;
const GPU_DECODE_LARGE_SLOT_INDEX_BASE: usize = 1 << 30;
const GPU_DECODE_V2_BATCH_DESC_WORDS: usize = 12;
const GPU_SPARSE_PACK_BATCH_DESC_WORDS: usize = 23;
const GPU_SPARSE_PACK_BATCH_RESULT_WORDS: usize = 4;
// Command stream cap heuristic for GPU section encode.
// If this underestimates a pathological section, kernel marks overflow and
// caller falls back to CPU for correctness.
const GPU_SECTION_CMD_CAP_MULTIPLIER: usize = 2;
const GPU_SECTION_CMD_CAP_PAD: usize = 64;

static GPU_RUNTIME: OnceLock<Result<GpuMatchRuntime, String>> = OnceLock::new();
static GPU_SUBMIT_STREAM_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
static BUILD_TABLE_STAGE_PROBE_SEQ: AtomicUsize = AtomicUsize::new(0);
static BUILD_TABLE_STAGE_PROBE_UNSUPPORTED_LOGGED: OnceLock<()> = OnceLock::new();
static SPARSE_KERNEL_TS_PROBE_SEQ: AtomicUsize = AtomicUsize::new(0);
static SPARSE_KERNEL_TS_PROBE_UNSUPPORTED_LOGGED: OnceLock<()> = OnceLock::new();
static BUILD_TABLE_BREAKDOWN_SEQ: AtomicUsize = AtomicUsize::new(0);
static RESOLVE_MAP_WAIT_PROBE_SEQ: AtomicUsize = AtomicUsize::new(0);
static SPARSE_LENS_WAIT_PROBE_SEQ: AtomicUsize = AtomicUsize::new(0);
static GPU_CALL_UNACCOUNTED_SEQ: AtomicUsize = AtomicUsize::new(0);
static GPU_INTEGRATED_SECTION_PROBE_LOGGED: AtomicBool = AtomicBool::new(false);
static GPU_INTEGRATED_FINAL_PROBE_LOGGED: AtomicBool = AtomicBool::new(false);
static GPU_DECODE_V2_WAIT_PROBE_SEQ: AtomicUsize = AtomicUsize::new(0);
static GPU_DECODE_V2_KERNEL_TS_UNSUPPORTED_LOGGED: OnceLock<()> = OnceLock::new();
static BUILD_TABLE_DEVICE_SEQ: AtomicU64 = AtomicU64::new(0);
static GPU_SUBMIT_SEQ: AtomicU64 = AtomicU64::new(0);

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim();
            !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    }
}

fn sparse_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("COZIP_PDEFLATE_PROFILE_SPARSE_PROBE"))
}

fn sparse_ready_first_spin_polls() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("COZIP_PDEFLATE_GPU_SPARSE_READY_FIRST_SPIN_POLLS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(16)
            .max(1)
    })
}

fn sparse_ready_first_yield_rounds() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("COZIP_PDEFLATE_GPU_SPARSE_READY_FIRST_YIELD_ROUNDS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
}

fn gpu_sparse_payload_validate_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("COZIP_PDEFLATE_GPU_SPARSE_VALIDATE_PAYLOAD"))
}

fn queue_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("COZIP_PDEFLATE_PROFILE_QUEUE_PROBE"))
}

fn table_stage_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_TABLE_STAGE_PROBE") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        },
    )
}

fn table_build_breakdown_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_TABLE_BUILD_BREAKDOWN") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        },
    )
}

fn resolve_map_wait_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_RESOLVE_MAP_WAIT_PROBE") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        },
    )
}

fn gpu_call_unaccounted_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_GPU_CALL_UNACCOUNTED") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        },
    )
}

fn resolve_wait_attribution_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("COZIP_PDEFLATE_PROFILE_RESOLVE_WAIT_ATTRIBUTION"))
}

fn sparse_lens_wait_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_SPARSE_LENS_WAIT_PROBE") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        },
    )
}

fn sparse_wait_attribution_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("COZIP_PDEFLATE_PROFILE_SPARSE_WAIT_ATTRIBUTION"))
}

fn sparse_kernel_ts_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_SPARSE_KERNEL_TS") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        },
    )
}

fn gpu_decode_v2_wait_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        match std::env::var("COZIP_PDEFLATE_PROFILE_GPU_DECODE_V2_WAIT_PROBE") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        }
    })
}

fn gpu_decode_v2_true_batch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("COZIP_PDEFLATE_GPU_DECODE_V2_TRUE_BATCH"))
}

fn gpu_decode_v2_true_batch_jobs(default_jobs: usize) -> usize {
    std::env::var("COZIP_PDEFLATE_GPU_DECODE_V2_TRUE_BATCH_JOBS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default_jobs.max(1))
}

fn gpu_decode_v2_true_batch_copy_jobs(default_jobs: usize) -> usize {
    let fallback = if default_jobs >= 8 {
        (default_jobs / 2).max(1)
    } else {
        default_jobs.max(1)
    };
    std::env::var("COZIP_PDEFLATE_GPU_DECODE_V2_TRUE_BATCH_COPY_JOBS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(fallback)
}

fn gpu_decode_v2_true_batch_readback_ring(default_depth: usize) -> usize {
    std::env::var("COZIP_PDEFLATE_GPU_DECODE_V2_TRUE_BATCH_READBACK_RING")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default_depth.max(1))
}

fn gpu_decode_v2_wait_probe_should_log(seq: usize) -> bool {
    seq < 8 || seq % 64 == 0
}

fn gpu_decode_v2_spin_polls_before_wait() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("COZIP_PDEFLATE_GPU_DECODE_V2_SPIN_POLLS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(16)
    })
}

fn gpu_decode_v2_target_inflight(max_inflight: usize) -> usize {
    if max_inflight <= 1 {
        return 1;
    }
    if let Ok(v) = std::env::var("COZIP_PDEFLATE_GPU_DECODE_V2_TARGET_INFLIGHT") {
        if let Ok(parsed) = v.trim().parse::<usize>() {
            return parsed.clamp(1, max_inflight);
        }
    }
    max_inflight
}

fn gpu_decode_v2_wait_high_watermark(max_inflight: usize, target_inflight: usize) -> usize {
    if max_inflight <= 1 {
        return 1;
    }
    if let Ok(v) = std::env::var("COZIP_PDEFLATE_GPU_DECODE_V2_WAIT_HIGH_WATERMARK") {
        if let Ok(parsed) = v.trim().parse::<usize>() {
            return parsed.clamp(1, max_inflight);
        }
    }
    max_inflight.max(target_inflight).clamp(1, max_inflight)
}

fn gpu_decode_v2_map_timeout_ms() -> Option<f64> {
    static VALUE: OnceLock<Option<f64>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("COZIP_PDEFLATE_GPU_DECODE_V2_MAP_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|ms| *ms > 0.0)
    })
}

fn sparse_inflight_submit_chunks() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_SPARSE_INFLIGHT_SUBMIT_CHUNKS") {
            Ok(v) => v
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|n| *n > 0)
                .unwrap_or(usize::MAX),
            Err(_) => 4,
        },
    )
}

fn table_stage_probe_should_log(seq: usize) -> bool {
    seq < 8 || seq % 64 == 0
}

fn next_gpu_submit_seq() -> u64 {
    GPU_SUBMIT_SEQ
        .fetch_add(1, Ordering::Relaxed)
        .saturating_add(1)
}

fn wait_for_queue_done(queue: &wgpu::Queue, device: &wgpu::Device) -> (f64, u64) {
    let t_wait = Instant::now();
    let (done_tx, done_rx) = mpsc::channel();
    queue.on_submitted_work_done(move || {
        let _ = done_tx.send(());
    });
    let mut poll_calls = 0u64;
    let mut done = false;
    while !done {
        device.poll(wgpu::Maintain::Wait);
        poll_calls = poll_calls.saturating_add(1);
        match done_rx.try_recv() {
            Ok(_) | Err(mpsc::TryRecvError::Disconnected) => done = true,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
    (elapsed_ms(t_wait), poll_calls)
}

fn round_capacity_bytes(required: u64) -> u64 {
    let req = required.max(4);
    req.checked_next_power_of_two().unwrap_or(req)
}

#[inline]
fn align_up4(v: usize) -> usize {
    (v + 3) & !3
}

#[inline]
fn estimate_section_cmd_cap_bytes(sec_len: usize) -> Result<usize, PDeflateError> {
    sec_len
        .checked_mul(GPU_SECTION_CMD_CAP_MULTIPLIER)
        .and_then(|v| v.checked_add(GPU_SECTION_CMD_CAP_PAD))
        .ok_or(PDeflateError::NumericOverflow)
}

#[inline]
fn estimate_token_cap_count(cmd_cap_bytes: u64) -> u64 {
    (cmd_cap_bytes / 2).max(1)
}

struct ByteWordPacker {
    words: Vec<u32>,
    len_bytes: usize,
}

impl ByteWordPacker {
    fn with_capacity_bytes(capacity: usize) -> Self {
        Self {
            words: Vec::with_capacity(capacity.div_ceil(4)),
            len_bytes: 0,
        }
    }

    fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    fn align4(&mut self) {
        let pad = align_up4(self.len_bytes).saturating_sub(self.len_bytes);
        if pad > 0 {
            self.append_zeros(pad);
        }
    }

    fn append_zeros(&mut self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let new_len = self.len_bytes.saturating_add(bytes);
        let need_words = new_len.div_ceil(4);
        if need_words > self.words.len() {
            self.words.resize(need_words, 0);
        }
        self.len_bytes = new_len;
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let new_len = self.len_bytes.saturating_add(bytes.len());
        let need_words = new_len.div_ceil(4);
        if need_words > self.words.len() {
            self.words.resize(need_words, 0);
        }
        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut self.words);
        dst[self.len_bytes..new_len].copy_from_slice(bytes);
        self.len_bytes = new_len;
    }

    fn write_bytes_at(&mut self, offset: usize, bytes: &[u8]) -> Result<(), PDeflateError> {
        let end = offset
            .checked_add(bytes.len())
            .ok_or(PDeflateError::NumericOverflow)?;
        if end > self.len_bytes {
            return Err(PDeflateError::NumericOverflow);
        }
        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut self.words);
        dst[offset..end].copy_from_slice(bytes);
        Ok(())
    }

    fn into_words(self) -> Vec<u32> {
        self.words
    }
}

fn storage_upload_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

impl GpuBatchScratch {
    fn fits(&self, caps: GpuBatchScratchCaps) -> bool {
        self.src_cap_bytes >= caps.src_bytes
            && self.chunk_starts_cap_bytes >= caps.chunk_starts_bytes
            && self.table_base_cap_bytes >= caps.table_base_bytes
            && self.table_count_cap_bytes >= caps.table_count_bytes
            && self.table_index_offsets_cap_bytes >= caps.table_index_offsets_bytes
            && self.table_data_offsets_cap_bytes >= caps.table_data_offsets_bytes
            && self.table_meta_cap_bytes >= caps.table_meta_bytes
            && self.table_index_cap_bytes >= caps.table_index_bytes
            && self.prefix_first_cap_bytes >= caps.prefix_first_bytes
            && self.table_lens_cap_bytes >= caps.table_lens_bytes
            && self.table_offsets_cap_bytes >= caps.table_offsets_bytes
            && self.table_data_cap_bytes >= caps.table_data_bytes
            && self.prep_params_cap_bytes >= caps.prep_params_bytes
            && self.params_cap_bytes >= caps.params_bytes
            && self.out_cap_bytes >= caps.out_bytes
    }

    fn covers(&self, other: &Self) -> bool {
        self.src_cap_bytes >= other.src_cap_bytes
            && self.chunk_starts_cap_bytes >= other.chunk_starts_cap_bytes
            && self.table_base_cap_bytes >= other.table_base_cap_bytes
            && self.table_count_cap_bytes >= other.table_count_cap_bytes
            && self.table_index_offsets_cap_bytes >= other.table_index_offsets_cap_bytes
            && self.table_data_offsets_cap_bytes >= other.table_data_offsets_cap_bytes
            && self.table_meta_cap_bytes >= other.table_meta_cap_bytes
            && self.table_index_cap_bytes >= other.table_index_cap_bytes
            && self.prefix_first_cap_bytes >= other.prefix_first_cap_bytes
            && self.table_lens_cap_bytes >= other.table_lens_cap_bytes
            && self.table_offsets_cap_bytes >= other.table_offsets_cap_bytes
            && self.table_data_cap_bytes >= other.table_data_cap_bytes
            && self.prep_params_cap_bytes >= other.prep_params_cap_bytes
            && self.params_cap_bytes >= other.params_cap_bytes
            && self.out_cap_bytes >= other.out_cap_bytes
    }

    fn new(runtime: &GpuMatchRuntime, required: GpuBatchScratchCaps) -> Self {
        let src_cap_bytes = round_capacity_bytes(required.src_bytes);
        let chunk_starts_cap_bytes = round_capacity_bytes(required.chunk_starts_bytes);
        let table_base_cap_bytes = round_capacity_bytes(required.table_base_bytes);
        let table_count_cap_bytes = round_capacity_bytes(required.table_count_bytes);
        let table_index_offsets_cap_bytes =
            round_capacity_bytes(required.table_index_offsets_bytes);
        let table_data_offsets_cap_bytes = round_capacity_bytes(required.table_data_offsets_bytes);
        let table_meta_cap_bytes = round_capacity_bytes(required.table_meta_bytes);
        let table_index_cap_bytes = round_capacity_bytes(required.table_index_bytes);
        let prefix_first_cap_bytes = round_capacity_bytes(required.prefix_first_bytes);
        let table_lens_cap_bytes = round_capacity_bytes(required.table_lens_bytes);
        let table_offsets_cap_bytes = round_capacity_bytes(required.table_offsets_bytes);
        let table_data_cap_bytes = round_capacity_bytes(required.table_data_bytes);
        let prep_params_cap_bytes = round_capacity_bytes(required.prep_params_bytes);
        let params_cap_bytes = round_capacity_bytes(required.params_bytes);
        let out_cap_bytes = round_capacity_bytes(required.out_bytes);

        let src_buffer =
            storage_upload_buffer(&runtime.device, "cozip-pdeflate-src", src_cap_bytes);
        let chunk_starts_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-chunk-starts",
            chunk_starts_cap_bytes,
        );
        let table_base_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-base",
            table_base_cap_bytes,
        );
        let table_count_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-count",
            table_count_cap_bytes,
        );
        let table_index_offsets_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-index-offsets",
            table_index_offsets_cap_bytes,
        );
        let table_data_offsets_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-data-offsets",
            table_data_offsets_cap_bytes,
        );
        let table_meta_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-meta",
            table_meta_cap_bytes,
        );
        let table_index_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-index",
            table_index_cap_bytes,
        );
        let prefix_first_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-prefix2-first",
            prefix_first_cap_bytes,
        );
        let table_lens_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-lens",
            table_lens_cap_bytes,
        );
        let table_offsets_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-offsets",
            table_offsets_cap_bytes,
        );
        let table_data_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-data",
            table_data_cap_bytes,
        );
        let prep_params_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-prep-params",
            prep_params_cap_bytes,
        );
        let params_buffer =
            storage_upload_buffer(&runtime.device, "cozip-pdeflate-params", params_cap_bytes);
        let out_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-out"),
            size: out_cap_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-readback"),
            size: out_cap_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-match-bg"),
                layout: &runtime.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: chunk_starts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: table_base_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: table_count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: prefix_first_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: table_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: table_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: table_data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: out_buffer.as_entire_binding(),
                    },
                ],
            });
        let prep_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-prep-table-bg"),
                layout: &runtime.match_prepare_table_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: table_base_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: table_count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: table_index_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: table_data_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: table_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: table_index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: table_data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: prep_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: prefix_first_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: table_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: table_offsets_buffer.as_entire_binding(),
                    },
                ],
            });

        Self {
            src_cap_bytes,
            chunk_starts_cap_bytes,
            table_base_cap_bytes,
            table_count_cap_bytes,
            table_index_offsets_cap_bytes,
            table_data_offsets_cap_bytes,
            table_meta_cap_bytes,
            table_index_cap_bytes,
            prefix_first_cap_bytes,
            table_lens_cap_bytes,
            table_offsets_cap_bytes,
            table_data_cap_bytes,
            prep_params_cap_bytes,
            params_cap_bytes,
            out_cap_bytes,
            src_buffer,
            chunk_starts_buffer,
            table_base_buffer,
            table_count_buffer,
            table_index_offsets_buffer,
            table_data_offsets_buffer,
            table_meta_buffer,
            table_index_buffer,
            prefix_first_buffer,
            table_lens_buffer,
            table_offsets_buffer,
            table_data_buffer,
            prep_params_buffer,
            params_buffer,
            out_buffer,
            readback_buffer,
            bind_group,
            prep_bind_group,
            section_slots: Vec::new(),
        }
    }
}

impl GpuTableBuildBatchScratch {
    fn new(runtime: &GpuMatchRuntime, required: GpuTableBuildBatchScratchCaps) -> Self {
        let caps = GpuTableBuildBatchScratchCaps {
            src_bytes: round_capacity_bytes(required.src_bytes),
            freq_params_bytes: round_capacity_bytes(required.freq_params_bytes),
            cand_params_bytes: round_capacity_bytes(required.cand_params_bytes),
            bucket_params_bytes: round_capacity_bytes(required.bucket_params_bytes),
            finalize_params_bytes: round_capacity_bytes(required.finalize_params_bytes),
            freq_bytes: round_capacity_bytes(required.freq_bytes),
            cand_bytes: round_capacity_bytes(required.cand_bytes),
            bucket_bytes: round_capacity_bytes(required.bucket_bytes),
            sorted_idx_bytes: round_capacity_bytes(required.sorted_idx_bytes),
            emit_desc_bytes: round_capacity_bytes(required.emit_desc_bytes),
            emit_word_bytes: round_capacity_bytes(required.emit_word_bytes),
        };
        Self {
            src_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-src",
                caps.src_bytes,
            ),
            freq_params_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-freq-params",
                caps.freq_params_bytes,
            ),
            cand_params_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-cand-params",
                caps.cand_params_bytes,
            ),
            bucket_params_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-bucket-params",
                caps.bucket_params_bytes,
            ),
            finalize_params_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-finalize-params",
                caps.finalize_params_bytes,
            ),
            freq_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-freq",
                caps.freq_bytes,
            ),
            cand_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-cand",
                caps.cand_bytes,
            ),
            bucket_count_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-bucket-count",
                caps.bucket_bytes,
            ),
            bucket_offset_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-bucket-offset",
                caps.bucket_bytes,
            ),
            bucket_cursor_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-bucket-cursor",
                caps.bucket_bytes,
            ),
            sorted_idx_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-sorted-idx",
                caps.sorted_idx_bytes,
            ),
            emit_desc_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-emit-desc",
                caps.emit_desc_bytes,
            ),
            emit_word_buffer: storage_upload_buffer(
                &runtime.device,
                "cozip-pdeflate-bt-batch-emit-word",
                caps.emit_word_bytes,
            ),
            caps,
        }
    }
}

impl GpuSectionEncodeSlot {
    fn fits(&self, section_count: usize, cmd_cap_bytes: u64) -> bool {
        self.section_count_cap >= section_count && self.cmd_cap_bytes >= cmd_cap_bytes
    }

    fn new(
        runtime: &GpuMatchRuntime,
        match_scratch: &GpuBatchScratch,
        section_count: usize,
        cmd_cap_bytes: u64,
    ) -> Result<Self, PDeflateError> {
        let section_count_u64 =
            u64::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
        let section_offsets_bytes = section_count_u64.saturating_mul(4);
        let section_caps_bytes = section_count_u64.saturating_mul(4);
        let section_params_bytes = 40u64;
        let out_lens_bytes = section_count_u64.saturating_mul(4);
        let section_prefix_bytes = section_count_u64.saturating_mul(4);
        let section_index_cap_bytes = u64::try_from(section_count.saturating_mul(5))
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let out_cmd_bytes = cmd_cap_bytes.max(4);
        let token_cap_count = estimate_token_cap_count(out_cmd_bytes);
        let token_buf_bytes = token_cap_count.saturating_mul(4).max(4);
        let out_cmd_byte_buf_bytes = out_cmd_bytes
            .checked_mul(4)
            .ok_or(PDeflateError::NumericOverflow)?
            .max(4);
        if token_buf_bytes > runtime.max_storage_binding_size
            || out_cmd_byte_buf_bytes > runtime.max_storage_binding_size
        {
            return Err(PDeflateError::Gpu(
                "gpu section encode staging buffer too large".to_string(),
            ));
        }
        let section_offsets_buffer = Arc::new(storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-se-offs",
            section_offsets_bytes.max(4),
        ));
        let section_caps_buffer = Arc::new(storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-se-caps",
            section_caps_bytes.max(4),
        ));
        let section_token_offsets_buffer = Arc::new(storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-se-token-offs",
            section_offsets_bytes.max(4),
        ));
        let section_token_caps_buffer = Arc::new(storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-se-token-caps",
            section_caps_bytes.max(4),
        ));
        let section_params_buffer = Arc::new(storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-se-params",
            section_params_bytes.max(4),
        ));
        let out_lens_buffer = Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-out-lens"),
            size: out_lens_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let out_lens_readback_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-out-lens-readback"),
                size: out_lens_bytes.max(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        let section_prefix_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-prefix"),
                size: section_prefix_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        let section_index_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-index"),
                size: section_index_cap_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        let section_meta_buffer = Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-meta"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let section_token_counts_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-token-counts"),
                size: out_lens_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        let section_token_meta_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-token-meta"),
                size: token_buf_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        let section_token_pos_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-token-pos"),
                size: token_buf_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        let section_token_cmd_offsets_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-token-cmd-offs"),
                size: token_buf_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        let out_cmd_byte_buffer = Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-out-cmd-byte"),
            size: out_cmd_byte_buf_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        let out_cmd_buffer = Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-out-cmd"),
            size: out_cmd_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let readback_meta_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-readback-meta"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        let bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-bg"),
                layout: &runtime.section_encode_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: match_scratch.src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: match_scratch.out_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: section_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_caps_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: out_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: out_cmd_buffer.as_entire_binding(),
                    },
                ],
            });
        let tokenize_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-tokenize-bg"),
                layout: &runtime.section_tokenize_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: match_scratch.src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: match_scratch.out_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: section_token_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_token_caps_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: section_token_counts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: section_token_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: section_token_pos_buffer.as_entire_binding(),
                    },
                ],
            });
        let prefix_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-prefix-bg"),
                layout: &runtime.section_prefix_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: section_token_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: section_token_caps_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_token_counts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: section_token_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: section_token_cmd_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: out_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: section_caps_buffer.as_entire_binding(),
                    },
                ],
            });
        let scatter_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-scatter-bg"),
                layout: &runtime.section_scatter_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: match_scratch.src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: section_token_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: section_token_counts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_token_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: section_token_pos_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: section_token_cmd_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: section_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: section_caps_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: out_cmd_byte_buffer.as_entire_binding(),
                    },
                ],
            });
        let pack_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-pack-bg"),
                layout: &runtime.section_cmd_pack_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out_cmd_byte_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: section_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_cmd_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: section_offsets_buffer.as_entire_binding(),
                    },
                ],
            });
        let meta_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-meta-bg"),
                layout: &runtime.section_meta_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: section_prefix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: section_index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                ],
            });
        Ok(Self {
            section_count_cap: section_count,
            cmd_cap_bytes: out_cmd_bytes,
            token_cap_count,
            section_index_cap_bytes: section_index_cap_bytes.max(4),
            section_offsets_buffer,
            section_caps_buffer,
            section_token_offsets_buffer,
            section_token_caps_buffer,
            section_params_buffer,
            out_lens_buffer,
            out_lens_readback_buffer,
            section_token_counts_buffer,
            section_token_meta_buffer,
            section_token_pos_buffer,
            section_token_cmd_offsets_buffer,
            out_cmd_byte_buffer,
            section_prefix_buffer,
            section_index_buffer,
            section_meta_buffer,
            out_cmd_buffer,
            readback_meta_buffer,
            bind_group,
            tokenize_bind_group,
            prefix_bind_group,
            scatter_bind_group,
            pack_bind_group,
            meta_bind_group,
        })
    }
}

impl GpuSparsePackScratch {
    fn fits(&self, caps: GpuSparsePackScratchCaps) -> bool {
        self.caps.fits(caps)
    }

    fn covers(&self, other: &Self) -> bool {
        self.caps.fits(other.caps)
    }

    fn new(runtime: &GpuMatchRuntime, caps: GpuSparsePackScratchCaps) -> Self {
        let caps = GpuSparsePackScratchCaps {
            desc_bytes: round_capacity_bytes(caps.desc_bytes),
            section_meta_bytes: round_capacity_bytes(caps.section_meta_bytes),
            table_index_bytes: round_capacity_bytes(caps.table_index_bytes),
            table_data_bytes: round_capacity_bytes(caps.table_data_bytes),
            section_index_bytes: round_capacity_bytes(caps.section_index_bytes),
            out_lens_bytes: round_capacity_bytes(caps.out_lens_bytes),
            section_prefix_bytes: round_capacity_bytes(caps.section_prefix_bytes),
            section_offsets_bytes: round_capacity_bytes(caps.section_offsets_bytes),
            out_cmd_bytes: round_capacity_bytes(caps.out_cmd_bytes),
            result_bytes: round_capacity_bytes(caps.result_bytes),
            out_bytes: round_capacity_bytes(caps.out_bytes),
        };
        let desc_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-pack-sparse-batch-desc",
            caps.desc_bytes,
        );
        let section_meta_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-pack-sparse-batch-section-meta",
            caps.section_meta_bytes,
        );
        let table_index_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-pack-sparse-batch-table-index",
            caps.table_index_bytes,
        );
        let table_data_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-pack-sparse-batch-table-data",
            caps.table_data_bytes,
        );
        let section_index_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-pack-sparse-batch-section-index",
            caps.section_index_bytes,
        );
        let out_lens_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-pack-sparse-batch-out-lens",
            caps.out_lens_bytes,
        );
        let section_prefix_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-pack-sparse-batch-section-prefix",
            caps.section_prefix_bytes,
        );
        let section_offsets_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-pack-sparse-batch-section-offsets",
            caps.section_offsets_bytes,
        );
        let out_cmd_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-pack-sparse-batch-out-cmd",
            caps.out_cmd_bytes,
        );
        let result_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-batch-result"),
            size: caps.result_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let result_readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-batch-result-readback"),
            size: caps.result_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-batch-out"),
            size: caps.out_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let payload_readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-batch-payload-readback"),
            size: caps.out_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let prepare_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-pack-sparse-batch-prepare-bg"),
                layout: &runtime.pack_sparse_prepare_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: desc_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: section_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: result_buffer.as_entire_binding(),
                    },
                ],
            });
        Self {
            caps,
            desc_buffer,
            section_meta_buffer,
            table_index_buffer,
            table_data_buffer,
            section_index_buffer,
            out_lens_buffer,
            section_prefix_buffer,
            section_offsets_buffer,
            out_cmd_buffer,
            result_buffer,
            result_readback_buffer,
            out_buffer,
            payload_readback_buffer,
            prepare_bind_group,
        }
    }
}

fn init_runtime() -> Result<GpuMatchRuntime, String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "adapter not found".to_string())?;

    let required_limits = adapter.limits();
    let adapter_features = adapter.features();
    let mut required_features = wgpu::Features::empty();
    if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("cozip-pdeflate-gpu-device"),
            required_features,
            required_limits,
        },
        None,
    ))
    .map_err(|e| format!("request_device failed: {e}"))?;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cozip-pdeflate-match-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-match-shader"),
        source: wgpu::ShaderSource::Wgsl(MATCH_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-match-pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cozip-pdeflate-match-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    let pack_chunk_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-pack-chunk-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let pack_chunk_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-pack-chunk-shader"),
        source: wgpu::ShaderSource::Wgsl(PACK_CHUNK_SHADER.into()),
    });
    let pack_chunk_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-pack-chunk-pl"),
        bind_group_layouts: &[&pack_chunk_bind_group_layout],
        push_constant_ranges: &[],
    });
    let pack_chunk_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cozip-pdeflate-pack-chunk-pipeline"),
        layout: Some(&pack_chunk_layout),
        module: &pack_chunk_shader,
        entry_point: "main",
    });
    let pack_sparse_prepare_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-prepare-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let pack_sparse_prepare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-prepare-shader"),
        source: wgpu::ShaderSource::Wgsl(PACK_CHUNK_SPARSE_PREPARE_SHADER.into()),
    });
    let pack_sparse_prepare_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-prepare-pl"),
            bind_group_layouts: &[&pack_sparse_prepare_bind_group_layout],
            push_constant_ranges: &[],
        });
    let pack_sparse_prepare_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-prepare-pipeline"),
            layout: Some(&pack_sparse_prepare_layout),
            module: &pack_sparse_prepare_shader,
            entry_point: "main",
        });

    let pack_sparse_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let pack_sparse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-shader"),
        source: wgpu::ShaderSource::Wgsl(PACK_CHUNK_SPARSE_SECTION_SHADER.into()),
    });
    let pack_sparse_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-pl"),
        bind_group_layouts: &[&pack_sparse_bind_group_layout],
        push_constant_ranges: &[],
    });
    let pack_sparse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-pipeline"),
        layout: Some(&pack_sparse_layout),
        module: &pack_sparse_shader,
        entry_point: "main",
    });

    let section_encode_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-encode-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let section_encode_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-encode-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_ENCODE_SHADER.into()),
    });
    let section_encode_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-encode-pl"),
        bind_group_layouts: &[&section_encode_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_encode_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-section-encode-pipeline"),
            layout: Some(&section_encode_layout),
            module: &section_encode_shader,
            entry_point: "main",
        });
    let section_tokenize_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-tokenize-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let section_tokenize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-tokenize-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_TOKENIZE_SHADER.into()),
    });
    let section_tokenize_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-tokenize-pl"),
        bind_group_layouts: &[&section_tokenize_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_tokenize_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-section-tokenize-pipeline"),
            layout: Some(&section_tokenize_layout),
            module: &section_tokenize_shader,
            entry_point: "main",
        });
    let section_prefix_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-prefix-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let section_prefix_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-prefix-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_TOKEN_PREFIX_SHADER.into()),
    });
    let section_prefix_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-prefix-pl"),
        bind_group_layouts: &[&section_prefix_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_prefix_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-section-prefix-pipeline"),
            layout: Some(&section_prefix_layout),
            module: &section_prefix_shader,
            entry_point: "main",
        });
    let section_scatter_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-scatter-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let section_scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-scatter-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_TOKEN_SCATTER_SHADER.into()),
    });
    let section_scatter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-scatter-pl"),
        bind_group_layouts: &[&section_scatter_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_scatter_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-section-scatter-pipeline"),
            layout: Some(&section_scatter_layout),
            module: &section_scatter_shader,
            entry_point: "main",
        });
    let section_cmd_pack_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-cmd-pack-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let section_cmd_pack_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-cmd-pack-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_CMD_PACK_SHADER.into()),
    });
    let section_cmd_pack_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-cmd-pack-pl"),
        bind_group_layouts: &[&section_cmd_pack_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_cmd_pack_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-section-cmd-pack-pipeline"),
            layout: Some(&section_cmd_pack_layout),
            module: &section_cmd_pack_shader,
            entry_point: "main",
        });
    let section_meta_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-meta-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let section_meta_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-meta-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_META_SHADER.into()),
    });
    let section_meta_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-meta-pl"),
        bind_group_layouts: &[&section_meta_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_meta_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cozip-pdeflate-section-meta-pipeline"),
        layout: Some(&section_meta_layout),
        module: &section_meta_shader,
        entry_point: "main",
    });
    let build_table_freq_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-freq-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_freq_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-build-table-freq-shader"),
        source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_FREQ_SHADER.into()),
    });
    let build_table_freq_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-build-table-freq-pl"),
        bind_group_layouts: &[&build_table_freq_bind_group_layout],
        push_constant_ranges: &[],
    });
    let build_table_freq_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-freq-pipeline"),
            layout: Some(&build_table_freq_layout),
            module: &build_table_freq_shader,
            entry_point: "main",
        });

    let build_table_candidate_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-candidate-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_candidate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-build-table-candidate-shader"),
        source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_CANDIDATE_SHADER.into()),
    });
    let build_table_candidate_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-candidate-pl"),
            bind_group_layouts: &[&build_table_candidate_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_candidate_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-candidate-pipeline"),
            layout: Some(&build_table_candidate_layout),
            module: &build_table_candidate_shader,
            entry_point: "main",
        });
    let build_table_bucket_count_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-count-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_bucket_count_shader =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-count-shader"),
            source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_BUCKET_COUNT_SHADER.into()),
        });
    let build_table_bucket_count_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-count-pl"),
            bind_group_layouts: &[&build_table_bucket_count_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_bucket_count_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-count-pipeline"),
            layout: Some(&build_table_bucket_count_layout),
            module: &build_table_bucket_count_shader,
            entry_point: "main",
        });
    let build_table_bucket_prefix_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-prefix-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_bucket_prefix_shader =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-prefix-shader"),
            source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_BUCKET_PREFIX_SHADER.into()),
        });
    let build_table_bucket_prefix_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-prefix-pl"),
            bind_group_layouts: &[&build_table_bucket_prefix_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_bucket_prefix_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-prefix-pipeline"),
            layout: Some(&build_table_bucket_prefix_layout),
            module: &build_table_bucket_prefix_shader,
            entry_point: "main",
        });

    let build_table_bucket_scatter_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-scatter-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_bucket_scatter_shader =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-scatter-shader"),
            source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_BUCKET_SCATTER_SHADER.into()),
        });
    let build_table_bucket_scatter_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-scatter-pl"),
            bind_group_layouts: &[&build_table_bucket_scatter_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_bucket_scatter_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-scatter-pipeline"),
            layout: Some(&build_table_bucket_scatter_layout),
            module: &build_table_bucket_scatter_shader,
            entry_point: "main",
        });
    let build_table_finalize_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-build-table-finalize-shader"),
        source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_FINALIZE_SHADER.into()),
    });
    let build_table_finalize_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-pl"),
            bind_group_layouts: &[&build_table_finalize_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_finalize_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-pipeline"),
            layout: Some(&build_table_finalize_layout),
            module: &build_table_finalize_shader,
            entry_point: "main",
        });
    let build_table_finalize_emit_desc_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-emit-desc-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_finalize_emit_shader =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-emit-shader"),
            source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_FINALIZE_EMIT_DESC_SHADER.into()),
        });
    let build_table_finalize_emit_desc_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-emit-desc-pl"),
            bind_group_layouts: &[&build_table_finalize_emit_desc_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_finalize_emit_desc_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-emit-desc-pipeline"),
            layout: Some(&build_table_finalize_emit_desc_layout),
            module: &build_table_finalize_emit_shader,
            entry_point: "main",
        });
    let build_table_finalize_emit_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-emit-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_finalize_emit_shader =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-emit-shader"),
            source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_FINALIZE_EMIT_SHADER.into()),
        });
    let build_table_finalize_emit_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-emit-pl"),
            bind_group_layouts: &[&build_table_finalize_emit_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_finalize_emit_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-emit-pipeline"),
            layout: Some(&build_table_finalize_emit_layout),
            module: &build_table_finalize_emit_shader,
            entry_point: "main",
        });
    let build_table_pack_index_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-pack-index-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_pack_index_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-build-table-pack-index-shader"),
        source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_PACK_INDEX_SHADER.into()),
    });
    let build_table_pack_index_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-pack-index-pl"),
            bind_group_layouts: &[&build_table_pack_index_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_pack_index_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-pack-index-pipeline"),
            layout: Some(&build_table_pack_index_layout),
            module: &build_table_pack_index_shader,
            entry_point: "main",
        });
    let match_prepare_table_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-match-prepare-table-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let match_prepare_table_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-match-prepare-table-shader"),
        source: wgpu::ShaderSource::Wgsl(MATCH_PREPARE_TABLE_SHADER.into()),
    });
    let match_prepare_table_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-match-prepare-table-pl"),
            bind_group_layouts: &[&match_prepare_table_bind_group_layout],
            push_constant_ranges: &[],
        });
    let match_prepare_table_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-match-prepare-table-pipeline"),
            layout: Some(&match_prepare_table_layout),
            module: &match_prepare_table_shader,
            entry_point: "main",
        });
    let decode_v2_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-decode-v2-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let decode_v2_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-decode-v2-shader"),
        source: wgpu::ShaderSource::Wgsl(DECODE_V2_SHADER.into()),
    });
    let decode_v2_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-decode-v2-pl"),
            bind_group_layouts: &[&decode_v2_bind_group_layout],
            push_constant_ranges: &[],
        });
    let decode_v2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cozip-pdeflate-decode-v2-pipeline"),
        layout: Some(&decode_v2_pipeline_layout),
        module: &decode_v2_shader,
        entry_point: "main",
    });
    let decode_v2_batch_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-decode-v2-batch-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let decode_v2_batch_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-decode-v2-batch-shader"),
        source: wgpu::ShaderSource::Wgsl(DECODE_V2_BATCH_SHADER.into()),
    });
    let decode_v2_batch_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-decode-v2-batch-pl"),
            bind_group_layouts: &[&decode_v2_batch_bind_group_layout],
            push_constant_ranges: &[],
        });
    let decode_v2_batch_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-decode-v2-batch-pipeline"),
            layout: Some(&decode_v2_batch_pipeline_layout),
            module: &decode_v2_batch_shader,
            entry_point: "main",
        });
    let pack_sparse_stats_zero_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-stats-zero"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let max_storage_binding_size = u64::from(device.limits().max_storage_buffer_binding_size);

    Ok(GpuMatchRuntime {
        device,
        queue,
        supports_timestamp_query: required_features.contains(wgpu::Features::TIMESTAMP_QUERY),
        bind_group_layout,
        pipeline,
        pack_chunk_bind_group_layout,
        pack_chunk_pipeline,
        pack_sparse_prepare_bind_group_layout,
        pack_sparse_prepare_pipeline,
        pack_sparse_bind_group_layout,
        pack_sparse_pipeline,
        pack_sparse_stats_zero_buffer,
        section_encode_bind_group_layout,
        section_encode_pipeline,
        section_tokenize_bind_group_layout,
        section_tokenize_pipeline,
        section_prefix_bind_group_layout,
        section_prefix_pipeline,
        section_scatter_bind_group_layout,
        section_scatter_pipeline,
        section_cmd_pack_bind_group_layout,
        section_cmd_pack_pipeline,
        section_meta_bind_group_layout,
        section_meta_pipeline,
        build_table_freq_bind_group_layout,
        build_table_freq_pipeline,
        build_table_candidate_bind_group_layout,
        build_table_candidate_pipeline,
        build_table_bucket_count_bind_group_layout,
        build_table_bucket_count_pipeline,
        build_table_bucket_prefix_bind_group_layout,
        build_table_bucket_prefix_pipeline,
        build_table_bucket_scatter_bind_group_layout,
        build_table_bucket_scatter_pipeline,
        build_table_finalize_bind_group_layout,
        build_table_finalize_pipeline,
        build_table_finalize_emit_desc_bind_group_layout,
        build_table_finalize_emit_desc_pipeline,
        build_table_finalize_emit_bind_group_layout,
        build_table_finalize_emit_pipeline,
        build_table_pack_index_bind_group_layout,
        build_table_pack_index_pipeline,
        match_prepare_table_bind_group_layout,
        match_prepare_table_pipeline,
        decode_v2_bind_group_layout,
        decode_v2_pipeline,
        decode_v2_batch_bind_group_layout,
        decode_v2_batch_pipeline,
        max_storage_binding_size,
        scratch_pool: Mutex::new(Vec::new()),
        scratch_hot: Mutex::new(None),
        table_build_batch_pool: Mutex::new(Vec::new()),
        table_build_batch_hot: Mutex::new(None),
        sparse_pack_pool: Mutex::new(Vec::new()),
        sparse_pack_hot: Mutex::new(None),
        decode_v2_true_batch_pool: Mutex::new(Vec::new()),
        decode_slot_pool: Mutex::new(GpuDecodeSlotPool::default()),
    })
}

fn runtime() -> Result<&'static GpuMatchRuntime, PDeflateError> {
    match GPU_RUNTIME.get_or_init(init_runtime) {
        Ok(r) => Ok(r),
        Err(e) => Err(PDeflateError::Gpu(e.clone())),
    }
}

fn acquire_batch_scratch(
    r: &GpuMatchRuntime,
    required_caps: GpuBatchScratchCaps,
) -> Result<GpuBatchScratch, PDeflateError> {
    if let Ok(mut hot) = r.scratch_hot.lock() {
        if let Some(scratch) = hot.take() {
            if scratch.fits(required_caps) {
                return Ok(scratch);
            }
            if let Ok(mut pool) = r.scratch_pool.lock() {
                if pool.len() < GPU_SCRATCH_POOL_LIMIT {
                    pool.push(scratch);
                }
            }
        }
    }
    let mut pool = r
        .scratch_pool
        .lock()
        .map_err(|_| PDeflateError::Gpu("gpu scratch pool mutex poisoned".to_string()))?;
    if let Some(pos) = pool.iter().position(|s| s.fits(required_caps)) {
        Ok(pool.swap_remove(pos))
    } else {
        Ok(GpuBatchScratch::new(r, required_caps))
    }
}

fn release_batch_scratch(r: &GpuMatchRuntime, scratch: GpuBatchScratch) {
    let mut spill = Some(scratch);
    if let Ok(mut hot) = r.scratch_hot.lock() {
        match hot.take() {
            None => {
                *hot = spill.take();
            }
            Some(existing) => {
                if let Some(candidate) = spill.take() {
                    if candidate.covers(&existing) {
                        *hot = Some(candidate);
                        spill = Some(existing);
                    } else {
                        *hot = Some(existing);
                        spill = Some(candidate);
                    }
                } else {
                    *hot = Some(existing);
                }
            }
        }
    }
    if let Some(s) = spill {
        if let Ok(mut pool) = r.scratch_pool.lock() {
            if pool.len() < GPU_SCRATCH_POOL_LIMIT {
                pool.push(s);
            }
        }
    }
}

fn acquire_table_build_batch_scratch(
    r: &GpuMatchRuntime,
    required_caps: GpuTableBuildBatchScratchCaps,
) -> Result<GpuTableBuildBatchScratch, PDeflateError> {
    if let Ok(mut hot) = r.table_build_batch_hot.lock() {
        if let Some(scratch) = hot.take() {
            if scratch.caps.fits(required_caps) {
                return Ok(scratch);
            }
            if let Ok(mut pool) = r.table_build_batch_pool.lock() {
                if pool.len() < GPU_TABLE_BUILD_BATCH_POOL_LIMIT {
                    pool.push(scratch);
                }
            }
        }
    }
    let mut pool = r
        .table_build_batch_pool
        .lock()
        .map_err(|_| PDeflateError::Gpu("gpu table-build batch pool mutex poisoned".to_string()))?;
    if let Some(pos) = pool.iter().position(|s| s.caps.fits(required_caps)) {
        Ok(pool.swap_remove(pos))
    } else {
        Ok(GpuTableBuildBatchScratch::new(r, required_caps))
    }
}

fn release_table_build_batch_scratch(r: &GpuMatchRuntime, scratch: GpuTableBuildBatchScratch) {
    let mut spill = Some(scratch);
    if let Ok(mut hot) = r.table_build_batch_hot.lock() {
        match hot.take() {
            None => *hot = spill.take(),
            Some(existing) => {
                if let Some(candidate) = spill.take() {
                    if candidate.caps.covers(existing.caps) {
                        *hot = Some(candidate);
                        spill = Some(existing);
                    } else {
                        *hot = Some(existing);
                        spill = Some(candidate);
                    }
                } else {
                    *hot = Some(existing);
                }
            }
        }
    }
    if let Some(s) = spill {
        if let Ok(mut pool) = r.table_build_batch_pool.lock() {
            if pool.len() < GPU_TABLE_BUILD_BATCH_POOL_LIMIT {
                pool.push(s);
            }
        }
    }
}

fn acquire_sparse_pack_scratch(
    r: &GpuMatchRuntime,
    caps: GpuSparsePackScratchCaps,
) -> Result<GpuSparsePackScratch, PDeflateError> {
    if let Ok(mut hot) = r.sparse_pack_hot.lock() {
        if let Some(scratch) = hot.take() {
            if scratch.fits(caps) {
                return Ok(scratch);
            }
            if let Ok(mut pool) = r.sparse_pack_pool.lock() {
                if pool.len() < GPU_SPARSE_PACK_POOL_LIMIT {
                    pool.push(scratch);
                }
            }
        }
    }
    let mut pool = r
        .sparse_pack_pool
        .lock()
        .map_err(|_| PDeflateError::Gpu("gpu sparse pack pool mutex poisoned".to_string()))?;
    if let Some(pos) = pool.iter().position(|s| s.fits(caps)) {
        Ok(pool.swap_remove(pos))
    } else {
        Ok(GpuSparsePackScratch::new(r, caps))
    }
}

fn release_sparse_pack_scratch(r: &GpuMatchRuntime, scratch: GpuSparsePackScratch) {
    let mut spill = Some(scratch);
    if let Ok(mut hot) = r.sparse_pack_hot.lock() {
        match hot.take() {
            None => {
                *hot = spill.take();
            }
            Some(existing) => {
                if let Some(candidate) = spill.take() {
                    if candidate.covers(&existing) {
                        *hot = Some(candidate);
                        spill = Some(existing);
                    } else {
                        *hot = Some(existing);
                        spill = Some(candidate);
                    }
                } else {
                    *hot = Some(existing);
                }
            }
        }
    }
    if let Some(s) = spill {
        if let Ok(mut pool) = r.sparse_pack_pool.lock() {
            if pool.len() < GPU_SPARSE_PACK_POOL_LIMIT {
                pool.push(s);
            }
        }
    }
}

fn create_gpu_decode_v2_true_batch_scratch(
    runtime: &GpuMatchRuntime,
    required: GpuDecodeV2TrueBatchScratchCaps,
) -> GpuDecodeV2TrueBatchScratch {
    let caps = GpuDecodeV2TrueBatchScratchCaps {
        desc_bytes: round_capacity_bytes(required.desc_bytes),
        section_meta_bytes: round_capacity_bytes(required.section_meta_bytes),
        cmd_bytes: round_capacity_bytes(required.cmd_bytes),
        table_bytes: round_capacity_bytes(required.table_bytes),
        out_bytes: round_capacity_bytes(required.out_bytes),
        error_bytes: round_capacity_bytes(required.error_bytes),
    };
    let desc_buffer = storage_upload_buffer(
        &runtime.device,
        "cozip-pdeflate-decode-v2-batch-desc-pool",
        caps.desc_bytes,
    );
    let section_meta_buffer = storage_upload_buffer(
        &runtime.device,
        "cozip-pdeflate-decode-v2-batch-section-meta-pool",
        caps.section_meta_bytes,
    );
    let cmd_buffer = storage_upload_buffer(
        &runtime.device,
        "cozip-pdeflate-decode-v2-batch-cmd-pool",
        caps.cmd_bytes,
    );
    let table_buffer = storage_upload_buffer(
        &runtime.device,
        "cozip-pdeflate-decode-v2-batch-table-pool",
        caps.table_bytes,
    );
    let out_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-decode-v2-batch-out-pool"),
        size: caps.out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let error_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-decode-v2-batch-error-pool"),
        size: caps.error_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-decode-v2-batch-out-readback-pool"),
        size: caps.out_bytes.saturating_add(caps.error_bytes),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-decode-v2-batch-bg-pool"),
            layout: &runtime.decode_v2_batch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: desc_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: section_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cmd_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: error_buffer.as_entire_binding(),
                },
            ],
        });
    let (ts_query_set, ts_resolve_buffer, ts_readback_buffer) = if runtime.supports_timestamp_query
    {
        (
            Some(runtime.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("cozip-pdeflate-decode-v2-batch-ts-query-pool"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })),
            Some(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-decode-v2-batch-ts-resolve-pool"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })),
            Some(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-decode-v2-batch-ts-readback-pool"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })),
        )
    } else {
        (None, None, None)
    };
    GpuDecodeV2TrueBatchScratch {
        caps,
        desc_buffer,
        section_meta_buffer,
        cmd_buffer,
        table_buffer,
        out_buffer,
        error_buffer,
        out_readback_buffer,
        bind_group,
        ts_query_set,
        ts_resolve_buffer,
        ts_readback_buffer,
    }
}

fn acquire_gpu_decode_v2_true_batch_scratch(
    runtime: &GpuMatchRuntime,
    required: GpuDecodeV2TrueBatchScratchCaps,
) -> Result<GpuDecodeV2TrueBatchScratch, PDeflateError> {
    let mut pool = runtime.decode_v2_true_batch_pool.lock().map_err(|_| {
        PDeflateError::Gpu("gpu decode-v2 true-batch pool mutex poisoned".to_string())
    })?;
    if let Some(pos) = pool.iter().position(|s| s.caps.fits(required)) {
        Ok(pool.swap_remove(pos))
    } else {
        Ok(create_gpu_decode_v2_true_batch_scratch(runtime, required))
    }
}

fn release_gpu_decode_v2_true_batch_scratch(
    runtime: &GpuMatchRuntime,
    scratch: GpuDecodeV2TrueBatchScratch,
) {
    if let Ok(mut pool) = runtime.decode_v2_true_batch_pool.lock() {
        if pool.len() < GPU_DECODE_V2_TRUE_BATCH_POOL_LIMIT {
            pool.push(scratch);
            return;
        }
        if let Some((idx, _)) = pool
            .iter()
            .enumerate()
            .find(|(_, existing)| scratch.caps.covers(&existing.caps))
        {
            pool[idx] = scratch;
        }
    }
}

pub(crate) fn lock_submit_stream() -> Result<std::sync::MutexGuard<'static, ()>, PDeflateError> {
    GPU_SUBMIT_STREAM_MUTEX
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| PDeflateError::Gpu("gpu submit stream mutex poisoned".to_string()))
}

fn pack_bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    if bytes.is_empty() {
        return Vec::new();
    }
    let mut words = vec![0u32; bytes.len().div_ceil(4)];
    for (i, &b) in bytes.iter().enumerate() {
        let wi = i >> 2;
        let shift = (i & 3) << 3;
        words[wi] |= u32::from(b) << shift;
    }
    words
}

fn readback_table_sizes_from_meta_buffer(
    runtime: &GpuMatchRuntime,
    table_meta_buffer: &wgpu::Buffer,
    table_meta_offset: u64,
    max_entries: usize,
    table_data_bytes_cap: usize,
) -> Result<(usize, usize, usize), PDeflateError> {
    if max_entries == 0 {
        return Ok((0, 0, 0));
    }
    let data_total_idx = 1usize
        .checked_add(
            max_entries
                .checked_mul(2)
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .ok_or(PDeflateError::NumericOverflow)?;
    let readback_sizes = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-readback-sizes"),
        size: 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-bt-size-readback-encoder"),
        });
    encoder.copy_buffer_to_buffer(table_meta_buffer, table_meta_offset, &readback_sizes, 0, 4);
    encoder.copy_buffer_to_buffer(
        table_meta_buffer,
        table_meta_offset
            .checked_add(
                u64::try_from(data_total_idx)
                    .map_err(|_| PDeflateError::NumericOverflow)?
                    .saturating_mul(4),
            )
            .ok_or(PDeflateError::NumericOverflow)?,
        &readback_sizes,
        4,
        4,
    );
    let submission = runtime.queue.submit(Some(encoder.finish()));

    let size_slice = readback_sizes.slice(..8);
    let (tx_size, rx_size) = mpsc::channel();
    size_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_size.send(res);
    });
    runtime.device.poll(wgpu::Maintain::wait_for(submission));
    rx_size
        .recv()
        .map_err(|_| PDeflateError::Gpu("gpu build-table size map channel closed".to_string()))?
        .map_err(|e| PDeflateError::Gpu(format!("gpu build-table size map failed: {e}")))?;
    let mapped_sizes = size_slice.get_mapped_range();
    let size_words: &[u32] = bytemuck::cast_slice(&mapped_sizes);
    let out_count = usize::try_from(*size_words.first().unwrap_or(&0))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .min(max_entries);
    let data_total = usize::try_from(*size_words.get(1).unwrap_or(&0))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .min(table_data_bytes_cap);
    drop(mapped_sizes);
    readback_sizes.unmap();
    Ok((out_count, out_count, data_total))
}

fn write_u16_le_fixed(dst: &mut [u8], off: usize, v: u16) {
    let b = v.to_le_bytes();
    dst[off..off + 2].copy_from_slice(&b);
}

fn write_u32_le_fixed(dst: &mut [u8], off: usize, v: u32) {
    let b = v.to_le_bytes();
    dst[off..off + 4].copy_from_slice(&b);
}

fn build_identity_huff_lut_block_for_pack() -> Vec<u8> {
    // Keep this in sync with decode-v2 packed entry format:
    // kind(2b) | bit_len(8b, <<2) | symbol(22b, <<10)
    const SYMBOL_COUNT: usize = 256;
    const ROOT_BITS: u8 = 8;
    const MAX_CODE_BITS: u8 = 8;
    const ENTRY_KIND_SYMBOL: u32 = 1;

    let mut out = Vec::with_capacity(12 + SYMBOL_COUNT * 4);
    out.extend_from_slice(&(SYMBOL_COUNT as u16).to_le_bytes());
    out.push(ROOT_BITS);
    out.push(MAX_CODE_BITS);
    out.extend_from_slice(&(SYMBOL_COUNT as u32).to_le_bytes()); // root_len
    out.extend_from_slice(&0u32.to_le_bytes()); // sub_len
    for symbol in 0..SYMBOL_COUNT {
        let packed = ENTRY_KIND_SYMBOL | (u32::from(ROOT_BITS) << 2) | ((symbol as u32) << 10);
        out.extend_from_slice(&packed.to_le_bytes());
    }
    out
}

fn finalize_sparse_packed_chunk_huffman(
    payload: &mut Vec<u8>,
    huffman_enabled: bool,
) -> Result<(), PDeflateError> {
    if payload.len() < PACK_CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream(
            "gpu sparse packed chunk header truncated",
        ));
    }
    let table_data_offset = usize::try_from(read_le_u32_at(payload, 20)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let huff_lut_offset = usize::try_from(read_le_u32_at(payload, 24)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let section_index_offset = usize::try_from(read_le_u32_at(payload, 28)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let section_bitstream_offset = usize::try_from(read_le_u32_at(payload, 32)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    if !(table_data_offset <= huff_lut_offset
        && huff_lut_offset <= section_index_offset
        && section_index_offset <= section_bitstream_offset
        && section_bitstream_offset <= payload.len())
    {
        return Err(PDeflateError::InvalidStream(
            "gpu sparse packed chunk offsets invalid",
        ));
    }

    let flags = if huffman_enabled {
        super::CHUNK_FLAG_HUFFMAN
    } else {
        0
    };
    write_u16_le_fixed(payload, 6, flags);

    if !huffman_enabled {
        return Ok(());
    }

    // Already has LUT payload.
    if huff_lut_offset < section_index_offset {
        return Ok(());
    }

    let lut = build_identity_huff_lut_block_for_pack();
    let lut_len = lut.len();
    payload.splice(huff_lut_offset..huff_lut_offset, lut);
    let new_section_index_offset = section_index_offset
        .checked_add(lut_len)
        .ok_or(PDeflateError::NumericOverflow)?;
    let new_section_bitstream_offset = section_bitstream_offset
        .checked_add(lut_len)
        .ok_or(PDeflateError::NumericOverflow)?;
    write_u32_le_fixed(
        payload,
        28,
        u32::try_from(new_section_index_offset).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    write_u32_le_fixed(
        payload,
        32,
        u32::try_from(new_section_bitstream_offset).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    Ok(())
}

pub(crate) fn pack_chunk_payload(
    chunk_len: usize,
    table_count: usize,
    section_count: usize,
    table_index: &[u8],
    table_data: &[u8],
    section_index: &[u8],
    section_cmd: &[u8],
) -> Result<(Vec<u8>, GpuMatchProfile), PDeflateError> {
    let r = runtime()?;
    let t_total = Instant::now();

    let table_index_offset = PACK_CHUNK_HEADER_SIZE;
    let table_data_offset = table_index_offset
        .checked_add(table_index.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let huff_lut_offset = table_data_offset
        .checked_add(table_data.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    // T01: reserve LUT area in header. Actual bytes are added in T03.
    let section_index_offset = huff_lut_offset;
    let section_bitstream_offset = section_index_offset
        .checked_add(section_index.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let total_len = section_bitstream_offset
        .checked_add(section_cmd.len())
        .ok_or(PDeflateError::NumericOverflow)?;

    let table_count_u16 = u16::try_from(table_count).map_err(|_| PDeflateError::NumericOverflow)?;
    let section_count_u16 =
        u16::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
    let chunk_len_u32 = u32::try_from(chunk_len).map_err(|_| PDeflateError::NumericOverflow)?;
    let table_index_offset_u32 =
        u32::try_from(table_index_offset).map_err(|_| PDeflateError::NumericOverflow)?;
    let table_data_offset_u32 =
        u32::try_from(table_data_offset).map_err(|_| PDeflateError::NumericOverflow)?;
    let huff_lut_offset_u32 =
        u32::try_from(huff_lut_offset).map_err(|_| PDeflateError::NumericOverflow)?;
    let section_index_offset_u32 =
        u32::try_from(section_index_offset).map_err(|_| PDeflateError::NumericOverflow)?;
    let section_bitstream_offset_u32 =
        u32::try_from(section_bitstream_offset).map_err(|_| PDeflateError::NumericOverflow)?;

    let mut header = [0u8; PACK_CHUNK_HEADER_SIZE];
    header[0..4].copy_from_slice(&PACK_CHUNK_MAGIC);
    write_u16_le_fixed(&mut header, 4, PACK_CHUNK_VERSION);
    write_u16_le_fixed(&mut header, 6, 0);
    write_u32_le_fixed(&mut header, 8, chunk_len_u32);
    write_u16_le_fixed(&mut header, 12, table_count_u16);
    write_u16_le_fixed(&mut header, 14, section_count_u16);
    write_u32_le_fixed(&mut header, 16, table_index_offset_u32);
    write_u32_le_fixed(&mut header, 20, table_data_offset_u32);
    write_u32_le_fixed(&mut header, 24, huff_lut_offset_u32);
    write_u32_le_fixed(&mut header, 28, section_index_offset_u32);
    write_u32_le_fixed(&mut header, 32, section_bitstream_offset_u32);

    let header_words = pack_bytes_to_words(&header);
    let table_index_words = pack_bytes_to_words(table_index);
    let table_data_words = pack_bytes_to_words(table_data);
    let section_index_words = pack_bytes_to_words(section_index);
    let section_cmd_words = pack_bytes_to_words(section_cmd);

    let total_words = total_len.div_ceil(4);
    let total_words_u32 = u32::try_from(total_words).map_err(|_| PDeflateError::NumericOverflow)?;
    let total_len_u32 = u32::try_from(total_len).map_err(|_| PDeflateError::NumericOverflow)?;
    let params: [u32; 11] = [
        total_len_u32,
        u32::try_from(PACK_CHUNK_HEADER_SIZE).map_err(|_| PDeflateError::NumericOverflow)?,
        table_index_offset_u32,
        u32::try_from(table_index.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        table_data_offset_u32,
        u32::try_from(table_data.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        section_index_offset_u32,
        u32::try_from(section_index.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        section_bitstream_offset_u32,
        u32::try_from(section_cmd.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        total_words_u32,
    ];

    let total_bytes = u64::try_from(total_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4)
        .max(4);
    if total_bytes > r.max_storage_binding_size {
        return Err(PDeflateError::Gpu(format!(
            "gpu chunk pack output too large ({} > {})",
            total_bytes, r.max_storage_binding_size
        )));
    }

    let header_bytes = u64::try_from(header_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_index_bytes = u64::try_from(table_index_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_data_bytes = u64::try_from(table_data_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let section_index_bytes = u64::try_from(section_index_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let section_cmd_bytes = u64::try_from(section_cmd_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let params_bytes = u64::try_from(params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);

    let params_buffer =
        storage_upload_buffer(&r.device, "cozip-pdeflate-pack-params", params_bytes);
    let header_buffer =
        storage_upload_buffer(&r.device, "cozip-pdeflate-pack-header", header_bytes);
    let table_index_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-pack-table-index",
        table_index_bytes,
    );
    let table_data_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-pack-table-data",
        table_data_bytes,
    );
    let section_index_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-pack-section-index",
        section_index_bytes,
    );
    let section_cmd_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-pack-section-cmd",
        section_cmd_bytes,
    );
    let out_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-pack-out"),
        size: total_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-pack-readback"),
        size: total_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-pack-bg"),
        layout: &r.pack_chunk_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: header_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: table_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: table_data_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: section_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: section_cmd_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: out_buffer.as_entire_binding(),
            },
        ],
    });

    let t_upload = Instant::now();
    r.queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));
    r.queue
        .write_buffer(&header_buffer, 0, bytemuck::cast_slice(&header_words));
    r.queue.write_buffer(
        &table_index_buffer,
        0,
        bytemuck::cast_slice(&table_index_words),
    );
    r.queue.write_buffer(
        &table_data_buffer,
        0,
        bytemuck::cast_slice(&table_data_words),
    );
    r.queue.write_buffer(
        &section_index_buffer,
        0,
        bytemuck::cast_slice(&section_index_words),
    );
    r.queue.write_buffer(
        &section_cmd_buffer,
        0,
        bytemuck::cast_slice(&section_cmd_words),
    );
    let upload_ms = elapsed_ms(t_upload);

    let groups_x = total_words_u32.div_ceil(256).max(1);
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-pack-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.pack_chunk_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buffer, 0, &readback_buffer, 0, total_bytes);
    let submission = r.queue.submit(Some(encoder.finish()));

    let t_wait = Instant::now();
    let slice = readback_buffer.slice(..total_bytes);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    r.device.poll(wgpu::Maintain::wait_for(submission));
    rx.recv()
        .map_err(|_| PDeflateError::Gpu("gpu chunk pack map channel closed".to_string()))?
        .map_err(|e| PDeflateError::Gpu(format!("gpu chunk pack map failed: {e}")))?;
    let wait_ms = elapsed_ms(t_wait);

    let t_copy = Instant::now();
    let mapped = slice.get_mapped_range();
    let mut out = vec![0u8; total_len];
    out.copy_from_slice(&mapped[..total_len]);
    drop(mapped);
    readback_buffer.unmap();
    let map_copy_ms = elapsed_ms(t_copy);

    Ok((
        out,
        GpuMatchProfile {
            upload_ms,
            wait_ms,
            map_copy_ms,
            total_ms: elapsed_ms(t_total),
        },
    ))
}

fn write_varint_u32(dst: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        dst.push(((v & 0x7f) as u8) | 0x80);
        v >>= 7;
    }
    dst.push((v & 0x7f) as u8);
}

pub(crate) fn pack_chunk_payload_from_device_sparse(
    chunk_len: usize,
    section_count: usize,
    table: &GpuPackedTableDevice,
    section: &GpuSectionCmdDeviceOutput,
) -> Result<(Vec<u8>, usize, GpuMatchProfile), PDeflateError> {
    let inputs = [GpuSparsePackInput {
        chunk_len,
        section_count,
        huffman_encode_enabled: false,
        table,
        section,
    }];
    let (mut packed, profile, _) = pack_chunk_payload_from_device_sparse_batch(&inputs)?;
    let first = packed.drain(..).next().ok_or(PDeflateError::Gpu(
        "gpu sparse pack single result missing".to_string(),
    ))?;
    Ok((first.payload, first.table_count, profile))
}

pub(crate) fn pack_chunk_payload_from_device_sparse_batch(
    inputs: &[GpuSparsePackInput<'_>],
) -> Result<
    (
        Vec<GpuSparsePackChunkOutput>,
        GpuMatchProfile,
        GpuSparsePackBatchProfile,
    ),
    PDeflateError,
> {
    if inputs.is_empty() {
        return Ok((
            Vec::new(),
            GpuMatchProfile::default(),
            GpuSparsePackBatchProfile::default(),
        ));
    }
    for input in inputs {
        if input.section.section_count != input.section_count {
            return Err(PDeflateError::InvalidOptions(
                "gpu sparse pack section_count mismatch",
            ));
        }
    }

    struct SparsePackPrepared {
        chunk_len: usize,
        huffman_encode_enabled: bool,
        section_count: usize,
        table_data_stream_offset: usize,
        section_index_stream_offset: usize,
        section_index_cap_len: usize,
        section_cmd_cap_len: usize,
        total_len_cap: usize,
        total_bytes_cap: u64,
        table_count: usize,
        table_index_len: usize,
        table_data_len: usize,
    }

    #[derive(Clone, Copy)]
    struct SparsePackHostJob {
        section_meta_words_off: usize,
        table_index_off: usize,
        table_data_off: usize,
        section_index_off: usize,
        out_lens_words_off: usize,
        section_prefix_words_off: usize,
        section_offsets_words_off: usize,
        out_cmd_off: usize,
        out_base_word: usize,
    }

    let r = runtime()?;
    let t_total = Instant::now();
    let mut upload_ms = 0.0_f64;
    let mut wait_ms = 0.0_f64;
    let mut map_copy_ms = 0.0_f64;
    let lens_bytes_total = inputs.iter().fold(0u64, |acc, input| {
        acc.saturating_add(input.section.out_lens_bytes.max(4))
    });
    let out_cmd_bytes_total = inputs.iter().fold(0u64, |acc, input| {
        acc.saturating_add(input.section.out_cmd_bytes.max(4))
    });

    let lens_submit_ms = 0.0_f64;
    let lens_wait_ms;
    let lens_copy_ms;

    let t_prepare = Instant::now();
    let packed_tables: Vec<&GpuPackedTableDevice> =
        inputs.iter().map(|input| input.table).collect();
    let (resolved_table_sizes, resolve_profile) = resolve_packed_table_sizes_batch(&packed_tables)?;
    let table_size_resolve_ms = resolve_profile.total_ms;
    let mut prepared = Vec::<SparsePackPrepared>::with_capacity(inputs.len());
    for (input, &(table_count, table_index_len, table_data_len)) in
        inputs.iter().zip(resolved_table_sizes.iter())
    {
        let table_index_offset = PACK_CHUNK_HEADER_SIZE;
        let table_data_stream_offset = table_index_offset
            .checked_add(table_index_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        let section_index_stream_offset = table_data_stream_offset
            .checked_add(table_data_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        let section_index_cap_len = usize::try_from(input.section.section_index_cap_bytes)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let section_cmd_cap_len = usize::try_from(input.section.out_cmd_bytes.max(4))
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let section_cmd_offset_cap = section_index_stream_offset
            .checked_add(section_index_cap_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        let total_len_cap = section_cmd_offset_cap
            .checked_add(section_cmd_cap_len)
            .ok_or(PDeflateError::NumericOverflow)?;

        let total_words = total_len_cap.div_ceil(4);
        let total_bytes = u64::try_from(total_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4)
            .max(4);
        if total_bytes > r.max_storage_binding_size {
            return Err(PDeflateError::Gpu(format!(
                "gpu sparse chunk pack output too large ({} > {})",
                total_bytes, r.max_storage_binding_size
            )));
        }

        prepared.push(SparsePackPrepared {
            chunk_len: input.chunk_len,
            huffman_encode_enabled: input.huffman_encode_enabled,
            section_count: input.section_count,
            table_data_stream_offset,
            section_index_stream_offset,
            section_index_cap_len,
            section_cmd_cap_len,
            total_len_cap,
            total_bytes_cap: total_bytes,
            table_count,
            table_index_len,
            table_data_len,
        });
    }
    let prepare_ms = elapsed_ms(t_prepare);
    let prepare_misc_ms = (prepare_ms - table_size_resolve_ms).max(0.0);

    let mut section_meta_total_bytes = 0u64;
    let mut table_index_total_bytes = 0u64;
    let mut table_data_total_bytes = 0u64;
    let mut section_index_total_bytes = 0u64;
    let mut out_lens_total_bytes = 0u64;
    let mut section_prefix_total_bytes = 0u64;
    let mut section_offsets_total_bytes = 0u64;
    let mut out_cmd_total_bytes = 0u64;
    let mut out_total_bytes = 0u64;
    let mut host_jobs = Vec::<SparsePackHostJob>::with_capacity(inputs.len());
    for (input, prep) in inputs.iter().zip(prepared.iter()) {
        let section_meta_words_off = usize::try_from(section_meta_total_bytes / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let table_index_off = align_up4(
            usize::try_from(table_index_total_bytes).map_err(|_| PDeflateError::NumericOverflow)?,
        );
        let table_data_off = align_up4(
            usize::try_from(table_data_total_bytes).map_err(|_| PDeflateError::NumericOverflow)?,
        );
        let section_index_off = usize::try_from(section_index_total_bytes)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let out_lens_words_off = usize::try_from(out_lens_total_bytes / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let section_prefix_words_off = usize::try_from(section_prefix_total_bytes / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let section_offsets_words_off = usize::try_from(section_offsets_total_bytes / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let out_cmd_off =
            usize::try_from(out_cmd_total_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
        let out_base_word = usize::try_from(out_total_bytes / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        host_jobs.push(SparsePackHostJob {
            section_meta_words_off,
            table_index_off,
            table_data_off,
            section_index_off,
            out_lens_words_off,
            section_prefix_words_off,
            section_offsets_words_off,
            out_cmd_off,
            out_base_word,
        });
        section_meta_total_bytes = section_meta_total_bytes.saturating_add(16);
        table_index_total_bytes = u64::try_from(
            table_index_off
                .checked_add(align_up4(prep.table_index_len))
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .map_err(|_| PDeflateError::NumericOverflow)?;
        table_data_total_bytes = u64::try_from(
            table_data_off
                .checked_add(align_up4(prep.table_data_len))
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .map_err(|_| PDeflateError::NumericOverflow)?;
        section_index_total_bytes =
            section_index_total_bytes.saturating_add(input.section.section_index_cap_bytes.max(4));
        out_lens_total_bytes =
            out_lens_total_bytes.saturating_add(input.section.out_lens_bytes.max(4));
        section_prefix_total_bytes =
            section_prefix_total_bytes.saturating_add(input.section.out_lens_bytes.max(4));
        section_offsets_total_bytes =
            section_offsets_total_bytes.saturating_add(input.section.out_lens_bytes.max(4));
        out_cmd_total_bytes =
            out_cmd_total_bytes.saturating_add(input.section.out_cmd_bytes.max(4));
        out_total_bytes = out_total_bytes.saturating_add(prep.total_bytes_cap.max(4));
    }

    let t_scratch_acquire = Instant::now();
    let scratch = acquire_sparse_pack_scratch(
        r,
        GpuSparsePackScratchCaps {
            desc_bytes: u64::try_from(
                4usize.saturating_add(
                    inputs
                        .len()
                        .saturating_mul(GPU_SPARSE_PACK_BATCH_DESC_WORDS * 4),
                ),
            )
            .map_err(|_| PDeflateError::NumericOverflow)?,
            section_meta_bytes: section_meta_total_bytes.max(4),
            table_index_bytes: table_index_total_bytes.max(4),
            table_data_bytes: table_data_total_bytes.max(4),
            section_index_bytes: section_index_total_bytes.max(4),
            out_lens_bytes: out_lens_total_bytes.max(4),
            section_prefix_bytes: section_prefix_total_bytes.max(4),
            section_offsets_bytes: section_offsets_total_bytes.max(4),
            out_cmd_bytes: out_cmd_total_bytes.max(4),
            result_bytes: u64::try_from(
                inputs
                    .len()
                    .saturating_mul(GPU_SPARSE_PACK_BATCH_RESULT_WORDS * 4),
            )
            .map_err(|_| PDeflateError::NumericOverflow)?
            .max(4),
            out_bytes: out_total_bytes.max(4),
        },
    )?;
    let scratch_acquire_ms = elapsed_ms(t_scratch_acquire);

    let sparse_probe = sparse_probe_enabled();
    let sparse_lens_wait_probe = sparse_lens_wait_probe_enabled();
    let sparse_wait_attr_probe = sparse_wait_attribution_probe_enabled();
    let sparse_kernel_ts_probe = sparse_kernel_ts_probe_enabled() && r.supports_timestamp_query;
    if sparse_kernel_ts_probe_enabled() && !r.supports_timestamp_query {
        let _ = SPARSE_KERNEL_TS_PROBE_UNSUPPORTED_LOGGED.get_or_init(|| {
            eprintln!(
                "[cozip_pdeflate][timing][sparse-kernel-ts] status=disabled reason=timestamp_query_not_supported"
            );
        });
    }
    let mut sparse_prebuild_build_id_min = u64::MAX;
    let mut sparse_prebuild_build_id_max = 0u64;
    let mut sparse_prebuild_submit_seq_min = u64::MAX;
    let mut sparse_prebuild_submit_seq_max = 0u64;
    let mut sparse_latest_build_submit: Option<(u64, u64, wgpu::SubmissionIndex)> = None;
    let mut sparse_build_ids = Vec::<u64>::with_capacity(inputs.len());
    for input in inputs {
        let table = input.table;
        if table.build_id != 0 {
            sparse_prebuild_build_id_min = sparse_prebuild_build_id_min.min(table.build_id);
            sparse_prebuild_build_id_max = sparse_prebuild_build_id_max.max(table.build_id);
            sparse_build_ids.push(table.build_id);
        }
        if table.build_submit_seq != 0 {
            sparse_prebuild_submit_seq_min =
                sparse_prebuild_submit_seq_min.min(table.build_submit_seq);
            sparse_prebuild_submit_seq_max =
                sparse_prebuild_submit_seq_max.max(table.build_submit_seq);
        }
        if let Some(submit_index) = table.build_submit_index.clone() {
            let should_replace = sparse_latest_build_submit
                .as_ref()
                .map(|(seq, _, _)| table.build_submit_seq > *seq)
                .unwrap_or(true);
            if should_replace {
                sparse_latest_build_submit =
                    Some((table.build_submit_seq, table.build_id, submit_index));
            }
        }
    }
    let sparse_stats_buffer = if sparse_probe {
        let buf = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-stats"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let zeros = [0u8; 32];
        r.queue.write_buffer(&buf, 0, &zeros);
        Some(buf)
    } else {
        None
    };
    let sparse_stats_readback_buffer = if sparse_probe {
        Some(r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-stats-readback"),
            size: 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    } else {
        None
    };
    let sparse_ts_query_count = if sparse_kernel_ts_probe { 4 } else { 0 };
    let mut sparse_ts_prepare_kernel_ms = 0.0_f64;
    let mut sparse_ts_kernel_ms = 0.0_f64;
    let mut sparse_ts_parse_ms = 0.0_f64;
    let mut sparse_ts_map_status: Option<String> = None;
    let mut sparse_ts_query_set = None;
    let mut sparse_ts_resolve_buffer = None;
    let mut sparse_ts_readback_buffer = None;
    if sparse_kernel_ts_probe && sparse_ts_query_count > 0 {
        let ts_bytes = u64::from(sparse_ts_query_count).saturating_mul(8).max(8);
        sparse_ts_query_set = Some(r.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-ts-qs"),
            ty: wgpu::QueryType::Timestamp,
            count: sparse_ts_query_count,
        }));
        sparse_ts_resolve_buffer = Some(r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-ts-resolve"),
            size: ts_bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        sparse_ts_readback_buffer = Some(r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-ts-readback"),
            size: ts_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }

    let mut desc_words =
        vec![0u32; 1usize.saturating_add(inputs.len() * GPU_SPARSE_PACK_BATCH_DESC_WORDS)];
    desc_words[0] = u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?;
    for (idx, (prep, host)) in prepared.iter().zip(host_jobs.iter()).enumerate() {
        let base = 1 + idx * GPU_SPARSE_PACK_BATCH_DESC_WORDS;
        desc_words[base] =
            u32::try_from(prep.section_count).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 1] =
            u32::try_from(prep.chunk_len).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 2] =
            u32::try_from(prep.table_count).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 3] =
            u32::try_from(host.table_index_off).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 4] =
            u32::try_from(prep.table_index_len).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 5] =
            u32::try_from(host.table_data_off).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 6] =
            u32::try_from(prep.table_data_len).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 7] = u32::try_from(host.section_meta_words_off)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 8] =
            u32::try_from(host.section_index_off).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 9] = u32::try_from(prep.section_index_cap_len)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 10] =
            u32::try_from(host.out_lens_words_off).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 11] = u32::try_from(host.section_prefix_words_off)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 12] = u32::try_from(host.section_offsets_words_off)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 13] =
            u32::try_from(host.out_cmd_off).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 14] =
            u32::try_from(prep.section_cmd_cap_len).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 15] =
            u32::try_from(host.out_base_word).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 16] =
            u32::try_from(prep.total_bytes_cap / 4).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 17] = if sparse_probe { 1 } else { 0 };
        desc_words[base + 18] = u32::try_from(prep.table_data_stream_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 19] = u32::try_from(prep.section_index_stream_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 20] = 0;
        desc_words[base + 21] =
            u32::try_from(host.out_base_word).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 22] = u32::try_from(prep.total_bytes_cap / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
    }

    let mut sparse_submit_seq = 0u64;
    let mut sparse_submit_seq_last = 0u64;
    let mut sparse_submit_count = 0usize;
    let mut submit_ms = 0.0_f64;
    let mut pre_wait_latest_build_submit_ms = 0.0_f64;
    if sparse_wait_attr_probe {
        if let Some((_, _, latest_submit_idx)) = sparse_latest_build_submit.as_ref() {
            let t_pre_wait = Instant::now();
            r.device
                .poll(wgpu::Maintain::wait_for(latest_submit_idx.clone()));
            pre_wait_latest_build_submit_ms = elapsed_ms(t_pre_wait);
        }
    }

    let result_readback_bytes = u64::try_from(
        inputs
            .len()
            .saturating_mul(GPU_SPARSE_PACK_BATCH_RESULT_WORDS * 4),
    )
    .map_err(|_| PDeflateError::NumericOverflow)?
    .max(4);
    let max_total_words_cap = prepared.iter().fold(1u32, |acc, prep| {
        acc.max(u32::try_from(prep.total_bytes_cap / 4).unwrap_or(u32::MAX))
    });
    let pack_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-batch-bg"),
        layout: &r.pack_sparse_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scratch.desc_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scratch.result_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scratch.table_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: scratch.table_data_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: scratch.section_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: scratch.out_lens_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: scratch.section_prefix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: scratch.section_offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: scratch.out_cmd_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: scratch.out_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: sparse_stats_buffer
                    .as_ref()
                    .unwrap_or(&r.pack_sparse_stats_zero_buffer)
                    .as_entire_binding(),
            },
        ],
    });

    let t_upload = Instant::now();
    r.queue
        .write_buffer(&scratch.desc_buffer, 0, bytemuck::cast_slice(&desc_words));
    if let Some(stats_buffer) = sparse_stats_buffer.as_ref() {
        let zeros = [0u8; 32];
        r.queue.write_buffer(stats_buffer, 0, &zeros);
    }
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-batch-encoder"),
        });
    for ((input, prep), host) in inputs.iter().zip(prepared.iter()).zip(host_jobs.iter()) {
        encoder.copy_buffer_to_buffer(
            &input.section.section_meta_buffer,
            0,
            &scratch.section_meta_buffer,
            u64::try_from(host.section_meta_words_off)
                .map_err(|_| PDeflateError::NumericOverflow)?
                .saturating_mul(4),
            16,
        );
        if prep.table_index_len > 0 {
            encoder.copy_buffer_to_buffer(
                &input.table.table_index_buffer,
                0,
                &scratch.table_index_buffer,
                u64::try_from(host.table_index_off).map_err(|_| PDeflateError::NumericOverflow)?,
                u64::try_from(align_up4(prep.table_index_len))
                    .map_err(|_| PDeflateError::NumericOverflow)?,
            );
        }
        if prep.table_data_len > 0 {
            encoder.copy_buffer_to_buffer(
                &input.table.table_data_buffer,
                0,
                &scratch.table_data_buffer,
                u64::try_from(host.table_data_off).map_err(|_| PDeflateError::NumericOverflow)?,
                u64::try_from(align_up4(prep.table_data_len))
                    .map_err(|_| PDeflateError::NumericOverflow)?,
            );
        }
        let section_index_copy_bytes = input.section.section_index_cap_bytes.max(4);
        encoder.copy_buffer_to_buffer(
            &input.section.section_index_buffer,
            0,
            &scratch.section_index_buffer,
            u64::try_from(host.section_index_off).map_err(|_| PDeflateError::NumericOverflow)?,
            section_index_copy_bytes,
        );
        let lens_copy_bytes = input.section.out_lens_bytes.max(4);
        let lens_copy_words = u64::try_from(host.out_lens_words_off)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        encoder.copy_buffer_to_buffer(
            &input.section.out_lens_buffer,
            0,
            &scratch.out_lens_buffer,
            lens_copy_words,
            lens_copy_bytes,
        );
        let prefix_copy_words = u64::try_from(host.section_prefix_words_off)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        encoder.copy_buffer_to_buffer(
            &input.section.section_prefix_buffer,
            0,
            &scratch.section_prefix_buffer,
            prefix_copy_words,
            lens_copy_bytes,
        );
        let offsets_copy_words = u64::try_from(host.section_offsets_words_off)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        encoder.copy_buffer_to_buffer(
            &input.section.section_offsets_buffer,
            0,
            &scratch.section_offsets_buffer,
            offsets_copy_words,
            lens_copy_bytes,
        );
        let out_cmd_copy_bytes = input.section.out_cmd_bytes.max(4);
        encoder.copy_buffer_to_buffer(
            &input.section.out_cmd_buffer,
            0,
            &scratch.out_cmd_buffer,
            u64::try_from(host.out_cmd_off).map_err(|_| PDeflateError::NumericOverflow)?,
            out_cmd_copy_bytes,
        );
    }
    if let Some(query_set) = sparse_ts_query_set.as_ref() {
        encoder.write_timestamp(query_set, 0);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-prepare-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.pack_sparse_prepare_pipeline);
        pass.set_bind_group(0, &scratch.prepare_bind_group, &[]);
        let groups_x = u32::try_from(inputs.len().div_ceil(64))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .max(1);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    if let Some(query_set) = sparse_ts_query_set.as_ref() {
        encoder.write_timestamp(query_set, 1);
    }
    encoder.copy_buffer_to_buffer(
        &scratch.result_buffer,
        0,
        &scratch.result_readback_buffer,
        0,
        result_readback_bytes,
    );
    if let Some(query_set) = sparse_ts_query_set.as_ref() {
        encoder.write_timestamp(query_set, 2);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.pack_sparse_pipeline);
        pass.set_bind_group(0, &pack_bind_group, &[]);
        pass.dispatch_workgroups(
            max_total_words_cap.div_ceil(256).max(1),
            u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?,
            1,
        );
    }
    if let Some(query_set) = sparse_ts_query_set.as_ref() {
        encoder.write_timestamp(query_set, 3);
    }
    if let (Some(stats_buffer), Some(stats_readback)) = (
        sparse_stats_buffer.as_ref(),
        sparse_stats_readback_buffer.as_ref(),
    ) {
        encoder.copy_buffer_to_buffer(stats_buffer, 0, stats_readback, 0, 32);
    }
    if let (Some(query_set), Some(resolve_buffer), Some(readback_buffer)) = (
        sparse_ts_query_set.as_ref(),
        sparse_ts_resolve_buffer.as_ref(),
        sparse_ts_readback_buffer.as_ref(),
    ) {
        let ts_bytes = u64::from(sparse_ts_query_count).saturating_mul(8).max(8);
        encoder.resolve_query_set(query_set, 0..sparse_ts_query_count, resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(resolve_buffer, 0, readback_buffer, 0, ts_bytes);
    }
    let t_submit_all = Instant::now();
    let submit_seq = next_gpu_submit_seq();
    sparse_submit_seq = submit_seq;
    sparse_submit_seq_last = submit_seq;
    r.queue.submit(Some(encoder.finish()));
    submit_ms += elapsed_ms(t_submit_all);
    sparse_submit_count = 1;
    let upload_dispatch_ms = (elapsed_ms(t_upload) - submit_ms).max(0.0);
    upload_ms += upload_dispatch_ms;

    let mut lens_submit_done_wait_ms = 0.0_f64;
    let mut lens_poll_calls = 0u64;
    let lens_yield_calls = 0u64;
    let mut phase1_wait_calls = 0u64;
    let mut phase2_wait_calls = 0u64;
    let mut phase1_wait_max_ms = 0.0_f64;
    let mut phase2_wait_max_ms = 0.0_f64;
    let mut phase1_wait_ready_total = 0usize;
    let mut phase2_wait_ready_total = 0usize;
    let mut phase1_submit_done_wait_ms = 0.0_f64;
    let phase1_wait_ms;
    let phase1_map_after_done_ms;

    let result_slice = scratch
        .result_readback_buffer
        .slice(..result_readback_bytes);
    let (result_tx, result_rx) = mpsc::channel();
    result_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = result_tx.send(res);
    });
    let mut sparse_stats_slice = None;
    let mut sparse_stats_receiver = None;
    let mut sparse_stats_pending = false;
    if let Some(stats_readback) = sparse_stats_readback_buffer.as_ref() {
        let stats_slice = stats_readback.slice(..32);
        let (stats_tx, stats_rx) = mpsc::channel();
        stats_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = stats_tx.send(res);
        });
        sparse_stats_slice = Some(stats_slice);
        sparse_stats_receiver = Some(stats_rx);
        sparse_stats_pending = true;
    }
    let mut sparse_ts_slice = None;
    let mut sparse_ts_receiver = None;
    let mut sparse_ts_pending = false;
    if let Some(ts_readback) = sparse_ts_readback_buffer.as_ref() {
        let ts_size = u64::from(sparse_ts_query_count).saturating_mul(8).max(8);
        let ts_slice = ts_readback.slice(..ts_size);
        let (ts_tx, ts_rx) = mpsc::channel();
        ts_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = ts_tx.send(res);
        });
        sparse_ts_slice = Some(ts_slice);
        sparse_ts_receiver = Some(ts_rx);
        sparse_ts_pending = true;
    }
    let t_lens_wait_phase1 = Instant::now();
    let mut result_pending = true;
    while result_pending || sparse_stats_pending || sparse_ts_pending {
        r.device.poll(wgpu::Maintain::Poll);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        let mut collected = 0usize;
        if result_pending {
            match result_rx.try_recv() {
                Ok(res) => {
                    res.map_err(|e| {
                        PDeflateError::Gpu(format!("gpu sparse size map failed: {e}"))
                    })?;
                    result_pending = false;
                    collected = collected.saturating_add(1);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(PDeflateError::Gpu(
                        "gpu sparse size map channel closed".to_string(),
                    ));
                }
            }
        }
        if sparse_stats_pending {
            if let Some(rx) = sparse_stats_receiver.as_ref() {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse stats map failed: {e}"))
                        })?;
                        sparse_stats_pending = false;
                        collected = collected.saturating_add(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(PDeflateError::Gpu(
                            "gpu sparse stats map channel closed".to_string(),
                        ));
                    }
                }
            }
        }
        if sparse_ts_pending {
            if let Some(rx) = sparse_ts_receiver.as_ref() {
                match rx.try_recv() {
                    Ok(res) => {
                        if let Err(e) = res {
                            sparse_ts_map_status = Some(format!("map failed: {e}"));
                        }
                        sparse_ts_pending = false;
                        collected = collected.saturating_add(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        sparse_ts_map_status = Some("map channel closed".to_string());
                        sparse_ts_pending = false;
                    }
                }
            }
        }
        if collected > 0 {
            continue;
        }
        let t_wait = Instant::now();
        r.device.poll(wgpu::Maintain::Wait);
        let wait_ms = elapsed_ms(t_wait);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        phase1_wait_calls = phase1_wait_calls.saturating_add(1);
        phase1_wait_max_ms = phase1_wait_max_ms.max(wait_ms);
        phase1_submit_done_wait_ms += wait_ms;
        let mut collected_after_wait = 0usize;
        if result_pending {
            match result_rx.try_recv() {
                Ok(res) => {
                    res.map_err(|e| {
                        PDeflateError::Gpu(format!("gpu sparse size map failed: {e}"))
                    })?;
                    result_pending = false;
                    collected_after_wait = collected_after_wait.saturating_add(1);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(PDeflateError::Gpu(
                        "gpu sparse size map channel closed".to_string(),
                    ));
                }
            }
        }
        if sparse_stats_pending {
            if let Some(rx) = sparse_stats_receiver.as_ref() {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse stats map failed: {e}"))
                        })?;
                        sparse_stats_pending = false;
                        collected_after_wait = collected_after_wait.saturating_add(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(PDeflateError::Gpu(
                            "gpu sparse stats map channel closed".to_string(),
                        ));
                    }
                }
            }
        }
        if sparse_ts_pending {
            if let Some(rx) = sparse_ts_receiver.as_ref() {
                match rx.try_recv() {
                    Ok(res) => {
                        if let Err(e) = res {
                            sparse_ts_map_status = Some(format!("map failed: {e}"));
                        }
                        sparse_ts_pending = false;
                        collected_after_wait = collected_after_wait.saturating_add(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        sparse_ts_map_status = Some("map channel closed".to_string());
                        sparse_ts_pending = false;
                    }
                }
            }
        }
        phase1_wait_ready_total = phase1_wait_ready_total.saturating_add(collected_after_wait);
        if collected_after_wait == 0
            && (result_pending || sparse_stats_pending || sparse_ts_pending)
        {
            return Err(PDeflateError::Gpu(
                "gpu sparse result/stats pending stalled after wait".to_string(),
            ));
        }
    }
    lens_submit_done_wait_ms += phase1_submit_done_wait_ms;
    phase1_wait_ms = elapsed_ms(t_lens_wait_phase1);
    phase1_map_after_done_ms = (phase1_wait_ms - phase1_submit_done_wait_ms).max(0.0);

    let t_lens_copy = Instant::now();
    let mut payload_lens = Vec::<usize>::with_capacity(inputs.len());
    let mut payload_table_counts = Vec::<usize>::with_capacity(inputs.len());
    let mut payload_copy_offsets = Vec::<u64>::with_capacity(inputs.len());
    let mut payload_copy_bytes = Vec::<u64>::with_capacity(inputs.len());
    let mut payload_readback_bytes = 0u64;
    let result_mapped = result_slice.get_mapped_range();
    let result_words: &[u32] = bytemuck::cast_slice(&result_mapped);
    for (idx, prep) in prepared.iter().enumerate() {
        let base = idx * GPU_SPARSE_PACK_BATCH_RESULT_WORDS;
        let total_len_u32 = *result_words.get(base + 2).unwrap_or(&0xffff_ffff);
        let total_words_u32 = *result_words.get(base + 3).unwrap_or(&0);
        if total_len_u32 == 0xffff_ffff || total_words_u32 == 0 {
            return Err(PDeflateError::Gpu(
                "gpu sparse size prepare overflow while packing".to_string(),
            ));
        }
        let total_len =
            usize::try_from(total_len_u32).map_err(|_| PDeflateError::NumericOverflow)?;
        let total_copy_len = align_up4(total_len);
        let total_copy_bytes =
            u64::try_from(total_copy_len).map_err(|_| PDeflateError::NumericOverflow)?;
        if total_copy_bytes > prep.total_bytes_cap {
            return Err(PDeflateError::Gpu(
                "gpu sparse pack produced oversized aligned payload".to_string(),
            ));
        }
        payload_copy_offsets.push(payload_readback_bytes);
        payload_copy_bytes.push(total_copy_bytes);
        payload_readback_bytes = payload_readback_bytes
            .checked_add(total_copy_bytes.max(4))
            .ok_or(PDeflateError::NumericOverflow)?;
        payload_lens.push(total_len);
        payload_table_counts.push(prep.table_count);
    }
    drop(result_mapped);
    scratch.result_readback_buffer.unmap();
    lens_copy_ms = elapsed_ms(t_lens_copy);

    let payload_readback_bytes = payload_readback_bytes.max(4);
    let payload_readback_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-payload-batch-readback"),
        size: payload_readback_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let t_payload_submit_all = Instant::now();
    let mut payload_encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-payload-batch-readback-encoder"),
        });
    for ((host, &dst_off), &copy_bytes) in host_jobs
        .iter()
        .zip(payload_copy_offsets.iter())
        .zip(payload_copy_bytes.iter())
    {
        payload_encoder.copy_buffer_to_buffer(
            &scratch.out_buffer,
            u64::try_from(host.out_base_word)
                .map_err(|_| PDeflateError::NumericOverflow)?
                .saturating_mul(4),
            &payload_readback_buffer,
            dst_off,
            copy_bytes.max(4),
        );
    }
    let t_payload_submit = Instant::now();
    let payload_submit_seq = next_gpu_submit_seq();
    sparse_submit_seq_last = payload_submit_seq;
    r.queue.submit(Some(payload_encoder.finish()));
    let payload_submit_ms = elapsed_ms(t_payload_submit);
    submit_ms += payload_submit_ms;
    sparse_submit_count = 2;
    upload_ms += (elapsed_ms(t_payload_submit_all) - payload_submit_ms).max(0.0);

    let payload_slice = payload_readback_buffer.slice(..payload_readback_bytes);
    let (out_tx, out_rx) = mpsc::channel();
    payload_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = out_tx.send(res);
    });
    let t_lens_wait_phase2 = Instant::now();
    let mut phase2_submit_done_wait_ms = 0.0_f64;
    let mut payload_pending = true;
    while payload_pending {
        r.device.poll(wgpu::Maintain::Poll);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        match out_rx.try_recv() {
            Ok(res) => {
                res.map_err(|e| {
                    PDeflateError::Gpu(format!("gpu sparse chunk pack map failed: {e}"))
                })?;
                payload_pending = false;
                continue;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(PDeflateError::Gpu(
                    "gpu sparse chunk pack map channel closed".to_string(),
                ));
            }
        }
        let t_wait = Instant::now();
        r.device.poll(wgpu::Maintain::Wait);
        let wait_ms = elapsed_ms(t_wait);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        phase2_wait_calls = phase2_wait_calls.saturating_add(1);
        phase2_wait_max_ms = phase2_wait_max_ms.max(wait_ms);
        phase2_submit_done_wait_ms += wait_ms;
        match out_rx.try_recv() {
            Ok(res) => {
                res.map_err(|e| {
                    PDeflateError::Gpu(format!("gpu sparse chunk pack map failed: {e}"))
                })?;
                payload_pending = false;
                phase2_wait_ready_total = phase2_wait_ready_total.saturating_add(1);
            }
            Err(mpsc::TryRecvError::Empty) => {
                return Err(PDeflateError::Gpu(
                    "gpu sparse payload pending stalled after wait".to_string(),
                ));
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(PDeflateError::Gpu(
                    "gpu sparse chunk pack map channel closed".to_string(),
                ));
            }
        }
    }
    lens_submit_done_wait_ms += phase2_submit_done_wait_ms;
    let phase2_wait_ms = elapsed_ms(t_lens_wait_phase2);
    let phase2_map_after_done_ms = (phase2_wait_ms - phase2_submit_done_wait_ms).max(0.0);
    let lens_map_after_done_ms = phase1_map_after_done_ms + phase2_map_after_done_ms;
    lens_wait_ms = phase1_wait_ms + phase2_wait_ms;

    let mut sparse_probe_counters = [0u64; 8];
    if let Some(stats_slice) = sparse_stats_slice {
        let mapped = stats_slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        for (i, dst) in sparse_probe_counters.iter_mut().enumerate() {
            *dst = u64::from(*words.get(i).unwrap_or(&0));
        }
        drop(mapped);
        if let Some(stats_readback) = sparse_stats_readback_buffer.as_ref() {
            stats_readback.unmap();
        }
    }
    if let Some(ts_slice) = sparse_ts_slice {
        if sparse_ts_map_status.is_none() {
            let t_parse = Instant::now();
            let mapped = ts_slice.get_mapped_range();
            let ticks: &[u64] = bytemuck::cast_slice(&mapped);
            if ticks.len() >= 4 {
                let period_ms = (r.queue.get_timestamp_period() as f64) / 1_000_000.0;
                if ticks[1] >= ticks[0] {
                    sparse_ts_prepare_kernel_ms = (ticks[1] - ticks[0]) as f64 * period_ms;
                }
                if ticks[3] >= ticks[2] {
                    sparse_ts_kernel_ms = (ticks[3] - ticks[2]) as f64 * period_ms;
                }
            }
            drop(mapped);
            sparse_ts_parse_ms = elapsed_ms(t_parse);
        }
        if let Some(ts_readback) = sparse_ts_readback_buffer.as_ref() {
            ts_readback.unmap();
        }
    }
    let sparse_ts_kernel_total_ms = sparse_ts_prepare_kernel_ms + sparse_ts_kernel_ms;
    let sparse_ts_queue_stall_est_ms = ((phase1_submit_done_wait_ms + phase2_submit_done_wait_ms)
        - sparse_ts_kernel_total_ms)
        .max(0.0);

    if sparse_lens_wait_probe {
        let seq = SPARSE_LENS_WAIT_PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
        if table_stage_probe_should_log(seq) {
            let size_readback_bytes = result_readback_bytes;
            let stats_readback_bytes = if sparse_probe { 32u64 } else { 0u64 };
            let ts_readback_bytes = if sparse_kernel_ts_probe {
                u64::from(sparse_ts_query_count).saturating_mul(8)
            } else {
                0u64
            };
            let readback_bytes = size_readback_bytes
                .saturating_add(stats_readback_bytes)
                .saturating_add(ts_readback_bytes)
                .saturating_add(payload_readback_bytes);
            let build_id_range = if sparse_prebuild_build_id_min != u64::MAX {
                format!(
                    "{}..{}",
                    sparse_prebuild_build_id_min, sparse_prebuild_build_id_max
                )
            } else {
                "none".to_string()
            };
            let submit_seq_range = if sparse_prebuild_submit_seq_min != u64::MAX {
                format!(
                    "{}..{}",
                    sparse_prebuild_submit_seq_min, sparse_prebuild_submit_seq_max
                )
            } else {
                "none".to_string()
            };
            let build_ids = if sparse_build_ids.is_empty() {
                "none".to_string()
            } else {
                sparse_build_ids
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let (latest_build_submit_seq, latest_build_id) = sparse_latest_build_submit
                .as_ref()
                .map(|(submit_seq, build_id, _)| (*submit_seq, *build_id))
                .unwrap_or((0, 0));
            let sparse_ts_status = if !sparse_kernel_ts_probe {
                "off".to_string()
            } else {
                sparse_ts_map_status
                    .clone()
                    .unwrap_or_else(|| "ok".to_string())
            };
            eprintln!(
                "[cozip_pdeflate][timing][sparse-lens-wait-breakdown] seq={} chunks={} pending_maps={} readback_kib={:.1} payload_readback_kib={:.1} build_id_range={} build_submit_seq_range={} build_ids={} latest_build_submit_seq={} latest_build_id={} sparse_submit_seq={} sparse_submit_seq_last={} sparse_submit_count={} payload_submit_seq={} pre_wait_latest_build_submit_ms={:.3} submit_done_wait_sparse_ms={:.3} submit_done_wait_payload_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_sparse_ms={:.3} map_callback_wait_payload_ms={:.3} map_callback_wait_ms={:.3} sparse_wait_calls={} payload_wait_calls={} sparse_wait_max_ms={:.3} payload_wait_max_ms={:.3} sparse_wait_ready={} payload_wait_ready={} sparse_ts_prepare_kernel_ms={:.3} sparse_ts_kernel_ms={:.3} sparse_ts_kernel_total_ms={:.3} sparse_ts_parse_ms={:.3} sparse_ts_queue_stall_est_ms={:.3} sparse_ts_status={} lens_wait_ms={:.3}",
                seq,
                inputs.len(),
                1usize
                    .saturating_add(if sparse_probe { 1 } else { 0 })
                    .saturating_add(if sparse_kernel_ts_probe { 1 } else { 0 })
                    .saturating_add(1),
                (readback_bytes as f64) / 1024.0,
                (payload_readback_bytes as f64) / 1024.0,
                build_id_range,
                submit_seq_range,
                build_ids,
                latest_build_submit_seq,
                latest_build_id,
                sparse_submit_seq,
                payload_submit_seq,
                sparse_submit_count,
                sparse_submit_seq_last,
                pre_wait_latest_build_submit_ms,
                phase1_submit_done_wait_ms,
                phase2_submit_done_wait_ms,
                lens_submit_done_wait_ms,
                phase1_map_after_done_ms,
                phase2_map_after_done_ms,
                lens_map_after_done_ms,
                phase1_wait_calls,
                phase2_wait_calls,
                phase1_wait_max_ms,
                phase2_wait_max_ms,
                phase1_wait_ready_total,
                phase2_wait_ready_total,
                sparse_ts_prepare_kernel_ms,
                sparse_ts_kernel_ms,
                sparse_ts_kernel_total_ms,
                sparse_ts_parse_ms,
                sparse_ts_queue_stall_est_ms,
                sparse_ts_status,
                lens_wait_ms,
            );
            if sparse_kernel_ts_probe {
                let ts_seq = SPARSE_KERNEL_TS_PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
                if table_stage_probe_should_log(ts_seq) {
                    eprintln!(
                        "[cozip_pdeflate][timing][sparse-kernel-ts] seq={} chunks={} prepare_kernel_ms={:.3} sparse_kernel_ms={:.3} kernel_total_ms={:.3} submit_done_wait_sparse_ms={:.3} queue_stall_est_ms={:.3} parse_ms={:.3} status={}",
                        ts_seq,
                        inputs.len(),
                        sparse_ts_prepare_kernel_ms,
                        sparse_ts_kernel_ms,
                        sparse_ts_kernel_total_ms,
                        phase1_submit_done_wait_ms,
                        sparse_ts_queue_stall_est_ms,
                        sparse_ts_parse_ms,
                        sparse_ts_map_status.as_deref().unwrap_or("ok"),
                    );
                }
            }
            if sparse_wait_attr_probe {
                eprintln!(
                    "[cozip_pdeflate][timing][sparse-wait-attribution] seq={} chunks={} sparse_submit_seq={} sparse_submit_seq_last={} sparse_submit_count={} pre_wait_latest_build_submit_ms={:.3} submit_done_wait_sparse_ms={:.3} submit_done_wait_payload_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_ms={:.3} lens_wait_ms={:.3} note=\"pre_wait captures known table-build completion; submit_done_wait includes residual queue+own sparse+payload submissions\"",
                    seq,
                    inputs.len(),
                    sparse_submit_seq,
                    sparse_submit_seq_last,
                    sparse_submit_count,
                    pre_wait_latest_build_submit_ms,
                    phase1_submit_done_wait_ms,
                    phase2_submit_done_wait_ms,
                    lens_submit_done_wait_ms,
                    lens_map_after_done_ms,
                    lens_wait_ms,
                );
            }
        }
    }

    let sparse_wait_ms = 0.0_f64;
    wait_ms += sparse_wait_ms;

    let t_copy = Instant::now();
    let mut out = Vec::<GpuSparsePackChunkOutput>::with_capacity(inputs.len());
        let payload_mapped = payload_slice.get_mapped_range();
        let payload_bytes: &[u8] = &payload_mapped;
        for (idx, host) in host_jobs.iter().enumerate() {
        let chunk_base = usize::try_from(payload_copy_offsets[idx])
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let chunk_copy_len = usize::try_from(payload_copy_bytes[idx])
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let chunk_end = chunk_base
            .checked_add(chunk_copy_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        let mapped = payload_bytes.get(chunk_base..chunk_end).ok_or_else(|| {
            PDeflateError::Gpu(format!(
                "gpu sparse payload batch readback truncated idx={} chunk_base={} chunk_copy_len={} payload_len={} payload_readback_bytes={} mapped_len={} out_base_word={}",
                idx,
                chunk_base,
                chunk_copy_len,
                payload_lens[idx],
                payload_readback_bytes,
                payload_bytes.len(),
                host.out_base_word,
            ))
        })?;
            let payload_len = payload_lens[idx];
            let mut payload = vec![0u8; payload_len];
            payload.copy_from_slice(&mapped[..payload_len]);
            finalize_sparse_packed_chunk_huffman(
                &mut payload,
                prepared[idx].huffman_encode_enabled,
            )?;
            if gpu_sparse_payload_validate_enabled() {
                let mut validate_out = vec![0u8; prepared[idx].chunk_len];
                super::decompress_chunk_into(&payload, &mut validate_out).map_err(|err| {
                    PDeflateError::Gpu(format!("sparse-batch/validate-payload: {err}"))
                })?;
            }
            out.push(GpuSparsePackChunkOutput {
                payload,
                table_count: payload_table_counts[idx],
            });
        }
    drop(payload_mapped);
    payload_readback_buffer.unmap();
    let sparse_copy_ms = elapsed_ms(t_copy);
    map_copy_ms += lens_copy_ms + sparse_copy_ms;

    if sparse_probe {
        let total_classified = sparse_probe_counters[0]
            .saturating_add(sparse_probe_counters[1])
            .saturating_add(sparse_probe_counters[2])
            .saturating_add(sparse_probe_counters[3])
            .saturating_add(sparse_probe_counters[4]);
        let section_bytes = sparse_probe_counters[4];
        let search_steps = sparse_probe_counters[5];
        let search_hits = sparse_probe_counters[6];
        let search_misses = sparse_probe_counters[7];
        let search_steps_per_section_byte = if section_bytes > 0 {
            (search_steps as f64) / (section_bytes as f64)
        } else {
            0.0
        };
        let section_byte_pct = if total_classified > 0 {
            100.0 * (section_bytes as f64) / (total_classified as f64)
        } else {
            0.0
        };
        eprintln!(
            "[cozip_pdeflate][timing][gpu-sparse-probe] chunks={} bytes_header={} bytes_table_index={} bytes_table_data={} bytes_section_index={} bytes_section_cmd={} section_byte_pct={:.1} search_steps={} search_hits={} search_misses={} search_steps_per_section_byte={:.3}",
            inputs.len(),
            sparse_probe_counters[0],
            sparse_probe_counters[1],
            sparse_probe_counters[2],
            sparse_probe_counters[3],
            section_bytes,
            section_byte_pct,
            search_steps,
            search_hits,
            search_misses,
            search_steps_per_section_byte,
        );
    }

    release_sparse_pack_scratch(r, scratch);

    let total_ms = elapsed_ms(t_total);
    Ok((
        out,
        GpuMatchProfile {
            upload_ms,
            wait_ms,
            map_copy_ms,
            total_ms,
        },
        GpuSparsePackBatchProfile {
            chunks: inputs.len(),
            lens_bytes_total,
            out_cmd_bytes_total,
            predicted_payload_bytes_total: 0,
            cap_payload_bytes_total: 0,
            actual_payload_bytes_total: 0,
            predicted_class_counts: [0; GPU_SPARSE_PAYLOAD_CLASS_COUNT],
            cap_class_counts: [0; GPU_SPARSE_PAYLOAD_CLASS_COUNT],
            actual_class_counts: [0; GPU_SPARSE_PAYLOAD_CLASS_COUNT],
            predicted_underestimate_chunks: 0,
            predicted_underestimate_bytes: 0,
            predicted_overestimate_bytes: 0,
            lens_submit_ms,
            lens_submit_done_wait_ms,
            lens_map_after_done_ms,
            lens_poll_calls,
            lens_yield_calls,
            lens_wait_ms,
            lens_copy_ms,
            prepare_ms,
            table_size_resolve_ms,
            prepare_misc_ms,
            scratch_acquire_ms,
            upload_dispatch_ms,
            submit_ms,
            wait_ms: sparse_wait_ms,
            copy_ms: sparse_copy_ms,
            total_ms,
        },
    ))
}

pub(crate) fn read_section_commands_from_device_batch(
    inputs: &[GpuSparsePackInput<'_>],
) -> Result<
    (
        Vec<GpuSparseSectionHostOutput>,
        GpuMatchProfile,
        GpuSparsePackBatchProfile,
    ),
    PDeflateError,
> {
    if inputs.is_empty() {
        return Ok((
            Vec::new(),
            GpuMatchProfile::default(),
            GpuSparsePackBatchProfile::default(),
        ));
    }
    for input in inputs {
        if input.section.section_count != input.section_count {
            return Err(PDeflateError::InvalidOptions(
                "gpu sparse readback section_count mismatch",
            ));
        }
    }

    struct ReadPrepared {
        section_offsets: Vec<usize>,
        out_lens_bytes: u64,
        out_cmd_bytes: u64,
    }

    let r = runtime()?;
    let t_total = Instant::now();
    let lens_bytes_total = inputs.iter().fold(0u64, |acc, input| {
        acc.saturating_add(input.section.out_lens_bytes.max(4))
    });

    let t_prepare = Instant::now();
    let mut prepared = Vec::<ReadPrepared>::with_capacity(inputs.len());
    for input in inputs {
        let min_lens_bytes = u64::try_from(input.section_count)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        if input.section.out_lens_bytes < min_lens_bytes {
            return Err(PDeflateError::Gpu(
                "gpu sparse readback lens buffer too small".to_string(),
            ));
        }

        let mut section_offsets = Vec::with_capacity(input.section_count);
        let mut total_cmd_cap_bytes = 0usize;
        for sec in 0..input.section_count {
            let s0 = section_start(sec, input.section_count, input.chunk_len);
            let s1 = section_start(sec + 1, input.section_count, input.chunk_len);
            let sec_len = s1.saturating_sub(s0);
            let cap_bytes = estimate_section_cmd_cap_bytes(sec_len)?;
            let aligned = align_up4(total_cmd_cap_bytes);
            section_offsets.push(aligned);
            total_cmd_cap_bytes = aligned
                .checked_add(cap_bytes)
                .ok_or(PDeflateError::NumericOverflow)?;
        }
        let expected_cmd_bytes = u64::try_from(align_up4(total_cmd_cap_bytes))
            .map_err(|_| PDeflateError::NumericOverflow)?;
        if expected_cmd_bytes > input.section.out_cmd_bytes.max(4) {
            return Err(PDeflateError::Gpu(
                "gpu sparse readback cmd buffer too small".to_string(),
            ));
        }
        prepared.push(ReadPrepared {
            section_offsets,
            out_lens_bytes: input.section.out_lens_bytes.max(4),
            out_cmd_bytes: input.section.out_cmd_bytes.max(4),
        });
    }
    let prepare_ms = elapsed_ms(t_prepare);

    let queue_probe = queue_probe_enabled();
    let mut pre_lens_queue_drain_ms = 0.0_f64;
    let mut pre_lens_queue_drain_poll_calls = 0u64;
    if queue_probe {
        let t_pre_drain = Instant::now();
        let (done_tx_pre, done_rx_pre) = mpsc::channel();
        r.queue.on_submitted_work_done(move || {
            let _ = done_tx_pre.send(());
        });
        let mut done = false;
        while !done {
            r.device.poll(wgpu::Maintain::Wait);
            pre_lens_queue_drain_poll_calls = pre_lens_queue_drain_poll_calls.saturating_add(1);
            match done_rx_pre.try_recv() {
                Ok(_) | Err(mpsc::TryRecvError::Disconnected) => done = true,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        pre_lens_queue_drain_ms = elapsed_ms(t_pre_drain);
    }

    struct LensHostChunk {
        section_cmd_lens: Vec<u32>,
        copy_offsets: Vec<usize>,
        total_cmd_len: usize,
        total_copy_bytes: u64,
    }

    let mut lens_offsets = Vec::with_capacity(prepared.len());
    let mut lens_total_bytes = 0u64;
    for prep in prepared.iter() {
        lens_offsets.push(lens_total_bytes);
        lens_total_bytes = lens_total_bytes
            .checked_add(prep.out_lens_bytes.max(4))
            .ok_or(PDeflateError::NumericOverflow)?;
    }
    let lens_total_bytes = lens_total_bytes.max(4);
    let lens_readback = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-sparse-lens-batch-readback"),
        size: lens_total_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let t_lens_submit_all = Instant::now();
    let mut lens_encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-sparse-lens-batch-readback-encoder"),
        });
    for (idx, input) in inputs.iter().enumerate() {
        lens_encoder.copy_buffer_to_buffer(
            &input.section.out_lens_buffer,
            0,
            &lens_readback,
            lens_offsets[idx],
            prepared[idx].out_lens_bytes.max(4),
        );
    }
    let t_lens_submit = Instant::now();
    r.queue.submit(Some(lens_encoder.finish()));
    let lens_submit_ms = elapsed_ms(t_lens_submit);
    let lens_upload_dispatch_ms = (elapsed_ms(t_lens_submit_all) - lens_submit_ms).max(0.0);

    let t_lens_wait = Instant::now();
    let lens_slice = lens_readback.slice(..lens_total_bytes);
    let (tx_lens, rx_lens) = mpsc::channel();
    lens_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_lens.send(res);
    });
    let mut lens_poll_calls = 0u64;
    loop {
        r.device.poll(wgpu::Maintain::Wait);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        match rx_lens.try_recv() {
            Ok(res) => {
                res.map_err(|e| {
                    PDeflateError::Gpu(format!("gpu sparse lens batch map failed: {e}"))
                })?;
                break;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(PDeflateError::Gpu(
                    "gpu sparse lens batch map channel closed".to_string(),
                ));
            }
        }
    }
    let lens_submit_done_wait_ms = elapsed_ms(t_lens_wait);
    let lens_wait_ms = lens_submit_done_wait_ms;
    let lens_map_after_done_ms = 0.0_f64;

    let t_lens_copy = Instant::now();
    let lens_mapped = lens_slice.get_mapped_range();
    let lens_bytes: &[u8] = &lens_mapped;
    let mut lens_chunks = Vec::<LensHostChunk>::with_capacity(inputs.len());
    for (idx, input) in inputs.iter().enumerate() {
        let prep = &prepared[idx];
        let lens_base =
            usize::try_from(lens_offsets[idx]).map_err(|_| PDeflateError::NumericOverflow)?;
        let lens_span = usize::try_from(prep.out_lens_bytes.max(4))
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let lens_end = lens_base
            .checked_add(lens_span)
            .ok_or(PDeflateError::NumericOverflow)?;
        if lens_end > lens_bytes.len() {
            return Err(PDeflateError::Gpu(
                "gpu sparse lens batch readback truncated".to_string(),
            ));
        }
        let chunk_lens = &lens_bytes[lens_base..lens_end];
        let needed_lens_bytes = input
            .section_count
            .checked_mul(4)
            .ok_or(PDeflateError::NumericOverflow)?;
        if chunk_lens.len() < needed_lens_bytes {
            return Err(PDeflateError::Gpu(
                "gpu sparse lens batch readback too small for section count".to_string(),
            ));
        }

        let mut section_cmd_lens = vec![0u32; input.section_count];
        for (sec, dst) in section_cmd_lens.iter_mut().enumerate() {
            let off = sec.checked_mul(4).ok_or(PDeflateError::NumericOverflow)?;
            *dst = u32::from_le_bytes([
                chunk_lens[off],
                chunk_lens[off + 1],
                chunk_lens[off + 2],
                chunk_lens[off + 3],
            ]);
        }

        let mut total_cmd_len = 0usize;
        let mut copy_offsets = Vec::with_capacity(input.section_count);
        let mut copy_cursor = 0usize;
        for &len_u32 in &section_cmd_lens {
            if len_u32 == 0xffff_ffff {
                return Err(PDeflateError::Gpu(
                    "gpu sparse lens overflow while rebuilding host section commands".to_string(),
                ));
            }
            let len = usize::try_from(len_u32).map_err(|_| PDeflateError::NumericOverflow)?;
            total_cmd_len = total_cmd_len
                .checked_add(len)
                .ok_or(PDeflateError::NumericOverflow)?;
            copy_offsets.push(copy_cursor);
            if len > 0 {
                copy_cursor = copy_cursor
                    .checked_add(align_up4(len))
                    .ok_or(PDeflateError::NumericOverflow)?;
            }
        }
        let total_copy_bytes =
            u64::try_from(copy_cursor.max(4)).map_err(|_| PDeflateError::NumericOverflow)?;
        if total_copy_bytes > prep.out_cmd_bytes {
            return Err(PDeflateError::Gpu(
                "gpu sparse cmd compact copy size exceeds cap".to_string(),
            ));
        }
        lens_chunks.push(LensHostChunk {
            section_cmd_lens,
            copy_offsets,
            total_cmd_len,
            total_copy_bytes,
        });
    }
    drop(lens_mapped);
    lens_readback.unmap();
    let lens_copy_ms = elapsed_ms(t_lens_copy);

    let mut cmd_offsets = Vec::with_capacity(lens_chunks.len());
    let mut cmd_total_bytes = 0u64;
    for lens in lens_chunks.iter() {
        cmd_offsets.push(cmd_total_bytes);
        cmd_total_bytes = cmd_total_bytes
            .checked_add(lens.total_copy_bytes.max(4))
            .ok_or(PDeflateError::NumericOverflow)?;
    }
    let cmd_total_bytes = cmd_total_bytes.max(4);
    let cmd_readback = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-sparse-cmd-batch-readback"),
        size: cmd_total_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let t_cmd_submit_all = Instant::now();
    let mut cmd_encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-sparse-cmd-batch-readback-encoder"),
        });
    for (idx, input) in inputs.iter().enumerate() {
        let prep = &prepared[idx];
        let lens = &lens_chunks[idx];
        let chunk_dst_base = cmd_offsets[idx];
        for sec in 0..input.section_count {
            let len = usize::try_from(lens.section_cmd_lens[sec])
                .map_err(|_| PDeflateError::NumericOverflow)?;
            if len == 0 {
                continue;
            }
            let src_off = u64::try_from(prep.section_offsets[sec])
                .map_err(|_| PDeflateError::NumericOverflow)?;
            let dst_off = chunk_dst_base
                .checked_add(
                    u64::try_from(lens.copy_offsets[sec])
                        .map_err(|_| PDeflateError::NumericOverflow)?,
                )
                .ok_or(PDeflateError::NumericOverflow)?;
            let copy_bytes =
                u64::try_from(align_up4(len)).map_err(|_| PDeflateError::NumericOverflow)?;
            if src_off.saturating_add(copy_bytes) > prep.out_cmd_bytes
                || dst_off.saturating_add(copy_bytes) > cmd_total_bytes
            {
                return Err(PDeflateError::Gpu(
                    "gpu sparse cmd batch copy range out of bounds".to_string(),
                ));
            }
            cmd_encoder.copy_buffer_to_buffer(
                &input.section.out_cmd_buffer,
                src_off,
                &cmd_readback,
                dst_off,
                copy_bytes,
            );
        }
    }
    let t_cmd_submit = Instant::now();
    r.queue.submit(Some(cmd_encoder.finish()));
    let cmd_submit_ms = elapsed_ms(t_cmd_submit);
    let cmd_upload_dispatch_ms = (elapsed_ms(t_cmd_submit_all) - cmd_submit_ms).max(0.0);

    let t_cmd_wait = Instant::now();
    let cmd_slice = cmd_readback.slice(..cmd_total_bytes);
    let (tx_cmd, rx_cmd) = mpsc::channel();
    cmd_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_cmd.send(res);
    });
    let mut cmd_poll_calls = 0u64;
    loop {
        r.device.poll(wgpu::Maintain::Wait);
        cmd_poll_calls = cmd_poll_calls.saturating_add(1);
        match rx_cmd.try_recv() {
            Ok(res) => {
                res.map_err(|e| {
                    PDeflateError::Gpu(format!("gpu sparse cmd batch map failed: {e}"))
                })?;
                break;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(PDeflateError::Gpu(
                    "gpu sparse cmd batch map channel closed".to_string(),
                ));
            }
        }
    }
    let cmd_wait_ms = elapsed_ms(t_cmd_wait);

    let t_cmd_copy = Instant::now();
    let cmd_mapped = cmd_slice.get_mapped_range();
    let cmd_bytes: &[u8] = &cmd_mapped;
    let mut out = Vec::<GpuSparseSectionHostOutput>::with_capacity(inputs.len());
    let mut copied_cmd_bytes_total = 0u64;
    for (idx, lens) in lens_chunks.into_iter().enumerate() {
        let chunk_base =
            usize::try_from(cmd_offsets[idx]).map_err(|_| PDeflateError::NumericOverflow)?;
        let chunk_span = usize::try_from(lens.total_copy_bytes.max(4))
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let chunk_end = chunk_base
            .checked_add(chunk_span)
            .ok_or(PDeflateError::NumericOverflow)?;
        if chunk_end > cmd_bytes.len() {
            return Err(PDeflateError::Gpu(
                "gpu sparse cmd batch readback truncated".to_string(),
            ));
        }
        let chunk_cmd = &cmd_bytes[chunk_base..chunk_end];
        let mut section_cmd = Vec::with_capacity(lens.total_cmd_len);
        for sec in 0..lens.section_cmd_lens.len() {
            let len = usize::try_from(lens.section_cmd_lens[sec])
                .map_err(|_| PDeflateError::NumericOverflow)?;
            if len == 0 {
                continue;
            }
            let off = lens.copy_offsets[sec];
            let end = off.checked_add(len).ok_or(PDeflateError::NumericOverflow)?;
            if end > chunk_cmd.len() {
                return Err(PDeflateError::Gpu(format!(
                    "gpu sparse cmd batch section {} truncated",
                    sec
                )));
            }
            section_cmd.extend_from_slice(&chunk_cmd[off..end]);
        }
        copied_cmd_bytes_total = copied_cmd_bytes_total.saturating_add(lens.total_copy_bytes);
        out.push(GpuSparseSectionHostOutput {
            section_cmd_lens: lens.section_cmd_lens,
            section_cmd,
        });
    }
    drop(cmd_mapped);
    cmd_readback.unmap();
    let cmd_copy_ms = elapsed_ms(t_cmd_copy);

    let lens_yield_calls = 0u64;
    let upload_dispatch_ms = lens_upload_dispatch_ms + cmd_upload_dispatch_ms;
    let submit_ms = lens_submit_ms + cmd_submit_ms;
    let wait_ms = lens_wait_ms + cmd_wait_ms;
    let map_copy_ms = lens_copy_ms + cmd_copy_ms;
    let total_ms = elapsed_ms(t_total);
    if queue_probe {
        eprintln!(
            "[cozip_pdeflate][timing][gpu-readback-probe] chunks={} pre_lens_queue_drain_ms={:.3} pre_lens_queue_drain_polls={} lens_submit_done_wait_ms={:.3} lens_wait_ms={:.3} cmd_wait_ms={:.3} lens_map_after_done_ms={:.3} lens_poll_calls={} cmd_poll_calls={}",
            inputs.len(),
            pre_lens_queue_drain_ms,
            pre_lens_queue_drain_poll_calls,
            lens_submit_done_wait_ms,
            lens_wait_ms,
            cmd_wait_ms,
            lens_map_after_done_ms,
            lens_poll_calls,
            cmd_poll_calls
        );
    }
    Ok((
        out,
        GpuMatchProfile {
            upload_ms: upload_dispatch_ms,
            wait_ms,
            map_copy_ms,
            total_ms,
        },
        GpuSparsePackBatchProfile {
            chunks: inputs.len(),
            lens_bytes_total,
            out_cmd_bytes_total: copied_cmd_bytes_total,
            predicted_payload_bytes_total: 0,
            cap_payload_bytes_total: 0,
            actual_payload_bytes_total: 0,
            predicted_class_counts: [0; GPU_SPARSE_PAYLOAD_CLASS_COUNT],
            cap_class_counts: [0; GPU_SPARSE_PAYLOAD_CLASS_COUNT],
            actual_class_counts: [0; GPU_SPARSE_PAYLOAD_CLASS_COUNT],
            predicted_underestimate_chunks: 0,
            predicted_underestimate_bytes: 0,
            predicted_overestimate_bytes: 0,
            lens_submit_ms,
            lens_submit_done_wait_ms,
            lens_map_after_done_ms,
            lens_poll_calls: lens_poll_calls.saturating_add(cmd_poll_calls),
            lens_yield_calls,
            lens_wait_ms,
            lens_copy_ms,
            prepare_ms,
            table_size_resolve_ms: 0.0,
            prepare_misc_ms: prepare_ms,
            scratch_acquire_ms: 0.0,
            upload_dispatch_ms,
            submit_ms,
            wait_ms: cmd_wait_ms,
            copy_ms: cmd_copy_ms,
            total_ms,
        },
    ))
}

pub(crate) fn is_runtime_available() -> bool {
    runtime().is_ok()
}

#[cfg(test)]
pub(crate) fn reset_decode_slot_pool_for_test() -> Result<(), PDeflateError> {
    let runtime = runtime()?;
    let mut pool = runtime
        .decode_slot_pool
        .lock()
        .map_err(|_| PDeflateError::Gpu("gpu decode slot pool mutex poisoned".to_string()))?;
    *pool = GpuDecodeSlotPool::default();
    Ok(())
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuDecodeSlot {
    pub slot_index: usize,
    pub generation: u64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuDecodeSectionMeta {
    pub cmd_offset: usize,
    pub cmd_len: usize,
    pub out_offset: usize,
    pub out_len: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct GpuDecodeJob<'a> {
    pub chunk_index: usize,
    pub payload: &'a [u8],
    pub table_count: usize,
    pub chunk_uncompressed_len: usize,
    pub out_offset: usize,
    pub out_len: usize,
    pub section_meta: Vec<GpuDecodeSectionMeta>,
    pub preferred_slot: Option<GpuDecodeSlot>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuDecodeDisposition {
    SubmittedGpu,
    CpuFallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuDecodeResult {
    pub chunk_index: usize,
    pub slot: Option<GpuDecodeSlot>,
    pub disposition: GpuDecodeDisposition,
    pub decoded_chunk: Option<Vec<u8>>,
}

fn decode_fallback_results(jobs: &[GpuDecodeJob<'_>]) -> Vec<GpuDecodeResult> {
    jobs.iter()
        .map(|job| GpuDecodeResult {
            chunk_index: job.chunk_index,
            slot: None,
            disposition: GpuDecodeDisposition::CpuFallback,
            decoded_chunk: None,
        })
        .collect()
}

fn read_le_u16_at(src: &[u8], offset: usize) -> Result<u16, PDeflateError> {
    let end = offset
        .checked_add(2)
        .ok_or(PDeflateError::NumericOverflow)?;
    let bytes = src.get(offset..end).ok_or(PDeflateError::InvalidStream(
        "gpu decode chunk header truncated",
    ))?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_le_u32_at(src: &[u8], offset: usize) -> Result<u32, PDeflateError> {
    let end = offset
        .checked_add(4)
        .ok_or(PDeflateError::NumericOverflow)?;
    let bytes = src.get(offset..end).ok_or(PDeflateError::InvalidStream(
        "gpu decode chunk header truncated",
    ))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn parse_gpu_decode_payload_layout(
    payload: &[u8],
) -> Result<GpuDecodePayloadLayout, PDeflateError> {
    if payload.len() < PACK_CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream(
            "gpu decode chunk header truncated",
        ));
    }
    let magic = payload.get(0..4).ok_or(PDeflateError::InvalidStream(
        "gpu decode chunk header truncated",
    ))?;
    if magic != PACK_CHUNK_MAGIC {
        return Err(PDeflateError::InvalidStream("gpu decode bad chunk magic"));
    }
    let version = read_le_u16_at(payload, 4)?;
    if version != PACK_CHUNK_VERSION {
        return Err(PDeflateError::InvalidStream(
            "gpu decode unsupported chunk version",
        ));
    }
    let flags = read_le_u16_at(payload, 6)?;
    if (flags & !(super::CHUNK_FLAG_HUFFMAN | super::CHUNK_FLAG_FINAL_STREAM)) != 0 {
        return Err(PDeflateError::InvalidStream(
            "gpu decode unsupported chunk flags",
        ));
    }
    let huffman_enabled = (flags & super::CHUNK_FLAG_HUFFMAN) != 0;
    let table_count = usize::from(read_le_u16_at(payload, 12)?);
    if table_count > GPU_DECODE_MAX_TABLE_ID {
        return Err(PDeflateError::InvalidStream(
            "gpu decode table_count too large",
        ));
    }
    let table_index_offset = usize::try_from(read_le_u32_at(payload, 16)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let table_data_offset = usize::try_from(read_le_u32_at(payload, 20)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let huff_lut_offset = usize::try_from(read_le_u32_at(payload, 24)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let section_index_offset = usize::try_from(read_le_u32_at(payload, 28)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let section_bitstream_offset = usize::try_from(read_le_u32_at(payload, 32)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    if !(table_index_offset <= table_data_offset
        && table_data_offset <= huff_lut_offset
        && huff_lut_offset <= section_index_offset
        && section_index_offset <= section_bitstream_offset
        && section_bitstream_offset <= payload.len())
    {
        return Err(PDeflateError::InvalidStream(
            "gpu decode chunk offsets invalid",
        ));
    }
    if table_index_offset != PACK_CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream(
            "gpu decode unexpected table_index_offset",
        ));
    }
    let table_index_len = table_data_offset.saturating_sub(table_index_offset);
    if table_index_len != table_count {
        return Err(PDeflateError::InvalidStream(
            "gpu decode table index length mismatch",
        ));
    }
    if huffman_enabled && huff_lut_offset == section_index_offset {
        return Err(PDeflateError::InvalidStream(
            "gpu decode huffman lut block is empty",
        ));
    }
    Ok(GpuDecodePayloadLayout {
        table_count,
        table_index_start: table_index_offset,
        table_data_start: table_data_offset,
        table_data_end: huff_lut_offset,
        huff_lut_start: huff_lut_offset,
        huff_lut_end: section_index_offset,
        huffman_enabled,
        cmd_start: section_bitstream_offset,
    })
}

fn gpu_decode_slot_split(slot_count: usize) -> (usize, usize) {
    let total = slot_count.max(1);
    if total == 1 {
        return (1, 0);
    }
    let large = if total >= 4 { (total / 4).max(1) } else { 1 };
    let normal = total.saturating_sub(large).max(1);
    (normal, total.saturating_sub(normal))
}

fn gpu_decode_normal_caps(options: &PDeflateOptions) -> Result<GpuDecodeSlotCaps, PDeflateError> {
    let base_table = options
        .max_table_entries
        .saturating_mul(GPU_DECODE_TABLE_REPEAT_STRIDE)
        .max(4096);
    Ok(GpuDecodeSlotCaps {
        cmd_cap_bytes: u64::try_from(options.chunk_size.saturating_mul(2).max(64 * 1024))
            .map_err(|_| PDeflateError::NumericOverflow)?,
        section_meta_cap: options.section_count.max(1),
        table_cap_bytes: u64::try_from(base_table).map_err(|_| PDeflateError::NumericOverflow)?,
        out_cap_bytes: u64::try_from(options.chunk_size.max(64 * 1024))
            .map_err(|_| PDeflateError::NumericOverflow)?,
        error_cap_words: options.section_count.max(1),
    })
}

fn gpu_decode_large_caps(base: GpuDecodeSlotCaps) -> GpuDecodeSlotCaps {
    GpuDecodeSlotCaps {
        cmd_cap_bytes: base.cmd_cap_bytes.saturating_mul(2),
        section_meta_cap: base.section_meta_cap.saturating_mul(2),
        table_cap_bytes: base.table_cap_bytes.saturating_mul(2),
        out_cap_bytes: base.out_cap_bytes.saturating_mul(2),
        error_cap_words: base.error_cap_words.saturating_mul(2),
    }
}

fn build_gpu_decode_slot_requirement(
    job: &GpuDecodeJob<'_>,
    layout: GpuDecodePayloadLayout,
) -> Result<GpuDecodeSlotRequirement, PDeflateError> {
    if job.table_count != layout.table_count {
        return Err(PDeflateError::InvalidStream(
            "gpu decode table_count mismatch",
        ));
    }
    if layout.cmd_start > job.payload.len()
        || layout.table_data_end > job.payload.len()
        || layout.huff_lut_end > job.payload.len()
    {
        return Err(PDeflateError::InvalidStream(
            "gpu decode payload layout out of bounds",
        ));
    }
    let cmd_bytes = u64::try_from(job.payload.len().saturating_sub(layout.cmd_start))
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let table_repeat_bytes = layout
        .table_count
        .checked_mul(GPU_DECODE_TABLE_REPEAT_STRIDE)
        .ok_or(PDeflateError::NumericOverflow)?;
    let lut_bytes = layout.huff_lut_end.saturating_sub(layout.huff_lut_start);
    let table_bytes = u64::try_from(
        table_repeat_bytes
            .checked_add(lut_bytes)
            .ok_or(PDeflateError::NumericOverflow)?,
    )
    .map_err(|_| PDeflateError::NumericOverflow)?;
    let out_bytes =
        u64::try_from(job.chunk_uncompressed_len).map_err(|_| PDeflateError::NumericOverflow)?;
    Ok(GpuDecodeSlotRequirement {
        cmd_bytes,
        section_meta_count: job.section_meta.len(),
        table_bytes,
        out_bytes,
        error_words: job.section_meta.len().max(1),
    })
}

fn create_gpu_decode_slot_entry(
    runtime: &GpuMatchRuntime,
    class: GpuDecodeSlotClass,
    local_index: usize,
    caps: GpuDecodeSlotCaps,
) -> Result<GpuDecodeSlotEntry, PDeflateError> {
    let index_base = match class {
        GpuDecodeSlotClass::Normal => 0usize,
        GpuDecodeSlotClass::Large => GPU_DECODE_LARGE_SLOT_INDEX_BASE,
    };
    let slot_index = index_base
        .checked_add(local_index)
        .ok_or(PDeflateError::NumericOverflow)?;
    let cmd_size = u64::try_from(align_up4(
        usize::try_from(caps.cmd_cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?,
    ))
    .map_err(|_| PDeflateError::NumericOverflow)?
    .max(4);
    let section_meta_size = u64::try_from(
        GPU_DECODE_SECTION_META_HEADER_WORDS
            .saturating_add(
                caps.section_meta_cap
                    .saturating_mul(GPU_DECODE_SECTION_META_WORDS),
            )
            .saturating_mul(std::mem::size_of::<u32>()),
    )
    .map_err(|_| PDeflateError::NumericOverflow)?
    .max(4);
    let table_size = u64::try_from(align_up4(
        usize::try_from(caps.table_cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?,
    ))
    .map_err(|_| PDeflateError::NumericOverflow)?
    .max(4);
    let out_size = u64::try_from(align_up4(
        usize::try_from(caps.out_cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?,
    ))
    .map_err(|_| PDeflateError::NumericOverflow)?
    .max(4);
    let error_size = u64::try_from(
        caps.error_cap_words
            .saturating_mul(std::mem::size_of::<u32>()),
    )
    .map_err(|_| PDeflateError::NumericOverflow)?
    .max(4);
    let class_label = match class {
        GpuDecodeSlotClass::Normal => "normal",
        GpuDecodeSlotClass::Large => "large",
    };
    let cmd_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-cmd-slot-{}",
            class_label, local_index
        )),
        size: cmd_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let section_meta_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-section-meta-slot-{}",
            class_label, local_index
        )),
        size: section_meta_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let table_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-table-slot-{}",
            class_label, local_index
        )),
        size: table_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-out-slot-{}",
            class_label, local_index
        )),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-out-readback-slot-{}",
            class_label, local_index
        )),
        size: out_size
            .checked_add(error_size)
            .ok_or(PDeflateError::NumericOverflow)?,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let error_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-error-slot-{}",
            class_label, local_index
        )),
        size: error_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let decode_bind_group = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!(
                "cozip-pdeflate-decode-v2-{}-bg-slot-{}",
                class_label, local_index
            )),
            layout: &runtime.decode_v2_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cmd_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: section_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: error_buffer.as_entire_binding(),
                },
            ],
        });
    Ok(GpuDecodeSlotEntry {
        slot_index,
        generation: 0,
        state: GpuDecodeSlotState::Free,
        caps,
        buffers: GpuDecodeSlotBuffers {
            cmd_buffer,
            section_meta_buffer,
            table_buffer,
            out_buffer,
            out_readback_buffer,
            error_buffer,
        },
        decode_bind_group,
    })
}

fn ensure_gpu_decode_slot_group(
    runtime: &GpuMatchRuntime,
    group: &mut GpuDecodeSlotGroup,
    class: GpuDecodeSlotClass,
    target_count: usize,
    target_caps: GpuDecodeSlotCaps,
) -> Result<(), PDeflateError> {
    if target_count == 0 {
        group.caps = target_caps;
        group.slots.clear();
        return Ok(());
    }
    let merged_caps = group.caps.max_with(target_caps);
    if group.slots.is_empty() || merged_caps != group.caps {
        group.caps = merged_caps;
        group.slots.clear();
    } else {
        group.caps = merged_caps;
    }
    while group.slots.len() < target_count {
        let local_index = group.slots.len();
        group.slots.push(create_gpu_decode_slot_entry(
            runtime,
            class,
            local_index,
            group.caps,
        )?);
    }
    Ok(())
}

fn recycle_gpu_decode_ready_slots(group: &mut GpuDecodeSlotGroup) {
    for slot in &mut group.slots {
        if slot.state == GpuDecodeSlotState::Ready {
            slot.state = GpuDecodeSlotState::Free;
        }
    }
}

fn acquire_gpu_decode_slot(
    group: &mut GpuDecodeSlotGroup,
    req: GpuDecodeSlotRequirement,
) -> Option<(usize, GpuDecodeSlot)> {
    if !group.caps.fits(req) {
        return None;
    }
    for (idx, slot) in group.slots.iter_mut().enumerate() {
        if slot.state == GpuDecodeSlotState::Free {
            slot.state = GpuDecodeSlotState::Inflight;
            slot.generation = slot.generation.saturating_add(1);
            return Some((
                idx,
                GpuDecodeSlot {
                    slot_index: slot.slot_index,
                    generation: slot.generation,
                },
            ));
        }
    }
    None
}

fn acquire_preferred_gpu_decode_slot(
    group: &mut GpuDecodeSlotGroup,
    req: GpuDecodeSlotRequirement,
    preferred: GpuDecodeSlot,
) -> Option<(usize, GpuDecodeSlot)> {
    if !group.caps.fits(req) {
        return None;
    }
    for (idx, slot) in group.slots.iter_mut().enumerate() {
        if slot.slot_index != preferred.slot_index
            || slot.generation != preferred.generation
            || slot.state != GpuDecodeSlotState::Free
        {
            continue;
        }
        slot.state = GpuDecodeSlotState::Inflight;
        slot.generation = slot.generation.saturating_add(1);
        return Some((
            idx,
            GpuDecodeSlot {
                slot_index: slot.slot_index,
                generation: slot.generation,
            },
        ));
    }
    None
}

fn mark_gpu_decode_slot_ready(group: &mut GpuDecodeSlotGroup, slot_idx: usize) {
    if let Some(slot) = group.slots.get_mut(slot_idx) {
        if slot.state == GpuDecodeSlotState::Inflight {
            slot.state = GpuDecodeSlotState::Ready;
        }
    }
}

fn mark_gpu_decode_slot_free(group: &mut GpuDecodeSlotGroup, slot_idx: usize) {
    if let Some(slot) = group.slots.get_mut(slot_idx) {
        slot.state = GpuDecodeSlotState::Free;
    }
}

fn try_acquire_gpu_decode_slot_for_job(
    pool: &mut GpuDecodeSlotPool,
    prepared: PreparedGpuDecodeJob<'_>,
) -> Option<(GpuDecodeSlotClass, usize, GpuDecodeSlot)> {
    let prefer_large = !pool.normal.caps.fits(prepared.req);
    if let Some(preferred) = prepared.job.preferred_slot {
        if preferred.slot_index >= GPU_DECODE_LARGE_SLOT_INDEX_BASE {
            if let Some((idx, slot)) =
                acquire_preferred_gpu_decode_slot(&mut pool.large, prepared.req, preferred)
            {
                return Some((GpuDecodeSlotClass::Large, idx, slot));
            }
        } else if let Some((idx, slot)) =
            acquire_preferred_gpu_decode_slot(&mut pool.normal, prepared.req, preferred)
        {
            return Some((GpuDecodeSlotClass::Normal, idx, slot));
        }
    }
    if prefer_large {
        if let Some((idx, slot)) = acquire_gpu_decode_slot(&mut pool.large, prepared.req) {
            return Some((GpuDecodeSlotClass::Large, idx, slot));
        }
        if let Some((idx, slot)) = acquire_gpu_decode_slot(&mut pool.normal, prepared.req) {
            return Some((GpuDecodeSlotClass::Normal, idx, slot));
        }
    } else {
        if let Some((idx, slot)) = acquire_gpu_decode_slot(&mut pool.normal, prepared.req) {
            return Some((GpuDecodeSlotClass::Normal, idx, slot));
        }
        if let Some((idx, slot)) = acquire_gpu_decode_slot(&mut pool.large, prepared.req) {
            return Some((GpuDecodeSlotClass::Large, idx, slot));
        }
    }
    None
}

fn build_gpu_decode_table_buffer_bytes(
    payload: &[u8],
    layout: GpuDecodePayloadLayout,
) -> Result<Vec<u8>, PDeflateError> {
    let table_index = payload
        .get(layout.table_index_start..layout.table_data_start)
        .ok_or(PDeflateError::InvalidStream(
            "gpu decode table index slice out of bounds",
        ))?;
    if table_index.len() != layout.table_count {
        return Err(PDeflateError::InvalidStream(
            "gpu decode table index length mismatch",
        ));
    }
    let table_data = payload
        .get(layout.table_data_start..layout.table_data_end)
        .ok_or(PDeflateError::InvalidStream(
            "gpu decode table data slice out of bounds",
        ))?;
    let huff_lut = if layout.huffman_enabled {
        let huff_lut = payload
            .get(layout.huff_lut_start..layout.huff_lut_end)
            .ok_or(PDeflateError::InvalidStream(
                "gpu decode huffman lut slice out of bounds",
            ))?;
        if huff_lut.is_empty() {
            return Err(PDeflateError::InvalidStream(
                "gpu decode huffman lut is empty",
            ));
        }
        huff_lut
    } else {
        &[][..]
    };
    let mut table_offsets = Vec::with_capacity(layout.table_count.saturating_add(1));
    table_offsets.push(0usize);
    for &entry_len_u8 in table_index {
        if entry_len_u8 == 0 {
            return Err(PDeflateError::InvalidStream(
                "gpu decode table entry len is zero",
            ));
        }
        let next = table_offsets
            .last()
            .copied()
            .ok_or(PDeflateError::NumericOverflow)?
            .checked_add(usize::from(entry_len_u8))
            .ok_or(PDeflateError::NumericOverflow)?;
        table_offsets.push(next);
    }
    if table_offsets.last().copied().unwrap_or(0) != table_data.len() {
        return Err(PDeflateError::InvalidStream(
            "gpu decode table data length mismatch",
        ));
    }
    let repeat_total = layout
        .table_count
        .checked_mul(GPU_DECODE_TABLE_REPEAT_STRIDE)
        .ok_or(PDeflateError::NumericOverflow)?;
    let mut table_repeat = vec![0u8; repeat_total];
    for id in 0..layout.table_count {
        let t0 = table_offsets[id];
        let t1 = table_offsets[id + 1];
        let entry = table_data.get(t0..t1).ok_or(PDeflateError::InvalidStream(
            "gpu decode table entry range invalid",
        ))?;
        if entry.is_empty() {
            return Err(PDeflateError::InvalidStream("gpu decode empty table entry"));
        }
        let dst = &mut table_repeat
            [id * GPU_DECODE_TABLE_REPEAT_STRIDE..(id + 1) * GPU_DECODE_TABLE_REPEAT_STRIDE];
        for i in 0..dst.len() {
            dst[i] = entry[i % entry.len()];
        }
    }
    let mut table_bytes = table_repeat;
    table_bytes.extend_from_slice(huff_lut);
    Ok(table_bytes)
}

fn upload_gpu_decode_job_to_slot(
    runtime: &GpuMatchRuntime,
    slot: &GpuDecodeSlotEntry,
    prepared: PreparedGpuDecodeJob<'_>,
) -> Result<(), PDeflateError> {
    let table_bytes = build_gpu_decode_table_buffer_bytes(prepared.job.payload, prepared.layout)?;
    let cmd_bytes = prepared
        .job
        .payload
        .get(prepared.layout.cmd_start..)
        .ok_or(PDeflateError::InvalidStream(
            "gpu decode cmd slice out of bounds",
        ))?;
    let mut section_meta_words = Vec::<u32>::with_capacity(
        GPU_DECODE_SECTION_META_HEADER_WORDS
            .saturating_add(prepared.job.section_meta.len() * GPU_DECODE_SECTION_META_WORDS),
    );
    section_meta_words.push(
        u32::try_from(prepared.job.section_meta.len())
            .map_err(|_| PDeflateError::NumericOverflow)?,
    );
    section_meta_words.push(
        u32::try_from(prepared.layout.table_count).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    section_meta_words.push(
        u32::try_from(GPU_DECODE_TABLE_REPEAT_STRIDE)
            .map_err(|_| PDeflateError::NumericOverflow)?,
    );
    section_meta_words.push(
        u32::try_from(
            prepared
                .layout
                .table_count
                .saturating_mul(GPU_DECODE_TABLE_REPEAT_STRIDE),
        )
        .map_err(|_| PDeflateError::NumericOverflow)?,
    );
    section_meta_words.push(
        u32::try_from(if prepared.layout.huffman_enabled {
            prepared
                .layout
                .huff_lut_end
                .saturating_sub(prepared.layout.huff_lut_start)
        } else {
            0
        })
        .map_err(|_| PDeflateError::NumericOverflow)?,
    );
    section_meta_words.push(0u32);
    for section in &prepared.job.section_meta {
        let cmd_end = section
            .cmd_offset
            .checked_add(section.cmd_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        if cmd_end > cmd_bytes.len() {
            return Err(PDeflateError::InvalidStream(
                "gpu decode section cmd range out of bounds",
            ));
        }
        let out_end = section
            .out_offset
            .checked_add(section.out_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        if out_end > prepared.job.chunk_uncompressed_len {
            return Err(PDeflateError::InvalidStream(
                "gpu decode section out range out of bounds",
            ));
        }
        section_meta_words
            .push(u32::try_from(section.cmd_offset).map_err(|_| PDeflateError::NumericOverflow)?);
        section_meta_words
            .push(u32::try_from(section.cmd_len).map_err(|_| PDeflateError::NumericOverflow)?);
        section_meta_words
            .push(u32::try_from(section.out_offset).map_err(|_| PDeflateError::NumericOverflow)?);
        section_meta_words
            .push(u32::try_from(section.out_len).map_err(|_| PDeflateError::NumericOverflow)?);
    }
    if !cmd_bytes.is_empty() {
        let cmd_words = pack_bytes_to_words(cmd_bytes);
        runtime.queue.write_buffer(
            &slot.buffers.cmd_buffer,
            0,
            bytemuck::cast_slice(&cmd_words),
        );
    }
    if !table_bytes.is_empty() {
        let table_words = pack_bytes_to_words(&table_bytes);
        runtime.queue.write_buffer(
            &slot.buffers.table_buffer,
            0,
            bytemuck::cast_slice(&table_words),
        );
    }
    runtime.queue.write_buffer(
        &slot.buffers.section_meta_buffer,
        0,
        bytemuck::cast_slice(&section_meta_words),
    );
    let zero = vec![0u32; prepared.job.section_meta.len().max(1)];
    runtime
        .queue
        .write_buffer(&slot.buffers.error_buffer, 0, bytemuck::cast_slice(&zero));
    Ok(())
}

fn encode_gpu_decode_job_on_slot(
    runtime: &GpuMatchRuntime,
    slot: &GpuDecodeSlotEntry,
    prepared: PreparedGpuDecodeJob<'_>,
    kernel_ts_probe: bool,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<PendingGpuDecodeMap, PDeflateError> {
    let out_data_copy_len = u64::try_from(align_up4(prepared.job.chunk_uncompressed_len))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .max(4);
    let error_words = prepared.job.section_meta.len().max(1);
    let err_copy_len = u64::try_from(error_words.saturating_mul(std::mem::size_of::<u32>()))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .max(4);
    let out_total_copy_len = out_data_copy_len
        .checked_add(err_copy_len)
        .ok_or(PDeflateError::NumericOverflow)?;
    let mut ts_readback_buffer = None;
    let mut ts_query_set = None;
    let mut ts_resolve_buffer = None;
    if kernel_ts_probe {
        ts_query_set = Some(runtime.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("cozip-pdeflate-decode-v2-ts-query"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        }));
        ts_resolve_buffer = Some(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-decode-v2-ts-resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        ts_readback_buffer = Some(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-decode-v2-ts-readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-decode-v2-pass"),
            timestamp_writes: ts_query_set.as_ref().map(|query_set| {
                wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }
            }),
        });
        pass.set_pipeline(&runtime.decode_v2_pipeline);
        pass.set_bind_group(0, &slot.decode_bind_group, &[]);
        let section_count = u32::try_from(prepared.job.section_meta.len())
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let groups_x = section_count.div_ceil(GPU_DECODE_V2_WORKGROUP_SIZE).max(1);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    if let (Some(query_set), Some(resolve_buffer), Some(readback_buffer)) = (
        ts_query_set.as_ref(),
        ts_resolve_buffer.as_ref(),
        ts_readback_buffer.as_ref(),
    ) {
        encoder.resolve_query_set(query_set, 0..2, resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(resolve_buffer, 0, readback_buffer, 0, 16);
    }
    encoder.copy_buffer_to_buffer(
        &slot.buffers.out_buffer,
        0,
        &slot.buffers.out_readback_buffer,
        0,
        out_data_copy_len,
    );
    encoder.copy_buffer_to_buffer(
        &slot.buffers.error_buffer,
        0,
        &slot.buffers.out_readback_buffer,
        out_data_copy_len,
        err_copy_len,
    );
    Ok(PendingGpuDecodeMap {
        out_data_copy_len,
        out_total_copy_len,
        out_len: prepared.job.chunk_uncompressed_len,
        error_words,
        submit_seq: next_gpu_submit_seq(),
        submit_ms: 0.0,
        kernel_timestamp_ms: if kernel_ts_probe {
            None
        } else {
            Some(f64::NAN)
        },
        ts_readback_buffer,
        ts_map_rx: None,
        submit_at: Instant::now(),
        completion_gate: Arc::new(AtomicBool::new(false)),
        map_rx: None,
    })
}

fn collect_gpu_decode_job_from_slot(
    slot: &GpuDecodeSlotEntry,
    pending: &PendingGpuDecodeMap,
) -> Result<(Option<Vec<u8>>, f64), PDeflateError> {
    let t_collect = Instant::now();
    let error_bytes = pending
        .error_words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(PDeflateError::NumericOverflow)?;
    let out_slice = slot
        .buffers
        .out_readback_buffer
        .slice(..pending.out_total_copy_len);
    let mapped = out_slice.get_mapped_range();
    let out_data_offset =
        usize::try_from(pending.out_data_copy_len).map_err(|_| PDeflateError::NumericOverflow)?;
    let err_end = out_data_offset
        .checked_add(error_bytes)
        .ok_or(PDeflateError::NumericOverflow)?
        .min(mapped.len());
    let mut has_error = false;
    if pending.out_len > out_data_offset {
        drop(mapped);
        slot.buffers.out_readback_buffer.unmap();
        return Err(PDeflateError::Gpu(
            "gpu decode readback payload length out of bounds".to_string(),
        ));
    }
    if pending.error_words > 0 {
        if err_end < out_data_offset || err_end > mapped.len() {
            drop(mapped);
            slot.buffers.out_readback_buffer.unmap();
            return Err(PDeflateError::Gpu(
                "gpu decode readback error trailer out of bounds".to_string(),
            ));
        }
        let err_words: &[u32] = bytemuck::cast_slice(&mapped[out_data_offset..err_end]);
        has_error = err_words
            .iter()
            .take(pending.error_words)
            .copied()
            .any(|code| code != 0);
    }
    let result = if has_error {
        Ok(None)
    } else {
        let mut decoded = vec![0u8; pending.out_len];
        if !decoded.is_empty() {
            decoded.copy_from_slice(&mapped[..pending.out_len]);
        }
        Ok(Some(decoded))
    };
    drop(mapped);
    slot.buffers.out_readback_buffer.unmap();
    result.map(|decoded_chunk| (decoded_chunk, elapsed_ms(t_collect)))
}

fn promote_ready_gpu_decode_true_batch<'a>(
    runtime: &GpuMatchRuntime,
    pending_batches: &mut VecDeque<PendingGpuDecodeTrueBatch<'a>>,
    ready_copy_batches: &mut VecDeque<PendingGpuDecodeTrueBatch<'a>>,
    map_timeout_duration: Option<Duration>,
) -> bool {
    let mut ready_idx = None;
    for idx in 0..pending_batches.len() {
        let pending = match pending_batches.get_mut(idx) {
            Some(p) => p,
            None => continue,
        };
        if pending.submit_done_wait_ms == 0.0 && pending.completion_gate.load(Ordering::Acquire) {
            pending.submit_done_wait_ms = elapsed_ms(pending.submit_at);
        }
        if pending.map_done.is_none() {
            match pending.map_rx.try_recv() {
                Ok(Ok(())) => pending.map_done = Some(Ok(())),
                Ok(Err(e)) => {
                    pending.map_done = Some(Err(PDeflateError::Gpu(format!(
                        "gpu decode-v2 true-batch map failed: {e}"
                    ))))
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    pending.map_done = Some(Err(PDeflateError::Gpu(
                        "gpu decode-v2 true-batch map channel closed".to_string(),
                    )))
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        if let Some(timeout) = map_timeout_duration {
            if pending.map_done.is_none() && pending.map_requested_at.elapsed() >= timeout {
                pending.map_done = Some(Err(PDeflateError::Gpu(format!(
                    "gpu decode-v2 true-batch map timed out after {:.3} ms",
                    timeout.as_secs_f64() * 1000.0
                ))));
                pending.map_timed_out = true;
            }
            if pending.map_timed_out && pending.kernel_timestamp_ms.is_none() {
                pending.kernel_timestamp_ms = Some(f64::NAN);
                pending.ts_map_rx = None;
            }
        }
        if pending.map_callback_wait_ms == 0.0 && pending.map_done.is_some() {
            pending.map_callback_wait_ms = elapsed_ms(pending.map_requested_at);
        }
        if pending.kernel_timestamp_ms.is_none() {
            if let Some(ts_rx) = pending.ts_map_rx.as_ref() {
                match ts_rx.try_recv() {
                    Ok(Ok(())) => {
                        let mut kernel_timestamp_ms = f64::NAN;
                        if let Some(ts_buffer) = pending.scratch.ts_readback_buffer.as_ref() {
                            let mapped = ts_buffer.slice(..).get_mapped_range();
                            let ts_words: &[u64] = bytemuck::cast_slice(&mapped);
                            if ts_words.len() >= 2 && ts_words[1] >= ts_words[0] {
                                let period_ms =
                                    (runtime.queue.get_timestamp_period() as f64) / 1_000_000.0;
                                kernel_timestamp_ms =
                                    (ts_words[1].saturating_sub(ts_words[0]) as f64) * period_ms;
                            }
                            drop(mapped);
                            ts_buffer.unmap();
                        }
                        pending.kernel_timestamp_ms = Some(kernel_timestamp_ms);
                        pending.ts_map_rx = None;
                    }
                    Ok(Err(_)) | Err(mpsc::TryRecvError::Disconnected) => {
                        pending.kernel_timestamp_ms = Some(f64::NAN);
                        pending.ts_map_rx = None;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            } else {
                pending.kernel_timestamp_ms = Some(f64::NAN);
            }
        }
        let maps_ready = pending.map_done.is_some();
        let ts_ready = pending.kernel_timestamp_ms.is_some();
        if maps_ready && ts_ready {
            ready_idx = Some(idx);
            break;
        }
    }
    if let Some(idx) = ready_idx {
        if let Some(pending) = pending_batches.remove(idx) {
            ready_copy_batches.push_back(pending);
            return true;
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn decode_chunks_gpu_v2_true_batch(
    runtime: &GpuMatchRuntime,
    prepared_jobs: &[PreparedGpuDecodeJob<'_>],
    default_batch_jobs: usize,
    wait_probe_enabled: bool,
    kernel_ts_probe: bool,
    target_inflight: usize,
    wait_high_watermark: usize,
    spin_polls_before_wait: usize,
    map_timeout_ms: Option<f64>,
) -> Result<Vec<GpuDecodeResult>, PDeflateError> {
    let mut out = Vec::with_capacity(prepared_jobs.len());
    let batch_jobs = gpu_decode_v2_true_batch_jobs(default_batch_jobs)
        .max(1)
        .min(prepared_jobs.len().max(1));
    let copy_jobs = gpu_decode_v2_true_batch_copy_jobs(batch_jobs)
        .max(1)
        .min(batch_jobs);
    let desired_inflight = target_inflight.max(1).min(prepared_jobs.len().max(1));
    let max_jobs_for_target_inflight = prepared_jobs
        .len()
        .saturating_add(desired_inflight.saturating_sub(1))
        / desired_inflight;
    // Keep super-batches large enough to amortize callback/map overhead, but small
    // enough that the current decode window can still realize the requested inflight.
    let super_batch_jobs = max_jobs_for_target_inflight
        .min(batch_jobs.saturating_mul(desired_inflight))
        .max(copy_jobs)
        .min(prepared_jobs.len().max(1));
    let readback_ring = gpu_decode_v2_true_batch_readback_ring(target_inflight.max(1))
        .clamp(1, wait_high_watermark.max(1));
    let deferred_copy_depth = target_inflight.max(1).min(readback_ring.max(1));
    let map_timeout_duration = map_timeout_ms
        .filter(|ms| ms.is_finite() && *ms >= 0.0)
        .map(|ms| Duration::from_secs_f64(ms / 1000.0));

    let mut wait_probe_submit_ms = 0.0_f64;
    let mut wait_probe_submit_done_wait_ms = 0.0_f64;
    let mut wait_probe_map_callback_wait_ms = 0.0_f64;
    let mut wait_probe_map_copy_ms = 0.0_f64;
    let mut wait_probe_kernel_timestamp_ms = 0.0_f64;
    let mut wait_probe_kernel_timestamp_samples = 0usize;
    let mut wait_probe_queue_stall_est_ms = 0.0_f64;
    let mut wait_probe_ready_any_hits = 0usize;
    let mut wait_probe_inflight_batches_sum = 0usize;
    let mut wait_probe_inflight_batches_max = 0usize;
    let mut wait_probe_inflight_samples = 0usize;
    let mut wait_probe_completed_jobs = 0usize;

    let fallback_capacity_jobs = 0usize;
    let mut fallback_submit_error_jobs = 0usize;
    let mut fallback_map_timeout_jobs = 0usize;
    let mut fallback_map_error_jobs = 0usize;
    let mut fallback_collect_error_jobs = 0usize;
    let mut fallback_kernel_error_jobs = 0usize;

    let mut pending_batches = VecDeque::<PendingGpuDecodeTrueBatch<'_>>::new();
    let mut ready_copy_batches = VecDeque::<PendingGpuDecodeTrueBatch<'_>>::new();
    let mut wait_probe_wait_loops = 0usize;

    let mut drain_pending = |mut pending: PendingGpuDecodeTrueBatch<'_>,
                             out: &mut Vec<GpuDecodeResult>| {
        if pending.submit_done_wait_ms == 0.0 {
            runtime
                .device
                .poll(wgpu::Maintain::wait_for(pending.submission));
            pending.submit_done_wait_ms = elapsed_ms(pending.submit_at);
        }
        if pending.map_done.is_none() {
            let map_wait = if let Some(timeout) = map_timeout_duration {
                match pending.map_rx.recv_timeout(timeout) {
                    Ok(Ok(())) => {
                        pending.map_done = Some(Ok(()));
                        false
                    }
                    Ok(Err(e)) => {
                        pending.map_done = Some(Err(PDeflateError::Gpu(format!(
                            "gpu decode-v2 true-batch map failed: {e}"
                        ))));
                        false
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        pending.map_done = Some(Err(PDeflateError::Gpu(format!(
                            "gpu decode-v2 true-batch map timed out after {:.3} ms",
                            timeout.as_secs_f64() * 1000.0
                        ))));
                        pending.map_timed_out = true;
                        true
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        pending.map_done = Some(Err(PDeflateError::Gpu(
                            "gpu decode-v2 true-batch map channel closed".to_string(),
                        )));
                        false
                    }
                }
            } else {
                match pending.map_rx.recv() {
                    Ok(Ok(())) => {
                        pending.map_done = Some(Ok(()));
                        false
                    }
                    Ok(Err(e)) => {
                        pending.map_done = Some(Err(PDeflateError::Gpu(format!(
                            "gpu decode-v2 true-batch map failed: {e}"
                        ))));
                        false
                    }
                    Err(_) => {
                        pending.map_done = Some(Err(PDeflateError::Gpu(
                            "gpu decode-v2 true-batch map channel closed".to_string(),
                        )));
                        false
                    }
                }
            };
            if map_wait {
                pending.kernel_timestamp_ms = Some(f64::NAN);
                pending.ts_map_rx = None;
            }
        }
        if pending.map_callback_wait_ms == 0.0 && pending.map_done.is_some() {
            pending.map_callback_wait_ms = elapsed_ms(pending.map_requested_at);
        }
        if pending.kernel_timestamp_ms.is_none() {
            if let Some(ts_rx) = pending.ts_map_rx.take() {
                let ts_ready = if let Some(timeout) = map_timeout_duration {
                    match ts_rx.recv_timeout(timeout) {
                        Ok(Ok(())) => true,
                        Ok(Err(_)) => false,
                        Err(_) => false,
                    }
                } else {
                    matches!(ts_rx.recv(), Ok(Ok(())))
                };
                if ts_ready {
                    let mut kernel_timestamp_ms = f64::NAN;
                    if let Some(ts_buffer) = pending.scratch.ts_readback_buffer.as_ref() {
                        let mapped = ts_buffer.slice(..).get_mapped_range();
                        let ts_words: &[u64] = bytemuck::cast_slice(&mapped);
                        if ts_words.len() >= 2 && ts_words[1] >= ts_words[0] {
                            let period_ms =
                                (runtime.queue.get_timestamp_period() as f64) / 1_000_000.0;
                            kernel_timestamp_ms =
                                (ts_words[1].saturating_sub(ts_words[0]) as f64) * period_ms;
                        }
                        drop(mapped);
                        ts_buffer.unmap();
                    }
                    pending.kernel_timestamp_ms = Some(kernel_timestamp_ms);
                } else {
                    pending.kernel_timestamp_ms = Some(f64::NAN);
                }
            } else {
                pending.kernel_timestamp_ms = Some(f64::NAN);
            }
        }

        let submit_done_wait_ms = pending.submit_done_wait_ms;
        let map_callback_wait_ms = pending.map_callback_wait_ms;
        let kernel_timestamp_ms = pending.kernel_timestamp_ms.unwrap_or(f64::NAN);
        let map_ok = matches!(pending.map_done.as_ref(), Some(Ok(())));
        let map_timed_out = pending.map_timed_out;

        if !map_ok {
            if map_timed_out {
                fallback_map_timeout_jobs =
                    fallback_map_timeout_jobs.saturating_add(pending.host_jobs.len());
            } else {
                fallback_map_error_jobs =
                    fallback_map_error_jobs.saturating_add(pending.host_jobs.len());
            }
            for host in &pending.host_jobs {
                out.push(GpuDecodeResult {
                    chunk_index: host.prepared.job.chunk_index,
                    slot: None,
                    disposition: GpuDecodeDisposition::CpuFallback,
                    decoded_chunk: None,
                });
            }
            if wait_probe_enabled {
                wait_probe_submit_done_wait_ms += submit_done_wait_ms;
                wait_probe_map_callback_wait_ms += map_callback_wait_ms;
                let queue_stall_est_ms = if kernel_timestamp_ms.is_finite() {
                    (submit_done_wait_ms - kernel_timestamp_ms).max(0.0)
                } else {
                    submit_done_wait_ms.max(0.0)
                };
                wait_probe_queue_stall_est_ms += queue_stall_est_ms;
                if kernel_timestamp_ms.is_finite() {
                    wait_probe_kernel_timestamp_ms += kernel_timestamp_ms;
                    wait_probe_kernel_timestamp_samples =
                        wait_probe_kernel_timestamp_samples.saturating_add(1);
                }
                if pending.completion_gate.load(Ordering::Acquire) {
                    wait_probe_ready_any_hits = wait_probe_ready_any_hits.saturating_add(1);
                }
                wait_probe_completed_jobs =
                    wait_probe_completed_jobs.saturating_add(pending.host_jobs.len());
            }
            release_gpu_decode_v2_true_batch_scratch(runtime, pending.scratch);
            return;
        }

        let t_map_copy = Instant::now();
        let mapped_out = pending
            .scratch
            .out_readback_buffer
            .slice(..pending.out_total_copy_bytes as u64)
            .get_mapped_range();
        let err_region = mapped_out
            .get(pending.out_copy_bytes..)
            .unwrap_or(&mapped_out[mapped_out.len()..]);
        let err_words: &[u32] = bytemuck::cast_slice(err_region);
        for host in &pending.host_jobs {
            let sec_count = host.prepared.job.section_meta.len();
            let err_end = host
                .error_base
                .saturating_add(host.error_words)
                .min(err_words.len());
            let has_error = if sec_count > 0 && host.error_base < err_end {
                err_words[host.error_base..err_end]
                    .iter()
                    .take(sec_count)
                    .copied()
                    .any(|v| v != 0)
            } else {
                false
            };
            if has_error {
                fallback_kernel_error_jobs = fallback_kernel_error_jobs.saturating_add(1);
                out.push(GpuDecodeResult {
                    chunk_index: host.prepared.job.chunk_index,
                    slot: None,
                    disposition: GpuDecodeDisposition::CpuFallback,
                    decoded_chunk: None,
                });
                continue;
            }
            let end = host.out_base.saturating_add(host.out_len);
            if end > mapped_out.len() || host.out_base > end {
                fallback_collect_error_jobs = fallback_collect_error_jobs.saturating_add(1);
                out.push(GpuDecodeResult {
                    chunk_index: host.prepared.job.chunk_index,
                    slot: None,
                    disposition: GpuDecodeDisposition::CpuFallback,
                    decoded_chunk: None,
                });
                continue;
            }
            let mut decoded = vec![0u8; host.out_len];
            if !decoded.is_empty() {
                decoded.copy_from_slice(&mapped_out[host.out_base..end]);
            }
            out.push(GpuDecodeResult {
                chunk_index: host.prepared.job.chunk_index,
                slot: None,
                disposition: GpuDecodeDisposition::SubmittedGpu,
                decoded_chunk: Some(decoded),
            });
        }
        drop(mapped_out);
        pending.scratch.out_readback_buffer.unmap();
        let map_copy_ms = elapsed_ms(t_map_copy);

        if wait_probe_enabled {
            wait_probe_submit_done_wait_ms += submit_done_wait_ms;
            wait_probe_map_callback_wait_ms += map_callback_wait_ms;
            wait_probe_map_copy_ms += map_copy_ms;
            if kernel_timestamp_ms.is_finite() {
                wait_probe_kernel_timestamp_ms += kernel_timestamp_ms;
                wait_probe_kernel_timestamp_samples =
                    wait_probe_kernel_timestamp_samples.saturating_add(1);
                wait_probe_queue_stall_est_ms +=
                    (submit_done_wait_ms - kernel_timestamp_ms).max(0.0);
            } else {
                wait_probe_queue_stall_est_ms += submit_done_wait_ms.max(0.0);
            }
            if pending.completion_gate.load(Ordering::Acquire) {
                wait_probe_ready_any_hits = wait_probe_ready_any_hits.saturating_add(1);
            }
            wait_probe_completed_jobs =
                wait_probe_completed_jobs.saturating_add(pending.host_jobs.len());
        }
        release_gpu_decode_v2_true_batch_scratch(runtime, pending.scratch);
    };

    for round in prepared_jobs.chunks(super_batch_jobs) {
        let mut desc_words = Vec::<u32>::with_capacity(
            1usize.saturating_add(round.len().saturating_mul(GPU_DECODE_V2_BATCH_DESC_WORDS)),
        );
        desc_words.push(u32::try_from(round.len()).map_err(|_| PDeflateError::NumericOverflow)?);
        let mut section_meta_words =
            Vec::<u32>::with_capacity(round.len().saturating_mul(GPU_DECODE_SECTION_META_WORDS));
        let mut cmd_bytes_all = Vec::<u8>::new();
        let mut table_bytes_all = Vec::<u8>::new();
        let mut host_jobs = Vec::<GpuDecodeBatchHostJob<'_>>::with_capacity(round.len());
        let mut out_total_bytes = 0usize;
        let mut error_total_words = 0usize;
        let mut max_sections = 0u32;
        let mut round_failed = false;

        for prepared in round {
            let table_bytes =
                match build_gpu_decode_table_buffer_bytes(prepared.job.payload, prepared.layout) {
                    Ok(v) => v,
                    Err(_) => {
                        round_failed = true;
                        break;
                    }
                };
            let cmd_bytes = match prepared.job.payload.get(prepared.layout.cmd_start..) {
                Some(v) => v,
                None => {
                    round_failed = true;
                    break;
                }
            };
            let section_meta_base = section_meta_words.len();
            let mut section_meta_ok = true;
            for section in &prepared.job.section_meta {
                let cmd_end = match section.cmd_offset.checked_add(section.cmd_len) {
                    Some(v) => v,
                    None => {
                        section_meta_ok = false;
                        break;
                    }
                };
                let out_end = match section.out_offset.checked_add(section.out_len) {
                    Some(v) => v,
                    None => {
                        section_meta_ok = false;
                        break;
                    }
                };
                if cmd_end > cmd_bytes.len() || out_end > prepared.job.chunk_uncompressed_len {
                    section_meta_ok = false;
                    break;
                }
                let cmd_off_u32 = match u32::try_from(section.cmd_offset) {
                    Ok(v) => v,
                    Err(_) => {
                        section_meta_ok = false;
                        break;
                    }
                };
                let cmd_len_u32 = match u32::try_from(section.cmd_len) {
                    Ok(v) => v,
                    Err(_) => {
                        section_meta_ok = false;
                        break;
                    }
                };
                let out_off_u32 = match u32::try_from(section.out_offset) {
                    Ok(v) => v,
                    Err(_) => {
                        section_meta_ok = false;
                        break;
                    }
                };
                let out_len_u32 = match u32::try_from(section.out_len) {
                    Ok(v) => v,
                    Err(_) => {
                        section_meta_ok = false;
                        break;
                    }
                };
                section_meta_words.push(cmd_off_u32);
                section_meta_words.push(cmd_len_u32);
                section_meta_words.push(out_off_u32);
                section_meta_words.push(out_len_u32);
            }
            if !section_meta_ok {
                round_failed = true;
                break;
            }

            let table_count_u32 = match u32::try_from(prepared.layout.table_count) {
                Ok(v) => v,
                Err(_) => {
                    round_failed = true;
                    break;
                }
            };
            let table_repeat_stride_u32 = match u32::try_from(GPU_DECODE_TABLE_REPEAT_STRIDE) {
                Ok(v) => v,
                Err(_) => {
                    round_failed = true;
                    break;
                }
            };
            let section_count_u32 = match u32::try_from(prepared.job.section_meta.len()) {
                Ok(v) => v,
                Err(_) => {
                    round_failed = true;
                    break;
                }
            };
            let huff_lut_off_u32 = match table_count_u32.checked_mul(table_repeat_stride_u32) {
                Some(v) => v,
                None => {
                    round_failed = true;
                    break;
                }
            };
            let huff_lut_len_u32 = match u32::try_from(if prepared.layout.huffman_enabled {
                prepared
                    .layout
                    .huff_lut_end
                    .saturating_sub(prepared.layout.huff_lut_start)
            } else {
                0
            }) {
                Ok(v) => v,
                Err(_) => {
                    round_failed = true;
                    break;
                }
            };
            let section_meta_base_u32 = match u32::try_from(section_meta_base) {
                Ok(v) => v,
                Err(_) => {
                    round_failed = true;
                    break;
                }
            };
            let cmd_base_u32 = match u32::try_from(cmd_bytes_all.len()) {
                Ok(v) => v,
                Err(_) => {
                    round_failed = true;
                    break;
                }
            };
            let cmd_len_u32 = match u32::try_from(cmd_bytes.len()) {
                Ok(v) => v,
                Err(_) => {
                    round_failed = true;
                    break;
                }
            };
            let table_base_u32 = match u32::try_from(table_bytes_all.len()) {
                Ok(v) => v,
                Err(_) => {
                    round_failed = true;
                    break;
                }
            };
            let out_base = align_up4(out_total_bytes);
            let out_base_u32 = match u32::try_from(out_base) {
                Ok(v) => v,
                Err(_) => {
                    round_failed = true;
                    break;
                }
            };
            let out_len = prepared.job.chunk_uncompressed_len;
            let out_len_u32 = match u32::try_from(out_len) {
                Ok(v) => v,
                Err(_) => {
                    round_failed = true;
                    break;
                }
            };
            let error_words = prepared.job.section_meta.len().max(1);
            let error_base = error_total_words;
            let error_base_u32 = match u32::try_from(error_base) {
                Ok(v) => v,
                Err(_) => {
                    round_failed = true;
                    break;
                }
            };
            out_total_bytes = match out_base.checked_add(out_len) {
                Some(v) => v,
                None => {
                    round_failed = true;
                    break;
                }
            };
            error_total_words = match error_total_words.checked_add(error_words) {
                Some(v) => v,
                None => {
                    round_failed = true;
                    break;
                }
            };
            max_sections = max_sections.max(section_count_u32.max(1));

            desc_words.push(section_count_u32);
            desc_words.push(table_count_u32);
            desc_words.push(table_repeat_stride_u32);
            desc_words.push(huff_lut_off_u32);
            desc_words.push(huff_lut_len_u32);
            desc_words.push(section_meta_base_u32);
            desc_words.push(cmd_base_u32);
            desc_words.push(cmd_len_u32);
            desc_words.push(table_base_u32);
            desc_words.push(out_base_u32);
            desc_words.push(out_len_u32);
            desc_words.push(error_base_u32);

            cmd_bytes_all.extend_from_slice(cmd_bytes);
            table_bytes_all.extend_from_slice(&table_bytes);
            host_jobs.push(GpuDecodeBatchHostJob {
                prepared: *prepared,
                out_base,
                out_len,
                error_base,
                error_words,
            });
        }

        if round_failed || host_jobs.is_empty() {
            for prepared in round {
                out.push(GpuDecodeResult {
                    chunk_index: prepared.job.chunk_index,
                    slot: None,
                    disposition: GpuDecodeDisposition::CpuFallback,
                    decoded_chunk: None,
                });
            }
            fallback_submit_error_jobs = fallback_submit_error_jobs.saturating_add(round.len());
            continue;
        }

        let desc_size = u64::try_from(desc_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4)
            .max(4);
        let section_meta_size = u64::try_from(section_meta_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4)
            .max(4);
        let cmd_words = pack_bytes_to_words(&cmd_bytes_all);
        let table_words = pack_bytes_to_words(&table_bytes_all);
        let cmd_size = u64::try_from(cmd_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4)
            .max(4);
        let table_size = u64::try_from(table_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4)
            .max(4);
        let out_copy_bytes = u64::try_from(align_up4(out_total_bytes))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .max(4);
        let err_copy_bytes = u64::try_from(error_total_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4)
            .max(4);
        let err_zero_words = vec![0u32; error_total_words.max(1)];
        let scratch_caps = GpuDecodeV2TrueBatchScratchCaps {
            desc_bytes: desc_size,
            section_meta_bytes: section_meta_size,
            cmd_bytes: cmd_size,
            table_bytes: table_size,
            out_bytes: out_copy_bytes,
            error_bytes: err_copy_bytes,
        };
        let scratch = match acquire_gpu_decode_v2_true_batch_scratch(runtime, scratch_caps) {
            Ok(s) => s,
            Err(_) => {
                for host in &host_jobs {
                    out.push(GpuDecodeResult {
                        chunk_index: host.prepared.job.chunk_index,
                        slot: None,
                        disposition: GpuDecodeDisposition::CpuFallback,
                        decoded_chunk: None,
                    });
                }
                fallback_submit_error_jobs =
                    fallback_submit_error_jobs.saturating_add(host_jobs.len());
                continue;
            }
        };

        runtime
            .queue
            .write_buffer(&scratch.desc_buffer, 0, bytemuck::cast_slice(&desc_words));
        runtime.queue.write_buffer(
            &scratch.section_meta_buffer,
            0,
            bytemuck::cast_slice(&section_meta_words),
        );
        runtime
            .queue
            .write_buffer(&scratch.cmd_buffer, 0, bytemuck::cast_slice(&cmd_words));
        runtime
            .queue
            .write_buffer(&scratch.table_buffer, 0, bytemuck::cast_slice(&table_words));
        runtime.queue.write_buffer(
            &scratch.error_buffer,
            0,
            bytemuck::cast_slice(&err_zero_words),
        );

        let mut ts_map_rx = None;

        let dispatch_x = max_sections.max(1);
        let dispatch_y =
            u32::try_from(host_jobs.len()).map_err(|_| PDeflateError::NumericOverflow)?;
        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-pdeflate-decode-v2-batch-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-decode-v2-batch-pass"),
                timestamp_writes: if kernel_ts_probe {
                    scratch.ts_query_set.as_ref().map(|query_set| {
                        wgpu::ComputePassTimestampWrites {
                            query_set,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }
                    })
                } else {
                    None
                },
            });
            pass.set_pipeline(&runtime.decode_v2_batch_pipeline);
            pass.set_bind_group(0, &scratch.bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        if kernel_ts_probe {
            if let (Some(query_set), Some(resolve_buffer), Some(readback_buffer)) = (
                scratch.ts_query_set.as_ref(),
                scratch.ts_resolve_buffer.as_ref(),
                scratch.ts_readback_buffer.as_ref(),
            ) {
                encoder.resolve_query_set(query_set, 0..2, resolve_buffer, 0);
                encoder.copy_buffer_to_buffer(resolve_buffer, 0, readback_buffer, 0, 16);
            }
        }
        encoder.copy_buffer_to_buffer(
            &scratch.out_buffer,
            0,
            &scratch.out_readback_buffer,
            0,
            out_copy_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &scratch.error_buffer,
            0,
            &scratch.out_readback_buffer,
            out_copy_bytes,
            err_copy_bytes,
        );
        let t_submit = Instant::now();
        let submission = runtime.queue.submit(Some(encoder.finish()));
        let submit_ms = elapsed_ms(t_submit);
        if wait_probe_enabled {
            wait_probe_submit_ms += submit_ms;
        }
        let submit_at = Instant::now();
        let completion_gate = Arc::new(AtomicBool::new(false));
        let completion_gate_cb = Arc::clone(&completion_gate);
        runtime.queue.on_submitted_work_done(move || {
            completion_gate_cb.store(true, Ordering::Release);
        });

        let out_total_copy_bytes = out_copy_bytes
            .checked_add(err_copy_bytes)
            .ok_or(PDeflateError::NumericOverflow)?;
        let out_copy_bytes_usize =
            usize::try_from(out_copy_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
        let out_total_copy_bytes_usize =
            usize::try_from(out_total_copy_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
        let map_slice = scratch.out_readback_buffer.slice(..out_total_copy_bytes);
        let (map_tx, map_rx) = mpsc::channel();
        map_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = map_tx.send(res);
        });
        if kernel_ts_probe {
            if let Some(ts_buffer) = scratch.ts_readback_buffer.as_ref() {
                let (ts_tx, rx) = mpsc::channel();
                ts_buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |res| {
                        let _ = ts_tx.send(res);
                    });
                ts_map_rx = Some(rx);
            }
        }

        pending_batches.push_back(PendingGpuDecodeTrueBatch {
            host_jobs,
            out_copy_bytes: out_copy_bytes_usize,
            out_total_copy_bytes: out_total_copy_bytes_usize,
            scratch,
            map_rx,
            ts_map_rx,
            submit_at,
            map_requested_at: Instant::now(),
            completion_gate,
            submission,
            map_done: None,
            map_timed_out: false,
            kernel_timestamp_ms: if kernel_ts_probe {
                None
            } else {
                Some(f64::NAN)
            },
            submit_done_wait_ms: 0.0,
            map_callback_wait_ms: 0.0,
        });

        if wait_probe_enabled {
            wait_probe_inflight_samples = wait_probe_inflight_samples.saturating_add(1);
            wait_probe_inflight_batches_sum =
                wait_probe_inflight_batches_sum.saturating_add(pending_batches.len());
            wait_probe_inflight_batches_max =
                wait_probe_inflight_batches_max.max(pending_batches.len());
        }

        while pending_batches.len() >= readback_ring
            || ready_copy_batches.len() >= deferred_copy_depth
        {
            if ready_copy_batches.len() >= deferred_copy_depth {
                if let Some(pending) = ready_copy_batches.pop_front() {
                    drain_pending(pending, &mut out);
                    continue;
                }
            }
            runtime.device.poll(wgpu::Maintain::Poll);
            if promote_ready_gpu_decode_true_batch(
                runtime,
                &mut pending_batches,
                &mut ready_copy_batches,
                map_timeout_duration,
            ) {
                continue;
            }
            if let Some(pending) = ready_copy_batches.pop_front() {
                drain_pending(pending, &mut out);
                continue;
            }
            wait_probe_wait_loops = wait_probe_wait_loops.saturating_add(1);
            runtime.device.poll(wgpu::Maintain::Wait);
        }
    }

    while !pending_batches.is_empty() || !ready_copy_batches.is_empty() {
        if pending_batches.is_empty() {
            if let Some(pending) = ready_copy_batches.pop_front() {
                drain_pending(pending, &mut out);
                continue;
            }
        }
        runtime.device.poll(wgpu::Maintain::Poll);
        if promote_ready_gpu_decode_true_batch(
            runtime,
            &mut pending_batches,
            &mut ready_copy_batches,
            map_timeout_duration,
        ) {
            continue;
        }
        if let Some(pending) = ready_copy_batches.pop_front() {
            drain_pending(pending, &mut out);
            continue;
        }
        wait_probe_wait_loops = wait_probe_wait_loops.saturating_add(1);
        runtime.device.poll(wgpu::Maintain::Wait);
    }

    out.sort_by_key(|r| r.chunk_index);

    if wait_probe_enabled {
        let inflight_avg = if wait_probe_inflight_samples > 0 {
            (wait_probe_inflight_batches_sum as f64) / (wait_probe_inflight_samples as f64)
        } else {
            0.0
        };
        let kernel_ts_status = if wait_probe_kernel_timestamp_samples > 0 {
            "ok"
        } else {
            "disabled"
        };
        eprintln!(
            "[cozip_pdeflate][timing][decode-v2-ready-any] jobs={} submit_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_ms={:.3} map_copy_ms={:.3} kernel_timestamp_ms={:.3} queue_stall_est_ms={:.3} ready_any_hits={} wait_loops={} inflight_batches_avg={:.2} inflight_batches_max={} inflight_samples={} kernel_ts_status={} ctrl_target_inflight={} ctrl_wait_high={} ctrl_spin_polls={}",
            wait_probe_completed_jobs,
            wait_probe_submit_ms,
            wait_probe_submit_done_wait_ms,
            wait_probe_map_callback_wait_ms,
            wait_probe_map_copy_ms,
            wait_probe_kernel_timestamp_ms,
            wait_probe_queue_stall_est_ms,
            wait_probe_ready_any_hits,
            wait_probe_wait_loops,
            inflight_avg,
            wait_probe_inflight_batches_max,
            wait_probe_inflight_samples,
            kernel_ts_status,
            target_inflight,
            wait_high_watermark,
            spin_polls_before_wait
        );
        eprintln!(
            "[cozip_pdeflate][timing][decode-v2-fallback-reasons] jobs={} capacity={} submit_err={} map_timeout={} map_err={} collect_err={} kernel_fallback={}",
            wait_probe_completed_jobs,
            fallback_capacity_jobs,
            fallback_submit_error_jobs,
            fallback_map_timeout_jobs,
            fallback_map_error_jobs,
            fallback_collect_error_jobs,
            fallback_kernel_error_jobs
        );
    }

    Ok(out)
}

fn gpu_decode_section_boundaries_word_aligned(section_meta: &[GpuDecodeSectionMeta]) -> bool {
    // decode-v2 section-parallel kernel writes bytes via u32 RMW; internal section boundaries
    // must be word-aligned to avoid cross-section word races.
    section_meta
        .iter()
        .enumerate()
        .all(|(idx, sec)| idx == 0 || (sec.out_offset & 3) == 0)
}

pub(crate) fn decode_chunks_gpu_v2(
    jobs: &[GpuDecodeJob<'_>],
    options: &PDeflateOptions,
) -> Result<Vec<GpuDecodeResult>, PDeflateError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let gpu_requested = options.gpu_decompress_enabled || options.gpu_decompress_force_gpu;
    if !gpu_requested || !is_runtime_available() {
        return Ok(decode_fallback_results(jobs));
    }
    let runtime = runtime()?;

    let mut out = Vec::with_capacity(jobs.len());
    let mut prepared_jobs = Vec::with_capacity(jobs.len());
    let mut pre_fallback_unaligned_count = 0usize;
    let mut pre_fallback_unaligned_first_chunk = None::<usize>;
    for job in jobs {
        if !gpu_decode_section_boundaries_word_aligned(&job.section_meta) {
            pre_fallback_unaligned_count = pre_fallback_unaligned_count.saturating_add(1);
            pre_fallback_unaligned_first_chunk.get_or_insert(job.chunk_index);
            out.push(GpuDecodeResult {
                chunk_index: job.chunk_index,
                slot: None,
                disposition: GpuDecodeDisposition::CpuFallback,
                decoded_chunk: None,
            });
            continue;
        }
        let layout = parse_gpu_decode_payload_layout(job.payload)?;
        let req = build_gpu_decode_slot_requirement(job, layout)?;
        prepared_jobs.push(PreparedGpuDecodeJob { job, req, layout });
    }
    if pre_fallback_unaligned_count > 0 {
        eprintln!(
            "[cozip_pdeflate][timing][decode-v2-fallback] reason=unaligned_section_boundary chunks={} first_chunk_idx={} gpu_force={} note=\"section out_offset not 4-byte aligned; cpu fallback\"",
            pre_fallback_unaligned_count,
            pre_fallback_unaligned_first_chunk.unwrap_or(usize::MAX),
            options.gpu_decompress_force_gpu
        );
    }
    if prepared_jobs.is_empty() {
        out.sort_by_key(|r| r.chunk_index);
        return Ok(out);
    }

    let micro_submit = options.gpu_submit_chunks.max(1);
    let max_inflight = options.gpu_slot_count.max(1);
    let target_inflight = gpu_decode_v2_target_inflight(max_inflight);
    let wait_high_watermark = gpu_decode_v2_wait_high_watermark(max_inflight, target_inflight);
    let spin_polls_before_wait = gpu_decode_v2_spin_polls_before_wait();
    let map_timeout_ms = gpu_decode_v2_map_timeout_ms();
    let wait_probe_enabled = gpu_decode_v2_wait_probe_enabled();
    let kernel_ts_probe = wait_probe_enabled && runtime.supports_timestamp_query;
    if wait_probe_enabled && !runtime.supports_timestamp_query {
        let _ = GPU_DECODE_V2_KERNEL_TS_UNSUPPORTED_LOGGED.get_or_init(|| {
            eprintln!(
                "[cozip_pdeflate][timing][decode-v2-wait-breakdown] status=disabled reason=timestamp_query_not_supported"
            );
        });
    }

    if gpu_decode_v2_true_batch_enabled() {
        let mut decoded = decode_chunks_gpu_v2_true_batch(
            runtime,
            &prepared_jobs,
            micro_submit,
            wait_probe_enabled,
            kernel_ts_probe,
            target_inflight,
            wait_high_watermark,
            spin_polls_before_wait,
            map_timeout_ms,
        )?;
        out.append(&mut decoded);
        out.sort_by_key(|r| r.chunk_index);
        return Ok(out);
    }

    let (normal_target_count, large_target_count) = gpu_decode_slot_split(options.gpu_slot_count);
    let normal_base = gpu_decode_normal_caps(options)?;
    let mut large_base = gpu_decode_large_caps(normal_base);
    let mut normal_req_max = GpuDecodeSlotCaps::default();
    let mut large_req_max = GpuDecodeSlotCaps::default();
    for prepared in &prepared_jobs {
        let req_caps = GpuDecodeSlotCaps {
            cmd_cap_bytes: prepared.req.cmd_bytes,
            section_meta_cap: prepared.req.section_meta_count,
            table_cap_bytes: prepared.req.table_bytes,
            out_cap_bytes: prepared.req.out_bytes,
            error_cap_words: prepared.req.error_words,
        };
        if normal_base.fits(prepared.req) {
            normal_req_max = normal_req_max.max_with(req_caps);
        } else {
            large_req_max = large_req_max.max_with(req_caps);
        }
    }
    let normal_target_caps = normal_base.max_with(normal_req_max);
    if large_req_max.cmd_cap_bytes > 0
        || large_req_max.section_meta_cap > 0
        || large_req_max.table_cap_bytes > 0
        || large_req_max.out_cap_bytes > 0
        || large_req_max.error_cap_words > 0
    {
        large_base = large_base.max_with(large_req_max);
    }

    let mut pool = runtime
        .decode_slot_pool
        .lock()
        .map_err(|_| PDeflateError::Gpu("gpu decode slot pool mutex poisoned".to_string()))?;
    ensure_gpu_decode_slot_group(
        runtime,
        &mut pool.normal,
        GpuDecodeSlotClass::Normal,
        normal_target_count,
        normal_target_caps,
    )?;
    ensure_gpu_decode_slot_group(
        runtime,
        &mut pool.large,
        GpuDecodeSlotClass::Large,
        large_target_count,
        large_base,
    )?;

    let mut next_submit_idx = 0usize;
    let mut inflight = Vec::<InflightGpuDecodeSubmission<'_>>::new();
    let mut completed_queue = VecDeque::<GpuDecodeResult>::new();
    let mut wait_probe_submit_ms = 0.0_f64;
    let mut wait_probe_submit_done_wait_ms = 0.0_f64;
    let mut wait_probe_map_callback_wait_ms = 0.0_f64;
    let mut wait_probe_map_copy_ms = 0.0_f64;
    let mut wait_probe_kernel_timestamp_ms = 0.0_f64;
    let mut wait_probe_kernel_timestamp_samples = 0usize;
    let mut wait_probe_queue_stall_est_ms = 0.0_f64;
    let mut wait_probe_ready_any_hits = 0usize;
    let mut wait_probe_wait_loops = 0usize;
    let mut wait_probe_inflight_batches_sum = 0usize;
    let mut wait_probe_inflight_batches_max = 0usize;
    let mut wait_probe_inflight_samples = 0usize;
    let mut wait_probe_completed_jobs = 0usize;
    let mut fallback_capacity_jobs = 0usize;
    let mut fallback_submit_error_jobs = 0usize;
    let mut fallback_map_timeout_jobs = 0usize;
    let mut fallback_map_error_jobs = 0usize;
    let mut fallback_collect_error_jobs = 0usize;
    let mut fallback_kernel_error_jobs = 0usize;
    let mut no_progress_loops = 0usize;

    while next_submit_idx < prepared_jobs.len() || !inflight.is_empty() {
        recycle_gpu_decode_ready_slots(&mut pool.normal);
        recycle_gpu_decode_ready_slots(&mut pool.large);

        // Stage 1: Complete (recover credit first)
        runtime.device.poll(wgpu::Maintain::Poll);
        let mut completed_in_round = 0usize;
        let mut i = 0usize;
        while i < inflight.len() {
            {
                let pending = &mut inflight[i];

                // Batch-level completion gate (single callback per submit batch).
                if pending.wait_profile.submit_done_wait_ms == 0.0
                    && pending.pending.completion_gate.load(Ordering::Acquire)
                {
                    pending.wait_profile.submit_seq = pending.pending.submit_seq;
                    pending.wait_profile.submit_ms = pending.pending.submit_ms;
                    pending.wait_profile.submit_done_wait_ms =
                        elapsed_ms(pending.pending.submit_at);
                }

                if pending.pending.kernel_timestamp_ms.is_none() {
                    if let Some(ts_rx) = pending.pending.ts_map_rx.as_ref() {
                        match ts_rx.try_recv() {
                            Ok(Ok(())) => {
                                if let Some(ts_buffer) = pending.pending.ts_readback_buffer.take() {
                                    let mapped = ts_buffer.slice(..).get_mapped_range();
                                    let ts_words: &[u64] = bytemuck::cast_slice(&mapped);
                                    let kernel_ms = if ts_words.len() >= 2
                                        && ts_words[1] >= ts_words[0]
                                    {
                                        let period_ms = (runtime.queue.get_timestamp_period()
                                            as f64)
                                            / 1_000_000.0;
                                        (ts_words[1].saturating_sub(ts_words[0]) as f64) * period_ms
                                    } else {
                                        f64::NAN
                                    };
                                    drop(mapped);
                                    ts_buffer.unmap();
                                    pending.pending.kernel_timestamp_ms = Some(kernel_ms);
                                } else {
                                    pending.pending.kernel_timestamp_ms = Some(f64::NAN);
                                }
                                pending.pending.ts_map_rx = None;
                            }
                            Ok(Err(_)) | Err(mpsc::TryRecvError::Disconnected) => {
                                let _ = pending.pending.ts_readback_buffer.take();
                                pending.pending.kernel_timestamp_ms = Some(f64::NAN);
                                pending.pending.ts_map_rx = None;
                            }
                            Err(mpsc::TryRecvError::Empty) => {}
                        }
                    }
                }

                if pending.map_done.is_none() {
                    if let Some(map_rx) = pending.pending.map_rx.as_ref() {
                        match map_rx.try_recv() {
                            Ok(res) => {
                                if let Some(t_map_wait) = pending.map_requested_at.take() {
                                    let wait_ms = elapsed_ms(t_map_wait);
                                    pending.wait_profile.map_callback_wait_ms = wait_ms;
                                    if pending.wait_profile.submit_done_wait_ms == 0.0 {
                                        pending.wait_profile.submit_seq =
                                            pending.pending.submit_seq;
                                        pending.wait_profile.submit_ms = pending.pending.submit_ms;
                                        pending.wait_profile.submit_done_wait_ms =
                                            elapsed_ms(pending.pending.submit_at);
                                    }
                                }
                                pending.map_done = Some(res.map_err(|e| {
                                    PDeflateError::Gpu(format!("gpu decode out map failed: {e}"))
                                }));
                            }
                            Err(mpsc::TryRecvError::Empty) => {}
                            Err(mpsc::TryRecvError::Disconnected) => {
                                if let Some(t_map_wait) = pending.map_requested_at.take() {
                                    let wait_ms = elapsed_ms(t_map_wait);
                                    pending.wait_profile.map_callback_wait_ms = wait_ms;
                                    if pending.wait_profile.submit_done_wait_ms == 0.0 {
                                        pending.wait_profile.submit_seq =
                                            pending.pending.submit_seq;
                                        pending.wait_profile.submit_ms = pending.pending.submit_ms;
                                        pending.wait_profile.submit_done_wait_ms =
                                            elapsed_ms(pending.pending.submit_at);
                                    }
                                }
                                pending.map_done = Some(Err(PDeflateError::Gpu(
                                    "gpu decode out map channel closed".to_string(),
                                )));
                            }
                        }
                    }
                }

                if pending.map_done.is_none() {
                    if let (Some(timeout_ms), Some(t_map_wait)) =
                        (map_timeout_ms, pending.map_requested_at)
                    {
                        if elapsed_ms(t_map_wait) >= timeout_ms {
                            pending.wait_profile.map_callback_wait_ms = elapsed_ms(t_map_wait);
                            if pending.wait_profile.submit_done_wait_ms == 0.0 {
                                pending.wait_profile.submit_seq = pending.pending.submit_seq;
                                pending.wait_profile.submit_ms = pending.pending.submit_ms;
                                pending.wait_profile.submit_done_wait_ms =
                                    elapsed_ms(pending.pending.submit_at);
                            }
                            pending.map_requested_at = None;
                            pending.pending.map_rx = None;
                            pending.map_timed_out = true;
                            pending.map_done = Some(Err(PDeflateError::Gpu(format!(
                                "gpu decode out map timed out after {:.3} ms",
                                timeout_ms
                            ))));
                        }
                    }
                }
            }

            let done =
                inflight[i].map_done.is_some() && inflight[i].pending.kernel_timestamp_ms.is_some();
            if !done {
                i = i.saturating_add(1);
                continue;
            }

            let mut submission = inflight.swap_remove(i);
            let map_done = submission.map_done.take();
            let map_ok = matches!(map_done, Some(Ok(())));

            let decode_result = if map_ok {
                let collected = match submission.class {
                    GpuDecodeSlotClass::Normal => {
                        let slot = &pool.normal.slots[submission.slot_idx];
                        collect_gpu_decode_job_from_slot(slot, &submission.pending)
                    }
                    GpuDecodeSlotClass::Large => {
                        let slot = &pool.large.slots[submission.slot_idx];
                        collect_gpu_decode_job_from_slot(slot, &submission.pending)
                    }
                };
                match collected {
                    Ok((decoded_chunk, map_copy_ms)) => {
                        submission.wait_profile.map_copy_ms = map_copy_ms;
                        if decoded_chunk.is_none() {
                            fallback_kernel_error_jobs =
                                fallback_kernel_error_jobs.saturating_add(1);
                        }
                        GpuDecodeResult {
                            chunk_index: submission.prepared.job.chunk_index,
                            slot: Some(submission.slot_handle),
                            disposition: if decoded_chunk.is_some() {
                                GpuDecodeDisposition::SubmittedGpu
                            } else {
                                GpuDecodeDisposition::CpuFallback
                            },
                            decoded_chunk,
                        }
                    }
                    Err(_) => {
                        fallback_collect_error_jobs = fallback_collect_error_jobs.saturating_add(1);
                        GpuDecodeResult {
                            chunk_index: submission.prepared.job.chunk_index,
                            slot: None,
                            disposition: GpuDecodeDisposition::CpuFallback,
                            decoded_chunk: None,
                        }
                    }
                }
            } else {
                if submission.map_timed_out {
                    fallback_map_timeout_jobs = fallback_map_timeout_jobs.saturating_add(1);
                } else {
                    fallback_map_error_jobs = fallback_map_error_jobs.saturating_add(1);
                }
                GpuDecodeResult {
                    chunk_index: submission.prepared.job.chunk_index,
                    slot: None,
                    disposition: GpuDecodeDisposition::CpuFallback,
                    decoded_chunk: None,
                }
            };
            if wait_probe_enabled {
                submission.wait_profile.kernel_timestamp_ms =
                    submission.pending.kernel_timestamp_ms.unwrap_or(f64::NAN);
                let kernel_ms = submission.wait_profile.kernel_timestamp_ms;
                submission.wait_profile.queue_stall_est_ms = if kernel_ms.is_finite() {
                    (submission.wait_profile.submit_done_wait_ms - kernel_ms).max(0.0)
                } else {
                    submission.wait_profile.submit_done_wait_ms.max(0.0)
                };
                wait_probe_submit_done_wait_ms += submission.wait_profile.submit_done_wait_ms;
                wait_probe_map_callback_wait_ms += submission.wait_profile.map_callback_wait_ms;
                wait_probe_map_copy_ms += submission.wait_profile.map_copy_ms;
                wait_probe_queue_stall_est_ms += submission.wait_profile.queue_stall_est_ms;
                if kernel_ms.is_finite() {
                    wait_probe_kernel_timestamp_ms += kernel_ms;
                    wait_probe_kernel_timestamp_samples =
                        wait_probe_kernel_timestamp_samples.saturating_add(1);
                }
                wait_probe_completed_jobs = wait_probe_completed_jobs.saturating_add(1);
                let seq = GPU_DECODE_V2_WAIT_PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
                if gpu_decode_v2_wait_probe_should_log(seq) {
                    let kernel_ts_status = if kernel_ms.is_finite() {
                        "ok"
                    } else {
                        "disabled"
                    };
                    eprintln!(
                        "[cozip_pdeflate][timing][decode-v2-wait-breakdown] seq={} chunk_idx={} slot_idx={} submit_seq={} submit_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_ms={:.3} map_copy_ms={:.3} kernel_timestamp_ms={:.3} queue_stall_est_ms={:.3} kernel_ts_status={}",
                        seq,
                        submission.prepared.job.chunk_index,
                        submission.slot_handle.slot_index,
                        submission.wait_profile.submit_seq,
                        submission.wait_profile.submit_ms,
                        submission.wait_profile.submit_done_wait_ms,
                        submission.wait_profile.map_callback_wait_ms,
                        submission.wait_profile.map_copy_ms,
                        if kernel_ms.is_finite() {
                            kernel_ms
                        } else {
                            0.0
                        },
                        submission.wait_profile.queue_stall_est_ms,
                        kernel_ts_status
                    );
                }
            }

            match submission.class {
                GpuDecodeSlotClass::Normal => {
                    mark_gpu_decode_slot_ready(&mut pool.normal, submission.slot_idx)
                }
                GpuDecodeSlotClass::Large => {
                    mark_gpu_decode_slot_ready(&mut pool.large, submission.slot_idx)
                }
            }
            completed_queue.push_back(decode_result);
            completed_in_round = completed_in_round.saturating_add(1);
        }

        if wait_probe_enabled {
            wait_probe_inflight_samples = wait_probe_inflight_samples.saturating_add(1);
            wait_probe_inflight_batches_sum =
                wait_probe_inflight_batches_sum.saturating_add(inflight.len());
            wait_probe_inflight_batches_max = wait_probe_inflight_batches_max.max(inflight.len());
        }

        // Stage 2: Submit (micro-submit; avoid oversized single-submit HOL stalls)
        let mut submitted_in_round = 0usize;
        let mut submit_stalled = false;
        let submit_inflight_limit = if inflight.len() < target_inflight {
            target_inflight
        } else {
            max_inflight
        };
        let submit_budget = micro_submit.max(submit_inflight_limit.saturating_sub(inflight.len()));
        while next_submit_idx < prepared_jobs.len()
            && submitted_in_round < submit_budget
            && inflight.len() < submit_inflight_limit
        {
            let prepared = prepared_jobs[next_submit_idx];
            let acquired = try_acquire_gpu_decode_slot_for_job(&mut pool, prepared);
            let Some((class, slot_idx, slot_handle)) = acquired else {
                let fit_any =
                    pool.normal.caps.fits(prepared.req) || pool.large.caps.fits(prepared.req);
                if !fit_any {
                    completed_queue.push_back(GpuDecodeResult {
                        chunk_index: prepared.job.chunk_index,
                        slot: None,
                        disposition: GpuDecodeDisposition::CpuFallback,
                        decoded_chunk: None,
                    });
                    fallback_capacity_jobs = fallback_capacity_jobs.saturating_add(1);
                    next_submit_idx = next_submit_idx.saturating_add(1);
                    submitted_in_round = submitted_in_round.saturating_add(1);
                    continue;
                }
                submit_stalled = true;
                break;
            };

            let mut encoder =
                runtime
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("cozip-pdeflate-decode-v2-encoder"),
                    });
            let stage_result = match class {
                GpuDecodeSlotClass::Normal => {
                    let slot = &pool.normal.slots[slot_idx];
                    upload_gpu_decode_job_to_slot(runtime, slot, prepared).and_then(|_| {
                        encode_gpu_decode_job_on_slot(
                            runtime,
                            slot,
                            prepared,
                            kernel_ts_probe,
                            &mut encoder,
                        )
                    })
                }
                GpuDecodeSlotClass::Large => {
                    let slot = &pool.large.slots[slot_idx];
                    upload_gpu_decode_job_to_slot(runtime, slot, prepared).and_then(|_| {
                        encode_gpu_decode_job_on_slot(
                            runtime,
                            slot,
                            prepared,
                            kernel_ts_probe,
                            &mut encoder,
                        )
                    })
                }
            };
            match stage_result {
                Ok(mut pending) => {
                    let t_submit = Instant::now();
                    let _submission = runtime.queue.submit(Some(encoder.finish()));
                    pending.submit_ms = elapsed_ms(t_submit);
                    if wait_probe_enabled {
                        wait_probe_submit_ms += pending.submit_ms;
                    }
                    pending.submit_at = Instant::now();
                    let completion_gate = Arc::new(AtomicBool::new(false));
                    let completion_gate_cb = Arc::clone(&completion_gate);
                    runtime.queue.on_submitted_work_done(move || {
                        completion_gate_cb.store(true, Ordering::Release);
                    });
                    pending.completion_gate = completion_gate;
                    if pending.map_rx.is_none() {
                        let out_slice = match class {
                            GpuDecodeSlotClass::Normal => pool.normal.slots[slot_idx]
                                .buffers
                                .out_readback_buffer
                                .slice(..pending.out_total_copy_len),
                            GpuDecodeSlotClass::Large => pool.large.slots[slot_idx]
                                .buffers
                                .out_readback_buffer
                                .slice(..pending.out_total_copy_len),
                        };
                        let (map_tx, map_rx) = mpsc::channel();
                        out_slice.map_async(wgpu::MapMode::Read, move |res| {
                            let _ = map_tx.send(res);
                        });
                        pending.map_rx = Some(map_rx);
                    }
                    if pending.kernel_timestamp_ms.is_none() && pending.ts_map_rx.is_none() {
                        if let Some(ts_buffer) = pending.ts_readback_buffer.as_ref() {
                            let (ts_tx, ts_rx) = mpsc::channel();
                            ts_buffer
                                .slice(..)
                                .map_async(wgpu::MapMode::Read, move |res| {
                                    let _ = ts_tx.send(res);
                                });
                            pending.ts_map_rx = Some(ts_rx);
                        }
                    }
                    inflight.push(InflightGpuDecodeSubmission {
                        prepared,
                        class,
                        slot_idx,
                        slot_handle,
                        pending: pending,
                        map_requested_at: Some(Instant::now()),
                        map_done: None,
                        map_timed_out: false,
                        wait_profile: GpuDecodeV2WaitProfile::default(),
                    });
                }
                Err(_) => {
                    match class {
                        GpuDecodeSlotClass::Normal => {
                            mark_gpu_decode_slot_free(&mut pool.normal, slot_idx)
                        }
                        GpuDecodeSlotClass::Large => {
                            mark_gpu_decode_slot_free(&mut pool.large, slot_idx)
                        }
                    }
                    completed_queue.push_back(GpuDecodeResult {
                        chunk_index: prepared.job.chunk_index,
                        slot: None,
                        disposition: GpuDecodeDisposition::CpuFallback,
                        decoded_chunk: None,
                    });
                    fallback_submit_error_jobs = fallback_submit_error_jobs.saturating_add(1);
                }
            }
            next_submit_idx = next_submit_idx.saturating_add(1);
            submitted_in_round = submitted_in_round.saturating_add(1);
        }

        // Stage 3: CPU-side finalize after submit prioritization.
        while let Some(result) = completed_queue.pop_front() {
            out.push(result);
        }

        if completed_in_round == 0 && submitted_in_round == 0 {
            no_progress_loops = no_progress_loops.saturating_add(1);
            let should_block_wait = submit_stalled || inflight.len() >= wait_high_watermark;
            if should_block_wait && no_progress_loops >= spin_polls_before_wait {
                if wait_probe_enabled {
                    wait_probe_wait_loops = wait_probe_wait_loops.saturating_add(1);
                }
                runtime.device.poll(wgpu::Maintain::Wait);
                no_progress_loops = 0;
            } else {
                std::thread::yield_now();
            }
        } else {
            no_progress_loops = 0;
            if wait_probe_enabled && completed_in_round > 0 {
                wait_probe_ready_any_hits = wait_probe_ready_any_hits.saturating_add(1);
            }
        }
    }

    out.sort_by_key(|r| r.chunk_index);
    recycle_gpu_decode_ready_slots(&mut pool.normal);
    recycle_gpu_decode_ready_slots(&mut pool.large);
    if wait_probe_enabled {
        let inflight_avg = if wait_probe_inflight_samples > 0 {
            (wait_probe_inflight_batches_sum as f64) / (wait_probe_inflight_samples as f64)
        } else {
            0.0
        };
        let kernel_ts_status = if wait_probe_kernel_timestamp_samples > 0 {
            "ok"
        } else {
            "disabled"
        };
        eprintln!(
            "[cozip_pdeflate][timing][decode-v2-ready-any] jobs={} submit_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_ms={:.3} map_copy_ms={:.3} kernel_timestamp_ms={:.3} queue_stall_est_ms={:.3} ready_any_hits={} wait_loops={} inflight_batches_avg={:.2} inflight_batches_max={} inflight_samples={} kernel_ts_status={} ctrl_target_inflight={} ctrl_wait_high={} ctrl_spin_polls={}",
            wait_probe_completed_jobs,
            wait_probe_submit_ms,
            wait_probe_submit_done_wait_ms,
            wait_probe_map_callback_wait_ms,
            wait_probe_map_copy_ms,
            wait_probe_kernel_timestamp_ms,
            wait_probe_queue_stall_est_ms,
            wait_probe_ready_any_hits,
            wait_probe_wait_loops,
            inflight_avg,
            wait_probe_inflight_batches_max,
            wait_probe_inflight_samples,
            kernel_ts_status,
            target_inflight,
            wait_high_watermark,
            spin_polls_before_wait
        );
        eprintln!(
            "[cozip_pdeflate][timing][decode-v2-fallback-reasons] jobs={} capacity={} submit_err={} map_timeout={} map_err={} collect_err={} kernel_fallback={}",
            wait_probe_completed_jobs,
            fallback_capacity_jobs,
            fallback_submit_error_jobs,
            fallback_map_timeout_jobs,
            fallback_map_error_jobs,
            fallback_collect_error_jobs,
            fallback_kernel_error_jobs
        );
    }
    Ok(out)
}

pub(crate) fn max_safe_match_batch_chunks(chunk_size: usize) -> Result<usize, PDeflateError> {
    if chunk_size == 0 {
        return Ok(1);
    }
    let r = runtime()?;
    let per_chunk_out = chunk_size
        .checked_mul(4)
        .ok_or(PDeflateError::NumericOverflow)?;
    if per_chunk_out == 0 {
        return Ok(1);
    }
    let max_binding =
        usize::try_from(r.max_storage_binding_size).map_err(|_| PDeflateError::NumericOverflow)?;
    Ok((max_binding / per_chunk_out).max(1))
}

pub(crate) fn max_storage_binding_size_bytes() -> Result<usize, PDeflateError> {
    let r = runtime()?;
    usize::try_from(r.max_storage_binding_size).map_err(|_| PDeflateError::NumericOverflow)
}

pub(crate) fn upload_packed_table_device(
    table_count: usize,
    table_index: &[u8],
    table_data: &[u8],
) -> Result<GpuPackedTableDevice, PDeflateError> {
    let r = runtime()?;
    let index_bytes = u64::try_from(table_index.len().div_ceil(4))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let data_bytes = u64::try_from(table_data.len().div_ceil(4))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_index_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-upload-table-index"),
        size: index_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let table_data_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-upload-table-data"),
        size: data_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    if !table_index.is_empty() {
        let table_index_words = pack_bytes_to_words(table_index);
        r.queue.write_buffer(
            &table_index_buffer,
            0,
            bytemuck::cast_slice(&table_index_words),
        );
    }
    if !table_data.is_empty() {
        let table_data_words = pack_bytes_to_words(table_data);
        r.queue.write_buffer(
            &table_data_buffer,
            0,
            bytemuck::cast_slice(&table_data_words),
        );
    }
    let table_meta_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-upload-table-meta"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    Ok(GpuPackedTableDevice {
        table_index_buffer: Arc::new(table_index_buffer),
        table_data_buffer: Arc::new(table_data_buffer),
        table_meta_buffer: Arc::new(table_meta_buffer),
        table_index_offset: 0,
        table_data_offset: 0,
        table_meta_offset: 0,
        table_count,
        table_index_len: table_index.len(),
        table_data_len: table_data.len(),
        sizes_known: true,
        max_entries: table_count,
        table_data_bytes_cap: table_data.len(),
        build_id: 0,
        build_submit_seq: 0,
        build_submit_index: None,
        build_breakdown: GpuTableBuildBreakdown::default(),
    })
}

pub(crate) fn build_table_gpu_device(
    chunk: &[u8],
    max_entries: usize,
    max_entry_len: usize,
    min_ref_len: usize,
    match_probe_limit: usize,
    hash_history_limit: usize,
    table_sample_stride: usize,
) -> Result<(GpuPackedTableDevice, GpuBuildTableProfile), PDeflateError> {
    let t_total_build = Instant::now();
    let r = runtime()?;
    if chunk.len() < 3 || max_entries == 0 {
        return Ok((
            GpuPackedTableDevice {
                table_index_buffer: Arc::new(storage_upload_buffer(
                    &r.device,
                    "cozip-pdeflate-bt-empty-table-index",
                    4,
                )),
                table_data_buffer: Arc::new(storage_upload_buffer(
                    &r.device,
                    "cozip-pdeflate-bt-empty-table-data",
                    4,
                )),
                table_meta_buffer: Arc::new(storage_upload_buffer(
                    &r.device,
                    "cozip-pdeflate-bt-empty-table-meta",
                    4,
                )),
                table_index_offset: 0,
                table_data_offset: 0,
                table_meta_offset: 0,
                table_count: 0,
                table_index_len: 0,
                table_data_len: 0,
                sizes_known: true,
                max_entries: 0,
                table_data_bytes_cap: 0,
                build_id: 0,
                build_submit_seq: 0,
                build_submit_index: None,
                build_breakdown: GpuTableBuildBreakdown::default(),
            },
            GpuBuildTableProfile::default(),
        ));
    }
    let build_id = BUILD_TABLE_DEVICE_SEQ
        .fetch_add(1, Ordering::Relaxed)
        .saturating_add(1);
    let capped_max_entry_len = max_entry_len.min(254).max(3);
    let sample_stride = table_sample_stride.max(1).saturating_mul(8);
    let sample_count = (chunk.len() - 2).div_ceil(sample_stride);
    let min_seed_match_len = min_ref_len.max(6).min(capped_max_entry_len);
    let history_limit = hash_history_limit.max(1).min(64);
    let probe_limit = match_probe_limit.max(1).min(8);

    let src_words = pack_bytes_to_words(chunk);
    let src_words_bytes = u64::try_from(src_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let freq_words = 256usize;
    let freq_bytes = u64::try_from(freq_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let cand_words = sample_count
        .checked_mul(4)
        .ok_or(PDeflateError::NumericOverflow)?;
    let cand_bytes = u64::try_from(cand_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let sort_bucket_words = BUILD_TABLE_SORT_BUCKETS;
    let sort_bucket_bytes = u64::try_from(sort_bucket_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let sorted_idx_words = sample_count;
    let sorted_idx_bytes = u64::try_from(sorted_idx_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_meta_words = 2usize
        .checked_add(
            max_entries
                .checked_mul(2)
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .ok_or(PDeflateError::NumericOverflow)?;
    let table_meta_bytes = u64::try_from(table_meta_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_data_bytes_cap = max_entries
        .checked_mul(capped_max_entry_len)
        .ok_or(PDeflateError::NumericOverflow)?;
    let table_data_words = table_data_bytes_cap.div_ceil(4);
    let table_data_bytes = u64::try_from(table_data_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let finalize_emit_desc_words = max_entries
        .checked_mul(3)
        .ok_or(PDeflateError::NumericOverflow)?;
    let finalize_emit_desc_bytes = u64::try_from(finalize_emit_desc_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let freq_params: [u32; 1] =
        [u32::try_from(chunk.len()).map_err(|_| PDeflateError::NumericOverflow)?];
    let freq_params_bytes = u64::try_from(freq_params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let cand_params: [u32; 7] = [
        u32::try_from(chunk.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(sample_stride).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(capped_max_entry_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(min_seed_match_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(history_limit).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(probe_limit).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(sample_count).map_err(|_| PDeflateError::NumericOverflow)?,
    ];
    let cand_params_bytes = u64::try_from(cand_params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let bucket_params: [u32; 1] =
        [u32::try_from(sample_count).map_err(|_| PDeflateError::NumericOverflow)?];
    let bucket_params_bytes = u64::try_from(bucket_params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let finalize_params: [u32; 8] = [
        u32::try_from(chunk.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(sample_count).map_err(|_| PDeflateError::NumericOverflow)?,
        0,
        u32::try_from(capped_max_entry_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(max_entries).map_err(|_| PDeflateError::NumericOverflow)?,
        64,
        8,
        u32::try_from(table_data_bytes_cap).map_err(|_| PDeflateError::NumericOverflow)?,
    ];
    let finalize_params_bytes = u64::try_from(finalize_params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let stage_probe_enabled = table_stage_probe_enabled() && r.supports_timestamp_query;
    if table_stage_probe_enabled() && !r.supports_timestamp_query {
        let _ = BUILD_TABLE_STAGE_PROBE_UNSUPPORTED_LOGGED.get_or_init(|| {
            eprintln!(
                "[cozip_pdeflate][timing][gpu-table-stage-probe] status=disabled reason=timestamp_query_not_supported"
            );
        });
    }
    let mut stage_gpu_ms = [0.0_f64; BUILD_TABLE_STAGE_COUNT];
    let mut stage_probe_error: Option<String> = None;
    let mut stage_probe_submit_ms = 0.0_f64;
    let mut stage_probe_wait_ms = 0.0_f64;
    let mut stage_probe_parse_ms = 0.0_f64;
    let mut stage_probe_query_set = None;
    let mut stage_probe_resolve_buffer = None;
    let mut stage_probe_readback_buffer = None;
    if stage_probe_enabled {
        let ts_bytes = u64::from(BUILD_TABLE_STAGE_QUERY_COUNT).saturating_mul(8);
        stage_probe_query_set = Some(r.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("cozip-pdeflate-bt-stage-ts-qs"),
            ty: wgpu::QueryType::Timestamp,
            count: BUILD_TABLE_STAGE_QUERY_COUNT,
        }));
        stage_probe_resolve_buffer = Some(r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-bt-stage-ts-resolve"),
            size: ts_bytes.max(8),
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        stage_probe_readback_buffer = Some(r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-bt-stage-ts-readback"),
            size: ts_bytes.max(8),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }

    let t_resource_setup = Instant::now();
    let pre_resource_ms = elapsed_ms(t_total_build);
    let src_buffer = storage_upload_buffer(&r.device, "cozip-pdeflate-bt-src", src_words_bytes);
    let freq_params_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-bt-freq-params",
        freq_params_bytes,
    );
    let cand_params_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-bt-cand-params",
        cand_params_bytes,
    );
    let bucket_params_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-bt-bucket-params",
        bucket_params_bytes,
    );
    let finalize_params_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-bt-finalize-params",
        finalize_params_bytes,
    );
    let freq_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-freq"),
        size: freq_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cand_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-cand"),
        size: cand_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bucket_count_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-count"),
        size: sort_bucket_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bucket_offset_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-offset"),
        size: sort_bucket_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bucket_cursor_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-cursor"),
        size: sort_bucket_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let sorted_idx_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-sorted-idx"),
        size: sorted_idx_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let table_meta_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-table-meta"),
        size: table_meta_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let table_index_words = max_entries.div_ceil(4);
    let table_index_bytes = u64::try_from(table_index_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_index_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-table-index"),
        size: table_index_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let table_data_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-table-data"),
        size: table_data_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let finalize_emit_desc_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-finalize-emit-desc"),
        size: finalize_emit_desc_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let finalize_emit_word_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-finalize-emit-word"),
        size: table_data_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let freq_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-freq-bg"),
        layout: &r.build_table_freq_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: freq_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: freq_buffer.as_entire_binding(),
            },
        ],
    });
    let cand_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-cand-bg"),
        layout: &r.build_table_candidate_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cand_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cand_buffer.as_entire_binding(),
            },
        ],
    });
    let bucket_count_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-count-bg"),
        layout: &r.build_table_bucket_count_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cand_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bucket_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bucket_count_buffer.as_entire_binding(),
            },
        ],
    });
    let bucket_prefix_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-prefix-bg"),
        layout: &r.build_table_bucket_prefix_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: bucket_count_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bucket_offset_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: finalize_params_buffer.as_entire_binding(),
            },
        ],
    });
    let bucket_scatter_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-scatter-bg"),
        layout: &r.build_table_bucket_scatter_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cand_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bucket_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bucket_offset_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bucket_cursor_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: sorted_idx_buffer.as_entire_binding(),
            },
        ],
    });
    let finalize_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-finalize-bg"),
        layout: &r.build_table_finalize_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: freq_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cand_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sorted_idx_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: finalize_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: table_meta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: finalize_emit_desc_buffer.as_entire_binding(),
            },
        ],
    });
    let finalize_emit_desc_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-finalize-emit-desc-bg"),
        layout: &r.build_table_finalize_emit_desc_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: table_meta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: finalize_emit_desc_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: finalize_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: finalize_emit_word_buffer.as_entire_binding(),
            },
        ],
    });
    let finalize_emit_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-finalize-emit-bg"),
        layout: &r.build_table_finalize_emit_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: finalize_emit_word_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: table_meta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: finalize_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: table_data_buffer.as_entire_binding(),
            },
        ],
    });
    let pack_index_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-pack-index-bg"),
        layout: &r.build_table_pack_index_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: table_meta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: finalize_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: table_index_buffer.as_entire_binding(),
            },
        ],
    });
    let resource_setup_ms = elapsed_ms(t_resource_setup);

    let t_upload = Instant::now();
    r.queue
        .write_buffer(&src_buffer, 0, bytemuck::cast_slice(&src_words));
    r.queue
        .write_buffer(&freq_params_buffer, 0, bytemuck::cast_slice(&freq_params));
    r.queue
        .write_buffer(&cand_params_buffer, 0, bytemuck::cast_slice(&cand_params));
    r.queue.write_buffer(
        &bucket_params_buffer,
        0,
        bytemuck::cast_slice(&bucket_params),
    );
    r.queue.write_buffer(
        &finalize_params_buffer,
        0,
        bytemuck::cast_slice(&finalize_params),
    );
    let upload_ms = elapsed_ms(t_upload);

    let t_encode_pass1 = Instant::now();
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-bt-encoder"),
        });
    encoder.clear_buffer(&freq_buffer, 0, None);
    encoder.clear_buffer(&cand_buffer, 0, None);

    let freq_groups = u32::try_from(chunk.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .div_ceil(256)
        .max(1);
    let t_freq = Instant::now();
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 0);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-freq-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_freq_pipeline);
        pass.set_bind_group(0, &freq_bind_group, &[]);
        pass.dispatch_workgroups(freq_groups, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 1);
    }
    let freq_kernel_ms = elapsed_ms(t_freq);

    let cand_groups = u32::try_from(sample_count)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .div_ceil(128)
        .max(1);
    let t_cand = Instant::now();
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 2);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-candidate-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_candidate_pipeline);
        pass.set_bind_group(0, &cand_bind_group, &[]);
        pass.dispatch_workgroups(cand_groups, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 3);
    }
    let mut candidate_kernel_ms = elapsed_ms(t_cand);

    encoder.clear_buffer(&bucket_count_buffer, 0, None);
    let t_bucket_count = Instant::now();
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 4);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-bucket-count-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_bucket_count_pipeline);
        pass.set_bind_group(0, &bucket_count_bind_group, &[]);
        pass.dispatch_workgroups(cand_groups, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 5);
    }
    candidate_kernel_ms += elapsed_ms(t_bucket_count);
    let encode_pass1_ms = elapsed_ms(t_encode_pass1);
    let t_submit_pass1 = Instant::now();
    let pass1_submit_seq = next_gpu_submit_seq();
    let _pass1_submission = r.queue.submit(Some(encoder.finish()));
    let submit_pass1_ms = elapsed_ms(t_submit_pass1);

    let readback_ms = 0.0;

    let t_encode_pass2 = Instant::now();
    let mut encoder2 = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-bt-finalize-encoder"),
        });
    encoder2.clear_buffer(&bucket_cursor_buffer, 0, None);
    encoder2.clear_buffer(&table_meta_buffer, 0, None);
    encoder2.clear_buffer(&table_data_buffer, 0, None);
    encoder2.clear_buffer(&table_index_buffer, 0, None);
    encoder2.clear_buffer(&finalize_emit_desc_buffer, 0, None);
    encoder2.clear_buffer(&finalize_emit_word_buffer, 0, None);
    let t_bucket_prefix = Instant::now();
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 6);
    }
    {
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-bucket-prefix-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_bucket_prefix_pipeline);
        pass.set_bind_group(0, &bucket_prefix_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 7);
    }
    candidate_kernel_ms += elapsed_ms(t_bucket_prefix);
    let t_bucket_scatter = Instant::now();
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 8);
    }
    {
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-bucket-scatter-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_bucket_scatter_pipeline);
        pass.set_bind_group(0, &bucket_scatter_bind_group, &[]);
        pass.dispatch_workgroups(cand_groups, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 9);
    }
    candidate_kernel_ms += elapsed_ms(t_bucket_scatter);
    let t_finalize_kernel = Instant::now();
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 10);
    }
    {
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-finalize-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_finalize_pipeline);
        pass.set_bind_group(0, &finalize_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    {
        let emit_groups = u32::try_from(table_data_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .div_ceil(64)
            .max(1);
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-finalize-emit-desc-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_finalize_emit_desc_pipeline);
        pass.set_bind_group(0, &finalize_emit_desc_bind_group, &[]);
        pass.dispatch_workgroups(emit_groups, 1, 1);
    }
    {
        let emit_groups = u32::try_from(table_data_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .div_ceil(256)
            .max(1);
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-finalize-emit-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_finalize_emit_pipeline);
        pass.set_bind_group(0, &finalize_emit_bind_group, &[]);
        pass.dispatch_workgroups(emit_groups, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 11);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 12);
    }
    {
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-pack-index-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_pack_index_pipeline);
        pass.set_bind_group(0, &pack_index_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 13);
    }
    let encode_pass2_ms = elapsed_ms(t_encode_pass2);
    let sort_ms = elapsed_ms(t_finalize_kernel);
    let t_submit_pass2 = Instant::now();
    let pass2_submit_seq = next_gpu_submit_seq();
    let pass2_submission = r.queue.submit(Some(encoder2.finish()));
    let submit_pass2_ms = elapsed_ms(t_submit_pass2);
    if let (Some(query_set), Some(resolve_buffer), Some(readback_buffer)) = (
        stage_probe_query_set.take(),
        stage_probe_resolve_buffer.take(),
        stage_probe_readback_buffer.take(),
    ) {
        let ts_bytes = u64::from(BUILD_TABLE_STAGE_QUERY_COUNT).saturating_mul(8);
        let mut ts_encoder = r
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-pdeflate-bt-stage-ts-resolve-encoder"),
            });
        ts_encoder.resolve_query_set(
            &query_set,
            0..BUILD_TABLE_STAGE_QUERY_COUNT,
            &resolve_buffer,
            0,
        );
        ts_encoder.copy_buffer_to_buffer(&resolve_buffer, 0, &readback_buffer, 0, ts_bytes.max(8));
        let t_probe_submit = Instant::now();
        let _stage_probe_submit_seq = next_gpu_submit_seq();
        let submission = r.queue.submit(Some(ts_encoder.finish()));
        stage_probe_submit_ms = elapsed_ms(t_probe_submit);
        let t_probe_readback = Instant::now();
        let t_probe_wait = Instant::now();
        let slice = readback_buffer.slice(..ts_bytes.max(8));
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        r.device.poll(wgpu::Maintain::wait_for(submission));
        stage_probe_wait_ms = elapsed_ms(t_probe_wait);
        match rx.recv() {
            Ok(Ok(())) => {
                let t_probe_parse = Instant::now();
                let mapped = slice.get_mapped_range();
                let ticks: &[u64] = bytemuck::cast_slice(&mapped);
                let period_ms = (r.queue.get_timestamp_period() as f64) / 1_000_000.0;
                for stage_idx in 0..BUILD_TABLE_STAGE_COUNT {
                    let begin_idx = stage_idx.saturating_mul(2);
                    let end_idx = begin_idx.saturating_add(1);
                    if end_idx < ticks.len() {
                        let begin = ticks[begin_idx];
                        let end = ticks[end_idx];
                        if end >= begin {
                            stage_gpu_ms[stage_idx] = (end - begin) as f64 * period_ms;
                        }
                    }
                }
                drop(mapped);
                readback_buffer.unmap();
                stage_probe_parse_ms = elapsed_ms(t_probe_parse);
            }
            Ok(Err(e)) => {
                stage_probe_error = Some(format!("map failed: {e}"));
            }
            Err(_) => {
                stage_probe_error = Some("map channel closed".to_string());
            }
        }
        let stage_probe_readback_ms = elapsed_ms(t_probe_readback);
        let seq = BUILD_TABLE_STAGE_PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
        if table_stage_probe_should_log(seq) {
            if let Some(err) = stage_probe_error.as_ref() {
                eprintln!(
                    "[cozip_pdeflate][timing][gpu-table-stage-probe] seq={} sample_count={} map_status=error err=\"{}\"",
                    seq, sample_count, err
                );
            } else {
                let total_stage_ms: f64 = stage_gpu_ms.iter().sum();
                eprintln!(
                    "[cozip_pdeflate][timing][gpu-table-stage-probe] seq={} sample_count={} freq_ms={:.3} candidate_ms={:.3} bucket_count_ms={:.3} bucket_prefix_ms={:.3} bucket_scatter_ms={:.3} finalize_ms={:.3} pack_index_ms={:.3} stage_sum_ms={:.3} ts_readback_ms={:.3}",
                    seq,
                    sample_count,
                    stage_gpu_ms[0],
                    stage_gpu_ms[1],
                    stage_gpu_ms[2],
                    stage_gpu_ms[3],
                    stage_gpu_ms[4],
                    stage_gpu_ms[5],
                    stage_gpu_ms[6],
                    total_stage_ms,
                    stage_probe_readback_ms
                );
            }
        }
    }
    let stage_sum_ms: f64 = stage_gpu_ms.iter().sum();
    let total_build_ms = elapsed_ms(t_total_build);
    let known_ms = pre_resource_ms
        + resource_setup_ms
        + upload_ms
        + encode_pass1_ms
        + submit_pass1_ms
        + encode_pass2_ms
        + submit_pass2_ms
        + stage_probe_submit_ms
        + stage_probe_wait_ms
        + stage_probe_parse_ms;
    let tail_ms = (total_build_ms - known_ms).max(0.0);
    let other_ms = pre_resource_ms + tail_ms;
    if table_build_breakdown_enabled() {
        let seq = BUILD_TABLE_BREAKDOWN_SEQ.fetch_add(1, Ordering::Relaxed);
        if table_stage_probe_should_log(seq) {
            eprintln!(
                "[cozip_pdeflate][timing][gpu-table-build-breakdown] seq={} build_id={} sample_count={} pass1_submit_seq={} pass2_submit_seq={} total_build_ms={:.3} pre_resource_ms={:.3} resource_setup_ms={:.3} upload_ms={:.3} encode_pass1_ms={:.3} submit_pass1_ms={:.3} encode_pass2_ms={:.3} submit_pass2_ms={:.3} stage_probe_submit_ms={:.3} stage_probe_wait_ms={:.3} stage_probe_parse_ms={:.3} stage_gpu_sum_ms={:.3} tail_ms={:.3} other_ms={:.3}",
                seq,
                build_id,
                sample_count,
                pass1_submit_seq,
                pass2_submit_seq,
                total_build_ms,
                pre_resource_ms,
                resource_setup_ms,
                upload_ms,
                encode_pass1_ms,
                submit_pass1_ms,
                encode_pass2_ms,
                submit_pass2_ms,
                stage_probe_submit_ms,
                stage_probe_wait_ms,
                stage_probe_parse_ms,
                stage_sum_ms,
                tail_ms,
                other_ms
            );
        }
    }
    Ok((
        GpuPackedTableDevice {
            table_index_buffer: Arc::new(table_index_buffer),
            table_data_buffer: Arc::new(table_data_buffer),
            table_meta_buffer: Arc::new(table_meta_buffer),
            table_index_offset: 0,
            table_data_offset: 0,
            table_meta_offset: 0,
            table_count: 0,
            table_index_len: 0,
            table_data_len: 0,
            sizes_known: false,
            max_entries,
            table_data_bytes_cap,
            build_id,
            build_submit_seq: pass2_submit_seq,
            build_submit_index: Some(pass2_submission),
            build_breakdown: GpuTableBuildBreakdown {
                total_ms: total_build_ms,
                pre_resource_ms,
                resource_setup_ms,
                upload_ms,
                encode_pass1_ms,
                submit_pass1_ms,
                encode_pass2_ms,
                submit_pass2_ms,
                stage_probe_submit_ms,
                stage_probe_wait_ms,
                stage_probe_parse_ms,
                stage_gpu_sum_ms: stage_sum_ms,
                tail_ms,
                other_ms,
            },
        },
        GpuBuildTableProfile {
            upload_ms,
            freq_kernel_ms,
            candidate_kernel_ms,
            readback_ms,
            materialize_ms: 0.0,
            sort_ms,
            sample_count,
        },
    ))
}

pub(crate) fn build_table_gpu_device_batch(
    chunks: &[&[u8]],
    max_entries: usize,
    max_entry_len: usize,
    min_ref_len: usize,
    match_probe_limit: usize,
    hash_history_limit: usize,
    table_sample_stride: usize,
) -> Result<Vec<(GpuPackedTableDevice, GpuBuildTableProfile)>, PDeflateError> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    if chunks.len() == 1 {
        return build_table_gpu_device(
            chunks[0],
            max_entries,
            max_entry_len,
            min_ref_len,
            match_probe_limit,
            hash_history_limit,
            table_sample_stride,
        )
        .map(|one| vec![one]);
    }

    struct BatchChunkDesc {
        chunk_len: usize,
        sample_count: usize,
        src_word_offset: usize,
        freq_word_offset: usize,
        cand_word_offset: usize,
        bucket_word_offset: usize,
        sorted_idx_word_offset: usize,
        meta_word_offset: usize,
        data_word_offset: usize,
        index_word_offset: usize,
        emit_desc_word_offset: usize,
        freq_param_word_offset: usize,
        cand_param_word_offset: usize,
        bucket_param_word_offset: usize,
        finalize_param_word_offset: usize,
        cand_groups: u32,
        emit_groups: u32,
        max_entry_len: usize,
        table_data_bytes_cap: usize,
        build_id: u64,
    }

    fn binding_resource<'a>(
        buffer: &'a wgpu::Buffer,
        offset: u64,
        size: u64,
    ) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset,
            size: Some(std::num::NonZeroU64::new(size.max(4)).unwrap()),
        })
    }

    let t_total_build = Instant::now();
    let r = runtime()?;
    let capped_max_entry_len = max_entry_len.min(254).max(3);
    let sample_stride = table_sample_stride.max(1).saturating_mul(8);
    let min_seed_match_len = min_ref_len.max(6).min(capped_max_entry_len);
    let history_limit = hash_history_limit.max(1).min(64);
    let probe_limit = match_probe_limit.max(1).min(8);
    let align_words32 = |v: usize| -> usize { (v + 7) & !7 };
    let build_id_base = BUILD_TABLE_DEVICE_SEQ
        .fetch_add(
            u64::try_from(chunks.len()).map_err(|_| PDeflateError::NumericOverflow)?,
            Ordering::Relaxed,
        )
        .saturating_add(1);

    let t_pre_resource = Instant::now();
    let mut src_bytes = Vec::<u8>::new();
    let mut descs = Vec::<BatchChunkDesc>::with_capacity(chunks.len());
    let mut total_src_word_offset = 0usize;
    let mut total_freq_words = 0usize;
    let mut total_cand_words = 0usize;
    let mut total_bucket_words = 0usize;
    let mut total_sorted_idx_words = 0usize;
    let mut total_meta_words = 0usize;
    let mut total_data_words = 0usize;
    let mut total_index_words = 0usize;
    let mut total_emit_desc_words = 0usize;
    let mut total_freq_param_words = 0usize;
    let mut total_cand_param_words = 0usize;
    let mut total_bucket_param_words = 0usize;
    let mut total_finalize_param_words = 0usize;
    for (idx, chunk) in chunks.iter().copied().enumerate() {
        if chunk.len() < 3 || max_entries == 0 {
            return Err(PDeflateError::InvalidOptions(
                "gpu table build batch requires chunks >= 3 bytes and max_entries > 0",
            ));
        }
        let sample_count = (chunk.len() - 2).div_ceil(sample_stride);
        let chunk_src_word_len = chunk.len().div_ceil(4);
        total_src_word_offset = align_words32(total_src_word_offset);
        let src_byte_offset = total_src_word_offset
            .checked_mul(4)
            .ok_or(PDeflateError::NumericOverflow)?;
        let chunk_src_padded_bytes = chunk_src_word_len
            .checked_mul(4)
            .ok_or(PDeflateError::NumericOverflow)?;
        if src_bytes.len() < src_byte_offset {
            src_bytes.resize(src_byte_offset, 0);
        }
        let src_byte_end = src_byte_offset
            .checked_add(chunk_src_padded_bytes)
            .ok_or(PDeflateError::NumericOverflow)?;
        if src_bytes.len() < src_byte_end {
            src_bytes.resize(src_byte_end, 0);
        }
        src_bytes[src_byte_offset..src_byte_offset + chunk.len()].copy_from_slice(chunk);
        total_freq_words = align_words32(total_freq_words);
        total_cand_words = align_words32(total_cand_words);
        total_bucket_words = align_words32(total_bucket_words);
        total_sorted_idx_words = align_words32(total_sorted_idx_words);
        total_meta_words = align_words32(total_meta_words);
        total_data_words = align_words32(total_data_words);
        total_index_words = align_words32(total_index_words);
        total_emit_desc_words = align_words32(total_emit_desc_words);
        total_freq_param_words = align_words32(total_freq_param_words);
        total_cand_param_words = align_words32(total_cand_param_words);
        total_bucket_param_words = align_words32(total_bucket_param_words);
        total_finalize_param_words = align_words32(total_finalize_param_words);
        let cand_words = sample_count
            .checked_mul(4)
            .ok_or(PDeflateError::NumericOverflow)?;
        let table_data_bytes_cap = max_entries
            .checked_mul(capped_max_entry_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        let table_data_words = table_data_bytes_cap.div_ceil(4);
        let table_meta_words = 2usize
            .checked_add(
                max_entries
                    .checked_mul(2)
                    .ok_or(PDeflateError::NumericOverflow)?,
            )
            .ok_or(PDeflateError::NumericOverflow)?;
        let emit_desc_words = max_entries
            .checked_mul(3)
            .ok_or(PDeflateError::NumericOverflow)?;
        let cand_groups = u32::try_from(sample_count)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .div_ceil(128)
            .max(1);
        let emit_groups = u32::try_from(table_data_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .div_ceil(64)
            .max(1);
        descs.push(BatchChunkDesc {
            chunk_len: chunk.len(),
            sample_count,
            src_word_offset: total_src_word_offset,
            freq_word_offset: total_freq_words,
            cand_word_offset: total_cand_words,
            bucket_word_offset: total_bucket_words,
            sorted_idx_word_offset: total_sorted_idx_words,
            meta_word_offset: total_meta_words,
            data_word_offset: total_data_words,
            index_word_offset: total_index_words,
            emit_desc_word_offset: total_emit_desc_words,
            freq_param_word_offset: total_freq_param_words,
            cand_param_word_offset: total_cand_param_words,
            bucket_param_word_offset: total_bucket_param_words,
            finalize_param_word_offset: total_finalize_param_words,
            cand_groups,
            emit_groups,
            max_entry_len: capped_max_entry_len,
            table_data_bytes_cap,
            build_id: build_id_base
                .saturating_add(u64::try_from(idx).map_err(|_| PDeflateError::NumericOverflow)?),
        });
        total_src_word_offset = total_src_word_offset
            .checked_add(chunk_src_word_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        total_freq_words = total_freq_words
            .checked_add(256)
            .ok_or(PDeflateError::NumericOverflow)?;
        total_cand_words = total_cand_words
            .checked_add(cand_words)
            .ok_or(PDeflateError::NumericOverflow)?;
        total_bucket_words = total_bucket_words
            .checked_add(BUILD_TABLE_SORT_BUCKETS)
            .ok_or(PDeflateError::NumericOverflow)?;
        total_sorted_idx_words = total_sorted_idx_words
            .checked_add(sample_count)
            .ok_or(PDeflateError::NumericOverflow)?;
        total_meta_words = total_meta_words
            .checked_add(table_meta_words)
            .ok_or(PDeflateError::NumericOverflow)?;
        total_data_words = total_data_words
            .checked_add(table_data_words)
            .ok_or(PDeflateError::NumericOverflow)?;
        total_index_words = total_index_words
            .checked_add(max_entries.div_ceil(4))
            .ok_or(PDeflateError::NumericOverflow)?;
        total_emit_desc_words = total_emit_desc_words
            .checked_add(emit_desc_words)
            .ok_or(PDeflateError::NumericOverflow)?;
        total_freq_param_words = total_freq_param_words
            .checked_add(1)
            .ok_or(PDeflateError::NumericOverflow)?;
        total_cand_param_words = total_cand_param_words
            .checked_add(7)
            .ok_or(PDeflateError::NumericOverflow)?;
        total_bucket_param_words = total_bucket_param_words
            .checked_add(1)
            .ok_or(PDeflateError::NumericOverflow)?;
        total_finalize_param_words = total_finalize_param_words
            .checked_add(8)
            .ok_or(PDeflateError::NumericOverflow)?;
    }
    let pre_resource_ms = elapsed_ms(t_pre_resource);

    let t_resource_setup = Instant::now();
    let scratch_caps = GpuTableBuildBatchScratchCaps {
        src_bytes: u64::try_from(src_bytes.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        freq_params_bytes: u64::try_from(total_freq_param_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        cand_params_bytes: u64::try_from(total_cand_param_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        bucket_params_bytes: u64::try_from(total_bucket_param_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        finalize_params_bytes: u64::try_from(total_finalize_param_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        freq_bytes: u64::try_from(total_freq_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        cand_bytes: u64::try_from(total_cand_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        bucket_bytes: u64::try_from(total_bucket_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        sorted_idx_bytes: u64::try_from(total_sorted_idx_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        emit_desc_bytes: u64::try_from(total_emit_desc_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        emit_word_bytes: u64::try_from(total_data_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
    };
    let scratch = acquire_table_build_batch_scratch(r, scratch_caps)?;
    let table_meta_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-batch-table-meta"),
        size: u64::try_from(total_meta_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4)
            .max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let table_data_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-batch-table-data"),
        size: u64::try_from(total_data_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4)
            .max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let table_index_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-batch-table-index"),
        size: u64::try_from(total_index_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4)
            .max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let resource_setup_ms = elapsed_ms(t_resource_setup);

    let mut freq_params_words = vec![0u32; total_freq_param_words];
    let mut cand_params_words = vec![0u32; total_cand_param_words];
    let mut bucket_params_words = vec![0u32; total_bucket_param_words];
    let mut finalize_params_words = vec![0u32; total_finalize_param_words];
    for desc in &descs {
        freq_params_words[desc.freq_param_word_offset] =
            u32::try_from(desc.chunk_len).map_err(|_| PDeflateError::NumericOverflow)?;
        let cand_base = desc.cand_param_word_offset;
        cand_params_words[cand_base + 0] =
            u32::try_from(desc.chunk_len).map_err(|_| PDeflateError::NumericOverflow)?;
        cand_params_words[cand_base + 1] =
            u32::try_from(sample_stride).map_err(|_| PDeflateError::NumericOverflow)?;
        cand_params_words[cand_base + 2] =
            u32::try_from(desc.max_entry_len).map_err(|_| PDeflateError::NumericOverflow)?;
        cand_params_words[cand_base + 3] =
            u32::try_from(min_seed_match_len).map_err(|_| PDeflateError::NumericOverflow)?;
        cand_params_words[cand_base + 4] =
            u32::try_from(history_limit).map_err(|_| PDeflateError::NumericOverflow)?;
        cand_params_words[cand_base + 5] =
            u32::try_from(probe_limit).map_err(|_| PDeflateError::NumericOverflow)?;
        cand_params_words[cand_base + 6] =
            u32::try_from(desc.sample_count).map_err(|_| PDeflateError::NumericOverflow)?;
        bucket_params_words[desc.bucket_param_word_offset] =
            u32::try_from(desc.sample_count).map_err(|_| PDeflateError::NumericOverflow)?;
        let finalize_base = desc.finalize_param_word_offset;
        finalize_params_words[finalize_base + 0] =
            u32::try_from(desc.chunk_len).map_err(|_| PDeflateError::NumericOverflow)?;
        finalize_params_words[finalize_base + 1] =
            u32::try_from(desc.sample_count).map_err(|_| PDeflateError::NumericOverflow)?;
        finalize_params_words[finalize_base + 2] = 0;
        finalize_params_words[finalize_base + 3] =
            u32::try_from(desc.max_entry_len).map_err(|_| PDeflateError::NumericOverflow)?;
        finalize_params_words[finalize_base + 4] =
            u32::try_from(max_entries).map_err(|_| PDeflateError::NumericOverflow)?;
        finalize_params_words[finalize_base + 5] = 64;
        finalize_params_words[finalize_base + 6] = 8;
        finalize_params_words[finalize_base + 7] =
            u32::try_from(desc.table_data_bytes_cap).map_err(|_| PDeflateError::NumericOverflow)?;
    }

    let t_upload = Instant::now();
    r.queue.write_buffer(&scratch.src_buffer, 0, &src_bytes);
    r.queue.write_buffer(
        &scratch.freq_params_buffer,
        0,
        bytemuck::cast_slice(&freq_params_words),
    );
    r.queue.write_buffer(
        &scratch.cand_params_buffer,
        0,
        bytemuck::cast_slice(&cand_params_words),
    );
    r.queue.write_buffer(
        &scratch.bucket_params_buffer,
        0,
        bytemuck::cast_slice(&bucket_params_words),
    );
    r.queue.write_buffer(
        &scratch.finalize_params_buffer,
        0,
        bytemuck::cast_slice(&finalize_params_words),
    );
    let upload_ms = elapsed_ms(t_upload);

    struct BatchBindGroups {
        freq: wgpu::BindGroup,
        candidate: wgpu::BindGroup,
        bucket_count: wgpu::BindGroup,
        bucket_prefix: wgpu::BindGroup,
        bucket_scatter: wgpu::BindGroup,
        finalize: wgpu::BindGroup,
        finalize_emit_desc: wgpu::BindGroup,
        finalize_emit: wgpu::BindGroup,
        pack_index: wgpu::BindGroup,
        cand_groups: u32,
        emit_groups: u32,
    }

    let t_encode = Instant::now();
    let mut groups = Vec::<BatchBindGroups>::with_capacity(descs.len());
    for desc in &descs {
        let src_offset = u64::try_from(desc.src_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let src_size = u64::try_from(desc.chunk_len.div_ceil(4))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let freq_size = 256_u64.saturating_mul(4);
        let freq_offset = u64::try_from(desc.freq_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let cand_offset = u64::try_from(desc.cand_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let cand_size = u64::try_from(
            desc.sample_count
                .checked_mul(4)
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
        let bucket_offset = u64::try_from(desc.bucket_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let bucket_size = u64::try_from(BUILD_TABLE_SORT_BUCKETS)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let sorted_idx_offset = u64::try_from(desc.sorted_idx_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let sorted_idx_size = u64::try_from(desc.sample_count)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let meta_offset = u64::try_from(desc.meta_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let meta_size = u64::try_from(
            2usize
                .checked_add(
                    max_entries
                        .checked_mul(2)
                        .ok_or(PDeflateError::NumericOverflow)?,
                )
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
        let data_offset = u64::try_from(desc.data_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let data_size = u64::try_from(desc.table_data_bytes_cap.div_ceil(4))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let index_offset = u64::try_from(desc.index_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let index_size = u64::try_from(max_entries.div_ceil(4))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let emit_desc_offset = u64::try_from(desc.emit_desc_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let emit_desc_size = u64::try_from(
            max_entries
                .checked_mul(3)
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
        let freq_param_offset = u64::try_from(desc.freq_param_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let cand_param_offset = u64::try_from(desc.cand_param_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let bucket_param_offset = u64::try_from(desc.bucket_param_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        let finalize_param_offset = u64::try_from(desc.finalize_param_word_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);

        let freq = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-bt-batch-freq-bg"),
            layout: &r.build_table_freq_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding_resource(&scratch.src_buffer, src_offset, src_size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: binding_resource(&scratch.freq_params_buffer, freq_param_offset, 4),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: binding_resource(&scratch.freq_buffer, freq_offset, freq_size),
                },
            ],
        });
        let candidate = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-bt-batch-cand-bg"),
            layout: &r.build_table_candidate_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding_resource(&scratch.src_buffer, src_offset, src_size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: binding_resource(&scratch.cand_params_buffer, cand_param_offset, 28),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: binding_resource(&scratch.cand_buffer, cand_offset, cand_size),
                },
            ],
        });
        let bucket_count = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-bt-batch-bucket-count-bg"),
            layout: &r.build_table_bucket_count_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding_resource(&scratch.cand_buffer, cand_offset, cand_size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: binding_resource(
                        &scratch.bucket_params_buffer,
                        bucket_param_offset,
                        4,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: binding_resource(
                        &scratch.bucket_count_buffer,
                        bucket_offset,
                        bucket_size,
                    ),
                },
            ],
        });
        let bucket_prefix = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-bt-batch-bucket-prefix-bg"),
            layout: &r.build_table_bucket_prefix_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding_resource(
                        &scratch.bucket_count_buffer,
                        bucket_offset,
                        bucket_size,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: binding_resource(
                        &scratch.bucket_offset_buffer,
                        bucket_offset,
                        bucket_size,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: binding_resource(
                        &scratch.finalize_params_buffer,
                        finalize_param_offset,
                        32,
                    ),
                },
            ],
        });
        let bucket_scatter = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-bt-batch-bucket-scatter-bg"),
            layout: &r.build_table_bucket_scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding_resource(&scratch.cand_buffer, cand_offset, cand_size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: binding_resource(
                        &scratch.bucket_params_buffer,
                        bucket_param_offset,
                        4,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: binding_resource(
                        &scratch.bucket_offset_buffer,
                        bucket_offset,
                        bucket_size,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: binding_resource(
                        &scratch.bucket_cursor_buffer,
                        bucket_offset,
                        bucket_size,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: binding_resource(
                        &scratch.sorted_idx_buffer,
                        sorted_idx_offset,
                        sorted_idx_size,
                    ),
                },
            ],
        });
        let finalize = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-bt-batch-finalize-bg"),
            layout: &r.build_table_finalize_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding_resource(&scratch.src_buffer, src_offset, src_size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: binding_resource(&scratch.freq_buffer, freq_offset, freq_size),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: binding_resource(&scratch.cand_buffer, cand_offset, cand_size),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: binding_resource(
                        &scratch.sorted_idx_buffer,
                        sorted_idx_offset,
                        sorted_idx_size,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: binding_resource(
                        &scratch.finalize_params_buffer,
                        finalize_param_offset,
                        32,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: binding_resource(&table_meta_buffer, meta_offset, meta_size),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: binding_resource(
                        &scratch.emit_desc_buffer,
                        emit_desc_offset,
                        emit_desc_size,
                    ),
                },
            ],
        });
        let finalize_emit_desc = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-bt-batch-finalize-emit-desc-bg"),
            layout: &r.build_table_finalize_emit_desc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding_resource(&scratch.src_buffer, src_offset, src_size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: binding_resource(&table_meta_buffer, meta_offset, meta_size),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: binding_resource(
                        &scratch.emit_desc_buffer,
                        emit_desc_offset,
                        emit_desc_size,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: binding_resource(
                        &scratch.finalize_params_buffer,
                        finalize_param_offset,
                        32,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: binding_resource(&scratch.emit_word_buffer, data_offset, data_size),
                },
            ],
        });
        let finalize_emit = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-bt-batch-finalize-emit-bg"),
            layout: &r.build_table_finalize_emit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding_resource(&scratch.emit_word_buffer, data_offset, data_size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: binding_resource(&table_meta_buffer, meta_offset, meta_size),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: binding_resource(
                        &scratch.finalize_params_buffer,
                        finalize_param_offset,
                        32,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: binding_resource(&table_data_buffer, data_offset, data_size),
                },
            ],
        });
        let pack_index = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-bt-batch-pack-index-bg"),
            layout: &r.build_table_pack_index_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding_resource(&table_meta_buffer, meta_offset, meta_size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: binding_resource(
                        &scratch.finalize_params_buffer,
                        finalize_param_offset,
                        32,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: binding_resource(&table_index_buffer, index_offset, index_size),
                },
            ],
        });
        groups.push(BatchBindGroups {
            freq,
            candidate,
            bucket_count,
            bucket_prefix,
            bucket_scatter,
            finalize,
            finalize_emit_desc,
            finalize_emit,
            pack_index,
            cand_groups: desc.cand_groups,
            emit_groups: desc.emit_groups,
        });
    }
    let encode_setup_ms = elapsed_ms(t_encode);

    let t_encode_pass1 = Instant::now();
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-bt-batch-encoder"),
        });
    encoder.clear_buffer(&scratch.freq_buffer, 0, None);
    encoder.clear_buffer(&scratch.cand_buffer, 0, None);
    encoder.clear_buffer(&scratch.bucket_count_buffer, 0, None);
    encoder.clear_buffer(&scratch.bucket_offset_buffer, 0, None);
    encoder.clear_buffer(&scratch.bucket_cursor_buffer, 0, None);
    encoder.clear_buffer(&scratch.sorted_idx_buffer, 0, None);
    encoder.clear_buffer(&table_meta_buffer, 0, None);
    encoder.clear_buffer(&table_data_buffer, 0, None);
    encoder.clear_buffer(&table_index_buffer, 0, None);
    encoder.clear_buffer(&scratch.emit_desc_buffer, 0, None);
    encoder.clear_buffer(&scratch.emit_word_buffer, 0, None);
    for (group, desc) in groups.iter().zip(descs.iter()) {
        let freq_groups = u32::try_from(desc.chunk_len)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .div_ceil(256)
            .max(1);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-bt-batch-freq-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.build_table_freq_pipeline);
            pass.set_bind_group(0, &group.freq, &[]);
            pass.dispatch_workgroups(freq_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-bt-batch-candidate-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.build_table_candidate_pipeline);
            pass.set_bind_group(0, &group.candidate, &[]);
            pass.dispatch_workgroups(group.cand_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-bt-batch-bucket-count-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.build_table_bucket_count_pipeline);
            pass.set_bind_group(0, &group.bucket_count, &[]);
            pass.dispatch_workgroups(group.cand_groups, 1, 1);
        }
    }
    let encode_pass1_ms = elapsed_ms(t_encode_pass1);

    let t_encode_pass2 = Instant::now();
    for group in &groups {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-bt-batch-bucket-prefix-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.build_table_bucket_prefix_pipeline);
            pass.set_bind_group(0, &group.bucket_prefix, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-bt-batch-bucket-scatter-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.build_table_bucket_scatter_pipeline);
            pass.set_bind_group(0, &group.bucket_scatter, &[]);
            pass.dispatch_workgroups(group.cand_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-bt-batch-finalize-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.build_table_finalize_pipeline);
            pass.set_bind_group(0, &group.finalize, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-bt-batch-finalize-emit-desc-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.build_table_finalize_emit_desc_pipeline);
            pass.set_bind_group(0, &group.finalize_emit_desc, &[]);
            pass.dispatch_workgroups(group.emit_groups, 1, 1);
        }
        {
            let emit_store_groups = group.emit_groups.div_ceil(4).max(1);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-bt-batch-finalize-emit-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.build_table_finalize_emit_pipeline);
            pass.set_bind_group(0, &group.finalize_emit, &[]);
            pass.dispatch_workgroups(emit_store_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-bt-batch-pack-index-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.build_table_pack_index_pipeline);
            pass.set_bind_group(0, &group.pack_index, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
    let encode_pass2_ms = elapsed_ms(t_encode_pass2);

    let t_submit = Instant::now();
    let pass1_submit_seq = next_gpu_submit_seq();
    let pass2_submit_seq = pass1_submit_seq;
    let submission = r.queue.submit(Some(encoder.finish()));
    let submit_ms = elapsed_ms(t_submit);
    let t_wait = Instant::now();
    r.device.poll(wgpu::Maintain::wait_for(submission.clone()));
    let wait_ms = elapsed_ms(t_wait);
    let total_build_ms = elapsed_ms(t_total_build);

    let chunk_count_f = chunks.len() as f64;
    let avg_total_ms = total_build_ms / chunk_count_f;
    let avg_pre_resource_ms = pre_resource_ms / chunk_count_f;
    let avg_resource_setup_ms = resource_setup_ms / chunk_count_f;
    let avg_upload_ms = upload_ms / chunk_count_f;
    let avg_encode_setup_ms = encode_setup_ms / chunk_count_f;
    let avg_encode_pass1_ms = encode_pass1_ms / chunk_count_f;
    let avg_encode_pass2_ms = encode_pass2_ms / chunk_count_f;
    let avg_submit_ms = submit_ms / chunk_count_f;
    let avg_wait_ms = wait_ms / chunk_count_f;
    let table_index_buffer = Arc::new(table_index_buffer);
    let table_data_buffer = Arc::new(table_data_buffer);
    let table_meta_buffer = Arc::new(table_meta_buffer);

    let mut out = Vec::with_capacity(descs.len());
    for desc in descs {
        out.push((
            GpuPackedTableDevice {
                table_index_buffer: Arc::clone(&table_index_buffer),
                table_data_buffer: Arc::clone(&table_data_buffer),
                table_meta_buffer: Arc::clone(&table_meta_buffer),
                table_index_offset: u64::try_from(desc.index_word_offset)
                    .map_err(|_| PDeflateError::NumericOverflow)?
                    .saturating_mul(4),
                table_data_offset: u64::try_from(desc.data_word_offset)
                    .map_err(|_| PDeflateError::NumericOverflow)?
                    .saturating_mul(4),
                table_meta_offset: u64::try_from(desc.meta_word_offset)
                    .map_err(|_| PDeflateError::NumericOverflow)?
                    .saturating_mul(4),
                table_count: 0,
                table_index_len: 0,
                table_data_len: 0,
                sizes_known: false,
                max_entries,
                table_data_bytes_cap: desc.table_data_bytes_cap,
                build_id: desc.build_id,
                build_submit_seq: pass2_submit_seq,
                build_submit_index: Some(submission.clone()),
                build_breakdown: GpuTableBuildBreakdown {
                    total_ms: avg_total_ms,
                    pre_resource_ms: avg_pre_resource_ms,
                    resource_setup_ms: avg_resource_setup_ms,
                    upload_ms: avg_upload_ms,
                    encode_pass1_ms: avg_encode_pass1_ms,
                    submit_pass1_ms: 0.0,
                    encode_pass2_ms: avg_encode_pass2_ms + avg_encode_setup_ms,
                    submit_pass2_ms: avg_submit_ms,
                    stage_probe_submit_ms: 0.0,
                    stage_probe_wait_ms: avg_wait_ms,
                    stage_probe_parse_ms: 0.0,
                    stage_gpu_sum_ms: avg_wait_ms,
                    tail_ms: 0.0,
                    other_ms: avg_pre_resource_ms,
                },
            },
            GpuBuildTableProfile {
                upload_ms: avg_upload_ms,
                freq_kernel_ms: 0.0,
                candidate_kernel_ms: 0.0,
                readback_ms: avg_wait_ms,
                materialize_ms: 0.0,
                sort_ms: 0.0,
                sample_count: desc.sample_count,
            },
        ));
    }
    release_table_build_batch_scratch(r, scratch);
    Ok(out)
}

fn resolve_packed_table_sizes(
    packed: &GpuPackedTableDevice,
) -> Result<(usize, usize, usize), PDeflateError> {
    if packed.sizes_known {
        return Ok((
            packed.table_count,
            packed.table_index_len,
            packed.table_data_len,
        ));
    }
    if packed.max_entries == 0 {
        return Ok((0, 0, 0));
    }
    let r = runtime()?;
    readback_table_sizes_from_meta_buffer(
        r,
        &packed.table_meta_buffer,
        packed.table_meta_offset,
        packed.max_entries,
        packed.table_data_bytes_cap,
    )
}

fn resolve_packed_table_sizes_batch(
    packed_tables: &[&GpuPackedTableDevice],
) -> Result<
    (
        Vec<(usize, usize, usize)>,
        ResolvePackedTableSizesBatchProfile,
    ),
    PDeflateError,
> {
    let t_total = Instant::now();
    let mut profile = ResolvePackedTableSizesBatchProfile::default();
    let mut out = vec![(0usize, 0usize, 0usize); packed_tables.len()];
    let mut unresolved = Vec::<(usize, &GpuPackedTableDevice)>::new();
    let t_scan = Instant::now();
    for (idx, packed) in packed_tables.iter().copied().enumerate() {
        if packed.sizes_known {
            out[idx] = (
                packed.table_count,
                packed.table_index_len,
                packed.table_data_len,
            );
        } else if packed.max_entries > 0 {
            unresolved.push((idx, packed));
        }
    }
    profile.scan_ms = elapsed_ms(t_scan);
    profile.unresolved_count = unresolved.len();
    if unresolved.is_empty() {
        profile.total_ms = elapsed_ms(t_total);
        return Ok((out, profile));
    }
    let mut prebuild_total_ms = 0.0_f64;
    let mut prebuild_pre_resource_ms = 0.0_f64;
    let mut prebuild_resource_setup_ms = 0.0_f64;
    let mut prebuild_upload_ms = 0.0_f64;
    let mut prebuild_encode_pass1_ms = 0.0_f64;
    let mut prebuild_submit_pass1_ms = 0.0_f64;
    let mut prebuild_encode_pass2_ms = 0.0_f64;
    let mut prebuild_submit_pass2_ms = 0.0_f64;
    let mut prebuild_stage_probe_submit_ms = 0.0_f64;
    let mut prebuild_stage_probe_wait_ms = 0.0_f64;
    let mut prebuild_stage_probe_parse_ms = 0.0_f64;
    let mut prebuild_stage_gpu_sum_ms = 0.0_f64;
    let mut prebuild_tail_ms = 0.0_f64;
    let mut prebuild_other_ms = 0.0_f64;
    let mut prebuild_build_id_min = u64::MAX;
    let mut prebuild_build_id_max = 0u64;
    let mut prebuild_submit_seq_min = u64::MAX;
    let mut prebuild_submit_seq_max = 0u64;
    let mut latest_build_submit: Option<(u64, u64, wgpu::SubmissionIndex)> = None;
    let mut unresolved_build_ids = Vec::<u64>::with_capacity(unresolved.len());
    for (_, packed) in unresolved.iter().copied() {
        let b = packed.build_breakdown;
        prebuild_total_ms += b.total_ms;
        prebuild_pre_resource_ms += b.pre_resource_ms;
        prebuild_resource_setup_ms += b.resource_setup_ms;
        prebuild_upload_ms += b.upload_ms;
        prebuild_encode_pass1_ms += b.encode_pass1_ms;
        prebuild_submit_pass1_ms += b.submit_pass1_ms;
        prebuild_encode_pass2_ms += b.encode_pass2_ms;
        prebuild_submit_pass2_ms += b.submit_pass2_ms;
        prebuild_stage_probe_submit_ms += b.stage_probe_submit_ms;
        prebuild_stage_probe_wait_ms += b.stage_probe_wait_ms;
        prebuild_stage_probe_parse_ms += b.stage_probe_parse_ms;
        prebuild_stage_gpu_sum_ms += b.stage_gpu_sum_ms;
        prebuild_tail_ms += b.tail_ms;
        prebuild_other_ms += b.other_ms;
        if packed.build_id != 0 {
            prebuild_build_id_min = prebuild_build_id_min.min(packed.build_id);
            prebuild_build_id_max = prebuild_build_id_max.max(packed.build_id);
            unresolved_build_ids.push(packed.build_id);
        }
        if packed.build_submit_seq != 0 {
            prebuild_submit_seq_min = prebuild_submit_seq_min.min(packed.build_submit_seq);
            prebuild_submit_seq_max = prebuild_submit_seq_max.max(packed.build_submit_seq);
        }
        if let Some(submit_index) = packed.build_submit_index.clone() {
            let should_replace = latest_build_submit
                .as_ref()
                .map(|(seq, _, _)| packed.build_submit_seq > *seq)
                .unwrap_or(true);
            if should_replace {
                latest_build_submit =
                    Some((packed.build_submit_seq, packed.build_id, submit_index));
            }
        }
    }

    let r = runtime()?;
    let t_setup = Instant::now();
    let readback_bytes = u64::try_from(unresolved.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(8);
    let readback = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-batch-size-readback"),
        size: readback_bytes.max(8),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-bt-batch-size-readback-encoder"),
        });
    for (slot, (_, packed)) in unresolved.iter().enumerate() {
        let dst = u64::try_from(slot)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(8);
        let data_total_idx = 1usize
            .checked_add(
                packed
                    .max_entries
                    .checked_mul(2)
                    .ok_or(PDeflateError::NumericOverflow)?,
            )
            .ok_or(PDeflateError::NumericOverflow)?;
        encoder.copy_buffer_to_buffer(
            &packed.table_meta_buffer,
            packed.table_meta_offset,
            &readback,
            dst,
            4,
        );
        encoder.copy_buffer_to_buffer(
            &packed.table_meta_buffer,
            packed
                .table_meta_offset
                .checked_add(
                    u64::try_from(data_total_idx)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .saturating_mul(4),
                )
                .ok_or(PDeflateError::NumericOverflow)?,
            &readback,
            dst + 4,
            4,
        );
    }
    profile.readback_setup_ms = elapsed_ms(t_setup);
    if resolve_wait_attribution_probe_enabled() {
        if let Some((latest_submit_seq, _, latest_submit_idx)) = latest_build_submit.as_ref() {
            let t_pre_wait = Instant::now();
            r.device
                .poll(wgpu::Maintain::wait_for(latest_submit_idx.clone()));
            profile.pre_wait_latest_build_submit_ms = elapsed_ms(t_pre_wait);
            profile.latest_build_submit_seq = *latest_submit_seq;
        }
    }
    let t_submit = Instant::now();
    let readback_submit_seq = next_gpu_submit_seq();
    let submission = r.queue.submit(Some(encoder.finish()));
    profile.readback_submit_seq = readback_submit_seq;
    profile.submit_ms = elapsed_ms(t_submit);

    let t_map_wait = Instant::now();
    let slice = readback.slice(..readback_bytes.max(8));
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let t_submit_done_wait = Instant::now();
    r.device.poll(wgpu::Maintain::wait_for(submission));
    profile.submit_done_wait_ms = elapsed_ms(t_submit_done_wait);
    let t_map_callback_wait = Instant::now();
    rx.recv()
        .map_err(|_| {
            PDeflateError::Gpu("gpu build-table batch size map channel closed".to_string())
        })?
        .map_err(|e| PDeflateError::Gpu(format!("gpu build-table batch size map failed: {e}")))?;
    profile.map_callback_wait_ms = elapsed_ms(t_map_callback_wait);
    profile.map_wait_ms = elapsed_ms(t_map_wait);
    if resolve_map_wait_probe_enabled() {
        let seq = RESOLVE_MAP_WAIT_PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
        if table_stage_probe_should_log(seq) {
            let build_id_range = if prebuild_build_id_min != u64::MAX {
                format!("{}..{}", prebuild_build_id_min, prebuild_build_id_max)
            } else {
                "none".to_string()
            };
            let submit_seq_range = if prebuild_submit_seq_min != u64::MAX {
                format!("{}..{}", prebuild_submit_seq_min, prebuild_submit_seq_max)
            } else {
                "none".to_string()
            };
            let build_ids = if unresolved_build_ids.is_empty() {
                "none".to_string()
            } else {
                unresolved_build_ids
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let (latest_submit_seq, latest_build_id) = latest_build_submit
                .as_ref()
                .map(|(submit_seq, build_id, _)| (*submit_seq, *build_id))
                .unwrap_or((0, 0));
            eprintln!(
                "[cozip_pdeflate][timing][resolve-map-wait-breakdown] seq={} unresolved={} readback_kib={:.1} latest_build_submit_seq={} latest_build_id={} readback_submit_seq={} pre_wait_latest_build_submit_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_ms={:.3} map_wait_ms={:.3}",
                seq,
                unresolved.len(),
                (readback_bytes as f64) / 1024.0,
                latest_submit_seq,
                latest_build_id,
                profile.readback_submit_seq,
                profile.pre_wait_latest_build_submit_ms,
                profile.submit_done_wait_ms,
                profile.map_callback_wait_ms,
                profile.map_wait_ms,
            );
            let unresolved_f = unresolved.len() as f64;
            eprintln!(
                "[cozip_pdeflate][timing][resolve-preceding-gpu-breakdown] seq={} unresolved={} build_id_range={} submit_seq_range={} build_ids={} prebuild_total_ms={:.3} prebuild_pre_resource_ms={:.3} prebuild_resource_setup_ms={:.3} prebuild_upload_ms={:.3} prebuild_encode_pass1_ms={:.3} prebuild_submit_pass1_ms={:.3} prebuild_encode_pass2_ms={:.3} prebuild_submit_pass2_ms={:.3} prebuild_stage_probe_submit_ms={:.3} prebuild_stage_probe_wait_ms={:.3} prebuild_stage_probe_parse_ms={:.3} prebuild_stage_gpu_sum_ms={:.3} prebuild_tail_ms={:.3} prebuild_other_ms={:.3} prebuild_avg_ms={:.3}",
                seq,
                unresolved.len(),
                build_id_range,
                submit_seq_range,
                build_ids,
                prebuild_total_ms,
                prebuild_pre_resource_ms,
                prebuild_resource_setup_ms,
                prebuild_upload_ms,
                prebuild_encode_pass1_ms,
                prebuild_submit_pass1_ms,
                prebuild_encode_pass2_ms,
                prebuild_submit_pass2_ms,
                prebuild_stage_probe_submit_ms,
                prebuild_stage_probe_wait_ms,
                prebuild_stage_probe_parse_ms,
                prebuild_stage_gpu_sum_ms,
                prebuild_tail_ms,
                prebuild_other_ms,
                if unresolved_f > 0.0 {
                    prebuild_total_ms / unresolved_f
                } else {
                    0.0
                },
            );
            if resolve_wait_attribution_probe_enabled() {
                eprintln!(
                    "[cozip_pdeflate][timing][resolve-wait-attribution] seq={} unresolved={} pre_wait_latest_build_submit_ms={:.3} submit_done_wait_ms={:.3} map_wait_ms={:.3} note=\"pre_wait captures unresolved build completion; submit_done_wait is residual after readback submit\"",
                    seq,
                    unresolved.len(),
                    profile.pre_wait_latest_build_submit_ms,
                    profile.submit_done_wait_ms,
                    profile.map_wait_ms,
                );
            }
        }
    }
    let t_parse = Instant::now();
    let mapped = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&mapped);
    for (slot, (idx, packed)) in unresolved.iter().enumerate() {
        let base = slot.saturating_mul(2);
        let count = usize::try_from(*words.get(base).unwrap_or(&0))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .min(packed.max_entries);
        let data_total = usize::try_from(*words.get(base + 1).unwrap_or(&0))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .min(packed.table_data_bytes_cap);
        out[*idx] = (count, count, data_total);
    }
    drop(mapped);
    readback.unmap();
    profile.parse_ms = elapsed_ms(t_parse);
    profile.total_ms = elapsed_ms(t_total);
    Ok((out, profile))
}

pub(crate) fn readback_packed_table_device(
    packed: &GpuPackedTableDevice,
) -> Result<GpuPackedTable, PDeflateError> {
    let (table_count, table_index_len, table_data_len) = resolve_packed_table_sizes(packed)?;
    if table_count == 0 || (table_index_len == 0 && table_data_len == 0) {
        return Ok(GpuPackedTable {
            table_index: Vec::new(),
            table_data: Vec::new(),
            table_count: 0,
        });
    }
    let r = runtime()?;
    let index_copy_bytes = u64::try_from(table_index_len.div_ceil(4))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let data_copy_bytes = u64::try_from(table_data_len.div_ceil(4))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let total_copy_bytes = index_copy_bytes
        .checked_add(data_copy_bytes)
        .ok_or(PDeflateError::NumericOverflow)?;
    let readback = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-readback-table-combined"),
        size: total_copy_bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut submission = None;
    if total_copy_bytes > 0 {
        let mut encoder = r
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-pdeflate-bt-readback-encoder"),
            });
        if index_copy_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &packed.table_index_buffer,
                packed.table_index_offset,
                &readback,
                0,
                index_copy_bytes,
            );
        }
        if data_copy_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &packed.table_data_buffer,
                packed.table_data_offset,
                &readback,
                index_copy_bytes,
                data_copy_bytes,
            );
        }
        submission = Some(r.queue.submit(Some(encoder.finish())));
    }

    let slice = readback.slice(..total_copy_bytes.max(4));
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    if let Some(submission) = submission {
        r.device.poll(wgpu::Maintain::wait_for(submission));
    } else {
        r.device.poll(wgpu::Maintain::Poll);
    }
    rx.recv()
        .map_err(|_| PDeflateError::Gpu("gpu build-table combined map channel closed".to_string()))?
        .map_err(|e| PDeflateError::Gpu(format!("gpu build-table combined map failed: {e}")))?;
    let mapped = slice.get_mapped_range();
    let mut table_index = vec![0u8; table_index_len];
    let mut table_data = vec![0u8; table_data_len];
    if table_index_len > 0 {
        table_index.copy_from_slice(&mapped[..table_index_len]);
    }
    if table_data_len > 0 {
        let data_off =
            usize::try_from(index_copy_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
        let data_end = data_off
            .checked_add(table_data_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        table_data.copy_from_slice(&mapped[data_off..data_end]);
    }
    drop(mapped);
    readback.unmap();

    Ok(GpuPackedTable {
        table_index,
        table_data,
        table_count,
    })
}

#[allow(dead_code)]
pub(crate) fn build_table_gpu(
    chunk: &[u8],
    max_entries: usize,
    max_entry_len: usize,
    min_ref_len: usize,
    match_probe_limit: usize,
    hash_history_limit: usize,
    table_sample_stride: usize,
) -> Result<(GpuPackedTable, GpuBuildTableProfile), PDeflateError> {
    let (packed_device, mut profile) = build_table_gpu_device(
        chunk,
        max_entries,
        max_entry_len,
        min_ref_len,
        match_probe_limit,
        hash_history_limit,
        table_sample_stride,
    )?;
    let t_readback = Instant::now();
    let packed = readback_packed_table_device(&packed_device)?;
    profile.materialize_ms += elapsed_ms(t_readback);
    Ok((packed, profile))
}

struct PackedBatchInputs<'a> {
    src_words: Vec<u32>,
    chunk_starts: Vec<u32>,
    table_chunk_bases: Vec<u32>,
    table_chunk_counts: Vec<u32>,
    table_chunk_index_offsets: Vec<u32>,
    table_chunk_data_offsets: Vec<u32>,
    table_chunk_meta_words: Vec<u32>,
    table_index_words: Vec<u32>,
    table_data_words: Vec<u32>,
    table_index_words_len: usize,
    table_data_words_len: usize,
    table_device_copies: Vec<PackedTableDeviceCopy<'a>>,
    table_meta_device_copies: Vec<PackedTableMetaDeviceCopy<'a>>,
    chunk_lens: Vec<usize>,
    total_src_len: usize,
    total_table_entries: usize,
    max_ref_len: usize,
    min_ref_len: usize,
    section_count: usize,
}

struct PackedTableDeviceCopy<'a> {
    index_src: &'a wgpu::Buffer,
    data_src: &'a wgpu::Buffer,
    index_src_offset: u64,
    data_src_offset: u64,
    index_dst_offset: u64,
    data_dst_offset: u64,
    index_len: usize,
    data_len: usize,
}

struct PackedTableMetaDeviceCopy<'a> {
    meta_src: &'a wgpu::Buffer,
    src_count_offset: u64,
    dst_count_offset: u64,
    dst_data_total_offset: u64,
    src_data_total_offset: u64,
}

struct PackedTableUpload<'a> {
    index_offset: u64,
    data_offset: u64,
    index_len: usize,
    data_len: usize,
    kind: PackedTableUploadKind<'a>,
}

enum PackedTableUploadKind<'a> {
    HostBorrow {
        index_bytes: &'a [u8],
        data_bytes: &'a [u8],
    },
    HostOwned {
        index_bytes: Vec<u8>,
        data_bytes: Vec<u8>,
    },
    Device {
        index_buffer: &'a wgpu::Buffer,
        data_buffer: &'a wgpu::Buffer,
        index_offset: u64,
        data_offset: u64,
    },
}

fn pack_match_batch_inputs<'a>(
    inputs: &'a [GpuMatchInput<'a>],
) -> Result<(PackedBatchInputs<'a>, PackMatchBatchInputsProfile), PDeflateError> {
    let t_total = Instant::now();
    if inputs.is_empty() {
        return Err(PDeflateError::InvalidOptions(
            "gpu match batch inputs must be non-empty",
        ));
    }
    let max_ref_len = inputs[0].max_ref_len;
    let min_ref_len = inputs[0].min_ref_len;
    let section_count = inputs[0].section_count;
    if max_ref_len == 0 || min_ref_len == 0 || section_count == 0 {
        return Err(PDeflateError::InvalidOptions(
            "gpu match params must be non-zero",
        ));
    }
    let t_alloc_setup = Instant::now();
    let total_src_bytes: usize = inputs.iter().map(|input| input.src.len()).sum();
    let mut src_words = vec![0u32; total_src_bytes.div_ceil(4)];
    let mut chunk_starts = Vec::<u32>::with_capacity(inputs.len() + 1);
    let mut table_chunk_bases = Vec::<u32>::with_capacity(inputs.len());
    let mut table_chunk_counts = Vec::<u32>::with_capacity(inputs.len());
    let mut table_chunk_index_offsets = Vec::<u32>::with_capacity(inputs.len());
    let mut table_chunk_data_offsets = Vec::<u32>::with_capacity(inputs.len());
    let mut table_chunk_meta_words = Vec::<u32>::with_capacity(inputs.len().saturating_mul(2));
    let mut table_uploads = Vec::<PackedTableUpload<'a>>::with_capacity(inputs.len());
    let mut table_meta_device_copies = Vec::<PackedTableMetaDeviceCopy<'a>>::new();
    let mut table_index_packer =
        ByteWordPacker::with_capacity_bytes(inputs.len().saturating_mul(256));
    let mut table_data_packer =
        ByteWordPacker::with_capacity_bytes(inputs.iter().map(|i| i.src.len() / 2).sum());
    let mut chunk_lens = Vec::<usize>::with_capacity(inputs.len());
    let mut global_table_base = 0usize;
    let alloc_setup_ms = elapsed_ms(t_alloc_setup);
    let resolve_profile = ResolvePackedTableSizesBatchProfile::default();
    let resolve_sizes_ms = 0.0_f64;
    chunk_starts.push(0);
    let mut src_len = 0usize;
    let mut src_copy_ms = 0.0_f64;
    let mut metadata_loop_ms = 0.0_f64;
    {
        let src_words_bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut src_words);
        for input in inputs {
            let t_src_copy = Instant::now();
            let next_src_len = src_len
                .checked_add(input.src.len())
                .ok_or(PDeflateError::NumericOverflow)?;
            src_words_bytes[src_len..next_src_len].copy_from_slice(input.src);
            src_len = next_src_len;
            chunk_starts.push(u32::try_from(src_len).map_err(|_| PDeflateError::NumericOverflow)?);
            src_copy_ms += elapsed_ms(t_src_copy);
            let t_meta = Instant::now();
            if input.max_ref_len != max_ref_len
                || input.min_ref_len != min_ref_len
                || input.section_count != section_count
            {
                return Err(PDeflateError::InvalidOptions(
                    "gpu match batch params must be consistent",
                ));
            }
            chunk_lens.push(input.src.len());
            table_chunk_bases.push(
                u32::try_from(global_table_base).map_err(|_| PDeflateError::NumericOverflow)?,
            );
            table_index_packer.align4();
            table_data_packer.align4();
            table_chunk_index_offsets.push(
                u32::try_from(table_index_packer.len_bytes())
                    .map_err(|_| PDeflateError::NumericOverflow)?,
            );
            table_chunk_data_offsets.push(
                u32::try_from(table_data_packer.len_bytes())
                    .map_err(|_| PDeflateError::NumericOverflow)?,
            );
            let meta_word_base = table_chunk_meta_words.len();
            table_chunk_meta_words.push(0);
            table_chunk_meta_words.push(0);
            let mut local_count = 0usize;
            let (index_len, data_len, kind) = if let Some(table_gpu) = input.table_gpu {
                let (count_hint, index_len_hint, data_len_hint, unresolved_gpu_sizes) = if table_gpu
                    .sizes_known
                    || table_gpu.table_count > 0
                    || table_gpu.table_index_len > 0
                    || table_gpu.table_data_len > 0
                {
                    (
                        table_gpu.table_count,
                        table_gpu.table_index_len,
                        table_gpu.table_data_len,
                        false,
                    )
                } else {
                    (
                        table_gpu.max_entries,
                        table_gpu.max_entries,
                        table_gpu.table_data_bytes_cap,
                        true,
                    )
                };
                local_count = count_hint;
                if !unresolved_gpu_sizes {
                    table_chunk_meta_words[meta_word_base] =
                        u32::try_from(count_hint).map_err(|_| PDeflateError::NumericOverflow)?;
                    table_chunk_meta_words[meta_word_base + 1] =
                        u32::try_from(data_len_hint).map_err(|_| PDeflateError::NumericOverflow)?;
                } else {
                    let dst_count_offset = u64::try_from(meta_word_base)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .saturating_mul(4);
                    let dst_data_total_offset = u64::try_from(meta_word_base + 1)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .saturating_mul(4);
                    let src_data_total_offset_words = 1usize
                        .checked_add(
                            table_gpu
                                .max_entries
                                .checked_mul(2)
                                .ok_or(PDeflateError::NumericOverflow)?,
                        )
                        .ok_or(PDeflateError::NumericOverflow)?;
                    let src_data_total_offset = u64::try_from(src_data_total_offset_words)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .saturating_mul(4);
                    table_meta_device_copies.push(PackedTableMetaDeviceCopy {
                        meta_src: &table_gpu.table_meta_buffer,
                        src_count_offset: table_gpu.table_meta_offset,
                        dst_count_offset,
                        dst_data_total_offset,
                        src_data_total_offset: table_gpu
                            .table_meta_offset
                            .checked_add(src_data_total_offset)
                            .ok_or(PDeflateError::NumericOverflow)?,
                    });
                }
                (
                    index_len_hint,
                    data_len_hint,
                    PackedTableUploadKind::Device {
                        index_buffer: &table_gpu.table_index_buffer,
                        data_buffer: &table_gpu.table_data_buffer,
                        index_offset: table_gpu.table_index_offset,
                        data_offset: table_gpu.table_data_offset,
                    },
                )
            } else if let (Some(index_bytes), Some(data_bytes)) =
                (input.table_index, input.table_data)
            {
                let expected_data_len: usize = index_bytes
                    .iter()
                    .map(|&v| usize::from(v))
                    .try_fold(0usize, |acc, v| acc.checked_add(v))
                    .ok_or(PDeflateError::NumericOverflow)?;
                if expected_data_len != data_bytes.len() {
                    return Err(PDeflateError::Gpu(
                        "gpu packed table index/data length mismatch".to_string(),
                    ));
                }
                local_count = index_bytes.len();
                table_chunk_meta_words[meta_word_base] =
                    u32::try_from(local_count).map_err(|_| PDeflateError::NumericOverflow)?;
                table_chunk_meta_words[meta_word_base + 1] =
                    u32::try_from(data_bytes.len()).map_err(|_| PDeflateError::NumericOverflow)?;
                (
                    index_bytes.len(),
                    data_bytes.len(),
                    PackedTableUploadKind::HostBorrow {
                        index_bytes,
                        data_bytes,
                    },
                )
            } else {
                let mut index_bytes = Vec::<u8>::new();
                let mut data_bytes = Vec::<u8>::new();
                index_bytes.reserve(input.table.len());
                let data_len: usize = input.table.iter().map(Vec::len).sum();
                data_bytes.reserve(data_len);
                for entry in input.table.iter() {
                    index_bytes.push(
                        u8::try_from(entry.len()).map_err(|_| PDeflateError::NumericOverflow)?,
                    );
                    data_bytes.extend_from_slice(entry);
                    local_count = local_count.saturating_add(1);
                }
                table_chunk_meta_words[meta_word_base] =
                    u32::try_from(local_count).map_err(|_| PDeflateError::NumericOverflow)?;
                table_chunk_meta_words[meta_word_base + 1] =
                    u32::try_from(data_bytes.len()).map_err(|_| PDeflateError::NumericOverflow)?;
                (
                    index_bytes.len(),
                    data_bytes.len(),
                    PackedTableUploadKind::HostOwned {
                        index_bytes,
                        data_bytes,
                    },
                )
            };
            table_uploads.push(PackedTableUpload {
                index_offset: u64::try_from(table_index_packer.len_bytes())
                    .map_err(|_| PDeflateError::NumericOverflow)?,
                data_offset: u64::try_from(table_data_packer.len_bytes())
                    .map_err(|_| PDeflateError::NumericOverflow)?,
                index_len,
                data_len,
                kind,
            });
            table_index_packer.append_zeros(index_len);
            table_data_packer.append_zeros(data_len);
            table_chunk_counts
                .push(u32::try_from(local_count).map_err(|_| PDeflateError::NumericOverflow)?);
            global_table_base = global_table_base
                .checked_add(local_count)
                .ok_or(PDeflateError::NumericOverflow)?;
            metadata_loop_ms += elapsed_ms(t_meta);
        }
    }
    let mut host_copy_ms = 0.0_f64;
    let mut device_copy_plan_ms = 0.0_f64;
    let mut table_device_copies = Vec::<PackedTableDeviceCopy<'a>>::new();
    for upload in table_uploads.into_iter() {
        match upload.kind {
            PackedTableUploadKind::HostBorrow {
                index_bytes,
                data_bytes,
            } => {
                let t_host_copy = Instant::now();
                if upload.index_len != 0 {
                    let off = usize::try_from(upload.index_offset)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    table_index_packer.write_bytes_at(off, index_bytes)?;
                }
                if upload.data_len != 0 {
                    let off = usize::try_from(upload.data_offset)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    table_data_packer.write_bytes_at(off, data_bytes)?;
                }
                host_copy_ms += elapsed_ms(t_host_copy);
            }
            PackedTableUploadKind::HostOwned {
                index_bytes,
                data_bytes,
            } => {
                let t_host_copy = Instant::now();
                if upload.index_len != 0 {
                    let off = usize::try_from(upload.index_offset)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    table_index_packer.write_bytes_at(off, &index_bytes)?;
                }
                if upload.data_len != 0 {
                    let off = usize::try_from(upload.data_offset)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    table_data_packer.write_bytes_at(off, &data_bytes)?;
                }
                host_copy_ms += elapsed_ms(t_host_copy);
            }
            PackedTableUploadKind::Device {
                index_buffer,
                data_buffer,
                index_offset,
                data_offset,
            } => {
                let t_device_plan = Instant::now();
                table_device_copies.push(PackedTableDeviceCopy {
                    index_src: index_buffer,
                    data_src: data_buffer,
                    index_src_offset: index_offset,
                    data_src_offset: data_offset,
                    index_dst_offset: upload.index_offset,
                    data_dst_offset: upload.data_offset,
                    index_len: upload.index_len,
                    data_len: upload.data_len,
                });
                device_copy_plan_ms += elapsed_ms(t_device_plan);
            }
        }
    }

    let t_finalize = Instant::now();
    let table_index_words_len = table_index_packer.len_bytes().div_ceil(4);
    let table_data_words_len = table_data_packer.len_bytes().div_ceil(4);
    let table_index_words = table_index_packer.into_words();
    let table_data_words = table_data_packer.into_words();
    let finalize_ms = elapsed_ms(t_finalize);

    let packed = PackedBatchInputs {
        src_words,
        chunk_starts,
        table_chunk_bases,
        table_chunk_counts,
        table_chunk_index_offsets,
        table_chunk_data_offsets,
        table_chunk_meta_words,
        table_index_words_len,
        table_data_words_len,
        table_index_words,
        table_data_words,
        table_device_copies,
        table_meta_device_copies,
        chunk_lens,
        total_src_len: total_src_bytes,
        total_table_entries: global_table_base,
        max_ref_len,
        min_ref_len,
        section_count,
    };
    let profile = PackMatchBatchInputsProfile {
        total_ms: elapsed_ms(t_total),
        alloc_setup_ms,
        resolve_sizes_ms,
        resolve_scan_ms: resolve_profile.scan_ms,
        resolve_readback_setup_ms: resolve_profile.readback_setup_ms,
        resolve_submit_ms: resolve_profile.submit_ms,
        resolve_map_wait_ms: resolve_profile.map_wait_ms,
        resolve_parse_ms: resolve_profile.parse_ms,
        src_copy_ms,
        metadata_loop_ms,
        host_copy_ms,
        device_copy_plan_ms,
        finalize_ms,
    };
    Ok((packed, profile))
}

fn encode_table_device_copies(
    encoder: &mut wgpu::CommandEncoder,
    scratch: &GpuBatchScratch,
    copies: &[PackedTableDeviceCopy<'_>],
    meta_copies: &[PackedTableMetaDeviceCopy<'_>],
) -> Result<(), PDeflateError> {
    for copy in copies {
        let index_copy_bytes = u64::try_from(copy.index_len.div_ceil(4))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        if index_copy_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &copy.index_src,
                copy.index_src_offset,
                &scratch.table_index_buffer,
                copy.index_dst_offset,
                index_copy_bytes,
            );
        }
        let data_copy_bytes = u64::try_from(copy.data_len.div_ceil(4))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        if data_copy_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &copy.data_src,
                copy.data_src_offset,
                &scratch.table_data_buffer,
                copy.data_dst_offset,
                data_copy_bytes,
            );
        }
    }
    for meta in meta_copies {
        encoder.copy_buffer_to_buffer(
            meta.meta_src,
            meta.src_count_offset,
            &scratch.table_meta_buffer,
            meta.dst_count_offset,
            4,
        );
        encoder.copy_buffer_to_buffer(
            meta.meta_src,
            meta.src_data_total_offset,
            &scratch.table_meta_buffer,
            meta.dst_data_total_offset,
            4,
        );
    }
    Ok(())
}

pub(crate) fn compute_matches_batch(
    inputs: &[GpuMatchInput<'_>],
) -> Result<GpuBatchMatchOutput, PDeflateError> {
    if inputs.is_empty() {
        return Ok(GpuBatchMatchOutput {
            chunks: Vec::new(),
            profile: GpuMatchProfile::default(),
        });
    }
    let r = runtime()?;
    let t_total = Instant::now();
    let (packed, _pack_profile) = pack_match_batch_inputs(inputs)?;
    let groups_total = (u32::try_from(packed.total_src_len)
        .map_err(|_| PDeflateError::NumericOverflow)?)
    .div_ceil(WORKGROUP_SIZE)
    .max(1);
    let groups_x = groups_total.min(65_535);
    let groups_y = groups_total.div_ceil(groups_x).max(1);
    let row_stride = groups_x.saturating_mul(WORKGROUP_SIZE);
    let params: [u32; 6] = [
        u32::try_from(packed.total_src_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(packed.max_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(packed.min_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        row_stride,
        u32::try_from(packed.section_count).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?,
    ];
    let out_len_u64 = u64::try_from(packed.total_src_len)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let required_caps = GpuBatchScratchCaps {
        src_bytes: u64::try_from(packed.src_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        chunk_starts_bytes: u64::try_from(packed.chunk_starts.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_base_bytes: u64::try_from(packed.table_chunk_bases.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_count_bytes: u64::try_from(packed.table_chunk_counts.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_index_offsets_bytes: u64::try_from(packed.table_chunk_index_offsets.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_data_offsets_bytes: u64::try_from(packed.table_chunk_data_offsets.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_meta_bytes: u64::try_from(packed.table_chunk_meta_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_index_bytes: u64::try_from(packed.table_index_words_len)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        prefix_first_bytes: u64::try_from(inputs.len().saturating_mul(PREFIX2_TABLE_SIZE))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_lens_bytes: u64::try_from(packed.total_table_entries)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_offsets_bytes: u64::try_from(packed.total_table_entries)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_data_bytes: u64::try_from(packed.table_data_words_len)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        prep_params_bytes: 4,
        params_bytes: u64::try_from(params.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        out_bytes: out_len_u64.max(4),
    };

    let scratch = acquire_batch_scratch(r, required_caps)?;

    let result = (|| -> Result<GpuBatchMatchOutput, PDeflateError> {
        let t_upload = Instant::now();
        r.queue.write_buffer(
            &scratch.src_buffer,
            0,
            bytemuck::cast_slice(&packed.src_words),
        );
        r.queue.write_buffer(
            &scratch.chunk_starts_buffer,
            0,
            bytemuck::cast_slice(&packed.chunk_starts),
        );
        r.queue.write_buffer(
            &scratch.table_base_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_bases),
        );
        r.queue.write_buffer(
            &scratch.table_count_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_counts),
        );
        r.queue.write_buffer(
            &scratch.table_index_offsets_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_index_offsets),
        );
        r.queue.write_buffer(
            &scratch.table_data_offsets_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_data_offsets),
        );
        if !packed.table_chunk_meta_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_meta_buffer,
                0,
                bytemuck::cast_slice(&packed.table_chunk_meta_words),
            );
        }
        if !packed.table_index_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_index_buffer,
                0,
                bytemuck::cast_slice(&packed.table_index_words),
            );
        }
        if !packed.table_data_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_data_buffer,
                0,
                bytemuck::cast_slice(&packed.table_data_words),
            );
        }
        let prep_params: [u32; 1] =
            [u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?];
        r.queue.write_buffer(
            &scratch.prep_params_buffer,
            0,
            bytemuck::cast_slice(&prep_params),
        );
        r.queue
            .write_buffer(&scratch.params_buffer, 0, bytemuck::cast_slice(&params));
        let upload_ms = elapsed_ms(t_upload);

        let mut encoder = r
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-pdeflate-match-encoder"),
            });
        encode_table_device_copies(
            &mut encoder,
            &scratch,
            &packed.table_device_copies,
            &packed.table_meta_device_copies,
        )?;
        encoder.clear_buffer(&scratch.prefix_first_buffer, 0, None);
        encoder.clear_buffer(&scratch.table_lens_buffer, 0, None);
        encoder.clear_buffer(&scratch.table_offsets_buffer, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-match-prepare-table-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.match_prepare_table_pipeline);
            pass.set_bind_group(0, &scratch.prep_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-match-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.pipeline);
            pass.set_bind_group(0, &scratch.bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        encoder.copy_buffer_to_buffer(
            &scratch.out_buffer,
            0,
            &scratch.readback_buffer,
            0,
            out_len_u64.max(4),
        );
        let submission = r.queue.submit(Some(encoder.finish()));

        let t_wait = Instant::now();
        let slice = scratch.readback_buffer.slice(..out_len_u64.max(4));
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        r.device.poll(wgpu::Maintain::wait_for(submission));
        let map_res = rx
            .recv()
            .map_err(|_| PDeflateError::Gpu("gpu map_async channel closed".to_string()))?;
        map_res.map_err(|e| PDeflateError::Gpu(format!("gpu map_async failed: {e}")))?;
        let wait_ms = elapsed_ms(t_wait);

        let t_copy = Instant::now();
        let mapped = slice.get_mapped_range();
        let mut packed_all = vec![0u32; packed.total_src_len];
        if !packed_all.is_empty() {
            packed_all
                .copy_from_slice(bytemuck::cast_slice(&mapped)[..packed.total_src_len].as_ref());
        }
        drop(mapped);
        scratch.readback_buffer.unmap();
        let map_copy_ms = elapsed_ms(t_copy);

        let mut chunks = Vec::with_capacity(packed.chunk_lens.len());
        let mut offset = 0usize;
        for len in packed.chunk_lens.iter().copied() {
            let end = offset
                .checked_add(len)
                .ok_or(PDeflateError::NumericOverflow)?;
            chunks.push(GpuBatchChunkOutput {
                packed_matches: packed_all[offset..end].to_vec(),
            });
            offset = end;
        }

        Ok(GpuBatchMatchOutput {
            chunks,
            profile: GpuMatchProfile {
                upload_ms,
                wait_ms,
                map_copy_ms,
                total_ms: elapsed_ms(t_total),
            },
        })
    })();

    release_batch_scratch(r, scratch);

    result
}

pub(crate) fn compute_matches_and_encode_sections_batch(
    inputs: &[GpuMatchInput<'_>],
    section_count: usize,
    max_cmd_len: usize,
) -> Result<GpuBatchSectionEncodeOutput, PDeflateError> {
    if inputs.is_empty() {
        return Ok(GpuBatchSectionEncodeOutput {
            chunks: Vec::new(),
            match_profile: GpuMatchProfile::default(),
            section_profile: GpuSectionEncodeProfile::default(),
            kernel_profile: GpuBatchKernelProfile::default(),
            keepalive: None,
        });
    }
    if section_count == 0 {
        return Err(PDeflateError::InvalidOptions(
            "gpu section encode requires section_count > 0",
        ));
    }

    let r = runtime()?;
    let t_total = Instant::now();
    let (packed, pack_profile) = pack_match_batch_inputs(inputs)?;
    let pack_inputs_ms = pack_profile.total_ms;
    if packed.section_count != section_count {
        return Err(PDeflateError::InvalidOptions(
            "gpu section_count mismatch between input and call",
        ));
    }
    let groups_total = (u32::try_from(packed.total_src_len)
        .map_err(|_| PDeflateError::NumericOverflow)?)
    .div_ceil(WORKGROUP_SIZE)
    .max(1);
    let groups_x = groups_total.min(65_535);
    let groups_y = groups_total.div_ceil(groups_x).max(1);
    let row_stride = groups_x.saturating_mul(WORKGROUP_SIZE);
    let match_params: [u32; 6] = [
        u32::try_from(packed.total_src_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(packed.max_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(packed.min_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        row_stride,
        u32::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?,
    ];
    let out_len_u64 = u64::try_from(packed.total_src_len)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let required_caps = GpuBatchScratchCaps {
        src_bytes: u64::try_from(packed.src_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        chunk_starts_bytes: u64::try_from(packed.chunk_starts.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_base_bytes: u64::try_from(packed.table_chunk_bases.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_count_bytes: u64::try_from(packed.table_chunk_counts.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_index_offsets_bytes: u64::try_from(packed.table_chunk_index_offsets.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_data_offsets_bytes: u64::try_from(packed.table_chunk_data_offsets.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_meta_bytes: u64::try_from(packed.table_chunk_meta_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_index_bytes: u64::try_from(packed.table_index_words_len)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        prefix_first_bytes: u64::try_from(inputs.len().saturating_mul(PREFIX2_TABLE_SIZE))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_lens_bytes: u64::try_from(packed.total_table_entries)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_offsets_bytes: u64::try_from(packed.total_table_entries)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_data_bytes: u64::try_from(packed.table_data_words_len)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        prep_params_bytes: 4,
        params_bytes: u64::try_from(match_params.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        out_bytes: out_len_u64.max(4),
    };

    let t_scratch = Instant::now();
    let mut scratch = acquire_batch_scratch(r, required_caps)?;
    let scratch_acquire_ms = elapsed_ms(t_scratch);

    struct SectionChunkState {
        slot_index: usize,
        section_count: usize,
        out_lens_bytes: u64,
        out_cmd_bytes: u64,
        section_index_cap_bytes: u64,
        max_tokens_per_section: u32,
        max_cmd_words_per_section: u32,
    }

    let result = (|| -> Result<GpuBatchSectionEncodeOutput, PDeflateError> {
        let t_match_upload = Instant::now();
        r.queue.write_buffer(
            &scratch.src_buffer,
            0,
            bytemuck::cast_slice(&packed.src_words),
        );
        r.queue.write_buffer(
            &scratch.chunk_starts_buffer,
            0,
            bytemuck::cast_slice(&packed.chunk_starts),
        );
        r.queue.write_buffer(
            &scratch.table_base_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_bases),
        );
        r.queue.write_buffer(
            &scratch.table_count_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_counts),
        );
        r.queue.write_buffer(
            &scratch.table_index_offsets_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_index_offsets),
        );
        r.queue.write_buffer(
            &scratch.table_data_offsets_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_data_offsets),
        );
        if !packed.table_chunk_meta_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_meta_buffer,
                0,
                bytemuck::cast_slice(&packed.table_chunk_meta_words),
            );
        }
        if !packed.table_index_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_index_buffer,
                0,
                bytemuck::cast_slice(&packed.table_index_words),
            );
        }
        if !packed.table_data_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_data_buffer,
                0,
                bytemuck::cast_slice(&packed.table_data_words),
            );
        }
        let prep_params: [u32; 1] =
            [u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?];
        r.queue.write_buffer(
            &scratch.prep_params_buffer,
            0,
            bytemuck::cast_slice(&prep_params),
        );
        r.queue.write_buffer(
            &scratch.params_buffer,
            0,
            bytemuck::cast_slice(&match_params),
        );
        let match_upload_ms = elapsed_ms(t_match_upload);
        let queue_probe = queue_probe_enabled();
        let mut pre_match_queue_drain_ms = 0.0_f64;
        let mut pre_match_queue_drain_poll_calls = 0u64;
        let mut match_submit_done_wait_ms = 0.0_f64;
        let mut match_submit_poll_calls = 0u64;
        let mut section_submit_done_wait_ms = 0.0_f64;
        let mut section_submit_poll_calls = 0u64;
        if queue_probe {
            let (wait_ms, poll_calls) = wait_for_queue_done(&r.queue, &r.device);
            pre_match_queue_drain_ms = wait_ms;
            pre_match_queue_drain_poll_calls = poll_calls;
        }

        let mut match_encoder = r
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-pdeflate-match-section-encoder"),
            });
        let t_table_copy = Instant::now();
        encode_table_device_copies(
            &mut match_encoder,
            &scratch,
            &packed.table_device_copies,
            &packed.table_meta_device_copies,
        )?;
        let match_table_copy_ms = elapsed_ms(t_table_copy);
        match_encoder.clear_buffer(&scratch.prefix_first_buffer, 0, None);
        match_encoder.clear_buffer(&scratch.table_lens_buffer, 0, None);
        match_encoder.clear_buffer(&scratch.table_offsets_buffer, 0, None);
        let match_prepare_dispatch_ms = {
            let t_prepare = Instant::now();
            let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-match-prepare-table-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.match_prepare_table_pipeline);
            pass.set_bind_group(0, &scratch.prep_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);
            elapsed_ms(t_prepare)
        };
        let match_kernel_dispatch_ms = {
            let t_match = Instant::now();
            let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-match-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.pipeline);
            pass.set_bind_group(0, &scratch.bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
            drop(pass);
            elapsed_ms(t_match)
        };

        let t_section_total = Instant::now();
        let t_section_setup = Instant::now();
        let mut section_states = Vec::<SectionChunkState>::with_capacity(packed.chunk_lens.len());
        let mut slot_in_use = vec![false; scratch.section_slots.len()];
        for (chunk_idx, &chunk_len) in packed.chunk_lens.iter().enumerate() {
            let src_base_u32 = packed.chunk_starts[chunk_idx];
            let mut section_offsets = vec![0u32; section_count];
            let mut section_caps = vec![0u32; section_count];
            let mut section_token_offsets = vec![0u32; section_count];
            let mut section_token_caps = vec![0u32; section_count];
            let mut total_cmd_cap_bytes: usize = 0;
            let mut total_token_cap: usize = 0;
            let mut max_tokens_per_section: usize = 1;
            let mut max_cmd_words_per_section: usize = 1;
            for sec in 0..section_count {
                let s0 = section_start(sec, section_count, chunk_len);
                let s1 = section_start(sec + 1, section_count, chunk_len);
                let sec_len = s1.saturating_sub(s0);
                let cap_bytes = estimate_section_cmd_cap_bytes(sec_len)?;
                let aligned = align_up4(total_cmd_cap_bytes);
                section_offsets[sec] =
                    u32::try_from(aligned).map_err(|_| PDeflateError::NumericOverflow)?;
                section_caps[sec] =
                    u32::try_from(cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
                total_cmd_cap_bytes = aligned
                    .checked_add(cap_bytes)
                    .ok_or(PDeflateError::NumericOverflow)?;
                section_token_offsets[sec] =
                    u32::try_from(total_token_cap).map_err(|_| PDeflateError::NumericOverflow)?;
                let token_cap = sec_len.max(1);
                section_token_caps[sec] =
                    u32::try_from(token_cap).map_err(|_| PDeflateError::NumericOverflow)?;
                total_token_cap = total_token_cap
                    .checked_add(token_cap)
                    .ok_or(PDeflateError::NumericOverflow)?;
                max_tokens_per_section = max_tokens_per_section.max(token_cap);
                max_cmd_words_per_section =
                    max_cmd_words_per_section.max(align_up4(cap_bytes).div_ceil(4));
            }

            let section_count_u64 =
                u64::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
            let section_params: [u32; 10] = [
                u32::try_from(chunk_len).map_err(|_| PDeflateError::NumericOverflow)?,
                u32::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?,
                u32::try_from(packed.min_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
                u32::try_from(packed.max_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
                u32::try_from(max_cmd_len).map_err(|_| PDeflateError::NumericOverflow)?,
                src_base_u32,
                0,
                0,
                0,
                0,
            ];
            let out_lens_bytes = section_count_u64.saturating_mul(4);
            let out_cmd_words = total_cmd_cap_bytes.div_ceil(4);
            let out_cmd_bytes = u64::try_from(out_cmd_words)
                .map_err(|_| PDeflateError::NumericOverflow)?
                .saturating_mul(4);
            if out_cmd_bytes > r.max_storage_binding_size {
                return Err(PDeflateError::Gpu(format!(
                    "gpu section encode output buffer too large ({} > {})",
                    out_cmd_bytes, r.max_storage_binding_size
                )));
            }

            let slot_index = if let Some((idx, _)) = scratch
                .section_slots
                .iter()
                .enumerate()
                .find(|(idx, slot)| !slot_in_use[*idx] && slot.fits(section_count, out_cmd_bytes))
            {
                idx
            } else {
                let slot = GpuSectionEncodeSlot::new(r, &scratch, section_count, out_cmd_bytes)?;
                scratch.section_slots.push(slot);
                slot_in_use.push(false);
                scratch.section_slots.len() - 1
            };
            slot_in_use[slot_index] = true;
            let slot = &scratch.section_slots[slot_index];
            if u64::try_from(total_token_cap).map_err(|_| PDeflateError::NumericOverflow)?
                > slot.token_cap_count
            {
                return Err(PDeflateError::Gpu(
                    "gpu section encode token staging cap exceeded".to_string(),
                ));
            }
            r.queue.write_buffer(
                &slot.section_offsets_buffer,
                0,
                bytemuck::cast_slice(&section_offsets),
            );
            r.queue.write_buffer(
                &slot.section_caps_buffer,
                0,
                bytemuck::cast_slice(&section_caps),
            );
            r.queue.write_buffer(
                &slot.section_token_offsets_buffer,
                0,
                bytemuck::cast_slice(&section_token_offsets),
            );
            r.queue.write_buffer(
                &slot.section_token_caps_buffer,
                0,
                bytemuck::cast_slice(&section_token_caps),
            );
            r.queue.write_buffer(
                &slot.section_params_buffer,
                0,
                bytemuck::cast_slice(&section_params),
            );

            section_states.push(SectionChunkState {
                slot_index,
                section_count,
                out_lens_bytes,
                out_cmd_bytes,
                section_index_cap_bytes: slot.section_index_cap_bytes,
                max_tokens_per_section: u32::try_from(max_tokens_per_section)
                    .map_err(|_| PDeflateError::NumericOverflow)?,
                max_cmd_words_per_section: u32::try_from(max_cmd_words_per_section)
                    .map_err(|_| PDeflateError::NumericOverflow)?,
            });
        }
        let section_upload_ms = elapsed_ms(t_section_setup);
        let mut section_pass_dispatch_ms = 0.0_f64;
        let mut section_tokenize_dispatch_ms = 0.0_f64;
        let mut section_prefix_dispatch_ms = 0.0_f64;
        let mut section_scatter_dispatch_ms = 0.0_f64;
        let mut section_pack_dispatch_ms = 0.0_f64;
        let mut section_meta_dispatch_ms = 0.0_f64;
        let section_copy_dispatch_ms = 0.0_f64;
        let mut section_tokenize_wait_ms = 0.0_f64;
        let mut section_prefix_wait_ms = 0.0_f64;
        let mut section_scatter_wait_ms = 0.0_f64;
        let mut section_pack_wait_ms = 0.0_f64;
        let mut section_meta_wait_ms = 0.0_f64;
        let mut section_stage_wait_poll_calls = 0u64;
        let (match_submit_ms, section_submit_ms) = if queue_probe {
            let t_match_submit = Instant::now();
            r.queue.submit(Some(match_encoder.finish()));
            let match_submit_ms = elapsed_ms(t_match_submit);
            let (wait_ms, poll_calls) = wait_for_queue_done(&r.queue, &r.device);
            match_submit_done_wait_ms = wait_ms;
            match_submit_poll_calls = poll_calls;

            let mut section_submit_ms = 0.0_f64;
            let mut submit_and_wait = |encoder: wgpu::CommandEncoder,
                                       stage_wait_ms: &mut f64|
             -> f64 {
                let t_submit = Instant::now();
                r.queue.submit(Some(encoder.finish()));
                let submit_ms = elapsed_ms(t_submit);
                let (wait_ms, poll_calls) = wait_for_queue_done(&r.queue, &r.device);
                *stage_wait_ms += wait_ms;
                section_submit_done_wait_ms += wait_ms;
                section_submit_poll_calls = section_submit_poll_calls.saturating_add(poll_calls);
                section_stage_wait_poll_calls =
                    section_stage_wait_poll_calls.saturating_add(poll_calls);
                submit_ms
            };

            {
                let mut encoder =
                    r.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-pdeflate-se-tokenize-encoder"),
                        });
                for state in section_states.iter() {
                    let slot = &scratch.section_slots[state.slot_index];
                    let groups_sections = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .div_ceil(64)
                        .max(1);
                    let t_stage = Instant::now();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-tokenize-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_tokenize_pipeline);
                    pass.set_bind_group(0, &slot.tokenize_bind_group, &[]);
                    pass.dispatch_workgroups(groups_sections, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_stage);
                    section_pass_dispatch_ms += dt;
                    section_tokenize_dispatch_ms += dt;
                }
                section_submit_ms += submit_and_wait(encoder, &mut section_tokenize_wait_ms);
            }
            {
                let mut encoder =
                    r.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-pdeflate-se-prefix-encoder"),
                        });
                for state in section_states.iter() {
                    let slot = &scratch.section_slots[state.slot_index];
                    let groups_sections = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .div_ceil(64)
                        .max(1);
                    let t_stage = Instant::now();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-prefix-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_prefix_pipeline);
                    pass.set_bind_group(0, &slot.prefix_bind_group, &[]);
                    pass.dispatch_workgroups(groups_sections, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_stage);
                    section_pass_dispatch_ms += dt;
                    section_prefix_dispatch_ms += dt;
                }
                section_submit_ms += submit_and_wait(encoder, &mut section_prefix_wait_ms);
            }
            {
                let mut encoder =
                    r.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-pdeflate-se-scatter-encoder"),
                        });
                for state in section_states.iter() {
                    let slot = &scratch.section_slots[state.slot_index];
                    let scatter_groups_x = state.max_tokens_per_section.div_ceil(64).max(1);
                    let scatter_groups_y = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    let t_stage = Instant::now();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-scatter-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_scatter_pipeline);
                    pass.set_bind_group(0, &slot.scatter_bind_group, &[]);
                    pass.dispatch_workgroups(scatter_groups_x, scatter_groups_y, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_stage);
                    section_pass_dispatch_ms += dt;
                    section_scatter_dispatch_ms += dt;
                }
                section_submit_ms += submit_and_wait(encoder, &mut section_scatter_wait_ms);
            }
            {
                let mut encoder =
                    r.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-pdeflate-se-pack-encoder"),
                        });
                for state in section_states.iter() {
                    let slot = &scratch.section_slots[state.slot_index];
                    let pack_groups_x = state.max_cmd_words_per_section.div_ceil(64).max(1);
                    let pack_groups_y = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    let t_stage = Instant::now();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-pack-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_cmd_pack_pipeline);
                    pass.set_bind_group(0, &slot.pack_bind_group, &[]);
                    pass.dispatch_workgroups(pack_groups_x, pack_groups_y, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_stage);
                    section_pass_dispatch_ms += dt;
                    section_pack_dispatch_ms += dt;
                }
                section_submit_ms += submit_and_wait(encoder, &mut section_pack_wait_ms);
            }
            {
                let mut encoder =
                    r.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-pdeflate-se-meta-encoder"),
                        });
                for state in section_states.iter() {
                    let slot = &scratch.section_slots[state.slot_index];
                    let t_stage = Instant::now();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-meta-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_meta_pipeline);
                    pass.set_bind_group(0, &slot.meta_bind_group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_stage);
                    section_pass_dispatch_ms += dt;
                    section_meta_dispatch_ms += dt;
                    encoder.copy_buffer_to_buffer(
                        &slot.section_meta_buffer,
                        0,
                        &slot.readback_meta_buffer,
                        0,
                        16,
                    );
                    encoder.copy_buffer_to_buffer(
                        &slot.out_lens_buffer,
                        0,
                        &slot.out_lens_readback_buffer,
                        0,
                        state.out_lens_bytes.max(4),
                    );
                }
                section_submit_ms += submit_and_wait(encoder, &mut section_meta_wait_ms);
            }

            (match_submit_ms, section_submit_ms)
        } else {
            for state in section_states.iter() {
                let slot = &scratch.section_slots[state.slot_index];
                let groups_sections = u32::try_from(state.section_count)
                    .map_err(|_| PDeflateError::NumericOverflow)?
                    .div_ceil(64)
                    .max(1);
                {
                    let t_pass = Instant::now();
                    let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-tokenize-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_tokenize_pipeline);
                    pass.set_bind_group(0, &slot.tokenize_bind_group, &[]);
                    pass.dispatch_workgroups(groups_sections, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_pass);
                    section_pass_dispatch_ms += dt;
                    section_tokenize_dispatch_ms += dt;
                }
                {
                    let t_prefix = Instant::now();
                    let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-prefix-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_prefix_pipeline);
                    pass.set_bind_group(0, &slot.prefix_bind_group, &[]);
                    pass.dispatch_workgroups(groups_sections, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_prefix);
                    section_pass_dispatch_ms += dt;
                    section_prefix_dispatch_ms += dt;
                }
                {
                    let t_scatter = Instant::now();
                    let scatter_groups_x = state.max_tokens_per_section.div_ceil(64).max(1);
                    let scatter_groups_y = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-scatter-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_scatter_pipeline);
                    pass.set_bind_group(0, &slot.scatter_bind_group, &[]);
                    pass.dispatch_workgroups(scatter_groups_x, scatter_groups_y, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_scatter);
                    section_pass_dispatch_ms += dt;
                    section_scatter_dispatch_ms += dt;
                }
                {
                    let t_pack = Instant::now();
                    let pack_groups_x = state.max_cmd_words_per_section.div_ceil(64).max(1);
                    let pack_groups_y = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-pack-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_cmd_pack_pipeline);
                    pass.set_bind_group(0, &slot.pack_bind_group, &[]);
                    pass.dispatch_workgroups(pack_groups_x, pack_groups_y, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_pack);
                    section_pass_dispatch_ms += dt;
                    section_pack_dispatch_ms += dt;
                }
                {
                    let t_meta = Instant::now();
                    let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-meta-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_meta_pipeline);
                    pass.set_bind_group(0, &slot.meta_bind_group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_meta);
                    section_pass_dispatch_ms += dt;
                    section_meta_dispatch_ms += dt;
                }
                match_encoder.copy_buffer_to_buffer(
                    &slot.section_meta_buffer,
                    0,
                    &slot.readback_meta_buffer,
                    0,
                    16,
                );
                match_encoder.copy_buffer_to_buffer(
                    &slot.out_lens_buffer,
                    0,
                    &slot.out_lens_readback_buffer,
                    0,
                    state.out_lens_bytes.max(4),
                );
            }
            let t_submit = Instant::now();
            r.queue.submit(Some(match_encoder.finish()));
            let submit_ms = elapsed_ms(t_submit);
            let match_weight =
                match_table_copy_ms + match_prepare_dispatch_ms + match_kernel_dispatch_ms;
            let section_weight = section_pass_dispatch_ms + section_copy_dispatch_ms;
            let weight_sum = match_weight + section_weight;
            let match_submit_ms = if weight_sum > 0.0 {
                submit_ms * (match_weight / weight_sum)
            } else {
                submit_ms
            };
            let section_submit_ms = if weight_sum > 0.0 {
                submit_ms * (section_weight / weight_sum)
            } else {
                0.0
            };
            (match_submit_ms, section_submit_ms)
        };
        if queue_probe {
            eprintln!(
                "[cozip_pdeflate][timing][gpu-stage-probe] chunks={} pre_match_queue_drain_ms={:.3} pre_match_queue_drain_polls={} match_submit_ms={:.3} match_submit_done_wait_ms={:.3} match_submit_polls={} section_submit_ms={:.3} section_submit_done_wait_ms={:.3} section_submit_polls={} match_dispatch_ms={:.3} section_dispatch_ms={:.3}",
                inputs.len(),
                pre_match_queue_drain_ms,
                pre_match_queue_drain_poll_calls,
                match_submit_ms,
                match_submit_done_wait_ms,
                match_submit_poll_calls,
                section_submit_ms,
                section_submit_done_wait_ms,
                section_submit_poll_calls,
                match_table_copy_ms + match_prepare_dispatch_ms + match_kernel_dispatch_ms,
                section_pass_dispatch_ms + section_copy_dispatch_ms
            );
            eprintln!(
                "[cozip_pdeflate][timing][gpu-section-stage-probe] chunks={} tokenize_dispatch_ms={:.3} prefix_dispatch_ms={:.3} scatter_dispatch_ms={:.3} pack_dispatch_ms={:.3} meta_dispatch_ms={:.3} tokenize_wait_ms={:.3} prefix_wait_ms={:.3} scatter_wait_ms={:.3} pack_wait_ms={:.3} meta_wait_ms={:.3} section_wait_polls={}",
                inputs.len(),
                section_tokenize_dispatch_ms,
                section_prefix_dispatch_ms,
                section_scatter_dispatch_ms,
                section_pack_dispatch_ms,
                section_meta_dispatch_ms,
                section_tokenize_wait_ms,
                section_prefix_wait_ms,
                section_scatter_wait_ms,
                section_pack_wait_ms,
                section_meta_wait_ms,
                section_stage_wait_poll_calls
            );
        }

        let t_wrap = Instant::now();
        let mut chunks = Vec::with_capacity(section_states.len());
        for state in section_states {
            let slot = &scratch.section_slots[state.slot_index];
            chunks.push(GpuBatchSectionEncodeChunkOutput {
                section_cmd_lens: Vec::new(),
                section_cmd: Vec::new(),
                section_cmd_device: Some(GpuSectionCmdDeviceOutput {
                    section_count: state.section_count,
                    section_offsets_buffer: slot.section_offsets_buffer.clone(),
                    out_lens_buffer: slot.out_lens_buffer.clone(),
                    out_lens_readback_buffer: slot.out_lens_readback_buffer.clone(),
                    section_prefix_buffer: slot.section_prefix_buffer.clone(),
                    section_index_buffer: slot.section_index_buffer.clone(),
                    section_meta_buffer: slot.section_meta_buffer.clone(),
                    section_meta_readback_buffer: slot.readback_meta_buffer.clone(),
                    out_cmd_buffer: slot.out_cmd_buffer.clone(),
                    out_lens_bytes: state.out_lens_bytes,
                    out_cmd_bytes: state.out_cmd_bytes,
                    section_index_cap_bytes: state.section_index_cap_bytes,
                }),
            });
        }
        let section_wrap_ms = elapsed_ms(t_wrap);

        Ok(GpuBatchSectionEncodeOutput {
            chunks,
            match_profile: GpuMatchProfile {
                upload_ms: match_upload_ms,
                wait_ms: 0.0,
                map_copy_ms: 0.0,
                total_ms: elapsed_ms(t_total),
            },
            section_profile: GpuSectionEncodeProfile {
                upload_ms: section_upload_ms,
                wait_ms: 0.0,
                map_copy_ms: 0.0,
                total_ms: elapsed_ms(t_section_total),
            },
            kernel_profile: GpuBatchKernelProfile {
                pack_inputs_ms,
                pack_alloc_setup_ms: pack_profile.alloc_setup_ms,
                pack_resolve_sizes_ms: pack_profile.resolve_sizes_ms,
                pack_resolve_scan_ms: pack_profile.resolve_scan_ms,
                pack_resolve_readback_setup_ms: pack_profile.resolve_readback_setup_ms,
                pack_resolve_submit_ms: pack_profile.resolve_submit_ms,
                pack_resolve_map_wait_ms: pack_profile.resolve_map_wait_ms,
                pack_resolve_parse_ms: pack_profile.resolve_parse_ms,
                pack_src_copy_ms: pack_profile.src_copy_ms,
                pack_metadata_loop_ms: pack_profile.metadata_loop_ms,
                pack_host_copy_ms: pack_profile.host_copy_ms,
                pack_device_copy_plan_ms: pack_profile.device_copy_plan_ms,
                pack_finalize_ms: pack_profile.finalize_ms,
                scratch_acquire_ms,
                match_table_copy_ms,
                match_prepare_dispatch_ms,
                match_kernel_dispatch_ms,
                match_submit_ms,
                section_setup_ms: section_upload_ms,
                section_pass_dispatch_ms,
                section_tokenize_dispatch_ms,
                section_prefix_dispatch_ms,
                section_scatter_dispatch_ms,
                section_pack_dispatch_ms,
                section_meta_dispatch_ms,
                section_copy_dispatch_ms,
                section_submit_ms,
                section_tokenize_wait_ms,
                section_prefix_wait_ms,
                section_scatter_wait_ms,
                section_pack_wait_ms,
                section_meta_wait_ms,
                section_wrap_ms,
                ..GpuBatchKernelProfile::default()
            },
            keepalive: None,
        })
    })();
    match result {
        Ok(mut out) => {
            out.keepalive = Some(GpuBatchSectionKeepalive::new(scratch));
            Ok(out)
        }
        Err(err) => {
            release_batch_scratch(r, scratch);
            Err(err)
        }
    }
}

pub(crate) fn submit_matches_encode_and_pack_sparse_batch(
    inputs: &[GpuMatchInput<'_>],
    section_count: usize,
    max_cmd_len: usize,
) -> Result<PendingGpuSparsePackBatch, PDeflateError> {
    let with_gpu_stage = |stage: &'static str, err: PDeflateError| -> PDeflateError {
        match err {
            PDeflateError::Gpu(msg) => PDeflateError::Gpu(format!("{stage}: {msg}")),
            other => other,
        }
    };
    if inputs.is_empty() {
        return Ok(PendingGpuSparsePackBatch {
            inputs_len: 0,
            compression_mode: super::PDeflateCompressionMode::Speed,
            direct_states: Vec::new(),
            payload_readback_offsets: Vec::new(),
            payload_readback_plan_bytes: Vec::new(),
            result_readback_bytes: 0,
            payload_readback_bytes: 0,
            sparse_prebuild_build_id_min: u64::MAX,
            sparse_prebuild_build_id_max: 0,
            sparse_prebuild_submit_seq_min: u64::MAX,
            sparse_prebuild_submit_seq_max: 0,
            sparse_latest_build_submit_seq: 0,
            sparse_latest_build_id: 0,
            sparse_build_ids: Vec::new(),
            result_rx: mpsc::channel().1,
            payload_rx: mpsc::channel().1,
            result_ready: true,
            payload_ready: true,
            submit_at: Instant::now(),
            pack_alloc_setup_ms: 0.0,
            pack_resolve_sizes_ms: 0.0,
            pack_resolve_scan_ms: 0.0,
            pack_resolve_readback_setup_ms: 0.0,
            pack_resolve_submit_ms: 0.0,
            pack_resolve_map_wait_ms: 0.0,
            pack_resolve_parse_ms: 0.0,
            pack_src_copy_ms: 0.0,
            pack_metadata_loop_ms: 0.0,
            pack_host_copy_ms: 0.0,
            pack_device_copy_plan_ms: 0.0,
            pack_finalize_ms: 0.0,
            match_upload_ms: 0.0,
            match_table_copy_ms: 0.0,
            match_prepare_dispatch_ms: 0.0,
            match_kernel_dispatch_ms: 0.0,
            section_upload_ms: 0.0,
            section_setup_ms: 0.0,
            section_pass_dispatch_ms: 0.0,
            section_tokenize_dispatch_ms: 0.0,
            section_prefix_dispatch_ms: 0.0,
            section_scatter_dispatch_ms: 0.0,
            section_pack_dispatch_ms: 0.0,
            section_meta_dispatch_ms: 0.0,
            pack_inputs_ms: 0.0,
            batch_scratch_acquire_ms: 0.0,
            sparse_scratch_acquire_ms: 0.0,
            sparse_prepare_ms: 0.0,
            table_size_resolve_ms: 0.0,
            pack_bind_group_ms: 0.0,
            encoder_create_ms: 0.0,
            table_stream_copy_ms: 0.0,
            match_clear_ms: 0.0,
            sparse_prepare_dispatch_ms: 0.0,
            sparse_pack_dispatch_ms: 0.0,
            result_readback_setup_ms: 0.0,
            payload_readback_setup_ms: 0.0,
            submit_ms: 0.0,
            phase1_submit_done_wait_ms: 0.0,
            phase1_result_wait_calls: 0,
            phase1_payload_wait_calls: 0,
            phase1_result_ready_hits: 0,
            phase1_payload_ready_hits: 0,
            phase1_yield_calls: 0,
            batch_scratch: None,
            sparse_scratch: None,
        });
    }
    if section_count == 0 {
        return Err(PDeflateError::InvalidOptions(
            "gpu section encode requires section_count > 0",
        ));
    }
    let compression_mode = inputs[0].compression_mode;

    type SparsePackPrepared = IntegratedSparsePackPrepared;
    type SparsePackHostJob = IntegratedSparsePackHostJob;
    type DirectSectionState = IntegratedDirectSectionState;

    struct DirectSectionGpu {
        state_idx: usize,
        bind_group: wgpu::BindGroup,
        tokenize_bind_group: wgpu::BindGroup,
        prefix_bind_group: wgpu::BindGroup,
        scatter_bind_group: wgpu::BindGroup,
        pack_bind_group: wgpu::BindGroup,
        meta_bind_group: wgpu::BindGroup,
        section_offsets_global_buffer: wgpu::Buffer,
        section_token_counts_buffer: wgpu::Buffer,
        section_token_meta_buffer: wgpu::Buffer,
        section_token_pos_buffer: wgpu::Buffer,
        section_token_cmd_offsets_buffer: wgpu::Buffer,
        out_cmd_byte_buffer: wgpu::Buffer,
    }

    let r = runtime().map_err(|err| with_gpu_stage("integrated/runtime", err))?;
    let (packed, pack_profile) =
        pack_match_batch_inputs(inputs).map_err(|err| with_gpu_stage("integrated/pack-inputs", err))?;
    let pack_inputs_ms = pack_profile.total_ms;

    let groups_total = (u32::try_from(packed.total_src_len)
        .map_err(|_| PDeflateError::NumericOverflow)?)
    .div_ceil(WORKGROUP_SIZE)
    .max(1);
    let groups_x = groups_total.min(65_535);
    let groups_y = groups_total.div_ceil(groups_x).max(1);
    let row_stride = groups_x.saturating_mul(WORKGROUP_SIZE);
    let match_params: [u32; 6] = [
        u32::try_from(packed.total_src_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(packed.max_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(packed.min_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        row_stride,
        u32::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?,
    ];
    let out_len_u64 = u64::try_from(packed.total_src_len)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let required_caps = GpuBatchScratchCaps {
        src_bytes: u64::try_from(packed.src_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        chunk_starts_bytes: u64::try_from(packed.chunk_starts.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_base_bytes: u64::try_from(packed.table_chunk_bases.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_count_bytes: u64::try_from(packed.table_chunk_counts.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_index_offsets_bytes: u64::try_from(packed.table_chunk_index_offsets.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_data_offsets_bytes: u64::try_from(packed.table_chunk_data_offsets.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_meta_bytes: u64::try_from(packed.table_chunk_meta_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_index_bytes: u64::try_from(packed.table_index_words_len)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        prefix_first_bytes: u64::try_from(inputs.len().saturating_mul(PREFIX2_TABLE_SIZE))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_lens_bytes: u64::try_from(packed.total_table_entries)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_offsets_bytes: u64::try_from(packed.total_table_entries)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_data_bytes: u64::try_from(packed.table_data_words_len)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        prep_params_bytes: 4,
        params_bytes: u64::try_from(match_params.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        out_bytes: out_len_u64.max(4),
    };

    let t_batch_scratch = Instant::now();
    let batch_scratch = acquire_batch_scratch(r, required_caps)
        .map_err(|err| with_gpu_stage("integrated/acquire-batch-scratch", err))?;
    let batch_scratch_acquire_ms = elapsed_ms(t_batch_scratch);

    let mut uploaded_tables = Vec::<GpuPackedTableDevice>::with_capacity(inputs.len());
    let mut uploaded_table_idx = Vec::<Option<usize>>::with_capacity(inputs.len());
    for input in inputs {
        if input.table_gpu.is_some() {
            uploaded_table_idx.push(None);
        } else {
            let table_index = input.table_index.ok_or_else(|| {
                PDeflateError::Gpu("gpu sparse direct path missing table_index".to_string())
            })?;
            let table_data = input.table_data.ok_or_else(|| {
                PDeflateError::Gpu("gpu sparse direct path missing table_data".to_string())
            })?;
            let gpu_table = upload_packed_table_device(input.table.len(), table_index, table_data)
                .map_err(|err| with_gpu_stage("integrated/upload-packed-table", err))?;
            uploaded_tables.push(gpu_table);
            uploaded_table_idx.push(Some(uploaded_tables.len() - 1));
        }
    }
    let packed_tables: Vec<&GpuPackedTableDevice> = inputs
        .iter()
        .enumerate()
        .map(|(idx, input)| match input.table_gpu {
            Some(table) => table,
            None => &uploaded_tables[uploaded_table_idx[idx].expect("uploaded table missing")],
        })
        .collect();

    let t_sparse_prepare = Instant::now();
    let (resolved_table_sizes, resolve_profile) = resolve_packed_table_sizes_batch(&packed_tables)
        .map_err(|err| with_gpu_stage("integrated/resolve-packed-table-sizes", err))?;
    let table_size_resolve_ms = resolve_profile.total_ms;

    let mut section_meta_total_bytes = 0u64;
    let mut table_index_total_bytes = 0u64;
    let mut table_data_total_bytes = 0u64;
    let mut section_index_total_bytes = 0u64;
    let mut out_lens_total_bytes = 0u64;
    let mut section_prefix_total_bytes = 0u64;
    let mut section_offsets_total_bytes = 0u64;
    let mut out_cmd_total_bytes = 0u64;
    let mut legacy_out_total_bytes = 0u64;
    let mut direct_states = Vec::<DirectSectionState>::with_capacity(inputs.len());

    for (chunk_idx, (input, &(table_count, table_index_len, table_data_len))) in
        inputs.iter().zip(resolved_table_sizes.iter()).enumerate()
    {
        let chunk_len = packed.chunk_lens[chunk_idx];
        let src_base_u32 = packed.chunk_starts[chunk_idx];
        let table_index_offset = PACK_CHUNK_HEADER_SIZE;
        let table_data_stream_offset = table_index_offset
            .checked_add(table_index_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        let section_index_stream_offset = table_data_stream_offset
            .checked_add(table_data_len)
            .ok_or(PDeflateError::NumericOverflow)?;

        let mut section_offsets_local = vec![0u32; section_count];
        let mut section_caps = vec![0u32; section_count];
        let mut section_token_offsets = vec![0u32; section_count];
        let mut section_token_caps = vec![0u32; section_count];
        let mut total_cmd_cap_bytes: usize = 0;
        let mut total_token_cap: usize = 0;
        let mut max_tokens_per_section: usize = 1;
        let mut max_cmd_words_per_section: usize = 1;
        for sec in 0..section_count {
            let s0 = section_start(sec, section_count, chunk_len);
            let s1 = section_start(sec + 1, section_count, chunk_len);
            let sec_len = s1.saturating_sub(s0);
            let cap_bytes = estimate_section_cmd_cap_bytes(sec_len)
                .map_err(|err| with_gpu_stage("integrated/estimate-section-cmd-cap", err))?;
            let aligned = align_up4(total_cmd_cap_bytes);
            section_offsets_local[sec] =
                u32::try_from(aligned).map_err(|_| PDeflateError::NumericOverflow)?;
            section_caps[sec] =
                u32::try_from(cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
            total_cmd_cap_bytes = aligned
                .checked_add(cap_bytes)
                .ok_or(PDeflateError::NumericOverflow)?;
            section_token_offsets[sec] =
                u32::try_from(total_token_cap).map_err(|_| PDeflateError::NumericOverflow)?;
            let token_cap = sec_len.max(1);
            section_token_caps[sec] =
                u32::try_from(token_cap).map_err(|_| PDeflateError::NumericOverflow)?;
            total_token_cap = total_token_cap
                .checked_add(token_cap)
                .ok_or(PDeflateError::NumericOverflow)?;
            max_tokens_per_section = max_tokens_per_section.max(token_cap);
            max_cmd_words_per_section =
                max_cmd_words_per_section.max(align_up4(cap_bytes).div_ceil(4));
        }

        let section_count_u64 =
            u64::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
        let out_lens_bytes = section_count_u64.saturating_mul(4);
        let section_index_cap_bytes = u64::try_from(section_count.saturating_mul(5))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .max(4);
        let out_cmd_words = total_cmd_cap_bytes.div_ceil(4);
        let out_cmd_bytes = u64::try_from(out_cmd_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        if out_cmd_bytes > r.max_storage_binding_size {
            return Err(PDeflateError::Gpu(format!(
                "gpu section encode output buffer too large ({} > {})",
                out_cmd_bytes, r.max_storage_binding_size
            )));
        }

        let prep = SparsePackPrepared {
            chunk_len,
            huffman_encode_enabled: input.huffman_encode_enabled,
            section_count,
            table_data_stream_offset,
            section_index_stream_offset,
            section_index_cap_len: usize::try_from(section_index_cap_bytes)
                .map_err(|_| PDeflateError::NumericOverflow)?,
            section_cmd_cap_len: usize::try_from(out_cmd_bytes.max(4))
                .map_err(|_| PDeflateError::NumericOverflow)?,
            total_len_cap: section_index_stream_offset
                .checked_add(
                    usize::try_from(section_index_cap_bytes)
                        .map_err(|_| PDeflateError::NumericOverflow)?,
                )
                .and_then(|v| v.checked_add(usize::try_from(out_cmd_bytes.max(4)).ok()?))
                .ok_or(PDeflateError::NumericOverflow)?,
            total_bytes_cap: u64::try_from(
                section_index_stream_offset
                    .checked_add(
                        usize::try_from(section_index_cap_bytes)
                            .map_err(|_| PDeflateError::NumericOverflow)?,
                    )
                    .and_then(|v| v.checked_add(usize::try_from(out_cmd_bytes.max(4)).ok()?))
                    .ok_or(PDeflateError::NumericOverflow)?
                    .div_ceil(4),
            )
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4)
            .max(4),
            table_count,
            table_index_len,
            table_data_len,
        };
        let predicted_payload_bytes = estimate_sparse_payload_bytes(
            compression_mode,
            chunk_len,
            section_count,
            table_count,
            table_index_len,
            table_data_len,
            prep.section_index_cap_len,
            prep.section_cmd_cap_len,
        );
        let predicted_payload_class = classify_sparse_payload_bytes(predicted_payload_bytes);
        let slot_cap_bytes = u64::try_from(sparse_payload_class_capacity_bytes(
            predicted_payload_class,
            predicted_payload_bytes,
        ))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .max(4);
        let cap_payload_class = classify_sparse_payload_bytes(
            usize::try_from(slot_cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?,
        );

        let section_meta_words_off = usize::try_from(section_meta_total_bytes / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let table_index_off = align_up4(
            usize::try_from(table_index_total_bytes).map_err(|_| PDeflateError::NumericOverflow)?,
        );
        let table_data_off = align_up4(
            usize::try_from(table_data_total_bytes).map_err(|_| PDeflateError::NumericOverflow)?,
        );
        let section_index_off = usize::try_from(section_index_total_bytes)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let out_lens_words_off = usize::try_from(out_lens_total_bytes / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let section_prefix_words_off = usize::try_from(section_prefix_total_bytes / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let section_offsets_words_off = usize::try_from(section_offsets_total_bytes / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let out_cmd_off =
            usize::try_from(out_cmd_total_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
        let out_base_word = usize::try_from(legacy_out_total_bytes / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let host = SparsePackHostJob {
            section_meta_words_off,
            table_index_off,
            table_data_off,
            section_index_off,
            out_lens_words_off,
            section_prefix_words_off,
            section_offsets_words_off,
            out_cmd_off,
            out_base_word,
        };

        let section_offsets_global: Vec<u32> = section_offsets_local
            .iter()
            .map(|&off| {
                u32::try_from(
                    out_cmd_off
                        .checked_add(
                            usize::try_from(off).map_err(|_| PDeflateError::NumericOverflow)?,
                        )
                        .ok_or(PDeflateError::NumericOverflow)?,
                )
                .map_err(|_| PDeflateError::NumericOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;

        section_meta_total_bytes = section_meta_total_bytes.saturating_add(16);
        table_index_total_bytes = u64::try_from(
            table_index_off
                .checked_add(align_up4(table_index_len))
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .map_err(|_| PDeflateError::NumericOverflow)?;
        table_data_total_bytes = u64::try_from(
            table_data_off
                .checked_add(align_up4(table_data_len))
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .map_err(|_| PDeflateError::NumericOverflow)?;
        section_index_total_bytes =
            section_index_total_bytes.saturating_add(section_index_cap_bytes.max(4));
        out_lens_total_bytes = out_lens_total_bytes.saturating_add(out_lens_bytes.max(4));
        section_prefix_total_bytes =
            section_prefix_total_bytes.saturating_add(out_lens_bytes.max(4));
        section_offsets_total_bytes =
            section_offsets_total_bytes.saturating_add(out_lens_bytes.max(4));
        out_cmd_total_bytes = out_cmd_total_bytes.saturating_add(out_cmd_bytes.max(4));
        legacy_out_total_bytes = legacy_out_total_bytes.saturating_add(prep.total_bytes_cap.max(4));

        direct_states.push(DirectSectionState {
            chunk_len,
            src_base_u32,
            out_lens_bytes,
            out_cmd_bytes,
            max_tokens_per_section: u32::try_from(max_tokens_per_section)
                .map_err(|_| PDeflateError::NumericOverflow)?,
            max_cmd_words_per_section: u32::try_from(max_cmd_words_per_section)
                .map_err(|_| PDeflateError::NumericOverflow)?,
            predicted_payload_bytes,
            predicted_payload_class,
            slot_cap_bytes,
            cap_payload_class,
            prep,
            host,
            section_offsets_local,
            section_offsets_global,
            section_caps,
            section_token_offsets,
            section_token_caps,
        });
        let _ = input;
    }

    let mut class_counts = [0usize; GPU_SPARSE_PAYLOAD_CLASS_COUNT];
    for state in &direct_states {
        class_counts[state.predicted_payload_class.idx()] =
            class_counts[state.predicted_payload_class.idx()].saturating_add(1);
    }
    let mut class_region_offsets = [0u64; GPU_SPARSE_PAYLOAD_CLASS_COUNT];
    let mut class_region_bytes = [0u64; GPU_SPARSE_PAYLOAD_CLASS_COUNT];
    let mut class_running_base = 0u64;
    for class_id in GPU_SPARSE_PAYLOAD_CLASS_ORDER {
        let idx = class_id.idx();
        class_region_offsets[idx] = class_running_base;
        class_region_bytes[idx] = direct_states
            .iter()
            .filter(|state| state.predicted_payload_class == class_id)
            .fold(0u64, |acc, state| acc.saturating_add(state.slot_cap_bytes));
        class_running_base = class_running_base.saturating_add(class_region_bytes[idx]);
    }
    let mut class_slot_indices = [0usize; GPU_SPARSE_PAYLOAD_CLASS_COUNT];
    let mut class_slot_offsets = class_region_offsets;
    for state in &mut direct_states {
        let class_idx = state.predicted_payload_class.idx();
        let slot_off = class_slot_offsets[class_idx];
        state.host.out_base_word =
            usize::try_from(slot_off / 4).map_err(|_| PDeflateError::NumericOverflow)?;
        state.prep.total_bytes_cap = state.slot_cap_bytes.max(4);
        class_slot_indices[class_idx] = class_slot_indices[class_idx].saturating_add(1);
        class_slot_offsets[class_idx] =
            class_slot_offsets[class_idx].saturating_add(state.slot_cap_bytes);
    }
    let out_total_bytes = class_running_base.max(4);
    if profile_enabled() {
        eprintln!(
            "[cozip_pdeflate][timing][sparse-class-layout] chunks={} class_counts={} class_region_mib=tiny:{:.2},small:{:.2},medium:{:.2},large:{:.2},xlarge:{:.2} out_total_mib={:.2} legacy_out_total_mib={:.2} reduction_pct={:.1}",
            inputs.len(),
            format_sparse_payload_class_counts(&[
                class_counts[0] as u32,
                class_counts[1] as u32,
                class_counts[2] as u32,
                class_counts[3] as u32,
                class_counts[4] as u32,
            ]),
            (class_region_bytes[GpuSparsePayloadClass::Tiny.idx()] as f64) / (1024.0 * 1024.0),
            (class_region_bytes[GpuSparsePayloadClass::Small.idx()] as f64) / (1024.0 * 1024.0),
            (class_region_bytes[GpuSparsePayloadClass::Medium.idx()] as f64) / (1024.0 * 1024.0),
            (class_region_bytes[GpuSparsePayloadClass::Large.idx()] as f64) / (1024.0 * 1024.0),
            (class_region_bytes[GpuSparsePayloadClass::XLarge.idx()] as f64) / (1024.0 * 1024.0),
            (out_total_bytes as f64) / (1024.0 * 1024.0),
            (legacy_out_total_bytes as f64) / (1024.0 * 1024.0),
            if legacy_out_total_bytes > 0 {
                100.0 * (1.0 - (out_total_bytes as f64 / legacy_out_total_bytes as f64))
            } else {
                0.0
            },
        );
    }

    let t_sparse_scratch = Instant::now();
    let scratch = acquire_sparse_pack_scratch(
        r,
        GpuSparsePackScratchCaps {
            desc_bytes: u64::try_from(
                4usize.saturating_add(
                    inputs
                        .len()
                        .saturating_mul(GPU_SPARSE_PACK_BATCH_DESC_WORDS * 4),
                ),
            )
            .map_err(|_| PDeflateError::NumericOverflow)?,
            section_meta_bytes: section_meta_total_bytes.max(4),
            table_index_bytes: table_index_total_bytes.max(4),
            table_data_bytes: table_data_total_bytes.max(4),
            section_index_bytes: section_index_total_bytes.max(4),
            out_lens_bytes: out_lens_total_bytes.max(4),
            section_prefix_bytes: section_prefix_total_bytes.max(4),
            section_offsets_bytes: section_offsets_total_bytes.max(4),
            out_cmd_bytes: out_cmd_total_bytes.max(4),
            result_bytes: u64::try_from(
                inputs
                    .len()
                    .saturating_mul(GPU_SPARSE_PACK_BATCH_RESULT_WORDS * 4),
            )
            .map_err(|_| PDeflateError::NumericOverflow)?
            .max(4),
            out_bytes: out_total_bytes.max(4),
        },
    )
    .map_err(|err| with_gpu_stage("integrated/acquire-sparse-pack-scratch", err))?;
    let sparse_scratch_acquire_ms = elapsed_ms(t_sparse_scratch);

    let sparse_probe = sparse_probe_enabled();
    let sparse_kernel_ts_probe = sparse_kernel_ts_probe_enabled() && r.supports_timestamp_query;
    let sparse_lens_wait_probe = sparse_lens_wait_probe_enabled();
    let sparse_wait_attr_probe = sparse_wait_attribution_probe_enabled();
    let gpu_call_unaccounted_probe = gpu_call_unaccounted_probe_enabled();
    let mut sparse_prebuild_build_id_min = u64::MAX;
    let mut sparse_prebuild_build_id_max = 0u64;
    let mut sparse_prebuild_submit_seq_min = u64::MAX;
    let mut sparse_prebuild_submit_seq_max = 0u64;
    let mut sparse_latest_build_submit: Option<(u64, u64, wgpu::SubmissionIndex)> = None;
    let mut sparse_build_ids = Vec::<u64>::with_capacity(inputs.len());
    for table in &packed_tables {
        if table.build_id != 0 {
            sparse_prebuild_build_id_min = sparse_prebuild_build_id_min.min(table.build_id);
            sparse_prebuild_build_id_max = sparse_prebuild_build_id_max.max(table.build_id);
            sparse_build_ids.push(table.build_id);
        }
        if table.build_submit_seq != 0 {
            sparse_prebuild_submit_seq_min =
                sparse_prebuild_submit_seq_min.min(table.build_submit_seq);
            sparse_prebuild_submit_seq_max =
                sparse_prebuild_submit_seq_max.max(table.build_submit_seq);
        }
        if let Some(submit_index) = table.build_submit_index.clone() {
            let should_replace = sparse_latest_build_submit
                .as_ref()
                .map(|(seq, _, _)| table.build_submit_seq > *seq)
                .unwrap_or(true);
            if should_replace {
                sparse_latest_build_submit =
                    Some((table.build_submit_seq, table.build_id, submit_index));
            }
        }
    }

    let mut desc_words =
        vec![0u32; 1usize.saturating_add(inputs.len() * GPU_SPARSE_PACK_BATCH_DESC_WORDS)];
    desc_words[0] = u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?;
    for (idx, state) in direct_states.iter().enumerate() {
        let base = 1 + idx * GPU_SPARSE_PACK_BATCH_DESC_WORDS;
        let prep = &state.prep;
        let host = &state.host;
        desc_words[base] =
            u32::try_from(prep.section_count).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 1] =
            u32::try_from(prep.chunk_len).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 2] =
            u32::try_from(prep.table_count).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 3] =
            u32::try_from(host.table_index_off).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 4] =
            u32::try_from(prep.table_index_len).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 5] =
            u32::try_from(host.table_data_off).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 6] =
            u32::try_from(prep.table_data_len).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 7] = u32::try_from(host.section_meta_words_off)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 8] =
            u32::try_from(host.section_index_off).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 9] = u32::try_from(prep.section_index_cap_len)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 10] =
            u32::try_from(host.out_lens_words_off).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 11] = u32::try_from(host.section_prefix_words_off)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 12] = u32::try_from(host.section_offsets_words_off)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 13] =
            u32::try_from(host.out_cmd_off).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 14] =
            u32::try_from(prep.section_cmd_cap_len).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 15] =
            u32::try_from(host.out_base_word).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 16] =
            u32::try_from(prep.total_bytes_cap / 4).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 17] = if sparse_probe { 1 } else { 0 };
        desc_words[base + 18] = u32::try_from(prep.table_data_stream_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 19] = u32::try_from(prep.section_index_stream_offset)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 20] = u32::try_from(state.predicted_payload_class.idx())
            .map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 21] =
            u32::try_from(host.out_base_word).map_err(|_| PDeflateError::NumericOverflow)?;
        desc_words[base + 22] = u32::try_from(prep.total_bytes_cap / 4)
            .map_err(|_| PDeflateError::NumericOverflow)?;
    }

    let t_pack_bind_group = Instant::now();
    let pack_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-batch-bg"),
        layout: &r.pack_sparse_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scratch.desc_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scratch.result_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scratch.table_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: scratch.table_data_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: scratch.section_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: scratch.out_lens_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: scratch.section_prefix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: scratch.section_offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: scratch.out_cmd_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: scratch.out_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: r.pack_sparse_stats_zero_buffer.as_entire_binding(),
            },
        ],
    });
    let pack_bind_group_ms = elapsed_ms(t_pack_bind_group);

    let t_section_setup = Instant::now();
    let mut section_upload_ms = 0.0_f64;
    let mut direct_gpu = Vec::<DirectSectionGpu>::with_capacity(direct_states.len());
    for (idx, state) in direct_states.iter().enumerate() {
        let section_count_u64 =
            u64::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
        let section_offsets_bytes = section_count_u64.saturating_mul(4);
        let section_caps_bytes = section_count_u64.saturating_mul(4);
        let section_params_bytes = 40u64;
        let token_buf_bytes = u64::try_from(
            usize::try_from(state.max_tokens_per_section)
                .map_err(|_| PDeflateError::NumericOverflow)?
                .saturating_mul(section_count)
                .saturating_mul(4),
        )
        .map_err(|_| PDeflateError::NumericOverflow)?
        .max(4);
        let out_cmd_byte_buf_bytes = state.out_cmd_bytes.max(4).saturating_mul(4);

        let section_offsets_buffer = storage_upload_buffer(
            &r.device,
            "cozip-pdeflate-se-direct-offs",
            section_offsets_bytes.max(4),
        );
        let section_caps_buffer = storage_upload_buffer(
            &r.device,
            "cozip-pdeflate-se-direct-caps",
            section_caps_bytes.max(4),
        );
        let section_token_offsets_buffer = storage_upload_buffer(
            &r.device,
            "cozip-pdeflate-se-direct-token-offs",
            section_offsets_bytes.max(4),
        );
        let section_token_caps_buffer = storage_upload_buffer(
            &r.device,
            "cozip-pdeflate-se-direct-token-caps",
            section_caps_bytes.max(4),
        );
        let section_params_buffer = storage_upload_buffer(
            &r.device,
            "cozip-pdeflate-se-direct-params",
            section_params_bytes.max(4),
        );
        let section_offsets_global_buffer = storage_upload_buffer(
            &r.device,
            "cozip-pdeflate-se-direct-offs-global",
            section_offsets_bytes.max(4),
        );
        let section_token_counts_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-direct-token-counts"),
            size: state.out_lens_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let section_token_meta_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-direct-token-meta"),
            size: token_buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let section_token_pos_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-direct-token-pos"),
            size: token_buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let section_token_cmd_offsets_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-direct-token-cmd-offs"),
            size: token_buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_cmd_byte_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-direct-out-cmd-byte"),
            size: out_cmd_byte_buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let section_params: [u32; 10] = [
            u32::try_from(state.chunk_len).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(packed.min_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(packed.max_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(max_cmd_len).map_err(|_| PDeflateError::NumericOverflow)?,
            state.src_base_u32,
            u32::try_from(state.host.out_lens_words_off)
                .map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(state.host.section_prefix_words_off)
                .map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(state.host.section_index_off)
                .map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(state.host.section_meta_words_off)
                .map_err(|_| PDeflateError::NumericOverflow)?,
        ];

        let t_upload = Instant::now();
        r.queue.write_buffer(
            &section_offsets_buffer,
            0,
            bytemuck::cast_slice(&state.section_offsets_local),
        );
        r.queue.write_buffer(
            &section_caps_buffer,
            0,
            bytemuck::cast_slice(&state.section_caps),
        );
        r.queue.write_buffer(
            &section_token_offsets_buffer,
            0,
            bytemuck::cast_slice(&state.section_token_offsets),
        );
        r.queue.write_buffer(
            &section_token_caps_buffer,
            0,
            bytemuck::cast_slice(&state.section_token_caps),
        );
        r.queue.write_buffer(
            &section_offsets_global_buffer,
            0,
            bytemuck::cast_slice(&state.section_offsets_global),
        );
        r.queue.write_buffer(
            &section_params_buffer,
            0,
            bytemuck::cast_slice(&section_params),
        );
        r.queue.write_buffer(
            &scratch.section_offsets_buffer,
            u64::try_from(state.host.section_offsets_words_off)
                .map_err(|_| PDeflateError::NumericOverflow)?
                .saturating_mul(4),
            bytemuck::cast_slice(&state.section_offsets_local),
        );
        section_upload_ms += elapsed_ms(t_upload);

        let bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-se-direct-bg"),
            layout: &r.section_encode_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: batch_scratch.src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: batch_scratch.out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: section_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: section_caps_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: section_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scratch.out_lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scratch.out_cmd_buffer.as_entire_binding(),
                },
            ],
        });
        let tokenize_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-se-direct-tokenize-bg"),
            layout: &r.section_tokenize_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: batch_scratch.src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: batch_scratch.out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: section_token_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: section_token_caps_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: section_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: section_token_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: section_token_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: section_token_pos_buffer.as_entire_binding(),
                },
            ],
        });
        let prefix_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-se-direct-prefix-bg"),
            layout: &r.section_prefix_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: section_token_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: section_token_caps_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: section_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: section_token_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: section_token_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: section_token_cmd_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scratch.out_lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: section_caps_buffer.as_entire_binding(),
                },
            ],
        });
        let scatter_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-se-direct-scatter-bg"),
            layout: &r.section_scatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: batch_scratch.src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: section_token_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: section_token_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: section_token_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: section_token_pos_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: section_token_cmd_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: section_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: section_caps_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: section_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: out_cmd_byte_buffer.as_entire_binding(),
                },
            ],
        });
        let pack_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-se-direct-pack-bg"),
            layout: &r.section_cmd_pack_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_cmd_byte_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: section_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scratch.out_lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: section_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scratch.out_cmd_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: section_offsets_global_buffer.as_entire_binding(),
                },
            ],
        });
        let meta_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-se-direct-meta-bg"),
            layout: &r.section_meta_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scratch.out_lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scratch.section_prefix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scratch.section_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch.section_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: section_params_buffer.as_entire_binding(),
                },
            ],
        });

        direct_gpu.push(DirectSectionGpu {
            state_idx: idx,
            bind_group,
            tokenize_bind_group,
            prefix_bind_group,
            scatter_bind_group,
            pack_bind_group,
            meta_bind_group,
            section_offsets_global_buffer,
            section_token_counts_buffer,
            section_token_meta_buffer,
            section_token_pos_buffer,
            section_token_cmd_offsets_buffer,
            out_cmd_byte_buffer,
        });
    }
    let section_setup_ms = elapsed_ms(t_section_setup);
    let sparse_prepare_total_ms = elapsed_ms(t_sparse_prepare);
    let sparse_prepare_ms = (sparse_prepare_total_ms - sparse_scratch_acquire_ms).max(0.0);
    let mut batch_scratch_slot = Some(batch_scratch);
    let mut scratch_slot = Some(scratch);

    let result = (|| -> Result<PendingGpuSparsePackBatch, PDeflateError> {
        let batch_scratch = batch_scratch_slot
            .take()
            .ok_or_else(|| PDeflateError::Gpu("integrated/submit: batch scratch missing".to_string()))?;
        let scratch = scratch_slot
            .take()
            .ok_or_else(|| PDeflateError::Gpu("integrated/submit: sparse scratch missing".to_string()))?;
        let validate_integrated_section_state =
            |state_idx: usize, state: &DirectSectionState| -> Result<(), PDeflateError> {
                let readback_lens_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cozip-pdeflate-integrated-section-lens-readback"),
                    size: state.out_lens_bytes.max(4),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let readback_cmd_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cozip-pdeflate-integrated-section-cmd-readback"),
                    size: state.out_cmd_bytes.max(4),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let mut encoder =
                    r.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-pdeflate-integrated-section-validate-encoder"),
                        });
                encoder.copy_buffer_to_buffer(
                    &scratch.out_lens_buffer,
                    u64::try_from(state.host.out_lens_words_off)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .saturating_mul(4),
                    &readback_lens_buffer,
                    0,
                    state.out_lens_bytes.max(4),
                );
                encoder.copy_buffer_to_buffer(
                    &scratch.out_cmd_buffer,
                    u64::try_from(state.host.out_cmd_off)
                        .map_err(|_| PDeflateError::NumericOverflow)?,
                    &readback_cmd_buffer,
                    0,
                    state.out_cmd_bytes.max(4),
                );
                let submission = r.queue.submit(Some(encoder.finish()));

                let lens_slice = readback_lens_buffer.slice(..state.out_lens_bytes.max(4));
                let cmd_slice = readback_cmd_buffer.slice(..state.out_cmd_bytes.max(4));
                let (tx_lens, rx_lens) = mpsc::channel();
                lens_slice.map_async(wgpu::MapMode::Read, move |res| {
                    let _ = tx_lens.send(res);
                });
                let (tx_cmd, rx_cmd) = mpsc::channel();
                cmd_slice.map_async(wgpu::MapMode::Read, move |res| {
                    let _ = tx_cmd.send(res);
                });
                r.device.poll(wgpu::Maintain::wait_for(submission));
                rx_lens
                    .recv()
                    .map_err(|_| {
                        PDeflateError::Gpu(
                            "integrated/validate-section-cmds: lens map channel closed"
                                .to_string(),
                        )
                    })?
                    .map_err(|e| {
                        PDeflateError::Gpu(format!(
                            "integrated/validate-section-cmds: lens map failed: {e}"
                        ))
                    })?;
                rx_cmd
                    .recv()
                    .map_err(|_| {
                        PDeflateError::Gpu(
                            "integrated/validate-section-cmds: cmd map channel closed"
                                .to_string(),
                        )
                    })?
                    .map_err(|e| {
                        PDeflateError::Gpu(format!(
                            "integrated/validate-section-cmds: cmd map failed: {e}"
                        ))
                    })?;

                let lens_mapped = lens_slice.get_mapped_range();
                let cmd_mapped = cmd_slice.get_mapped_range();
                let lens_u32: &[u32] = bytemuck::cast_slice(&lens_mapped);
                let cmd_bytes: &[u8] = &cmd_mapped;
                let mut section_cmd_lens = vec![0u32; state.prep.section_count];
                section_cmd_lens.copy_from_slice(&lens_u32[..state.prep.section_count]);
                let mut section_cmd =
                    Vec::with_capacity(usize::try_from(state.out_cmd_bytes).unwrap_or(0));
                for sec in 0..state.prep.section_count {
                    let off = usize::try_from(state.section_offsets_global[sec])
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .checked_sub(state.host.out_cmd_off)
                        .ok_or_else(|| {
                            PDeflateError::Gpu(format!(
                                "integrated/validate-section-cmds: section {} offset underflow",
                                sec
                            ))
                        })?;
                    let len = usize::try_from(section_cmd_lens[sec])
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    let cap = usize::try_from(state.section_caps[sec])
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    if len > cap {
                        return Err(PDeflateError::Gpu(format!(
                            "integrated/validate-section-cmds: section {} len {} exceeds cap {}",
                            sec, len, cap
                        )));
                    }
                    let end = off.checked_add(len).ok_or(PDeflateError::NumericOverflow)?;
                    if end > cmd_bytes.len() {
                        return Err(PDeflateError::Gpu(format!(
                            "integrated/validate-section-cmds: section {} range {}..{} exceeds cmd bytes {}",
                            sec,
                            off,
                            end,
                            cmd_bytes.len()
                        )));
                    }
                    section_cmd.extend_from_slice(&cmd_bytes[off..end]);
                }
                drop(lens_mapped);
                drop(cmd_mapped);
                readback_lens_buffer.unmap();
                readback_cmd_buffer.unmap();
                validate_section_commands(
                    state.chunk_len,
                    state.prep.section_count,
                    &section_cmd_lens,
                    &section_cmd,
                )
                .inspect_err(|_| {
                    if GPU_INTEGRATED_SECTION_PROBE_LOGGED
                        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                        .is_err()
                    {
                        return;
                    }
                    let probe_sec = 32usize.min(state.prep.section_count.saturating_sub(1));
                    let gpu_state = &direct_gpu[state_idx];
                    let map_read = |buffer: &wgpu::Buffer,
                                    offset: u64,
                                    size: u64,
                                    stage: &'static str|
                     -> Result<Vec<u8>, PDeflateError> {
                        let copy_size = size.max(4);
                        let readback = r.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("cozip-pdeflate-integrated-section-probe-readback"),
                            size: copy_size,
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        let mut encoder =
                            r.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some(
                                        "cozip-pdeflate-integrated-section-probe-encoder",
                                    ),
                                });
                        encoder.copy_buffer_to_buffer(buffer, offset, &readback, 0, copy_size);
                        let submission = r.queue.submit(Some(encoder.finish()));
                        let slice = readback.slice(..copy_size);
                        let (tx, rx) = mpsc::channel();
                        slice.map_async(wgpu::MapMode::Read, move |res| {
                            let _ = tx.send(res);
                        });
                        r.device.poll(wgpu::Maintain::wait_for(submission));
                        rx.recv()
                            .map_err(|_| {
                                PDeflateError::Gpu(format!(
                                    "integrated/section-probe-{stage}: map channel closed"
                                ))
                            })
                            .and_then(|res| {
                                res.map_err(|e| {
                                    PDeflateError::Gpu(format!(
                                        "integrated/section-probe-{stage}: map failed: {e}"
                                    ))
                                })
                            })?;
                        let mapped = slice.get_mapped_range();
                        let out = mapped.to_vec();
                        drop(mapped);
                        readback.unmap();
                        Ok(out)
                    };
                    let token_count_bytes = match map_read(
                        &gpu_state.section_token_counts_buffer,
                        u64::try_from(probe_sec).unwrap_or(0).saturating_mul(4),
                        4,
                        "token-count",
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            eprintln!(
                                "[cozip_pdeflate][timing][gpu-integrated-section-probe] sec={} probe_error={}",
                                probe_sec, err
                            );
                            return;
                        }
                    };
                    let token_count =
                        bytemuck::cast_slice::<u8, u32>(&token_count_bytes)[0] as usize;
                    let token_base = usize::try_from(state.section_token_offsets[probe_sec])
                        .unwrap_or(0);
                    let first_token_index = token_base;
                    let first_token_meta = if token_count > 0 {
                        map_read(
                            &gpu_state.section_token_meta_buffer,
                            u64::try_from(first_token_index).unwrap_or(0).saturating_mul(4),
                            4,
                            "token-meta",
                        )
                        .ok()
                        .map(|v| bytemuck::cast_slice::<u8, u32>(&v)[0])
                        .unwrap_or(0)
                    } else {
                        0
                    };
                    let first_token_pos = if token_count > 0 {
                        map_read(
                            &gpu_state.section_token_pos_buffer,
                            u64::try_from(first_token_index).unwrap_or(0).saturating_mul(4),
                            4,
                            "token-pos",
                        )
                        .ok()
                        .map(|v| bytemuck::cast_slice::<u8, u32>(&v)[0])
                        .unwrap_or(0)
                    } else {
                        0
                    };
                    let first_token_cmd_off = if token_count > 0 {
                        map_read(
                            &gpu_state.section_token_cmd_offsets_buffer,
                            u64::try_from(first_token_index).unwrap_or(0).saturating_mul(4),
                            4,
                            "token-cmd-off",
                        )
                        .ok()
                        .map(|v| bytemuck::cast_slice::<u8, u32>(&v)[0])
                        .unwrap_or(0)
                    } else {
                        0
                    };
                    let out_len = section_cmd_lens[probe_sec];
                    let sec_cap = state.section_caps[probe_sec];
                    let local_off = state.section_offsets_local[probe_sec];
                    let global_off = state.section_offsets_global[probe_sec];
                    let local_probe_len = u64::from(out_len.max(8).min(sec_cap.max(8)).min(16));
                    let local_bytes = map_read(
                        &gpu_state.out_cmd_byte_buffer,
                        u64::from(local_off).saturating_mul(4),
                        local_probe_len.saturating_mul(4),
                        "local-bytes",
                    )
                    .unwrap_or_default();
                    let packed_bytes = map_read(
                        &scratch.out_cmd_buffer,
                        u64::from(global_off),
                        local_probe_len,
                        "packed-bytes",
                    )
                    .unwrap_or_default();
                    let fmt_hex = |bytes: &[u8], stride4: bool| {
                        let logical: Vec<u8> = if stride4 {
                            bytes.chunks_exact(4).map(|w| w[0]).collect()
                        } else {
                            bytes.to_vec()
                        };
                        logical.iter()
                            .map(|b| format!("{b:02x}"))
                            .collect::<Vec<_>>()
                            .join("")
                    };
                    eprintln!(
                        "[cozip_pdeflate][timing][gpu-integrated-section-probe] sec={} token_count={} first_token_meta={} first_token_pos={} first_token_cmd_off={} out_len={} sec_cap={} local_off={} global_off={} local_bytes={} packed_bytes={}",
                        probe_sec,
                        token_count,
                        first_token_meta,
                        first_token_pos,
                        first_token_cmd_off,
                        out_len,
                        sec_cap,
                        local_off,
                        global_off,
                        fmt_hex(&local_bytes, true),
                        fmt_hex(&packed_bytes, false),
                    );
                })
            };
        let mut encoder_create_ms = 0.0_f64;
        let mut table_stream_copy_ms = 0.0_f64;
        let mut match_clear_ms = 0.0_f64;
        let mut sparse_prepare_dispatch_ms = 0.0_f64;
        let mut sparse_pack_dispatch_ms = 0.0_f64;
        let mut result_readback_setup_ms = 0.0_f64;
        let mut payload_readback_setup_ms = 0.0_f64;
        let t_match_upload = Instant::now();
        r.queue.write_buffer(
            &batch_scratch.src_buffer,
            0,
            bytemuck::cast_slice(&packed.src_words),
        );
        r.queue.write_buffer(
            &batch_scratch.chunk_starts_buffer,
            0,
            bytemuck::cast_slice(&packed.chunk_starts),
        );
        r.queue.write_buffer(
            &batch_scratch.table_base_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_bases),
        );
        r.queue.write_buffer(
            &batch_scratch.table_count_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_counts),
        );
        r.queue.write_buffer(
            &batch_scratch.table_index_offsets_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_index_offsets),
        );
        r.queue.write_buffer(
            &batch_scratch.table_data_offsets_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_data_offsets),
        );
        if !packed.table_chunk_meta_words.is_empty() {
            r.queue.write_buffer(
                &batch_scratch.table_meta_buffer,
                0,
                bytemuck::cast_slice(&packed.table_chunk_meta_words),
            );
        }
        if !packed.table_index_words.is_empty() {
            r.queue.write_buffer(
                &batch_scratch.table_index_buffer,
                0,
                bytemuck::cast_slice(&packed.table_index_words),
            );
        }
        if !packed.table_data_words.is_empty() {
            r.queue.write_buffer(
                &batch_scratch.table_data_buffer,
                0,
                bytemuck::cast_slice(&packed.table_data_words),
            );
        }
        let prep_params: [u32; 1] =
            [u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?];
        r.queue.write_buffer(
            &batch_scratch.prep_params_buffer,
            0,
            bytemuck::cast_slice(&prep_params),
        );
        r.queue.write_buffer(
            &batch_scratch.params_buffer,
            0,
            bytemuck::cast_slice(&match_params),
        );
        r.queue
            .write_buffer(&scratch.desc_buffer, 0, bytemuck::cast_slice(&desc_words));
        let match_upload_ms = elapsed_ms(t_match_upload);

        let t_encoder_create = Instant::now();
        let mut encoder = r
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-pdeflate-match-section-sparse-direct-encoder"),
            });
        encoder_create_ms += elapsed_ms(t_encoder_create);
        let t_table_copy = Instant::now();
        encode_table_device_copies(
            &mut encoder,
            &batch_scratch,
            &packed.table_device_copies,
            &packed.table_meta_device_copies,
        )?;
        let match_table_copy_ms = elapsed_ms(t_table_copy);
        let t_match_clear = Instant::now();
        encoder.clear_buffer(&batch_scratch.prefix_first_buffer, 0, None);
        encoder.clear_buffer(&batch_scratch.table_lens_buffer, 0, None);
        encoder.clear_buffer(&batch_scratch.table_offsets_buffer, 0, None);
        match_clear_ms = elapsed_ms(t_match_clear);

        let match_prepare_dispatch_ms = {
            let t_prepare = Instant::now();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-match-prepare-table-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.match_prepare_table_pipeline);
            pass.set_bind_group(0, &batch_scratch.prep_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);
            elapsed_ms(t_prepare)
        };
        let match_kernel_dispatch_ms = {
            let t_match = Instant::now();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-match-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.pipeline);
            pass.set_bind_group(0, &batch_scratch.bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
            drop(pass);
            elapsed_ms(t_match)
        };

        let t_table_stream_copy = Instant::now();
        for (idx, state) in direct_states.iter().enumerate() {
            let table = packed_tables[idx];
            if state.prep.table_index_len > 0 {
                encoder.copy_buffer_to_buffer(
                    &table.table_index_buffer,
                    0,
                    &scratch.table_index_buffer,
                    u64::try_from(state.host.table_index_off)
                        .map_err(|_| PDeflateError::NumericOverflow)?,
                    u64::try_from(align_up4(state.prep.table_index_len))
                        .map_err(|_| PDeflateError::NumericOverflow)?,
                );
            }
            if state.prep.table_data_len > 0 {
                encoder.copy_buffer_to_buffer(
                    &table.table_data_buffer,
                    0,
                    &scratch.table_data_buffer,
                    u64::try_from(state.host.table_data_off)
                        .map_err(|_| PDeflateError::NumericOverflow)?,
                    u64::try_from(align_up4(state.prep.table_data_len))
                        .map_err(|_| PDeflateError::NumericOverflow)?,
                );
            }
        }
        table_stream_copy_ms = elapsed_ms(t_table_stream_copy);

        let mut section_pass_dispatch_ms = 0.0_f64;
        let mut section_tokenize_dispatch_ms = 0.0_f64;
        let mut section_prefix_dispatch_ms = 0.0_f64;
        let mut section_scatter_dispatch_ms = 0.0_f64;
        let mut section_pack_dispatch_ms = 0.0_f64;
        let mut section_meta_dispatch_ms = 0.0_f64;

        for gpu_state in &direct_gpu {
            let state = &direct_states[gpu_state.state_idx];
            let groups_sections = u32::try_from(section_count)
                .map_err(|_| PDeflateError::NumericOverflow)?
                .div_ceil(64)
                .max(1);
            {
                let t_stage = Instant::now();
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-pdeflate-se-direct-tokenize-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&r.section_tokenize_pipeline);
                pass.set_bind_group(0, &gpu_state.tokenize_bind_group, &[]);
                pass.dispatch_workgroups(groups_sections, 1, 1);
                drop(pass);
                let dt = elapsed_ms(t_stage);
                section_pass_dispatch_ms += dt;
                section_tokenize_dispatch_ms += dt;
            }
            {
                let t_stage = Instant::now();
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-pdeflate-se-direct-prefix-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&r.section_prefix_pipeline);
                pass.set_bind_group(0, &gpu_state.prefix_bind_group, &[]);
                pass.dispatch_workgroups(groups_sections, 1, 1);
                drop(pass);
                let dt = elapsed_ms(t_stage);
                section_pass_dispatch_ms += dt;
                section_prefix_dispatch_ms += dt;
            }
            {
                let t_stage = Instant::now();
                let scatter_groups_x = state.max_tokens_per_section.div_ceil(64).max(1);
                let scatter_groups_y =
                    u32::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-pdeflate-se-direct-scatter-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&r.section_scatter_pipeline);
                pass.set_bind_group(0, &gpu_state.scatter_bind_group, &[]);
                pass.dispatch_workgroups(scatter_groups_x, scatter_groups_y, 1);
                drop(pass);
                let dt = elapsed_ms(t_stage);
                section_pass_dispatch_ms += dt;
                section_scatter_dispatch_ms += dt;
            }
            {
                let t_stage = Instant::now();
                let pack_groups_x = state.max_cmd_words_per_section.div_ceil(64).max(1);
                let pack_groups_y =
                    u32::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-pdeflate-se-direct-pack-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&r.section_cmd_pack_pipeline);
                pass.set_bind_group(0, &gpu_state.pack_bind_group, &[]);
                pass.dispatch_workgroups(pack_groups_x, pack_groups_y, 1);
                drop(pass);
                let dt = elapsed_ms(t_stage);
                section_pass_dispatch_ms += dt;
                section_pack_dispatch_ms += dt;
            }
            {
                let t_stage = Instant::now();
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-pdeflate-se-direct-meta-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&r.section_meta_pipeline);
                pass.set_bind_group(0, &gpu_state.meta_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                drop(pass);
                let dt = elapsed_ms(t_stage);
                section_pass_dispatch_ms += dt;
                section_meta_dispatch_ms += dt;
            }
        }

        let result_readback_bytes = u64::try_from(
            inputs
                .len()
                .saturating_mul(GPU_SPARSE_PACK_BATCH_RESULT_WORDS * 4),
        )
        .map_err(|_| PDeflateError::NumericOverflow)?
        .max(4);
        let payload_readback_plan_bytes: Vec<u64> = direct_states
            .iter()
            .map(|state| {
                planned_sparse_payload_readback_bytes(
                    state.predicted_payload_bytes,
                    state.slot_cap_bytes,
                )
            })
            .collect();
        let mut payload_readback_offsets = Vec::<u64>::with_capacity(direct_states.len());
        let mut payload_readback_bytes = 0u64;
        for &copy_bytes in &payload_readback_plan_bytes {
            payload_readback_offsets.push(payload_readback_bytes);
            payload_readback_bytes = payload_readback_bytes
                .checked_add(copy_bytes.max(4))
                .ok_or(PDeflateError::NumericOverflow)?;
        }
        let payload_readback_bytes = payload_readback_bytes.max(4);
        let t_payload_readback_setup = Instant::now();
        let payload_readback_buffer = &scratch.payload_readback_buffer;
        payload_readback_setup_ms = elapsed_ms(t_payload_readback_setup);
        let max_total_words_cap = direct_states.iter().fold(1u32, |acc, state| {
            acc.max(u32::try_from(state.prep.total_bytes_cap / 4).unwrap_or(u32::MAX))
        });
        {
            let t_stage = Instant::now();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-pack-sparse-prepare-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.pack_sparse_prepare_pipeline);
            pass.set_bind_group(0, &scratch.prepare_bind_group, &[]);
            let groups_x = u32::try_from(inputs.len().div_ceil(64))
                .map_err(|_| PDeflateError::NumericOverflow)?
                .max(1);
            pass.dispatch_workgroups(groups_x, 1, 1);
            drop(pass);
            sparse_prepare_dispatch_ms = elapsed_ms(t_stage);
        }
        encoder.copy_buffer_to_buffer(
            &scratch.result_buffer,
            0,
            &scratch.result_readback_buffer,
            0,
            result_readback_bytes,
        );
        {
            let t_stage = Instant::now();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-pack-sparse-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.pack_sparse_pipeline);
            pass.set_bind_group(0, &pack_bind_group, &[]);
            pass.dispatch_workgroups(
                max_total_words_cap.div_ceil(256).max(1),
                u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?,
                1,
            );
            drop(pass);
            sparse_pack_dispatch_ms = elapsed_ms(t_stage);
        }
        for ((state, &dst_off), &copy_bytes) in direct_states
            .iter()
            .zip(payload_readback_offsets.iter())
            .zip(payload_readback_plan_bytes.iter())
        {
            encoder.copy_buffer_to_buffer(
                &scratch.out_buffer,
                u64::try_from(state.host.out_base_word)
                    .map_err(|_| PDeflateError::NumericOverflow)?
                    .saturating_mul(4),
                payload_readback_buffer,
                dst_off,
                copy_bytes.max(4),
            );
        }

        let t_submit = Instant::now();
        r.queue.submit(Some(encoder.finish()));
        let submit_ms = elapsed_ms(t_submit);

        let t_result_readback_setup = Instant::now();
        let result_slice = scratch.result_readback_buffer.slice(..result_readback_bytes);
        let (result_tx, result_rx) = mpsc::channel();
        result_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = result_tx.send(res);
        });
        result_readback_setup_ms = elapsed_ms(t_result_readback_setup);
        let payload_slice = payload_readback_buffer.slice(..payload_readback_bytes);
        let (out_tx, out_rx) = mpsc::channel();
        payload_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = out_tx.send(res);
        });
        let _ = sparse_wait_attr_probe;
        let (sparse_latest_build_submit_seq, sparse_latest_build_id) = sparse_latest_build_submit
            .as_ref()
            .map(|(submit_seq, build_id, _)| (*submit_seq, *build_id))
            .unwrap_or((0, 0));
        Ok(PendingGpuSparsePackBatch {
            inputs_len: inputs.len(),
            compression_mode,
            direct_states,
            payload_readback_offsets,
            payload_readback_plan_bytes,
            result_readback_bytes,
            payload_readback_bytes,
            sparse_prebuild_build_id_min,
            sparse_prebuild_build_id_max,
            sparse_prebuild_submit_seq_min,
            sparse_prebuild_submit_seq_max,
            sparse_latest_build_submit_seq,
            sparse_latest_build_id,
            sparse_build_ids,
            result_rx,
            payload_rx: out_rx,
            result_ready: false,
            payload_ready: false,
            submit_at: Instant::now(),
            pack_alloc_setup_ms: pack_profile.alloc_setup_ms,
            pack_resolve_sizes_ms: pack_profile.resolve_sizes_ms,
            pack_resolve_scan_ms: pack_profile.resolve_scan_ms,
            pack_resolve_readback_setup_ms: pack_profile.resolve_readback_setup_ms,
            pack_resolve_submit_ms: pack_profile.resolve_submit_ms,
            pack_resolve_map_wait_ms: pack_profile.resolve_map_wait_ms,
            pack_resolve_parse_ms: pack_profile.resolve_parse_ms,
            pack_src_copy_ms: pack_profile.src_copy_ms,
            pack_metadata_loop_ms: pack_profile.metadata_loop_ms,
            pack_host_copy_ms: pack_profile.host_copy_ms,
            pack_device_copy_plan_ms: pack_profile.device_copy_plan_ms,
            pack_finalize_ms: pack_profile.finalize_ms,
            match_upload_ms,
            match_table_copy_ms,
            match_prepare_dispatch_ms,
            match_kernel_dispatch_ms,
            section_upload_ms,
            section_setup_ms,
            section_pass_dispatch_ms,
            section_tokenize_dispatch_ms,
            section_prefix_dispatch_ms,
            section_scatter_dispatch_ms,
            section_pack_dispatch_ms,
            section_meta_dispatch_ms,
            pack_inputs_ms,
            batch_scratch_acquire_ms,
            sparse_scratch_acquire_ms,
            sparse_prepare_ms,
            table_size_resolve_ms,
            pack_bind_group_ms,
            encoder_create_ms,
            table_stream_copy_ms,
            match_clear_ms,
            sparse_prepare_dispatch_ms,
            sparse_pack_dispatch_ms,
            result_readback_setup_ms,
            payload_readback_setup_ms,
            submit_ms,
            phase1_submit_done_wait_ms: 0.0,
            phase1_result_wait_calls: 0,
            phase1_payload_wait_calls: 0,
            phase1_result_ready_hits: 0,
            phase1_payload_ready_hits: 0,
            phase1_yield_calls: 0,
            batch_scratch: Some(batch_scratch),
            sparse_scratch: Some(scratch),
        })
    })();

    match result {
        Ok(pending) => Ok(pending),
        Err(err) => {
            if let Some(scratch) = scratch_slot.take() {
                release_sparse_pack_scratch(r, scratch);
            }
            if let Some(batch_scratch) = batch_scratch_slot.take() {
                release_batch_scratch(r, batch_scratch);
            }
            Err(err)
        }
    }
}

fn release_pending_gpu_sparse_pack_batch(
    runtime: &GpuMatchRuntime,
    pending: &mut PendingGpuSparsePackBatch,
) {
    if let Some(scratch) = pending.sparse_scratch.take() {
        release_sparse_pack_scratch(runtime, scratch);
    }
    if let Some(scratch) = pending.batch_scratch.take() {
        release_batch_scratch(runtime, scratch);
    }
}

fn update_pending_map_ready(
    pending: &mut PendingGpuSparsePackBatch,
) -> Result<(), PDeflateError> {
    if !pending.result_ready {
        match pending.result_rx.try_recv() {
            Ok(Ok(())) => {
                pending.result_ready = true;
                pending.phase1_result_ready_hits =
                    pending.phase1_result_ready_hits.saturating_add(1);
            }
            Ok(Err(e)) => {
                return Err(PDeflateError::Gpu(format!(
                    "integrated/result-readback-map: map_async failed: {e}"
                )));
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(PDeflateError::Gpu(
                    "integrated/result-readback-map: channel closed while waiting".to_string(),
                ));
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
    if !pending.payload_ready {
        match pending.payload_rx.try_recv() {
            Ok(Ok(())) => {
                pending.payload_ready = true;
                pending.phase1_payload_ready_hits =
                    pending.phase1_payload_ready_hits.saturating_add(1);
            }
            Ok(Err(e)) => {
                return Err(PDeflateError::Gpu(format!(
                    "integrated/payload-readback-map: map_async failed: {e}"
                )));
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(PDeflateError::Gpu(
                    "integrated/payload-readback-map: channel closed while waiting".to_string(),
                ));
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
    Ok(())
}

fn collect_pending_gpu_sparse_pack_batch(
    pending: &mut PendingGpuSparsePackBatch,
    block: bool,
) -> Result<Option<GpuBatchSparsePackOutput>, PDeflateError> {
    if pending.inputs_len == 0 {
        return Ok(Some(GpuBatchSparsePackOutput {
            chunks: Vec::new(),
            match_profile: GpuMatchProfile::default(),
            section_profile: GpuSectionEncodeProfile::default(),
            sparse_profile: GpuMatchProfile::default(),
            sparse_batch_profile: GpuSparsePackBatchProfile::default(),
            kernel_profile: GpuBatchKernelProfile::default(),
        }));
    }

    let r = runtime()?;
    r.device.poll(wgpu::Maintain::Poll);
    update_pending_map_ready(pending)?;
    if !(pending.result_ready && pending.payload_ready) {
        if !block {
            return Ok(None);
        }
        let spin_polls_before_wait = sparse_ready_first_spin_polls();
        let yield_rounds_before_wait = sparse_ready_first_yield_rounds();
        let mut yield_rounds = 0usize;
        while !(pending.result_ready && pending.payload_ready) {
            for _ in 0..spin_polls_before_wait {
                r.device.poll(wgpu::Maintain::Poll);
                update_pending_map_ready(pending)?;
                if pending.result_ready && pending.payload_ready {
                    break;
                }
            }
            if pending.result_ready && pending.payload_ready {
                break;
            }
            if yield_rounds < yield_rounds_before_wait {
                std::thread::yield_now();
                pending.phase1_yield_calls = pending.phase1_yield_calls.saturating_add(1);
                yield_rounds = yield_rounds.saturating_add(1);
                continue;
            }
            if !pending.result_ready {
                pending.phase1_result_wait_calls =
                    pending.phase1_result_wait_calls.saturating_add(1);
            }
            if !pending.payload_ready {
                pending.phase1_payload_wait_calls =
                    pending.phase1_payload_wait_calls.saturating_add(1);
            }
            let t_wait = Instant::now();
            r.device.poll(wgpu::Maintain::Wait);
            pending.phase1_submit_done_wait_ms += elapsed_ms(t_wait);
            yield_rounds = 0;
            update_pending_map_ready(pending)?;
        }
    }

    let Some(sparse_scratch) = pending.sparse_scratch.as_mut() else {
        return Err(PDeflateError::Gpu(
            "integrated/collect: sparse scratch missing".to_string(),
        ));
    };
    let Some(_batch_scratch) = pending.batch_scratch.as_ref() else {
        return Err(PDeflateError::Gpu(
            "integrated/collect: batch scratch missing".to_string(),
        ));
    };

    let result_slice = sparse_scratch
        .result_readback_buffer
        .slice(..pending.result_readback_bytes.max(4));
    let payload_slice = sparse_scratch
        .payload_readback_buffer
        .slice(..pending.payload_readback_bytes.max(4));

    let t_lens_copy = Instant::now();
    let (payload_lens, payload_table_counts, payload_copy_bytes) = {
        let result_mapped = result_slice.get_mapped_range();
        let result_words: &[u32] = bytemuck::cast_slice(&result_mapped);
        let parse_result = (|| -> Result<_, PDeflateError> {
            let mut payload_lens = Vec::<usize>::with_capacity(pending.inputs_len);
            let mut payload_table_counts = Vec::<usize>::with_capacity(pending.inputs_len);
            let mut payload_copy_bytes = Vec::<u64>::with_capacity(pending.inputs_len);
            for (idx, state) in pending.direct_states.iter().enumerate() {
                let base = idx * GPU_SPARSE_PACK_BATCH_RESULT_WORDS;
                let total_len_u32 = *result_words.get(base + 2).unwrap_or(&0xffff_ffff);
                let total_words_u32 = *result_words.get(base + 3).unwrap_or(&0);
                if total_len_u32 == 0xffff_ffff || total_words_u32 == 0 {
                    return Err(PDeflateError::Gpu(
                        "integrated/result-parse: sparse size prepare overflow while packing"
                            .to_string(),
                    ));
                }
                let total_len =
                    usize::try_from(total_len_u32).map_err(|_| PDeflateError::NumericOverflow)?;
                let total_copy_len = align_up4(total_len);
                let total_copy_bytes = u64::try_from(total_copy_len)
                    .map_err(|_| PDeflateError::NumericOverflow)?;
                if total_copy_bytes > state.prep.total_bytes_cap {
                    return Err(PDeflateError::Gpu(
                        "gpu sparse pack produced oversized aligned payload".to_string(),
                    ));
                }
                if total_copy_bytes > pending.payload_readback_plan_bytes[idx] {
                    return Err(PDeflateError::Gpu(format!(
                        "integrated/result-parse: planned payload readback too small idx={} planned={} actual={}",
                        idx,
                        pending.payload_readback_plan_bytes[idx],
                        total_copy_bytes,
                    )));
                }
                payload_copy_bytes.push(total_copy_bytes);
                payload_lens.push(total_len);
                payload_table_counts.push(state.prep.table_count);
            }
            Ok((payload_lens, payload_table_counts, payload_copy_bytes))
        })();
        drop(result_mapped);
        sparse_scratch.result_readback_buffer.unmap();
        parse_result?
    };
    let lens_copy_ms = elapsed_ms(t_lens_copy);

    let t_copy = Instant::now();
    let (
        out,
        predicted_payload_bytes_total,
        cap_payload_bytes_total,
        actual_payload_bytes_total,
        predicted_class_counts,
        cap_class_counts,
        actual_class_counts,
        predicted_underestimate_chunks,
        predicted_underestimate_bytes,
        predicted_overestimate_bytes,
    ) = {
        let payload_mapped = payload_slice.get_mapped_range();
        let payload_bytes: &[u8] = &payload_mapped;
        let copy_result = (|| -> Result<_, PDeflateError> {
            let mut out = Vec::<GpuSparsePackChunkOutput>::with_capacity(pending.inputs_len);
            let mut predicted_payload_bytes_total = 0u64;
            let mut cap_payload_bytes_total = 0u64;
            let mut actual_payload_bytes_total = 0u64;
            let mut predicted_class_counts = [0u32; GPU_SPARSE_PAYLOAD_CLASS_COUNT];
            let mut cap_class_counts = [0u32; GPU_SPARSE_PAYLOAD_CLASS_COUNT];
            let mut actual_class_counts = [0u32; GPU_SPARSE_PAYLOAD_CLASS_COUNT];
            let mut predicted_underestimate_chunks = 0u32;
            let mut predicted_underestimate_bytes = 0u64;
            let mut predicted_overestimate_bytes = 0u64;
            for (idx, state) in pending.direct_states.iter().enumerate() {
                let chunk_base = usize::try_from(pending.payload_readback_offsets[idx])
                    .map_err(|_| PDeflateError::NumericOverflow)?;
                let chunk_end = chunk_base
                    .checked_add(
                        usize::try_from(payload_copy_bytes[idx])
                            .map_err(|_| PDeflateError::NumericOverflow)?,
                    )
                    .ok_or(PDeflateError::NumericOverflow)?;
                let mapped = payload_bytes.get(chunk_base..chunk_end).ok_or_else(|| {
                    PDeflateError::Gpu(format!(
                        "integrated/payload-readback-copy: payload batch readback truncated idx={} chunk_base={} chunk_copy_len={} payload_len={} payload_readback_bytes={} mapped_len={} out_base_word={}",
                        idx,
                        chunk_base,
                        payload_copy_bytes[idx],
                        payload_lens[idx],
                        pending.payload_readback_bytes,
                        payload_bytes.len(),
                        state.host.out_base_word,
                    ))
                })?;
                let payload_len = payload_lens[idx];
                let actual_payload_class = classify_sparse_payload_bytes(payload_len);
                predicted_payload_bytes_total = predicted_payload_bytes_total
                    .saturating_add(u64::try_from(state.predicted_payload_bytes).unwrap_or(u64::MAX));
                cap_payload_bytes_total =
                    cap_payload_bytes_total.saturating_add(state.prep.total_bytes_cap);
                actual_payload_bytes_total = actual_payload_bytes_total
                    .saturating_add(u64::try_from(payload_len).unwrap_or(u64::MAX));
                predicted_class_counts[state.predicted_payload_class.idx()] =
                    predicted_class_counts[state.predicted_payload_class.idx()].saturating_add(1);
                cap_class_counts[state.cap_payload_class.idx()] =
                    cap_class_counts[state.cap_payload_class.idx()].saturating_add(1);
                actual_class_counts[actual_payload_class.idx()] =
                    actual_class_counts[actual_payload_class.idx()].saturating_add(1);
                if state.predicted_payload_bytes < payload_len {
                    predicted_underestimate_chunks =
                        predicted_underestimate_chunks.saturating_add(1);
                    predicted_underestimate_bytes = predicted_underestimate_bytes.saturating_add(
                        u64::try_from(payload_len - state.predicted_payload_bytes)
                            .unwrap_or(u64::MAX),
                    );
                } else {
                    predicted_overestimate_bytes = predicted_overestimate_bytes.saturating_add(
                        u64::try_from(state.predicted_payload_bytes - payload_len)
                            .unwrap_or(u64::MAX),
                    );
                }
                let mut payload = vec![0u8; payload_len];
                payload.copy_from_slice(&mapped[..payload_len]);
                finalize_sparse_packed_chunk_huffman(
                    &mut payload,
                    state.prep.huffman_encode_enabled,
                )?;
                if gpu_sparse_payload_validate_enabled() {
                    let mut validate_out = vec![0u8; state.chunk_len];
                    super::decompress_chunk_into(&payload, &mut validate_out).map_err(|err| {
                        PDeflateError::Gpu(format!(
                            "integrated/validate-final-payload: {err}"
                        ))
                    })?;
                }
                out.push(GpuSparsePackChunkOutput {
                    payload,
                    table_count: payload_table_counts[idx],
                });
            }
            Ok((
                out,
                predicted_payload_bytes_total,
                cap_payload_bytes_total,
                actual_payload_bytes_total,
                predicted_class_counts,
                cap_class_counts,
                actual_class_counts,
                predicted_underestimate_chunks,
                predicted_underestimate_bytes,
                predicted_overestimate_bytes,
            ))
        })();
        drop(payload_mapped);
        sparse_scratch.payload_readback_buffer.unmap();
        copy_result?
    };
    let sparse_copy_ms = elapsed_ms(t_copy);

    let match_stage_weight =
        pending.match_table_copy_ms + pending.match_prepare_dispatch_ms + pending.match_kernel_dispatch_ms;
    let section_stage_weight = pending.section_setup_ms + pending.section_pass_dispatch_ms;
    let sparse_stage_weight = pending.sparse_prepare_dispatch_ms + pending.sparse_pack_dispatch_ms;
    let stage_weight_sum = match_stage_weight + section_stage_weight + sparse_stage_weight;
    let match_submit_ms = if stage_weight_sum > 0.0 {
        pending.submit_ms * (match_stage_weight / stage_weight_sum)
    } else {
        pending.submit_ms
    };
    let section_submit_ms = if stage_weight_sum > 0.0 {
        pending.submit_ms * (section_stage_weight / stage_weight_sum)
    } else {
        0.0
    };
    let sparse_submit_ms = if stage_weight_sum > 0.0 {
        pending.submit_ms * (sparse_stage_weight / stage_weight_sum)
    } else {
        0.0
    };
    let sparse_wait_ms = pending.phase1_submit_done_wait_ms;
    let sparse_total_ms = pending.sparse_prepare_ms
        + pending.sparse_scratch_acquire_ms
        + sparse_submit_ms
        + sparse_wait_ms
        + lens_copy_ms
        + sparse_copy_ms;
    let sparse_profile = GpuMatchProfile {
        upload_ms: 0.0,
        wait_ms: sparse_wait_ms,
        map_copy_ms: lens_copy_ms + sparse_copy_ms,
        total_ms: sparse_total_ms,
    };
    let sparse_batch_profile = GpuSparsePackBatchProfile {
        chunks: pending.inputs_len,
        lens_bytes_total: pending
            .direct_states
            .iter()
            .fold(0u64, |acc, s| acc.saturating_add(s.out_lens_bytes.max(4))),
        out_cmd_bytes_total: pending
            .direct_states
            .iter()
            .fold(0u64, |acc, s| acc.saturating_add(s.out_cmd_bytes.max(4))),
        predicted_payload_bytes_total,
        cap_payload_bytes_total,
        actual_payload_bytes_total,
        predicted_class_counts,
        cap_class_counts,
        actual_class_counts,
        predicted_underestimate_chunks,
        predicted_underestimate_bytes,
        predicted_overestimate_bytes,
        lens_submit_ms: sparse_submit_ms,
        lens_submit_done_wait_ms: sparse_wait_ms,
        lens_map_after_done_ms: 0.0,
        lens_poll_calls: 0,
        lens_yield_calls: pending.phase1_yield_calls,
        lens_wait_ms: sparse_wait_ms,
        lens_copy_ms,
        prepare_ms: pending.sparse_prepare_ms,
        table_size_resolve_ms: pending.table_size_resolve_ms,
        prepare_misc_ms: (pending.sparse_prepare_ms - pending.table_size_resolve_ms).max(0.0),
        scratch_acquire_ms: pending.sparse_scratch_acquire_ms,
        upload_dispatch_ms: 0.0,
        submit_ms: sparse_submit_ms,
        wait_ms: sparse_wait_ms,
        copy_ms: sparse_copy_ms,
        total_ms: sparse_total_ms,
    };

    let sparse_kernel_ts_probe = sparse_kernel_ts_probe_enabled() && r.supports_timestamp_query;
    if sparse_lens_wait_probe_enabled() {
        let seq = SPARSE_LENS_WAIT_PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
        if table_stage_probe_should_log(seq) {
            let readback_bytes =
                pending.result_readback_bytes.saturating_add(pending.payload_readback_bytes);
            let build_id_range = if pending.sparse_prebuild_build_id_min != u64::MAX {
                format!(
                    "{}..{}",
                    pending.sparse_prebuild_build_id_min, pending.sparse_prebuild_build_id_max
                )
            } else {
                "none".to_string()
            };
            let submit_seq_range = if pending.sparse_prebuild_submit_seq_min != u64::MAX {
                format!(
                    "{}..{}",
                    pending.sparse_prebuild_submit_seq_min, pending.sparse_prebuild_submit_seq_max
                )
            } else {
                "none".to_string()
            };
            let build_ids = if pending.sparse_build_ids.is_empty() {
                "none".to_string()
            } else {
                pending
                    .sparse_build_ids
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            eprintln!(
                "[cozip_pdeflate][timing][sparse-lens-wait-breakdown] seq={} chunks={} pending_maps={} readback_kib={:.1} payload_readback_kib={:.1} build_id_range={} build_submit_seq_range={} build_ids={} latest_build_submit_seq={} latest_build_id={} sparse_submit_seq=0 sparse_submit_seq_last=0 sparse_submit_count=1 payload_submit_seq=0 pre_wait_latest_build_submit_ms=0.000 submit_done_wait_sparse_ms={:.3} submit_done_wait_payload_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_sparse_ms=0.000 map_callback_wait_payload_ms=0.000 map_callback_wait_ms=0.000 sparse_wait_calls={} payload_wait_calls={} sparse_wait_max_ms={:.3} payload_wait_max_ms={:.3} sparse_wait_ready={} payload_wait_ready={} sparse_ts_prepare_kernel_ms=0.000 sparse_ts_kernel_ms=0.000 sparse_ts_kernel_total_ms=0.000 sparse_ts_parse_ms=0.000 sparse_ts_queue_stall_est_ms=0.000 sparse_ts_status={} lens_wait_ms={:.3}",
                seq,
                pending.inputs_len,
                2,
                (readback_bytes as f64) / 1024.0,
                (pending.payload_readback_bytes as f64) / 1024.0,
                build_id_range,
                submit_seq_range,
                build_ids,
                pending.sparse_latest_build_submit_seq,
                pending.sparse_latest_build_id,
                pending.phase1_submit_done_wait_ms,
                0.0,
                sparse_wait_ms,
                pending.phase1_result_wait_calls,
                pending.phase1_payload_wait_calls,
                0.0,
                0.0,
                pending.phase1_result_ready_hits,
                pending.phase1_payload_ready_hits,
                if sparse_kernel_ts_probe { "on" } else { "off" },
                sparse_wait_ms,
            );
        }
    }
    if profile_enabled() {
        let chunks_f = (pending.inputs_len as f64).max(1.0);
        eprintln!(
            "[cozip_pdeflate][timing][sparse-class-breakdown] chunks={} mode={:?} predicted_hist={} actual_hist={} cap_hist={} predicted_avg_kib={:.1} actual_avg_kib={:.1} cap_avg_kib={:.1} predicted_under_chunks={} predicted_under_mib={:.3} predicted_over_mib={:.3}",
            pending.inputs_len,
            pending.compression_mode,
            format_sparse_payload_class_counts(&predicted_class_counts),
            format_sparse_payload_class_counts(&actual_class_counts),
            format_sparse_payload_class_counts(&cap_class_counts),
            (predicted_payload_bytes_total as f64) / chunks_f / 1024.0,
            (actual_payload_bytes_total as f64) / chunks_f / 1024.0,
            (cap_payload_bytes_total as f64) / chunks_f / 1024.0,
            predicted_underestimate_chunks,
            (predicted_underestimate_bytes as f64) / (1024.0 * 1024.0),
            (predicted_overestimate_bytes as f64) / (1024.0 * 1024.0),
        );
    }
    if gpu_call_unaccounted_probe_enabled() {
        let seq = GPU_CALL_UNACCOUNTED_SEQ.fetch_add(1, Ordering::Relaxed);
        if table_stage_probe_should_log(seq) {
            let host_unaccounted_ms = pending.encoder_create_ms
                + pending.pack_bind_group_ms
                + pending.table_stream_copy_ms
                + pending.match_clear_ms
                + pending.sparse_prepare_dispatch_ms
                + pending.sparse_pack_dispatch_ms
                + pending.result_readback_setup_ms
                + pending.payload_readback_setup_ms;
            eprintln!(
                "[cozip_pdeflate][timing][gpu-call-unaccounted-breakdown] seq={} chunks={} pack_bind_group_ms={:.3} encoder_create_ms={:.3} table_stream_copy_ms={:.3} match_clear_ms={:.3} sparse_prepare_dispatch_ms={:.3} sparse_pack_dispatch_ms={:.3} result_readback_setup_ms={:.3} payload_readback_setup_ms={:.3} host_unaccounted_ms={:.3} profile_call_ms={:.3} profile_total_ms={:.3} call_minus_profile_total_ms={:.3}",
                seq,
                pending.inputs_len,
                pending.pack_bind_group_ms,
                pending.encoder_create_ms,
                pending.table_stream_copy_ms,
                pending.match_clear_ms,
                pending.sparse_prepare_dispatch_ms,
                pending.sparse_pack_dispatch_ms,
                pending.result_readback_setup_ms,
                pending.payload_readback_setup_ms,
                host_unaccounted_ms,
                pending.match_upload_ms
                    + pending.match_table_copy_ms
                    + pending.match_prepare_dispatch_ms
                    + pending.match_kernel_dispatch_ms
                    + pending.section_upload_ms
                    + pending.section_setup_ms
                    + pending.section_pass_dispatch_ms
                    + sparse_total_ms,
                pending.match_upload_ms + pending.section_upload_ms + sparse_profile.total_ms,
                (pending.match_upload_ms
                    + pending.match_table_copy_ms
                    + pending.match_prepare_dispatch_ms
                    + pending.match_kernel_dispatch_ms
                    + pending.section_upload_ms
                    + pending.section_setup_ms
                    + pending.section_pass_dispatch_ms
                    + sparse_total_ms
                    - (pending.match_upload_ms + pending.section_upload_ms + sparse_profile.total_ms))
                    .max(0.0),
            );
        }
    }

    let out_result = GpuBatchSparsePackOutput {
        chunks: out,
        match_profile: GpuMatchProfile {
            upload_ms: pending.match_upload_ms,
            wait_ms: 0.0,
            map_copy_ms: 0.0,
            total_ms: pending.match_upload_ms
                + pending.match_table_copy_ms
                + pending.match_prepare_dispatch_ms
                + pending.match_kernel_dispatch_ms
                + match_submit_ms,
        },
        section_profile: GpuSectionEncodeProfile {
            upload_ms: pending.section_upload_ms,
            wait_ms: 0.0,
            map_copy_ms: 0.0,
            total_ms: pending.section_upload_ms
                + pending.section_setup_ms
                + pending.section_pass_dispatch_ms
                + section_submit_ms
                + pending.section_meta_dispatch_ms,
        },
        sparse_profile,
        sparse_batch_profile,
        kernel_profile: GpuBatchKernelProfile {
            pack_inputs_ms: pending.pack_inputs_ms,
            pack_alloc_setup_ms: pending.pack_alloc_setup_ms,
            pack_resolve_sizes_ms: pending.pack_resolve_sizes_ms,
            pack_resolve_scan_ms: pending.pack_resolve_scan_ms,
            pack_resolve_readback_setup_ms: pending.pack_resolve_readback_setup_ms,
            pack_resolve_submit_ms: pending.pack_resolve_submit_ms,
            pack_resolve_map_wait_ms: pending.pack_resolve_map_wait_ms,
            pack_resolve_parse_ms: pending.pack_resolve_parse_ms,
            pack_src_copy_ms: pending.pack_src_copy_ms,
            pack_metadata_loop_ms: pending.pack_metadata_loop_ms,
            pack_host_copy_ms: pending.pack_host_copy_ms,
            pack_device_copy_plan_ms: pending.pack_device_copy_plan_ms,
            pack_finalize_ms: pending.pack_finalize_ms,
            scratch_acquire_ms: pending.batch_scratch_acquire_ms,
            match_table_copy_ms: pending.match_table_copy_ms,
            match_prepare_dispatch_ms: pending.match_prepare_dispatch_ms,
            match_kernel_dispatch_ms: pending.match_kernel_dispatch_ms,
            match_submit_ms,
            section_setup_ms: pending.section_setup_ms,
            section_pass_dispatch_ms: pending.section_pass_dispatch_ms,
            section_tokenize_dispatch_ms: pending.section_tokenize_dispatch_ms,
            section_prefix_dispatch_ms: pending.section_prefix_dispatch_ms,
            section_scatter_dispatch_ms: pending.section_scatter_dispatch_ms,
            section_pack_dispatch_ms: pending.section_pack_dispatch_ms,
            section_meta_dispatch_ms: pending.section_meta_dispatch_ms,
            section_submit_ms,
            sparse_prepare_ms: pending.sparse_prepare_ms,
            sparse_scratch_acquire_ms: pending.sparse_scratch_acquire_ms,
            sparse_submit_ms,
            sparse_lens_submit_ms: sparse_submit_ms,
            sparse_lens_submit_done_wait_ms: sparse_wait_ms,
            sparse_lens_map_after_done_ms: 0.0,
            sparse_lens_poll_calls: 0,
            sparse_lens_yield_calls: pending.phase1_yield_calls,
            sparse_lens_wait_ms: sparse_wait_ms,
            sparse_lens_copy_ms: lens_copy_ms,
            sparse_wait_ms,
            sparse_copy_ms,
            sparse_total_ms,
            call_pack_bind_group_ms: pending.pack_bind_group_ms,
            call_encoder_create_ms: pending.encoder_create_ms,
            call_table_stream_copy_ms: pending.table_stream_copy_ms,
            call_match_clear_ms: pending.match_clear_ms,
            call_sparse_prepare_dispatch_ms: pending.sparse_prepare_dispatch_ms,
            call_sparse_pack_dispatch_ms: pending.sparse_pack_dispatch_ms,
            call_result_readback_setup_ms: pending.result_readback_setup_ms,
            call_payload_readback_setup_ms: pending.payload_readback_setup_ms,
            ..GpuBatchKernelProfile::default()
        },
    };
    release_pending_gpu_sparse_pack_batch(r, pending);
    Ok(Some(out_result))
}

pub(crate) fn try_collect_matches_encode_and_pack_sparse_batch(
    pending: &mut PendingGpuSparsePackBatch,
) -> Result<Option<GpuBatchSparsePackOutput>, PDeflateError> {
    let r = runtime()?;
    match collect_pending_gpu_sparse_pack_batch(pending, false) {
        Ok(out) => Ok(out),
        Err(err) => {
            release_pending_gpu_sparse_pack_batch(r, pending);
            Err(err)
        }
    }
}

pub(crate) fn collect_matches_encode_and_pack_sparse_batch(
    mut pending: PendingGpuSparsePackBatch,
) -> Result<GpuBatchSparsePackOutput, PDeflateError> {
    let r = runtime()?;
    match collect_pending_gpu_sparse_pack_batch(&mut pending, true) {
        Ok(Some(out)) => Ok(out),
        Ok(None) => Err(PDeflateError::Gpu(
            "integrated/collect: pending batch was not ready after blocking collect".to_string(),
        )),
        Err(err) => {
            release_pending_gpu_sparse_pack_batch(r, &mut pending);
            Err(err)
        }
    }
}

pub(crate) fn compute_matches_encode_and_pack_sparse_batch(
    inputs: &[GpuMatchInput<'_>],
    section_count: usize,
    max_cmd_len: usize,
) -> Result<GpuBatchSparsePackOutput, PDeflateError> {
    let pending = submit_matches_encode_and_pack_sparse_batch(inputs, section_count, max_cmd_len)?;
    collect_matches_encode_and_pack_sparse_batch(pending)
}

pub(crate) fn compute_matches(input: &GpuMatchInput<'_>) -> Result<GpuMatchOutput, PDeflateError> {
    let batch = compute_matches_batch(std::slice::from_ref(input))?;
    let mut iter = batch.chunks.into_iter();
    let Some(chunk) = iter.next() else {
        return Err(PDeflateError::Gpu(
            "gpu batch returned empty result for single chunk".to_string(),
        ));
    };
    Ok(GpuMatchOutput {
        packed_matches: chunk.packed_matches,
        profile: batch.profile,
    })
}

#[inline]
fn section_start(sec: usize, section_count: usize, len: usize) -> usize {
    if section_count == 0 || len == 0 {
        return 0;
    }
    if sec == 0 {
        return 0;
    }
    if sec >= section_count {
        return len;
    }
    let raw = sec.saturating_mul(len) / section_count;
    raw & !3usize
}

#[allow(dead_code)]
pub(crate) fn encode_section_commands(
    src: &[u8],
    packed_matches: &[u32],
    section_count: usize,
    min_ref_len: usize,
    max_ref_len: usize,
    max_cmd_len: usize,
) -> Result<GpuSectionEncodeOutput, PDeflateError> {
    if src.len() != packed_matches.len() {
        return Err(PDeflateError::InvalidOptions(
            "gpu section encode input length mismatch",
        ));
    }
    if src.is_empty() || section_count == 0 {
        return Ok(GpuSectionEncodeOutput {
            section_cmd_lens: vec![0; section_count],
            section_cmd: Vec::new(),
            profile: GpuSectionEncodeProfile::default(),
        });
    }
    let r = runtime()?;
    let t_total = Instant::now();

    let mut section_offsets = vec![0u32; section_count];
    let mut section_caps = vec![0u32; section_count];
    let mut total_cmd_cap_bytes: usize = 0;
    for sec in 0..section_count {
        let s0 = section_start(sec, section_count, src.len());
        let s1 = section_start(sec + 1, section_count, src.len());
        let sec_len = s1.saturating_sub(s0);
        // Tightened cap to reduce oversized command buffer traffic.
        let cap_bytes = estimate_section_cmd_cap_bytes(sec_len)?;
        let aligned = align_up4(total_cmd_cap_bytes);
        section_offsets[sec] =
            u32::try_from(aligned).map_err(|_| PDeflateError::NumericOverflow)?;
        section_caps[sec] = u32::try_from(cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
        total_cmd_cap_bytes = aligned
            .checked_add(cap_bytes)
            .ok_or(PDeflateError::NumericOverflow)?;
    }

    let src_words = pack_bytes_to_words(src);
    let src_words_bytes = u64::try_from(src_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let packed_bytes = u64::try_from(packed_matches.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let section_count_u64 =
        u64::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
    let section_offsets_bytes = section_count_u64.saturating_mul(4);
    let section_caps_bytes = section_count_u64.saturating_mul(4);
    let params: [u32; 10] = [
        u32::try_from(src.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(min_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(max_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(max_cmd_len).map_err(|_| PDeflateError::NumericOverflow)?,
        0,
        0,
        0,
        0,
        0,
    ];
    let params_bytes = u64::try_from(params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let out_lens_bytes = section_count_u64.saturating_mul(4);
    let out_cmd_words = total_cmd_cap_bytes.div_ceil(4);
    let out_cmd_bytes = u64::try_from(out_cmd_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    if out_cmd_bytes > r.max_storage_binding_size {
        return Err(PDeflateError::Gpu(format!(
            "gpu section encode output buffer too large ({} > {})",
            out_cmd_bytes, r.max_storage_binding_size
        )));
    }

    let src_buffer = storage_upload_buffer(&r.device, "cozip-pdeflate-se-src", src_words_bytes);
    let packed_buffer = storage_upload_buffer(&r.device, "cozip-pdeflate-se-packed", packed_bytes);
    let section_offsets_buffer =
        storage_upload_buffer(&r.device, "cozip-pdeflate-se-offs", section_offsets_bytes);
    let section_caps_buffer =
        storage_upload_buffer(&r.device, "cozip-pdeflate-se-caps", section_caps_bytes);
    let params_buffer = storage_upload_buffer(&r.device, "cozip-pdeflate-se-params", params_bytes);
    let out_lens_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-se-out-lens"),
        size: out_lens_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_cmd_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-se-out-cmd"),
        size: out_cmd_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_lens_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-se-readback-lens"),
        size: out_lens_bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_cmd_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-se-readback-cmd"),
        size: out_cmd_bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-se-bg"),
        layout: &r.section_encode_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: packed_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: section_offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: section_caps_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: out_lens_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: out_cmd_buffer.as_entire_binding(),
            },
        ],
    });

    let t_upload = Instant::now();
    r.queue
        .write_buffer(&src_buffer, 0, bytemuck::cast_slice(&src_words));
    r.queue
        .write_buffer(&packed_buffer, 0, bytemuck::cast_slice(packed_matches));
    r.queue.write_buffer(
        &section_offsets_buffer,
        0,
        bytemuck::cast_slice(&section_offsets),
    );
    r.queue
        .write_buffer(&section_caps_buffer, 0, bytemuck::cast_slice(&section_caps));
    r.queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));
    let upload_ms = elapsed_ms(t_upload);

    let groups_x = u32::try_from(section_count)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .div_ceil(64)
        .max(1);
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-se-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-se-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.section_encode_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &out_lens_buffer,
        0,
        &readback_lens_buffer,
        0,
        out_lens_bytes.max(4),
    );
    encoder.copy_buffer_to_buffer(
        &out_cmd_buffer,
        0,
        &readback_cmd_buffer,
        0,
        out_cmd_bytes.max(4),
    );
    let submission = r.queue.submit(Some(encoder.finish()));

    let t_wait = Instant::now();
    let lens_slice = readback_lens_buffer.slice(..out_lens_bytes.max(4));
    let cmd_slice = readback_cmd_buffer.slice(..out_cmd_bytes.max(4));
    let (tx_lens, rx_lens) = mpsc::channel();
    lens_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_lens.send(res);
    });
    let (tx_cmd, rx_cmd) = mpsc::channel();
    cmd_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_cmd.send(res);
    });
    r.device.poll(wgpu::Maintain::wait_for(submission));
    rx_lens
        .recv()
        .map_err(|_| PDeflateError::Gpu("gpu section encode lens map channel closed".to_string()))?
        .map_err(|e| PDeflateError::Gpu(format!("gpu section encode lens map failed: {e}")))?;
    rx_cmd
        .recv()
        .map_err(|_| PDeflateError::Gpu("gpu section encode cmd map channel closed".to_string()))?
        .map_err(|e| PDeflateError::Gpu(format!("gpu section encode cmd map failed: {e}")))?;
    let wait_ms = elapsed_ms(t_wait);

    let t_copy = Instant::now();
    let lens_mapped = lens_slice.get_mapped_range();
    let cmd_mapped = cmd_slice.get_mapped_range();
    let lens_u32: &[u32] = bytemuck::cast_slice(&lens_mapped);
    let cmd_bytes: &[u8] = &cmd_mapped;
    let mut section_cmd_lens = vec![0u32; section_count];
    section_cmd_lens.copy_from_slice(&lens_u32[..section_count]);
    let mut lens_error: Option<PDeflateError> = None;
    for (sec, &len_u32) in section_cmd_lens.iter().enumerate() {
        if len_u32 == 0xffff_ffff {
            lens_error = Some(PDeflateError::Gpu(format!(
                "gpu section encode overflow on section {}",
                sec
            )));
            break;
        }
        let cap = section_caps[sec];
        if len_u32 > cap {
            lens_error = Some(PDeflateError::Gpu(format!(
                "gpu section encode len exceeds cap on section {}",
                sec
            )));
            break;
        }
    }
    let total_cmd_len: usize = section_cmd_lens
        .iter()
        .try_fold(0usize, |acc, &v| acc.checked_add(usize::try_from(v).ok()?))
        .ok_or(PDeflateError::NumericOverflow)?;
    let mut section_cmd = Vec::with_capacity(total_cmd_len);
    for sec in 0..section_count {
        let off =
            usize::try_from(section_offsets[sec]).map_err(|_| PDeflateError::NumericOverflow)?;
        let len =
            usize::try_from(section_cmd_lens[sec]).map_err(|_| PDeflateError::NumericOverflow)?;
        if off.checked_add(len).ok_or(PDeflateError::NumericOverflow)? > cmd_bytes.len() {
            return Err(PDeflateError::Gpu(format!(
                "gpu section encode cmd bytes truncated on section {}",
                sec
            )));
        }
        section_cmd.extend_from_slice(&cmd_bytes[off..off + len]);
    }
    drop(lens_mapped);
    drop(cmd_mapped);
    readback_lens_buffer.unmap();
    readback_cmd_buffer.unmap();
    if let Some(err) = lens_error {
        return Err(err);
    }
    validate_section_commands(src.len(), section_count, &section_cmd_lens, &section_cmd)?;
    let map_copy_ms = elapsed_ms(t_copy);

    Ok(GpuSectionEncodeOutput {
        section_cmd_lens,
        section_cmd,
        profile: GpuSectionEncodeProfile {
            upload_ms,
            wait_ms,
            map_copy_ms,
            total_ms: elapsed_ms(t_total),
        },
    })
}

fn host_section_start(sec: usize, section_count: usize, len: usize) -> usize {
    if section_count == 0 || len == 0 {
        return 0;
    }
    if sec == 0 {
        return 0;
    }
    if sec >= section_count {
        return len;
    }
    let raw = ((sec as u128) * (len as u128) / (section_count as u128)) as usize;
    raw & !3usize
}

fn validate_section_commands(
    src_len: usize,
    section_count: usize,
    section_cmd_lens: &[u32],
    section_cmd: &[u8],
) -> Result<(), PDeflateError> {
    if section_cmd_lens.len() != section_count {
        return Err(PDeflateError::Gpu(
            "gpu section encode validation failed: lens count mismatch".to_string(),
        ));
    }
    let mut offset = 0usize;
    for sec in 0..section_count {
        let sec_cmd_len =
            usize::try_from(section_cmd_lens[sec]).map_err(|_| PDeflateError::NumericOverflow)?;
        let end = offset
            .checked_add(sec_cmd_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        let cmd = section_cmd.get(offset..end).ok_or_else(|| {
            PDeflateError::Gpu(format!(
                "gpu section encode validation failed: section {} range oob",
                sec
            ))
        })?;
        let s0 = host_section_start(sec, section_count, src_len);
        let s1 = host_section_start(sec + 1, section_count, src_len);
        let mut out_pos = 0usize;
        let mut cursor = 0usize;
        while out_pos < (s1 - s0) {
            let h_end = cursor
                .checked_add(2)
                .ok_or(PDeflateError::NumericOverflow)?;
            let header = cmd.get(cursor..h_end).ok_or_else(|| {
                PDeflateError::Gpu(format!(
                    "gpu section encode validation failed: section {} header truncated",
                    sec
                ))
            })?;
            let word = u16::from_le_bytes([header[0], header[1]]);
            cursor = h_end;
            let tag = word & 0x0fff;
            let len4 = usize::from(word >> 12);
            let mut len = len4;
            if len4 == 0x0f {
                let ext = *cmd.get(cursor).ok_or_else(|| {
                    PDeflateError::Gpu(format!(
                        "gpu section encode validation failed: section {} ext truncated",
                        sec
                    ))
                })?;
                cursor += 1;
                len = len.saturating_add(usize::from(ext));
            }
            if len == 0 {
                return Err(PDeflateError::Gpu(format!(
                    "gpu section encode validation failed: section {} has zero-length command",
                    sec
                )));
            }
            if tag == 0x0fff {
                let lit_end = cursor
                    .checked_add(len)
                    .ok_or(PDeflateError::NumericOverflow)?;
                if lit_end > cmd.len() {
                    return Err(PDeflateError::Gpu(format!(
                        "gpu section encode validation failed: section {} literal oob",
                        sec
                    )));
                }
                cursor = lit_end;
            }
            out_pos = out_pos.saturating_add(len);
            if out_pos > (s1 - s0) {
                return Err(PDeflateError::Gpu(format!(
                    "gpu section encode validation failed: section {} output overrun",
                    sec
                )));
            }
        }
        if out_pos != (s1 - s0) || cursor != cmd.len() {
            return Err(PDeflateError::Gpu(format!(
                "gpu section encode validation failed: section {} length mismatch",
                sec
            )));
        }
        offset = end;
    }
    if offset != section_cmd.len() {
        return Err(PDeflateError::Gpu(
            "gpu section encode validation failed: trailing command bytes".to_string(),
        ));
    }
    Ok(())
}
