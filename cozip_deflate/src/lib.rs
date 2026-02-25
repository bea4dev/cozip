use std::borrow::Cow;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use flate2::Compression;
use thiserror::Error;

const FRAME_MAGIC: [u8; 4] = *b"CZDF";
const FRAME_VERSION: u8 = 3;
const HEADER_LEN: usize = 22;
const CHUNK_META_LEN_V3: usize = 11;
const CHUNK_META_LEN_V2: usize = 10;
const CHUNK_META_LEN_V1: usize = 9;
const TRANSFORM_LANES: usize = 2;

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const MAX_DISTANCE: usize = 32_768;
const HASH_SIZE: usize = 1 << 15;
const HASH_MASK: usize = HASH_SIZE - 1;
const MAX_HASH_CANDIDATES: usize = 32;
const MAX_RUN_HINTS: usize = 8;
const LITLEN_SYMBOL_COUNT: usize = 286;
const DIST_SYMBOL_COUNT: usize = 30;
const DYN_TABLE_U32_COUNT: usize = (LITLEN_SYMBOL_COUNT * 2) + (DIST_SYMBOL_COUNT * 2);
const GPU_BATCH_CHUNKS: usize = 16;
const GPU_PIPELINED_SUBMIT_CHUNKS: usize = 4;
const GPU_DEFLATE_SLOT_POOL: usize = GPU_BATCH_CHUNKS;
const PREFIX_SCAN_BLOCK_SIZE: usize = 256;
const TOKEN_FINALIZE_SEGMENT_SIZE: usize = 4096;
const GPU_DEFLATE_MAX_BITS_PER_BYTE: usize = 12;
const MAX_DISPATCH_WORKGROUPS_PER_DIM: u32 = 65_535;
const GPU_RESERVATION_TIMEOUT_MS: u64 = 3;
const SCHEDULER_WAIT_MS: u64 = 1;

#[derive(Debug, Clone)]
pub struct HybridOptions {
    pub chunk_size: usize,
    pub gpu_subchunk_size: usize,
    pub compression_level: u32,
    pub compression_mode: CompressionMode,
    pub prefer_gpu: bool,
    pub gpu_fraction: f32,
    pub gpu_min_chunk_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMode {
    Speed,
    Balanced,
    Ratio,
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

#[derive(Debug, Clone)]
struct ChunkTask {
    index: usize,
    preferred_gpu: bool,
    raw: Vec<u8>,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompressTaskState {
    Pending = 0,
    ReservedGpu = 1,
    RunningGpu = 2,
    RunningCpu = 3,
    Done = 4,
}

#[derive(Debug)]
struct ScheduledCompressTask {
    index: usize,
    preferred_gpu: bool,
    raw: Vec<u8>,
    state: AtomicU8,
    reserved_at_ms: AtomicU64,
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

#[derive(Debug)]
struct GpuAssist {
    device: wgpu::Device,
    queue: wgpu::Queue,
    tokenize_bind_group_layout: wgpu::BindGroupLayout,
    tokenize_pipeline: wgpu::ComputePipeline,
    token_finalize_pipeline: wgpu::ComputePipeline,
    freq_bind_group_layout: wgpu::BindGroupLayout,
    freq_pipeline: wgpu::ComputePipeline,
    dyn_map_bind_group_layout: wgpu::BindGroupLayout,
    dyn_map_pipeline: wgpu::ComputePipeline,
    dyn_finalize_bind_group_layout: wgpu::BindGroupLayout,
    dyn_finalize_pipeline: wgpu::ComputePipeline,
    litlen_bind_group_layout: wgpu::BindGroupLayout,
    litlen_pipeline: wgpu::ComputePipeline,
    bitpack_bind_group_layout: wgpu::BindGroupLayout,
    bitpack_pipeline: wgpu::ComputePipeline,
    match_bind_group_layout: wgpu::BindGroupLayout,
    match_pipeline: wgpu::ComputePipeline,
    count_bind_group_layout: wgpu::BindGroupLayout,
    count_pipeline: wgpu::ComputePipeline,
    prefix_bind_group_layout: wgpu::BindGroupLayout,
    prefix_pipeline: wgpu::ComputePipeline,
    scan_blocks_bind_group_layout: wgpu::BindGroupLayout,
    scan_blocks_pipeline: wgpu::ComputePipeline,
    scan_add_bind_group_layout: wgpu::BindGroupLayout,
    scan_add_pipeline: wgpu::ComputePipeline,
    emit_bind_group_layout: wgpu::BindGroupLayout,
    emit_pipeline: wgpu::ComputePipeline,
    deflate_slots: Mutex<Vec<DeflateSlot>>,
    deflate_header_buffer: wgpu::Buffer,
}

#[derive(Debug)]
struct DeflateSlot {
    len_capacity: usize,
    output_storage_size: u64,
    input_buffer: wgpu::Buffer,
    codes_buffer: wgpu::Buffer,
    token_flags_buffer: wgpu::Buffer,
    token_kind_buffer: wgpu::Buffer,
    token_len_buffer: wgpu::Buffer,
    token_dist_buffer: wgpu::Buffer,
    token_lit_buffer: wgpu::Buffer,
    litlen_freq_buffer: wgpu::Buffer,
    dist_freq_buffer: wgpu::Buffer,
    dyn_table_buffer: wgpu::Buffer,
    dyn_meta_buffer: wgpu::Buffer,
    dyn_overflow_buffer: wgpu::Buffer,
    token_prefix_buffer: wgpu::Buffer,
    token_total_buffer: wgpu::Buffer,
    bitlens_buffer: wgpu::Buffer,
    bit_offsets_buffer: wgpu::Buffer,
    total_bits_buffer: wgpu::Buffer,
    output_words_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    readback: wgpu::Buffer,
    litlen_bg: wgpu::BindGroup,
    tokenize_bg: wgpu::BindGroup,
    freq_bg: wgpu::BindGroup,
    dyn_map_bg: wgpu::BindGroup,
    dyn_finalize_bg: wgpu::BindGroup,
    bitpack_bg: wgpu::BindGroup,
}

impl GpuAssist {
    fn new() -> Result<Self, CozipDeflateError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, CozipDeflateError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| CozipDeflateError::GpuUnavailable("adapter not found".to_string()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("cozip-deflate-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|err| CozipDeflateError::GpuUnavailable(err.to_string()))?;

        let tokenize_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-tokenize-bgl"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let tokenize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-tokenize-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    block_size: u32,
    mode: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;

@group(0) @binding(1)
var<storage, read_write> token_flags: array<u32>;

@group(0) @binding(2)
var<storage, read_write> token_kind: array<u32>;

@group(0) @binding(3)
var<storage, read_write> token_len: array<u32>;

@group(0) @binding(4)
var<storage, read_write> token_dist: array<u32>;

@group(0) @binding(5)
var<storage, read_write> token_lit: array<u32>;

@group(0) @binding(6)
var<uniform> params: Params;

fn mode_max_match_scan(mode: u32) -> u32 {
    switch (mode) {
        case 1u: { return 128u; } // Balanced
        case 2u: { return 192u; } // Ratio
        default: { return 64u; }  // Speed
    }
}

fn mode_max_match_len(mode: u32) -> u32 {
    switch (mode) {
        case 1u: { return 128u; } // Balanced
        case 2u: { return 258u; } // Ratio
        default: { return 64u; }  // Speed
    }
}

fn mode_dist_candidate_count(mode: u32) -> u32 {
    switch (mode) {
        case 1u: { return 28u; } // up to 8192
        case 2u: { return 32u; } // up to 32768
        default: { return 20u; } // up to 512
    }
}

fn byte_at(index: u32) -> u32 {
    let word = input_words[index >> 2u];
    let shift = (index & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn dist_candidate(slot: u32) -> u32 {
    switch (slot) {
        case 0u: { return 1u; }
        case 1u: { return 2u; }
        case 2u: { return 3u; }
        case 3u: { return 4u; }
        case 4u: { return 5u; }
        case 5u: { return 6u; }
        case 6u: { return 8u; }
        case 7u: { return 10u; }
        case 8u: { return 12u; }
        case 9u: { return 16u; }
        case 10u: { return 24u; }
        case 11u: { return 32u; }
        case 12u: { return 48u; }
        case 13u: { return 64u; }
        case 14u: { return 96u; }
        case 15u: { return 128u; }
        case 16u: { return 192u; }
        case 17u: { return 256u; }
        case 18u: { return 384u; }
        case 19u: { return 512u; }
        case 20u: { return 768u; }
        case 21u: { return 1024u; }
        case 22u: { return 1536u; }
        case 23u: { return 2048u; }
        case 24u: { return 3072u; }
        case 25u: { return 4096u; }
        case 26u: { return 6144u; }
        case 27u: { return 8192u; }
        case 28u: { return 12288u; }
        case 29u: { return 16384u; }
        case 30u: { return 24576u; }
        default: { return 32768u; }
    }
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + (id.y * 8388480u);
    if (i >= params.len) {
        return;
    }

    let lit = byte_at(i);
    token_lit[i] = lit;
    token_flags[i] = 0u;
    token_kind[i] = 0u;

    var best_dist: u32 = 0u;
    var best_len: u32 = 0u;
    let match_scan_limit = mode_max_match_scan(params.mode);
    let match_len_limit = mode_max_match_len(params.mode);
    let dist_limit = mode_dist_candidate_count(params.mode);
    var c: u32 = 0u;
    loop {
        if (c >= dist_limit) {
            break;
        }
        let dist = dist_candidate(c);
        if (dist <= i && i + 2u < params.len) {
            if (byte_at(i) == byte_at(i - dist)
                && byte_at(i + 1u) == byte_at(i + 1u - dist)
                && byte_at(i + 2u) == byte_at(i + 2u - dist))
            {
                var mlen: u32 = 3u;
                var p = i + 3u;
                var scanned: u32 = 0u;
                loop {
                    if (p >= params.len || mlen >= match_len_limit || scanned >= match_scan_limit) {
                        break;
                    }
                    if (byte_at(p) != byte_at(p - dist)) {
                        break;
                    }
                    mlen = mlen + 1u;
                    p = p + 1u;
                    scanned = scanned + 1u;
                }
                if (mlen > best_len || (mlen == best_len && (best_dist == 0u || dist < best_dist))) {
                    best_len = mlen;
                    best_dist = dist;
                }
            }
        }
        c = c + 1u;
    }

    if (best_len >= 3u && best_dist > 0u) {
        token_len[i] = best_len;
        token_dist[i] = best_dist;
    } else {
        token_len[i] = 0u;
        token_dist[i] = 0u;
    }
}
"#,
            )),
        });

        let tokenize_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-tokenize-layout"),
            bind_group_layouts: &[&tokenize_bind_group_layout],
            push_constant_ranges: &[],
        });

        let tokenize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-tokenize-pipeline"),
            layout: Some(&tokenize_layout),
            module: &tokenize_shader,
            entry_point: "main",
        });

        let token_finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-token-finalize-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    block_size: u32,
    mode: u32,
    _pad1: u32,
}

@group(0) @binding(1)
var<storage, read_write> token_flags: array<u32>;

@group(0) @binding(2)
var<storage, read_write> token_kind: array<u32>;

@group(0) @binding(3)
var<storage, read_write> token_len: array<u32>;

@group(0) @binding(4)
var<storage, read_write> token_dist: array<u32>;

@group(0) @binding(6)
var<uniform> params: Params;

fn mode_lazy_delta(mode: u32) -> u32 {
    switch (mode) {
        case 1u: { return 1u; } // Balanced
        case 2u: { return 2u; } // Ratio
        default: { return 0u; } // Speed
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let segment_size = max(params.block_size, 1u);
    let seg_start = id.x * segment_size;
    if (seg_start >= params.len) {
        return;
    }
    let seg_end = min(seg_start + segment_size, params.len);

    var i: u32 = seg_start;
    let lazy_delta = mode_lazy_delta(params.mode);
    loop {
        if (i >= seg_end) {
            break;
        }

        if (token_len[i] >= 3u && token_dist[i] > 0u) {
            let mlen = min(token_len[i], seg_end - i);
            var take_match = true;
            if (lazy_delta > 0u && (i + 1u) < seg_end && token_len[i + 1u] >= 3u && token_dist[i + 1u] > 0u) {
                let next_len = min(token_len[i + 1u], seg_end - (i + 1u));
                if (next_len >= mlen + lazy_delta) {
                    take_match = false;
                }
            }

            if (take_match) {
                token_flags[i] = 1u;
                token_kind[i] = 1u;
                token_len[i] = mlen;

                var j: u32 = 1u;
                loop {
                    if (j >= mlen || (i + j) >= seg_end) {
                        break;
                    }
                    token_flags[i + j] = 0u;
                    token_kind[i + j] = 0u;
                    token_len[i + j] = 0u;
                    token_dist[i + j] = 0u;
                    j = j + 1u;
                }
                i = i + mlen;
            } else {
                token_flags[i] = 1u;
                token_kind[i] = 0u;
                token_len[i] = 0u;
                token_dist[i] = 0u;
                i = i + 1u;
            }
        } else {
            token_flags[i] = 1u;
            token_kind[i] = 0u;
            token_len[i] = 0u;
            token_dist[i] = 0u;
            i = i + 1u;
        }
    }
}
"#,
            )),
        });

        let token_finalize_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cozip-token-finalize-pipeline"),
                layout: Some(&tokenize_layout),
                module: &token_finalize_shader,
                entry_point: "main",
            });

        let freq_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-freq-bgl"),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let freq_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-freq-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    block_size: u32,
    mode: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> token_flags: array<u32>;

@group(0) @binding(1)
var<storage, read> token_kind: array<u32>;

@group(0) @binding(2)
var<storage, read> token_match_len: array<u32>;

@group(0) @binding(3)
var<storage, read> token_match_dist: array<u32>;

@group(0) @binding(4)
var<storage, read> token_lit: array<u32>;

@group(0) @binding(5)
var<storage, read_write> litlen_freq: array<atomic<u32>>;

@group(0) @binding(6)
var<storage, read_write> dist_freq: array<atomic<u32>>;

@group(0) @binding(7)
var<uniform> params: Params;

fn litlen_symbol_for_len(mlen_in: u32) -> u32 {
    let mlen = min(max(mlen_in, 3u), 258u);
    if (mlen <= 10u) {
        return 254u + mlen;
    }
    if (mlen == 258u) {
        return 285u;
    }

    var symbol: u32 = 265u;
    var base: u32 = 11u;
    var extra: u32 = 1u;
    loop {
        if (extra > 5u) {
            break;
        }
        var j: u32 = 0u;
        loop {
            if (j >= 4u) {
                break;
            }
            let maxv = base + ((1u << extra) - 1u);
            if (mlen >= base && mlen <= maxv) {
                return symbol;
            }
            base = maxv + 1u;
            symbol = symbol + 1u;
            j = j + 1u;
        }
        extra = extra + 1u;
    }
    return 285u;
}

fn dist_symbol_for_dist(mdist_in: u32) -> u32 {
    let mdist = max(mdist_in, 1u);
    if (mdist <= 1u) {
        return 0u;
    }
    if (mdist <= 4u) {
        return mdist - 1u;
    }

    var symbol: u32 = 4u;
    var base: u32 = 5u;
    var extra: u32 = 1u;
    loop {
        if (extra > 13u) {
            break;
        }
        var j: u32 = 0u;
        loop {
            if (j >= 2u) {
                break;
            }
            let maxv = base + ((1u << extra) - 1u);
            if (mdist >= base && mdist <= maxv) {
                return symbol;
            }
            base = maxv + 1u;
            symbol = symbol + 1u;
            j = j + 1u;
        }
        extra = extra + 1u;
    }
    return 29u;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 8388480u);
    if (idx >= params.len || token_flags[idx] == 0u) {
        return;
    }

    if (token_kind[idx] == 0u) {
        let lit = min(token_lit[idx], 255u);
        atomicAdd(&litlen_freq[lit], 1u);
        return;
    }

    let len_symbol = litlen_symbol_for_len(token_match_len[idx]);
    let dist_symbol = dist_symbol_for_dist(token_match_dist[idx]);
    atomicAdd(&litlen_freq[len_symbol], 1u);
    atomicAdd(&dist_freq[dist_symbol], 1u);
}
"#,
            )),
        });

        let freq_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-freq-layout"),
            bind_group_layouts: &[&freq_bind_group_layout],
            push_constant_ranges: &[],
        });

        let freq_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-freq-pipeline"),
            layout: Some(&freq_layout),
            module: &freq_shader,
            entry_point: "main",
        });

        let dyn_map_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-dyn-map-bgl"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let dyn_map_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-dyn-map-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    block_size: u32,
    mode: u32,
    header_bits: u32,
}

@group(0) @binding(0)
var<storage, read> token_flags: array<u32>;

@group(0) @binding(1)
var<storage, read> token_match_len: array<u32>;

@group(0) @binding(2)
var<storage, read> token_match_dist: array<u32>;

@group(0) @binding(3)
var<storage, read> token_lit: array<u32>;

@group(0) @binding(4)
var<storage, read> dyn_table: array<u32>;

@group(0) @binding(5)
var<storage, read_write> out_codes: array<u32>;

@group(0) @binding(6)
var<storage, read_write> out_overflow: array<u32>;

@group(0) @binding(7)
var<storage, read_write> out_bitlens: array<u32>;

@group(0) @binding(8)
var<uniform> params: Params;

fn litlen_code(sym: u32) -> u32 {
    return dyn_table[sym];
}

fn litlen_bits(sym: u32) -> u32 {
    return dyn_table[286u + sym];
}

fn dist_code(sym: u32) -> u32 {
    return dyn_table[572u + sym];
}

fn dist_bits(sym: u32) -> u32 {
    return dyn_table[602u + sym];
}

fn litlen_symbol_for_len(mlen_in: u32) -> vec3<u32> {
    let mlen = min(max(mlen_in, 3u), 258u);
    if (mlen <= 10u) {
        return vec3<u32>(254u + mlen, 0u, 0u);
    }
    if (mlen == 258u) {
        return vec3<u32>(285u, 0u, 0u);
    }

    var symbol: u32 = 265u;
    var base: u32 = 11u;
    var extra: u32 = 1u;
    loop {
        if (extra > 5u) {
            break;
        }
        var j: u32 = 0u;
        loop {
            if (j >= 4u) {
                break;
            }
            let maxv = base + ((1u << extra) - 1u);
            if (mlen >= base && mlen <= maxv) {
                return vec3<u32>(symbol, mlen - base, extra);
            }
            base = maxv + 1u;
            symbol = symbol + 1u;
            j = j + 1u;
        }
        extra = extra + 1u;
    }
    return vec3<u32>(285u, 0u, 0u);
}

fn dist_symbol_for_dist(mdist_in: u32) -> vec3<u32> {
    let mdist = max(mdist_in, 1u);
    if (mdist <= 1u) {
        return vec3<u32>(0u, 0u, 0u);
    }
    if (mdist <= 4u) {
        return vec3<u32>(mdist - 1u, 0u, 0u);
    }

    var symbol: u32 = 4u;
    var base: u32 = 5u;
    var extra: u32 = 1u;
    loop {
        if (extra > 13u) {
            break;
        }
        var j: u32 = 0u;
        loop {
            if (j >= 2u) {
                break;
            }
            let maxv = base + ((1u << extra) - 1u);
            if (mdist >= base && mdist <= maxv) {
                return vec3<u32>(symbol, mdist - base, extra);
            }
            base = maxv + 1u;
            symbol = symbol + 1u;
            j = j + 1u;
        }
        extra = extra + 1u;
    }
    return vec3<u32>(29u, 0u, 0u);
}

fn append_bits(
    value: u32,
    bits: u32,
    code_lo: ptr<function, u32>,
    code_hi: ptr<function, u32>,
    bitlen: ptr<function, u32>,
) {
    if (bits == 0u) {
        return;
    }
    let cur = *bitlen;
    if (cur < 32u) {
        if (cur + bits <= 32u) {
            *code_lo = *code_lo | (value << cur);
        } else {
            let low_bits = 32u - cur;
            *code_lo = *code_lo | (value << cur);
            *code_hi = *code_hi | (value >> low_bits);
        }
    } else {
        let hi_shift = cur - 32u;
        *code_hi = *code_hi | (value << hi_shift);
    }
    *bitlen = cur + bits;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 8388480u);
    if (idx >= params.len || token_flags[idx] == 0u) {
        return;
    }

    var code_lo: u32 = 0u;
    var code_hi: u32 = 0u;
    var bits: u32 = 0u;

    if (token_match_len[idx] < 3u || token_match_dist[idx] == 0u) {
        let sym = min(token_lit[idx], 255u);
        append_bits(litlen_code(sym), litlen_bits(sym), &code_lo, &code_hi, &bits);
    } else {
        let len_info = litlen_symbol_for_len(token_match_len[idx]);
        let len_sym = len_info.x;
        let len_extra_val = len_info.y;
        let len_extra_bits = len_info.z;
        append_bits(
            litlen_code(len_sym),
            litlen_bits(len_sym),
            &code_lo,
            &code_hi,
            &bits,
        );
        append_bits(len_extra_val, len_extra_bits, &code_lo, &code_hi, &bits);

        let dist_info = dist_symbol_for_dist(token_match_dist[idx]);
        let dist_sym = dist_info.x;
        let dist_extra_val = dist_info.y;
        let dist_extra_bits = dist_info.z;
        append_bits(dist_code(dist_sym), dist_bits(dist_sym), &code_lo, &code_hi, &bits);
        append_bits(dist_extra_val, dist_extra_bits, &code_lo, &code_hi, &bits);
    }

    out_codes[idx] = code_lo;
    out_overflow[idx] = code_hi;
    out_bitlens[idx] = bits;
}
"#,
            )),
        });

        let dyn_map_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-dyn-map-layout"),
            bind_group_layouts: &[&dyn_map_bind_group_layout],
            push_constant_ranges: &[],
        });

        let dyn_map_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-dyn-map-pipeline"),
            layout: Some(&dyn_map_layout),
            module: &dyn_map_shader,
            entry_point: "main",
        });

        let dyn_finalize_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-dyn-finalize-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let dyn_finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-dyn-finalize-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
@group(0) @binding(0)
var<storage, read_write> out_words: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read_write> total_bits: array<u32>;

@group(0) @binding(2)
var<storage, read> dyn_meta: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) {
        return;
    }

    let header_bits = dyn_meta[0];
    let eob_code = dyn_meta[1];
    let eob_bits = dyn_meta[2];

    let token_bits = total_bits[0];
    let bit_offset = header_bits + token_bits;
    let word_index = bit_offset >> 5u;
    let shift = bit_offset & 31u;

    atomicOr(&out_words[word_index], eob_code << shift);
    if (shift + eob_bits > 32u) {
        atomicOr(&out_words[word_index + 1u], eob_code >> (32u - shift));
    }

    total_bits[0] = bit_offset + eob_bits;
}
"#,
            )),
        });

        let dyn_finalize_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-dyn-finalize-layout"),
            bind_group_layouts: &[&dyn_finalize_bind_group_layout],
            push_constant_ranges: &[],
        });

        let dyn_finalize_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cozip-dyn-finalize-pipeline"),
                layout: Some(&dyn_finalize_layout),
                module: &dyn_finalize_shader,
                entry_point: "main",
            });

        let litlen_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-litlen-bgl"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let litlen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-litlen-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> token_flags: array<u32>;

@group(0) @binding(1)
var<storage, read> token_kind: array<u32>;

@group(0) @binding(2)
var<storage, read> token_match_len: array<u32>;

@group(0) @binding(3)
var<storage, read> token_match_dist: array<u32>;

@group(0) @binding(4)
var<storage, read> token_lit: array<u32>;

@group(0) @binding(5)
var<storage, read> token_prefix: array<u32>;

@group(0) @binding(6)
var<storage, read_write> codes: array<u32>;

@group(0) @binding(7)
var<storage, read_write> bitlens: array<u32>;

@group(0) @binding(8)
var<uniform> params: Params;

fn reverse_bits_u32(value: u32, bit_len: u32) -> u32 {
    var out: u32 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= bit_len) {
            break;
        }
        out = (out << 1u) | ((value >> i) & 1u);
        i = i + 1u;
    }
    return out;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 8388480u);
    if (idx >= params.len) {
        return;
    }

    if (token_flags[idx] == 0u) {
        return;
    }

    let token_index = token_prefix[idx];
    let kind = token_kind[idx];

    if (kind == 0u) {
        let lit = token_lit[idx];
        var code: u32 = 0u;
        var bits: u32 = 0u;
        if (lit <= 143u) {
            code = 0x30u + lit;
            bits = 8u;
        } else {
            code = 0x190u + (lit - 144u);
            bits = 9u;
        }

        codes[token_index] = reverse_bits_u32(code, bits);
        bitlens[token_index] = bits;
    } else {
        let mlen = token_match_len[idx];
        let mdist = token_match_dist[idx];

        var len_symbol: u32 = 257u;
        var len_extra_bits: u32 = 0u;
        var len_extra_value: u32 = 0u;

        if (mlen <= 10u) {
            len_symbol = 254u + mlen;
        } else if (mlen == 258u) {
            len_symbol = 285u;
        } else {
            var symbol: u32 = 265u;
            var base: u32 = 11u;
            var extra: u32 = 1u;
            var found: bool = false;

            loop {
                if (extra > 5u || found) {
                    break;
                }

                var j: u32 = 0u;
                loop {
                    if (j >= 4u) {
                        break;
                    }
                    let maxv = base + ((1u << extra) - 1u);
                    if (mlen >= base && mlen <= maxv) {
                        len_symbol = symbol;
                        len_extra_bits = extra;
                        len_extra_value = mlen - base;
                        found = true;
                        break;
                    }
                    base = maxv + 1u;
                    symbol = symbol + 1u;
                    j = j + 1u;
                }

                extra = extra + 1u;
            }
        }

        var len_code: u32 = 0u;
        var len_bits: u32 = 0u;
        if (len_symbol <= 279u) {
            len_code = len_symbol - 256u;
            len_bits = 7u;
        } else {
            len_code = 0xC0u + (len_symbol - 280u);
            len_bits = 8u;
        }

        var out_code: u32 = 0u;
        var out_bits: u32 = 0u;

        out_code = out_code | (reverse_bits_u32(len_code, len_bits) << out_bits);
        out_bits = out_bits + len_bits;

        if (len_extra_bits > 0u) {
            out_code = out_code | (len_extra_value << out_bits);
            out_bits = out_bits + len_extra_bits;
        }

        var dist_symbol: u32 = 0u;
        var dist_extra_bits: u32 = 0u;
        var dist_extra_value: u32 = 0u;

        if (mdist <= 1u) {
            dist_symbol = 0u;
        } else if (mdist <= 4u) {
            dist_symbol = mdist - 1u;
        } else {
            var symbol: u32 = 4u;
            var base: u32 = 5u;
            var extra: u32 = 1u;
            var found: bool = false;

            loop {
                if (extra > 13u || found) {
                    break;
                }

                var j: u32 = 0u;
                loop {
                    if (j >= 2u) {
                        break;
                    }
                    let maxv = base + ((1u << extra) - 1u);
                    if (mdist >= base && mdist <= maxv) {
                        dist_symbol = symbol;
                        dist_extra_bits = extra;
                        dist_extra_value = mdist - base;
                        found = true;
                        break;
                    }
                    base = maxv + 1u;
                    symbol = symbol + 1u;
                    j = j + 1u;
                }

                extra = extra + 1u;
            }
        }

        let dist_code = reverse_bits_u32(dist_symbol, 5u);
        out_code = out_code | (dist_code << out_bits);
        out_bits = out_bits + 5u;
        if (dist_extra_bits > 0u) {
            out_code = out_code | (dist_extra_value << out_bits);
            out_bits = out_bits + dist_extra_bits;
        }

        codes[token_index] = out_code;
        bitlens[token_index] = out_bits;
    }
}
"#,
            )),
        });

        let litlen_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-litlen-layout"),
            bind_group_layouts: &[&litlen_bind_group_layout],
            push_constant_ranges: &[],
        });

        let litlen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-litlen-pipeline"),
            layout: Some(&litlen_layout),
            module: &litlen_shader,
            entry_point: "main",
        });

        let bitpack_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-bitpack-bgl"),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bitpack_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-bitpack-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> codes: array<u32>;

@group(0) @binding(1)
var<storage, read> codes_hi: array<u32>;

@group(0) @binding(2)
var<storage, read> bitlens: array<u32>;

@group(0) @binding(3)
var<storage, read> bit_offsets: array<u32>;

@group(0) @binding(4)
var<storage, read_write> out_words: array<atomic<u32>>;

@group(0) @binding(5)
var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 8388480u);
    if (idx >= params.len) {
        return;
    }

    let bits = bitlens[idx];
    if (bits == 0u) {
        return;
    }

    let code = codes[idx];
    let code_hi = codes_hi[idx];
    let bit_offset = bit_offsets[idx] + params._pad1;
    let word_index = bit_offset >> 5u;
    let shift = bit_offset & 31u;

    atomicOr(&out_words[word_index], code << shift);
    if (shift + bits > 32u) {
        atomicOr(&out_words[word_index + 1u], code >> (32u - shift));
    }

    if (bits > 32u) {
        let hi_bits = bits - 32u;
        let hi_offset = bit_offset + 32u;
        let hi_word_index = hi_offset >> 5u;
        let hi_shift = hi_offset & 31u;

        atomicOr(&out_words[hi_word_index], code_hi << hi_shift);
        if (hi_shift + hi_bits > 32u) {
            atomicOr(&out_words[hi_word_index + 1u], code_hi >> (32u - hi_shift));
        }
    }
}
"#,
            )),
        });

        let bitpack_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-bitpack-layout"),
            bind_group_layouts: &[&bitpack_bind_group_layout],
            push_constant_ranges: &[],
        });

        let bitpack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-bitpack-pipeline"),
            layout: Some(&bitpack_layout),
            module: &bitpack_shader,
            entry_point: "main",
        });

        let match_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-match-bgl"),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let match_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-match-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
@group(0) @binding(0)
var<storage, read> input_words: array<u32>;

@group(0) @binding(1)
var<storage, read_write> eq_prev: array<u32>;

struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(2)
var<uniform> params: Params;

fn byte_at(index: u32) -> u32 {
    let word = input_words[index >> 2u];
    let shift = (index & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 8388480u);
    if (idx >= params.len) {
        return;
    }

    if (idx == 0u) {
        eq_prev[idx] = 0u;
    } else {
        eq_prev[idx] = select(0u, 1u, byte_at(idx) == byte_at(idx - 1u));
    }
}
"#,
            )),
        });

        let match_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-match-layout"),
            bind_group_layouts: &[&match_bind_group_layout],
            push_constant_ranges: &[],
        });

        let match_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-match-pipeline"),
            layout: Some(&match_layout),
            module: &match_shader,
            entry_point: "main",
        });

        let count_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-count-bgl"),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let count_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-count-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> eq_prev: array<u32>;

@group(0) @binding(1)
var<storage, read_write> starts: array<u32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 8388480u);
    if (idx >= params.len) {
        return;
    }

    let block_start = select(0u, 1u, params.block_size > 0u && (idx % params.block_size) == 0u);
    let is_start = select(0u, 1u, idx == 0u || block_start == 1u || eq_prev[idx] == 0u);
    starts[idx] = is_start;
}
"#,
            )),
        });

        let count_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-count-layout"),
            bind_group_layouts: &[&count_bind_group_layout],
            push_constant_ranges: &[],
        });

        let count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-count-pipeline"),
            layout: Some(&count_layout),
            module: &count_shader,
            entry_point: "main",
        });

        let prefix_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-prefix-bgl"),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let prefix_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-prefix-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> starts: array<u32>;

@group(0) @binding(1)
var<storage, read_write> prefix: array<u32>;

@group(0) @binding(2)
var<storage, read_write> total: array<u32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) {
        return;
    }

    var sum: u32 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= params.len) {
            break;
        }
        prefix[i] = sum;
        sum = sum + starts[i];
        i = i + 1u;
    }
    total[0] = sum;
}
"#,
            )),
        });

        let prefix_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-prefix-layout"),
            bind_group_layouts: &[&prefix_bind_group_layout],
            push_constant_ranges: &[],
        });

        let prefix_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-prefix-pipeline"),
            layout: Some(&prefix_layout),
            module: &prefix_shader,
            entry_point: "main",
        });

        let scan_blocks_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-scan-blocks-bgl"),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let scan_blocks_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-scan-blocks-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> input_data: array<u32>;

@group(0) @binding(1)
var<storage, read_write> prefix_data: array<u32>;

@group(0) @binding(2)
var<storage, read_write> block_sums: array<u32>;

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> scratch: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let lid = local_id.x;
    let gid = wg_id.x + (wg_id.y * 65535u);
    let idx = gid * 256u + lid;
    let value = select(0u, input_data[idx], idx < params.len);

    scratch[lid] = value;
    workgroupBarrier();

    var offset: u32 = 1u;
    loop {
        if (offset >= 256u) {
            break;
        }
        var addend: u32 = 0u;
        if (lid >= offset) {
            addend = scratch[lid - offset];
        }
        workgroupBarrier();
        scratch[lid] = scratch[lid] + addend;
        workgroupBarrier();
        offset = offset << 1u;
    }

    if (idx < params.len) {
        prefix_data[idx] = scratch[lid] - value;
    }

    if (lid == 255u && (gid * 256u) < params.len) {
        block_sums[gid] = scratch[255u];
    }
}
"#,
            )),
        });

        let scan_blocks_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-scan-blocks-layout"),
            bind_group_layouts: &[&scan_blocks_bind_group_layout],
            push_constant_ranges: &[],
        });

        let scan_blocks_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-scan-blocks-pipeline"),
            layout: Some(&scan_blocks_layout),
            module: &scan_blocks_shader,
            entry_point: "main",
        });

        let scan_add_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-scan-add-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let scan_add_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-scan-add-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read_write> prefix_data: array<u32>;

@group(0) @binding(1)
var<storage, read> block_offsets: array<u32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 16776960u);
    if (idx >= params.len) {
        return;
    }

    let block_idx = idx / 256u;
    prefix_data[idx] = prefix_data[idx] + block_offsets[block_idx];
}
"#,
            )),
        });

        let scan_add_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-scan-add-layout"),
            bind_group_layouts: &[&scan_add_bind_group_layout],
            push_constant_ranges: &[],
        });

        let scan_add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-scan-add-pipeline"),
            layout: Some(&scan_add_layout),
            module: &scan_add_shader,
            entry_point: "main",
        });

        let emit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-emit-bgl"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let emit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-emit-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> starts: array<u32>;

@group(0) @binding(1)
var<storage, read> prefix: array<u32>;

@group(0) @binding(2)
var<storage, read_write> positions: array<u32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 8388480u);
    if (idx >= params.len) {
        return;
    }

    if (starts[idx] == 1u) {
        let pos = prefix[idx];
        positions[pos] = idx;
    }
}
"#,
            )),
        });

        let emit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-emit-layout"),
            bind_group_layouts: &[&emit_bind_group_layout],
            push_constant_ranges: &[],
        });

        let emit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-emit-pipeline"),
            layout: Some(&emit_layout),
            module: &emit_shader,
            entry_point: "main",
        });

        let deflate_header_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-header"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&deflate_header_buffer, 0, bytemuck::bytes_of(&0b011_u32));

        Ok(Self {
            device,
            queue,
            tokenize_bind_group_layout,
            tokenize_pipeline,
            token_finalize_pipeline,
            freq_bind_group_layout,
            freq_pipeline,
            dyn_map_bind_group_layout,
            dyn_map_pipeline,
            dyn_finalize_bind_group_layout,
            dyn_finalize_pipeline,
            litlen_bind_group_layout,
            litlen_pipeline,
            bitpack_bind_group_layout,
            bitpack_pipeline,
            match_bind_group_layout,
            match_pipeline,
            count_bind_group_layout,
            count_pipeline,
            prefix_bind_group_layout,
            prefix_pipeline,
            scan_blocks_bind_group_layout,
            scan_blocks_pipeline,
            scan_add_bind_group_layout,
            scan_add_pipeline,
            emit_bind_group_layout,
            emit_pipeline,
            deflate_slots: Mutex::new(Vec::new()),
            deflate_header_buffer,
        })
    }

    fn dispatch_parallel_prefix_scan(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_buffer: &wgpu::Buffer,
        prefix_buffer: &wgpu::Buffer,
        total_buffer: &wgpu::Buffer,
        len: usize,
        label: &str,
    ) -> Result<(), CozipDeflateError> {
        if len == 0 {
            return Ok(());
        }

        let blocks = len.div_ceil(PREFIX_SCAN_BLOCK_SIZE);
        let block_storage_size = bytes_len::<u32>(blocks)?;

        let block_sums_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-scan-block-sums"),
            size: block_storage_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [
            u32::try_from(len).map_err(|_| CozipDeflateError::DataTooLarge)?,
            u32::try_from(PREFIX_SCAN_BLOCK_SIZE).map_err(|_| CozipDeflateError::DataTooLarge)?,
            0,
            0,
        ];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-scan-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

        let scan_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-scan-blocks-bg"),
            layout: &self.scan_blocks_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prefix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: block_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let block_groups =
                u32::try_from(blocks).map_err(|_| CozipDeflateError::DataTooLarge)?;
            let (dispatch_x, dispatch_y) = dispatch_grid_for_groups(block_groups);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-scan-blocks-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scan_blocks_pipeline);
            pass.set_bind_group(0, &scan_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        if blocks == 1 {
            encoder.copy_buffer_to_buffer(&block_sums_buffer, 0, total_buffer, 0, 4);
            return Ok(());
        }

        let block_prefix_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-scan-block-prefix"),
            size: block_storage_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let block_total_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-scan-block-total"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&block_total_buffer, 0, bytemuck::bytes_of(&0_u32));

        self.dispatch_parallel_prefix_scan(
            encoder,
            &block_sums_buffer,
            &block_prefix_buffer,
            &block_total_buffer,
            blocks,
            label,
        )?;

        let add_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-scan-add-bg"),
            layout: &self.scan_add_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: block_prefix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let (dispatch_x, dispatch_y) = dispatch_grid_for_items(len, PREFIX_SCAN_BLOCK_SIZE)?;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-scan-add-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scan_add_pipeline);
            pass.set_bind_group(0, &add_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        encoder.copy_buffer_to_buffer(&block_total_buffer, 0, total_buffer, 0, 4);
        let _ = label;
        Ok(())
    }

    fn create_deflate_slot(&self, len_capacity: usize) -> Result<DeflateSlot, CozipDeflateError> {
        let len_u32 = u32::try_from(len_capacity).map_err(|_| CozipDeflateError::DataTooLarge)?;
        let input_words = len_capacity.div_ceil(std::mem::size_of::<u32>());
        let input_storage_size = bytes_len::<u32>(input_words)?;
        let lane_storage_size = bytes_len::<u32>(len_capacity)?;
        let max_total_bits = len_capacity
            .checked_mul(GPU_DEFLATE_MAX_BITS_PER_BYTE)
            .and_then(|value| value.checked_add(2048))
            .ok_or(CozipDeflateError::DataTooLarge)?;
        let output_words = max_total_bits.div_ceil(32);
        let output_storage_size = bytes_len::<u32>(output_words)?;

        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-input"),
            size: input_storage_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let codes_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-codes"),
            size: lane_storage_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let token_flags_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-token-flags"),
            size: lane_storage_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let token_kind_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-token-kind"),
            size: lane_storage_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let token_len_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-token-len"),
            size: lane_storage_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let token_dist_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-token-dist"),
            size: lane_storage_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let token_lit_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-token-lit"),
            size: lane_storage_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let litlen_freq_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-litlen-freq"),
            size: bytes_len::<u32>(LITLEN_SYMBOL_COUNT)?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dist_freq_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-dist-freq"),
            size: bytes_len::<u32>(DIST_SYMBOL_COUNT)?,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dyn_table_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-dyn-table"),
            size: bytes_len::<u32>(DYN_TABLE_U32_COUNT)?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dyn_meta_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-dyn-meta"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dyn_overflow_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-dyn-overflow"),
            size: lane_storage_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let token_prefix_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-token-prefix"),
            size: lane_storage_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let token_total_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-token-total"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bitlens_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-bitlens"),
            size: lane_storage_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bit_offsets_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-bit-offsets"),
            size: lane_storage_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let total_bits_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-total-bits"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_words_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-out-words"),
            size: output_storage_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_size = output_storage_size
            .checked_add(4)
            .ok_or(CozipDeflateError::DataTooLarge)?;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-deflate-readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let litlen_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-deflate-litlen-bg"),
            layout: &self.litlen_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: token_flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: token_kind_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: token_len_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: token_dist_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: token_lit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: token_prefix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: codes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: bitlens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let tokenize_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-deflate-tokenize-bg"),
            layout: &self.tokenize_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: token_flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: token_kind_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: token_len_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: token_dist_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: token_lit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let freq_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-deflate-freq-bg"),
            layout: &self.freq_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: token_flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: token_kind_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: token_len_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: token_dist_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: token_lit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: litlen_freq_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dist_freq_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let bitpack_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-deflate-bitpack-bg"),
            layout: &self.bitpack_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: codes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dyn_overflow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bitlens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bit_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_words_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let dyn_map_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-deflate-dyn-map-bg"),
            layout: &self.dyn_map_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: token_flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: token_len_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: token_dist_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: token_lit_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dyn_table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: codes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dyn_overflow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: bitlens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let dyn_finalize_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-deflate-dyn-finalize-bg"),
            layout: &self.dyn_finalize_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_words_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: total_bits_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dyn_meta_buffer.as_entire_binding(),
                },
            ],
        });

        // Set static params for maximum-capacity slot; actual len is set per dispatch call.
        let params = [
            len_u32,
            u32::try_from(TOKEN_FINALIZE_SEGMENT_SIZE)
                .map_err(|_| CozipDeflateError::DataTooLarge)?,
            0,
            3,
        ];
        self.queue
            .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

        Ok(DeflateSlot {
            len_capacity,
            output_storage_size,
            input_buffer,
            codes_buffer,
            token_flags_buffer,
            token_kind_buffer,
            token_len_buffer,
            token_dist_buffer,
            token_lit_buffer,
            litlen_freq_buffer,
            dist_freq_buffer,
            dyn_table_buffer,
            dyn_meta_buffer,
            dyn_overflow_buffer,
            token_prefix_buffer,
            token_total_buffer,
            bitlens_buffer,
            bit_offsets_buffer,
            total_bits_buffer,
            output_words_buffer,
            params_buffer,
            readback,
            litlen_bg,
            tokenize_bg,
            freq_bg,
            dyn_map_bg,
            dyn_finalize_bg,
            bitpack_bg,
        })
    }

    fn run_start_positions(
        &self,
        data: &[u8],
        block_size: usize,
    ) -> Result<Vec<usize>, CozipDeflateError> {
        let mut batch = self.run_start_positions_batch(&[data], block_size)?;
        Ok(batch.pop().unwrap_or_default())
    }

    fn run_start_positions_batch(
        &self,
        chunks: &[&[u8]],
        block_size: usize,
    ) -> Result<Vec<Vec<usize>>, CozipDeflateError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        struct PendingReadback {
            chunk_index: usize,
            len: usize,
            total_readback: wgpu::Buffer,
            positions_readback: wgpu::Buffer,
        }

        let mut results = vec![Vec::new(); chunks.len()];
        let mut pending = Vec::with_capacity(chunks.len());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-run-batch-encoder"),
            });
        let mut dispatched = false;

        for (chunk_index, data) in chunks.iter().enumerate() {
            if data.is_empty() {
                continue;
            }

            let len = data.len();
            let input_words = len.div_ceil(std::mem::size_of::<u32>());
            let input_storage_size = bytes_len::<u32>(input_words)?;
            let storage_size = bytes_len::<u32>(len)?;

            let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-run-input"),
                size: input_storage_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if data.len() % std::mem::size_of::<u32>() == 0 {
                self.queue.write_buffer(&input_buffer, 0, data);
            } else {
                let mut padded = vec![0_u8; input_words * std::mem::size_of::<u32>()];
                padded[..data.len()].copy_from_slice(data);
                self.queue.write_buffer(&input_buffer, 0, &padded);
            }

            let eq_prev_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-run-eq-prev"),
                size: storage_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let starts_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-run-starts"),
                size: storage_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let prefix_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-run-prefix"),
                size: storage_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let positions_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-run-positions"),
                size: storage_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let total_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-run-total"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&total_buffer, 0, &[0_u8, 0, 0, 0]);

            let params = [
                u32::try_from(len).map_err(|_| CozipDeflateError::DataTooLarge)?,
                u32::try_from(block_size.max(1)).map_err(|_| CozipDeflateError::DataTooLarge)?,
                0,
                0,
            ];
            let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-run-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

            let total_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-run-total-readback"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let positions_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-run-pos-readback"),
                size: storage_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let match_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-run-match-bg"),
                layout: &self.match_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: eq_prev_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let count_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-run-count-bg"),
                layout: &self.count_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: eq_prev_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: starts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let prefix_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-run-prefix-bg"),
                layout: &self.prefix_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: starts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: prefix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: total_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let emit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-run-emit-bg"),
                layout: &self.emit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: starts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: prefix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: positions_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            {
                let (dispatch_x, dispatch_y) = dispatch_grid_for_items(len, 128)?;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-run-match-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.match_pipeline);
                pass.set_bind_group(0, &match_bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            {
                let (dispatch_x, dispatch_y) = dispatch_grid_for_items(len, 128)?;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-run-count-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.count_pipeline);
                pass.set_bind_group(0, &count_bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-run-prefix-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.prefix_pipeline);
                pass.set_bind_group(0, &prefix_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            {
                let (dispatch_x, dispatch_y) = dispatch_grid_for_items(len, 128)?;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-run-emit-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.emit_pipeline);
                pass.set_bind_group(0, &emit_bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            encoder.copy_buffer_to_buffer(&total_buffer, 0, &total_readback, 0, 4);
            encoder.copy_buffer_to_buffer(&positions_buffer, 0, &positions_readback, 0, storage_size);

            pending.push(PendingReadback {
                chunk_index,
                len,
                total_readback,
                positions_readback,
            });
            dispatched = true;
        }

        if !dispatched {
            return Ok(results);
        }

        self.queue.submit(Some(encoder.finish()));

        let mut total_receivers = Vec::with_capacity(pending.len());
        let mut pos_receivers = Vec::with_capacity(pending.len());
        for item in &pending {
            let (tx_total, rx_total) = std::sync::mpsc::channel();
            item.total_readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx_total.send(result);
                });
            total_receivers.push(rx_total);

            let (tx_pos, rx_pos) = std::sync::mpsc::channel();
            item.positions_readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx_pos.send(result);
                });
            pos_receivers.push(rx_pos);
        }

        self.device.poll(wgpu::Maintain::Wait);

        for (item, (total_rx, pos_rx)) in pending
            .into_iter()
            .zip(total_receivers.into_iter().zip(pos_receivers.into_iter()))
        {
            match total_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                Err(err) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
            }

            match pos_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                Err(err) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
            }

            let total_mapped = item.total_readback.slice(..).get_mapped_range();
            let total_words: &[u32] = bytemuck::cast_slice(&total_mapped);
            let total_count = total_words.first().copied().unwrap_or(0) as usize;
            drop(total_mapped);
            item.total_readback.unmap();

            let pos_mapped = item.positions_readback.slice(..).get_mapped_range();
            let pos_words: &[u32] = bytemuck::cast_slice(&pos_mapped);
            let capped = total_count.min(pos_words.len());
            let mut out = Vec::with_capacity(capped);
            for value in &pos_words[..capped] {
                out.push(*value as usize);
            }
            drop(pos_mapped);
            item.positions_readback.unmap();

            out.sort_unstable();
            out.dedup();
            out.retain(|idx| *idx < item.len);

            if out.first().copied() != Some(0) {
                out.insert(0, 0);
            }

            results[item.chunk_index] = out;
        }

        Ok(results)
    }

    fn deflate_fixed_literals_batch(
        &self,
        chunks: &[&[u8]],
        mode: CompressionMode,
        compression_level: u32,
    ) -> Result<Vec<Vec<u8>>, CozipDeflateError> {
        if mode == CompressionMode::Ratio {
            return self.deflate_dynamic_hybrid_batch(chunks, mode, compression_level);
        }
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = vec![Vec::new(); chunks.len()];
        let mut slots = lock(&self.deflate_slots)?;
        let mut work_indices = Vec::with_capacity(chunks.len());
        for (chunk_index, data) in chunks.iter().enumerate() {
            if data.is_empty() {
                results[chunk_index] = vec![0x03, 0x00];
                continue;
            }
            work_indices.push(chunk_index);
            if slots.len() <= chunk_index {
                slots.push(self.create_deflate_slot(data.len())?);
            } else if slots[chunk_index].len_capacity < data.len() {
                slots[chunk_index] = self.create_deflate_slot(data.len())?;
            }
        }

        for chunk_group in work_indices.chunks(GPU_PIPELINED_SUBMIT_CHUNKS.max(1)) {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("cozip-deflate-batch-encoder"),
                });

            for &chunk_index in chunk_group {
                let data = chunks[chunk_index];
                let len = data.len();
                let len_u32 = u32::try_from(len).map_err(|_| CozipDeflateError::DataTooLarge)?;
                let slot = &mut slots[chunk_index];
                let input_words = len.div_ceil(std::mem::size_of::<u32>());

                if data.len() % std::mem::size_of::<u32>() == 0 {
                    self.queue.write_buffer(&slot.input_buffer, 0, data);
                } else {
                    let mut padded = vec![0_u8; input_words * std::mem::size_of::<u32>()];
                    padded[..data.len()].copy_from_slice(data);
                    self.queue.write_buffer(&slot.input_buffer, 0, &padded);
                }
                let params = [
                    len_u32,
                    u32::try_from(TOKEN_FINALIZE_SEGMENT_SIZE)
                        .map_err(|_| CozipDeflateError::DataTooLarge)?,
                    compression_mode_id(mode),
                    3,
                ];
                self.queue
                    .write_buffer(&slot.params_buffer, 0, bytemuck::cast_slice(&params));

                encoder.clear_buffer(&slot.token_total_buffer, 0, None);
                encoder.clear_buffer(&slot.bitlens_buffer, 0, None);
                encoder.clear_buffer(&slot.total_bits_buffer, 0, None);
                encoder.clear_buffer(&slot.dyn_overflow_buffer, 0, None);
                encoder.clear_buffer(&slot.output_words_buffer, 0, None);
                encoder.copy_buffer_to_buffer(
                    &self.deflate_header_buffer,
                    0,
                    &slot.output_words_buffer,
                    0,
                    4,
                );

                {
                    let (dispatch_x, dispatch_y) = dispatch_grid_for_items(len, 128)?;
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-deflate-tokenize-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.tokenize_pipeline);
                    pass.set_bind_group(0, &slot.tokenize_bg, &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }

                {
                    let (dispatch_x, dispatch_y) =
                        dispatch_grid_for_items(len, TOKEN_FINALIZE_SEGMENT_SIZE)?;
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-deflate-token-finalize-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.token_finalize_pipeline);
                    pass.set_bind_group(0, &slot.tokenize_bg, &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }

                self.dispatch_parallel_prefix_scan(
                    &mut encoder,
                    &slot.token_flags_buffer,
                    &slot.token_prefix_buffer,
                    &slot.token_total_buffer,
                    len,
                    "token-prefix",
                )?;

                {
                    let (dispatch_x, dispatch_y) = dispatch_grid_for_items(len, 128)?;
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-deflate-litlen-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.litlen_pipeline);
                    pass.set_bind_group(0, &slot.litlen_bg, &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }

                self.dispatch_parallel_prefix_scan(
                    &mut encoder,
                    &slot.bitlens_buffer,
                    &slot.bit_offsets_buffer,
                    &slot.total_bits_buffer,
                    len,
                    "bitlen-prefix",
                )?;

                {
                    let (dispatch_x, dispatch_y) = dispatch_grid_for_items(len, 128)?;
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-deflate-bitpack-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.bitpack_pipeline);
                    pass.set_bind_group(0, &slot.bitpack_bg, &[]);
                    pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }

                encoder.copy_buffer_to_buffer(&slot.total_bits_buffer, 0, &slot.readback, 0, 4);
            }

            self.queue.submit(Some(encoder.finish()));

            let mut bit_receivers = Vec::with_capacity(chunk_group.len());
            for &chunk_index in chunk_group {
                let (tx, rx) = std::sync::mpsc::channel();
                slots[chunk_index]
                    .readback
                    .slice(0..4)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = tx.send(result);
                    });
                bit_receivers.push((chunk_index, rx));
            }
            self.device.poll(wgpu::Maintain::Wait);

            let mut payload_jobs: Vec<(usize, usize, u64)> = Vec::new();
            let mut cpu_fallback = Vec::new();
            for (chunk_index, rx) in bit_receivers {
                match rx.recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                    Err(err) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                }
                let mapped = slots[chunk_index].readback.slice(0..4).get_mapped_range();
                let total_words: &[u32] = bytemuck::cast_slice(&mapped[..4]);
                let literal_bits = total_words.first().copied().unwrap_or(0) as usize;
                let total_bits = literal_bits
                    .checked_add(10)
                    .ok_or(CozipDeflateError::DataTooLarge)?;
                let total_bytes = total_bits.div_ceil(8);
                drop(mapped);
                slots[chunk_index].readback.unmap();

                let output_storage_size = usize::try_from(slots[chunk_index].output_storage_size)
                    .map_err(|_| CozipDeflateError::DataTooLarge)?;
                if total_bytes > output_storage_size {
                    cpu_fallback.push(chunk_index);
                    continue;
                }

                let copy_words = total_bytes.div_ceil(std::mem::size_of::<u32>());
                let copy_size = bytes_len::<u32>(copy_words)?;
                payload_jobs.push((chunk_index, total_bytes, copy_size));
            }

            if !payload_jobs.is_empty() {
                let mut payload_encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("cozip-deflate-payload-readback-encoder"),
                    });
                for (chunk_index, _total_bytes, copy_size) in &payload_jobs {
                    let slot = &slots[*chunk_index];
                    payload_encoder.copy_buffer_to_buffer(
                        &slot.output_words_buffer,
                        0,
                        &slot.readback,
                        0,
                        *copy_size,
                    );
                }
                self.queue.submit(Some(payload_encoder.finish()));

                let mut payload_receivers = Vec::with_capacity(payload_jobs.len());
                for (chunk_index, _total_bytes, copy_size) in &payload_jobs {
                    let (tx, rx) = std::sync::mpsc::channel();
                    slots[*chunk_index]
                        .readback
                        .slice(0..*copy_size)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            let _ = tx.send(result);
                        });
                    payload_receivers.push(rx);
                }
                self.device.poll(wgpu::Maintain::Wait);

                for ((chunk_index, total_bytes, copy_size), rx) in
                    payload_jobs.into_iter().zip(payload_receivers.into_iter())
                {
                    match rx.recv() {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => {
                            return Err(CozipDeflateError::GpuExecution(err.to_string()));
                        }
                        Err(err) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                    }
                    let mapped = slots[chunk_index].readback.slice(0..copy_size).get_mapped_range();
                    let mut compressed = Vec::with_capacity(total_bytes);
                    compressed.extend_from_slice(&mapped[..total_bytes]);
                    drop(mapped);
                    slots[chunk_index].readback.unmap();
                    results[chunk_index] = compressed;
                }
            }

            for chunk_index in cpu_fallback {
                results[chunk_index] =
                    deflate_compress_cpu(chunks[chunk_index], compression_level)?;
            }
        }

        Ok(results)
    }

    fn deflate_dynamic_hybrid_batch(
        &self,
        chunks: &[&[u8]],
        mode: CompressionMode,
        compression_level: u32,
    ) -> Result<Vec<Vec<u8>>, CozipDeflateError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        struct PendingDynFreqReadback {
            chunk_index: usize,
            slot_index: usize,
            litlen_freq_readback: wgpu::Buffer,
            dist_freq_readback: wgpu::Buffer,
            litlen_receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
            dist_receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
        }

        struct PreparedDynamicPack {
            chunk_index: usize,
            slot_index: usize,
            len_u32: u32,
            header_bits: u32,
            header_bytes: Vec<u8>,
            dyn_table: Vec<u32>,
            eob_code: u16,
            eob_bits: u8,
        }

        struct PendingDynPackBitsReadback {
            chunk_index: usize,
            slot_index: usize,
            receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
        }

        let mut results = vec![Vec::new(); chunks.len()];
        let mut slots = lock(&self.deflate_slots)?;
        let litlen_freq_size = bytes_len::<u32>(LITLEN_SYMBOL_COUNT)?;
        let dist_freq_size = bytes_len::<u32>(DIST_SYMBOL_COUNT)?;

        let mut freq_pending: Vec<PendingDynFreqReadback> = Vec::new();
        let mut staged_freq_readbacks: Vec<(usize, usize, wgpu::Buffer, wgpu::Buffer)> = Vec::new();
        let mut freq_submit_chunk_count = 0usize;
        let mut freq_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-deflate-dynamic-freq-batch-encoder"),
            });

        for (chunk_index, data) in chunks.iter().enumerate() {
            if data.is_empty() {
                results[chunk_index] = vec![0x03, 0x00];
                continue;
            }

            let len = data.len();
            let len_u32 = u32::try_from(len).map_err(|_| CozipDeflateError::DataTooLarge)?;
            if slots.len() <= chunk_index {
                slots.push(self.create_deflate_slot(len)?);
            } else if slots[chunk_index].len_capacity < len {
                slots[chunk_index] = self.create_deflate_slot(len)?;
            }

            let slot = &slots[chunk_index];
            let slot_index = chunk_index;

            let input_words = len.div_ceil(std::mem::size_of::<u32>());
            if data.len() % std::mem::size_of::<u32>() == 0 {
                self.queue.write_buffer(&slot.input_buffer, 0, data);
            } else {
                let mut padded = vec![0_u8; input_words * std::mem::size_of::<u32>()];
                padded[..data.len()].copy_from_slice(data);
                self.queue.write_buffer(&slot.input_buffer, 0, &padded);
            }

            let params = [
                len_u32,
                u32::try_from(TOKEN_FINALIZE_SEGMENT_SIZE)
                    .map_err(|_| CozipDeflateError::DataTooLarge)?,
                compression_mode_id(mode),
                0,
            ];
            self.queue
                .write_buffer(&slot.params_buffer, 0, bytemuck::cast_slice(&params));

            let litlen_freq_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-deflate-litlen-freq-rb"),
                size: litlen_freq_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let dist_freq_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-deflate-dist-freq-rb"),
                size: dist_freq_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            freq_encoder.clear_buffer(&slot.litlen_freq_buffer, 0, None);
            freq_encoder.clear_buffer(&slot.dist_freq_buffer, 0, None);

            {
                let (dispatch_x, dispatch_y) = dispatch_grid_for_items(len, 128)?;
                let mut pass = freq_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-deflate-tokenize-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.tokenize_pipeline);
                pass.set_bind_group(0, &slot.tokenize_bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            {
                let (dispatch_x, dispatch_y) =
                    dispatch_grid_for_items(len, TOKEN_FINALIZE_SEGMENT_SIZE)?;
                let mut pass = freq_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-deflate-token-finalize-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.token_finalize_pipeline);
                pass.set_bind_group(0, &slot.tokenize_bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            {
                let (dispatch_x, dispatch_y) = dispatch_grid_for_items(len, 128)?;
                let mut pass = freq_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-deflate-freq-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.freq_pipeline);
                pass.set_bind_group(0, &slot.freq_bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            freq_encoder.copy_buffer_to_buffer(
                &slot.litlen_freq_buffer,
                0,
                &litlen_freq_readback,
                0,
                litlen_freq_size,
            );
            freq_encoder.copy_buffer_to_buffer(
                &slot.dist_freq_buffer,
                0,
                &dist_freq_readback,
                0,
                dist_freq_size,
            );

            staged_freq_readbacks.push((
                chunk_index,
                slot_index,
                litlen_freq_readback,
                dist_freq_readback,
            ));
            freq_submit_chunk_count += 1;

            if freq_submit_chunk_count >= GPU_PIPELINED_SUBMIT_CHUNKS {
                self.queue.submit(Some(freq_encoder.finish()));
                for (pending_chunk_index, pending_slot_index, lit_rb, dist_rb) in
                    staged_freq_readbacks.drain(..)
                {
                    let (lit_tx, lit_rx) = std::sync::mpsc::channel();
                    lit_rb.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                        let _ = lit_tx.send(result);
                    });
                    let (dist_tx, dist_rx) = std::sync::mpsc::channel();
                    dist_rb.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                        let _ = dist_tx.send(result);
                    });
                    freq_pending.push(PendingDynFreqReadback {
                        chunk_index: pending_chunk_index,
                        slot_index: pending_slot_index,
                        litlen_freq_readback: lit_rb,
                        dist_freq_readback: dist_rb,
                        litlen_receiver: lit_rx,
                        dist_receiver: dist_rx,
                    });
                }
                self.device.poll(wgpu::Maintain::Poll);
                freq_encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("cozip-deflate-dynamic-freq-batch-encoder"),
                    });
                freq_submit_chunk_count = 0;
            }
        }

        if freq_submit_chunk_count > 0 {
            self.queue.submit(Some(freq_encoder.finish()));
            for (pending_chunk_index, pending_slot_index, lit_rb, dist_rb) in
                staged_freq_readbacks.drain(..)
            {
                let (lit_tx, lit_rx) = std::sync::mpsc::channel();
                lit_rb.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    let _ = lit_tx.send(result);
                });
                let (dist_tx, dist_rx) = std::sync::mpsc::channel();
                dist_rb.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    let _ = dist_tx.send(result);
                });
                freq_pending.push(PendingDynFreqReadback {
                    chunk_index: pending_chunk_index,
                    slot_index: pending_slot_index,
                    litlen_freq_readback: lit_rb,
                    dist_freq_readback: dist_rb,
                    litlen_receiver: lit_rx,
                    dist_receiver: dist_rx,
                });
            }
        }

        let mut prepared = Vec::with_capacity(freq_pending.len());
        if !freq_pending.is_empty() {
            self.device.poll(wgpu::Maintain::Wait);
        }
        for pending in freq_pending {
            match pending.litlen_receiver.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                Err(err) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
            }
            match pending.dist_receiver.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                Err(err) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
            }

            let litlen_freq_map = pending.litlen_freq_readback.slice(..).get_mapped_range();
            let dist_freq_map = pending.dist_freq_readback.slice(..).get_mapped_range();
            let mut litlen_freq = vec![0_u32; LITLEN_SYMBOL_COUNT];
            let mut dist_freq = vec![0_u32; DIST_SYMBOL_COUNT];
            let litlen_words: &[u32] = bytemuck::cast_slice(&litlen_freq_map);
            let dist_words_freq: &[u32] = bytemuck::cast_slice(&dist_freq_map);
            litlen_freq.copy_from_slice(&litlen_words[..LITLEN_SYMBOL_COUNT]);
            dist_freq.copy_from_slice(&dist_words_freq[..DIST_SYMBOL_COUNT]);
            drop(litlen_freq_map);
            drop(dist_freq_map);
            pending.litlen_freq_readback.unmap();
            pending.dist_freq_readback.unmap();

            let plan = build_dynamic_huffman_plan(&litlen_freq, &dist_freq)?;
            let mut dyn_table = Vec::with_capacity(DYN_TABLE_U32_COUNT);
            dyn_table.extend_from_slice(&plan.litlen_codes);
            dyn_table.extend_from_slice(&plan.litlen_bits);
            dyn_table.extend_from_slice(&plan.dist_codes);
            dyn_table.extend_from_slice(&plan.dist_bits);

            let len_u32 = u32::try_from(chunks[pending.chunk_index].len())
                .map_err(|_| CozipDeflateError::DataTooLarge)?;
            prepared.push(PreparedDynamicPack {
                chunk_index: pending.chunk_index,
                slot_index: pending.slot_index,
                len_u32,
                header_bits: plan.header_bits,
                header_bytes: plan.header_bytes,
                dyn_table,
                eob_code: plan.eob_code,
                eob_bits: plan.eob_bits,
            });
        }

        let mut pack_pending: Vec<PendingDynPackBitsReadback> = Vec::with_capacity(prepared.len());
        let mut staged_pack: Vec<(usize, usize)> = Vec::new();
        let mut pack_submit_chunk_count = 0usize;
        let mut pack_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-deflate-dynamic-pack-batch-encoder"),
            });

        for item in prepared {
            let slot = &slots[item.slot_index];
            self.queue
                .write_buffer(&slot.dyn_table_buffer, 0, bytemuck::cast_slice(&item.dyn_table));
            let meta = [
                item.header_bits,
                u32::from(item.eob_code),
                u32::from(item.eob_bits),
                0,
            ];
            self.queue
                .write_buffer(&slot.dyn_meta_buffer, 0, bytemuck::cast_slice(&meta));
            let params = [
                item.len_u32,
                u32::try_from(TOKEN_FINALIZE_SEGMENT_SIZE)
                    .map_err(|_| CozipDeflateError::DataTooLarge)?,
                compression_mode_id(mode),
                item.header_bits,
            ];
            self.queue
                .write_buffer(&slot.params_buffer, 0, bytemuck::cast_slice(&params));

            let header_words = item.header_bytes.len().div_ceil(std::mem::size_of::<u32>());
            let header_copy_size = bytes_len::<u32>(header_words)?;
            let header_copy_size_usize =
                usize::try_from(header_copy_size).map_err(|_| CozipDeflateError::DataTooLarge)?;
            if header_copy_size > slot.output_storage_size {
                return Err(CozipDeflateError::DataTooLarge);
            }
            let mut header_padded = vec![0_u8; header_copy_size_usize];
            header_padded[..item.header_bytes.len()].copy_from_slice(&item.header_bytes);
            let header_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-deflate-dyn-header"),
                size: header_copy_size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&header_staging, 0, &header_padded);

            pack_encoder.clear_buffer(&slot.token_total_buffer, 0, None);
            pack_encoder.clear_buffer(&slot.bitlens_buffer, 0, None);
            pack_encoder.clear_buffer(&slot.total_bits_buffer, 0, None);
            pack_encoder.clear_buffer(&slot.dyn_overflow_buffer, 0, None);
            pack_encoder.clear_buffer(&slot.output_words_buffer, 0, None);
            pack_encoder.copy_buffer_to_buffer(
                &header_staging,
                0,
                &slot.output_words_buffer,
                0,
                header_copy_size,
            );

            self.dispatch_parallel_prefix_scan(
                &mut pack_encoder,
                &slot.token_flags_buffer,
                &slot.token_prefix_buffer,
                &slot.token_total_buffer,
                item.len_u32 as usize,
                "token-prefix-dynamic",
            )?;

            {
                let (dispatch_x, dispatch_y) =
                    dispatch_grid_for_items(item.len_u32 as usize, 128)?;
                let mut pass = pack_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-deflate-dyn-map-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.dyn_map_pipeline);
                pass.set_bind_group(0, &slot.dyn_map_bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            self.dispatch_parallel_prefix_scan(
                &mut pack_encoder,
                &slot.bitlens_buffer,
                &slot.bit_offsets_buffer,
                &slot.total_bits_buffer,
                item.len_u32 as usize,
                "bitlen-prefix-dynamic",
            )?;

            {
                let (dispatch_x, dispatch_y) =
                    dispatch_grid_for_items(item.len_u32 as usize, 128)?;
                let mut pass = pack_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-deflate-bitpack-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bitpack_pipeline);
                pass.set_bind_group(0, &slot.bitpack_bg, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            {
                let mut pass = pack_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-deflate-dyn-finalize-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.dyn_finalize_pipeline);
                pass.set_bind_group(0, &slot.dyn_finalize_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            pack_encoder.copy_buffer_to_buffer(&slot.total_bits_buffer, 0, &slot.readback, 0, 4);

            staged_pack.push((item.chunk_index, item.slot_index));
            pack_submit_chunk_count += 1;

            if pack_submit_chunk_count >= GPU_PIPELINED_SUBMIT_CHUNKS {
                self.queue.submit(Some(pack_encoder.finish()));
                for (pending_chunk_index, pending_slot_index) in staged_pack.drain(..) {
                    let (tx, rx) = std::sync::mpsc::channel();
                    slots[pending_slot_index]
                        .readback
                        .slice(0..4)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            let _ = tx.send(result);
                        });
                    pack_pending.push(PendingDynPackBitsReadback {
                        chunk_index: pending_chunk_index,
                        slot_index: pending_slot_index,
                        receiver: rx,
                    });
                }
                self.device.poll(wgpu::Maintain::Poll);
                pack_encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("cozip-deflate-dynamic-pack-batch-encoder"),
                    });
                pack_submit_chunk_count = 0;
            }
        }

        if pack_submit_chunk_count > 0 {
            self.queue.submit(Some(pack_encoder.finish()));
            for (pending_chunk_index, pending_slot_index) in staged_pack.drain(..) {
                let (tx, rx) = std::sync::mpsc::channel();
                slots[pending_slot_index]
                    .readback
                    .slice(0..4)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = tx.send(result);
                    });
                pack_pending.push(PendingDynPackBitsReadback {
                    chunk_index: pending_chunk_index,
                    slot_index: pending_slot_index,
                    receiver: rx,
                });
            }
        }

        let mut payload_jobs: Vec<(usize, usize, u64)> = Vec::new();
        let mut cpu_fallback = Vec::new();
        if !pack_pending.is_empty() {
            self.device.poll(wgpu::Maintain::Wait);
        }
        for pending in pack_pending {
            match pending.receiver.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                Err(err) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
            }
            let mapped = slots[pending.slot_index].readback.slice(0..4).get_mapped_range();
            let total_words: &[u32] = bytemuck::cast_slice(&mapped[..4]);
            let total_bits = total_words.first().copied().unwrap_or(0) as usize;
            let total_bytes = total_bits.div_ceil(8);
            drop(mapped);
            slots[pending.slot_index].readback.unmap();

            let output_storage_size = usize::try_from(slots[pending.slot_index].output_storage_size)
                .map_err(|_| CozipDeflateError::DataTooLarge)?;
            if total_bytes > output_storage_size {
                cpu_fallback.push(pending.chunk_index);
                continue;
            }

            let copy_words = total_bytes.div_ceil(std::mem::size_of::<u32>());
            let copy_size = bytes_len::<u32>(copy_words)?;
            payload_jobs.push((pending.chunk_index, total_bytes, copy_size));
        }

        if !payload_jobs.is_empty() {
            let mut payload_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("cozip-deflate-dynamic-payload-readback-encoder"),
                });
            for (chunk_index, _total_bytes, copy_size) in &payload_jobs {
                let slot = &slots[*chunk_index];
                payload_encoder.copy_buffer_to_buffer(
                    &slot.output_words_buffer,
                    0,
                    &slot.readback,
                    0,
                    *copy_size,
                );
            }
            self.queue.submit(Some(payload_encoder.finish()));

            let mut payload_receivers = Vec::with_capacity(payload_jobs.len());
            for (chunk_index, _total_bytes, copy_size) in &payload_jobs {
                let (tx, rx) = std::sync::mpsc::channel();
                slots[*chunk_index]
                    .readback
                    .slice(0..*copy_size)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = tx.send(result);
                    });
                payload_receivers.push(rx);
            }
            self.device.poll(wgpu::Maintain::Wait);

            for ((chunk_index, total_bytes, copy_size), rx) in
                payload_jobs.into_iter().zip(payload_receivers.into_iter())
            {
                match rx.recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                    Err(err) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                }
                let mapped = slots[chunk_index].readback.slice(0..copy_size).get_mapped_range();
                let mut compressed = Vec::with_capacity(total_bytes);
                compressed.extend_from_slice(&mapped[..total_bytes]);
                drop(mapped);
                slots[chunk_index].readback.unmap();
                results[chunk_index] = compressed;
            }
        }

        for chunk_index in cpu_fallback {
            results[chunk_index] = deflate_compress_cpu(chunks[chunk_index], compression_level)?;
        }

        Ok(results)
    }
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

fn bytes_len<T>(items: usize) -> Result<u64, CozipDeflateError> {
    let bytes = items
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CozipDeflateError::DataTooLarge)?;
    u64::try_from(bytes).map_err(|_| CozipDeflateError::DataTooLarge)
}

fn collect_deflate_readback(
    slot: &DeflateSlot,
    output_storage_size: u64,
) -> Result<Vec<u8>, CozipDeflateError> {
    let mapped = slot.readback.slice(..).get_mapped_range();
    if mapped.len() < 4 {
        return Err(CozipDeflateError::Internal("gpu readback too small"));
    }

    let total_words: &[u32] = bytemuck::cast_slice(&mapped[..4]);
    let literal_bits = total_words.first().copied().unwrap_or(0) as usize;
    let total_bits = literal_bits
        .checked_add(10)
        .ok_or(CozipDeflateError::DataTooLarge)?;
    let total_bytes = total_bits.div_ceil(8);

    let output_storage_size =
        usize::try_from(output_storage_size).map_err(|_| CozipDeflateError::DataTooLarge)?;
    if total_bytes > output_storage_size {
        return Err(CozipDeflateError::Internal(
            "gpu compressed output exceeded allocated readback",
        ));
    }

    let payload_start = 4usize;
    let payload_end = payload_start
        .checked_add(output_storage_size)
        .ok_or(CozipDeflateError::DataTooLarge)?;
    if payload_end > mapped.len() {
        return Err(CozipDeflateError::Internal("gpu readback payload out of range"));
    }

    let payload = &mapped[payload_start..payload_end];
    let mut compressed = Vec::with_capacity(total_bytes);
    compressed.extend_from_slice(&payload[..total_bytes]);
    drop(mapped);
    slot.readback.unmap();
    Ok(compressed)
}

fn collect_deflate_readback_dynamic(
    slot: &DeflateSlot,
    output_storage_size: u64,
) -> Result<Vec<u8>, CozipDeflateError> {
    let mapped = slot.readback.slice(..).get_mapped_range();
    if mapped.len() < 4 {
        return Err(CozipDeflateError::Internal("gpu readback too small"));
    }

    let total_words: &[u32] = bytemuck::cast_slice(&mapped[..4]);
    let total_bits = total_words.first().copied().unwrap_or(0) as usize;
    let total_bytes = total_bits.div_ceil(8);

    let output_storage_size =
        usize::try_from(output_storage_size).map_err(|_| CozipDeflateError::DataTooLarge)?;
    if total_bytes > output_storage_size {
        return Err(CozipDeflateError::Internal(
            "gpu compressed output exceeded allocated readback",
        ));
    }

    let payload_start = 4usize;
    let payload_end = payload_start
        .checked_add(output_storage_size)
        .ok_or(CozipDeflateError::DataTooLarge)?;
    if payload_end > mapped.len() {
        return Err(CozipDeflateError::Internal("gpu readback payload out of range"));
    }

    let payload = &mapped[payload_start..payload_end];
    let mut compressed = Vec::with_capacity(total_bytes);
    compressed.extend_from_slice(&payload[..total_bytes]);
    drop(mapped);
    slot.readback.unmap();
    Ok(compressed)
}

fn lock<'a, T>(mutex: &'a Mutex<T>) -> Result<std::sync::MutexGuard<'a, T>, CozipDeflateError> {
    mutex
        .lock()
        .map_err(|_| CozipDeflateError::Internal("mutex poisoned"))
}

static GPU_CONTEXT: OnceLock<Mutex<Option<Arc<GpuAssist>>>> = OnceLock::new();

fn shared_gpu_context(prefer_gpu: bool) -> Option<Arc<GpuAssist>> {
    if !prefer_gpu {
        return None;
    }

    let holder = GPU_CONTEXT.get_or_init(|| Mutex::new(None));
    let mut guard = holder.lock().ok()?;

    if let Some(existing) = guard.as_ref() {
        return Some(existing.clone());
    }

    let created = GpuAssist::new().ok().map(Arc::new);
    *guard = created.clone();
    created
}

pub fn deflate_compress_cpu(input: &[u8], level: u32) -> Result<Vec<u8>, CozipDeflateError> {
    let mut encoder =
        flate2::write::DeflateEncoder::new(Vec::new(), Compression::new(level.clamp(0, 9)));
    encoder.write_all(input)?;
    Ok(encoder.finish()?)
}

pub fn deflate_decompress_cpu(input: &[u8]) -> Result<Vec<u8>, CozipDeflateError> {
    let mut decoder = flate2::write::DeflateDecoder::new(Vec::new());
    decoder.write_all(input)?;
    Ok(decoder.finish()?)
}

pub fn compress_hybrid(
    input: &[u8],
    options: &HybridOptions,
) -> Result<CompressedFrame, CozipDeflateError> {
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

    let gpu_context = if options.gpu_fraction <= 0.0 {
        None
    } else {
        shared_gpu_context(options.prefer_gpu)
    };

    let tasks = build_chunk_tasks(input, options, gpu_context.is_some())?;
    if let Some(gpu) = gpu_context.clone() {
        return compress_hybrid_adaptive_scheduler(input.len(), tasks, options, gpu);
    }

    let chunk_count = tasks.len();
    let queue = Arc::new(Mutex::new(VecDeque::from(tasks)));
    let results = Arc::new(Mutex::new(vec![None; chunk_count]));
    let error = Arc::new(Mutex::new(None::<CozipDeflateError>));

    let cpu_workers = cpu_worker_count(gpu_context.is_some());
    let mut handles = Vec::new();

    for _ in 0..cpu_workers {
        let queue_ref = Arc::clone(&queue);
        let result_ref = Arc::clone(&results);
        let err_ref = Arc::clone(&error);
        let opts = options.clone();
        let gpu_enabled = gpu_context.is_some();

        handles.push(std::thread::spawn(move || {
            compress_cpu_worker(queue_ref, result_ref, err_ref, &opts, gpu_enabled)
        }));
    }

    if let Some(gpu) = gpu_context.clone() {
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

    let stats = summarize_encoded_chunks(&chunks, gpu_context.is_some());
    let frame = encode_frame(input.len(), &chunks)?;

    Ok(CompressedFrame {
        bytes: frame,
        stats,
    })
}

pub fn decompress_hybrid(
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

    let gpu_context = if options.gpu_fraction <= 0.0 {
        None
    } else {
        shared_gpu_context(options.prefer_gpu)
    };

    let stats = summarize_descriptors(&descriptors, gpu_context.is_some());

    let queue = Arc::new(Mutex::new(VecDeque::from(descriptors)));
    let results = Arc::new(Mutex::new(vec![None; stats.chunk_count]));
    let error = Arc::new(Mutex::new(None::<CozipDeflateError>));

    let cpu_workers = cpu_worker_count(gpu_context.is_some());
    let mut handles = Vec::new();

    for _ in 0..cpu_workers {
        let queue_ref = Arc::clone(&queue);
        let result_ref = Arc::clone(&results);
        let err_ref = Arc::clone(&error);
        let opts = options.clone();
        let gpu_enabled = gpu_context.is_some();

        handles.push(std::thread::spawn(move || {
            decompress_cpu_worker(queue_ref, result_ref, err_ref, &opts, gpu_enabled)
        }));
    }

    if let Some(gpu) = gpu_context.clone() {
        let queue_ref = Arc::clone(&queue);
        let result_ref = Arc::clone(&results);
        let err_ref = Arc::clone(&error);
        let opts = options.clone();

        handles.push(std::thread::spawn(move || {
            decompress_gpu_worker(queue_ref, result_ref, err_ref, &opts, gpu)
        }));
    }

    for handle in handles {
        let _ = handle.join();
    }

    if let Some(err) = lock(&error)?.take() {
        return Err(err);
    }

    let mut decoded = Vec::with_capacity(original_len);
    for item in lock(&results)?.drain(..) {
        let chunk = item.ok_or(CozipDeflateError::Internal("missing decompressed chunk"))?;
        decoded.extend_from_slice(&chunk.raw);
    }

    if decoded.len() != original_len {
        return Err(CozipDeflateError::InvalidFrame(
            "decoded size does not match frame header",
        ));
    }

    Ok(DecompressedFrame {
        bytes: decoded,
        stats,
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

    Ok(())
}

fn compression_mode_id(mode: CompressionMode) -> u32 {
    match mode {
        CompressionMode::Speed => 0,
        CompressionMode::Balanced => 1,
        CompressionMode::Ratio => 2,
    }
}

fn should_validate_gpu_chunk(mode: CompressionMode) -> bool {
    mode != CompressionMode::Speed
}

fn gpu_chunk_roundtrip_matches(raw: &[u8], compressed: &[u8]) -> bool {
    match deflate_decompress_cpu(compressed) {
        Ok(decoded) => decoded == raw,
        Err(_) => false,
    }
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

fn compress_hybrid_adaptive_scheduler(
    original_len: usize,
    tasks: Vec<ChunkTask>,
    options: &HybridOptions,
    gpu: Arc<GpuAssist>,
) -> Result<CompressedFrame, CozipDeflateError> {
    let chunk_count = tasks.len();
    let start = Arc::new(Instant::now());
    let now_ms = monotonic_ms(&start);
    let scheduled_tasks = Arc::new(
        tasks
            .into_iter()
            .map(|task| {
                let state = if task.preferred_gpu && task.raw.len() >= options.gpu_min_chunk_size {
                    CompressTaskState::ReservedGpu
                } else {
                    CompressTaskState::Pending
                };
                ScheduledCompressTask {
                    index: task.index,
                    preferred_gpu: task.preferred_gpu,
                    raw: task.raw,
                    state: AtomicU8::new(state as u8),
                    reserved_at_ms: AtomicU64::new(now_ms),
                }
            })
            .collect::<Vec<_>>(),
    );

    let results = Arc::new(Mutex::new(vec![None; chunk_count]));
    let error = Arc::new(Mutex::new(None::<CozipDeflateError>));
    let remaining = Arc::new(AtomicUsize::new(chunk_count));
    let active_cpu = Arc::new(AtomicUsize::new(0));
    let wake = Arc::new((Mutex::new(()), Condvar::new()));
    let cpu_workers = cpu_worker_count(true);

    let mut handles = Vec::new();
    for _ in 0..cpu_workers {
        let tasks_ref = Arc::clone(&scheduled_tasks);
        let results_ref = Arc::clone(&results);
        let err_ref = Arc::clone(&error);
        let rem_ref = Arc::clone(&remaining);
        let active_ref = Arc::clone(&active_cpu);
        let wake_ref = Arc::clone(&wake);
        let opts = options.clone();

        handles.push(std::thread::spawn(move || {
            compress_cpu_worker_adaptive(
                tasks_ref,
                results_ref,
                err_ref,
                rem_ref,
                active_ref,
                wake_ref,
                &opts,
            )
        }));
    }

    {
        let tasks_ref = Arc::clone(&scheduled_tasks);
        let results_ref = Arc::clone(&results);
        let err_ref = Arc::clone(&error);
        let rem_ref = Arc::clone(&remaining);
        let wake_ref = Arc::clone(&wake);
        let opts = options.clone();
        let gpu_ref = Arc::clone(&gpu);

        handles.push(std::thread::spawn(move || {
            compress_gpu_worker_adaptive(tasks_ref, results_ref, err_ref, rem_ref, wake_ref, &opts, gpu_ref)
        }));
    }

    {
        let tasks_ref = Arc::clone(&scheduled_tasks);
        let err_ref = Arc::clone(&error);
        let rem_ref = Arc::clone(&remaining);
        let active_ref = Arc::clone(&active_cpu);
        let wake_ref = Arc::clone(&wake);
        let start_ref = Arc::clone(&start);

        handles.push(std::thread::spawn(move || {
            compress_scheduler_watchdog(
                tasks_ref, err_ref, rem_ref, active_ref, wake_ref, cpu_workers, start_ref,
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

    let stats = summarize_encoded_chunks(&chunks, true);
    let frame = encode_frame(original_len, &chunks)?;

    Ok(CompressedFrame { bytes: frame, stats })
}

fn compress_cpu_worker_adaptive(
    tasks: Arc<Vec<ScheduledCompressTask>>,
    results: Arc<Mutex<Vec<Option<ChunkMember>>>>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    remaining: Arc<AtomicUsize>,
    active_cpu: Arc<AtomicUsize>,
    wake: Arc<(Mutex<()>, Condvar)>,
    options: &HybridOptions,
) {
    loop {
        if has_error(&error) || remaining.load(Ordering::Acquire) == 0 {
            break;
        }

        let Some(task_index) = claim_cpu_task(&tasks, options.gpu_min_chunk_size) else {
            wait_for_scheduler(&wake);
            continue;
        };

        active_cpu.fetch_add(1, Ordering::AcqRel);
        let task = &tasks[task_index];
        let raw_len = match u32::try_from(task.raw.len()) {
            Ok(value) => value,
            Err(_) => {
                set_error(&error, CozipDeflateError::DataTooLarge);
                return;
            }
        };
        let compressed = deflate_compress_cpu(&task.raw, options.compression_level).map(|compressed| {
            ChunkMember {
                index: task.index,
                backend: ChunkBackend::Cpu,
                transform: ChunkTransform::None,
                codec: ChunkCodec::DeflateCpu,
                raw_len,
                compressed,
            }
        });
        active_cpu.fetch_sub(1, Ordering::AcqRel);

        match compressed {
            Ok(encoded) => {
                if let Err(err) = store_encoded_result(&results, encoded) {
                    set_error(&error, err);
                    break;
                }
                finish_scheduled_task(task, &remaining, &wake);
            }
            Err(err) => {
                set_error(&error, err);
                break;
            }
        }
    }
}

fn compress_gpu_worker_adaptive(
    tasks: Arc<Vec<ScheduledCompressTask>>,
    results: Arc<Mutex<Vec<Option<ChunkMember>>>>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    remaining: Arc<AtomicUsize>,
    wake: Arc<(Mutex<()>, Condvar)>,
    options: &HybridOptions,
    gpu: Arc<GpuAssist>,
) {
    loop {
        if has_error(&error) || remaining.load(Ordering::Acquire) == 0 {
            break;
        }

        let batch = claim_gpu_batch_tasks(&tasks, options.gpu_min_chunk_size, GPU_BATCH_CHUNKS);
        if batch.is_empty() {
            wait_for_scheduler(&wake);
            continue;
        }

        let task_data: Vec<&[u8]> = batch.iter().map(|idx| tasks[*idx].raw.as_slice()).collect();
        let compressed_batch = gpu.deflate_fixed_literals_batch(
            &task_data,
            options.compression_mode,
            options.compression_level,
        );

        match compressed_batch {
            Ok(compressed_items) if compressed_items.len() == batch.len() => {
                for (task_index, compressed) in batch.iter().copied().zip(compressed_items.into_iter()) {
                    let task = &tasks[task_index];
                    let raw_len = match u32::try_from(task.raw.len()) {
                        Ok(value) => value,
                        Err(_) => {
                            set_error(&error, CozipDeflateError::DataTooLarge);
                            return;
                        }
                    };
                    let encoded = if should_validate_gpu_chunk(options.compression_mode)
                        && !gpu_chunk_roundtrip_matches(&task.raw, &compressed)
                    {
                        match deflate_compress_cpu(&task.raw, options.compression_level) {
                            Ok(cpu_compressed) => ChunkMember {
                                index: task.index,
                                backend: ChunkBackend::Cpu,
                                transform: ChunkTransform::None,
                                codec: ChunkCodec::DeflateCpu,
                                raw_len,
                                compressed: cpu_compressed,
                            },
                            Err(err) => {
                                set_error(&error, err);
                                return;
                            }
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
                    if let Err(err) = store_encoded_result(&results, encoded) {
                        set_error(&error, err);
                        return;
                    }
                    finish_scheduled_task(task, &remaining, &wake);
                }
            }
            _ => {
                for task_index in batch {
                    let task = &tasks[task_index];
                    let raw_len = match u32::try_from(task.raw.len()) {
                        Ok(value) => value,
                        Err(_) => {
                            set_error(&error, CozipDeflateError::DataTooLarge);
                            return;
                        }
                    };
                    match deflate_compress_cpu(&task.raw, options.compression_level) {
                        Ok(compressed) => {
                            let encoded = ChunkMember {
                                index: task.index,
                                backend: ChunkBackend::Cpu,
                                transform: ChunkTransform::None,
                                codec: ChunkCodec::DeflateCpu,
                                raw_len,
                                compressed,
                            };
                            if let Err(err) = store_encoded_result(&results, encoded) {
                                set_error(&error, err);
                                return;
                            }
                            finish_scheduled_task(task, &remaining, &wake);
                        }
                        Err(err) => {
                            set_error(&error, err);
                            return;
                        }
                    }
                }
            }
        }
    }
}

fn compress_scheduler_watchdog(
    tasks: Arc<Vec<ScheduledCompressTask>>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    remaining: Arc<AtomicUsize>,
    active_cpu: Arc<AtomicUsize>,
    wake: Arc<(Mutex<()>, Condvar)>,
    cpu_workers: usize,
    start: Arc<Instant>,
) {
    loop {
        if has_error(&error) || remaining.load(Ordering::Acquire) == 0 {
            break;
        }

        let idle_cpu = cpu_workers.saturating_sub(active_cpu.load(Ordering::Acquire));
        if idle_cpu > 0 {
            let now_ms = monotonic_ms(&start);
            let mut demoted = 0usize;
            for task in tasks.iter() {
                if demoted >= idle_cpu {
                    break;
                }
                if task.state.load(Ordering::Acquire) == CompressTaskState::ReservedGpu as u8 {
                    let reserved_at = task.reserved_at_ms.load(Ordering::Acquire);
                    if now_ms.saturating_sub(reserved_at) >= GPU_RESERVATION_TIMEOUT_MS
                        && task
                            .state
                            .compare_exchange(
                                CompressTaskState::ReservedGpu as u8,
                                CompressTaskState::Pending as u8,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                    {
                        demoted += 1;
                    }
                }
            }
            if demoted > 0 {
                wake.1.notify_all();
            }
        }

        wait_for_scheduler(&wake);
    }
}

fn monotonic_ms(start: &Instant) -> u64 {
    let elapsed = start.elapsed().as_millis();
    elapsed.min(u128::from(u64::MAX)) as u64
}

fn claim_cpu_task(tasks: &[ScheduledCompressTask], gpu_min_chunk_size: usize) -> Option<usize> {
    for (index, task) in tasks.iter().enumerate() {
        if task.state.load(Ordering::Acquire) == CompressTaskState::Pending as u8
            && (!task.preferred_gpu || task.raw.len() < gpu_min_chunk_size)
            && task
                .state
                .compare_exchange(
                    CompressTaskState::Pending as u8,
                    CompressTaskState::RunningCpu as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            return Some(index);
        }
    }

    for (index, task) in tasks.iter().enumerate() {
        if task.state.load(Ordering::Acquire) == CompressTaskState::Pending as u8
            && task
                .state
                .compare_exchange(
                    CompressTaskState::Pending as u8,
                    CompressTaskState::RunningCpu as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            return Some(index);
        }
    }

    None
}

fn claim_gpu_batch_tasks(
    tasks: &[ScheduledCompressTask],
    gpu_min_chunk_size: usize,
    max_batch_chunks: usize,
) -> Vec<usize> {
    let mut batch = Vec::with_capacity(max_batch_chunks.max(1));
    let batch_limit = max_batch_chunks.max(1);

    for (index, task) in tasks.iter().enumerate() {
        if batch.len() >= batch_limit {
            break;
        }
        if task.state.load(Ordering::Acquire) == CompressTaskState::ReservedGpu as u8
            && task
                .state
                .compare_exchange(
                    CompressTaskState::ReservedGpu as u8,
                    CompressTaskState::RunningGpu as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            batch.push(index);
        }
    }

    for (index, task) in tasks.iter().enumerate() {
        if batch.len() >= batch_limit {
            break;
        }
        if task.raw.len() < gpu_min_chunk_size {
            continue;
        }
        if task.state.load(Ordering::Acquire) == CompressTaskState::Pending as u8
            && task
                .state
                .compare_exchange(
                    CompressTaskState::Pending as u8,
                    CompressTaskState::RunningGpu as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            batch.push(index);
        }
    }

    batch
}

fn finish_scheduled_task(
    task: &ScheduledCompressTask,
    remaining: &AtomicUsize,
    wake: &(Mutex<()>, Condvar),
) {
    task.state.store(CompressTaskState::Done as u8, Ordering::Release);
    remaining.fetch_sub(1, Ordering::AcqRel);
    wake.1.notify_all();
}

fn wait_for_scheduler(wake: &(Mutex<()>, Condvar)) {
    if let Ok(guard) = wake.0.lock() {
        let _ = wake
            .1
            .wait_timeout(guard, Duration::from_millis(SCHEDULER_WAIT_MS));
    }
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

fn summarize_descriptors(descriptors: &[ChunkDescriptor], gpu_available: bool) -> HybridStats {
    let mut stats = HybridStats {
        chunk_count: descriptors.len(),
        cpu_chunks: 0,
        gpu_chunks: 0,
        gpu_available,
        cpu_bytes: 0,
        gpu_bytes: 0,
    };

    for descriptor in descriptors {
        match descriptor.backend {
            ChunkBackend::Cpu => {
                stats.cpu_chunks += 1;
                stats.cpu_bytes += descriptor.raw_len as usize;
            }
            ChunkBackend::GpuAssisted => {
                stats.gpu_chunks += 1;
                stats.gpu_bytes += descriptor.raw_len as usize;
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
    let compressed_batch = gpu.deflate_fixed_literals_batch(
        &task_data,
        options.compression_mode,
        options.compression_level,
    )?;

    if compressed_batch.len() != tasks.len() {
        return Err(CozipDeflateError::Internal(
            "gpu batch returned mismatched compressed vectors",
        ));
    }

    let mut out = Vec::with_capacity(tasks.len());
    for (task, compressed) in tasks.iter().zip(compressed_batch.into_iter()) {
        let raw_len = u32::try_from(task.raw.len()).map_err(|_| CozipDeflateError::DataTooLarge)?;
        let member = if should_validate_gpu_chunk(options.compression_mode)
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

fn decompress_cpu_worker(
    queue: Arc<Mutex<VecDeque<ChunkDescriptor>>>,
    results: Arc<Mutex<Vec<Option<DecodedChunk>>>>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    options: &HybridOptions,
    gpu_enabled: bool,
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
            pop_cpu_descriptor(&mut guard, gpu_enabled)
        };

        let Some(descriptor) = descriptor else { break };

        match decode_descriptor_cpu(descriptor, options) {
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

fn decompress_gpu_worker(
    queue: Arc<Mutex<VecDeque<ChunkDescriptor>>>,
    results: Arc<Mutex<Vec<Option<DecodedChunk>>>>,
    error: Arc<Mutex<Option<CozipDeflateError>>>,
    options: &HybridOptions,
    gpu: Arc<GpuAssist>,
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
            pop_gpu_descriptor(&mut guard)
        };

        let Some(descriptor) = descriptor else { break };

        let decoded = decode_descriptor_gpu(descriptor.clone(), options, &gpu)
            .or_else(|_| decode_descriptor_cpu(descriptor, options));

        match decoded {
            Ok(value) => {
                if let Err(err) = store_decoded_result(&results, value) {
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
        return None;
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

fn pop_cpu_descriptor(
    queue: &mut VecDeque<ChunkDescriptor>,
    gpu_enabled: bool,
) -> Option<ChunkDescriptor> {
    if gpu_enabled
        && let Some(pos) = queue
            .iter()
            .position(|descriptor| descriptor.backend == ChunkBackend::Cpu)
    {
        return queue.remove(pos);
    }
    queue.pop_front()
}

fn pop_gpu_descriptor(queue: &mut VecDeque<ChunkDescriptor>) -> Option<ChunkDescriptor> {
    if let Some(pos) = queue
        .iter()
        .position(|descriptor| descriptor.backend == ChunkBackend::GpuAssisted)
    {
        return queue.remove(pos);
    }
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

fn decode_deflate_by_codec(codec: ChunkCodec, compressed: &[u8]) -> Result<Vec<u8>, CozipDeflateError> {
    match codec {
        ChunkCodec::DeflateCpu | ChunkCodec::DeflateGpuFast => deflate_decompress_cpu(compressed),
    }
}

fn decode_descriptor_cpu(
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
        return Err(CozipDeflateError::InvalidFrame(
            "raw chunk length mismatch in cpu path",
        ));
    }

    Ok(DecodedChunk {
        index: descriptor.index,
        raw,
    })
}

fn decode_descriptor_gpu(
    descriptor: ChunkDescriptor,
    options: &HybridOptions,
    _gpu: &GpuAssist,
) -> Result<DecodedChunk, CozipDeflateError> {
    let inflated = decode_deflate_by_codec(descriptor.codec, &descriptor.compressed)?;

    let raw = match descriptor.transform {
        ChunkTransform::None => inflated,
        ChunkTransform::EvenOdd => {
            even_odd_transform_cpu(&inflated, options.gpu_subchunk_size, true)
        }
    };

    if raw.len() != descriptor.raw_len as usize {
        return Err(CozipDeflateError::InvalidFrame(
            "raw chunk length mismatch in gpu path",
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

#[derive(Debug, Clone, Copy)]
enum DeflateToken {
    Literal(u8),
    Match { len: usize, dist: usize },
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

fn hash4(data: &[u8], pos: usize) -> usize {
    if pos + 3 >= data.len() {
        return 0;
    }
    let a = data[pos] as usize;
    let b = data[pos + 1] as usize;
    let c = data[pos + 2] as usize;
    let d = data[pos + 3] as usize;
    ((a << 11) ^ (b << 7) ^ (c << 3) ^ d) & HASH_MASK
}

fn match_len(data: &[u8], left: usize, right: usize, max_len: usize) -> usize {
    let mut len = 0;
    while len < max_len
        && right + len < data.len()
        && left + len < data.len()
        && data[left + len] == data[right + len]
    {
        len += 1;
    }
    len
}

fn push_hash_candidate(table: &mut [Vec<usize>], data: &[u8], pos: usize) {
    if pos + MIN_MATCH >= data.len() {
        return;
    }
    let slot = hash4(data, pos);
    let bucket = &mut table[slot];
    bucket.push(pos);
    if bucket.len() > MAX_HASH_CANDIDATES {
        let drop_count = bucket.len() - MAX_HASH_CANDIDATES;
        bucket.drain(0..drop_count);
    }
}

fn build_tokens_from_run_starts(data: &[u8], run_starts: &[usize]) -> Vec<DeflateToken> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut starts = run_starts.to_vec();
    starts.sort_unstable();
    starts.dedup();
    starts.retain(|value| *value < data.len());
    if starts.first().copied() != Some(0) {
        starts.insert(0, 0);
    }

    let mut run_by_byte = vec![Vec::<usize>::new(); 256];
    for &pos in &starts {
        run_by_byte[data[pos] as usize].push(pos);
    }

    let mut table: Vec<Vec<usize>> = vec![Vec::new(); HASH_SIZE];
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        let max_len = (data.len() - i).min(MAX_MATCH);

        if max_len >= MIN_MATCH {
            let slot = hash4(data, i);
            let bucket = &table[slot];

            for &candidate in bucket.iter().rev() {
                if candidate >= i {
                    continue;
                }
                let dist = i - candidate;
                if dist > MAX_DISTANCE {
                    continue;
                }

                let len = match_len(data, candidate, i, max_len);
                if len >= MIN_MATCH && len > best_len {
                    best_len = len;
                    best_dist = dist;
                    if len == max_len {
                        break;
                    }
                }
            }

            let hints = &run_by_byte[data[i] as usize];
            if !hints.is_empty() {
                let mut take = 0usize;
                let mut idx = hints.partition_point(|&value| value < i);
                while idx > 0 && take < MAX_RUN_HINTS {
                    idx -= 1;
                    let candidate = hints[idx];
                    let dist = i - candidate;
                    if dist == 0 || dist > MAX_DISTANCE {
                        continue;
                    }
                    let len = match_len(data, candidate, i, max_len);
                    if len >= MIN_MATCH && len > best_len {
                        best_len = len;
                        best_dist = dist;
                        if len == max_len {
                            break;
                        }
                    }
                    take += 1;
                }
            }
        }

        if best_len >= MIN_MATCH {
            tokens.push(DeflateToken::Match {
                len: best_len,
                dist: best_dist,
            });

            for offset in 0..best_len {
                push_hash_candidate(&mut table, data, i + offset);
            }
            i += best_len;
        } else {
            tokens.push(DeflateToken::Literal(data[i]));
            push_hash_candidate(&mut table, data, i);
            i += 1;
        }
    }

    tokens
}

fn fixed_litlen_code(symbol: u16) -> Result<(u16, u8), CozipDeflateError> {
    match symbol {
        0..=143 => Ok((0x30 + symbol, 8)),
        144..=255 => Ok((0x190 + (symbol - 144), 9)),
        256..=279 => Ok((symbol - 256, 7)),
        280..=287 => Ok((0xC0 + (symbol - 280), 8)),
        _ => Err(CozipDeflateError::Internal("invalid literal/length symbol")),
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

fn write_fixed_litlen_symbol(writer: &mut BitWriter, symbol: u16) -> Result<(), CozipDeflateError> {
    let (code, bits) = fixed_litlen_code(symbol)?;
    let rev = reverse_bits(code, bits);
    writer.write_bits(rev as u32, bits);
    Ok(())
}

fn length_symbol_extra(len: usize) -> Result<(u16, u32, u8), CozipDeflateError> {
    if !(MIN_MATCH..=MAX_MATCH).contains(&len) {
        return Err(CozipDeflateError::Internal("length out of deflate range"));
    }

    for index in 0..LEN_BASE.len() {
        let base = LEN_BASE[index] as usize;
        let extra = LEN_EXTRA[index];
        let max = if extra == 0 {
            base
        } else {
            base + ((1_usize << extra) - 1)
        };

        if (base..=max).contains(&len) {
            let symbol = 257 + index as u16;
            let extra_value = len - base;
            return Ok((symbol, extra_value as u32, extra));
        }
    }

    Err(CozipDeflateError::Internal("length symbol not found"))
}

fn distance_symbol_extra(dist: usize) -> Result<(u16, u32, u8), CozipDeflateError> {
    if !(1..=MAX_DISTANCE).contains(&dist) {
        return Err(CozipDeflateError::Internal("distance out of deflate range"));
    }

    for index in 0..DIST_BASE.len() {
        let base = DIST_BASE[index] as usize;
        let extra = DIST_EXTRA[index];
        let max = if extra == 0 {
            base
        } else {
            base + ((1_usize << extra) - 1)
        };

        if (base..=max).contains(&dist) {
            let extra_value = dist - base;
            return Ok((index as u16, extra_value as u32, extra));
        }
    }

    Err(CozipDeflateError::Internal("distance symbol not found"))
}

fn write_length_distance(
    writer: &mut BitWriter,
    len: usize,
    dist: usize,
) -> Result<(), CozipDeflateError> {
    let (len_symbol, len_extra_value, len_extra_bits) = length_symbol_extra(len)?;

    write_fixed_litlen_symbol(writer, len_symbol)?;
    if len_extra_bits > 0 {
        writer.write_bits(len_extra_value, len_extra_bits);
    }

    let (dist_symbol, dist_extra_value, dist_extra_bits) = distance_symbol_extra(dist)?;

    let dist_code = reverse_bits(dist_symbol, 5);
    writer.write_bits(dist_code as u32, 5);
    if dist_extra_bits > 0 {
        writer.write_bits(dist_extra_value, dist_extra_bits);
    }

    Ok(())
}

fn encode_deflate_fixed_from_tokens(tokens: &[DeflateToken]) -> Result<Vec<u8>, CozipDeflateError> {
    let mut writer = BitWriter::new();
    writer.write_bits(1, 1);
    writer.write_bits(0b01, 2);

    for token in tokens {
        match token {
            DeflateToken::Literal(byte) => write_fixed_litlen_symbol(&mut writer, *byte as u16)?,
            DeflateToken::Match { len, dist } => write_length_distance(&mut writer, *len, *dist)?,
        }
    }

    write_fixed_litlen_symbol(&mut writer, 256)?;
    Ok(writer.finish())
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

    let litlen_lengths = build_huffman_code_lengths(&litlen_freq, MAX_BITS)
        .ok_or(CozipDeflateError::Internal("failed to build litlen huffman lengths"))?;
    let dist_lengths = build_huffman_code_lengths(&dist_freq, MAX_BITS)
        .ok_or(CozipDeflateError::Internal("failed to build dist huffman lengths"))?;

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

    let cl_lengths = build_huffman_code_lengths(&cl_freq, CODELEN_MAX_BITS)
        .ok_or(CozipDeflateError::Internal("failed to build codelen huffman lengths"))?;
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

    let header_bits = u32::try_from(writer.bit_len()).map_err(|_| CozipDeflateError::DataTooLarge)?;
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

fn encode_deflate_dynamic_from_tokens(
    tokens: &[DeflateToken],
    litlen_freq_in: &[u32],
    dist_freq_in: &[u32],
) -> Result<Vec<u8>, CozipDeflateError> {
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

    let litlen_lengths = build_huffman_code_lengths(&litlen_freq, MAX_BITS)
        .ok_or(CozipDeflateError::Internal("failed to build litlen huffman lengths"))?;
    let dist_lengths = build_huffman_code_lengths(&dist_freq, MAX_BITS)
        .ok_or(CozipDeflateError::Internal("failed to build dist huffman lengths"))?;

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

    let cl_lengths = build_huffman_code_lengths(&cl_freq, CODELEN_MAX_BITS)
        .ok_or(CozipDeflateError::Internal("failed to build codelen huffman lengths"))?;
    let hclen_count = CODELEN_ORDER
        .iter()
        .rposition(|&sym| cl_lengths[sym] != 0)
        .map(|index| (index + 1).max(4))
        .unwrap_or(4);

    let litlen_codes = build_canonical_codes(&litlen_lengths, MAX_BITS)
        .ok_or(CozipDeflateError::Internal("failed to build litlen codes"))?;
    let dist_codes = build_canonical_codes(&dist_lengths, MAX_BITS)
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

    for token in tokens {
        match token {
            DeflateToken::Literal(byte) => {
                let (code, bits) = litlen_codes[*byte as usize];
                if bits == 0 {
                    return Err(CozipDeflateError::Internal("missing literal code"));
                }
                writer.write_bits(code as u32, bits);
            }
            DeflateToken::Match { len, dist } => {
                let (len_symbol, len_extra_value, len_extra_bits) = length_symbol_extra(*len)?;
                let (len_code, len_bits) = litlen_codes[len_symbol as usize];
                if len_bits == 0 {
                    return Err(CozipDeflateError::Internal("missing length code"));
                }
                writer.write_bits(len_code as u32, len_bits);
                if len_extra_bits > 0 {
                    writer.write_bits(len_extra_value, len_extra_bits);
                }

                let (dist_symbol, dist_extra_value, dist_extra_bits) = distance_symbol_extra(*dist)?;
                let (dist_code, dist_bits) = dist_codes[dist_symbol as usize];
                if dist_bits == 0 {
                    return Err(CozipDeflateError::Internal("missing distance code"));
                }
                writer.write_bits(dist_code as u32, dist_bits);
                if dist_extra_bits > 0 {
                    writer.write_bits(dist_extra_value, dist_extra_bits);
                }
            }
        }
    }

    let (eob_code, eob_bits) = litlen_codes[256];
    if eob_bits == 0 {
        return Err(CozipDeflateError::Internal("missing end-of-block code"));
    }
    writer.write_bits(eob_code as u32, eob_bits);
    Ok(writer.finish())
}

fn encode_deflate_fixed_from_runs(
    data: &[u8],
    run_starts: &[usize],
) -> Result<Vec<u8>, CozipDeflateError> {
    let tokens = build_tokens_from_run_starts(data, run_starts);

    let mut writer = BitWriter::new();
    writer.write_bits(1, 1);
    writer.write_bits(0b01, 2);

    for token in tokens {
        match token {
            DeflateToken::Literal(byte) => {
                write_fixed_litlen_symbol(&mut writer, byte as u16)?;
            }
            DeflateToken::Match { len, dist } => {
                write_length_distance(&mut writer, len, dist)?;
            }
        }
    }

    write_fixed_litlen_symbol(&mut writer, 256)?;
    Ok(writer.finish())
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

fn encode_frame(original_len: usize, chunks: &[ChunkMember]) -> Result<Vec<u8>, CozipDeflateError> {
    let mut out = Vec::new();
    out.extend_from_slice(&FRAME_MAGIC);
    out.push(FRAME_VERSION);
    out.push(0);
    write_u32(
        &mut out,
        u32::try_from(chunks.len()).map_err(|_| CozipDeflateError::DataTooLarge)?,
    );
    write_u32(
        &mut out,
        chunks.iter().map(|chunk| chunk.raw_len).max().unwrap_or(0),
    );
    write_u64(
        &mut out,
        u64::try_from(original_len).map_err(|_| CozipDeflateError::DataTooLarge)?,
    );

    for chunk in chunks {
        out.push(chunk.backend.to_u8());
        out.push(chunk.transform.to_u8());
        out.push(chunk.codec.to_u8());
        write_u32(&mut out, chunk.raw_len);
        write_u32(
            &mut out,
            u32::try_from(chunk.compressed.len()).map_err(|_| CozipDeflateError::DataTooLarge)?,
        );
    }

    for chunk in chunks {
        out.extend_from_slice(&chunk.compressed);
    }

    Ok(out)
}

fn parse_frame(frame: &[u8]) -> Result<(usize, Vec<ChunkDescriptor>), CozipDeflateError> {
    if frame.len() < HEADER_LEN {
        return Err(CozipDeflateError::InvalidFrame("frame too short"));
    }

    if frame[..4] != FRAME_MAGIC {
        return Err(CozipDeflateError::InvalidFrame("bad magic"));
    }

    let version = frame[4];
    if version != 1 && version != 2 && version != FRAME_VERSION {
        return Err(CozipDeflateError::InvalidFrame("unsupported frame version"));
    }

    let chunk_count = read_u32(frame, 6)? as usize;
    let original_len = usize::try_from(read_u64(frame, 14)?)
        .map_err(|_| CozipDeflateError::InvalidFrame("original length overflow"))?;

    let chunk_meta_len = match version {
        1 => CHUNK_META_LEN_V1,
        2 => CHUNK_META_LEN_V2,
        _ => CHUNK_META_LEN_V3,
    };

    let meta_len = chunk_count
        .checked_mul(chunk_meta_len)
        .ok_or(CozipDeflateError::InvalidFrame("metadata overflow"))?;
    let payload_start = HEADER_LEN
        .checked_add(meta_len)
        .ok_or(CozipDeflateError::InvalidFrame("metadata overflow"))?;
    if frame.len() < payload_start {
        return Err(CozipDeflateError::InvalidFrame("incomplete chunk metadata"));
    }

    let mut descriptors = Vec::with_capacity(chunk_count);
    let mut cursor = HEADER_LEN;
    let mut payload_cursor = payload_start;

    for index in 0..chunk_count {
        let backend = ChunkBackend::from_u8(frame[cursor])?;
        let (transform, codec, raw_off, comp_off) = match version {
            1 => (
                ChunkTransform::None,
                ChunkCodec::DeflateCpu,
                1_usize,
                5_usize,
            ),
            2 => (
                ChunkTransform::from_u8(frame[cursor + 1])?,
                if backend == ChunkBackend::GpuAssisted {
                    ChunkCodec::DeflateGpuFast
                } else {
                    ChunkCodec::DeflateCpu
                },
                2_usize,
                6_usize,
            ),
            _ => (
                ChunkTransform::from_u8(frame[cursor + 1])?,
                ChunkCodec::from_u8(frame[cursor + 2])?,
                3_usize,
                7_usize,
            ),
        };

        let raw_len = read_u32(frame, cursor + raw_off)?;
        let compressed_len = read_u32(frame, cursor + comp_off)? as usize;
        cursor += chunk_meta_len;

        let payload_end = payload_cursor
            .checked_add(compressed_len)
            .ok_or(CozipDeflateError::InvalidFrame("payload overflow"))?;
        if payload_end > frame.len() {
            return Err(CozipDeflateError::InvalidFrame(
                "chunk payload out of range",
            ));
        }

        descriptors.push(ChunkDescriptor {
            index,
            backend,
            transform,
            codec,
            raw_len,
            compressed: frame[payload_cursor..payload_end].to_vec(),
        });
        payload_cursor = payload_end;
    }

    if payload_cursor != frame.len() {
        return Err(CozipDeflateError::InvalidFrame("trailing bytes in frame"));
    }

    let total_raw: usize = descriptors.iter().map(|chunk| chunk.raw_len as usize).sum();
    if total_raw != original_len {
        return Err(CozipDeflateError::InvalidFrame(
            "sum(raw_len) does not match original length",
        ));
    }

    Ok((original_len, descriptors))
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, CozipDeflateError> {
    let end = offset
        .checked_add(4)
        .ok_or(CozipDeflateError::InvalidFrame("u32 overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or(CozipDeflateError::InvalidFrame("u32 out of range"))?;
    let array: [u8; 4] = slice
        .try_into()
        .map_err(|_| CozipDeflateError::InvalidFrame("u32 parse failed"))?;
    Ok(u32::from_le_bytes(array))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, CozipDeflateError> {
    let end = offset
        .checked_add(8)
        .ok_or(CozipDeflateError::InvalidFrame("u64 overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or(CozipDeflateError::InvalidFrame("u64 out of range"))?;
    let array: [u8; 8] = slice
        .try_into()
        .map_err(|_| CozipDeflateError::InvalidFrame("u64 parse failed"))?;
    Ok(u64::from_le_bytes(array))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterned_data(len: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push(((i as u32 * 31 + 7) % 251) as u8);
        }
        data
    }

    #[test]
    fn raw_deflate_roundtrip() {
        let input = b"cozip-cozip-cozip-cozip-cozip";
        let compressed = deflate_compress_cpu(input, 6).expect("compression should succeed");
        let restored = deflate_decompress_cpu(&compressed).expect("decompression should succeed");
        assert_eq!(restored, input);
    }

    #[test]
    fn fixed_run_encoder_roundtrip() {
        let input = b"AAAAAAABBBBBBBBBBBCCCCCCDDDDDDDDDxxxxxx";
        let runs = vec![0, 7, 18, 24, 33, 39];
        let encoded = encode_deflate_fixed_from_runs(input, &runs).expect("encode should succeed");
        let decoded = deflate_decompress_cpu(&encoded).expect("decode should succeed");
        assert_eq!(decoded, input);
    }

    #[test]
    fn even_odd_cpu_transform_roundtrip() {
        let input = patterned_data(1024 * 17 + 5);
        let encoded = even_odd_transform_cpu(&input, 333, false);
        let decoded = even_odd_transform_cpu(&encoded, 333, true);
        assert_eq!(decoded, input);
    }

    #[test]
    fn hybrid_roundtrip_default_options() {
        let input = patterned_data(1024 * 1024 + 137);
        let options = HybridOptions::default();

        let compressed = compress_hybrid(&input, &options).expect("hybrid compress should succeed");
        let decompressed = decompress_hybrid(&compressed.bytes, &options)
            .expect("hybrid decompress should succeed");

        assert_eq!(decompressed.bytes, input);
        assert_eq!(compressed.stats.chunk_count, decompressed.stats.chunk_count);
    }

    #[test]
    fn frame_corruption_is_detected() {
        let input = b"hello cozip";
        let options = HybridOptions::default();
        let mut frame = compress_hybrid(input, &options)
            .expect("compress should succeed")
            .bytes;
        frame[0] = b'X';

        let error = decompress_hybrid(&frame, &options).expect_err("invalid frame should fail");
        assert!(matches!(error, CozipDeflateError::InvalidFrame(_)));
    }

    #[test]
    fn ratio_mode_roundtrip() {
        let input = patterned_data(1024 * 1024 + 73);
        let options = HybridOptions {
            compression_mode: CompressionMode::Ratio,
            prefer_gpu: false,
            gpu_fraction: 0.0,
            ..HybridOptions::default()
        };

        let compressed = compress_hybrid(&input, &options).expect("compress should succeed");
        let decompressed =
            decompress_hybrid(&compressed.bytes, &options).expect("decompress should succeed");

        assert_eq!(decompressed.bytes, input);
    }

    #[test]
    fn decode_v2_frame_compatibility() {
        let input = b"v2-frame-compat-v2-frame-compat-v2";
        let compressed = deflate_compress_cpu(input, 6).expect("compress should succeed");

        let mut frame = Vec::new();
        frame.extend_from_slice(&FRAME_MAGIC);
        frame.push(2);
        frame.push(0);
        write_u32(&mut frame, 1);
        write_u32(&mut frame, u32::try_from(input.len()).expect("len fits"));
        write_u64(&mut frame, u64::try_from(input.len()).expect("len fits"));
        frame.push(ChunkBackend::Cpu.to_u8());
        frame.push(ChunkTransform::None.to_u8());
        write_u32(&mut frame, u32::try_from(input.len()).expect("len fits"));
        write_u32(
            &mut frame,
            u32::try_from(compressed.len()).expect("compressed len fits"),
        );
        frame.extend_from_slice(&compressed);

        let decoded =
            decompress_hybrid(&frame, &HybridOptions::default()).expect("decode should succeed");
        assert_eq!(decoded.bytes, input);
    }
}
