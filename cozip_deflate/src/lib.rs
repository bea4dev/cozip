use std::borrow::Cow;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
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
pub struct CoZipInitStats {
    pub gpu_context_init_ms: f64,
    pub gpu_available: bool,
}

#[derive(Debug, Clone)]
pub struct CoZip {
    options: HybridOptions,
    gpu_context: Option<Arc<GpuAssist>>,
    init_stats: CoZipInitStats,
}

impl CoZip {
    pub fn init(options: HybridOptions) -> Result<Self, CozipDeflateError> {
        validate_options(&options)?;
        let mut init_stats = CoZipInitStats::default();
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

    pub fn init_stats(&self) -> CoZipInitStats {
        self.init_stats
    }

    pub fn gpu_context_init_ms(&self) -> f64 {
        self.init_stats.gpu_context_init_ms
    }

    pub fn compress(&self, input: &[u8]) -> Result<CompressedFrame, CozipDeflateError> {
        compress_hybrid_with_context(input, &self.options, self.gpu_context.clone())
    }

    pub fn decompress(&self, frame: &[u8]) -> Result<DecompressedFrame, CozipDeflateError> {
        decompress_hybrid_with_context(frame, &self.options, self.gpu_context.clone())
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

struct GpuFixedTiming {
    call_id: u64,
    mode: CompressionMode,
    groups: usize,
    chunks: usize,
    input_bytes: usize,
    output_bytes: usize,
    payload_chunks: usize,
    fallback_chunks: usize,
    bits_readback_bytes: usize,
    payload_readback_bytes: usize,
    encode_submit_ms: f64,
    bits_readback_ms: f64,
    payload_submit_ms: f64,
    payload_readback_ms: f64,
    cpu_fallback_ms: f64,
}

struct GpuDynamicTiming {
    call_id: u64,
    mode: CompressionMode,
    chunks: usize,
    input_bytes: usize,
    output_bytes: usize,
    payload_chunks: usize,
    fallback_chunks: usize,
    bits_readback_bytes: usize,
    payload_readback_bytes: usize,
    freq_submit_ms: f64,
    freq_poll_wait_ms: f64,
    freq_recv_ms: f64,
    freq_map_copy_ms: f64,
    freq_plan_ms: f64,
    freq_pending_samples: usize,
    freq_pending_sum_chunks: usize,
    freq_pending_max_chunks: usize,
    freq_submit_collect_samples: usize,
    freq_submit_collect_sum_ms: f64,
    freq_submit_collect_max_ms: f64,
    freq_recv_immediate: usize,
    freq_recv_blocked: usize,
    freq_recv_blocked_ms: f64,
    pack_submit_ms: f64,
    pack_bits_readback_ms: f64,
    payload_submit_ms: f64,
    payload_readback_ms: f64,
    cpu_fallback_ms: f64,
}

static GPU_TIMING_CALL_SEQ: AtomicU64 = AtomicU64::new(1);
static DEEP_DYNAMIC_PROBE_TAKEN: AtomicU8 = AtomicU8::new(0);

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

#[derive(Debug)]
struct GpuAssist {
    device: wgpu::Device,
    queue: wgpu::Queue,
    tokenize_bind_group_layout: wgpu::BindGroupLayout,
    tokenize_pipeline: wgpu::ComputePipeline,
    phase1_fused_bind_group_layout: wgpu::BindGroupLayout,
    phase1_fused_pipeline: wgpu::ComputePipeline,
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
    scan_blocks_bind_group_layout: wgpu::BindGroupLayout,
    scan_blocks_pipeline: wgpu::ComputePipeline,
    scan_add_bind_group_layout: wgpu::BindGroupLayout,
    scan_add_pipeline: wgpu::ComputePipeline,
    deflate_slots: Mutex<Vec<DeflateSlot>>,
    deflate_header_buffer: wgpu::Buffer,
    dump_bad_chunk_seq: AtomicUsize,
}

#[derive(Debug)]
struct DeflateSlot {
    len_capacity: usize,
    output_storage_size: u64,
    input_buffer: wgpu::Buffer,
    token_flags_buffer: wgpu::Buffer,
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
    phase1_bg: wgpu::BindGroup,
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
        if timing_profile_enabled() {
            let info = adapter.get_info();
            eprintln!(
                "[cozip][timing] gpu_adapter name=\"{}\" vendor=0x{:x} device=0x{:x} backend={:?} type={:?}",
                info.name, info.vendor, info.device, info.backend, info.device_type
            );
        }

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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/tokenize.wgsl"))),
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

        let phase1_fused_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-phase1-fused-bgl"),
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

        let phase1_fused_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-phase1-fused-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/phase1_fused.wgsl"))),
        });

        let phase1_fused_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-phase1-fused-layout"),
            bind_group_layouts: &[&phase1_fused_bind_group_layout],
            push_constant_ranges: &[],
        });

        let phase1_fused_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cozip-phase1-fused-pipeline"),
                layout: Some(&phase1_fused_layout),
                module: &phase1_fused_shader,
                entry_point: "main",
            });

        let token_finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-token-finalize-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/token_finalize.wgsl"))),
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/freq.wgsl"))),
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/dyn_map.wgsl"))),
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/dyn_finalize.wgsl"))),
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/litlen.wgsl"))),
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/bitpack.wgsl"))),
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/scan_blocks.wgsl"))),
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/scan_add.wgsl"))),
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

        // run_start_positions*専用だったmatch/count/prefix/emitパイプラインは未使用のため
        // 初期化コスト削減のため生成しない。

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
            phase1_fused_bind_group_layout,
            phase1_fused_pipeline,
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
            scan_blocks_bind_group_layout,
            scan_blocks_pipeline,
            scan_add_bind_group_layout,
            scan_add_pipeline,
            deflate_slots: Mutex::new(Vec::new()),
            deflate_header_buffer,
            dump_bad_chunk_seq: AtomicUsize::new(0),
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

        let phase1_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-deflate-phase1-bg"),
            layout: &self.phase1_fused_bind_group_layout,
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
                    resource: litlen_freq_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: dist_freq_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
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
            token_flags_buffer,
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
            phase1_bg,
            freq_bg,
            dyn_map_bg,
            dyn_finalize_bg,
            bitpack_bg,
        })
    }

    fn profile_dynamic_phase1_probe(
        &self,
        slot: &DeflateSlot,
        len: usize,
        mode: CompressionMode,
        chunk_index: usize,
    ) -> Result<(), CozipDeflateError> {
        if len == 0 {
            return Ok(());
        }

        let call_id = GPU_TIMING_CALL_SEQ.fetch_add(1, Ordering::Relaxed);
        let (tokenize_x, tokenize_y) = dispatch_grid_for_items(len, 128)?;
        let (finalize_x, finalize_y) = dispatch_grid_for_items(len, TOKEN_FINALIZE_SEGMENT_SIZE)?;
        let (freq_x, freq_y) =
            dispatch_grid_for_items_capped(len, 128, GPU_FREQ_MAX_WORKGROUPS)?;
        let len_u32 = u32::try_from(len).map_err(|_| CozipDeflateError::DataTooLarge)?;
        let mode_id = compression_mode_id(mode);
        let head_only_mode = match mode {
            CompressionMode::Speed => 101,
            CompressionMode::Balanced => 102,
            CompressionMode::Ratio => 103,
        };

        let tokenize_lit_ms = self.profile_tokenize_probe_pass(
            slot,
            len_u32,
            100,
            tokenize_x,
            tokenize_y,
            "cozip-deflate-dyn-probe-tokenize-lit",
        )?;
        let tokenize_head_total_ms = self.profile_tokenize_probe_pass(
            slot,
            len_u32,
            head_only_mode,
            tokenize_x,
            tokenize_y,
            "cozip-deflate-dyn-probe-tokenize-head",
        )?;
        let tokenize_full_ms = self.profile_tokenize_probe_pass(
            slot,
            len_u32,
            mode_id,
            tokenize_x,
            tokenize_y,
            "cozip-deflate-dyn-probe-tokenize-full",
        )?;
        let tokenize_head_only_ms = (tokenize_head_total_ms - tokenize_lit_ms).max(0.0);
        let tokenize_extend_only_ms = (tokenize_full_ms - tokenize_head_total_ms).max(0.0);

        let params = [
            len_u32,
            u32::try_from(TOKEN_FINALIZE_SEGMENT_SIZE).map_err(|_| CozipDeflateError::DataTooLarge)?,
            mode_id,
            0,
        ];
        self.queue
            .write_buffer(&slot.params_buffer, 0, bytemuck::cast_slice(&params));

        let finalize_start = Instant::now();
        let mut finalize_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-deflate-dyn-probe-finalize"),
            });
        {
            let mut pass = finalize_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-deflate-dyn-probe-finalize-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.token_finalize_pipeline);
            pass.set_bind_group(0, &slot.tokenize_bg, &[]);
            pass.dispatch_workgroups(finalize_x, finalize_y, 1);
        }
        self.queue.submit(Some(finalize_encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        let finalize_ms = elapsed_ms(finalize_start);

        let freq_start = Instant::now();
        let mut freq_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-deflate-dyn-probe-freq"),
            });
        freq_encoder.clear_buffer(&slot.litlen_freq_buffer, 0, None);
        freq_encoder.clear_buffer(&slot.dist_freq_buffer, 0, None);
        {
            let mut pass = freq_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-deflate-dyn-probe-freq-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.freq_pipeline);
            pass.set_bind_group(0, &slot.freq_bg, &[]);
            pass.dispatch_workgroups(freq_x, freq_y, 1);
        }
        self.queue.submit(Some(freq_encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        let freq_ms = elapsed_ms(freq_start);

        eprintln!(
            "[cozip][timing][gpu-dynamic-probe] call={} mode={:?} chunk_index={} len_mib={:.2} t_tokenize_lit_ms={:.3} t_tokenize_head_total_ms={:.3} t_tokenize_full_ms={:.3} t_tokenize_head_only_ms={:.3} t_tokenize_extend_only_ms={:.3} t_finalize_ms={:.3} t_freq_ms={:.3} t_phase1_ms={:.3}",
            call_id,
            mode,
            chunk_index,
            len as f64 / (1024.0 * 1024.0),
            tokenize_lit_ms,
            tokenize_head_total_ms,
            tokenize_full_ms,
            tokenize_head_only_ms,
            tokenize_extend_only_ms,
            finalize_ms,
            freq_ms,
            tokenize_full_ms + finalize_ms + freq_ms,
        );

        Ok(())
    }

    fn profile_tokenize_probe_pass(
        &self,
        slot: &DeflateSlot,
        len_u32: u32,
        mode_id: u32,
        dispatch_x: u32,
        dispatch_y: u32,
        label: &'static str,
    ) -> Result<f64, CozipDeflateError> {
        let params = [
            len_u32,
            u32::try_from(TOKEN_FINALIZE_SEGMENT_SIZE).map_err(|_| CozipDeflateError::DataTooLarge)?,
            mode_id,
            0,
        ];
        self.queue
            .write_buffer(&slot.params_buffer, 0, bytemuck::cast_slice(&params));

        // Warmup pass to reduce one-time effects in deep profiling.
        let mut warmup_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = warmup_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tokenize_pipeline);
            pass.set_bind_group(0, &slot.tokenize_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        self.queue.submit(Some(warmup_encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tokenize_pipeline);
            pass.set_bind_group(0, &slot.tokenize_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        Ok(elapsed_ms(start))
    }

    fn deflate_fixed_literals_batch(
        &self,
        chunks: &[&[u8]],
        options: &HybridOptions,
    ) -> Result<Vec<Vec<u8>>, CozipDeflateError> {
        let mode = options.compression_mode;
        let compression_level = options.compression_level;
        if mode == CompressionMode::Ratio {
            return self.deflate_dynamic_hybrid_batch(chunks, options);
        }
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let timing_enabled = timing_profile_enabled();
        let mut timing = GpuFixedTiming {
            call_id: GPU_TIMING_CALL_SEQ.fetch_add(1, Ordering::Relaxed),
            mode,
            groups: 0,
            chunks: 0,
            input_bytes: chunks.iter().map(|chunk| chunk.len()).sum(),
            output_bytes: 0,
            payload_chunks: 0,
            fallback_chunks: 0,
            bits_readback_bytes: 0,
            payload_readback_bytes: 0,
            encode_submit_ms: 0.0,
            bits_readback_ms: 0.0,
            payload_submit_ms: 0.0,
            payload_readback_ms: 0.0,
            cpu_fallback_ms: 0.0,
        };

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
            timing.groups += 1;
            timing.chunks += chunk_group.len();
            let encode_start = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
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
            if let Some(start) = encode_start {
                timing.encode_submit_ms += elapsed_ms(start);
            }

            let bits_readback_start = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
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
            timing.bits_readback_bytes += 4 * bit_receivers.len();
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
            if let Some(start) = bits_readback_start {
                timing.bits_readback_ms += elapsed_ms(start);
            }

            if !payload_jobs.is_empty() {
                timing.payload_chunks += payload_jobs.len();
                timing.payload_readback_bytes += payload_jobs
                    .iter()
                    .try_fold(0usize, |acc, (_, _, copy_size)| {
                        let copy_size = usize::try_from(*copy_size)
                            .map_err(|_| CozipDeflateError::DataTooLarge)?;
                        acc.checked_add(copy_size).ok_or(CozipDeflateError::DataTooLarge)
                    })?;
                let payload_submit_start = if timing_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
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
                if let Some(start) = payload_submit_start {
                    timing.payload_submit_ms += elapsed_ms(start);
                }

                let payload_readback_start = if timing_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
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
                    timing.output_bytes = timing
                        .output_bytes
                        .checked_add(total_bytes)
                        .ok_or(CozipDeflateError::DataTooLarge)?;
                    results[chunk_index] = compressed;
                }
                if let Some(start) = payload_readback_start {
                    timing.payload_readback_ms += elapsed_ms(start);
                }
            }

            timing.fallback_chunks += cpu_fallback.len();
            let fallback_start = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
            for chunk_index in cpu_fallback {
                results[chunk_index] =
                    deflate_compress_cpu(chunks[chunk_index], compression_level)?;
                timing.output_bytes = timing
                    .output_bytes
                    .checked_add(results[chunk_index].len())
                    .ok_or(CozipDeflateError::DataTooLarge)?;
            }
            if let Some(start) = fallback_start {
                timing.cpu_fallback_ms += elapsed_ms(start);
            }
        }

        if timing_enabled {
            eprintln!(
                "[cozip][timing][gpu-fixed] call={} mode={:?} groups={} chunks={} in_mib={:.2} out_mib={:.2} payload_chunks={} fallback_chunks={} bits_rb_kib={:.1} payload_rb_mib={:.2} t_encode_submit_ms={:.3} t_bits_rb_ms={:.3} t_payload_submit_ms={:.3} t_payload_rb_ms={:.3} t_cpu_fallback_ms={:.3}",
                timing.call_id,
                timing.mode,
                timing.groups,
                timing.chunks,
                timing.input_bytes as f64 / (1024.0 * 1024.0),
                timing.output_bytes as f64 / (1024.0 * 1024.0),
                timing.payload_chunks,
                timing.fallback_chunks,
                timing.bits_readback_bytes as f64 / 1024.0,
                timing.payload_readback_bytes as f64 / (1024.0 * 1024.0),
                timing.encode_submit_ms,
                timing.bits_readback_ms,
                timing.payload_submit_ms,
                timing.payload_readback_ms,
                timing.cpu_fallback_ms,
            );
        }

        Ok(results)
    }

    fn deflate_dynamic_hybrid_batch(
        &self,
        chunks: &[&[u8]],
        options: &HybridOptions,
    ) -> Result<Vec<Vec<u8>>, CozipDeflateError> {
        let mode = options.compression_mode;
        let compression_level = options.compression_level;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let timing_enabled = timing_profile_enabled();
        let timing_detail_enabled = timing_enabled && timing_profile_detail_enabled();
        let mut timing = GpuDynamicTiming {
            call_id: GPU_TIMING_CALL_SEQ.fetch_add(1, Ordering::Relaxed),
            mode,
            chunks: 0,
            input_bytes: chunks.iter().map(|chunk| chunk.len()).sum(),
            output_bytes: 0,
            payload_chunks: 0,
            fallback_chunks: 0,
            bits_readback_bytes: 0,
            payload_readback_bytes: 0,
            freq_submit_ms: 0.0,
            freq_poll_wait_ms: 0.0,
            freq_recv_ms: 0.0,
            freq_map_copy_ms: 0.0,
            freq_plan_ms: 0.0,
            freq_pending_samples: 0,
            freq_pending_sum_chunks: 0,
            freq_pending_max_chunks: 0,
            freq_submit_collect_samples: 0,
            freq_submit_collect_sum_ms: 0.0,
            freq_submit_collect_max_ms: 0.0,
            freq_recv_immediate: 0,
            freq_recv_blocked: 0,
            freq_recv_blocked_ms: 0.0,
            pack_submit_ms: 0.0,
            pack_bits_readback_ms: 0.0,
            payload_submit_ms: 0.0,
            payload_readback_ms: 0.0,
            cpu_fallback_ms: 0.0,
        };
        let deep_timing_enabled = deep_timing_profile_enabled();

        struct PendingDynFreqReadback {
            chunk_index: usize,
            slot_index: usize,
            litlen_freq_readback: wgpu::Buffer,
            dist_freq_readback: wgpu::Buffer,
            litlen_receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
            dist_receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
            litlen_ready: bool,
            dist_ready: bool,
            submitted_at: Option<Instant>,
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

        let mut freq_pending: VecDeque<PendingDynFreqReadback> = VecDeque::new();
        let mut prepared: Vec<PreparedDynamicPack> = Vec::with_capacity(chunks.len().max(1));
        let mut staged_freq_readbacks: Vec<(usize, usize, wgpu::Buffer, wgpu::Buffer)> = Vec::new();
        let mut freq_submit_chunk_count = 0usize;
        let mut freq_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-deflate-dynamic-freq-batch-encoder"),
            });
        let collect_ready_freq_front = |freq_pending: &mut VecDeque<PendingDynFreqReadback>,
                                        prepared: &mut Vec<PreparedDynamicPack>,
                                        timing: &mut GpuDynamicTiming|
         -> Result<usize, CozipDeflateError> {
            let mut collected = 0usize;
            loop {
                let freq_recv_start = if timing_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                let ready = {
                    let Some(front) = freq_pending.front_mut() else {
                        break;
                    };
                    if !front.litlen_ready {
                        match front.litlen_receiver.try_recv() {
                            Ok(Ok(())) => {
                                front.litlen_ready = true;
                                if timing_detail_enabled {
                                    timing.freq_recv_immediate += 1;
                                }
                            }
                            Ok(Err(err)) => {
                                return Err(CozipDeflateError::GpuExecution(err.to_string()));
                            }
                            Err(std::sync::mpsc::TryRecvError::Empty) => {}
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                return Err(CozipDeflateError::GpuExecution(
                                    "litlen map_async receiver disconnected".to_string(),
                                ));
                            }
                        }
                    }
                    if !front.dist_ready {
                        match front.dist_receiver.try_recv() {
                            Ok(Ok(())) => {
                                front.dist_ready = true;
                                if timing_detail_enabled {
                                    timing.freq_recv_immediate += 1;
                                }
                            }
                            Ok(Err(err)) => {
                                return Err(CozipDeflateError::GpuExecution(err.to_string()));
                            }
                            Err(std::sync::mpsc::TryRecvError::Empty) => {}
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                return Err(CozipDeflateError::GpuExecution(
                                    "dist map_async receiver disconnected".to_string(),
                                ));
                            }
                        }
                    }
                    front.litlen_ready && front.dist_ready
                };
                if let Some(start) = freq_recv_start {
                    timing.freq_recv_ms += elapsed_ms(start);
                }
                if !ready {
                    break;
                }

                let pending = freq_pending
                    .pop_front()
                    .ok_or(CozipDeflateError::Internal("freq pending pop failed"))?;
                if timing_detail_enabled {
                    if let Some(submitted_at) = pending.submitted_at {
                        let delay_ms = elapsed_ms(submitted_at);
                        timing.freq_submit_collect_samples += 1;
                        timing.freq_submit_collect_sum_ms += delay_ms;
                        timing.freq_submit_collect_max_ms =
                            timing.freq_submit_collect_max_ms.max(delay_ms);
                    }
                }

                let freq_map_copy_start = if timing_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
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
                if let Some(start) = freq_map_copy_start {
                    timing.freq_map_copy_ms += elapsed_ms(start);
                }

                let freq_plan_start = if timing_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                let plan = build_dynamic_huffman_plan(&litlen_freq, &dist_freq)?;
                let mut dyn_table = Vec::with_capacity(DYN_TABLE_U32_COUNT);
                dyn_table.extend_from_slice(&plan.litlen_codes);
                dyn_table.extend_from_slice(&plan.litlen_bits);
                dyn_table.extend_from_slice(&plan.dist_codes);
                dyn_table.extend_from_slice(&plan.dist_bits);
                if let Some(start) = freq_plan_start {
                    timing.freq_plan_ms += elapsed_ms(start);
                }

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
                collected += 1;
            }
            Ok(collected)
        };

        for (chunk_index, data) in chunks.iter().enumerate() {
            if data.is_empty() {
                results[chunk_index] = vec![0x03, 0x00];
                continue;
            }
            timing.chunks += 1;

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

            if deep_timing_enabled
                && DEEP_DYNAMIC_PROBE_TAKEN
                    .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                self.profile_dynamic_phase1_probe(slot, len, mode, chunk_index)?;
            }

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
                let (dispatch_x, dispatch_y) =
                    dispatch_grid_for_items(len, TOKEN_FINALIZE_SEGMENT_SIZE)?;
                let mut pass = freq_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-deflate-phase1-fused-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.phase1_fused_pipeline);
                pass.set_bind_group(0, &slot.phase1_bg, &[]);
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
                let freq_submit_start = if timing_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                self.queue.submit(Some(freq_encoder.finish()));
                if let Some(start) = freq_submit_start {
                    timing.freq_submit_ms += elapsed_ms(start);
                }
                let submitted_at = if timing_detail_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
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
                    freq_pending.push_back(PendingDynFreqReadback {
                        chunk_index: pending_chunk_index,
                        slot_index: pending_slot_index,
                        litlen_freq_readback: lit_rb,
                        dist_freq_readback: dist_rb,
                        litlen_receiver: lit_rx,
                        dist_receiver: dist_rx,
                        litlen_ready: false,
                        dist_ready: false,
                        submitted_at,
                    });
                }
                if timing_detail_enabled {
                    let pending = freq_pending.len();
                    timing.freq_pending_samples += 1;
                    timing.freq_pending_sum_chunks += pending;
                    timing.freq_pending_max_chunks = timing.freq_pending_max_chunks.max(pending);
                }
                self.device.poll(wgpu::Maintain::Poll);
                if !freq_pending.is_empty() {
                    let _ = collect_ready_freq_front(&mut freq_pending, &mut prepared, &mut timing)?;
                }
                freq_encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("cozip-deflate-dynamic-freq-batch-encoder"),
                    });
                freq_submit_chunk_count = 0;
            }
        }

        if freq_submit_chunk_count > 0 {
            let freq_submit_start = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
            self.queue.submit(Some(freq_encoder.finish()));
            if let Some(start) = freq_submit_start {
                timing.freq_submit_ms += elapsed_ms(start);
            }
            let submitted_at = if timing_detail_enabled {
                Some(Instant::now())
            } else {
                None
            };
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
                freq_pending.push_back(PendingDynFreqReadback {
                    chunk_index: pending_chunk_index,
                    slot_index: pending_slot_index,
                    litlen_freq_readback: lit_rb,
                    dist_freq_readback: dist_rb,
                    litlen_receiver: lit_rx,
                    dist_receiver: dist_rx,
                    litlen_ready: false,
                    dist_ready: false,
                    submitted_at,
                });
            }
            if timing_detail_enabled {
                let pending = freq_pending.len();
                timing.freq_pending_samples += 1;
                timing.freq_pending_sum_chunks += pending;
                timing.freq_pending_max_chunks = timing.freq_pending_max_chunks.max(pending);
            }
        }

        if prepared.capacity() < prepared.len() + freq_pending.len() {
            prepared.reserve(freq_pending.len());
        }
        while !freq_pending.is_empty() {
            self.device.poll(wgpu::Maintain::Poll);
            let collected = collect_ready_freq_front(&mut freq_pending, &mut prepared, &mut timing)?;
            if collected == 0 {
                let freq_poll_wait_start = if timing_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                self.device.poll(wgpu::Maintain::Wait);
                if let Some(start) = freq_poll_wait_start {
                    timing.freq_poll_wait_ms += elapsed_ms(start);
                }
                let collected_after_wait =
                    collect_ready_freq_front(&mut freq_pending, &mut prepared, &mut timing)?;
                if collected_after_wait == 0 {
                    return Err(CozipDeflateError::Internal(
                        "freq pending stalled after gpu wait",
                    ));
                }
            }
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
                let pack_submit_start = if timing_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                self.queue.submit(Some(pack_encoder.finish()));
                if let Some(start) = pack_submit_start {
                    timing.pack_submit_ms += elapsed_ms(start);
                }
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
            let pack_submit_start = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
            self.queue.submit(Some(pack_encoder.finish()));
            if let Some(start) = pack_submit_start {
                timing.pack_submit_ms += elapsed_ms(start);
            }
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
        let pack_bits_readback_start = if timing_enabled {
            Some(Instant::now())
        } else {
            None
        };
        if !pack_pending.is_empty() {
            self.device.poll(wgpu::Maintain::Wait);
        }
        timing.bits_readback_bytes += 4 * pack_pending.len();
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
        if let Some(start) = pack_bits_readback_start {
            timing.pack_bits_readback_ms += elapsed_ms(start);
        }

        if !payload_jobs.is_empty() {
            timing.payload_chunks += payload_jobs.len();
            timing.payload_readback_bytes += payload_jobs
                .iter()
                .try_fold(0usize, |acc, (_, _, copy_size)| {
                    let copy_size = usize::try_from(*copy_size)
                        .map_err(|_| CozipDeflateError::DataTooLarge)?;
                    acc.checked_add(copy_size).ok_or(CozipDeflateError::DataTooLarge)
                })?;
            let payload_submit_start = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
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
            if let Some(start) = payload_submit_start {
                timing.payload_submit_ms += elapsed_ms(start);
            }

            let payload_readback_start = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
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
                if mode == CompressionMode::Ratio && options.gpu_dynamic_self_check {
                    if let Err(issue) = gpu_chunk_roundtrip_diagnose(chunks[chunk_index], &compressed)
                    {
                        let raw_hash = fnv1a64(chunks[chunk_index]);
                        let gpu_hash = fnv1a64(&compressed);
                        match &issue {
                            GpuRoundtripIssue::DecodeFailed(message) => eprintln!(
                                "[cozip][warn][gpu-dynamic] call={} chunk_index={} raw_len={} gpu_comp_len={} gpu_payload_bytes={} raw_hash={:016x} gpu_hash={:016x} reason=decode_failed msg=\"{}\" action=cpu_fallback",
                                timing.call_id,
                                chunk_index,
                                chunks[chunk_index].len(),
                                compressed.len(),
                                total_bytes,
                                raw_hash,
                                gpu_hash,
                                message,
                            ),
                            GpuRoundtripIssue::LengthMismatch {
                                decoded_len,
                                prefix_match_len,
                                expected_next,
                                actual_next,
                            } => eprintln!(
                                "[cozip][warn][gpu-dynamic] call={} chunk_index={} raw_len={} gpu_comp_len={} gpu_payload_bytes={} raw_hash={:016x} gpu_hash={:016x} reason=length_mismatch decoded_len={} prefix_match_len={} expected_next={:?} actual_next={:?} action=cpu_fallback",
                                timing.call_id,
                                chunk_index,
                                chunks[chunk_index].len(),
                                compressed.len(),
                                total_bytes,
                                raw_hash,
                                gpu_hash,
                                decoded_len,
                                prefix_match_len,
                                expected_next,
                                actual_next,
                            ),
                            GpuRoundtripIssue::ContentMismatch {
                                first_diff,
                                expected,
                                actual,
                            } => eprintln!(
                                "[cozip][warn][gpu-dynamic] call={} chunk_index={} raw_len={} gpu_comp_len={} gpu_payload_bytes={} raw_hash={:016x} gpu_hash={:016x} reason=content_mismatch first_diff={} expected={} actual={} action=cpu_fallback",
                                timing.call_id,
                                chunk_index,
                                chunks[chunk_index].len(),
                                compressed.len(),
                                total_bytes,
                                raw_hash,
                                gpu_hash,
                                first_diff,
                                expected,
                                actual,
                            ),
                        }
                        let gpu_compressed = compressed.clone();
                        let fallback_start = if timing_enabled {
                            Some(Instant::now())
                        } else {
                            None
                        };
                        compressed = deflate_compress_cpu(chunks[chunk_index], compression_level)?;
                        dump_gpu_dynamic_bad_chunk(
                            options,
                            &self.dump_bad_chunk_seq,
                            timing.call_id,
                            chunk_index,
                            chunks[chunk_index],
                            &gpu_compressed,
                            &compressed,
                            &issue,
                        );
                        timing.fallback_chunks += 1;
                        if let Some(start) = fallback_start {
                            timing.cpu_fallback_ms += elapsed_ms(start);
                        }
                    }
                }
                timing.output_bytes = timing
                    .output_bytes
                    .checked_add(compressed.len())
                    .ok_or(CozipDeflateError::DataTooLarge)?;
                results[chunk_index] = compressed;
            }
            if let Some(start) = payload_readback_start {
                timing.payload_readback_ms += elapsed_ms(start);
            }
        }

        timing.fallback_chunks += cpu_fallback.len();
        let cpu_fallback_start = if timing_enabled {
            Some(Instant::now())
        } else {
            None
        };
        for chunk_index in cpu_fallback {
            results[chunk_index] = deflate_compress_cpu(chunks[chunk_index], compression_level)?;
            timing.output_bytes = timing
                .output_bytes
                .checked_add(results[chunk_index].len())
                .ok_or(CozipDeflateError::DataTooLarge)?;
        }
        if let Some(start) = cpu_fallback_start {
            timing.cpu_fallback_ms += elapsed_ms(start);
        }

        if timing_enabled {
            if timing_detail_enabled {
                let pending_avg_chunks = if timing.freq_pending_samples > 0 {
                    timing.freq_pending_sum_chunks as f64 / timing.freq_pending_samples as f64
                } else {
                    0.0
                };
                let submit_collect_avg_ms = if timing.freq_submit_collect_samples > 0 {
                    timing.freq_submit_collect_sum_ms / timing.freq_submit_collect_samples as f64
                } else {
                    0.0
                };
                eprintln!(
                    "[cozip][timing][gpu-dynamic] call={} mode={:?} chunks={} in_mib={:.2} out_mib={:.2} payload_chunks={} fallback_chunks={} bits_rb_kib={:.1} payload_rb_mib={:.2} pending_avg_chunks={:.2} pending_max_chunks={} submit_collect_avg_ms={:.3} submit_collect_max_ms={:.3} recv_immediate={} recv_blocked={} recv_blocked_ms={:.3} t_freq_submit_ms={:.3} t_freq_poll_wait_ms={:.3} t_freq_recv_ms={:.3} t_freq_map_copy_ms={:.3} t_freq_plan_ms={:.3} t_pack_submit_ms={:.3} t_pack_bits_rb_ms={:.3} t_payload_submit_ms={:.3} t_payload_rb_ms={:.3} t_cpu_fallback_ms={:.3}",
                    timing.call_id,
                    timing.mode,
                    timing.chunks,
                    timing.input_bytes as f64 / (1024.0 * 1024.0),
                    timing.output_bytes as f64 / (1024.0 * 1024.0),
                    timing.payload_chunks,
                    timing.fallback_chunks,
                    timing.bits_readback_bytes as f64 / 1024.0,
                    timing.payload_readback_bytes as f64 / (1024.0 * 1024.0),
                    pending_avg_chunks,
                    timing.freq_pending_max_chunks,
                    submit_collect_avg_ms,
                    timing.freq_submit_collect_max_ms,
                    timing.freq_recv_immediate,
                    timing.freq_recv_blocked,
                    timing.freq_recv_blocked_ms,
                    timing.freq_submit_ms,
                    timing.freq_poll_wait_ms,
                    timing.freq_recv_ms,
                    timing.freq_map_copy_ms,
                    timing.freq_plan_ms,
                    timing.pack_submit_ms,
                    timing.pack_bits_readback_ms,
                    timing.payload_submit_ms,
                    timing.payload_readback_ms,
                    timing.cpu_fallback_ms,
                );
            } else {
                eprintln!(
                    "[cozip][timing][gpu-dynamic] call={} mode={:?} chunks={} in_mib={:.2} out_mib={:.2} payload_chunks={} fallback_chunks={} bits_rb_kib={:.1} payload_rb_mib={:.2} t_freq_submit_ms={:.3} t_freq_poll_wait_ms={:.3} t_freq_recv_ms={:.3} t_freq_map_copy_ms={:.3} t_freq_plan_ms={:.3} t_pack_submit_ms={:.3} t_pack_bits_rb_ms={:.3} t_payload_submit_ms={:.3} t_payload_rb_ms={:.3} t_cpu_fallback_ms={:.3}",
                    timing.call_id,
                    timing.mode,
                    timing.chunks,
                    timing.input_bytes as f64 / (1024.0 * 1024.0),
                    timing.output_bytes as f64 / (1024.0 * 1024.0),
                    timing.payload_chunks,
                    timing.fallback_chunks,
                    timing.bits_readback_bytes as f64 / 1024.0,
                    timing.payload_readback_bytes as f64 / (1024.0 * 1024.0),
                    timing.freq_submit_ms,
                    timing.freq_poll_wait_ms,
                    timing.freq_recv_ms,
                    timing.freq_map_copy_ms,
                    timing.freq_plan_ms,
                    timing.pack_submit_ms,
                    timing.pack_bits_readback_ms,
                    timing.payload_submit_ms,
                    timing.payload_readback_ms,
                    timing.cpu_fallback_ms,
                );
            }
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

pub fn deflate_decompress_cpu(input: &[u8]) -> Result<Vec<u8>, CozipDeflateError> {
    let mut decoder = flate2::write::DeflateDecoder::new(Vec::new());
    decoder.write_all(input)?;
    Ok(decoder.finish()?)
}

pub fn compress_hybrid(
    input: &[u8],
    options: &HybridOptions,
) -> Result<CompressedFrame, CozipDeflateError> {
    let cozip = CoZip::init(options.clone())?;
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
    let has_gpu_tasks = gpu_available && gpu_requested && tasks.iter().any(|task| task.preferred_gpu);
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

    Ok(CompressedFrame { bytes: frame, stats })
}

pub fn decompress_hybrid(
    frame: &[u8],
    options: &HybridOptions,
) -> Result<DecompressedFrame, CozipDeflateError> {
    let cozip = CoZip::init(options.clone())?;
    cozip.decompress(frame)
}

fn decompress_hybrid_with_context(
    frame: &[u8],
    options: &HybridOptions,
    gpu_context: Option<Arc<GpuAssist>>,
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

    let gpu_context = if options.gpu_fraction > 0.0 {
        gpu_context
    } else {
        None
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
    let decoded = deflate_decompress_cpu(compressed)
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
        } => format!(
            "content_mismatch first_diff={first_diff} expected={expected} actual={actual}"
        ),
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
    meta.push_str(&format!("cpu_fallback_fnv1a64={:016x}\n", fnv1a64(cpu_fallback)));
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
        eprintln!(
            "[cozip][error] gpu_decode_len_mismatch index={} backend={:?} codec={:?} raw_len={} decoded_len={} compressed_len={}",
            descriptor.index,
            descriptor.backend,
            descriptor.codec,
            descriptor.raw_len,
            raw.len(),
            descriptor.compressed.len(),
        );
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
