use super::*;
use std::borrow::Cow;
use std::sync::atomic::{AtomicU8, AtomicU64};

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
pub(super) struct GpuCompressedChunk {
    pub(super) compressed: Vec<u8>,
    pub(super) end_bit: Option<usize>,
    pub(super) used_gpu: bool,
}

#[derive(Debug)]
pub(super) struct GpuAssist {
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
    pub(super) fn new() -> Result<Self, CozipDeflateError> {
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/phase1_fused.wgsl"
            ))),
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/token_finalize.wgsl"
            ))),
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/dyn_finalize.wgsl"
            ))),
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/scan_blocks.wgsl"
            ))),
        });

        let scan_blocks_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-scan-blocks-layout"),
            bind_group_layouts: &[&scan_blocks_bind_group_layout],
            push_constant_ranges: &[],
        });

        let scan_blocks_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
        let (freq_x, freq_y) = dispatch_grid_for_items_capped(len, 128, GPU_FREQ_MAX_WORKGROUPS)?;
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
            u32::try_from(TOKEN_FINALIZE_SEGMENT_SIZE)
                .map_err(|_| CozipDeflateError::DataTooLarge)?,
            mode_id,
            0,
        ];
        self.queue
            .write_buffer(&slot.params_buffer, 0, bytemuck::cast_slice(&params));

        let finalize_start = Instant::now();
        let mut finalize_encoder =
            self.device
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
        let mut freq_encoder =
            self.device
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
            u32::try_from(TOKEN_FINALIZE_SEGMENT_SIZE)
                .map_err(|_| CozipDeflateError::DataTooLarge)?,
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
    pub(super) fn deflate_fixed_literals_batch(
        &self,
        chunks: &[&[u8]],
        options: &HybridOptions,
    ) -> Result<Vec<GpuCompressedChunk>, CozipDeflateError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let slot_limit = options.gpu_slot_count.max(1);
        let mode = options.compression_mode;
        {
            let mut slots = lock(&self.deflate_slots)?;
            if slots.len() > slot_limit {
                slots.truncate(slot_limit);
            }
        }
        let mut out = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(slot_limit) {
            let mut encoded = if mode == CompressionMode::Ratio {
                self.deflate_dynamic_hybrid_batch_impl(batch, options)?
            } else {
                self.deflate_fixed_literals_batch_impl(batch, options)?
            };
            out.append(&mut encoded);
        }
        Ok(out)
    }

    fn deflate_fixed_literals_batch_impl(
        &self,
        chunks: &[&[u8]],
        options: &HybridOptions,
    ) -> Result<Vec<GpuCompressedChunk>, CozipDeflateError> {
        let mode = options.compression_mode;
        let compression_level = options.compression_level;
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

        let mut results = vec![
            GpuCompressedChunk {
                compressed: Vec::new(),
                end_bit: None,
                used_gpu: false,
            };
            chunks.len()
        ];
        let submit_group = options.gpu_pipelined_submit_chunks.max(1);
        let mut slots = lock(&self.deflate_slots)?;
        let mut work_indices = Vec::with_capacity(chunks.len());
        for (chunk_index, data) in chunks.iter().enumerate() {
            if data.is_empty() {
                results[chunk_index] = GpuCompressedChunk {
                    compressed: vec![0x03, 0x00],
                    end_bit: Some(10),
                    used_gpu: true,
                };
                continue;
            }
            work_indices.push(chunk_index);
            if slots.len() <= chunk_index {
                slots.push(self.create_deflate_slot(data.len())?);
            } else if slots[chunk_index].len_capacity < data.len() {
                slots[chunk_index] = self.create_deflate_slot(data.len())?;
            }
        }

        for chunk_group in work_indices.chunks(submit_group) {
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
                slots[chunk_index].readback.slice(0..4).map_async(
                    wgpu::MapMode::Read,
                    move |result| {
                        let _ = tx.send(result);
                    },
                );
                bit_receivers.push((chunk_index, rx));
            }
            timing.bits_readback_bytes += 4 * bit_receivers.len();
            self.device.poll(wgpu::Maintain::Wait);

            let mut payload_jobs: Vec<(usize, usize, usize, u64)> = Vec::new();
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
                payload_jobs.push((chunk_index, total_bits, total_bytes, copy_size));
            }
            if let Some(start) = bits_readback_start {
                timing.bits_readback_ms += elapsed_ms(start);
            }

            if !payload_jobs.is_empty() {
                timing.payload_chunks += payload_jobs.len();
                timing.payload_readback_bytes +=
                    payload_jobs
                        .iter()
                        .try_fold(0usize, |acc, (_, _, _, copy_size)| {
                            let copy_size = usize::try_from(*copy_size)
                                .map_err(|_| CozipDeflateError::DataTooLarge)?;
                            acc.checked_add(copy_size)
                                .ok_or(CozipDeflateError::DataTooLarge)
                        })?;
                let payload_submit_start = if timing_enabled {
                    Some(Instant::now())
                } else {
                    None
                };
                let mut payload_encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-deflate-payload-readback-encoder"),
                        });
                for (chunk_index, _total_bits, _total_bytes, copy_size) in &payload_jobs {
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
                for (chunk_index, _total_bits, _total_bytes, copy_size) in &payload_jobs {
                    let (tx, rx) = std::sync::mpsc::channel();
                    slots[*chunk_index].readback.slice(0..*copy_size).map_async(
                        wgpu::MapMode::Read,
                        move |result| {
                            let _ = tx.send(result);
                        },
                    );
                    payload_receivers.push(rx);
                }
                self.device.poll(wgpu::Maintain::Wait);

                for ((chunk_index, total_bits, total_bytes, copy_size), rx) in
                    payload_jobs.into_iter().zip(payload_receivers.into_iter())
                {
                    match rx.recv() {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => {
                            return Err(CozipDeflateError::GpuExecution(err.to_string()));
                        }
                        Err(err) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                    }
                    let mapped = slots[chunk_index]
                        .readback
                        .slice(0..copy_size)
                        .get_mapped_range();
                    let mut compressed = Vec::with_capacity(total_bytes);
                    compressed.extend_from_slice(&mapped[..total_bytes]);
                    drop(mapped);
                    slots[chunk_index].readback.unmap();
                    timing.output_bytes = timing
                        .output_bytes
                        .checked_add(total_bytes)
                        .ok_or(CozipDeflateError::DataTooLarge)?;
                    results[chunk_index] = GpuCompressedChunk {
                        compressed,
                        end_bit: Some(total_bits),
                        used_gpu: true,
                    };
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
                results[chunk_index] = GpuCompressedChunk {
                    compressed: deflate_compress_cpu(chunks[chunk_index], compression_level)?,
                    end_bit: None,
                    used_gpu: false,
                };
                timing.output_bytes = timing
                    .output_bytes
                    .checked_add(results[chunk_index].compressed.len())
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

    fn deflate_dynamic_hybrid_batch_impl(
        &self,
        chunks: &[&[u8]],
        options: &HybridOptions,
    ) -> Result<Vec<GpuCompressedChunk>, CozipDeflateError> {
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

        let mut results = vec![
            GpuCompressedChunk {
                compressed: Vec::new(),
                end_bit: None,
                used_gpu: false,
            };
            chunks.len()
        ];
        let submit_group = options.gpu_pipelined_submit_chunks.max(1);
        let mut slots = lock(&self.deflate_slots)?;
        let litlen_freq_size = bytes_len::<u32>(LITLEN_SYMBOL_COUNT)?;
        let dist_freq_size = bytes_len::<u32>(DIST_SYMBOL_COUNT)?;

        let mut freq_pending: VecDeque<PendingDynFreqReadback> = VecDeque::new();
        let mut prepared: Vec<PreparedDynamicPack> = Vec::with_capacity(chunks.len().max(1));
        let mut staged_freq_readbacks: Vec<(usize, usize, wgpu::Buffer, wgpu::Buffer)> = Vec::new();
        let mut freq_submit_chunk_count = 0usize;
        let mut freq_encoder =
            self.device
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
                results[chunk_index] = GpuCompressedChunk {
                    compressed: vec![0x03, 0x00],
                    end_bit: Some(10),
                    used_gpu: true,
                };
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

            if freq_submit_chunk_count >= submit_group {
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
                    lit_rb
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            let _ = lit_tx.send(result);
                        });
                    let (dist_tx, dist_rx) = std::sync::mpsc::channel();
                    dist_rb
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
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
                    let _ =
                        collect_ready_freq_front(&mut freq_pending, &mut prepared, &mut timing)?;
                }
                freq_encoder =
                    self.device
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
                lit_rb
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = lit_tx.send(result);
                    });
                let (dist_tx, dist_rx) = std::sync::mpsc::channel();
                dist_rb
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
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
            let collected =
                collect_ready_freq_front(&mut freq_pending, &mut prepared, &mut timing)?;
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
        let mut pack_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("cozip-deflate-dynamic-pack-batch-encoder"),
                });

        for item in prepared {
            let slot = &slots[item.slot_index];
            self.queue.write_buffer(
                &slot.dyn_table_buffer,
                0,
                bytemuck::cast_slice(&item.dyn_table),
            );
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
                let (dispatch_x, dispatch_y) = dispatch_grid_for_items(item.len_u32 as usize, 128)?;
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
                let (dispatch_x, dispatch_y) = dispatch_grid_for_items(item.len_u32 as usize, 128)?;
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

            if pack_submit_chunk_count >= submit_group {
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
                    slots[pending_slot_index].readback.slice(0..4).map_async(
                        wgpu::MapMode::Read,
                        move |result| {
                            let _ = tx.send(result);
                        },
                    );
                    pack_pending.push(PendingDynPackBitsReadback {
                        chunk_index: pending_chunk_index,
                        slot_index: pending_slot_index,
                        receiver: rx,
                    });
                }
                self.device.poll(wgpu::Maintain::Poll);
                pack_encoder =
                    self.device
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
                slots[pending_slot_index].readback.slice(0..4).map_async(
                    wgpu::MapMode::Read,
                    move |result| {
                        let _ = tx.send(result);
                    },
                );
                pack_pending.push(PendingDynPackBitsReadback {
                    chunk_index: pending_chunk_index,
                    slot_index: pending_slot_index,
                    receiver: rx,
                });
            }
        }

        let mut payload_jobs: Vec<(usize, usize, usize, u64)> = Vec::new();
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
            let mapped = slots[pending.slot_index]
                .readback
                .slice(0..4)
                .get_mapped_range();
            let total_words: &[u32] = bytemuck::cast_slice(&mapped[..4]);
            let total_bits = total_words.first().copied().unwrap_or(0) as usize;
            let total_bytes = total_bits.div_ceil(8);
            drop(mapped);
            slots[pending.slot_index].readback.unmap();

            let output_storage_size =
                usize::try_from(slots[pending.slot_index].output_storage_size)
                    .map_err(|_| CozipDeflateError::DataTooLarge)?;
            if total_bytes > output_storage_size {
                cpu_fallback.push(pending.chunk_index);
                continue;
            }

            let copy_words = total_bytes.div_ceil(std::mem::size_of::<u32>());
            let copy_size = bytes_len::<u32>(copy_words)?;
            payload_jobs.push((pending.chunk_index, total_bits, total_bytes, copy_size));
        }
        if let Some(start) = pack_bits_readback_start {
            timing.pack_bits_readback_ms += elapsed_ms(start);
        }

        if !payload_jobs.is_empty() {
            timing.payload_chunks += payload_jobs.len();
            timing.payload_readback_bytes +=
                payload_jobs
                    .iter()
                    .try_fold(0usize, |acc, (_, _, _, copy_size)| {
                        let copy_size = usize::try_from(*copy_size)
                            .map_err(|_| CozipDeflateError::DataTooLarge)?;
                        acc.checked_add(copy_size)
                            .ok_or(CozipDeflateError::DataTooLarge)
                    })?;
            let payload_submit_start = if timing_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let mut payload_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("cozip-deflate-dynamic-payload-readback-encoder"),
                    });
            for (chunk_index, _total_bits, _total_bytes, copy_size) in &payload_jobs {
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
            for (chunk_index, _total_bits, _total_bytes, copy_size) in &payload_jobs {
                let (tx, rx) = std::sync::mpsc::channel();
                slots[*chunk_index].readback.slice(0..*copy_size).map_async(
                    wgpu::MapMode::Read,
                    move |result| {
                        let _ = tx.send(result);
                    },
                );
                payload_receivers.push(rx);
            }
            self.device.poll(wgpu::Maintain::Wait);

            for ((chunk_index, total_bits, total_bytes, copy_size), rx) in
                payload_jobs.into_iter().zip(payload_receivers.into_iter())
            {
                match rx.recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                    Err(err) => return Err(CozipDeflateError::GpuExecution(err.to_string())),
                }
                let mapped = slots[chunk_index]
                    .readback
                    .slice(0..copy_size)
                    .get_mapped_range();
                let mut compressed = Vec::with_capacity(total_bytes);
                compressed.extend_from_slice(&mapped[..total_bytes]);
                drop(mapped);
                slots[chunk_index].readback.unmap();
                if mode == CompressionMode::Ratio && options.gpu_dynamic_self_check {
                    if let Err(issue) =
                        gpu_chunk_roundtrip_diagnose(chunks[chunk_index], &compressed)
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
                results[chunk_index] = GpuCompressedChunk {
                    compressed,
                    end_bit: Some(total_bits),
                    used_gpu: true,
                };
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
            results[chunk_index] = GpuCompressedChunk {
                compressed: deflate_compress_cpu(chunks[chunk_index], compression_level)?,
                end_bit: None,
                used_gpu: false,
            };
            timing.output_bytes = timing
                .output_bytes
                .checked_add(results[chunk_index].compressed.len())
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
