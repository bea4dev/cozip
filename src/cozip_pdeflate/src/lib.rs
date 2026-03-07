use std::io::{Read, Write};
use std::time::Instant;

use thiserror::Error;

mod pdeflate;

pub use pdeflate::{
    pdeflate_compress, pdeflate_compress_with_stats, pdeflate_decompress,
    pdeflate_decompress_into, pdeflate_decompress_into_with_stats,
    pdeflate_decompress_into_with_stats_with_options, pdeflate_decompress_with_stats,
    pdeflate_gpu_init, PDeflateCompressionMode as CompressionMode,
    PDeflateHybridSchedulerPolicy as HybridSchedulerPolicy,
    PDeflateOptions as HybridOptions, PDeflateStats as DeflateCpuStreamStats,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct CoZipDeflateInitStats {
    pub gpu_context_init_ms: f64,
    pub gpu_available: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeflateChunkIndex;

#[derive(Debug, Clone)]
pub struct DeflateHybridCompressResult {
    pub stats: DeflateCpuStreamStats,
    pub index: Option<DeflateChunkIndex>,
}

#[derive(Debug, Error)]
pub enum CozipDeflateError {
    #[error("invalid options: {0}")]
    InvalidOptions(&'static str),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("pdeflate failed: {0}")]
    PDeflate(String),
}

#[derive(Debug, Clone)]
pub struct CoZipDeflate {
    options: HybridOptions,
    init_stats: CoZipDeflateInitStats,
}

impl CoZipDeflate {
    pub fn init(options: HybridOptions) -> Result<Self, CozipDeflateError> {
        validate_options(&options)?;
        let gpu_requested = options.gpu_compress_enabled
            || options.gpu_decompress_enabled
            || options.gpu_decompress_force_gpu;
        let mut init_stats = CoZipDeflateInitStats::default();
        if gpu_requested {
            let t0 = Instant::now();
            init_stats.gpu_available = pdeflate_gpu_init();
            init_stats.gpu_context_init_ms = elapsed_ms(t0);
        }
        Ok(Self {
            options,
            init_stats,
        })
    }

    pub fn init_stats(&self) -> CoZipDeflateInitStats {
        self.init_stats
    }

    pub fn gpu_context_init_ms(&self) -> f64 {
        self.init_stats.gpu_context_init_ms
    }

    pub fn deflate_compress_stream_zip_compatible<R: Read + Send, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
        let result = self.deflate_compress_stream_zip_compatible_with_index(reader, writer)?;
        Ok(result.stats)
    }

    pub fn deflate_compress_stream_zip_compatible_with_index<R: Read + Send, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<DeflateHybridCompressResult, CozipDeflateError> {
        let mut input = Vec::new();
        reader.read_to_end(&mut input)?;
        let (compressed, stats) =
            pdeflate_compress_with_stats(&input, &self.options).map_err(map_pdeflate_error)?;
        writer.write_all(&compressed)?;
        Ok(DeflateHybridCompressResult { stats, index: None })
    }

    pub fn deflate_decompress_stream_zip_compatible_with_index<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
        _index: &DeflateChunkIndex,
    ) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
        self.pdeflate_decompress_stream(reader, writer)
    }

    pub fn deflate_decompress_stream_zip_compatible_with_index_cpu<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
        _index: &DeflateChunkIndex,
    ) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
        let mut cpu_options = self.options.clone();
        cpu_options.gpu_decompress_enabled = false;
        cpu_options.gpu_decompress_force_gpu = false;
        pdeflate_decompress_stream_with_options(reader, writer, &cpu_options)
    }

    pub fn pdeflate_decompress_stream<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
        pdeflate_decompress_stream_with_options(reader, writer, &self.options)
    }

    pub fn pdeflate_decompress_bytes(
        &self,
        stream: &[u8],
        output: &mut Vec<u8>,
    ) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
        pdeflate::pdeflate_decompress_into_with_stats_with_options(
            stream,
            output,
            &self.options,
        )
        .map_err(map_pdeflate_error)
    }
}

pub fn deflate_compress_stream_hybrid_zip_compatible<R: Read + Send, W: Write>(
    reader: &mut R,
    writer: &mut W,
    _level: u32,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    let cozip = CoZipDeflate::init(HybridOptions::default())?;
    cozip.deflate_compress_stream_zip_compatible(reader, writer)
}

pub fn deflate_compress_stream_hybrid_zip_compatible_with_index<R: Read + Send, W: Write>(
    reader: &mut R,
    writer: &mut W,
    _level: u32,
) -> Result<DeflateHybridCompressResult, CozipDeflateError> {
    let cozip = CoZipDeflate::init(HybridOptions::default())?;
    cozip.deflate_compress_stream_zip_compatible_with_index(reader, writer)
}

pub fn deflate_decompress_stream_indexed_on_cpu<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    _index: &DeflateChunkIndex,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    let mut options = HybridOptions::default();
    options.gpu_decompress_enabled = false;
    options.gpu_decompress_force_gpu = false;
    pdeflate_decompress_stream_with_options(reader, writer, &options)
}

pub fn deflate_decompress_stream_hybrid_indexed<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    _index: &DeflateChunkIndex,
    options: &HybridOptions,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    pdeflate_decompress_stream_with_options(reader, writer, options)
}

fn pdeflate_decompress_stream_with_options<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &HybridOptions,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    let mut stream = Vec::new();
    reader.read_to_end(&mut stream)?;
    let mut restored = Vec::new();
    let stats = pdeflate::pdeflate_decompress_into_with_stats_with_options(
        &stream,
        &mut restored,
        options,
    )
    .map_err(map_pdeflate_error)?;
    writer.write_all(&restored)?;
    Ok(stats)
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn validate_options(options: &HybridOptions) -> Result<(), CozipDeflateError> {
    if options.chunk_size == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "chunk_size must be greater than 0",
        ));
    }
    if options.section_count == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "section_count must be greater than 0",
        ));
    }
    if options.gpu_slot_count == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_slot_count must be greater than 0",
        ));
    }
    if options.gpu_submit_chunks == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_submit_chunks must be greater than 0",
        ));
    }
    if options.gpu_pipelined_submit_chunks == 0 {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_pipelined_submit_chunks must be greater than 0",
        ));
    }
    if !(0.0..=1.0).contains(&options.gpu_tail_stop_ratio) {
        return Err(CozipDeflateError::InvalidOptions(
            "gpu_tail_stop_ratio must be within 0.0..=1.0",
        ));
    }
    Ok(())
}

fn map_pdeflate_error(err: pdeflate::PDeflateError) -> CozipDeflateError {
    CozipDeflateError::PDeflate(err.to_string())
}

#[cfg(test)]
mod tests;
