use std::collections::VecDeque;
use std::fs::File as StdFile;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

mod pdeflate;

pub use pdeflate::{
    PDeflateCompressionMode as CompressionMode, PDeflateError,
    PDeflateHybridSchedulerPolicy as HybridSchedulerPolicy, PDeflateOptions, PDeflateStats,
    pdeflate_compress, pdeflate_compress_with_stats, pdeflate_decompress,
    pdeflate_decompress_into, pdeflate_decompress_into_with_stats,
    pdeflate_decompress_into_with_stats_with_options, pdeflate_decompress_with_stats,
    pdeflate_gpu_init,
};

pub type HybridOptions = PDeflateOptions;
pub type DeflateCpuStreamStats = PDeflateStats;
pub type CoZipPDeflate = CoZipDeflate;
pub type CoZipPDeflateError = CozipDeflateError;

const DEFAULT_STREAM_IO_BUFFER_SIZE: usize = 8 * 1024 * 1024;
const DEFAULT_ASYNC_STREAM_BUFFER_CAPACITY: usize = 16 * 1024 * 1024;
const DEFAULT_ASYNC_STREAM_LOW_WATERMARK: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct StreamOptions {
    pub io_buffer_size: usize,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            io_buffer_size: DEFAULT_STREAM_IO_BUFFER_SIZE,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AsyncStreamOptions {
    pub buffer_capacity: usize,
    pub low_watermark: usize,
}

impl Default for AsyncStreamOptions {
    fn default() -> Self {
        Self {
            buffer_capacity: DEFAULT_ASYNC_STREAM_BUFFER_CAPACITY,
            low_watermark: DEFAULT_ASYNC_STREAM_LOW_WATERMARK,
        }
    }
}

pub struct AsyncStream<R> {
    inner: R,
    options: AsyncStreamOptions,
    buffer: VecDeque<u8>,
    eof: bool,
}

impl<R> AsyncStream<R> {
    pub fn new(inner: R, options: AsyncStreamOptions) -> Self {
        Self {
            inner,
            options,
            buffer: VecDeque::with_capacity(options.buffer_capacity.max(1)),
            eof: false,
        }
    }
}

impl<R: AsyncRead + Unpin> AsyncStream<R> {
    pub async fn next_chunk(&mut self, max_len: usize) -> Result<Option<Vec<u8>>, std::io::Error> {
        let target_low = self
            .options
            .low_watermark
            .max(max_len.min(self.options.buffer_capacity.max(1)));
        while !self.eof && self.buffer.len() < target_low {
            let want = self
                .options
                .buffer_capacity
                .saturating_sub(self.buffer.len())
                .max(1);
            let mut tmp = vec![0u8; want.min(target_low.max(1))];
            let read = self.inner.read(&mut tmp).await?;
            if read == 0 {
                self.eof = true;
                break;
            }
            self.buffer.extend(&tmp[..read]);
        }
        if self.buffer.is_empty() {
            return Ok(None);
        }
        let take = max_len.min(self.buffer.len()).max(1);
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(byte) = self.buffer.pop_front() {
                out.push(byte);
            }
        }
        Ok(Some(out))
    }
}

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
    #[error("async task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
    #[error("pdeflate failed: {0}")]
    PDeflate(String),
}

#[derive(Debug, Clone)]
pub struct CoZipDeflate {
    options: PDeflateOptions,
    init_stats: CoZipDeflateInitStats,
}

impl CoZipDeflate {
    pub fn init(options: PDeflateOptions) -> Result<Self, CozipDeflateError> {
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

    pub fn compress_stream<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        self.compress_stream_with_options(reader, writer, StreamOptions::default())
    }

    pub fn compress_stream_with_options<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
        stream_options: StreamOptions,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        let _ = stream_options;
        pdeflate::pdeflate_compress_reader_with_stats(reader, writer, &self.options)
            .map_err(map_pdeflate_error)
    }

    pub fn decompress_stream<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        self.decompress_stream_with_options(reader, writer, StreamOptions::default())
    }

    pub fn decompress_stream_with_options<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
        stream_options: StreamOptions,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        let _ = stream_options;
        pdeflate::pdeflate_decompress_reader_with_stats(reader, writer, &self.options)
            .map_err(map_pdeflate_error)
    }

    pub fn compress_file(
        &self,
        input_file: StdFile,
        output_file: StdFile,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        self.compress_file_with_options(input_file, output_file, StreamOptions::default())
    }

    pub fn compress_file_with_options(
        &self,
        input_file: StdFile,
        output_file: StdFile,
        stream_options: StreamOptions,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        let mut reader = BufReader::with_capacity(stream_options.io_buffer_size.max(1), input_file);
        let mut writer =
            BufWriter::with_capacity(stream_options.io_buffer_size.max(1), output_file);
        let stats = self.compress_stream_with_options(&mut reader, &mut writer, stream_options)?;
        writer.flush()?;
        Ok(stats)
    }

    pub fn compress_file_with_name(
        &self,
        input_file: StdFile,
        output_file: StdFile,
        _entry_name: &str,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        self.compress_file(input_file, output_file)
    }

    pub fn compress_file_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        let input = StdFile::open(input_path)?;
        let output = StdFile::create(output_path)?;
        self.compress_file(input, output)
    }

    pub async fn compress_file_async(
        &self,
        input_file: tokio::fs::File,
        output_file: tokio::fs::File,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        self.compress_file_async_with_options(
            input_file,
            output_file,
            StreamOptions::default(),
            AsyncStreamOptions::default(),
        )
        .await
    }

    pub async fn compress_file_async_with_options(
        &self,
        input_file: tokio::fs::File,
        output_file: tokio::fs::File,
        stream_options: StreamOptions,
        _async_stream_options: AsyncStreamOptions,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        let this = self.clone();
        let input_std = input_file.into_std().await;
        let output_std = output_file.into_std().await;
        tokio::task::spawn_blocking(move || {
            this.compress_file_with_options(input_std, output_std, stream_options)
        })
        .await?
    }

    pub async fn compress_file_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        let input = tokio::fs::File::open(input_path).await?;
        let output = tokio::fs::File::create(output_path).await?;
        self.compress_file_async(input, output).await
    }

    pub fn decompress_file(
        &self,
        input_file: StdFile,
        output_file: StdFile,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        self.decompress_file_with_options(input_file, output_file, StreamOptions::default())
    }

    pub fn decompress_file_with_options(
        &self,
        input_file: StdFile,
        output_file: StdFile,
        stream_options: StreamOptions,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        let mut reader = BufReader::with_capacity(stream_options.io_buffer_size.max(1), input_file);
        let mut writer =
            BufWriter::with_capacity(stream_options.io_buffer_size.max(1), output_file);
        let stats = self.decompress_stream_with_options(&mut reader, &mut writer, stream_options)?;
        writer.flush()?;
        Ok(stats)
    }

    pub fn decompress_file_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        let input = StdFile::open(input_path)?;
        let output = StdFile::create(output_path)?;
        self.decompress_file(input, output)
    }

    pub async fn decompress_file_async(
        &self,
        input_file: tokio::fs::File,
        output_file: tokio::fs::File,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        self.decompress_file_async_with_options(
            input_file,
            output_file,
            StreamOptions::default(),
            AsyncStreamOptions::default(),
        )
        .await
    }

    pub async fn decompress_file_async_with_options(
        &self,
        input_file: tokio::fs::File,
        output_file: tokio::fs::File,
        stream_options: StreamOptions,
        _async_stream_options: AsyncStreamOptions,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        let this = self.clone();
        let input_std = input_file.into_std().await;
        let output_std = output_file.into_std().await;
        tokio::task::spawn_blocking(move || {
            this.decompress_file_with_options(input_std, output_std, stream_options)
        })
        .await?
    }

    pub async fn decompress_file_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
    ) -> Result<PDeflateStats, CozipDeflateError> {
        let input = tokio::fs::File::open(input_path).await?;
        let output = tokio::fs::File::create(output_path).await?;
        self.decompress_file_async(input, output).await
    }

    pub async fn compress_stream_async<R, W>(
        &self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<PDeflateStats, CozipDeflateError>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
    {
        self.compress_stream_async_with_options(
            reader,
            writer,
            StreamOptions::default(),
            AsyncStreamOptions::default(),
        )
        .await
    }

    pub async fn compress_stream_async_with_options<R, W>(
        &self,
        reader: &mut R,
        writer: &mut W,
        stream_options: StreamOptions,
        async_stream_options: AsyncStreamOptions,
    ) -> Result<PDeflateStats, CozipDeflateError>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
    {
        let input_path = create_temp_path("cozip-pdeflate-async-in")?;
        let output_path = create_temp_path("cozip-pdeflate-async-out")?;
        let mut input_file = tokio::fs::File::create(&input_path).await?;
        let mut stream = AsyncStream::new(reader, async_stream_options);
        while let Some(chunk) = stream.next_chunk(self.options.chunk_size.max(1)).await? {
            input_file.write_all(&chunk).await?;
        }
        input_file.flush().await?;
        drop(input_file);

        let input_std = StdFile::open(&input_path)?;
        let output_std = StdFile::create(&output_path)?;
        let this = self.clone();
        let stats = tokio::task::spawn_blocking(move || {
            this.compress_file_with_options(input_std, output_std, stream_options)
        })
        .await??;

        let mut output_file = tokio::fs::File::open(&output_path).await?;
        tokio::io::copy(&mut output_file, writer).await?;
        writer.flush().await?;
        let _ = tokio::fs::remove_file(&input_path).await;
        let _ = tokio::fs::remove_file(&output_path).await;
        Ok(stats)
    }

    pub async fn decompress_stream_async<R, W>(
        &self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<PDeflateStats, CozipDeflateError>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
    {
        self.decompress_stream_async_with_options(
            reader,
            writer,
            StreamOptions::default(),
            AsyncStreamOptions::default(),
        )
        .await
    }

    pub async fn decompress_stream_async_with_options<R, W>(
        &self,
        reader: &mut R,
        writer: &mut W,
        stream_options: StreamOptions,
        async_stream_options: AsyncStreamOptions,
    ) -> Result<PDeflateStats, CozipDeflateError>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
    {
        let input_path = create_temp_path("cozip-pdeflate-async-in")?;
        let output_path = create_temp_path("cozip-pdeflate-async-out")?;
        let mut input_file = tokio::fs::File::create(&input_path).await?;
        let mut stream = AsyncStream::new(reader, async_stream_options);
        while let Some(chunk) = stream.next_chunk(self.options.chunk_size.max(1)).await? {
            input_file.write_all(&chunk).await?;
        }
        input_file.flush().await?;
        drop(input_file);

        let input_std = StdFile::open(&input_path)?;
        let output_std = StdFile::create(&output_path)?;
        let this = self.clone();
        let stats = tokio::task::spawn_blocking(move || {
            this.decompress_file_with_options(input_std, output_std, stream_options)
        })
        .await??;

        let mut output_file = tokio::fs::File::open(&output_path).await?;
        tokio::io::copy(&mut output_file, writer).await?;
        writer.flush().await?;
        let _ = tokio::fs::remove_file(&input_path).await;
        let _ = tokio::fs::remove_file(&output_path).await;
        Ok(stats)
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
        let stats = self.compress_stream(reader, writer)?;
        Ok(DeflateHybridCompressResult { stats, index: None })
    }

    pub fn deflate_decompress_stream_zip_compatible_with_index<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
        _index: &DeflateChunkIndex,
    ) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
        self.decompress_stream(reader, writer)
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
        pdeflate::pdeflate_decompress_reader_with_stats(reader, writer, &cpu_options)
            .map_err(map_pdeflate_error)
    }

    pub fn pdeflate_decompress_stream<R: Read, W: Write>(
        &self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
        self.decompress_stream(reader, writer)
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
    let cozip = CoZipDeflate::init(PDeflateOptions::default())?;
    cozip.deflate_compress_stream_zip_compatible(reader, writer)
}

pub fn deflate_compress_stream_hybrid_zip_compatible_with_index<R: Read + Send, W: Write>(
    reader: &mut R,
    writer: &mut W,
    _level: u32,
) -> Result<DeflateHybridCompressResult, CozipDeflateError> {
    let cozip = CoZipDeflate::init(PDeflateOptions::default())?;
    cozip.deflate_compress_stream_zip_compatible_with_index(reader, writer)
}

pub fn deflate_decompress_stream_indexed_on_cpu<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    _index: &DeflateChunkIndex,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    let mut options = PDeflateOptions::default();
    options.gpu_decompress_enabled = false;
    options.gpu_decompress_force_gpu = false;
    pdeflate::pdeflate_decompress_reader_with_stats(reader, writer, &options)
        .map_err(map_pdeflate_error)
}

pub fn deflate_decompress_stream_hybrid_indexed<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    _index: &DeflateChunkIndex,
    options: &PDeflateOptions,
) -> Result<DeflateCpuStreamStats, CozipDeflateError> {
    pdeflate::pdeflate_decompress_reader_with_stats(reader, writer, options)
        .map_err(map_pdeflate_error)
}

fn create_temp_path(prefix: &str) -> Result<PathBuf, std::io::Error> {
    let mut path = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    path.push(format!("{prefix}-{pid}-{nanos}.tmp"));
    Ok(path)
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn validate_options(options: &PDeflateOptions) -> Result<(), CozipDeflateError> {
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

fn map_pdeflate_error(err: PDeflateError) -> CozipDeflateError {
    match err {
        PDeflateError::Io(io) => CozipDeflateError::Io(io),
        other => CozipDeflateError::PDeflate(other.to_string()),
    }
}

#[cfg(test)]
mod tests;
