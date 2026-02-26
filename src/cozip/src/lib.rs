use std::collections::VecDeque;
use std::fs::File as StdFile;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

use cozip_deflate::{
    CozipDeflateError, DeflateStreamMode, deflate_compress_cpu, deflate_compress_stream,
    deflate_decompress_on_cpu, deflate_decompress_stream_on_cpu,
};
use thiserror::Error;

const LOCAL_FILE_HEADER_SIG: u32 = 0x0403_4b50;
const CENTRAL_DIR_HEADER_SIG: u32 = 0x0201_4b50;
const EOCD_SIG: u32 = 0x0605_4b50;
const DATA_DESCRIPTOR_SIG: u32 = 0x0807_4b50;

const GP_FLAG_DATA_DESCRIPTOR: u16 = 1 << 3;
const GP_FLAG_UTF8: u16 = 1 << 11;

const DEFLATE_METHOD: u16 = 8;
const STORED_METHOD: u16 = 0;
const ZIP_VERSION: u16 = 20;
const DEFAULT_ENTRY_NAME: &str = "payload.bin";
const STREAM_BUF_SIZE: usize = 256 * 1024;

#[derive(Debug, Clone)]
pub struct ZipOptions {
    pub compression_level: u32,
    pub deflate_mode: ZipDeflateMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZipDeflateMode {
    Hybrid,
    Cpu,
}

impl Default for ZipOptions {
    fn default() -> Self {
        Self {
            compression_level: 6,
            deflate_mode: ZipDeflateMode::Hybrid,
        }
    }
}

#[derive(Debug, Clone)]
pub enum CoZipOptions {
    Zip { options: ZipOptions },
}

impl Default for CoZipOptions {
    fn default() -> Self {
        Self::Zip {
            options: ZipOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CoZipStats {
    pub entries: usize,
    pub input_bytes: u64,
    pub output_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct ZipEntry {
    pub name: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum CoZipError {
    #[error("invalid zip: {0}")]
    InvalidZip(&'static str),
    #[error("unsupported zip: {0}")]
    Unsupported(&'static str),
    #[error("invalid entry name: {0}")]
    InvalidEntryName(&'static str),
    #[error("deflate error: {0}")]
    Deflate(#[from] CozipDeflateError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("path contains non-utf8 bytes")]
    NonUtf8Name,
    #[error("data too large for zip32")]
    DataTooLarge,
    #[error("async task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

pub type CozipZipError = CoZipError;

#[derive(Debug, Clone)]
pub struct CoZip {
    backend: CoZipBackend,
}

#[derive(Debug, Clone)]
enum CoZipBackend {
    Zip { options: ZipOptions },
}

impl CoZip {
    pub fn init(options: CoZipOptions) -> Self {
        let backend = match options {
            CoZipOptions::Zip { options } => CoZipBackend::Zip { options },
        };
        Self { backend }
    }

    pub fn compress_file(
        &self,
        input_file: StdFile,
        output_file: StdFile,
    ) -> Result<CoZipStats, CoZipError> {
        self.compress_file_with_name(input_file, output_file, DEFAULT_ENTRY_NAME)
    }

    pub fn compress_file_with_name(
        &self,
        input_file: StdFile,
        output_file: StdFile,
        entry_name: &str,
    ) -> Result<CoZipStats, CoZipError> {
        let options = self.zip_options();
        let entry_name = normalize_zip_entry_name(entry_name)?;

        let mut reader = BufReader::new(input_file);
        let mut writer = BufWriter::new(output_file);
        let mut state = ZipWriteState::default();
        state.write_entry_from_reader(&mut writer, &entry_name, &mut reader, options)?;
        let stats = state.finish(&mut writer)?;
        writer.flush()?;
        Ok(stats)
    }

    pub fn compress_file_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
    ) -> Result<CoZipStats, CoZipError> {
        let entry_name = file_name_from_path(input_path.as_ref())?;
        let input = StdFile::open(input_path)?;
        let output = StdFile::create(output_path)?;
        self.compress_file_with_name(input, output, &entry_name)
    }

    pub async fn compress_file_async(
        &self,
        input_file: tokio::fs::File,
        output_file: tokio::fs::File,
    ) -> Result<CoZipStats, CoZipError> {
        self.compress_file_async_with_name(input_file, output_file, DEFAULT_ENTRY_NAME)
            .await
    }

    pub async fn compress_file_async_with_name(
        &self,
        input_file: tokio::fs::File,
        output_file: tokio::fs::File,
        entry_name: impl Into<String>,
    ) -> Result<CoZipStats, CoZipError> {
        let entry_name = entry_name.into();
        let this = self.clone();
        let input_std = input_file.into_std().await;
        let output_std = output_file.into_std().await;
        tokio::task::spawn_blocking(move || {
            this.compress_file_with_name(input_std, output_std, &entry_name)
        })
        .await?
    }

    pub async fn compress_file_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
    ) -> Result<CoZipStats, CoZipError> {
        let input_path = input_path.as_ref().to_path_buf();
        let output_path = output_path.as_ref().to_path_buf();
        let entry_name = file_name_from_path(&input_path)?;

        let input = tokio::fs::File::open(&input_path).await?;
        let output = tokio::fs::File::create(&output_path).await?;
        self.compress_file_async_with_name(input, output, entry_name)
            .await
    }

    pub fn compress_directory<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_dir: PIn,
        output_path: POut,
    ) -> Result<CoZipStats, CoZipError> {
        let options = self.zip_options();
        let input_dir = input_dir.as_ref();
        if !input_dir.is_dir() {
            return Err(CoZipError::InvalidZip("input path is not a directory"));
        }

        let files = collect_files_recursively(input_dir)?;
        let output = StdFile::create(output_path)?;
        let mut writer = BufWriter::new(output);
        let mut state = ZipWriteState::default();

        for file in files {
            let rel = file
                .strip_prefix(input_dir)
                .map_err(|_| CoZipError::InvalidZip("failed to compute relative path"))?;
            let entry_name = zip_name_from_relative_path(rel)?;
            let mut reader = BufReader::new(StdFile::open(&file)?);
            state.write_entry_from_reader(&mut writer, &entry_name, &mut reader, options)?;
        }

        let stats = state.finish(&mut writer)?;
        writer.flush()?;
        Ok(stats)
    }

    pub async fn compress_directory_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_dir: PIn,
        output_path: POut,
    ) -> Result<CoZipStats, CoZipError> {
        let input_dir = input_dir.as_ref().to_path_buf();
        let output_path = output_path.as_ref().to_path_buf();
        let this = self.clone();
        tokio::task::spawn_blocking(move || this.compress_directory(input_dir, output_path)).await?
    }

    pub fn decompress_file(
        &self,
        input_file: StdFile,
        output_file: StdFile,
    ) -> Result<CoZipStats, CoZipError> {
        let mut reader = BufReader::new(input_file);
        let mut writer = BufWriter::new(output_file);
        let (entries, input_len) = read_central_directory_entries(&mut reader)?;
        if entries.len() != 1 {
            return Err(CoZipError::Unsupported(
                "decompress_file expects exactly one file in archive",
            ));
        }

        let output_bytes = extract_entry_to_writer(&mut reader, &entries[0], &mut writer)?;
        writer.flush()?;

        Ok(CoZipStats {
            entries: 1,
            input_bytes: input_len,
            output_bytes,
        })
    }

    pub fn decompress_file_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
    ) -> Result<CoZipStats, CoZipError> {
        let input = StdFile::open(input_path)?;
        let output = StdFile::create(output_path)?;
        self.decompress_file(input, output)
    }

    pub async fn decompress_file_async(
        &self,
        input_file: tokio::fs::File,
        output_file: tokio::fs::File,
    ) -> Result<CoZipStats, CoZipError> {
        let this = self.clone();
        let input_std = input_file.into_std().await;
        let output_std = output_file.into_std().await;
        tokio::task::spawn_blocking(move || this.decompress_file(input_std, output_std)).await?
    }

    pub async fn decompress_file_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_path: POut,
    ) -> Result<CoZipStats, CoZipError> {
        let input = tokio::fs::File::open(input_path).await?;
        let output = tokio::fs::File::create(output_path).await?;
        self.decompress_file_async(input, output).await
    }

    pub fn decompress_directory<POut: AsRef<Path>>(
        &self,
        input_file: StdFile,
        output_dir: POut,
    ) -> Result<CoZipStats, CoZipError> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        let mut reader = BufReader::new(input_file);
        let (entries, input_len) = read_central_directory_entries(&mut reader)?;
        let mut stats = CoZipStats {
            entries: 0,
            input_bytes: input_len,
            output_bytes: 0,
        };

        for entry in entries {
            let rel_path = entry_path_from_zip_name(&entry.name)?;
            let out_path = output_dir.join(rel_path);
            if entry.name.ends_with('/') {
                std::fs::create_dir_all(&out_path)?;
                continue;
            }

            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let out_file = StdFile::create(&out_path)?;
            let mut out_writer = BufWriter::new(out_file);
            let written = extract_entry_to_writer(&mut reader, &entry, &mut out_writer)?;
            out_writer.flush()?;

            stats.entries = stats.entries.saturating_add(1);
            stats.output_bytes = stats.output_bytes.saturating_add(written);
        }

        Ok(stats)
    }

    pub fn decompress_directory_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_dir: POut,
    ) -> Result<CoZipStats, CoZipError> {
        let input = StdFile::open(input_path)?;
        self.decompress_directory(input, output_dir)
    }

    pub async fn decompress_directory_async<POut: AsRef<Path>>(
        &self,
        input_file: tokio::fs::File,
        output_dir: POut,
    ) -> Result<CoZipStats, CoZipError> {
        let output_dir = output_dir.as_ref().to_path_buf();
        let this = self.clone();
        let input_std = input_file.into_std().await;
        tokio::task::spawn_blocking(move || this.decompress_directory(input_std, output_dir))
            .await?
    }

    pub async fn decompress_directory_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
        &self,
        input_path: PIn,
        output_dir: POut,
    ) -> Result<CoZipStats, CoZipError> {
        let input = tokio::fs::File::open(input_path).await?;
        self.decompress_directory_async(input, output_dir).await
    }

    fn zip_options(&self) -> &ZipOptions {
        match &self.backend {
            CoZipBackend::Zip { options } => options,
        }
    }
}

pub fn zip_compress_single(
    file_name: &str,
    data: &[u8],
    options: &ZipOptions,
) -> Result<Vec<u8>, CoZipError> {
    if file_name.is_empty() {
        return Err(CoZipError::InvalidZip("file name is empty"));
    }

    let name = normalize_zip_entry_name(file_name)?;
    let name_bytes = name.as_bytes();
    let name_len = u16::try_from(name_bytes.len()).map_err(|_| CoZipError::DataTooLarge)?;

    let compressed = deflate_compress_cpu(data, options.compression_level)?;
    let crc = crc32fast::hash(data);
    let compressed_size = u32::try_from(compressed.len()).map_err(|_| CoZipError::DataTooLarge)?;
    let uncompressed_size = u32::try_from(data.len()).map_err(|_| CoZipError::DataTooLarge)?;

    let local_header_len = 30_usize
        .checked_add(name_bytes.len())
        .ok_or(CoZipError::DataTooLarge)?;
    let central_header_offset = local_header_len
        .checked_add(compressed.len())
        .ok_or(CoZipError::DataTooLarge)?;
    let central_header_len = 46_usize
        .checked_add(name_bytes.len())
        .ok_or(CoZipError::DataTooLarge)?;

    let mut out = Vec::with_capacity(
        central_header_offset
            .checked_add(central_header_len)
            .and_then(|v| v.checked_add(22))
            .ok_or(CoZipError::DataTooLarge)?,
    );

    write_u32(&mut out, LOCAL_FILE_HEADER_SIG)?;
    write_u16(&mut out, ZIP_VERSION)?;
    write_u16(&mut out, GP_FLAG_UTF8)?;
    write_u16(&mut out, DEFLATE_METHOD)?;
    write_u16(&mut out, 0)?;
    write_u16(&mut out, 0)?;
    write_u32(&mut out, crc)?;
    write_u32(&mut out, compressed_size)?;
    write_u32(&mut out, uncompressed_size)?;
    write_u16(&mut out, name_len)?;
    write_u16(&mut out, 0)?;
    out.extend_from_slice(name_bytes);
    out.extend_from_slice(&compressed);

    let central_offset_u32 =
        u32::try_from(central_header_offset).map_err(|_| CoZipError::DataTooLarge)?;
    let central_size_u32 =
        u32::try_from(central_header_len).map_err(|_| CoZipError::DataTooLarge)?;

    write_u32(&mut out, CENTRAL_DIR_HEADER_SIG)?;
    write_u16(&mut out, ZIP_VERSION)?;
    write_u16(&mut out, ZIP_VERSION)?;
    write_u16(&mut out, GP_FLAG_UTF8)?;
    write_u16(&mut out, DEFLATE_METHOD)?;
    write_u16(&mut out, 0)?;
    write_u16(&mut out, 0)?;
    write_u32(&mut out, crc)?;
    write_u32(&mut out, compressed_size)?;
    write_u32(&mut out, uncompressed_size)?;
    write_u16(&mut out, name_len)?;
    write_u16(&mut out, 0)?;
    write_u16(&mut out, 0)?;
    write_u16(&mut out, 0)?;
    write_u16(&mut out, 0)?;
    write_u32(&mut out, 0)?;
    write_u32(&mut out, 0)?;
    out.extend_from_slice(name_bytes);

    write_u32(&mut out, EOCD_SIG)?;
    write_u16(&mut out, 0)?;
    write_u16(&mut out, 0)?;
    write_u16(&mut out, 1)?;
    write_u16(&mut out, 1)?;
    write_u32(&mut out, central_size_u32)?;
    write_u32(&mut out, central_offset_u32)?;
    write_u16(&mut out, 0)?;

    Ok(out)
}

pub fn zip_decompress_single(zip_bytes: &[u8]) -> Result<ZipEntry, CoZipError> {
    let eocd_offset = find_eocd(zip_bytes).ok_or(CoZipError::InvalidZip("EOCD not found"))?;
    if read_u32(zip_bytes, eocd_offset)? != EOCD_SIG {
        return Err(CoZipError::InvalidZip("invalid EOCD signature"));
    }

    let entries = read_u16(zip_bytes, eocd_offset + 10)?;
    if entries != 1 {
        return Err(CoZipError::Unsupported(
            "zip_decompress_single expects exactly one file",
        ));
    }

    let central_size = read_u32(zip_bytes, eocd_offset + 12)? as usize;
    let central_offset = read_u32(zip_bytes, eocd_offset + 16)? as usize;
    let central_end = central_offset
        .checked_add(central_size)
        .ok_or(CoZipError::InvalidZip("central directory overflow"))?;
    if central_end > zip_bytes.len() {
        return Err(CoZipError::InvalidZip("central directory out of range"));
    }

    if read_u32(zip_bytes, central_offset)? != CENTRAL_DIR_HEADER_SIG {
        return Err(CoZipError::InvalidZip(
            "invalid central directory signature",
        ));
    }

    let method = read_u16(zip_bytes, central_offset + 10)?;
    if method != DEFLATE_METHOD && method != STORED_METHOD {
        return Err(CoZipError::Unsupported(
            "only deflate/store methods are supported",
        ));
    }

    let crc = read_u32(zip_bytes, central_offset + 16)?;
    let compressed_size = read_u32(zip_bytes, central_offset + 20)? as usize;
    let uncompressed_size = read_u32(zip_bytes, central_offset + 24)? as usize;
    let file_name_len = read_u16(zip_bytes, central_offset + 28)? as usize;
    let extra_len = read_u16(zip_bytes, central_offset + 30)? as usize;
    let comment_len = read_u16(zip_bytes, central_offset + 32)? as usize;
    let local_header_offset = read_u32(zip_bytes, central_offset + 42)? as usize;

    let name_start = central_offset + 46;
    let name_end = name_start
        .checked_add(file_name_len)
        .ok_or(CoZipError::InvalidZip("name range overflow"))?;
    let file_name = zip_bytes
        .get(name_start..name_end)
        .ok_or(CoZipError::InvalidZip("name out of range"))?;
    let file_name = String::from_utf8(file_name.to_vec()).map_err(|_| CoZipError::NonUtf8Name)?;

    let local_name_len = read_u16(zip_bytes, local_header_offset + 26)? as usize;
    let local_extra_len = read_u16(zip_bytes, local_header_offset + 28)? as usize;
    if read_u32(zip_bytes, local_header_offset)? != LOCAL_FILE_HEADER_SIG {
        return Err(CoZipError::InvalidZip(
            "invalid local file header signature",
        ));
    }

    let data_start = local_header_offset
        .checked_add(30)
        .and_then(|v| v.checked_add(local_name_len))
        .and_then(|v| v.checked_add(local_extra_len))
        .ok_or(CoZipError::InvalidZip("local data range overflow"))?;
    let data_end = data_start
        .checked_add(compressed_size)
        .ok_or(CoZipError::InvalidZip("compressed data range overflow"))?;
    let compressed = zip_bytes
        .get(data_start..data_end)
        .ok_or(CoZipError::InvalidZip("compressed data out of range"))?;

    let data = if method == DEFLATE_METHOD {
        deflate_decompress_on_cpu(compressed)?
    } else {
        compressed.to_vec()
    };

    if data.len() != uncompressed_size {
        return Err(CoZipError::InvalidZip(
            "decompressed size mismatch against directory",
        ));
    }

    let actual_crc = crc32fast::hash(&data);
    if actual_crc != crc {
        return Err(CoZipError::InvalidZip("crc32 mismatch"));
    }

    let consumed = 46_usize
        .checked_add(file_name_len)
        .and_then(|v| v.checked_add(extra_len))
        .and_then(|v| v.checked_add(comment_len))
        .ok_or(CoZipError::InvalidZip("central record length overflow"))?;
    if central_offset + consumed > central_end {
        return Err(CoZipError::InvalidZip("central record is truncated"));
    }

    Ok(ZipEntry {
        name: file_name,
        data,
    })
}

pub fn compress_file(
    cozip: &CoZip,
    input_file: StdFile,
    output_file: StdFile,
) -> Result<CoZipStats, CoZipError> {
    cozip.compress_file(input_file, output_file)
}

pub fn compress_file_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
    cozip: &CoZip,
    input_path: PIn,
    output_path: POut,
) -> Result<CoZipStats, CoZipError> {
    cozip.compress_file_from_name(input_path, output_path)
}

pub async fn compress_file_async(
    cozip: &CoZip,
    input_file: tokio::fs::File,
    output_file: tokio::fs::File,
) -> Result<CoZipStats, CoZipError> {
    cozip.compress_file_async(input_file, output_file).await
}

pub async fn compress_file_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
    cozip: &CoZip,
    input_path: PIn,
    output_path: POut,
) -> Result<CoZipStats, CoZipError> {
    cozip
        .compress_file_from_name_async(input_path, output_path)
        .await
}

pub fn compress_directory<PIn: AsRef<Path>, POut: AsRef<Path>>(
    cozip: &CoZip,
    input_dir: PIn,
    output_path: POut,
) -> Result<CoZipStats, CoZipError> {
    cozip.compress_directory(input_dir, output_path)
}

pub async fn compress_directory_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
    cozip: &CoZip,
    input_dir: PIn,
    output_path: POut,
) -> Result<CoZipStats, CoZipError> {
    cozip.compress_directory_async(input_dir, output_path).await
}

pub fn decompress_file(
    cozip: &CoZip,
    input_file: StdFile,
    output_file: StdFile,
) -> Result<CoZipStats, CoZipError> {
    cozip.decompress_file(input_file, output_file)
}

pub fn decompress_file_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
    cozip: &CoZip,
    input_path: PIn,
    output_path: POut,
) -> Result<CoZipStats, CoZipError> {
    cozip.decompress_file_from_name(input_path, output_path)
}

pub async fn decompress_file_async(
    cozip: &CoZip,
    input_file: tokio::fs::File,
    output_file: tokio::fs::File,
) -> Result<CoZipStats, CoZipError> {
    cozip.decompress_file_async(input_file, output_file).await
}

pub async fn decompress_file_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
    cozip: &CoZip,
    input_path: PIn,
    output_path: POut,
) -> Result<CoZipStats, CoZipError> {
    cozip
        .decompress_file_from_name_async(input_path, output_path)
        .await
}

pub fn decompress_directory<POut: AsRef<Path>>(
    cozip: &CoZip,
    input_file: StdFile,
    output_dir: POut,
) -> Result<CoZipStats, CoZipError> {
    cozip.decompress_directory(input_file, output_dir)
}

pub fn decompress_directory_from_name<PIn: AsRef<Path>, POut: AsRef<Path>>(
    cozip: &CoZip,
    input_path: PIn,
    output_dir: POut,
) -> Result<CoZipStats, CoZipError> {
    cozip.decompress_directory_from_name(input_path, output_dir)
}

pub async fn decompress_directory_async<POut: AsRef<Path>>(
    cozip: &CoZip,
    input_file: tokio::fs::File,
    output_dir: POut,
) -> Result<CoZipStats, CoZipError> {
    cozip
        .decompress_directory_async(input_file, output_dir)
        .await
}

pub async fn decompress_directory_from_name_async<PIn: AsRef<Path>, POut: AsRef<Path>>(
    cozip: &CoZip,
    input_path: PIn,
    output_dir: POut,
) -> Result<CoZipStats, CoZipError> {
    cozip
        .decompress_directory_from_name_async(input_path, output_dir)
        .await
}

#[derive(Debug, Clone)]
struct ZipCentralWriteEntry {
    name: String,
    crc: u32,
    compressed_size: u32,
    uncompressed_size: u32,
    local_header_offset: u32,
}

#[derive(Debug, Default)]
struct ZipWriteState {
    central_entries: Vec<ZipCentralWriteEntry>,
    offset: u64,
    stats: CoZipStats,
}

impl ZipWriteState {
    fn write_entry_from_reader<W: Write, R: Read>(
        &mut self,
        writer: &mut W,
        entry_name: &str,
        reader: &mut R,
        options: &ZipOptions,
    ) -> Result<(), CoZipError> {
        let name = normalize_zip_entry_name(entry_name)?;
        let name_bytes = name.as_bytes();
        let name_len = u16::try_from(name_bytes.len()).map_err(|_| CoZipError::DataTooLarge)?;

        let local_header_offset =
            u32::try_from(self.offset).map_err(|_| CoZipError::DataTooLarge)?;

        let gp_flags = GP_FLAG_DATA_DESCRIPTOR | GP_FLAG_UTF8;
        write_u32(writer, LOCAL_FILE_HEADER_SIG)?;
        write_u16(writer, ZIP_VERSION)?;
        write_u16(writer, gp_flags)?;
        write_u16(writer, DEFLATE_METHOD)?;
        write_u16(writer, 0)?;
        write_u16(writer, 0)?;
        write_u32(writer, 0)?;
        write_u32(writer, 0)?;
        write_u32(writer, 0)?;
        write_u16(writer, name_len)?;
        write_u16(writer, 0)?;
        writer.write_all(name_bytes)?;

        self.offset = self
            .offset
            .checked_add(30)
            .and_then(|v| v.checked_add(u64::try_from(name_bytes.len()).ok()?))
            .ok_or(CoZipError::DataTooLarge)?;

        let (crc, compressed_size_u64, uncompressed_size_u64) = stream_deflate_from_reader(
            writer,
            reader,
            options.compression_level,
            options.deflate_mode,
        )?;

        let compressed_size =
            u32::try_from(compressed_size_u64).map_err(|_| CoZipError::DataTooLarge)?;
        let uncompressed_size =
            u32::try_from(uncompressed_size_u64).map_err(|_| CoZipError::DataTooLarge)?;

        self.offset = self
            .offset
            .checked_add(compressed_size_u64)
            .ok_or(CoZipError::DataTooLarge)?;

        write_u32(writer, DATA_DESCRIPTOR_SIG)?;
        write_u32(writer, crc)?;
        write_u32(writer, compressed_size)?;
        write_u32(writer, uncompressed_size)?;

        self.offset = self
            .offset
            .checked_add(16)
            .ok_or(CoZipError::DataTooLarge)?;

        self.central_entries.push(ZipCentralWriteEntry {
            name,
            crc,
            compressed_size,
            uncompressed_size,
            local_header_offset,
        });

        self.stats.entries = self.stats.entries.saturating_add(1);
        self.stats.input_bytes = self
            .stats
            .input_bytes
            .checked_add(uncompressed_size_u64)
            .ok_or(CoZipError::DataTooLarge)?;

        Ok(())
    }

    fn finish<W: Write>(mut self, writer: &mut W) -> Result<CoZipStats, CoZipError> {
        let central_dir_offset = self.offset;

        for entry in &self.central_entries {
            let name_bytes = entry.name.as_bytes();
            let name_len = u16::try_from(name_bytes.len()).map_err(|_| CoZipError::DataTooLarge)?;

            write_u32(writer, CENTRAL_DIR_HEADER_SIG)?;
            write_u16(writer, ZIP_VERSION)?;
            write_u16(writer, ZIP_VERSION)?;
            write_u16(writer, GP_FLAG_UTF8)?;
            write_u16(writer, DEFLATE_METHOD)?;
            write_u16(writer, 0)?;
            write_u16(writer, 0)?;
            write_u32(writer, entry.crc)?;
            write_u32(writer, entry.compressed_size)?;
            write_u32(writer, entry.uncompressed_size)?;
            write_u16(writer, name_len)?;
            write_u16(writer, 0)?;
            write_u16(writer, 0)?;
            write_u16(writer, 0)?;
            write_u16(writer, 0)?;
            write_u32(writer, 0)?;
            write_u32(writer, entry.local_header_offset)?;
            writer.write_all(name_bytes)?;

            self.offset = self
                .offset
                .checked_add(46)
                .and_then(|v| v.checked_add(u64::try_from(name_bytes.len()).ok()?))
                .ok_or(CoZipError::DataTooLarge)?;
        }

        let central_dir_size = self
            .offset
            .checked_sub(central_dir_offset)
            .ok_or(CoZipError::DataTooLarge)?;

        let entries_u16 =
            u16::try_from(self.central_entries.len()).map_err(|_| CoZipError::DataTooLarge)?;
        let central_dir_size_u32 =
            u32::try_from(central_dir_size).map_err(|_| CoZipError::DataTooLarge)?;
        let central_dir_offset_u32 =
            u32::try_from(central_dir_offset).map_err(|_| CoZipError::DataTooLarge)?;

        write_u32(writer, EOCD_SIG)?;
        write_u16(writer, 0)?;
        write_u16(writer, 0)?;
        write_u16(writer, entries_u16)?;
        write_u16(writer, entries_u16)?;
        write_u32(writer, central_dir_size_u32)?;
        write_u32(writer, central_dir_offset_u32)?;
        write_u16(writer, 0)?;

        self.offset = self
            .offset
            .checked_add(22)
            .ok_or(CoZipError::DataTooLarge)?;
        self.stats.output_bytes = self.offset;
        Ok(self.stats)
    }
}

#[derive(Debug, Clone)]
struct ZipCentralReadEntry {
    name: String,
    gp_flags: u16,
    method: u16,
    crc: u32,
    compressed_size: u32,
    uncompressed_size: u32,
    local_header_offset: u32,
}

fn stream_deflate_from_reader<W: Write, R: Read>(
    writer: &mut W,
    reader: &mut R,
    level: u32,
    mode: ZipDeflateMode,
) -> Result<(u32, u64, u64), CoZipError> {
    let stream_mode = match mode {
        ZipDeflateMode::Hybrid => DeflateStreamMode::Hybrid,
        ZipDeflateMode::Cpu => DeflateStreamMode::Cpu,
    };
    let stats = deflate_compress_stream(reader, writer, level, stream_mode)?;
    Ok((stats.input_crc32, stats.output_bytes, stats.input_bytes))
}

fn read_central_directory_entries<R: Read + Seek>(
    reader: &mut R,
) -> Result<(Vec<ZipCentralReadEntry>, u64), CoZipError> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    let eocd = read_eocd(reader, file_len)?;

    if eocd
        .central_offset
        .checked_add(eocd.central_size)
        .ok_or(CoZipError::InvalidZip("central directory overflow"))?
        > file_len
    {
        return Err(CoZipError::InvalidZip("central directory out of range"));
    }

    reader.seek(SeekFrom::Start(eocd.central_offset))?;
    let mut entries = Vec::with_capacity(eocd.entries as usize);

    for _ in 0..eocd.entries {
        let mut fixed = [0_u8; 46];
        reader.read_exact(&mut fixed)?;
        if u32::from_le_bytes(
            fixed[0..4]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("failed to parse central header signature"))?,
        ) != CENTRAL_DIR_HEADER_SIG
        {
            return Err(CoZipError::InvalidZip(
                "invalid central directory signature",
            ));
        }

        let gp_flags = u16::from_le_bytes(
            fixed[8..10]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("flags parse failed"))?,
        );
        let method = u16::from_le_bytes(
            fixed[10..12]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("method parse failed"))?,
        );
        if method != DEFLATE_METHOD && method != STORED_METHOD {
            return Err(CoZipError::Unsupported(
                "only deflate/store methods are supported",
            ));
        }
        if (gp_flags & 0x0001) != 0 {
            return Err(CoZipError::Unsupported(
                "encrypted zip entries are unsupported",
            ));
        }

        let crc = u32::from_le_bytes(
            fixed[16..20]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("crc parse failed"))?,
        );
        let compressed_size = u32::from_le_bytes(
            fixed[20..24]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("compressed size parse failed"))?,
        );
        let uncompressed_size = u32::from_le_bytes(
            fixed[24..28]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("uncompressed size parse failed"))?,
        );
        let name_len = u16::from_le_bytes(
            fixed[28..30]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("name len parse failed"))?,
        ) as usize;
        let extra_len = u16::from_le_bytes(
            fixed[30..32]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("extra len parse failed"))?,
        ) as usize;
        let comment_len = u16::from_le_bytes(
            fixed[32..34]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("comment len parse failed"))?,
        ) as usize;
        let local_header_offset = u32::from_le_bytes(
            fixed[42..46]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("local offset parse failed"))?,
        );

        let mut name = vec![0_u8; name_len];
        reader.read_exact(&mut name)?;
        let name = String::from_utf8(name).map_err(|_| CoZipError::NonUtf8Name)?;

        let skip = i64::try_from(extra_len.checked_add(comment_len).ok_or(
            CoZipError::InvalidZip("central variable-length fields overflow"),
        )?)
        .map_err(|_| CoZipError::DataTooLarge)?;
        reader.seek(SeekFrom::Current(skip))?;

        entries.push(ZipCentralReadEntry {
            name,
            gp_flags,
            method,
            crc,
            compressed_size,
            uncompressed_size,
            local_header_offset,
        });
    }

    Ok((entries, file_len))
}

fn extract_entry_to_writer<R: Read + Seek, W: Write>(
    reader: &mut R,
    entry: &ZipCentralReadEntry,
    writer: &mut W,
) -> Result<u64, CoZipError> {
    reader.seek(SeekFrom::Start(u64::from(entry.local_header_offset)))?;

    let mut local_fixed = [0_u8; 30];
    reader.read_exact(&mut local_fixed)?;
    let local_sig = u32::from_le_bytes(
        local_fixed[0..4]
            .try_into()
            .map_err(|_| CoZipError::InvalidZip("local signature parse failed"))?,
    );
    if local_sig != LOCAL_FILE_HEADER_SIG {
        return Err(CoZipError::InvalidZip(
            "invalid local file header signature",
        ));
    }

    let local_name_len = u16::from_le_bytes(
        local_fixed[26..28]
            .try_into()
            .map_err(|_| CoZipError::InvalidZip("local name len parse failed"))?,
    ) as usize;
    let local_extra_len = u16::from_le_bytes(
        local_fixed[28..30]
            .try_into()
            .map_err(|_| CoZipError::InvalidZip("local extra len parse failed"))?,
    ) as usize;
    let skip = i64::try_from(local_name_len.checked_add(local_extra_len).ok_or(
        CoZipError::InvalidZip("local variable-length fields overflow"),
    )?)
    .map_err(|_| CoZipError::DataTooLarge)?;
    reader.seek(SeekFrom::Current(skip))?;

    let mut limited = reader.take(u64::from(entry.compressed_size));
    let mut written: u64;
    let mut buf = vec![0_u8; STREAM_BUF_SIZE];

    match entry.method {
        DEFLATE_METHOD => {
            let stats = deflate_decompress_stream_on_cpu(&mut limited, writer)?;
            written = stats.output_bytes;
            if stats.output_crc32 != entry.crc {
                return Err(CoZipError::InvalidZip("crc32 mismatch"));
            }
        }
        STORED_METHOD => {
            let mut crc = crc32fast::Hasher::new();
            written = 0;
            loop {
                let read = limited.read(&mut buf)?;
                if read == 0 {
                    break;
                }
                writer.write_all(&buf[..read])?;
                crc.update(&buf[..read]);
                written = written
                    .checked_add(u64::try_from(read).map_err(|_| CoZipError::DataTooLarge)?)
                    .ok_or(CoZipError::DataTooLarge)?;
            }
            let actual_crc = crc.finalize();
            if actual_crc != entry.crc {
                return Err(CoZipError::InvalidZip("crc32 mismatch"));
            }
        }
        _ => {
            return Err(CoZipError::Unsupported(
                "only deflate/store methods are supported",
            ));
        }
    }

    let mut sink = io::sink();
    let leftover = io::copy(&mut limited, &mut sink)?;
    if leftover != 0 {
        return Err(CoZipError::InvalidZip(
            "compressed stream did not consume declared size",
        ));
    }

    if written != u64::from(entry.uncompressed_size) {
        return Err(CoZipError::InvalidZip("decompressed size mismatch"));
    }

    if (entry.gp_flags & 0x0001) != 0 {
        return Err(CoZipError::Unsupported(
            "encrypted zip entries are unsupported",
        ));
    }

    Ok(written)
}

#[derive(Debug, Clone, Copy)]
struct Eocd {
    entries: u16,
    central_size: u64,
    central_offset: u64,
}

fn read_eocd<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<Eocd, CoZipError> {
    if file_len < 22 {
        return Err(CoZipError::InvalidZip("file too small for EOCD"));
    }

    let search_len = file_len.min(22 + 65_535);
    let search_start = file_len - search_len;

    reader.seek(SeekFrom::Start(search_start))?;
    let mut tail = vec![0_u8; usize::try_from(search_len).map_err(|_| CoZipError::DataTooLarge)?];
    reader.read_exact(&mut tail)?;

    let rel = find_eocd(&tail).ok_or(CoZipError::InvalidZip("EOCD not found"))?;
    let eocd_offset = search_start
        .checked_add(u64::try_from(rel).map_err(|_| CoZipError::DataTooLarge)?)
        .ok_or(CoZipError::DataTooLarge)?;

    let entries = read_u16(&tail, rel + 10)?;
    let central_size = u64::from(read_u32(&tail, rel + 12)?);
    let central_offset = u64::from(read_u32(&tail, rel + 16)?);

    if entries == u16::MAX
        || central_size == u64::from(u32::MAX)
        || central_offset == u64::from(u32::MAX)
    {
        return Err(CoZipError::Unsupported("zip64 archives are unsupported"));
    }

    let min_eocd_end = eocd_offset
        .checked_add(22)
        .ok_or(CoZipError::DataTooLarge)?;
    if min_eocd_end > file_len {
        return Err(CoZipError::InvalidZip("EOCD out of range"));
    }

    Ok(Eocd {
        entries,
        central_size,
        central_offset,
    })
}

fn collect_files_recursively(root: &Path) -> Result<Vec<PathBuf>, CoZipError> {
    let mut files = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(root.to_path_buf());

    while let Some(dir) = queue.pop_front() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                queue.push_back(path);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }

    files.sort();
    Ok(files)
}

fn zip_name_from_relative_path(path: &Path) -> Result<String, CoZipError> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let part = part.to_str().ok_or(CoZipError::NonUtf8Name)?;
                parts.push(part.to_string());
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(CoZipError::InvalidEntryName(
                    "relative path contains invalid component",
                ));
            }
        }
    }

    if parts.is_empty() {
        return Err(CoZipError::InvalidEntryName("entry name is empty"));
    }
    Ok(parts.join("/"))
}

fn entry_path_from_zip_name(name: &str) -> Result<PathBuf, CoZipError> {
    let normalized = normalize_zip_entry_name(name)?;
    let mut out = PathBuf::new();
    for part in normalized.split('/') {
        out.push(part);
    }
    Ok(out)
}

fn file_name_from_path(path: &Path) -> Result<String, CoZipError> {
    let file_name = path
        .file_name()
        .ok_or(CoZipError::InvalidEntryName("file name is missing"))?;
    let file_name = file_name.to_str().ok_or(CoZipError::NonUtf8Name)?;
    normalize_zip_entry_name(file_name)
}

fn normalize_zip_entry_name(name: &str) -> Result<String, CoZipError> {
    let sanitized = name.replace('\\', "/");
    let mut parts: Vec<String> = Vec::new();
    for component in Path::new(&sanitized).components() {
        match component {
            Component::Normal(part) => {
                let part = part.to_str().ok_or(CoZipError::NonUtf8Name)?;
                parts.push(part.to_string());
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(CoZipError::InvalidEntryName(
                    "entry name must be relative without parent traversal",
                ));
            }
        }
    }

    if parts.is_empty() {
        return Err(CoZipError::InvalidEntryName("entry name is empty"));
    }

    Ok(parts.join("/"))
}

fn find_eocd(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 22 {
        return None;
    }

    (0..=bytes.len() - 22)
        .rev()
        .find(|offset| bytes[*offset..*offset + 4] == EOCD_SIG.to_le_bytes())
}

fn write_u16<W: Write>(out: &mut W, value: u16) -> Result<(), CoZipError> {
    out.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u32<W: Write>(out: &mut W, value: u32) -> Result<(), CoZipError> {
    out.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, CoZipError> {
    let end = offset
        .checked_add(2)
        .ok_or(CoZipError::InvalidZip("u16 overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or(CoZipError::InvalidZip("u16 out of range"))?;
    let array: [u8; 2] = slice
        .try_into()
        .map_err(|_| CoZipError::InvalidZip("u16 parse failed"))?;
    Ok(u16::from_le_bytes(array))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, CoZipError> {
    let end = offset
        .checked_add(4)
        .ok_or(CoZipError::InvalidZip("u32 overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or(CoZipError::InvalidZip("u32 out of range"))?;
    let array: [u8; 4] = slice
        .try_into()
        .map_err(|_| CoZipError::InvalidZip("u32 parse failed"))?;
    Ok(u32::from_le_bytes(array))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_single_roundtrip() {
        let input = b"cozip zip test cozip zip test cozip zip test".to_vec();
        let zip = zip_compress_single("message.txt", &input, &ZipOptions::default())
            .expect("zip compression should succeed");

        let entry = zip_decompress_single(&zip).expect("zip decompression should succeed");
        assert_eq!(entry.name, "message.txt");
        assert_eq!(entry.data, input);
    }

    #[test]
    fn cozip_compress_file_roundtrip() {
        let cozip = CoZip::init(CoZipOptions::default());
        let mut input = std::env::temp_dir();
        input.push(format!("cozip-input-{}.txt", std::process::id()));
        let mut output = std::env::temp_dir();
        output.push(format!("cozip-output-{}.zip", std::process::id()));
        let mut restored = std::env::temp_dir();
        restored.push(format!("cozip-restored-{}.txt", std::process::id()));

        std::fs::write(&input, b"hello cozip").expect("write input");
        cozip
            .compress_file_from_name(&input, &output)
            .expect("compress file");
        cozip
            .decompress_file_from_name(&output, &restored)
            .expect("decompress file");

        let restored_data = std::fs::read(&restored).expect("read restored");
        assert_eq!(restored_data, b"hello cozip");

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_file(restored);
    }

    #[test]
    fn cozip_directory_roundtrip() {
        let cozip = CoZip::init(CoZipOptions::default());
        let base = std::env::temp_dir().join(format!("cozip-dir-{}", std::process::id()));
        let input_dir = base.join("input");
        let nested = input_dir.join("nested");
        let output_zip = base.join("archive.zip");
        let restore_dir = base.join("restored");

        std::fs::create_dir_all(&nested).expect("create nested dir");
        std::fs::write(input_dir.join("a.txt"), b"aaa").expect("write a");
        std::fs::write(nested.join("b.txt"), b"bbb").expect("write b");

        cozip
            .compress_directory(&input_dir, &output_zip)
            .expect("compress directory");
        cozip
            .decompress_directory_from_name(&output_zip, &restore_dir)
            .expect("decompress directory");

        assert_eq!(
            std::fs::read(restore_dir.join("a.txt")).expect("read restored a"),
            b"aaa"
        );
        assert_eq!(
            std::fs::read(restore_dir.join("nested").join("b.txt")).expect("read restored b"),
            b"bbb"
        );

        let _ = std::fs::remove_dir_all(base);
    }
}
