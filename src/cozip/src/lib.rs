use std::collections::VecDeque;
use std::fs::File as StdFile;
use std::io::{self, BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

use cozip_deflate::{
    CoZipDeflate, CompressionMode, CozipDeflateError, DeflateChunkIndex, HybridOptions,
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
const ZIP_VERSION_ZIP64: u16 = 45;
const DEFAULT_ENTRY_NAME: &str = "payload.bin";
const STREAM_BUF_SIZE: usize = 256 * 1024;

const ZIP64_EXTRA_FIELD_TAG: u16 = 0x0001;
const ZIP64_EOCD_SIG: u32 = 0x0606_4b50;
const ZIP64_EOCD_LOCATOR_SIG: u32 = 0x0706_4b50;
const CZDI_EXTRA_FIELD_TAG: u16 = 0x435A;
const CZDI_EXTRA_VERSION_V1: u8 = 1;
const CZDI_STORAGE_INLINE: u8 = 0;
const CZDI_STORAGE_EOCD64: u8 = 1;
const CZDI_STORAGE_NONE: u8 = 2;
const CZDI_EOCD64_MAGIC: [u8; 4] = *b"CZDG";

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

fn resolve_czdi_write_plan(
    entries: &[ZipCentralWriteEntry],
) -> Result<(Vec<CzdiResolvedPlan>, Vec<u8>), CoZipError> {
    let mut plans = Vec::with_capacity(entries.len());
    let mut eocd_blob_area = Vec::new();
    let max_inline_blob_len = usize::from(u16::MAX)
        .saturating_sub(28) // ZIP64 extra field in CD
        .saturating_sub(4) // CZDI tag + len
        .saturating_sub(12); // inline payload fixed bytes

    for entry in entries {
        let Some(blob) = entry.czdi_blob.as_ref() else {
            plans.push(CzdiResolvedPlan {
                kind: CzdiExtraKind::None,
                inline_blob: None,
            });
            continue;
        };
        if blob.len() <= max_inline_blob_len {
            plans.push(CzdiResolvedPlan {
                kind: CzdiExtraKind::Inline {
                    blob_len: u32::try_from(blob.len()).map_err(|_| CoZipError::DataTooLarge)?,
                    blob_crc32: crc32fast::hash(blob),
                },
                inline_blob: Some(blob.clone()),
            });
            continue;
        }

        let blob_offset =
            u32::try_from(eocd_blob_area.len()).map_err(|_| CoZipError::DataTooLarge)?;
        let blob_len = u32::try_from(blob.len()).map_err(|_| CoZipError::DataTooLarge)?;
        let blob_crc32 = crc32fast::hash(blob);
        eocd_blob_area.extend_from_slice(blob);
        plans.push(CzdiResolvedPlan {
            kind: CzdiExtraKind::Eocd64Ref {
                blob_offset,
                blob_len,
                blob_crc32,
            },
            inline_blob: None,
        });
    }

    let eocd_payload = if eocd_blob_area.is_empty() {
        Vec::new()
    } else {
        encode_czdi_eocd64_blob(&eocd_blob_area)?
    };
    Ok((plans, eocd_payload))
}

fn encode_czdi_extra_field(plan: &CzdiResolvedPlan) -> Result<Vec<u8>, CoZipError> {
    let mut payload = Vec::new();
    payload.push(CZDI_EXTRA_VERSION_V1);
    match plan.kind {
        CzdiExtraKind::Inline {
            blob_len,
            blob_crc32,
        } => {
            payload.push(CZDI_STORAGE_INLINE);
            payload.extend_from_slice(&0_u16.to_le_bytes());
            payload.extend_from_slice(&blob_len.to_le_bytes());
            payload.extend_from_slice(&blob_crc32.to_le_bytes());
            let inline = plan
                .inline_blob
                .as_ref()
                .ok_or(CoZipError::InvalidZip("missing inline czdi payload"))?;
            payload.extend_from_slice(inline);
        }
        CzdiExtraKind::Eocd64Ref {
            blob_offset,
            blob_len,
            blob_crc32,
        } => {
            payload.push(CZDI_STORAGE_EOCD64);
            payload.extend_from_slice(&0_u16.to_le_bytes());
            payload.extend_from_slice(&blob_offset.to_le_bytes());
            payload.extend_from_slice(&blob_len.to_le_bytes());
            payload.extend_from_slice(&blob_crc32.to_le_bytes());
        }
        CzdiExtraKind::None => {
            payload.push(CZDI_STORAGE_NONE);
            payload.extend_from_slice(&0_u16.to_le_bytes());
        }
    }

    let payload_len = u16::try_from(payload.len()).map_err(|_| CoZipError::DataTooLarge)?;
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&CZDI_EXTRA_FIELD_TAG.to_le_bytes());
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

fn encode_czdi_eocd64_blob(blob_area: &[u8]) -> Result<Vec<u8>, CoZipError> {
    let mut out = Vec::with_capacity(12 + blob_area.len());
    out.extend_from_slice(&CZDI_EOCD64_MAGIC);
    out.push(CZDI_EXTRA_VERSION_V1);
    out.push(0);
    out.extend_from_slice(&0_u16.to_le_bytes());
    out.extend_from_slice(
        &u32::try_from(blob_area.len())
            .map_err(|_| CoZipError::DataTooLarge)?
            .to_le_bytes(),
    );
    out.extend_from_slice(blob_area);
    Ok(out)
}

fn decode_czdi_eocd64_blob(blob: &[u8]) -> Result<Option<Vec<u8>>, CoZipError> {
    if blob.is_empty() {
        return Ok(None);
    }
    if blob.len() < 12 {
        return Err(CoZipError::InvalidZip("czdi eocd64 blob truncated"));
    }
    if blob[..4] != CZDI_EOCD64_MAGIC {
        return Ok(None);
    }
    let version = blob[4];
    if version != CZDI_EXTRA_VERSION_V1 {
        return Err(CoZipError::InvalidZip(
            "unsupported czdi eocd64 blob version",
        ));
    }
    let area_len = u32::from_le_bytes(
        blob[8..12]
            .try_into()
            .map_err(|_| CoZipError::InvalidZip("czdi eocd64 length parse failed"))?,
    ) as usize;
    let area_end = 12_usize
        .checked_add(area_len)
        .ok_or(CoZipError::InvalidZip("czdi eocd64 length overflow"))?;
    let area = blob
        .get(12..area_end)
        .ok_or(CoZipError::InvalidZip("czdi eocd64 payload truncated"))?;
    Ok(Some(area.to_vec()))
}

fn parse_czdi_extra_field(extra: &[u8]) -> Result<Option<CzdiParsedExtra>, CoZipError> {
    let mut pos = 0_usize;
    while pos + 4 <= extra.len() {
        let tag = u16::from_le_bytes(
            extra[pos..pos + 2]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("czdi tag parse failed"))?,
        );
        let size = usize::from(u16::from_le_bytes(
            extra[pos + 2..pos + 4]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("czdi size parse failed"))?,
        ));
        pos += 4;
        let end = pos
            .checked_add(size)
            .ok_or(CoZipError::InvalidZip("czdi field overflow"))?;
        let data = extra
            .get(pos..end)
            .ok_or(CoZipError::InvalidZip("czdi field truncated"))?;
        if tag == CZDI_EXTRA_FIELD_TAG {
            if data.len() < 4 {
                return Err(CoZipError::InvalidZip("czdi payload too short"));
            }
            if data[0] != CZDI_EXTRA_VERSION_V1 {
                return Err(CoZipError::InvalidZip("unsupported czdi extra version"));
            }
            let storage = data[1];
            match storage {
                CZDI_STORAGE_INLINE => {
                    if data.len() < 12 {
                        return Err(CoZipError::InvalidZip("czdi inline header truncated"));
                    }
                    let blob_len = u32::from_le_bytes(
                        data[4..8]
                            .try_into()
                            .map_err(|_| CoZipError::InvalidZip("czdi inline len parse failed"))?,
                    );
                    let blob_crc32 = u32::from_le_bytes(
                        data[8..12]
                            .try_into()
                            .map_err(|_| CoZipError::InvalidZip("czdi inline crc parse failed"))?,
                    );
                    let blob_end =
                        12_usize
                            .checked_add(usize::try_from(blob_len).map_err(|_| {
                                CoZipError::InvalidZip("czdi inline length too large")
                            })?)
                            .ok_or(CoZipError::InvalidZip("czdi inline length overflow"))?;
                    let blob = data
                        .get(12..blob_end)
                        .ok_or(CoZipError::InvalidZip("czdi inline payload truncated"))?;
                    if crc32fast::hash(blob) != blob_crc32 {
                        return Err(CoZipError::InvalidZip("czdi inline crc mismatch"));
                    }
                    return Ok(Some(CzdiParsedExtra {
                        kind: CzdiExtraKind::Inline {
                            blob_len,
                            blob_crc32,
                        },
                        inline_blob: Some(blob.to_vec()),
                    }));
                }
                CZDI_STORAGE_EOCD64 => {
                    if data.len() < 16 {
                        return Err(CoZipError::InvalidZip("czdi eocd64 ref truncated"));
                    }
                    let blob_offset = u32::from_le_bytes(
                        data[4..8]
                            .try_into()
                            .map_err(|_| CoZipError::InvalidZip("czdi ref offset parse failed"))?,
                    );
                    let blob_len = u32::from_le_bytes(
                        data[8..12]
                            .try_into()
                            .map_err(|_| CoZipError::InvalidZip("czdi ref len parse failed"))?,
                    );
                    let blob_crc32 = u32::from_le_bytes(
                        data[12..16]
                            .try_into()
                            .map_err(|_| CoZipError::InvalidZip("czdi ref crc parse failed"))?,
                    );
                    return Ok(Some(CzdiParsedExtra {
                        kind: CzdiExtraKind::Eocd64Ref {
                            blob_offset,
                            blob_len,
                            blob_crc32,
                        },
                        inline_blob: None,
                    }));
                }
                CZDI_STORAGE_NONE => {
                    return Ok(Some(CzdiParsedExtra {
                        kind: CzdiExtraKind::None,
                        inline_blob: None,
                    }));
                }
                _ => return Err(CoZipError::InvalidZip("unknown czdi storage kind")),
            }
        }
        pos = end;
    }
    Ok(None)
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
    Zip { deflate: CoZipDeflate },
}

impl CoZip {
    pub fn init(options: CoZipOptions) -> Result<Self, CoZipError> {
        let backend = match options {
            CoZipOptions::Zip { options } => {
                let mut hybrid_opts = HybridOptions::default();
                let compression_level = options.compression_level.clamp(0, 9);
                hybrid_opts.compression_level = compression_level;
                hybrid_opts.compression_mode = compression_mode_from_level(compression_level);
                hybrid_opts.prefer_gpu = matches!(options.deflate_mode, ZipDeflateMode::Hybrid);
                let deflate = CoZipDeflate::init(hybrid_opts)?;
                CoZipBackend::Zip { deflate }
            }
        };
        Ok(Self { backend })
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
        let deflate = self.zip_deflate();
        let entry_name = normalize_zip_entry_name(entry_name)?;

        let mut reader = BufReader::new(input_file);
        let mut writer = BufWriter::new(output_file);
        let mut state = ZipWriteState::default();
        state.write_entry_from_reader(&mut writer, &entry_name, &mut reader, deflate)?;
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
        let deflate = self.zip_deflate();
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
            state.write_entry_from_reader(&mut writer, &entry_name, &mut reader, deflate)?;
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

        let deflate = self.zip_deflate();
        let output_bytes = extract_entry_to_writer(&mut reader, &entries[0], &mut writer, deflate)?;
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
            let deflate = self.zip_deflate();
            let written = extract_entry_to_writer(&mut reader, &entry, &mut out_writer, deflate)?;
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

    fn zip_deflate(&self) -> &CoZipDeflate {
        match &self.backend {
            CoZipBackend::Zip { deflate } => deflate,
        }
    }
}

fn compression_mode_from_level(level: u32) -> CompressionMode {
    match level {
        0..=3 => CompressionMode::Speed,
        4..=6 => CompressionMode::Balanced,
        _ => CompressionMode::Ratio,
    }
}

pub fn zip_compress_single(
    file_name: &str,
    data: &[u8],
    deflate: &CoZipDeflate,
) -> Result<Vec<u8>, CoZipError> {
    if file_name.is_empty() {
        return Err(CoZipError::InvalidZip("file name is empty"));
    }

    let name = normalize_zip_entry_name(file_name)?;
    let name_bytes = name.as_bytes();
    let name_len = u16::try_from(name_bytes.len()).map_err(|_| CoZipError::DataTooLarge)?;

    let mut cursor = std::io::Cursor::new(data);
    let mut compressed = Vec::new();
    let stats = deflate.deflate_compress_stream_zip_compatible(&mut cursor, &mut compressed)?;
    let crc = stats.input_crc32;
    let compressed_size = compressed.len() as u64;
    let uncompressed_size = data.len() as u64;

    // LFH ZIP64 extra: tag(2) + size(2) + uncompressed(8) + compressed(8) = 20
    let lfh_extra_len: u16 = 20;
    // CD ZIP64 extra: tag(2) + size(2) + uncompressed(8) + compressed(8) + offset(8) = 28
    let cd_extra_len: u16 = 28;

    let local_header_len: u64 = 30 + u64::from(lfh_extra_len) + name_bytes.len() as u64;
    let central_header_offset: u64 = local_header_len + compressed_size;
    let central_header_len: u64 = 46 + u64::from(cd_extra_len) + name_bytes.len() as u64;

    let mut out = Vec::new();

    // Local File Header (no data descriptor — sizes are known)
    write_u32(&mut out, LOCAL_FILE_HEADER_SIG)?;
    write_u16(&mut out, ZIP_VERSION_ZIP64)?;
    write_u16(&mut out, GP_FLAG_UTF8)?;
    write_u16(&mut out, DEFLATE_METHOD)?;
    write_u16(&mut out, 0)?; // mod time
    write_u16(&mut out, 0)?; // mod date
    write_u32(&mut out, crc)?;
    write_u32(&mut out, 0xFFFF_FFFF)?; // compressed size (ZIP64)
    write_u32(&mut out, 0xFFFF_FFFF)?; // uncompressed size (ZIP64)
    write_u16(&mut out, name_len)?;
    write_u16(&mut out, lfh_extra_len)?;
    out.extend_from_slice(name_bytes);

    // ZIP64 extra field
    write_u16(&mut out, ZIP64_EXTRA_FIELD_TAG)?;
    write_u16(&mut out, 16)?; // data size
    write_u64(&mut out, uncompressed_size)?;
    write_u64(&mut out, compressed_size)?;

    out.extend_from_slice(&compressed);

    // Central Directory Header
    write_u32(&mut out, CENTRAL_DIR_HEADER_SIG)?;
    write_u16(&mut out, ZIP_VERSION_ZIP64)?;
    write_u16(&mut out, ZIP_VERSION_ZIP64)?;
    write_u16(&mut out, GP_FLAG_UTF8)?;
    write_u16(&mut out, DEFLATE_METHOD)?;
    write_u16(&mut out, 0)?; // mod time
    write_u16(&mut out, 0)?; // mod date
    write_u32(&mut out, crc)?;
    write_u32(&mut out, 0xFFFF_FFFF)?; // compressed size (ZIP64)
    write_u32(&mut out, 0xFFFF_FFFF)?; // uncompressed size (ZIP64)
    write_u16(&mut out, name_len)?;
    write_u16(&mut out, cd_extra_len)?;
    write_u16(&mut out, 0)?; // comment len
    write_u16(&mut out, 0)?; // disk number start
    write_u16(&mut out, 0)?; // internal file attributes
    write_u32(&mut out, 0)?; // external file attributes
    write_u32(&mut out, 0xFFFF_FFFF)?; // local header offset (ZIP64)
    out.extend_from_slice(name_bytes);

    // ZIP64 extra field
    write_u16(&mut out, ZIP64_EXTRA_FIELD_TAG)?;
    write_u16(&mut out, 24)?; // data size
    write_u64(&mut out, uncompressed_size)?;
    write_u64(&mut out, compressed_size)?;
    write_u64(&mut out, 0)?; // local header offset

    // ZIP64 EOCD (56 bytes)
    let zip64_eocd_offset = central_header_offset + central_header_len;
    write_u32(&mut out, ZIP64_EOCD_SIG)?;
    write_u64(&mut out, 44)?; // size of remaining record
    write_u16(&mut out, ZIP_VERSION_ZIP64)?;
    write_u16(&mut out, ZIP_VERSION_ZIP64)?;
    write_u32(&mut out, 0)?; // disk number
    write_u32(&mut out, 0)?; // disk with central dir
    write_u64(&mut out, 1)?; // entries on this disk
    write_u64(&mut out, 1)?; // total entries
    write_u64(&mut out, central_header_len)?;
    write_u64(&mut out, central_header_offset)?;

    // ZIP64 EOCD Locator (20 bytes)
    write_u32(&mut out, ZIP64_EOCD_LOCATOR_SIG)?;
    write_u32(&mut out, 0)?;
    write_u64(&mut out, zip64_eocd_offset)?;
    write_u32(&mut out, 1)?;

    // ZIP32 EOCD (22 bytes)
    let cd_size_u32 = u32::try_from(central_header_len).unwrap_or(0xFFFF_FFFF);
    let cd_offset_u32 = u32::try_from(central_header_offset).unwrap_or(0xFFFF_FFFF);
    write_u32(&mut out, EOCD_SIG)?;
    write_u16(&mut out, 0)?;
    write_u16(&mut out, 0)?;
    write_u16(&mut out, 1)?;
    write_u16(&mut out, 1)?;
    write_u32(&mut out, cd_size_u32)?;
    write_u32(&mut out, cd_offset_u32)?;
    write_u16(&mut out, 0)?;

    Ok(out)
}

pub fn zip_decompress_single(zip_bytes: &[u8]) -> Result<ZipEntry, CoZipError> {
    let eocd_offset = find_eocd(zip_bytes).ok_or(CoZipError::InvalidZip("EOCD not found"))?;
    if read_u32(zip_bytes, eocd_offset)? != EOCD_SIG {
        return Err(CoZipError::InvalidZip("invalid EOCD signature"));
    }

    let entries_u16 = read_u16(zip_bytes, eocd_offset + 10)?;
    let central_size_u32 = read_u32(zip_bytes, eocd_offset + 12)?;
    let central_offset_u32 = read_u32(zip_bytes, eocd_offset + 16)?;

    // Check for ZIP64
    let (entry_count, central_size, central_offset) = if entries_u16 == u16::MAX
        || central_size_u32 == u32::MAX
        || central_offset_u32 == u32::MAX
    {
        // Read ZIP64 EOCD Locator (20 bytes before EOCD)
        if eocd_offset < 20 {
            return Err(CoZipError::InvalidZip("ZIP64 EOCD locator not found"));
        }
        let loc_offset = eocd_offset - 20;
        if read_u32(zip_bytes, loc_offset)? != ZIP64_EOCD_LOCATOR_SIG {
            return Err(CoZipError::InvalidZip("ZIP64 EOCD locator not found"));
        }
        let z64_eocd_off = usize_from_u64(
            read_u64(zip_bytes, loc_offset + 8)?,
            "zip64 eocd offset out of range",
        )?;

        if read_u32(zip_bytes, z64_eocd_off)? != ZIP64_EOCD_SIG {
            return Err(CoZipError::InvalidZip("invalid ZIP64 EOCD signature"));
        }

        let entries = read_u64(zip_bytes, z64_eocd_off + 32)?;
        let cd_size = usize_from_u64(
            read_u64(zip_bytes, z64_eocd_off + 40)?,
            "zip64 central directory size out of range",
        )?;
        let cd_offset = usize_from_u64(
            read_u64(zip_bytes, z64_eocd_off + 48)?,
            "zip64 central directory offset out of range",
        )?;
        (entries, cd_size, cd_offset)
    } else {
        (
            u64::from(entries_u16),
            usize::try_from(central_size_u32)
                .map_err(|_| CoZipError::InvalidZip("central directory size out of range"))?,
            usize::try_from(central_offset_u32)
                .map_err(|_| CoZipError::InvalidZip("central directory offset out of range"))?,
        )
    };

    if entry_count != 1 {
        return Err(CoZipError::Unsupported(
            "zip_decompress_single expects exactly one file",
        ));
    }

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
    let compressed_size_u32 = read_u32(zip_bytes, central_offset + 20)?;
    let uncompressed_size_u32 = read_u32(zip_bytes, central_offset + 24)?;
    let file_name_len = read_u16(zip_bytes, central_offset + 28)? as usize;
    let extra_len = read_u16(zip_bytes, central_offset + 30)? as usize;
    let comment_len = read_u16(zip_bytes, central_offset + 32)? as usize;
    let local_header_offset_u32 = read_u32(zip_bytes, central_offset + 42)?;

    let name_start = central_offset + 46;
    let name_end = name_start
        .checked_add(file_name_len)
        .ok_or(CoZipError::InvalidZip("name range overflow"))?;
    let file_name = zip_bytes
        .get(name_start..name_end)
        .ok_or(CoZipError::InvalidZip("name out of range"))?;
    let file_name = String::from_utf8(file_name.to_vec()).map_err(|_| CoZipError::NonUtf8Name)?;

    // Parse ZIP64 extra field from central directory
    let mut compressed_size = usize::try_from(compressed_size_u32)
        .map_err(|_| CoZipError::InvalidZip("compressed size out of range"))?;
    let mut uncompressed_size = usize::try_from(uncompressed_size_u32)
        .map_err(|_| CoZipError::InvalidZip("uncompressed size out of range"))?;
    let mut local_header_offset = usize::try_from(local_header_offset_u32)
        .map_err(|_| CoZipError::InvalidZip("local header offset out of range"))?;

    let extra_start = name_end;
    let extra_end = extra_start
        .checked_add(extra_len)
        .ok_or(CoZipError::InvalidZip("extra range overflow"))?;
    let extra_data = zip_bytes
        .get(extra_start..extra_end)
        .ok_or(CoZipError::InvalidZip("extra out of range"))?;
    let z64 = parse_zip64_extra_field(
        extra_data,
        uncompressed_size_u32 == u32::MAX,
        compressed_size_u32 == u32::MAX,
        local_header_offset_u32 == u32::MAX,
    )?;
    if uncompressed_size_u32 == u32::MAX {
        let value = z64
            .as_ref()
            .and_then(|field| field.uncompressed_size)
            .ok_or(CoZipError::InvalidZip(
                "missing zip64 uncompressed size in central directory",
            ))?;
        uncompressed_size = usize_from_u64(value, "zip64 uncompressed size out of range")?;
    }
    if compressed_size_u32 == u32::MAX {
        let value =
            z64.as_ref()
                .and_then(|field| field.compressed_size)
                .ok_or(CoZipError::InvalidZip(
                    "missing zip64 compressed size in central directory",
                ))?;
        compressed_size = usize_from_u64(value, "zip64 compressed size out of range")?;
    }
    if local_header_offset_u32 == u32::MAX {
        let value = z64
            .as_ref()
            .and_then(|field| field.local_header_offset)
            .ok_or(CoZipError::InvalidZip(
                "missing zip64 local header offset in central directory",
            ))?;
        local_header_offset = usize_from_u64(value, "zip64 local header offset out of range")?;
    }

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
    gp_flags: u16,
    crc: u32,
    compressed_size: u64,
    uncompressed_size: u64,
    local_header_offset: u64,
    czdi_blob: Option<Vec<u8>>,
}

#[derive(Debug, Default)]
struct ZipWriteState {
    central_entries: Vec<ZipCentralWriteEntry>,
    offset: u64,
    stats: CoZipStats,
}

#[derive(Debug, Clone, Copy)]
enum CzdiExtraKind {
    Inline {
        blob_len: u32,
        blob_crc32: u32,
    },
    Eocd64Ref {
        blob_offset: u32,
        blob_len: u32,
        blob_crc32: u32,
    },
    None,
}

#[derive(Debug, Clone)]
struct CzdiResolvedPlan {
    kind: CzdiExtraKind,
    inline_blob: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct CzdiParsedExtra {
    kind: CzdiExtraKind,
    inline_blob: Option<Vec<u8>>,
}

impl ZipWriteState {
    fn write_entry_from_reader<W: Write, R: Read>(
        &mut self,
        writer: &mut W,
        entry_name: &str,
        reader: &mut R,
        deflate: &CoZipDeflate,
    ) -> Result<(), CoZipError> {
        let name = normalize_zip_entry_name(entry_name)?;
        let name_bytes = name.as_bytes();
        let name_len = u16::try_from(name_bytes.len()).map_err(|_| CoZipError::DataTooLarge)?;

        let local_header_offset = self.offset;

        // ZIP64 Extra Field in LFH: tag(2) + size(2) + uncompressed(8) + compressed(8) = 20
        let zip64_extra_len: u16 = 20;

        let gp_flags = GP_FLAG_DATA_DESCRIPTOR | GP_FLAG_UTF8;
        write_u32(writer, LOCAL_FILE_HEADER_SIG)?;
        write_u16(writer, ZIP_VERSION_ZIP64)?;
        write_u16(writer, gp_flags)?;
        write_u16(writer, DEFLATE_METHOD)?;
        write_u16(writer, 0)?; // mod time
        write_u16(writer, 0)?; // mod date
        write_u32(writer, 0)?; // crc (unknown, data descriptor)
        write_u32(writer, 0xFFFF_FFFF)?; // compressed size (ZIP64)
        write_u32(writer, 0xFFFF_FFFF)?; // uncompressed size (ZIP64)
        write_u16(writer, name_len)?;
        write_u16(writer, zip64_extra_len)?;
        writer.write_all(name_bytes)?;

        // ZIP64 extra field (placeholder — sizes unknown before compression)
        write_u16(writer, ZIP64_EXTRA_FIELD_TAG)?;
        write_u16(writer, 16)?; // data size: uncompressed(8) + compressed(8)
        write_u64(writer, 0)?; // uncompressed placeholder
        write_u64(writer, 0)?; // compressed placeholder

        self.offset = self
            .offset
            .checked_add(30)
            .and_then(|v| v.checked_add(u64::from(zip64_extra_len)))
            .and_then(|v| v.checked_add(u64::try_from(name_bytes.len()).ok()?))
            .ok_or(CoZipError::DataTooLarge)?;

        let (crc, compressed_size, uncompressed_size, czdi_blob) =
            stream_deflate_from_reader(writer, reader, deflate)?;

        self.offset = self
            .offset
            .checked_add(compressed_size)
            .ok_or(CoZipError::DataTooLarge)?;

        // ZIP64 Data Descriptor: sig(4) + crc(4) + compressed(8) + uncompressed(8) = 24
        write_u32(writer, DATA_DESCRIPTOR_SIG)?;
        write_u32(writer, crc)?;
        write_u64(writer, compressed_size)?;
        write_u64(writer, uncompressed_size)?;

        self.offset = self
            .offset
            .checked_add(24)
            .ok_or(CoZipError::DataTooLarge)?;

        self.central_entries.push(ZipCentralWriteEntry {
            name,
            gp_flags,
            crc,
            compressed_size,
            uncompressed_size,
            local_header_offset,
            czdi_blob,
        });

        self.stats.entries = self.stats.entries.saturating_add(1);
        self.stats.input_bytes = self
            .stats
            .input_bytes
            .checked_add(uncompressed_size)
            .ok_or(CoZipError::DataTooLarge)?;

        Ok(())
    }

    fn finish<W: Write>(mut self, writer: &mut W) -> Result<CoZipStats, CoZipError> {
        let (czdi_plans, eocd64_czdi_blob) = resolve_czdi_write_plan(&self.central_entries)?;
        let central_dir_offset = self.offset;

        // ZIP64 Extra Field in CD: tag(2) + size(2) + uncompressed(8) + compressed(8) + offset(8) = 28
        let zip64_cd_extra_len: u16 = 28;

        for (entry, czdi_plan) in self.central_entries.iter().zip(czdi_plans.iter()) {
            let name_bytes = entry.name.as_bytes();
            let name_len = u16::try_from(name_bytes.len()).map_err(|_| CoZipError::DataTooLarge)?;
            let czdi_extra = encode_czdi_extra_field(czdi_plan)?;
            let extra_len_total = usize::from(zip64_cd_extra_len)
                .checked_add(czdi_extra.len())
                .ok_or(CoZipError::DataTooLarge)?;
            let extra_len_total_u16 =
                u16::try_from(extra_len_total).map_err(|_| CoZipError::DataTooLarge)?;

            write_u32(writer, CENTRAL_DIR_HEADER_SIG)?;
            write_u16(writer, ZIP_VERSION_ZIP64)?; // version made by
            write_u16(writer, ZIP_VERSION_ZIP64)?; // version needed
            write_u16(writer, entry.gp_flags)?;
            write_u16(writer, DEFLATE_METHOD)?;
            write_u16(writer, 0)?; // mod time
            write_u16(writer, 0)?; // mod date
            write_u32(writer, entry.crc)?;
            write_u32(writer, 0xFFFF_FFFF)?; // compressed size (ZIP64)
            write_u32(writer, 0xFFFF_FFFF)?; // uncompressed size (ZIP64)
            write_u16(writer, name_len)?;
            write_u16(writer, extra_len_total_u16)?;
            write_u16(writer, 0)?; // comment len
            write_u16(writer, 0)?; // disk number start
            write_u16(writer, 0)?; // internal file attributes
            write_u32(writer, 0)?; // external file attributes
            write_u32(writer, 0xFFFF_FFFF)?; // local header offset (ZIP64)
            writer.write_all(name_bytes)?;

            // ZIP64 extra field
            write_u16(writer, ZIP64_EXTRA_FIELD_TAG)?;
            write_u16(writer, 24)?; // data size: uncompressed(8) + compressed(8) + offset(8)
            write_u64(writer, entry.uncompressed_size)?;
            write_u64(writer, entry.compressed_size)?;
            write_u64(writer, entry.local_header_offset)?;
            writer.write_all(&czdi_extra)?;

            self.offset = self
                .offset
                .checked_add(46)
                .and_then(|v| v.checked_add(u64::from(extra_len_total_u16)))
                .and_then(|v| v.checked_add(u64::try_from(name_bytes.len()).ok()?))
                .ok_or(CoZipError::DataTooLarge)?;
        }

        let central_dir_size = self
            .offset
            .checked_sub(central_dir_offset)
            .ok_or(CoZipError::DataTooLarge)?;

        let entry_count = self.central_entries.len() as u64;

        // ZIP64 EOCD (56 + extensible data bytes)
        let zip64_eocd_offset = self.offset;
        let zip64_ext_len_u64 =
            u64::try_from(eocd64_czdi_blob.len()).map_err(|_| CoZipError::DataTooLarge)?;
        write_u32(writer, ZIP64_EOCD_SIG)?;
        write_u64(
            writer,
            44_u64
                .checked_add(zip64_ext_len_u64)
                .ok_or(CoZipError::DataTooLarge)?,
        )?; // size of remaining record
        write_u16(writer, ZIP_VERSION_ZIP64)?; // version made by
        write_u16(writer, ZIP_VERSION_ZIP64)?; // version needed
        write_u32(writer, 0)?; // disk number
        write_u32(writer, 0)?; // disk with central dir
        write_u64(writer, entry_count)?; // entries on this disk
        write_u64(writer, entry_count)?; // total entries
        write_u64(writer, central_dir_size)?;
        write_u64(writer, central_dir_offset)?;
        if !eocd64_czdi_blob.is_empty() {
            writer.write_all(&eocd64_czdi_blob)?;
        }

        self.offset = self
            .offset
            .checked_add(56)
            .and_then(|v| v.checked_add(zip64_ext_len_u64))
            .ok_or(CoZipError::DataTooLarge)?;

        // ZIP64 EOCD Locator (20 bytes)
        write_u32(writer, ZIP64_EOCD_LOCATOR_SIG)?;
        write_u32(writer, 0)?; // disk with ZIP64 EOCD
        write_u64(writer, zip64_eocd_offset)?;
        write_u32(writer, 1)?; // total disks

        self.offset = self
            .offset
            .checked_add(20)
            .ok_or(CoZipError::DataTooLarge)?;

        // ZIP32 EOCD (22 bytes) with sentinel values
        let entries_u16 = if entry_count > u64::from(u16::MAX - 1) {
            0xFFFF
        } else {
            entry_count as u16
        };
        let cd_size_u32 = u32::try_from(central_dir_size).unwrap_or(0xFFFF_FFFF);
        let cd_offset_u32 = u32::try_from(central_dir_offset).unwrap_or(0xFFFF_FFFF);

        write_u32(writer, EOCD_SIG)?;
        write_u16(writer, 0)?; // disk number
        write_u16(writer, 0)?; // disk with central dir
        write_u16(writer, entries_u16)?;
        write_u16(writer, entries_u16)?;
        write_u32(writer, cd_size_u32)?;
        write_u32(writer, cd_offset_u32)?;
        write_u16(writer, 0)?; // comment len

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
    compressed_size: u64,
    uncompressed_size: u64,
    local_header_offset: u64,
    _czdi_index: Option<DeflateChunkIndex>,
}

fn stream_deflate_from_reader<W: Write, R: Read>(
    writer: &mut W,
    reader: &mut R,
    deflate: &CoZipDeflate,
) -> Result<(u32, u64, u64, Option<Vec<u8>>), CoZipError> {
    let result = deflate.deflate_compress_stream_zip_compatible_with_index(reader, writer)?;
    let index_blob = result
        .index
        .map(|index| index.encode_czdi_v1())
        .transpose()?;
    Ok((
        result.stats.input_crc32,
        result.stats.output_bytes,
        result.stats.input_bytes,
        index_blob,
    ))
}

fn read_central_directory_entries<R: Read + Seek>(
    reader: &mut R,
) -> Result<(Vec<ZipCentralReadEntry>, u64), CoZipError> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    let eocd = read_eocd(reader, file_len)?;
    let czdi_eocd_blob = match eocd.zip64_extensible_data.as_deref() {
        Some(ext) => decode_czdi_eocd64_blob(ext)?,
        None => None,
    };

    if eocd
        .central_offset
        .checked_add(eocd.central_size)
        .ok_or(CoZipError::InvalidZip("central directory overflow"))?
        > file_len
    {
        return Err(CoZipError::InvalidZip("central directory out of range"));
    }

    reader.seek(SeekFrom::Start(eocd.central_offset))?;
    let mut entries = Vec::with_capacity(usize_from_u64(eocd.entries, "entry count too large")?);

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
        let compressed_size_u32 = u32::from_le_bytes(
            fixed[20..24]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("compressed size parse failed"))?,
        );
        let uncompressed_size_u32 = u32::from_le_bytes(
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
        let local_header_offset_u32 = u32::from_le_bytes(
            fixed[42..46]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("local offset parse failed"))?,
        );

        let mut name = vec![0_u8; name_len];
        reader.read_exact(&mut name)?;
        let name = String::from_utf8(name).map_err(|_| CoZipError::NonUtf8Name)?;

        // Read extra field data
        let mut extra_data = vec![0_u8; extra_len];
        reader.read_exact(&mut extra_data)?;

        // Parse ZIP64 extra field if present
        let mut compressed_size = u64::from(compressed_size_u32);
        let mut uncompressed_size = u64::from(uncompressed_size_u32);
        let mut local_header_offset = u64::from(local_header_offset_u32);

        let z64 = parse_zip64_extra_field(
            &extra_data,
            uncompressed_size_u32 == u32::MAX,
            compressed_size_u32 == u32::MAX,
            local_header_offset_u32 == u32::MAX,
        )?;
        if uncompressed_size_u32 == u32::MAX {
            uncompressed_size = z64
                .as_ref()
                .and_then(|field| field.uncompressed_size)
                .ok_or(CoZipError::InvalidZip(
                    "missing zip64 uncompressed size in central directory",
                ))?;
        }
        if compressed_size_u32 == u32::MAX {
            compressed_size = z64.as_ref().and_then(|field| field.compressed_size).ok_or(
                CoZipError::InvalidZip("missing zip64 compressed size in central directory"),
            )?;
        }
        if local_header_offset_u32 == u32::MAX {
            local_header_offset = z64
                .as_ref()
                .and_then(|field| field.local_header_offset)
                .ok_or(CoZipError::InvalidZip(
                    "missing zip64 local header offset in central directory",
                ))?;
        }

        let czdi_parsed = parse_czdi_extra_field(&extra_data)?;
        let czdi_blob = match czdi_parsed {
            Some(CzdiParsedExtra {
                kind:
                    CzdiExtraKind::Inline {
                        blob_len: _,
                        blob_crc32: _,
                    },
                inline_blob,
            }) => inline_blob,
            Some(CzdiParsedExtra {
                kind:
                    CzdiExtraKind::Eocd64Ref {
                        blob_offset,
                        blob_len,
                        blob_crc32,
                    },
                inline_blob: _,
            }) => {
                let area = czdi_eocd_blob
                    .as_ref()
                    .ok_or(CoZipError::InvalidZip("czdi eocd64 blob is missing"))?;
                let start = usize::try_from(blob_offset)
                    .map_err(|_| CoZipError::InvalidZip("czdi blob offset out of range"))?;
                let len = usize::try_from(blob_len)
                    .map_err(|_| CoZipError::InvalidZip("czdi blob length out of range"))?;
                let end = start
                    .checked_add(len)
                    .ok_or(CoZipError::InvalidZip("czdi blob range overflow"))?;
                let blob = area
                    .get(start..end)
                    .ok_or(CoZipError::InvalidZip("czdi blob range is invalid"))?;
                if crc32fast::hash(blob) != blob_crc32 {
                    return Err(CoZipError::InvalidZip("czdi eocd64 blob crc mismatch"));
                }
                Some(blob.to_vec())
            }
            Some(CzdiParsedExtra {
                kind: CzdiExtraKind::None,
                inline_blob: _,
            })
            | None => None,
        };
        let czdi_index = czdi_blob
            .as_deref()
            .map(DeflateChunkIndex::decode_czdi_v1)
            .transpose()
            .map_err(|_| CoZipError::InvalidZip("czdi index decode failed"))?;

        // Skip comment
        if comment_len > 0 {
            let skip = i64::try_from(comment_len).map_err(|_| CoZipError::DataTooLarge)?;
            reader.seek(SeekFrom::Current(skip))?;
        }

        entries.push(ZipCentralReadEntry {
            name,
            gp_flags,
            method,
            crc,
            compressed_size,
            uncompressed_size,
            local_header_offset,
            _czdi_index: czdi_index,
        });
    }

    Ok((entries, file_len))
}

fn extract_entry_to_writer<R: Read + Seek, W: Write>(
    reader: &mut R,
    entry: &ZipCentralReadEntry,
    writer: &mut W,
    deflate: &CoZipDeflate,
) -> Result<u64, CoZipError> {
    reader.seek(SeekFrom::Start(entry.local_header_offset))?;

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

    // Skip name, read extra field
    let name_skip = i64::try_from(local_name_len).map_err(|_| CoZipError::DataTooLarge)?;
    reader.seek(SeekFrom::Current(name_skip))?;

    let mut local_extra = vec![0_u8; local_extra_len];
    reader.read_exact(&mut local_extra)?;

    // Use compressed_size from central directory (already ZIP64-resolved)
    let mut compressed_size = entry.compressed_size;

    // If central directory had 0xFFFFFFFF, also check local extra field
    if compressed_size == u64::from(u32::MAX) {
        let z64 = parse_zip64_extra_field(&local_extra, false, true, false)?;
        compressed_size =
            z64.and_then(|field| field.compressed_size)
                .ok_or(CoZipError::InvalidZip(
                    "missing zip64 compressed size in local header",
                ))?;
    }

    let mut limited = reader.take(compressed_size);
    let mut written: u64;
    let mut buf = vec![0_u8; STREAM_BUF_SIZE];

    match entry.method {
        DEFLATE_METHOD => {
            let mut compressed_payload = Vec::new();
            limited.read_to_end(&mut compressed_payload)?;
            if u64::try_from(compressed_payload.len()).unwrap_or(u64::MAX) != compressed_size {
                return Err(CoZipError::InvalidZip(
                    "compressed stream did not consume declared size",
                ));
            }

            let decode_cpu_legacy =
                |writer: &mut W| -> Result<cozip_deflate::DeflateCpuStreamStats, CoZipError> {
                    let mut reader = Cursor::new(compressed_payload.as_slice());
                    Ok(deflate_decompress_stream_on_cpu(&mut reader, writer)?)
                };

            let decode_indexed_to_vec =
                || -> Result<(cozip_deflate::DeflateCpuStreamStats, Vec<u8>), CoZipError> {
                    let index = entry._czdi_index.as_ref().ok_or(CoZipError::InvalidZip(
                        "indexed decode requested without CZDI",
                    ))?;
                    let mut out = Vec::new();
                    let mut reader = Cursor::new(compressed_payload.as_slice());
                    match deflate.deflate_decompress_stream_zip_compatible_with_index(
                        &mut reader,
                        &mut out,
                        index,
                    ) {
                        Ok(stats) => Ok((stats, out)),
                        Err(CozipDeflateError::GpuExecution(_))
                        | Err(CozipDeflateError::GpuUnavailable(_)) => {
                            let mut out = Vec::new();
                            let mut cpu_reader = Cursor::new(compressed_payload.as_slice());
                            let stats = deflate
                                .deflate_decompress_stream_zip_compatible_with_index_cpu(
                                    &mut cpu_reader,
                                    &mut out,
                                    index,
                                )?;
                            Ok((stats, out))
                        }
                        Err(err) => Err(CoZipError::Deflate(err)),
                    }
                };

            let stats = if entry._czdi_index.is_some() {
                match decode_indexed_to_vec() {
                    Ok((stats, decoded)) => {
                        if stats.output_crc32 != entry.crc {
                            return Err(CoZipError::InvalidZip("crc32 mismatch"));
                        }
                        writer.write_all(&decoded)?;
                        written = stats.output_bytes;
                        stats
                    }
                    Err(_) => {
                        let stats = decode_cpu_legacy(writer)?;
                        written = stats.output_bytes;
                        stats
                    }
                }
            } else {
                let stats = decode_cpu_legacy(writer)?;
                written = stats.output_bytes;
                stats
            };

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

    if written != entry.uncompressed_size {
        return Err(CoZipError::InvalidZip("decompressed size mismatch"));
    }

    if (entry.gp_flags & 0x0001) != 0 {
        return Err(CoZipError::Unsupported(
            "encrypted zip entries are unsupported",
        ));
    }

    Ok(written)
}

#[derive(Debug, Clone)]
struct Eocd {
    entries: u64,
    central_size: u64,
    central_offset: u64,
    zip64_extensible_data: Option<Vec<u8>>,
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

    let min_eocd_end = eocd_offset
        .checked_add(22)
        .ok_or(CoZipError::DataTooLarge)?;
    if min_eocd_end > file_len {
        return Err(CoZipError::InvalidZip("EOCD out of range"));
    }

    let entries_u16 = read_u16(&tail, rel + 10)?;
    let central_size_u32 = read_u32(&tail, rel + 12)?;
    let central_offset_u32 = read_u32(&tail, rel + 16)?;

    let needs_zip64 =
        entries_u16 == u16::MAX || central_size_u32 == u32::MAX || central_offset_u32 == u32::MAX;

    if needs_zip64 {
        // Look for ZIP64 EOCD Locator at eocd_offset - 20
        if eocd_offset < 20 {
            return Err(CoZipError::InvalidZip(
                "ZIP64 EOCD locator not found (file too small)",
            ));
        }
        let locator_offset = eocd_offset - 20;
        reader.seek(SeekFrom::Start(locator_offset))?;
        let mut locator_buf = [0_u8; 20];
        reader.read_exact(&mut locator_buf)?;

        let locator_sig = u32::from_le_bytes(
            locator_buf[0..4]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("locator sig parse failed"))?,
        );
        if locator_sig != ZIP64_EOCD_LOCATOR_SIG {
            return Err(CoZipError::InvalidZip("ZIP64 EOCD locator not found"));
        }

        let zip64_eocd_offset = u64::from_le_bytes(
            locator_buf[8..16]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("zip64 eocd offset parse failed"))?,
        );

        // Read ZIP64 EOCD header prefix (sig + size)
        reader.seek(SeekFrom::Start(zip64_eocd_offset))?;
        let mut z64_prefix = [0_u8; 12];
        reader.read_exact(&mut z64_prefix)?;
        let z64_sig = u32::from_le_bytes(
            z64_prefix[0..4]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("zip64 eocd sig parse failed"))?,
        );
        if z64_sig != ZIP64_EOCD_SIG {
            return Err(CoZipError::InvalidZip("invalid ZIP64 EOCD signature"));
        }
        let z64_record_size = u64::from_le_bytes(
            z64_prefix[4..12]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("zip64 eocd size parse failed"))?,
        );
        if z64_record_size < 44 {
            return Err(CoZipError::InvalidZip("zip64 eocd record too short"));
        }
        let z64_tail_len = usize_from_u64(z64_record_size, "zip64 eocd size too large")?;
        let mut z64_tail = vec![0_u8; z64_tail_len];
        reader.read_exact(&mut z64_tail)?;

        let entries = u64::from_le_bytes(
            z64_tail[20..28]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("zip64 entries parse failed"))?,
        );
        let central_size = u64::from_le_bytes(
            z64_tail[28..36]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("zip64 cd size parse failed"))?,
        );
        let central_offset = u64::from_le_bytes(
            z64_tail[36..44]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("zip64 cd offset parse failed"))?,
        );
        let zip64_extensible_data = if z64_tail.len() > 44 {
            Some(z64_tail[44..].to_vec())
        } else {
            None
        };

        Ok(Eocd {
            entries,
            central_size,
            central_offset,
            zip64_extensible_data,
        })
    } else {
        Ok(Eocd {
            entries: u64::from(entries_u16),
            central_size: u64::from(central_size_u32),
            central_offset: u64::from(central_offset_u32),
            zip64_extensible_data: None,
        })
    }
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

fn write_u64<W: Write>(out: &mut W, value: u64) -> Result<(), CoZipError> {
    out.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn usize_from_u64(value: u64, message: &'static str) -> Result<usize, CoZipError> {
    usize::try_from(value).map_err(|_| CoZipError::InvalidZip(message))
}

#[derive(Debug)]
struct Zip64ExtraField {
    uncompressed_size: Option<u64>,
    compressed_size: Option<u64>,
    local_header_offset: Option<u64>,
}

fn parse_zip64_extra_field(
    extra: &[u8],
    needs_uncompressed_size: bool,
    needs_compressed_size: bool,
    needs_local_header_offset: bool,
) -> Result<Option<Zip64ExtraField>, CoZipError> {
    let mut pos = 0;
    while pos + 4 <= extra.len() {
        let tag = u16::from_le_bytes(
            extra[pos..pos + 2]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("zip64 extra tag parse failed"))?,
        );
        let size = usize::from(u16::from_le_bytes(
            extra[pos + 2..pos + 4]
                .try_into()
                .map_err(|_| CoZipError::InvalidZip("zip64 extra size parse failed"))?,
        ));
        pos += 4;
        let end = pos
            .checked_add(size)
            .ok_or(CoZipError::InvalidZip("zip64 extra field overflow"))?;
        let data = extra
            .get(pos..end)
            .ok_or(CoZipError::InvalidZip("zip64 extra field truncated"))?;
        if tag == ZIP64_EXTRA_FIELD_TAG {
            let mut offset: usize = 0;
            let mut uncompressed_size = None;
            let mut compressed_size = None;
            let mut local_header_offset = None;

            if needs_uncompressed_size {
                let next = offset
                    .checked_add(8)
                    .ok_or(CoZipError::InvalidZip("zip64 uncompressed size overflow"))?;
                let bytes = data
                    .get(offset..next)
                    .ok_or(CoZipError::InvalidZip("zip64 uncompressed size missing"))?;
                let v =
                    u64::from_le_bytes(bytes.try_into().map_err(|_| {
                        CoZipError::InvalidZip("zip64 uncompressed size parse failed")
                    })?);
                offset += 8;
                uncompressed_size = Some(v);
            }
            if needs_compressed_size {
                let next = offset
                    .checked_add(8)
                    .ok_or(CoZipError::InvalidZip("zip64 compressed size overflow"))?;
                let bytes = data
                    .get(offset..next)
                    .ok_or(CoZipError::InvalidZip("zip64 compressed size missing"))?;
                let v =
                    u64::from_le_bytes(bytes.try_into().map_err(|_| {
                        CoZipError::InvalidZip("zip64 compressed size parse failed")
                    })?);
                offset += 8;
                compressed_size = Some(v);
            }
            if needs_local_header_offset {
                let next = offset
                    .checked_add(8)
                    .ok_or(CoZipError::InvalidZip("zip64 local offset overflow"))?;
                let bytes = data
                    .get(offset..next)
                    .ok_or(CoZipError::InvalidZip("zip64 local offset missing"))?;
                let v = u64::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| CoZipError::InvalidZip("zip64 local offset parse failed"))?,
                );
                local_header_offset = Some(v);
            }
            return Ok(Some(Zip64ExtraField {
                uncompressed_size,
                compressed_size,
                local_header_offset,
            }));
        }
        pos = end;
    }
    Ok(None)
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

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, CoZipError> {
    let end = offset
        .checked_add(8)
        .ok_or(CoZipError::InvalidZip("u64 overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or(CoZipError::InvalidZip("u64 out of range"))?;
    let array: [u8; 8] = slice
        .try_into()
        .map_err(|_| CoZipError::InvalidZip("u64 parse failed"))?;
    Ok(u64::from_le_bytes(array))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_single_roundtrip() {
        let input = b"cozip zip test cozip zip test cozip zip test".to_vec();
        let mut opts = HybridOptions::default();
        opts.prefer_gpu = false;
        let deflate = CoZipDeflate::init(opts).expect("deflate init");
        let zip = zip_compress_single("message.txt", &input, &deflate)
            .expect("zip compression should succeed");

        let entry = zip_decompress_single(&zip).expect("zip decompression should succeed");
        assert_eq!(entry.name, "message.txt");
        assert_eq!(entry.data, input);
    }

    #[test]
    fn cozip_compress_file_roundtrip() {
        let cozip = CoZip::init(CoZipOptions::default()).expect("init");
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
        let cozip = CoZip::init(CoZipOptions::default()).expect("init");
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

    #[test]
    fn compression_mode_mapping_matches_level_ranges() {
        assert_eq!(compression_mode_from_level(0), CompressionMode::Speed);
        assert_eq!(compression_mode_from_level(3), CompressionMode::Speed);
        assert_eq!(compression_mode_from_level(4), CompressionMode::Balanced);
        assert_eq!(compression_mode_from_level(6), CompressionMode::Balanced);
        assert_eq!(compression_mode_from_level(7), CompressionMode::Ratio);
        assert_eq!(compression_mode_from_level(9), CompressionMode::Ratio);
        assert_eq!(
            compression_mode_from_level(ZipOptions::default().compression_level),
            CompressionMode::Balanced
        );
    }

    #[test]
    fn zip64_extra_field_parses_offset_only_layout() {
        let offset_value = 0x0102_0304_0506_0708_u64;
        let mut extra = Vec::new();
        extra.extend_from_slice(&ZIP64_EXTRA_FIELD_TAG.to_le_bytes());
        extra.extend_from_slice(&8_u16.to_le_bytes());
        extra.extend_from_slice(&offset_value.to_le_bytes());

        let parsed = parse_zip64_extra_field(&extra, false, false, true)
            .expect("zip64 extra parse should succeed")
            .expect("zip64 extra should be found");
        assert_eq!(parsed.local_header_offset, Some(offset_value));
        assert_eq!(parsed.uncompressed_size, None);
        assert_eq!(parsed.compressed_size, None);
    }

    #[test]
    fn zip64_extra_field_errors_when_required_value_is_missing() {
        let mut extra = Vec::new();
        extra.extend_from_slice(&ZIP64_EXTRA_FIELD_TAG.to_le_bytes());
        extra.extend_from_slice(&8_u16.to_le_bytes());
        extra.extend_from_slice(&123_u64.to_le_bytes());

        let err = parse_zip64_extra_field(&extra, true, true, false)
            .expect_err("missing required compressed size should fail");
        assert!(matches!(err, CoZipError::InvalidZip(_)));
    }

    #[test]
    fn czdi_inline_extra_roundtrip() {
        let index = DeflateChunkIndex {
            chunk_size: 4 * 1024 * 1024,
            chunk_count: 1,
            uncompressed_size: 1234,
            compressed_size: 567,
            entries: vec![cozip_deflate::DeflateChunkIndexEntry {
                comp_bit_off: 0,
                comp_bit_len: 567 * 8,
                final_header_rel_bit: 0,
                raw_len: 1234,
            }],
        };
        let blob = index.encode_czdi_v1().expect("encode czdi");
        let plan = CzdiResolvedPlan {
            kind: CzdiExtraKind::Inline {
                blob_len: u32::try_from(blob.len()).expect("blob len"),
                blob_crc32: crc32fast::hash(&blob),
            },
            inline_blob: Some(blob.clone()),
        };
        let extra = encode_czdi_extra_field(&plan).expect("encode extra");
        let parsed = parse_czdi_extra_field(&extra)
            .expect("parse extra")
            .expect("czdi extra exists");
        let inline = parsed.inline_blob.expect("inline payload");
        assert_eq!(inline, blob);
    }

    #[test]
    fn czdi_overflow_uses_eocd64_blob_storage() {
        let large_blob = vec![0xAB; 70_000];
        let entries = vec![ZipCentralWriteEntry {
            name: "big.bin".to_string(),
            gp_flags: GP_FLAG_DATA_DESCRIPTOR | GP_FLAG_UTF8,
            crc: 0,
            compressed_size: 0,
            uncompressed_size: 0,
            local_header_offset: 0,
            czdi_blob: Some(large_blob.clone()),
        }];
        let (plans, eocd_blob) = resolve_czdi_write_plan(&entries).expect("resolve plan");
        assert_eq!(plans.len(), 1);
        let CzdiExtraKind::Eocd64Ref {
            blob_offset,
            blob_len,
            blob_crc32,
        } = plans[0].kind
        else {
            panic!("expected eocd64 ref plan");
        };
        let area = decode_czdi_eocd64_blob(&eocd_blob)
            .expect("decode eocd blob")
            .expect("eocd area");
        let start = usize::try_from(blob_offset).expect("offset");
        let len = usize::try_from(blob_len).expect("len");
        let end = start + len;
        let slice = &area[start..end];
        assert_eq!(crc32fast::hash(slice), blob_crc32);
        assert_eq!(slice, large_blob.as_slice());
    }

    #[test]
    fn cozip_written_zip_contains_czdi_index_in_central_directory() {
        let cozip = CoZip::init(CoZipOptions::default()).expect("init");
        let mut input = std::env::temp_dir();
        input.push(format!("cozip-czdi-input-{}.bin", std::process::id()));
        let mut output = std::env::temp_dir();
        output.push(format!("cozip-czdi-output-{}.zip", std::process::id()));

        std::fs::write(&input, vec![1_u8; 128 * 1024]).expect("write input");
        cozip
            .compress_file_from_name(&input, &output)
            .expect("compress file");

        let file = StdFile::open(&output).expect("open output zip");
        let mut reader = BufReader::new(file);
        let (entries, _) = read_central_directory_entries(&mut reader).expect("read central dir");
        assert_eq!(entries.len(), 1);
        assert!(entries[0]._czdi_index.is_some());

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }
}
