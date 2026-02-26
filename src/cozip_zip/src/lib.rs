use cozip_deflate::{CozipDeflateError, deflate_compress_cpu, deflate_decompress_on_cpu};
use thiserror::Error;

const LOCAL_FILE_HEADER_SIG: u32 = 0x0403_4b50;
const CENTRAL_DIR_HEADER_SIG: u32 = 0x0201_4b50;
const EOCD_SIG: u32 = 0x0605_4b50;

#[derive(Debug, Clone)]
pub struct ZipOptions {
    pub compression_level: u32,
}

impl Default for ZipOptions {
    fn default() -> Self {
        Self {
            compression_level: 6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZipEntry {
    pub name: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum CozipZipError {
    #[error("invalid zip: {0}")]
    InvalidZip(&'static str),
    #[error("unsupported zip: {0}")]
    Unsupported(&'static str),
    #[error("deflate error: {0}")]
    Deflate(#[from] CozipDeflateError),
    #[error("path contains non-utf8 bytes")]
    NonUtf8Name,
    #[error("data too large for zip32")]
    DataTooLarge,
}

pub fn zip_compress_single(
    file_name: &str,
    data: &[u8],
    options: &ZipOptions,
) -> Result<Vec<u8>, CozipZipError> {
    if file_name.is_empty() {
        return Err(CozipZipError::InvalidZip("file name is empty"));
    }

    let name_bytes = file_name.as_bytes();
    let name_len = u16::try_from(name_bytes.len()).map_err(|_| CozipZipError::DataTooLarge)?;
    let compressed = deflate_compress_cpu(data, options.compression_level)?;

    let crc = crc32fast::hash(data);
    let compressed_size =
        u32::try_from(compressed.len()).map_err(|_| CozipZipError::DataTooLarge)?;
    let uncompressed_size = u32::try_from(data.len()).map_err(|_| CozipZipError::DataTooLarge)?;

    let local_header_len = 30_usize
        .checked_add(name_bytes.len())
        .ok_or(CozipZipError::DataTooLarge)?;
    let central_header_offset = local_header_len
        .checked_add(compressed.len())
        .ok_or(CozipZipError::DataTooLarge)?;
    let central_header_len = 46_usize
        .checked_add(name_bytes.len())
        .ok_or(CozipZipError::DataTooLarge)?;

    let mut out = Vec::with_capacity(
        central_header_offset
            .checked_add(central_header_len)
            .and_then(|v| v.checked_add(22))
            .ok_or(CozipZipError::DataTooLarge)?,
    );

    write_u32(&mut out, LOCAL_FILE_HEADER_SIG);
    write_u16(&mut out, 20);
    write_u16(&mut out, 0);
    write_u16(&mut out, 8);
    write_u16(&mut out, 0);
    write_u16(&mut out, 0);
    write_u32(&mut out, crc);
    write_u32(&mut out, compressed_size);
    write_u32(&mut out, uncompressed_size);
    write_u16(&mut out, name_len);
    write_u16(&mut out, 0);
    out.extend_from_slice(name_bytes);
    out.extend_from_slice(&compressed);

    let central_offset_u32 =
        u32::try_from(central_header_offset).map_err(|_| CozipZipError::DataTooLarge)?;
    let central_size_u32 =
        u32::try_from(central_header_len).map_err(|_| CozipZipError::DataTooLarge)?;

    write_u32(&mut out, CENTRAL_DIR_HEADER_SIG);
    write_u16(&mut out, 20);
    write_u16(&mut out, 20);
    write_u16(&mut out, 0);
    write_u16(&mut out, 8);
    write_u16(&mut out, 0);
    write_u16(&mut out, 0);
    write_u32(&mut out, crc);
    write_u32(&mut out, compressed_size);
    write_u32(&mut out, uncompressed_size);
    write_u16(&mut out, name_len);
    write_u16(&mut out, 0);
    write_u16(&mut out, 0);
    write_u16(&mut out, 0);
    write_u16(&mut out, 0);
    write_u32(&mut out, 0);
    write_u32(&mut out, 0);
    out.extend_from_slice(name_bytes);

    write_u32(&mut out, EOCD_SIG);
    write_u16(&mut out, 0);
    write_u16(&mut out, 0);
    write_u16(&mut out, 1);
    write_u16(&mut out, 1);
    write_u32(&mut out, central_size_u32);
    write_u32(&mut out, central_offset_u32);
    write_u16(&mut out, 0);

    Ok(out)
}

pub fn zip_decompress_single(zip_bytes: &[u8]) -> Result<ZipEntry, CozipZipError> {
    let eocd_offset = find_eocd(zip_bytes).ok_or(CozipZipError::InvalidZip("EOCD not found"))?;
    if read_u32(zip_bytes, eocd_offset)? != EOCD_SIG {
        return Err(CozipZipError::InvalidZip("invalid EOCD signature"));
    }

    let entries = read_u16(zip_bytes, eocd_offset + 10)?;
    if entries != 1 {
        return Err(CozipZipError::Unsupported(
            "zip_decompress_single expects exactly one file",
        ));
    }

    let central_size = read_u32(zip_bytes, eocd_offset + 12)? as usize;
    let central_offset = read_u32(zip_bytes, eocd_offset + 16)? as usize;
    let central_end = central_offset
        .checked_add(central_size)
        .ok_or(CozipZipError::InvalidZip("central directory overflow"))?;
    if central_end > zip_bytes.len() {
        return Err(CozipZipError::InvalidZip("central directory out of range"));
    }

    if read_u32(zip_bytes, central_offset)? != CENTRAL_DIR_HEADER_SIG {
        return Err(CozipZipError::InvalidZip(
            "invalid central directory signature",
        ));
    }

    let method = read_u16(zip_bytes, central_offset + 10)?;
    if method != 8 && method != 0 {
        return Err(CozipZipError::Unsupported(
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
        .ok_or(CozipZipError::InvalidZip("name range overflow"))?;
    let file_name = zip_bytes
        .get(name_start..name_end)
        .ok_or(CozipZipError::InvalidZip("name out of range"))?;
    let file_name =
        String::from_utf8(file_name.to_vec()).map_err(|_| CozipZipError::NonUtf8Name)?;

    let local_name_len = read_u16(zip_bytes, local_header_offset + 26)? as usize;
    let local_extra_len = read_u16(zip_bytes, local_header_offset + 28)? as usize;
    if read_u32(zip_bytes, local_header_offset)? != LOCAL_FILE_HEADER_SIG {
        return Err(CozipZipError::InvalidZip(
            "invalid local file header signature",
        ));
    }

    let data_start = local_header_offset
        .checked_add(30)
        .and_then(|v| v.checked_add(local_name_len))
        .and_then(|v| v.checked_add(local_extra_len))
        .ok_or(CozipZipError::InvalidZip("local data range overflow"))?;
    let data_end = data_start
        .checked_add(compressed_size)
        .ok_or(CozipZipError::InvalidZip("compressed data range overflow"))?;
    let compressed = zip_bytes
        .get(data_start..data_end)
        .ok_or(CozipZipError::InvalidZip("compressed data out of range"))?;

    let data = if method == 8 {
        deflate_decompress_on_cpu(compressed)?
    } else {
        compressed.to_vec()
    };

    if data.len() != uncompressed_size {
        return Err(CozipZipError::InvalidZip(
            "decompressed size mismatch against directory",
        ));
    }

    let actual_crc = crc32fast::hash(&data);
    if actual_crc != crc {
        return Err(CozipZipError::InvalidZip("crc32 mismatch"));
    }

    let consumed = 46_usize
        .checked_add(file_name_len)
        .and_then(|v| v.checked_add(extra_len))
        .and_then(|v| v.checked_add(comment_len))
        .ok_or(CozipZipError::InvalidZip("central record length overflow"))?;
    if central_offset + consumed > central_end {
        return Err(CozipZipError::InvalidZip("central record is truncated"));
    }

    Ok(ZipEntry {
        name: file_name,
        data,
    })
}

fn find_eocd(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 22 {
        return None;
    }

    let search_start = bytes.len().saturating_sub(22 + 65_535);
    (search_start..=bytes.len() - 22)
        .rev()
        .find(|offset| bytes[*offset..*offset + 4] == EOCD_SIG.to_le_bytes())
}

fn write_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, CozipZipError> {
    let end = offset
        .checked_add(2)
        .ok_or(CozipZipError::InvalidZip("u16 overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or(CozipZipError::InvalidZip("u16 out of range"))?;
    let array: [u8; 2] = slice
        .try_into()
        .map_err(|_| CozipZipError::InvalidZip("u16 parse failed"))?;
    Ok(u16::from_le_bytes(array))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, CozipZipError> {
    let end = offset
        .checked_add(4)
        .ok_or(CozipZipError::InvalidZip("u32 overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or(CozipZipError::InvalidZip("u32 out of range"))?;
    let array: [u8; 4] = slice
        .try_into()
        .map_err(|_| CozipZipError::InvalidZip("u32 parse failed"))?;
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
}
