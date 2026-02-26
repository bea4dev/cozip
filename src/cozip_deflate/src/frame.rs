use super::*;

pub(super) fn encode_frame(
    original_len: usize,
    chunks: &[ChunkMember],
) -> Result<Vec<u8>, CozipDeflateError> {
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

pub(super) fn parse_frame(
    frame: &[u8],
) -> Result<(usize, Vec<ChunkDescriptor>), CozipDeflateError> {
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
