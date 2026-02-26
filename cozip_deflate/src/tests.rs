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
    let restored = deflate_decompress_on_cpu(&compressed).expect("decompression should succeed");
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
    let options = HybridOptions {
        prefer_gpu: false,
        gpu_fraction: 0.0,
        ..HybridOptions::default()
    };

    let compressed = compress_hybrid(&input, &options).expect("hybrid compress should succeed");
    let decompressed =
        decompress_hybrid(&compressed.bytes, &options).expect("hybrid decompress should succeed");

    assert_eq!(decompressed.bytes, input);
    assert_eq!(compressed.stats.chunk_count, decompressed.stats.chunk_count);
}

#[test]
fn frame_corruption_is_detected() {
    let input = b"hello cozip";
    let options = HybridOptions {
        prefer_gpu: false,
        gpu_fraction: 0.0,
        ..HybridOptions::default()
    };
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
    frame.extend_from_slice(&1_u32.to_le_bytes());
    frame.extend_from_slice(&u32::try_from(input.len()).expect("len fits").to_le_bytes());
    frame.extend_from_slice(&u64::try_from(input.len()).expect("len fits").to_le_bytes());
    frame.push(ChunkBackend::Cpu.to_u8());
    frame.push(ChunkTransform::None.to_u8());
    frame.extend_from_slice(&u32::try_from(input.len()).expect("len fits").to_le_bytes());
    frame.extend_from_slice(
        &u32::try_from(compressed.len())
            .expect("compressed len fits")
            .to_le_bytes(),
    );
    frame.extend_from_slice(&compressed);

    let decoded = decompress_hybrid(
        &frame,
        &HybridOptions {
            prefer_gpu: false,
            gpu_fraction: 0.0,
            ..HybridOptions::default()
        },
    )
    .expect("decode should succeed");
    assert_eq!(decoded.bytes, input);
}
