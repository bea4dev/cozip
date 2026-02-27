use super::*;

fn patterned_data(len: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(len);
    for i in 0..len {
        data.push(((i as u32 * 31 + 7) % 251) as u8);
    }
    data
}

fn bench_mixed_data(bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes);
    let mut state: u32 = 0x1234_5678;
    while out.len() < bytes {
        let zone = (out.len() / 4096) % 3;
        match zone {
            0 => out.extend_from_slice(b"cozip-cpu-gpu-deflate-"),
            1 => out.extend_from_slice(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            _ => {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                out.push((state >> 24) as u8);
            }
        }
    }
    out.truncate(bytes);
    out
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

#[test]
fn stream_roundtrip_with_chunked_io() {
    let options = HybridOptions {
        prefer_gpu: false,
        gpu_fraction: 0.0,
        ..HybridOptions::default()
    };
    let cozip = CoZipDeflate::init(options).expect("init should succeed");
    let input = patterned_data((1024 * 1024 * 3) + 123);

    let mut compressed_stream = Vec::new();
    let mut src_reader = std::io::Cursor::new(input.clone());
    let compress_stats = cozip
        .compress_stream(
            &mut src_reader,
            &mut compressed_stream,
            StreamOptions {
                frame_input_size: 256 * 1024,
            },
        )
        .expect("stream compression should succeed");

    assert!(compress_stats.frames > 1);

    let mut restored = Vec::new();
    let mut stream_reader = std::io::Cursor::new(compressed_stream);
    let decompress_stats = cozip
        .decompress_stream(&mut stream_reader, &mut restored)
        .expect("stream decompression should succeed");

    assert_eq!(restored, input);
    assert_eq!(decompress_stats.frames, compress_stats.frames);
    assert_eq!(decompress_stats.output_bytes, input.len());
}

#[test]
fn stream_frame_input_size_must_be_positive() {
    let options = HybridOptions {
        prefer_gpu: false,
        gpu_fraction: 0.0,
        ..HybridOptions::default()
    };
    let cozip = CoZipDeflate::init(options).expect("init should succeed");
    let mut input = std::io::Cursor::new(vec![1_u8, 2, 3, 4]);
    let mut output = Vec::new();

    let err = cozip
        .compress_stream(
            &mut input,
            &mut output,
            StreamOptions {
                frame_input_size: 0,
            },
        )
        .expect_err("frame_input_size=0 should fail");

    assert!(matches!(err, CozipDeflateError::InvalidOptions(_)));
}

#[test]
fn raw_deflate_hybrid_stream_is_zip_compatible() {
    let input = patterned_data((1024 * 1024) + 321);
    let mut reader = std::io::Cursor::new(input.clone());
    let mut compressed = Vec::new();

    let stats = deflate_compress_stream_hybrid_zip_compatible(&mut reader, &mut compressed, 6)
        .expect("hybrid stream compress should succeed");
    let restored =
        deflate_decompress_on_cpu(&compressed).expect("raw deflate stream should be decodable");

    assert_eq!(restored, input);
    assert_eq!(usize::try_from(stats.input_bytes).ok(), Some(input.len()));
}

#[test]
fn raw_deflate_hybrid_stream_handles_empty_input() {
    let input = Vec::<u8>::new();
    let mut reader = std::io::Cursor::new(input.clone());
    let mut compressed = Vec::new();

    let stats = deflate_compress_stream_hybrid_zip_compatible(&mut reader, &mut compressed, 6)
        .expect("empty hybrid stream compress should succeed");
    let restored =
        deflate_decompress_on_cpu(&compressed).expect("raw deflate stream should decode");

    assert!(restored.is_empty());
    assert_eq!(stats.input_bytes, 0);
}

#[test]
fn raw_deflate_stream_roundtrip_chunk_5mib_cpu_only() {
    let input = bench_mixed_data(16 * 1024 * 1024);
    let options = HybridOptions {
        chunk_size: 5 * 1024 * 1024,
        compression_mode: CompressionMode::Ratio,
        scheduler_policy: HybridSchedulerPolicy::GlobalQueueLocalBuffers,
        stream_batch_chunks: 0,
        stream_max_inflight_chunks: 0,
        stream_max_inflight_bytes: 0,
        prefer_gpu: false,
        gpu_fraction: 0.0,
        ..HybridOptions::default()
    };
    let cozip = CoZipDeflate::init(options).expect("init should succeed");
    let mut src = std::io::Cursor::new(input.clone());
    let mut compressed = Vec::new();
    cozip
        .deflate_compress_stream_zip_compatible(&mut src, &mut compressed)
        .expect("stream deflate should succeed");
    let restored = deflate_decompress_on_cpu(&compressed).expect("stream must be decodable");
    assert_eq!(restored, input);
}
