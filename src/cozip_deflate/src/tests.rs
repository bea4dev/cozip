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

#[test]
fn prepared_non_final_link_roundtrip_at_chunk_boundary() {
    fn find_unaligned_input() -> Vec<u8> {
        for len in 257..4096 {
            let candidate = patterned_data(len);
            let compressed = deflate_compress_cpu(&candidate, 6).expect("compress should succeed");
            let layout =
                parse_deflate_stream_layout(&compressed).expect("layout parse should work");
            if layout.end_bit % 8 != 0 {
                return candidate;
            }
        }
        panic!("failed to find an input that produces non-byte-aligned deflate end bit");
    }

    let input1 = find_unaligned_input();
    let input2 = patterned_data(3333);

    let mut chunk1 = deflate_compress_cpu(&input1, 6).expect("compress 1 should succeed");
    let layout1 = parse_deflate_stream_layout(&chunk1).expect("layout parse 1 should succeed");
    assert_eq!(bit_at(&chunk1, layout1.final_header_bit), 1);
    let non_final_end1 =
        prepare_chunk_bits_for_non_final_stream(&mut chunk1, layout1).expect("prepare should work");
    assert_eq!(bit_at(&chunk1, layout1.final_header_bit), 0);
    assert_eq!(non_final_end1 % 8, 0);
    assert!(non_final_end1 >= layout1.end_bit);

    let mut chunk2 = deflate_compress_cpu(&input2, 6).expect("compress 2 should succeed");
    let layout2 = parse_deflate_stream_layout(&chunk2).expect("layout parse 2 should succeed");
    let _ = prepare_chunk_bits_for_non_final_stream(&mut chunk2, layout2)
        .expect("prepare 2 should work");

    let mut out = Vec::new();
    {
        let mut writer = DeflateBitWriter::new(&mut out);
        writer
            .write_bits_from_slice(&chunk1, 0, non_final_end1)
            .expect("write chunk1");
        write_chunk_bits_with_final_override(&mut writer, &chunk2, layout2, 1)
            .expect("write final chunk2");
        writer.finish().expect("finish writer");
    }

    let restored = deflate_decompress_on_cpu(&out).expect("composed stream should decode");
    let mut expected = input1;
    expected.extend_from_slice(&input2);
    assert_eq!(restored, expected);
}

#[test]
fn write_chunk_bits_with_final_override_handles_nonzero_final_bit_position() {
    let chunk = vec![0b1010_1100_u8, 0b0110_1010_u8, 0b0000_1111_u8];
    let layout = DeflateStreamLayout {
        final_header_bit: 9,
        end_bit: 21,
    };

    let mut expected = chunk.clone();
    let final_byte = layout.final_header_bit / 8;
    let final_mask = 1_u8 << (layout.final_header_bit % 8);
    expected[final_byte] |= final_mask;

    let mut out = Vec::new();
    {
        let mut writer = DeflateBitWriter::new(&mut out);
        write_chunk_bits_with_final_override(&mut writer, &chunk, layout, 1)
            .expect("override write should succeed");
        writer.finish().expect("finish should succeed");
    }

    for bit in 0..layout.end_bit {
        assert_eq!(
            bit_at(&out, bit),
            bit_at(&expected, bit),
            "bit mismatch at {bit}"
        );
    }
}
