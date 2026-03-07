@group(0) @binding(0) var<storage, read> out_cmd_bytes: array<u32>;
@group(0) @binding(1) var<storage, read> section_offsets_local: array<u32>;
@group(0) @binding(2) var<storage, read> out_lens: array<u32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_cmd_words: array<u32>;
@group(0) @binding(5) var<storage, read> section_offsets_global: array<u32>;

fn read_cmd_byte(idx: u32) -> u32 {
    return out_cmd_bytes[idx] & 0xffu;
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let word = gid3.x;
    let sec = gid3.y;
    let section_count = params[1];
    if (sec >= section_count) {
        return;
    }
    let out_lens_base = params[6];
    let len = out_lens[out_lens_base + sec];
    if (len == 0xffffffffu) {
        return;
    }
    let sec_words = (len + 3u) >> 2u;
    if (word >= sec_words) {
        return;
    }

    let sec_off_local = section_offsets_local[sec];
    let sec_off_global = section_offsets_global[sec];
    let local_byte = word << 2u;
    let abs_byte = sec_off_local + local_byte;

    let b0 = select(0u, read_cmd_byte(abs_byte), local_byte < len);
    let b1 = select(0u, read_cmd_byte(abs_byte + 1u), (local_byte + 1u) < len);
    let b2 = select(0u, read_cmd_byte(abs_byte + 2u), (local_byte + 2u) < len);
    let b3 = select(0u, read_cmd_byte(abs_byte + 3u), (local_byte + 3u) < len);
    let out_word = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);

    let dst_word = (sec_off_global >> 2u) + word;
    out_cmd_words[dst_word] = out_word;
}
