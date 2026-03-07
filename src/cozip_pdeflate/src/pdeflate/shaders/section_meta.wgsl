@group(0) @binding(0) var<storage, read> out_lens: array<u32>;
@group(0) @binding(1) var<storage, read_write> section_prefix: array<u32>;
@group(0) @binding(2) var<storage, read_write> section_index_words: array<u32>;
@group(0) @binding(3) var<storage, read_write> section_meta_words: array<u32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;

fn write_section_index_byte(idx: u32, value: u32) {
    let base = params[8];
    let abs = base + idx;
    let wi = abs >> 2u;
    let shift = (abs & 3u) * 8u;
    let mask = 0xffu << shift;
    let prev = section_index_words[wi];
    section_index_words[wi] = (prev & (~mask)) | ((value & 0xffu) << shift);
}

fn append_varint(cursor_in: u32, value: u32) -> u32 {
    var cursor = cursor_in;
    var v = value;
    loop {
        var b = v & 0x7fu;
        v = v >> 7u;
        if (v != 0u) {
            b = b | 0x80u;
        }
        write_section_index_byte(cursor, b);
        cursor = cursor + 1u;
        if (v == 0u) {
            break;
        }
    }
    return cursor;
}

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u) {
        return;
    }
    let section_count = params[1];
    let out_lens_base = params[6];
    let section_prefix_base = params[7];
    let section_meta_base = params[9];
    var total_cmd_len = 0u;
    var section_index_len = 0u;
    var overflow = 0u;
    var sec = 0u;
    loop {
        if (sec >= section_count) {
            break;
        }
        let len = out_lens[out_lens_base + sec];
        if (len == 0xffffffffu) {
            overflow = 1u;
            break;
        }
        section_prefix[section_prefix_base + sec] = total_cmd_len;
        total_cmd_len = total_cmd_len + len;
        // section index stores bit_len (not byte_len) for each section.
        if (len > 0x1fffffffu) {
            overflow = 1u;
            break;
        }
        let bit_len = len << 3u;
        section_index_len = append_varint(section_index_len, bit_len);
        sec = sec + 1u;
    }
    if (overflow != 0u) {
        section_meta_words[section_meta_base + 0u] = 0xffffffffu;
        section_meta_words[section_meta_base + 1u] = 0xffffffffu;
        section_meta_words[section_meta_base + 2u] = 1u;
        section_meta_words[section_meta_base + 3u] = 0u;
    } else {
        section_meta_words[section_meta_base + 0u] = total_cmd_len;
        section_meta_words[section_meta_base + 1u] = section_index_len;
        section_meta_words[section_meta_base + 2u] = 0u;
        section_meta_words[section_meta_base + 3u] = 0u;
    }
}
