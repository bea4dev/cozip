@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> header_words: array<u32>;
@group(0) @binding(2) var<storage, read> table_index_words: array<u32>;
@group(0) @binding(3) var<storage, read> table_data_words: array<u32>;
@group(0) @binding(4) var<storage, read> section_index_words: array<u32>;
@group(0) @binding(5) var<storage, read> section_cmd_words: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_words: array<u32>;

fn load_header_byte(idx: u32) -> u32 {
    let w = header_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_table_index_byte(idx: u32) -> u32 {
    let w = table_index_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_table_data_byte(idx: u32) -> u32 {
    let w = table_data_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_section_index_byte(idx: u32) -> u32 {
    let w = section_index_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_section_cmd_byte(idx: u32) -> u32 {
    let w = section_cmd_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn read_output_byte(pos: u32) -> u32 {
    let header_len = params[1];
    let table_index_off = params[2];
    let table_index_len = params[3];
    let table_data_off = params[4];
    let table_data_len = params[5];
    let section_index_off = params[6];
    let section_index_len = params[7];
    let section_cmd_off = params[8];
    let section_cmd_len = params[9];

    if (pos < header_len) {
        return load_header_byte(pos);
    }
    if (pos >= table_index_off && pos < table_index_off + table_index_len) {
        return load_table_index_byte(pos - table_index_off);
    }
    if (pos >= table_data_off && pos < table_data_off + table_data_len) {
        return load_table_data_byte(pos - table_data_off);
    }
    if (pos >= section_index_off && pos < section_index_off + section_index_len) {
        return load_section_index_byte(pos - section_index_off);
    }
    if (pos >= section_cmd_off && pos < section_cmd_off + section_cmd_len) {
        return load_section_cmd_byte(pos - section_cmd_off);
    }
    return 0u;
}

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let word_idx = gid3.x;
    let total_len = params[0];
    let out_words_len = params[10];
    if (word_idx >= out_words_len) {
        return;
    }
    let base = word_idx * 4u;
    var b0 = 0u;
    var b1 = 0u;
    var b2 = 0u;
    var b3 = 0u;
    if (base < total_len) {
        b0 = read_output_byte(base);
    }
    if (base + 1u < total_len) {
        b1 = read_output_byte(base + 1u);
    }
    if (base + 2u < total_len) {
        b2 = read_output_byte(base + 2u);
    }
    if (base + 3u < total_len) {
        b3 = read_output_byte(base + 3u);
    }
    out_words[word_idx] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}
