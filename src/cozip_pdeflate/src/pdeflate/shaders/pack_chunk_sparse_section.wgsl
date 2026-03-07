const SPARSE_BATCH_DESC_WORDS: u32 = 23u;
const SPARSE_BATCH_RESULT_WORDS: u32 = 4u;

@group(0) @binding(0) var<storage, read> desc_words: array<u32>;
@group(0) @binding(1) var<storage, read> result_words: array<u32>;
@group(0) @binding(2) var<storage, read> table_index_words: array<u32>;
@group(0) @binding(3) var<storage, read> table_data_words: array<u32>;
@group(0) @binding(4) var<storage, read> section_index_words: array<u32>;
@group(0) @binding(5) var<storage, read> section_lens_words: array<u32>;
@group(0) @binding(6) var<storage, read> section_prefix_words: array<u32>;
@group(0) @binding(7) var<storage, read> section_offsets_words: array<u32>;
@group(0) @binding(8) var<storage, read> section_cmd_sparse_words: array<u32>;
@group(0) @binding(9) var<storage, read_write> out_words: array<u32>;
@group(0) @binding(10) var<storage, read_write> sparse_stats_words: array<atomic<u32>, 8>;

fn low8(v: u32, s: u32) -> u32 {
    return (v >> s) & 0xffu;
}

fn sparse_probe_enabled(desc_base: u32) -> bool {
    return desc_words[desc_base + 17u] != 0u;
}

fn sparse_stat_add(desc_base: u32, idx: u32, value: u32) {
    if (sparse_probe_enabled(desc_base)) {
        atomicAdd(&sparse_stats_words[idx], value);
    }
}

fn load_table_index_byte(desc_base: u32, idx: u32) -> u32 {
    let abs_idx = desc_words[desc_base + 3u] + idx;
    let w = table_index_words[abs_idx >> 2u];
    let shift = (abs_idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_table_data_byte(desc_base: u32, idx: u32) -> u32 {
    let abs_idx = desc_words[desc_base + 5u] + idx;
    let w = table_data_words[abs_idx >> 2u];
    let shift = (abs_idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_section_index_byte(desc_base: u32, idx: u32) -> u32 {
    let abs_idx = desc_words[desc_base + 8u] + idx;
    let w = section_index_words[abs_idx >> 2u];
    let shift = (abs_idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_section_cmd_sparse_byte(desc_base: u32, idx: u32) -> u32 {
    let abs_idx = desc_words[desc_base + 13u] + idx;
    let w = section_cmd_sparse_words[abs_idx >> 2u];
    let shift = (abs_idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_header_byte(desc_base: u32, section_cmd_off: u32, idx: u32) -> u32 {
    let chunk_len = desc_words[desc_base + 1u];
    let table_count = desc_words[desc_base + 2u];
    let section_count = desc_words[desc_base];
    let table_index_off = 36u;
    let table_data_off = desc_words[desc_base + 18u];
    let section_index_off = desc_words[desc_base + 19u];
    let huff_lut_off = section_index_off;

    if (idx == 0u) { return 0x50u; }
    if (idx == 1u) { return 0x44u; }
    if (idx == 2u) { return 0x46u; }
    if (idx == 3u) { return 0x30u; }
    if (idx == 4u || idx == 5u || idx == 6u || idx == 7u) { return 0u; }
    if (idx >= 8u && idx < 12u) { return low8(chunk_len, (idx - 8u) * 8u); }
    if (idx >= 12u && idx < 14u) { return low8(table_count, (idx - 12u) * 8u); }
    if (idx >= 14u && idx < 16u) { return low8(section_count, (idx - 14u) * 8u); }
    if (idx >= 16u && idx < 20u) { return low8(table_index_off, (idx - 16u) * 8u); }
    if (idx >= 20u && idx < 24u) { return low8(table_data_off, (idx - 20u) * 8u); }
    if (idx >= 24u && idx < 28u) { return low8(huff_lut_off, (idx - 24u) * 8u); }
    if (idx >= 28u && idx < 32u) { return low8(section_index_off, (idx - 28u) * 8u); }
    if (idx >= 32u && idx < 36u) { return low8(section_cmd_off, (idx - 32u) * 8u); }
    return 0u;
}

fn find_section_for_local(desc_base: u32, local: u32) -> u32 {
    let section_count = desc_words[desc_base];
    if (section_count == 0u) {
        sparse_stat_add(desc_base, 7u, 1u);
        return 0xffffffffu;
    }
    let prefix_base = desc_words[desc_base + 11u];
    let lens_base = desc_words[desc_base + 10u];
    var lo = 0u;
    var hi = section_count;
    loop {
        sparse_stat_add(desc_base, 5u, 1u);
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) >> 1u;
        let start = section_prefix_words[prefix_base + mid];
        let end = start + section_lens_words[lens_base + mid];
        if (local < start) {
            hi = mid;
            continue;
        }
        if (local >= end) {
            lo = mid + 1u;
            continue;
        }
        sparse_stat_add(desc_base, 6u, 1u);
        return mid;
    }
    sparse_stat_add(desc_base, 7u, 1u);
    return 0xffffffffu;
}

fn load_section_cmd_compact_word(desc_base: u32, local_base: u32, remaining: u32) -> u32 {
    let section_count = desc_words[desc_base];
    let prefix_base = desc_words[desc_base + 11u];
    let lens_base = desc_words[desc_base + 10u];
    let offsets_base = desc_words[desc_base + 12u];
    var sec_idx = find_section_for_local(desc_base, local_base);
    var sec_start = 0u;
    var sec_end = 0u;
    if (sec_idx != 0xffffffffu) {
        sec_start = section_prefix_words[prefix_base + sec_idx];
        sec_end = sec_start + section_lens_words[lens_base + sec_idx];
    }

    var out = 0u;
    var i = 0u;
    loop {
        if (i >= 4u || i >= remaining) {
            break;
        }
        let local = local_base + i;
        loop {
            if (sec_idx == 0xffffffffu || local < sec_end) {
                break;
            }
            if (sec_idx + 1u >= section_count) {
                sec_idx = 0xffffffffu;
                break;
            }
            sec_idx = sec_idx + 1u;
            sec_start = section_prefix_words[prefix_base + sec_idx];
            sec_end = sec_start + section_lens_words[lens_base + sec_idx];
        }
        if (sec_idx != 0xffffffffu && local >= sec_start && local < sec_end) {
            sparse_stat_add(desc_base, 4u, 1u);
            let src = section_offsets_words[offsets_base + sec_idx] + (local - sec_start);
            let byte = load_section_cmd_sparse_byte(desc_base, src);
            out = out | (byte << (i * 8u));
        } else {
            sparse_stat_add(desc_base, 7u, 1u);
        }
        i = i + 1u;
    }
    return out;
}

fn load_section_cmd_compact_byte(desc_base: u32, local: u32) -> u32 {
    return load_section_cmd_compact_word(desc_base, local, 1u) & 0xffu;
}

fn read_output_word(desc_base: u32, result_base: u32, byte_base: u32) -> u32 {
    let total_len = result_words[result_base + 2u];
    let section_index_len = result_words[result_base + 1u];
    let section_cmd_len = result_words[result_base];
    let table_index_off = 36u;
    let table_index_len = desc_words[desc_base + 4u];
    let table_data_off = desc_words[desc_base + 18u];
    let table_data_len = desc_words[desc_base + 6u];
    let section_index_off = desc_words[desc_base + 19u];
    let section_cmd_off = section_index_off + section_index_len;
    let section_cmd_end = section_cmd_off + section_cmd_len;

    if (total_len == 0xffffffffu || byte_base >= total_len) {
        return 0u;
    }
    if (byte_base >= section_cmd_off && byte_base < section_cmd_end && desc_words[desc_base] > 0u) {
        let remaining = section_cmd_end - byte_base;
        return load_section_cmd_compact_word(desc_base, byte_base - section_cmd_off, remaining);
    }

    var out = 0u;
    var i = 0u;
    loop {
        if (i >= 4u) {
            break;
        }
        let pos = byte_base + i;
        if (pos >= total_len) {
            break;
        }
        var byte = 0u;
        if (pos < 36u) {
            sparse_stat_add(desc_base, 0u, 1u);
            byte = load_header_byte(desc_base, section_cmd_off, pos);
        } else if (pos >= table_index_off && pos < table_index_off + table_index_len) {
            sparse_stat_add(desc_base, 1u, 1u);
            byte = load_table_index_byte(desc_base, pos - table_index_off);
        } else if (pos >= table_data_off && pos < table_data_off + table_data_len) {
            sparse_stat_add(desc_base, 2u, 1u);
            byte = load_table_data_byte(desc_base, pos - table_data_off);
        } else if (pos >= section_index_off && pos < section_index_off + section_index_len) {
            sparse_stat_add(desc_base, 3u, 1u);
            byte = load_section_index_byte(desc_base, pos - section_index_off);
        } else if (pos >= section_cmd_off && pos < section_cmd_end) {
            sparse_stat_add(desc_base, 4u, 1u);
            byte = load_section_cmd_compact_byte(desc_base, pos - section_cmd_off);
        }
        out = out | (byte << (i * 8u));
        i = i + 1u;
    }
    return out;
}

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let job_count = desc_words[0];
    let job = gid3.y;
    if (gid3.z != 0u || job >= job_count) {
        return;
    }
    let desc_base = 1u + job * SPARSE_BATCH_DESC_WORDS;
    let result_base = job * SPARSE_BATCH_RESULT_WORDS;
    let word_idx = gid3.x;
    let out_words_len = result_words[result_base + 3u];
    if (word_idx >= out_words_len) {
        return;
    }
    let out_base_word = desc_words[desc_base + 21u];
    let slot_words_len = desc_words[desc_base + 22u];
    if (word_idx >= slot_words_len) {
        return;
    }
    out_words[out_base_word + word_idx] = read_output_word(desc_base, result_base, word_idx * 4u);
}
