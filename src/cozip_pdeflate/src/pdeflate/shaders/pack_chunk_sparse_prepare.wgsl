const SPARSE_BATCH_DESC_WORDS: u32 = 23u;
const SPARSE_BATCH_RESULT_WORDS: u32 = 4u;

@group(0) @binding(0) var<storage, read> desc_words: array<u32>;
@group(0) @binding(1) var<storage, read> section_meta_words: array<u32>;
@group(0) @binding(2) var<storage, read_write> result_words: array<u32>;

fn align_up4(v: u32) -> u32 {
    return (v + 3u) & 0xfffffffcu;
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let job = gid3.x;
    let job_count = desc_words[0];
    if (gid3.y != 0u || gid3.z != 0u || job >= job_count) {
        return;
    }
    let desc_base = 1u + job * SPARSE_BATCH_DESC_WORDS;
    let result_base = job * SPARSE_BATCH_RESULT_WORDS;
    let section_meta_base = desc_words[desc_base + 7u];
    let section_index_off = desc_words[desc_base + 19u];
    let section_index_cap_len = desc_words[desc_base + 9u];
    let section_cmd_cap_len = desc_words[desc_base + 14u];
    let slot_cap_words = desc_words[desc_base + 22u];
    let section_cmd_len = section_meta_words[section_meta_base];
    let section_index_len = section_meta_words[section_meta_base + 1u];
    let overflow = section_meta_words[section_meta_base + 2u];
    let section_cmd_off_cap = section_index_off + section_index_cap_len;
    let total_len_cap = section_cmd_off_cap + section_cmd_cap_len;

    var total_len = 0xffffffffu;
    if (overflow == 0u && section_cmd_len != 0xffffffffu && section_index_len != 0xffffffffu) {
        let section_cmd_off = section_index_off + section_index_len;
        total_len = section_cmd_off + section_cmd_len;
        if (total_len > total_len_cap) {
            total_len = 0xffffffffu;
        }
    }
    let total_words = select(max(1u, align_up4(total_len) >> 2u), 1u, total_len == 0xffffffffu);
    if (total_len != 0xffffffffu && total_words > slot_cap_words) {
        result_words[result_base] = section_cmd_len;
        result_words[result_base + 1u] = section_index_len;
        result_words[result_base + 2u] = 0xffffffffu;
        result_words[result_base + 3u] = slot_cap_words;
        return;
    }
    result_words[result_base] = section_cmd_len;
    result_words[result_base + 1u] = section_index_len;
    result_words[result_base + 2u] = total_len;
    result_words[result_base + 3u] = total_words;
}
