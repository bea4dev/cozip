@group(0) @binding(0) var<storage, read> table_chunk_bases: array<u32>;
@group(0) @binding(1) var<storage, read> table_chunk_counts: array<u32>;
@group(0) @binding(2) var<storage, read> table_chunk_index_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> table_chunk_data_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> table_chunk_meta_words: array<u32>;
@group(0) @binding(5) var<storage, read> table_index_words_in: array<u32>;
@group(0) @binding(6) var<storage, read> table_data_words_in: array<u32>;
@group(0) @binding(7) var<storage, read> params: array<u32>;
@group(0) @binding(8) var<storage, read_write> prefix2_first_ids: array<u32>;
@group(0) @binding(9) var<storage, read_write> table_entry_lens: array<u32>;
@group(0) @binding(10) var<storage, read_write> table_entry_offsets: array<u32>;

fn load_table_index_byte(idx: u32) -> u32 {
    let w = table_index_words_in[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_table_data_byte(idx: u32) -> u32 {
    let w = table_data_words_in[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u || gid3.y != 0u || gid3.z != 0u) {
        return;
    }
    let chunk_count = params[0];
    var chunk = 0u;
    loop {
        if (chunk >= chunk_count) {
            break;
        }
        let chunk_base = table_chunk_bases[chunk];
        let chunk_entries_hint = table_chunk_counts[chunk];
        let meta_base = chunk * 2u;
        let meta_count = table_chunk_meta_words[meta_base];
        var chunk_entries = chunk_entries_hint;
        if (meta_count != 0u) {
            chunk_entries = min(chunk_entries_hint, meta_count);
        }
        let table_index_base = table_chunk_index_offsets[chunk];
        let mut_data_base = table_chunk_data_offsets[chunk];
        let prefix_base = chunk << 16u;

        var local_id = 0u;
        var data_cursor = mut_data_base;
        loop {
            if (local_id >= chunk_entries) {
                break;
            }
            let len = load_table_index_byte(table_index_base + local_id);
            if (len == 0u) {
                break;
            }
            let global_id = chunk_base + local_id;
            table_entry_lens[global_id] = len;
            table_entry_offsets[global_id] = data_cursor;
            if (len >= 2u) {
                let key = (load_table_data_byte(data_cursor) << 8u) | load_table_data_byte(data_cursor + 1u);
                let slot = prefix_base + key;
                if (prefix2_first_ids[slot] == 0u) {
                    prefix2_first_ids[slot] = local_id + 1u;
                }
            }
            data_cursor = data_cursor + len;
            local_id = local_id + 1u;
        }
        chunk = chunk + 1u;
    }
}
