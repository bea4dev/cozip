@group(0) @binding(0) var<storage, read> table_meta: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> table_index_words: array<u32>;

fn store_table_index_byte(idx: u32, value: u32) {
    let wi = idx >> 2u;
    let shift = (idx & 3u) * 8u;
    let mask = 0xffu << shift;
    let prev = table_index_words[wi];
    table_index_words[wi] = (prev & (~mask)) | ((value & 0xffu) << shift);
}

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u || gid3.y != 0u || gid3.z != 0u) {
        return;
    }
    let max_entries = params[4];
    let out_count = min(table_meta[0], max_entries);
    var i = 0u;
    loop {
        if (i >= out_count) {
            break;
        }
        let len = table_meta[2u + i * 2u];
        store_table_index_byte(i, len);
        i = i + 1u;
    }
}
