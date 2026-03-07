@group(0) @binding(0) var<storage, read> emit_word_words: array<u32>;
@group(0) @binding(1) var<storage, read> table_meta: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_data: array<u32>;

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let max_entries = params[4];
    let total_bytes = table_meta[1u + max_entries * 2u];
    let total_words = (total_bytes + 3u) >> 2u;
    let word_idx = gid3.x;
    if (word_idx >= total_words) {
        return;
    }
    out_data[word_idx] = emit_word_words[word_idx];
}
