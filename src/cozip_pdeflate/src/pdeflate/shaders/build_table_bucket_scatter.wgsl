@group(0) @binding(0) var<storage, read> cand_words: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read> bucket_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> bucket_cursors: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> out_sorted_indices: array<u32>;

fn key_for(score: u32, len: u32) -> u32 {
    let s = min(score, 255u);
    let l = min(len, 255u);
    return (s << 8u) | l;
}

@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let sid = gid3.x;
    let sample_count = params[0];
    if (sid >= sample_count) {
        return;
    }
    let base = sid * 4u;
    let score = cand_words[base + 0u];
    if (score == 0u) {
        return;
    }
    let len = cand_words[base + 2u];
    if (len == 0u) {
        return;
    }
    let key = key_for(score, len);
    let local = atomicAdd(&bucket_cursors[key], 1u);
    let out_idx = bucket_offsets[key] + local;
    out_sorted_indices[out_idx] = sid;
}
