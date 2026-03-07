@group(0) @binding(0) var<storage, read> bucket_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> bucket_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> params: array<u32>;

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u || gid3.y != 0u || gid3.z != 0u) {
        return;
    }
    var sorted_count = 0u;
    var key: i32 = 65535;
    loop {
        if (key < 0) {
            break;
        }
        let k = u32(key);
        let c = bucket_counts[k];
        bucket_offsets[k] = sorted_count;
        sorted_count = sorted_count + c;
        key = key - 1;
    }
    // params[2] is sorted_count for BUILD_TABLE_FINALIZE_SHADER.
    params[2] = sorted_count;
}
