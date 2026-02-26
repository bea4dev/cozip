struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read_write> prefix_data: array<u32>;

@group(0) @binding(1)
var<storage, read> block_offsets: array<u32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 16776960u);
    if (idx >= params.len) {
        return;
    }

    let block_idx = idx / 256u;
    prefix_data[idx] = prefix_data[idx] + block_offsets[block_idx];
}
