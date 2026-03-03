struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> input_data: array<u32>;

@group(0) @binding(1)
var<storage, read_write> prefix_data: array<u32>;

@group(0) @binding(2)
var<storage, read_write> block_sums: array<u32>;

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> scratch: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let lid = local_id.x;
    let gid = wg_id.x + (wg_id.y * 65535u);
    let idx = gid * 256u + lid;
    let value = select(0u, input_data[idx], idx < params.len);

    scratch[lid] = value;
    workgroupBarrier();

    var offset: u32 = 1u;
    loop {
        if (offset >= 256u) {
            break;
        }
        var addend: u32 = 0u;
        if (lid >= offset) {
            addend = scratch[lid - offset];
        }
        workgroupBarrier();
        scratch[lid] = scratch[lid] + addend;
        workgroupBarrier();
        offset = offset << 1u;
    }

    if (idx < params.len) {
        prefix_data[idx] = scratch[lid] - value;
    }

    if (lid == 255u && (gid * 256u) < params.len) {
        block_sums[gid] = scratch[255u];
    }
}
