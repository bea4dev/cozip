@group(0) @binding(0) var<storage, read> token_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> token_caps: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@group(0) @binding(3) var<storage, read> token_counts: array<u32>;
@group(0) @binding(4) var<storage, read> token_meta: array<u32>;
@group(0) @binding(5) var<storage, read_write> token_cmd_offsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_lens: array<u32>;
@group(0) @binding(7) var<storage, read> section_caps: array<u32>;

fn cmd_size_bytes(token_word: u32) -> u32 {
    let tag = (token_word >> 16u) & 0x0fffu;
    let len = token_word & 0xffffu;
    var size = 2u;
    if (len > 14u) {
        size = 3u;
    }
    if (tag == 0x0fffu) {
        size = size + len;
    }
    return size;
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let sec = gid3.x;
    let section_count = params[1];
    if (sec >= section_count) {
        return;
    }
    let out_lens_base = params[6];
    let count = token_counts[sec];
    if (count == 0xffffffffu) {
        out_lens[out_lens_base + sec] = 0xffffffffu;
        return;
    }
    if (count > token_caps[sec]) {
        out_lens[out_lens_base + sec] = 0xffffffffu;
        return;
    }

    let base = token_offsets[sec];
    let cap = section_caps[sec];
    var cursor: u32 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= count) {
            break;
        }
        let idx = base + i;
        token_cmd_offsets[idx] = cursor;
        cursor = cursor + cmd_size_bytes(token_meta[idx]);
        if (cursor > cap) {
            out_lens[out_lens_base + sec] = 0xffffffffu;
            return;
        }
        i = i + 1u;
    }
    out_lens[out_lens_base + sec] = cursor;
}
