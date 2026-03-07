@group(0) @binding(0) var<storage, read> src_words: array<u32>;
@group(0) @binding(1) var<storage, read> token_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> token_counts: array<u32>;
@group(0) @binding(3) var<storage, read> token_meta: array<u32>;
@group(0) @binding(4) var<storage, read> token_pos: array<u32>;
@group(0) @binding(5) var<storage, read> token_cmd_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> section_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> section_caps: array<u32>;
@group(0) @binding(8) var<storage, read> params: array<u32>;
@group(0) @binding(9) var<storage, read_write> out_cmd_bytes: array<u32>;

fn load_src(idx: u32) -> u32 {
    let w = src_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn write_cmd_byte(idx: u32, value: u32) {
    out_cmd_bytes[idx] = value & 0xffu;
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let tok = gid3.x;
    let sec = gid3.y;
    let section_count = params[1];
    let src_base = params[5];
    if (sec >= section_count) {
        return;
    }
    let count = token_counts[sec];
    if (count == 0xffffffffu || tok >= count) {
        return;
    }

    let base = token_offsets[sec];
    let tidx = base + tok;
    let token_word = token_meta[tidx];
    let tag = (token_word >> 16u) & 0x0fffu;
    let len = token_word & 0xffffu;
    var header_len: u32 = 2u;
    if (len > 14u) {
        header_len = 3u;
    }
    let payload_len = select(0u, len, tag == 0x0fffu);
    let cmd_len = header_len + payload_len;

    let sec_cap = section_caps[sec];
    let local_off = token_cmd_offsets[tidx];
    if (local_off + cmd_len > sec_cap) {
        return;
    }

    let dst = section_offsets[sec] + local_off;
    var len4 = len;
    if (len4 > 14u) {
        len4 = 15u;
    }
    let header = (len4 << 12u) | tag;
    write_cmd_byte(dst, header & 0xffu);
    write_cmd_byte(dst + 1u, (header >> 8u) & 0xffu);
    if (header_len == 3u) {
        write_cmd_byte(dst + 2u, len - 15u);
    }
    if (tag == 0x0fffu) {
        let lit_base = token_pos[tidx];
        var i: u32 = 0u;
        loop {
            if (i >= len) {
                break;
            }
            write_cmd_byte(dst + header_len + i, load_src(src_base + lit_base + i));
            i = i + 1u;
        }
    }
}
