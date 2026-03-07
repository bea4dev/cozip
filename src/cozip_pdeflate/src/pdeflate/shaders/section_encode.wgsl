@group(0) @binding(0) var<storage, read> src_words: array<u32>;
@group(0) @binding(1) var<storage, read> match_words: array<u32>;
@group(0) @binding(2) var<storage, read> section_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> section_caps: array<u32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_lens: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_cmd: array<u32>;

fn load_src(idx: u32) -> u32 {
    let w = src_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn section_start(sec: u32, section_count: u32, total_len: u32) -> u32 {
    if (section_count == 0u || total_len == 0u) {
        return 0u;
    }
    if (sec == 0u) {
        return 0u;
    }
    if (sec >= section_count) {
        return total_len;
    }
    let raw = (sec * total_len) / section_count;
    return raw & 0xfffffffcu;
}

fn match_len_or_zero(word: u32, remaining: u32, min_ref_len: u32, max_ref_len: u32) -> u32 {
    if (word == 0u) {
        return 0u;
    }
    let mlen = word & 0xffffu;
    if mlen < min_ref_len || mlen > max_ref_len {
        return 0u;
    }
    if (mlen > remaining) {
        return 0u;
    }
    return mlen;
}

fn is_valid_match(word: u32, remaining: u32, min_ref_len: u32, max_ref_len: u32) -> bool {
    return match_len_or_zero(word, remaining, min_ref_len, max_ref_len) != 0u;
}

fn emit_byte(base: u32, cursor: u32, cap: u32, value: u32) -> u32 {
    let byte_idx = base + cursor;
    let wi = byte_idx >> 2u;
    let shift = (byte_idx & 3u) * 8u;
    let mask = 0xffu << shift;
    let prev = out_cmd[wi];
    out_cmd[wi] = (prev & (~mask)) | ((value & 0xffu) << shift);
    return cursor + 1u;
}

fn emit_cmd_byte(base: u32, cursor: u32, cap: u32, value: u32) -> u32 {
    if (cursor + 1u > cap) {
        return 0xffffffffu;
    }
    return emit_byte(base, cursor, cap, value);
}

fn emit_header(base: u32, cursor: u32, cap: u32, tag: u32, len: u32) -> u32 {
    var len4 = len;
    if (len4 > 14u) {
        len4 = 15u;
    }
    let header = (len4 << 12u) | (tag & 0x0fffu);
    var next = emit_cmd_byte(base, cursor, cap, header & 0xffu);
    if (next == 0xffffffffu) {
        return next;
    }
    next = emit_cmd_byte(base, next, cap, (header >> 8u) & 0xffu);
    if (next == 0xffffffffu) {
        return next;
    }
    if (len4 == 15u) {
        next = emit_cmd_byte(base, next, cap, len - 15u);
        if (next == 0xffffffffu) {
            return next;
        }
    }
    return next;
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let sec = gid3.x;
    let total_len = params[0];
    let section_count = params[1];
    let min_ref_len = params[2];
    let max_ref_len = params[3];
    let max_cmd_len = params[4];
    let src_base = params[5];
    let out_lens_base = params[6];
    if (sec >= section_count) {
        return;
    }
    let base = section_offsets[sec];
    let cap = section_caps[sec];
    let s0 = section_start(sec, section_count, total_len);
    let s1 = section_start(sec + 1u, section_count, total_len);
    var cursor: u32 = 0u;
    var pos: u32 = s0;

    loop {
        if (pos >= s1) {
            break;
        }
        let word = match_words[src_base + pos];
        let remaining = s1 - pos;
        if (is_valid_match(word, remaining, min_ref_len, max_ref_len)) {
            let tag = word >> 16u;
            let mlen = word & 0xffffu;
            cursor = emit_header(base, cursor, cap, tag, mlen);
            if (cursor == 0xffffffffu) {
                out_lens[out_lens_base + sec] = 0xffffffffu;
                return;
            }
            pos = pos + mlen;
            continue;
        }

        let lit_start = pos;
        pos = pos + 1u;
        loop {
            if (pos >= s1 || (pos - lit_start) >= max_cmd_len) {
                break;
            }
            let p2 = match_words[src_base + pos];
            let rem2 = s1 - pos;
            if (is_valid_match(p2, rem2, min_ref_len, max_ref_len)) {
                break;
            }
            pos = pos + 1u;
        }
        let lit_len = pos - lit_start;
        cursor = emit_header(base, cursor, cap, 0x0fffu, lit_len);
        if (cursor == 0xffffffffu) {
            out_lens[out_lens_base + sec] = 0xffffffffu;
            return;
        }
        var i: u32 = 0u;
        loop {
            if (i >= lit_len) {
                break;
            }
            cursor = emit_cmd_byte(base, cursor, cap, load_src(src_base + lit_start + i));
            if (cursor == 0xffffffffu) {
                out_lens[out_lens_base + sec] = 0xffffffffu;
                return;
            }
            i = i + 1u;
        }
    }

    out_lens[out_lens_base + sec] = cursor;
}
