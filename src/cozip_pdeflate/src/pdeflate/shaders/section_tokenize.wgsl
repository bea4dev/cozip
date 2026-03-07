@group(0) @binding(0) var<storage, read> src_words: array<u32>;
@group(0) @binding(1) var<storage, read> match_words: array<u32>;
@group(0) @binding(2) var<storage, read> token_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> token_caps: array<u32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@group(0) @binding(5) var<storage, read_write> token_counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> token_meta: array<u32>;
@group(0) @binding(7) var<storage, read_write> token_pos: array<u32>;

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

fn probe_stride(lit_len: u32) -> u32 {
    if (lit_len < 8u) {
        return 1u;
    }
    if (lit_len < 32u) {
        return 8u;
    }
    if (lit_len < 128u) {
        return 32u;
    }
    if (lit_len < 512u) {
        return 64u;
    }
    // Similar to zlib's "reduce search effort once we have a good lead":
    // lower probe density for long literal runs.
    return 128u;
}

fn next_probe_step(lit_len: u32) -> u32 {
    let stride = probe_stride(lit_len);
    if (stride == 1u) {
        return 1u;
    }
    let rem = lit_len & (stride - 1u);
    if (rem == 0u) {
        return stride;
    }
    return stride - rem;
}

fn push_token(sec: u32, idx: u32, token_word: u32, pos: u32) {
    let base = token_offsets[sec];
    let out = base + idx;
    token_meta[out] = token_word;
    token_pos[out] = pos;
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
    if (sec >= section_count) {
        return;
    }
    let s0 = section_start(sec, section_count, total_len);
    let s1 = section_start(sec + 1u, section_count, total_len);
    let cap = token_caps[sec];
    var tok: u32 = 0u;
    var pos: u32 = s0;

    loop {
        if (pos >= s1) {
            break;
        }
        if (tok >= cap) {
            token_counts[sec] = 0xffffffffu;
            return;
        }
        let word = match_words[src_base + pos];
        let remaining = s1 - pos;
        let mlen = match_len_or_zero(word, remaining, min_ref_len, max_ref_len);
        if (mlen != 0u) {
            let tag = (word >> 16u) & 0x0fffu;
            push_token(sec, tok, (tag << 16u) | mlen, pos);
            tok = tok + 1u;
            pos = pos + mlen;
            continue;
        }

        let lit_start = pos;
        pos = pos + 1u;
        var lit_len = 1u;
        loop {
            if (pos >= s1 || lit_len >= max_cmd_len) {
                break;
            }
            // If remaining bytes are below min_ref_len, no further matches can
            // be emitted. Consume the tail as a single literal span.
            let sec_rem = s1 - pos;
            if (sec_rem < min_ref_len) {
                let cmd_rem = max_cmd_len - lit_len;
                var tail = sec_rem;
                if (tail > cmd_rem) {
                    tail = cmd_rem;
                }
                pos = pos + tail;
                lit_len = lit_len + tail;
                break;
            }

            // Like CPU lazy lookahead: reduce probe frequency as literal span
            // grows to cap tokenize-stage work.
            let stride = probe_stride(lit_len);
            let should_probe = stride == 1u || ((lit_len & (stride - 1u)) == 0u);
            if (!should_probe) {
                // Skip directly to the next probe boundary instead of
                // advancing byte-by-byte on non-probe regions.
                var step = next_probe_step(lit_len);
                let cmd_rem = max_cmd_len - lit_len;
                if (step > cmd_rem) {
                    step = cmd_rem;
                }
                if (step > sec_rem) {
                    step = sec_rem;
                }
                pos = pos + step;
                lit_len = lit_len + step;
                continue;
            }

            let p2 = match_words[src_base + pos];
            let rem2 = sec_rem;
            if (match_len_or_zero(p2, rem2, min_ref_len, max_ref_len) != 0u) {
                break;
            }
            // No hit at this probe point: jump straight to next probe point.
            var step = stride;
            let cmd_rem = max_cmd_len - lit_len;
            if (step > cmd_rem) {
                step = cmd_rem;
            }
            if (step > sec_rem) {
                step = sec_rem;
            }
            if (step == 0u) {
                break;
            }
            pos = pos + step;
            lit_len = lit_len + step;
        }
        push_token(sec, tok, (0x0fffu << 16u) | lit_len, lit_start);
        tok = tok + 1u;
    }

    token_counts[sec] = tok;
}
