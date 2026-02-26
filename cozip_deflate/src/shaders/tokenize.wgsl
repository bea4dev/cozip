struct Params {
    len: u32,
    block_size: u32,
    mode: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;

@group(0) @binding(1)
var<storage, read_write> token_flags: array<u32>;

@group(0) @binding(2)
var<storage, read_write> token_kind: array<u32>;

@group(0) @binding(3)
var<storage, read_write> token_len: array<u32>;

@group(0) @binding(4)
var<storage, read_write> token_dist: array<u32>;

@group(0) @binding(5)
var<storage, read_write> token_lit: array<u32>;

@group(0) @binding(6)
var<uniform> params: Params;

fn mode_max_match_scan(mode: u32) -> u32 {
    switch (mode) {
        case 100u: { return 0u; } // profile: literal-only
        case 101u: { return 0u; } // profile: head-only speed
        case 102u: { return 0u; } // profile: head-only balanced
        case 103u: { return 0u; } // profile: head-only ratio
        case 1u: { return 128u; } // Balanced
        case 2u: { return 192u; } // Ratio
        default: { return 64u; }  // Speed
    }
}

fn mode_max_match_len(mode: u32) -> u32 {
    switch (mode) {
        case 100u: { return 3u; } // profile: literal-only
        case 101u: { return 3u; } // profile: head-only speed
        case 102u: { return 3u; } // profile: head-only balanced
        case 103u: { return 3u; } // profile: head-only ratio
        case 1u: { return 128u; } // Balanced
        case 2u: { return 258u; } // Ratio
        default: { return 64u; }  // Speed
    }
}

fn mode_dist_candidate_count(mode: u32) -> u32 {
    switch (mode) {
        case 100u: { return 0u; }  // profile: literal-only
        case 101u: { return 20u; } // profile: head-only speed
        case 102u: { return 28u; } // profile: head-only balanced
        case 103u: { return 32u; } // profile: head-only ratio
        case 1u: { return 28u; } // up to 8192
        case 2u: { return 32u; } // up to 32768
        default: { return 20u; } // up to 512
    }
}

fn byte_at(index: u32) -> u32 {
    let word = input_words[index >> 2u];
    let shift = (index & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn load_u32_unaligned(index: u32) -> u32 {
    let word_index = index >> 2u;
    let byte_offset = index & 3u;
    if (byte_offset == 0u) {
        return input_words[word_index];
    }
    let low = input_words[word_index] >> (byte_offset * 8u);
    let high_shift = (4u - byte_offset) * 8u;
    let high = input_words[word_index + 1u] << high_shift;
    return low | high;
}

fn dist_candidate(slot: u32) -> u32 {
    switch (slot) {
        case 0u: { return 1u; }
        case 1u: { return 2u; }
        case 2u: { return 3u; }
        case 3u: { return 4u; }
        case 4u: { return 5u; }
        case 5u: { return 6u; }
        case 6u: { return 8u; }
        case 7u: { return 10u; }
        case 8u: { return 12u; }
        case 9u: { return 16u; }
        case 10u: { return 24u; }
        case 11u: { return 32u; }
        case 12u: { return 48u; }
        case 13u: { return 64u; }
        case 14u: { return 96u; }
        case 15u: { return 128u; }
        case 16u: { return 192u; }
        case 17u: { return 256u; }
        case 18u: { return 384u; }
        case 19u: { return 512u; }
        case 20u: { return 768u; }
        case 21u: { return 1024u; }
        case 22u: { return 1536u; }
        case 23u: { return 2048u; }
        case 24u: { return 3072u; }
        case 25u: { return 4096u; }
        case 26u: { return 6144u; }
        case 27u: { return 8192u; }
        case 28u: { return 12288u; }
        case 29u: { return 16384u; }
        case 30u: { return 24576u; }
        default: { return 32768u; }
    }
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + (id.y * 8388480u);
    if (i >= params.len) {
        return;
    }

    let lit = byte_at(i);
    token_lit[i] = lit;
    token_flags[i] = 0u;
    token_kind[i] = 0u;

    var best_dist: u32 = 0u;
    var best_len: u32 = 0u;
    let match_scan_limit = mode_max_match_scan(params.mode);
    let match_len_limit = mode_max_match_len(params.mode);
    let dist_limit = mode_dist_candidate_count(params.mode);
    let b0 = lit;
    var b1: u32 = 0u;
    var b2: u32 = 0u;
    if (i + 2u < params.len) {
        b1 = byte_at(i + 1u);
        b2 = byte_at(i + 2u);
    }
    var c: u32 = 0u;
    loop {
        if (c >= dist_limit) {
            break;
        }
        let dist = dist_candidate(c);
        if (dist <= i && i + 2u < params.len) {
            if (b0 == byte_at(i - dist)
                && b1 == byte_at(i + 1u - dist)
                && b2 == byte_at(i + 2u - dist))
            {
                var mlen: u32 = 3u;
                var p = i + 3u;
                var scanned: u32 = 0u;

                loop {
                    if (p + 15u >= params.len || mlen + 16u > match_len_limit || scanned + 16u > match_scan_limit) {
                        break;
                    }

                    var mismatch = false;
                    let diff0 = load_u32_unaligned(p) ^ load_u32_unaligned(p - dist);
                    if (diff0 != 0u) {
                        let same = countTrailingZeros(diff0) >> 3u;
                        mlen = mlen + same;
                        p = p + same;
                        scanned = scanned + same;
                        mismatch = true;
                    }

                    if (!mismatch) {
                        let diff1 = load_u32_unaligned(p + 4u) ^ load_u32_unaligned(p + 4u - dist);
                        if (diff1 != 0u) {
                            let same = countTrailingZeros(diff1) >> 3u;
                            mlen = mlen + 4u + same;
                            p = p + 4u + same;
                            scanned = scanned + 4u + same;
                            mismatch = true;
                        }
                    }

                    if (!mismatch) {
                        let diff2 = load_u32_unaligned(p + 8u) ^ load_u32_unaligned(p + 8u - dist);
                        if (diff2 != 0u) {
                            let same = countTrailingZeros(diff2) >> 3u;
                            mlen = mlen + 8u + same;
                            p = p + 8u + same;
                            scanned = scanned + 8u + same;
                            mismatch = true;
                        }
                    }

                    if (!mismatch) {
                        let diff3 = load_u32_unaligned(p + 12u) ^ load_u32_unaligned(p + 12u - dist);
                        if (diff3 != 0u) {
                            let same = countTrailingZeros(diff3) >> 3u;
                            mlen = mlen + 12u + same;
                            p = p + 12u + same;
                            scanned = scanned + 12u + same;
                            mismatch = true;
                        }
                    }

                    if (mismatch) {
                        break;
                    }

                    mlen = mlen + 16u;
                    p = p + 16u;
                    scanned = scanned + 16u;
                }

                loop {
                    if (p + 3u >= params.len || mlen + 4u > match_len_limit || scanned + 4u > match_scan_limit) {
                        break;
                    }
                    let left4 = load_u32_unaligned(p);
                    let right4 = load_u32_unaligned(p - dist);
                    if (left4 == right4) {
                        mlen = mlen + 4u;
                        p = p + 4u;
                        scanned = scanned + 4u;
                    } else {
                        let diff = left4 ^ right4;
                        let same_bytes = countTrailingZeros(diff) >> 3u;
                        mlen = mlen + same_bytes;
                        p = p + same_bytes;
                        scanned = scanned + same_bytes;
                        break;
                    }
                }

                loop {
                    if (p >= params.len || mlen >= match_len_limit || scanned >= match_scan_limit) {
                        break;
                    }
                    if (byte_at(p) != byte_at(p - dist)) {
                        break;
                    }
                    mlen = mlen + 1u;
                    p = p + 1u;
                    scanned = scanned + 1u;
                }
                if (mlen > best_len || (mlen == best_len && (best_dist == 0u || dist < best_dist))) {
                    best_len = mlen;
                    best_dist = dist;
                }
            }
        }
        c = c + 1u;
    }

    if (best_len >= 3u && best_dist > 0u) {
        token_len[i] = best_len;
        token_dist[i] = best_dist;
    } else {
        token_len[i] = 0u;
        token_dist[i] = 0u;
    }
}
