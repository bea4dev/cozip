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
var<storage, read_write> litlen_freq: array<atomic<u32>>;

@group(0) @binding(7)
var<storage, read_write> dist_freq: array<atomic<u32>>;

@group(0) @binding(8)
var<uniform> params: Params;

const PHASE1_WG_SIZE: u32 = 128u;
const LITLEN_SYMBOLS: u32 = 286u;
const DIST_SYMBOLS: u32 = 30u;

var<workgroup> local_litlen_freq: array<u32, 286>;
var<workgroup> local_dist_freq: array<u32, 30>;

fn mode_max_match_scan(mode: u32) -> u32 {
    switch (mode) {
        case 1u: { return 128u; } // Balanced
        case 2u: { return 192u; } // Ratio
        default: { return 64u; }  // Speed
    }
}

fn mode_max_match_len(mode: u32) -> u32 {
    switch (mode) {
        case 1u: { return 128u; } // Balanced
        case 2u: { return 258u; } // Ratio
        default: { return 64u; }  // Speed
    }
}

fn mode_dist_candidate_count(mode: u32) -> u32 {
    switch (mode) {
        case 1u: { return 28u; } // Balanced
        case 2u: { return 32u; } // Ratio
        default: { return 20u; } // Speed
    }
}

fn mode_lazy_delta(mode: u32) -> u32 {
    switch (mode) {
        case 1u: { return 1u; } // Balanced
        case 2u: { return 2u; } // Ratio
        default: { return 0u; } // Speed
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

fn litlen_symbol_for_len(mlen_in: u32) -> u32 {
    let mlen = min(max(mlen_in, 3u), 258u);
    if (mlen <= 10u) {
        return 254u + mlen;
    }
    if (mlen == 258u) {
        return 285u;
    }

    var symbol: u32 = 265u;
    var base: u32 = 11u;
    var extra: u32 = 1u;
    loop {
        if (extra > 5u) {
            break;
        }
        var j: u32 = 0u;
        loop {
            if (j >= 4u) {
                break;
            }
            let maxv = base + ((1u << extra) - 1u);
            if (mlen >= base && mlen <= maxv) {
                return symbol;
            }
            base = maxv + 1u;
            symbol = symbol + 1u;
            j = j + 1u;
        }
        extra = extra + 1u;
    }
    return 285u;
}

fn dist_symbol_for_dist(mdist_in: u32) -> u32 {
    let mdist = max(mdist_in, 1u);
    if (mdist <= 1u) {
        return 0u;
    }
    if (mdist <= 4u) {
        return mdist - 1u;
    }

    var symbol: u32 = 4u;
    var base: u32 = 5u;
    var extra: u32 = 1u;
    loop {
        if (extra > 13u) {
            break;
        }
        var j: u32 = 0u;
        loop {
            if (j >= 2u) {
                break;
            }
            let maxv = base + ((1u << extra) - 1u);
            if (mdist >= base && mdist <= maxv) {
                return symbol;
            }
            base = maxv + 1u;
            symbol = symbol + 1u;
            j = j + 1u;
        }
        extra = extra + 1u;
    }
    return 29u;
}

@compute @workgroup_size(128)
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    var s: u32 = lid;
    loop {
        if (s >= LITLEN_SYMBOLS) {
            break;
        }
        local_litlen_freq[s] = 0u;
        s = s + PHASE1_WG_SIZE;
    }
    s = lid;
    loop {
        if (s >= DIST_SYMBOLS) {
            break;
        }
        local_dist_freq[s] = 0u;
        s = s + PHASE1_WG_SIZE;
    }
    workgroupBarrier();

    let segment_size = max(params.block_size, 1u);
    let seg_id = wid.x + (wid.y * max(1u, num_wg.x));
    let seg_start = seg_id * segment_size;
    if (seg_start >= params.len) {
        return;
    }
    let seg_end = min(seg_start + segment_size, params.len);

    let match_scan_limit = mode_max_match_scan(params.mode);
    let match_len_limit = mode_max_match_len(params.mode);
    let dist_limit = mode_dist_candidate_count(params.mode);

    var i = seg_start + lid;
    loop {
        if (i >= seg_end) {
            break;
        }

        let lit = byte_at(i);
        token_lit[i] = lit;
        token_flags[i] = 0u;
        token_kind[i] = 0u;

        var best_dist: u32 = 0u;
        var best_len: u32 = 0u;
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
                        // Cannot improve beyond mode cap; stop checking farther candidates.
                        if (best_len >= match_len_limit) {
                            break;
                        }
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

        i = i + PHASE1_WG_SIZE;
    }

    workgroupBarrier();

    if (lid == 0u) {
        let lazy_delta = mode_lazy_delta(params.mode);
        var pos = seg_start;
        loop {
            if (pos >= seg_end) {
                break;
            }

            if (token_len[pos] >= 3u && token_dist[pos] > 0u) {
                let mlen = min(token_len[pos], seg_end - pos);
                // Segment tail can clamp to 1-2 bytes; those must stay literals.
                var take_match = (mlen >= 3u);
                if (take_match && lazy_delta > 0u && (pos + 1u) < seg_end && token_len[pos + 1u] >= 3u && token_dist[pos + 1u] > 0u) {
                    let next_len = min(token_len[pos + 1u], seg_end - (pos + 1u));
                    if (next_len >= mlen + lazy_delta) {
                        take_match = false;
                    }
                }

                if (take_match) {
                    token_flags[pos] = 1u;
                    token_kind[pos] = 1u;
                    token_len[pos] = mlen;
                    let len_symbol = litlen_symbol_for_len(mlen);
                    let dist_symbol = dist_symbol_for_dist(token_dist[pos]);
                    local_litlen_freq[len_symbol] = local_litlen_freq[len_symbol] + 1u;
                    local_dist_freq[dist_symbol] = local_dist_freq[dist_symbol] + 1u;

                    var j: u32 = 1u;
                    loop {
                        if (j >= mlen || (pos + j) >= seg_end) {
                            break;
                        }
                        token_flags[pos + j] = 0u;
                        token_kind[pos + j] = 0u;
                        token_len[pos + j] = 0u;
                        token_dist[pos + j] = 0u;
                        j = j + 1u;
                    }
                    pos = pos + mlen;
                } else {
                    token_flags[pos] = 1u;
                    token_kind[pos] = 0u;
                    token_len[pos] = 0u;
                    token_dist[pos] = 0u;
                    let lit = min(token_lit[pos], 255u);
                    local_litlen_freq[lit] = local_litlen_freq[lit] + 1u;
                    pos = pos + 1u;
                }
            } else {
                token_flags[pos] = 1u;
                token_kind[pos] = 0u;
                token_len[pos] = 0u;
                token_dist[pos] = 0u;
                let lit = min(token_lit[pos], 255u);
                local_litlen_freq[lit] = local_litlen_freq[lit] + 1u;
                pos = pos + 1u;
            }
        }
    }

    workgroupBarrier();

    s = lid;
    loop {
        if (s >= LITLEN_SYMBOLS) {
            break;
        }
        let value = local_litlen_freq[s];
        if (value > 0u) {
            atomicAdd(&litlen_freq[s], value);
        }
        s = s + PHASE1_WG_SIZE;
    }

    s = lid;
    loop {
        if (s >= DIST_SYMBOLS) {
            break;
        }
        let value = local_dist_freq[s];
        if (value > 0u) {
            atomicAdd(&dist_freq[s], value);
        }
        s = s + PHASE1_WG_SIZE;
    }
}
