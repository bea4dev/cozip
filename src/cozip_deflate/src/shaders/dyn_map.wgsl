struct Params {
    len: u32,
    block_size: u32,
    mode: u32,
    header_bits: u32,
}

@group(0) @binding(0)
var<storage, read> token_flags: array<u32>;

@group(0) @binding(1)
var<storage, read> token_match_len: array<u32>;

@group(0) @binding(2)
var<storage, read> token_match_dist: array<u32>;

@group(0) @binding(3)
var<storage, read> token_lit: array<u32>;

@group(0) @binding(4)
var<storage, read> dyn_table: array<u32>;

@group(0) @binding(5)
var<storage, read_write> out_codes: array<u32>;

@group(0) @binding(6)
var<storage, read_write> out_overflow: array<u32>;

@group(0) @binding(7)
var<storage, read_write> out_bitlens: array<u32>;

@group(0) @binding(8)
var<uniform> params: Params;

fn litlen_code(sym: u32) -> u32 {
    return dyn_table[sym];
}

fn litlen_bits(sym: u32) -> u32 {
    return dyn_table[286u + sym];
}

fn dist_code(sym: u32) -> u32 {
    return dyn_table[572u + sym];
}

fn dist_bits(sym: u32) -> u32 {
    return dyn_table[602u + sym];
}

fn litlen_symbol_for_len(mlen_in: u32) -> vec3<u32> {
    let mlen = min(max(mlen_in, 3u), 258u);
    if (mlen <= 10u) {
        return vec3<u32>(254u + mlen, 0u, 0u);
    }
    if (mlen == 258u) {
        return vec3<u32>(285u, 0u, 0u);
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
                return vec3<u32>(symbol, mlen - base, extra);
            }
            base = maxv + 1u;
            symbol = symbol + 1u;
            j = j + 1u;
        }
        extra = extra + 1u;
    }
    return vec3<u32>(285u, 0u, 0u);
}

fn dist_symbol_for_dist(mdist_in: u32) -> vec3<u32> {
    let mdist = max(mdist_in, 1u);
    if (mdist <= 1u) {
        return vec3<u32>(0u, 0u, 0u);
    }
    if (mdist <= 4u) {
        return vec3<u32>(mdist - 1u, 0u, 0u);
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
                return vec3<u32>(symbol, mdist - base, extra);
            }
            base = maxv + 1u;
            symbol = symbol + 1u;
            j = j + 1u;
        }
        extra = extra + 1u;
    }
    return vec3<u32>(29u, 0u, 0u);
}

fn append_bits(
    value: u32,
    bits: u32,
    code_lo: ptr<function, u32>,
    code_hi: ptr<function, u32>,
    bitlen: ptr<function, u32>,
) {
    if (bits == 0u) {
        return;
    }
    let cur = *bitlen;
    if (cur < 32u) {
        if (cur + bits <= 32u) {
            *code_lo = *code_lo | (value << cur);
        } else {
            let low_bits = 32u - cur;
            *code_lo = *code_lo | (value << cur);
            *code_hi = *code_hi | (value >> low_bits);
        }
    } else {
        let hi_shift = cur - 32u;
        *code_hi = *code_hi | (value << hi_shift);
    }
    *bitlen = cur + bits;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 8388480u);
    if (idx >= params.len || token_flags[idx] == 0u) {
        return;
    }

    var code_lo: u32 = 0u;
    var code_hi: u32 = 0u;
    var bits: u32 = 0u;

    if (token_match_len[idx] < 3u || token_match_dist[idx] == 0u) {
        let sym = min(token_lit[idx], 255u);
        append_bits(litlen_code(sym), litlen_bits(sym), &code_lo, &code_hi, &bits);
    } else {
        let len_info = litlen_symbol_for_len(token_match_len[idx]);
        let len_sym = len_info.x;
        let len_extra_val = len_info.y;
        let len_extra_bits = len_info.z;
        append_bits(
            litlen_code(len_sym),
            litlen_bits(len_sym),
            &code_lo,
            &code_hi,
            &bits,
        );
        append_bits(len_extra_val, len_extra_bits, &code_lo, &code_hi, &bits);

        let dist_info = dist_symbol_for_dist(token_match_dist[idx]);
        let dist_sym = dist_info.x;
        let dist_extra_val = dist_info.y;
        let dist_extra_bits = dist_info.z;
        append_bits(dist_code(dist_sym), dist_bits(dist_sym), &code_lo, &code_hi, &bits);
        append_bits(dist_extra_val, dist_extra_bits, &code_lo, &code_hi, &bits);
    }

    out_codes[idx] = code_lo;
    out_overflow[idx] = code_hi;
    out_bitlens[idx] = bits;
}
