struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> token_flags: array<u32>;

@group(0) @binding(1)
var<storage, read> token_kind: array<u32>;

@group(0) @binding(2)
var<storage, read> token_match_len: array<u32>;

@group(0) @binding(3)
var<storage, read> token_match_dist: array<u32>;

@group(0) @binding(4)
var<storage, read> token_lit: array<u32>;

@group(0) @binding(5)
var<storage, read> token_prefix: array<u32>;

@group(0) @binding(6)
var<storage, read_write> codes: array<u32>;

@group(0) @binding(7)
var<storage, read_write> bitlens: array<u32>;

@group(0) @binding(8)
var<uniform> params: Params;

fn reverse_bits_u32(value: u32, bit_len: u32) -> u32 {
    var out: u32 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= bit_len) {
            break;
        }
        out = (out << 1u) | ((value >> i) & 1u);
        i = i + 1u;
    }
    return out;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 8388480u);
    if (idx >= params.len) {
        return;
    }

    if (token_flags[idx] == 0u) {
        return;
    }

    let token_index = token_prefix[idx];
    let kind = token_kind[idx];

    if (kind == 0u) {
        let lit = token_lit[idx];
        var code: u32 = 0u;
        var bits: u32 = 0u;
        if (lit <= 143u) {
            code = 0x30u + lit;
            bits = 8u;
        } else {
            code = 0x190u + (lit - 144u);
            bits = 9u;
        }

        codes[token_index] = reverse_bits_u32(code, bits);
        bitlens[token_index] = bits;
    } else {
        let mlen = token_match_len[idx];
        let mdist = token_match_dist[idx];

        var len_symbol: u32 = 257u;
        var len_extra_bits: u32 = 0u;
        var len_extra_value: u32 = 0u;

        if (mlen <= 10u) {
            len_symbol = 254u + mlen;
        } else if (mlen == 258u) {
            len_symbol = 285u;
        } else {
            var symbol: u32 = 265u;
            var base: u32 = 11u;
            var extra: u32 = 1u;
            var found: bool = false;

            loop {
                if (extra > 5u || found) {
                    break;
                }

                var j: u32 = 0u;
                loop {
                    if (j >= 4u) {
                        break;
                    }
                    let maxv = base + ((1u << extra) - 1u);
                    if (mlen >= base && mlen <= maxv) {
                        len_symbol = symbol;
                        len_extra_bits = extra;
                        len_extra_value = mlen - base;
                        found = true;
                        break;
                    }
                    base = maxv + 1u;
                    symbol = symbol + 1u;
                    j = j + 1u;
                }

                extra = extra + 1u;
            }
        }

        var len_code: u32 = 0u;
        var len_bits: u32 = 0u;
        if (len_symbol <= 279u) {
            len_code = len_symbol - 256u;
            len_bits = 7u;
        } else {
            len_code = 0xC0u + (len_symbol - 280u);
            len_bits = 8u;
        }

        var out_code: u32 = 0u;
        var out_bits: u32 = 0u;

        out_code = out_code | (reverse_bits_u32(len_code, len_bits) << out_bits);
        out_bits = out_bits + len_bits;

        if (len_extra_bits > 0u) {
            out_code = out_code | (len_extra_value << out_bits);
            out_bits = out_bits + len_extra_bits;
        }

        var dist_symbol: u32 = 0u;
        var dist_extra_bits: u32 = 0u;
        var dist_extra_value: u32 = 0u;

        if (mdist <= 1u) {
            dist_symbol = 0u;
        } else if (mdist <= 4u) {
            dist_symbol = mdist - 1u;
        } else {
            var symbol: u32 = 4u;
            var base: u32 = 5u;
            var extra: u32 = 1u;
            var found: bool = false;

            loop {
                if (extra > 13u || found) {
                    break;
                }

                var j: u32 = 0u;
                loop {
                    if (j >= 2u) {
                        break;
                    }
                    let maxv = base + ((1u << extra) - 1u);
                    if (mdist >= base && mdist <= maxv) {
                        dist_symbol = symbol;
                        dist_extra_bits = extra;
                        dist_extra_value = mdist - base;
                        found = true;
                        break;
                    }
                    base = maxv + 1u;
                    symbol = symbol + 1u;
                    j = j + 1u;
                }

                extra = extra + 1u;
            }
        }

        let dist_code = reverse_bits_u32(dist_symbol, 5u);
        out_code = out_code | (dist_code << out_bits);
        out_bits = out_bits + 5u;
        if (dist_extra_bits > 0u) {
            out_code = out_code | (dist_extra_value << out_bits);
            out_bits = out_bits + dist_extra_bits;
        }

        codes[token_index] = out_code;
        bitlens[token_index] = out_bits;
    }
}
