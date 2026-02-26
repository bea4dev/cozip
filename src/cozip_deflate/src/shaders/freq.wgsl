struct Params {
    len: u32,
    block_size: u32,
    mode: u32,
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
var<storage, read_write> litlen_freq: array<atomic<u32>>;

@group(0) @binding(6)
var<storage, read_write> dist_freq: array<atomic<u32>>;

@group(0) @binding(7)
var<uniform> params: Params;

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

const LITLEN_SYMBOLS: u32 = 286u;
const DIST_SYMBOLS: u32 = 30u;
const FREQ_WORKGROUP_SIZE: u32 = 128u;

var<workgroup> local_litlen_freq: array<atomic<u32>, 286>;
var<workgroup> local_dist_freq: array<atomic<u32>, 30>;

@compute @workgroup_size(128)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    var s: u32 = lid;
    loop {
        if (s >= LITLEN_SYMBOLS) {
            break;
        }
        atomicStore(&local_litlen_freq[s], 0u);
        s = s + FREQ_WORKGROUP_SIZE;
    }

    s = lid;
    loop {
        if (s >= DIST_SYMBOLS) {
            break;
        }
        atomicStore(&local_dist_freq[s], 0u);
        s = s + FREQ_WORKGROUP_SIZE;
    }
    workgroupBarrier();

    let thread_count = max(1u, num_wg.x * num_wg.y * FREQ_WORKGROUP_SIZE);
    var idx = id.x + (id.y * 8388480u);
    loop {
        if (idx >= params.len) {
            break;
        }
        if (token_flags[idx] != 0u) {
            if (token_kind[idx] == 0u) {
                let lit = min(token_lit[idx], 255u);
                atomicAdd(&local_litlen_freq[lit], 1u);
            } else {
                let len_symbol = litlen_symbol_for_len(token_match_len[idx]);
                let dist_symbol = dist_symbol_for_dist(token_match_dist[idx]);
                atomicAdd(&local_litlen_freq[len_symbol], 1u);
                atomicAdd(&local_dist_freq[dist_symbol], 1u);
            }
        }
        idx = idx + thread_count;
    }
    workgroupBarrier();

    s = lid;
    loop {
        if (s >= LITLEN_SYMBOLS) {
            break;
        }
        let value = atomicLoad(&local_litlen_freq[s]);
        if (value > 0u) {
            atomicAdd(&litlen_freq[s], value);
        }
        s = s + FREQ_WORKGROUP_SIZE;
    }

    s = lid;
    loop {
        if (s >= DIST_SYMBOLS) {
            break;
        }
        let value = atomicLoad(&local_dist_freq[s]);
        if (value > 0u) {
            atomicAdd(&dist_freq[s], value);
        }
        s = s + FREQ_WORKGROUP_SIZE;
    }
}
