struct Params {
    len: u32,
    block_size: u32,
    mode: u32,
    _pad1: u32,
}

@group(0) @binding(1)
var<storage, read_write> token_flags: array<u32>;

@group(0) @binding(2)
var<storage, read_write> token_kind: array<u32>;

@group(0) @binding(3)
var<storage, read_write> token_len: array<u32>;

@group(0) @binding(4)
var<storage, read_write> token_dist: array<u32>;

@group(0) @binding(6)
var<uniform> params: Params;

fn mode_lazy_delta(mode: u32) -> u32 {
    switch (mode) {
        case 1u: { return 1u; } // Balanced
        case 2u: { return 2u; } // Ratio
        default: { return 0u; } // Speed
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let segment_size = max(params.block_size, 1u);
    let seg_start = id.x * segment_size;
    if (seg_start >= params.len) {
        return;
    }
    let seg_end = min(seg_start + segment_size, params.len);

    var i: u32 = seg_start;
    let lazy_delta = mode_lazy_delta(params.mode);
    loop {
        if (i >= seg_end) {
            break;
        }

        if (token_len[i] >= 3u && token_dist[i] > 0u) {
            let mlen = min(token_len[i], seg_end - i);
            // Segment tail can clamp to 1-2 bytes; those must stay literals.
            var take_match = (mlen >= 3u);
            if (take_match && lazy_delta > 0u && (i + 1u) < seg_end && token_len[i + 1u] >= 3u && token_dist[i + 1u] > 0u) {
                let next_len = min(token_len[i + 1u], seg_end - (i + 1u));
                if (next_len >= mlen + lazy_delta) {
                    take_match = false;
                }
            }

            if (take_match) {
                token_flags[i] = 1u;
                token_kind[i] = 1u;
                token_len[i] = mlen;

                var j: u32 = 1u;
                loop {
                    if (j >= mlen || (i + j) >= seg_end) {
                        break;
                    }
                    token_flags[i + j] = 0u;
                    token_kind[i + j] = 0u;
                    token_len[i + j] = 0u;
                    token_dist[i + j] = 0u;
                    j = j + 1u;
                }
                i = i + mlen;
            } else {
                token_flags[i] = 1u;
                token_kind[i] = 0u;
                token_len[i] = 0u;
                token_dist[i] = 0u;
                i = i + 1u;
            }
        } else {
            token_flags[i] = 1u;
            token_kind[i] = 0u;
            token_len[i] = 0u;
            token_dist[i] = 0u;
            i = i + 1u;
        }
    }
}
