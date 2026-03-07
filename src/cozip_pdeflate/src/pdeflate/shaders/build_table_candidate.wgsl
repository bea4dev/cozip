@group(0) @binding(0) var<storage, read> src_words: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_words: array<u32>;

fn load_src(idx: u32) -> u32 {
    let w = src_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn key3(pos: u32) -> u32 {
    return (load_src(pos) << 16u) | (load_src(pos + 1u) << 8u) | load_src(pos + 2u);
}

fn choose_entry_len(match_len: u32, max_entry_len: u32) -> u32 {
    var capped = match_len;
    if (capped > max_entry_len) {
        capped = max_entry_len;
    }
    var best = 1u;
    if (3u <= capped) { best = 3u; }
    if (4u <= capped) { best = 4u; }
    if (6u <= capped) { best = 6u; }
    if (8u <= capped) { best = 8u; }
    if (12u <= capped) { best = 12u; }
    if (16u <= capped) { best = 16u; }
    if (24u <= capped) { best = 24u; }
    if (32u <= capped) { best = 32u; }
    return best;
}

@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let sid = gid3.x;
    let total_len = params[0];
    let sample_stride = params[1];
    let max_entry_len = params[2];
    let min_seed_match_len = params[3];
    let history_limit = params[4];
    let probe_limit = params[5];
    let sample_count = params[6];
    if (sid >= sample_count) {
        return;
    }

    let out_base = sid * 4u;
    out_words[out_base + 0u] = 0u;
    out_words[out_base + 1u] = 0u;
    out_words[out_base + 2u] = 0u;
    out_words[out_base + 3u] = 0u;

    let pos = sid * sample_stride;
    if (pos + 2u >= total_len) {
        return;
    }

    let key = key3(pos);
    var best_mlen = 0u;
    var probes = 0u;
    var dist = 1u;
    loop {
        if (dist > history_limit || probes >= probe_limit || dist > pos) {
            break;
        }
        let prev = pos - dist;
        if (prev + 2u >= total_len) {
            dist = dist + 1u;
            continue;
        }
        if (key3(prev) != key) {
            dist = dist + 1u;
            continue;
        }
        probes = probes + 1u;
        if (pos + 3u >= total_len || prev + 3u >= total_len || load_src(pos + 3u) != load_src(prev + 3u)) {
            dist = dist + 1u;
            continue;
        }
        var m = 4u;
        var limit = max_entry_len;
        let remain = total_len - pos;
        if (remain < limit) {
            limit = remain;
        }
        loop {
            if (m >= limit) {
                break;
            }
            if (load_src(pos + m) != load_src(prev + m)) {
                break;
            }
            m = m + 1u;
        }
        if (m > best_mlen) {
            best_mlen = m;
            if (best_mlen >= max_entry_len) {
                break;
            }
        }
        dist = dist + 1u;
    }

    if (best_mlen < min_seed_match_len) {
        return;
    }
    let cand_len = choose_entry_len(best_mlen, max_entry_len);
    let score = best_mlen - 2u;
    out_words[out_base + 0u] = score;
    out_words[out_base + 1u] = pos;
    out_words[out_base + 2u] = cand_len;
    out_words[out_base + 3u] = best_mlen;
}
