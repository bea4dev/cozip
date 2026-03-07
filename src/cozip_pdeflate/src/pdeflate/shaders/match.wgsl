@group(0) @binding(0) var<storage, read> src_words: array<u32>;
@group(0) @binding(1) var<storage, read> chunk_starts: array<u32>;
@group(0) @binding(2) var<storage, read> table_chunk_bases: array<u32>;
@group(0) @binding(3) var<storage, read> table_chunk_counts: array<u32>;
@group(0) @binding(4) var<storage, read> prefix2_first_ids: array<u32>;
@group(0) @binding(5) var<storage, read> table_entry_lens: array<u32>;
@group(0) @binding(6) var<storage, read> table_entry_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> table_data_words: array<u32>;
@group(0) @binding(8) var<storage, read> params: array<u32>;
@group(0) @binding(9) var<storage, read_write> out_matches: array<u32>;

fn load_src(idx: u32) -> u32 {
    let w = src_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_table_data(idx: u32) -> u32 {
    let w = table_data_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn chunk_index_for_pos(pos: u32, chunk_count: u32) -> u32 {
    var lo = 0u;
    var hi = chunk_count;
    loop {
        if (lo + 1u >= hi) {
            break;
        }
        let mid = (lo + hi) >> 1u;
        if (pos < chunk_starts[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return lo;
}

fn load_rep(table_id: u32, idx: u32) -> u32 {
    let entry_len = table_entry_lens[table_id];
    if (entry_len == 0u) {
        return 0u;
    }
    let off = table_entry_offsets[table_id];
    let rel = idx % entry_len;
    return load_table_data(off + rel);
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

fn section_index_for_local_pos(local_pos: u32, section_count: u32, total_len: u32) -> u32 {
    var sec = (local_pos * section_count) / total_len;
    if (sec >= section_count) {
        sec = section_count - 1u;
    }
    loop {
        let s0 = section_start(sec, section_count, total_len);
        if (local_pos >= s0 || sec == 0u) {
            break;
        }
        sec = sec - 1u;
    }
    loop {
        let s1 = section_start(sec + 1u, section_count, total_len);
        if (local_pos < s1 || (sec + 1u) >= section_count) {
            break;
        }
        sec = sec + 1u;
    }
    return sec;
}

@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x + gid3.y * params[3];
    let total_len = params[0];
    if (gid >= total_len) {
        return;
    }

    let max_ref_len = params[1];
    let min_ref_len = params[2];
    let section_count = params[4];
    let chunk_count = params[5];
    let chunk_idx = chunk_index_for_pos(gid, chunk_count);
    if (chunk_idx >= chunk_count) {
        out_matches[gid] = 0u;
        return;
    }
    let chunk_start = chunk_starts[chunk_idx];
    let chunk_end = chunk_starts[chunk_idx + 1u];
    let chunk_len = chunk_end - chunk_start;
    if (chunk_len == 0u) {
        out_matches[gid] = 0u;
        return;
    }
    let local_pos = gid - chunk_start;
    let sec = section_index_for_local_pos(local_pos, section_count, chunk_len);
    let sec_end = chunk_start + section_start(sec + 1u, section_count, chunk_len);
    if (gid + min_ref_len > sec_end) {
        out_matches[gid] = 0u;
        return;
    }
    if (gid + 1u >= sec_end) {
        out_matches[gid] = 0u;
        return;
    }

    let table_base = table_chunk_bases[chunk_idx];
    let table_count = table_chunk_counts[chunk_idx];
    if (table_count == 0u) {
        out_matches[gid] = 0u;
        return;
    }
    let k2 = (load_src(gid) << 8u) | load_src(gid + 1u);
    let prefix_base = chunk_idx << 16u;
    let cand_enc = prefix2_first_ids[prefix_base + k2];
    if (cand_enc == 0u) {
        out_matches[gid] = 0u;
        return;
    }
    let cand = cand_enc - 1u;
    if (cand >= table_count) {
        out_matches[gid] = 0u;
        return;
    }
    let table_id = table_base + cand;

    var max_len = max_ref_len;
    let remain = sec_end - gid;
    if (remain < max_len) {
        max_len = remain;
    }
    if (max_len < min_ref_len) {
        out_matches[gid] = 0u;
        return;
    }

    var m: u32 = 0u;
    loop {
        if (m >= max_len) {
            break;
        }
        if (load_src(gid + m) != load_rep(table_id, m)) {
            break;
        }
        m = m + 1u;
    }

    if (m >= min_ref_len) {
        out_matches[gid] = (cand << 16u) | (m & 0xffffu);
    } else {
        out_matches[gid] = 0u;
    }
}
