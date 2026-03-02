use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::PDeflateError;

const WORKGROUP_SIZE: u32 = 128;
const PREFIX2_TABLE_SIZE: usize = 1 << 16;
const PACK_CHUNK_HEADER_SIZE: usize = 32;
const PACK_CHUNK_MAGIC: [u8; 4] = *b"PDF0";
const PACK_CHUNK_VERSION: u16 = 0;

const MATCH_SHADER: &str = r#"
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
    let sec = (local_pos * section_count) / chunk_len;
    let sec_end = chunk_start + (((sec + 1u) * chunk_len) / section_count);
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
"#;

const BUILD_TABLE_FREQ_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> src_words: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> freq: array<atomic<u32>>;

fn load_src(idx: u32) -> u32 {
    let w = src_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let i = gid3.x;
    let total_len = params[0];
    if (i >= total_len) {
        return;
    }
    let b = load_src(i);
    atomicAdd(&freq[b], 1u);
}
"#;

const BUILD_TABLE_CANDIDATE_SHADER: &str = r#"
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
"#;

const BUILD_TABLE_BUCKET_COUNT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> cand_words: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> bucket_counts: array<atomic<u32>>;

fn key_for(score: u32, len: u32) -> u32 {
    let s = min(score, 255u);
    let l = min(len, 255u);
    return (s << 8u) | l;
}

@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let sid = gid3.x;
    let sample_count = params[0];
    if (sid >= sample_count) {
        return;
    }
    let base = sid * 4u;
    let score = cand_words[base + 0u];
    if (score == 0u) {
        return;
    }
    let len = cand_words[base + 2u];
    if (len == 0u) {
        return;
    }
    let key = key_for(score, len);
    atomicAdd(&bucket_counts[key], 1u);
}
"#;

const BUILD_TABLE_BUCKET_SCATTER_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> cand_words: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read> bucket_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> bucket_cursors: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> out_sorted_indices: array<u32>;

fn key_for(score: u32, len: u32) -> u32 {
    let s = min(score, 255u);
    let l = min(len, 255u);
    return (s << 8u) | l;
}

@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let sid = gid3.x;
    let sample_count = params[0];
    if (sid >= sample_count) {
        return;
    }
    let base = sid * 4u;
    let score = cand_words[base + 0u];
    if (score == 0u) {
        return;
    }
    let len = cand_words[base + 2u];
    if (len == 0u) {
        return;
    }
    let key = key_for(score, len);
    let local = atomicAdd(&bucket_cursors[key], 1u);
    let out_idx = bucket_offsets[key] + local;
    out_sorted_indices[out_idx] = sid;
}
"#;

const BUILD_TABLE_BUCKET_PREFIX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> bucket_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> bucket_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> params: array<u32>;

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u || gid3.y != 0u || gid3.z != 0u) {
        return;
    }
    var sorted_count = 0u;
    var key: i32 = 65535;
    loop {
        if (key < 0) {
            break;
        }
        let k = u32(key);
        let c = bucket_counts[k];
        bucket_offsets[k] = sorted_count;
        sorted_count = sorted_count + c;
        key = key - 1;
    }
    // params[2] is sorted_count for BUILD_TABLE_FINALIZE_SHADER.
    params[2] = sorted_count;
}
"#;

const BUILD_TABLE_FINALIZE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> src_words: array<u32>;
@group(0) @binding(1) var<storage, read> freq_words: array<u32>;
@group(0) @binding(2) var<storage, read> cand_words: array<u32>;
@group(0) @binding(3) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_meta: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_data: array<u32>;

fn load_src(idx: u32) -> u32 {
    let w = src_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_out_data(idx: u32) -> u32 {
    let w = out_data[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn store_out_data(idx: u32, value: u32) {
    let wi = idx >> 2u;
    let shift = (idx & 3u) * 8u;
    let mask = 0xffu << shift;
    let prev = out_data[wi];
    out_data[wi] = (prev & (~mask)) | ((value & 0xffu) << shift);
}

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u || gid3.y != 0u || gid3.z != 0u) {
        return;
    }

    let total_len = params[0];
    let sample_count = params[1];
    let sorted_count = params[2];
    let max_entry_len = params[3];
    let max_entries = params[4];
    let literal_limit = params[5];
    let min_literal_freq = params[6];
    let max_data_bytes = params[7];

    var taken_literals: array<u32, 256>;
    var b_init = 0u;
    loop {
        if (b_init >= 256u) {
            break;
        }
        taken_literals[b_init] = 0u;
        b_init = b_init + 1u;
    }

    var out_count = 0u;
    var data_cursor = 0u;

    var lit_rank = 0u;
    loop {
        if (lit_rank >= literal_limit || out_count >= max_entries) {
            break;
        }
        var best_byte = 0u;
        var best_freq = 0u;
        var has_best = 0u;
        var b = 0u;
        loop {
            if (b >= 256u) {
                break;
            }
            if (taken_literals[b] == 0u) {
                let f = freq_words[b];
                if (f >= min_literal_freq) {
                    if (has_best == 0u || f > best_freq || (f == best_freq && b < best_byte)) {
                        has_best = 1u;
                        best_byte = b;
                        best_freq = f;
                    }
                }
            }
            b = b + 1u;
        }
        if (has_best == 0u) {
            break;
        }
        if (data_cursor + 1u > max_data_bytes) {
            break;
        }
        out_meta[1u + out_count * 2u] = data_cursor;
        out_meta[2u + out_count * 2u] = 1u;
        store_out_data(data_cursor, best_byte);
        data_cursor = data_cursor + 1u;
        out_count = out_count + 1u;
        taken_literals[best_byte] = 1u;
        lit_rank = lit_rank + 1u;
    }

    var sid_i = 0u;
    loop {
        if (sid_i >= sorted_count || out_count >= max_entries) {
            break;
        }
        let sid = sorted_indices[sid_i];
        sid_i = sid_i + 1u;
        if (sid >= sample_count) {
            continue;
        }
        let base = sid * 4u;
        let score = cand_words[base + 0u];
        if (score == 0u) {
            continue;
        }
        let pos = cand_words[base + 1u];
        let len = cand_words[base + 2u];
        if (len == 0u || len > max_entry_len || pos + len > total_len) {
            continue;
        }
        if (data_cursor + len > max_data_bytes) {
            break;
        }

        var is_dup = 0u;
        var e = 0u;
        loop {
            if (e >= out_count) {
                break;
            }
            let e_off = out_meta[1u + e * 2u];
            let e_len = out_meta[2u + e * 2u];
            if (e_len == len) {
                var same = 1u;
                var j = 0u;
                loop {
                    if (j >= len) {
                        break;
                    }
                    if (load_src(pos + j) != load_out_data(e_off + j)) {
                        same = 0u;
                        break;
                    }
                    j = j + 1u;
                }
                if (same == 1u) {
                    is_dup = 1u;
                    break;
                }
            }
            e = e + 1u;
        }
        if (is_dup == 1u) {
            continue;
        }

        out_meta[1u + out_count * 2u] = data_cursor;
        out_meta[2u + out_count * 2u] = len;
        var j = 0u;
        loop {
            if (j >= len) {
                break;
            }
            store_out_data(data_cursor + j, load_src(pos + j));
            j = j + 1u;
        }
        data_cursor = data_cursor + len;
        out_count = out_count + 1u;
    }

    out_meta[0] = out_count;
    out_meta[1u + max_entries * 2u] = data_cursor;
}
"#;

const BUILD_TABLE_PACK_INDEX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> table_meta: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> table_index_words: array<u32>;

fn store_table_index_byte(idx: u32, value: u32) {
    let wi = idx >> 2u;
    let shift = (idx & 3u) * 8u;
    let mask = 0xffu << shift;
    let prev = table_index_words[wi];
    table_index_words[wi] = (prev & (~mask)) | ((value & 0xffu) << shift);
}

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u || gid3.y != 0u || gid3.z != 0u) {
        return;
    }
    let max_entries = params[4];
    let out_count = min(table_meta[0], max_entries);
    var i = 0u;
    loop {
        if (i >= out_count) {
            break;
        }
        let len = table_meta[2u + i * 2u];
        store_table_index_byte(i, len);
        i = i + 1u;
    }
}
"#;

const BUILD_TABLE_SORT_BUCKETS: usize = 256 * 256;

const MATCH_PREPARE_TABLE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> table_chunk_bases: array<u32>;
@group(0) @binding(1) var<storage, read> table_chunk_counts: array<u32>;
@group(0) @binding(2) var<storage, read> table_chunk_index_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> table_chunk_data_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> table_index_words_in: array<u32>;
@group(0) @binding(5) var<storage, read> table_data_words_in: array<u32>;
@group(0) @binding(6) var<storage, read> params: array<u32>;
@group(0) @binding(7) var<storage, read_write> prefix2_first_ids: array<u32>;
@group(0) @binding(8) var<storage, read_write> table_entry_lens: array<u32>;
@group(0) @binding(9) var<storage, read_write> table_entry_offsets: array<u32>;

fn load_table_index_byte(idx: u32) -> u32 {
    let w = table_index_words_in[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_table_data_byte(idx: u32) -> u32 {
    let w = table_data_words_in[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u || gid3.y != 0u || gid3.z != 0u) {
        return;
    }
    let chunk_count = params[0];
    var chunk = 0u;
    loop {
        if (chunk >= chunk_count) {
            break;
        }
        let chunk_base = table_chunk_bases[chunk];
        let chunk_entries = table_chunk_counts[chunk];
        let table_index_base = table_chunk_index_offsets[chunk];
        let mut_data_base = table_chunk_data_offsets[chunk];
        let prefix_base = chunk << 16u;

        var local_id = 0u;
        var data_cursor = mut_data_base;
        loop {
            if (local_id >= chunk_entries) {
                break;
            }
            let len = load_table_index_byte(table_index_base + local_id);
            let global_id = chunk_base + local_id;
            table_entry_lens[global_id] = len;
            table_entry_offsets[global_id] = data_cursor;
            if (len >= 2u) {
                let key = (load_table_data_byte(data_cursor) << 8u) | load_table_data_byte(data_cursor + 1u);
                let slot = prefix_base + key;
                if (prefix2_first_ids[slot] == 0u) {
                    prefix2_first_ids[slot] = local_id + 1u;
                }
            }
            data_cursor = data_cursor + len;
            local_id = local_id + 1u;
        }
        chunk = chunk + 1u;
    }
}
"#;

const SECTION_ENCODE_SHADER: &str = r#"
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
    return (sec * total_len) / section_count;
}

fn is_valid_match(word: u32, remaining: u32, min_ref_len: u32, max_ref_len: u32) -> bool {
    if (word == 0u) {
        return false;
    }
    let mlen = word & 0xffffu;
    if mlen < min_ref_len || mlen > max_ref_len {
        return false;
    }
    return mlen <= remaining;
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
                out_lens[sec] = 0xffffffffu;
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
            out_lens[sec] = 0xffffffffu;
            return;
        }
        var i: u32 = 0u;
        loop {
            if (i >= lit_len) {
                break;
            }
            cursor = emit_cmd_byte(base, cursor, cap, load_src(src_base + lit_start + i));
            if (cursor == 0xffffffffu) {
                out_lens[sec] = 0xffffffffu;
                return;
            }
            i = i + 1u;
        }
    }

    out_lens[sec] = cursor;
}
"#;

const SECTION_META_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> out_lens: array<u32>;
@group(0) @binding(1) var<storage, read_write> section_prefix: array<u32>;
@group(0) @binding(2) var<storage, read_write> section_index_words: array<u32>;
@group(0) @binding(3) var<storage, read_write> section_meta_words: array<u32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;

fn write_section_index_byte(idx: u32, value: u32) {
    let wi = idx >> 2u;
    let shift = (idx & 3u) * 8u;
    let mask = 0xffu << shift;
    let prev = section_index_words[wi];
    section_index_words[wi] = (prev & (~mask)) | ((value & 0xffu) << shift);
}

fn append_varint(cursor_in: u32, value: u32) -> u32 {
    var cursor = cursor_in;
    var v = value;
    loop {
        var b = v & 0x7fu;
        v = v >> 7u;
        if (v != 0u) {
            b = b | 0x80u;
        }
        write_section_index_byte(cursor, b);
        cursor = cursor + 1u;
        if (v == 0u) {
            break;
        }
    }
    return cursor;
}

@compute
@workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    if (gid3.x != 0u) {
        return;
    }
    let section_count = params[1];
    var total_cmd_len = 0u;
    var section_index_len = 0u;
    var overflow = 0u;
    var sec = 0u;
    loop {
        if (sec >= section_count) {
            break;
        }
        let len = out_lens[sec];
        if (len == 0xffffffffu) {
            overflow = 1u;
            break;
        }
        section_prefix[sec] = total_cmd_len;
        total_cmd_len = total_cmd_len + len;
        section_index_len = append_varint(section_index_len, len);
        sec = sec + 1u;
    }
    if (overflow != 0u) {
        section_meta_words[0] = 0xffffffffu;
        section_meta_words[1] = 0xffffffffu;
        section_meta_words[2] = 1u;
        section_meta_words[3] = 0u;
    } else {
        section_meta_words[0] = total_cmd_len;
        section_meta_words[1] = section_index_len;
        section_meta_words[2] = 0u;
        section_meta_words[3] = 0u;
    }
}
"#;

const PACK_CHUNK_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> header_words: array<u32>;
@group(0) @binding(2) var<storage, read> table_index_words: array<u32>;
@group(0) @binding(3) var<storage, read> table_data_words: array<u32>;
@group(0) @binding(4) var<storage, read> section_index_words: array<u32>;
@group(0) @binding(5) var<storage, read> section_cmd_words: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_words: array<u32>;

fn load_header_byte(idx: u32) -> u32 {
    let w = header_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_table_index_byte(idx: u32) -> u32 {
    let w = table_index_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_table_data_byte(idx: u32) -> u32 {
    let w = table_data_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_section_index_byte(idx: u32) -> u32 {
    let w = section_index_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_section_cmd_byte(idx: u32) -> u32 {
    let w = section_cmd_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn read_output_byte(pos: u32) -> u32 {
    let header_len = params[1];
    let table_index_off = params[2];
    let table_index_len = params[3];
    let table_data_off = params[4];
    let table_data_len = params[5];
    let section_index_off = params[6];
    let section_index_len = params[7];
    let section_cmd_off = params[8];
    let section_cmd_len = params[9];

    if (pos < header_len) {
        return load_header_byte(pos);
    }
    if (pos >= table_index_off && pos < table_index_off + table_index_len) {
        return load_table_index_byte(pos - table_index_off);
    }
    if (pos >= table_data_off && pos < table_data_off + table_data_len) {
        return load_table_data_byte(pos - table_data_off);
    }
    if (pos >= section_index_off && pos < section_index_off + section_index_len) {
        return load_section_index_byte(pos - section_index_off);
    }
    if (pos >= section_cmd_off && pos < section_cmd_off + section_cmd_len) {
        return load_section_cmd_byte(pos - section_cmd_off);
    }
    return 0u;
}

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let word_idx = gid3.x;
    let total_len = params[0];
    let out_words_len = params[10];
    if (word_idx >= out_words_len) {
        return;
    }
    let base = word_idx * 4u;
    var b0 = 0u;
    var b1 = 0u;
    var b2 = 0u;
    var b3 = 0u;
    if (base < total_len) {
        b0 = read_output_byte(base);
    }
    if (base + 1u < total_len) {
        b1 = read_output_byte(base + 1u);
    }
    if (base + 2u < total_len) {
        b2 = read_output_byte(base + 2u);
    }
    if (base + 3u < total_len) {
        b3 = read_output_byte(base + 3u);
    }
    out_words[word_idx] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}
"#;

const PACK_CHUNK_SPARSE_SECTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> section_meta_words: array<u32>;
@group(0) @binding(2) var<storage, read> table_index_words: array<u32>;
@group(0) @binding(3) var<storage, read> table_data_words: array<u32>;
@group(0) @binding(4) var<storage, read> section_index_words: array<u32>;
@group(0) @binding(5) var<storage, read> section_lens_words: array<u32>;
@group(0) @binding(6) var<storage, read> section_prefix_words: array<u32>;
@group(0) @binding(7) var<storage, read> section_offsets_words: array<u32>;
@group(0) @binding(8) var<storage, read> section_cmd_sparse_words: array<u32>;
@group(0) @binding(9) var<storage, read_write> out_words: array<u32>;

fn low8(v: u32, s: u32) -> u32 {
    return (v >> s) & 0xffu;
}

fn load_header_byte(idx: u32, section_cmd_off: u32) -> u32 {
    let chunk_len = params[0];
    let table_count = params[1];
    let section_count = params[2];
    let table_index_off = params[3];
    let table_data_off = params[5];
    let section_index_off = params[7];

    if (idx == 0u) {
        return 0x50u;
    }
    if (idx == 1u) {
        return 0x44u;
    }
    if (idx == 2u) {
        return 0x46u;
    }
    if (idx == 3u) {
        return 0x30u;
    }
    if (idx == 4u || idx == 5u || idx == 6u || idx == 7u) {
        return 0u;
    }
    if (idx >= 8u && idx < 12u) {
        return low8(chunk_len, (idx - 8u) * 8u);
    }
    if (idx >= 12u && idx < 14u) {
        return low8(table_count, (idx - 12u) * 8u);
    }
    if (idx >= 14u && idx < 16u) {
        return low8(section_count, (idx - 14u) * 8u);
    }
    if (idx >= 16u && idx < 20u) {
        return low8(table_index_off, (idx - 16u) * 8u);
    }
    if (idx >= 20u && idx < 24u) {
        return low8(table_data_off, (idx - 20u) * 8u);
    }
    if (idx >= 24u && idx < 28u) {
        return low8(section_index_off, (idx - 24u) * 8u);
    }
    if (idx >= 28u && idx < 32u) {
        return low8(section_cmd_off, (idx - 28u) * 8u);
    }
    return 0u;
}

fn load_table_index_byte(idx: u32) -> u32 {
    let w = table_index_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_table_data_byte(idx: u32) -> u32 {
    let w = table_data_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_section_index_byte(idx: u32) -> u32 {
    let w = section_index_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_section_cmd_sparse_byte(idx: u32) -> u32 {
    let w = section_cmd_sparse_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_section_cmd_compact_byte(local: u32, section_count: u32) -> u32 {
    if (section_count == 0u) {
        return 0u;
    }
    var lo = 0u;
    var hi = section_count;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) >> 1u;
        let start = section_prefix_words[mid];
        let end = start + section_lens_words[mid];
        if (local < start) {
            hi = mid;
            continue;
        }
        if (local >= end) {
            lo = mid + 1u;
            continue;
        }
        let rel = local - start;
        let src = section_offsets_words[mid] + rel;
        return load_section_cmd_sparse_byte(src);
    }
    return 0u;
}

fn read_output_byte(pos: u32) -> u32 {
    let section_cmd_len = section_meta_words[0];
    let section_index_len = section_meta_words[1];
    let overflow = section_meta_words[2];
    let header_len = 32u;
    let table_index_off = params[3];
    let table_index_len = params[4];
    let table_data_off = params[5];
    let table_data_len = params[6];
    let section_index_off = params[7];
    let section_count = params[2];
    let section_cmd_off = section_index_off + section_index_len;
    let total_len = section_cmd_off + section_cmd_len;

    if (overflow != 0u || section_cmd_len == 0xffffffffu || section_index_len == 0xffffffffu) {
        return 0u;
    }
    if (pos >= total_len) {
        return 0u;
    }

    if (pos < header_len) {
        return load_header_byte(pos, section_cmd_off);
    }
    if (pos >= table_index_off && pos < table_index_off + table_index_len) {
        return load_table_index_byte(pos - table_index_off);
    }
    if (pos >= table_data_off && pos < table_data_off + table_data_len) {
        return load_table_data_byte(pos - table_data_off);
    }
    if (pos >= section_index_off && pos < section_index_off + section_index_len) {
        return load_section_index_byte(pos - section_index_off);
    }
    if (pos >= section_cmd_off && pos < section_cmd_off + section_cmd_len) {
        return load_section_cmd_compact_byte(pos - section_cmd_off, section_count);
    }
    return 0u;
}

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let word_idx = gid3.x;
    let out_words_len = params[11];
    if (word_idx >= out_words_len) {
        return;
    }
    let base = word_idx * 4u;
    let b0 = read_output_byte(base);
    let b1 = read_output_byte(base + 1u);
    let b2 = read_output_byte(base + 2u);
    let b3 = read_output_byte(base + 3u);
    out_words[word_idx] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}
"#;

const PACK_CHUNK_SPARSE_PREPARE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> params: array<u32>;
@group(0) @binding(1) var<storage, read> section_meta_words: array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch_words: array<u32>;

fn align_up4(v: u32) -> u32 {
    return (v + 3u) & 0xfffffffcu;
}

@compute
@workgroup_size(1, 1, 1)
fn main() {
    let section_index_off = params[7];
    let section_index_cap_len = params[8];
    let section_cmd_cap_len = params[9];
    let total_len_cap = section_index_off + section_index_cap_len + section_cmd_cap_len;

    let section_cmd_len = section_meta_words[0];
    let section_index_len = section_meta_words[1];
    let overflow = section_meta_words[2];

    var total_len = 0xffffffffu;
    if (overflow == 0u && section_cmd_len != 0xffffffffu && section_index_len != 0xffffffffu) {
        let section_cmd_off = section_index_off + section_index_len;
        total_len = section_cmd_off + section_cmd_len;
        if (total_len > total_len_cap) {
            total_len = 0xffffffffu;
        }
    }

    let total_words = select(max(1u, align_up4(total_len) >> 2u), 1u, total_len == 0xffffffffu);
    let groups_x = max(1u, (total_words + 255u) / 256u);

    params[10] = total_len;
    params[11] = total_words;
    dispatch_words[0] = groups_x;
    dispatch_words[1] = 1u;
    dispatch_words[2] = 1u;
}
"#;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuMatchProfile {
    pub(crate) upload_ms: f64,
    pub(crate) wait_ms: f64,
    pub(crate) map_copy_ms: f64,
    pub(crate) total_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuBatchKernelProfile {
    pub(crate) pack_inputs_ms: f64,
    pub(crate) scratch_acquire_ms: f64,
    pub(crate) match_table_copy_ms: f64,
    pub(crate) match_prepare_dispatch_ms: f64,
    pub(crate) match_kernel_dispatch_ms: f64,
    pub(crate) match_submit_ms: f64,
    pub(crate) section_setup_ms: f64,
    pub(crate) section_pass_dispatch_ms: f64,
    pub(crate) section_copy_dispatch_ms: f64,
    pub(crate) section_submit_ms: f64,
    pub(crate) section_wrap_ms: f64,
    pub(crate) sparse_lens_submit_ms: f64,
    pub(crate) sparse_lens_submit_done_wait_ms: f64,
    pub(crate) sparse_lens_map_after_done_ms: f64,
    pub(crate) sparse_lens_poll_calls: u64,
    pub(crate) sparse_lens_yield_calls: u64,
    pub(crate) sparse_lens_wait_ms: f64,
    pub(crate) sparse_lens_copy_ms: f64,
    pub(crate) sparse_prepare_ms: f64,
    pub(crate) sparse_scratch_acquire_ms: f64,
    pub(crate) sparse_upload_dispatch_ms: f64,
    pub(crate) sparse_submit_ms: f64,
    pub(crate) sparse_wait_ms: f64,
    pub(crate) sparse_copy_ms: f64,
    pub(crate) sparse_total_ms: f64,
}

pub(crate) struct GpuMatchInput<'a> {
    pub(crate) src: &'a [u8],
    pub(crate) table: &'a [Vec<u8>],
    pub(crate) table_gpu: Option<&'a GpuPackedTableDevice>,
    pub(crate) table_index: Option<&'a [u8]>,
    pub(crate) table_data: Option<&'a [u8]>,
    pub(crate) max_ref_len: usize,
    pub(crate) min_ref_len: usize,
    pub(crate) section_count: usize,
}

pub(crate) struct GpuMatchOutput {
    pub(crate) packed_matches: Vec<u32>,
    pub(crate) profile: GpuMatchProfile,
}

pub(crate) struct GpuBatchChunkOutput {
    pub(crate) packed_matches: Vec<u32>,
}

pub(crate) struct GpuBatchMatchOutput {
    pub(crate) chunks: Vec<GpuBatchChunkOutput>,
    pub(crate) profile: GpuMatchProfile,
}

pub(crate) struct GpuBatchSectionEncodeChunkOutput {
    pub(crate) section_cmd_lens: Vec<u32>,
    pub(crate) section_cmd: Vec<u8>,
    pub(crate) section_cmd_device: Option<GpuSectionCmdDeviceOutput>,
}

pub(crate) struct GpuBatchSectionEncodeOutput {
    pub(crate) chunks: Vec<GpuBatchSectionEncodeChunkOutput>,
    pub(crate) match_profile: GpuMatchProfile,
    pub(crate) section_profile: GpuSectionEncodeProfile,
    pub(crate) kernel_profile: GpuBatchKernelProfile,
    pub(crate) keepalive: Option<GpuBatchSectionKeepalive>,
}

pub(crate) struct GpuSectionCmdDeviceOutput {
    pub(crate) section_count: usize,
    pub(crate) section_offsets_buffer: Arc<wgpu::Buffer>,
    pub(crate) out_lens_buffer: Arc<wgpu::Buffer>,
    pub(crate) section_prefix_buffer: Arc<wgpu::Buffer>,
    pub(crate) section_index_buffer: Arc<wgpu::Buffer>,
    pub(crate) section_meta_buffer: Arc<wgpu::Buffer>,
    pub(crate) section_meta_readback_buffer: Arc<wgpu::Buffer>,
    pub(crate) out_cmd_buffer: Arc<wgpu::Buffer>,
    pub(crate) out_lens_bytes: u64,
    pub(crate) out_cmd_bytes: u64,
    pub(crate) section_index_cap_bytes: u64,
}

pub(crate) struct GpuBatchSectionKeepalive {
    scratch: Option<GpuBatchScratch>,
}

impl GpuBatchSectionKeepalive {
    fn new(scratch: GpuBatchScratch) -> Self {
        Self {
            scratch: Some(scratch),
        }
    }
}

impl Drop for GpuBatchSectionKeepalive {
    fn drop(&mut self) {
        let Some(scratch) = self.scratch.take() else {
            return;
        };
        let Ok(r) = runtime() else {
            return;
        };
        if let Ok(mut pool) = r.scratch_pool.lock() {
            if pool.len() < GPU_SCRATCH_POOL_LIMIT {
                pool.push(scratch);
            }
        }
    }
}

pub(crate) struct GpuSparsePackInput<'a> {
    pub(crate) chunk_len: usize,
    pub(crate) section_count: usize,
    pub(crate) table: &'a GpuPackedTableDevice,
    pub(crate) section: &'a GpuSectionCmdDeviceOutput,
}

pub(crate) struct GpuSparsePackChunkOutput {
    pub(crate) payload: Vec<u8>,
    pub(crate) table_count: usize,
}

pub(crate) struct GpuSparseSectionHostOutput {
    pub(crate) section_cmd_lens: Vec<u32>,
    pub(crate) section_cmd: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuSparsePackBatchProfile {
    pub(crate) chunks: usize,
    pub(crate) lens_bytes_total: u64,
    pub(crate) out_cmd_bytes_total: u64,
    pub(crate) lens_submit_ms: f64,
    pub(crate) lens_submit_done_wait_ms: f64,
    pub(crate) lens_map_after_done_ms: f64,
    pub(crate) lens_poll_calls: u64,
    pub(crate) lens_yield_calls: u64,
    pub(crate) lens_wait_ms: f64,
    pub(crate) lens_copy_ms: f64,
    pub(crate) prepare_ms: f64,
    pub(crate) scratch_acquire_ms: f64,
    pub(crate) upload_dispatch_ms: f64,
    pub(crate) submit_ms: f64,
    pub(crate) wait_ms: f64,
    pub(crate) copy_ms: f64,
    pub(crate) total_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuSectionEncodeProfile {
    pub(crate) upload_ms: f64,
    pub(crate) wait_ms: f64,
    pub(crate) map_copy_ms: f64,
    pub(crate) total_ms: f64,
}

#[allow(dead_code)]
pub(crate) struct GpuSectionEncodeOutput {
    pub(crate) section_cmd_lens: Vec<u32>,
    pub(crate) section_cmd: Vec<u8>,
    pub(crate) profile: GpuSectionEncodeProfile,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuBuildTableProfile {
    pub(crate) upload_ms: f64,
    pub(crate) freq_kernel_ms: f64,
    pub(crate) candidate_kernel_ms: f64,
    pub(crate) readback_ms: f64,
    pub(crate) materialize_ms: f64,
    pub(crate) sort_ms: f64,
    pub(crate) sample_count: usize,
}

pub(crate) struct GpuPackedTable {
    pub(crate) table_index: Vec<u8>,
    pub(crate) table_data: Vec<u8>,
    pub(crate) table_count: usize,
}

pub(crate) struct GpuPackedTableDevice {
    pub(crate) table_index_buffer: wgpu::Buffer,
    pub(crate) table_data_buffer: wgpu::Buffer,
    pub(crate) table_meta_buffer: wgpu::Buffer,
    pub(crate) table_count: usize,
    pub(crate) table_index_len: usize,
    pub(crate) table_data_len: usize,
    pub(crate) sizes_known: bool,
    pub(crate) max_entries: usize,
    pub(crate) table_data_bytes_cap: usize,
}

struct GpuMatchRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    pack_chunk_bind_group_layout: wgpu::BindGroupLayout,
    pack_chunk_pipeline: wgpu::ComputePipeline,
    pack_sparse_prepare_bind_group_layout: wgpu::BindGroupLayout,
    pack_sparse_prepare_pipeline: wgpu::ComputePipeline,
    pack_sparse_bind_group_layout: wgpu::BindGroupLayout,
    pack_sparse_pipeline: wgpu::ComputePipeline,
    section_encode_bind_group_layout: wgpu::BindGroupLayout,
    section_encode_pipeline: wgpu::ComputePipeline,
    section_meta_bind_group_layout: wgpu::BindGroupLayout,
    section_meta_pipeline: wgpu::ComputePipeline,
    build_table_freq_bind_group_layout: wgpu::BindGroupLayout,
    build_table_freq_pipeline: wgpu::ComputePipeline,
    build_table_candidate_bind_group_layout: wgpu::BindGroupLayout,
    build_table_candidate_pipeline: wgpu::ComputePipeline,
    build_table_bucket_count_bind_group_layout: wgpu::BindGroupLayout,
    build_table_bucket_count_pipeline: wgpu::ComputePipeline,
    build_table_bucket_prefix_bind_group_layout: wgpu::BindGroupLayout,
    build_table_bucket_prefix_pipeline: wgpu::ComputePipeline,
    build_table_bucket_scatter_bind_group_layout: wgpu::BindGroupLayout,
    build_table_bucket_scatter_pipeline: wgpu::ComputePipeline,
    build_table_finalize_bind_group_layout: wgpu::BindGroupLayout,
    build_table_finalize_pipeline: wgpu::ComputePipeline,
    build_table_pack_index_bind_group_layout: wgpu::BindGroupLayout,
    build_table_pack_index_pipeline: wgpu::ComputePipeline,
    match_prepare_table_bind_group_layout: wgpu::BindGroupLayout,
    match_prepare_table_pipeline: wgpu::ComputePipeline,
    max_storage_binding_size: u64,
    scratch_pool: Mutex<Vec<GpuBatchScratch>>,
    sparse_pack_pool: Mutex<Vec<GpuSparsePackScratch>>,
}

struct GpuBatchScratch {
    src_cap_bytes: u64,
    chunk_starts_cap_bytes: u64,
    table_base_cap_bytes: u64,
    table_count_cap_bytes: u64,
    table_index_offsets_cap_bytes: u64,
    table_data_offsets_cap_bytes: u64,
    table_index_cap_bytes: u64,
    prefix_first_cap_bytes: u64,
    table_lens_cap_bytes: u64,
    table_offsets_cap_bytes: u64,
    table_data_cap_bytes: u64,
    prep_params_cap_bytes: u64,
    params_cap_bytes: u64,
    out_cap_bytes: u64,
    src_buffer: wgpu::Buffer,
    chunk_starts_buffer: wgpu::Buffer,
    table_base_buffer: wgpu::Buffer,
    table_count_buffer: wgpu::Buffer,
    table_index_offsets_buffer: wgpu::Buffer,
    table_data_offsets_buffer: wgpu::Buffer,
    table_index_buffer: wgpu::Buffer,
    prefix_first_buffer: wgpu::Buffer,
    table_lens_buffer: wgpu::Buffer,
    table_offsets_buffer: wgpu::Buffer,
    table_data_buffer: wgpu::Buffer,
    prep_params_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    prep_bind_group: wgpu::BindGroup,
    section_slots: Vec<GpuSectionEncodeSlot>,
}

struct GpuSectionEncodeSlot {
    section_count_cap: usize,
    cmd_cap_bytes: u64,
    section_index_cap_bytes: u64,
    section_offsets_buffer: Arc<wgpu::Buffer>,
    section_caps_buffer: Arc<wgpu::Buffer>,
    section_params_buffer: Arc<wgpu::Buffer>,
    out_lens_buffer: Arc<wgpu::Buffer>,
    section_prefix_buffer: Arc<wgpu::Buffer>,
    section_index_buffer: Arc<wgpu::Buffer>,
    section_meta_buffer: Arc<wgpu::Buffer>,
    out_cmd_buffer: Arc<wgpu::Buffer>,
    readback_meta_buffer: Arc<wgpu::Buffer>,
    bind_group: wgpu::BindGroup,
    meta_bind_group: wgpu::BindGroup,
}

struct GpuSparsePackScratch {
    params_cap_bytes: u64,
    out_cap_bytes: u64,
    params_buffer: wgpu::Buffer,
    params_readback_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    dispatch_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
}

#[derive(Clone, Copy)]
struct GpuSparsePackScratchCaps {
    params_bytes: u64,
    out_bytes: u64,
}

#[derive(Clone, Copy)]
struct GpuBatchScratchCaps {
    src_bytes: u64,
    chunk_starts_bytes: u64,
    table_base_bytes: u64,
    table_count_bytes: u64,
    table_index_offsets_bytes: u64,
    table_data_offsets_bytes: u64,
    table_index_bytes: u64,
    prefix_first_bytes: u64,
    table_lens_bytes: u64,
    table_offsets_bytes: u64,
    table_data_bytes: u64,
    prep_params_bytes: u64,
    params_bytes: u64,
    out_bytes: u64,
}

const GPU_SCRATCH_POOL_LIMIT: usize = 4;
const GPU_SPARSE_PACK_POOL_LIMIT: usize = 16;
// Command stream cap heuristic for GPU section encode.
// If this underestimates a pathological section, kernel marks overflow and
// caller falls back to CPU for correctness.
const GPU_SECTION_CMD_CAP_MULTIPLIER: usize = 2;
const GPU_SECTION_CMD_CAP_PAD: usize = 64;

static GPU_RUNTIME: OnceLock<Result<GpuMatchRuntime, String>> = OnceLock::new();
static GPU_SUBMIT_STREAM_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn round_capacity_bytes(required: u64) -> u64 {
    let req = required.max(4);
    req.checked_next_power_of_two().unwrap_or(req)
}

#[inline]
fn align_up4(v: usize) -> usize {
    (v + 3) & !3
}

#[inline]
fn estimate_section_cmd_cap_bytes(sec_len: usize) -> Result<usize, PDeflateError> {
    sec_len
        .checked_mul(GPU_SECTION_CMD_CAP_MULTIPLIER)
        .and_then(|v| v.checked_add(GPU_SECTION_CMD_CAP_PAD))
        .ok_or(PDeflateError::NumericOverflow)
}

struct ByteWordPacker {
    words: Vec<u32>,
    len_bytes: usize,
}

impl ByteWordPacker {
    fn with_capacity_bytes(capacity: usize) -> Self {
        Self {
            words: Vec::with_capacity(capacity.div_ceil(4)),
            len_bytes: 0,
        }
    }

    fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    fn align4(&mut self) {
        let pad = align_up4(self.len_bytes).saturating_sub(self.len_bytes);
        if pad > 0 {
            self.append_zeros(pad);
        }
    }

    fn append_zeros(&mut self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let new_len = self.len_bytes.saturating_add(bytes);
        let need_words = new_len.div_ceil(4);
        if need_words > self.words.len() {
            self.words.resize(need_words, 0);
        }
        self.len_bytes = new_len;
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let new_len = self.len_bytes.saturating_add(bytes.len());
        let need_words = new_len.div_ceil(4);
        if need_words > self.words.len() {
            self.words.resize(need_words, 0);
        }
        let mut pos = self.len_bytes;
        for &b in bytes {
            let wi = pos >> 2;
            let shift = (pos & 3) << 3;
            self.words[wi] |= u32::from(b) << shift;
            pos = pos.saturating_add(1);
        }
        self.len_bytes = new_len;
    }

    fn write_bytes_at(&mut self, offset: usize, bytes: &[u8]) -> Result<(), PDeflateError> {
        let end = offset
            .checked_add(bytes.len())
            .ok_or(PDeflateError::NumericOverflow)?;
        if end > self.len_bytes {
            return Err(PDeflateError::NumericOverflow);
        }
        let mut pos = offset;
        for &b in bytes {
            let wi = pos >> 2;
            let shift = (pos & 3) << 3;
            let mask = !(0xff_u32 << shift);
            let old = self.words[wi] & mask;
            self.words[wi] = old | (u32::from(b) << shift);
            pos = pos.saturating_add(1);
        }
        Ok(())
    }

    fn into_words(self) -> Vec<u32> {
        self.words
    }
}

fn storage_upload_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl GpuBatchScratch {
    fn fits(&self, caps: GpuBatchScratchCaps) -> bool {
        self.src_cap_bytes >= caps.src_bytes
            && self.chunk_starts_cap_bytes >= caps.chunk_starts_bytes
            && self.table_base_cap_bytes >= caps.table_base_bytes
            && self.table_count_cap_bytes >= caps.table_count_bytes
            && self.table_index_offsets_cap_bytes >= caps.table_index_offsets_bytes
            && self.table_data_offsets_cap_bytes >= caps.table_data_offsets_bytes
            && self.table_index_cap_bytes >= caps.table_index_bytes
            && self.prefix_first_cap_bytes >= caps.prefix_first_bytes
            && self.table_lens_cap_bytes >= caps.table_lens_bytes
            && self.table_offsets_cap_bytes >= caps.table_offsets_bytes
            && self.table_data_cap_bytes >= caps.table_data_bytes
            && self.prep_params_cap_bytes >= caps.prep_params_bytes
            && self.params_cap_bytes >= caps.params_bytes
            && self.out_cap_bytes >= caps.out_bytes
    }

    fn new(runtime: &GpuMatchRuntime, required: GpuBatchScratchCaps) -> Self {
        let src_cap_bytes = round_capacity_bytes(required.src_bytes);
        let chunk_starts_cap_bytes = round_capacity_bytes(required.chunk_starts_bytes);
        let table_base_cap_bytes = round_capacity_bytes(required.table_base_bytes);
        let table_count_cap_bytes = round_capacity_bytes(required.table_count_bytes);
        let table_index_offsets_cap_bytes =
            round_capacity_bytes(required.table_index_offsets_bytes);
        let table_data_offsets_cap_bytes = round_capacity_bytes(required.table_data_offsets_bytes);
        let table_index_cap_bytes = round_capacity_bytes(required.table_index_bytes);
        let prefix_first_cap_bytes = round_capacity_bytes(required.prefix_first_bytes);
        let table_lens_cap_bytes = round_capacity_bytes(required.table_lens_bytes);
        let table_offsets_cap_bytes = round_capacity_bytes(required.table_offsets_bytes);
        let table_data_cap_bytes = round_capacity_bytes(required.table_data_bytes);
        let prep_params_cap_bytes = round_capacity_bytes(required.prep_params_bytes);
        let params_cap_bytes = round_capacity_bytes(required.params_bytes);
        let out_cap_bytes = round_capacity_bytes(required.out_bytes);

        let src_buffer =
            storage_upload_buffer(&runtime.device, "cozip-pdeflate-src", src_cap_bytes);
        let chunk_starts_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-chunk-starts",
            chunk_starts_cap_bytes,
        );
        let table_base_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-base",
            table_base_cap_bytes,
        );
        let table_count_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-count",
            table_count_cap_bytes,
        );
        let table_index_offsets_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-index-offsets",
            table_index_offsets_cap_bytes,
        );
        let table_data_offsets_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-data-offsets",
            table_data_offsets_cap_bytes,
        );
        let table_index_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-index",
            table_index_cap_bytes,
        );
        let prefix_first_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-prefix2-first",
            prefix_first_cap_bytes,
        );
        let table_lens_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-lens",
            table_lens_cap_bytes,
        );
        let table_offsets_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-offsets",
            table_offsets_cap_bytes,
        );
        let table_data_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-data",
            table_data_cap_bytes,
        );
        let prep_params_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-prep-params",
            prep_params_cap_bytes,
        );
        let params_buffer =
            storage_upload_buffer(&runtime.device, "cozip-pdeflate-params", params_cap_bytes);
        let out_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-out"),
            size: out_cap_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-readback"),
            size: out_cap_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-match-bg"),
                layout: &runtime.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: chunk_starts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: table_base_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: table_count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: prefix_first_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: table_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: table_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: table_data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: out_buffer.as_entire_binding(),
                    },
                ],
            });
        let prep_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-prep-table-bg"),
                layout: &runtime.match_prepare_table_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: table_base_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: table_count_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: table_index_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: table_data_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: table_index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: table_data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: prep_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: prefix_first_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: table_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: table_offsets_buffer.as_entire_binding(),
                    },
                ],
            });

        Self {
            src_cap_bytes,
            chunk_starts_cap_bytes,
            table_base_cap_bytes,
            table_count_cap_bytes,
            table_index_offsets_cap_bytes,
            table_data_offsets_cap_bytes,
            table_index_cap_bytes,
            prefix_first_cap_bytes,
            table_lens_cap_bytes,
            table_offsets_cap_bytes,
            table_data_cap_bytes,
            prep_params_cap_bytes,
            params_cap_bytes,
            out_cap_bytes,
            src_buffer,
            chunk_starts_buffer,
            table_base_buffer,
            table_count_buffer,
            table_index_offsets_buffer,
            table_data_offsets_buffer,
            table_index_buffer,
            prefix_first_buffer,
            table_lens_buffer,
            table_offsets_buffer,
            table_data_buffer,
            prep_params_buffer,
            params_buffer,
            out_buffer,
            readback_buffer,
            bind_group,
            prep_bind_group,
            section_slots: Vec::new(),
        }
    }
}

impl GpuSectionEncodeSlot {
    fn fits(&self, section_count: usize, cmd_cap_bytes: u64) -> bool {
        self.section_count_cap >= section_count && self.cmd_cap_bytes >= cmd_cap_bytes
    }

    fn new(
        runtime: &GpuMatchRuntime,
        match_scratch: &GpuBatchScratch,
        section_count: usize,
        cmd_cap_bytes: u64,
    ) -> Result<Self, PDeflateError> {
        let section_count_u64 =
            u64::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
        let section_offsets_bytes = section_count_u64.saturating_mul(4);
        let section_caps_bytes = section_count_u64.saturating_mul(4);
        let section_params_bytes = 24u64;
        let out_lens_bytes = section_count_u64.saturating_mul(4);
        let section_prefix_bytes = section_count_u64.saturating_mul(4);
        let section_index_cap_bytes = u64::try_from(section_count.saturating_mul(5))
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let out_cmd_bytes = cmd_cap_bytes.max(4);
        let section_offsets_buffer = Arc::new(storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-se-offs",
            section_offsets_bytes.max(4),
        ));
        let section_caps_buffer = Arc::new(storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-se-caps",
            section_caps_bytes.max(4),
        ));
        let section_params_buffer = Arc::new(storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-se-params",
            section_params_bytes.max(4),
        ));
        let out_lens_buffer = Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-out-lens"),
            size: out_lens_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let section_prefix_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-prefix"),
                size: section_prefix_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        let section_index_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-index"),
                size: section_index_cap_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        let section_meta_buffer = Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-meta"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let out_cmd_buffer = Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-out-cmd"),
            size: out_cmd_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let readback_meta_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-readback-meta"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        let bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-bg"),
                layout: &runtime.section_encode_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: match_scratch.src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: match_scratch.out_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: section_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_caps_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: out_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: out_cmd_buffer.as_entire_binding(),
                    },
                ],
            });
        let meta_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-meta-bg"),
                layout: &runtime.section_meta_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: section_prefix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: section_index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                ],
            });
        Ok(Self {
            section_count_cap: section_count,
            cmd_cap_bytes: out_cmd_bytes,
            section_index_cap_bytes: section_index_cap_bytes.max(4),
            section_offsets_buffer,
            section_caps_buffer,
            section_params_buffer,
            out_lens_buffer,
            section_prefix_buffer,
            section_index_buffer,
            section_meta_buffer,
            out_cmd_buffer,
            readback_meta_buffer,
            bind_group,
            meta_bind_group,
        })
    }
}

impl GpuSparsePackScratch {
    fn fits(&self, caps: GpuSparsePackScratchCaps) -> bool {
        self.params_cap_bytes >= caps.params_bytes && self.out_cap_bytes >= caps.out_bytes
    }

    fn new(runtime: &GpuMatchRuntime, caps: GpuSparsePackScratchCaps) -> Self {
        let params_cap_bytes = round_capacity_bytes(caps.params_bytes);
        let out_cap_bytes = round_capacity_bytes(caps.out_bytes);
        let params_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-params"),
            size: params_cap_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-params-readback"),
            size: params_cap_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-out"),
            size: out_cap_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dispatch_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-dispatch"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-readback"),
            size: out_cap_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            params_cap_bytes,
            out_cap_bytes,
            params_buffer,
            params_readback_buffer,
            out_buffer,
            dispatch_buffer,
            readback_buffer,
        }
    }
}

fn init_runtime() -> Result<GpuMatchRuntime, String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "adapter not found".to_string())?;

    let required_limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("cozip-pdeflate-gpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits,
        },
        None,
    ))
    .map_err(|e| format!("request_device failed: {e}"))?;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cozip-pdeflate-match-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-match-shader"),
        source: wgpu::ShaderSource::Wgsl(MATCH_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-match-pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cozip-pdeflate-match-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let pack_chunk_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-pack-chunk-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let pack_chunk_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-pack-chunk-shader"),
        source: wgpu::ShaderSource::Wgsl(PACK_CHUNK_SHADER.into()),
    });
    let pack_chunk_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-pack-chunk-pl"),
        bind_group_layouts: &[&pack_chunk_bind_group_layout],
        push_constant_ranges: &[],
    });
    let pack_chunk_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cozip-pdeflate-pack-chunk-pipeline"),
        layout: Some(&pack_chunk_layout),
        module: &pack_chunk_shader,
        entry_point: "main",
    });
    let pack_sparse_prepare_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-prepare-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let pack_sparse_prepare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-prepare-shader"),
        source: wgpu::ShaderSource::Wgsl(PACK_CHUNK_SPARSE_PREPARE_SHADER.into()),
    });
    let pack_sparse_prepare_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-prepare-pl"),
            bind_group_layouts: &[&pack_sparse_prepare_bind_group_layout],
            push_constant_ranges: &[],
        });
    let pack_sparse_prepare_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-prepare-pipeline"),
            layout: Some(&pack_sparse_prepare_layout),
            module: &pack_sparse_prepare_shader,
            entry_point: "main",
        });

    let pack_sparse_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let pack_sparse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-shader"),
        source: wgpu::ShaderSource::Wgsl(PACK_CHUNK_SPARSE_SECTION_SHADER.into()),
    });
    let pack_sparse_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-pl"),
        bind_group_layouts: &[&pack_sparse_bind_group_layout],
        push_constant_ranges: &[],
    });
    let pack_sparse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-pipeline"),
        layout: Some(&pack_sparse_layout),
        module: &pack_sparse_shader,
        entry_point: "main",
    });

    let section_encode_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-encode-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let section_encode_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-encode-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_ENCODE_SHADER.into()),
    });
    let section_encode_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-encode-pl"),
        bind_group_layouts: &[&section_encode_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_encode_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-section-encode-pipeline"),
            layout: Some(&section_encode_layout),
            module: &section_encode_shader,
            entry_point: "main",
        });
    let section_meta_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-meta-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let section_meta_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-meta-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_META_SHADER.into()),
    });
    let section_meta_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-meta-pl"),
        bind_group_layouts: &[&section_meta_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_meta_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cozip-pdeflate-section-meta-pipeline"),
        layout: Some(&section_meta_layout),
        module: &section_meta_shader,
        entry_point: "main",
    });
    let build_table_freq_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-freq-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_freq_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-build-table-freq-shader"),
        source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_FREQ_SHADER.into()),
    });
    let build_table_freq_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-build-table-freq-pl"),
        bind_group_layouts: &[&build_table_freq_bind_group_layout],
        push_constant_ranges: &[],
    });
    let build_table_freq_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-freq-pipeline"),
            layout: Some(&build_table_freq_layout),
            module: &build_table_freq_shader,
            entry_point: "main",
        });

    let build_table_candidate_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-candidate-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_candidate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-build-table-candidate-shader"),
        source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_CANDIDATE_SHADER.into()),
    });
    let build_table_candidate_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-candidate-pl"),
            bind_group_layouts: &[&build_table_candidate_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_candidate_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-candidate-pipeline"),
            layout: Some(&build_table_candidate_layout),
            module: &build_table_candidate_shader,
            entry_point: "main",
        });
    let build_table_bucket_count_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-count-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_bucket_count_shader =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-count-shader"),
            source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_BUCKET_COUNT_SHADER.into()),
        });
    let build_table_bucket_count_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-count-pl"),
            bind_group_layouts: &[&build_table_bucket_count_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_bucket_count_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-count-pipeline"),
            layout: Some(&build_table_bucket_count_layout),
            module: &build_table_bucket_count_shader,
            entry_point: "main",
        });
    let build_table_bucket_prefix_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-prefix-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_bucket_prefix_shader =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-prefix-shader"),
            source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_BUCKET_PREFIX_SHADER.into()),
        });
    let build_table_bucket_prefix_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-prefix-pl"),
            bind_group_layouts: &[&build_table_bucket_prefix_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_bucket_prefix_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-prefix-pipeline"),
            layout: Some(&build_table_bucket_prefix_layout),
            module: &build_table_bucket_prefix_shader,
            entry_point: "main",
        });

    let build_table_bucket_scatter_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-scatter-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_bucket_scatter_shader =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-scatter-shader"),
            source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_BUCKET_SCATTER_SHADER.into()),
        });
    let build_table_bucket_scatter_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-scatter-pl"),
            bind_group_layouts: &[&build_table_bucket_scatter_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_bucket_scatter_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-bucket-scatter-pipeline"),
            layout: Some(&build_table_bucket_scatter_layout),
            module: &build_table_bucket_scatter_shader,
            entry_point: "main",
        });
    let build_table_finalize_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-build-table-finalize-shader"),
        source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_FINALIZE_SHADER.into()),
    });
    let build_table_finalize_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-pl"),
            bind_group_layouts: &[&build_table_finalize_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_finalize_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-finalize-pipeline"),
            layout: Some(&build_table_finalize_layout),
            module: &build_table_finalize_shader,
            entry_point: "main",
        });
    let build_table_pack_index_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-pack-index-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let build_table_pack_index_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-build-table-pack-index-shader"),
        source: wgpu::ShaderSource::Wgsl(BUILD_TABLE_PACK_INDEX_SHADER.into()),
    });
    let build_table_pack_index_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-build-table-pack-index-pl"),
            bind_group_layouts: &[&build_table_pack_index_bind_group_layout],
            push_constant_ranges: &[],
        });
    let build_table_pack_index_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-build-table-pack-index-pipeline"),
            layout: Some(&build_table_pack_index_layout),
            module: &build_table_pack_index_shader,
            entry_point: "main",
        });
    let match_prepare_table_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-match-prepare-table-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let match_prepare_table_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-match-prepare-table-shader"),
        source: wgpu::ShaderSource::Wgsl(MATCH_PREPARE_TABLE_SHADER.into()),
    });
    let match_prepare_table_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-match-prepare-table-pl"),
            bind_group_layouts: &[&match_prepare_table_bind_group_layout],
            push_constant_ranges: &[],
        });
    let match_prepare_table_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-match-prepare-table-pipeline"),
            layout: Some(&match_prepare_table_layout),
            module: &match_prepare_table_shader,
            entry_point: "main",
        });
    let max_storage_binding_size = u64::from(device.limits().max_storage_buffer_binding_size);

    Ok(GpuMatchRuntime {
        device,
        queue,
        bind_group_layout,
        pipeline,
        pack_chunk_bind_group_layout,
        pack_chunk_pipeline,
        pack_sparse_prepare_bind_group_layout,
        pack_sparse_prepare_pipeline,
        pack_sparse_bind_group_layout,
        pack_sparse_pipeline,
        section_encode_bind_group_layout,
        section_encode_pipeline,
        section_meta_bind_group_layout,
        section_meta_pipeline,
        build_table_freq_bind_group_layout,
        build_table_freq_pipeline,
        build_table_candidate_bind_group_layout,
        build_table_candidate_pipeline,
        build_table_bucket_count_bind_group_layout,
        build_table_bucket_count_pipeline,
        build_table_bucket_prefix_bind_group_layout,
        build_table_bucket_prefix_pipeline,
        build_table_bucket_scatter_bind_group_layout,
        build_table_bucket_scatter_pipeline,
        build_table_finalize_bind_group_layout,
        build_table_finalize_pipeline,
        build_table_pack_index_bind_group_layout,
        build_table_pack_index_pipeline,
        match_prepare_table_bind_group_layout,
        match_prepare_table_pipeline,
        max_storage_binding_size,
        scratch_pool: Mutex::new(Vec::new()),
        sparse_pack_pool: Mutex::new(Vec::new()),
    })
}

fn runtime() -> Result<&'static GpuMatchRuntime, PDeflateError> {
    match GPU_RUNTIME.get_or_init(init_runtime) {
        Ok(r) => Ok(r),
        Err(e) => Err(PDeflateError::Gpu(e.clone())),
    }
}

pub(crate) fn lock_submit_stream() -> Result<std::sync::MutexGuard<'static, ()>, PDeflateError> {
    GPU_SUBMIT_STREAM_MUTEX
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| PDeflateError::Gpu("gpu submit stream mutex poisoned".to_string()))
}

fn pack_bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    if bytes.is_empty() {
        return Vec::new();
    }
    let mut words = vec![0u32; bytes.len().div_ceil(4)];
    for (i, &b) in bytes.iter().enumerate() {
        let wi = i >> 2;
        let shift = (i & 3) << 3;
        words[wi] |= u32::from(b) << shift;
    }
    words
}

fn readback_table_sizes_from_meta_buffer(
    runtime: &GpuMatchRuntime,
    table_meta_buffer: &wgpu::Buffer,
    max_entries: usize,
    table_data_bytes_cap: usize,
) -> Result<(usize, usize, usize), PDeflateError> {
    if max_entries == 0 {
        return Ok((0, 0, 0));
    }
    let data_total_idx = 1usize
        .checked_add(
            max_entries
                .checked_mul(2)
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .ok_or(PDeflateError::NumericOverflow)?;
    let readback_sizes = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-readback-sizes"),
        size: 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-bt-size-readback-encoder"),
        });
    encoder.copy_buffer_to_buffer(table_meta_buffer, 0, &readback_sizes, 0, 4);
    encoder.copy_buffer_to_buffer(
        table_meta_buffer,
        u64::try_from(data_total_idx)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        &readback_sizes,
        4,
        4,
    );
    let submission = runtime.queue.submit(Some(encoder.finish()));

    let size_slice = readback_sizes.slice(..8);
    let (tx_size, rx_size) = mpsc::channel();
    size_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_size.send(res);
    });
    runtime.device.poll(wgpu::Maintain::wait_for(submission));
    rx_size
        .recv()
        .map_err(|_| PDeflateError::Gpu("gpu build-table size map channel closed".to_string()))?
        .map_err(|e| PDeflateError::Gpu(format!("gpu build-table size map failed: {e}")))?;
    let mapped_sizes = size_slice.get_mapped_range();
    let size_words: &[u32] = bytemuck::cast_slice(&mapped_sizes);
    let out_count = usize::try_from(*size_words.first().unwrap_or(&0))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .min(max_entries);
    let data_total = usize::try_from(*size_words.get(1).unwrap_or(&0))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .min(table_data_bytes_cap);
    drop(mapped_sizes);
    readback_sizes.unmap();
    Ok((out_count, out_count, data_total))
}

fn write_u16_le_fixed(dst: &mut [u8], off: usize, v: u16) {
    let b = v.to_le_bytes();
    dst[off..off + 2].copy_from_slice(&b);
}

fn write_u32_le_fixed(dst: &mut [u8], off: usize, v: u32) {
    let b = v.to_le_bytes();
    dst[off..off + 4].copy_from_slice(&b);
}

pub(crate) fn pack_chunk_payload(
    chunk_len: usize,
    table_count: usize,
    section_count: usize,
    table_index: &[u8],
    table_data: &[u8],
    section_index: &[u8],
    section_cmd: &[u8],
) -> Result<(Vec<u8>, GpuMatchProfile), PDeflateError> {
    let r = runtime()?;
    let t_total = Instant::now();

    let table_index_offset = PACK_CHUNK_HEADER_SIZE;
    let table_data_offset = table_index_offset
        .checked_add(table_index.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let section_index_offset = table_data_offset
        .checked_add(table_data.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let section_cmd_offset = section_index_offset
        .checked_add(section_index.len())
        .ok_or(PDeflateError::NumericOverflow)?;
    let total_len = section_cmd_offset
        .checked_add(section_cmd.len())
        .ok_or(PDeflateError::NumericOverflow)?;

    let table_count_u16 = u16::try_from(table_count).map_err(|_| PDeflateError::NumericOverflow)?;
    let section_count_u16 =
        u16::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
    let chunk_len_u32 = u32::try_from(chunk_len).map_err(|_| PDeflateError::NumericOverflow)?;
    let table_index_offset_u32 =
        u32::try_from(table_index_offset).map_err(|_| PDeflateError::NumericOverflow)?;
    let table_data_offset_u32 =
        u32::try_from(table_data_offset).map_err(|_| PDeflateError::NumericOverflow)?;
    let section_index_offset_u32 =
        u32::try_from(section_index_offset).map_err(|_| PDeflateError::NumericOverflow)?;
    let section_cmd_offset_u32 =
        u32::try_from(section_cmd_offset).map_err(|_| PDeflateError::NumericOverflow)?;

    let mut header = [0u8; PACK_CHUNK_HEADER_SIZE];
    header[0..4].copy_from_slice(&PACK_CHUNK_MAGIC);
    write_u16_le_fixed(&mut header, 4, PACK_CHUNK_VERSION);
    write_u16_le_fixed(&mut header, 6, 0);
    write_u32_le_fixed(&mut header, 8, chunk_len_u32);
    write_u16_le_fixed(&mut header, 12, table_count_u16);
    write_u16_le_fixed(&mut header, 14, section_count_u16);
    write_u32_le_fixed(&mut header, 16, table_index_offset_u32);
    write_u32_le_fixed(&mut header, 20, table_data_offset_u32);
    write_u32_le_fixed(&mut header, 24, section_index_offset_u32);
    write_u32_le_fixed(&mut header, 28, section_cmd_offset_u32);

    let header_words = pack_bytes_to_words(&header);
    let table_index_words = pack_bytes_to_words(table_index);
    let table_data_words = pack_bytes_to_words(table_data);
    let section_index_words = pack_bytes_to_words(section_index);
    let section_cmd_words = pack_bytes_to_words(section_cmd);

    let total_words = total_len.div_ceil(4);
    let total_words_u32 = u32::try_from(total_words).map_err(|_| PDeflateError::NumericOverflow)?;
    let total_len_u32 = u32::try_from(total_len).map_err(|_| PDeflateError::NumericOverflow)?;
    let params: [u32; 11] = [
        total_len_u32,
        u32::try_from(PACK_CHUNK_HEADER_SIZE).map_err(|_| PDeflateError::NumericOverflow)?,
        table_index_offset_u32,
        u32::try_from(table_index.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        table_data_offset_u32,
        u32::try_from(table_data.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        section_index_offset_u32,
        u32::try_from(section_index.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        section_cmd_offset_u32,
        u32::try_from(section_cmd.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        total_words_u32,
    ];

    let total_bytes = u64::try_from(total_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4)
        .max(4);
    if total_bytes > r.max_storage_binding_size {
        return Err(PDeflateError::Gpu(format!(
            "gpu chunk pack output too large ({} > {})",
            total_bytes, r.max_storage_binding_size
        )));
    }

    let header_bytes = u64::try_from(header_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_index_bytes = u64::try_from(table_index_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_data_bytes = u64::try_from(table_data_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let section_index_bytes = u64::try_from(section_index_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let section_cmd_bytes = u64::try_from(section_cmd_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let params_bytes = u64::try_from(params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);

    let params_buffer =
        storage_upload_buffer(&r.device, "cozip-pdeflate-pack-params", params_bytes);
    let header_buffer =
        storage_upload_buffer(&r.device, "cozip-pdeflate-pack-header", header_bytes);
    let table_index_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-pack-table-index",
        table_index_bytes,
    );
    let table_data_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-pack-table-data",
        table_data_bytes,
    );
    let section_index_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-pack-section-index",
        section_index_bytes,
    );
    let section_cmd_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-pack-section-cmd",
        section_cmd_bytes,
    );
    let out_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-pack-out"),
        size: total_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-pack-readback"),
        size: total_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-pack-bg"),
        layout: &r.pack_chunk_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: header_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: table_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: table_data_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: section_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: section_cmd_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: out_buffer.as_entire_binding(),
            },
        ],
    });

    let t_upload = Instant::now();
    r.queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));
    r.queue
        .write_buffer(&header_buffer, 0, bytemuck::cast_slice(&header_words));
    r.queue.write_buffer(
        &table_index_buffer,
        0,
        bytemuck::cast_slice(&table_index_words),
    );
    r.queue.write_buffer(
        &table_data_buffer,
        0,
        bytemuck::cast_slice(&table_data_words),
    );
    r.queue.write_buffer(
        &section_index_buffer,
        0,
        bytemuck::cast_slice(&section_index_words),
    );
    r.queue.write_buffer(
        &section_cmd_buffer,
        0,
        bytemuck::cast_slice(&section_cmd_words),
    );
    let upload_ms = elapsed_ms(t_upload);

    let groups_x = total_words_u32.div_ceil(256).max(1);
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-pack-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-pack-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.pack_chunk_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buffer, 0, &readback_buffer, 0, total_bytes);
    let submission = r.queue.submit(Some(encoder.finish()));

    let t_wait = Instant::now();
    let slice = readback_buffer.slice(..total_bytes);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    r.device.poll(wgpu::Maintain::wait_for(submission));
    rx.recv()
        .map_err(|_| PDeflateError::Gpu("gpu chunk pack map channel closed".to_string()))?
        .map_err(|e| PDeflateError::Gpu(format!("gpu chunk pack map failed: {e}")))?;
    let wait_ms = elapsed_ms(t_wait);

    let t_copy = Instant::now();
    let mapped = slice.get_mapped_range();
    let mut out = vec![0u8; total_len];
    out.copy_from_slice(&mapped[..total_len]);
    drop(mapped);
    readback_buffer.unmap();
    let map_copy_ms = elapsed_ms(t_copy);

    Ok((
        out,
        GpuMatchProfile {
            upload_ms,
            wait_ms,
            map_copy_ms,
            total_ms: elapsed_ms(t_total),
        },
    ))
}

fn write_varint_u32(dst: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        dst.push(((v & 0x7f) as u8) | 0x80);
        v >>= 7;
    }
    dst.push((v & 0x7f) as u8);
}

pub(crate) fn pack_chunk_payload_from_device_sparse(
    chunk_len: usize,
    section_count: usize,
    table: &GpuPackedTableDevice,
    section: &GpuSectionCmdDeviceOutput,
) -> Result<(Vec<u8>, usize, GpuMatchProfile), PDeflateError> {
    let inputs = [GpuSparsePackInput {
        chunk_len,
        section_count,
        table,
        section,
    }];
    let (mut packed, profile, _) = pack_chunk_payload_from_device_sparse_batch(&inputs)?;
    let first = packed.drain(..).next().ok_or(PDeflateError::Gpu(
        "gpu sparse pack single result missing".to_string(),
    ))?;
    Ok((first.payload, first.table_count, profile))
}

pub(crate) fn pack_chunk_payload_from_device_sparse_batch(
    inputs: &[GpuSparsePackInput<'_>],
) -> Result<
    (
        Vec<GpuSparsePackChunkOutput>,
        GpuMatchProfile,
        GpuSparsePackBatchProfile,
    ),
    PDeflateError,
> {
    if inputs.is_empty() {
        return Ok((
            Vec::new(),
            GpuMatchProfile::default(),
            GpuSparsePackBatchProfile::default(),
        ));
    }
    for input in inputs {
        if input.section.section_count != input.section_count {
            return Err(PDeflateError::InvalidOptions(
                "gpu sparse pack section_count mismatch",
            ));
        }
    }

    struct SparsePackPrepared {
        chunk_len: usize,
        section_count: usize,
        table_index_offset: usize,
        table_data_offset: usize,
        section_index_offset: usize,
        total_len_cap: usize,
        total_bytes_cap: u64,
        table_count: usize,
        table_index_len: usize,
        table_data_len: usize,
    }

    struct SparsePackDispatchJob {
        table_count: usize,
        scratch: GpuSparsePackScratch,
    }

    let r = runtime()?;
    let t_total = Instant::now();
    let mut upload_ms = 0.0_f64;
    let mut wait_ms = 0.0_f64;
    let mut map_copy_ms = 0.0_f64;
    let lens_bytes_total = inputs.iter().fold(0u64, |acc, input| {
        acc.saturating_add(input.section.out_lens_bytes.max(4))
    });
    let out_cmd_bytes_total = inputs.iter().fold(0u64, |acc, input| {
        acc.saturating_add(input.section.out_cmd_bytes.max(4))
    });

    let lens_submit_ms = 0.0_f64;
    let lens_wait_ms;
    let lens_copy_ms;

    let t_prepare = Instant::now();
    let mut prepared = Vec::<SparsePackPrepared>::with_capacity(inputs.len());
    for input in inputs {
        let (table_count, table_index_len, table_data_len) =
            resolve_packed_table_sizes(input.table)?;
        let table_index_offset = PACK_CHUNK_HEADER_SIZE;
        let table_data_offset = table_index_offset
            .checked_add(table_index_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        let section_index_offset = table_data_offset
            .checked_add(table_data_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        let section_index_cap_len = usize::try_from(input.section.section_index_cap_bytes)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let section_cmd_cap_len = usize::try_from(input.section.out_cmd_bytes.max(4))
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let section_cmd_offset_cap = section_index_offset
            .checked_add(section_index_cap_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        let total_len_cap = section_cmd_offset_cap
            .checked_add(section_cmd_cap_len)
            .ok_or(PDeflateError::NumericOverflow)?;

        let total_words = total_len_cap.div_ceil(4);
        let total_bytes = u64::try_from(total_words)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4)
            .max(4);
        if total_bytes > r.max_storage_binding_size {
            return Err(PDeflateError::Gpu(format!(
                "gpu sparse chunk pack output too large ({} > {})",
                total_bytes, r.max_storage_binding_size
            )));
        }

        prepared.push(SparsePackPrepared {
            chunk_len: input.chunk_len,
            section_count: input.section_count,
            table_index_offset,
            table_data_offset,
            section_index_offset,
            total_len_cap,
            total_bytes_cap: total_bytes,
            table_count,
            table_index_len,
            table_data_len,
        });
    }
    let prepare_ms = elapsed_ms(t_prepare);

    let t_scratch_acquire = Instant::now();
    let mut acquired_scratch = Vec::<GpuSparsePackScratch>::with_capacity(prepared.len());
    {
        let mut pool = r
            .sparse_pack_pool
            .lock()
            .map_err(|_| PDeflateError::Gpu("gpu sparse pack pool mutex poisoned".to_string()))?;
        for (_input, prep) in inputs.iter().zip(prepared.iter()) {
            let caps = GpuSparsePackScratchCaps {
                params_bytes: 48,
                out_bytes: prep.total_bytes_cap,
            };
            if let Some(pos) = pool.iter().position(|s| s.fits(caps)) {
                acquired_scratch.push(pool.swap_remove(pos));
            } else {
                acquired_scratch.push(GpuSparsePackScratch::new(r, caps));
            }
        }
    }
    let scratch_acquire_ms = elapsed_ms(t_scratch_acquire);

    let t_upload = Instant::now();
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-batch-encoder"),
        });
    let mut dispatch_jobs = Vec::<SparsePackDispatchJob>::with_capacity(inputs.len());
    for ((input, prep), scratch) in inputs
        .iter()
        .zip(prepared.iter())
        .zip(acquired_scratch.into_iter())
    {
        let section_index_cap_len = usize::try_from(input.section.section_index_cap_bytes)
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let section_cmd_cap_len = usize::try_from(input.section.out_cmd_bytes.max(4))
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let total_words_cap =
            u32::try_from(prep.total_bytes_cap / 4).map_err(|_| PDeflateError::NumericOverflow)?;
        let params_cap = [
            u32::try_from(prep.chunk_len).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(prep.table_count).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(prep.section_count).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(prep.table_index_offset).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(prep.table_index_len).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(prep.table_data_offset).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(prep.table_data_len).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(prep.section_index_offset).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(section_index_cap_len).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(section_cmd_cap_len).map_err(|_| PDeflateError::NumericOverflow)?,
            u32::try_from(prep.total_len_cap).map_err(|_| PDeflateError::NumericOverflow)?,
            total_words_cap,
        ];

        let prepare_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-prepare-bg"),
            layout: &r.pack_sparse_prepare_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scratch.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.section.section_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scratch.dispatch_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-bg"),
            layout: &r.pack_sparse_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scratch.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.section.section_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input.table.table_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input.table.table_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: input.section.section_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: input.section.out_lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: input.section.section_prefix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: input.section.section_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: input.section.out_cmd_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: scratch.out_buffer.as_entire_binding(),
                },
            ],
        });

        r.queue
            .write_buffer(&scratch.params_buffer, 0, bytemuck::cast_slice(&params_cap));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-pack-sparse-prepare-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.pack_sparse_prepare_pipeline);
            pass.set_bind_group(0, &prepare_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &scratch.params_buffer,
            0,
            &scratch.params_readback_buffer,
            0,
            48,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-pack-sparse-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.pack_sparse_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups_indirect(&scratch.dispatch_buffer, 0);
        }
        encoder.copy_buffer_to_buffer(
            &scratch.out_buffer,
            0,
            &scratch.readback_buffer,
            0,
            prep.total_bytes_cap,
        );
        dispatch_jobs.push(SparsePackDispatchJob {
            table_count: prep.table_count,
            scratch,
        });
    }
    let t_submit = Instant::now();
    r.queue.submit(Some(encoder.finish()));
    let submit_ms = elapsed_ms(t_submit);
    let upload_dispatch_ms = (elapsed_ms(t_upload) - submit_ms).max(0.0);
    upload_ms += upload_dispatch_ms;

    let t_lens_wait = Instant::now();
    let lens_submit_done_wait_ms = 0.0_f64;
    let mut lens_poll_calls = 0u64;
    let lens_yield_calls = 0u64;
    let mut size_slices = Vec::with_capacity(dispatch_jobs.len());
    let mut size_receivers = Vec::with_capacity(dispatch_jobs.len());
    let mut out_slices = Vec::with_capacity(dispatch_jobs.len());
    let mut out_receivers = Vec::with_capacity(dispatch_jobs.len());
    for dispatch in &dispatch_jobs {
        let size_slice = dispatch.scratch.params_readback_buffer.slice(..48);
        let (tx, rx) = mpsc::channel();
        size_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        size_slices.push(size_slice);
        size_receivers.push(rx);

        let out_slice = dispatch
            .scratch
            .readback_buffer
            .slice(..dispatch.scratch.out_cap_bytes.max(4));
        let (out_tx, out_rx) = mpsc::channel();
        out_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = out_tx.send(res);
        });
        out_slices.push(out_slice);
        out_receivers.push(out_rx);
    }
    let mut pending_sizes = vec![true; size_receivers.len()];
    let mut remaining_sizes = size_receivers.len();
    let mut pending_out = vec![true; out_receivers.len()];
    let mut remaining_out = out_receivers.len();
    while remaining_sizes > 0 || remaining_out > 0 {
        r.device.poll(wgpu::Maintain::Wait);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        for (i, rx) in size_receivers.iter().enumerate() {
            if pending_sizes[i] {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse size map failed: {e}"))
                        })?;
                        pending_sizes[i] = false;
                        remaining_sizes = remaining_sizes.saturating_sub(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(PDeflateError::Gpu(
                            "gpu sparse size map channel closed".to_string(),
                        ));
                    }
                }
            }
        }
        for (i, rx) in out_receivers.iter().enumerate() {
            if pending_out[i] {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse chunk pack map failed: {e}"))
                        })?;
                        pending_out[i] = false;
                        remaining_out = remaining_out.saturating_sub(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(PDeflateError::Gpu(
                            "gpu sparse chunk pack map channel closed".to_string(),
                        ));
                    }
                }
            }
        }
    }
    lens_wait_ms = elapsed_ms(t_lens_wait);
    let lens_map_after_done_ms = 0.0_f64;

    let t_lens_copy = Instant::now();
    let mut payload_lens = Vec::<usize>::with_capacity(dispatch_jobs.len());
    for (idx, prep) in prepared.iter().enumerate() {
        let size_mapped = size_slices[idx].get_mapped_range();
        let size_words: &[u32] = bytemuck::cast_slice(&size_mapped);
        let total_len_u32 = *size_words.get(10).unwrap_or(&0xffff_ffff);
        let total_words_u32 = *size_words.get(11).unwrap_or(&0);
        drop(size_mapped);
        dispatch_jobs[idx].scratch.params_readback_buffer.unmap();

        if total_len_u32 == 0xffff_ffff || total_words_u32 == 0 {
            return Err(PDeflateError::Gpu(
                "gpu sparse size prepare overflow while packing".to_string(),
            ));
        }

        let total_len =
            usize::try_from(total_len_u32).map_err(|_| PDeflateError::NumericOverflow)?;
        let total_copy_len = align_up4(total_len);
        let total_copy_bytes =
            u64::try_from(total_copy_len).map_err(|_| PDeflateError::NumericOverflow)?;
        if total_copy_bytes > prep.total_bytes_cap {
            return Err(PDeflateError::Gpu(
                "gpu sparse pack produced oversized aligned payload".to_string(),
            ));
        }
        payload_lens.push(total_len);
    }
    lens_copy_ms = elapsed_ms(t_lens_copy);
    let sparse_wait_ms = 0.0_f64;
    wait_ms += sparse_wait_ms;

    let t_copy = Instant::now();
    let mut out = Vec::<GpuSparsePackChunkOutput>::with_capacity(dispatch_jobs.len());
    for (idx, dispatch) in dispatch_jobs.iter().enumerate() {
        let mapped = out_slices[idx].get_mapped_range();
        let payload_len = payload_lens[idx];
        let mut payload = vec![0u8; payload_len];
        payload.copy_from_slice(&mapped[..payload_len]);
        drop(mapped);
        dispatch.scratch.readback_buffer.unmap();
        out.push(GpuSparsePackChunkOutput {
            payload,
            table_count: dispatch.table_count,
        });
    }
    let sparse_copy_ms = elapsed_ms(t_copy);
    map_copy_ms += lens_copy_ms + sparse_copy_ms;

    {
        let mut pool = r
            .sparse_pack_pool
            .lock()
            .map_err(|_| PDeflateError::Gpu("gpu sparse pack pool mutex poisoned".to_string()))?;
        for job in dispatch_jobs.drain(..) {
            if pool.len() < GPU_SPARSE_PACK_POOL_LIMIT {
                pool.push(job.scratch);
            }
        }
    }

    let total_ms = elapsed_ms(t_total);
    Ok((
        out,
        GpuMatchProfile {
            upload_ms,
            wait_ms,
            map_copy_ms,
            total_ms,
        },
        GpuSparsePackBatchProfile {
            chunks: inputs.len(),
            lens_bytes_total,
            out_cmd_bytes_total,
            lens_submit_ms,
            lens_submit_done_wait_ms,
            lens_map_after_done_ms,
            lens_poll_calls,
            lens_yield_calls,
            lens_wait_ms,
            lens_copy_ms,
            prepare_ms,
            scratch_acquire_ms,
            upload_dispatch_ms,
            submit_ms,
            wait_ms: sparse_wait_ms,
            copy_ms: sparse_copy_ms,
            total_ms,
        },
    ))
}

pub(crate) fn read_section_commands_from_device_batch(
    inputs: &[GpuSparsePackInput<'_>],
) -> Result<
    (
        Vec<GpuSparseSectionHostOutput>,
        GpuMatchProfile,
        GpuSparsePackBatchProfile,
    ),
    PDeflateError,
> {
    if inputs.is_empty() {
        return Ok((
            Vec::new(),
            GpuMatchProfile::default(),
            GpuSparsePackBatchProfile::default(),
        ));
    }
    for input in inputs {
        if input.section.section_count != input.section_count {
            return Err(PDeflateError::InvalidOptions(
                "gpu sparse readback section_count mismatch",
            ));
        }
    }

    struct ReadPrepared {
        section_offsets: Vec<usize>,
        out_lens_bytes: u64,
        out_cmd_bytes: u64,
    }
    struct LensReadbackJob {
        lens_buffer: wgpu::Buffer,
    }
    struct LensHostChunk {
        section_cmd_lens: Vec<u32>,
        copy_offsets: Vec<usize>,
        total_cmd_len: usize,
        total_copy_bytes: u64,
    }
    struct CmdReadbackJob {
        cmd_buffer: wgpu::Buffer,
    }

    let r = runtime()?;
    let t_total = Instant::now();
    let lens_bytes_total = inputs.iter().fold(0u64, |acc, input| {
        acc.saturating_add(input.section.out_lens_bytes.max(4))
    });

    let t_prepare = Instant::now();
    let mut prepared = Vec::<ReadPrepared>::with_capacity(inputs.len());
    for input in inputs {
        let min_lens_bytes = u64::try_from(input.section_count)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        if input.section.out_lens_bytes < min_lens_bytes {
            return Err(PDeflateError::Gpu(
                "gpu sparse readback lens buffer too small".to_string(),
            ));
        }

        let mut section_offsets = Vec::with_capacity(input.section_count);
        let mut total_cmd_cap_bytes = 0usize;
        for sec in 0..input.section_count {
            let s0 = section_start(sec, input.section_count, input.chunk_len);
            let s1 = section_start(sec + 1, input.section_count, input.chunk_len);
            let sec_len = s1.saturating_sub(s0);
            let cap_bytes = estimate_section_cmd_cap_bytes(sec_len)?;
            let aligned = align_up4(total_cmd_cap_bytes);
            section_offsets.push(aligned);
            total_cmd_cap_bytes = aligned
                .checked_add(cap_bytes)
                .ok_or(PDeflateError::NumericOverflow)?;
        }
        let expected_cmd_bytes = u64::try_from(align_up4(total_cmd_cap_bytes))
            .map_err(|_| PDeflateError::NumericOverflow)?;
        if expected_cmd_bytes > input.section.out_cmd_bytes.max(4) {
            return Err(PDeflateError::Gpu(
                "gpu sparse readback cmd buffer too small".to_string(),
            ));
        }
        prepared.push(ReadPrepared {
            section_offsets,
            out_lens_bytes: input.section.out_lens_bytes.max(4),
            out_cmd_bytes: input.section.out_cmd_bytes.max(4),
        });
    }
    let prepare_ms = elapsed_ms(t_prepare);

    let t_lens_upload = Instant::now();
    let mut lens_readbacks = Vec::<LensReadbackJob>::with_capacity(inputs.len());
    let mut lens_encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-sparse-lens-readback-encoder"),
        });
    for (input, prep) in inputs.iter().zip(prepared.iter()) {
        let lens_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-sparse-readback-lens"),
            size: prep.out_lens_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        lens_encoder.copy_buffer_to_buffer(
            &input.section.out_lens_buffer,
            0,
            &lens_buffer,
            0,
            prep.out_lens_bytes,
        );
        lens_readbacks.push(LensReadbackJob { lens_buffer });
    }
    let t_lens_submit = Instant::now();
    r.queue.submit(Some(lens_encoder.finish()));
    let lens_submit_ms = elapsed_ms(t_lens_submit);
    let lens_upload_dispatch_ms = (elapsed_ms(t_lens_upload) - lens_submit_ms).max(0.0);

    let t_lens_wait = Instant::now();
    let (done_tx_lens, done_rx_lens) = mpsc::channel();
    r.queue.on_submitted_work_done(move || {
        let _ = done_tx_lens.send(());
    });

    let mut lens_rxs = Vec::with_capacity(lens_readbacks.len());
    for (rb, prep) in lens_readbacks.iter().zip(prepared.iter()) {
        let lens_slice = rb.lens_buffer.slice(..prep.out_lens_bytes);
        let (tx_lens, rx_lens) = mpsc::channel();
        lens_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx_lens.send(res);
        });
        lens_rxs.push(rx_lens);
    }

    let mut lens_submit_done_wait_ms = 0.0_f64;
    let mut lens_submit_done_seen = false;
    let mut lens_poll_calls = 0u64;
    let lens_yield_calls = 0u64;
    let mut pending_lens = vec![true; lens_rxs.len()];
    let mut remaining_lens = lens_rxs.len();
    while remaining_lens > 0 || !lens_submit_done_seen {
        r.device.poll(wgpu::Maintain::Wait);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        if !lens_submit_done_seen {
            match done_rx_lens.try_recv() {
                Ok(_) => {
                    lens_submit_done_seen = true;
                    lens_submit_done_wait_ms = elapsed_ms(t_lens_wait);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    lens_submit_done_seen = true;
                    lens_submit_done_wait_ms = elapsed_ms(t_lens_wait);
                }
            }
        }
        for (i, rx) in lens_rxs.iter().enumerate() {
            if pending_lens[i] {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse lens readback map failed: {e}"))
                        })?;
                        pending_lens[i] = false;
                        remaining_lens = remaining_lens.saturating_sub(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(PDeflateError::Gpu(
                            "gpu sparse lens readback channel closed".to_string(),
                        ));
                    }
                }
            }
        }
    }
    let lens_wait_ms = elapsed_ms(t_lens_wait);
    let lens_map_after_done_ms = (lens_wait_ms - lens_submit_done_wait_ms).max(0.0);

    let t_lens_copy = Instant::now();
    let mut lens_host = Vec::<LensHostChunk>::with_capacity(inputs.len());
    let mut copied_cmd_bytes_total = 0u64;
    for (idx, (input, prep)) in inputs.iter().zip(prepared.iter()).enumerate() {
        let lens_slice = lens_readbacks[idx].lens_buffer.slice(..prep.out_lens_bytes);
        let lens_mapped = lens_slice.get_mapped_range();
        let lens_words: &[u32] = bytemuck::cast_slice(&lens_mapped);
        if lens_words.len() < input.section_count {
            return Err(PDeflateError::Gpu(
                "gpu sparse lens readback truncated".to_string(),
            ));
        }
        let mut section_cmd_lens = vec![0u32; input.section_count];
        section_cmd_lens.copy_from_slice(&lens_words[..input.section_count]);

        let mut total_cmd_len = 0usize;
        let mut copy_offsets = Vec::with_capacity(input.section_count);
        let mut copy_cursor = 0usize;
        for &len_u32 in &section_cmd_lens {
            if len_u32 == 0xffff_ffff {
                return Err(PDeflateError::Gpu(
                    "gpu sparse lens overflow while rebuilding host section commands".to_string(),
                ));
            }
            let len = usize::try_from(len_u32).map_err(|_| PDeflateError::NumericOverflow)?;
            total_cmd_len = total_cmd_len
                .checked_add(len)
                .ok_or(PDeflateError::NumericOverflow)?;
            copy_offsets.push(copy_cursor);
            if len > 0 {
                copy_cursor = copy_cursor
                    .checked_add(align_up4(len))
                    .ok_or(PDeflateError::NumericOverflow)?;
            }
        }
        let total_copy_bytes =
            u64::try_from(copy_cursor.max(4)).map_err(|_| PDeflateError::NumericOverflow)?;
        if total_copy_bytes > prep.out_cmd_bytes {
            return Err(PDeflateError::Gpu(
                "gpu sparse cmd compact copy size exceeds cap".to_string(),
            ));
        }
        copied_cmd_bytes_total = copied_cmd_bytes_total.saturating_add(total_copy_bytes);
        drop(lens_mapped);
        lens_readbacks[idx].lens_buffer.unmap();
        lens_host.push(LensHostChunk {
            section_cmd_lens,
            copy_offsets,
            total_cmd_len,
            total_copy_bytes,
        });
    }
    let lens_copy_ms = elapsed_ms(t_lens_copy);

    let t_cmd_upload = Instant::now();
    let mut cmd_readbacks = Vec::<CmdReadbackJob>::with_capacity(inputs.len());
    let mut cmd_encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-sparse-cmd-readback-encoder"),
        });
    for ((input, prep), lens) in inputs.iter().zip(prepared.iter()).zip(lens_host.iter()) {
        let cmd_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-sparse-readback-cmd-compact"),
            size: lens.total_copy_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for sec in 0..input.section_count {
            let len = usize::try_from(lens.section_cmd_lens[sec])
                .map_err(|_| PDeflateError::NumericOverflow)?;
            if len == 0 {
                continue;
            }
            let src_off = u64::try_from(prep.section_offsets[sec])
                .map_err(|_| PDeflateError::NumericOverflow)?;
            let dst_off = u64::try_from(lens.copy_offsets[sec])
                .map_err(|_| PDeflateError::NumericOverflow)?;
            let copy_bytes =
                u64::try_from(align_up4(len)).map_err(|_| PDeflateError::NumericOverflow)?;
            if src_off.saturating_add(copy_bytes) > prep.out_cmd_bytes
                || dst_off.saturating_add(copy_bytes) > lens.total_copy_bytes
            {
                return Err(PDeflateError::Gpu(
                    "gpu sparse cmd compact copy range out of bounds".to_string(),
                ));
            }
            cmd_encoder.copy_buffer_to_buffer(
                &input.section.out_cmd_buffer,
                src_off,
                &cmd_buffer,
                dst_off,
                copy_bytes,
            );
        }
        cmd_readbacks.push(CmdReadbackJob { cmd_buffer });
    }
    let t_cmd_submit = Instant::now();
    r.queue.submit(Some(cmd_encoder.finish()));
    let cmd_submit_ms = elapsed_ms(t_cmd_submit);
    let cmd_upload_dispatch_ms = (elapsed_ms(t_cmd_upload) - cmd_submit_ms).max(0.0);

    let t_cmd_wait = Instant::now();
    let (done_tx_cmd, done_rx_cmd) = mpsc::channel();
    r.queue.on_submitted_work_done(move || {
        let _ = done_tx_cmd.send(());
    });
    let mut cmd_rxs = Vec::with_capacity(cmd_readbacks.len());
    for (rb, lens) in cmd_readbacks.iter().zip(lens_host.iter()) {
        let cmd_slice = rb.cmd_buffer.slice(..lens.total_copy_bytes);
        let (tx_cmd, rx_cmd) = mpsc::channel();
        cmd_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx_cmd.send(res);
        });
        cmd_rxs.push(rx_cmd);
    }
    let mut cmd_done_seen = false;
    let mut cmd_poll_calls = 0u64;
    let mut cmd_pending = vec![true; cmd_rxs.len()];
    let mut cmd_remaining = cmd_rxs.len();
    while cmd_remaining > 0 || !cmd_done_seen {
        r.device.poll(wgpu::Maintain::Wait);
        cmd_poll_calls = cmd_poll_calls.saturating_add(1);
        if !cmd_done_seen {
            match done_rx_cmd.try_recv() {
                Ok(_) | Err(mpsc::TryRecvError::Disconnected) => cmd_done_seen = true,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        for (i, rx) in cmd_rxs.iter().enumerate() {
            if cmd_pending[i] {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse compact cmd map failed: {e}"))
                        })?;
                        cmd_pending[i] = false;
                        cmd_remaining = cmd_remaining.saturating_sub(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(PDeflateError::Gpu(
                            "gpu sparse compact cmd readback channel closed".to_string(),
                        ));
                    }
                }
            }
        }
    }
    let cmd_wait_ms = elapsed_ms(t_cmd_wait);

    let t_cmd_copy = Instant::now();
    let mut out = Vec::<GpuSparseSectionHostOutput>::with_capacity(inputs.len());
    for (idx, lens) in lens_host.iter().enumerate() {
        let cmd_slice = cmd_readbacks[idx].cmd_buffer.slice(..lens.total_copy_bytes);
        let cmd_mapped = cmd_slice.get_mapped_range();
        let compact_bytes: &[u8] = &cmd_mapped;
        let mut section_cmd = Vec::with_capacity(lens.total_cmd_len);
        for sec in 0..lens.section_cmd_lens.len() {
            let len = usize::try_from(lens.section_cmd_lens[sec])
                .map_err(|_| PDeflateError::NumericOverflow)?;
            if len == 0 {
                continue;
            }
            let off = lens.copy_offsets[sec];
            let end = off.checked_add(len).ok_or(PDeflateError::NumericOverflow)?;
            if end > compact_bytes.len() {
                return Err(PDeflateError::Gpu(format!(
                    "gpu sparse compact cmd readback truncated on section {}",
                    sec
                )));
            }
            section_cmd.extend_from_slice(&compact_bytes[off..end]);
        }
        drop(cmd_mapped);
        cmd_readbacks[idx].cmd_buffer.unmap();
        out.push(GpuSparseSectionHostOutput {
            section_cmd_lens: lens.section_cmd_lens.clone(),
            section_cmd,
        });
    }
    let cmd_copy_ms = elapsed_ms(t_cmd_copy);

    let upload_dispatch_ms = lens_upload_dispatch_ms + cmd_upload_dispatch_ms;
    let submit_ms = lens_submit_ms + cmd_submit_ms;
    let wait_ms = lens_wait_ms + cmd_wait_ms;
    let map_copy_ms = lens_copy_ms + cmd_copy_ms;
    let total_ms = elapsed_ms(t_total);
    Ok((
        out,
        GpuMatchProfile {
            upload_ms: upload_dispatch_ms,
            wait_ms,
            map_copy_ms,
            total_ms,
        },
        GpuSparsePackBatchProfile {
            chunks: inputs.len(),
            lens_bytes_total,
            out_cmd_bytes_total: copied_cmd_bytes_total,
            lens_submit_ms,
            lens_submit_done_wait_ms,
            lens_map_after_done_ms,
            lens_poll_calls: lens_poll_calls.saturating_add(cmd_poll_calls),
            lens_yield_calls,
            lens_wait_ms,
            lens_copy_ms,
            prepare_ms,
            scratch_acquire_ms: 0.0,
            upload_dispatch_ms,
            submit_ms,
            wait_ms: cmd_wait_ms,
            copy_ms: cmd_copy_ms,
            total_ms,
        },
    ))
}

pub(crate) fn is_runtime_available() -> bool {
    runtime().is_ok()
}

pub(crate) fn max_safe_match_batch_chunks(chunk_size: usize) -> Result<usize, PDeflateError> {
    if chunk_size == 0 {
        return Ok(1);
    }
    let r = runtime()?;
    let per_chunk_out = chunk_size
        .checked_mul(4)
        .ok_or(PDeflateError::NumericOverflow)?;
    if per_chunk_out == 0 {
        return Ok(1);
    }
    let max_binding =
        usize::try_from(r.max_storage_binding_size).map_err(|_| PDeflateError::NumericOverflow)?;
    Ok((max_binding / per_chunk_out).max(1))
}

pub(crate) fn upload_packed_table_device(
    table_count: usize,
    table_index: &[u8],
    table_data: &[u8],
) -> Result<GpuPackedTableDevice, PDeflateError> {
    let r = runtime()?;
    let index_bytes = u64::try_from(table_index.len().div_ceil(4))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let data_bytes = u64::try_from(table_data.len().div_ceil(4))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_index_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-upload-table-index"),
        size: index_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let table_data_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-upload-table-data"),
        size: data_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    if !table_index.is_empty() {
        let table_index_words = pack_bytes_to_words(table_index);
        r.queue.write_buffer(
            &table_index_buffer,
            0,
            bytemuck::cast_slice(&table_index_words),
        );
    }
    if !table_data.is_empty() {
        let table_data_words = pack_bytes_to_words(table_data);
        r.queue.write_buffer(
            &table_data_buffer,
            0,
            bytemuck::cast_slice(&table_data_words),
        );
    }
    let table_meta_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-upload-table-meta"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    Ok(GpuPackedTableDevice {
        table_index_buffer,
        table_data_buffer,
        table_meta_buffer,
        table_count,
        table_index_len: table_index.len(),
        table_data_len: table_data.len(),
        sizes_known: true,
        max_entries: table_count,
        table_data_bytes_cap: table_data.len(),
    })
}

pub(crate) fn build_table_gpu_device(
    chunk: &[u8],
    max_entries: usize,
    max_entry_len: usize,
    min_ref_len: usize,
    match_probe_limit: usize,
    hash_history_limit: usize,
    table_sample_stride: usize,
) -> Result<(GpuPackedTableDevice, GpuBuildTableProfile), PDeflateError> {
    let r = runtime()?;
    if chunk.len() < 3 || max_entries == 0 {
        return Ok((
            GpuPackedTableDevice {
                table_index_buffer: storage_upload_buffer(
                    &r.device,
                    "cozip-pdeflate-bt-empty-table-index",
                    4,
                ),
                table_data_buffer: storage_upload_buffer(
                    &r.device,
                    "cozip-pdeflate-bt-empty-table-data",
                    4,
                ),
                table_meta_buffer: storage_upload_buffer(
                    &r.device,
                    "cozip-pdeflate-bt-empty-table-meta",
                    4,
                ),
                table_count: 0,
                table_index_len: 0,
                table_data_len: 0,
                sizes_known: true,
                max_entries: 0,
                table_data_bytes_cap: 0,
            },
            GpuBuildTableProfile::default(),
        ));
    }
    let capped_max_entry_len = max_entry_len.min(254).max(3);
    let sample_stride = table_sample_stride.max(1).saturating_mul(8);
    let sample_count = (chunk.len() - 2).div_ceil(sample_stride);
    let min_seed_match_len = min_ref_len.max(6).min(capped_max_entry_len);
    let history_limit = hash_history_limit.max(1).min(64);
    let probe_limit = match_probe_limit.max(1).min(8);

    let src_words = pack_bytes_to_words(chunk);
    let src_words_bytes = u64::try_from(src_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let freq_words = 256usize;
    let freq_bytes = u64::try_from(freq_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let cand_words = sample_count
        .checked_mul(4)
        .ok_or(PDeflateError::NumericOverflow)?;
    let cand_bytes = u64::try_from(cand_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let sort_bucket_words = BUILD_TABLE_SORT_BUCKETS;
    let sort_bucket_bytes = u64::try_from(sort_bucket_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let sorted_idx_words = sample_count;
    let sorted_idx_bytes = u64::try_from(sorted_idx_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_meta_words = 2usize
        .checked_add(
            max_entries
                .checked_mul(2)
                .ok_or(PDeflateError::NumericOverflow)?,
        )
        .ok_or(PDeflateError::NumericOverflow)?;
    let table_meta_bytes = u64::try_from(table_meta_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_data_bytes_cap = max_entries
        .checked_mul(capped_max_entry_len)
        .ok_or(PDeflateError::NumericOverflow)?;
    let table_data_words = table_data_bytes_cap.div_ceil(4);
    let table_data_bytes = u64::try_from(table_data_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let freq_params: [u32; 1] =
        [u32::try_from(chunk.len()).map_err(|_| PDeflateError::NumericOverflow)?];
    let freq_params_bytes = u64::try_from(freq_params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let cand_params: [u32; 7] = [
        u32::try_from(chunk.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(sample_stride).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(capped_max_entry_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(min_seed_match_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(history_limit).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(probe_limit).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(sample_count).map_err(|_| PDeflateError::NumericOverflow)?,
    ];
    let cand_params_bytes = u64::try_from(cand_params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let bucket_params: [u32; 1] =
        [u32::try_from(sample_count).map_err(|_| PDeflateError::NumericOverflow)?];
    let bucket_params_bytes = u64::try_from(bucket_params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let finalize_params: [u32; 8] = [
        u32::try_from(chunk.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(sample_count).map_err(|_| PDeflateError::NumericOverflow)?,
        0,
        u32::try_from(capped_max_entry_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(max_entries).map_err(|_| PDeflateError::NumericOverflow)?,
        64,
        8,
        u32::try_from(table_data_bytes_cap).map_err(|_| PDeflateError::NumericOverflow)?,
    ];
    let finalize_params_bytes = u64::try_from(finalize_params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);

    let src_buffer = storage_upload_buffer(&r.device, "cozip-pdeflate-bt-src", src_words_bytes);
    let freq_params_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-bt-freq-params",
        freq_params_bytes,
    );
    let cand_params_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-bt-cand-params",
        cand_params_bytes,
    );
    let bucket_params_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-bt-bucket-params",
        bucket_params_bytes,
    );
    let finalize_params_buffer = storage_upload_buffer(
        &r.device,
        "cozip-pdeflate-bt-finalize-params",
        finalize_params_bytes,
    );
    let freq_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-freq"),
        size: freq_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cand_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-cand"),
        size: cand_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bucket_count_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-count"),
        size: sort_bucket_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bucket_offset_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-offset"),
        size: sort_bucket_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bucket_cursor_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-cursor"),
        size: sort_bucket_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let sorted_idx_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-sorted-idx"),
        size: sorted_idx_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let table_meta_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-table-meta"),
        size: table_meta_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let table_index_words = max_entries.div_ceil(4);
    let table_index_bytes = u64::try_from(table_index_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let table_index_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-table-index"),
        size: table_index_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let table_data_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-table-data"),
        size: table_data_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let freq_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-freq-bg"),
        layout: &r.build_table_freq_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: freq_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: freq_buffer.as_entire_binding(),
            },
        ],
    });
    let cand_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-cand-bg"),
        layout: &r.build_table_candidate_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cand_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cand_buffer.as_entire_binding(),
            },
        ],
    });
    let bucket_count_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-count-bg"),
        layout: &r.build_table_bucket_count_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cand_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bucket_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bucket_count_buffer.as_entire_binding(),
            },
        ],
    });
    let bucket_prefix_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-prefix-bg"),
        layout: &r.build_table_bucket_prefix_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: bucket_count_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bucket_offset_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: finalize_params_buffer.as_entire_binding(),
            },
        ],
    });
    let bucket_scatter_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-bucket-scatter-bg"),
        layout: &r.build_table_bucket_scatter_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cand_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bucket_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bucket_offset_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bucket_cursor_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: sorted_idx_buffer.as_entire_binding(),
            },
        ],
    });
    let finalize_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-finalize-bg"),
        layout: &r.build_table_finalize_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: freq_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cand_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sorted_idx_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: finalize_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: table_meta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: table_data_buffer.as_entire_binding(),
            },
        ],
    });
    let pack_index_bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-bt-pack-index-bg"),
        layout: &r.build_table_pack_index_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: table_meta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: finalize_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: table_index_buffer.as_entire_binding(),
            },
        ],
    });

    let t_upload = Instant::now();
    r.queue
        .write_buffer(&src_buffer, 0, bytemuck::cast_slice(&src_words));
    r.queue
        .write_buffer(&freq_params_buffer, 0, bytemuck::cast_slice(&freq_params));
    r.queue
        .write_buffer(&cand_params_buffer, 0, bytemuck::cast_slice(&cand_params));
    r.queue.write_buffer(
        &bucket_params_buffer,
        0,
        bytemuck::cast_slice(&bucket_params),
    );
    r.queue.write_buffer(
        &finalize_params_buffer,
        0,
        bytemuck::cast_slice(&finalize_params),
    );
    let upload_ms = elapsed_ms(t_upload);

    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-bt-encoder"),
        });
    encoder.clear_buffer(&freq_buffer, 0, None);
    encoder.clear_buffer(&cand_buffer, 0, None);

    let freq_groups = u32::try_from(chunk.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .div_ceil(256)
        .max(1);
    let t_freq = Instant::now();
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-freq-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_freq_pipeline);
        pass.set_bind_group(0, &freq_bind_group, &[]);
        pass.dispatch_workgroups(freq_groups, 1, 1);
    }
    let freq_kernel_ms = elapsed_ms(t_freq);

    let cand_groups = u32::try_from(sample_count)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .div_ceil(128)
        .max(1);
    let t_cand = Instant::now();
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-candidate-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_candidate_pipeline);
        pass.set_bind_group(0, &cand_bind_group, &[]);
        pass.dispatch_workgroups(cand_groups, 1, 1);
    }
    let mut candidate_kernel_ms = elapsed_ms(t_cand);

    encoder.clear_buffer(&bucket_count_buffer, 0, None);
    let t_bucket_count = Instant::now();
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-bucket-count-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_bucket_count_pipeline);
        pass.set_bind_group(0, &bucket_count_bind_group, &[]);
        pass.dispatch_workgroups(cand_groups, 1, 1);
    }
    candidate_kernel_ms += elapsed_ms(t_bucket_count);
    r.queue.submit(Some(encoder.finish()));

    let readback_ms = 0.0;

    let mut encoder2 = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-bt-finalize-encoder"),
        });
    encoder2.clear_buffer(&bucket_cursor_buffer, 0, None);
    encoder2.clear_buffer(&table_meta_buffer, 0, None);
    encoder2.clear_buffer(&table_data_buffer, 0, None);
    encoder2.clear_buffer(&table_index_buffer, 0, None);
    let t_bucket_prefix = Instant::now();
    {
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-bucket-prefix-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_bucket_prefix_pipeline);
        pass.set_bind_group(0, &bucket_prefix_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    candidate_kernel_ms += elapsed_ms(t_bucket_prefix);
    let t_bucket_scatter = Instant::now();
    {
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-bucket-scatter-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_bucket_scatter_pipeline);
        pass.set_bind_group(0, &bucket_scatter_bind_group, &[]);
        pass.dispatch_workgroups(cand_groups, 1, 1);
    }
    candidate_kernel_ms += elapsed_ms(t_bucket_scatter);
    let t_finalize_kernel = Instant::now();
    {
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-finalize-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_finalize_pipeline);
        pass.set_bind_group(0, &finalize_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    {
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-pack-index-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_pack_index_pipeline);
        pass.set_bind_group(0, &pack_index_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    let sort_ms = elapsed_ms(t_finalize_kernel);
    r.queue.submit(Some(encoder2.finish()));
    Ok((
        GpuPackedTableDevice {
            table_index_buffer,
            table_data_buffer,
            table_meta_buffer,
            table_count: 0,
            table_index_len: 0,
            table_data_len: 0,
            sizes_known: false,
            max_entries,
            table_data_bytes_cap,
        },
        GpuBuildTableProfile {
            upload_ms,
            freq_kernel_ms,
            candidate_kernel_ms,
            readback_ms,
            materialize_ms: 0.0,
            sort_ms,
            sample_count,
        },
    ))
}

fn resolve_packed_table_sizes(
    packed: &GpuPackedTableDevice,
) -> Result<(usize, usize, usize), PDeflateError> {
    if packed.sizes_known {
        return Ok((
            packed.table_count,
            packed.table_index_len,
            packed.table_data_len,
        ));
    }
    if packed.max_entries == 0 {
        return Ok((0, 0, 0));
    }
    let r = runtime()?;
    readback_table_sizes_from_meta_buffer(
        r,
        &packed.table_meta_buffer,
        packed.max_entries,
        packed.table_data_bytes_cap,
    )
}

fn resolve_packed_table_sizes_batch(
    packed_tables: &[&GpuPackedTableDevice],
) -> Result<Vec<(usize, usize, usize)>, PDeflateError> {
    let mut out = vec![(0usize, 0usize, 0usize); packed_tables.len()];
    let mut unresolved = Vec::<(usize, &GpuPackedTableDevice)>::new();
    for (idx, packed) in packed_tables.iter().copied().enumerate() {
        if packed.sizes_known {
            out[idx] = (
                packed.table_count,
                packed.table_index_len,
                packed.table_data_len,
            );
        } else if packed.max_entries > 0 {
            unresolved.push((idx, packed));
        }
    }
    if unresolved.is_empty() {
        return Ok(out);
    }

    let r = runtime()?;
    let readback_bytes = u64::try_from(unresolved.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(8);
    let readback = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-batch-size-readback"),
        size: readback_bytes.max(8),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-bt-batch-size-readback-encoder"),
        });
    for (slot, (_, packed)) in unresolved.iter().enumerate() {
        let dst = u64::try_from(slot)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(8);
        let data_total_idx = 1usize
            .checked_add(
                packed
                    .max_entries
                    .checked_mul(2)
                    .ok_or(PDeflateError::NumericOverflow)?,
            )
            .ok_or(PDeflateError::NumericOverflow)?;
        encoder.copy_buffer_to_buffer(&packed.table_meta_buffer, 0, &readback, dst, 4);
        encoder.copy_buffer_to_buffer(
            &packed.table_meta_buffer,
            u64::try_from(data_total_idx)
                .map_err(|_| PDeflateError::NumericOverflow)?
                .saturating_mul(4),
            &readback,
            dst + 4,
            4,
        );
    }
    let submission = r.queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..readback_bytes.max(8));
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    r.device.poll(wgpu::Maintain::wait_for(submission));
    rx.recv()
        .map_err(|_| {
            PDeflateError::Gpu("gpu build-table batch size map channel closed".to_string())
        })?
        .map_err(|e| PDeflateError::Gpu(format!("gpu build-table batch size map failed: {e}")))?;
    let mapped = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&mapped);
    for (slot, (idx, packed)) in unresolved.iter().enumerate() {
        let base = slot.saturating_mul(2);
        let count = usize::try_from(*words.get(base).unwrap_or(&0))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .min(packed.max_entries);
        let data_total = usize::try_from(*words.get(base + 1).unwrap_or(&0))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .min(packed.table_data_bytes_cap);
        out[*idx] = (count, count, data_total);
    }
    drop(mapped);
    readback.unmap();
    Ok(out)
}

pub(crate) fn readback_packed_table_device(
    packed: &GpuPackedTableDevice,
) -> Result<GpuPackedTable, PDeflateError> {
    let (table_count, table_index_len, table_data_len) = resolve_packed_table_sizes(packed)?;
    if table_count == 0 || (table_index_len == 0 && table_data_len == 0) {
        return Ok(GpuPackedTable {
            table_index: Vec::new(),
            table_data: Vec::new(),
            table_count: 0,
        });
    }
    let r = runtime()?;
    let index_copy_bytes = u64::try_from(table_index_len.div_ceil(4))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let data_copy_bytes = u64::try_from(table_data_len.div_ceil(4))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let total_copy_bytes = index_copy_bytes
        .checked_add(data_copy_bytes)
        .ok_or(PDeflateError::NumericOverflow)?;
    let readback = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-bt-readback-table-combined"),
        size: total_copy_bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut submission = None;
    if total_copy_bytes > 0 {
        let mut encoder = r
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-pdeflate-bt-readback-encoder"),
            });
        if index_copy_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &packed.table_index_buffer,
                0,
                &readback,
                0,
                index_copy_bytes,
            );
        }
        if data_copy_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &packed.table_data_buffer,
                0,
                &readback,
                index_copy_bytes,
                data_copy_bytes,
            );
        }
        submission = Some(r.queue.submit(Some(encoder.finish())));
    }

    let slice = readback.slice(..total_copy_bytes.max(4));
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    if let Some(submission) = submission {
        r.device.poll(wgpu::Maintain::wait_for(submission));
    } else {
        r.device.poll(wgpu::Maintain::Poll);
    }
    rx.recv()
        .map_err(|_| PDeflateError::Gpu("gpu build-table combined map channel closed".to_string()))?
        .map_err(|e| PDeflateError::Gpu(format!("gpu build-table combined map failed: {e}")))?;
    let mapped = slice.get_mapped_range();
    let mut table_index = vec![0u8; table_index_len];
    let mut table_data = vec![0u8; table_data_len];
    if table_index_len > 0 {
        table_index.copy_from_slice(&mapped[..table_index_len]);
    }
    if table_data_len > 0 {
        let data_off =
            usize::try_from(index_copy_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
        let data_end = data_off
            .checked_add(table_data_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        table_data.copy_from_slice(&mapped[data_off..data_end]);
    }
    drop(mapped);
    readback.unmap();

    Ok(GpuPackedTable {
        table_index,
        table_data,
        table_count,
    })
}

#[allow(dead_code)]
pub(crate) fn build_table_gpu(
    chunk: &[u8],
    max_entries: usize,
    max_entry_len: usize,
    min_ref_len: usize,
    match_probe_limit: usize,
    hash_history_limit: usize,
    table_sample_stride: usize,
) -> Result<(GpuPackedTable, GpuBuildTableProfile), PDeflateError> {
    let (packed_device, mut profile) = build_table_gpu_device(
        chunk,
        max_entries,
        max_entry_len,
        min_ref_len,
        match_probe_limit,
        hash_history_limit,
        table_sample_stride,
    )?;
    let t_readback = Instant::now();
    let packed = readback_packed_table_device(&packed_device)?;
    profile.materialize_ms += elapsed_ms(t_readback);
    Ok((packed, profile))
}

struct PackedBatchInputs<'a> {
    src_words: Vec<u32>,
    chunk_starts: Vec<u32>,
    table_chunk_bases: Vec<u32>,
    table_chunk_counts: Vec<u32>,
    table_chunk_index_offsets: Vec<u32>,
    table_chunk_data_offsets: Vec<u32>,
    table_index_words: Vec<u32>,
    table_data_words: Vec<u32>,
    table_index_words_len: usize,
    table_data_words_len: usize,
    table_device_copies: Vec<PackedTableDeviceCopy<'a>>,
    chunk_lens: Vec<usize>,
    total_src_len: usize,
    total_table_entries: usize,
    max_ref_len: usize,
    min_ref_len: usize,
    section_count: usize,
}

struct PackedTableDeviceCopy<'a> {
    index_src: &'a wgpu::Buffer,
    data_src: &'a wgpu::Buffer,
    index_dst_offset: u64,
    data_dst_offset: u64,
    index_len: usize,
    data_len: usize,
}

struct PackedTableUpload<'a> {
    index_offset: u64,
    data_offset: u64,
    index_len: usize,
    data_len: usize,
    kind: PackedTableUploadKind<'a>,
}

enum PackedTableUploadKind<'a> {
    Host {
        index_bytes: Vec<u8>,
        data_bytes: Vec<u8>,
    },
    Device {
        index_buffer: &'a wgpu::Buffer,
        data_buffer: &'a wgpu::Buffer,
    },
}

fn pack_match_batch_inputs<'a>(
    inputs: &'a [GpuMatchInput<'a>],
) -> Result<PackedBatchInputs<'a>, PDeflateError> {
    if inputs.is_empty() {
        return Err(PDeflateError::InvalidOptions(
            "gpu match batch inputs must be non-empty",
        ));
    }
    let max_ref_len = inputs[0].max_ref_len;
    let min_ref_len = inputs[0].min_ref_len;
    let section_count = inputs[0].section_count;
    if max_ref_len == 0 || min_ref_len == 0 || section_count == 0 {
        return Err(PDeflateError::InvalidOptions(
            "gpu match params must be non-zero",
        ));
    }
    let total_src_bytes: usize = inputs.iter().map(|input| input.src.len()).sum();
    let mut src_bytes = Vec::<u8>::with_capacity(total_src_bytes);
    let mut chunk_starts = Vec::<u32>::with_capacity(inputs.len() + 1);
    let mut table_chunk_bases = Vec::<u32>::with_capacity(inputs.len());
    let mut table_chunk_counts = Vec::<u32>::with_capacity(inputs.len());
    let mut table_chunk_index_offsets = Vec::<u32>::with_capacity(inputs.len());
    let mut table_chunk_data_offsets = Vec::<u32>::with_capacity(inputs.len());
    let mut table_uploads = Vec::<PackedTableUpload<'a>>::with_capacity(inputs.len());
    let mut table_index_packer =
        ByteWordPacker::with_capacity_bytes(inputs.len().saturating_mul(256));
    let mut table_data_packer =
        ByteWordPacker::with_capacity_bytes(inputs.iter().map(|i| i.src.len() / 2).sum());
    let mut chunk_lens = Vec::<usize>::with_capacity(inputs.len());
    let mut global_table_base = 0usize;
    let unresolved_tables: Vec<&GpuPackedTableDevice> = inputs
        .iter()
        .filter_map(|input| input.table_gpu)
        .filter(|table_gpu| !table_gpu.sizes_known && table_gpu.max_entries > 0)
        .collect();
    let unresolved_sizes = if unresolved_tables.is_empty() {
        Vec::new()
    } else {
        resolve_packed_table_sizes_batch(&unresolved_tables)?
    };
    let mut unresolved_cursor = 0usize;
    chunk_starts.push(0);
    for input in inputs {
        if input.max_ref_len != max_ref_len
            || input.min_ref_len != min_ref_len
            || input.section_count != section_count
        {
            return Err(PDeflateError::InvalidOptions(
                "gpu match batch params must be consistent",
            ));
        }
        src_bytes.extend_from_slice(input.src);
        chunk_lens.push(input.src.len());
        chunk_starts
            .push(u32::try_from(src_bytes.len()).map_err(|_| PDeflateError::NumericOverflow)?);
        table_chunk_bases
            .push(u32::try_from(global_table_base).map_err(|_| PDeflateError::NumericOverflow)?);
        table_index_packer.align4();
        table_data_packer.align4();
        table_chunk_index_offsets.push(
            u32::try_from(table_index_packer.len_bytes())
                .map_err(|_| PDeflateError::NumericOverflow)?,
        );
        table_chunk_data_offsets.push(
            u32::try_from(table_data_packer.len_bytes())
                .map_err(|_| PDeflateError::NumericOverflow)?,
        );
        let mut local_count = 0usize;
        let (index_len, data_len, kind) = if let Some(table_gpu) = input.table_gpu {
            let (resolved_count, resolved_index_len, resolved_data_len) = if table_gpu.table_count
                > 0
                || table_gpu.table_index_len > 0
                || table_gpu.table_data_len > 0
            {
                (
                    table_gpu.table_count,
                    table_gpu.table_index_len,
                    table_gpu.table_data_len,
                )
            } else {
                let sizes = *unresolved_sizes
                    .get(unresolved_cursor)
                    .ok_or(PDeflateError::NumericOverflow)?;
                unresolved_cursor = unresolved_cursor.saturating_add(1);
                sizes
            };
            local_count = resolved_count;
            (
                resolved_index_len,
                resolved_data_len,
                PackedTableUploadKind::Device {
                    index_buffer: &table_gpu.table_index_buffer,
                    data_buffer: &table_gpu.table_data_buffer,
                },
            )
        } else if let (Some(index_bytes), Some(data_bytes)) = (input.table_index, input.table_data)
        {
            let expected_data_len: usize = index_bytes
                .iter()
                .map(|&v| usize::from(v))
                .try_fold(0usize, |acc, v| acc.checked_add(v))
                .ok_or(PDeflateError::NumericOverflow)?;
            if expected_data_len != data_bytes.len() {
                return Err(PDeflateError::Gpu(
                    "gpu packed table index/data length mismatch".to_string(),
                ));
            }
            local_count = index_bytes.len();
            (
                index_bytes.len(),
                data_bytes.len(),
                PackedTableUploadKind::Host {
                    index_bytes: index_bytes.to_vec(),
                    data_bytes: data_bytes.to_vec(),
                },
            )
        } else {
            let mut index_bytes = Vec::<u8>::new();
            let mut data_bytes = Vec::<u8>::new();
            index_bytes.reserve(input.table.len());
            let data_len: usize = input.table.iter().map(Vec::len).sum();
            data_bytes.reserve(data_len);
            for entry in input.table.iter() {
                index_bytes
                    .push(u8::try_from(entry.len()).map_err(|_| PDeflateError::NumericOverflow)?);
                data_bytes.extend_from_slice(entry);
                local_count = local_count.saturating_add(1);
            }
            (
                index_bytes.len(),
                data_bytes.len(),
                PackedTableUploadKind::Host {
                    index_bytes,
                    data_bytes,
                },
            )
        };
        table_uploads.push(PackedTableUpload {
            index_offset: u64::try_from(table_index_packer.len_bytes())
                .map_err(|_| PDeflateError::NumericOverflow)?,
            data_offset: u64::try_from(table_data_packer.len_bytes())
                .map_err(|_| PDeflateError::NumericOverflow)?,
            index_len,
            data_len,
            kind,
        });
        table_index_packer.append_zeros(index_len);
        table_data_packer.append_zeros(data_len);
        table_chunk_counts
            .push(u32::try_from(local_count).map_err(|_| PDeflateError::NumericOverflow)?);
        global_table_base = global_table_base
            .checked_add(local_count)
            .ok_or(PDeflateError::NumericOverflow)?;
    }
    let mut table_device_copies = Vec::<PackedTableDeviceCopy<'a>>::new();
    for upload in table_uploads.into_iter() {
        match upload.kind {
            PackedTableUploadKind::Host {
                index_bytes,
                data_bytes,
            } => {
                if upload.index_len != 0 {
                    let off = usize::try_from(upload.index_offset)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    table_index_packer.write_bytes_at(off, &index_bytes)?;
                }
                if upload.data_len != 0 {
                    let off = usize::try_from(upload.data_offset)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    table_data_packer.write_bytes_at(off, &data_bytes)?;
                }
            }
            PackedTableUploadKind::Device {
                index_buffer,
                data_buffer,
            } => {
                table_device_copies.push(PackedTableDeviceCopy {
                    index_src: index_buffer,
                    data_src: data_buffer,
                    index_dst_offset: upload.index_offset,
                    data_dst_offset: upload.data_offset,
                    index_len: upload.index_len,
                    data_len: upload.data_len,
                });
            }
        }
    }

    Ok(PackedBatchInputs {
        src_words: pack_bytes_to_words(&src_bytes),
        chunk_starts,
        table_chunk_bases,
        table_chunk_counts,
        table_chunk_index_offsets,
        table_chunk_data_offsets,
        table_index_words_len: table_index_packer.len_bytes().div_ceil(4),
        table_data_words_len: table_data_packer.len_bytes().div_ceil(4),
        table_index_words: table_index_packer.into_words(),
        table_data_words: table_data_packer.into_words(),
        table_device_copies,
        chunk_lens,
        total_src_len: total_src_bytes,
        total_table_entries: global_table_base,
        max_ref_len,
        min_ref_len,
        section_count,
    })
}

fn encode_table_device_copies(
    encoder: &mut wgpu::CommandEncoder,
    scratch: &GpuBatchScratch,
    copies: &[PackedTableDeviceCopy<'_>],
) -> Result<(), PDeflateError> {
    for copy in copies {
        let index_copy_bytes = u64::try_from(copy.index_len.div_ceil(4))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        if index_copy_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &copy.index_src,
                0,
                &scratch.table_index_buffer,
                copy.index_dst_offset,
                index_copy_bytes,
            );
        }
        let data_copy_bytes = u64::try_from(copy.data_len.div_ceil(4))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4);
        if data_copy_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &copy.data_src,
                0,
                &scratch.table_data_buffer,
                copy.data_dst_offset,
                data_copy_bytes,
            );
        }
    }
    Ok(())
}

pub(crate) fn compute_matches_batch(
    inputs: &[GpuMatchInput<'_>],
) -> Result<GpuBatchMatchOutput, PDeflateError> {
    if inputs.is_empty() {
        return Ok(GpuBatchMatchOutput {
            chunks: Vec::new(),
            profile: GpuMatchProfile::default(),
        });
    }
    let t_total = Instant::now();
    let r = runtime()?;
    let packed = pack_match_batch_inputs(inputs)?;
    let groups_total = (u32::try_from(packed.total_src_len)
        .map_err(|_| PDeflateError::NumericOverflow)?)
    .div_ceil(WORKGROUP_SIZE)
    .max(1);
    let groups_x = groups_total.min(65_535);
    let groups_y = groups_total.div_ceil(groups_x).max(1);
    let row_stride = groups_x.saturating_mul(WORKGROUP_SIZE);
    let params: [u32; 6] = [
        u32::try_from(packed.total_src_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(packed.max_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(packed.min_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        row_stride,
        u32::try_from(packed.section_count).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?,
    ];
    let out_len_u64 = u64::try_from(packed.total_src_len)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let required_caps = GpuBatchScratchCaps {
        src_bytes: u64::try_from(packed.src_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        chunk_starts_bytes: u64::try_from(packed.chunk_starts.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_base_bytes: u64::try_from(packed.table_chunk_bases.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_count_bytes: u64::try_from(packed.table_chunk_counts.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_index_offsets_bytes: u64::try_from(packed.table_chunk_index_offsets.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_data_offsets_bytes: u64::try_from(packed.table_chunk_data_offsets.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_index_bytes: u64::try_from(packed.table_index_words_len)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        prefix_first_bytes: u64::try_from(inputs.len().saturating_mul(PREFIX2_TABLE_SIZE))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_lens_bytes: u64::try_from(packed.total_table_entries)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_offsets_bytes: u64::try_from(packed.total_table_entries)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_data_bytes: u64::try_from(packed.table_data_words_len)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        prep_params_bytes: 4,
        params_bytes: u64::try_from(params.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        out_bytes: out_len_u64.max(4),
    };

    let scratch = {
        let mut pool = r
            .scratch_pool
            .lock()
            .map_err(|_| PDeflateError::Gpu("gpu scratch pool mutex poisoned".to_string()))?;
        if let Some(pos) = pool.iter().position(|s| s.fits(required_caps)) {
            pool.swap_remove(pos)
        } else {
            GpuBatchScratch::new(r, required_caps)
        }
    };

    let result = (|| -> Result<GpuBatchMatchOutput, PDeflateError> {
        let t_upload = Instant::now();
        r.queue.write_buffer(
            &scratch.src_buffer,
            0,
            bytemuck::cast_slice(&packed.src_words),
        );
        r.queue.write_buffer(
            &scratch.chunk_starts_buffer,
            0,
            bytemuck::cast_slice(&packed.chunk_starts),
        );
        r.queue.write_buffer(
            &scratch.table_base_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_bases),
        );
        r.queue.write_buffer(
            &scratch.table_count_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_counts),
        );
        r.queue.write_buffer(
            &scratch.table_index_offsets_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_index_offsets),
        );
        r.queue.write_buffer(
            &scratch.table_data_offsets_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_data_offsets),
        );
        if !packed.table_index_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_index_buffer,
                0,
                bytemuck::cast_slice(&packed.table_index_words),
            );
        }
        if !packed.table_data_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_data_buffer,
                0,
                bytemuck::cast_slice(&packed.table_data_words),
            );
        }
        let prep_params: [u32; 1] =
            [u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?];
        r.queue.write_buffer(
            &scratch.prep_params_buffer,
            0,
            bytemuck::cast_slice(&prep_params),
        );
        r.queue
            .write_buffer(&scratch.params_buffer, 0, bytemuck::cast_slice(&params));
        let upload_ms = elapsed_ms(t_upload);

        let mut encoder = r
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-pdeflate-match-encoder"),
            });
        encode_table_device_copies(&mut encoder, &scratch, &packed.table_device_copies)?;
        encoder.clear_buffer(&scratch.prefix_first_buffer, 0, None);
        encoder.clear_buffer(&scratch.table_lens_buffer, 0, None);
        encoder.clear_buffer(&scratch.table_offsets_buffer, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-match-prepare-table-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.match_prepare_table_pipeline);
            pass.set_bind_group(0, &scratch.prep_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-match-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.pipeline);
            pass.set_bind_group(0, &scratch.bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        encoder.copy_buffer_to_buffer(
            &scratch.out_buffer,
            0,
            &scratch.readback_buffer,
            0,
            out_len_u64.max(4),
        );
        let submission = r.queue.submit(Some(encoder.finish()));

        let t_wait = Instant::now();
        let slice = scratch.readback_buffer.slice(..out_len_u64.max(4));
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        r.device.poll(wgpu::Maintain::wait_for(submission));
        let map_res = rx
            .recv()
            .map_err(|_| PDeflateError::Gpu("gpu map_async channel closed".to_string()))?;
        map_res.map_err(|e| PDeflateError::Gpu(format!("gpu map_async failed: {e}")))?;
        let wait_ms = elapsed_ms(t_wait);

        let t_copy = Instant::now();
        let mapped = slice.get_mapped_range();
        let mut packed_all = vec![0u32; packed.total_src_len];
        if !packed_all.is_empty() {
            packed_all
                .copy_from_slice(bytemuck::cast_slice(&mapped)[..packed.total_src_len].as_ref());
        }
        drop(mapped);
        scratch.readback_buffer.unmap();
        let map_copy_ms = elapsed_ms(t_copy);

        let mut chunks = Vec::with_capacity(packed.chunk_lens.len());
        let mut offset = 0usize;
        for len in packed.chunk_lens.iter().copied() {
            let end = offset
                .checked_add(len)
                .ok_or(PDeflateError::NumericOverflow)?;
            chunks.push(GpuBatchChunkOutput {
                packed_matches: packed_all[offset..end].to_vec(),
            });
            offset = end;
        }

        Ok(GpuBatchMatchOutput {
            chunks,
            profile: GpuMatchProfile {
                upload_ms,
                wait_ms,
                map_copy_ms,
                total_ms: elapsed_ms(t_total),
            },
        })
    })();

    {
        let mut pool = r
            .scratch_pool
            .lock()
            .map_err(|_| PDeflateError::Gpu("gpu scratch pool mutex poisoned".to_string()))?;
        if pool.len() < GPU_SCRATCH_POOL_LIMIT {
            pool.push(scratch);
        }
    }

    result
}

pub(crate) fn compute_matches_and_encode_sections_batch(
    inputs: &[GpuMatchInput<'_>],
    section_count: usize,
    max_cmd_len: usize,
) -> Result<GpuBatchSectionEncodeOutput, PDeflateError> {
    if inputs.is_empty() {
        return Ok(GpuBatchSectionEncodeOutput {
            chunks: Vec::new(),
            match_profile: GpuMatchProfile::default(),
            section_profile: GpuSectionEncodeProfile::default(),
            kernel_profile: GpuBatchKernelProfile::default(),
            keepalive: None,
        });
    }
    if section_count == 0 {
        return Err(PDeflateError::InvalidOptions(
            "gpu section encode requires section_count > 0",
        ));
    }

    let r = runtime()?;
    let t_total = Instant::now();
    let t_pack = Instant::now();
    let packed = pack_match_batch_inputs(inputs)?;
    let pack_inputs_ms = elapsed_ms(t_pack);
    if packed.section_count != section_count {
        return Err(PDeflateError::InvalidOptions(
            "gpu section_count mismatch between input and call",
        ));
    }
    let groups_total = (u32::try_from(packed.total_src_len)
        .map_err(|_| PDeflateError::NumericOverflow)?)
    .div_ceil(WORKGROUP_SIZE)
    .max(1);
    let groups_x = groups_total.min(65_535);
    let groups_y = groups_total.div_ceil(groups_x).max(1);
    let row_stride = groups_x.saturating_mul(WORKGROUP_SIZE);
    let match_params: [u32; 6] = [
        u32::try_from(packed.total_src_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(packed.max_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(packed.min_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        row_stride,
        u32::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?,
    ];
    let out_len_u64 = u64::try_from(packed.total_src_len)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let required_caps = GpuBatchScratchCaps {
        src_bytes: u64::try_from(packed.src_words.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        chunk_starts_bytes: u64::try_from(packed.chunk_starts.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_base_bytes: u64::try_from(packed.table_chunk_bases.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_count_bytes: u64::try_from(packed.table_chunk_counts.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_index_offsets_bytes: u64::try_from(packed.table_chunk_index_offsets.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_data_offsets_bytes: u64::try_from(packed.table_chunk_data_offsets.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_index_bytes: u64::try_from(packed.table_index_words_len)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        prefix_first_bytes: u64::try_from(inputs.len().saturating_mul(PREFIX2_TABLE_SIZE))
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_lens_bytes: u64::try_from(packed.total_table_entries)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_offsets_bytes: u64::try_from(packed.total_table_entries)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        table_data_bytes: u64::try_from(packed.table_data_words_len)
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        prep_params_bytes: 4,
        params_bytes: u64::try_from(match_params.len())
            .map_err(|_| PDeflateError::NumericOverflow)?
            .saturating_mul(4),
        out_bytes: out_len_u64.max(4),
    };

    let t_scratch = Instant::now();
    let mut scratch = {
        let mut pool = r
            .scratch_pool
            .lock()
            .map_err(|_| PDeflateError::Gpu("gpu scratch pool mutex poisoned".to_string()))?;
        if let Some(pos) = pool.iter().position(|s| s.fits(required_caps)) {
            pool.swap_remove(pos)
        } else {
            GpuBatchScratch::new(r, required_caps)
        }
    };
    let scratch_acquire_ms = elapsed_ms(t_scratch);

    struct SectionChunkState {
        slot_index: usize,
        section_count: usize,
        out_lens_bytes: u64,
        out_cmd_bytes: u64,
        section_index_cap_bytes: u64,
    }

    let result = (|| -> Result<GpuBatchSectionEncodeOutput, PDeflateError> {
        let t_match_upload = Instant::now();
        r.queue.write_buffer(
            &scratch.src_buffer,
            0,
            bytemuck::cast_slice(&packed.src_words),
        );
        r.queue.write_buffer(
            &scratch.chunk_starts_buffer,
            0,
            bytemuck::cast_slice(&packed.chunk_starts),
        );
        r.queue.write_buffer(
            &scratch.table_base_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_bases),
        );
        r.queue.write_buffer(
            &scratch.table_count_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_counts),
        );
        r.queue.write_buffer(
            &scratch.table_index_offsets_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_index_offsets),
        );
        r.queue.write_buffer(
            &scratch.table_data_offsets_buffer,
            0,
            bytemuck::cast_slice(&packed.table_chunk_data_offsets),
        );
        if !packed.table_index_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_index_buffer,
                0,
                bytemuck::cast_slice(&packed.table_index_words),
            );
        }
        if !packed.table_data_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_data_buffer,
                0,
                bytemuck::cast_slice(&packed.table_data_words),
            );
        }
        let prep_params: [u32; 1] =
            [u32::try_from(inputs.len()).map_err(|_| PDeflateError::NumericOverflow)?];
        r.queue.write_buffer(
            &scratch.prep_params_buffer,
            0,
            bytemuck::cast_slice(&prep_params),
        );
        r.queue.write_buffer(
            &scratch.params_buffer,
            0,
            bytemuck::cast_slice(&match_params),
        );
        let match_upload_ms = elapsed_ms(t_match_upload);
        let mut encoder = r
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-pdeflate-match-section-encoder"),
            });
        let t_table_copy = Instant::now();
        encode_table_device_copies(&mut encoder, &scratch, &packed.table_device_copies)?;
        let match_table_copy_ms = elapsed_ms(t_table_copy);
        encoder.clear_buffer(&scratch.prefix_first_buffer, 0, None);
        encoder.clear_buffer(&scratch.table_lens_buffer, 0, None);
        encoder.clear_buffer(&scratch.table_offsets_buffer, 0, None);
        let match_prepare_dispatch_ms = {
            let t_prepare = Instant::now();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-match-prepare-table-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.match_prepare_table_pipeline);
            pass.set_bind_group(0, &scratch.prep_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);
            elapsed_ms(t_prepare)
        };
        let match_kernel_dispatch_ms = {
            let t_match = Instant::now();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-match-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.pipeline);
            pass.set_bind_group(0, &scratch.bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
            drop(pass);
            elapsed_ms(t_match)
        };

        let t_section_total = Instant::now();
        let t_section_setup = Instant::now();
        let mut section_states = Vec::<SectionChunkState>::with_capacity(packed.chunk_lens.len());
        let mut slot_in_use = vec![false; scratch.section_slots.len()];
        for (chunk_idx, &chunk_len) in packed.chunk_lens.iter().enumerate() {
            let src_base_u32 = packed.chunk_starts[chunk_idx];
            let mut section_offsets = vec![0u32; section_count];
            let mut section_caps = vec![0u32; section_count];
            let mut total_cmd_cap_bytes: usize = 0;
            for sec in 0..section_count {
                let s0 = section_start(sec, section_count, chunk_len);
                let s1 = section_start(sec + 1, section_count, chunk_len);
                let sec_len = s1.saturating_sub(s0);
                let cap_bytes = estimate_section_cmd_cap_bytes(sec_len)?;
                let aligned = align_up4(total_cmd_cap_bytes);
                section_offsets[sec] =
                    u32::try_from(aligned).map_err(|_| PDeflateError::NumericOverflow)?;
                section_caps[sec] =
                    u32::try_from(cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
                total_cmd_cap_bytes = aligned
                    .checked_add(cap_bytes)
                    .ok_or(PDeflateError::NumericOverflow)?;
            }

            let section_count_u64 =
                u64::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
            let section_params: [u32; 6] = [
                u32::try_from(chunk_len).map_err(|_| PDeflateError::NumericOverflow)?,
                u32::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?,
                u32::try_from(packed.min_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
                u32::try_from(packed.max_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
                u32::try_from(max_cmd_len).map_err(|_| PDeflateError::NumericOverflow)?,
                src_base_u32,
            ];
            let out_lens_bytes = section_count_u64.saturating_mul(4);
            let out_cmd_words = total_cmd_cap_bytes.div_ceil(4);
            let out_cmd_bytes = u64::try_from(out_cmd_words)
                .map_err(|_| PDeflateError::NumericOverflow)?
                .saturating_mul(4);
            if out_cmd_bytes > r.max_storage_binding_size {
                return Err(PDeflateError::Gpu(format!(
                    "gpu section encode output buffer too large ({} > {})",
                    out_cmd_bytes, r.max_storage_binding_size
                )));
            }

            let slot_index = if let Some((idx, _)) = scratch
                .section_slots
                .iter()
                .enumerate()
                .find(|(idx, slot)| !slot_in_use[*idx] && slot.fits(section_count, out_cmd_bytes))
            {
                idx
            } else {
                let slot = GpuSectionEncodeSlot::new(r, &scratch, section_count, out_cmd_bytes)?;
                scratch.section_slots.push(slot);
                slot_in_use.push(false);
                scratch.section_slots.len() - 1
            };
            slot_in_use[slot_index] = true;
            let slot = &scratch.section_slots[slot_index];
            r.queue.write_buffer(
                &slot.section_offsets_buffer,
                0,
                bytemuck::cast_slice(&section_offsets),
            );
            r.queue.write_buffer(
                &slot.section_caps_buffer,
                0,
                bytemuck::cast_slice(&section_caps),
            );
            r.queue.write_buffer(
                &slot.section_params_buffer,
                0,
                bytemuck::cast_slice(&section_params),
            );

            section_states.push(SectionChunkState {
                slot_index,
                section_count,
                out_lens_bytes,
                out_cmd_bytes,
                section_index_cap_bytes: slot.section_index_cap_bytes,
            });
        }
        let section_upload_ms = elapsed_ms(t_section_setup);
        let mut section_pass_dispatch_ms = 0.0_f64;
        let section_copy_dispatch_ms = 0.0_f64;

        for state in section_states.iter() {
            let slot = &scratch.section_slots[state.slot_index];
            let groups_x = u32::try_from(state.section_count)
                .map_err(|_| PDeflateError::NumericOverflow)?
                .div_ceil(64)
                .max(1);
            {
                let t_pass = Instant::now();
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-pdeflate-se-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&r.section_encode_pipeline);
                pass.set_bind_group(0, &slot.bind_group, &[]);
                pass.dispatch_workgroups(groups_x, 1, 1);
                drop(pass);
                section_pass_dispatch_ms += elapsed_ms(t_pass);
            }
            {
                let t_meta = Instant::now();
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cozip-pdeflate-se-meta-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&r.section_meta_pipeline);
                pass.set_bind_group(0, &slot.meta_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                drop(pass);
                section_pass_dispatch_ms += elapsed_ms(t_meta);
            }
            encoder.copy_buffer_to_buffer(
                &slot.section_meta_buffer,
                0,
                &slot.readback_meta_buffer,
                0,
                16,
            );
        }
        let t_submit = Instant::now();
        r.queue.submit(Some(encoder.finish()));
        let submit_ms = elapsed_ms(t_submit);
        let match_weight =
            match_table_copy_ms + match_prepare_dispatch_ms + match_kernel_dispatch_ms;
        let section_weight = section_pass_dispatch_ms + section_copy_dispatch_ms;
        let weight_sum = match_weight + section_weight;
        let match_submit_ms = if weight_sum > 0.0 {
            submit_ms * (match_weight / weight_sum)
        } else {
            submit_ms
        };
        let section_submit_ms = if weight_sum > 0.0 {
            submit_ms * (section_weight / weight_sum)
        } else {
            0.0
        };

        let t_wrap = Instant::now();
        let mut chunks = Vec::with_capacity(section_states.len());
        for state in section_states {
            let slot = &scratch.section_slots[state.slot_index];
            chunks.push(GpuBatchSectionEncodeChunkOutput {
                section_cmd_lens: Vec::new(),
                section_cmd: Vec::new(),
                section_cmd_device: Some(GpuSectionCmdDeviceOutput {
                    section_count: state.section_count,
                    section_offsets_buffer: slot.section_offsets_buffer.clone(),
                    out_lens_buffer: slot.out_lens_buffer.clone(),
                    section_prefix_buffer: slot.section_prefix_buffer.clone(),
                    section_index_buffer: slot.section_index_buffer.clone(),
                    section_meta_buffer: slot.section_meta_buffer.clone(),
                    section_meta_readback_buffer: slot.readback_meta_buffer.clone(),
                    out_cmd_buffer: slot.out_cmd_buffer.clone(),
                    out_lens_bytes: state.out_lens_bytes,
                    out_cmd_bytes: state.out_cmd_bytes,
                    section_index_cap_bytes: state.section_index_cap_bytes,
                }),
            });
        }
        let section_wrap_ms = elapsed_ms(t_wrap);

        Ok(GpuBatchSectionEncodeOutput {
            chunks,
            match_profile: GpuMatchProfile {
                upload_ms: match_upload_ms,
                wait_ms: 0.0,
                map_copy_ms: 0.0,
                total_ms: elapsed_ms(t_total),
            },
            section_profile: GpuSectionEncodeProfile {
                upload_ms: section_upload_ms,
                wait_ms: 0.0,
                map_copy_ms: 0.0,
                total_ms: elapsed_ms(t_section_total),
            },
            kernel_profile: GpuBatchKernelProfile {
                pack_inputs_ms,
                scratch_acquire_ms,
                match_table_copy_ms,
                match_prepare_dispatch_ms,
                match_kernel_dispatch_ms,
                match_submit_ms,
                section_setup_ms: section_upload_ms,
                section_pass_dispatch_ms,
                section_copy_dispatch_ms,
                section_submit_ms,
                section_wrap_ms,
                ..GpuBatchKernelProfile::default()
            },
            keepalive: None,
        })
    })();
    match result {
        Ok(mut out) => {
            out.keepalive = Some(GpuBatchSectionKeepalive::new(scratch));
            Ok(out)
        }
        Err(err) => {
            if let Ok(mut pool) = r.scratch_pool.lock() {
                if pool.len() < GPU_SCRATCH_POOL_LIMIT {
                    pool.push(scratch);
                }
            }
            Err(err)
        }
    }
}

pub(crate) fn compute_matches(input: &GpuMatchInput<'_>) -> Result<GpuMatchOutput, PDeflateError> {
    let batch = compute_matches_batch(std::slice::from_ref(input))?;
    let mut iter = batch.chunks.into_iter();
    let Some(chunk) = iter.next() else {
        return Err(PDeflateError::Gpu(
            "gpu batch returned empty result for single chunk".to_string(),
        ));
    };
    Ok(GpuMatchOutput {
        packed_matches: chunk.packed_matches,
        profile: batch.profile,
    })
}

#[inline]
fn section_start(sec: usize, section_count: usize, len: usize) -> usize {
    if section_count == 0 {
        return 0;
    }
    sec.saturating_mul(len) / section_count
}

#[allow(dead_code)]
pub(crate) fn encode_section_commands(
    src: &[u8],
    packed_matches: &[u32],
    section_count: usize,
    min_ref_len: usize,
    max_ref_len: usize,
    max_cmd_len: usize,
) -> Result<GpuSectionEncodeOutput, PDeflateError> {
    if src.len() != packed_matches.len() {
        return Err(PDeflateError::InvalidOptions(
            "gpu section encode input length mismatch",
        ));
    }
    if src.is_empty() || section_count == 0 {
        return Ok(GpuSectionEncodeOutput {
            section_cmd_lens: vec![0; section_count],
            section_cmd: Vec::new(),
            profile: GpuSectionEncodeProfile::default(),
        });
    }
    let r = runtime()?;
    let t_total = Instant::now();

    let mut section_offsets = vec![0u32; section_count];
    let mut section_caps = vec![0u32; section_count];
    let mut total_cmd_cap_bytes: usize = 0;
    for sec in 0..section_count {
        let s0 = section_start(sec, section_count, src.len());
        let s1 = section_start(sec + 1, section_count, src.len());
        let sec_len = s1.saturating_sub(s0);
        // Tightened cap to reduce oversized command buffer traffic.
        let cap_bytes = estimate_section_cmd_cap_bytes(sec_len)?;
        let aligned = align_up4(total_cmd_cap_bytes);
        section_offsets[sec] =
            u32::try_from(aligned).map_err(|_| PDeflateError::NumericOverflow)?;
        section_caps[sec] = u32::try_from(cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
        total_cmd_cap_bytes = aligned
            .checked_add(cap_bytes)
            .ok_or(PDeflateError::NumericOverflow)?;
    }

    let src_words = pack_bytes_to_words(src);
    let src_words_bytes = u64::try_from(src_words.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let packed_bytes = u64::try_from(packed_matches.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let section_count_u64 =
        u64::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?;
    let section_offsets_bytes = section_count_u64.saturating_mul(4);
    let section_caps_bytes = section_count_u64.saturating_mul(4);
    let params: [u32; 6] = [
        u32::try_from(src.len()).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(section_count).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(min_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(max_ref_len).map_err(|_| PDeflateError::NumericOverflow)?,
        u32::try_from(max_cmd_len).map_err(|_| PDeflateError::NumericOverflow)?,
        0,
    ];
    let params_bytes = u64::try_from(params.len())
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    let out_lens_bytes = section_count_u64.saturating_mul(4);
    let out_cmd_words = total_cmd_cap_bytes.div_ceil(4);
    let out_cmd_bytes = u64::try_from(out_cmd_words)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .saturating_mul(4);
    if out_cmd_bytes > r.max_storage_binding_size {
        return Err(PDeflateError::Gpu(format!(
            "gpu section encode output buffer too large ({} > {})",
            out_cmd_bytes, r.max_storage_binding_size
        )));
    }

    let src_buffer = storage_upload_buffer(&r.device, "cozip-pdeflate-se-src", src_words_bytes);
    let packed_buffer = storage_upload_buffer(&r.device, "cozip-pdeflate-se-packed", packed_bytes);
    let section_offsets_buffer =
        storage_upload_buffer(&r.device, "cozip-pdeflate-se-offs", section_offsets_bytes);
    let section_caps_buffer =
        storage_upload_buffer(&r.device, "cozip-pdeflate-se-caps", section_caps_bytes);
    let params_buffer = storage_upload_buffer(&r.device, "cozip-pdeflate-se-params", params_bytes);
    let out_lens_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-se-out-lens"),
        size: out_lens_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_cmd_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-se-out-cmd"),
        size: out_cmd_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_lens_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-se-readback-lens"),
        size: out_lens_bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback_cmd_buffer = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-se-readback-cmd"),
        size: out_cmd_bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cozip-pdeflate-se-bg"),
        layout: &r.section_encode_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: packed_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: section_offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: section_caps_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: out_lens_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: out_cmd_buffer.as_entire_binding(),
            },
        ],
    });

    let t_upload = Instant::now();
    r.queue
        .write_buffer(&src_buffer, 0, bytemuck::cast_slice(&src_words));
    r.queue
        .write_buffer(&packed_buffer, 0, bytemuck::cast_slice(packed_matches));
    r.queue.write_buffer(
        &section_offsets_buffer,
        0,
        bytemuck::cast_slice(&section_offsets),
    );
    r.queue
        .write_buffer(&section_caps_buffer, 0, bytemuck::cast_slice(&section_caps));
    r.queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));
    let upload_ms = elapsed_ms(t_upload);

    let groups_x = u32::try_from(section_count)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .div_ceil(64)
        .max(1);
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-se-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-se-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.section_encode_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &out_lens_buffer,
        0,
        &readback_lens_buffer,
        0,
        out_lens_bytes.max(4),
    );
    encoder.copy_buffer_to_buffer(
        &out_cmd_buffer,
        0,
        &readback_cmd_buffer,
        0,
        out_cmd_bytes.max(4),
    );
    let submission = r.queue.submit(Some(encoder.finish()));

    let t_wait = Instant::now();
    let lens_slice = readback_lens_buffer.slice(..out_lens_bytes.max(4));
    let cmd_slice = readback_cmd_buffer.slice(..out_cmd_bytes.max(4));
    let (tx_lens, rx_lens) = mpsc::channel();
    lens_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_lens.send(res);
    });
    let (tx_cmd, rx_cmd) = mpsc::channel();
    cmd_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_cmd.send(res);
    });
    r.device.poll(wgpu::Maintain::wait_for(submission));
    rx_lens
        .recv()
        .map_err(|_| PDeflateError::Gpu("gpu section encode lens map channel closed".to_string()))?
        .map_err(|e| PDeflateError::Gpu(format!("gpu section encode lens map failed: {e}")))?;
    rx_cmd
        .recv()
        .map_err(|_| PDeflateError::Gpu("gpu section encode cmd map channel closed".to_string()))?
        .map_err(|e| PDeflateError::Gpu(format!("gpu section encode cmd map failed: {e}")))?;
    let wait_ms = elapsed_ms(t_wait);

    let t_copy = Instant::now();
    let lens_mapped = lens_slice.get_mapped_range();
    let cmd_mapped = cmd_slice.get_mapped_range();
    let lens_u32: &[u32] = bytemuck::cast_slice(&lens_mapped);
    let cmd_bytes: &[u8] = &cmd_mapped;
    let mut section_cmd_lens = vec![0u32; section_count];
    section_cmd_lens.copy_from_slice(&lens_u32[..section_count]);
    let mut lens_error: Option<PDeflateError> = None;
    for (sec, &len_u32) in section_cmd_lens.iter().enumerate() {
        if len_u32 == 0xffff_ffff {
            lens_error = Some(PDeflateError::Gpu(format!(
                "gpu section encode overflow on section {}",
                sec
            )));
            break;
        }
        let cap = section_caps[sec];
        if len_u32 > cap {
            lens_error = Some(PDeflateError::Gpu(format!(
                "gpu section encode len exceeds cap on section {}",
                sec
            )));
            break;
        }
    }
    let total_cmd_len: usize = section_cmd_lens
        .iter()
        .try_fold(0usize, |acc, &v| acc.checked_add(usize::try_from(v).ok()?))
        .ok_or(PDeflateError::NumericOverflow)?;
    let mut section_cmd = Vec::with_capacity(total_cmd_len);
    for sec in 0..section_count {
        let off =
            usize::try_from(section_offsets[sec]).map_err(|_| PDeflateError::NumericOverflow)?;
        let len =
            usize::try_from(section_cmd_lens[sec]).map_err(|_| PDeflateError::NumericOverflow)?;
        if off.checked_add(len).ok_or(PDeflateError::NumericOverflow)? > cmd_bytes.len() {
            return Err(PDeflateError::Gpu(format!(
                "gpu section encode cmd bytes truncated on section {}",
                sec
            )));
        }
        section_cmd.extend_from_slice(&cmd_bytes[off..off + len]);
    }
    drop(lens_mapped);
    drop(cmd_mapped);
    readback_lens_buffer.unmap();
    readback_cmd_buffer.unmap();
    if let Some(err) = lens_error {
        return Err(err);
    }
    validate_section_commands(src.len(), section_count, &section_cmd_lens, &section_cmd)?;
    let map_copy_ms = elapsed_ms(t_copy);

    Ok(GpuSectionEncodeOutput {
        section_cmd_lens,
        section_cmd,
        profile: GpuSectionEncodeProfile {
            upload_ms,
            wait_ms,
            map_copy_ms,
            total_ms: elapsed_ms(t_total),
        },
    })
}

fn host_section_start(sec: usize, section_count: usize, len: usize) -> usize {
    if section_count == 0 || len == 0 {
        return 0;
    }
    ((sec as u128) * (len as u128) / (section_count as u128)) as usize
}

fn validate_section_commands(
    src_len: usize,
    section_count: usize,
    section_cmd_lens: &[u32],
    section_cmd: &[u8],
) -> Result<(), PDeflateError> {
    if section_cmd_lens.len() != section_count {
        return Err(PDeflateError::Gpu(
            "gpu section encode validation failed: lens count mismatch".to_string(),
        ));
    }
    let mut offset = 0usize;
    for sec in 0..section_count {
        let sec_cmd_len =
            usize::try_from(section_cmd_lens[sec]).map_err(|_| PDeflateError::NumericOverflow)?;
        let end = offset
            .checked_add(sec_cmd_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        let cmd = section_cmd.get(offset..end).ok_or_else(|| {
            PDeflateError::Gpu(format!(
                "gpu section encode validation failed: section {} range oob",
                sec
            ))
        })?;
        let s0 = host_section_start(sec, section_count, src_len);
        let s1 = host_section_start(sec + 1, section_count, src_len);
        let mut out_pos = 0usize;
        let mut cursor = 0usize;
        while out_pos < (s1 - s0) {
            let h_end = cursor
                .checked_add(2)
                .ok_or(PDeflateError::NumericOverflow)?;
            let header = cmd.get(cursor..h_end).ok_or_else(|| {
                PDeflateError::Gpu(format!(
                    "gpu section encode validation failed: section {} header truncated",
                    sec
                ))
            })?;
            let word = u16::from_le_bytes([header[0], header[1]]);
            cursor = h_end;
            let tag = word & 0x0fff;
            let len4 = usize::from(word >> 12);
            let mut len = len4;
            if len4 == 0x0f {
                let ext = *cmd.get(cursor).ok_or_else(|| {
                    PDeflateError::Gpu(format!(
                        "gpu section encode validation failed: section {} ext truncated",
                        sec
                    ))
                })?;
                cursor += 1;
                len = len.saturating_add(usize::from(ext));
            }
            if len == 0 {
                return Err(PDeflateError::Gpu(format!(
                    "gpu section encode validation failed: section {} has zero-length command",
                    sec
                )));
            }
            if tag == 0x0fff {
                let lit_end = cursor
                    .checked_add(len)
                    .ok_or(PDeflateError::NumericOverflow)?;
                if lit_end > cmd.len() {
                    return Err(PDeflateError::Gpu(format!(
                        "gpu section encode validation failed: section {} literal oob",
                        sec
                    )));
                }
                cursor = lit_end;
            }
            out_pos = out_pos.saturating_add(len);
            if out_pos > (s1 - s0) {
                return Err(PDeflateError::Gpu(format!(
                    "gpu section encode validation failed: section {} output overrun",
                    sec
                )));
            }
        }
        if out_pos != (s1 - s0) || cursor != cmd.len() {
            return Err(PDeflateError::Gpu(format!(
                "gpu section encode validation failed: section {} length mismatch",
                sec
            )));
        }
        offset = end;
    }
    if offset != section_cmd.len() {
        return Err(PDeflateError::Gpu(
            "gpu section encode validation failed: trailing command bytes".to_string(),
        ));
    }
    Ok(())
}
