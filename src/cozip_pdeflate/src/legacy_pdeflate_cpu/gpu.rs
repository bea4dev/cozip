use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::{PDeflateError, PDeflateOptions};

const WORKGROUP_SIZE: u32 = 128;
const PREFIX2_TABLE_SIZE: usize = 1 << 16;
const PACK_CHUNK_HEADER_SIZE: usize = 32;
const PACK_CHUNK_MAGIC: [u8; 4] = *b"PDF0";
const PACK_CHUNK_VERSION: u16 = 0;
const BUILD_TABLE_STAGE_COUNT: usize = 7;
const BUILD_TABLE_STAGE_QUERY_COUNT: u32 = (BUILD_TABLE_STAGE_COUNT as u32) * 2;
const SPARSE_KERNEL_STAGE_COUNT: usize = 2;
const SPARSE_KERNEL_STAGE_QUERY_COUNT: u32 = (SPARSE_KERNEL_STAGE_COUNT as u32) * 2;
const GPU_DECODE_TABLE_REPEAT_STRIDE: usize = 270;
const GPU_DECODE_SECTION_META_HEADER_WORDS: usize = 4;
const GPU_DECODE_SECTION_META_WORDS: usize = 4;
const GPU_DECODE_MAX_TABLE_ID: usize = 0x0fff - 1;
const GPU_DECODE_V2_WORKGROUP_SIZE: u32 = 64;

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

const FINALIZE_SHARDS: u32 = 128u;

var<workgroup> shard_pos_0: array<u32, 128>;
var<workgroup> shard_len_0: array<u32, 128>;
var<workgroup> shard_score_0: array<u32, 128>;
var<workgroup> shard_pos_1: array<u32, 128>;
var<workgroup> shard_len_1: array<u32, 128>;
var<workgroup> shard_score_1: array<u32, 128>;
var<workgroup> shard_pos_2: array<u32, 128>;
var<workgroup> shard_len_2: array<u32, 128>;
var<workgroup> shard_score_2: array<u32, 128>;

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

fn better_candidate(
    score: u32,
    len: u32,
    pos: u32,
    best_score: u32,
    best_len: u32,
    best_pos: u32,
) -> bool {
    if (score > best_score) {
        return true;
    }
    if (score < best_score) {
        return false;
    }
    if (len > best_len) {
        return true;
    }
    if (len < best_len) {
        return false;
    }
    return pos < best_pos;
}

@compute
@workgroup_size(128, 1, 1)
fn main(
    @builtin(global_invocation_id) gid3: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {
    let total_len = params[0];
    let sample_count = params[1];
    let sorted_count = params[2];
    let max_entry_len = params[3];
    let max_entries = params[4];
    let literal_limit = params[5];
    let min_literal_freq = params[6];
    let max_data_bytes = params[7];

    let lane = lid3.x;

    var top_score_0 = 0u;
    var top_len_0 = 0u;
    var top_pos_0 = 0u;
    var top_score_1 = 0u;
    var top_len_1 = 0u;
    var top_pos_1 = 0u;
    var top_score_2 = 0u;
    var top_len_2 = 0u;
    var top_pos_2 = 0u;

    var sid_i = lane;
    loop {
        if (sid_i >= sorted_count) {
            break;
        }
        let sid = sorted_indices[sid_i];
        sid_i = sid_i + FINALIZE_SHARDS;
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

        if (better_candidate(score, len, pos, top_score_0, top_len_0, top_pos_0)) {
            top_score_2 = top_score_1;
            top_len_2 = top_len_1;
            top_pos_2 = top_pos_1;
            top_score_1 = top_score_0;
            top_len_1 = top_len_0;
            top_pos_1 = top_pos_0;
            top_score_0 = score;
            top_len_0 = len;
            top_pos_0 = pos;
        } else if (better_candidate(score, len, pos, top_score_1, top_len_1, top_pos_1)) {
            top_score_2 = top_score_1;
            top_len_2 = top_len_1;
            top_pos_2 = top_pos_1;
            top_score_1 = score;
            top_len_1 = len;
            top_pos_1 = pos;
        } else if (better_candidate(score, len, pos, top_score_2, top_len_2, top_pos_2)) {
            top_score_2 = score;
            top_len_2 = len;
            top_pos_2 = pos;
        }
    }

    shard_pos_0[lane] = top_pos_0;
    shard_len_0[lane] = top_len_0;
    shard_score_0[lane] = top_score_0;
    shard_pos_1[lane] = top_pos_1;
    shard_len_1[lane] = top_len_1;
    shard_score_1[lane] = top_score_1;
    shard_pos_2[lane] = top_pos_2;
    shard_len_2[lane] = top_len_2;
    shard_score_2[lane] = top_score_2;
    workgroupBarrier();

    if (gid3.x != 0u || gid3.y != 0u || gid3.z != 0u || lane != 0u) {
        return;
    }

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

    var rank = 0u;
    loop {
        if (rank >= 3u || out_count >= max_entries) {
            break;
        }
        var shard = 0u;
        loop {
            if (shard >= FINALIZE_SHARDS || out_count >= max_entries) {
                break;
            }
            var score = 0u;
            var pos = 0u;
            var len = 0u;
            if (rank == 0u) {
                score = shard_score_0[shard];
                pos = shard_pos_0[shard];
                len = shard_len_0[shard];
            } else if (rank == 1u) {
                score = shard_score_1[shard];
                pos = shard_pos_1[shard];
                len = shard_len_1[shard];
            } else {
                score = shard_score_2[shard];
                pos = shard_pos_2[shard];
                len = shard_len_2[shard];
            }
            shard = shard + 1u;
            if (score == 0u) {
                continue;
            }
            if (len == 0u || len > max_entry_len || pos + len > total_len) {
                continue;
            }
            if (data_cursor + len > max_data_bytes) {
                continue;
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
        rank = rank + 1u;
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
@group(0) @binding(4) var<storage, read> table_chunk_meta_words: array<u32>;
@group(0) @binding(5) var<storage, read> table_index_words_in: array<u32>;
@group(0) @binding(6) var<storage, read> table_data_words_in: array<u32>;
@group(0) @binding(7) var<storage, read> params: array<u32>;
@group(0) @binding(8) var<storage, read_write> prefix2_first_ids: array<u32>;
@group(0) @binding(9) var<storage, read_write> table_entry_lens: array<u32>;
@group(0) @binding(10) var<storage, read_write> table_entry_offsets: array<u32>;

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
        let chunk_entries_hint = table_chunk_counts[chunk];
        let meta_base = chunk * 2u;
        let meta_count = table_chunk_meta_words[meta_base];
        var chunk_entries = chunk_entries_hint;
        if (meta_count != 0u) {
            chunk_entries = min(chunk_entries_hint, meta_count);
        }
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
            if (len == 0u) {
                break;
            }
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

fn is_valid_match(word: u32, remaining: u32, min_ref_len: u32, max_ref_len: u32) -> bool {
    return match_len_or_zero(word, remaining, min_ref_len, max_ref_len) != 0u;
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

const SECTION_TOKENIZE_SHADER: &str = r#"
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
"#;

const SECTION_TOKEN_PREFIX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> token_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> token_caps: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@group(0) @binding(3) var<storage, read> token_counts: array<u32>;
@group(0) @binding(4) var<storage, read> token_meta: array<u32>;
@group(0) @binding(5) var<storage, read_write> token_cmd_offsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_lens: array<u32>;
@group(0) @binding(7) var<storage, read> section_caps: array<u32>;

fn cmd_size_bytes(token_word: u32) -> u32 {
    let tag = (token_word >> 16u) & 0x0fffu;
    let len = token_word & 0xffffu;
    var size = 2u;
    if (len > 14u) {
        size = 3u;
    }
    if (tag == 0x0fffu) {
        size = size + len;
    }
    return size;
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let sec = gid3.x;
    let section_count = params[1];
    if (sec >= section_count) {
        return;
    }
    let count = token_counts[sec];
    if (count == 0xffffffffu) {
        out_lens[sec] = 0xffffffffu;
        return;
    }
    if (count > token_caps[sec]) {
        out_lens[sec] = 0xffffffffu;
        return;
    }

    let base = token_offsets[sec];
    let cap = section_caps[sec];
    var cursor: u32 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= count) {
            break;
        }
        let idx = base + i;
        token_cmd_offsets[idx] = cursor;
        cursor = cursor + cmd_size_bytes(token_meta[idx]);
        if (cursor > cap) {
            out_lens[sec] = 0xffffffffu;
            return;
        }
        i = i + 1u;
    }
    out_lens[sec] = cursor;
}
"#;

const SECTION_TOKEN_SCATTER_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> src_words: array<u32>;
@group(0) @binding(1) var<storage, read> token_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> token_counts: array<u32>;
@group(0) @binding(3) var<storage, read> token_meta: array<u32>;
@group(0) @binding(4) var<storage, read> token_pos: array<u32>;
@group(0) @binding(5) var<storage, read> token_cmd_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> section_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> section_caps: array<u32>;
@group(0) @binding(8) var<storage, read> params: array<u32>;
@group(0) @binding(9) var<storage, read_write> out_cmd_bytes: array<u32>;

fn load_src(idx: u32) -> u32 {
    let w = src_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn write_cmd_byte(idx: u32, value: u32) {
    out_cmd_bytes[idx] = value & 0xffu;
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let tok = gid3.x;
    let sec = gid3.y;
    let section_count = params[1];
    let src_base = params[5];
    if (sec >= section_count) {
        return;
    }
    let count = token_counts[sec];
    if (count == 0xffffffffu || tok >= count) {
        return;
    }

    let base = token_offsets[sec];
    let tidx = base + tok;
    let token_word = token_meta[tidx];
    let tag = (token_word >> 16u) & 0x0fffu;
    let len = token_word & 0xffffu;
    var header_len: u32 = 2u;
    if (len > 14u) {
        header_len = 3u;
    }
    let payload_len = select(0u, len, tag == 0x0fffu);
    let cmd_len = header_len + payload_len;

    let sec_cap = section_caps[sec];
    let local_off = token_cmd_offsets[tidx];
    if (local_off + cmd_len > sec_cap) {
        return;
    }

    let dst = section_offsets[sec] + local_off;
    var len4 = len;
    if (len4 > 14u) {
        len4 = 15u;
    }
    let header = (len4 << 12u) | tag;
    write_cmd_byte(dst, header & 0xffu);
    write_cmd_byte(dst + 1u, (header >> 8u) & 0xffu);
    if (header_len == 3u) {
        write_cmd_byte(dst + 2u, len - 15u);
    }
    if (tag == 0x0fffu) {
        let lit_base = token_pos[tidx];
        var i: u32 = 0u;
        loop {
            if (i >= len) {
                break;
            }
            write_cmd_byte(dst + header_len + i, load_src(src_base + lit_base + i));
            i = i + 1u;
        }
    }
}
"#;

const SECTION_CMD_PACK_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> out_cmd_bytes: array<u32>;
@group(0) @binding(1) var<storage, read> section_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> out_lens: array<u32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_cmd_words: array<u32>;

fn read_cmd_byte(idx: u32) -> u32 {
    return out_cmd_bytes[idx] & 0xffu;
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let word = gid3.x;
    let sec = gid3.y;
    let section_count = params[1];
    if (sec >= section_count) {
        return;
    }
    let len = out_lens[sec];
    if (len == 0xffffffffu) {
        return;
    }
    let sec_words = (len + 3u) >> 2u;
    if (word >= sec_words) {
        return;
    }

    let sec_off = section_offsets[sec];
    let local_byte = word << 2u;
    let abs_byte = sec_off + local_byte;

    let b0 = select(0u, read_cmd_byte(abs_byte), local_byte < len);
    let b1 = select(0u, read_cmd_byte(abs_byte + 1u), (local_byte + 1u) < len);
    let b2 = select(0u, read_cmd_byte(abs_byte + 2u), (local_byte + 2u) < len);
    let b3 = select(0u, read_cmd_byte(abs_byte + 3u), (local_byte + 3u) < len);
    let out_word = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);

    let dst_word = (sec_off >> 2u) + word;
    out_cmd_words[dst_word] = out_word;
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
@group(0) @binding(10) var<storage, read_write> sparse_stats_words: array<atomic<u32>, 8>;

fn low8(v: u32, s: u32) -> u32 {
    return (v >> s) & 0xffu;
}

fn sparse_probe_enabled() -> bool {
    return params[12] != 0u;
}

fn sparse_stat_add(idx: u32, value: u32) {
    if (sparse_probe_enabled()) {
        atomicAdd(&sparse_stats_words[idx], value);
    }
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

fn find_section_for_local(local: u32, section_count: u32) -> u32 {
    if (section_count == 0u) {
        sparse_stat_add(7u, 1u);
        return 0xffffffffu;
    }
    var lo = 0u;
    var hi = section_count;
    loop {
        sparse_stat_add(5u, 1u);
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
        sparse_stat_add(6u, 1u);
        return mid;
    }
    sparse_stat_add(7u, 1u);
    return 0xffffffffu;
}

fn load_section_cmd_compact_byte(local: u32, section_count: u32) -> u32 {
    if (section_count == 0u) {
        sparse_stat_add(7u, 1u);
        return 0u;
    }
    var lo = 0u;
    var hi = section_count;
    loop {
        sparse_stat_add(5u, 1u);
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
        sparse_stat_add(6u, 1u);
        return load_section_cmd_sparse_byte(src);
    }
    sparse_stat_add(7u, 1u);
    return 0u;
}

fn load_section_cmd_compact_word(local_base: u32, section_count: u32, remaining: u32) -> u32 {
    var sec_idx = find_section_for_local(local_base, section_count);
    var sec_start = 0u;
    var sec_end = 0u;
    if (sec_idx != 0xffffffffu) {
        sec_start = section_prefix_words[sec_idx];
        sec_end = sec_start + section_lens_words[sec_idx];
    }

    var b0 = 0u;
    var b1 = 0u;
    var b2 = 0u;
    var b3 = 0u;

    if (remaining > 0u) {
        let local0 = local_base;
        loop {
            if (sec_idx == 0xffffffffu || local0 < sec_end) {
                break;
            }
            if (sec_idx + 1u >= section_count) {
                sec_idx = 0xffffffffu;
                break;
            }
            sec_idx = sec_idx + 1u;
            sec_start = section_prefix_words[sec_idx];
            sec_end = sec_start + section_lens_words[sec_idx];
        }
        if (sec_idx != 0xffffffffu && local0 >= sec_start && local0 < sec_end) {
            sparse_stat_add(4u, 1u);
            b0 = load_section_cmd_sparse_byte(section_offsets_words[sec_idx] + (local0 - sec_start));
        } else {
            sparse_stat_add(7u, 1u);
        }
    }

    if (remaining > 1u) {
        let local1 = local_base + 1u;
        loop {
            if (sec_idx == 0xffffffffu || local1 < sec_end) {
                break;
            }
            if (sec_idx + 1u >= section_count) {
                sec_idx = 0xffffffffu;
                break;
            }
            sec_idx = sec_idx + 1u;
            sec_start = section_prefix_words[sec_idx];
            sec_end = sec_start + section_lens_words[sec_idx];
        }
        if (sec_idx != 0xffffffffu && local1 >= sec_start && local1 < sec_end) {
            sparse_stat_add(4u, 1u);
            b1 = load_section_cmd_sparse_byte(section_offsets_words[sec_idx] + (local1 - sec_start));
        } else {
            sparse_stat_add(7u, 1u);
        }
    }

    if (remaining > 2u) {
        let local2 = local_base + 2u;
        loop {
            if (sec_idx == 0xffffffffu || local2 < sec_end) {
                break;
            }
            if (sec_idx + 1u >= section_count) {
                sec_idx = 0xffffffffu;
                break;
            }
            sec_idx = sec_idx + 1u;
            sec_start = section_prefix_words[sec_idx];
            sec_end = sec_start + section_lens_words[sec_idx];
        }
        if (sec_idx != 0xffffffffu && local2 >= sec_start && local2 < sec_end) {
            sparse_stat_add(4u, 1u);
            b2 = load_section_cmd_sparse_byte(section_offsets_words[sec_idx] + (local2 - sec_start));
        } else {
            sparse_stat_add(7u, 1u);
        }
    }

    if (remaining > 3u) {
        let local3 = local_base + 3u;
        loop {
            if (sec_idx == 0xffffffffu || local3 < sec_end) {
                break;
            }
            if (sec_idx + 1u >= section_count) {
                sec_idx = 0xffffffffu;
                break;
            }
            sec_idx = sec_idx + 1u;
            sec_start = section_prefix_words[sec_idx];
            sec_end = sec_start + section_lens_words[sec_idx];
        }
        if (sec_idx != 0xffffffffu && local3 >= sec_start && local3 < sec_end) {
            sparse_stat_add(4u, 1u);
            b3 = load_section_cmd_sparse_byte(section_offsets_words[sec_idx] + (local3 - sec_start));
        } else {
            sparse_stat_add(7u, 1u);
        }
    }

    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
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
        sparse_stat_add(0u, 1u);
        return load_header_byte(pos, section_cmd_off);
    }
    if (pos >= table_index_off && pos < table_index_off + table_index_len) {
        sparse_stat_add(1u, 1u);
        return load_table_index_byte(pos - table_index_off);
    }
    if (pos >= table_data_off && pos < table_data_off + table_data_len) {
        sparse_stat_add(2u, 1u);
        return load_table_data_byte(pos - table_data_off);
    }
    if (pos >= section_index_off && pos < section_index_off + section_index_len) {
        sparse_stat_add(3u, 1u);
        return load_section_index_byte(pos - section_index_off);
    }
    if (pos >= section_cmd_off && pos < section_cmd_off + section_cmd_len) {
        sparse_stat_add(4u, 1u);
        return load_section_cmd_compact_byte(pos - section_cmd_off, section_count);
    }
    return 0u;
}

fn read_output_word(base: u32) -> u32 {
    let section_cmd_len = section_meta_words[0];
    let section_index_len = section_meta_words[1];
    let overflow = section_meta_words[2];
    let section_count = params[2];
    let section_index_off = params[7];
    let section_cmd_off = section_index_off + section_index_len;
    let section_cmd_end = section_cmd_off + section_cmd_len;
    let total_len = section_cmd_end;

    if (overflow == 0u
        && section_cmd_len != 0xffffffffu
        && section_index_len != 0xffffffffu
        && base >= section_cmd_off
        && base < section_cmd_end
        && section_count > 0u) {
        let local_base = base - section_cmd_off;
        let remaining = section_cmd_end - base;
        return load_section_cmd_compact_word(local_base, section_count, remaining);
    }

    var b0 = read_output_byte(base);
    var b1 = 0u;
    var b2 = 0u;
    var b3 = 0u;
    if (base + 1u < total_len) {
        b1 = read_output_byte(base + 1u);
    }
    if (base + 2u < total_len) {
        b2 = read_output_byte(base + 2u);
    }
    if (base + 3u < total_len) {
        b3 = read_output_byte(base + 3u);
    }
    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
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
    out_words[word_idx] = read_output_word(base);
}
"#;

const PACK_CHUNK_SPARSE_PREPARE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> params: array<u32>;
@group(0) @binding(1) var<storage, read> section_meta_words: array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch_words: array<u32>;
@group(0) @binding(3) var<storage, read> table_meta_words: array<u32>;

fn align_up4(v: u32) -> u32 {
    return (v + 3u) & 0xfffffffcu;
}

@compute
@workgroup_size(1, 1, 1)
fn main() {
    let table_count_hint = params[1];
    let table_index_off = params[3];
    var table_index_len = params[4];
    var table_data_len = params[6];
    if (table_count_hint != 0u) {
        let meta_count = table_meta_words[0];
        let meta_data_total_idx = 1u + table_count_hint * 2u;
        let meta_data_total = table_meta_words[meta_data_total_idx];
        if (meta_count != 0u || meta_data_total != 0u) {
            let capped_count = min(meta_count, table_count_hint);
            table_index_len = min(table_index_len, capped_count);
            table_data_len = min(table_data_len, meta_data_total);
            params[1] = capped_count;
            params[4] = table_index_len;
            params[6] = table_data_len;
        }
    }

    let table_data_off = table_index_off + table_index_len;
    let section_index_off = table_data_off + table_data_len;
    let section_index_cap_len = params[8];
    let section_cmd_cap_len = params[9];
    let section_cmd_off_cap = section_index_off + section_index_cap_len;
    let total_len_cap = section_cmd_off_cap + section_cmd_cap_len;

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

    params[5] = table_data_off;
    params[7] = section_index_off;
    params[10] = total_len;
    params[11] = total_words;
    dispatch_words[0] = groups_x;
    dispatch_words[1] = 1u;
    dispatch_words[2] = 1u;
}
"#;

const DECODE_V2_SHADER: &str = r#"
const LITERAL_TAG: u32 = 0x0fffu;
const MAX_CMD_LEN: u32 = 270u;
const TABLE_REPEAT_STRIDE: u32 = 270u;
const META_HEADER_WORDS: u32 = 4u;
const META_WORDS: u32 = 4u;

@group(0) @binding(0) var<storage, read> cmd_words: array<u32>;
@group(0) @binding(1) var<storage, read> section_meta_words: array<u32>;
@group(0) @binding(2) var<storage, read> table_words: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_words: array<u32>;
@group(0) @binding(4) var<storage, read_write> error_words: array<u32>;

fn load_cmd_u8(idx: u32) -> u32 {
    let w = cmd_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_table_u8(idx: u32) -> u32 {
    let w = table_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_cmd_u16(idx: u32) -> u32 {
    let b0 = load_cmd_u8(idx);
    let b1 = load_cmd_u8(idx + 1u);
    return b0 | (b1 << 8u);
}

fn store_out_u8(idx: u32, value: u32) {
    let word_idx = idx >> 2u;
    let shift = (idx & 3u) * 8u;
    let mask = ~(0xffu << shift);
    let cur = out_words[word_idx];
    out_words[word_idx] = (cur & mask) | ((value & 0xffu) << shift);
}

fn mark_error(sec: u32, code: u32) {
    error_words[sec] = code;
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let section_count = section_meta_words[0];
    let table_count = section_meta_words[1];
    if (gid3.y != 0u || gid3.z != 0u || gid3.x >= section_count) {
        return;
    }

    let sec = gid3.x;
    let meta_base = META_HEADER_WORDS + sec * META_WORDS;
    let cmd_start = section_meta_words[meta_base];
    let cmd_len = section_meta_words[meta_base + 1u];
    let out_start = section_meta_words[meta_base + 2u];
    let out_len = section_meta_words[meta_base + 3u];
    let cmd_end = cmd_start + cmd_len;
    let out_end = out_start + out_len;

    var cmd_cursor = cmd_start;
    var out_cursor = out_start;
    var err: u32 = 0u;
    loop {
        if (out_cursor >= out_end) {
            break;
        }
        if (cmd_cursor + 2u > cmd_end) {
            err = 1u;
            break;
        }
        let cmd = load_cmd_u16(cmd_cursor);
        cmd_cursor = cmd_cursor + 2u;

        let tag = cmd & 0x0fffu;
        let len4 = (cmd >> 12u) & 0x0fu;
        var len = len4;
        if (len4 == 0x0fu) {
            if (cmd_cursor >= cmd_end) {
                err = 2u;
                break;
            }
            len = 15u + load_cmd_u8(cmd_cursor);
            cmd_cursor = cmd_cursor + 1u;
        }
        if (len == 0u) {
            err = 3u;
            break;
        }
        if (len > MAX_CMD_LEN) {
            err = 4u;
            break;
        }
        if (len > (out_end - out_cursor)) {
            err = 5u;
            break;
        }

        if (tag == LITERAL_TAG) {
            if (len > (cmd_end - cmd_cursor)) {
                err = 6u;
                break;
            }
            var i = 0u;
            loop {
                if (i >= len) {
                    break;
                }
                let b = load_cmd_u8(cmd_cursor + i);
                store_out_u8(out_cursor + i, b);
                i = i + 1u;
            }
            cmd_cursor = cmd_cursor + len;
            out_cursor = out_cursor + len;
            continue;
        }

        if (tag >= table_count) {
            err = 7u;
            break;
        }
        if (len < 3u) {
            err = 8u;
            break;
        }

        let table_base = tag * TABLE_REPEAT_STRIDE;
        var j = 0u;
        loop {
            if (j >= len) {
                break;
            }
            let b = load_table_u8(table_base + j);
            store_out_u8(out_cursor + j, b);
            j = j + 1u;
        }
        out_cursor = out_cursor + len;
    }

    if (err == 0u && cmd_cursor != cmd_end) {
        err = 9u;
    }
    mark_error(sec, err);
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
    pub(crate) pack_alloc_setup_ms: f64,
    pub(crate) pack_resolve_sizes_ms: f64,
    pub(crate) pack_resolve_scan_ms: f64,
    pub(crate) pack_resolve_readback_setup_ms: f64,
    pub(crate) pack_resolve_submit_ms: f64,
    pub(crate) pack_resolve_map_wait_ms: f64,
    pub(crate) pack_resolve_parse_ms: f64,
    pub(crate) pack_src_copy_ms: f64,
    pub(crate) pack_metadata_loop_ms: f64,
    pub(crate) pack_host_copy_ms: f64,
    pub(crate) pack_device_copy_plan_ms: f64,
    pub(crate) pack_finalize_ms: f64,
    pub(crate) scratch_acquire_ms: f64,
    pub(crate) match_table_copy_ms: f64,
    pub(crate) match_prepare_dispatch_ms: f64,
    pub(crate) match_kernel_dispatch_ms: f64,
    pub(crate) match_submit_ms: f64,
    pub(crate) section_setup_ms: f64,
    pub(crate) section_pass_dispatch_ms: f64,
    pub(crate) section_tokenize_dispatch_ms: f64,
    pub(crate) section_prefix_dispatch_ms: f64,
    pub(crate) section_scatter_dispatch_ms: f64,
    pub(crate) section_pack_dispatch_ms: f64,
    pub(crate) section_meta_dispatch_ms: f64,
    pub(crate) section_copy_dispatch_ms: f64,
    pub(crate) section_submit_ms: f64,
    pub(crate) section_tokenize_wait_ms: f64,
    pub(crate) section_prefix_wait_ms: f64,
    pub(crate) section_scatter_wait_ms: f64,
    pub(crate) section_pack_wait_ms: f64,
    pub(crate) section_meta_wait_ms: f64,
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

#[derive(Debug, Clone, Copy, Default)]
struct ResolvePackedTableSizesBatchProfile {
    unresolved_count: usize,
    scan_ms: f64,
    readback_setup_ms: f64,
    submit_ms: f64,
    submit_done_wait_ms: f64,
    map_callback_wait_ms: f64,
    map_wait_ms: f64,
    pre_wait_latest_build_submit_ms: f64,
    latest_build_submit_seq: u64,
    readback_submit_seq: u64,
    parse_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct PackMatchBatchInputsProfile {
    total_ms: f64,
    alloc_setup_ms: f64,
    resolve_sizes_ms: f64,
    resolve_scan_ms: f64,
    resolve_readback_setup_ms: f64,
    resolve_submit_ms: f64,
    resolve_map_wait_ms: f64,
    resolve_parse_ms: f64,
    src_copy_ms: f64,
    metadata_loop_ms: f64,
    host_copy_ms: f64,
    device_copy_plan_ms: f64,
    finalize_ms: f64,
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
    pub(crate) out_lens_readback_buffer: Arc<wgpu::Buffer>,
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
        release_batch_scratch(r, scratch);
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
    pub(crate) table_size_resolve_ms: f64,
    pub(crate) prepare_misc_ms: f64,
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

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuTableBuildBreakdown {
    pub(crate) total_ms: f64,
    pub(crate) pre_resource_ms: f64,
    pub(crate) resource_setup_ms: f64,
    pub(crate) upload_ms: f64,
    pub(crate) encode_pass1_ms: f64,
    pub(crate) submit_pass1_ms: f64,
    pub(crate) encode_pass2_ms: f64,
    pub(crate) submit_pass2_ms: f64,
    pub(crate) stage_probe_submit_ms: f64,
    pub(crate) stage_probe_wait_ms: f64,
    pub(crate) stage_probe_parse_ms: f64,
    pub(crate) stage_gpu_sum_ms: f64,
    pub(crate) tail_ms: f64,
    pub(crate) other_ms: f64,
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
    pub(crate) build_id: u64,
    pub(crate) build_submit_seq: u64,
    pub(crate) build_submit_index: Option<wgpu::SubmissionIndex>,
    pub(crate) build_breakdown: GpuTableBuildBreakdown,
}

struct GpuMatchRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    supports_timestamp_query: bool,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    pack_chunk_bind_group_layout: wgpu::BindGroupLayout,
    pack_chunk_pipeline: wgpu::ComputePipeline,
    pack_sparse_prepare_bind_group_layout: wgpu::BindGroupLayout,
    pack_sparse_prepare_pipeline: wgpu::ComputePipeline,
    pack_sparse_bind_group_layout: wgpu::BindGroupLayout,
    pack_sparse_pipeline: wgpu::ComputePipeline,
    pack_sparse_stats_zero_buffer: wgpu::Buffer,
    section_encode_bind_group_layout: wgpu::BindGroupLayout,
    section_encode_pipeline: wgpu::ComputePipeline,
    section_tokenize_bind_group_layout: wgpu::BindGroupLayout,
    section_tokenize_pipeline: wgpu::ComputePipeline,
    section_prefix_bind_group_layout: wgpu::BindGroupLayout,
    section_prefix_pipeline: wgpu::ComputePipeline,
    section_scatter_bind_group_layout: wgpu::BindGroupLayout,
    section_scatter_pipeline: wgpu::ComputePipeline,
    section_cmd_pack_bind_group_layout: wgpu::BindGroupLayout,
    section_cmd_pack_pipeline: wgpu::ComputePipeline,
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
    decode_v2_bind_group_layout: wgpu::BindGroupLayout,
    decode_v2_pipeline: wgpu::ComputePipeline,
    max_storage_binding_size: u64,
    scratch_pool: Mutex<Vec<GpuBatchScratch>>,
    scratch_hot: Mutex<Option<GpuBatchScratch>>,
    sparse_pack_pool: Mutex<Vec<GpuSparsePackScratch>>,
    sparse_pack_hot: Mutex<Option<GpuSparsePackScratch>>,
    decode_slot_pool: Mutex<GpuDecodeSlotPool>,
}

struct GpuBatchScratch {
    src_cap_bytes: u64,
    chunk_starts_cap_bytes: u64,
    table_base_cap_bytes: u64,
    table_count_cap_bytes: u64,
    table_index_offsets_cap_bytes: u64,
    table_data_offsets_cap_bytes: u64,
    table_meta_cap_bytes: u64,
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
    table_meta_buffer: wgpu::Buffer,
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
    token_cap_count: u64,
    section_index_cap_bytes: u64,
    section_offsets_buffer: Arc<wgpu::Buffer>,
    section_caps_buffer: Arc<wgpu::Buffer>,
    section_token_offsets_buffer: Arc<wgpu::Buffer>,
    section_token_caps_buffer: Arc<wgpu::Buffer>,
    section_params_buffer: Arc<wgpu::Buffer>,
    out_lens_buffer: Arc<wgpu::Buffer>,
    out_lens_readback_buffer: Arc<wgpu::Buffer>,
    section_token_counts_buffer: Arc<wgpu::Buffer>,
    section_token_meta_buffer: Arc<wgpu::Buffer>,
    section_token_pos_buffer: Arc<wgpu::Buffer>,
    section_token_cmd_offsets_buffer: Arc<wgpu::Buffer>,
    out_cmd_byte_buffer: Arc<wgpu::Buffer>,
    section_prefix_buffer: Arc<wgpu::Buffer>,
    section_index_buffer: Arc<wgpu::Buffer>,
    section_meta_buffer: Arc<wgpu::Buffer>,
    out_cmd_buffer: Arc<wgpu::Buffer>,
    readback_meta_buffer: Arc<wgpu::Buffer>,
    bind_group: wgpu::BindGroup,
    tokenize_bind_group: wgpu::BindGroup,
    prefix_bind_group: wgpu::BindGroup,
    scatter_bind_group: wgpu::BindGroup,
    pack_bind_group: wgpu::BindGroup,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuDecodeSlotClass {
    Normal,
    Large,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuDecodeSlotState {
    Free,
    Inflight,
    Ready,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct GpuDecodeSlotCaps {
    cmd_cap_bytes: u64,
    section_meta_cap: usize,
    table_cap_bytes: u64,
    out_cap_bytes: u64,
    error_cap_words: usize,
}

impl GpuDecodeSlotCaps {
    fn fits(&self, req: GpuDecodeSlotRequirement) -> bool {
        req.cmd_bytes <= self.cmd_cap_bytes
            && req.section_meta_count <= self.section_meta_cap
            && req.table_bytes <= self.table_cap_bytes
            && req.out_bytes <= self.out_cap_bytes
            && req.error_words <= self.error_cap_words
    }

    fn max_with(self, rhs: GpuDecodeSlotCaps) -> GpuDecodeSlotCaps {
        GpuDecodeSlotCaps {
            cmd_cap_bytes: self.cmd_cap_bytes.max(rhs.cmd_cap_bytes),
            section_meta_cap: self.section_meta_cap.max(rhs.section_meta_cap),
            table_cap_bytes: self.table_cap_bytes.max(rhs.table_cap_bytes),
            out_cap_bytes: self.out_cap_bytes.max(rhs.out_cap_bytes),
            error_cap_words: self.error_cap_words.max(rhs.error_cap_words),
        }
    }
}

struct GpuDecodeSlotBuffers {
    cmd_buffer: wgpu::Buffer,
    section_meta_buffer: wgpu::Buffer,
    table_buffer: wgpu::Buffer,
    out_buffer: wgpu::Buffer,
    out_readback_buffer: wgpu::Buffer,
    error_buffer: wgpu::Buffer,
}

struct GpuDecodeSlotEntry {
    slot_index: usize,
    generation: u64,
    state: GpuDecodeSlotState,
    caps: GpuDecodeSlotCaps,
    buffers: GpuDecodeSlotBuffers,
    decode_bind_group: wgpu::BindGroup,
}

#[derive(Default)]
struct GpuDecodeSlotGroup {
    caps: GpuDecodeSlotCaps,
    slots: Vec<GpuDecodeSlotEntry>,
}

#[derive(Default)]
struct GpuDecodeSlotPool {
    normal: GpuDecodeSlotGroup,
    large: GpuDecodeSlotGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct GpuDecodeSlotRequirement {
    cmd_bytes: u64,
    section_meta_count: usize,
    table_bytes: u64,
    out_bytes: u64,
    error_words: usize,
}

#[derive(Debug, Clone, Copy)]
struct GpuDecodePayloadLayout {
    table_count: usize,
    table_index_start: usize,
    table_data_start: usize,
    table_end: usize,
    cmd_start: usize,
}

#[derive(Debug, Clone, Copy)]
struct PreparedGpuDecodeJob<'a> {
    job: &'a GpuDecodeJob<'a>,
    req: GpuDecodeSlotRequirement,
    layout: GpuDecodePayloadLayout,
}

struct PendingGpuDecodeMap {
    out_data_copy_len: u64,
    out_total_copy_len: u64,
    out_len: usize,
    error_words: usize,
    submit_seq: u64,
    submit_ms: f64,
    kernel_timestamp_ms: Option<f64>,
    ts_readback_buffer: Option<wgpu::Buffer>,
    ts_map_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    submit_at: Instant,
    completion_rx: mpsc::Receiver<()>,
    map_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

#[derive(Debug, Clone, Copy, Default)]
struct GpuDecodeV2WaitProfile {
    submit_seq: u64,
    submit_ms: f64,
    submit_done_wait_ms: f64,
    map_callback_wait_ms: f64,
    map_copy_ms: f64,
    kernel_timestamp_ms: f64,
    queue_stall_est_ms: f64,
}

struct InflightGpuDecodeSubmission<'a> {
    prepared: PreparedGpuDecodeJob<'a>,
    class: GpuDecodeSlotClass,
    slot_idx: usize,
    slot_handle: GpuDecodeSlot,
    pending: PendingGpuDecodeMap,
    map_requested_at: Option<Instant>,
    completion_done: bool,
    map_done: Option<Result<(), PDeflateError>>,
    wait_profile: GpuDecodeV2WaitProfile,
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
    table_meta_bytes: u64,
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
const GPU_DECODE_LARGE_SLOT_INDEX_BASE: usize = 1 << 30;
// Command stream cap heuristic for GPU section encode.
// If this underestimates a pathological section, kernel marks overflow and
// caller falls back to CPU for correctness.
const GPU_SECTION_CMD_CAP_MULTIPLIER: usize = 2;
const GPU_SECTION_CMD_CAP_PAD: usize = 64;

static GPU_RUNTIME: OnceLock<Result<GpuMatchRuntime, String>> = OnceLock::new();
static GPU_SUBMIT_STREAM_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
static BUILD_TABLE_STAGE_PROBE_SEQ: AtomicUsize = AtomicUsize::new(0);
static BUILD_TABLE_STAGE_PROBE_UNSUPPORTED_LOGGED: OnceLock<()> = OnceLock::new();
static SPARSE_KERNEL_TS_PROBE_SEQ: AtomicUsize = AtomicUsize::new(0);
static SPARSE_KERNEL_TS_PROBE_UNSUPPORTED_LOGGED: OnceLock<()> = OnceLock::new();
static BUILD_TABLE_BREAKDOWN_SEQ: AtomicUsize = AtomicUsize::new(0);
static RESOLVE_MAP_WAIT_PROBE_SEQ: AtomicUsize = AtomicUsize::new(0);
static SPARSE_LENS_WAIT_PROBE_SEQ: AtomicUsize = AtomicUsize::new(0);
static GPU_DECODE_V2_WAIT_PROBE_SEQ: AtomicUsize = AtomicUsize::new(0);
static GPU_DECODE_V2_KERNEL_TS_UNSUPPORTED_LOGGED: OnceLock<()> = OnceLock::new();
static BUILD_TABLE_DEVICE_SEQ: AtomicU64 = AtomicU64::new(0);
static GPU_SUBMIT_SEQ: AtomicU64 = AtomicU64::new(0);

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim();
            !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    }
}

fn sparse_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("COZIP_PDEFLATE_PROFILE_SPARSE_PROBE"))
}

fn queue_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("COZIP_PDEFLATE_PROFILE_QUEUE_PROBE"))
}

fn table_stage_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_TABLE_STAGE_PROBE") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        },
    )
}

fn table_build_breakdown_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_TABLE_BUILD_BREAKDOWN") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        },
    )
}

fn resolve_map_wait_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_RESOLVE_MAP_WAIT_PROBE") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        },
    )
}

fn resolve_wait_attribution_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("COZIP_PDEFLATE_PROFILE_RESOLVE_WAIT_ATTRIBUTION"))
}

fn sparse_lens_wait_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_SPARSE_LENS_WAIT_PROBE") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        },
    )
}

fn sparse_wait_attribution_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("COZIP_PDEFLATE_PROFILE_SPARSE_WAIT_ATTRIBUTION"))
}

fn sparse_kernel_ts_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_PROFILE_SPARSE_KERNEL_TS") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        },
    )
}

fn gpu_decode_v2_wait_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        match std::env::var("COZIP_PDEFLATE_PROFILE_GPU_DECODE_V2_WAIT_PROBE") {
            Ok(v) => {
                let s = v.trim();
                !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
            }
            Err(_) => env_flag_enabled("COZIP_PDEFLATE_PROFILE"),
        }
    })
}

fn gpu_decode_v2_wait_probe_should_log(seq: usize) -> bool {
    seq < 8 || seq % 64 == 0
}

fn sparse_inflight_submit_chunks() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(
        || match std::env::var("COZIP_PDEFLATE_SPARSE_INFLIGHT_SUBMIT_CHUNKS") {
            Ok(v) => v
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|n| *n > 0)
                .unwrap_or(usize::MAX),
            Err(_) => 4,
        },
    )
}

fn table_stage_probe_should_log(seq: usize) -> bool {
    seq < 8 || seq % 64 == 0
}

fn next_gpu_submit_seq() -> u64 {
    GPU_SUBMIT_SEQ
        .fetch_add(1, Ordering::Relaxed)
        .saturating_add(1)
}

fn wait_for_queue_done(queue: &wgpu::Queue, device: &wgpu::Device) -> (f64, u64) {
    let t_wait = Instant::now();
    let (done_tx, done_rx) = mpsc::channel();
    queue.on_submitted_work_done(move || {
        let _ = done_tx.send(());
    });
    let mut poll_calls = 0u64;
    let mut done = false;
    while !done {
        device.poll(wgpu::Maintain::Wait);
        poll_calls = poll_calls.saturating_add(1);
        match done_rx.try_recv() {
            Ok(_) | Err(mpsc::TryRecvError::Disconnected) => done = true,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
    (elapsed_ms(t_wait), poll_calls)
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

#[inline]
fn estimate_token_cap_count(cmd_cap_bytes: u64) -> u64 {
    (cmd_cap_bytes / 2).max(1)
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
        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut self.words);
        dst[self.len_bytes..new_len].copy_from_slice(bytes);
        self.len_bytes = new_len;
    }

    fn write_bytes_at(&mut self, offset: usize, bytes: &[u8]) -> Result<(), PDeflateError> {
        let end = offset
            .checked_add(bytes.len())
            .ok_or(PDeflateError::NumericOverflow)?;
        if end > self.len_bytes {
            return Err(PDeflateError::NumericOverflow);
        }
        let dst = bytemuck::cast_slice_mut::<u32, u8>(&mut self.words);
        dst[offset..end].copy_from_slice(bytes);
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
            && self.table_meta_cap_bytes >= caps.table_meta_bytes
            && self.table_index_cap_bytes >= caps.table_index_bytes
            && self.prefix_first_cap_bytes >= caps.prefix_first_bytes
            && self.table_lens_cap_bytes >= caps.table_lens_bytes
            && self.table_offsets_cap_bytes >= caps.table_offsets_bytes
            && self.table_data_cap_bytes >= caps.table_data_bytes
            && self.prep_params_cap_bytes >= caps.prep_params_bytes
            && self.params_cap_bytes >= caps.params_bytes
            && self.out_cap_bytes >= caps.out_bytes
    }

    fn covers(&self, other: &Self) -> bool {
        self.src_cap_bytes >= other.src_cap_bytes
            && self.chunk_starts_cap_bytes >= other.chunk_starts_cap_bytes
            && self.table_base_cap_bytes >= other.table_base_cap_bytes
            && self.table_count_cap_bytes >= other.table_count_cap_bytes
            && self.table_index_offsets_cap_bytes >= other.table_index_offsets_cap_bytes
            && self.table_data_offsets_cap_bytes >= other.table_data_offsets_cap_bytes
            && self.table_meta_cap_bytes >= other.table_meta_cap_bytes
            && self.table_index_cap_bytes >= other.table_index_cap_bytes
            && self.prefix_first_cap_bytes >= other.prefix_first_cap_bytes
            && self.table_lens_cap_bytes >= other.table_lens_cap_bytes
            && self.table_offsets_cap_bytes >= other.table_offsets_cap_bytes
            && self.table_data_cap_bytes >= other.table_data_cap_bytes
            && self.prep_params_cap_bytes >= other.prep_params_cap_bytes
            && self.params_cap_bytes >= other.params_cap_bytes
            && self.out_cap_bytes >= other.out_cap_bytes
    }

    fn new(runtime: &GpuMatchRuntime, required: GpuBatchScratchCaps) -> Self {
        let src_cap_bytes = round_capacity_bytes(required.src_bytes);
        let chunk_starts_cap_bytes = round_capacity_bytes(required.chunk_starts_bytes);
        let table_base_cap_bytes = round_capacity_bytes(required.table_base_bytes);
        let table_count_cap_bytes = round_capacity_bytes(required.table_count_bytes);
        let table_index_offsets_cap_bytes =
            round_capacity_bytes(required.table_index_offsets_bytes);
        let table_data_offsets_cap_bytes = round_capacity_bytes(required.table_data_offsets_bytes);
        let table_meta_cap_bytes = round_capacity_bytes(required.table_meta_bytes);
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
        let table_meta_buffer = storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-table-meta",
            table_meta_cap_bytes,
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
                        resource: table_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: table_index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: table_data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: prep_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: prefix_first_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: table_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
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
            table_meta_cap_bytes,
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
            table_meta_buffer,
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
        let token_cap_count = estimate_token_cap_count(out_cmd_bytes);
        let token_buf_bytes = token_cap_count.saturating_mul(4).max(4);
        let out_cmd_byte_buf_bytes = out_cmd_bytes
            .checked_mul(4)
            .ok_or(PDeflateError::NumericOverflow)?
            .max(4);
        if token_buf_bytes > runtime.max_storage_binding_size
            || out_cmd_byte_buf_bytes > runtime.max_storage_binding_size
        {
            return Err(PDeflateError::Gpu(
                "gpu section encode staging buffer too large".to_string(),
            ));
        }
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
        let section_token_offsets_buffer = Arc::new(storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-se-token-offs",
            section_offsets_bytes.max(4),
        ));
        let section_token_caps_buffer = Arc::new(storage_upload_buffer(
            &runtime.device,
            "cozip-pdeflate-se-token-caps",
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
        let out_lens_readback_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-out-lens-readback"),
                size: out_lens_bytes.max(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
        let section_token_counts_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-token-counts"),
                size: out_lens_bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        let section_token_meta_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-token-meta"),
                size: token_buf_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        let section_token_pos_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-token-pos"),
                size: token_buf_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        let section_token_cmd_offsets_buffer =
            Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cozip-pdeflate-se-token-cmd-offs"),
                size: token_buf_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        let out_cmd_byte_buffer = Arc::new(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-se-out-cmd-byte"),
            size: out_cmd_byte_buf_bytes,
            usage: wgpu::BufferUsages::STORAGE,
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
        let tokenize_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-tokenize-bg"),
                layout: &runtime.section_tokenize_bind_group_layout,
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
                        resource: section_token_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_token_caps_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: section_token_counts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: section_token_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: section_token_pos_buffer.as_entire_binding(),
                    },
                ],
            });
        let prefix_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-prefix-bg"),
                layout: &runtime.section_prefix_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: section_token_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: section_token_caps_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_token_counts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: section_token_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: section_token_cmd_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: out_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: section_caps_buffer.as_entire_binding(),
                    },
                ],
            });
        let scatter_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-scatter-bg"),
                layout: &runtime.section_scatter_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: match_scratch.src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: section_token_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: section_token_counts_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_token_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: section_token_pos_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: section_token_cmd_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: section_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: section_caps_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: out_cmd_byte_buffer.as_entire_binding(),
                    },
                ],
            });
        let pack_bind_group = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cozip-pdeflate-se-pack-bg"),
                layout: &runtime.section_cmd_pack_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: out_cmd_byte_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: section_offsets_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_lens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: section_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
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
            token_cap_count,
            section_index_cap_bytes: section_index_cap_bytes.max(4),
            section_offsets_buffer,
            section_caps_buffer,
            section_token_offsets_buffer,
            section_token_caps_buffer,
            section_params_buffer,
            out_lens_buffer,
            out_lens_readback_buffer,
            section_token_counts_buffer,
            section_token_meta_buffer,
            section_token_pos_buffer,
            section_token_cmd_offsets_buffer,
            out_cmd_byte_buffer,
            section_prefix_buffer,
            section_index_buffer,
            section_meta_buffer,
            out_cmd_buffer,
            readback_meta_buffer,
            bind_group,
            tokenize_bind_group,
            prefix_bind_group,
            scatter_bind_group,
            pack_bind_group,
            meta_bind_group,
        })
    }
}

impl GpuSparsePackScratch {
    fn fits(&self, caps: GpuSparsePackScratchCaps) -> bool {
        self.params_cap_bytes >= caps.params_bytes && self.out_cap_bytes >= caps.out_bytes
    }

    fn covers(&self, other: &Self) -> bool {
        self.params_cap_bytes >= other.params_cap_bytes && self.out_cap_bytes >= other.out_cap_bytes
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
    let adapter_features = adapter.features();
    let mut required_features = wgpu::Features::empty();
    if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("cozip-pdeflate-gpu-device"),
            required_features,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
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
    let section_tokenize_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-tokenize-bgl"),
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
            ],
        });
    let section_tokenize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-tokenize-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_TOKENIZE_SHADER.into()),
    });
    let section_tokenize_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-tokenize-pl"),
        bind_group_layouts: &[&section_tokenize_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_tokenize_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-section-tokenize-pipeline"),
            layout: Some(&section_tokenize_layout),
            module: &section_tokenize_shader,
            entry_point: "main",
        });
    let section_prefix_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-prefix-bgl"),
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
            ],
        });
    let section_prefix_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-prefix-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_TOKEN_PREFIX_SHADER.into()),
    });
    let section_prefix_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-prefix-pl"),
        bind_group_layouts: &[&section_prefix_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_prefix_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-section-prefix-pipeline"),
            layout: Some(&section_prefix_layout),
            module: &section_prefix_shader,
            entry_point: "main",
        });
    let section_scatter_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-scatter-bgl"),
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
    let section_scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-scatter-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_TOKEN_SCATTER_SHADER.into()),
    });
    let section_scatter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-scatter-pl"),
        bind_group_layouts: &[&section_scatter_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_scatter_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-section-scatter-pipeline"),
            layout: Some(&section_scatter_layout),
            module: &section_scatter_shader,
            entry_point: "main",
        });
    let section_cmd_pack_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-section-cmd-pack-bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let section_cmd_pack_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-section-cmd-pack-shader"),
        source: wgpu::ShaderSource::Wgsl(SECTION_CMD_PACK_SHADER.into()),
    });
    let section_cmd_pack_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cozip-pdeflate-section-cmd-pack-pl"),
        bind_group_layouts: &[&section_cmd_pack_bind_group_layout],
        push_constant_ranges: &[],
    });
    let section_cmd_pack_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-pdeflate-section-cmd-pack-pipeline"),
            layout: Some(&section_cmd_pack_layout),
            module: &section_cmd_pack_shader,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
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
    let decode_v2_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-pdeflate-decode-v2-bgl"),
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
    let decode_v2_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cozip-pdeflate-decode-v2-shader"),
        source: wgpu::ShaderSource::Wgsl(DECODE_V2_SHADER.into()),
    });
    let decode_v2_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-pdeflate-decode-v2-pl"),
            bind_group_layouts: &[&decode_v2_bind_group_layout],
            push_constant_ranges: &[],
        });
    let decode_v2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cozip-pdeflate-decode-v2-pipeline"),
        layout: Some(&decode_v2_pipeline_layout),
        module: &decode_v2_shader,
        entry_point: "main",
    });
    let pack_sparse_stats_zero_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-pack-sparse-stats-zero"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let max_storage_binding_size = u64::from(device.limits().max_storage_buffer_binding_size);

    Ok(GpuMatchRuntime {
        device,
        queue,
        supports_timestamp_query: required_features.contains(wgpu::Features::TIMESTAMP_QUERY),
        bind_group_layout,
        pipeline,
        pack_chunk_bind_group_layout,
        pack_chunk_pipeline,
        pack_sparse_prepare_bind_group_layout,
        pack_sparse_prepare_pipeline,
        pack_sparse_bind_group_layout,
        pack_sparse_pipeline,
        pack_sparse_stats_zero_buffer,
        section_encode_bind_group_layout,
        section_encode_pipeline,
        section_tokenize_bind_group_layout,
        section_tokenize_pipeline,
        section_prefix_bind_group_layout,
        section_prefix_pipeline,
        section_scatter_bind_group_layout,
        section_scatter_pipeline,
        section_cmd_pack_bind_group_layout,
        section_cmd_pack_pipeline,
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
        decode_v2_bind_group_layout,
        decode_v2_pipeline,
        max_storage_binding_size,
        scratch_pool: Mutex::new(Vec::new()),
        scratch_hot: Mutex::new(None),
        sparse_pack_pool: Mutex::new(Vec::new()),
        sparse_pack_hot: Mutex::new(None),
        decode_slot_pool: Mutex::new(GpuDecodeSlotPool::default()),
    })
}

fn runtime() -> Result<&'static GpuMatchRuntime, PDeflateError> {
    match GPU_RUNTIME.get_or_init(init_runtime) {
        Ok(r) => Ok(r),
        Err(e) => Err(PDeflateError::Gpu(e.clone())),
    }
}

fn acquire_batch_scratch(
    r: &GpuMatchRuntime,
    required_caps: GpuBatchScratchCaps,
) -> Result<GpuBatchScratch, PDeflateError> {
    if let Ok(mut hot) = r.scratch_hot.lock() {
        if let Some(scratch) = hot.take() {
            if scratch.fits(required_caps) {
                return Ok(scratch);
            }
            if let Ok(mut pool) = r.scratch_pool.lock() {
                if pool.len() < GPU_SCRATCH_POOL_LIMIT {
                    pool.push(scratch);
                }
            }
        }
    }
    let mut pool = r
        .scratch_pool
        .lock()
        .map_err(|_| PDeflateError::Gpu("gpu scratch pool mutex poisoned".to_string()))?;
    if let Some(pos) = pool.iter().position(|s| s.fits(required_caps)) {
        Ok(pool.swap_remove(pos))
    } else {
        Ok(GpuBatchScratch::new(r, required_caps))
    }
}

fn release_batch_scratch(r: &GpuMatchRuntime, scratch: GpuBatchScratch) {
    let mut spill = Some(scratch);
    if let Ok(mut hot) = r.scratch_hot.lock() {
        match hot.take() {
            None => {
                *hot = spill.take();
            }
            Some(existing) => {
                if let Some(candidate) = spill.take() {
                    if candidate.covers(&existing) {
                        *hot = Some(candidate);
                        spill = Some(existing);
                    } else {
                        *hot = Some(existing);
                        spill = Some(candidate);
                    }
                } else {
                    *hot = Some(existing);
                }
            }
        }
    }
    if let Some(s) = spill {
        if let Ok(mut pool) = r.scratch_pool.lock() {
            if pool.len() < GPU_SCRATCH_POOL_LIMIT {
                pool.push(s);
            }
        }
    }
}

fn acquire_sparse_pack_scratch(
    r: &GpuMatchRuntime,
    caps: GpuSparsePackScratchCaps,
) -> Result<GpuSparsePackScratch, PDeflateError> {
    if let Ok(mut hot) = r.sparse_pack_hot.lock() {
        if let Some(scratch) = hot.take() {
            if scratch.fits(caps) {
                return Ok(scratch);
            }
            if let Ok(mut pool) = r.sparse_pack_pool.lock() {
                if pool.len() < GPU_SPARSE_PACK_POOL_LIMIT {
                    pool.push(scratch);
                }
            }
        }
    }
    let mut pool = r
        .sparse_pack_pool
        .lock()
        .map_err(|_| PDeflateError::Gpu("gpu sparse pack pool mutex poisoned".to_string()))?;
    if let Some(pos) = pool.iter().position(|s| s.fits(caps)) {
        Ok(pool.swap_remove(pos))
    } else {
        Ok(GpuSparsePackScratch::new(r, caps))
    }
}

fn release_sparse_pack_scratch(r: &GpuMatchRuntime, scratch: GpuSparsePackScratch) {
    let mut spill = Some(scratch);
    if let Ok(mut hot) = r.sparse_pack_hot.lock() {
        match hot.take() {
            None => {
                *hot = spill.take();
            }
            Some(existing) => {
                if let Some(candidate) = spill.take() {
                    if candidate.covers(&existing) {
                        *hot = Some(candidate);
                        spill = Some(existing);
                    } else {
                        *hot = Some(existing);
                        spill = Some(candidate);
                    }
                } else {
                    *hot = Some(existing);
                }
            }
        }
    }
    if let Some(s) = spill {
        if let Ok(mut pool) = r.sparse_pack_pool.lock() {
            if pool.len() < GPU_SPARSE_PACK_POOL_LIMIT {
                pool.push(s);
            }
        }
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
    let table_size_resolve_ms = 0.0_f64;
    let mut prepared = Vec::<SparsePackPrepared>::with_capacity(inputs.len());
    for input in inputs {
        let (table_count, table_index_len, table_data_len) = if input.table.sizes_known
            || input.table.table_count > 0
            || input.table.table_index_len > 0
            || input.table.table_data_len > 0
        {
            (
                input.table.table_count,
                input.table.table_index_len,
                input.table.table_data_len,
            )
        } else {
            (
                input.table.max_entries,
                input.table.max_entries,
                input.table.table_data_bytes_cap,
            )
        };
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
    let prepare_misc_ms = (prepare_ms - table_size_resolve_ms).max(0.0);

    let t_scratch_acquire = Instant::now();
    let mut acquired_scratch = Vec::<GpuSparsePackScratch>::with_capacity(prepared.len());
    for prep in prepared.iter() {
        let caps = GpuSparsePackScratchCaps {
            params_bytes: 52,
            out_bytes: prep.total_bytes_cap,
        };
        acquired_scratch.push(acquire_sparse_pack_scratch(r, caps)?);
    }
    let scratch_acquire_ms = elapsed_ms(t_scratch_acquire);

    let sparse_probe = sparse_probe_enabled();
    let sparse_lens_wait_probe = sparse_lens_wait_probe_enabled();
    let sparse_wait_attr_probe = sparse_wait_attribution_probe_enabled();
    let sparse_kernel_ts_probe = sparse_kernel_ts_probe_enabled() && r.supports_timestamp_query;
    if sparse_kernel_ts_probe_enabled() && !r.supports_timestamp_query {
        let _ = SPARSE_KERNEL_TS_PROBE_UNSUPPORTED_LOGGED.get_or_init(|| {
            eprintln!(
                "[cozip_pdeflate][timing][sparse-kernel-ts] status=disabled reason=timestamp_query_not_supported"
            );
        });
    }
    let mut sparse_prebuild_build_id_min = u64::MAX;
    let mut sparse_prebuild_build_id_max = 0u64;
    let mut sparse_prebuild_submit_seq_min = u64::MAX;
    let mut sparse_prebuild_submit_seq_max = 0u64;
    let mut sparse_latest_build_submit: Option<(u64, u64, wgpu::SubmissionIndex)> = None;
    let mut sparse_build_ids = Vec::<u64>::with_capacity(inputs.len());
    for input in inputs {
        let table = input.table;
        if table.build_id != 0 {
            sparse_prebuild_build_id_min = sparse_prebuild_build_id_min.min(table.build_id);
            sparse_prebuild_build_id_max = sparse_prebuild_build_id_max.max(table.build_id);
            sparse_build_ids.push(table.build_id);
        }
        if table.build_submit_seq != 0 {
            sparse_prebuild_submit_seq_min =
                sparse_prebuild_submit_seq_min.min(table.build_submit_seq);
            sparse_prebuild_submit_seq_max =
                sparse_prebuild_submit_seq_max.max(table.build_submit_seq);
        }
        if let Some(submit_index) = table.build_submit_index.clone() {
            let should_replace = sparse_latest_build_submit
                .as_ref()
                .map(|(seq, _, _)| table.build_submit_seq > *seq)
                .unwrap_or(true);
            if should_replace {
                sparse_latest_build_submit =
                    Some((table.build_submit_seq, table.build_id, submit_index));
            }
        }
    }
    let sparse_stats_buffer = if sparse_probe {
        let buf = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-stats"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let zeros = [0u8; 32];
        r.queue.write_buffer(&buf, 0, &zeros);
        Some(buf)
    } else {
        None
    };
    let sparse_stats_readback_buffer = if sparse_probe {
        Some(r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-stats-readback"),
            size: 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    } else {
        None
    };
    let sparse_ts_query_count = u32::try_from(inputs.len())
        .ok()
        .and_then(|n| n.checked_mul(SPARSE_KERNEL_STAGE_QUERY_COUNT))
        .unwrap_or(0);
    let mut sparse_ts_prepare_kernel_ms = 0.0_f64;
    let mut sparse_ts_kernel_ms = 0.0_f64;
    let mut sparse_ts_parse_ms = 0.0_f64;
    let mut sparse_ts_map_status: Option<String> = None;
    let mut sparse_ts_query_set = None;
    let mut sparse_ts_resolve_buffer = None;
    let mut sparse_ts_readback_buffer = None;
    if sparse_kernel_ts_probe && sparse_ts_query_count > 0 {
        let ts_bytes = u64::from(sparse_ts_query_count).saturating_mul(8).max(8);
        sparse_ts_query_set = Some(r.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-ts-qs"),
            ty: wgpu::QueryType::Timestamp,
            count: sparse_ts_query_count,
        }));
        sparse_ts_resolve_buffer = Some(r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-ts-resolve"),
            size: ts_bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        sparse_ts_readback_buffer = Some(r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-ts-readback"),
            size: ts_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }

    let t_upload = Instant::now();
    let sparse_submit_chunk = sparse_inflight_submit_chunks().max(1);
    let mut encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-batch-encoder"),
        });
    let mut encoder_has_work = false;
    let mut chunk_encoded = 0usize;
    let mut dispatch_jobs = Vec::<SparsePackDispatchJob>::with_capacity(inputs.len());
    let mut sparse_submit_seq = 0u64;
    let mut sparse_submit_seq_last = 0u64;
    let mut sparse_submit_count = 0usize;
    let mut submit_ms = 0.0_f64;
    let mut pre_wait_latest_build_submit_ms = 0.0_f64;
    if sparse_wait_attr_probe {
        if let Some((_, _, latest_submit_idx)) = sparse_latest_build_submit.as_ref() {
            let t_pre_wait = Instant::now();
            r.device
                .poll(wgpu::Maintain::wait_for(latest_submit_idx.clone()));
            pre_wait_latest_build_submit_ms = elapsed_ms(t_pre_wait);
        }
    }
    for ((dispatch_idx, (input, prep)), scratch) in inputs
        .iter()
        .zip(prepared.iter())
        .enumerate()
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
            if sparse_probe { 1 } else { 0 },
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input.table.table_meta_buffer.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: sparse_stats_buffer
                        .as_ref()
                        .unwrap_or(&r.pack_sparse_stats_zero_buffer)
                        .as_entire_binding(),
                },
            ],
        });

        r.queue
            .write_buffer(&scratch.params_buffer, 0, bytemuck::cast_slice(&params_cap));

        let stage_base = u32::try_from(dispatch_idx)
            .ok()
            .and_then(|v| v.checked_mul(SPARSE_KERNEL_STAGE_QUERY_COUNT))
            .unwrap_or(0);
        if let Some(query_set) = sparse_ts_query_set.as_ref() {
            if stage_base.saturating_add(3) < sparse_ts_query_count {
                encoder.write_timestamp(query_set, stage_base);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cozip-pdeflate-pack-sparse-prepare-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&r.pack_sparse_prepare_pipeline);
            pass.set_bind_group(0, &prepare_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        if let Some(query_set) = sparse_ts_query_set.as_ref() {
            if stage_base.saturating_add(3) < sparse_ts_query_count {
                encoder.write_timestamp(query_set, stage_base + 1);
                encoder.write_timestamp(query_set, stage_base + 2);
            }
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
        if let Some(query_set) = sparse_ts_query_set.as_ref() {
            if stage_base.saturating_add(3) < sparse_ts_query_count {
                encoder.write_timestamp(query_set, stage_base + 3);
            }
        }
        dispatch_jobs.push(SparsePackDispatchJob { scratch });
        encoder_has_work = true;
        chunk_encoded = chunk_encoded.saturating_add(1);
        if chunk_encoded >= sparse_submit_chunk {
            let t_submit_chunk = Instant::now();
            let submit_seq = next_gpu_submit_seq();
            if sparse_submit_seq == 0 {
                sparse_submit_seq = submit_seq;
            }
            sparse_submit_seq_last = submit_seq;
            r.queue.submit(Some(encoder.finish()));
            submit_ms += elapsed_ms(t_submit_chunk);
            sparse_submit_count = sparse_submit_count.saturating_add(1);
            encoder = r
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("cozip-pdeflate-pack-sparse-batch-encoder"),
                });
            encoder_has_work = false;
            chunk_encoded = 0;
        }
    }
    if encoder_has_work {
        let t_submit_chunk = Instant::now();
        let submit_seq = next_gpu_submit_seq();
        if sparse_submit_seq == 0 {
            sparse_submit_seq = submit_seq;
        }
        sparse_submit_seq_last = submit_seq;
        r.queue.submit(Some(encoder.finish()));
        submit_ms += elapsed_ms(t_submit_chunk);
        sparse_submit_count = sparse_submit_count.saturating_add(1);
    }

    let mut post_encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-post-encoder"),
        });
    let mut post_encoder_has_work = false;
    if let (Some(stats_buffer), Some(stats_readback)) = (
        sparse_stats_buffer.as_ref(),
        sparse_stats_readback_buffer.as_ref(),
    ) {
        post_encoder.copy_buffer_to_buffer(stats_buffer, 0, stats_readback, 0, 32);
        post_encoder_has_work = true;
    }
    if let (Some(query_set), Some(resolve_buffer), Some(readback_buffer)) = (
        sparse_ts_query_set.as_ref(),
        sparse_ts_resolve_buffer.as_ref(),
        sparse_ts_readback_buffer.as_ref(),
    ) {
        let ts_bytes = u64::from(sparse_ts_query_count).saturating_mul(8).max(8);
        post_encoder.resolve_query_set(query_set, 0..sparse_ts_query_count, resolve_buffer, 0);
        post_encoder.copy_buffer_to_buffer(resolve_buffer, 0, readback_buffer, 0, ts_bytes);
        post_encoder_has_work = true;
    }
    if post_encoder_has_work {
        let t_submit_post = Instant::now();
        let submit_seq = next_gpu_submit_seq();
        if sparse_submit_seq == 0 {
            sparse_submit_seq = submit_seq;
        }
        sparse_submit_seq_last = submit_seq;
        r.queue.submit(Some(post_encoder.finish()));
        submit_ms += elapsed_ms(t_submit_post);
        sparse_submit_count = sparse_submit_count.saturating_add(1);
    }
    if sparse_submit_seq == 0 {
        sparse_submit_seq = sparse_submit_seq_last;
    }
    let mut upload_dispatch_ms = (elapsed_ms(t_upload) - submit_ms).max(0.0);
    upload_ms += upload_dispatch_ms;

    let mut lens_submit_done_wait_ms = 0.0_f64;
    let mut lens_poll_calls = 0u64;
    let lens_yield_calls = 0u64;
    let mut phase1_wait_calls = 0u64;
    let mut phase2_wait_calls = 0u64;
    let mut phase1_wait_max_ms = 0.0_f64;
    let mut phase2_wait_max_ms = 0.0_f64;
    let mut phase1_wait_ready_total = 0usize;
    let mut phase2_wait_ready_total = 0usize;
    let mut size_slices = Vec::with_capacity(dispatch_jobs.len());
    let mut size_receivers = Vec::with_capacity(dispatch_jobs.len());
    let mut sparse_stats_slice = None;
    let mut sparse_stats_receiver = None;
    let mut sparse_stats_pending = false;
    let mut sparse_ts_slice = None;
    let mut sparse_ts_receiver = None;
    let mut sparse_ts_pending = false;
    if let Some(stats_readback) = sparse_stats_readback_buffer.as_ref() {
        let stats_slice = stats_readback.slice(..32);
        let (stats_tx, stats_rx) = mpsc::channel();
        stats_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = stats_tx.send(res);
        });
        sparse_stats_slice = Some(stats_slice);
        sparse_stats_receiver = Some(stats_rx);
        sparse_stats_pending = true;
    }
    if let Some(ts_readback) = sparse_ts_readback_buffer.as_ref() {
        let ts_size = u64::from(sparse_ts_query_count).saturating_mul(8).max(8);
        let ts_slice = ts_readback.slice(..ts_size);
        let (ts_tx, ts_rx) = mpsc::channel();
        ts_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = ts_tx.send(res);
        });
        sparse_ts_slice = Some(ts_slice);
        sparse_ts_receiver = Some(ts_rx);
        sparse_ts_pending = true;
    }
    for dispatch in &dispatch_jobs {
        let size_slice = dispatch.scratch.params_readback_buffer.slice(..48);
        let (tx, rx) = mpsc::channel();
        size_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        size_slices.push(size_slice);
        size_receivers.push(rx);
    }
    let t_lens_wait_phase1 = Instant::now();
    let mut phase1_submit_done_wait_ms = 0.0_f64;
    let mut pending_sizes = vec![true; size_receivers.len()];
    let mut remaining_sizes = size_receivers.len();
    while remaining_sizes > 0 || sparse_stats_pending || sparse_ts_pending {
        r.device.poll(wgpu::Maintain::Poll);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        let mut collected = 0usize;
        for (i, rx) in size_receivers.iter().enumerate() {
            if pending_sizes[i] {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse size map failed: {e}"))
                        })?;
                        pending_sizes[i] = false;
                        remaining_sizes = remaining_sizes.saturating_sub(1);
                        collected = collected.saturating_add(1);
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
        if sparse_stats_pending {
            if let Some(rx) = sparse_stats_receiver.as_ref() {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse stats map failed: {e}"))
                        })?;
                        sparse_stats_pending = false;
                        collected = collected.saturating_add(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(PDeflateError::Gpu(
                            "gpu sparse stats map channel closed".to_string(),
                        ));
                    }
                }
            }
        }
        if sparse_ts_pending {
            if let Some(rx) = sparse_ts_receiver.as_ref() {
                match rx.try_recv() {
                    Ok(res) => {
                        if let Err(e) = res {
                            sparse_ts_map_status = Some(format!("map failed: {e}"));
                        }
                        sparse_ts_pending = false;
                        collected = collected.saturating_add(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        sparse_ts_map_status = Some("map channel closed".to_string());
                        sparse_ts_pending = false;
                    }
                }
            }
        }
        if collected > 0 {
            continue;
        }
        let t_wait = Instant::now();
        r.device.poll(wgpu::Maintain::Wait);
        let wait_ms = elapsed_ms(t_wait);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        phase1_wait_calls = phase1_wait_calls.saturating_add(1);
        phase1_wait_max_ms = phase1_wait_max_ms.max(wait_ms);
        phase1_submit_done_wait_ms += wait_ms;
        let mut collected_after_wait = 0usize;
        for (i, rx) in size_receivers.iter().enumerate() {
            if pending_sizes[i] {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse size map failed: {e}"))
                        })?;
                        pending_sizes[i] = false;
                        remaining_sizes = remaining_sizes.saturating_sub(1);
                        collected_after_wait = collected_after_wait.saturating_add(1);
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
        if sparse_stats_pending {
            if let Some(rx) = sparse_stats_receiver.as_ref() {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse stats map failed: {e}"))
                        })?;
                        sparse_stats_pending = false;
                        collected_after_wait = collected_after_wait.saturating_add(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(PDeflateError::Gpu(
                            "gpu sparse stats map channel closed".to_string(),
                        ));
                    }
                }
            }
        }
        if sparse_ts_pending {
            if let Some(rx) = sparse_ts_receiver.as_ref() {
                match rx.try_recv() {
                    Ok(res) => {
                        if let Err(e) = res {
                            sparse_ts_map_status = Some(format!("map failed: {e}"));
                        }
                        sparse_ts_pending = false;
                        collected_after_wait = collected_after_wait.saturating_add(1);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        sparse_ts_map_status = Some("map channel closed".to_string());
                        sparse_ts_pending = false;
                    }
                }
            }
        }
        phase1_wait_ready_total = phase1_wait_ready_total.saturating_add(collected_after_wait);
        if collected_after_wait == 0
            && (remaining_sizes > 0 || sparse_stats_pending || sparse_ts_pending)
        {
            return Err(PDeflateError::Gpu(
                "gpu sparse size/stats pending stalled after wait".to_string(),
            ));
        }
    }
    lens_submit_done_wait_ms += phase1_submit_done_wait_ms;
    let phase1_wait_ms = elapsed_ms(t_lens_wait_phase1);
    let phase1_map_after_done_ms = (phase1_wait_ms - phase1_submit_done_wait_ms).max(0.0);
    let mut lens_map_after_done_ms = phase1_map_after_done_ms;

    if let Some(ts_slice) = sparse_ts_slice {
        if sparse_ts_map_status.is_none() {
            let t_parse = Instant::now();
            let mapped = ts_slice.get_mapped_range();
            let ticks: &[u64] = bytemuck::cast_slice(&mapped);
            let period_ms = (r.queue.get_timestamp_period() as f64) / 1_000_000.0;
            for dispatch_idx in 0..inputs.len() {
                let stage_base =
                    dispatch_idx.saturating_mul(SPARSE_KERNEL_STAGE_QUERY_COUNT as usize);
                if stage_base + 3 < ticks.len() {
                    let t0 = ticks[stage_base];
                    let t1 = ticks[stage_base + 1];
                    let t2 = ticks[stage_base + 2];
                    let t3 = ticks[stage_base + 3];
                    if t1 >= t0 {
                        sparse_ts_prepare_kernel_ms += (t1 - t0) as f64 * period_ms;
                    }
                    if t3 >= t2 {
                        sparse_ts_kernel_ms += (t3 - t2) as f64 * period_ms;
                    }
                }
            }
            drop(mapped);
            sparse_ts_parse_ms = elapsed_ms(t_parse);
        }
        if let Some(ts_readback) = sparse_ts_readback_buffer.as_ref() {
            ts_readback.unmap();
        }
    }
    let sparse_ts_kernel_total_ms = sparse_ts_prepare_kernel_ms + sparse_ts_kernel_ms;
    let sparse_ts_queue_stall_est_ms =
        (phase1_submit_done_wait_ms - sparse_ts_kernel_total_ms).max(0.0);

    let mut sparse_probe_counters = [0u64; 8];
    if let Some(stats_slice) = sparse_stats_slice {
        let mapped = stats_slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        for (i, dst) in sparse_probe_counters.iter_mut().enumerate() {
            *dst = u64::from(*words.get(i).unwrap_or(&0));
        }
        drop(mapped);
        if let Some(stats_readback) = sparse_stats_readback_buffer.as_ref() {
            stats_readback.unmap();
        }
    }

    let t_lens_copy = Instant::now();
    let mut payload_lens = Vec::<usize>::with_capacity(dispatch_jobs.len());
    let mut payload_copy_bytes = Vec::<u64>::with_capacity(dispatch_jobs.len());
    let mut payload_table_counts = Vec::<usize>::with_capacity(dispatch_jobs.len());
    for (idx, prep) in prepared.iter().enumerate() {
        let size_mapped = size_slices[idx].get_mapped_range();
        let size_words: &[u32] = bytemuck::cast_slice(&size_mapped);
        let table_count_u32 = *size_words.get(1).unwrap_or(&0);
        let total_len_u32 = *size_words.get(10).unwrap_or(&0xffff_ffff);
        let total_words_u32 = *size_words.get(11).unwrap_or(&0);
        drop(size_mapped);
        dispatch_jobs[idx].scratch.params_readback_buffer.unmap();
        payload_table_counts
            .push(usize::try_from(table_count_u32).map_err(|_| PDeflateError::NumericOverflow)?);

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
        payload_copy_bytes.push(total_copy_bytes.max(4));
    }
    lens_copy_ms = elapsed_ms(t_lens_copy);

    let t_payload_submit_all = Instant::now();
    let mut payload_encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-pack-sparse-payload-readback-encoder"),
        });
    for (idx, dispatch) in dispatch_jobs.iter().enumerate() {
        payload_encoder.copy_buffer_to_buffer(
            &dispatch.scratch.out_buffer,
            0,
            &dispatch.scratch.readback_buffer,
            0,
            payload_copy_bytes[idx],
        );
    }
    let t_payload_submit = Instant::now();
    let payload_readback_submit_seq = next_gpu_submit_seq();
    r.queue.submit(Some(payload_encoder.finish()));
    let payload_submit_ms = elapsed_ms(t_payload_submit);
    submit_ms += payload_submit_ms;
    let payload_upload_dispatch_ms =
        (elapsed_ms(t_payload_submit_all) - payload_submit_ms).max(0.0);
    upload_ms += payload_upload_dispatch_ms;
    upload_dispatch_ms += payload_upload_dispatch_ms;

    let t_lens_wait_phase2 = Instant::now();
    let mut phase2_submit_done_wait_ms = 0.0_f64;
    let mut out_slices = Vec::with_capacity(dispatch_jobs.len());
    let mut out_receivers = Vec::with_capacity(dispatch_jobs.len());
    for (idx, dispatch) in dispatch_jobs.iter().enumerate() {
        let out_slice = dispatch
            .scratch
            .readback_buffer
            .slice(..payload_copy_bytes[idx]);
        let (out_tx, out_rx) = mpsc::channel();
        out_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = out_tx.send(res);
        });
        out_slices.push(out_slice);
        out_receivers.push(out_rx);
    }
    let mut pending_out = vec![true; out_receivers.len()];
    let mut remaining_out = out_receivers.len();
    while remaining_out > 0 {
        r.device.poll(wgpu::Maintain::Poll);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        let mut collected = 0usize;
        for (i, rx) in out_receivers.iter().enumerate() {
            if pending_out[i] {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse chunk pack map failed: {e}"))
                        })?;
                        pending_out[i] = false;
                        remaining_out = remaining_out.saturating_sub(1);
                        collected = collected.saturating_add(1);
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
        if collected > 0 {
            continue;
        }
        let t_wait = Instant::now();
        r.device.poll(wgpu::Maintain::Wait);
        let wait_ms = elapsed_ms(t_wait);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        phase2_wait_calls = phase2_wait_calls.saturating_add(1);
        phase2_wait_max_ms = phase2_wait_max_ms.max(wait_ms);
        phase2_submit_done_wait_ms += wait_ms;
        let mut collected_after_wait = 0usize;
        for (i, rx) in out_receivers.iter().enumerate() {
            if pending_out[i] {
                match rx.try_recv() {
                    Ok(res) => {
                        res.map_err(|e| {
                            PDeflateError::Gpu(format!("gpu sparse chunk pack map failed: {e}"))
                        })?;
                        pending_out[i] = false;
                        remaining_out = remaining_out.saturating_sub(1);
                        collected_after_wait = collected_after_wait.saturating_add(1);
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
        phase2_wait_ready_total = phase2_wait_ready_total.saturating_add(collected_after_wait);
        if collected_after_wait == 0 && remaining_out > 0 {
            return Err(PDeflateError::Gpu(
                "gpu sparse payload pending stalled after wait".to_string(),
            ));
        }
    }
    lens_submit_done_wait_ms += phase2_submit_done_wait_ms;
    let phase2_wait_ms = elapsed_ms(t_lens_wait_phase2);
    let phase2_map_after_done_ms = (phase2_wait_ms - phase2_submit_done_wait_ms).max(0.0);
    lens_map_after_done_ms += phase2_map_after_done_ms;
    lens_wait_ms = phase1_wait_ms + phase2_wait_ms;

    if sparse_lens_wait_probe {
        let seq = SPARSE_LENS_WAIT_PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
        if table_stage_probe_should_log(seq) {
            let size_readback_bytes = (dispatch_jobs.len() as u64).saturating_mul(48);
            let stats_readback_bytes = if sparse_probe { 32u64 } else { 0u64 };
            let ts_readback_bytes = if sparse_kernel_ts_probe {
                u64::from(sparse_ts_query_count).saturating_mul(8)
            } else {
                0u64
            };
            let payload_readback_bytes = payload_copy_bytes
                .iter()
                .fold(0u64, |acc, v| acc.saturating_add(*v));
            let readback_bytes = size_readback_bytes
                .saturating_add(stats_readback_bytes)
                .saturating_add(ts_readback_bytes)
                .saturating_add(payload_readback_bytes);
            let build_id_range = if sparse_prebuild_build_id_min != u64::MAX {
                format!(
                    "{}..{}",
                    sparse_prebuild_build_id_min, sparse_prebuild_build_id_max
                )
            } else {
                "none".to_string()
            };
            let submit_seq_range = if sparse_prebuild_submit_seq_min != u64::MAX {
                format!(
                    "{}..{}",
                    sparse_prebuild_submit_seq_min, sparse_prebuild_submit_seq_max
                )
            } else {
                "none".to_string()
            };
            let build_ids = if sparse_build_ids.is_empty() {
                "none".to_string()
            } else {
                sparse_build_ids
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let (latest_build_submit_seq, latest_build_id) = sparse_latest_build_submit
                .as_ref()
                .map(|(submit_seq, build_id, _)| (*submit_seq, *build_id))
                .unwrap_or((0, 0));
            let sparse_ts_status = if !sparse_kernel_ts_probe {
                "off".to_string()
            } else {
                sparse_ts_map_status
                    .clone()
                    .unwrap_or_else(|| "ok".to_string())
            };
            eprintln!(
                "[cozip_pdeflate][timing][sparse-lens-wait-breakdown] seq={} chunks={} pending_maps={} readback_kib={:.1} payload_readback_kib={:.1} build_id_range={} build_submit_seq_range={} build_ids={} latest_build_submit_seq={} latest_build_id={} sparse_submit_seq={} sparse_submit_seq_last={} sparse_submit_count={} payload_submit_seq={} pre_wait_latest_build_submit_ms={:.3} submit_done_wait_sparse_ms={:.3} submit_done_wait_payload_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_sparse_ms={:.3} map_callback_wait_payload_ms={:.3} map_callback_wait_ms={:.3} sparse_wait_calls={} payload_wait_calls={} sparse_wait_max_ms={:.3} payload_wait_max_ms={:.3} sparse_wait_ready={} payload_wait_ready={} sparse_ts_prepare_kernel_ms={:.3} sparse_ts_kernel_ms={:.3} sparse_ts_kernel_total_ms={:.3} sparse_ts_parse_ms={:.3} sparse_ts_queue_stall_est_ms={:.3} sparse_ts_status={} lens_wait_ms={:.3}",
                seq,
                inputs.len(),
                dispatch_jobs
                    .len()
                    .saturating_add(if sparse_probe { 1 } else { 0 })
                    .saturating_add(dispatch_jobs.len()),
                (readback_bytes as f64) / 1024.0,
                (payload_readback_bytes as f64) / 1024.0,
                build_id_range,
                submit_seq_range,
                build_ids,
                latest_build_submit_seq,
                latest_build_id,
                sparse_submit_seq,
                sparse_submit_seq_last,
                sparse_submit_count,
                payload_readback_submit_seq,
                pre_wait_latest_build_submit_ms,
                phase1_submit_done_wait_ms,
                phase2_submit_done_wait_ms,
                lens_submit_done_wait_ms,
                phase1_map_after_done_ms,
                phase2_map_after_done_ms,
                lens_map_after_done_ms,
                phase1_wait_calls,
                phase2_wait_calls,
                phase1_wait_max_ms,
                phase2_wait_max_ms,
                phase1_wait_ready_total,
                phase2_wait_ready_total,
                sparse_ts_prepare_kernel_ms,
                sparse_ts_kernel_ms,
                sparse_ts_kernel_total_ms,
                sparse_ts_parse_ms,
                sparse_ts_queue_stall_est_ms,
                sparse_ts_status,
                lens_wait_ms,
            );
            if sparse_kernel_ts_probe {
                let ts_seq = SPARSE_KERNEL_TS_PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
                if table_stage_probe_should_log(ts_seq) {
                    eprintln!(
                        "[cozip_pdeflate][timing][sparse-kernel-ts] seq={} chunks={} prepare_kernel_ms={:.3} sparse_kernel_ms={:.3} kernel_total_ms={:.3} submit_done_wait_sparse_ms={:.3} queue_stall_est_ms={:.3} parse_ms={:.3} status={}",
                        ts_seq,
                        inputs.len(),
                        sparse_ts_prepare_kernel_ms,
                        sparse_ts_kernel_ms,
                        sparse_ts_kernel_total_ms,
                        phase1_submit_done_wait_ms,
                        sparse_ts_queue_stall_est_ms,
                        sparse_ts_parse_ms,
                        sparse_ts_map_status.as_deref().unwrap_or("ok"),
                    );
                }
            }
            if sparse_wait_attr_probe {
                eprintln!(
                    "[cozip_pdeflate][timing][sparse-wait-attribution] seq={} chunks={} sparse_submit_seq={} sparse_submit_seq_last={} sparse_submit_count={} pre_wait_latest_build_submit_ms={:.3} submit_done_wait_sparse_ms={:.3} submit_done_wait_payload_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_ms={:.3} lens_wait_ms={:.3} note=\"pre_wait captures known table-build completion; submit_done_wait includes residual queue+own sparse+payload submissions\"",
                    seq,
                    inputs.len(),
                    sparse_submit_seq,
                    sparse_submit_seq_last,
                    sparse_submit_count,
                    pre_wait_latest_build_submit_ms,
                    phase1_submit_done_wait_ms,
                    phase2_submit_done_wait_ms,
                    lens_submit_done_wait_ms,
                    lens_map_after_done_ms,
                    lens_wait_ms,
                );
            }
        }
    }

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
            table_count: payload_table_counts[idx],
        });
    }
    let sparse_copy_ms = elapsed_ms(t_copy);
    map_copy_ms += lens_copy_ms + sparse_copy_ms;

    if sparse_probe {
        let total_classified = sparse_probe_counters[0]
            .saturating_add(sparse_probe_counters[1])
            .saturating_add(sparse_probe_counters[2])
            .saturating_add(sparse_probe_counters[3])
            .saturating_add(sparse_probe_counters[4]);
        let section_bytes = sparse_probe_counters[4];
        let search_steps = sparse_probe_counters[5];
        let search_hits = sparse_probe_counters[6];
        let search_misses = sparse_probe_counters[7];
        let search_steps_per_section_byte = if section_bytes > 0 {
            (search_steps as f64) / (section_bytes as f64)
        } else {
            0.0
        };
        let section_byte_pct = if total_classified > 0 {
            100.0 * (section_bytes as f64) / (total_classified as f64)
        } else {
            0.0
        };
        eprintln!(
            "[cozip_pdeflate][timing][gpu-sparse-probe] chunks={} bytes_header={} bytes_table_index={} bytes_table_data={} bytes_section_index={} bytes_section_cmd={} section_byte_pct={:.1} search_steps={} search_hits={} search_misses={} search_steps_per_section_byte={:.3}",
            inputs.len(),
            sparse_probe_counters[0],
            sparse_probe_counters[1],
            sparse_probe_counters[2],
            sparse_probe_counters[3],
            section_bytes,
            section_byte_pct,
            search_steps,
            search_hits,
            search_misses,
            search_steps_per_section_byte,
        );
    }

    for job in dispatch_jobs.drain(..) {
        release_sparse_pack_scratch(r, job.scratch);
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
            table_size_resolve_ms,
            prepare_misc_ms,
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

    let queue_probe = queue_probe_enabled();
    let mut pre_lens_queue_drain_ms = 0.0_f64;
    let mut pre_lens_queue_drain_poll_calls = 0u64;
    if queue_probe {
        let t_pre_drain = Instant::now();
        let (done_tx_pre, done_rx_pre) = mpsc::channel();
        r.queue.on_submitted_work_done(move || {
            let _ = done_tx_pre.send(());
        });
        let mut done = false;
        while !done {
            r.device.poll(wgpu::Maintain::Wait);
            pre_lens_queue_drain_poll_calls = pre_lens_queue_drain_poll_calls.saturating_add(1);
            match done_rx_pre.try_recv() {
                Ok(_) | Err(mpsc::TryRecvError::Disconnected) => done = true,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        pre_lens_queue_drain_ms = elapsed_ms(t_pre_drain);
    }

    struct LensHostChunk {
        section_cmd_lens: Vec<u32>,
        copy_offsets: Vec<usize>,
        total_cmd_len: usize,
        total_copy_bytes: u64,
    }

    let mut lens_offsets = Vec::with_capacity(prepared.len());
    let mut lens_total_bytes = 0u64;
    for prep in prepared.iter() {
        lens_offsets.push(lens_total_bytes);
        lens_total_bytes = lens_total_bytes
            .checked_add(prep.out_lens_bytes.max(4))
            .ok_or(PDeflateError::NumericOverflow)?;
    }
    let lens_total_bytes = lens_total_bytes.max(4);
    let lens_readback = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-sparse-lens-batch-readback"),
        size: lens_total_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let t_lens_submit_all = Instant::now();
    let mut lens_encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-sparse-lens-batch-readback-encoder"),
        });
    for (idx, input) in inputs.iter().enumerate() {
        lens_encoder.copy_buffer_to_buffer(
            &input.section.out_lens_buffer,
            0,
            &lens_readback,
            lens_offsets[idx],
            prepared[idx].out_lens_bytes.max(4),
        );
    }
    let t_lens_submit = Instant::now();
    r.queue.submit(Some(lens_encoder.finish()));
    let lens_submit_ms = elapsed_ms(t_lens_submit);
    let lens_upload_dispatch_ms = (elapsed_ms(t_lens_submit_all) - lens_submit_ms).max(0.0);

    let t_lens_wait = Instant::now();
    let lens_slice = lens_readback.slice(..lens_total_bytes);
    let (tx_lens, rx_lens) = mpsc::channel();
    lens_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_lens.send(res);
    });
    let mut lens_poll_calls = 0u64;
    loop {
        r.device.poll(wgpu::Maintain::Wait);
        lens_poll_calls = lens_poll_calls.saturating_add(1);
        match rx_lens.try_recv() {
            Ok(res) => {
                res.map_err(|e| {
                    PDeflateError::Gpu(format!("gpu sparse lens batch map failed: {e}"))
                })?;
                break;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(PDeflateError::Gpu(
                    "gpu sparse lens batch map channel closed".to_string(),
                ));
            }
        }
    }
    let lens_submit_done_wait_ms = elapsed_ms(t_lens_wait);
    let lens_wait_ms = lens_submit_done_wait_ms;
    let lens_map_after_done_ms = 0.0_f64;

    let t_lens_copy = Instant::now();
    let lens_mapped = lens_slice.get_mapped_range();
    let lens_bytes: &[u8] = &lens_mapped;
    let mut lens_chunks = Vec::<LensHostChunk>::with_capacity(inputs.len());
    for (idx, input) in inputs.iter().enumerate() {
        let prep = &prepared[idx];
        let lens_base =
            usize::try_from(lens_offsets[idx]).map_err(|_| PDeflateError::NumericOverflow)?;
        let lens_span = usize::try_from(prep.out_lens_bytes.max(4))
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let lens_end = lens_base
            .checked_add(lens_span)
            .ok_or(PDeflateError::NumericOverflow)?;
        if lens_end > lens_bytes.len() {
            return Err(PDeflateError::Gpu(
                "gpu sparse lens batch readback truncated".to_string(),
            ));
        }
        let chunk_lens = &lens_bytes[lens_base..lens_end];
        let needed_lens_bytes = input
            .section_count
            .checked_mul(4)
            .ok_or(PDeflateError::NumericOverflow)?;
        if chunk_lens.len() < needed_lens_bytes {
            return Err(PDeflateError::Gpu(
                "gpu sparse lens batch readback too small for section count".to_string(),
            ));
        }

        let mut section_cmd_lens = vec![0u32; input.section_count];
        for (sec, dst) in section_cmd_lens.iter_mut().enumerate() {
            let off = sec.checked_mul(4).ok_or(PDeflateError::NumericOverflow)?;
            *dst = u32::from_le_bytes([
                chunk_lens[off],
                chunk_lens[off + 1],
                chunk_lens[off + 2],
                chunk_lens[off + 3],
            ]);
        }

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
        lens_chunks.push(LensHostChunk {
            section_cmd_lens,
            copy_offsets,
            total_cmd_len,
            total_copy_bytes,
        });
    }
    drop(lens_mapped);
    lens_readback.unmap();
    let lens_copy_ms = elapsed_ms(t_lens_copy);

    let mut cmd_offsets = Vec::with_capacity(lens_chunks.len());
    let mut cmd_total_bytes = 0u64;
    for lens in lens_chunks.iter() {
        cmd_offsets.push(cmd_total_bytes);
        cmd_total_bytes = cmd_total_bytes
            .checked_add(lens.total_copy_bytes.max(4))
            .ok_or(PDeflateError::NumericOverflow)?;
    }
    let cmd_total_bytes = cmd_total_bytes.max(4);
    let cmd_readback = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-pdeflate-sparse-cmd-batch-readback"),
        size: cmd_total_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let t_cmd_submit_all = Instant::now();
    let mut cmd_encoder = r
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-sparse-cmd-batch-readback-encoder"),
        });
    for (idx, input) in inputs.iter().enumerate() {
        let prep = &prepared[idx];
        let lens = &lens_chunks[idx];
        let chunk_dst_base = cmd_offsets[idx];
        for sec in 0..input.section_count {
            let len = usize::try_from(lens.section_cmd_lens[sec])
                .map_err(|_| PDeflateError::NumericOverflow)?;
            if len == 0 {
                continue;
            }
            let src_off = u64::try_from(prep.section_offsets[sec])
                .map_err(|_| PDeflateError::NumericOverflow)?;
            let dst_off = chunk_dst_base
                .checked_add(
                    u64::try_from(lens.copy_offsets[sec])
                        .map_err(|_| PDeflateError::NumericOverflow)?,
                )
                .ok_or(PDeflateError::NumericOverflow)?;
            let copy_bytes =
                u64::try_from(align_up4(len)).map_err(|_| PDeflateError::NumericOverflow)?;
            if src_off.saturating_add(copy_bytes) > prep.out_cmd_bytes
                || dst_off.saturating_add(copy_bytes) > cmd_total_bytes
            {
                return Err(PDeflateError::Gpu(
                    "gpu sparse cmd batch copy range out of bounds".to_string(),
                ));
            }
            cmd_encoder.copy_buffer_to_buffer(
                &input.section.out_cmd_buffer,
                src_off,
                &cmd_readback,
                dst_off,
                copy_bytes,
            );
        }
    }
    let t_cmd_submit = Instant::now();
    r.queue.submit(Some(cmd_encoder.finish()));
    let cmd_submit_ms = elapsed_ms(t_cmd_submit);
    let cmd_upload_dispatch_ms = (elapsed_ms(t_cmd_submit_all) - cmd_submit_ms).max(0.0);

    let t_cmd_wait = Instant::now();
    let cmd_slice = cmd_readback.slice(..cmd_total_bytes);
    let (tx_cmd, rx_cmd) = mpsc::channel();
    cmd_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx_cmd.send(res);
    });
    let mut cmd_poll_calls = 0u64;
    loop {
        r.device.poll(wgpu::Maintain::Wait);
        cmd_poll_calls = cmd_poll_calls.saturating_add(1);
        match rx_cmd.try_recv() {
            Ok(res) => {
                res.map_err(|e| {
                    PDeflateError::Gpu(format!("gpu sparse cmd batch map failed: {e}"))
                })?;
                break;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(PDeflateError::Gpu(
                    "gpu sparse cmd batch map channel closed".to_string(),
                ));
            }
        }
    }
    let cmd_wait_ms = elapsed_ms(t_cmd_wait);

    let t_cmd_copy = Instant::now();
    let cmd_mapped = cmd_slice.get_mapped_range();
    let cmd_bytes: &[u8] = &cmd_mapped;
    let mut out = Vec::<GpuSparseSectionHostOutput>::with_capacity(inputs.len());
    let mut copied_cmd_bytes_total = 0u64;
    for (idx, lens) in lens_chunks.into_iter().enumerate() {
        let chunk_base =
            usize::try_from(cmd_offsets[idx]).map_err(|_| PDeflateError::NumericOverflow)?;
        let chunk_span = usize::try_from(lens.total_copy_bytes.max(4))
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let chunk_end = chunk_base
            .checked_add(chunk_span)
            .ok_or(PDeflateError::NumericOverflow)?;
        if chunk_end > cmd_bytes.len() {
            return Err(PDeflateError::Gpu(
                "gpu sparse cmd batch readback truncated".to_string(),
            ));
        }
        let chunk_cmd = &cmd_bytes[chunk_base..chunk_end];
        let mut section_cmd = Vec::with_capacity(lens.total_cmd_len);
        for sec in 0..lens.section_cmd_lens.len() {
            let len = usize::try_from(lens.section_cmd_lens[sec])
                .map_err(|_| PDeflateError::NumericOverflow)?;
            if len == 0 {
                continue;
            }
            let off = lens.copy_offsets[sec];
            let end = off.checked_add(len).ok_or(PDeflateError::NumericOverflow)?;
            if end > chunk_cmd.len() {
                return Err(PDeflateError::Gpu(format!(
                    "gpu sparse cmd batch section {} truncated",
                    sec
                )));
            }
            section_cmd.extend_from_slice(&chunk_cmd[off..end]);
        }
        copied_cmd_bytes_total = copied_cmd_bytes_total.saturating_add(lens.total_copy_bytes);
        out.push(GpuSparseSectionHostOutput {
            section_cmd_lens: lens.section_cmd_lens,
            section_cmd,
        });
    }
    drop(cmd_mapped);
    cmd_readback.unmap();
    let cmd_copy_ms = elapsed_ms(t_cmd_copy);

    let lens_yield_calls = 0u64;
    let upload_dispatch_ms = lens_upload_dispatch_ms + cmd_upload_dispatch_ms;
    let submit_ms = lens_submit_ms + cmd_submit_ms;
    let wait_ms = lens_wait_ms + cmd_wait_ms;
    let map_copy_ms = lens_copy_ms + cmd_copy_ms;
    let total_ms = elapsed_ms(t_total);
    if queue_probe {
        eprintln!(
            "[cozip_pdeflate][timing][gpu-readback-probe] chunks={} pre_lens_queue_drain_ms={:.3} pre_lens_queue_drain_polls={} lens_submit_done_wait_ms={:.3} lens_wait_ms={:.3} cmd_wait_ms={:.3} lens_map_after_done_ms={:.3} lens_poll_calls={} cmd_poll_calls={}",
            inputs.len(),
            pre_lens_queue_drain_ms,
            pre_lens_queue_drain_poll_calls,
            lens_submit_done_wait_ms,
            lens_wait_ms,
            cmd_wait_ms,
            lens_map_after_done_ms,
            lens_poll_calls,
            cmd_poll_calls
        );
    }
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
            table_size_resolve_ms: 0.0,
            prepare_misc_ms: prepare_ms,
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

#[cfg(test)]
pub(crate) fn reset_decode_slot_pool_for_test() -> Result<(), PDeflateError> {
    let runtime = runtime()?;
    let mut pool = runtime
        .decode_slot_pool
        .lock()
        .map_err(|_| PDeflateError::Gpu("gpu decode slot pool mutex poisoned".to_string()))?;
    *pool = GpuDecodeSlotPool::default();
    Ok(())
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuDecodeSlot {
    pub slot_index: usize,
    pub generation: u64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuDecodeSectionMeta {
    pub cmd_offset: usize,
    pub cmd_len: usize,
    pub out_offset: usize,
    pub out_len: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct GpuDecodeJob<'a> {
    pub chunk_index: usize,
    pub payload: &'a [u8],
    pub table_count: usize,
    pub chunk_uncompressed_len: usize,
    pub out_offset: usize,
    pub out_len: usize,
    pub section_meta: Vec<GpuDecodeSectionMeta>,
    pub preferred_slot: Option<GpuDecodeSlot>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuDecodeDisposition {
    SubmittedGpu,
    CpuFallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuDecodeResult {
    pub chunk_index: usize,
    pub slot: Option<GpuDecodeSlot>,
    pub disposition: GpuDecodeDisposition,
    pub decoded_chunk: Option<Vec<u8>>,
}

fn decode_fallback_results(jobs: &[GpuDecodeJob<'_>]) -> Vec<GpuDecodeResult> {
    jobs.iter()
        .map(|job| GpuDecodeResult {
            chunk_index: job.chunk_index,
            slot: None,
            disposition: GpuDecodeDisposition::CpuFallback,
            decoded_chunk: None,
        })
        .collect()
}

fn read_le_u16_at(src: &[u8], offset: usize) -> Result<u16, PDeflateError> {
    let end = offset
        .checked_add(2)
        .ok_or(PDeflateError::NumericOverflow)?;
    let bytes = src.get(offset..end).ok_or(PDeflateError::InvalidStream(
        "gpu decode chunk header truncated",
    ))?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_le_u32_at(src: &[u8], offset: usize) -> Result<u32, PDeflateError> {
    let end = offset
        .checked_add(4)
        .ok_or(PDeflateError::NumericOverflow)?;
    let bytes = src.get(offset..end).ok_or(PDeflateError::InvalidStream(
        "gpu decode chunk header truncated",
    ))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn parse_gpu_decode_payload_layout(
    payload: &[u8],
) -> Result<GpuDecodePayloadLayout, PDeflateError> {
    if payload.len() < PACK_CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream(
            "gpu decode chunk header truncated",
        ));
    }
    let magic = payload.get(0..4).ok_or(PDeflateError::InvalidStream(
        "gpu decode chunk header truncated",
    ))?;
    if magic != PACK_CHUNK_MAGIC {
        return Err(PDeflateError::InvalidStream("gpu decode bad chunk magic"));
    }
    let version = read_le_u16_at(payload, 4)?;
    if version != PACK_CHUNK_VERSION {
        return Err(PDeflateError::InvalidStream(
            "gpu decode unsupported chunk version",
        ));
    }
    let table_count = usize::from(read_le_u16_at(payload, 12)?);
    if table_count > GPU_DECODE_MAX_TABLE_ID {
        return Err(PDeflateError::InvalidStream(
            "gpu decode table_count too large",
        ));
    }
    let table_index_offset = usize::try_from(read_le_u32_at(payload, 16)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let table_data_offset = usize::try_from(read_le_u32_at(payload, 20)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let section_index_offset = usize::try_from(read_le_u32_at(payload, 24)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let section_cmd_offset = usize::try_from(read_le_u32_at(payload, 28)?)
        .map_err(|_| PDeflateError::NumericOverflow)?;
    if !(table_index_offset <= table_data_offset
        && table_data_offset <= section_index_offset
        && section_index_offset <= section_cmd_offset
        && section_cmd_offset <= payload.len())
    {
        return Err(PDeflateError::InvalidStream(
            "gpu decode chunk offsets invalid",
        ));
    }
    if table_index_offset != PACK_CHUNK_HEADER_SIZE {
        return Err(PDeflateError::InvalidStream(
            "gpu decode unexpected table_index_offset",
        ));
    }
    let table_index_len = table_data_offset.saturating_sub(table_index_offset);
    if table_index_len != table_count {
        return Err(PDeflateError::InvalidStream(
            "gpu decode table index length mismatch",
        ));
    }
    Ok(GpuDecodePayloadLayout {
        table_count,
        table_index_start: table_index_offset,
        table_data_start: table_data_offset,
        table_end: section_index_offset,
        cmd_start: section_cmd_offset,
    })
}

fn gpu_decode_slot_split(slot_count: usize) -> (usize, usize) {
    let total = slot_count.max(1);
    if total == 1 {
        return (1, 0);
    }
    let large = if total >= 4 { (total / 4).max(1) } else { 1 };
    let normal = total.saturating_sub(large).max(1);
    (normal, total.saturating_sub(normal))
}

fn gpu_decode_normal_caps(options: &PDeflateOptions) -> Result<GpuDecodeSlotCaps, PDeflateError> {
    let base_table = options
        .max_table_entries
        .saturating_mul(GPU_DECODE_TABLE_REPEAT_STRIDE)
        .max(4096);
    Ok(GpuDecodeSlotCaps {
        cmd_cap_bytes: u64::try_from(options.chunk_size.saturating_mul(2).max(64 * 1024))
            .map_err(|_| PDeflateError::NumericOverflow)?,
        section_meta_cap: options.section_count.max(1),
        table_cap_bytes: u64::try_from(base_table).map_err(|_| PDeflateError::NumericOverflow)?,
        out_cap_bytes: u64::try_from(options.chunk_size.max(64 * 1024))
            .map_err(|_| PDeflateError::NumericOverflow)?,
        error_cap_words: options.section_count.max(1),
    })
}

fn gpu_decode_large_caps(base: GpuDecodeSlotCaps) -> GpuDecodeSlotCaps {
    GpuDecodeSlotCaps {
        cmd_cap_bytes: base.cmd_cap_bytes.saturating_mul(2),
        section_meta_cap: base.section_meta_cap.saturating_mul(2),
        table_cap_bytes: base.table_cap_bytes.saturating_mul(2),
        out_cap_bytes: base.out_cap_bytes.saturating_mul(2),
        error_cap_words: base.error_cap_words.saturating_mul(2),
    }
}

fn build_gpu_decode_slot_requirement(
    job: &GpuDecodeJob<'_>,
    layout: GpuDecodePayloadLayout,
) -> Result<GpuDecodeSlotRequirement, PDeflateError> {
    if job.table_count != layout.table_count {
        return Err(PDeflateError::InvalidStream(
            "gpu decode table_count mismatch",
        ));
    }
    if layout.cmd_start > job.payload.len() || layout.table_end > job.payload.len() {
        return Err(PDeflateError::InvalidStream(
            "gpu decode payload layout out of bounds",
        ));
    }
    let cmd_bytes = u64::try_from(job.payload.len().saturating_sub(layout.cmd_start))
        .map_err(|_| PDeflateError::NumericOverflow)?;
    let table_repeat_bytes = layout
        .table_count
        .checked_mul(GPU_DECODE_TABLE_REPEAT_STRIDE)
        .ok_or(PDeflateError::NumericOverflow)?;
    let table_bytes =
        u64::try_from(table_repeat_bytes).map_err(|_| PDeflateError::NumericOverflow)?;
    let out_bytes =
        u64::try_from(job.chunk_uncompressed_len).map_err(|_| PDeflateError::NumericOverflow)?;
    Ok(GpuDecodeSlotRequirement {
        cmd_bytes,
        section_meta_count: job.section_meta.len(),
        table_bytes,
        out_bytes,
        error_words: job.section_meta.len().max(1),
    })
}

fn create_gpu_decode_slot_entry(
    runtime: &GpuMatchRuntime,
    class: GpuDecodeSlotClass,
    local_index: usize,
    caps: GpuDecodeSlotCaps,
) -> Result<GpuDecodeSlotEntry, PDeflateError> {
    let index_base = match class {
        GpuDecodeSlotClass::Normal => 0usize,
        GpuDecodeSlotClass::Large => GPU_DECODE_LARGE_SLOT_INDEX_BASE,
    };
    let slot_index = index_base
        .checked_add(local_index)
        .ok_or(PDeflateError::NumericOverflow)?;
    let cmd_size = u64::try_from(align_up4(
        usize::try_from(caps.cmd_cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?,
    ))
    .map_err(|_| PDeflateError::NumericOverflow)?
    .max(4);
    let section_meta_size = u64::try_from(
        GPU_DECODE_SECTION_META_HEADER_WORDS
            .saturating_add(
                caps.section_meta_cap
                    .saturating_mul(GPU_DECODE_SECTION_META_WORDS),
            )
            .saturating_mul(std::mem::size_of::<u32>()),
    )
    .map_err(|_| PDeflateError::NumericOverflow)?
    .max(4);
    let table_size = u64::try_from(align_up4(
        usize::try_from(caps.table_cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?,
    ))
    .map_err(|_| PDeflateError::NumericOverflow)?
    .max(4);
    let out_size = u64::try_from(align_up4(
        usize::try_from(caps.out_cap_bytes).map_err(|_| PDeflateError::NumericOverflow)?,
    ))
    .map_err(|_| PDeflateError::NumericOverflow)?
    .max(4);
    let error_size = u64::try_from(
        caps.error_cap_words
            .saturating_mul(std::mem::size_of::<u32>()),
    )
    .map_err(|_| PDeflateError::NumericOverflow)?
    .max(4);
    let class_label = match class {
        GpuDecodeSlotClass::Normal => "normal",
        GpuDecodeSlotClass::Large => "large",
    };
    let cmd_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-cmd-slot-{}",
            class_label, local_index
        )),
        size: cmd_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let section_meta_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-section-meta-slot-{}",
            class_label, local_index
        )),
        size: section_meta_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let table_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-table-slot-{}",
            class_label, local_index
        )),
        size: table_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-out-slot-{}",
            class_label, local_index
        )),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-out-readback-slot-{}",
            class_label, local_index
        )),
        size: out_size
            .checked_add(error_size)
            .ok_or(PDeflateError::NumericOverflow)?,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let error_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!(
            "cozip-pdeflate-decode-v2-{}-error-slot-{}",
            class_label, local_index
        )),
        size: error_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let decode_bind_group = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!(
                "cozip-pdeflate-decode-v2-{}-bg-slot-{}",
                class_label, local_index
            )),
            layout: &runtime.decode_v2_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cmd_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: section_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: error_buffer.as_entire_binding(),
                },
            ],
        });
    Ok(GpuDecodeSlotEntry {
        slot_index,
        generation: 0,
        state: GpuDecodeSlotState::Free,
        caps,
        buffers: GpuDecodeSlotBuffers {
            cmd_buffer,
            section_meta_buffer,
            table_buffer,
            out_buffer,
            out_readback_buffer,
            error_buffer,
        },
        decode_bind_group,
    })
}

fn ensure_gpu_decode_slot_group(
    runtime: &GpuMatchRuntime,
    group: &mut GpuDecodeSlotGroup,
    class: GpuDecodeSlotClass,
    target_count: usize,
    target_caps: GpuDecodeSlotCaps,
) -> Result<(), PDeflateError> {
    if target_count == 0 {
        group.caps = target_caps;
        group.slots.clear();
        return Ok(());
    }
    let merged_caps = group.caps.max_with(target_caps);
    if group.slots.is_empty() || merged_caps != group.caps {
        group.caps = merged_caps;
        group.slots.clear();
    } else {
        group.caps = merged_caps;
    }
    while group.slots.len() < target_count {
        let local_index = group.slots.len();
        group.slots.push(create_gpu_decode_slot_entry(
            runtime,
            class,
            local_index,
            group.caps,
        )?);
    }
    Ok(())
}

fn recycle_gpu_decode_ready_slots(group: &mut GpuDecodeSlotGroup) {
    for slot in &mut group.slots {
        if slot.state == GpuDecodeSlotState::Ready {
            slot.state = GpuDecodeSlotState::Free;
        }
    }
}

fn acquire_gpu_decode_slot(
    group: &mut GpuDecodeSlotGroup,
    req: GpuDecodeSlotRequirement,
) -> Option<(usize, GpuDecodeSlot)> {
    if !group.caps.fits(req) {
        return None;
    }
    for (idx, slot) in group.slots.iter_mut().enumerate() {
        if slot.state == GpuDecodeSlotState::Free {
            slot.state = GpuDecodeSlotState::Inflight;
            slot.generation = slot.generation.saturating_add(1);
            return Some((
                idx,
                GpuDecodeSlot {
                    slot_index: slot.slot_index,
                    generation: slot.generation,
                },
            ));
        }
    }
    None
}

fn acquire_preferred_gpu_decode_slot(
    group: &mut GpuDecodeSlotGroup,
    req: GpuDecodeSlotRequirement,
    preferred: GpuDecodeSlot,
) -> Option<(usize, GpuDecodeSlot)> {
    if !group.caps.fits(req) {
        return None;
    }
    for (idx, slot) in group.slots.iter_mut().enumerate() {
        if slot.slot_index != preferred.slot_index
            || slot.generation != preferred.generation
            || slot.state != GpuDecodeSlotState::Free
        {
            continue;
        }
        slot.state = GpuDecodeSlotState::Inflight;
        slot.generation = slot.generation.saturating_add(1);
        return Some((
            idx,
            GpuDecodeSlot {
                slot_index: slot.slot_index,
                generation: slot.generation,
            },
        ));
    }
    None
}

fn mark_gpu_decode_slot_ready(group: &mut GpuDecodeSlotGroup, slot_idx: usize) {
    if let Some(slot) = group.slots.get_mut(slot_idx) {
        if slot.state == GpuDecodeSlotState::Inflight {
            slot.state = GpuDecodeSlotState::Ready;
        }
    }
}

fn mark_gpu_decode_slot_free(group: &mut GpuDecodeSlotGroup, slot_idx: usize) {
    if let Some(slot) = group.slots.get_mut(slot_idx) {
        slot.state = GpuDecodeSlotState::Free;
    }
}

fn try_acquire_gpu_decode_slot_for_job(
    pool: &mut GpuDecodeSlotPool,
    prepared: PreparedGpuDecodeJob<'_>,
) -> Option<(GpuDecodeSlotClass, usize, GpuDecodeSlot)> {
    let prefer_large = !pool.normal.caps.fits(prepared.req);
    if let Some(preferred) = prepared.job.preferred_slot {
        if preferred.slot_index >= GPU_DECODE_LARGE_SLOT_INDEX_BASE {
            if let Some((idx, slot)) =
                acquire_preferred_gpu_decode_slot(&mut pool.large, prepared.req, preferred)
            {
                return Some((GpuDecodeSlotClass::Large, idx, slot));
            }
        } else if let Some((idx, slot)) =
            acquire_preferred_gpu_decode_slot(&mut pool.normal, prepared.req, preferred)
        {
            return Some((GpuDecodeSlotClass::Normal, idx, slot));
        }
    }
    if prefer_large {
        if let Some((idx, slot)) = acquire_gpu_decode_slot(&mut pool.large, prepared.req) {
            return Some((GpuDecodeSlotClass::Large, idx, slot));
        }
        if let Some((idx, slot)) = acquire_gpu_decode_slot(&mut pool.normal, prepared.req) {
            return Some((GpuDecodeSlotClass::Normal, idx, slot));
        }
    } else {
        if let Some((idx, slot)) = acquire_gpu_decode_slot(&mut pool.normal, prepared.req) {
            return Some((GpuDecodeSlotClass::Normal, idx, slot));
        }
        if let Some((idx, slot)) = acquire_gpu_decode_slot(&mut pool.large, prepared.req) {
            return Some((GpuDecodeSlotClass::Large, idx, slot));
        }
    }
    None
}

fn build_gpu_decode_table_repeat_bytes(
    payload: &[u8],
    layout: GpuDecodePayloadLayout,
) -> Result<Vec<u8>, PDeflateError> {
    let table_index = payload
        .get(layout.table_index_start..layout.table_data_start)
        .ok_or(PDeflateError::InvalidStream(
            "gpu decode table index slice out of bounds",
        ))?;
    if table_index.len() != layout.table_count {
        return Err(PDeflateError::InvalidStream(
            "gpu decode table index length mismatch",
        ));
    }
    let table_data = payload
        .get(layout.table_data_start..layout.table_end)
        .ok_or(PDeflateError::InvalidStream(
            "gpu decode table data slice out of bounds",
        ))?;
    let mut table_offsets = Vec::with_capacity(layout.table_count.saturating_add(1));
    table_offsets.push(0usize);
    for &entry_len_u8 in table_index {
        if entry_len_u8 == 0 {
            return Err(PDeflateError::InvalidStream(
                "gpu decode table entry len is zero",
            ));
        }
        let next = table_offsets
            .last()
            .copied()
            .ok_or(PDeflateError::NumericOverflow)?
            .checked_add(usize::from(entry_len_u8))
            .ok_or(PDeflateError::NumericOverflow)?;
        table_offsets.push(next);
    }
    if table_offsets.last().copied().unwrap_or(0) != table_data.len() {
        return Err(PDeflateError::InvalidStream(
            "gpu decode table data length mismatch",
        ));
    }
    let repeat_total = layout
        .table_count
        .checked_mul(GPU_DECODE_TABLE_REPEAT_STRIDE)
        .ok_or(PDeflateError::NumericOverflow)?;
    let mut table_repeat = vec![0u8; repeat_total];
    for id in 0..layout.table_count {
        let t0 = table_offsets[id];
        let t1 = table_offsets[id + 1];
        let entry = table_data.get(t0..t1).ok_or(PDeflateError::InvalidStream(
            "gpu decode table entry range invalid",
        ))?;
        if entry.is_empty() {
            return Err(PDeflateError::InvalidStream("gpu decode empty table entry"));
        }
        let dst = &mut table_repeat
            [id * GPU_DECODE_TABLE_REPEAT_STRIDE..(id + 1) * GPU_DECODE_TABLE_REPEAT_STRIDE];
        for i in 0..dst.len() {
            dst[i] = entry[i % entry.len()];
        }
    }
    Ok(table_repeat)
}

fn upload_gpu_decode_job_to_slot(
    runtime: &GpuMatchRuntime,
    slot: &GpuDecodeSlotEntry,
    prepared: PreparedGpuDecodeJob<'_>,
) -> Result<(), PDeflateError> {
    let table_repeat_bytes =
        build_gpu_decode_table_repeat_bytes(prepared.job.payload, prepared.layout)?;
    let cmd_bytes = prepared
        .job
        .payload
        .get(prepared.layout.cmd_start..)
        .ok_or(PDeflateError::InvalidStream(
            "gpu decode cmd slice out of bounds",
        ))?;
    let mut section_meta_words = Vec::<u32>::with_capacity(
        GPU_DECODE_SECTION_META_HEADER_WORDS
            .saturating_add(prepared.job.section_meta.len() * GPU_DECODE_SECTION_META_WORDS),
    );
    section_meta_words.push(
        u32::try_from(prepared.job.section_meta.len())
            .map_err(|_| PDeflateError::NumericOverflow)?,
    );
    section_meta_words.push(
        u32::try_from(prepared.layout.table_count).map_err(|_| PDeflateError::NumericOverflow)?,
    );
    section_meta_words.push(
        u32::try_from(GPU_DECODE_TABLE_REPEAT_STRIDE)
            .map_err(|_| PDeflateError::NumericOverflow)?,
    );
    section_meta_words.push(0u32);
    for section in &prepared.job.section_meta {
        let cmd_end = section
            .cmd_offset
            .checked_add(section.cmd_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        if cmd_end > cmd_bytes.len() {
            return Err(PDeflateError::InvalidStream(
                "gpu decode section cmd range out of bounds",
            ));
        }
        let out_end = section
            .out_offset
            .checked_add(section.out_len)
            .ok_or(PDeflateError::NumericOverflow)?;
        if out_end > prepared.job.chunk_uncompressed_len {
            return Err(PDeflateError::InvalidStream(
                "gpu decode section out range out of bounds",
            ));
        }
        section_meta_words
            .push(u32::try_from(section.cmd_offset).map_err(|_| PDeflateError::NumericOverflow)?);
        section_meta_words
            .push(u32::try_from(section.cmd_len).map_err(|_| PDeflateError::NumericOverflow)?);
        section_meta_words
            .push(u32::try_from(section.out_offset).map_err(|_| PDeflateError::NumericOverflow)?);
        section_meta_words
            .push(u32::try_from(section.out_len).map_err(|_| PDeflateError::NumericOverflow)?);
    }
    if !cmd_bytes.is_empty() {
        let cmd_words = pack_bytes_to_words(cmd_bytes);
        runtime.queue.write_buffer(
            &slot.buffers.cmd_buffer,
            0,
            bytemuck::cast_slice(&cmd_words),
        );
    }
    if !table_repeat_bytes.is_empty() {
        let table_words = pack_bytes_to_words(&table_repeat_bytes);
        runtime.queue.write_buffer(
            &slot.buffers.table_buffer,
            0,
            bytemuck::cast_slice(&table_words),
        );
    }
    runtime.queue.write_buffer(
        &slot.buffers.section_meta_buffer,
        0,
        bytemuck::cast_slice(&section_meta_words),
    );
    let zero = vec![0u32; prepared.job.section_meta.len().max(1)];
    runtime
        .queue
        .write_buffer(&slot.buffers.error_buffer, 0, bytemuck::cast_slice(&zero));
    Ok(())
}

fn submit_gpu_decode_job_on_slot(
    runtime: &GpuMatchRuntime,
    slot: &GpuDecodeSlotEntry,
    prepared: PreparedGpuDecodeJob<'_>,
    kernel_ts_probe: bool,
) -> Result<PendingGpuDecodeMap, PDeflateError> {
    let out_data_copy_len = u64::try_from(align_up4(prepared.job.chunk_uncompressed_len))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .max(4);
    let error_words = prepared.job.section_meta.len().max(1);
    let err_copy_len = u64::try_from(error_words.saturating_mul(std::mem::size_of::<u32>()))
        .map_err(|_| PDeflateError::NumericOverflow)?
        .max(4);
    let out_total_copy_len = out_data_copy_len
        .checked_add(err_copy_len)
        .ok_or(PDeflateError::NumericOverflow)?;
    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-pdeflate-decode-v2-encoder"),
        });
    let mut ts_readback_buffer = None;
    let mut ts_query_set = None;
    let mut ts_resolve_buffer = None;
    if kernel_ts_probe {
        ts_query_set = Some(runtime.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("cozip-pdeflate-decode-v2-ts-query"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        }));
        ts_resolve_buffer = Some(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-decode-v2-ts-resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        ts_readback_buffer = Some(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-decode-v2-ts-readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-decode-v2-pass"),
            timestamp_writes: ts_query_set.as_ref().map(|query_set| {
                wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }
            }),
        });
        pass.set_pipeline(&runtime.decode_v2_pipeline);
        pass.set_bind_group(0, &slot.decode_bind_group, &[]);
        let section_count = u32::try_from(prepared.job.section_meta.len())
            .map_err(|_| PDeflateError::NumericOverflow)?;
        let groups_x = section_count.div_ceil(GPU_DECODE_V2_WORKGROUP_SIZE).max(1);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    if let (Some(query_set), Some(resolve_buffer), Some(readback_buffer)) = (
        ts_query_set.as_ref(),
        ts_resolve_buffer.as_ref(),
        ts_readback_buffer.as_ref(),
    ) {
        encoder.resolve_query_set(query_set, 0..2, resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(resolve_buffer, 0, readback_buffer, 0, 16);
    }
    encoder.copy_buffer_to_buffer(
        &slot.buffers.out_buffer,
        0,
        &slot.buffers.out_readback_buffer,
        0,
        out_data_copy_len,
    );
    encoder.copy_buffer_to_buffer(
        &slot.buffers.error_buffer,
        0,
        &slot.buffers.out_readback_buffer,
        out_data_copy_len,
        err_copy_len,
    );
    let submit_seq = next_gpu_submit_seq();
    let t_submit = Instant::now();
    let _submission = runtime.queue.submit(Some(encoder.finish()));
    let submit_ms = elapsed_ms(t_submit);
    let (done_tx, done_rx) = mpsc::channel();
    runtime.queue.on_submitted_work_done(move || {
        let _ = done_tx.send(());
    });
    Ok(PendingGpuDecodeMap {
        out_data_copy_len,
        out_total_copy_len,
        out_len: prepared.job.chunk_uncompressed_len,
        error_words,
        submit_seq,
        submit_ms,
        kernel_timestamp_ms: if kernel_ts_probe {
            None
        } else {
            Some(f64::NAN)
        },
        ts_readback_buffer,
        ts_map_rx: None,
        submit_at: Instant::now(),
        completion_rx: done_rx,
        map_rx: None,
    })
}

fn collect_gpu_decode_job_from_slot(
    slot: &GpuDecodeSlotEntry,
    pending: &PendingGpuDecodeMap,
) -> Result<(Option<Vec<u8>>, f64), PDeflateError> {
    let t_collect = Instant::now();
    let error_bytes = pending
        .error_words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(PDeflateError::NumericOverflow)?;
    let out_slice = slot
        .buffers
        .out_readback_buffer
        .slice(..pending.out_total_copy_len);
    let mapped = out_slice.get_mapped_range();
    let out_data_offset =
        usize::try_from(pending.out_data_copy_len).map_err(|_| PDeflateError::NumericOverflow)?;
    let err_end = out_data_offset
        .checked_add(error_bytes)
        .ok_or(PDeflateError::NumericOverflow)?
        .min(mapped.len());
    let mut has_error = false;
    if pending.out_len > out_data_offset {
        drop(mapped);
        slot.buffers.out_readback_buffer.unmap();
        return Err(PDeflateError::Gpu(
            "gpu decode readback payload length out of bounds".to_string(),
        ));
    }
    if pending.error_words > 0 {
        if err_end < out_data_offset || err_end > mapped.len() {
            drop(mapped);
            slot.buffers.out_readback_buffer.unmap();
            return Err(PDeflateError::Gpu(
                "gpu decode readback error trailer out of bounds".to_string(),
            ));
        }
        let err_words: &[u32] = bytemuck::cast_slice(&mapped[out_data_offset..err_end]);
        has_error = err_words
            .iter()
            .take(pending.error_words)
            .copied()
            .any(|code| code != 0);
    }
    let result = if has_error {
        Ok(None)
    } else {
        let mut decoded = vec![0u8; pending.out_len];
        if !decoded.is_empty() {
            decoded.copy_from_slice(&mapped[..pending.out_len]);
        }
        Ok(Some(decoded))
    };
    drop(mapped);
    slot.buffers.out_readback_buffer.unmap();
    result.map(|decoded_chunk| (decoded_chunk, elapsed_ms(t_collect)))
}

fn gpu_decode_section_boundaries_word_aligned(section_meta: &[GpuDecodeSectionMeta]) -> bool {
    // decode-v2 section-parallel kernel writes bytes via u32 RMW; internal section boundaries
    // must be word-aligned to avoid cross-section word races.
    section_meta
        .iter()
        .enumerate()
        .all(|(idx, sec)| idx == 0 || (sec.out_offset & 3) == 0)
}

pub(crate) fn decode_chunks_gpu_v2(
    jobs: &[GpuDecodeJob<'_>],
    options: &PDeflateOptions,
) -> Result<Vec<GpuDecodeResult>, PDeflateError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let gpu_requested = options.gpu_decompress_enabled || options.gpu_decompress_force_gpu;
    if !gpu_requested || !is_runtime_available() {
        return Ok(decode_fallback_results(jobs));
    }
    let runtime = runtime()?;

    let mut out = Vec::with_capacity(jobs.len());
    let mut prepared_jobs = Vec::with_capacity(jobs.len());
    let mut pre_fallback_unaligned_count = 0usize;
    let mut pre_fallback_unaligned_first_chunk = None::<usize>;
    for job in jobs {
        if !gpu_decode_section_boundaries_word_aligned(&job.section_meta) {
            pre_fallback_unaligned_count = pre_fallback_unaligned_count.saturating_add(1);
            pre_fallback_unaligned_first_chunk.get_or_insert(job.chunk_index);
            out.push(GpuDecodeResult {
                chunk_index: job.chunk_index,
                slot: None,
                disposition: GpuDecodeDisposition::CpuFallback,
                decoded_chunk: None,
            });
            continue;
        }
        let layout = parse_gpu_decode_payload_layout(job.payload)?;
        let req = build_gpu_decode_slot_requirement(job, layout)?;
        prepared_jobs.push(PreparedGpuDecodeJob { job, req, layout });
    }
    if pre_fallback_unaligned_count > 0 {
        eprintln!(
            "[cozip_pdeflate][timing][decode-v2-fallback] reason=unaligned_section_boundary chunks={} first_chunk_idx={} gpu_force={} note=\"section out_offset not 4-byte aligned; cpu fallback\"",
            pre_fallback_unaligned_count,
            pre_fallback_unaligned_first_chunk.unwrap_or(usize::MAX),
            options.gpu_decompress_force_gpu
        );
    }
    if prepared_jobs.is_empty() {
        out.sort_by_key(|r| r.chunk_index);
        return Ok(out);
    }

    let (normal_target_count, large_target_count) = gpu_decode_slot_split(options.gpu_slot_count);
    let normal_base = gpu_decode_normal_caps(options)?;
    let mut large_base = gpu_decode_large_caps(normal_base);
    let mut normal_req_max = GpuDecodeSlotCaps::default();
    let mut large_req_max = GpuDecodeSlotCaps::default();
    for prepared in &prepared_jobs {
        let req_caps = GpuDecodeSlotCaps {
            cmd_cap_bytes: prepared.req.cmd_bytes,
            section_meta_cap: prepared.req.section_meta_count,
            table_cap_bytes: prepared.req.table_bytes,
            out_cap_bytes: prepared.req.out_bytes,
            error_cap_words: prepared.req.error_words,
        };
        if normal_base.fits(prepared.req) {
            normal_req_max = normal_req_max.max_with(req_caps);
        } else {
            large_req_max = large_req_max.max_with(req_caps);
        }
    }
    let normal_target_caps = normal_base.max_with(normal_req_max);
    if large_req_max.cmd_cap_bytes > 0
        || large_req_max.section_meta_cap > 0
        || large_req_max.table_cap_bytes > 0
        || large_req_max.out_cap_bytes > 0
        || large_req_max.error_cap_words > 0
    {
        large_base = large_base.max_with(large_req_max);
    }

    let mut pool = runtime
        .decode_slot_pool
        .lock()
        .map_err(|_| PDeflateError::Gpu("gpu decode slot pool mutex poisoned".to_string()))?;
    ensure_gpu_decode_slot_group(
        runtime,
        &mut pool.normal,
        GpuDecodeSlotClass::Normal,
        normal_target_count,
        normal_target_caps,
    )?;
    ensure_gpu_decode_slot_group(
        runtime,
        &mut pool.large,
        GpuDecodeSlotClass::Large,
        large_target_count,
        large_base,
    )?;

    let micro_submit = options.gpu_submit_chunks.max(1);
    let max_inflight = options.gpu_slot_count.max(1);
    let wait_probe_enabled = gpu_decode_v2_wait_probe_enabled();
    let kernel_ts_probe = wait_probe_enabled && runtime.supports_timestamp_query;
    if wait_probe_enabled && !runtime.supports_timestamp_query {
        let _ = GPU_DECODE_V2_KERNEL_TS_UNSUPPORTED_LOGGED.get_or_init(|| {
            eprintln!(
                "[cozip_pdeflate][timing][decode-v2-wait-breakdown] status=disabled reason=timestamp_query_not_supported"
            );
        });
    }
    let mut next_submit_idx = 0usize;
    let mut inflight = Vec::<InflightGpuDecodeSubmission<'_>>::new();
    let mut wait_probe_submit_ms = 0.0_f64;
    let mut wait_probe_submit_done_wait_ms = 0.0_f64;
    let mut wait_probe_map_callback_wait_ms = 0.0_f64;
    let mut wait_probe_map_copy_ms = 0.0_f64;
    let mut wait_probe_kernel_timestamp_ms = 0.0_f64;
    let mut wait_probe_kernel_timestamp_samples = 0usize;
    let mut wait_probe_queue_stall_est_ms = 0.0_f64;
    let mut wait_probe_ready_any_hits = 0usize;
    let mut wait_probe_wait_loops = 0usize;
    let mut wait_probe_inflight_batches_sum = 0usize;
    let mut wait_probe_inflight_batches_max = 0usize;
    let mut wait_probe_inflight_samples = 0usize;
    let mut wait_probe_completed_jobs = 0usize;

    while next_submit_idx < prepared_jobs.len() || !inflight.is_empty() {
        recycle_gpu_decode_ready_slots(&mut pool.normal);
        recycle_gpu_decode_ready_slots(&mut pool.large);

        let mut submitted_in_round = 0usize;
        while next_submit_idx < prepared_jobs.len()
            && submitted_in_round < micro_submit
            && inflight.len() < max_inflight
        {
            let prepared = prepared_jobs[next_submit_idx];
            let acquired = try_acquire_gpu_decode_slot_for_job(&mut pool, prepared);
            let Some((class, slot_idx, slot_handle)) = acquired else {
                let fit_any =
                    pool.normal.caps.fits(prepared.req) || pool.large.caps.fits(prepared.req);
                if !fit_any {
                    out.push(GpuDecodeResult {
                        chunk_index: prepared.job.chunk_index,
                        slot: None,
                        disposition: GpuDecodeDisposition::CpuFallback,
                        decoded_chunk: None,
                    });
                    next_submit_idx = next_submit_idx.saturating_add(1);
                    submitted_in_round = submitted_in_round.saturating_add(1);
                    continue;
                }
                break;
            };

            let submit_result = match class {
                GpuDecodeSlotClass::Normal => {
                    let slot = &pool.normal.slots[slot_idx];
                    upload_gpu_decode_job_to_slot(runtime, slot, prepared).and_then(|_| {
                        submit_gpu_decode_job_on_slot(runtime, slot, prepared, kernel_ts_probe)
                    })
                }
                GpuDecodeSlotClass::Large => {
                    let slot = &pool.large.slots[slot_idx];
                    upload_gpu_decode_job_to_slot(runtime, slot, prepared).and_then(|_| {
                        submit_gpu_decode_job_on_slot(runtime, slot, prepared, kernel_ts_probe)
                    })
                }
            };
            match submit_result {
                Ok(pending) => {
                    if wait_probe_enabled {
                        wait_probe_submit_ms += pending.submit_ms;
                    }
                    inflight.push(InflightGpuDecodeSubmission {
                        prepared,
                        class,
                        slot_idx,
                        slot_handle,
                        pending,
                        map_requested_at: None,
                        completion_done: false,
                        map_done: None,
                        wait_profile: GpuDecodeV2WaitProfile::default(),
                    });
                }
                Err(_) => {
                    match class {
                        GpuDecodeSlotClass::Normal => {
                            mark_gpu_decode_slot_free(&mut pool.normal, slot_idx)
                        }
                        GpuDecodeSlotClass::Large => {
                            mark_gpu_decode_slot_free(&mut pool.large, slot_idx)
                        }
                    }
                    out.push(GpuDecodeResult {
                        chunk_index: prepared.job.chunk_index,
                        slot: None,
                        disposition: GpuDecodeDisposition::CpuFallback,
                        decoded_chunk: None,
                    });
                }
            }
            next_submit_idx = next_submit_idx.saturating_add(1);
            submitted_in_round = submitted_in_round.saturating_add(1);
        }

        if inflight.is_empty() {
            continue;
        }
        if wait_probe_enabled {
            wait_probe_inflight_samples = wait_probe_inflight_samples.saturating_add(1);
            wait_probe_inflight_batches_sum =
                wait_probe_inflight_batches_sum.saturating_add(inflight.len());
            wait_probe_inflight_batches_max = wait_probe_inflight_batches_max.max(inflight.len());
        }

        runtime.device.poll(wgpu::Maintain::Poll);
        let mut completed_in_round = 0usize;
        let mut i = 0usize;
        while i < inflight.len() {
            {
                let pending = &mut inflight[i];
                if !pending.completion_done {
                    match pending.pending.completion_rx.try_recv() {
                        Ok(_) | Err(mpsc::TryRecvError::Disconnected) => {
                            pending.completion_done = true;
                            pending.wait_profile.submit_seq = pending.pending.submit_seq;
                            pending.wait_profile.submit_ms = pending.pending.submit_ms;
                            pending.wait_profile.submit_done_wait_ms =
                                elapsed_ms(pending.pending.submit_at);
                        }
                        Err(mpsc::TryRecvError::Empty) => {}
                    }
                }

                if pending.completion_done
                    && pending.pending.kernel_timestamp_ms.is_none()
                    && pending.pending.ts_map_rx.is_none()
                {
                    if let Some(ts_buffer) = pending.pending.ts_readback_buffer.as_ref() {
                        let (ts_tx, ts_rx) = mpsc::channel();
                        ts_buffer
                            .slice(..)
                            .map_async(wgpu::MapMode::Read, move |res| {
                                let _ = ts_tx.send(res);
                            });
                        pending.pending.ts_map_rx = Some(ts_rx);
                    }
                }

                if pending.pending.kernel_timestamp_ms.is_none() {
                    if let Some(ts_rx) = pending.pending.ts_map_rx.as_ref() {
                        match ts_rx.try_recv() {
                            Ok(Ok(())) => {
                                if let Some(ts_buffer) = pending.pending.ts_readback_buffer.take() {
                                    let mapped = ts_buffer.slice(..).get_mapped_range();
                                    let ts_words: &[u64] = bytemuck::cast_slice(&mapped);
                                    let kernel_ms = if ts_words.len() >= 2
                                        && ts_words[1] >= ts_words[0]
                                    {
                                        let period_ms = (runtime.queue.get_timestamp_period()
                                            as f64)
                                            / 1_000_000.0;
                                        (ts_words[1].saturating_sub(ts_words[0]) as f64) * period_ms
                                    } else {
                                        f64::NAN
                                    };
                                    drop(mapped);
                                    ts_buffer.unmap();
                                    pending.pending.kernel_timestamp_ms = Some(kernel_ms);
                                } else {
                                    pending.pending.kernel_timestamp_ms = Some(f64::NAN);
                                }
                                pending.pending.ts_map_rx = None;
                            }
                            Ok(Err(_)) | Err(mpsc::TryRecvError::Disconnected) => {
                                let _ = pending.pending.ts_readback_buffer.take();
                                pending.pending.kernel_timestamp_ms = Some(f64::NAN);
                                pending.pending.ts_map_rx = None;
                            }
                            Err(mpsc::TryRecvError::Empty) => {}
                        }
                    }
                }

                if pending.completion_done && pending.pending.map_rx.is_none() {
                    let out_slice = match pending.class {
                        GpuDecodeSlotClass::Normal => pool.normal.slots[pending.slot_idx]
                            .buffers
                            .out_readback_buffer
                            .slice(..pending.pending.out_total_copy_len),
                        GpuDecodeSlotClass::Large => pool.large.slots[pending.slot_idx]
                            .buffers
                            .out_readback_buffer
                            .slice(..pending.pending.out_total_copy_len),
                    };
                    let (map_tx, map_rx) = mpsc::channel();
                    out_slice.map_async(wgpu::MapMode::Read, move |res| {
                        let _ = map_tx.send(res);
                    });
                    pending.map_requested_at = Some(Instant::now());
                    pending.pending.map_rx = Some(map_rx);
                }

                if pending.map_done.is_none() {
                    if let Some(map_rx) = pending.pending.map_rx.as_ref() {
                        match map_rx.try_recv() {
                            Ok(res) => {
                                if let Some(t_map_wait) = pending.map_requested_at.take() {
                                    pending.wait_profile.map_callback_wait_ms =
                                        elapsed_ms(t_map_wait);
                                }
                                pending.map_done = Some(res.map_err(|e| {
                                    PDeflateError::Gpu(format!("gpu decode out map failed: {e}"))
                                }));
                            }
                            Err(mpsc::TryRecvError::Empty) => {}
                            Err(mpsc::TryRecvError::Disconnected) => {
                                if let Some(t_map_wait) = pending.map_requested_at.take() {
                                    pending.wait_profile.map_callback_wait_ms =
                                        elapsed_ms(t_map_wait);
                                }
                                pending.map_done = Some(Err(PDeflateError::Gpu(
                                    "gpu decode out map channel closed".to_string(),
                                )));
                            }
                        }
                    }
                }
            }

            let done =
                inflight[i].map_done.is_some() && inflight[i].pending.kernel_timestamp_ms.is_some();
            if !done {
                i = i.saturating_add(1);
                continue;
            }

            let mut submission = inflight.swap_remove(i);
            let map_done = submission.map_done.take();
            let map_ok = matches!(map_done, Some(Ok(())));

            let decode_result = if map_ok {
                let collected = match submission.class {
                    GpuDecodeSlotClass::Normal => {
                        let slot = &pool.normal.slots[submission.slot_idx];
                        collect_gpu_decode_job_from_slot(slot, &submission.pending)
                    }
                    GpuDecodeSlotClass::Large => {
                        let slot = &pool.large.slots[submission.slot_idx];
                        collect_gpu_decode_job_from_slot(slot, &submission.pending)
                    }
                };
                match collected {
                    Ok((decoded_chunk, map_copy_ms)) => {
                        submission.wait_profile.map_copy_ms = map_copy_ms;
                        GpuDecodeResult {
                            chunk_index: submission.prepared.job.chunk_index,
                            slot: Some(submission.slot_handle),
                            disposition: if decoded_chunk.is_some() {
                                GpuDecodeDisposition::SubmittedGpu
                            } else {
                                GpuDecodeDisposition::CpuFallback
                            },
                            decoded_chunk,
                        }
                    }
                    Err(_) => GpuDecodeResult {
                        chunk_index: submission.prepared.job.chunk_index,
                        slot: None,
                        disposition: GpuDecodeDisposition::CpuFallback,
                        decoded_chunk: None,
                    },
                }
            } else {
                GpuDecodeResult {
                    chunk_index: submission.prepared.job.chunk_index,
                    slot: None,
                    disposition: GpuDecodeDisposition::CpuFallback,
                    decoded_chunk: None,
                }
            };
            if wait_probe_enabled {
                submission.wait_profile.kernel_timestamp_ms =
                    submission.pending.kernel_timestamp_ms.unwrap_or(f64::NAN);
                let kernel_ms = submission.wait_profile.kernel_timestamp_ms;
                submission.wait_profile.queue_stall_est_ms = if kernel_ms.is_finite() {
                    (submission.wait_profile.submit_done_wait_ms - kernel_ms).max(0.0)
                } else {
                    submission.wait_profile.submit_done_wait_ms.max(0.0)
                };
                wait_probe_submit_done_wait_ms += submission.wait_profile.submit_done_wait_ms;
                wait_probe_map_callback_wait_ms += submission.wait_profile.map_callback_wait_ms;
                wait_probe_map_copy_ms += submission.wait_profile.map_copy_ms;
                wait_probe_queue_stall_est_ms += submission.wait_profile.queue_stall_est_ms;
                if kernel_ms.is_finite() {
                    wait_probe_kernel_timestamp_ms += kernel_ms;
                    wait_probe_kernel_timestamp_samples =
                        wait_probe_kernel_timestamp_samples.saturating_add(1);
                }
                wait_probe_completed_jobs = wait_probe_completed_jobs.saturating_add(1);
                let seq = GPU_DECODE_V2_WAIT_PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
                if gpu_decode_v2_wait_probe_should_log(seq) {
                    let kernel_ts_status = if kernel_ms.is_finite() {
                        "ok"
                    } else {
                        "disabled"
                    };
                    eprintln!(
                        "[cozip_pdeflate][timing][decode-v2-wait-breakdown] seq={} chunk_idx={} slot_idx={} submit_seq={} submit_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_ms={:.3} map_copy_ms={:.3} kernel_timestamp_ms={:.3} queue_stall_est_ms={:.3} kernel_ts_status={}",
                        seq,
                        submission.prepared.job.chunk_index,
                        submission.slot_handle.slot_index,
                        submission.wait_profile.submit_seq,
                        submission.wait_profile.submit_ms,
                        submission.wait_profile.submit_done_wait_ms,
                        submission.wait_profile.map_callback_wait_ms,
                        submission.wait_profile.map_copy_ms,
                        if kernel_ms.is_finite() {
                            kernel_ms
                        } else {
                            0.0
                        },
                        submission.wait_profile.queue_stall_est_ms,
                        kernel_ts_status
                    );
                }
            }

            match submission.class {
                GpuDecodeSlotClass::Normal => {
                    mark_gpu_decode_slot_ready(&mut pool.normal, submission.slot_idx)
                }
                GpuDecodeSlotClass::Large => {
                    mark_gpu_decode_slot_ready(&mut pool.large, submission.slot_idx)
                }
            }
            out.push(decode_result);
            completed_in_round = completed_in_round.saturating_add(1);
        }

        if completed_in_round == 0 {
            if wait_probe_enabled {
                wait_probe_wait_loops = wait_probe_wait_loops.saturating_add(1);
            }
            runtime.device.poll(wgpu::Maintain::Wait);
        } else if wait_probe_enabled {
            wait_probe_ready_any_hits = wait_probe_ready_any_hits.saturating_add(1);
        }
    }

    out.sort_by_key(|r| r.chunk_index);
    recycle_gpu_decode_ready_slots(&mut pool.normal);
    recycle_gpu_decode_ready_slots(&mut pool.large);
    if wait_probe_enabled {
        let inflight_avg = if wait_probe_inflight_samples > 0 {
            (wait_probe_inflight_batches_sum as f64) / (wait_probe_inflight_samples as f64)
        } else {
            0.0
        };
        let kernel_ts_status = if wait_probe_kernel_timestamp_samples > 0 {
            "ok"
        } else {
            "disabled"
        };
        eprintln!(
            "[cozip_pdeflate][timing][decode-v2-ready-any] jobs={} submit_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_ms={:.3} map_copy_ms={:.3} kernel_timestamp_ms={:.3} queue_stall_est_ms={:.3} ready_any_hits={} wait_loops={} inflight_batches_avg={:.2} inflight_batches_max={} inflight_samples={} kernel_ts_status={}",
            wait_probe_completed_jobs,
            wait_probe_submit_ms,
            wait_probe_submit_done_wait_ms,
            wait_probe_map_callback_wait_ms,
            wait_probe_map_copy_ms,
            wait_probe_kernel_timestamp_ms,
            wait_probe_queue_stall_est_ms,
            wait_probe_ready_any_hits,
            wait_probe_wait_loops,
            inflight_avg,
            wait_probe_inflight_batches_max,
            wait_probe_inflight_samples,
            kernel_ts_status
        );
    }
    Ok(out)
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
        build_id: 0,
        build_submit_seq: 0,
        build_submit_index: None,
        build_breakdown: GpuTableBuildBreakdown::default(),
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
    let t_total_build = Instant::now();
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
                build_id: 0,
                build_submit_seq: 0,
                build_submit_index: None,
                build_breakdown: GpuTableBuildBreakdown::default(),
            },
            GpuBuildTableProfile::default(),
        ));
    }
    let build_id = BUILD_TABLE_DEVICE_SEQ
        .fetch_add(1, Ordering::Relaxed)
        .saturating_add(1);
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
    let stage_probe_enabled = table_stage_probe_enabled() && r.supports_timestamp_query;
    if table_stage_probe_enabled() && !r.supports_timestamp_query {
        let _ = BUILD_TABLE_STAGE_PROBE_UNSUPPORTED_LOGGED.get_or_init(|| {
            eprintln!(
                "[cozip_pdeflate][timing][gpu-table-stage-probe] status=disabled reason=timestamp_query_not_supported"
            );
        });
    }
    let mut stage_gpu_ms = [0.0_f64; BUILD_TABLE_STAGE_COUNT];
    let mut stage_probe_error: Option<String> = None;
    let mut stage_probe_submit_ms = 0.0_f64;
    let mut stage_probe_wait_ms = 0.0_f64;
    let mut stage_probe_parse_ms = 0.0_f64;
    let mut stage_probe_query_set = None;
    let mut stage_probe_resolve_buffer = None;
    let mut stage_probe_readback_buffer = None;
    if stage_probe_enabled {
        let ts_bytes = u64::from(BUILD_TABLE_STAGE_QUERY_COUNT).saturating_mul(8);
        stage_probe_query_set = Some(r.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("cozip-pdeflate-bt-stage-ts-qs"),
            ty: wgpu::QueryType::Timestamp,
            count: BUILD_TABLE_STAGE_QUERY_COUNT,
        }));
        stage_probe_resolve_buffer = Some(r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-bt-stage-ts-resolve"),
            size: ts_bytes.max(8),
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        stage_probe_readback_buffer = Some(r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-pdeflate-bt-stage-ts-readback"),
            size: ts_bytes.max(8),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }

    let t_resource_setup = Instant::now();
    let pre_resource_ms = elapsed_ms(t_total_build);
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
    let resource_setup_ms = elapsed_ms(t_resource_setup);

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

    let t_encode_pass1 = Instant::now();
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
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 0);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-freq-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_freq_pipeline);
        pass.set_bind_group(0, &freq_bind_group, &[]);
        pass.dispatch_workgroups(freq_groups, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 1);
    }
    let freq_kernel_ms = elapsed_ms(t_freq);

    let cand_groups = u32::try_from(sample_count)
        .map_err(|_| PDeflateError::NumericOverflow)?
        .div_ceil(128)
        .max(1);
    let t_cand = Instant::now();
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 2);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-candidate-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_candidate_pipeline);
        pass.set_bind_group(0, &cand_bind_group, &[]);
        pass.dispatch_workgroups(cand_groups, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 3);
    }
    let mut candidate_kernel_ms = elapsed_ms(t_cand);

    encoder.clear_buffer(&bucket_count_buffer, 0, None);
    let t_bucket_count = Instant::now();
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 4);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-bucket-count-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_bucket_count_pipeline);
        pass.set_bind_group(0, &bucket_count_bind_group, &[]);
        pass.dispatch_workgroups(cand_groups, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder.write_timestamp(query_set, 5);
    }
    candidate_kernel_ms += elapsed_ms(t_bucket_count);
    let encode_pass1_ms = elapsed_ms(t_encode_pass1);
    let t_submit_pass1 = Instant::now();
    let pass1_submit_seq = next_gpu_submit_seq();
    let _pass1_submission = r.queue.submit(Some(encoder.finish()));
    let submit_pass1_ms = elapsed_ms(t_submit_pass1);

    let readback_ms = 0.0;

    let t_encode_pass2 = Instant::now();
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
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 6);
    }
    {
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-bucket-prefix-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_bucket_prefix_pipeline);
        pass.set_bind_group(0, &bucket_prefix_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 7);
    }
    candidate_kernel_ms += elapsed_ms(t_bucket_prefix);
    let t_bucket_scatter = Instant::now();
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 8);
    }
    {
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-bucket-scatter-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_bucket_scatter_pipeline);
        pass.set_bind_group(0, &bucket_scatter_bind_group, &[]);
        pass.dispatch_workgroups(cand_groups, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 9);
    }
    candidate_kernel_ms += elapsed_ms(t_bucket_scatter);
    let t_finalize_kernel = Instant::now();
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 10);
    }
    {
        let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-pdeflate-bt-finalize-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&r.build_table_finalize_pipeline);
        pass.set_bind_group(0, &finalize_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 11);
    }
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 12);
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
    if let Some(query_set) = stage_probe_query_set.as_ref() {
        encoder2.write_timestamp(query_set, 13);
    }
    let encode_pass2_ms = elapsed_ms(t_encode_pass2);
    let sort_ms = elapsed_ms(t_finalize_kernel);
    let t_submit_pass2 = Instant::now();
    let pass2_submit_seq = next_gpu_submit_seq();
    let pass2_submission = r.queue.submit(Some(encoder2.finish()));
    let submit_pass2_ms = elapsed_ms(t_submit_pass2);
    if let (Some(query_set), Some(resolve_buffer), Some(readback_buffer)) = (
        stage_probe_query_set.take(),
        stage_probe_resolve_buffer.take(),
        stage_probe_readback_buffer.take(),
    ) {
        let ts_bytes = u64::from(BUILD_TABLE_STAGE_QUERY_COUNT).saturating_mul(8);
        let mut ts_encoder = r
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-pdeflate-bt-stage-ts-resolve-encoder"),
            });
        ts_encoder.resolve_query_set(
            &query_set,
            0..BUILD_TABLE_STAGE_QUERY_COUNT,
            &resolve_buffer,
            0,
        );
        ts_encoder.copy_buffer_to_buffer(&resolve_buffer, 0, &readback_buffer, 0, ts_bytes.max(8));
        let t_probe_submit = Instant::now();
        let _stage_probe_submit_seq = next_gpu_submit_seq();
        let submission = r.queue.submit(Some(ts_encoder.finish()));
        stage_probe_submit_ms = elapsed_ms(t_probe_submit);
        let t_probe_readback = Instant::now();
        let t_probe_wait = Instant::now();
        let slice = readback_buffer.slice(..ts_bytes.max(8));
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        r.device.poll(wgpu::Maintain::wait_for(submission));
        stage_probe_wait_ms = elapsed_ms(t_probe_wait);
        match rx.recv() {
            Ok(Ok(())) => {
                let t_probe_parse = Instant::now();
                let mapped = slice.get_mapped_range();
                let ticks: &[u64] = bytemuck::cast_slice(&mapped);
                let period_ms = (r.queue.get_timestamp_period() as f64) / 1_000_000.0;
                for stage_idx in 0..BUILD_TABLE_STAGE_COUNT {
                    let begin_idx = stage_idx.saturating_mul(2);
                    let end_idx = begin_idx.saturating_add(1);
                    if end_idx < ticks.len() {
                        let begin = ticks[begin_idx];
                        let end = ticks[end_idx];
                        if end >= begin {
                            stage_gpu_ms[stage_idx] = (end - begin) as f64 * period_ms;
                        }
                    }
                }
                drop(mapped);
                readback_buffer.unmap();
                stage_probe_parse_ms = elapsed_ms(t_probe_parse);
            }
            Ok(Err(e)) => {
                stage_probe_error = Some(format!("map failed: {e}"));
            }
            Err(_) => {
                stage_probe_error = Some("map channel closed".to_string());
            }
        }
        let stage_probe_readback_ms = elapsed_ms(t_probe_readback);
        let seq = BUILD_TABLE_STAGE_PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
        if table_stage_probe_should_log(seq) {
            if let Some(err) = stage_probe_error.as_ref() {
                eprintln!(
                    "[cozip_pdeflate][timing][gpu-table-stage-probe] seq={} sample_count={} map_status=error err=\"{}\"",
                    seq, sample_count, err
                );
            } else {
                let total_stage_ms: f64 = stage_gpu_ms.iter().sum();
                eprintln!(
                    "[cozip_pdeflate][timing][gpu-table-stage-probe] seq={} sample_count={} freq_ms={:.3} candidate_ms={:.3} bucket_count_ms={:.3} bucket_prefix_ms={:.3} bucket_scatter_ms={:.3} finalize_ms={:.3} pack_index_ms={:.3} stage_sum_ms={:.3} ts_readback_ms={:.3}",
                    seq,
                    sample_count,
                    stage_gpu_ms[0],
                    stage_gpu_ms[1],
                    stage_gpu_ms[2],
                    stage_gpu_ms[3],
                    stage_gpu_ms[4],
                    stage_gpu_ms[5],
                    stage_gpu_ms[6],
                    total_stage_ms,
                    stage_probe_readback_ms
                );
            }
        }
    }
    let stage_sum_ms: f64 = stage_gpu_ms.iter().sum();
    let total_build_ms = elapsed_ms(t_total_build);
    let known_ms = pre_resource_ms
        + resource_setup_ms
        + upload_ms
        + encode_pass1_ms
        + submit_pass1_ms
        + encode_pass2_ms
        + submit_pass2_ms
        + stage_probe_submit_ms
        + stage_probe_wait_ms
        + stage_probe_parse_ms;
    let tail_ms = (total_build_ms - known_ms).max(0.0);
    let other_ms = pre_resource_ms + tail_ms;
    if table_build_breakdown_enabled() {
        let seq = BUILD_TABLE_BREAKDOWN_SEQ.fetch_add(1, Ordering::Relaxed);
        if table_stage_probe_should_log(seq) {
            eprintln!(
                "[cozip_pdeflate][timing][gpu-table-build-breakdown] seq={} build_id={} sample_count={} pass1_submit_seq={} pass2_submit_seq={} total_build_ms={:.3} pre_resource_ms={:.3} resource_setup_ms={:.3} upload_ms={:.3} encode_pass1_ms={:.3} submit_pass1_ms={:.3} encode_pass2_ms={:.3} submit_pass2_ms={:.3} stage_probe_submit_ms={:.3} stage_probe_wait_ms={:.3} stage_probe_parse_ms={:.3} stage_gpu_sum_ms={:.3} tail_ms={:.3} other_ms={:.3}",
                seq,
                build_id,
                sample_count,
                pass1_submit_seq,
                pass2_submit_seq,
                total_build_ms,
                pre_resource_ms,
                resource_setup_ms,
                upload_ms,
                encode_pass1_ms,
                submit_pass1_ms,
                encode_pass2_ms,
                submit_pass2_ms,
                stage_probe_submit_ms,
                stage_probe_wait_ms,
                stage_probe_parse_ms,
                stage_sum_ms,
                tail_ms,
                other_ms
            );
        }
    }
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
            build_id,
            build_submit_seq: pass2_submit_seq,
            build_submit_index: Some(pass2_submission),
            build_breakdown: GpuTableBuildBreakdown {
                total_ms: total_build_ms,
                pre_resource_ms,
                resource_setup_ms,
                upload_ms,
                encode_pass1_ms,
                submit_pass1_ms,
                encode_pass2_ms,
                submit_pass2_ms,
                stage_probe_submit_ms,
                stage_probe_wait_ms,
                stage_probe_parse_ms,
                stage_gpu_sum_ms: stage_sum_ms,
                tail_ms,
                other_ms,
            },
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
) -> Result<
    (
        Vec<(usize, usize, usize)>,
        ResolvePackedTableSizesBatchProfile,
    ),
    PDeflateError,
> {
    let t_total = Instant::now();
    let mut profile = ResolvePackedTableSizesBatchProfile::default();
    let mut out = vec![(0usize, 0usize, 0usize); packed_tables.len()];
    let mut unresolved = Vec::<(usize, &GpuPackedTableDevice)>::new();
    let t_scan = Instant::now();
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
    profile.scan_ms = elapsed_ms(t_scan);
    profile.unresolved_count = unresolved.len();
    if unresolved.is_empty() {
        profile.total_ms = elapsed_ms(t_total);
        return Ok((out, profile));
    }
    let mut prebuild_total_ms = 0.0_f64;
    let mut prebuild_pre_resource_ms = 0.0_f64;
    let mut prebuild_resource_setup_ms = 0.0_f64;
    let mut prebuild_upload_ms = 0.0_f64;
    let mut prebuild_encode_pass1_ms = 0.0_f64;
    let mut prebuild_submit_pass1_ms = 0.0_f64;
    let mut prebuild_encode_pass2_ms = 0.0_f64;
    let mut prebuild_submit_pass2_ms = 0.0_f64;
    let mut prebuild_stage_probe_submit_ms = 0.0_f64;
    let mut prebuild_stage_probe_wait_ms = 0.0_f64;
    let mut prebuild_stage_probe_parse_ms = 0.0_f64;
    let mut prebuild_stage_gpu_sum_ms = 0.0_f64;
    let mut prebuild_tail_ms = 0.0_f64;
    let mut prebuild_other_ms = 0.0_f64;
    let mut prebuild_build_id_min = u64::MAX;
    let mut prebuild_build_id_max = 0u64;
    let mut prebuild_submit_seq_min = u64::MAX;
    let mut prebuild_submit_seq_max = 0u64;
    let mut latest_build_submit: Option<(u64, u64, wgpu::SubmissionIndex)> = None;
    let mut unresolved_build_ids = Vec::<u64>::with_capacity(unresolved.len());
    for (_, packed) in unresolved.iter().copied() {
        let b = packed.build_breakdown;
        prebuild_total_ms += b.total_ms;
        prebuild_pre_resource_ms += b.pre_resource_ms;
        prebuild_resource_setup_ms += b.resource_setup_ms;
        prebuild_upload_ms += b.upload_ms;
        prebuild_encode_pass1_ms += b.encode_pass1_ms;
        prebuild_submit_pass1_ms += b.submit_pass1_ms;
        prebuild_encode_pass2_ms += b.encode_pass2_ms;
        prebuild_submit_pass2_ms += b.submit_pass2_ms;
        prebuild_stage_probe_submit_ms += b.stage_probe_submit_ms;
        prebuild_stage_probe_wait_ms += b.stage_probe_wait_ms;
        prebuild_stage_probe_parse_ms += b.stage_probe_parse_ms;
        prebuild_stage_gpu_sum_ms += b.stage_gpu_sum_ms;
        prebuild_tail_ms += b.tail_ms;
        prebuild_other_ms += b.other_ms;
        if packed.build_id != 0 {
            prebuild_build_id_min = prebuild_build_id_min.min(packed.build_id);
            prebuild_build_id_max = prebuild_build_id_max.max(packed.build_id);
            unresolved_build_ids.push(packed.build_id);
        }
        if packed.build_submit_seq != 0 {
            prebuild_submit_seq_min = prebuild_submit_seq_min.min(packed.build_submit_seq);
            prebuild_submit_seq_max = prebuild_submit_seq_max.max(packed.build_submit_seq);
        }
        if let Some(submit_index) = packed.build_submit_index.clone() {
            let should_replace = latest_build_submit
                .as_ref()
                .map(|(seq, _, _)| packed.build_submit_seq > *seq)
                .unwrap_or(true);
            if should_replace {
                latest_build_submit =
                    Some((packed.build_submit_seq, packed.build_id, submit_index));
            }
        }
    }

    let r = runtime()?;
    let t_setup = Instant::now();
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
    profile.readback_setup_ms = elapsed_ms(t_setup);
    if resolve_wait_attribution_probe_enabled() {
        if let Some((latest_submit_seq, _, latest_submit_idx)) = latest_build_submit.as_ref() {
            let t_pre_wait = Instant::now();
            r.device
                .poll(wgpu::Maintain::wait_for(latest_submit_idx.clone()));
            profile.pre_wait_latest_build_submit_ms = elapsed_ms(t_pre_wait);
            profile.latest_build_submit_seq = *latest_submit_seq;
        }
    }
    let t_submit = Instant::now();
    let readback_submit_seq = next_gpu_submit_seq();
    let submission = r.queue.submit(Some(encoder.finish()));
    profile.readback_submit_seq = readback_submit_seq;
    profile.submit_ms = elapsed_ms(t_submit);

    let t_map_wait = Instant::now();
    let slice = readback.slice(..readback_bytes.max(8));
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let t_submit_done_wait = Instant::now();
    r.device.poll(wgpu::Maintain::wait_for(submission));
    profile.submit_done_wait_ms = elapsed_ms(t_submit_done_wait);
    let t_map_callback_wait = Instant::now();
    rx.recv()
        .map_err(|_| {
            PDeflateError::Gpu("gpu build-table batch size map channel closed".to_string())
        })?
        .map_err(|e| PDeflateError::Gpu(format!("gpu build-table batch size map failed: {e}")))?;
    profile.map_callback_wait_ms = elapsed_ms(t_map_callback_wait);
    profile.map_wait_ms = elapsed_ms(t_map_wait);
    if resolve_map_wait_probe_enabled() {
        let seq = RESOLVE_MAP_WAIT_PROBE_SEQ.fetch_add(1, Ordering::Relaxed);
        if table_stage_probe_should_log(seq) {
            let build_id_range = if prebuild_build_id_min != u64::MAX {
                format!("{}..{}", prebuild_build_id_min, prebuild_build_id_max)
            } else {
                "none".to_string()
            };
            let submit_seq_range = if prebuild_submit_seq_min != u64::MAX {
                format!("{}..{}", prebuild_submit_seq_min, prebuild_submit_seq_max)
            } else {
                "none".to_string()
            };
            let build_ids = if unresolved_build_ids.is_empty() {
                "none".to_string()
            } else {
                unresolved_build_ids
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let (latest_submit_seq, latest_build_id) = latest_build_submit
                .as_ref()
                .map(|(submit_seq, build_id, _)| (*submit_seq, *build_id))
                .unwrap_or((0, 0));
            eprintln!(
                "[cozip_pdeflate][timing][resolve-map-wait-breakdown] seq={} unresolved={} readback_kib={:.1} latest_build_submit_seq={} latest_build_id={} readback_submit_seq={} pre_wait_latest_build_submit_ms={:.3} submit_done_wait_ms={:.3} map_callback_wait_ms={:.3} map_wait_ms={:.3}",
                seq,
                unresolved.len(),
                (readback_bytes as f64) / 1024.0,
                latest_submit_seq,
                latest_build_id,
                profile.readback_submit_seq,
                profile.pre_wait_latest_build_submit_ms,
                profile.submit_done_wait_ms,
                profile.map_callback_wait_ms,
                profile.map_wait_ms,
            );
            let unresolved_f = unresolved.len() as f64;
            eprintln!(
                "[cozip_pdeflate][timing][resolve-preceding-gpu-breakdown] seq={} unresolved={} build_id_range={} submit_seq_range={} build_ids={} prebuild_total_ms={:.3} prebuild_pre_resource_ms={:.3} prebuild_resource_setup_ms={:.3} prebuild_upload_ms={:.3} prebuild_encode_pass1_ms={:.3} prebuild_submit_pass1_ms={:.3} prebuild_encode_pass2_ms={:.3} prebuild_submit_pass2_ms={:.3} prebuild_stage_probe_submit_ms={:.3} prebuild_stage_probe_wait_ms={:.3} prebuild_stage_probe_parse_ms={:.3} prebuild_stage_gpu_sum_ms={:.3} prebuild_tail_ms={:.3} prebuild_other_ms={:.3} prebuild_avg_ms={:.3}",
                seq,
                unresolved.len(),
                build_id_range,
                submit_seq_range,
                build_ids,
                prebuild_total_ms,
                prebuild_pre_resource_ms,
                prebuild_resource_setup_ms,
                prebuild_upload_ms,
                prebuild_encode_pass1_ms,
                prebuild_submit_pass1_ms,
                prebuild_encode_pass2_ms,
                prebuild_submit_pass2_ms,
                prebuild_stage_probe_submit_ms,
                prebuild_stage_probe_wait_ms,
                prebuild_stage_probe_parse_ms,
                prebuild_stage_gpu_sum_ms,
                prebuild_tail_ms,
                prebuild_other_ms,
                if unresolved_f > 0.0 {
                    prebuild_total_ms / unresolved_f
                } else {
                    0.0
                },
            );
            if resolve_wait_attribution_probe_enabled() {
                eprintln!(
                    "[cozip_pdeflate][timing][resolve-wait-attribution] seq={} unresolved={} pre_wait_latest_build_submit_ms={:.3} submit_done_wait_ms={:.3} map_wait_ms={:.3} note=\"pre_wait captures unresolved build completion; submit_done_wait is residual after readback submit\"",
                    seq,
                    unresolved.len(),
                    profile.pre_wait_latest_build_submit_ms,
                    profile.submit_done_wait_ms,
                    profile.map_wait_ms,
                );
            }
        }
    }
    let t_parse = Instant::now();
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
    profile.parse_ms = elapsed_ms(t_parse);
    profile.total_ms = elapsed_ms(t_total);
    Ok((out, profile))
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
    table_chunk_meta_words: Vec<u32>,
    table_index_words: Vec<u32>,
    table_data_words: Vec<u32>,
    table_index_words_len: usize,
    table_data_words_len: usize,
    table_device_copies: Vec<PackedTableDeviceCopy<'a>>,
    table_meta_device_copies: Vec<PackedTableMetaDeviceCopy<'a>>,
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

struct PackedTableMetaDeviceCopy<'a> {
    meta_src: &'a wgpu::Buffer,
    dst_count_offset: u64,
    dst_data_total_offset: u64,
    src_data_total_offset: u64,
}

struct PackedTableUpload<'a> {
    index_offset: u64,
    data_offset: u64,
    index_len: usize,
    data_len: usize,
    kind: PackedTableUploadKind<'a>,
}

enum PackedTableUploadKind<'a> {
    HostBorrow {
        index_bytes: &'a [u8],
        data_bytes: &'a [u8],
    },
    HostOwned {
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
) -> Result<(PackedBatchInputs<'a>, PackMatchBatchInputsProfile), PDeflateError> {
    let t_total = Instant::now();
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
    let t_alloc_setup = Instant::now();
    let total_src_bytes: usize = inputs.iter().map(|input| input.src.len()).sum();
    let mut src_words = vec![0u32; total_src_bytes.div_ceil(4)];
    let mut chunk_starts = Vec::<u32>::with_capacity(inputs.len() + 1);
    let mut table_chunk_bases = Vec::<u32>::with_capacity(inputs.len());
    let mut table_chunk_counts = Vec::<u32>::with_capacity(inputs.len());
    let mut table_chunk_index_offsets = Vec::<u32>::with_capacity(inputs.len());
    let mut table_chunk_data_offsets = Vec::<u32>::with_capacity(inputs.len());
    let mut table_chunk_meta_words = Vec::<u32>::with_capacity(inputs.len().saturating_mul(2));
    let mut table_uploads = Vec::<PackedTableUpload<'a>>::with_capacity(inputs.len());
    let mut table_meta_device_copies = Vec::<PackedTableMetaDeviceCopy<'a>>::new();
    let mut table_index_packer =
        ByteWordPacker::with_capacity_bytes(inputs.len().saturating_mul(256));
    let mut table_data_packer =
        ByteWordPacker::with_capacity_bytes(inputs.iter().map(|i| i.src.len() / 2).sum());
    let mut chunk_lens = Vec::<usize>::with_capacity(inputs.len());
    let mut global_table_base = 0usize;
    let alloc_setup_ms = elapsed_ms(t_alloc_setup);
    let resolve_profile = ResolvePackedTableSizesBatchProfile::default();
    let resolve_sizes_ms = 0.0_f64;
    chunk_starts.push(0);
    let mut src_len = 0usize;
    let mut src_copy_ms = 0.0_f64;
    let mut metadata_loop_ms = 0.0_f64;
    {
        let src_words_bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut src_words);
        for input in inputs {
            let t_src_copy = Instant::now();
            let next_src_len = src_len
                .checked_add(input.src.len())
                .ok_or(PDeflateError::NumericOverflow)?;
            src_words_bytes[src_len..next_src_len].copy_from_slice(input.src);
            src_len = next_src_len;
            chunk_starts.push(u32::try_from(src_len).map_err(|_| PDeflateError::NumericOverflow)?);
            src_copy_ms += elapsed_ms(t_src_copy);
            let t_meta = Instant::now();
            if input.max_ref_len != max_ref_len
                || input.min_ref_len != min_ref_len
                || input.section_count != section_count
            {
                return Err(PDeflateError::InvalidOptions(
                    "gpu match batch params must be consistent",
                ));
            }
            chunk_lens.push(input.src.len());
            table_chunk_bases.push(
                u32::try_from(global_table_base).map_err(|_| PDeflateError::NumericOverflow)?,
            );
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
            let meta_word_base = table_chunk_meta_words.len();
            table_chunk_meta_words.push(0);
            table_chunk_meta_words.push(0);
            let mut local_count = 0usize;
            let (index_len, data_len, kind) = if let Some(table_gpu) = input.table_gpu {
                let (count_hint, index_len_hint, data_len_hint, unresolved_gpu_sizes) = if table_gpu
                    .sizes_known
                    || table_gpu.table_count > 0
                    || table_gpu.table_index_len > 0
                    || table_gpu.table_data_len > 0
                {
                    (
                        table_gpu.table_count,
                        table_gpu.table_index_len,
                        table_gpu.table_data_len,
                        false,
                    )
                } else {
                    (
                        table_gpu.max_entries,
                        table_gpu.max_entries,
                        table_gpu.table_data_bytes_cap,
                        true,
                    )
                };
                local_count = count_hint;
                if !unresolved_gpu_sizes {
                    table_chunk_meta_words[meta_word_base] =
                        u32::try_from(count_hint).map_err(|_| PDeflateError::NumericOverflow)?;
                    table_chunk_meta_words[meta_word_base + 1] =
                        u32::try_from(data_len_hint).map_err(|_| PDeflateError::NumericOverflow)?;
                } else {
                    let dst_count_offset = u64::try_from(meta_word_base)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .saturating_mul(4);
                    let dst_data_total_offset = u64::try_from(meta_word_base + 1)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .saturating_mul(4);
                    let src_data_total_offset_words = 1usize
                        .checked_add(
                            table_gpu
                                .max_entries
                                .checked_mul(2)
                                .ok_or(PDeflateError::NumericOverflow)?,
                        )
                        .ok_or(PDeflateError::NumericOverflow)?;
                    let src_data_total_offset = u64::try_from(src_data_total_offset_words)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .saturating_mul(4);
                    table_meta_device_copies.push(PackedTableMetaDeviceCopy {
                        meta_src: &table_gpu.table_meta_buffer,
                        dst_count_offset,
                        dst_data_total_offset,
                        src_data_total_offset,
                    });
                }
                (
                    index_len_hint,
                    data_len_hint,
                    PackedTableUploadKind::Device {
                        index_buffer: &table_gpu.table_index_buffer,
                        data_buffer: &table_gpu.table_data_buffer,
                    },
                )
            } else if let (Some(index_bytes), Some(data_bytes)) =
                (input.table_index, input.table_data)
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
                table_chunk_meta_words[meta_word_base] =
                    u32::try_from(local_count).map_err(|_| PDeflateError::NumericOverflow)?;
                table_chunk_meta_words[meta_word_base + 1] =
                    u32::try_from(data_bytes.len()).map_err(|_| PDeflateError::NumericOverflow)?;
                (
                    index_bytes.len(),
                    data_bytes.len(),
                    PackedTableUploadKind::HostBorrow {
                        index_bytes,
                        data_bytes,
                    },
                )
            } else {
                let mut index_bytes = Vec::<u8>::new();
                let mut data_bytes = Vec::<u8>::new();
                index_bytes.reserve(input.table.len());
                let data_len: usize = input.table.iter().map(Vec::len).sum();
                data_bytes.reserve(data_len);
                for entry in input.table.iter() {
                    index_bytes.push(
                        u8::try_from(entry.len()).map_err(|_| PDeflateError::NumericOverflow)?,
                    );
                    data_bytes.extend_from_slice(entry);
                    local_count = local_count.saturating_add(1);
                }
                table_chunk_meta_words[meta_word_base] =
                    u32::try_from(local_count).map_err(|_| PDeflateError::NumericOverflow)?;
                table_chunk_meta_words[meta_word_base + 1] =
                    u32::try_from(data_bytes.len()).map_err(|_| PDeflateError::NumericOverflow)?;
                (
                    index_bytes.len(),
                    data_bytes.len(),
                    PackedTableUploadKind::HostOwned {
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
            metadata_loop_ms += elapsed_ms(t_meta);
        }
    }
    let mut host_copy_ms = 0.0_f64;
    let mut device_copy_plan_ms = 0.0_f64;
    let mut table_device_copies = Vec::<PackedTableDeviceCopy<'a>>::new();
    for upload in table_uploads.into_iter() {
        match upload.kind {
            PackedTableUploadKind::HostBorrow {
                index_bytes,
                data_bytes,
            } => {
                let t_host_copy = Instant::now();
                if upload.index_len != 0 {
                    let off = usize::try_from(upload.index_offset)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    table_index_packer.write_bytes_at(off, index_bytes)?;
                }
                if upload.data_len != 0 {
                    let off = usize::try_from(upload.data_offset)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    table_data_packer.write_bytes_at(off, data_bytes)?;
                }
                host_copy_ms += elapsed_ms(t_host_copy);
            }
            PackedTableUploadKind::HostOwned {
                index_bytes,
                data_bytes,
            } => {
                let t_host_copy = Instant::now();
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
                host_copy_ms += elapsed_ms(t_host_copy);
            }
            PackedTableUploadKind::Device {
                index_buffer,
                data_buffer,
            } => {
                let t_device_plan = Instant::now();
                table_device_copies.push(PackedTableDeviceCopy {
                    index_src: index_buffer,
                    data_src: data_buffer,
                    index_dst_offset: upload.index_offset,
                    data_dst_offset: upload.data_offset,
                    index_len: upload.index_len,
                    data_len: upload.data_len,
                });
                device_copy_plan_ms += elapsed_ms(t_device_plan);
            }
        }
    }

    let t_finalize = Instant::now();
    let table_index_words_len = table_index_packer.len_bytes().div_ceil(4);
    let table_data_words_len = table_data_packer.len_bytes().div_ceil(4);
    let table_index_words = table_index_packer.into_words();
    let table_data_words = table_data_packer.into_words();
    let finalize_ms = elapsed_ms(t_finalize);

    let packed = PackedBatchInputs {
        src_words,
        chunk_starts,
        table_chunk_bases,
        table_chunk_counts,
        table_chunk_index_offsets,
        table_chunk_data_offsets,
        table_chunk_meta_words,
        table_index_words_len,
        table_data_words_len,
        table_index_words,
        table_data_words,
        table_device_copies,
        table_meta_device_copies,
        chunk_lens,
        total_src_len: total_src_bytes,
        total_table_entries: global_table_base,
        max_ref_len,
        min_ref_len,
        section_count,
    };
    let profile = PackMatchBatchInputsProfile {
        total_ms: elapsed_ms(t_total),
        alloc_setup_ms,
        resolve_sizes_ms,
        resolve_scan_ms: resolve_profile.scan_ms,
        resolve_readback_setup_ms: resolve_profile.readback_setup_ms,
        resolve_submit_ms: resolve_profile.submit_ms,
        resolve_map_wait_ms: resolve_profile.map_wait_ms,
        resolve_parse_ms: resolve_profile.parse_ms,
        src_copy_ms,
        metadata_loop_ms,
        host_copy_ms,
        device_copy_plan_ms,
        finalize_ms,
    };
    Ok((packed, profile))
}

fn encode_table_device_copies(
    encoder: &mut wgpu::CommandEncoder,
    scratch: &GpuBatchScratch,
    copies: &[PackedTableDeviceCopy<'_>],
    meta_copies: &[PackedTableMetaDeviceCopy<'_>],
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
    for meta in meta_copies {
        encoder.copy_buffer_to_buffer(
            meta.meta_src,
            0,
            &scratch.table_meta_buffer,
            meta.dst_count_offset,
            4,
        );
        encoder.copy_buffer_to_buffer(
            meta.meta_src,
            meta.src_data_total_offset,
            &scratch.table_meta_buffer,
            meta.dst_data_total_offset,
            4,
        );
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
    let (packed, _pack_profile) = pack_match_batch_inputs(inputs)?;
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
        table_meta_bytes: u64::try_from(packed.table_chunk_meta_words.len())
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

    let scratch = acquire_batch_scratch(r, required_caps)?;

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
        if !packed.table_chunk_meta_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_meta_buffer,
                0,
                bytemuck::cast_slice(&packed.table_chunk_meta_words),
            );
        }
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
        encode_table_device_copies(
            &mut encoder,
            &scratch,
            &packed.table_device_copies,
            &packed.table_meta_device_copies,
        )?;
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

    release_batch_scratch(r, scratch);

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
    let (packed, pack_profile) = pack_match_batch_inputs(inputs)?;
    let pack_inputs_ms = pack_profile.total_ms;
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
        table_meta_bytes: u64::try_from(packed.table_chunk_meta_words.len())
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
    let mut scratch = acquire_batch_scratch(r, required_caps)?;
    let scratch_acquire_ms = elapsed_ms(t_scratch);

    struct SectionChunkState {
        slot_index: usize,
        section_count: usize,
        out_lens_bytes: u64,
        out_cmd_bytes: u64,
        section_index_cap_bytes: u64,
        max_tokens_per_section: u32,
        max_cmd_words_per_section: u32,
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
        if !packed.table_chunk_meta_words.is_empty() {
            r.queue.write_buffer(
                &scratch.table_meta_buffer,
                0,
                bytemuck::cast_slice(&packed.table_chunk_meta_words),
            );
        }
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
        let queue_probe = queue_probe_enabled();
        let mut pre_match_queue_drain_ms = 0.0_f64;
        let mut pre_match_queue_drain_poll_calls = 0u64;
        let mut match_submit_done_wait_ms = 0.0_f64;
        let mut match_submit_poll_calls = 0u64;
        let mut section_submit_done_wait_ms = 0.0_f64;
        let mut section_submit_poll_calls = 0u64;
        if queue_probe {
            let (wait_ms, poll_calls) = wait_for_queue_done(&r.queue, &r.device);
            pre_match_queue_drain_ms = wait_ms;
            pre_match_queue_drain_poll_calls = poll_calls;
        }

        let mut match_encoder = r
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-pdeflate-match-section-encoder"),
            });
        let t_table_copy = Instant::now();
        encode_table_device_copies(
            &mut match_encoder,
            &scratch,
            &packed.table_device_copies,
            &packed.table_meta_device_copies,
        )?;
        let match_table_copy_ms = elapsed_ms(t_table_copy);
        match_encoder.clear_buffer(&scratch.prefix_first_buffer, 0, None);
        match_encoder.clear_buffer(&scratch.table_lens_buffer, 0, None);
        match_encoder.clear_buffer(&scratch.table_offsets_buffer, 0, None);
        let match_prepare_dispatch_ms = {
            let t_prepare = Instant::now();
            let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
            let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
            let mut section_token_offsets = vec![0u32; section_count];
            let mut section_token_caps = vec![0u32; section_count];
            let mut total_cmd_cap_bytes: usize = 0;
            let mut total_token_cap: usize = 0;
            let mut max_tokens_per_section: usize = 1;
            let mut max_cmd_words_per_section: usize = 1;
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
                section_token_offsets[sec] =
                    u32::try_from(total_token_cap).map_err(|_| PDeflateError::NumericOverflow)?;
                let token_cap = sec_len.max(1);
                section_token_caps[sec] =
                    u32::try_from(token_cap).map_err(|_| PDeflateError::NumericOverflow)?;
                total_token_cap = total_token_cap
                    .checked_add(token_cap)
                    .ok_or(PDeflateError::NumericOverflow)?;
                max_tokens_per_section = max_tokens_per_section.max(token_cap);
                max_cmd_words_per_section =
                    max_cmd_words_per_section.max(align_up4(cap_bytes).div_ceil(4));
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
            if u64::try_from(total_token_cap).map_err(|_| PDeflateError::NumericOverflow)?
                > slot.token_cap_count
            {
                return Err(PDeflateError::Gpu(
                    "gpu section encode token staging cap exceeded".to_string(),
                ));
            }
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
                &slot.section_token_offsets_buffer,
                0,
                bytemuck::cast_slice(&section_token_offsets),
            );
            r.queue.write_buffer(
                &slot.section_token_caps_buffer,
                0,
                bytemuck::cast_slice(&section_token_caps),
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
                max_tokens_per_section: u32::try_from(max_tokens_per_section)
                    .map_err(|_| PDeflateError::NumericOverflow)?,
                max_cmd_words_per_section: u32::try_from(max_cmd_words_per_section)
                    .map_err(|_| PDeflateError::NumericOverflow)?,
            });
        }
        let section_upload_ms = elapsed_ms(t_section_setup);
        let mut section_pass_dispatch_ms = 0.0_f64;
        let mut section_tokenize_dispatch_ms = 0.0_f64;
        let mut section_prefix_dispatch_ms = 0.0_f64;
        let mut section_scatter_dispatch_ms = 0.0_f64;
        let mut section_pack_dispatch_ms = 0.0_f64;
        let mut section_meta_dispatch_ms = 0.0_f64;
        let section_copy_dispatch_ms = 0.0_f64;
        let mut section_tokenize_wait_ms = 0.0_f64;
        let mut section_prefix_wait_ms = 0.0_f64;
        let mut section_scatter_wait_ms = 0.0_f64;
        let mut section_pack_wait_ms = 0.0_f64;
        let mut section_meta_wait_ms = 0.0_f64;
        let mut section_stage_wait_poll_calls = 0u64;
        let (match_submit_ms, section_submit_ms) = if queue_probe {
            let t_match_submit = Instant::now();
            r.queue.submit(Some(match_encoder.finish()));
            let match_submit_ms = elapsed_ms(t_match_submit);
            let (wait_ms, poll_calls) = wait_for_queue_done(&r.queue, &r.device);
            match_submit_done_wait_ms = wait_ms;
            match_submit_poll_calls = poll_calls;

            let mut section_submit_ms = 0.0_f64;
            let mut submit_and_wait = |encoder: wgpu::CommandEncoder,
                                       stage_wait_ms: &mut f64|
             -> f64 {
                let t_submit = Instant::now();
                r.queue.submit(Some(encoder.finish()));
                let submit_ms = elapsed_ms(t_submit);
                let (wait_ms, poll_calls) = wait_for_queue_done(&r.queue, &r.device);
                *stage_wait_ms += wait_ms;
                section_submit_done_wait_ms += wait_ms;
                section_submit_poll_calls = section_submit_poll_calls.saturating_add(poll_calls);
                section_stage_wait_poll_calls =
                    section_stage_wait_poll_calls.saturating_add(poll_calls);
                submit_ms
            };

            {
                let mut encoder =
                    r.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-pdeflate-se-tokenize-encoder"),
                        });
                for state in section_states.iter() {
                    let slot = &scratch.section_slots[state.slot_index];
                    let groups_sections = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .div_ceil(64)
                        .max(1);
                    let t_stage = Instant::now();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-tokenize-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_tokenize_pipeline);
                    pass.set_bind_group(0, &slot.tokenize_bind_group, &[]);
                    pass.dispatch_workgroups(groups_sections, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_stage);
                    section_pass_dispatch_ms += dt;
                    section_tokenize_dispatch_ms += dt;
                }
                section_submit_ms += submit_and_wait(encoder, &mut section_tokenize_wait_ms);
            }
            {
                let mut encoder =
                    r.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-pdeflate-se-prefix-encoder"),
                        });
                for state in section_states.iter() {
                    let slot = &scratch.section_slots[state.slot_index];
                    let groups_sections = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?
                        .div_ceil(64)
                        .max(1);
                    let t_stage = Instant::now();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-prefix-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_prefix_pipeline);
                    pass.set_bind_group(0, &slot.prefix_bind_group, &[]);
                    pass.dispatch_workgroups(groups_sections, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_stage);
                    section_pass_dispatch_ms += dt;
                    section_prefix_dispatch_ms += dt;
                }
                section_submit_ms += submit_and_wait(encoder, &mut section_prefix_wait_ms);
            }
            {
                let mut encoder =
                    r.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-pdeflate-se-scatter-encoder"),
                        });
                for state in section_states.iter() {
                    let slot = &scratch.section_slots[state.slot_index];
                    let scatter_groups_x = state.max_tokens_per_section.div_ceil(64).max(1);
                    let scatter_groups_y = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    let t_stage = Instant::now();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-scatter-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_scatter_pipeline);
                    pass.set_bind_group(0, &slot.scatter_bind_group, &[]);
                    pass.dispatch_workgroups(scatter_groups_x, scatter_groups_y, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_stage);
                    section_pass_dispatch_ms += dt;
                    section_scatter_dispatch_ms += dt;
                }
                section_submit_ms += submit_and_wait(encoder, &mut section_scatter_wait_ms);
            }
            {
                let mut encoder =
                    r.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-pdeflate-se-pack-encoder"),
                        });
                for state in section_states.iter() {
                    let slot = &scratch.section_slots[state.slot_index];
                    let pack_groups_x = state.max_cmd_words_per_section.div_ceil(64).max(1);
                    let pack_groups_y = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    let t_stage = Instant::now();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-pack-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_cmd_pack_pipeline);
                    pass.set_bind_group(0, &slot.pack_bind_group, &[]);
                    pass.dispatch_workgroups(pack_groups_x, pack_groups_y, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_stage);
                    section_pass_dispatch_ms += dt;
                    section_pack_dispatch_ms += dt;
                }
                section_submit_ms += submit_and_wait(encoder, &mut section_pack_wait_ms);
            }
            {
                let mut encoder =
                    r.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("cozip-pdeflate-se-meta-encoder"),
                        });
                for state in section_states.iter() {
                    let slot = &scratch.section_slots[state.slot_index];
                    let t_stage = Instant::now();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-meta-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_meta_pipeline);
                    pass.set_bind_group(0, &slot.meta_bind_group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_stage);
                    section_pass_dispatch_ms += dt;
                    section_meta_dispatch_ms += dt;
                    encoder.copy_buffer_to_buffer(
                        &slot.section_meta_buffer,
                        0,
                        &slot.readback_meta_buffer,
                        0,
                        16,
                    );
                    encoder.copy_buffer_to_buffer(
                        &slot.out_lens_buffer,
                        0,
                        &slot.out_lens_readback_buffer,
                        0,
                        state.out_lens_bytes.max(4),
                    );
                }
                section_submit_ms += submit_and_wait(encoder, &mut section_meta_wait_ms);
            }

            (match_submit_ms, section_submit_ms)
        } else {
            for state in section_states.iter() {
                let slot = &scratch.section_slots[state.slot_index];
                let groups_sections = u32::try_from(state.section_count)
                    .map_err(|_| PDeflateError::NumericOverflow)?
                    .div_ceil(64)
                    .max(1);
                {
                    let t_pass = Instant::now();
                    let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-tokenize-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_tokenize_pipeline);
                    pass.set_bind_group(0, &slot.tokenize_bind_group, &[]);
                    pass.dispatch_workgroups(groups_sections, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_pass);
                    section_pass_dispatch_ms += dt;
                    section_tokenize_dispatch_ms += dt;
                }
                {
                    let t_prefix = Instant::now();
                    let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-prefix-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_prefix_pipeline);
                    pass.set_bind_group(0, &slot.prefix_bind_group, &[]);
                    pass.dispatch_workgroups(groups_sections, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_prefix);
                    section_pass_dispatch_ms += dt;
                    section_prefix_dispatch_ms += dt;
                }
                {
                    let t_scatter = Instant::now();
                    let scatter_groups_x = state.max_tokens_per_section.div_ceil(64).max(1);
                    let scatter_groups_y = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-scatter-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_scatter_pipeline);
                    pass.set_bind_group(0, &slot.scatter_bind_group, &[]);
                    pass.dispatch_workgroups(scatter_groups_x, scatter_groups_y, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_scatter);
                    section_pass_dispatch_ms += dt;
                    section_scatter_dispatch_ms += dt;
                }
                {
                    let t_pack = Instant::now();
                    let pack_groups_x = state.max_cmd_words_per_section.div_ceil(64).max(1);
                    let pack_groups_y = u32::try_from(state.section_count)
                        .map_err(|_| PDeflateError::NumericOverflow)?;
                    let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-pack-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_cmd_pack_pipeline);
                    pass.set_bind_group(0, &slot.pack_bind_group, &[]);
                    pass.dispatch_workgroups(pack_groups_x, pack_groups_y, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_pack);
                    section_pass_dispatch_ms += dt;
                    section_pack_dispatch_ms += dt;
                }
                {
                    let t_meta = Instant::now();
                    let mut pass = match_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("cozip-pdeflate-se-meta-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&r.section_meta_pipeline);
                    pass.set_bind_group(0, &slot.meta_bind_group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                    drop(pass);
                    let dt = elapsed_ms(t_meta);
                    section_pass_dispatch_ms += dt;
                    section_meta_dispatch_ms += dt;
                }
                match_encoder.copy_buffer_to_buffer(
                    &slot.section_meta_buffer,
                    0,
                    &slot.readback_meta_buffer,
                    0,
                    16,
                );
                match_encoder.copy_buffer_to_buffer(
                    &slot.out_lens_buffer,
                    0,
                    &slot.out_lens_readback_buffer,
                    0,
                    state.out_lens_bytes.max(4),
                );
            }
            let t_submit = Instant::now();
            r.queue.submit(Some(match_encoder.finish()));
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
            (match_submit_ms, section_submit_ms)
        };
        if queue_probe {
            eprintln!(
                "[cozip_pdeflate][timing][gpu-stage-probe] chunks={} pre_match_queue_drain_ms={:.3} pre_match_queue_drain_polls={} match_submit_ms={:.3} match_submit_done_wait_ms={:.3} match_submit_polls={} section_submit_ms={:.3} section_submit_done_wait_ms={:.3} section_submit_polls={} match_dispatch_ms={:.3} section_dispatch_ms={:.3}",
                inputs.len(),
                pre_match_queue_drain_ms,
                pre_match_queue_drain_poll_calls,
                match_submit_ms,
                match_submit_done_wait_ms,
                match_submit_poll_calls,
                section_submit_ms,
                section_submit_done_wait_ms,
                section_submit_poll_calls,
                match_table_copy_ms + match_prepare_dispatch_ms + match_kernel_dispatch_ms,
                section_pass_dispatch_ms + section_copy_dispatch_ms
            );
            eprintln!(
                "[cozip_pdeflate][timing][gpu-section-stage-probe] chunks={} tokenize_dispatch_ms={:.3} prefix_dispatch_ms={:.3} scatter_dispatch_ms={:.3} pack_dispatch_ms={:.3} meta_dispatch_ms={:.3} tokenize_wait_ms={:.3} prefix_wait_ms={:.3} scatter_wait_ms={:.3} pack_wait_ms={:.3} meta_wait_ms={:.3} section_wait_polls={}",
                inputs.len(),
                section_tokenize_dispatch_ms,
                section_prefix_dispatch_ms,
                section_scatter_dispatch_ms,
                section_pack_dispatch_ms,
                section_meta_dispatch_ms,
                section_tokenize_wait_ms,
                section_prefix_wait_ms,
                section_scatter_wait_ms,
                section_pack_wait_ms,
                section_meta_wait_ms,
                section_stage_wait_poll_calls
            );
        }

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
                    out_lens_readback_buffer: slot.out_lens_readback_buffer.clone(),
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
                pack_alloc_setup_ms: pack_profile.alloc_setup_ms,
                pack_resolve_sizes_ms: pack_profile.resolve_sizes_ms,
                pack_resolve_scan_ms: pack_profile.resolve_scan_ms,
                pack_resolve_readback_setup_ms: pack_profile.resolve_readback_setup_ms,
                pack_resolve_submit_ms: pack_profile.resolve_submit_ms,
                pack_resolve_map_wait_ms: pack_profile.resolve_map_wait_ms,
                pack_resolve_parse_ms: pack_profile.resolve_parse_ms,
                pack_src_copy_ms: pack_profile.src_copy_ms,
                pack_metadata_loop_ms: pack_profile.metadata_loop_ms,
                pack_host_copy_ms: pack_profile.host_copy_ms,
                pack_device_copy_plan_ms: pack_profile.device_copy_plan_ms,
                pack_finalize_ms: pack_profile.finalize_ms,
                scratch_acquire_ms,
                match_table_copy_ms,
                match_prepare_dispatch_ms,
                match_kernel_dispatch_ms,
                match_submit_ms,
                section_setup_ms: section_upload_ms,
                section_pass_dispatch_ms,
                section_tokenize_dispatch_ms,
                section_prefix_dispatch_ms,
                section_scatter_dispatch_ms,
                section_pack_dispatch_ms,
                section_meta_dispatch_ms,
                section_copy_dispatch_ms,
                section_submit_ms,
                section_tokenize_wait_ms,
                section_prefix_wait_ms,
                section_scatter_wait_ms,
                section_pack_wait_ms,
                section_meta_wait_ms,
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
            release_batch_scratch(r, scratch);
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
    if section_count == 0 || len == 0 {
        return 0;
    }
    if sec == 0 {
        return 0;
    }
    if sec >= section_count {
        return len;
    }
    let raw = sec.saturating_mul(len) / section_count;
    raw & !3usize
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
    if sec == 0 {
        return 0;
    }
    if sec >= section_count {
        return len;
    }
    let raw = ((sec as u128) * (len as u128) / (section_count as u128)) as usize;
    raw & !3usize
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
