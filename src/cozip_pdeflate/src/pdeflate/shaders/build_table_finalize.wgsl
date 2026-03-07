@group(0) @binding(0) var<storage, read> src_words: array<u32>;
@group(0) @binding(1) var<storage, read> freq_words: array<u32>;
@group(0) @binding(2) var<storage, read> cand_words: array<u32>;
@group(0) @binding(3) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_meta: array<u32>;
@group(0) @binding(6) var<storage, read_write> emit_desc_words: array<u32>;

const FINALIZE_SHARDS: u32 = 128u;
const FINALIZE_LITERAL_FLAG: u32 = 0x80000000u;
const FINALIZE_EMIT_DESC_WORDS: u32 = 3u;

var<workgroup> shard_pos_0: array<u32, 128>;
var<workgroup> shard_len_0: array<u32, 128>;
var<workgroup> shard_score_0: array<u32, 128>;
var<workgroup> shard_pos_1: array<u32, 128>;
var<workgroup> shard_len_1: array<u32, 128>;
var<workgroup> shard_score_1: array<u32, 128>;
var<workgroup> shard_pos_2: array<u32, 128>;
var<workgroup> shard_len_2: array<u32, 128>;
var<workgroup> shard_score_2: array<u32, 128>;
var<workgroup> taken_literals_wg: array<u32, 256>;
var<workgroup> out_count_wg: u32;
var<workgroup> data_cursor_wg: u32;
var<workgroup> rank_pos_wg: array<u32, 128>;
var<workgroup> rank_len_wg: array<u32, 128>;
var<workgroup> rank_score_wg: array<u32, 128>;
var<workgroup> rank_sig0_wg: array<u32, 128>;
var<workgroup> rank_sig1_wg: array<u32, 128>;
var<workgroup> rank_active_wg: array<u32, 128>;
var<workgroup> reduce_pos_wg: array<u32, 128>;
var<workgroup> reduce_len_wg: array<u32, 128>;
var<workgroup> reduce_score_wg: array<u32, 128>;
var<workgroup> reduce_idx_wg: array<u32, 128>;
var<workgroup> reduce_found_wg: array<u32, 128>;
var<workgroup> reduce_byte_wg: array<u32, 128>;
var<workgroup> reduce_freq_wg: array<u32, 128>;
var<workgroup> literal_stop_wg: u32;
var<workgroup> rank_stop_wg: u32;

fn load_src(idx: u32) -> u32 {
    let w = src_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
}

fn load_src_packed(pos: u32, len: u32) -> u32 {
    var out = 0u;
    var i = 0u;
    loop {
        if (i >= 4u || i >= len) {
            break;
        }
        out = out | (load_src(pos + i) << (i * 8u));
        i = i + 1u;
    }
    return out;
}

fn load_src_prefix_sig(pos: u32, len: u32) -> u32 {
    return load_src_packed(pos, min(len, 4u));
}

fn load_src_suffix_sig(pos: u32, len: u32) -> u32 {
    if (len <= 4u) {
        return load_src_packed(pos, len);
    }
    return load_src_packed(pos + len - 4u, 4u);
}

fn emit_desc_base(entry_idx: u32) -> u32 {
    return entry_idx * FINALIZE_EMIT_DESC_WORDS;
}

fn emit_desc_is_literal(desc: u32) -> bool {
    return (desc & FINALIZE_LITERAL_FLAG) != 0u;
}

fn emit_desc_literal_byte(desc: u32) -> u32 {
    return desc & 0xffu;
}

fn entry_matches_src(
    pos: u32,
    len: u32,
    sig0: u32,
    sig1: u32,
    prev_desc: u32,
    prev_len: u32,
    prev_sig0: u32,
    prev_sig1: u32,
) -> bool {
    if (prev_len != len) {
        return false;
    }
    if (emit_desc_is_literal(prev_desc)) {
        return len == 1u && emit_desc_literal_byte(prev_desc) == (sig0 & 0xffu);
    }
    if (prev_sig0 != sig0 || prev_sig1 != sig1) {
        return false;
    }

    let prev_pos = prev_desc;
    let word_count = len >> 2u;
    var word_idx = 0u;
    loop {
        if (word_idx >= word_count) {
            break;
        }
        let byte_off = word_idx << 2u;
        if (load_src_packed(pos + byte_off, 4u) != load_src_packed(prev_pos + byte_off, 4u)) {
            return false;
        }
        word_idx = word_idx + 1u;
    }
    let rem = len & 3u;
    if (rem != 0u) {
        let byte_off = word_count << 2u;
        if (load_src_packed(pos + byte_off, rem) != load_src_packed(prev_pos + byte_off, rem)) {
            return false;
        }
    }
    return true;
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

fn better_literal(
    freq: u32,
    byte: u32,
    best_freq: u32,
    best_byte: u32,
) -> bool {
    if (freq > best_freq) {
        return true;
    }
    if (freq < best_freq) {
        return false;
    }
    return byte < best_byte;
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

    if (gid3.x != 0u || gid3.y != 0u || gid3.z != 0u) {
        return;
    }

    var b_init = lane;
    loop {
        if (b_init >= 256u) {
            break;
        }
        taken_literals_wg[b_init] = 0u;
        b_init = b_init + 128u;
    }
    if (lane == 0u) {
        out_count_wg = 0u;
        data_cursor_wg = 0u;
    }
    workgroupBarrier();

    var lit_rank = 0u;
    loop {
        if (lit_rank >= literal_limit || out_count_wg >= max_entries) {
            break;
        }
        var best_byte = 0u;
        var best_freq = 0u;
        var has_best = 0u;
        var b = lane;
        loop {
            if (b >= 256u) {
                break;
            }
            if (taken_literals_wg[b] == 0u) {
                let f = freq_words[b];
                if (f >= min_literal_freq) {
                    if (has_best == 0u || better_literal(f, b, best_freq, best_byte)) {
                        has_best = 1u;
                        best_byte = b;
                        best_freq = f;
                    }
                }
            }
            b = b + 128u;
        }
        reduce_found_wg[lane] = has_best;
        reduce_byte_wg[lane] = best_byte;
        reduce_freq_wg[lane] = best_freq;
        workgroupBarrier();

        var lit_step = 64u;
        loop {
            if (lit_step == 0u) {
                break;
            }
            if (lane < lit_step) {
                let rhs_found = reduce_found_wg[lane + lit_step];
                if (rhs_found != 0u) {
                    let lhs_found = reduce_found_wg[lane];
                    let rhs_byte = reduce_byte_wg[lane + lit_step];
                    let rhs_freq = reduce_freq_wg[lane + lit_step];
                    if (
                        lhs_found == 0u ||
                        better_literal(
                            rhs_freq,
                            rhs_byte,
                            reduce_freq_wg[lane],
                            reduce_byte_wg[lane],
                        )
                    ) {
                        reduce_found_wg[lane] = rhs_found;
                        reduce_byte_wg[lane] = rhs_byte;
                        reduce_freq_wg[lane] = rhs_freq;
                    }
                }
            }
            workgroupBarrier();
            lit_step = lit_step >> 1u;
        }

        if (lane == 0u) {
            if (reduce_found_wg[0] != 0u && data_cursor_wg + 1u <= max_data_bytes) {
                let chosen_byte = reduce_byte_wg[0];
                out_meta[1u + out_count_wg * 2u] = data_cursor_wg;
                out_meta[2u + out_count_wg * 2u] = 1u;
                let desc_base = emit_desc_base(out_count_wg);
                emit_desc_words[desc_base + 0u] = FINALIZE_LITERAL_FLAG | chosen_byte;
                emit_desc_words[desc_base + 1u] = chosen_byte;
                emit_desc_words[desc_base + 2u] = chosen_byte;
                data_cursor_wg = data_cursor_wg + 1u;
                out_count_wg = out_count_wg + 1u;
                taken_literals_wg[chosen_byte] = 1u;
                literal_stop_wg = 0u;
            } else {
                literal_stop_wg = 1u;
            }
        }
        workgroupBarrier();
        if (literal_stop_wg != 0u) {
            break;
        }
        lit_rank = lit_rank + 1u;
    }

    var rank = 0u;
    loop {
        if (rank >= 3u || out_count_wg >= max_entries) {
            break;
        }
        var score = 0u;
        var pos = 0u;
        var len = 0u;
        if (rank == 0u) {
            score = shard_score_0[lane];
            pos = shard_pos_0[lane];
            len = shard_len_0[lane];
        } else if (rank == 1u) {
            score = shard_score_1[lane];
            pos = shard_pos_1[lane];
            len = shard_len_1[lane];
        } else {
            score = shard_score_2[lane];
            pos = shard_pos_2[lane];
            len = shard_len_2[lane];
        }
        if (score != 0u && len != 0u && len <= max_entry_len && pos + len <= total_len) {
            rank_pos_wg[lane] = pos;
            rank_len_wg[lane] = len;
            rank_score_wg[lane] = score;
            rank_sig0_wg[lane] = load_src_prefix_sig(pos, len);
            rank_sig1_wg[lane] = load_src_suffix_sig(pos, len);
            rank_active_wg[lane] = 1u;
        } else {
            rank_pos_wg[lane] = 0u;
            rank_len_wg[lane] = 0u;
            rank_score_wg[lane] = 0u;
            rank_sig0_wg[lane] = 0u;
            rank_sig1_wg[lane] = 0u;
            rank_active_wg[lane] = 0u;
        }
        workgroupBarrier();

        loop {
            if (out_count_wg >= max_entries) {
                break;
            }
            var keep = rank_active_wg[lane];
            if (keep != 0u) {
                let candidate_len = rank_len_wg[lane];
                if (data_cursor_wg + candidate_len > max_data_bytes) {
                    keep = 0u;
                }
            }
            if (keep != 0u) {
                var e = 0u;
                loop {
                    if (e >= out_count_wg) {
                        break;
                    }
                    let e_len = out_meta[2u + e * 2u];
                    let desc_base = emit_desc_base(e);
                    let prev_desc = emit_desc_words[desc_base + 0u];
                    let prev_sig0 = emit_desc_words[desc_base + 1u];
                    let prev_sig1 = emit_desc_words[desc_base + 2u];
                    if (entry_matches_src(
                        rank_pos_wg[lane],
                        rank_len_wg[lane],
                        rank_sig0_wg[lane],
                        rank_sig1_wg[lane],
                        prev_desc,
                        e_len,
                        prev_sig0,
                        prev_sig1,
                    )) {
                        keep = 0u;
                        break;
                    }
                    e = e + 1u;
                }
            }
            if (keep != 0u) {
                reduce_score_wg[lane] = rank_score_wg[lane];
                reduce_len_wg[lane] = rank_len_wg[lane];
                reduce_pos_wg[lane] = rank_pos_wg[lane];
                reduce_idx_wg[lane] = lane;
            } else {
                reduce_score_wg[lane] = 0u;
                reduce_len_wg[lane] = 0u;
                reduce_pos_wg[lane] = 0u;
                reduce_idx_wg[lane] = 0xffffffffu;
            }
            workgroupBarrier();

            var step = 64u;
            loop {
                if (step == 0u) {
                    break;
                }
                if (lane < step) {
                    let rhs_score = reduce_score_wg[lane + step];
                    if (
                        rhs_score != 0u &&
                        (reduce_score_wg[lane] == 0u ||
                            better_candidate(
                                rhs_score,
                                reduce_len_wg[lane + step],
                                reduce_pos_wg[lane + step],
                                reduce_score_wg[lane],
                                reduce_len_wg[lane],
                                reduce_pos_wg[lane],
                            ))
                    ) {
                        reduce_score_wg[lane] = rhs_score;
                        reduce_len_wg[lane] = reduce_len_wg[lane + step];
                        reduce_pos_wg[lane] = reduce_pos_wg[lane + step];
                        reduce_idx_wg[lane] = reduce_idx_wg[lane + step];
                    }
                }
                workgroupBarrier();
                step = step >> 1u;
            }

            if (lane == 0u) {
                let chosen_score = reduce_score_wg[0];
                let chosen_idx = reduce_idx_wg[0];
                if (chosen_score == 0u || chosen_idx == 0xffffffffu) {
                    rank_stop_wg = 1u;
                } else {
                    let chosen_len = rank_len_wg[chosen_idx];
                    out_meta[1u + out_count_wg * 2u] = data_cursor_wg;
                    out_meta[2u + out_count_wg * 2u] = chosen_len;
                    let desc_base = emit_desc_base(out_count_wg);
                    emit_desc_words[desc_base + 0u] = rank_pos_wg[chosen_idx];
                    emit_desc_words[desc_base + 1u] = rank_sig0_wg[chosen_idx];
                    emit_desc_words[desc_base + 2u] = rank_sig1_wg[chosen_idx];
                    data_cursor_wg = data_cursor_wg + chosen_len;
                    out_count_wg = out_count_wg + 1u;
                    rank_active_wg[chosen_idx] = 0u;
                    rank_stop_wg = 0u;
                }
            }
            workgroupBarrier();
            if (rank_stop_wg != 0u) {
                break;
            }
        }
        rank = rank + 1u;
    }

    if (lane == 0u) {
        out_meta[0] = out_count_wg;
        out_meta[1u + max_entries * 2u] = data_cursor_wg;
    }
}
