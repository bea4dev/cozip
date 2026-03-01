use std::sync::{
    Mutex, OnceLock,
    mpsc::{self, TryRecvError},
};
use std::time::Instant;

use super::{
    BitWriter, GDEFLATE_NUM_STREAMS, GDEFLATE_TILE_SIZE, GDeflateError, serialize_lanes_words,
};

const STORED_LANE_PAYLOAD_STRIDE: usize = (GDEFLATE_TILE_SIZE / GDEFLATE_NUM_STREAMS) + 1;
const STORED_OUTPUT_TILE_BYTES: usize = GDEFLATE_NUM_STREAMS * STORED_LANE_PAYLOAD_STRIDE;

// Split a 64KiB tile into smaller static blocks to reduce long per-invocation
// sequential work in GPU static encode.
const STATIC_SUBBLOCK_SIZE: usize = 2048;
const STATIC_SUBBLOCKS_PER_TILE: usize = GDEFLATE_TILE_SIZE / STATIC_SUBBLOCK_SIZE; // 32
// Per-subblock per-lane scratch budget for emitted byte stream.
const STATIC_GPU_BLOCK_MAX_LANE_BYTES: usize = 192;
const STATIC_GPU_BLOCK_MAX_LANE_WORDS: usize = STATIC_GPU_BLOCK_MAX_LANE_BYTES.div_ceil(4);
// Intermediates are packed as 4 bytes per u32 word.
const STATIC_BLOCK_OUTPUT_TILE_WORDS: usize =
    GDEFLATE_NUM_STREAMS * STATIC_GPU_BLOCK_MAX_LANE_WORDS;
const STATIC_OUTPUT_TILE_WORDS: usize = STATIC_SUBBLOCKS_PER_TILE * STATIC_BLOCK_OUTPUT_TILE_WORDS;
const STATIC_BITS_TILE_WORDS: usize = STATIC_SUBBLOCKS_PER_TILE * GDEFLATE_NUM_STREAMS;
const STATIC_AUX_WORDS_PER_TILE: usize = GDEFLATE_NUM_STREAMS + 1;
const STATIC_INTERM_TILE_WORDS: usize =
    STATIC_OUTPUT_TILE_WORDS + STATIC_BITS_TILE_WORDS + STATIC_AUX_WORDS_PER_TILE;
const STATIC_GPU_MAX_LANE_BYTES: usize =
    STATIC_SUBBLOCKS_PER_TILE * STATIC_GPU_BLOCK_MAX_LANE_BYTES;
const STATIC_SERIALIZED_MAX_PAGE_BYTES: usize =
    (GDEFLATE_NUM_STREAMS * STATIC_GPU_MAX_LANE_BYTES.div_ceil(4) * 4) + (GDEFLATE_NUM_STREAMS * 4);
const STATIC_SERIALIZED_MAX_PAGE_WORDS: usize = STATIC_SERIALIZED_MAX_PAGE_BYTES / 4;
const STATIC_LZ_HASH_SIZE: usize = 1 << 15;
const STATIC_LZ_PREV_WORDS: usize = GDEFLATE_TILE_SIZE;
const STATIC_LZ_BEST_LEN_WORDS: usize = GDEFLATE_TILE_SIZE;
const STATIC_LZ_BEST_DIST_WORDS: usize = GDEFLATE_TILE_SIZE;
const STATIC_LZ_SCRATCH_WORDS_PER_TILE: usize = STATIC_LZ_HASH_SIZE
    + STATIC_LZ_PREV_WORDS
    + STATIC_LZ_BEST_LEN_WORDS
    + STATIC_LZ_BEST_DIST_WORDS;

const WORKGROUP_SIZE: u32 = 256;
const GPU_DECODE_TILE_WORDS: usize = GDEFLATE_TILE_SIZE;

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

const STORED_SCATTER_SHADER: &str = r#"
const TILE_SIZE: u32 = 65536u;
const NUM_STREAMS: u32 = 32u;
const LANE_PAYLOAD_STRIDE: u32 = 2049u;
const OUTPUT_TILE_BYTES: u32 = NUM_STREAMS * LANE_PAYLOAD_STRIDE;

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;
@group(0) @binding(1)
var<storage, read> tile_lens: array<u32>;
@group(0) @binding(2)
var<storage, read_write> output_words: array<u32>;
@group(0) @binding(3)
var<storage, read_write> _scratch_words_unused: array<u32>;

fn read_input_byte(byte_index: u32) -> u32 {
    let word = input_words[byte_index / 4u];
    let shift = (byte_index % 4u) * 8u;
    return (word >> shift) & 0xffu;
}

@compute @workgroup_size(256, 1, 1)
fn scatter_stored(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_word_count = arrayLength(&output_words);
    if (gid.x >= out_word_count) {
        return;
    }
    let tile_count = arrayLength(&tile_lens);
    let total_out_bytes = tile_count * OUTPUT_TILE_BYTES;
    let out_byte_base = gid.x * 4u;

    var out_word: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let out_byte_index = out_byte_base + k;
        var out_b: u32 = 0u;
        if (out_byte_index < total_out_bytes) {
            let tile = out_byte_index / OUTPUT_TILE_BYTES;
            let tile_local = out_byte_index % OUTPUT_TILE_BYTES;
            let lane = tile_local / LANE_PAYLOAD_STRIDE;
            let slot = tile_local % LANE_PAYLOAD_STRIDE;
            if (lane < NUM_STREAMS) {
                let tile_len = tile_lens[tile];
                var src_valid = false;
                var src_in_tile: u32 = 0u;
                if (tile_len == TILE_SIZE) {
                    if (lane == 0u && slot == (LANE_PAYLOAD_STRIDE - 1u)) {
                        src_valid = true;
                        src_in_tile = TILE_SIZE - 1u;
                    } else {
                        let idx = slot * NUM_STREAMS + lane;
                        if (idx < (TILE_SIZE - 1u)) {
                            src_valid = true;
                            src_in_tile = idx;
                        }
                    }
                } else {
                    let idx = slot * NUM_STREAMS + lane;
                    if (idx < tile_len) {
                        src_valid = true;
                        src_in_tile = idx;
                    }
                }
                if (src_valid) {
                    out_b = read_input_byte(tile * TILE_SIZE + src_in_tile);
                }
            }
        }
        out_word = out_word | ((out_b & 0xffu) << (k * 8u));
    }
    output_words[gid.x] = out_word;
}
"#;

const STATIC_HASH_RESET_SHADER: &str = r#"
const HASH_BITS: u32 = 15u;
const HASH_SIZE: u32 = 1u << HASH_BITS;
const NUM_STREAMS: u32 = 32u;
const SUBBLOCKS_PER_TILE: u32 = 32u;
const BLOCK_MAX_LANE_BYTES: u32 = 192u;
const BLOCK_MAX_LANE_WORDS: u32 = BLOCK_MAX_LANE_BYTES / 4u;
const BLOCK_OUTPUT_WORDS: u32 = NUM_STREAMS * BLOCK_MAX_LANE_WORDS;
const OUTPUT_TILE_WORDS: u32 = SUBBLOCKS_PER_TILE * BLOCK_OUTPUT_WORDS;
const TILE_SIZE: u32 = 65536u;
const SCRATCH_WORDS_PER_TILE: u32 = HASH_SIZE + TILE_SIZE + TILE_SIZE + TILE_SIZE;
const INVALID_POS: u32 = 0xffffffffu;

@group(0) @binding(0)
var<storage, read> _input_words: array<u32>;
@group(0) @binding(1)
var<storage, read> tile_lens: array<u32>;
@group(0) @binding(2)
var<storage, read_write> _interm_words_unused: array<u32>;
@group(0) @binding(3)
var<storage, read_write> scratch_words: array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1)
fn reset_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_count = arrayLength(&tile_lens);
    let total = tile_count * HASH_SIZE;
    if (gid.x >= total) {
        return;
    }
    let tile = gid.x / HASH_SIZE;
    let slot = gid.x % HASH_SIZE;
    let scratch_base = tile * SCRATCH_WORDS_PER_TILE;
    let head_base = scratch_base;
    atomicStore(&scratch_words[head_base + slot], INVALID_POS);
}
"#;

const STATIC_HASH_BUILD_SHADER: &str = r#"
const TILE_SIZE: u32 = 65536u;
const HASH_BITS: u32 = 15u;
const HASH_SIZE: u32 = 1u << HASH_BITS;
const NUM_STREAMS: u32 = 32u;
const SUBBLOCKS_PER_TILE: u32 = 32u;
const BLOCK_MAX_LANE_BYTES: u32 = 192u;
const BLOCK_MAX_LANE_WORDS: u32 = BLOCK_MAX_LANE_BYTES / 4u;
const BLOCK_OUTPUT_WORDS: u32 = NUM_STREAMS * BLOCK_MAX_LANE_WORDS;
const OUTPUT_TILE_WORDS: u32 = SUBBLOCKS_PER_TILE * BLOCK_OUTPUT_WORDS;
const SCRATCH_WORDS_PER_TILE: u32 = HASH_SIZE + TILE_SIZE + TILE_SIZE + TILE_SIZE;
const INVALID_POS: u32 = 0xffffffffu;

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;
@group(0) @binding(1)
var<storage, read> tile_lens: array<u32>;
@group(0) @binding(2)
var<storage, read_write> _interm_words_unused: array<u32>;
@group(0) @binding(3)
var<storage, read_write> scratch_words: array<atomic<u32>>;

fn read_input_byte(byte_index: u32) -> u32 {
    let word = input_words[byte_index / 4u];
    let shift = (byte_index % 4u) * 8u;
    return (word >> shift) & 0xffu;
}

fn hash3_at(tile_base: u32, pos: u32) -> u32 {
    let b0 = read_input_byte(tile_base + pos);
    let b1 = read_input_byte(tile_base + pos + 1u);
    let b2 = read_input_byte(tile_base + pos + 2u);
    let v = b0 | (b1 << 8u) | (b2 << 16u);
    return ((v * 2654435761u) >> (32u - HASH_BITS)) & (HASH_SIZE - 1u);
}

@compute @workgroup_size(256, 1, 1)
fn build_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_count = arrayLength(&tile_lens);
    let total_pos = tile_count * TILE_SIZE;
    if (gid.x >= total_pos) {
        return;
    }
    let tile = gid.x / TILE_SIZE;
    let pos = gid.x % TILE_SIZE;
    let tile_len = tile_lens[tile];
    if (pos + 2u >= tile_len) {
        return;
    }

    let scratch_base = tile * SCRATCH_WORDS_PER_TILE;
    let head_base = scratch_base;
    let prev_base = scratch_base + HASH_SIZE;
    let tile_base = tile * TILE_SIZE;
    let h = hash3_at(tile_base, pos);
    let old = atomicExchange(&scratch_words[head_base + h], pos);
    atomicStore(&scratch_words[prev_base + pos], old);
}
"#;

const STATIC_MATCH_SHADER: &str = r#"
const TILE_SIZE: u32 = 65536u;
const HASH_BITS: u32 = 15u;
const HASH_SIZE: u32 = 1u << HASH_BITS;
const NUM_STREAMS: u32 = 32u;
const SUBBLOCKS_PER_TILE: u32 = 32u;
const BLOCK_MAX_LANE_BYTES: u32 = 192u;
const BLOCK_MAX_LANE_WORDS: u32 = BLOCK_MAX_LANE_BYTES / 4u;
const BLOCK_OUTPUT_WORDS: u32 = NUM_STREAMS * BLOCK_MAX_LANE_WORDS;
const OUTPUT_TILE_WORDS: u32 = SUBBLOCKS_PER_TILE * BLOCK_OUTPUT_WORDS;
const SCRATCH_WORDS_PER_TILE: u32 = HASH_SIZE + TILE_SIZE + TILE_SIZE + TILE_SIZE;
const WINDOW_SIZE: u32 = 32768u;
const MIN_MATCH: u32 = 3u;
const MAX_MATCH: u32 = 258u;
// Speed-first tuning for GPU static path: shallower chain search.
const MAX_CHAIN: u32 = 16u;
const FAST_WINDOW_SIZE: u32 = 8192u;
const EARLY_ACCEPT_LEN: u32 = 32u;
const INVALID_POS: u32 = 0xffffffffu;

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;
@group(0) @binding(1)
var<storage, read> tile_lens: array<u32>;
@group(0) @binding(2)
var<storage, read_write> _interm_words_unused: array<u32>;
@group(0) @binding(3)
var<storage, read_write> scratch_words: array<atomic<u32>>;

fn read_input_byte(byte_index: u32) -> u32 {
    let word = input_words[byte_index / 4u];
    let shift = (byte_index % 4u) * 8u;
    return (word >> shift) & 0xffu;
}

fn read_input_u32_le(byte_index: u32) -> u32 {
    let word_idx = byte_index / 4u;
    let shift = (byte_index % 4u) * 8u;
    let w0 = input_words[word_idx];
    if (shift == 0u) {
        return w0;
    }
    let w1 = input_words[word_idx + 1u];
    return (w0 >> shift) | (w1 << (32u - shift));
}

fn hash3_at(tile_base: u32, pos: u32) -> u32 {
    let b0 = read_input_byte(tile_base + pos);
    let b1 = read_input_byte(tile_base + pos + 1u);
    let b2 = read_input_byte(tile_base + pos + 2u);
    let v = b0 | (b1 << 8u) | (b2 << 16u);
    return ((v * 2654435761u) >> (32u - HASH_BITS)) & (HASH_SIZE - 1u);
}

@compute @workgroup_size(256, 1, 1)
fn match_best(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_count = arrayLength(&tile_lens);
    let total_pos = tile_count * TILE_SIZE;
    if (gid.x >= total_pos) {
        return;
    }
    let tile = gid.x / TILE_SIZE;
    let pos = gid.x % TILE_SIZE;
    let tile_len = tile_lens[tile];

    let scratch_base = tile * SCRATCH_WORDS_PER_TILE;
    let head_base = scratch_base;
    let prev_base = scratch_base + HASH_SIZE;
    let best_len_base = prev_base + TILE_SIZE;
    let best_dist_base = best_len_base + TILE_SIZE;

    if (pos >= tile_len || (pos + MIN_MATCH) > tile_len) {
        atomicStore(&scratch_words[best_len_base + pos], 0u);
        atomicStore(&scratch_words[best_dist_base + pos], 0u);
        return;
    }

    let tile_base = tile * TILE_SIZE;
    let h = hash3_at(tile_base, pos);
    var candidate = atomicLoad(&scratch_words[head_base + h]);
    var best_len: u32 = 0u;
    var best_dist: u32 = 0u;
    var chain: u32 = 0u;

    loop {
        if (candidate == INVALID_POS || chain >= MAX_CHAIN) {
            break;
        }
        if (best_len >= EARLY_ACCEPT_LEN && chain >= 4u) {
            break;
        }
        if (candidate < pos) {
            let dist = pos - candidate;
            if (dist <= FAST_WINDOW_SIZE) {
                let max_len = min(MAX_MATCH, tile_len - pos);
                // Early reject: if candidate can't beat current best, skip quickly.
                if (best_len > 0u && best_len < max_len) {
                    let a_next = read_input_byte(tile_base + candidate + best_len);
                    let b_next = read_input_byte(tile_base + pos + best_len);
                    if (a_next != b_next) {
                        candidate = atomicLoad(&scratch_words[prev_base + candidate]);
                        chain = chain + 1u;
                        continue;
                    }
                }

                var l: u32 = 0u;
                // Compare 4 bytes at a time first.
                loop {
                    if ((l + 4u) > max_len) {
                        break;
                    }
                    let a4 = read_input_u32_le(tile_base + candidate + l);
                    let b4 = read_input_u32_le(tile_base + pos + l);
                    if (a4 != b4) {
                        break;
                    }
                    l = l + 4u;
                }
                // Finish remaining tail bytes.
                loop {
                    if (l >= max_len) {
                        break;
                    }
                    let a = read_input_byte(tile_base + candidate + l);
                    let b = read_input_byte(tile_base + pos + l);
                    if (a != b) {
                        break;
                    }
                    l = l + 1u;
                }
                if (l >= MIN_MATCH && l > best_len) {
                    best_len = l;
                    best_dist = dist;
                    if (l == MAX_MATCH) {
                        break;
                    }
                    if (l >= EARLY_ACCEPT_LEN) {
                        break;
                    }
                }
            }
        }
        candidate = atomicLoad(&scratch_words[prev_base + candidate]);
        chain = chain + 1u;
    }

    atomicStore(&scratch_words[best_len_base + pos], best_len);
    atomicStore(&scratch_words[best_dist_base + pos], best_dist);
}
"#;

const STATIC_SCATTER_SHADER: &str = r#"
const TILE_SIZE: u32 = 65536u;
const SUBBLOCK_SIZE: u32 = 2048u;
const SUBBLOCKS_PER_TILE: u32 = 32u;
const NUM_STREAMS: u32 = 32u;
const BLOCK_MAX_LANE_BYTES: u32 = 192u;
const BLOCK_MAX_LANE_WORDS: u32 = BLOCK_MAX_LANE_BYTES / 4u;
const BLOCK_OUTPUT_WORDS: u32 = NUM_STREAMS * BLOCK_MAX_LANE_WORDS;
const OUTPUT_TILE_WORDS: u32 = SUBBLOCKS_PER_TILE * BLOCK_OUTPUT_WORDS;
const HASH_BITS: u32 = 15u;
const HASH_SIZE: u32 = 1u << HASH_BITS;
const MIN_MATCH: u32 = 3u;
const MAX_MATCH: u32 = 258u;
const WINDOW_SIZE: u32 = 32768u;
const INVALID_POS: u32 = 0xffffffffu;
const SCRATCH_WORDS_PER_TILE: u32 = HASH_SIZE + TILE_SIZE + TILE_SIZE + TILE_SIZE;

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;
@group(0) @binding(1)
var<storage, read> tile_lens: array<u32>;
@group(0) @binding(2)
var<storage, read_write> output_words: array<u32>;
@group(0) @binding(3)
var<storage, read_write> scratch_words: array<atomic<u32>>;

fn read_input_byte(byte_index: u32) -> u32 {
    let word = input_words[byte_index / 4u];
    let shift = (byte_index % 4u) * 8u;
    return (word >> shift) & 0xffu;
}

fn reverse_bits(v_in: u32, n: u32) -> u32 {
    return reverseBits(v_in) >> (32u - n);
}

fn static_ll_code(sym: u32) -> vec2<u32> {
    if (sym <= 143u) {
        let msb = 0x30u + sym;
        return vec2<u32>(reverse_bits(msb, 8u), 8u);
    }
    if (sym <= 255u) {
        let msb = 0x190u + (sym - 144u);
        return vec2<u32>(reverse_bits(msb, 9u), 9u);
    }
    if (sym <= 279u) {
        let msb = sym - 256u;
        return vec2<u32>(reverse_bits(msb, 7u), 7u);
    }
    let msb = 0xC0u + (sym - 280u);
    return vec2<u32>(reverse_bits(msb, 8u), 8u);
}

fn encode_length_symbol(len: u32) -> vec3<u32> {
    if (len < MIN_MATCH || len > MAX_MATCH) {
        return vec3<u32>(285u, 0u, 0u);
    }
    if (len <= 10u) {
        return vec3<u32>(254u + len, 0u, 0u);
    }
    if (len == 258u) {
        return vec3<u32>(285u, 0u, 0u);
    }

    var sym: u32 = 265u;
    var base: u32 = 11u;
    var eb: u32 = 1u;
    loop {
        let span = 1u << eb;
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            if (len < (base + span)) {
                return vec3<u32>(sym, len - base, eb);
            }
            sym = sym + 1u;
            base = base + span;
            if (sym > 284u) {
                break;
            }
        }
        if (sym > 284u) {
            break;
        }
        eb = eb + 1u;
    }
    return vec3<u32>(285u, 0u, 0u);
}

fn encode_distance_symbol(dist: u32) -> vec3<u32> {
    if (dist < 1u || dist > WINDOW_SIZE) {
        return vec3<u32>(0u, 0u, 0u);
    }
    if (dist <= 4u) {
        return vec3<u32>(dist - 1u, 0u, 0u);
    }
    let d = dist - 1u;
    let msb = 31u - countLeadingZeros(d);
    let eb = msb - 1u;
    let span = 1u << eb;
    let base_first = (1u << (eb + 1u)) + 1u;
    let off = dist - base_first;
    let second = select(0u, 1u, off >= span);
    let sym = 2u * eb + 2u + second;
    let extra = off - second * span;
    return vec3<u32>(sym, extra, eb);
}

fn push_bits_lane(
    lane: u32,
    v_in: u32,
    n_in: u32,
    lane_base: u32,
    out_word_pos: ptr<function, array<u32, 32>>,
    pack_word: ptr<function, array<u32, 32>>,
    pack_count: ptr<function, array<u32, 32>>,
    out_bytes: ptr<function, array<u32, 32>>,
    bitbuf: ptr<function, array<u32, 32>>,
    bitcnt: ptr<function, array<u32, 32>>,
    overflow: ptr<function, bool>,
) {
    var v = v_in;
    var n = n_in;

    loop {
        if (n == 0u) {
            break;
        }
        let room = 8u - (*bitcnt)[lane];
        let take = min(room, n);
        let mask = (1u << take) - 1u;
        (*bitbuf)[lane] = (*bitbuf)[lane] | ((v & mask) << (*bitcnt)[lane]);
        (*bitcnt)[lane] = (*bitcnt)[lane] + take;
        v = v >> take;
        n = n - take;

        if ((*bitcnt)[lane] == 8u) {
            let bytes = (*out_bytes)[lane];
            if (bytes < BLOCK_MAX_LANE_BYTES) {
                let shift = (*pack_count)[lane] * 8u;
                (*pack_word)[lane] = (*pack_word)[lane] | (((*bitbuf)[lane] & 0xffu) << shift);
                (*pack_count)[lane] = (*pack_count)[lane] + 1u;
                (*out_bytes)[lane] = bytes + 1u;
                if ((*pack_count)[lane] == 4u) {
                    let idx = lane_base + lane * BLOCK_MAX_LANE_WORDS + (*out_word_pos)[lane];
                    output_words[idx] = (*pack_word)[lane];
                    (*out_word_pos)[lane] = (*out_word_pos)[lane] + 1u;
                    (*pack_word)[lane] = 0u;
                    (*pack_count)[lane] = 0u;
                }
            } else {
                *overflow = true;
            }
            (*bitbuf)[lane] = 0u;
            (*bitcnt)[lane] = 0u;
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn scatter_static(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_count = arrayLength(&tile_lens);
    let total_blocks = tile_count * SUBBLOCKS_PER_TILE;
    let tile_block = gid.x;
    if (tile_block >= total_blocks) {
        return;
    }
    let tile = tile_block / SUBBLOCKS_PER_TILE;
    let block = tile_block % SUBBLOCKS_PER_TILE;

    let tile_len = tile_lens[tile];
    let block_start = block * SUBBLOCK_SIZE;
    let lane_base = tile * OUTPUT_TILE_WORDS + block * BLOCK_OUTPUT_WORDS;
    let bits_base = tile_count * OUTPUT_TILE_WORDS + tile * (SUBBLOCKS_PER_TILE * NUM_STREAMS) + block * NUM_STREAMS;
    let scratch_base = tile * SCRATCH_WORDS_PER_TILE;
    let best_len_base = scratch_base + HASH_SIZE + TILE_SIZE;
    let best_dist_base = best_len_base + TILE_SIZE;

    if (block_start >= tile_len) {
        for (var lane: u32 = 0u; lane < NUM_STREAMS; lane = lane + 1u) {
            output_words[bits_base + lane] = 0u;
        }
        return;
    }
    let block_end = min(block_start + SUBBLOCK_SIZE, tile_len);
    let tile_base = tile * TILE_SIZE;

    var out_word_pos: array<u32, 32>;
    var pack_word: array<u32, 32>;
    var pack_count: array<u32, 32>;
    var out_bytes: array<u32, 32>;
    var bitbuf: array<u32, 32>;
    var bitcnt: array<u32, 32>;
    for (var lane: u32 = 0u; lane < NUM_STREAMS; lane = lane + 1u) {
        out_word_pos[lane] = 0u;
        pack_word[lane] = 0u;
        pack_count[lane] = 0u;
        out_bytes[lane] = 0u;
        bitbuf[lane] = 0u;
        bitcnt[lane] = 0u;
    }
    var overflow = false;
    var lane_cursor: u32 = 0u;

    // BFINAL/BTYPE on lane 0 (single static block per sub-block).
    var bfinal: u32 = 0u;
    if (block_end == tile_len) {
        bfinal = 1u;
    }
    push_bits_lane(
        0u,
        bfinal,
        1u,
        lane_base,
        &out_word_pos,
        &pack_word,
        &pack_count,
        &out_bytes,
        &bitbuf,
        &bitcnt,
        &overflow,
    );
    push_bits_lane(
        0u,
        1u,
        2u,
        lane_base,
        &out_word_pos,
        &pack_word,
        &pack_count,
        &out_bytes,
        &bitbuf,
        &bitcnt,
        &overflow,
    );

    var pos: u32 = block_start;
    loop {
        if (pos >= block_end) {
            break;
        }

        var best_len = atomicLoad(&scratch_words[best_len_base + pos]);
        var best_dist = atomicLoad(&scratch_words[best_dist_base + pos]);

        if (best_len >= MIN_MATCH) {
            let emit_len = min(best_len, block_end - pos);
            if (emit_len < MIN_MATCH) {
                let lane = lane_cursor;
                let b = read_input_byte(tile_base + pos);
                let ll = static_ll_code(b);
                push_bits_lane(
                    lane,
                    ll.x,
                    ll.y,
                    lane_base,
                    &out_word_pos,
                    &pack_word,
                    &pack_count,
                    &out_bytes,
                    &bitbuf,
                    &bitcnt,
                    &overflow,
                );
                pos = pos + 1u;
                lane_cursor = (lane_cursor + 1u) & (NUM_STREAMS - 1u);
                continue;
            }
            let lane = lane_cursor;
            let len_enc = encode_length_symbol(emit_len);
            let ll = static_ll_code(len_enc.x);
            push_bits_lane(
                lane,
                ll.x,
                ll.y,
                lane_base,
                &out_word_pos,
                &pack_word,
                &pack_count,
                &out_bytes,
                &bitbuf,
                &bitcnt,
                &overflow,
            );
            if (len_enc.z > 0u) {
                push_bits_lane(
                    lane,
                    len_enc.y,
                    len_enc.z,
                    lane_base,
                    &out_word_pos,
                    &pack_word,
                    &pack_count,
                    &out_bytes,
                    &bitbuf,
                    &bitcnt,
                    &overflow,
                );
            }
            let dist_enc = encode_distance_symbol(best_dist);
            let dist_code_lsb = reverse_bits(dist_enc.x, 5u);
            push_bits_lane(
                lane,
                dist_code_lsb,
                5u,
                lane_base,
                &out_word_pos,
                &pack_word,
                &pack_count,
                &out_bytes,
                &bitbuf,
                &bitcnt,
                &overflow,
            );
            if (dist_enc.z > 0u) {
                push_bits_lane(
                    lane,
                    dist_enc.y,
                    dist_enc.z,
                    lane_base,
                    &out_word_pos,
                    &pack_word,
                    &pack_count,
                    &out_bytes,
                    &bitbuf,
                    &bitcnt,
                    &overflow,
                );
            }

            pos = pos + emit_len;
        } else {
            let lane = lane_cursor;
            let b = read_input_byte(tile_base + pos);
            let ll = static_ll_code(b);
            push_bits_lane(
                lane,
                ll.x,
                ll.y,
                lane_base,
                &out_word_pos,
                &pack_word,
                &pack_count,
                &out_bytes,
                &bitbuf,
                &bitcnt,
                &overflow,
            );
            pos = pos + 1u;
        }

        lane_cursor = (lane_cursor + 1u) & (NUM_STREAMS - 1u);
    }

    let eob_lane = lane_cursor;
    let eob = static_ll_code(256u);
    push_bits_lane(
        eob_lane,
        eob.x,
        eob.y,
        lane_base,
        &out_word_pos,
        &pack_word,
        &pack_count,
        &out_bytes,
        &bitbuf,
        &bitcnt,
        &overflow,
    );

    for (var lane: u32 = 0u; lane < NUM_STREAMS; lane = lane + 1u) {
        if (bitcnt[lane] > 0u) {
            let bytes = out_bytes[lane];
            if (bytes < BLOCK_MAX_LANE_BYTES) {
                let shift = pack_count[lane] * 8u;
                pack_word[lane] = pack_word[lane] | ((bitbuf[lane] & 0xffu) << shift);
                pack_count[lane] = pack_count[lane] + 1u;
                out_bytes[lane] = bytes + 1u;
                if (pack_count[lane] == 4u) {
                    let idx = lane_base + lane * BLOCK_MAX_LANE_WORDS + out_word_pos[lane];
                    output_words[idx] = pack_word[lane];
                    out_word_pos[lane] = out_word_pos[lane] + 1u;
                    pack_word[lane] = 0u;
                    pack_count[lane] = 0u;
                }
                bitbuf[lane] = 0u;
                bitcnt[lane] = 0u;
            } else {
                overflow = true;
            }
        }
        if (pack_count[lane] > 0u) {
            let idx = lane_base + lane * BLOCK_MAX_LANE_WORDS + out_word_pos[lane];
            output_words[idx] = pack_word[lane];
            out_word_pos[lane] = out_word_pos[lane] + 1u;
            pack_word[lane] = 0u;
            pack_count[lane] = 0u;
        }
    }

    for (var lane: u32 = 0u; lane < NUM_STREAMS; lane = lane + 1u) {
        if (overflow) {
            output_words[bits_base + lane] = INVALID_POS;
        } else {
            // Align each lane to byte boundary at sub-block boundary.
            output_words[bits_base + lane] = out_bytes[lane] * 8u;
        }
    }
}
"#;

const STATIC_REDUCE_SHADER: &str = r#"
const NUM_STREAMS: u32 = 32u;
const SUBBLOCK_SIZE: u32 = 2048u;
const SUBBLOCKS_PER_TILE: u32 = 32u;
const BLOCK_MAX_LANE_BYTES: u32 = 192u;
const BLOCK_MAX_LANE_WORDS: u32 = BLOCK_MAX_LANE_BYTES / 4u;
const BLOCK_OUTPUT_WORDS: u32 = NUM_STREAMS * BLOCK_MAX_LANE_WORDS;
const OUTPUT_TILE_WORDS: u32 = SUBBLOCKS_PER_TILE * BLOCK_OUTPUT_WORDS;
const AUX_WORDS_PER_TILE: u32 = NUM_STREAMS + 1u;
const MAX_LANE_BYTES: u32 = SUBBLOCKS_PER_TILE * BLOCK_MAX_LANE_BYTES;
const INVALID_POS: u32 = 0xffffffffu;

@group(0) @binding(0)
var<storage, read> _input_words_unused: array<u32>;
@group(0) @binding(1)
var<storage, read> tile_lens: array<u32>;
@group(0) @binding(2)
var<storage, read_write> interm_words: array<u32>;
@group(0) @binding(3)
var<storage, read_write> _scratch_words_unused: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn reduce_static(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_count = arrayLength(&tile_lens);
    let tile = gid.x;
    if (tile >= tile_count) {
        return;
    }

    let tile_len = tile_lens[tile];
    let block_count = (tile_len + (SUBBLOCK_SIZE - 1u)) / SUBBLOCK_SIZE;
    let bits_base = tile_count * OUTPUT_TILE_WORDS + tile * (SUBBLOCKS_PER_TILE * NUM_STREAMS);
    let aux_base = tile_count * OUTPUT_TILE_WORDS
        + tile_count * (SUBBLOCKS_PER_TILE * NUM_STREAMS)
        + tile * AUX_WORDS_PER_TILE;

    var max_lane_words: u32 = 0u;
    for (var lane: u32 = 0u; lane < NUM_STREAMS; lane = lane + 1u) {
        var total_bytes: u32 = 0u;
        for (var blk: u32 = 0u; blk < block_count; blk = blk + 1u) {
            let bit_len = interm_words[bits_base + blk * NUM_STREAMS + lane];
            if (bit_len == INVALID_POS) {
                interm_words[aux_base + NUM_STREAMS] = INVALID_POS;
                return;
            }
            total_bytes = total_bytes + ((bit_len + 7u) / 8u);
        }
        if (total_bytes > MAX_LANE_BYTES) {
            interm_words[aux_base + NUM_STREAMS] = INVALID_POS;
            return;
        }
        let lane_words = (total_bytes + 3u) / 4u;
        interm_words[aux_base + lane] = total_bytes;
        if (lane_words > max_lane_words) {
            max_lane_words = lane_words;
        }
    }
    interm_words[aux_base + NUM_STREAMS] = max_lane_words;
}
"#;

const STATIC_SERIALIZE_SHADER: &str = r#"
const NUM_STREAMS: u32 = 32u;
const SUBBLOCK_SIZE: u32 = 2048u;
const SUBBLOCKS_PER_TILE: u32 = 32u;
const BLOCK_MAX_LANE_BYTES: u32 = 192u;
const BLOCK_MAX_LANE_WORDS: u32 = BLOCK_MAX_LANE_BYTES / 4u;
const BLOCK_OUTPUT_WORDS: u32 = NUM_STREAMS * BLOCK_MAX_LANE_WORDS;
const OUTPUT_TILE_WORDS: u32 = SUBBLOCKS_PER_TILE * BLOCK_OUTPUT_WORDS;
const AUX_WORDS_PER_TILE: u32 = NUM_STREAMS + 1u;
const MAX_LANE_BYTES: u32 = SUBBLOCKS_PER_TILE * BLOCK_MAX_LANE_BYTES;
const MAX_LANE_WORDS: u32 = MAX_LANE_BYTES / 4u;
const MAX_PAGE_WORDS: u32 = (MAX_LANE_WORDS * NUM_STREAMS) + 32u;
const INVALID_POS: u32 = 0xffffffffu;

@group(0) @binding(0)
var<storage, read> interm_words: array<u32>;
@group(0) @binding(1)
var<storage, read> tile_lens: array<u32>;
@group(0) @binding(2)
var<storage, read_write> final_words: array<u32>;
@group(0) @binding(3)
var<storage, read_write> _scratch_words_unused: array<u32>;

@compute @workgroup_size(128, 1, 1)
fn serialize_static(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_count = arrayLength(&tile_lens);
    let pair = gid.x;
    let tile = pair / NUM_STREAMS;
    if (tile >= tile_count) {
        return;
    }
    let lane = pair % NUM_STREAMS;

    let tile_len = tile_lens[tile];
    let block_count = (tile_len + (SUBBLOCK_SIZE - 1u)) / SUBBLOCK_SIZE;
    let lane_word_base = tile * OUTPUT_TILE_WORDS;
    let bits_base = tile_count * OUTPUT_TILE_WORDS + tile * (SUBBLOCKS_PER_TILE * NUM_STREAMS);
    let aux_base = tile_count * OUTPUT_TILE_WORDS
        + tile_count * (SUBBLOCKS_PER_TILE * NUM_STREAMS)
        + tile * AUX_WORDS_PER_TILE;
    let dst_base = tile * MAX_PAGE_WORDS;
    let meta_base = tile_count * MAX_PAGE_WORDS;

    let byte_len = interm_words[aux_base + lane];
    let max_lane_words = interm_words[aux_base + NUM_STREAMS];
    if (byte_len == INVALID_POS || max_lane_words == INVALID_POS
        || byte_len > MAX_LANE_BYTES || max_lane_words > MAX_LANE_WORDS) {
        final_words[meta_base + tile] = 0xffffffffu;
        return;
    }
    let lane_words = (byte_len + 3u) / 4u;
    if (lane_words > max_lane_words) {
        final_words[meta_base + tile] = 0xffffffffu;
        return;
    }

    var produced_bytes: u32 = 0u;
    var dst_word_index: u32 = 0u;
    var packed_word: u32 = 0u;
    var packed_count: u32 = 0u;
    for (var blk: u32 = 0u; blk < block_count; blk = blk + 1u) {
        if (produced_bytes >= byte_len) {
            break;
        }
        let blk_bits = interm_words[bits_base + blk * NUM_STREAMS + lane];
        if (blk_bits == INVALID_POS) {
            final_words[meta_base + tile] = INVALID_POS;
            return;
        }
        let blk_bytes = (blk_bits + 7u) / 8u;
        let src_lane_base = lane_word_base + blk * BLOCK_OUTPUT_WORDS + lane * BLOCK_MAX_LANE_WORDS;
        for (var bi: u32 = 0u; bi < blk_bytes; bi = bi + 1u) {
            if (produced_bytes >= byte_len) {
                break;
            }
            let src_word = interm_words[src_lane_base + (bi / 4u)];
            let b = (src_word >> ((bi % 4u) * 8u)) & 0xffu;
            packed_word = packed_word | (b << (packed_count * 8u));
            packed_count = packed_count + 1u;
            produced_bytes = produced_bytes + 1u;
            if (packed_count == 4u) {
                let row_base = dst_base + dst_word_index * NUM_STREAMS;
                final_words[row_base + lane] = packed_word;
                dst_word_index = dst_word_index + 1u;
                packed_word = 0u;
                packed_count = 0u;
            }
        }
    }
    if (produced_bytes != byte_len) {
        final_words[meta_base + tile] = INVALID_POS;
        return;
    }
    if (packed_count > 0u) {
        let row_base = dst_base + dst_word_index * NUM_STREAMS;
        final_words[row_base + lane] = packed_word;
        dst_word_index = dst_word_index + 1u;
    }
    if (dst_word_index != lane_words) {
        final_words[meta_base + tile] = INVALID_POS;
        return;
    }

    // Lane 0 finalizes metadata and trailing padding.
    if (lane != 0u) {
        return;
    }

    let pad_base = dst_base + max_lane_words * NUM_STREAMS;
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        final_words[pad_base + i] = 0u;
    }

    let page_words = (max_lane_words * NUM_STREAMS) + 32u;
    final_words[meta_base + tile] = page_words;
}
"#;

const STATIC_DECODE_SHADER: &str = r#"
const TILE_SIZE: u32 = 65536u;
const NUM_STREAMS: u32 = 32u;
const TRAILING_PAD_BYTES: u32 = 128u;
const STATIC_LITERAL_MAX_BITS: u32 = 9u;

const ERR_UNSUPPORTED_BTYPE: u32 = 2u;
const ERR_BIT_UNDERFLOW: u32 = 3u;
const ERR_INVALID_STATIC_SYM: u32 = 4u;
const ERR_INVALID_LENGTH_SYM: u32 = 5u;
const ERR_INVALID_DIST_SYM: u32 = 6u;
const ERR_INVALID_DISTANCE: u32 = 7u;
const ERR_OUTPUT_OVERFLOW: u32 = 8u;
const ERR_DECODE_GUARD: u32 = 9u;

fn length_base(idx: u32) -> u32 {
    switch idx {
        case 0u: { return 3u; }
        case 1u: { return 4u; }
        case 2u: { return 5u; }
        case 3u: { return 6u; }
        case 4u: { return 7u; }
        case 5u: { return 8u; }
        case 6u: { return 9u; }
        case 7u: { return 10u; }
        case 8u: { return 11u; }
        case 9u: { return 13u; }
        case 10u: { return 15u; }
        case 11u: { return 17u; }
        case 12u: { return 19u; }
        case 13u: { return 23u; }
        case 14u: { return 27u; }
        case 15u: { return 31u; }
        case 16u: { return 35u; }
        case 17u: { return 43u; }
        case 18u: { return 51u; }
        case 19u: { return 59u; }
        case 20u: { return 67u; }
        case 21u: { return 83u; }
        case 22u: { return 99u; }
        case 23u: { return 115u; }
        case 24u: { return 131u; }
        case 25u: { return 163u; }
        case 26u: { return 195u; }
        case 27u: { return 227u; }
        default: { return 258u; }
    }
}

fn length_extra(idx: u32) -> u32 {
    switch idx {
        case 8u, 9u, 10u, 11u: { return 1u; }
        case 12u, 13u, 14u, 15u: { return 2u; }
        case 16u, 17u, 18u, 19u: { return 3u; }
        case 20u, 21u, 22u, 23u: { return 4u; }
        case 24u, 25u, 26u, 27u: { return 5u; }
        default: { return 0u; }
    }
}

fn dist_base(sym: u32) -> u32 {
    switch sym {
        case 0u: { return 1u; }
        case 1u: { return 2u; }
        case 2u: { return 3u; }
        case 3u: { return 4u; }
        case 4u: { return 5u; }
        case 5u: { return 7u; }
        case 6u: { return 9u; }
        case 7u: { return 13u; }
        case 8u: { return 17u; }
        case 9u: { return 25u; }
        case 10u: { return 33u; }
        case 11u: { return 49u; }
        case 12u: { return 65u; }
        case 13u: { return 97u; }
        case 14u: { return 129u; }
        case 15u: { return 193u; }
        case 16u: { return 257u; }
        case 17u: { return 385u; }
        case 18u: { return 513u; }
        case 19u: { return 769u; }
        case 20u: { return 1025u; }
        case 21u: { return 1537u; }
        case 22u: { return 2049u; }
        case 23u: { return 3073u; }
        case 24u: { return 4097u; }
        case 25u: { return 6145u; }
        case 26u: { return 8193u; }
        case 27u: { return 12289u; }
        case 28u: { return 16385u; }
        default: { return 24577u; }
    }
}

fn dist_extra(sym: u32) -> u32 {
    switch sym {
        case 4u, 5u: { return 1u; }
        case 6u, 7u: { return 2u; }
        case 8u, 9u: { return 3u; }
        case 10u, 11u: { return 4u; }
        case 12u, 13u: { return 5u; }
        case 14u, 15u: { return 6u; }
        case 16u, 17u: { return 7u; }
        case 18u, 19u: { return 8u; }
        case 20u, 21u: { return 9u; }
        case 22u, 23u: { return 10u; }
        case 24u, 25u: { return 11u; }
        case 26u, 27u: { return 12u; }
        case 28u, 29u: { return 13u; }
        default: { return 0u; }
    }
}

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;
@group(0) @binding(1)
var<storage, read> tile_meta_words: array<u32>; // [page_off, page_len, expected_len] * tiles
@group(0) @binding(2)
var<storage, read_write> output_words: array<u32>; // [status*tiles][produced*tiles][tile_out_words]

var<workgroup> lane_lo: array<u32, 32>;
var<workgroup> lane_hi: array<u32, 32>;
var<workgroup> lane_bits: array<u32, 32>;
var<workgroup> lane_cursor: array<u32, 32>;
var<workgroup> decode_error: u32;
var<workgroup> cur_page_off: u32;
var<workgroup> cur_words_per_lane: u32;
var<workgroup> produced_total: u32;
var<workgroup> block_bfinal: u32;
var<workgroup> block_btype: u32;
var<workgroup> block_done: u32;
var<workgroup> eob_lane: u32;
var<workgroup> round_len_sum: u32;
var<workgroup> lane_sym: array<u32, 32>;
var<workgroup> lane_sym_bits: array<u32, 32>;
var<workgroup> lane_kind: array<u32, 32>; // 0=lit 1=end 2=copy 4=oob
var<workgroup> lane_lit: array<u32, 32>;
var<workgroup> lane_len: array<u32, 32>;
var<workgroup> lane_dist: array<u32, 32>;
var<workgroup> lane_off: array<u32, 32>;
var<workgroup> copy_active: u32;
var<workgroup> copy_dist: u32;
var<workgroup> copy_len: u32;
var<workgroup> copy_dst: u32;

fn reverse_bits_n(code_in: u32, len: u32) -> u32 {
    var out: u32 = 0u;
    var code: u32 = code_in;
    var i: u32 = 0u;
    loop {
        if (i >= len) { break; }
        out = (out << 1u) | (code & 1u);
        code = code >> 1u;
        i = i + 1u;
    }
    return out;
}

fn read_input_word(byte_off: u32) -> u32 {
    return input_words[byte_off / 4u];
}

fn lane_refill(lane: u32) -> bool {
    if (lane_bits[lane] > 32u) {
        return true;
    }
    let cursor = lane_cursor[lane];
    if (cursor >= cur_words_per_lane) {
        return false;
    }
    let payload_word_index = cursor * NUM_STREAMS + lane;
    let byte_off = cur_page_off + payload_word_index * 4u;
    let w = read_input_word(byte_off);
    lane_cursor[lane] = cursor + 1u;
    let bits = lane_bits[lane];
    if (bits == 32u) {
        lane_hi[lane] = w;
        lane_bits[lane] = 64u;
        return true;
    }
    if (bits == 0u) {
        lane_lo[lane] = w;
        lane_hi[lane] = 0u;
        lane_bits[lane] = 32u;
        return true;
    }
    lane_lo[lane] = lane_lo[lane] | (w << bits);
    if (bits > 0u) {
        lane_hi[lane] = w >> (32u - bits);
    } else {
        lane_hi[lane] = w;
    }
    lane_bits[lane] = bits + 32u;
    return true;
}

fn lane_ensure_bits(lane: u32, need_bits: u32) -> bool {
    loop {
        if (lane_bits[lane] >= need_bits) {
            return true;
        }
        if (!lane_refill(lane)) {
            return false;
        }
    }
    return false;
}

fn lane_try_ensure_bits(lane: u32, need_bits: u32) -> bool {
    loop {
        if (lane_bits[lane] >= need_bits) {
            return true;
        }
        if (!lane_refill(lane)) {
            return false;
        }
    }
    return false;
}

fn lane_read_bits_raw(lane: u32, bits: u32) -> u32 {
    if (bits == 0u) {
        return 0u;
    }
    var m: u32 = 0xffffffffu;
    if (bits < 32u) {
        m = (1u << bits) - 1u;
    }
    let v = lane_lo[lane] & m;
    if (bits < 32u) {
        lane_lo[lane] = (lane_lo[lane] >> bits) | (lane_hi[lane] << (32u - bits));
        lane_hi[lane] = lane_hi[lane] >> bits;
    } else {
        lane_lo[lane] = lane_hi[lane];
        lane_hi[lane] = 0u;
    }
    lane_bits[lane] = lane_bits[lane] - bits;
    return v;
}

fn lane_peek_bits_try(lane: u32, bits: u32) -> vec2<u32> {
    if (!lane_try_ensure_bits(lane, bits)) {
        return vec2<u32>(0u, 0u);
    }
    var m: u32 = 0xffffffffu;
    if (bits < 32u) {
        m = (1u << bits) - 1u;
    }
    return vec2<u32>(lane_lo[lane] & m, 1u);
}

fn lane_peek_bits_checked(lane: u32, bits: u32) -> u32 {
    if (!lane_ensure_bits(lane, bits)) {
        if (decode_error == 0u) {
            decode_error = ERR_BIT_UNDERFLOW;
        }
        return 0u;
    }
    var m: u32 = 0xffffffffu;
    if (bits < 32u) {
        m = (1u << bits) - 1u;
    }
    return lane_lo[lane] & m;
}

fn lane_consume_bits(lane: u32, bits: u32) {
    let _consumed = lane_read_bits_raw(lane, bits);
}

fn lane_read_bits_checked(lane: u32, bits: u32) -> u32 {
    let v = lane_peek_bits_checked(lane, bits);
    if (decode_error == 0u) {
        lane_consume_bits(lane, bits);
    }
    return v;
}

fn decode_static_ll_symbol_peek(lane: u32) -> vec2<u32> {
    var len: u32 = 1u;
    loop {
        if (len > STATIC_LITERAL_MAX_BITS) { break; }
        let peek = lane_peek_bits_try(lane, len);
        if (peek.y == 0u) {
            return vec2<u32>(0xffffffffu, 0u);
        }
        let code_lsb = peek.x;
        let msb = reverse_bits_n(code_lsb, len);
        if (len == 7u && msb <= 23u) {
            return vec2<u32>(256u + msb, len);
        }
        if (len == 8u && msb >= 48u && msb <= 191u) {
            return vec2<u32>(msb - 48u, len);
        }
        if (len == 8u && msb >= 192u && msb <= 199u) {
            return vec2<u32>(280u + (msb - 192u), len);
        }
        if (len == 9u && msb >= 400u && msb <= 511u) {
            return vec2<u32>(144u + (msb - 400u), len);
        }
        len = len + 1u;
    }
    return vec2<u32>(0xffffffffu, 0u);
}

fn decode_length_from_symbol(sym: u32, lane: u32) -> u32 {
    if (sym == 285u) {
        return 258u;
    }
    if (sym < 257u || sym > 284u) {
        if (decode_error == 0u) {
            decode_error = ERR_INVALID_LENGTH_SYM;
        }
        return 0u;
    }
    let idx = sym - 257u;
    let base = length_base(idx);
    let extra_bits = length_extra(idx);
    var extra: u32 = 0u;
    if (extra_bits != 0u) {
        extra = lane_read_bits_checked(lane, extra_bits);
    }
    return base + extra;
}

fn decode_static_distance(lane: u32) -> u32 {
    let code_lsb = lane_read_bits_checked(lane, 5u);
    if (decode_error != 0u) {
        return 0u;
    }
    let sym = reverse_bits_n(code_lsb, 5u);
    if (sym > 29u) {
        if (decode_error == 0u) {
            decode_error = ERR_INVALID_DIST_SYM;
        }
        return 0u;
    }
    let base = dist_base(sym);
    let extra_bits = dist_extra(sym);
    var extra: u32 = 0u;
    if (extra_bits != 0u) {
        extra = lane_read_bits_checked(lane, extra_bits);
    }
    return base + extra;
}

fn lane_align_byte(lane: u32) {
    let rem = lane_bits[lane] & 7u;
    if (rem != 0u) {
        lane_consume_bits(lane, rem);
    }
}

@compute @workgroup_size(32, 1, 1)
fn decode_static_tile(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tile = wid.x;
    let lane = lid.x;
    let tile_count = arrayLength(&tile_meta_words) / 3u;
    if (tile >= tile_count) {
        return;
    }
    let meta_base = tile * 3u;
    let page_off = tile_meta_words[meta_base + 0u];
    let page_len = tile_meta_words[meta_base + 1u];
    let expected_len = tile_meta_words[meta_base + 2u];
    let status_base = 0u;
    let produced_base = tile_count;
    let out_words_base = tile_count * 2u;
    let out_base = out_words_base + tile * TILE_SIZE;

    if (lane == 0u) {
        output_words[status_base + tile] = 0u;
        output_words[produced_base + tile] = 0u;
        decode_error = 0u;
        produced_total = 0u;
        block_bfinal = 0u;
        block_btype = 0u;
        block_done = 0u;
        eob_lane = NUM_STREAMS;
        round_len_sum = 0u;
        copy_active = 0u;
    }
    workgroupBarrier();

    if (page_len < TRAILING_PAD_BYTES) {
        if (lane == 0u) {
            output_words[status_base + tile] = ERR_UNSUPPORTED_BTYPE;
        }
        return;
    }
    let payload_len = page_len - TRAILING_PAD_BYTES;
    if ((payload_len % (NUM_STREAMS * 4u)) != 0u) {
        if (lane == 0u) {
            output_words[status_base + tile] = ERR_UNSUPPORTED_BTYPE;
        }
        return;
    }

    if (lane == 0u) {
        cur_page_off = page_off;
        cur_words_per_lane = payload_len / (NUM_STREAMS * 4u);
    }
    workgroupBarrier();

    lane_lo[lane] = 0u;
    lane_hi[lane] = 0u;
    lane_bits[lane] = 0u;
    lane_cursor[lane] = 0u;
    if (cur_words_per_lane > 0u) {
        let byte_off = cur_page_off + lane * 4u;
        lane_lo[lane] = read_input_word(byte_off);
        lane_bits[lane] = 32u;
        lane_cursor[lane] = 1u;
    }
    workgroupBarrier();

    if (lane == 0u) {
        block_bfinal = lane_read_bits_checked(0u, 1u);
        block_btype = lane_read_bits_checked(0u, 2u);
    }
    workgroupBarrier();

    var block_guard: u32 = 0u;
    loop {
        if (decode_error != 0u) { break; }
        if (lane == 0u) {
            if (block_guard > 4096u) {
                decode_error = ERR_DECODE_GUARD;
            }
            block_guard = block_guard + 1u;
            block_done = 0u;
            eob_lane = NUM_STREAMS;
            if (decode_error == 0u && block_btype != 1u) {
                decode_error = ERR_UNSUPPORTED_BTYPE;
            }
            copy_active = 0u;
        }
        workgroupBarrier();
        if (decode_error != 0u) { break; }

        var lane_loop_guard: u32 = 0u;
        loop {
            if (decode_error != 0u || block_done != 0u) { break; }
            if (lane == 0u) {
                if (lane_loop_guard > (TILE_SIZE * 4u)) {
                    decode_error = ERR_DECODE_GUARD;
                }
                lane_loop_guard = lane_loop_guard + 1u;
            }
            workgroupBarrier();
            if (decode_error != 0u) { break; }

            let sym_bits = decode_static_ll_symbol_peek(lane);
            lane_sym[lane] = sym_bits.x;
            lane_sym_bits[lane] = sym_bits.y;
            lane_kind[lane] = 4u;
            lane_lit[lane] = 0u;
            lane_len[lane] = 0u;
            lane_dist[lane] = 0u;
            lane_off[lane] = 0u;
            workgroupBarrier();

            if (lane == 0u) {
                var first_eob = NUM_STREAMS;
                var i: u32 = 0u;
                loop {
                    if (i >= NUM_STREAMS) { break; }
                    if (lane_sym[i] == 256u) {
                        first_eob = i;
                        break;
                    }
                    i = i + 1u;
                }
                eob_lane = first_eob;
            }
            workgroupBarrier();

            let lane_active = (eob_lane == NUM_STREAMS) || (lane <= eob_lane);
            if (lane_active && decode_error == 0u) {
                let sym = lane_sym[lane];
                let sym_consumed_bits = lane_sym_bits[lane];
                if (sym == 0xffffffffu || sym_consumed_bits == 0u) {
                    if (decode_error == 0u) {
                        decode_error = ERR_BIT_UNDERFLOW;
                    }
                } else {
                    lane_consume_bits(lane, sym_consumed_bits);
                    if (sym < 256u) {
                        lane_kind[lane] = 0u;
                        lane_lit[lane] = sym;
                        lane_len[lane] = 1u;
                    } else if (sym == 256u) {
                        lane_kind[lane] = 1u;
                    } else if (sym <= 285u) {
                        let mlen = decode_length_from_symbol(sym, lane);
                        let mdist = decode_static_distance(lane);
                        if (decode_error == 0u) {
                            lane_kind[lane] = 2u;
                            lane_len[lane] = mlen;
                            lane_dist[lane] = mdist;
                        }
                    } else {
                        decode_error = ERR_INVALID_STATIC_SYM;
                    }
                }
            } else {
                lane_kind[lane] = 4u;
            }
            workgroupBarrier();
            if (decode_error != 0u) { break; }

            if (lane == 0u) {
                var run_off: u32 = 0u;
                var i: u32 = 0u;
                loop {
                    if (i >= NUM_STREAMS) { break; }
                    lane_off[i] = run_off;
                    let kind = lane_kind[i];
                    var add: u32 = 0u;
                    if (kind == 0u) {
                        add = 1u;
                    } else if (kind == 2u) {
                        add = lane_len[i];
                    }
                    run_off = run_off + add;
                    i = i + 1u;
                }
                round_len_sum = run_off;
                if (produced_total + round_len_sum > expected_len || produced_total + round_len_sum > TILE_SIZE) {
                    decode_error = ERR_OUTPUT_OVERFLOW;
                }
            }
            workgroupBarrier();
            if (decode_error != 0u) { break; }

            if (lane_kind[lane] == 0u) {
                let dst = out_base + produced_total + lane_off[lane];
                output_words[dst] = lane_lit[lane] & 0xffu;
            }
            workgroupBarrier();

            var c: u32 = 0u;
            loop {
                if (c >= NUM_STREAMS) { break; }
                if (lane == 0u) {
                    if (lane_kind[c] == 2u) {
                        let dst_off = produced_total + lane_off[c];
                        let dist = lane_dist[c];
                        let mlen = lane_len[c];
                        if (dist == 0u || dist > dst_off) {
                            decode_error = ERR_INVALID_DISTANCE;
                            copy_active = 0u;
                        } else {
                            copy_active = 1u;
                            copy_dist = dist;
                            copy_len = mlen;
                            copy_dst = dst_off;
                        }
                    } else {
                        copy_active = 0u;
                    }
                }
                workgroupBarrier();
                if (decode_error != 0u) { break; }
                if (copy_active != 0u) {
                    var j = lane;
                    loop {
                        if (j >= copy_len) { break; }
                        let src = out_base + copy_dst + (j % copy_dist) - copy_dist;
                        let b = output_words[src] & 0xffu;
                        output_words[out_base + copy_dst + j] = b;
                        j = j + NUM_STREAMS;
                    }
                }
                workgroupBarrier();
                c = c + 1u;
            }
            if (decode_error != 0u) { break; }

            if (lane == 0u) {
                produced_total = produced_total + round_len_sum;
                if (eob_lane != NUM_STREAMS) {
                    block_done = 1u;
                }
            }
            workgroupBarrier();
            if (block_done != 0u) { break; }
        }

        if (decode_error != 0u) { break; }
        if (block_bfinal != 0u) {
            break;
        }

        lane_align_byte(lane);
        workgroupBarrier();
        if (lane == 0u) {
            block_bfinal = lane_read_bits_checked(0u, 1u);
            block_btype = lane_read_bits_checked(0u, 2u);
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        if (decode_error == 0u && produced_total != expected_len) {
            decode_error = ERR_OUTPUT_OVERFLOW;
        }
        output_words[produced_base + tile] = produced_total;
        output_words[status_base + tile] = decode_error;
    }
}
"#;

struct StoredGpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    timestamp_query_supported: bool,
    static_timestamp_profile_enabled: bool,
    decode_timestamp_profile_enabled: bool,
    timestamp_period_ns: f64,
    stored_pipeline: wgpu::ComputePipeline,
    static_hash_reset_pipeline: wgpu::ComputePipeline,
    static_hash_build_pipeline: wgpu::ComputePipeline,
    static_match_pipeline: wgpu::ComputePipeline,
    static_pipeline: wgpu::ComputePipeline,
    static_reduce_pipeline: wgpu::ComputePipeline,
    static_serialize_pipeline: wgpu::ComputePipeline,
    static_decode_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    decode_bind_group_layout: wgpu::BindGroupLayout,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct GpuBatchStats {
    pub tiles: usize,
    pub upload_ms: f64,
    pub submit_wait_ms: f64,
    pub map_copy_ms: f64,
    pub repack_ms: f64,
    pub total_ms: f64,
    pub static_hash_reset_ms: f64,
    pub static_hash_build_ms: f64,
    pub static_match_ms: f64,
    pub static_scatter_ms: f64,
    pub static_serialize_ms: f64,
    pub static_copy_ms: f64,
    pub static_profiled: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct GpuDecodeBatchStats {
    pub tiles: usize,
    pub upload_ms: f64,
    pub submit_wait_ms: f64,
    pub map_copy_ms: f64,
    pub total_ms: f64,
    pub static_decode_ms: f64,
    pub static_copy_ms: f64,
    pub gpu_exec_ms: f64,
    pub submit_overhead_ms: f64,
    pub static_profiled: bool,
}

impl StoredGpuRuntime {
    fn init() -> Result<Self, GDeflateError> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| GDeflateError::Gpu("no suitable GPU adapter".to_string()))?;

        let supported_features = adapter.features();
        let timestamp_query_supported =
            supported_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let static_timestamp_profile_enabled = std::env::var("COZIP_GDEFLATE_GPU_STATIC_PROFILE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let decode_timestamp_profile_enabled = std::env::var("COZIP_GDEFLATE_GPU_DECODE_PROFILE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut required_features = wgpu::Features::empty();
        if timestamp_query_supported
            && (static_timestamp_profile_enabled || decode_timestamp_profile_enabled)
        {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cozip-gdeflate-device"),
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|e| GDeflateError::Gpu(format!("request_device failed: {e}")))?;
        let timestamp_period_ns = f64::from(queue.get_timestamp_period());

        let stored_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-gdeflate-stored-scatter"),
            source: wgpu::ShaderSource::Wgsl(STORED_SCATTER_SHADER.into()),
        });
        let static_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-gdeflate-static-scatter"),
            source: wgpu::ShaderSource::Wgsl(STATIC_SCATTER_SHADER.into()),
        });
        let static_hash_reset_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-gdeflate-static-hash-reset"),
            source: wgpu::ShaderSource::Wgsl(STATIC_HASH_RESET_SHADER.into()),
        });
        let static_hash_build_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-gdeflate-static-hash-build"),
            source: wgpu::ShaderSource::Wgsl(STATIC_HASH_BUILD_SHADER.into()),
        });
        let static_match_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-gdeflate-static-match"),
            source: wgpu::ShaderSource::Wgsl(STATIC_MATCH_SHADER.into()),
        });
        let static_reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-gdeflate-static-reduce"),
            source: wgpu::ShaderSource::Wgsl(STATIC_REDUCE_SHADER.into()),
        });
        let static_serialize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-gdeflate-static-serialize"),
            source: wgpu::ShaderSource::Wgsl(STATIC_SERIALIZE_SHADER.into()),
        });
        let static_decode_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cozip-gdeflate-static-decode"),
            source: wgpu::ShaderSource::Wgsl(STATIC_DECODE_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cozip-gdeflate-bgl"),
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
            ],
        });
        let decode_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cozip-gdeflate-decode-bgl"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cozip-gdeflate-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let decode_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cozip-gdeflate-decode-pl"),
                bind_group_layouts: &[&decode_bind_group_layout],
                push_constant_ranges: &[],
            });

        let stored_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-gdeflate-stored-cp"),
            layout: Some(&pipeline_layout),
            module: &stored_shader,
            entry_point: "scatter_stored",
        });
        let static_hash_reset_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cozip-gdeflate-static-hash-reset-cp"),
                layout: Some(&pipeline_layout),
                module: &static_hash_reset_shader,
                entry_point: "reset_hash",
            });
        let static_hash_build_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cozip-gdeflate-static-hash-build-cp"),
                layout: Some(&pipeline_layout),
                module: &static_hash_build_shader,
                entry_point: "build_hash",
            });
        let static_match_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cozip-gdeflate-static-match-cp"),
                layout: Some(&pipeline_layout),
                module: &static_match_shader,
                entry_point: "match_best",
            });
        let static_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cozip-gdeflate-static-cp"),
            layout: Some(&pipeline_layout),
            module: &static_shader,
            entry_point: "scatter_static",
        });
        let static_reduce_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cozip-gdeflate-static-reduce-cp"),
                layout: Some(&pipeline_layout),
                module: &static_reduce_shader,
                entry_point: "reduce_static",
            });
        let static_serialize_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cozip-gdeflate-static-serialize-cp"),
                layout: Some(&pipeline_layout),
                module: &static_serialize_shader,
                entry_point: "serialize_static",
            });
        let static_decode_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cozip-gdeflate-static-decode-cp"),
                layout: Some(&decode_pipeline_layout),
                module: &static_decode_shader,
                entry_point: "decode_static_tile",
            });

        Ok(Self {
            device,
            queue,
            timestamp_query_supported,
            static_timestamp_profile_enabled,
            decode_timestamp_profile_enabled,
            timestamp_period_ns,
            stored_pipeline,
            static_hash_reset_pipeline,
            static_hash_build_pipeline,
            static_match_pipeline,
            static_pipeline,
            static_reduce_pipeline,
            static_serialize_pipeline,
            static_decode_pipeline,
            bind_group_layout,
            decode_bind_group_layout,
        })
    }
}

fn runtime() -> Result<&'static StoredGpuRuntime, GDeflateError> {
    static RUNTIME: OnceLock<Result<StoredGpuRuntime, GDeflateError>> = OnceLock::new();
    RUNTIME
        .get_or_init(StoredGpuRuntime::init)
        .as_ref()
        .map_err(|e| GDeflateError::Gpu(format!("{e}")))
}

fn pack_bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    let mut words = Vec::with_capacity(bytes.len().div_ceil(4));
    for chunk in bytes.chunks(4) {
        let mut tmp = [0_u8; 4];
        tmp[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(tmp));
    }
    words
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len().saturating_mul(4));
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn stored_lane_payload_len(tile_len: usize, lane: usize) -> usize {
    if tile_len == GDEFLATE_TILE_SIZE {
        match lane {
            0 => 2049,
            31 => 2047,
            _ => 2048,
        }
    } else if lane >= tile_len {
        0
    } else {
        ((tile_len - 1 - lane) / GDEFLATE_NUM_STREAMS) + 1
    }
}

fn build_stored_page_from_lane_block(
    tile_len: usize,
    lane_block: &[u8],
) -> Result<Vec<u8>, GDeflateError> {
    if lane_block.len() != STORED_OUTPUT_TILE_BYTES {
        return Err(GDeflateError::Gpu(
            "stored lane block size mismatch".to_string(),
        ));
    }

    let mut payload_lanes: [Vec<u8>; GDEFLATE_NUM_STREAMS] = std::array::from_fn(|_| Vec::new());
    for lane in 0..GDEFLATE_NUM_STREAMS {
        let count = stored_lane_payload_len(tile_len, lane);
        let start = lane.saturating_mul(STORED_LANE_PAYLOAD_STRIDE);
        let end = start.saturating_add(count);
        payload_lanes[lane] = lane_block
            .get(start..end)
            .ok_or_else(|| {
                GDeflateError::Gpu("stored lane payload slice out of bounds".to_string())
            })?
            .to_vec();
    }

    let mut lanes: [BitWriter; GDEFLATE_NUM_STREAMS] =
        std::array::from_fn(|_| BitWriter::default());
    for lane in 1..GDEFLATE_NUM_STREAMS {
        lanes[lane].bytes = payload_lanes[lane].clone();
        lanes[lane].bit_pos = lanes[lane].bytes.len().saturating_mul(8);
    }

    if tile_len == GDEFLATE_TILE_SIZE {
        if payload_lanes[0].len() != 2049 {
            return Err(GDeflateError::Gpu(
                "stored lane0 payload length mismatch for full tile".to_string(),
            ));
        }
        let mut lane0 = Vec::with_capacity(5 + 2048 + 5 + 1);
        lane0.extend_from_slice(&[0x00, 0xFF, 0xFF, 0x00, 0x00]);
        lane0.extend_from_slice(&payload_lanes[0][..2048]);
        lane0.extend_from_slice(&[0x01, 0x01, 0x00, 0xFE, 0xFF]);
        lane0.push(payload_lanes[0][2048]);
        lanes[0].bytes = lane0;
    } else {
        let len = u16::try_from(tile_len).map_err(|_| GDeflateError::DataTooLarge)?;
        let nlen = !len;
        let mut lane0 = Vec::with_capacity(5 + payload_lanes[0].len());
        lane0.push(0x01);
        lane0.extend_from_slice(&len.to_le_bytes());
        lane0.extend_from_slice(&nlen.to_le_bytes());
        lane0.extend_from_slice(&payload_lanes[0]);
        lanes[0].bytes = lane0;
    }
    lanes[0].bit_pos = lanes[0].bytes.len().saturating_mul(8);

    Ok(serialize_lanes_words(&lanes))
}

fn read_u32_le_at(bytes: &[u8], byte_off: usize) -> Result<u32, GDeflateError> {
    let end = byte_off.checked_add(4).ok_or(GDeflateError::DataTooLarge)?;
    let raw = bytes
        .get(byte_off..end)
        .ok_or_else(|| GDeflateError::Gpu("u32 read out of bounds".to_string()))?;
    let mut tmp = [0_u8; 4];
    tmp.copy_from_slice(raw);
    Ok(u32::from_le_bytes(tmp))
}

fn parse_static_page_words_from_meta(
    bytes: &[u8],
    tile_count: usize,
) -> Result<Vec<u32>, GDeflateError> {
    let mut page_words = Vec::with_capacity(tile_count);
    for tile_index in 0..tile_count {
        let meta_index_bytes = tile_index
            .checked_mul(4)
            .ok_or(GDeflateError::DataTooLarge)?;
        let words = read_u32_le_at(bytes, meta_index_bytes)?;
        if words == u32::MAX {
            return Err(GDeflateError::Gpu(
                "serialized static page overflow marker received".to_string(),
            ));
        }
        let words_usize = usize::try_from(words).map_err(|_| GDeflateError::DataTooLarge)?;
        if words_usize == 0 || words_usize > STATIC_SERIALIZED_MAX_PAGE_WORDS {
            return Err(GDeflateError::Gpu(format!(
                "serialized static page word length out of range (tile_index={tile_index}, page_words={words}, max={})",
                STATIC_SERIALIZED_MAX_PAGE_WORDS
            )));
        }
        page_words.push(words);
    }
    Ok(page_words)
}

fn build_static_pages_from_compact_bytes(
    page_words: &[u32],
    bytes: &[u8],
) -> Result<Vec<Vec<u8>>, GDeflateError> {
    let mut pages = Vec::with_capacity(page_words.len());
    let mut off = 0usize;
    for (tile_index, &pw) in page_words.iter().enumerate() {
        let page_len = usize::try_from(pw)
            .map_err(|_| GDeflateError::DataTooLarge)?
            .checked_mul(4)
            .ok_or(GDeflateError::DataTooLarge)?;
        let end = off
            .checked_add(page_len)
            .ok_or(GDeflateError::DataTooLarge)?;
        let page = bytes.get(off..end).ok_or_else(|| {
            GDeflateError::Gpu(format!(
                "serialized static compact page out of bounds (tile_index={tile_index})"
            ))
        })?;
        pages.push(page.to_vec());
        off = end;
    }
    Ok(pages)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GpuEncodeMode {
    Stored,
    Static,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GpuDecodeMode {
    Static,
}

pub(super) fn max_submit_tiles_for_mode(mode: GpuEncodeMode) -> Result<usize, GDeflateError> {
    let runtime = runtime()?;
    let max_binding_bytes = runtime.device.limits().max_storage_buffer_binding_size as usize;
    let per_tile_bytes = match mode {
        GpuEncodeMode::Stored => STORED_OUTPUT_TILE_BYTES,
        GpuEncodeMode::Static => {
            // Static path uses separate storage bindings for lane output and LZ scratch.
            // Effective cap is bounded by the largest per-tile binding.
            let per_tile_interm_words = STATIC_INTERM_TILE_WORDS;
            let per_tile_interm_bytes = per_tile_interm_words.saturating_mul(4);
            let per_tile_scratch_bytes = STATIC_LZ_SCRATCH_WORDS_PER_TILE.saturating_mul(4);
            per_tile_interm_bytes.max(per_tile_scratch_bytes)
        }
    };
    let cap = max_binding_bytes / per_tile_bytes.max(1);
    Ok(cap.max(1))
}

pub(super) fn max_submit_tiles_for_decode_mode(
    mode: GpuDecodeMode,
) -> Result<usize, GDeflateError> {
    let runtime = runtime()?;
    let max_binding_bytes = runtime.device.limits().max_storage_buffer_binding_size as usize;
    let per_tile_bytes = match mode {
        // Decode output binding packs:
        // - status words (4 bytes/tile)
        // - produced words (4 bytes/tile)
        // - payload words (GPU_DECODE_TILE_WORDS * 4 bytes/tile)
        // Bind group uses the entire output buffer range, so cap must include all of them.
        GpuDecodeMode::Static => GPU_DECODE_TILE_WORDS.saturating_mul(4).saturating_add(8),
    };
    let cap = max_binding_bytes / per_tile_bytes.max(1);
    Ok(cap.max(1))
}

pub(super) struct PendingGpuEncodeBatch {
    mode: GpuEncodeMode,
    tile_lens: Vec<usize>,
    stored_readback_buffer: Option<wgpu::Buffer>,
    readback_len_bytes: usize,
    map_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    query_map_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    query_count: u32,
    timestamp_period_ns: f64,
    total_start: Instant,
    submit_start: Instant,
    upload_ms: f64,
    static_page_words: Option<Vec<u32>>,
    static_buffers: Option<StaticGpuBatchBuffers>,
}

pub(super) struct PendingGpuDecodeBatch {
    tile_expected_lens: Vec<usize>,
    readback_len_bytes: usize,
    readback_output_words_off: usize,
    decode_buffers: Option<DecodeGpuBatchBuffers>,
    map_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    query_map_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    query_count: u32,
    timestamp_period_ns: f64,
    total_start: Instant,
    submit_start: Instant,
    upload_ms: f64,
}

pub(super) fn submit_encode_tiles_gpu(
    mode: GpuEncodeMode,
    tiles: &[&[u8]],
) -> Result<PendingGpuEncodeBatch, GDeflateError> {
    match mode {
        GpuEncodeMode::Stored => submit_stored_tiles_gpu(tiles),
        GpuEncodeMode::Static => submit_static_tiles_gpu(tiles),
    }
}

pub(super) fn poll_encode_tiles_gpu(
    pending: &mut PendingGpuEncodeBatch,
    block: bool,
) -> Result<Option<(Vec<Vec<u8>>, GpuBatchStats)>, GDeflateError> {
    let runtime = runtime()?;
    runtime.device.poll(if block {
        wgpu::Maintain::Wait
    } else {
        wgpu::Maintain::Poll
    });

    let map_ready =
        if block {
            Some(pending.map_rx.recv().map_err(|_| {
                GDeflateError::Gpu("map_async completion channel closed".to_string())
            })?)
        } else {
            match pending.map_rx.try_recv() {
                Ok(v) => Some(v),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    return Err(GDeflateError::Gpu(
                        "map_async completion channel disconnected".to_string(),
                    ));
                }
            }
        };

    let Some(map_result) = map_ready else {
        return Ok(None);
    };

    if pending.mode == GpuEncodeMode::Static && pending.static_page_words.is_none() {
        map_result.map_err(|e| GDeflateError::Gpu(format!("map_async failed: {e}")))?;
        let static_buffers = pending
            .static_buffers
            .as_ref()
            .ok_or_else(|| GDeflateError::Gpu("missing static buffers".to_string()))?;
        let meta_slice = static_buffers.meta_readback_buffer.slice(
            ..u64::try_from(static_buffers.meta_readback_len_bytes)
                .map_err(|_| GDeflateError::DataTooLarge)?,
        );
        let meta_mapped = meta_slice.get_mapped_range();
        let page_words = parse_static_page_words_from_meta(&meta_mapped, pending.tile_lens.len())?;
        drop(meta_mapped);
        static_buffers.meta_readback_buffer.unmap();

        let mut payload_bytes = 0usize;
        for &w in &page_words {
            let add = usize::try_from(w)
                .map_err(|_| GDeflateError::DataTooLarge)?
                .checked_mul(4)
                .ok_or(GDeflateError::DataTooLarge)?;
            payload_bytes = payload_bytes
                .checked_add(add)
                .ok_or(GDeflateError::DataTooLarge)?;
        }
        if payload_bytes > static_buffers.readback_len_bytes {
            return Err(GDeflateError::Gpu(
                "compact static payload exceeds readback capacity".to_string(),
            ));
        }

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cozip-gdeflate-static-compact-copy-encoder"),
            });
        let mut dst_off = 0usize;
        for (tile_index, &words) in page_words.iter().enumerate() {
            let src_off_words = tile_index
                .checked_mul(STATIC_SERIALIZED_MAX_PAGE_WORDS)
                .ok_or(GDeflateError::DataTooLarge)?;
            let src_off = src_off_words
                .checked_mul(4)
                .ok_or(GDeflateError::DataTooLarge)?;
            let copy_bytes = usize::try_from(words)
                .map_err(|_| GDeflateError::DataTooLarge)?
                .checked_mul(4)
                .ok_or(GDeflateError::DataTooLarge)?;
            encoder.copy_buffer_to_buffer(
                &static_buffers.final_buffer,
                u64::try_from(src_off).map_err(|_| GDeflateError::DataTooLarge)?,
                &static_buffers.readback_buffer,
                u64::try_from(dst_off).map_err(|_| GDeflateError::DataTooLarge)?,
                u64::try_from(copy_bytes).map_err(|_| GDeflateError::DataTooLarge)?,
            );
            dst_off = dst_off
                .checked_add(copy_bytes)
                .ok_or(GDeflateError::DataTooLarge)?;
        }
        runtime.queue.submit(Some(encoder.finish()));
        let payload_slice = static_buffers
            .readback_buffer
            .slice(..u64::try_from(payload_bytes).map_err(|_| GDeflateError::DataTooLarge)?);
        let (tx, rx) = mpsc::channel();
        payload_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        pending.map_rx = rx;
        pending.readback_len_bytes = payload_bytes;
        pending.static_page_words = Some(page_words);
        return Ok(None);
    }

    let done = (|| -> Result<(Vec<Vec<u8>>, GpuBatchStats), GDeflateError> {
        map_result.map_err(|e| GDeflateError::Gpu(format!("map_async failed: {e}")))?;
        let submit_wait_ms = elapsed_ms(pending.submit_start);
        let readback_buffer = if let Some(static_buffers) = pending.static_buffers.as_ref() {
            &static_buffers.readback_buffer
        } else {
            pending
                .stored_readback_buffer
                .as_ref()
                .ok_or_else(|| GDeflateError::Gpu("missing stored readback buffer".to_string()))?
        };

        let mut static_hash_reset_ms = 0.0;
        let mut static_hash_build_ms = 0.0;
        let mut static_match_ms = 0.0;
        let mut static_scatter_ms = 0.0;
        let mut static_serialize_ms = 0.0;
        let mut static_copy_ms = 0.0;
        let mut static_profiled = false;
        if pending.mode == GpuEncodeMode::Static {
            if let Some(query_rx) = pending.query_map_rx.as_ref() {
                let query_buf = pending
                    .static_buffers
                    .as_ref()
                    .and_then(|b| b.query_readback_buffer.as_ref())
                    .ok_or_else(|| {
                        GDeflateError::Gpu(
                            "static query readback buffer missing while profiling enabled"
                                .to_string(),
                        )
                    })?;
                let query_result = query_rx.recv().map_err(|_| {
                    GDeflateError::Gpu("query map_async completion channel closed".to_string())
                })?;
                query_result
                    .map_err(|e| GDeflateError::Gpu(format!("query map_async failed: {e}")))?;
                let query_slice = query_buf.slice(..);
                let mapped = query_slice.get_mapped_range();
                let mut ts = Vec::with_capacity(pending.query_count as usize);
                for chunk in mapped.chunks_exact(8) {
                    let mut tmp = [0_u8; 8];
                    tmp.copy_from_slice(chunk);
                    ts.push(u64::from_le_bytes(tmp));
                }
                drop(mapped);
                query_buf.unmap();
                if ts.len() >= 7 {
                    let p = pending.timestamp_period_ns / 1_000_000.0;
                    let dt = |a: usize, b: usize| -> f64 {
                        if ts[b] >= ts[a] {
                            (ts[b] - ts[a]) as f64 * p
                        } else {
                            0.0
                        }
                    };
                    static_hash_reset_ms = dt(0, 1);
                    static_hash_build_ms = dt(1, 2);
                    static_match_ms = dt(2, 3);
                    static_scatter_ms = dt(3, 4);
                    static_serialize_ms = dt(4, 5);
                    static_copy_ms = dt(5, 6);
                    static_profiled = true;
                }
            }
        }

        let map_copy_start = Instant::now();
        let slice = readback_buffer.slice(
            ..u64::try_from(pending.readback_len_bytes).map_err(|_| GDeflateError::DataTooLarge)?,
        );
        let mapped = slice.get_mapped_range();
        let map_copy_ms = elapsed_ms(map_copy_start);

        let repack_start = Instant::now();
        let pages_res: Result<Vec<Vec<u8>>, GDeflateError> = match pending.mode {
            GpuEncodeMode::Stored => {
                let mut pages = Vec::with_capacity(pending.tile_lens.len());
                for (i, tile_len) in pending.tile_lens.iter().enumerate() {
                    let start = i
                        .checked_mul(STORED_OUTPUT_TILE_BYTES)
                        .ok_or(GDeflateError::DataTooLarge)?;
                    let end = start
                        .checked_add(STORED_OUTPUT_TILE_BYTES)
                        .ok_or(GDeflateError::DataTooLarge)?;
                    let tile_lane_block = mapped.get(start..end).ok_or_else(|| {
                        GDeflateError::Gpu("tile lane block out of bounds".to_string())
                    })?;
                    let page = build_stored_page_from_lane_block(*tile_len, tile_lane_block)?;
                    pages.push(page);
                }
                Ok(pages)
            }
            GpuEncodeMode::Static => {
                let page_words = pending.static_page_words.as_ref().ok_or_else(|| {
                    GDeflateError::Gpu("missing static compact page metadata".to_string())
                })?;
                build_static_pages_from_compact_bytes(page_words, &mapped)
            }
        };
        drop(mapped);
        readback_buffer.unmap();
        let pages = pages_res?;
        let repack_ms = elapsed_ms(repack_start);

        Ok((
            pages,
            GpuBatchStats {
                tiles: pending.tile_lens.len(),
                upload_ms: pending.upload_ms,
                submit_wait_ms,
                map_copy_ms,
                repack_ms,
                total_ms: elapsed_ms(pending.total_start),
                static_hash_reset_ms,
                static_hash_build_ms,
                static_match_ms,
                static_scatter_ms,
                static_serialize_ms,
                static_copy_ms,
                static_profiled,
            },
        ))
    })();

    if let Some(static_buffers) = pending.static_buffers.take() {
        return_static_batch_buffers(static_buffers);
    }

    done.map(Some)
}

pub(super) fn submit_decode_tiles_gpu(
    mode: GpuDecodeMode,
    pages: &[&[u8]],
    expected_lens: &[usize],
) -> Result<PendingGpuDecodeBatch, GDeflateError> {
    match mode {
        GpuDecodeMode::Static => submit_decode_static_tiles_gpu(pages, expected_lens),
    }
}

pub(super) fn poll_runtime_device(block: bool) -> Result<(), GDeflateError> {
    let runtime = runtime()?;
    runtime.device.poll(if block {
        wgpu::Maintain::Wait
    } else {
        wgpu::Maintain::Poll
    });
    Ok(())
}

pub(super) fn poll_decode_tiles_gpu_no_poll(
    pending: &mut PendingGpuDecodeBatch,
    block: bool,
) -> Result<Option<(Vec<Vec<u8>>, GpuDecodeBatchStats)>, GDeflateError> {
    poll_decode_tiles_gpu_impl(pending, block, false)
}

fn poll_decode_tiles_gpu_impl(
    pending: &mut PendingGpuDecodeBatch,
    block: bool,
    do_device_poll: bool,
) -> Result<Option<(Vec<Vec<u8>>, GpuDecodeBatchStats)>, GDeflateError> {
    let runtime = runtime()?;
    if do_device_poll {
        runtime.device.poll(if block {
            wgpu::Maintain::Wait
        } else {
            wgpu::Maintain::Poll
        });
    }

    let map_ready = if block {
        Some(pending.map_rx.recv().map_err(|_| {
            GDeflateError::Gpu("decode map_async completion channel closed".to_string())
        })?)
    } else {
        match pending.map_rx.try_recv() {
            Ok(v) => Some(v),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                return Err(GDeflateError::Gpu(
                    "decode map_async completion channel disconnected".to_string(),
                ));
            }
        }
    };
    let Some(map_result) = map_ready else {
        return Ok(None);
    };

    let done = (|| -> Result<(Vec<Vec<u8>>, GpuDecodeBatchStats), GDeflateError> {
        map_result.map_err(|e| GDeflateError::Gpu(format!("decode map_async failed: {e}")))?;
        let decode_buffers = pending
            .decode_buffers
            .as_ref()
            .ok_or_else(|| GDeflateError::Gpu("missing decode buffers".to_string()))?;
        let submit_wait_ms = elapsed_ms(pending.submit_start);
        let mut static_decode_ms = 0.0;
        let mut static_copy_ms = 0.0;
        let mut gpu_exec_ms = 0.0;
        let mut submit_overhead_ms = 0.0;
        let mut static_profiled = false;
        if let Some(query_rx) = pending.query_map_rx.as_ref() {
            let query_buf = decode_buffers
                .query_readback_buffer
                .as_ref()
                .ok_or_else(|| {
                    GDeflateError::Gpu(
                        "decode query readback buffer missing while profiling enabled".to_string(),
                    )
                })?;
            let query_result = query_rx.recv().map_err(|_| {
                GDeflateError::Gpu("decode query map_async completion channel closed".to_string())
            })?;
            query_result
                .map_err(|e| GDeflateError::Gpu(format!("decode query map_async failed: {e}")))?;
            let query_slice = query_buf.slice(..);
            let mapped = query_slice.get_mapped_range();
            let mut ts = Vec::with_capacity(pending.query_count as usize);
            for chunk in mapped.chunks_exact(8) {
                let mut tmp = [0_u8; 8];
                tmp.copy_from_slice(chunk);
                ts.push(u64::from_le_bytes(tmp));
            }
            drop(mapped);
            query_buf.unmap();
            if ts.len() >= 3 {
                let p = pending.timestamp_period_ns / 1_000_000.0;
                let dt = |a: usize, b: usize| -> f64 {
                    if ts[b] >= ts[a] {
                        (ts[b] - ts[a]) as f64 * p
                    } else {
                        0.0
                    }
                };
                static_decode_ms = dt(0, 1);
                static_copy_ms = dt(1, 2);
                gpu_exec_ms = dt(0, 2);
                submit_overhead_ms = (submit_wait_ms - gpu_exec_ms).max(0.0);
                static_profiled = true;
            }
        }
        let map_copy_start = Instant::now();
        let slice = decode_buffers.readback_buffer.slice(
            ..u64::try_from(pending.readback_len_bytes).map_err(|_| GDeflateError::DataTooLarge)?,
        );
        let mapped = slice.get_mapped_range();
        let map_copy_ms = elapsed_ms(map_copy_start);

        let tile_count = pending.tile_expected_lens.len();
        let status_words_end = tile_count
            .checked_mul(4)
            .ok_or(GDeflateError::DataTooLarge)?;
        let produced_words_end = status_words_end
            .checked_add(
                tile_count
                    .checked_mul(4)
                    .ok_or(GDeflateError::DataTooLarge)?,
            )
            .ok_or(GDeflateError::DataTooLarge)?;
        let output_words_off = pending.readback_output_words_off;
        let output_words_end = output_words_off
            .checked_add(
                tile_count
                    .checked_mul(GPU_DECODE_TILE_WORDS)
                    .and_then(|x| x.checked_mul(4))
                    .ok_or(GDeflateError::DataTooLarge)?,
            )
            .ok_or(GDeflateError::DataTooLarge)?;
        if mapped.len() < output_words_end {
            drop(mapped);
            decode_buffers.readback_buffer.unmap();
            return Err(GDeflateError::Gpu(
                "decode readback buffer shorter than expected".to_string(),
            ));
        }

        let status_bytes = &mapped[..status_words_end];
        let produced_bytes = &mapped[status_words_end..produced_words_end];
        let output_bytes = &mapped[output_words_off..output_words_end];

        let mut tiles = Vec::with_capacity(tile_count);
        for tile_idx in 0..tile_count {
            let st_off = tile_idx.checked_mul(4).ok_or(GDeflateError::DataTooLarge)?;
            let status = u32::from_le_bytes(
                status_bytes[st_off..st_off + 4]
                    .try_into()
                    .map_err(|_| GDeflateError::Gpu("decode status parse failed".to_string()))?,
            );
            let produced = u32::from_le_bytes(
                produced_bytes[st_off..st_off + 4]
                    .try_into()
                    .map_err(|_| GDeflateError::Gpu("decode produced parse failed".to_string()))?,
            ) as usize;
            let expected = pending.tile_expected_lens[tile_idx];
            if status != 0 {
                drop(mapped);
                decode_buffers.readback_buffer.unmap();
                return Err(GDeflateError::Gpu(format!(
                    "static gpu decode failed: tile={tile_idx} status={status}"
                )));
            }
            if produced != expected {
                drop(mapped);
                decode_buffers.readback_buffer.unmap();
                return Err(GDeflateError::Gpu(format!(
                    "static gpu decode length mismatch: tile={tile_idx} produced={produced} expected={expected}"
                )));
            }
            let tile_out_words_off = tile_idx
                .checked_mul(GPU_DECODE_TILE_WORDS)
                .and_then(|x| x.checked_mul(4))
                .ok_or(GDeflateError::DataTooLarge)?;
            let tile_out_words = output_bytes
                .get(tile_out_words_off..tile_out_words_off + (expected * 4))
                .ok_or_else(|| {
                    GDeflateError::Gpu("decode output slice out of bounds".to_string())
                })?;
            let mut tile = Vec::with_capacity(expected);
            for wchunk in tile_out_words.chunks_exact(4) {
                let w =
                    u32::from_le_bytes(wchunk.try_into().map_err(|_| {
                        GDeflateError::Gpu("decode output parse failed".to_string())
                    })?);
                tile.push((w & 0xff) as u8);
            }
            tiles.push(tile);
        }

        drop(mapped);
        decode_buffers.readback_buffer.unmap();
        Ok((
            tiles,
            GpuDecodeBatchStats {
                tiles: tile_count,
                upload_ms: pending.upload_ms,
                submit_wait_ms,
                map_copy_ms,
                total_ms: elapsed_ms(pending.total_start),
                static_decode_ms,
                static_copy_ms,
                gpu_exec_ms,
                submit_overhead_ms,
                static_profiled,
            },
        ))
    })();

    if let Some(decode_buffers) = pending.decode_buffers.take() {
        return_decode_batch_buffers(decode_buffers);
    }

    done.map(Some)
}

fn submit_stored_tiles_gpu(tiles: &[&[u8]]) -> Result<PendingGpuEncodeBatch, GDeflateError> {
    let total_start = Instant::now();
    if tiles.is_empty() {
        return Err(GDeflateError::Gpu(
            "submit_stored_tiles_gpu called with empty tiles".to_string(),
        ));
    }
    let runtime = runtime()?;
    let batch_tiles = tiles.len();

    let input_bytes_len = batch_tiles
        .checked_mul(GDEFLATE_TILE_SIZE)
        .ok_or(GDeflateError::DataTooLarge)?;
    let mut input_bytes = vec![0_u8; input_bytes_len];
    let mut tile_lens = Vec::with_capacity(batch_tiles);
    let mut lens_words = Vec::with_capacity(batch_tiles);
    let mut static_tile_inputs = Vec::with_capacity(batch_tiles);
    for (i, tile) in tiles.iter().enumerate() {
        let dst = i
            .checked_mul(GDEFLATE_TILE_SIZE)
            .ok_or(GDeflateError::DataTooLarge)?;
        input_bytes[dst..dst + tile.len()].copy_from_slice(tile);
        tile_lens.push(tile.len());
        lens_words.push(u32::try_from(tile.len()).map_err(|_| GDeflateError::DataTooLarge)?);
        static_tile_inputs.push((*tile).to_vec());
    }
    let input_words = pack_bytes_to_words(&input_bytes);

    let output_word_count = batch_tiles
        .checked_mul(STORED_OUTPUT_TILE_BYTES)
        .ok_or(GDeflateError::DataTooLarge)?
        .div_ceil(4);
    let output_bytes_size = output_word_count
        .checked_mul(4)
        .ok_or(GDeflateError::DataTooLarge)?;

    let input_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-input"),
        size: u64::try_from(input_words.len().saturating_mul(4))
            .map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let lens_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-lens"),
        size: u64::try_from(lens_words.len().saturating_mul(4))
            .map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-output"),
        size: u64::try_from(output_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let scratch_dummy_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-scratch-dummy"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-readback"),
        size: u64::try_from(output_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let upload_start = Instant::now();
    runtime
        .queue
        .write_buffer(&input_buffer, 0, &words_to_bytes(&input_words));
    runtime
        .queue
        .write_buffer(&lens_buffer, 0, &words_to_bytes(&lens_words));
    let upload_ms = elapsed_ms(upload_start);

    let bind_group = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-gdeflate-stored-bg"),
            layout: &runtime.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch_dummy_buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-gdeflate-stored-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-gdeflate-stored-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.stored_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let words_u32 =
            u32::try_from(output_word_count).map_err(|_| GDeflateError::DataTooLarge)?;
        let groups = words_u32.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &readback_buffer,
        0,
        u64::try_from(output_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
    );

    let submit_start = Instant::now();
    runtime.queue.submit(Some(encoder.finish()));
    let slice = readback_buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    Ok(PendingGpuEncodeBatch {
        mode: GpuEncodeMode::Stored,
        tile_lens,
        stored_readback_buffer: Some(readback_buffer),
        readback_len_bytes: output_bytes_size,
        map_rx: rx,
        query_map_rx: None,
        query_count: 0,
        timestamp_period_ns: runtime.timestamp_period_ns,
        total_start,
        submit_start,
        upload_ms,
        static_page_words: None,
        static_buffers: None,
    })
}

struct StaticGpuBatchBuffers {
    batch_tiles: usize,
    readback_len_bytes: usize,
    meta_readback_len_bytes: usize,
    input_buffer: wgpu::Buffer,
    lens_buffer: wgpu::Buffer,
    final_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    meta_readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    serialize_bind_group: wgpu::BindGroup,
    query_set: Option<wgpu::QuerySet>,
    query_resolve_buffer: Option<wgpu::Buffer>,
    query_readback_buffer: Option<wgpu::Buffer>,
    query_count: u32,
}

fn static_batch_pool() -> &'static Mutex<Vec<StaticGpuBatchBuffers>> {
    static STATIC_BATCH_POOL: OnceLock<Mutex<Vec<StaticGpuBatchBuffers>>> = OnceLock::new();
    STATIC_BATCH_POOL.get_or_init(|| Mutex::new(Vec::new()))
}

fn acquire_static_batch_buffers(
    runtime: &StoredGpuRuntime,
    batch_tiles: usize,
) -> Result<StaticGpuBatchBuffers, GDeflateError> {
    let mut pool = static_batch_pool()
        .lock()
        .map_err(|_| GDeflateError::Gpu("static batch pool lock poisoned".to_string()))?;
    if let Some(pos) = pool
        .iter()
        .position(|entry| entry.batch_tiles == batch_tiles)
    {
        return Ok(pool.swap_remove(pos));
    }
    drop(pool);
    create_static_batch_buffers(runtime, batch_tiles)
}

fn return_static_batch_buffers(buffers: StaticGpuBatchBuffers) {
    if let Ok(mut pool) = static_batch_pool().lock() {
        pool.push(buffers);
    }
}

fn create_static_batch_buffers(
    runtime: &StoredGpuRuntime,
    batch_tiles: usize,
) -> Result<StaticGpuBatchBuffers, GDeflateError> {
    let input_word_count = batch_tiles
        .checked_mul(GDEFLATE_TILE_SIZE)
        .ok_or(GDeflateError::DataTooLarge)?
        .div_ceil(4);
    let input_bytes_size = input_word_count
        .checked_mul(4)
        .ok_or(GDeflateError::DataTooLarge)?;
    let lens_bytes_size = batch_tiles
        .checked_mul(4)
        .ok_or(GDeflateError::DataTooLarge)?;
    let interm_word_count = batch_tiles
        .checked_mul(STATIC_INTERM_TILE_WORDS)
        .ok_or(GDeflateError::DataTooLarge)?;
    let interm_bytes_size = interm_word_count
        .checked_mul(4)
        .ok_or(GDeflateError::DataTooLarge)?;
    let scratch_word_count = batch_tiles
        .checked_mul(STATIC_LZ_SCRATCH_WORDS_PER_TILE)
        .ok_or(GDeflateError::DataTooLarge)?;
    let scratch_bytes_size = scratch_word_count
        .checked_mul(4)
        .ok_or(GDeflateError::DataTooLarge)?;
    let final_word_count = batch_tiles
        .checked_mul(STATIC_SERIALIZED_MAX_PAGE_WORDS)
        .and_then(|x| x.checked_add(batch_tiles))
        .ok_or(GDeflateError::DataTooLarge)?;
    let final_bytes_size = final_word_count
        .checked_mul(4)
        .ok_or(GDeflateError::DataTooLarge)?;
    let meta_readback_len_bytes = batch_tiles
        .checked_mul(4)
        .ok_or(GDeflateError::DataTooLarge)?;

    let input_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-static-input"),
        size: u64::try_from(input_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let lens_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-static-lens"),
        size: u64::try_from(lens_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let interm_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-static-output"),
        size: u64::try_from(interm_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let scratch_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-static-scratch"),
        size: u64::try_from(scratch_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let final_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-static-final"),
        size: u64::try_from(final_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-static-readback"),
        size: u64::try_from(final_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let meta_readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-gpu-static-meta-readback"),
        size: u64::try_from(meta_readback_len_bytes).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-gdeflate-static-scatter-bg"),
            layout: &runtime.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: interm_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch_buffer.as_entire_binding(),
                },
            ],
        });
    let serialize_bind_group = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-gdeflate-static-serialize-bg"),
            layout: &runtime.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: interm_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lens_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: final_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch_buffer.as_entire_binding(),
                },
            ],
        });

    let timestamp_query_count: u32 = 7;
    let mut query_set: Option<wgpu::QuerySet> = None;
    let mut query_resolve_buffer: Option<wgpu::Buffer> = None;
    let mut query_readback_buffer: Option<wgpu::Buffer> = None;
    let mut query_count_effective = 0;
    if runtime.timestamp_query_supported && runtime.static_timestamp_profile_enabled {
        query_set = Some(runtime.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("cozip-gdeflate-static-timestamps"),
            count: timestamp_query_count,
            ty: wgpu::QueryType::Timestamp,
        }));
        let query_resolve_size = usize::try_from(timestamp_query_count)
            .map_err(|_| GDeflateError::DataTooLarge)?
            .saturating_mul(8);
        query_resolve_buffer = Some(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-gdeflate-static-ts-resolve"),
            size: u64::try_from(query_resolve_size).map_err(|_| GDeflateError::DataTooLarge)?,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        query_readback_buffer = Some(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-gdeflate-static-ts-readback"),
            size: u64::try_from(query_resolve_size).map_err(|_| GDeflateError::DataTooLarge)?,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        query_count_effective = timestamp_query_count;
    }

    Ok(StaticGpuBatchBuffers {
        batch_tiles,
        readback_len_bytes: final_bytes_size,
        meta_readback_len_bytes,
        input_buffer,
        lens_buffer,
        final_buffer,
        readback_buffer,
        meta_readback_buffer,
        bind_group,
        serialize_bind_group,
        query_set,
        query_resolve_buffer,
        query_readback_buffer,
        query_count: query_count_effective,
    })
}

fn submit_static_tiles_gpu(tiles: &[&[u8]]) -> Result<PendingGpuEncodeBatch, GDeflateError> {
    let total_start = Instant::now();
    if tiles.is_empty() {
        return Err(GDeflateError::Gpu(
            "submit_static_tiles_gpu called with empty tiles".to_string(),
        ));
    }
    let runtime = runtime()?;
    let batch_tiles = tiles.len();

    let input_bytes_len = batch_tiles
        .checked_mul(GDEFLATE_TILE_SIZE)
        .ok_or(GDeflateError::DataTooLarge)?;
    let mut input_bytes = vec![0_u8; input_bytes_len];
    let mut tile_lens = Vec::with_capacity(batch_tiles);
    let mut lens_words = Vec::with_capacity(batch_tiles);
    for (i, tile) in tiles.iter().enumerate() {
        let dst = i
            .checked_mul(GDEFLATE_TILE_SIZE)
            .ok_or(GDeflateError::DataTooLarge)?;
        input_bytes[dst..dst + tile.len()].copy_from_slice(tile);
        tile_lens.push(tile.len());
        lens_words.push(u32::try_from(tile.len()).map_err(|_| GDeflateError::DataTooLarge)?);
    }
    let input_words = pack_bytes_to_words(&input_bytes);

    let static_buffers = acquire_static_batch_buffers(runtime, batch_tiles)?;

    let upload_start = Instant::now();
    runtime.queue.write_buffer(
        &static_buffers.input_buffer,
        0,
        &words_to_bytes(&input_words),
    );
    runtime
        .queue
        .write_buffer(&static_buffers.lens_buffer, 0, &words_to_bytes(&lens_words));
    let upload_ms = elapsed_ms(upload_start);

    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-gdeflate-static-encoder"),
        });
    let mut query_map_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> = None;
    if let Some(qs) = static_buffers.query_set.as_ref() {
        encoder.write_timestamp(qs, 0);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-gdeflate-static-hash-reset-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.static_hash_reset_pipeline);
        pass.set_bind_group(0, &static_buffers.bind_group, &[]);
        let heads = u32::try_from(batch_tiles.saturating_mul(STATIC_LZ_HASH_SIZE))
            .map_err(|_| GDeflateError::DataTooLarge)?;
        let groups = heads.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
    if let Some(qs) = static_buffers.query_set.as_ref() {
        encoder.write_timestamp(qs, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-gdeflate-static-hash-build-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.static_hash_build_pipeline);
        pass.set_bind_group(0, &static_buffers.bind_group, &[]);
        let positions = u32::try_from(batch_tiles.saturating_mul(GDEFLATE_TILE_SIZE))
            .map_err(|_| GDeflateError::DataTooLarge)?;
        let groups = positions.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
    if let Some(qs) = static_buffers.query_set.as_ref() {
        encoder.write_timestamp(qs, 2);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-gdeflate-static-match-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.static_match_pipeline);
        pass.set_bind_group(0, &static_buffers.bind_group, &[]);
        let positions = u32::try_from(batch_tiles.saturating_mul(GDEFLATE_TILE_SIZE))
            .map_err(|_| GDeflateError::DataTooLarge)?;
        let groups = positions.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
    if let Some(qs) = static_buffers.query_set.as_ref() {
        encoder.write_timestamp(qs, 3);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-gdeflate-static-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.static_pipeline);
        pass.set_bind_group(0, &static_buffers.bind_group, &[]);
        let block_count_u32 = u32::try_from(batch_tiles.saturating_mul(STATIC_SUBBLOCKS_PER_TILE))
            .map_err(|_| GDeflateError::DataTooLarge)?;
        let groups = block_count_u32.div_ceil(64);
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-gdeflate-static-reduce-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.static_reduce_pipeline);
        pass.set_bind_group(0, &static_buffers.bind_group, &[]);
        let tile_count_u32 = u32::try_from(batch_tiles).map_err(|_| GDeflateError::DataTooLarge)?;
        let groups = tile_count_u32.div_ceil(64);
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
    if let Some(qs) = static_buffers.query_set.as_ref() {
        encoder.write_timestamp(qs, 4);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-gdeflate-static-serialize-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.static_serialize_pipeline);
        pass.set_bind_group(0, &static_buffers.serialize_bind_group, &[]);
        let tile_count_u32 = u32::try_from(batch_tiles).map_err(|_| GDeflateError::DataTooLarge)?;
        let lane_tasks = tile_count_u32
            .checked_mul(
                u32::try_from(GDEFLATE_NUM_STREAMS).map_err(|_| GDeflateError::DataTooLarge)?,
            )
            .ok_or(GDeflateError::DataTooLarge)?;
        let groups = lane_tasks.div_ceil(128);
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
    if let Some(qs) = static_buffers.query_set.as_ref() {
        encoder.write_timestamp(qs, 5);
    }
    let meta_src_off = batch_tiles
        .checked_mul(STATIC_SERIALIZED_MAX_PAGE_WORDS)
        .and_then(|x| x.checked_mul(4))
        .ok_or(GDeflateError::DataTooLarge)?;
    encoder.copy_buffer_to_buffer(
        &static_buffers.final_buffer,
        u64::try_from(meta_src_off).map_err(|_| GDeflateError::DataTooLarge)?,
        &static_buffers.meta_readback_buffer,
        0,
        u64::try_from(static_buffers.meta_readback_len_bytes)
            .map_err(|_| GDeflateError::DataTooLarge)?,
    );
    if let Some(qs) = static_buffers.query_set.as_ref() {
        encoder.write_timestamp(qs, 6);
    }
    if let (Some(qs), Some(q_resolve), Some(q_readback)) = (
        static_buffers.query_set.as_ref(),
        static_buffers.query_resolve_buffer.as_ref(),
        static_buffers.query_readback_buffer.as_ref(),
    ) {
        let query_resolve_size = usize::try_from(static_buffers.query_count)
            .map_err(|_| GDeflateError::DataTooLarge)?
            .saturating_mul(8);
        encoder.resolve_query_set(qs, 0..static_buffers.query_count, q_resolve, 0);
        encoder.copy_buffer_to_buffer(
            q_resolve,
            0,
            q_readback,
            0,
            u64::try_from(query_resolve_size).map_err(|_| GDeflateError::DataTooLarge)?,
        );
    }

    let submit_start = Instant::now();
    runtime.queue.submit(Some(encoder.finish()));
    let slice = static_buffers.meta_readback_buffer.slice(
        ..u64::try_from(static_buffers.meta_readback_len_bytes)
            .map_err(|_| GDeflateError::DataTooLarge)?,
    );
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    if let Some(query_rb) = static_buffers.query_readback_buffer.as_ref() {
        let query_slice = query_rb.slice(..);
        let (qtx, qrx) = mpsc::channel();
        query_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = qtx.send(result);
        });
        query_map_rx = Some(qrx);
    }

    Ok(PendingGpuEncodeBatch {
        mode: GpuEncodeMode::Static,
        tile_lens,
        stored_readback_buffer: None,
        readback_len_bytes: static_buffers.readback_len_bytes,
        map_rx: rx,
        query_map_rx,
        query_count: static_buffers.query_count,
        timestamp_period_ns: runtime.timestamp_period_ns,
        total_start,
        submit_start,
        upload_ms,
        static_page_words: None,
        static_buffers: Some(static_buffers),
    })
}

struct DecodeGpuBatchBuffers {
    batch_tiles: usize,
    input_buffer_len_bytes: usize,
    meta_buffer_len_bytes: usize,
    output_buffer_len_bytes: usize,
    input_buffer: wgpu::Buffer,
    meta_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    query_set: Option<wgpu::QuerySet>,
    query_resolve_buffer: Option<wgpu::Buffer>,
    query_readback_buffer: Option<wgpu::Buffer>,
    query_count: u32,
}

fn decode_batch_pool() -> &'static Mutex<Vec<DecodeGpuBatchBuffers>> {
    static DECODE_BATCH_POOL: OnceLock<Mutex<Vec<DecodeGpuBatchBuffers>>> = OnceLock::new();
    DECODE_BATCH_POOL.get_or_init(|| Mutex::new(Vec::new()))
}

fn acquire_decode_batch_buffers(
    runtime: &StoredGpuRuntime,
    batch_tiles: usize,
    input_bytes_size: usize,
    meta_bytes_size: usize,
    output_buffer_bytes_size: usize,
) -> Result<DecodeGpuBatchBuffers, GDeflateError> {
    let mut pool = decode_batch_pool()
        .lock()
        .map_err(|_| GDeflateError::Gpu("decode batch pool lock poisoned".to_string()))?;
    if let Some(pos) = pool.iter().position(|entry| {
        entry.batch_tiles == batch_tiles
            && entry.input_buffer_len_bytes >= input_bytes_size
            && entry.meta_buffer_len_bytes >= meta_bytes_size
            && entry.output_buffer_len_bytes >= output_buffer_bytes_size
    }) {
        return Ok(pool.swap_remove(pos));
    }
    drop(pool);
    create_decode_batch_buffers(
        runtime,
        batch_tiles,
        input_bytes_size,
        meta_bytes_size,
        output_buffer_bytes_size,
    )
}

fn return_decode_batch_buffers(buffers: DecodeGpuBatchBuffers) {
    if let Ok(mut pool) = decode_batch_pool().lock() {
        pool.push(buffers);
    }
}

fn create_decode_batch_buffers(
    runtime: &StoredGpuRuntime,
    batch_tiles: usize,
    input_bytes_size: usize,
    meta_bytes_size: usize,
    output_buffer_bytes_size: usize,
) -> Result<DecodeGpuBatchBuffers, GDeflateError> {
    let input_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-decode-input"),
        size: u64::try_from(input_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let meta_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-decode-meta"),
        size: u64::try_from(meta_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-decode-output"),
        size: u64::try_from(output_buffer_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cozip-gdeflate-decode-readback"),
        size: u64::try_from(output_buffer_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cozip-gdeflate-decode-bg"),
            layout: &runtime.decode_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

    let decode_timestamp_query_count: u32 = 3;
    let mut query_set: Option<wgpu::QuerySet> = None;
    let mut query_resolve_buffer: Option<wgpu::Buffer> = None;
    let mut query_readback_buffer: Option<wgpu::Buffer> = None;
    let mut query_count_effective = 0;
    if runtime.timestamp_query_supported && runtime.decode_timestamp_profile_enabled {
        query_set = Some(runtime.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("cozip-gdeflate-decode-timestamps"),
            count: decode_timestamp_query_count,
            ty: wgpu::QueryType::Timestamp,
        }));
        let query_resolve_size = usize::try_from(decode_timestamp_query_count)
            .map_err(|_| GDeflateError::DataTooLarge)?
            .saturating_mul(8);
        query_resolve_buffer = Some(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-gdeflate-decode-ts-resolve"),
            size: u64::try_from(query_resolve_size).map_err(|_| GDeflateError::DataTooLarge)?,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        query_readback_buffer = Some(runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cozip-gdeflate-decode-ts-readback"),
            size: u64::try_from(query_resolve_size).map_err(|_| GDeflateError::DataTooLarge)?,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        query_count_effective = decode_timestamp_query_count;
    }

    Ok(DecodeGpuBatchBuffers {
        batch_tiles,
        input_buffer_len_bytes: input_bytes_size,
        meta_buffer_len_bytes: meta_bytes_size,
        output_buffer_len_bytes: output_buffer_bytes_size,
        input_buffer,
        meta_buffer,
        output_buffer,
        readback_buffer,
        bind_group,
        query_set,
        query_resolve_buffer,
        query_readback_buffer,
        query_count: query_count_effective,
    })
}

fn submit_decode_static_tiles_gpu(
    pages: &[&[u8]],
    expected_lens: &[usize],
) -> Result<PendingGpuDecodeBatch, GDeflateError> {
    let total_start = Instant::now();
    if pages.is_empty() {
        return Err(GDeflateError::Gpu(
            "submit_decode_static_tiles_gpu called with empty pages".to_string(),
        ));
    }
    if pages.len() != expected_lens.len() {
        return Err(GDeflateError::Gpu(
            "decode page/expected length count mismatch".to_string(),
        ));
    }
    let runtime = runtime()?;
    let batch_tiles = pages.len();

    let mut meta_words = Vec::with_capacity(batch_tiles.saturating_mul(3));
    let total_input_bytes = pages
        .iter()
        .try_fold(0usize, |acc, page| acc.checked_add(page.len()))
        .ok_or(GDeflateError::DataTooLarge)?;
    let mut input_bytes = Vec::with_capacity(total_input_bytes);
    let mut cur_off = 0usize;
    for (page, &expected_len) in pages.iter().zip(expected_lens.iter()) {
        if expected_len > GDEFLATE_TILE_SIZE {
            return Err(GDeflateError::InvalidOptions(
                "decode expected length exceeds tile size",
            ));
        }
        meta_words.push(u32::try_from(cur_off).map_err(|_| GDeflateError::DataTooLarge)?);
        meta_words.push(u32::try_from(page.len()).map_err(|_| GDeflateError::DataTooLarge)?);
        meta_words.push(u32::try_from(expected_len).map_err(|_| GDeflateError::DataTooLarge)?);
        input_bytes.extend_from_slice(page);
        cur_off = cur_off
            .checked_add(page.len())
            .ok_or(GDeflateError::DataTooLarge)?;
    }
    let input_words = pack_bytes_to_words(&input_bytes);
    let input_bytes_padded = words_to_bytes(&input_words);

    let meta_bytes_size = meta_words
        .len()
        .checked_mul(4)
        .ok_or(GDeflateError::DataTooLarge)?;
    let output_words_count = batch_tiles
        .checked_mul(GPU_DECODE_TILE_WORDS)
        .ok_or(GDeflateError::DataTooLarge)?;
    let output_payload_bytes_size = output_words_count
        .checked_mul(4)
        .ok_or(GDeflateError::DataTooLarge)?;
    let status_bytes_size = batch_tiles
        .checked_mul(4)
        .ok_or(GDeflateError::DataTooLarge)?;
    let produced_bytes_size = status_bytes_size;
    let output_words_off = status_bytes_size
        .checked_add(produced_bytes_size)
        .ok_or(GDeflateError::DataTooLarge)?;
    let output_buffer_bytes_size = output_words_off
        .checked_add(output_payload_bytes_size)
        .ok_or(GDeflateError::DataTooLarge)?;

    let decode_buffers = acquire_decode_batch_buffers(
        runtime,
        batch_tiles,
        input_bytes_padded.len(),
        meta_bytes_size,
        output_buffer_bytes_size,
    )?;

    let upload_start = Instant::now();
    runtime
        .queue
        .write_buffer(&decode_buffers.input_buffer, 0, &input_bytes_padded);
    runtime
        .queue
        .write_buffer(&decode_buffers.meta_buffer, 0, &words_to_bytes(&meta_words));
    let upload_ms = elapsed_ms(upload_start);

    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cozip-gdeflate-decode-encoder"),
        });
    if let Some(qs) = decode_buffers.query_set.as_ref() {
        encoder.write_timestamp(qs, 0);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cozip-gdeflate-decode-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&runtime.static_decode_pipeline);
        pass.set_bind_group(0, &decode_buffers.bind_group, &[]);
        let groups = u32::try_from(batch_tiles).map_err(|_| GDeflateError::DataTooLarge)?;
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
    if let Some(qs) = decode_buffers.query_set.as_ref() {
        encoder.write_timestamp(qs, 1);
    }
    encoder.copy_buffer_to_buffer(
        &decode_buffers.output_buffer,
        0,
        &decode_buffers.readback_buffer,
        0,
        u64::try_from(output_buffer_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?,
    );
    if let Some(qs) = decode_buffers.query_set.as_ref() {
        encoder.write_timestamp(qs, 2);
    }
    if let (Some(qs), Some(q_resolve), Some(q_readback)) = (
        decode_buffers.query_set.as_ref(),
        decode_buffers.query_resolve_buffer.as_ref(),
        decode_buffers.query_readback_buffer.as_ref(),
    ) {
        let query_resolve_size = usize::try_from(decode_buffers.query_count)
            .map_err(|_| GDeflateError::DataTooLarge)?
            .saturating_mul(8);
        encoder.resolve_query_set(qs, 0..decode_buffers.query_count, q_resolve, 0);
        encoder.copy_buffer_to_buffer(
            q_resolve,
            0,
            q_readback,
            0,
            u64::try_from(query_resolve_size).map_err(|_| GDeflateError::DataTooLarge)?,
        );
    }

    let submit_start = Instant::now();
    runtime.queue.submit(Some(encoder.finish()));
    let slice = decode_buffers
        .readback_buffer
        .slice(..u64::try_from(output_buffer_bytes_size).map_err(|_| GDeflateError::DataTooLarge)?);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let mut query_map_rx: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> = None;
    if let Some(query_rb) = decode_buffers.query_readback_buffer.as_ref() {
        let query_slice = query_rb.slice(..);
        let (qtx, qrx) = mpsc::channel();
        query_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = qtx.send(result);
        });
        query_map_rx = Some(qrx);
    }
    let query_count = decode_buffers.query_count;

    Ok(PendingGpuDecodeBatch {
        tile_expected_lens: expected_lens.to_vec(),
        readback_len_bytes: output_buffer_bytes_size,
        readback_output_words_off: output_words_off,
        decode_buffers: Some(decode_buffers),
        map_rx: rx,
        query_map_rx,
        query_count,
        timestamp_period_ns: runtime.timestamp_period_ns,
        total_start,
        submit_start,
        upload_ms,
    })
}
