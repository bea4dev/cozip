const LITERAL_TAG: u32 = 0x0fffu;
const MAX_CMD_LEN: u32 = 270u;
const TABLE_REPEAT_STRIDE: u32 = 270u;
const HUFF_LUT_HEADER_SIZE: u32 = 12u;
const META_HEADER_WORDS: u32 = 6u;
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

fn load_table_u16(idx: u32) -> u32 {
    let b0 = load_table_u8(idx);
    let b1 = load_table_u8(idx + 1u);
    return b0 | (b1 << 8u);
}

fn load_table_u32(idx: u32) -> u32 {
    if ((idx & 3u) == 0u) {
        return table_words[idx >> 2u];
    }
    let b0 = load_table_u8(idx);
    let b1 = load_table_u8(idx + 1u);
    let b2 = load_table_u8(idx + 2u);
    let b3 = load_table_u8(idx + 3u);
    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
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

fn read_cmd_bit(bit_pos: u32) -> u32 {
    let byte = load_cmd_u8(bit_pos >> 3u);
    let shift = bit_pos & 7u;
    return (byte >> shift) & 1u;
}

fn peek_cmd_bits(bit_cursor: u32, bit_end: u32, bit_len: u32) -> u32 {
    if (bit_len == 0u || bit_cursor >= bit_end) {
        return 0u;
    }
    let available = bit_end - bit_cursor;
    if (bit_len > available) {
        var out = 0u;
        var i = 0u;
        loop {
            if (i >= bit_len) {
                break;
            }
            let p = bit_cursor + i;
            if (p >= bit_end) {
                break;
            }
            out = out | (read_cmd_bit(p) << i);
            i = i + 1u;
        }
        return out;
    }

    let byte_idx = bit_cursor >> 3u;
    let bit_shift = bit_cursor & 7u;
    let word_idx = byte_idx >> 2u;
    let word_shift = ((byte_idx & 3u) << 3u) + bit_shift;
    var out = cmd_words[word_idx] >> word_shift;
    if (word_shift + bit_len > 32u) {
        out = out | (cmd_words[word_idx + 1u] << (32u - word_shift));
    }
    if (bit_len >= 32u) {
        return out;
    }
    return out & ((1u << bit_len) - 1u);
}

fn copy_table_repeat_to_output(table_src_base: u32, out_dst_base: u32, len: u32) {
    var src = table_src_base;
    var dst = out_dst_base;
    var remaining = len;

    loop {
        if (remaining == 0u || ((src | dst) & 3u) == 0u) {
            break;
        }
        store_out_u8(dst, load_table_u8(src));
        src = src + 1u;
        dst = dst + 1u;
        remaining = remaining - 1u;
    }

    loop {
        if (remaining < 12u) {
            break;
        }
        let src_word = src >> 2u;
        let dst_word = dst >> 2u;
        out_words[dst_word] = table_words[src_word];
        out_words[dst_word + 1u] = table_words[src_word + 1u];
        out_words[dst_word + 2u] = table_words[src_word + 2u];
        src = src + 12u;
        dst = dst + 12u;
        remaining = remaining - 12u;
    }

    loop {
        if (remaining < 4u) {
            break;
        }
        out_words[dst >> 2u] = table_words[src >> 2u];
        src = src + 4u;
        dst = dst + 4u;
        remaining = remaining - 4u;
    }

    loop {
        if (remaining == 0u) {
            break;
        }
        store_out_u8(dst, load_table_u8(src));
        src = src + 1u;
        dst = dst + 1u;
        remaining = remaining - 1u;
    }
}

struct DecodedSymbol {
    ok: u32,
    symbol: u32,
    next_bit: u32,
    err: u32,
};

fn decode_huffman_symbol(
    bit_cursor: u32,
    bit_end: u32,
    huff_lut_off: u32,
    huff_lut_len: u32,
) -> DecodedSymbol {
    if (huff_lut_len < HUFF_LUT_HEADER_SIZE) {
        return DecodedSymbol(0u, 0u, bit_cursor, 10u);
    }
    let lut_end = huff_lut_off + huff_lut_len;
    let root_bits = load_table_u8(huff_lut_off + 2u);
    let max_code_bits = load_table_u8(huff_lut_off + 3u);
    let root_len = load_table_u32(huff_lut_off + 4u);
    let sub_len = load_table_u32(huff_lut_off + 8u);
    if (root_bits == 0u || root_bits > max_code_bits || max_code_bits > 31u) {
        return DecodedSymbol(0u, 0u, bit_cursor, 11u);
    }
    let expected_root_len = (1u << root_bits);
    if (root_len != expected_root_len) {
        return DecodedSymbol(0u, 0u, bit_cursor, 12u);
    }
    let entry_bytes = (root_len + sub_len) * 4u;
    let entries_off = huff_lut_off + HUFF_LUT_HEADER_SIZE;
    if (entries_off + entry_bytes > lut_end) {
        return DecodedSymbol(0u, 0u, bit_cursor, 13u);
    }
    if (bit_cursor >= bit_end) {
        return DecodedSymbol(0u, 0u, bit_cursor, 14u);
    }

    let root_mask = (1u << root_bits) - 1u;
    let root_idx = peek_cmd_bits(bit_cursor, bit_end, root_bits) & root_mask;
    if (root_idx >= root_len) {
        return DecodedSymbol(0u, 0u, bit_cursor, 15u);
    }
    let root_entry = load_table_u32(entries_off + root_idx * 4u);
    let root_kind = root_entry & 0x3u;
    if (root_kind == 1u) {
        let bit_len = (root_entry >> 2u) & 0xffu;
        let symbol = (root_entry >> 10u) & 0xffffu;
        if (bit_len == 0u || bit_cursor + bit_len > bit_end || symbol > 255u) {
            return DecodedSymbol(0u, 0u, bit_cursor, 16u);
        }
        return DecodedSymbol(1u, symbol, bit_cursor + bit_len, 0u);
    }
    if (root_kind == 2u) {
        let sub_bits = (root_entry >> 2u) & 0xffu;
        let sub_off = root_entry >> 10u;
        if (sub_bits == 0u || root_bits + sub_bits > max_code_bits) {
            return DecodedSymbol(0u, 0u, bit_cursor, 17u);
        }
        let full = peek_cmd_bits(bit_cursor, bit_end, root_bits + sub_bits);
        let sub_mask = (1u << sub_bits) - 1u;
        let sub_idx = (full >> root_bits) & sub_mask;
        let abs_idx = root_len + sub_off + sub_idx;
        if (abs_idx >= root_len + sub_len) {
            return DecodedSymbol(0u, 0u, bit_cursor, 18u);
        }
        let sub_entry = load_table_u32(entries_off + abs_idx * 4u);
        let sub_kind = sub_entry & 0x3u;
        if (sub_kind != 1u) {
            return DecodedSymbol(0u, 0u, bit_cursor, 19u);
        }
        let bit_len = (sub_entry >> 2u) & 0xffu;
        let symbol = (sub_entry >> 10u) & 0xffffu;
        if (bit_len == 0u || bit_cursor + bit_len > bit_end || symbol > 255u) {
            return DecodedSymbol(0u, 0u, bit_cursor, 20u);
        }
        return DecodedSymbol(1u, symbol, bit_cursor + bit_len, 0u);
    }
    return DecodedSymbol(0u, 0u, bit_cursor, 21u);
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let section_count = section_meta_words[0];
    let table_count = section_meta_words[1];
    let table_repeat_stride = section_meta_words[2];
    let huff_lut_off = section_meta_words[3];
    let huff_lut_len = section_meta_words[4];
    if (gid3.y != 0u || gid3.z != 0u || gid3.x >= section_count) {
        return;
    }
    if (table_repeat_stride == 0u) {
        mark_error(gid3.x, 22u);
        return;
    }

    let sec = gid3.x;
    let meta_base = META_HEADER_WORDS + sec * META_WORDS;
    let cmd_start = section_meta_words[meta_base];
    let cmd_len = section_meta_words[meta_base + 1u];
    let out_start = section_meta_words[meta_base + 2u];
    let out_len = section_meta_words[meta_base + 3u];
    let cmd_end = cmd_start + cmd_len;
    let bit_end = cmd_end * 8u;
    let out_end = out_start + out_len;

    var bit_cursor = cmd_start * 8u;
    var out_cursor = out_start;
    var err: u32 = 0u;
    loop {
        if (out_cursor >= out_end) {
            break;
        }
        let sym0 = decode_huffman_symbol(bit_cursor, bit_end, huff_lut_off, huff_lut_len);
        if (sym0.ok == 0u) {
            err = 1u + sym0.err;
            break;
        }
        let sym1 = decode_huffman_symbol(sym0.next_bit, bit_end, huff_lut_off, huff_lut_len);
        if (sym1.ok == 0u) {
            err = 1u + sym1.err;
            break;
        }
        bit_cursor = sym1.next_bit;
        let cmd = sym0.symbol | (sym1.symbol << 8u);

        let tag = cmd & 0x0fffu;
        let len4 = (cmd >> 12u) & 0x0fu;
        var len = len4;
        if (len4 == 0x0fu) {
            let ext_sym = decode_huffman_symbol(bit_cursor, bit_end, huff_lut_off, huff_lut_len);
            if (ext_sym.ok == 0u) {
                err = 2u + ext_sym.err;
                break;
            }
            len = 15u + ext_sym.symbol;
            bit_cursor = ext_sym.next_bit;
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
            var i = 0u;
            loop {
                if (i >= len) {
                    break;
                }
                let lit_sym =
                    decode_huffman_symbol(bit_cursor, bit_end, huff_lut_off, huff_lut_len);
                if (lit_sym.ok == 0u) {
                    err = 6u + lit_sym.err;
                    break;
                }
                store_out_u8(out_cursor + i, lit_sym.symbol);
                bit_cursor = lit_sym.next_bit;
                i = i + 1u;
            }
            if (err != 0u) {
                break;
            }
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

        let table_base = tag * table_repeat_stride;
        copy_table_repeat_to_output(table_base, out_cursor, len);
        out_cursor = out_cursor + len;
    }

    if (err == 0u && bit_cursor != bit_end) {
        err = 9u;
    }
    mark_error(sec, err);
}
