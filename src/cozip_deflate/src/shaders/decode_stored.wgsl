struct Params {
    chunk_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct HuffmanTree {
    counts: array<u32, 16>,
    symbols: array<u32, 320>,
    max_bits: u32,
    symbol_count: u32,
};

var<private> CODELEN_ORDER: array<u32, 19> = array<u32, 19>(
    16u, 17u, 18u, 0u, 8u, 7u, 9u, 6u, 10u, 5u, 11u, 4u, 12u, 3u, 13u, 2u, 14u, 1u, 15u,
);

var<private> LENGTH_BASE: array<u32, 29> = array<u32, 29>(
    3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 13u, 15u, 17u, 19u, 23u, 27u,
    31u, 35u, 43u, 51u, 59u, 67u, 83u, 99u, 115u, 131u, 163u, 195u, 227u, 258u,
);

var<private> LENGTH_EXTRA_BITS: array<u32, 29> = array<u32, 29>(
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u,
    2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 0u,
);

var<private> DIST_BASE: array<u32, 30> = array<u32, 30>(
    1u, 2u, 3u, 4u, 5u, 7u, 9u, 13u, 17u, 25u, 33u, 49u, 65u, 97u, 129u,
    193u, 257u, 385u, 513u, 769u, 1025u, 1537u, 2049u, 3073u, 4097u,
    6145u, 8193u, 12289u, 16385u, 24577u,
);

var<private> DIST_EXTRA_BITS: array<u32, 30> = array<u32, 30>(
    0u, 0u, 0u, 0u, 1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u, 6u,
    6u, 7u, 7u, 8u, 8u, 9u, 9u, 10u, 10u, 11u, 11u, 12u, 12u, 13u, 13u,
);

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;
@group(0) @binding(1)
var<storage, read> meta_words: array<u32>;
@group(0) @binding(2)
var<storage, read_write> output_words: array<u32>;
@group(0) @binding(3)
var<storage, read_write> out_lens: array<u32>;
@group(0) @binding(4)
var<storage, read_write> status_codes: array<u32>;
@group(0) @binding(5)
var<uniform> params: Params;

fn read_input_byte(byte_off: u32) -> u32 {
    let word = input_words[byte_off >> 2u];
    let shift = (byte_off & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

fn read_input_bit(comp_off: u32, bit_pos: u32) -> u32 {
    let byte_off = comp_off + (bit_pos >> 3u);
    let shift = bit_pos & 7u;
    return (read_input_byte(byte_off) >> shift) & 1u;
}

fn write_output_byte(byte_off: u32, value: u32) {
    let word_index = byte_off >> 2u;
    let shift = (byte_off & 3u) * 8u;
    let mask = 0xffu << shift;
    let prior = output_words[word_index];
    output_words[word_index] = (prior & (~mask)) | ((value & 0xffu) << shift);
}

fn read_output_byte(byte_off: u32) -> u32 {
    let word = output_words[byte_off >> 2u];
    let shift = (byte_off & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

fn read_bit_checked(comp_off: u32, comp_bits: u32, bit_pos: ptr<function, u32>, status: ptr<function, u32>) -> u32 {
    if (*status != 0u) {
        return 0u;
    }
    if (*bit_pos >= comp_bits) {
        *status = 2u; // truncated bitstream
        return 0u;
    }
    let b = read_input_bit(comp_off, *bit_pos);
    *bit_pos = *bit_pos + 1u;
    return b;
}

fn read_bits_checked(
    comp_off: u32,
    comp_bits: u32,
    bit_pos: ptr<function, u32>,
    bit_count: u32,
    status: ptr<function, u32>,
) -> u32 {
    var out = 0u;
    var i = 0u;
    loop {
        if (i >= bit_count) {
            break;
        }
        let b = read_bit_checked(comp_off, comp_bits, bit_pos, status);
        if (*status != 0u) {
            return 0u;
        }
        out = out | (b << i);
        i = i + 1u;
    }
    return out;
}

fn build_huffman_tree(
    lengths: ptr<function, array<u32, 320>>,
    num_symbols: u32,
    tree: ptr<function, HuffmanTree>,
) -> bool {
    var i = 0u;
    loop {
        if (i >= 16u) {
            break;
        }
        (*tree).counts[i] = 0u;
        i = i + 1u;
    }
    (*tree).max_bits = 0u;
    (*tree).symbol_count = 0u;

    i = 0u;
    loop {
        if (i >= num_symbols) {
            break;
        }
        let len = (*lengths)[i];
        if (len > 15u) {
            return false;
        }
        if (len > 0u) {
            (*tree).counts[len] = (*tree).counts[len] + 1u;
            (*tree).symbol_count = (*tree).symbol_count + 1u;
            if (len > (*tree).max_bits) {
                (*tree).max_bits = len;
            }
        }
        i = i + 1u;
    }

    var offsets: array<u32, 16>;
    i = 0u;
    loop {
        if (i >= 16u) {
            break;
        }
        offsets[i] = 0u;
        i = i + 1u;
    }
    var bits = 1u;
    loop {
        if (bits >= 15u) {
            break;
        }
        offsets[bits + 1u] = offsets[bits] + (*tree).counts[bits];
        bits = bits + 1u;
    }

    i = 0u;
    loop {
        if (i >= num_symbols) {
            break;
        }
        let len = (*lengths)[i];
        if (len > 0u) {
            let slot = offsets[len];
            if (slot >= 320u) {
                return false;
            }
            (*tree).symbols[slot] = i;
            offsets[len] = slot + 1u;
        }
        i = i + 1u;
    }

    return true;
}

fn decode_huffman_symbol(
    comp_off: u32,
    comp_bits: u32,
    bit_pos: ptr<function, u32>,
    tree: ptr<function, HuffmanTree>,
    status: ptr<function, u32>,
) -> u32 {
    if (*status != 0u) {
        return 0u;
    }
    if ((*tree).max_bits == 0u) {
        *status = 6u;
        return 0u;
    }

    var code = 0u;
    var first = 0u;
    var index = 0u;
    var len = 1u;
    loop {
        if (len > (*tree).max_bits) {
            break;
        }
        let bit = read_bit_checked(comp_off, comp_bits, bit_pos, status);
        if (*status != 0u) {
            return 0u;
        }
        code = code | bit;

        let count = (*tree).counts[len];
        if (code >= first && (code - first) < count) {
            let off = index + (code - first);
            if (off >= (*tree).symbol_count) {
                *status = 6u;
                return 0u;
            }
            return (*tree).symbols[off];
        }

        index = index + count;
        first = (first + count) << 1u;
        code = code << 1u;
        len = len + 1u;
    }

    *status = 6u;
    return 0u;
}

fn build_fixed_trees(litlen_tree: ptr<function, HuffmanTree>, dist_tree: ptr<function, HuffmanTree>) -> bool {
    var lit_lengths: array<u32, 320>;
    var i = 0u;
    loop {
        if (i >= 320u) {
            break;
        }
        lit_lengths[i] = 0u;
        i = i + 1u;
    }
    i = 0u;
    loop {
        if (i >= 144u) {
            break;
        }
        lit_lengths[i] = 8u;
        i = i + 1u;
    }
    i = 144u;
    loop {
        if (i >= 256u) {
            break;
        }
        lit_lengths[i] = 9u;
        i = i + 1u;
    }
    i = 256u;
    loop {
        if (i >= 280u) {
            break;
        }
        lit_lengths[i] = 7u;
        i = i + 1u;
    }
    i = 280u;
    loop {
        if (i >= 288u) {
            break;
        }
        lit_lengths[i] = 8u;
        i = i + 1u;
    }

    var dist_lengths: array<u32, 320>;
    i = 0u;
    loop {
        if (i >= 320u) {
            break;
        }
        dist_lengths[i] = 0u;
        i = i + 1u;
    }
    i = 0u;
    loop {
        if (i >= 32u) {
            break;
        }
        dist_lengths[i] = 5u;
        i = i + 1u;
    }

    if (!build_huffman_tree(&lit_lengths, 288u, litlen_tree)) {
        return false;
    }
    if (!build_huffman_tree(&dist_lengths, 32u, dist_tree)) {
        return false;
    }
    return true;
}

fn build_dynamic_trees(
    comp_off: u32,
    comp_bits: u32,
    bit_pos: ptr<function, u32>,
    litlen_tree: ptr<function, HuffmanTree>,
    dist_tree: ptr<function, HuffmanTree>,
    status: ptr<function, u32>,
) -> bool {
    let hlit = read_bits_checked(comp_off, comp_bits, bit_pos, 5u, status) + 257u;
    let hdist = read_bits_checked(comp_off, comp_bits, bit_pos, 5u, status) + 1u;
    let hclen = read_bits_checked(comp_off, comp_bits, bit_pos, 4u, status) + 4u;
    if (*status != 0u) {
        return false;
    }
    if (hlit > 286u || hdist > 32u || hclen > 19u) {
        *status = 8u;
        return false;
    }
    let total = hlit + hdist;
    if (total > 320u) {
        *status = 8u;
        return false;
    }

    var codelen_lengths: array<u32, 320>;
    var i = 0u;
    loop {
        if (i >= 320u) {
            break;
        }
        codelen_lengths[i] = 0u;
        i = i + 1u;
    }

    i = 0u;
    loop {
        if (i >= hclen) {
            break;
        }
        let sym = CODELEN_ORDER[i];
        codelen_lengths[sym] = read_bits_checked(comp_off, comp_bits, bit_pos, 3u, status);
        if (*status != 0u) {
            return false;
        }
        i = i + 1u;
    }

    var codelen_tree: HuffmanTree;
    if (!build_huffman_tree(&codelen_lengths, 19u, &codelen_tree)) {
        *status = 8u;
        return false;
    }

    var lengths: array<u32, 320>;
    i = 0u;
    loop {
        if (i >= 320u) {
            break;
        }
        lengths[i] = 0u;
        i = i + 1u;
    }

    var out_count = 0u;
    loop {
        if (out_count >= total) {
            break;
        }
        let sym = decode_huffman_symbol(comp_off, comp_bits, bit_pos, &codelen_tree, status);
        if (*status != 0u) {
            return false;
        }

        if (sym <= 15u) {
            lengths[out_count] = sym;
            out_count = out_count + 1u;
            continue;
        }

        if (sym == 16u) {
            if (out_count == 0u) {
                *status = 8u;
                return false;
            }
            let repeat = read_bits_checked(comp_off, comp_bits, bit_pos, 2u, status) + 3u;
            if (*status != 0u) {
                return false;
            }
            if (out_count + repeat > total) {
                *status = 8u;
                return false;
            }
            let prev = lengths[out_count - 1u];
            var r = 0u;
            loop {
                if (r >= repeat) {
                    break;
                }
                lengths[out_count] = prev;
                out_count = out_count + 1u;
                r = r + 1u;
            }
            continue;
        }

        if (sym == 17u) {
            let repeat = read_bits_checked(comp_off, comp_bits, bit_pos, 3u, status) + 3u;
            if (*status != 0u) {
                return false;
            }
            if (out_count + repeat > total) {
                *status = 8u;
                return false;
            }
            var r = 0u;
            loop {
                if (r >= repeat) {
                    break;
                }
                lengths[out_count] = 0u;
                out_count = out_count + 1u;
                r = r + 1u;
            }
            continue;
        }

        if (sym == 18u) {
            let repeat = read_bits_checked(comp_off, comp_bits, bit_pos, 7u, status) + 11u;
            if (*status != 0u) {
                return false;
            }
            if (out_count + repeat > total) {
                *status = 8u;
                return false;
            }
            var r = 0u;
            loop {
                if (r >= repeat) {
                    break;
                }
                lengths[out_count] = 0u;
                out_count = out_count + 1u;
                r = r + 1u;
            }
            continue;
        }

        *status = 8u;
        return false;
    }

    var lit_lengths: array<u32, 320>;
    var dist_lengths: array<u32, 320>;
    i = 0u;
    loop {
        if (i >= 320u) {
            break;
        }
        lit_lengths[i] = 0u;
        dist_lengths[i] = 0u;
        i = i + 1u;
    }

    i = 0u;
    loop {
        if (i >= hlit) {
            break;
        }
        lit_lengths[i] = lengths[i];
        i = i + 1u;
    }
    i = 0u;
    loop {
        if (i >= hdist) {
            break;
        }
        dist_lengths[i] = lengths[hlit + i];
        i = i + 1u;
    }

    if (!build_huffman_tree(&lit_lengths, hlit, litlen_tree)) {
        *status = 8u;
        return false;
    }
    if (!build_huffman_tree(&dist_lengths, hdist, dist_tree)) {
        *status = 8u;
        return false;
    }
    return true;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.chunk_count) {
        return;
    }

    let base = idx * 4u;
    let comp_off = meta_words[base + 0u];
    let comp_bits = meta_words[base + 1u];
    let out_off = meta_words[base + 2u];
    let out_cap = meta_words[base + 3u];

    var bit_pos = 0u;
    var out_pos = 0u;
    var status = 0u;

    loop {
        if (bit_pos + 3u > comp_bits) {
            status = 2u;
            break;
        }

        let bfinal = read_bit_checked(comp_off, comp_bits, &bit_pos, &status);
        let btype = read_bits_checked(comp_off, comp_bits, &bit_pos, 2u, &status);
        if (status != 0u) {
            break;
        }

        if (btype == 0u) {
            bit_pos = (bit_pos + 7u) & (~7u);
            if (bit_pos + 32u > comp_bits) {
                status = 2u;
                break;
            }
            let len_byte_off = comp_off + (bit_pos >> 3u);
            let len = read_input_byte(len_byte_off + 0u) | (read_input_byte(len_byte_off + 1u) << 8u);
            let nlen = read_input_byte(len_byte_off + 2u) | (read_input_byte(len_byte_off + 3u) << 8u);
            if (((len ^ nlen) & 0xffffu) != 0xffffu) {
                status = 3u;
                break;
            }
            bit_pos = bit_pos + 32u;

            if (out_pos + len > out_cap) {
                status = 4u;
                break;
            }
            if (bit_pos + len * 8u > comp_bits) {
                status = 2u;
                break;
            }

            let src_off = comp_off + (bit_pos >> 3u);
            var i = 0u;
            loop {
                if (i >= len) {
                    break;
                }
                let b = read_input_byte(src_off + i);
                write_output_byte(out_off + out_pos + i, b);
                i = i + 1u;
            }
            out_pos = out_pos + len;
            bit_pos = bit_pos + (len * 8u);
        } else if (btype == 1u || btype == 2u) {
            var litlen_tree: HuffmanTree;
            var dist_tree: HuffmanTree;

            if (btype == 1u) {
                if (!build_fixed_trees(&litlen_tree, &dist_tree)) {
                    status = 6u;
                    break;
                }
            } else {
                if (!build_dynamic_trees(comp_off, comp_bits, &bit_pos, &litlen_tree, &dist_tree, &status)) {
                    if (status == 0u) {
                        status = 8u;
                    }
                    break;
                }
            }

            loop {
                let sym = decode_huffman_symbol(comp_off, comp_bits, &bit_pos, &litlen_tree, &status);
                if (status != 0u) {
                    break;
                }

                if (sym < 256u) {
                    if (out_pos >= out_cap) {
                        status = 4u;
                        break;
                    }
                    write_output_byte(out_off + out_pos, sym);
                    out_pos = out_pos + 1u;
                    continue;
                }

                if (sym == 256u) {
                    break;
                }

                if (sym < 257u || sym > 285u) {
                    status = 7u;
                    break;
                }

                let len_index = sym - 257u;
                var match_len = LENGTH_BASE[len_index];
                let extra_len_bits = LENGTH_EXTRA_BITS[len_index];
                if (extra_len_bits > 0u) {
                    match_len = match_len + read_bits_checked(comp_off, comp_bits, &bit_pos, extra_len_bits, &status);
                    if (status != 0u) {
                        break;
                    }
                }

                let dist_sym = decode_huffman_symbol(comp_off, comp_bits, &bit_pos, &dist_tree, &status);
                if (status != 0u) {
                    break;
                }
                if (dist_sym >= 30u) {
                    status = 7u;
                    break;
                }

                var dist = DIST_BASE[dist_sym];
                let extra_dist_bits = DIST_EXTRA_BITS[dist_sym];
                if (extra_dist_bits > 0u) {
                    dist = dist + read_bits_checked(comp_off, comp_bits, &bit_pos, extra_dist_bits, &status);
                    if (status != 0u) {
                        break;
                    }
                }

                if (dist == 0u || dist > out_pos) {
                    status = 7u;
                    break;
                }
                if (out_pos + match_len > out_cap) {
                    status = 4u;
                    break;
                }

                var m = 0u;
                loop {
                    if (m >= match_len) {
                        break;
                    }
                    let b = read_output_byte(out_off + out_pos - dist);
                    write_output_byte(out_off + out_pos, b);
                    out_pos = out_pos + 1u;
                    m = m + 1u;
                }
            }

            if (status != 0u) {
                break;
            }
        } else {
            status = 1u; // invalid block type
            break;
        }

        if (bfinal == 1u) {
            break;
        }
    }

    if (status == 0u && out_pos != out_cap) {
        status = 5u;
    }

    out_lens[idx] = out_pos;
    status_codes[idx] = status;
}
