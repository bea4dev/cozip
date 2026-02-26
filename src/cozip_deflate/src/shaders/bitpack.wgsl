struct Params {
    len: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> codes: array<u32>;

@group(0) @binding(1)
var<storage, read> codes_hi: array<u32>;

@group(0) @binding(2)
var<storage, read> bitlens: array<u32>;

@group(0) @binding(3)
var<storage, read> bit_offsets: array<u32>;

@group(0) @binding(4)
var<storage, read_write> out_words: array<atomic<u32>>;

@group(0) @binding(5)
var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + (id.y * 8388480u);
    if (idx >= params.len) {
        return;
    }

    let bits = bitlens[idx];
    if (bits == 0u) {
        return;
    }

    let code = codes[idx];
    let code_hi = codes_hi[idx];
    let bit_offset = bit_offsets[idx] + params._pad1;
    let word_index = bit_offset >> 5u;
    let shift = bit_offset & 31u;

    atomicOr(&out_words[word_index], code << shift);
    // Avoid implementation-defined shift-by-32 when shift == 0 and bits > 32.
    // The high part for bits > 32 is handled by the code_hi path below.
    if (shift > 0u && shift + bits > 32u) {
        atomicOr(&out_words[word_index + 1u], code >> (32u - shift));
    }

    if (bits > 32u) {
        let hi_bits = bits - 32u;
        let hi_offset = bit_offset + 32u;
        let hi_word_index = hi_offset >> 5u;
        let hi_shift = hi_offset & 31u;

        atomicOr(&out_words[hi_word_index], code_hi << hi_shift);
        if (hi_shift + hi_bits > 32u) {
            atomicOr(&out_words[hi_word_index + 1u], code_hi >> (32u - hi_shift));
        }
    }
}
