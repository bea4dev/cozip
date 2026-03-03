@group(0) @binding(0)
var<storage, read_write> out_words: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read_write> total_bits: array<u32>;

@group(0) @binding(2)
var<storage, read> dyn_meta: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) {
        return;
    }

    let header_bits = dyn_meta[0];
    let eob_code = dyn_meta[1];
    let eob_bits = dyn_meta[2];

    let token_bits = total_bits[0];
    let bit_offset = header_bits + token_bits;
    let word_index = bit_offset >> 5u;
    let shift = bit_offset & 31u;

    atomicOr(&out_words[word_index], eob_code << shift);
    if (shift + eob_bits > 32u) {
        atomicOr(&out_words[word_index + 1u], eob_code >> (32u - shift));
    }

    total_bits[0] = bit_offset + eob_bits;
}
