@group(0) @binding(0) var<storage, read> src_words: array<u32>;
@group(0) @binding(1) var<storage, read> table_meta: array<u32>;
@group(0) @binding(2) var<storage, read> emit_desc_words: array<u32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_words: array<u32>;

const FINALIZE_LITERAL_FLAG: u32 = 0x80000000u;
const FINALIZE_EMIT_DESC_WORDS: u32 = 3u;

fn load_src(idx: u32) -> u32 {
    let w = src_words[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (w >> shift) & 0xffu;
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

fn output_byte_for_entry(entry_idx: u32, local: u32) -> u32 {
    let desc = emit_desc_words[emit_desc_base(entry_idx)];
    if (emit_desc_is_literal(desc)) {
        return emit_desc_literal_byte(desc);
    }
    return load_src(desc + local);
}

fn find_entry_for_pos(out_count: u32, pos: u32) -> u32 {
    var lo = 0u;
    var hi = out_count;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) >> 1u;
        let meta_base = 1u + mid * 2u;
        let off = table_meta[meta_base];
        let len = table_meta[meta_base + 1u];
        if (pos < off) {
            hi = mid;
        } else if (pos >= off + len) {
            lo = mid + 1u;
        } else {
            return mid;
        }
    }
    return 0xffffffffu;
}

@compute
@workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid3: vec3<u32>,
) {
    let max_entries = params[4];
    let out_count = min(table_meta[0], max_entries);
    let total_bytes = table_meta[1u + max_entries * 2u];
    let total_words = (total_bytes + 3u) >> 2u;
    let word_idx = gid3.x;
    if (word_idx >= total_words) {
        return;
    }
    let base_pos = word_idx << 2u;
    var word = 0u;
    var byte_idx = 0u;
    loop {
        if (byte_idx >= 4u) {
            break;
        }
        let pos = base_pos + byte_idx;
        if (pos < total_bytes) {
            let entry_idx = find_entry_for_pos(out_count, pos);
            if (entry_idx != 0xffffffffu) {
                let meta_base = 1u + entry_idx * 2u;
                let off = table_meta[meta_base];
                let byte = output_byte_for_entry(entry_idx, pos - off) & 0xffu;
                word = word | (byte << (byte_idx * 8u));
            }
        }
        byte_idx = byte_idx + 1u;
    }
    out_words[word_idx] = word;
}
