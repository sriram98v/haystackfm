// GPU SA resolve: one thread per match.
//
// Each invocation finds its query via binary search on match_offsets, then
// resolves the SA position using LF-mapping over the sampled SA, and finally
// maps the text position to (seq_id, pos_in_seq) via binary search on seq_bounds.
//
// c_array values are embedded in the uniform to stay within
// max_storage_buffers_per_shader_stage=8.

const BLOCK_SIZE: u32 = 64u;
const ALPHA: u32 = 16u;  // full IUPAC: $=0,A=1,C=2,G=3,T=4,N=5,R=6,Y=7,S=8,W=9,K=10,M=11,B=12,D=13,H=14,V=15
const U32_MAX: u32 = 0xFFFFFFFFu;
const MAX_IVS: u32 = 16u;

// 24 × u32 = 96 bytes (multiple of 16).
struct Params {
    num_queries:   u32,
    text_len:      u32,
    num_blocks:    u32,
    num_seqs:      u32,
    sample_rate:   u32,
    total_matches: u32,
    // C-array: C[c] = number of chars lexicographically < c
    c0: u32, c1: u32, c2: u32,  c3: u32,  c4: u32,  c5: u32,
    c6: u32, c7: u32, c8: u32,  c9: u32,  c10: u32, c11: u32,
    c12: u32, c13: u32, c14: u32, c15: u32,
    _pad: u32, _pad2: u32,
}

// Storage bindings: 8 total (at the max_storage_buffers_per_shader_stage limit)
@group(0) @binding(0) var<storage, read>       bwt:           array<u32>;
@group(0) @binding(1) var<storage, read>       checkpoints:   array<u32>; // [num_blocks * ALPHA]
@group(0) @binding(2) var<storage, read>       bitvectors:    array<u32>; // [num_blocks * ALPHA * 2]
@group(0) @binding(3) var<storage, read>       sa_samples:    array<u32>; // U32_MAX = unsampled
@group(0) @binding(4) var<storage, read>       seq_bounds:    array<u32>; // cumulative seq end positions
@group(0) @binding(5) var<storage, read>       intervals:     array<u32>; // [num_queries * MAX_IVS * 2]
@group(0) @binding(6) var<storage, read>       match_offsets: array<u32>; // [num_queries + 1]
@group(0) @binding(7) var<storage, read_write> results:       array<u32>; // [total_matches * 2]
@group(0) @binding(8) var<uniform>             params:         Params;

fn c_val(c: u32) -> u32 {
    switch c {
        case 0u:  { return params.c0; }
        case 1u:  { return params.c1; }
        case 2u:  { return params.c2; }
        case 3u:  { return params.c3; }
        case 4u:  { return params.c4; }
        case 5u:  { return params.c5; }
        case 6u:  { return params.c6; }
        case 7u:  { return params.c7; }
        case 8u:  { return params.c8; }
        case 9u:  { return params.c9; }
        case 10u: { return params.c10; }
        case 11u: { return params.c11; }
        case 12u: { return params.c12; }
        case 13u: { return params.c13; }
        case 14u: { return params.c14; }
        case 15u: { return params.c15; }
        default:  { return 0u; }
    }
}

fn occ_rank(c: u32, i: u32) -> u32 {
    if i == 0u { return 0u; }
    let block  = (i - 1u) / BLOCK_SIZE;
    let offset = (i - 1u) % BLOCK_SIZE;

    let checkpoint = checkpoints[block * ALPHA + c];
    let bv_lo      = bitvectors[(block * ALPHA + c) * 2u];
    let bv_hi      = bitvectors[(block * ALPHA + c) * 2u + 1u];

    var count = checkpoint;
    if offset < 32u {
        var mask_lo: u32;
        if offset == 31u {
            mask_lo = 0xFFFFFFFFu;
        } else {
            mask_lo = (1u << (offset + 1u)) - 1u;
        }
        count += countOneBits(bv_lo & mask_lo);
    } else if offset == 63u {
        count += countOneBits(bv_lo) + countOneBits(bv_hi);
    } else {
        let hi_bits = offset - 31u;
        let mask_hi = (1u << hi_bits) - 1u;
        count += countOneBits(bv_lo) + countOneBits(bv_hi & mask_hi);
    }
    return count;
}

// LF-mapping: LF(i) = C[BWT[i]] + Occ(BWT[i], i)
fn lf_map(i: u32) -> u32 {
    let c = bwt[i];
    return c_val(c) + occ_rank(c, i);
}

@compute @workgroup_size(64)
fn locate_resolve(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let tid = gid.x;
    if tid >= params.total_matches { return; }

    // Upper-bound of tid in match_offsets → find qid such that
    // match_offsets[qid] <= tid < match_offsets[qid+1].
    var blo = 0u;
    var bhi = params.num_queries + 1u;
    loop {
        if blo >= bhi { break; }
        let bmid = (blo + bhi) / 2u;
        if match_offsets[bmid] <= tid {
            blo = bmid + 1u;
        } else {
            bhi = bmid;
        }
    }
    let qid = blo - 1u;

    // Scan the per-query interval block to map local offset → SA position.
    // intervals layout: [qid * MAX_IVS * 2 + i*2] = lo_i, [+i*2+1] = hi_i
    // Zero-padded slots (lo==hi==0) contribute size 0 and are skipped.
    let iv_base = qid * MAX_IVS * 2u;
    var remaining = tid - match_offsets[qid];
    var bwt_pos = 0u;
    for (var ii = 0u; ii < MAX_IVS; ii++) {
        let iv_lo = intervals[iv_base + ii * 2u];
        let iv_hi = intervals[iv_base + ii * 2u + 1u];
        if iv_hi <= iv_lo { continue; }
        let size = iv_hi - iv_lo;
        if remaining < size {
            bwt_pos = iv_lo + remaining;
            break;
        }
        remaining -= size;
    }

    // Walk LF-mapping until a sampled SA position is found.
    var steps = 0u;
    loop {
        let sa_val = sa_samples[bwt_pos];
        if sa_val != U32_MAX {
            let text_pos = sa_val + steps;

            // partition_point: find first idx where seq_bounds[idx] > text_pos
            var slo = 0u;
            var shi = params.num_seqs;
            loop {
                if slo >= shi { break; }
                let smid = (slo + shi) / 2u;
                if seq_bounds[smid] <= text_pos {
                    slo = smid + 1u;
                } else {
                    shi = smid;
                }
            }
            let seq_idx = slo;

            var seq_start = 0u;
            if seq_idx > 0u {
                seq_start = seq_bounds[seq_idx - 1u];
            }

            results[tid * 2u]      = seq_idx;
            results[tid * 2u + 1u] = text_pos - seq_start;
            return;
        }
        bwt_pos = lf_map(bwt_pos);
        steps += 1u;
        if steps > params.sample_rate {
            // Unreachable with a correct sampled SA; guard against infinite loop.
            results[tid * 2u]      = U32_MAX;
            results[tid * 2u + 1u] = U32_MAX;
            return;
        }
    }
}
