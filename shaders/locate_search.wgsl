// GPU backward search: one thread per query.
//
// Each invocation reads its pattern from queries_flat[query_offsets[qid]..query_offsets[qid+1]],
// walks right-to-left, and narrows the SA interval using C[] and Occ[].
// Output: intervals[qid*2] = lo, intervals[qid*2+1] = hi.

const BLOCK_SIZE: u32 = 64u;
const ALPHA: u32 = 16u;  // full IUPAC: $=0,A=1,C=2,G=3,T=4,N=5,R=6,Y=7,S=8,W=9,K=10,M=11,B=12,D=13,H=14,V=15

// c_array values are embedded in the uniform to stay within the
// max_storage_buffers_per_shader_stage=8 limit.
// 20 × u32 = 80 bytes (multiple of 16).
struct Params {
    num_queries: u32,
    text_len:    u32,
    num_blocks:  u32,
    _pad:        u32,
    // C-array: C[c] = number of chars lexicographically < c
    c0: u32, c1: u32, c2: u32,  c3: u32,  c4: u32,  c5: u32,
    c6: u32, c7: u32, c8: u32,  c9: u32,  c10: u32, c11: u32,
    c12: u32, c13: u32, c14: u32, c15: u32,
}

@group(0) @binding(0) var<storage, read>       queries_flat:  array<u32>;
@group(0) @binding(1) var<storage, read>       query_offsets: array<u32>; // len = num_queries + 1
@group(0) @binding(2) var<storage, read>       checkpoints:   array<u32>; // [num_blocks * ALPHA]
@group(0) @binding(3) var<storage, read>       bitvectors:    array<u32>; // [num_blocks * ALPHA * 2]
@group(0) @binding(4) var<storage, read_write> intervals:     array<u32>; // [num_queries * 2]
@group(0) @binding(5) var<uniform>             params:         Params;

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

// Count occurrences of character c in bwt[0..i).
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
        // offset in [32, 62]: all of bv_lo + lower (offset-31) bits of bv_hi
        let hi_bits = offset - 31u; // in [1, 31] — no overflow
        let mask_hi = (1u << hi_bits) - 1u;
        count += countOneBits(bv_lo) + countOneBits(bv_hi & mask_hi);
    }
    return count;
}

@compute @workgroup_size(64)
fn locate_search(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let qid = gid.x;
    if qid >= params.num_queries { return; }

    let pat_start = query_offsets[qid];
    let pat_end   = query_offsets[qid + 1u];
    let pat_len   = pat_end - pat_start;

    if pat_len == 0u {
        intervals[qid * 2u]      = 0u;
        intervals[qid * 2u + 1u] = params.text_len;
        return;
    }

    var lo = 0u;
    var hi = params.text_len;

    var k = pat_len;
    loop {
        if k == 0u { break; }
        if lo >= hi { break; }
        k -= 1u;
        let c = queries_flat[pat_start + k];
        if c >= ALPHA {
            lo = 0u;
            hi = 0u;
            break;
        }
        let cv = c_val(c);
        lo = cv + occ_rank(c, lo);
        hi = cv + occ_rank(c, hi);
    }

    intervals[qid * 2u]      = lo;
    intervals[qid * 2u + 1u] = hi;
}
