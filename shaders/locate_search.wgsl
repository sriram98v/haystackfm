// GPU backward search: one thread per query.
//
// Each invocation reads its pattern from queries_flat[query_offsets[qid]..query_offsets[qid+1]],
// walks right-to-left, and narrows the SA interval using C[] and Occ[].
// Output: intervals[qid*2] = lo, intervals[qid*2+1] = hi.

const BLOCK_SIZE: u32 = 64u;
const ALPHA: u32 = 16u;  // full IUPAC: $=0,A=1,C=2,G=3,T=4,N=5,R=6,Y=7,S=8,W=9,K=10,M=11,B=12,D=13,H=14,V=15
const MAX_IVS: u32 = 16u;

// IUPAC compatible-symbol table (mirrors alphabet::compatible_symbols in Rust).
// COMPAT[code * 16 + k] = k-th symbol compatible with `code`; 0 = padding.
// COMPAT_LEN[code]      = number of valid entries for `code`.
// Two codes are compatible when their IUPAC base sets share ≥1 nucleotide.
const COMPAT_LEN: array<u32, 16> = array<u32, 16>(
     0u,  8u,  8u,  8u,  8u, 15u, 12u, 12u,
    12u, 12u, 12u, 12u, 14u, 14u, 14u, 14u,
);
const COMPAT: array<u32, 256> = array<u32, 256>(
    // code  0 ($): no compatible symbols
     0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,
    // code  1 (A): A N R W M D H V
     1u,  5u,  6u,  9u, 11u, 13u, 14u, 15u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,
    // code  2 (C): C N Y S M B H V
     2u,  5u,  7u,  8u, 11u, 12u, 14u, 15u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,
    // code  3 (G): G N R S K B D V
     3u,  5u,  6u,  8u, 10u, 12u, 13u, 15u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,
    // code  4 (T): T N Y W K B D H
     4u,  5u,  7u,  9u, 10u, 12u, 13u, 14u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,  0u,
    // code  5 (N): A C G T N R Y S W K M B D H V
     1u,  2u,  3u,  4u,  5u,  6u,  7u,  8u,  9u, 10u, 11u, 12u, 13u, 14u, 15u,  0u,
    // code  6 (R=A|G): A G N R S W K M B D H V
     1u,  3u,  5u,  6u,  8u,  9u, 10u, 11u, 12u, 13u, 14u, 15u,  0u,  0u,  0u,  0u,
    // code  7 (Y=C|T): C T N Y S W K M B D H V
     2u,  4u,  5u,  7u,  8u,  9u, 10u, 11u, 12u, 13u, 14u, 15u,  0u,  0u,  0u,  0u,
    // code  8 (S=G|C): C G N R Y S K M B D H V
     2u,  3u,  5u,  6u,  7u,  8u, 10u, 11u, 12u, 13u, 14u, 15u,  0u,  0u,  0u,  0u,
    // code  9 (W=A|T): A T N R Y W K M B D H V
     1u,  4u,  5u,  6u,  7u,  9u, 10u, 11u, 12u, 13u, 14u, 15u,  0u,  0u,  0u,  0u,
    // code 10 (K=G|T): G T N R Y S W K B D H V
     3u,  4u,  5u,  6u,  7u,  8u,  9u, 10u, 12u, 13u, 14u, 15u,  0u,  0u,  0u,  0u,
    // code 11 (M=A|C): A C N R Y S W M B D H V
     1u,  2u,  5u,  6u,  7u,  8u,  9u, 11u, 12u, 13u, 14u, 15u,  0u,  0u,  0u,  0u,
    // code 12 (B=C|G|T): C G T N R Y S W K M B D H V
     2u,  3u,  4u,  5u,  6u,  7u,  8u,  9u, 10u, 11u, 12u, 13u, 14u, 15u,  0u,  0u,
    // code 13 (D=A|G|T): A G T N R Y S W K M B D H V
     1u,  3u,  4u,  5u,  6u,  7u,  8u,  9u, 10u, 11u, 12u, 13u, 14u, 15u,  0u,  0u,
    // code 14 (H=A|C|T): A C T N R Y S W K M B D H V
     1u,  2u,  4u,  5u,  6u,  7u,  8u,  9u, 10u, 11u, 12u, 13u, 14u, 15u,  0u,  0u,
    // code 15 (V=A|C|G): A C G N R Y S W K M B D H V
     1u,  2u,  3u,  5u,  6u,  7u,  8u,  9u, 10u, 11u, 12u, 13u, 14u, 15u,  0u,  0u,
);

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
@group(0) @binding(4) var<storage, read_write> intervals:     array<u32>; // [num_queries * MAX_IVS * 2]
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

    let base = qid * MAX_IVS * 2u;

    if pat_len == 0u {
        intervals[base]      = 0u;
        intervals[base + 1u] = params.text_len;
        for (var zi = 1u; zi < MAX_IVS; zi++) {
            intervals[base + zi * 2u]      = 0u;
            intervals[base + zi * 2u + 1u] = 0u;
        }
        return;
    }

    var ivs: array<vec2u, 16>;
    var n_ivs: u32 = 1u;
    ivs[0] = vec2u(0u, params.text_len);

    var k = pat_len;
    loop {
        if k == 0u { break; }
        if n_ivs == 0u { break; }
        k -= 1u;
        let c = queries_flat[pat_start + k];
        if c >= ALPHA {
            n_ivs = 0u;
            break;
        }

        // Expand: each active interval × each compatible symbol → child intervals
        var scratch: array<vec2u, 16>;
        var ns: u32 = 0u;
        let clen = COMPAT_LEN[c];
        for (var ii = 0u; ii < n_ivs; ii++) {
            let iv_lo = ivs[ii].x;
            let iv_hi = ivs[ii].y;
            for (var ki = 0u; ki < clen; ki++) {
                let r   = COMPAT[c * 16u + ki];
                let cv  = c_val(r);
                let nlo = cv + occ_rank(r, iv_lo);
                let nhi = cv + occ_rank(r, iv_hi);
                if nlo < nhi && ns < MAX_IVS {
                    scratch[ns] = vec2u(nlo, nhi);
                    ns += 1u;
                }
            }
        }

        if ns == 0u {
            n_ivs = 0u;
            break;
        }

        // Insertion sort by lo
        for (var i = 1u; i < ns; i++) {
            let key = scratch[i];
            var j   = i;
            loop {
                if j == 0u { break; }
                if scratch[j - 1u].x <= key.x { break; }
                scratch[j] = scratch[j - 1u];
                j -= 1u;
            }
            scratch[j] = key;
        }

        // Coalesce overlapping / adjacent intervals
        var merged: array<vec2u, 16>;
        var nm: u32 = 1u;
        merged[0] = scratch[0];
        for (var i = 1u; i < ns; i++) {
            if scratch[i].x <= merged[nm - 1u].y {
                if scratch[i].y > merged[nm - 1u].y {
                    merged[nm - 1u].y = scratch[i].y;
                }
            } else {
                merged[nm] = scratch[i];
                nm += 1u;
            }
        }

        n_ivs = nm;
        for (var i = 0u; i < nm; i++) {
            ivs[i] = merged[i];
        }
    }

    // Write MAX_IVS slots; zero-pad unused
    for (var i = 0u; i < MAX_IVS; i++) {
        if i < n_ivs {
            intervals[base + i * 2u]      = ivs[i].x;
            intervals[base + i * 2u + 1u] = ivs[i].y;
        } else {
            intervals[base + i * 2u]      = 0u;
            intervals[base + i * 2u + 1u] = 0u;
        }
    }
}
