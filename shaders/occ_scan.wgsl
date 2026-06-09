// Occ table block construction.
//
// For each block of BLOCK_SIZE (64) BWT characters, one workgroup of 64 threads:
//   - Sets bits in per-character bitvectors (bit j = BWT[block_start + j] == c)
//   - Counts occurrences per character in this block
//
// Outputs (for CPU to prefix-sum into checkpoints):
//   block_counts[block * ALPHA + c]  = count of c in this block
//   bitvectors[(block * ALPHA + c)*2 + 0] = low  32 bits of presence bitvector for c
//   bitvectors[(block * ALPHA + c)*2 + 1] = high 32 bits of presence bitvector for c

const BLOCK_SIZE: u32 = 64u;
const ALPHA: u32 = 16u;  // IUPAC alphabet: $=0,A=1,C=2,G=3,T=4,N=5,R=6,Y=7,S=8,W=9,K=10,M=11,B=12,D=13,H=14,V=15

struct Params {
    n: u32,
    num_blocks: u32,
}

@group(0) @binding(0) var<storage, read>       bwt:          array<u32>;
@group(0) @binding(1) var<storage, read_write> block_counts: array<u32>;  // [num_blocks * ALPHA]
@group(0) @binding(2) var<storage, read_write> bitvectors:   array<u32>;  // [num_blocks * ALPHA * 2]
@group(0) @binding(3) var<uniform>             params:        Params;

// Workgroup-shared accumulators (atomic for correctness across the 64 threads)
var<workgroup> ws_counts: array<atomic<u32>, 16>;
var<workgroup> ws_bv_lo:  array<atomic<u32>, 16>;  // bits 0-31
var<workgroup> ws_bv_hi:  array<atomic<u32>, 16>;  // bits 32-63

@compute @workgroup_size(64)
fn occ_block(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id)        gid: vec3u,
) {
    let block_id = gid.x;

    // --- Initialise shared accumulators (first 5 threads) ---
    if (lid.x < ALPHA) {
        atomicStore(&ws_counts[lid.x], 0u);
        atomicStore(&ws_bv_lo[lid.x],  0u);
        atomicStore(&ws_bv_hi[lid.x],  0u);
    }
    workgroupBarrier();

    // --- Each thread processes one BWT position in this block ---
    let pos = block_id * BLOCK_SIZE + lid.x;
    if (pos < params.n) {
        let ch = bwt[pos];
        if (ch < ALPHA) {
            atomicAdd(&ws_counts[ch], 1u);
            // Set the bit for this position (lid.x in [0,63])
            if (lid.x < 32u) {
                atomicOr(&ws_bv_lo[ch], 1u << lid.x);
            } else {
                atomicOr(&ws_bv_hi[ch], 1u << (lid.x - 32u));
            }
        }
    }
    workgroupBarrier();

    // --- Write results to global memory (first 5 threads) ---
    if (lid.x < ALPHA) {
        let c = lid.x;
        block_counts[block_id * ALPHA + c]         = atomicLoad(&ws_counts[c]);
        bitvectors[(block_id * ALPHA + c) * 2u]     = atomicLoad(&ws_bv_lo[c]);
        bitvectors[(block_id * ALPHA + c) * 2u + 1u] = atomicLoad(&ws_bv_hi[c]);
    }
}
