// GPU SA resolve for MEM intervals.
//
// One thread per SA position across all MEMs.
// Each thread binary-searches `offsets_in` to find its owning MEM,
// computes its SA index (fwd_lo + local offset), then walks back to
// the nearest sampled SA entry to resolve the raw text position.
//
// Binding layout (7 storage + 1 uniform = within max_storage_buffers_per_shader_stage=8):
//   0: bwt          — BWT encoded as u32 (one u8 per entry, packed as u32)
//   1: checkpoints  — Occ checkpoints [num_blocks * ALPHA]
//   2: bitvectors   — Occ bitvectors  [num_blocks * ALPHA * 2]
//   3: sa_samples   — sampled SA; U32_MAX = unsampled
//   4: intervals_in — flat [fwd_lo, fwd_hi] * num_mems
//   5: offsets_in   — prefix sums of interval sizes [num_mems + 1]
//   6: positions_out — raw text positions [total_positions]; U32_MAX on error
//   7: params       — uniform

const BLOCK_SIZE: u32 = 64u;
const ALPHA: u32      = 16u;  // full IUPAC: $=0,A=1,C=2,G=3,T=4,N=5,R=6,Y=7,S=8,W=9,K=10,M=11,B=12,D=13,H=14,V=15
const U32_MAX: u32    = 0xFFFFFFFFu;

// 24 × u32 = 96 bytes (multiple of 16 — satisfies WGSL uniform alignment).
struct Params {
    num_mems:    u32,
    text_len:    u32,
    num_blocks:  u32,
    sample_rate: u32,
    total_pos:   u32,
    _pad0:       u32,
    c0: u32, c1: u32, c2: u32,  c3: u32,  c4: u32,  c5: u32,
    c6: u32, c7: u32, c8: u32,  c9: u32,  c10: u32, c11: u32,
    c12: u32, c13: u32, c14: u32, c15: u32,
    _pad1: u32, _pad2: u32,
}

@group(0) @binding(0) var<storage, read>       bwt:          array<u32>;
@group(0) @binding(1) var<storage, read>       checkpoints:  array<u32>;
@group(0) @binding(2) var<storage, read>       bitvectors:   array<u32>;
@group(0) @binding(3) var<storage, read>       sa_samples:   array<u32>;
@group(0) @binding(4) var<storage, read>       intervals_in: array<u32>; // [lo, hi] per MEM
@group(0) @binding(5) var<storage, read>       offsets_in:   array<u32>; // prefix sums
@group(0) @binding(6) var<storage, read_write> positions_out: array<u32>;
@group(0) @binding(7) var<uniform>             params:        Params;

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

fn lf_map(pos: u32) -> u32 {
    let c = bwt[pos];
    return c_val(c) + occ_rank(c, pos);
}

@compute @workgroup_size(64)
fn resolve_positions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if tid >= params.total_pos { return; }

    // Binary search offsets_in to find owning MEM.
    var lo = 0u;
    var hi = params.num_mems;
    loop {
        if lo >= hi { break; }
        let mid = (lo + hi) / 2u;
        if offsets_in[mid + 1u] <= tid {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let mem_id    = lo;
    let fwd_lo    = intervals_in[mem_id * 2u];
    let local_idx = tid - offsets_in[mem_id];
    var bwt_pos   = fwd_lo + local_idx;

    // Walk back to nearest sampled SA entry.
    var steps = 0u;
    loop {
        if bwt_pos < arrayLength(&sa_samples) && sa_samples[bwt_pos] != U32_MAX {
            positions_out[tid] = sa_samples[bwt_pos] + steps;
            return;
        }
        bwt_pos = lf_map(bwt_pos);
        steps  += 1u;
        if steps > params.sample_rate {
            positions_out[tid] = U32_MAX;
            return;
        }
    }
}
