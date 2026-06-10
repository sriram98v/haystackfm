// GPU bidirectional SMEM/MEM finding.
// Two entry points share the same bind group layout:
//   count_mems  (pass 1): writes [mem_count, iv_count] per query to pass_buf_a (stride 2)
//   write_mems  (pass 2): reads [mem_offset, iv_offset] from pass_buf_a, writes to mems_out + iv_buf
//
// One thread per query.
// mems_out: [query_start, query_end, iv_offset, n_ivs] packed as 4 u32 per MEM.
// iv_buf:   [fwd_lo, fwd_hi] per fwd interval across all MEMs (flat, stride 2).
//
// Extension formulae (Lam 2009 Lemma 3):
//   extend_right(c): update rev interval via rev-OCC/rev-C,
//                    update fwd_lo by count_smaller_than(c, rev_lo, rev_hi, rev_occ)
//   extend_left(c):  update fwd interval via fwd-OCC/fwd-C,
//                    update rev_lo by count_smaller_than(c, fwd_lo, fwd_hi, fwd_occ)

const BLOCK_SIZE: u32 = 64u;
const ALPHA: u32 = 16u;  // full IUPAC: $=0,A=1,C=2,G=3,T=4,N=5,R=6,Y=7,S=8,W=9,K=10,M=11,B=12,D=13,H=14,V=15
const MODE_SMEM: u32 = 0u;
const MODE_MEM: u32 = 1u;
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

// 40 × u32 = 160 bytes (multiple of 16).
struct Params {
    n_queries:      u32,
    min_len:        u32,
    fwd_text_len:   u32,
    rev_text_len:   u32,
    fwd_num_blocks: u32,
    rev_num_blocks: u32,
    mode:           u32,
    total_mems:     u32,
    fwd_c0:  u32, fwd_c1:  u32, fwd_c2:  u32, fwd_c3:  u32,
    fwd_c4:  u32, fwd_c5:  u32, fwd_c6:  u32, fwd_c7:  u32,
    fwd_c8:  u32, fwd_c9:  u32, fwd_c10: u32, fwd_c11: u32,
    fwd_c12: u32, fwd_c13: u32, fwd_c14: u32, fwd_c15: u32,
    rev_c0:  u32, rev_c1:  u32, rev_c2:  u32, rev_c3:  u32,
    rev_c4:  u32, rev_c5:  u32, rev_c6:  u32, rev_c7:  u32,
    rev_c8:  u32, rev_c9:  u32, rev_c10: u32, rev_c11: u32,
    rev_c12: u32, rev_c13: u32, rev_c14: u32, rev_c15: u32,
}

// 7 storage bindings + 1 uniform = 8 total (at max_storage_buffers_per_shader_stage=8)
@group(0) @binding(0) var<storage, read>       queries_flat:    array<u32>;
@group(0) @binding(1) var<storage, read>       query_offsets:   array<u32>;
@group(0) @binding(2) var<storage, read>       all_checkpoints: array<u32>; // fwd then rev
@group(0) @binding(3) var<storage, read>       all_bitvectors:  array<u32>; // fwd then rev
@group(0) @binding(4) var<storage, read_write> pass_buf_a:      array<u32>; // pass1: [mem_count,iv_count]*n_queries; pass2: [mem_offset,iv_offset]*n_queries
@group(0) @binding(5) var<storage, read_write> mems_out:        array<u32>; // pass2: [qs,qe,iv_offset,n_ivs]*total_mems
@group(0) @binding(6) var<storage, read_write> iv_buf:          array<u32>; // pass2: [fwd_lo,fwd_hi]*total_ivs
@group(0) @binding(7) var<uniform>             params:          Params;

fn fwd_c_val(c: u32) -> u32 {
    switch c {
        case 0u:  { return params.fwd_c0; }
        case 1u:  { return params.fwd_c1; }
        case 2u:  { return params.fwd_c2; }
        case 3u:  { return params.fwd_c3; }
        case 4u:  { return params.fwd_c4; }
        case 5u:  { return params.fwd_c5; }
        case 6u:  { return params.fwd_c6; }
        case 7u:  { return params.fwd_c7; }
        case 8u:  { return params.fwd_c8; }
        case 9u:  { return params.fwd_c9; }
        case 10u: { return params.fwd_c10; }
        case 11u: { return params.fwd_c11; }
        case 12u: { return params.fwd_c12; }
        case 13u: { return params.fwd_c13; }
        case 14u: { return params.fwd_c14; }
        case 15u: { return params.fwd_c15; }
        default:  { return 0u; }
    }
}

fn rev_c_val(c: u32) -> u32 {
    switch c {
        case 0u:  { return params.rev_c0; }
        case 1u:  { return params.rev_c1; }
        case 2u:  { return params.rev_c2; }
        case 3u:  { return params.rev_c3; }
        case 4u:  { return params.rev_c4; }
        case 5u:  { return params.rev_c5; }
        case 6u:  { return params.rev_c6; }
        case 7u:  { return params.rev_c7; }
        case 8u:  { return params.rev_c8; }
        case 9u:  { return params.rev_c9; }
        case 10u: { return params.rev_c10; }
        case 11u: { return params.rev_c11; }
        case 12u: { return params.rev_c12; }
        case 13u: { return params.rev_c13; }
        case 14u: { return params.rev_c14; }
        case 15u: { return params.rev_c15; }
        default:  { return 0u; }
    }
}

fn occ_rank_at(c: u32, i: u32, chk_base: u32, bv_base: u32) -> u32 {
    if i == 0u { return 0u; }
    let block  = (i - 1u) / BLOCK_SIZE;
    let offset = (i - 1u) % BLOCK_SIZE;

    let checkpoint = all_checkpoints[chk_base + block * ALPHA + c];
    let bv_lo      = all_bitvectors[bv_base + (block * ALPHA + c) * 2u];
    let bv_hi      = all_bitvectors[bv_base + (block * ALPHA + c) * 2u + 1u];

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

fn fwd_occ(c: u32, i: u32) -> u32 {
    return occ_rank_at(c, i, 0u, 0u);
}

fn rev_occ(c: u32, i: u32) -> u32 {
    let chk_base = params.fwd_num_blocks * ALPHA;
    let bv_base  = params.fwd_num_blocks * ALPHA * 2u;
    return occ_rank_at(c, i, chk_base, bv_base);
}

// Extend right by c: pure function returning vec4u(fwd_lo, fwd_hi, rev_lo, rev_hi).
// Returns vec4u(0,0,0,0) on collapse; success check: result.y > result.x.
fn try_extend_right_pure(fwd_lo: u32, fwd_hi: u32, rev_lo: u32, rev_hi: u32, c: u32) -> vec4u {
    let cv  = rev_c_val(c);
    let nrl = cv + rev_occ(c, rev_lo);
    let nrh = cv + rev_occ(c, rev_hi);
    if nrl >= nrh { return vec4u(0u, 0u, 0u, 0u); }

    var offset = 0u;
    var b = 0u;
    loop {
        if b >= c { break; }
        offset += rev_occ(b, rev_hi) - rev_occ(b, rev_lo);
        b += 1u;
    }

    let new_flo = fwd_lo + offset;
    return vec4u(new_flo, new_flo + (nrh - nrl), nrl, nrh);
}

// Extend all active bidir intervals right by IUPAC code c.
// Collects one child per (iv, compatible base); deduplicates by fwd range; caps at MAX_IVS.
fn extend_multi_right_step(
    ivs: ptr<function, array<vec4u, 16>>,
    n_ivs: ptr<function, u32>,
    c: u32,
) {
    var scratch: array<vec4u, 16>;
    var ns: u32 = 0u;
    let clen = COMPAT_LEN[c];
    for (var ii = 0u; ii < *n_ivs; ii++) {
        let iv = (*ivs)[ii];
        for (var ki = 0u; ki < clen; ki++) {
            let r = COMPAT[c * 16u + ki];
            let new_iv = try_extend_right_pure(iv.x, iv.y, iv.z, iv.w, r);
            if new_iv.y > new_iv.x {
                var dup = false;
                for (var di = 0u; di < ns; di++) {
                    if scratch[di].x == new_iv.x && scratch[di].y == new_iv.y {
                        dup = true;
                        break;
                    }
                }
                if !dup && ns < MAX_IVS {
                    scratch[ns] = new_iv;
                    ns += 1u;
                }
            }
        }
    }
    *n_ivs = ns;
    for (var i = 0u; i < ns; i++) { (*ivs)[i] = scratch[i]; }
}

// Returns true if extending left by c would succeed (read-only check).
fn can_extend_left(fwd_lo: u32, fwd_hi: u32, c: u32) -> bool {
    let cv  = fwd_c_val(c);
    let nfl = cv + fwd_occ(c, fwd_lo);
    let nfh = cv + fwd_occ(c, fwd_hi);
    return nfl < nfh;
}

// Returns true if ANY active interval can extend left by ANY base compatible with c.
fn any_can_extend_left_multi(
    ivs: ptr<function, array<vec4u, 16>>,
    n_ivs: u32,
    c: u32,
) -> bool {
    let clen = COMPAT_LEN[c];
    for (var ii = 0u; ii < n_ivs; ii++) {
        let iv = (*ivs)[ii];
        for (var ki = 0u; ki < clen; ki++) {
            let r = COMPAT[c * 16u + ki];
            if can_extend_left(iv.x, iv.y, r) { return true; }
        }
    }
    return false;
}

// Core per-query algorithm.
// write_output=false: write [mem_count, iv_count] to pass_buf_a[qid*2..qid*2+2].
// write_output=true:  read [mem_offset, iv_offset] from pass_buf_a[qid*2..qid*2+2],
//                     write MEM headers to mems_out and fwd intervals to iv_buf.
fn process_query(qid: u32, write_output: bool) -> u32 {
    let pat_start = query_offsets[qid];
    let pat_end   = query_offsets[qid + 1u];
    let n         = pat_end - pat_start;
    if n == 0u {
        if !write_output {
            pass_buf_a[qid * 2u]      = 0u;
            pass_buf_a[qid * 2u + 1u] = 0u;
        }
        return 0u;
    }

    var mem_count    = 0u;
    var iv_count     = 0u;
    var mem_out_base = 0u;
    var iv_out       = 0u;
    if write_output {
        mem_out_base = pass_buf_a[qid * 2u];
        iv_out       = pass_buf_a[qid * 2u + 1u];
    }

    var i = 0u;
    loop {
        if i >= n { break; }

        // ── Right-extension (uses rev OCC) ──────────────────────────────────
        var ivs: array<vec4u, 16>;
        var n_ivs: u32 = 1u;
        ivs[0] = vec4u(0u, params.fwd_text_len, 0u, params.rev_text_len);

        var last_ivs: array<vec4u, 16>;
        var last_n_ivs: u32 = 0u;
        var last_j    = i;
        var has_valid = false;
        var j         = i;

        loop {
            if j >= n { break; }
            let c = queries_flat[pat_start + j];
            if c >= ALPHA { break; }
            extend_multi_right_step(&ivs, &n_ivs, c);
            if n_ivs == 0u { break; }
            j += 1u;
            for (var ci = 0u; ci < n_ivs; ci++) { last_ivs[ci] = ivs[ci]; }
            last_n_ivs = n_ivs;
            last_j     = j;
            has_valid  = true;
        }

        if !has_valid || (last_j - i) < params.min_len {
            i += 1u;
            continue;
        }

        // ── Left-maximality check (uses fwd OCC) ────────────────────────────
        var is_left_max: bool;
        if i == 0u {
            is_left_max = true;
        } else {
            let c_left = queries_flat[pat_start + i - 1u];
            if c_left >= ALPHA {
                is_left_max = true;
            } else {
                is_left_max = !any_can_extend_left_multi(&last_ivs, last_n_ivs, c_left);
            }
        }

        if !is_left_max {
            i += 1u;
            continue;
        }

        // ── Emit MEM ─────────────────────────────────────────────────────────
        if write_output {
            let mem_slot = (mem_out_base + mem_count) * 4u;
            mems_out[mem_slot]      = i;
            mems_out[mem_slot + 1u] = last_j;
            mems_out[mem_slot + 2u] = iv_out;
            mems_out[mem_slot + 3u] = last_n_ivs;
            for (var ki = 0u; ki < last_n_ivs; ki++) {
                iv_buf[(iv_out + ki) * 2u]      = last_ivs[ki].x;
                iv_buf[(iv_out + ki) * 2u + 1u] = last_ivs[ki].y;
            }
            iv_out += last_n_ivs;
        } else {
            iv_count += last_n_ivs;
        }
        mem_count += 1u;

        if params.mode == MODE_SMEM {
            i = last_j;
        } else {
            i += 1u;
        }
    }

    if !write_output {
        pass_buf_a[qid * 2u]      = mem_count;
        pass_buf_a[qid * 2u + 1u] = iv_count;
    }
    return mem_count;
}

@compute @workgroup_size(64)
fn count_mems(@builtin(global_invocation_id) gid: vec3u) {
    let qid = gid.x;
    if qid >= params.n_queries { return; }
    _ = process_query(qid, false);
}

@compute @workgroup_size(64)
fn write_mems(@builtin(global_invocation_id) gid: vec3u) {
    let qid = gid.x;
    if qid >= params.n_queries { return; }
    _ = process_query(qid, true);
}
