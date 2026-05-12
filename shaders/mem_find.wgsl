// GPU bidirectional SMEM/MEM finding.
// Two entry points share the same bind group layout:
//   count_mems  (pass 1): writes per-query MEM counts to pass_buf_a
//   write_mems  (pass 2): reads offsets from pass_buf_a, writes results to mems_out
//
// One thread per query.
// Results: [query_start, query_end, match_count] packed as 3 u32 per MEM.
//
// Extension formulae (Lam 2009 Lemma 3):
//   extend_right(c): update rev interval via rev-OCC/rev-C,
//                    update fwd_lo by count_smaller_than(c, rev_lo, rev_hi, rev_occ)
//   extend_left(c):  update fwd interval via fwd-OCC/fwd-C,
//                    update rev_lo by count_smaller_than(c, fwd_lo, fwd_hi, fwd_occ)

const BLOCK_SIZE: u32 = 64u;
const ALPHA: u32 = 6u;
const MODE_SMEM: u32 = 0u;
const MODE_MEM: u32 = 1u;

struct Params {
    n_queries:      u32,
    min_len:        u32,
    fwd_text_len:   u32,
    rev_text_len:   u32,
    fwd_num_blocks: u32,
    rev_num_blocks: u32,
    mode:           u32,
    total_mems:     u32,
    fwd_c0: u32, fwd_c1: u32, fwd_c2: u32, fwd_c3: u32, fwd_c4: u32, fwd_c5: u32,
    rev_c0: u32, rev_c1: u32, rev_c2: u32, rev_c3: u32, rev_c4: u32, rev_c5: u32,
}

// 6 storage bindings + 1 uniform = 7 total (within max_storage_buffers_per_shader_stage=8)
@group(0) @binding(0) var<storage, read>       queries_flat:    array<u32>;
@group(0) @binding(1) var<storage, read>       query_offsets:   array<u32>;
@group(0) @binding(2) var<storage, read>       all_checkpoints: array<u32>; // fwd then rev
@group(0) @binding(3) var<storage, read>       all_bitvectors:  array<u32>; // fwd then rev
@group(0) @binding(4) var<storage, read_write> pass_buf_a:      array<u32>; // pass1: mem_counts; pass2: mem_offsets
@group(0) @binding(5) var<storage, read_write> mems_out:        array<u32>; // pass2: [start,end,count]*total_mems
@group(0) @binding(6) var<uniform>             params:          Params;

fn fwd_c_val(c: u32) -> u32 {
    switch c {
        case 0u: { return params.fwd_c0; }
        case 1u: { return params.fwd_c1; }
        case 2u: { return params.fwd_c2; }
        case 3u: { return params.fwd_c3; }
        case 4u: { return params.fwd_c4; }
        case 5u: { return params.fwd_c5; }
        default: { return 0u; }
    }
}

fn rev_c_val(c: u32) -> u32 {
    switch c {
        case 0u: { return params.rev_c0; }
        case 1u: { return params.rev_c1; }
        case 2u: { return params.rev_c2; }
        case 3u: { return params.rev_c3; }
        case 4u: { return params.rev_c4; }
        case 5u: { return params.rev_c5; }
        default: { return 0u; }
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

// Extend right by c (P → Pc): updates rev interval via rev-OCC/C,
// updates fwd interval via count_smaller_than on rev.
// Returns false (leaving state unchanged) if the interval collapses.
fn try_extend_right(
    fwd_lo: ptr<function, u32>, fwd_hi: ptr<function, u32>,
    rev_lo: ptr<function, u32>, rev_hi: ptr<function, u32>,
    c: u32
) -> bool {
    let cv  = rev_c_val(c);
    let nrl = cv + rev_occ(c, *rev_lo);
    let nrh = cv + rev_occ(c, *rev_hi);
    if nrl >= nrh { return false; }

    // offset = Σ_{b=0}^{c-1} (rev_occ(b, rev_hi) − rev_occ(b, rev_lo))
    var offset = 0u;
    var b = 0u;
    loop {
        if b >= c { break; }
        offset += rev_occ(b, *rev_hi) - rev_occ(b, *rev_lo);
        b += 1u;
    }

    let old_flo = *fwd_lo;
    *fwd_lo = old_flo + offset;
    *fwd_hi = old_flo + offset + (nrh - nrl);
    *rev_lo = nrl;
    *rev_hi = nrh;
    return true;
}

// Returns true if extending left by c would succeed (read-only check).
fn can_extend_left(fwd_lo: u32, fwd_hi: u32, c: u32) -> bool {
    let cv  = fwd_c_val(c);
    let nfl = cv + fwd_occ(c, fwd_lo);
    let nfh = cv + fwd_occ(c, fwd_hi);
    return nfl < nfh;
}

// Core per-query algorithm.
// write_output=false: count MEMs, return count.
// write_output=true:  read base offset from pass_buf_a[qid], write to mems_out.
fn process_query(qid: u32, write_output: bool) -> u32 {
    let pat_start = query_offsets[qid];
    let pat_end   = query_offsets[qid + 1u];
    let n         = pat_end - pat_start;
    if n == 0u { return 0u; }

    var mem_count = 0u;
    var out_base  = 0u;
    if write_output {
        out_base = pass_buf_a[qid]; // prefix-summed offset from CPU
    }

    var i = 0u;
    loop {
        if i >= n { break; }

        // ── Right-extension (uses rev OCC) ──────────────────────────────────
        var fwd_lo = 0u; var fwd_hi = params.fwd_text_len;
        var rev_lo = 0u; var rev_hi = params.rev_text_len;
        var j        = i;
        var last_flo = 0u; var last_fhi = params.fwd_text_len;
        var last_j   = i;
        var has_valid = false;

        loop {
            if j >= n { break; }
            let c = queries_flat[pat_start + j];
            if c >= ALPHA { break; }
            if !try_extend_right(&fwd_lo, &fwd_hi, &rev_lo, &rev_hi, c) { break; }
            j        += 1u;
            last_flo  = fwd_lo;
            last_fhi  = fwd_hi;
            last_j    = j;
            has_valid = true;
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
                is_left_max = !can_extend_left(last_flo, last_fhi, c_left);
            }
        }

        if !is_left_max {
            i += 1u;
            continue;
        }

        // ── Emit MEM ─────────────────────────────────────────────────────────
        if write_output {
            let slot = (out_base + mem_count) * 3u;
            mems_out[slot]      = i;
            mems_out[slot + 1u] = last_j;
            mems_out[slot + 2u] = last_fhi - last_flo; // match_count = fwd interval size
        }
        mem_count += 1u;

        if params.mode == MODE_SMEM {
            i = last_j; // advance past the SMEM right boundary
        } else {
            i += 1u;    // MEM mode: advance by 1, report all maximal matches
        }
    }

    return mem_count;
}

@compute @workgroup_size(64)
fn count_mems(@builtin(global_invocation_id) gid: vec3u) {
    let qid = gid.x;
    if qid >= params.n_queries { return; }
    pass_buf_a[qid] = process_query(qid, false);
}

@compute @workgroup_size(64)
fn write_mems(@builtin(global_invocation_id) gid: vec3u) {
    let qid = gid.x;
    if qid >= params.n_queries { return; }
    _ = process_query(qid, true);
}
