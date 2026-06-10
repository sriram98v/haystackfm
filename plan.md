# GPU IUPAC Ambiguity Support — Implementation Plan

## Root Problem

GPU emits **one `(lo,hi)` interval** per query; CPU `backward_search` returns **a union of disjoint intervals**. Everything downstream (count, locate, MEM/SMEM) is wrong for any IUPAC ambiguity code in a query.

CPU path (`src/fm_index/query.rs:42`):
- Calls `compatible_symbols(c)` → expands each code to all matching bases
- Iterates over compatible bases, collects multiple SA intervals, merges them

GPU path (`shaders/locate_search.wgsl`, `shaders/mem_find.wgsl`):
- Treats query char as a single exact code — `c_val(c)` with no expansion
- No branching, no merging → wrong results for N, R, Y, S, W, K, M, B, D, H, V

## Key Design Choice: `MAX_IVS = 16`

C-array partitions BWT by symbol → intervals for different compatible symbols never overlap.
After merging, ≤15 disjoint intervals exist per backward step.
Use fixed-size `array<vec2u, 16>` in-register; **incremental merge after each char** (bounds register pressure vs 240-elem scratch).

---

## Phase 1 — Shared IUPAC table in WGSL (~0.5d, Low risk)

Encode `compatible_symbols` as two WGSL constants:
- `const COMPAT: array<u32, 256>` — indexed by `code * 16 + k`
- `const COMPAT_LEN: array<u32, 16>` — number of compatible symbols per code

Must exactly mirror `src/alphabet.rs::compatible_symbols`. Add a Rust parity test
(`#[cfg(test)]`) that builds the same table from `compatible_symbols` and asserts equality.

**Files:** `shaders/locate_search.wgsl`, `shaders/mem_find.wgsl`, `src/alphabet.rs` (test only)

Status: [x] complete

---

## Phase 2 — `locate_search.wgsl` multi-interval (~1.5d, High)

Replace scalar `lo/hi` with `var ivs: array<vec2u, MAX_IVS>; var n_ivs: u32`.

Per query char:
1. For each active interval `ivs[k]` × each `r in compat(c)`:
   - Compute `new_lo = C[r] + occ(r, lo)`, `new_hi = C[r] + occ(r, hi)`
   - Collect if `new_lo < new_hi`
2. Incremental merge (insertion sort by lo + linear coalesce, mirrors `merge_intervals` at `query.rs:133`)
3. Break if empty

Output layout change: emit `MAX_IVS` intervals per query (zero-pad unused slots).
Buffer: `[num_queries * MAX_IVS * 2]` u32s. Fixed stride, no second prefix-sum needed.

Driver (`src/gpu/locate.rs`):
- `match_count[q] = Σ(hi−lo)` over its `MAX_IVS` interval block
- Prefix sum and resolve phase unchanged

**Files:** `shaders/locate_search.wgsl`, `src/gpu/locate.rs`

Status: [x] complete

---

## Phase 3 — `locate_resolve.wgsl` interval-list aware (~1d, High)

Map flat thread index `k = tid - match_offsets[qid]` to `(interval, within_offset)`:
- Walk per-query `MAX_IVS` block, subtract each `hi−lo` until `k` lands inside
- `bwt_pos = iv_lo + k_local`

**Binding workaround (8-binding limit already hit):** Use zero-terminated intervals —
real intervals always satisfy `hi > lo`, so sentinel `(0,0)` terminates the scan.
No extra binding needed.

**Files:** `shaders/locate_resolve.wgsl`, `src/gpu/locate.rs`

Status: [x] complete

---

## Phase 4 — `mem_find.wgsl` multi-interval bidirectional (~2–3d, Highest)

Replace scalar `(fwd_lo, fwd_hi, rev_lo, rev_hi)` with `var ivs: array<vec4u, MAX_IVS>; var n_ivs`.

`extend_multi_right(c)`:
- For each iv × each `r in compat(c)`: run `try_extend_right` math, collect non-empty
- Merge by fwd interval

Left-maximality: not-left-maximal if ANY iv extends left by ANY `r in compat(query[i-1])`
(mirrors `smem.rs:175`).

`match_count = Σ fwd sizes`.

**Two-level MEM output layout:**
- MEM header: `[qs, qe, iv_offset, iv_count]`
- Flat fwd-interval buffer (separate)

Pass 1 outputs both `mem_count` and `total_iv_count` per query; CPU prefix-sums both.
Single `process_query` fn shared by both passes (existing pattern).

Update downstream: `shaders/mem_resolve.wgsl`, `shaders/ref_map.wgsl`, `src/gpu/mem_find.rs`.

**Files:** `shaders/mem_find.wgsl`, `shaders/mem_resolve.wgsl`, `shaders/ref_map.wgsl`, `src/gpu/mem_find.rs`

Status: [x] complete

---

## Phase 5 — Parity Tests (~1d, Medium)

- WGSL compat table == `alphabet.rs::compatible_symbols` (CI-enforced, fails on drift)
- `count_gpu` / `locate_gpu` == CPU across all IUPAC query × ref combos
- `find_smems_gpu` / `find_mems_gpu` == CPU (sorted positions)
- All GPU tests skip cleanly when no GPU adapter available

**Files:** `src/gpu/locate.rs` (tests), `src/gpu/mem_find.rs` (tests)

Status: [x] complete

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Register pressure with 16-interval arrays | High | Incremental merge, not 240-elem scratch |
| 8-binding limit in resolve | Medium | Zero-terminated interval blocks |
| MEM pass 1/2 count divergence | Medium | Single `process_query` fn, write-branch only |
| `MAX_IVS` overflow on N-heavy queries | Low | Debug clamp + stress test; bump to 32 if exceeded |
| WGSL compat table drifts from `alphabet.rs` | Low | Parity test in CI |

---

## Estimate

| Phase | Effort | Risk |
|-------|--------|------|
| 1 — WGSL IUPAC table | 0.5d | Low |
| 2 — locate_search multi-interval | 1.5d | High |
| 3 — locate_resolve interval-list | 1d | High |
| 4 — mem_find multi-interval bidir | 2–3d | Highest |
| 5 — Parity tests | 1d | Medium |
| **Total** | **~6–7d** | |

**Phases 1–3 (locate count/locate) are independently shippable before Phase 4 (MEM/SMEM).**

---

## Success Criteria

- [ ] `count_gpu` / `locate_gpu` match CPU across all IUPAC query/ref combos
- [ ] `find_smems_gpu` / `find_mems_gpu` match CPU (counts + positions)
- [ ] WGSL compat table proven equal to `alphabet.rs::compatible_symbols`
- [ ] No exact-ACGT benchmark regression
- [ ] Resolve stays ≤8 storage bindings
- [ ] Parity tests skip cleanly when no GPU adapter
