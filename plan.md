# Plan: Pure-Rust Construction Speedup + Memory (WASM-Safe)

## Requirements
- Improve construction speed (currently ~2136s / ~35 min for chromosome)
- Reduce peak memory (currently 8276 MB, 27× input)
- **All changes must be pure Rust** — no C FFI, no `cc`/`cmake` deps
- WASM build (`wasm-pack build --features wasm`) must continue working

---

## Phase 1 — Drop Full SA Before Occ Build — LOW EFFORT

**Expected: ~1.2 GB peak reduction for chromosome (drops 4n bytes earlier)**

One-line change in `src/fm_index/mod.rs` (around line 79–91):
```rust
let sa = build_suffix_array(&text);
let bwt = build_bwt(&text, &sa);
let sa_samples = SampledSuffixArray::from_full(&sa, config.sa_sample_rate);
drop(sa);   // ← free 4n bytes before building Occ
let occ = build_occ(&bwt);
```

**Risk:** None — purely mechanical change.

---

## Phase 2 — Replace Prefix-Doubling with `psacak` ✨ HIGH IMPACT

**Expected: O(n log² n) → O(n), ~20-50× construction speedup**

`psacak` (SACA-K algorithm) is a pure-Rust linear-time SA construction crate — already present as a transitive dep in the benchmark. No C FFI, works on `wasm32-unknown-unknown`.

**Changes:**
- Add `psacak = "0.1"` to `[dependencies]` in `Cargo.toml`
- Replace `build_suffix_array()` in `src/suffix_array/cpu.rs`:
  ```rust
  pub fn build_suffix_array(text: &[u8]) -> SuffixArray {
      let mut sa = vec![0usize; text.len()];
      psacak::saca_k(text, &mut sa);
      SuffixArray { data: sa.into_iter().map(|x| x as u32).collect() }
  }
  ```
  *(exact API shape needs verification against `psacak` docs before coding)*
- All existing SA tests remain valid (same output, different algorithm)

**Risk:** psacak uses `usize` SA entries — need to verify API and cast to u32. Low risk.

---

## Phase 3 — Compact `SampledSuffixArray` — MEDIUM IMPACT

**Expected: ~75% reduction in sampled SA memory (4n → ~1.25n bytes)**

`bitvec` is pure Rust — WASM-safe.

**Changes:**
- Add `bitvec = "1"` to `[dependencies]` in `Cargo.toml`
- Replace flat `Vec<u32>` (length n, u32::MAX sentinels) in `src/suffix_array/mod.rs`:
  ```rust
  pub struct SampledSuffixArray {
      is_sampled: bitvec::vec::BitVec,  // n/8 bytes
      samples: Vec<u32>,                 // only n/sample_rate entries
      sample_rate: u32,
  }
  ```
- Update `from_full()`, `is_sampled()`, `get()` — rank into `samples` via `is_sampled.count_ones(..)` prefix sum
- Update `locate()` in `src/fm_index/mod.rs` to use new indexing
- All existing locate tests cover correctness

**Risk:** rank query must match LF-walk logic. Existing locate tests catch regressions.

---

## Estimated Impact

| Phase | Effort | Speedup | Memory |
|-------|--------|---------|--------|
| 1 — drop SA early | ~15min | neutral | −1.2 GB peak |
| 2 — psacak | ~1h | ~20-50× construction | neutral |
| 3 — compact SA | ~2h | neutral | −75% sampled SA (~6 GB saved) |

**Total: LOW–MEDIUM. All three phases together: ~35 min → ~1–3 min, ~8 GB → ~2–3 GB.**

---

## WASM Safety

| Component | WASM-safe? | Reason |
|-----------|-----------|--------|
| `psacak` | ✅ | Pure Rust, no `cc` dep |
| `bitvec` | ✅ | Pure Rust |
| `drop(sa)` | ✅ | Language feature |

No platform-conditional cfg guards needed.
