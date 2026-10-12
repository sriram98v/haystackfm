# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build (CPU only, default)
cargo build

# Build with GPU support
cargo build --features gpu

# Build with WASM bindings
cargo build --features wasm

# Run all tests
cargo test

# Run tests with GPU feature
cargo test --features gpu

# Run a single test by name
cargo test <test_name> --features gpu

# Run benchmarks
cargo bench --features gpu

# Run a specific benchmark
cargo bench --bench locate_bench --features gpu
cargo bench --bench mem_bench --features gpu
cargo bench --bench mem_positions_bench --features gpu

# Lint
cargo clippy --all-features

# Build WASM package
wasm-pack build --target web --features wasm
```

## Architecture

GPU-accelerated FM-index for DNA sequences. Compiles to native (Vulkan/Metal/DX12) and WASM (browser WebGPU).

### Construction pipeline
1. **Input**: `[DnaSequence]` — concatenated with per-sequence sentinel bytes
2. **Suffix array** (GPU): iterative prefix-doubling via `radix_sort.wgsl` + `sa_*.wgsl`
3. **BWT** (GPU): `bwt_gather.wgsl`
4. **C array** (CPU): trivial 5-value histogram over the alphabet
5. **Occ table** (GPU): `occ_scan.wgsl` + `prefix_sum.wgsl`
6. **SA sampling**: every `k`-th position stored for locate

### Query pipeline (CPU)
- `count`: backward search → O(m) SA interval
- `locate`: LF-walk from interval positions to nearest SA sample → O(m + occ·k)

### GPU MEM/SMEM finding (3-pass pipeline)
1. `mem_find.wgsl` — bidirectional extension, emits raw SA intervals (4-field format: `[fwd_lo, fwd_hi, rev_lo, rev_hi]`)
2. `mem_resolve.wgsl` — resolves SA intervals to text positions
3. `ref_map.wgsl` — maps positions to reference sequences via binary search

### Key modules
| Path | Role |
|------|------|
| `src/fm_index/` | `FmIndex`, `BidirFmIndex`, backward search, SMEM/MEM logic (`smem.rs`: sweep generic over `BidirSearch`, text-free left-maximality via `BWT[row]`), cursor (`bidir.rs`: extend/contract); `lookup.rs` depth-k table with per-k-mer exact slot + wildcard variants |
| `src/rlbwt/` | `RlBwt`: run-length BWT as 64-byte blocks of 8 runs (starts, head nibbles, per-symbol rank samples) behind a row→run bucket directory, implements `BwtRank` (incl. `extend_ranks` run scan); `from_runs`/`from_bwt`/`from_occ`/`build_cpu` |
| `src/rl_fm_index/` | `RlFmIndex` (count/locate via sampled SA, no text; `sa_sample_rate = 0` = count-only; `sequence` extracts by LF-walk from per-sequence sentinel rows `seq_end_rows`; optional lookup table), `BidirRlFmIndex` (cursor via `FmHalf`, `find_smems`/`find_mems` through `smem::BidirSearch`), `"HRL\x02"`/`"HRB\x02"` serialization (runs only; blocks rebuilt) with full load-time validation; RL-only `BidirSearch` overrides (pruned MEM sweep, lockstep SMEM backward phase, fused `Fanout` children, `ShallowCache` k-mer seeding, batched locate); no LCP contraction |
| `src/move_table/` | `MoveTable`: permutation of `[0,n)` as 16-byte rows `(p, q, idx, head)` + sentinel row; `step` = row read + fast-forward; optional Nishimoto–Tabei balancing; serde validates it is a permutation |
| `src/move_index/` | `MoveLf` (LF move table, sentinel runs split into single rows, walk-then-select run lookup), `MoveLocate` (run-boundary SA samples, balanced `φ`/`φ⁻¹` tables, PLCP samples; toehold enumeration both ways), `MoveFmIndex` (r-index), `BMoveIndex`/`BMoveCursor` (b-move; toehold = text position of one occurrence), `"HMV\x01"`/`"HBM\x01"` |
| `src/occ/rank.rs`, `src/fm_index/half.rs` | `BwtRank` (rank/access trait over `OccTable` and `RlBwt`) and sealed `FmHalf` (index half the `BidirInterval` cursor is generic over; contraction stays `FmIndex`-only) |
| `src/fm_index/prepare.rs` | Construction prefix shared by `FmIndex` and `RlFmIndex` (concatenate, headers, C array, SA, forced-sample seq starts) |
| `src/fm_index/fwd_interval.rs` | `FwdInterval`: forward-only row range — LF step on sub-ranges, LCP parent (ancestor walk), locate |
| `src/fm_index/serialize.rs`, `src/serde_raw.rs` | On-disk format (`HFM` + version byte, v3 current: alphabet tables stored inline, CSR lookup table; v2/v1/legacy readers kept, fixtures under `tests/fixtures/` written by `examples/gen_fixtures.rs`); `serde_raw` codecs store big `Vec<u8/u16/u32/(u32,u32)>` as LE byte blobs so `from_bytes` copies in bulk |
| `src/lcp.rs` | Capped u16 LCP array (Kasai at CPU build) + block minima / sparse table for psv/nsv; one per bidir half, opt-in via `FmIndexConfig::build_lcp` (default off), backs `contract_left` (fwd) / `contract_right` (rev) |
| `src/gpu/` | WebGPU pipeline setup, buffer management, `GpuContext` (process-wide OnceLock cache) |
| `src/suffix_array/`, `src/bwt/`, `src/occ/` | CPU and GPU implementations of each index component |
| `src/wasm/` | `wasm-bindgen` JS/TS bindings |
| `shaders/` | WGSL compute shaders |

### Feature flags
- `cpu` (default): CPU construction and queries
- `gpu`: adds `wgpu`-based GPU acceleration; required for all GPU benchmarks and tests
- `wasm`: adds `wasm-bindgen` JS bindings

### DNA encoding
Alphabet is `{A=1, C=2, G=3, T=4}`. Sentinel `0` marks sequence boundaries. See `src/alphabet.rs`.

### GpuContext caching
`src/gpu/context_cache.rs` holds a process-wide `OnceLock<GpuContext>`. GPU functions accept an optional pre-initialized context; pass `None` on first call, reuse thereafter. Avoids adapter/device init overhead in benchmarks and tests.
