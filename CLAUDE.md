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
| `src/fm_index/` | `FmIndex`, `BidirFmIndex`, backward search, SMEM/MEM logic |
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
