# Architecture

How the pieces fit together, from construction through the GPU query pipelines.

## Construction pipeline

```text
Input: [DnaSequence]
  → concatenate + add sentinels
  → GPU: build suffix array          radix_sort.wgsl + sa_*.wgsl
  → GPU: derive BWT                  bwt_gather.wgsl
  → CPU: compute C array             (trivial histogram, 16 values)
  → GPU: build Occ table             occ_scan.wgsl + prefix_sum.wgsl
  → sample SA at rate k
```

## CPU query pipeline

`O(m)` count, `O(m + occ·k)` locate:

```text
backward_search (IUPAC multi-interval) → SA interval set [lo, hi)
locate: LF-walk from each position to nearest SA sample, fused symbol+rank
        lookup per step (OccTable::lf_step — one block-plane read instead of two)
```

## GPU locate pipeline (2 passes)

```text
Pass 1  locate_search.wgsl   — backward search, IUPAC multi-interval → (match_count, intervals)
Pass 2  locate_resolve.wgsl  — LF-walk to sampled SA position → (seq_id, offset)
```

## GPU MEM / SMEM pipeline (3 passes)

```text
Pass 1  mem_find.wgsl        — bidirectional extension, IUPAC multi-interval → raw SA intervals
Pass 2  mem_resolve.wgsl     — resolve each SA position via LF-walk
Pass 3  ref_map.wgsl         — binary-search reference boundaries → (ref_id, offset)
```

## Key modules

| Path | Role |
|------|------|
| `src/fm_index/` | `FmIndex`, `BidirFmIndex`, backward search, SMEM/MEM logic |
| `src/fm_index/lookup.rs` | `LookupTable` — depth-*k* prefix table seeding `backward_search` in O(1) for core-symbol k-mers |
| `src/alphabet.rs` | IUPAC encoding, `compatible_symbols`, `DnaSequence`, `Alphabet` trait (`IupacDna`, `ExactDna`) |
| `src/gpu/` | WebGPU pipeline setup, buffer management, `GpuContext` |
| `src/gpu/locate.rs` | `locate_batch_gpu` — 2-pass GPU locate |
| `src/gpu/mem_find.rs` | `find_mems_batch_gpu` / `find_smems_batch_gpu` / `find_all_mems_batch_gpu` |
| `src/gpu/mem_resolve.rs` | SA position resolve pass |
| `src/gpu/ref_map.rs` | Reference boundary mapping pass |
| `src/suffix_array/`, `src/bwt/`, `src/occ/` | CPU and GPU implementations of each component |
| `src/wasm/` | `wasm-bindgen` JS/TS bindings |
| `shaders/` | WGSL compute shaders |

## Compact Occ table

No resident `Bwt` is kept after construction. The `OccTable` compacts to the *effective*
symbol alphabet (bitplane-encoded lanes, not the full 16-symbol IUPAC space) and reconstructs
BWT rows/bytes on demand for GPU upload. This keeps the index small while still feeding the
fixed-lane layout the shaders expect.
