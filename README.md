# webgpu-fmidx

A GPU-accelerated FM-index library for DNA sequence alignment. Runs on native targets (Vulkan / Metal / DX12) via **wgpu** and compiles to **WebAssembly** for in-browser WebGPU use.

Supports the full **16-symbol IUPAC ambiguity alphabet** (A C G T N R Y S W K M B D H V) in both the CPU and GPU query paths, with a pluggable `Alphabet` trait for swapping in custom matching semantics (e.g. exact-match ACGT-only for peer-comparable benchmarks).

---

## Features

| Feature | Description |
|---------|-------------|
| CPU construction | BWT, suffix array, and Occ table built on CPU |
| GPU construction | All three steps GPU-accelerated via WGSL compute shaders |
| `count` / `locate` | Exact-match and position queries (CPU, O(m) and O(m + occ·k)) |
| GPU batch locate | IUPAC-aware backward search + SA resolve on GPU |
| MEM / SMEM finding | Bidirectional FM-index; CPU and GPU paths |
| GPU MEM pipeline | 3-pass: find intervals → resolve SA positions → map to references |
| IUPAC ambiguity | Full 16-symbol alphabet in query and reference; GPU parity-tested |
| Pluggable `Alphabet` trait | Swap matching semantics (`IupacDna` default, `ExactDna` ACGT-only) via `build_cpu_with::<A>` |
| Lookup-table seeding | Optional depth-k prefix table (`FmIndexConfig::lookup_depth`) skips the first k `backward_search` steps |
| WASM bindings | `wasm-bindgen` JS/TS API for browser use |
| Serialization | `to_bytes` / `from_bytes` for index persistence |
| Compact Occ table | Bitplane-encoded lanes over the effective symbol alphabet (not the full 16-symbol IUPAC space) — no resident BWT kept after construction |

---

## Quick Start

### Native — CPU only

```bash
cargo add webgpu-fmidx
```

```rust
use webgpu_fmidx::{DnaSequence, FmIndex, FmIndexConfig};

let seq = DnaSequence::from_str("ACGTACGT")?;
let config = FmIndexConfig::default();
let index = FmIndex::build_cpu(&[seq], &config)?;

// Encode a query pattern (a DnaSequence holds the IUPAC-encoded bytes).
let query = DnaSequence::from_str("ACGT")?;
assert_eq!(index.count(query.as_slice()), 2);
let positions = index.locate(query.as_slice()); // Vec<(seq_id, offset)>
```

### Native — GPU (Vulkan / Metal / DX12)

```bash
cargo add webgpu-fmidx --features gpu
```

```rust
use webgpu_fmidx::{DnaSequence, FmIndex, FmIndexConfig};
use webgpu_fmidx::gpu::locate::locate_batch_gpu;

let seqs = vec![DnaSequence::from_str("ACGTACGT")?];
let config = FmIndexConfig::default();

// Async GPU build
let index = FmIndex::build(&seqs, &config).await?;

// GPU batch locate (IUPAC-aware)
let ctx = webgpu_fmidx::gpu::GpuContext::new().await?;
let query = DnaSequence::from_str("ACGT")?;
let queries: Vec<&[u8]> = vec![query.as_slice()];
let hits = locate_batch_gpu(&ctx, &index, &queries).await?;
// hits[i] = Vec<(seq_id, offset_within_seq)>
```

### MEM / SMEM finding

```rust
use webgpu_fmidx::{DnaSequence, BidirFmIndex, FmIndexConfig};

let refs = vec![DnaSequence::from_str("ACGTACGTACGT")?];
let config = FmIndexConfig::default();
let bidir = BidirFmIndex::build_cpu(&refs, &config)?;

let query = DnaSequence::from_str("ACGT")?;

// CPU — returns Vec<Mem>
let smems = bidir.find_smems(query.as_slice(), /*min_len=*/18, /*locate=*/true);
let mems  = bidir.find_mems(query.as_slice(),  /*min_len=*/18, /*locate=*/true);

// GPU batch — returns Vec<Vec<MemHit>>. The GPU context is drawn from a
// process-wide cache, so no `GpuContext` argument is passed.
#[cfg(feature = "gpu")]
{
    let boundaries = bidir.seq_boundaries();      // reference-sequence boundaries
    let queries = [query.clone()];                // &[DnaSequence]
    let smem_hits = bidir.find_smems_gpu(&queries, /*min_len=*/18, boundaries, /*max_hits_per_mem=*/1024).await?;
    let mem_hits  = bidir.find_mems_gpu(&queries,  18, boundaries, 1024).await?;
    // smem_hits[query_i] = Vec<MemHit> with resolved reference positions
}
```

### WebAssembly (browser WebGPU)

```bash
cargo add webgpu-fmidx --features wasm
wasm-pack build --target web --features wasm
```

```typescript
import init, { FmIndexBuilder, FmIndexHandle } from "./pkg/webgpu_fmidx.js";

await init();

const builder = new FmIndexBuilder(/*sa_sample_rate=*/32);
builder.add_fasta(`>seq1\nACGTACGT\n>seq2\nTGCATGCA`);

const handle = await builder.build_gpu();    // or builder.build_cpu()

console.log(handle.count("ACGT"));           // number of occurrences
console.log(handle.locate("ACGT"));          // Array of [seqId, offset] pairs
console.log(handle.text_len());              // total text length
console.log(handle.num_sequences());         // number of indexed sequences

const bytes = handle.to_bytes();             // Uint8Array — serialize
const restored = FmIndexHandle.from_bytes(bytes);
```

---

## Rust API

### `FmIndex`

```rust
// Build
FmIndex::build_cpu(sequences, config)?          // sync, CPU, IupacDna alphabet (default)
FmIndex::build_cpu_with::<ExactDna>(sequences, config)?  // sync, CPU, custom alphabet
FmIndex::build(sequences, config).await?        // async, GPU (feature = "gpu")

// Query
index.count(pattern)                            // u32 — number of occurrences
index.locate(pattern)                           // Vec<(seq_id, offset)>
index.text_len()                                // u32
index.num_sequences()                           // u32

// Persistence
index.to_bytes()?                               // Vec<u8>
FmIndex::from_bytes(bytes)?                     // FmIndex
```

### `BidirFmIndex`

```rust
BidirFmIndex::build_cpu(sequences, config)?               // IupacDna alphabet (default)
BidirFmIndex::build_cpu_with::<ExactDna>(sequences, config)?  // custom alphabet

// CPU MEM finding — IUPAC-aware; N matches any of A/C/G/T
bidir.find_smems(query, min_len, locate)        // Vec<Mem>
bidir.find_mems(query, min_len, locate)         // Vec<Mem>

// GPU MEM finding (feature = "gpu"); `queries: &[DnaSequence]`,
// `ref_boundaries` from `bidir.seq_boundaries()`, GPU context from the cache
bidir.find_smems_gpu(queries, min_len, ref_boundaries, max_hits_per_mem).await?  // Vec<Vec<MemHit>>
bidir.find_mems_gpu(queries,  min_len, ref_boundaries, max_hits_per_mem).await?  // Vec<Vec<MemHit>>
```

### `Mem` / `MemHit`

```rust
pub struct Mem {
    pub query_start: usize,       // 0-based inclusive
    pub query_end:   usize,       // 0-based exclusive
    pub match_count: u32,         // SA interval size
    pub positions:   Vec<(u32, u32)>, // (seq_id, offset) — empty when locate=false
}

pub struct MemHit {               // GPU result type
    pub query_start: u32,
    pub query_end:   u32,
    pub match_count: u32,
    pub positions:   Vec<(u32, u32)>, // (ref_id, offset_within_ref)
    pub truncated:   bool,        // true if positions capped at max_hits_per_mem
}
```

---

## IUPAC Ambiguity

Queries and references may contain any of the 16 IUPAC symbols:

| Code | Bases | Code | Bases |
|------|-------|------|-------|
| A | A | N | A C G T |
| C | C | R | A G |
| G | G | Y | C T |
| T | T | S | G C |
| | | W | A T |
| | | K | G T |
| | | M | A C |
| | | B | C G T |
| | | D | A G T |
| | | H | A C T |
| | | V | A C G |

Two symbols match when their base sets share at least one nucleotide. Both the CPU (`compatible_symbols`) and GPU (WGSL `COMPAT` table) paths use the same lookup, and parity tests enforce they stay in sync.

---

## Alphabet Trait

Matching semantics are pluggable via the `Alphabet` trait (`src/alphabet.rs`). `FmIndex` and `BidirFmIndex` store a runtime `AlphabetFns` bundle (function pointers + a serialization tag) rather than a generic type parameter, so the index type itself stays alphabet-agnostic.

| Alphabet | Behavior |
|----------|----------|
| `IupacDna` (default) | Full 16-symbol IUPAC matching — `N` and other ambiguity codes expand to base-set overlap. Used by `build_cpu` / `build`. |
| `ExactDna` | Only A/C/G/T match themselves; any ambiguity code (including `N`) produces zero hits. Useful for peer-comparable benchmarks where other tools don't treat `N` as a wildcard. |

```rust
use webgpu_fmidx::alphabet::ExactDna;

let index = FmIndex::build_cpu_with::<ExactDna>(&seqs, &config)?;
```

Implement `Alphabet` for a custom type to define your own symbol set and match rules — see the trait docs in `src/alphabet.rs` for the safety contract (stable function pointers, unique serialization tag ≥ 128).

---

## Architecture

```
Construction pipeline
  ┌───────────────────────────────────────┐
  │  Input: [DnaSequence]                 │
  │  → concatenate + add sentinels        │
  │  → GPU: build suffix array            │  radix_sort.wgsl + sa_*.wgsl
  │  → GPU: derive BWT                    │  bwt_gather.wgsl
  │  → CPU: compute C array               │  (trivial histogram, 16 values)
  │  → GPU: build Occ table               │  occ_scan.wgsl + prefix_sum.wgsl
  │  → sample SA at rate k                │
  └───────────────────────────────────────┘

CPU query pipeline  (O(m) count / O(m + occ·k) locate)
  backward_search (IUPAC multi-interval) → SA interval set [lo, hi)
  locate: LF-walk from each position to nearest SA sample, fused symbol+rank
          lookup per step (`OccTable::lf_step` — one block-plane read instead
          of two)

GPU locate pipeline  (2 passes)
  Pass 1  locate_search.wgsl   — backward search, IUPAC multi-interval → (match_count, intervals)
  Pass 2  locate_resolve.wgsl  — LF-walk to sampled SA position → (seq_id, offset)

GPU MEM/SMEM pipeline  (3 passes)
  Pass 1  mem_find.wgsl        — bidirectional extension, IUPAC multi-interval → raw SA intervals
  Pass 2  mem_resolve.wgsl     — resolve each SA position via LF-walk
  Pass 3  ref_map.wgsl         — binary-search reference boundaries → (ref_id, offset)
```

### Key modules

| Path | Role |
|------|------|
| `src/fm_index/` | `FmIndex`, `BidirFmIndex`, backward search, SMEM/MEM logic |
| `src/fm_index/lookup.rs` | `LookupTable` — depth-k prefix table seeding `backward_search` in O(1) for core-symbol k-mers |
| `src/alphabet.rs` | IUPAC encoding, `compatible_symbols`, `DnaSequence`, `Alphabet` trait (`IupacDna`, `ExactDna`) |
| `src/gpu/` | WebGPU pipeline setup, buffer management, `GpuContext` |
| `src/gpu/locate.rs` | `locate_batch_gpu` — 2-pass GPU locate |
| `src/gpu/mem_find.rs` | `find_mems_batch_gpu` / `find_smems_batch_gpu` / `find_all_mems_batch_gpu` |
| `src/gpu/mem_resolve.rs` | SA position resolve pass |
| `src/gpu/ref_map.rs` | Reference boundary mapping pass |
| `src/suffix_array/`, `src/bwt/`, `src/occ/` | CPU and GPU implementations. No resident `Bwt` is kept after construction — `OccTable` compacts to the effective symbol alphabet (bitplane-encoded lanes) and reconstructs BWT rows/bytes on demand for GPU upload |
| `src/wasm/` | `wasm-bindgen` JS/TS bindings |
| `shaders/` | WGSL compute shaders |

### `GpuContext` caching

`src/gpu/context_cache.rs` holds a process-wide `OnceLock<GpuContext>`. GPU functions accept an optional pre-initialized context; pass `None` on first call and reuse thereafter to avoid adapter/device init overhead in benchmarks and tests.

---

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `cpu` | yes | CPU construction and queries |
| `gpu` | no | WebGPU-accelerated construction and queries via `wgpu` |
| `wasm` | no | `wasm-bindgen` JS/TS bindings |

---

## Browser Compatibility

Requires a browser with **WebGPU** support:

| Browser | Support |
|---------|---------|
| Chrome 113+ | Stable |
| Edge 113+ | Stable |
| Safari 18+ | Stable |
| Firefox | Behind flag |

---

## Benchmarks

```bash
cargo bench --features gpu --bench locate_bench
cargo bench --features gpu --bench mem_bench
cargo bench --features gpu --bench mem_positions_bench
```

---

## License

MIT
