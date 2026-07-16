# Quick Start

The fastest path: build an index on the CPU and run a couple of queries.

## Build and query (CPU)

```rust
use haystackfm::{DnaSequence, FmIndex, FmIndexConfig};

let seq = DnaSequence::from_str("ACGTACGT")?;
let config = FmIndexConfig::default();
let index = FmIndex::build_cpu(&[seq], &config)?;

// A DnaSequence holds IUPAC-encoded bytes; encode the query the same way.
let query = DnaSequence::from_str("ACGT")?;

assert_eq!(index.count(query.as_slice()), 2);
let positions = index.locate(query.as_slice()); // Vec<(String, u32)> — (seq name, offset)
```

`count` returns how many times the pattern occurs; `locate` returns the `(seq_id, offset)`
of each occurrence. See [Count & Locate](../guide/count-locate.md) for the full surface.

## Build on the GPU

GPU construction is async and returns the same `FmIndex` type:

```rust
use haystackfm::{DnaSequence, FmIndex, FmIndexConfig};
use haystackfm::gpu::locate::locate_batch_gpu;

let seqs = vec![DnaSequence::from_str("ACGTACGT")?];
let config = FmIndexConfig::default();

let index = FmIndex::build(&seqs, &config).await?;   // async GPU build

// Batched, IUPAC-aware GPU locate
let ctx = haystackfm::gpu::GpuContext::new().await?;
let query = DnaSequence::from_str("ACGT")?;
let queries: Vec<&[u8]> = vec![query.as_slice()];
let hits = locate_batch_gpu(&ctx, &index, &queries).await?;
// hits[i] = Vec<(seq_id, offset_within_seq)>
```

Requires the `gpu` feature. See [GPU Acceleration](../guide/gpu.md).

## Prefer the browser?

The [live demo](../demo.md) does all of this — build (CPU or GPU), search, serialize —
without leaving the page. It's the quickest way to see haystackfm run.

## Configuration

`FmIndexConfig::default()` is a sensible starting point. The knobs that matter most:

| Field | Meaning |
|-------|---------|
| `sa_sample_rate` | Suffix-array sampling rate *k*. Higher = smaller index, slower `locate`. |
| `use_gpu` | Whether construction offloads to the GPU. |
| `lookup_depth` | Depth-*k* prefix lookup table that seeds `backward_search` (0 disables). |
| `occ_encoding` | Occ-table encoding (`Bitplane` / `OneHot`). |

```rust
let config = FmIndexConfig {
    sa_sample_rate: 4,
    ..Default::default()
};
```
