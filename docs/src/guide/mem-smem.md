# MEM / SMEM Finding

Exact-match queries answer "is this whole pattern present?". Seeding a read aligner instead
needs the *longest exact stretches* shared between a query and the reference — Maximal Exact
Matches (MEMs) and Super-Maximal Exact Matches (SMEMs). These require the bidirectional index,
`BidirFmIndex`.

## Build a bidirectional index

```rust
use haystackfm::{DnaSequence, BidirFmIndex, FmIndexConfig};

let refs = vec![DnaSequence::from_str("ACGTACGTACGT")?];
let config = FmIndexConfig::default();
let bidir = BidirFmIndex::build_cpu(&refs, &config)?;
```

## CPU MEM / SMEM

```rust
let query = DnaSequence::from_str("ACGT")?;

// Vec<Mem>. `min_len` filters short matches; `locate` resolves positions.
let smems = bidir.find_smems(query.as_slice(), /*min_len=*/18, /*locate=*/true);
let mems  = bidir.find_mems(query.as_slice(),  /*min_len=*/18, /*locate=*/true);
```

Both are IUPAC-aware (`N` matches any of A/C/G/T). Passing `locate = false` skips position
resolution and leaves `Mem::positions` empty — cheaper when you only need match extents and
counts.

### `Mem`

```rust
pub struct Mem {
    pub query_start: usize,              // 0-based inclusive
    pub query_end:   usize,              // 0-based exclusive
    pub match_count: u32,                // number of occurrences
    pub positions:   Vec<(String, u32)>, // (seq name, offset) — empty when locate=false
}
```

## GPU MEM / SMEM

For batches of queries, run the GPU pipeline. Queries are passed as `&[DnaSequence]`, and
reference boundaries come from the index:

```rust
# #[cfg(feature = "gpu")]
# async fn run(bidir: &haystackfm::BidirFmIndex, query: haystackfm::DnaSequence) -> Result<(), Box<dyn std::error::Error>> {
let boundaries = bidir.seq_boundaries();     // reference-sequence boundaries
let queries = [query.clone()];               // &[DnaSequence]

// Vec<Vec<MemHit>> — the GPU context is drawn from a process-wide cache.
let smem_hits = bidir.find_smems_gpu(&queries, /*min_len=*/18, boundaries, /*max_hits_per_mem=*/1024).await?;
let mem_hits  = bidir.find_mems_gpu(&queries,  18, boundaries, 1024).await?;
// smem_hits[query_i] = Vec<MemHit> with resolved reference positions
# Ok(()) }
```

### `MemHit`

```rust
pub struct MemHit {                   // GPU result type
    pub query_start: u32,
    pub query_end:   u32,
    pub match_count: u32,
    pub positions:   Vec<(u32, u32)>, // (ref_id, offset_within_ref)
    pub truncated:   bool,            // true if positions capped at max_hits_per_mem
}
```

`max_hits_per_mem` caps how many positions each MEM resolves; when a MEM has more occurrences
than the cap, `truncated` is set. GPU results are parity-tested against the CPU
`find_mems` / `find_smems` output.
