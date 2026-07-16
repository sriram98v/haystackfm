# Count & Locate

The two core exact-match queries. Both operate on IUPAC-encoded byte slices
(`DnaSequence::as_slice()`).

## `count`

```rust
let n = index.count(query.as_slice()); // u32 — number of occurrences
```

Runs a backward search and returns the final SA-interval size — *O(|P|)*, independent of how
many times the pattern occurs.

## `locate`

```rust
let hits = index.locate(query.as_slice()); // Vec<(String, u32)>
```

Returns every occurrence as `(seq_id, offset_within_seq)`, where `seq_id` is the sequence's
name (a `String`) and `offset` is a `u32`. Cost is *O(|P| + occ·k)*, where `occ` is the
number of hits and `k` is `sa_sample_rate` — each hit LF-walks at most *k* steps to the
nearest SA sample.

## Index metadata

```rust
index.text_len();        // u32 — total length of the concatenated text
index.num_sequences();   // u32 — number of indexed sequences
```

## GPU batch locate

When you have many patterns, resolve them together on the GPU. It is IUPAC-aware and returns
results per query:

```rust
use haystackfm::gpu::locate::locate_batch_gpu;

let ctx = haystackfm::gpu::GpuContext::new().await?;
let queries: Vec<&[u8]> = vec![q1.as_slice(), q2.as_slice()];
let hits = locate_batch_gpu(&ctx, &index, &queries).await?;
// hits[i] = Vec<(seq_id, offset_within_seq)> for queries[i]
```

Requires the `gpu` feature. GPU results carry the same occurrences as the CPU `locate` output
(order is not guaranteed), reported as numeric `(seq_id, offset)` indices. For when the GPU
path is worthwhile, see [GPU Acceleration](./gpu.md).
