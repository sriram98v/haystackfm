# API Reference

The complete, always-current type-level API reference is generated from the source by
rustdoc and hosted on docs.rs:

### 👉 [docs.rs/haystackfm](https://docs.rs/haystackfm)

There you'll find every public type, trait, and function with its signatures and doc
comments, including:

- `FmIndex` — `build_cpu`, `build_cpu_with`, `build`, `count`, `locate`, `to_bytes`,
  `from_bytes`, the id/header accessors `seq_headers`, `seq_header`, `seq_id`, and the
  base accessors `sequence`, `sequence_by_header`.
- `BidirFmIndex` — `build_cpu`, `build_cpu_with`, `find_mems`, `find_smems`, the GPU
  variants, and the same id/header and base accessors. The cursor API: `full_interval`,
  `extend_right` / `extend_left`, `children_right` / `children_left`,
  `extend_right_compatible` / `extend_left_compatible`, the class counts
  `count_wild_right` / `count_wild_left` and `count_right_in` / `count_left_in`,
  `compatible_set`, `count_interval`, `locate_interval`.
- `BidirInterval` — the paired SA interval a cursor walks; the same operations taking the
  forward or reverse `FmIndex` half explicitly.
- `SeqId` — a reference's stable 0-based id, reported by every query in place of its FASTA
  header. See [Sequence ids vs. headers](../guide/concepts.md#sequence-ids-vs-headers).
- `Mem` / `MemHit` — result types for MEM/SMEM finding.
- `FmIndexConfig` — construction knobs.
- `alphabet` — the `Alphabet` trait, `IupacDna`, `ExactDna`, `DnaSequence`,
  `compatible_symbols`, and `SymbolSet` (a set of codes; `SymbolSet::WILDCARDS` is every
  ambiguity code).
- `gpu` — `GpuContext`, `locate_batch_gpu`, and the GPU MEM/SMEM functions (behind the
  `gpu` feature).

This guide covers the *how* and *why*; docs.rs is the exhaustive *what*.
