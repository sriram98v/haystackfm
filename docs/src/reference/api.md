# API Reference

The complete, always-current type-level API reference is generated from the source by
rustdoc and hosted on docs.rs:

### 👉 [docs.rs/haystackfm](https://docs.rs/haystackfm)

There you'll find every public type, trait, and function with its signatures and doc
comments, including:

- `FmIndex` — `build_cpu`, `build_cpu_with`, `build`, `count`, `locate`, `to_bytes`,
  `from_bytes`.
- `BidirFmIndex` — `build_cpu`, `build_cpu_with`, `find_mems`, `find_smems`, and the GPU
  variants.
- `Mem` / `MemHit` — result types for MEM/SMEM finding.
- `FmIndexConfig` — construction knobs.
- `alphabet` — the `Alphabet` trait, `IupacDna`, `ExactDna`, `DnaSequence`, and
  `compatible_symbols`.
- `gpu` — `GpuContext`, `locate_batch_gpu`, and the GPU MEM/SMEM functions (behind the
  `gpu` feature).

This guide covers the *how* and *why*; docs.rs is the exhaustive *what*.
