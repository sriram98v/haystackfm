# Concepts

A quick tour of the moving parts, so the rest of the guide has names to hang on.

## The building blocks

An FM-index is assembled from a few classic components:

- **Suffix Array (SA)** — the sorted order of all suffixes of the text. It's the backbone
  that turns "where does *P* occur" into a contiguous range.
- **Burrows–Wheeler Transform (BWT)** — a reversible permutation of the text derived from
  the SA. It clusters similar contexts together, which is what makes the index compressible
  and searchable.
- **C array** — for each symbol, the number of text characters that sort strictly before it.
  A tiny histogram (16 values for the IUPAC alphabet).
- **Occ table** — *rank* support: `Occ(c, i)` = how many times symbol `c` appears in the
  BWT up to position `i`. This is the hot data structure every query step touches.

Together, the **LF-mapping** (`C[c] + Occ(c, i)`) lets a backward search walk the text one
symbol at a time, shrinking an SA interval until it exactly covers every occurrence of the
pattern.

## Backward search

`count` and `locate` both start with a backward search: process the pattern right-to-left,
and at each step apply LF-mapping to narrow the `[lo, hi)` SA interval. When the interval is
empty the pattern is absent; otherwise its size is the occurrence count.

Because haystackfm is IUPAC-aware, a single query symbol can match several BWT symbols, so
the search tracks a **union of intervals** rather than one — see
[IUPAC Ambiguity](./iupac.md).

## Locate

`count` only needs the final interval size. `locate` must turn each SA-interval position
into a text coordinate. Storing the whole SA would defeat compression, so haystackfm samples
it every *k* positions (`sa_sample_rate`) and **LF-walks** from an unsampled position until
it lands on a sample — trading a little query time for a much smaller index.

## Bidirectional index

Maximal Exact Match (MEM) and Super-Maximal Exact Match (SMEM) finding need to extend a
match in *both* directions. That requires a **bidirectional** FM-index — `BidirFmIndex` —
which maintains synchronized intervals over the forward and reverse texts. See
[MEM / SMEM Finding](./mem-smem.md).

## Where the GPU comes in

Construction (SA, BWT, Occ) and the query hot loops are all data-parallel, which maps well
to GPU compute. haystackfm implements each stage as WGSL compute shaders and keeps the CPU
implementation as the correctness oracle — every GPU path is parity-tested against it. The
full pipeline is laid out in [Architecture](../reference/architecture.md).
