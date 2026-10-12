# Run-Length Indexes

A collection of closely related genomes (strains of one species, a pangenome, many
assemblies of one individual) has a BWT made of long runs of equal symbols. The number of
runs `r` grows with the collection's *variation*, not its length `n`. The run-length
indexes store the BWT as runs, so their rank structure is `O(r)` instead of `O(n)`.

| Type | Counterpart | Supports |
|------|-------------|----------|
| [`RlBwt`] | `Bwt` + `OccTable` | rank / access / LF over runs |
| [`RlFmIndex`] | `FmIndex` | `count`, `locate`, `locate_positions`, `locate_rows` |
| [`BidirRlFmIndex`] | `BidirFmIndex` | the `BidirInterval` cursor: `extend_*`, `children_*`, `count_*_in`, `*_compatible`; `count`, `locate_interval` |

They produce **the same SA intervals** as the plain indexes over the same references, and
IUPAC/`ExactDna` matching works the same way.

## Building

```rust,ignore
use haystackfm::{BidirRlFmIndex, DnaSequence, RlBwt, RlFmIndex, RlFmIndexConfig, OccEncoding};

// Just the run-length BWT.
let rlbwt = RlBwt::build_cpu(&seqs, OccEncoding::Bitplane)?;
println!("n = {}, r = {}, n/r = {:.1}", rlbwt.len(), rlbwt.num_runs(), rlbwt.run_ratio());

// Run-length FM-index (count + locate) and its bidirectional form.
let config = RlFmIndexConfig { sa_sample_rate: 32, ..Default::default() };
let rl = RlFmIndex::build_cpu(&seqs, &config)?;
let bidir = BidirRlFmIndex::build_cpu(&seqs, &config)?;

// Or convert an index you already have, without rebuilding the suffix array.
let rl = RlFmIndex::from(&fm_index);
let bidir = BidirRlFmIndex::from(&bidir_fm_index);
```

Other ways to produce an `RlBwt`: `RlBwt::from_bwt(&bwt, enc)`, `RlBwt::from_occ(&occ, enc)`,
and `RlBwt::from_runs(iter_of_(symbol, length), enc)`. `Bwt::runs()` iterates the maximal runs
of a plain BWT.

`sa_sample_rate = 0` builds a **count-only** index, with no SA samples, and its `locate*`
methods panic. The sample bitvector alone costs about 0.22 bytes per base, which on a
repetitive collection is more than the run-length rank structure. The reverse half of a
`BidirRlFmIndex` is always count-only.

## Layout and space

```text
H          run-head string (one symbol per run), with an occ table over it
starts     start row of each run (r + 1 u32s), plus a bucket directory (~r/4 u32s)
cum[c]     per symbol, prefix sums of that symbol's run lengths (r + 16 u32s in total)

rank_c(i) = cum[c][rank_H(c, k)] + (H[k] == c ? i − starts[k] : 0),   k = run containing i
```

That is about **9.6 bytes per run** with the default `Bitplane` head table (`OneHot` adds a few bytes per run). The plain occ table costs about 0.6 bytes per base
(Bitplane), so the run-length structure is smaller once `n / r` is above roughly 16.
Collections of dozens of genomes of one species are usually far above that. A single
genome, or random text, is far below it (`n / r ≈ 1.3` on random DNA), so use `FmIndex` there.

A rank touches the head occ table, the run-start directory and a prefix-sum entry: three
dependent memory reads instead of one. Queries are therefore slower than `FmIndex` when
both fit in cache. The payoff is fitting collections that the plain index cannot.

## Serialization

`RlFmIndex::to_bytes` writes the magic `"HRL\x01"`, and `BidirRlFmIndex::to_bytes` writes
`"HRB\x01"`, independently of the plain index's `"HFM"` format. `from_bytes` validates the
run structure, the head table and the SA samples before returning an index.

## Not yet supported

Compared with the plain indexes, these are missing: the k-mer lookup table, LCP-backed
`contract_*` / `parent`, SMEM/MEM finding, and GPU/WASM paths.

Construction still builds the full suffix array, so peak memory is `O(n)`. Prefix-free
parsing would lower that, and it is future work.

## Move and b-move

The move structure builds on these runs. Each run's LF image is stored as a table row, so
LF is a row read instead of a rank. SA samples at run boundaries (toehold + `φ`) make
locate `O(r)`-space. See [Move Structure & b-move](./move.md) for `MoveFmIndex` and the
bidirectional `BMoveIndex`.

[`RlBwt`]: https://docs.rs/haystackfm/latest/haystackfm/rlbwt/struct.RlBwt.html
[`RlFmIndex`]: https://docs.rs/haystackfm/latest/haystackfm/rl_fm_index/struct.RlFmIndex.html
[`BidirRlFmIndex`]: https://docs.rs/haystackfm/latest/haystackfm/rl_fm_index/struct.BidirRlFmIndex.html
