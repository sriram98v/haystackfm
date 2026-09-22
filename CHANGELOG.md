# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, a breaking change bumps the **minor** version.

## [Unreleased]

### Added
- `BidirInterval::contract_left` / `BidirFmIndex::contract_left`: the inverse of
  `extend_left` (cP → P) without re-walking. One `OccTable::select` on the forward half
  gives ψ of the interval's first row, the forward LCP array widens that sub-range to the
  full interval of P (O(1) when every occurrence of P is preceded by c, O(log n) otherwise
  via block minima + a sparse table), and the existing class rank recovers the reverse
  interval. Works across sentinel-preceded occurrences and IUPAC reference codes.
  `has_lcp()` reports whether contraction is available.
- `LcpArray` (`src/lcp.rs`): Kasai LCP over the SA rows stored capped at `u16::MAX`, with
  `psv_below` / `nsv_below`. Built during CPU construction while the full suffix array is
  resident; `FmIndexConfig::build_lcp` (default `true`) controls it. Forward half only;
  the reverse half of a `BidirFmIndex` never builds one.
- `OccTable::select(c, r)`: position of the r-th occurrence of `c` in the BWT, from the
  existing superblock/block/lane-mask layout plus a small hint array (rebuilt on load,
  not serialized).
- Serialized indexes now start with a 4-byte format marker (`"HFM\x01"`). Legacy blobs
  without it still load, with `has_lcp() == false`.
- `FmIndexError::LcpNotBuilt`, `PatternTooLong`, `InvalidContraction`.
- `SymbolSet`: a 16-bit set of alphabet codes with `EMPTY`, `ALL`, `BASES`, `WILDCARDS`
  (codes 5..=15), `NON_SENTINEL`, `single`, `below`, `from_codes`, set algebra and
  iteration. `AlphabetFns::compatible_set(q)` returns the codes `q` matches under that
  alphabet as a `SymbolSet`.
- `OccTable::rank_set`, `rank_set_pair` and `rank_all`: class ranks over the occurrence
  table. Every block record already holds all lanes, so counting "any symbol in this set"
  costs one block touch regardless of the set's size, and `rank_all` returns every symbol's
  rank at once.
- Wildcard-aware bidirectional cursor operations, on both `BidirInterval` (taking the
  relevant `&FmIndex` half) and `BidirFmIndex`:
  - `count_wild_right` / `count_wild_left`: occurrences followed / preceded in the reference
    by an ambiguity code (`N` or a degenerate IUPAC symbol). Two occ-block touches, no text
    access, independent of how many wildcard codes exist — a cursor walk can ask at every
    step and fan out only when the answer is non-zero.
  - `count_right_in` / `count_left_in`: the same for an arbitrary `SymbolSet`, e.g.
    `compatible_set(base).intersection(SymbolSet::WILDCARDS)`.
  - `children_right` / `children_left`: every child interval for all 16 codes from two
    `rank_all` calls, bit-identical to the per-code `extend_*` results; slot 0 is the
    sentinel child (occurrences at a reference end / start) and child sizes sum to the
    parent's size.
  - `extend_right_compatible` / `extend_left_compatible`: the compatible-symbol fan-out
    `find_smems` / `find_mems` use internally, now public, driven by the index's own
    alphabet; `BidirFmIndex::compatible_set` exposes that alphabet's match set.

### Changed
- **Breaking.** `BidirInterval` gains a `len: u32` field (the matched pattern length,
  maintained by every extension and contraction); struct literals must supply it.
- **Breaking.** `FmIndexConfig` gains `build_lcp: bool` (default `true`); exhaustive
  struct literals must supply it. CPU-built indexes grow by ~2.7 bytes per base on the
  forward half unless it is set to `false`. GPU construction never builds the LCP.
- `BidirInterval::extend_right` / `extend_left` compute the paired-interval offset with a
  single class rank per border instead of one scalar rank per smaller symbol. Results are
  unchanged; extending by a high IUPAC code no longer costs up to 30 extra rank calls.

### Fixed
- `benches/query.rs` compiles again (its `FmIndexConfig` literals predated the
  `lookup_depth` / `build_threads` / `occ_encoding` fields) and gains a `wild_counts` group
  comparing `count_wild_right` against the 11-code `extend_right` fan-out.

## [0.4.0] - 2026-07-28

Breaking behavior change. The public API is unchanged — `cargo semver-checks` reports no
semver update required — but `find_mems` returns materially different results, so this takes
a minor bump under the pre-1.0 policy above rather than the patch bump an API-only check
would allow.

### Fixed
- **Breaking.** `BidirFmIndex::find_mems` now enumerates MEMs in the MUMmer / BWA sense:
  a query interval is reported when **at least one** of its occurrences is maximal in both
  directions *at that occurrence*. It previously required **every** occurrence to be
  maximal, so a single extendable occurrence in any one reference deleted the interval.
  Both the right-maximality test (the forward loop stopped only when the occurrence set went
  empty, discarding occurrences that dropped out mid-extension) and the left-maximality test
  (rejecting the interval if *any* occurrence was left-extendable) were affected.

  Consequences of the old behavior: at most one interval per query start position, and a
  result set that collapsed onto `find_smems`, making MEM-vs-SMEM comparisons through this
  API a guaranteed null result. In a database of near-identical references, per-reference
  match recovery was systematically sparse.

  For `query = ACGTACGTAC` against `ACGTACGTAC` and `ACGTACT` with `min_len = 2`, the result
  goes from `{(0,10)}` to `{(0,2), (0,6), (0,10), (4,10), (8,10)}`.

  `find_smems` is unaffected and unchanged: the containment-maximal MEMs under the new
  definition are exactly the SMEMs it already returned.

### Changed
- **Breaking.** `Mem::match_count` and `Mem::positions` from `find_mems` now cover only the
  maximal occurrences of a match, not every occurrence of the matched substring.
- `find_mems` output size is now bounded by O(|query|²) rather than |query|; `min_len` is the
  only guard. Expect `benches/mem_bench.rs` and `benches/mem_positions_bench.rs` timings to
  move accordingly.

### Known issues
- `find_mems_gpu` still runs the previous whole-set algorithm and is **not** equivalent to
  `find_mems`. The `shaders/mem_find.wgsl` MODE_MEM port is tracked in `KNOWN-ISSUES.md`.

## [0.3.0] - 2026-07-24

Breaking release: the index now retains the indexed text and can serve the bases of any
sequence, so callers no longer need to keep their own copy of every reference.

### Added
- `FmIndex::sequence(SeqId)` / `BidirFmIndex::sequence(SeqId)` — the bases of one indexed
  sequence as a borrowed slice, in O(1), with the trailing sentinel excluded. Returns
  alphabet codes (`A = 1`, `C = 2`, …), not ASCII; use `decode_char` to render them.
- `FmIndex::sequence_by_header(&str)` / `BidirFmIndex::sequence_by_header(&str)` — the same
  by header, equivalent to `seq_id(h).and_then(|id| self.sequence(id))`.
- `encode_byte`, `encode_char` and `decode_char` re-exported at the crate root, so callers
  can move queries into (and results out of) the alphabet's code space without reaching
  into the `alphabet` module.

### Changed
- **Breaking.** The serialized layout gains a `text` field, so indexes written by 0.2.0 and
  earlier are rejected by `from_bytes` and must be rebuilt.
- The concatenated text is no longer dropped after BWT construction. This costs ~n bytes of
  resident and serialized size and forgoes a build-time peak-memory reduction, in exchange
  for random-access substrings. An FM-index can otherwise only recover text by LF-walking
  backwards one symbol at a time — far too slow for a caller rescoring a read against a
  candidate diagonal, which is the case this exists to serve.
- Only the forward half of a `BidirFmIndex` retains text. The reverse half's copy is a
  redundant reversal that is never served, so it is released at build time, keeping the
  overhead at ~n rather than ~2n.

## [0.2.0] - 2026-07-24

Breaking release: match locations are now reported by integer sequence id instead of by
FASTA header string.

### Changed
- **Breaking.** Queries report match locations as `(SeqId, offset)` rather than
  `(String, offset)`, so no header string is allocated per occurrence — the cost was per
  occurrence rather than per seed, and scaled with seed multiplicity. Affects
  `FmIndex::locate`, `FmIndex::locate_gpu`, `FmIndex::map_position`,
  `BidirFmIndex::locate_interval`, `Mem::positions` and `MemHit::positions`, and the WASM
  `locate`. Callers that need labels resolve ids through `seq_header()`, or build an
  `id -> label` table once from `seq_headers()` and index it by `SeqId::index()`.
- **Breaking.** Sequence headers must now be unique; a collision fails the build with
  `FmIndexError::DuplicateHeader`. This is what makes `seq_id` an exact inverse of
  `seq_header`. Sequences supplied without a header are still auto-named `seq_{i}`, so this
  only fires on genuinely repeated names.

### Added
- `SeqId`, a stable 0-based identifier for an indexed reference. Assigned in build order and
  preserved across `to_bytes` / `from_bytes`.
- Sequence-id accessors on `FmIndex` and `BidirFmIndex` — `seq_headers()`, `seq_header(id)`
  and `seq_id(header)`. Both directions are O(1), backed by a header map built at
  construction and rebuilt on deserialization (it is derived, so it is not serialized).
- `FmIndexError::DuplicateHeader`.
- WASM bindings for the accessors: `seq_header`, `seq_id`, `seq_headers`.

The on-disk index format is unchanged — a `SeqId` is the position in the already-serialized
header list, and the header map is rebuilt on load rather than stored — so indexes written by
0.1.0 deserialize under 0.2.0, with one exception: an index built by 0.1.0 from *duplicate*
headers is now rejected by `from_bytes` with `DuplicateHeader`, since 0.1.0 permitted
collisions that 0.2.0 does not. Rebuild those indexes with unique headers.

## [0.1.0] - 2026-07-16

First release under the `haystackfm` name. The project was previously published as
`webgpu-fmidx` (versions 0.1.0–0.5.1); version numbering restarts at 0.1.0 under the new
name, and its earlier history is not carried over here.

### Added
- GPU-accelerated FM-index construction (suffix array, BWT, Occ table) via WebGPU compute
  shaders, alongside CPU construction.
- `count` / `locate` queries; bidirectional index with MEM / SMEM finding (CPU and GPU paths).
- Full 16-symbol IUPAC ambiguity alphabet with a pluggable `Alphabet` trait
  (`IupacDna` default, `ExactDna` for exact ACGT matching).
- WASM bindings for in-browser WebGPU use; index serialization (`to_bytes` / `from_bytes`).
- Community health files, CI (fmt / clippy / build / test on `--all-features`), and Dependabot.

### Changed
- Licensed under Apache-2.0.

[Unreleased]: https://github.com/sriram98v/haystackfm/commits/main
[0.4.0]: https://crates.io/crates/haystackfm/0.4.0
[0.1.0]: https://crates.io/crates/haystackfm/0.1.0
