# Move Structure & b-move

The move-structure indexes are the fast, `O(r)`-space end of the run-length family. Here `r` is
the number of BWT runs.

| Type | Counterpart | Supports |
|------|-------------|----------|
| `MoveFmIndex` | `FmIndex` | `count`, `locate`, `locate_positions` (r-index: toehold + `φ`) |
| `BMoveIndex` + `BMoveCursor` | `BidirFmIndex` + `BidirInterval` | `extend_*`, `children_*`, `count_*_in`, `count_wild_*`, `*_compatible`, `locate(&cursor)`, `count` |
| `MoveTable` | — | the move table under both (inspection only: `domain`, `num_rows`, `heap_size_bytes`) |

They produce the **same SA intervals** as the plain indexes, and IUPAC/`ExactDna` matching
works the same way. `BMoveCursor::interval()` converts a cursor to a `BidirInterval`.

## Building and querying

```rust,ignore
use haystackfm::{BMoveIndex, MoveFmIndex, MoveFmIndexConfig};

let mv = MoveFmIndex::build_cpu(&seqs, &MoveFmIndexConfig::default())?;
mv.count(&pattern);
mv.locate(&pattern);                   // Vec<(SeqId, u32)>, unordered

let bm = BMoveIndex::build_cpu(&seqs, &MoveFmIndexConfig::default())?;
let cur = bm.full_cursor();
let cur = bm.extend_right(&cur, code).and_then(|c| bm.extend_left(&c, code2));
if let Some(cur) = cur {
    let hits = bm.locate(&cur);        // from the cursor's toehold, no SA samples per row
}
```

`MoveFmIndexConfig { locate: false, .. }` builds a count-only index, which is about 4× smaller.
Its `locate` methods panic.

## How it works

**LF as a table step.** Each BWT run maps under LF to a contiguous range. The index stores
one row per run: the run start `p`, its image `q = LF(p)`, the row `idx` containing `q`, and
the run symbol. LF of a position in row `k` is `q_k + (i − p_k)`. Its row is found by
starting at `idx_k` and scanning forward a few rows, so no rank query is needed. A search
step finds the first and last row holding the extension symbol inside the interval. It
walks a few rows and falls back to rank/select over the run heads. Then it steps both ends.

**Locate in `O(r)` space.** SA values are stored only at the first and last row of each
run. Every extension updates a *toehold*: the text position of one occurrence of the
current pattern.
- It usually comes from a sampled row inside the interval that holds the extension symbol.
- When the interval lies inside a single run, it is carried from the previous toehold.

A bidirectional toehold can sit anywhere in its interval. Locate therefore walks both ways:
- `φ` (the previous row's SA value) upward, while the permuted LCP shows the previous
  suffix still starts with the pattern;
- `φ⁻¹` downward, on the same condition.

`φ`, `φ⁻¹` and the permuted LCP are piecewise linear over `O(r)` pieces. They are stored
as balanced move tables with one LCP value per piece. The design follows the r-index
(Gagie, Navarro & Prezza), bidirectional toeholds (Arakawa, Navarro & Sadakane) and b-move
(Depuydt et al.).

**Sentinels.** All sequence ends share one sentinel code, which makes LF ambiguous where
it wraps around from text position 0.
- Toeholds only step through non-sentinel symbols.
- Every sentinel row is its own move-table row, so it is sampled. This keeps `φ` and the
  permuted LCP correct at sequence boundaries.
- A b-move cursor extended *by* the sentinel still gives correct intervals, but it cannot
  be located.

## Space and speed

These figures come from `benches/move_query.rs`: 400 kbp inputs, 16-byte move rows, u32 values.

| Input | n / r | `FmIndex` | `RlFmIndex` | `MoveFmIndex` (LF only) |
|-------|-------|-----------|-------------|-------------------------|
| random | 1.3 | ~2.2 B/base (occ 0.89, SA samples 0.35, text 1.0) | 7.5 | 48.7 (12.7) |
| 40 × 10 kbp genome, 0.1% SNPs | 41.6 | ~2.2 | 0.58 | **1.78 (0.41)** |

Per BWT run, the LF table costs about 17 bytes. The locate structures cost about 57 more:
run samples, `φ` and `φ⁻¹` with their balancing splits, and PLCP samples. Against
`FmIndex` the break-even is around `n / r ≈ 35`. Pangenome collections are usually far
above that.

On the repetitive input:
- **locate** is about **6× faster** than `FmIndex` (31 µs vs 191 µs for 16 patterns). `φ`
  steps replace LF walks to the nearest SA sample.
- **count** is about 1.6× slower than `FmIndex`, and 1.6× faster than `RlFmIndex`.
- **b-move cursor extension** is about 1.75× slower than `FmIndex`.

These inputs fit in cache, which favours the plain occ table. On indexes large enough to be
DRAM-bound, a move step's single row read is the advantage.

## Serialization

The magic is `"HMV\x01"` for `MoveFmIndex` and `"HBM\x01"` for `BMoveIndex`. Loading
validates every table before a query can touch it:
- each move table is a permutation, with correct row pointers;
- the LF rows are consistent with the C array;
- sample, `φ` and PLCP sizes match the tables, and PLCP bounds are checked.

## Not yet supported

- Bit-packed rows.
- SMEM/MEM finding on b-move.
- Cursor contraction.
- Prefix-free-parsing construction: the build is still `O(n)` memory.
- GPU/WASM.
