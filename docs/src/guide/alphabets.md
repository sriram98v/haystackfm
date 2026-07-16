# Custom Alphabets

Matching semantics are pluggable through the `Alphabet` trait (`src/alphabet.rs`). Rather
than carrying a generic type parameter, `FmIndex` and `BidirFmIndex` store a runtime
`AlphabetFns` bundle (function pointers plus a serialization tag), so the index type itself
stays alphabet-agnostic while the match rules are chosen at build time.

## Built-in alphabets

| Alphabet | Behavior |
|----------|----------|
| `IupacDna` (default) | Full 16-symbol IUPAC matching — `N` and other ambiguity codes expand to base-set overlap. Used by `build_cpu` / `build`. |
| `ExactDna` | Only A/C/G/T match themselves; any ambiguity code (including `N`) produces zero hits. Useful for peer-comparable benchmarks where other tools don't treat `N` as a wildcard. |

## Choosing one

The default `build_cpu` / `build` use `IupacDna`. To pick a different alphabet, use the
`_with` constructors:

```rust
use haystackfm::alphabet::ExactDna;
use haystackfm::{FmIndex, BidirFmIndex};

let index = FmIndex::build_cpu_with::<ExactDna>(&seqs, &config)?;
let bidir = BidirFmIndex::build_cpu_with::<ExactDna>(&seqs, &config)?;
```

## Implementing your own

Implement `Alphabet` for a custom type to define your own symbol set and match rules. The
trait carries a safety contract — stable function pointers and a unique serialization tag
(≥ 128) so serialized indices can be matched back to their alphabet. See the trait docs in
[`src/alphabet.rs` on docs.rs](https://docs.rs/haystackfm) for the exact requirements.
