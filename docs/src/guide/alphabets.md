# Custom Alphabets

Matching semantics are pluggable through the `Alphabet` trait (`src/alphabet.rs`). Rather
than carrying a generic type parameter, `FmIndex` and `BidirFmIndex` store a runtime
`AlphabetFns` value — one `SymbolSet` of compatible reference codes per query code, the core
(unambiguous) symbol set the lookup table is built over, and a tag — so the index type itself
stays alphabet-agnostic while the match rules are chosen at build time. Every query path
expands a query code through that mask (`AlphabetFns::compatible`), and the tables are
written into the serialized index, so a custom alphabet loads back exactly as built.

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

Implement `Alphabet` for a custom type to define your own symbol set and match rules: build
the value with `AlphabetFns::new` (explicit masks) or `AlphabetFns::from_compatible_fn` (a
`fn(u8) -> &'static [u8]` evaluated once per code). The trait's contract is that `fns()`
returns an equal value every call and that the tag is ≥ 128 and unique within a program;
tags 2..=127 are reserved and rejected by `from_bytes`. Fan-out order at query time is
ascending code order regardless of how the function lists its codes. See the trait docs in
[`src/alphabet.rs` on docs.rs](https://docs.rs/haystackfm) for the exact requirements.

Custom alphabets are CPU-only: GPU construction and queries always use `IupacDna`.
