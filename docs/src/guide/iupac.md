# IUPAC Ambiguity

Queries and references may contain any of the 16 IUPAC nucleotide symbols. Two symbols
**match** when their base sets share at least one nucleotide.

| Code | Bases | Code | Bases |
|------|-------|------|-------|
| A | A | N | A C G T |
| C | C | R | A G |
| G | G | Y | C T |
| T | T | S | G C |
| | | W | A T |
| | | K | G T |
| | | M | A C |
| | | B | C G T |
| | | D | A G T |
| | | H | A C T |
| | | V | A C G |

For example, a query `N` matches any base; a query `R` matches `A` or `G` (and any ambiguity
code whose base set includes one of them).

## Wildcards in the reference

A reference database often carries ambiguity codes of its own (short runs of `R`, `Y`, `M`,
`W`, …). `find_smems` / `find_mems` already let a read base match them. If you drive the
bidirectional cursor yourself, ask at each step whether any occurrence continues into a
wildcard before paying for the per-code fan-out:

```rust,ignore
use haystackfm::{alphabet, SymbolSet};

let mut iv = bidir.full_interval();
for &base in read {
    // Cheap: two occurrence-table touches, independent of how many wildcard codes exist.
    let wild_here = bidir.count_right_in(&iv, bidir.compatible_set(base) & SymbolSet::WILDCARDS);
    if wild_here == 0 {
        iv = match bidir.extend_right(iv, base) { Some(next) => next, None => break };
    } else {
        // Some occurrences continue into an ambiguity code: fork over every compatible code.
        let children: Vec<_> = bidir.extend_right_compatible(iv, base).collect();
        // … keep the children you care about
    }
}
```

`count_wild_right` / `count_wild_left` are the same query with `SymbolSet::WILDCARDS`, and
`children_right` / `children_left` return every child interval for all 16 codes at roughly
the cost of a single extension. A sequence boundary (sentinel) never counts as wild.

## CPU/GPU parity

Both paths use the same compatibility relation — the `IupacDna` masks on the CPU and the
`COMPAT` table in WGSL on the GPU. A unit test parses the shader sources and checks the
table against `IupacDna`, and GPU parity tests compare query results, so a query returns
the same matches whether it runs on the CPU or the GPU.

## Opting out

If you want strict A/C/G/T matching where ambiguity codes never match (useful for
peer-comparable benchmarks), build with the `ExactDna` alphabet instead of the default. See
[Custom Alphabets](./alphabets.md).
