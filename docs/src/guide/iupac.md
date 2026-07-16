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

## CPU/GPU parity

Both paths use the same compatibility lookup — `compatible_symbols` on the CPU and the
`COMPAT` table in WGSL on the GPU. Parity tests enforce that the two stay in sync, so a query
returns the same matches whether it runs on the CPU or the GPU.

## Opting out

If you want strict A/C/G/T matching where ambiguity codes never match (useful for
peer-comparable benchmarks), build with the `ExactDna` alphabet instead of the default. See
[Custom Alphabets](./alphabets.md).
