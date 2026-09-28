//! [`MoveLf`]: the BWT's LF mapping as a move table, with the per-run lookups backward
//! search and bidirectional extension need.

use serde::{Deserialize, Serialize};

use crate::alphabet::{SymbolSet, ALPHABET_SIZE, SENTINEL};
use crate::bwt::Bwt;
use crate::c_array::CArray;
use crate::move_table::MoveTable;
use crate::occ::cpu::build_occ_table;
use crate::occ::{OccEncoding, OccTable};

/// Rows scanned linearly before falling back to rank/select on the run heads when looking
/// for the next or previous run of a symbol (Movi's "repositioning").
const WALK: usize = 8;

/// One side of a cursor: the SA interval `[s, e]` (inclusive) and the move-table rows
/// holding `s` and `e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Span {
    pub s: u32,
    pub ks: u32,
    pub e: u32,
    pub ke: u32,
}

impl Span {
    /// Number of rows.
    #[inline]
    pub fn size(&self) -> u32 {
        self.e - self.s + 1
    }
}

/// Where the toehold of an extended interval comes from: the SA value at a sampled row
/// (start or end of a run whose symbol is the extension symbol), or — when the whole
/// interval lies inside one run — the previous toehold carried through the LF step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Toehold {
    RunStart(u32),
    RunEnd(u32),
    Carry,
}

/// The BWT's LF mapping as a move table: one row per BWT run, whose head is the run symbol
/// (sentinel runs are split into single rows, so every sentinel row is sampled — see
/// [`crate::move_index`]), plus a rank/select table over the row heads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "MoveLfRepr", into = "MoveLfRepr")]
pub struct MoveLf {
    pub(crate) table: MoveTable,
    pub(crate) c_array: CArray,
    /// Rank/select over the row heads. Derived from `table`.
    heads: OccTable,
    /// Symbols occurring in the BWT. Derived.
    present: SymbolSet,
}

impl MoveLf {
    /// Build from a BWT and its C array.
    pub(crate) fn from_bwt(bwt: &Bwt, c_array: &CArray) -> Self {
        let mut counts = [0u32; ALPHABET_SIZE];
        let mut runs: Vec<(u32, u32, u8)> = Vec::new();
        let mut p = 0u32;
        for (c, len) in bwt.runs() {
            let q = c_array.get(c) + counts[c as usize];
            if c == SENTINEL {
                // Every sentinel row becomes its own row: LF is only trusted to map text
                // positions across non-sentinel symbols, and a sampled row per sentinel
                // keeps φ / PLCP piecewise-linear around sequence boundaries.
                runs.extend((0..len).map(|off| (p + off, q + off, c)));
            } else {
                runs.push((p, q, c));
            }
            counts[c as usize] += len;
            p += len;
        }
        let table = MoveTable::build(&runs, p, None);
        Self::from_parts(table, c_array.clone()).expect("LF move table built from a BWT is valid")
    }

    /// Assemble from a table and C array, checking they describe an LF mapping: each row's
    /// image is `C[head] + rank_head(start)`, and the head counts match the C array.
    fn from_parts(table: MoveTable, c_array: CArray) -> Result<Self, String> {
        let n = table.domain();
        if c_array.data.windows(2).any(|w| w[0] > w[1]) || c_array.data[ALPHABET_SIZE - 1] > n {
            return Err("C array is not a prefix sum within the text".into());
        }
        let rows = table.num_rows() as usize;
        let mut counts = [0u32; ALPHABET_SIZE];
        let mut heads = Vec::with_capacity(rows);
        for k in 0..rows {
            let row = table.row(k);
            let c = row.head as usize;
            if c >= ALPHABET_SIZE {
                return Err("LF row head out of range".into());
            }
            if row.q != c_array.data[c] + counts[c] {
                return Err(format!("LF row {k} does not map to C[head] + rank"));
            }
            counts[c] += table.end(k) - table.start(k) + 1;
            heads.push(row.head);
        }
        for c in 0..ALPHABET_SIZE as u8 {
            if c_array.symbol_count(c, n) != counts[c as usize] {
                return Err(format!("LF rows disagree with the C array on symbol {c}"));
            }
        }
        let heads = build_occ_table(&Bwt::from_unpacked(heads), OccEncoding::Bitplane);
        let present = c_array.present_symbols(n);
        Ok(Self {
            table,
            c_array,
            heads,
            present,
        })
    }

    /// Number of BWT rows.
    pub fn len(&self) -> u32 {
        self.table.domain()
    }

    /// True for an empty BWT.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of move-table rows (BWT runs, with sentinel runs split).
    pub fn num_rows(&self) -> u32 {
        self.table.num_rows()
    }

    /// Symbols occurring in the BWT.
    pub(crate) fn present(&self) -> SymbolSet {
        self.present
    }

    /// Heap bytes of the move table and the run-head rank table.
    pub fn heap_size_bytes(&self) -> usize {
        self.table.heap_size_bytes() + self.heads.heap_size_bytes()
    }

    /// The span of the whole BWT (empty pattern). `None` for an empty BWT.
    pub(crate) fn full_span(&self) -> Option<Span> {
        (!self.is_empty()).then(|| Span {
            s: 0,
            ks: 0,
            e: self.len() - 1,
            ke: self.num_rows() - 1,
        })
    }

    /// First row in `[ks, ke]` whose head is `c`.
    #[inline]
    fn find_first(&self, c: u8, ks: usize, ke: usize) -> Option<usize> {
        let limit = ke.min(ks + WALK);
        for k in ks..=limit {
            if self.table.head(k) == c {
                return Some(k);
            }
        }
        if limit == ke {
            return None;
        }
        let before = self.heads.rank(c, (limit + 1) as u32);
        let k = self.heads.select(c, before + 1)? as usize;
        (k <= ke).then_some(k)
    }

    /// Last row in `[ks, ke]` whose head is `c`.
    #[inline]
    fn find_last(&self, c: u8, ks: usize, ke: usize) -> Option<usize> {
        let lower = ks.max(ke.saturating_sub(WALK));
        for k in (lower..=ke).rev() {
            if self.table.head(k) == c {
                return Some(k);
            }
        }
        if lower == ks {
            return None;
        }
        let before = self.heads.rank(c, lower as u32);
        let k = self.heads.select(c, before)? as usize;
        (k >= ks).then_some(k)
    }

    /// The first and last rows of `span` holding `c`, and the sub-interval `[s_c, e_c]`
    /// between the first and last occurrence of `c` in `bwt[s..=e]`.
    #[inline]
    fn occurrences(&self, c: u8, span: &Span) -> Option<(usize, u32, usize, u32)> {
        let (ks, ke) = (span.ks as usize, span.ke as usize);
        let kf = self.find_first(c, ks, ke)?;
        let kl = self
            .find_last(c, kf, ke)
            .expect("a row holding c exists at or after the first one");
        let s_c = if kf == ks {
            span.s
        } else {
            self.table.start(kf)
        };
        let e_c = if kl == ke { span.e } else { self.table.end(kl) };
        Some((kf, s_c, kl, e_c))
    }

    /// Occurrences of `c` in `bwt[s..=e]`.
    #[inline]
    pub(crate) fn count_in(&self, c: u8, span: &Span) -> u32 {
        match self.occurrences(c, span) {
            Some((kf, s_c, kl, e_c)) => self.table.map(e_c, kl) - self.table.map(s_c, kf) + 1,
            None => 0,
        }
    }

    /// Occurrences in `bwt[s..=e]` of any present symbol in `set`.
    ///
    /// A span over few rows is summed row by row (clipped to `[s, e]`); a wide one asks
    /// [`count_in`](Self::count_in) per member.
    pub(crate) fn count_set_in(&self, set: SymbolSet, span: &Span) -> u32 {
        let set = set & self.present;
        let (ks, ke) = (span.ks as usize, span.ke as usize);
        if ke - ks <= 2 * WALK {
            return (ks..=ke)
                .filter(|&k| set.contains(self.table.head(k)))
                .map(|k| {
                    let lo = self.table.start(k).max(span.s);
                    let hi = self.table.end(k).min(span.e);
                    hi - lo + 1
                })
                .sum();
        }
        set.iter().map(|c| self.count_in(c, span)).sum()
    }

    /// Backward-search step: the span of `c·P` given the span of `P`, and where its toehold
    /// comes from. `None` when `c` does not occur in `bwt[s..=e]`.
    #[inline]
    pub(crate) fn extend(&self, c: u8, span: &Span) -> Option<(Span, Toehold)> {
        let (kf, s_c, kl, e_c) = self.occurrences(c, span)?;
        let (s, ks) = self.table.step(s_c, kf);
        let (e, ke) = self.table.step(e_c, kl);
        let toehold = if kf != span.ks as usize {
            Toehold::RunStart(kf as u32)
        } else if kl != span.ke as usize {
            Toehold::RunEnd(kl as u32)
        } else if span.ks != span.ke {
            // Row ks holds c and ends inside the interval.
            Toehold::RunEnd(span.ks)
        } else {
            Toehold::Carry
        };
        Some((
            Span {
                s,
                ks: ks as u32,
                e,
                ke: ke as u32,
            },
            toehold,
        ))
    }

    /// The span `[s, e]` with its rows resolved, searching from the rows of an enclosing
    /// span `outer` (`outer.s <= s <= e <= outer.e`).
    #[inline]
    pub(crate) fn narrow(&self, outer: &Span, s: u32, e: u32) -> Span {
        Span {
            s,
            ks: self.table.row_from(s, outer.ks as usize) as u32,
            e,
            ke: self.table.row_back_from(e, outer.ke as usize) as u32,
        }
    }
}

/// Serialized layout of a [`MoveLf`]: the table and C array; the head rank table and
/// present set are rebuilt.
#[derive(Serialize, Deserialize)]
struct MoveLfRepr {
    table: MoveTable,
    c_array: CArray,
}

impl From<MoveLf> for MoveLfRepr {
    fn from(lf: MoveLf) -> Self {
        Self {
            table: lf.table,
            c_array: lf.c_array,
        }
    }
}

impl TryFrom<MoveLfRepr> for MoveLf {
    type Error = String;

    fn try_from(r: MoveLfRepr) -> Result<Self, String> {
        Self::from_parts(r.table, r.c_array)
    }
}
