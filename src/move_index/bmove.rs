//! b-move: a bidirectional move structure. Forward and reverse LF move tables drive a
//! cursor that carries both intervals, their move-table rows, and a toehold; locate runs on
//! the forward half's `φ` / `φ⁻¹` tables.

use super::lf::{MoveLf, Span, Toehold};
use super::locate::RunSamples;
use super::{MoveFmIndex, MoveFmIndexConfig, UNKNOWN_TOEHOLD};
use crate::alphabet::{Alphabet, DnaSequence, IupacDna, SymbolSet, ALPHABET_SIZE, SENTINEL};
use crate::bwt::cpu::build_bwt;
use crate::error::FmIndexError;
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::bidir_index::reverse_as_sequence;
use crate::fm_index::prepare::{in_build_pool, prepare_cpu};
use crate::fm_index::seq_id::SeqId;

/// A b-move search state: the SA interval of the matched pattern `P` in the forward index
/// and of `P^R` in the reverse one, the move-table rows of their borders, `|P|`, and a
/// toehold (the text position of one occurrence of `P`).
///
/// Produced and consumed by [`BMoveIndex`]; convert to a [`BidirInterval`] with
/// [`interval`](Self::interval) to compare with the other bidirectional indexes (both
/// intervals are identical across index types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BMoveCursor {
    fwd: Span,
    rev: Span,
    len: u32,
    toehold: u32,
}

impl BMoveCursor {
    /// Number of occurrences.
    pub fn size(&self) -> u32 {
        self.fwd.size()
    }

    /// Length of the matched pattern.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// True for the empty pattern's cursor.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The same search state as a [`BidirInterval`] (exclusive upper bounds).
    pub fn interval(&self) -> BidirInterval {
        BidirInterval {
            fwd_lo: self.fwd.s,
            fwd_hi: self.fwd.e + 1,
            rev_lo: self.rev.s,
            rev_hi: self.rev.e + 1,
            len: self.len,
        }
    }

    /// Text position of one occurrence of the matched pattern, or `None` when none is known
    /// (index without locate support, or a pattern extended by the sentinel).
    pub fn toehold(&self) -> Option<u32> {
        (self.toehold != UNKNOWN_TOEHOLD).then_some(self.toehold)
    }
}

/// Bidirectional move structure (b-move): a forward [`MoveFmIndex`] (LF, locate) plus a
/// reverse LF move table with its run-boundary SA samples. See [`crate::move_index`].
///
/// Cursor steps return intervals identical to
/// [`BidirFmIndex`](crate::BidirFmIndex)'s; locate needs only the toehold the cursor
/// carries. Extending by the sentinel (code 0) is allowed but such cursors cannot be
/// located ([`locate`](BMoveIndex::locate) panics).
#[derive(Debug, Clone)]
pub struct BMoveIndex {
    pub(crate) fwd: MoveFmIndex,
    pub(crate) rev: MoveLf,
    /// Present exactly when `fwd` has locate support.
    pub(crate) rev_samples: Option<RunSamples>,
}

impl BMoveIndex {
    /// Build on the CPU with the [`IupacDna`] alphabet.
    pub fn build_cpu(
        sequences: &[DnaSequence],
        config: &MoveFmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        Self::build_cpu_with::<IupacDna>(sequences, config)
    }

    /// Build on the CPU with a custom [`Alphabet`].
    pub fn build_cpu_with<A: Alphabet>(
        sequences: &[DnaSequence],
        config: &MoveFmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }
        in_build_pool(config.build_threads, || {
            let prepared = prepare_cpu(sequences)?;
            let fwd = MoveFmIndex::from_prepared(&prepared, A::fns(), config.locate);
            // Reverse half: the byte-reversal of the same text as one sequence, as for
            // `BidirFmIndex`; only its LF table and run samples are kept.
            let rev_prepared = prepare_cpu(&[reverse_as_sequence(&prepared.text)?])?;
            drop(prepared);
            let rev_bwt = build_bwt(&rev_prepared.text, &rev_prepared.sa);
            let rev = MoveLf::from_bwt(&rev_bwt, &rev_prepared.c_array);
            let rev_samples = config
                .locate
                .then(|| RunSamples::build(&rev.table, &rev_prepared.sa.data));
            Ok(Self {
                fwd,
                rev,
                rev_samples,
            })
        })
    }

    /// The forward half.
    pub fn fwd(&self) -> &MoveFmIndex {
        &self.fwd
    }

    /// The reverse half's LF move structure.
    pub fn rev(&self) -> &MoveLf {
        &self.rev
    }

    /// The cursor of the empty pattern (`None` only for an empty index, which cannot be
    /// built).
    pub fn full_cursor(&self) -> BMoveCursor {
        BMoveCursor {
            fwd: self.fwd.lf.full_span().expect("index is non-empty"),
            rev: self.rev.full_span().expect("index is non-empty"),
            len: 0,
            toehold: self
                .fwd
                .locate
                .as_ref()
                .map_or(UNKNOWN_TOEHOLD, |l| l.sa_first()),
        }
    }

    /// Extend left by `c` (P → cP): a move LF step on the forward half; the reverse
    /// interval narrows by the occurrences of symbols `< c`. `None` when cP does not occur.
    pub fn extend_left(&self, cur: &BMoveCursor, c: u8) -> Option<BMoveCursor> {
        let (fwd, toehold) = self.fwd.lf.extend(c, &cur.fwd)?;
        let below = self.fwd.lf.count_set_in(SymbolSet::below(c), &cur.fwd);
        Some(self.left_child(cur, c, fwd, toehold, below))
    }

    /// Extend right by `c` (P → Pc): a move LF step on the reverse half; the forward
    /// interval narrows by the occurrences of symbols `< c`. `None` when Pc does not occur.
    pub fn extend_right(&self, cur: &BMoveCursor, c: u8) -> Option<BMoveCursor> {
        let (rev, toehold) = self.rev.extend(c, &cur.rev)?;
        let below = self.rev.count_set_in(SymbolSet::below(c), &cur.rev);
        Some(self.right_child(cur, c, rev, toehold, below))
    }

    fn left_child(
        &self,
        cur: &BMoveCursor,
        c: u8,
        fwd: Span,
        toehold: Toehold,
        below: u32,
    ) -> BMoveCursor {
        let s = cur.rev.s + below;
        let rev = self.rev.narrow(&cur.rev, s, s + fwd.size() - 1);
        // The occurrence moves one to the left (`SA[LF(t)] = SA[t] − 1`, trusted only for
        // non-sentinel symbols).
        let toehold = match &self.fwd.locate {
            Some(locate) if c != SENTINEL && cur.toehold != UNKNOWN_TOEHOLD => locate
                .samples
                .get(toehold)
                .unwrap_or(cur.toehold)
                .checked_sub(1)
                .unwrap_or(UNKNOWN_TOEHOLD),
            _ => UNKNOWN_TOEHOLD,
        };
        BMoveCursor {
            fwd,
            rev,
            len: cur.len + 1,
            toehold,
        }
    }

    fn right_child(
        &self,
        cur: &BMoveCursor,
        c: u8,
        rev: Span,
        toehold: Toehold,
        below: u32,
    ) -> BMoveCursor {
        let s = cur.fwd.s + below;
        let fwd = self.fwd.lf.narrow(&cur.fwd, s, s + rev.size() - 1);
        let len = cur.len + 1;
        let toehold = match &self.rev_samples {
            Some(samples) if c != SENTINEL && cur.toehold != UNKNOWN_TOEHOLD => {
                match samples.get(toehold) {
                    // `y` starts an occurrence of (Pc)^R in T^R, and T^R[j] = T[n − 2 − j].
                    Some(v) => v
                        .checked_sub(1)
                        .and_then(|y| (self.fwd.text_len - 1).checked_sub(y + len))
                        .unwrap_or(UNKNOWN_TOEHOLD),
                    // Every occurrence of P is followed by c: the toehold still starts one.
                    None => cur.toehold,
                }
            }
            _ => UNKNOWN_TOEHOLD,
        };
        BMoveCursor {
            fwd,
            rev,
            len,
            toehold,
        }
    }

    /// Every left child at once: `children_left(cur)[c] == extend_left(cur, c)`.
    pub fn children_left(&self, cur: &BMoveCursor) -> [Option<BMoveCursor>; ALPHABET_SIZE] {
        let mut out = [None; ALPHABET_SIZE];
        let mut below = 0;
        for c in self.fwd.lf.present().iter() {
            if let Some((fwd, toehold)) = self.fwd.lf.extend(c, &cur.fwd) {
                let size = fwd.size();
                out[c as usize] = Some(self.left_child(cur, c, fwd, toehold, below));
                below += size;
            }
        }
        out
    }

    /// Every right child at once: `children_right(cur)[c] == extend_right(cur, c)`.
    pub fn children_right(&self, cur: &BMoveCursor) -> [Option<BMoveCursor>; ALPHABET_SIZE] {
        let mut out = [None; ALPHABET_SIZE];
        let mut below = 0;
        for c in self.rev.present().iter() {
            if let Some((rev, toehold)) = self.rev.extend(c, &cur.rev) {
                let size = rev.size();
                out[c as usize] = Some(self.right_child(cur, c, rev, toehold, below));
                below += size;
            }
        }
        out
    }

    /// Extend left by every reference code query code `q` matches.
    pub fn extend_left_compatible<'a>(
        &'a self,
        cur: &BMoveCursor,
        q: u8,
    ) -> impl Iterator<Item = BMoveCursor> + 'a {
        let cur = *cur;
        self.compatible_set(q)
            .iter()
            .filter_map(move |c| self.extend_left(&cur, c))
    }

    /// Extend right by every reference code query code `q` matches.
    pub fn extend_right_compatible<'a>(
        &'a self,
        cur: &BMoveCursor,
        q: u8,
    ) -> impl Iterator<Item = BMoveCursor> + 'a {
        let cur = *cur;
        self.compatible_set(q)
            .iter()
            .filter_map(move |c| self.extend_right(&cur, c))
    }

    /// Occurrences of the cursor's pattern preceded by a symbol in `set`.
    pub fn count_left_in(&self, cur: &BMoveCursor, set: SymbolSet) -> u32 {
        self.fwd.lf.count_set_in(set, &cur.fwd)
    }

    /// Occurrences of the cursor's pattern followed by a symbol in `set`.
    pub fn count_right_in(&self, cur: &BMoveCursor, set: SymbolSet) -> u32 {
        self.rev.count_set_in(set, &cur.rev)
    }

    /// Occurrences preceded by an ambiguity code (codes 5..=15).
    pub fn count_wild_left(&self, cur: &BMoveCursor) -> u32 {
        self.count_left_in(cur, SymbolSet::WILDCARDS)
    }

    /// Occurrences followed by an ambiguity code (codes 5..=15).
    pub fn count_wild_right(&self, cur: &BMoveCursor) -> u32 {
        self.count_right_in(cur, SymbolSet::WILDCARDS)
    }

    /// The reference codes query code `q` matches under this index's alphabet.
    pub fn compatible_set(&self, q: u8) -> SymbolSet {
        self.fwd.alphabet_fns.compatible_set(q)
    }

    /// Locate every occurrence of the cursor's pattern, in no particular order.
    ///
    /// # Panics
    /// If the index was built without locate support, or the cursor has no toehold (its
    /// pattern was extended by the sentinel).
    pub fn locate(&self, cur: &BMoveCursor) -> Vec<(SeqId, u32)> {
        let locate = self.fwd.locate_support();
        assert!(
            cur.toehold != UNKNOWN_TOEHOLD,
            "cannot locate a b-move cursor extended by the sentinel"
        );
        let mut out = Vec::with_capacity(cur.size() as usize);
        locate.enumerate(cur.toehold, cur.len, cur.size(), &mut out);
        out.into_iter()
            .map(|pos| {
                self.fwd
                    .map_position(pos)
                    .expect("located position must be within text bounds")
            })
            .collect()
    }

    /// Count occurrences of `pattern` (backward search on the forward half).
    pub fn count(&self, pattern: &[u8]) -> u32 {
        self.fwd.count(pattern)
    }

    /// Total text length (sentinels included).
    pub fn text_len(&self) -> u32 {
        self.fwd.text_len
    }

    /// Number of sequences indexed.
    pub fn num_sequences(&self) -> u32 {
        self.fwd.num_sequences
    }

    /// FASTA headers in build order.
    pub fn seq_headers(&self) -> &[String] {
        self.fwd.seq_headers()
    }

    /// Header for a sequence id.
    pub fn seq_header(&self, id: SeqId) -> Option<&str> {
        self.fwd.seq_header(id)
    }

    /// Id for a header.
    pub fn seq_id(&self, header: &str) -> Option<SeqId> {
        self.fwd.seq_id(header)
    }

    /// Heap bytes of both halves.
    pub fn heap_size_bytes(&self) -> usize {
        self.fwd.heap_size_bytes()
            + self.rev.heap_size_bytes()
            + self
                .rev_samples
                .as_ref()
                .map_or(0, RunSamples::heap_size_bytes)
    }
}
