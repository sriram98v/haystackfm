//! Row-level locate for the [`BidirInterval`] backends (occ-table and run-length), and the
//! occ-table backend itself.

use super::{BidirSearch, Mem, RawMem};
use crate::alphabet::{AlphabetFns, SymbolSet, ALPHABET_SIZE};
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::bidir_index::BidirFmIndex;
use crate::fm_index::half::FmHalf;
use crate::fm_index::seq_id::SeqId;

/// Locate for a backend whose cursor is a [`BidirInterval`]: forward SA rows resolve one by
/// one, and the forward BWT tells which rows are left-maximal. The symbol preceding the
/// occurrence at forward row `r` is `BWT[r]`, so the filter runs before locating and needs
/// no text; an occurrence at a reference start has a sentinel there, which no query symbol
/// is compatible with, so it is always kept.
pub(crate) trait RowLocate: BidirSearch<Cursor = BidirInterval> {
    /// Forward rows of `ivs` whose preceding symbol (`BWT_fwd[row]`) is not in `compat`, in
    /// order: the left-maximal occurrences of a match whose left context `compat` matches.
    fn left_maximal_rows(&self, ivs: &[BidirInterval], compat: SymbolSet) -> Vec<u32>;

    /// `(sequence id, offset)` of each listed forward SA row, in order.
    fn locate_row_list(&self, rows: &[u32]) -> Vec<(SeqId, u32)>;

    /// [`BidirSearch::search_locate`] with one [`locate_row_list`](Self::locate_row_list)
    /// call for the rows of every match, so a backend that interleaves LF walks gets the
    /// whole query's rows at once instead of a few per match. Positions in row order.
    fn locate_by_rows(&self, raws: Vec<RawMem>) -> Vec<Mem> {
        let mut rows: Vec<u32> = Vec::new();
        let mut ends: Vec<usize> = Vec::with_capacity(raws.len());
        for raw in &raws {
            if let Some(c) = raw.left_ctx {
                let compat = self.search_alphabet().compatible(c);
                rows.extend(self.left_maximal_rows(&raw.ivs, compat));
            } else {
                rows.extend(raw.ivs.iter().flat_map(|iv| iv.fwd_lo..iv.fwd_hi));
            }
            ends.push(rows.len());
        }
        let mut located = self.locate_row_list(&rows).into_iter();
        let mut start = 0;
        raws.into_iter()
            .zip(ends)
            .map(|(raw, end)| {
                let positions = located.by_ref().take(end - start).collect();
                start = end;
                raw.into_mem(positions)
            })
            .collect()
    }
}

impl BidirSearch for BidirFmIndex {
    type Cursor = BidirInterval;

    const FANOUT_STEPS: bool = false;

    fn search_alphabet(&self) -> &AlphabetFns {
        self.fwd.alphabet()
    }

    fn search_full_cursor(&self) -> BidirInterval {
        self.full_interval()
    }

    #[inline]
    fn search_extend_right(&self, cur: &BidirInterval, c: u8) -> Option<BidirInterval> {
        cur.extend_right(c, &self.rev)
    }

    #[inline]
    fn search_extend_left(&self, cur: &BidirInterval, c: u8) -> Option<BidirInterval> {
        cur.extend_left(c, &self.fwd)
    }

    #[inline]
    fn search_children_right(&self, cur: &BidirInterval) -> [Option<BidirInterval>; ALPHABET_SIZE] {
        cur.children_right(&self.rev)
    }

    #[inline]
    fn search_children_left(&self, cur: &BidirInterval) -> [Option<BidirInterval>; ALPHABET_SIZE] {
        cur.children_left(&self.fwd)
    }

    fn shallow(&self, kmer: &[u8]) -> Option<&[BidirInterval]> {
        self.shallow.get(kmer)
    }

    fn search_locate(&self, raws: Vec<RawMem>) -> Vec<Mem> {
        self.locate_by_rows(raws)
    }
}

impl RowLocate for BidirFmIndex {
    fn left_maximal_rows(&self, ivs: &[BidirInterval], compat: SymbolSet) -> Vec<u32> {
        let mut rows = Vec::new();
        for iv in ivs {
            self.fwd
                .occ
                .rows_not_in(compat, iv.fwd_lo, iv.fwd_hi, &mut rows);
        }
        rows
    }

    fn locate_row_list(&self, rows: &[u32]) -> Vec<(SeqId, u32)> {
        let mut text_positions = Vec::with_capacity(rows.len());
        self.fwd.resolve_sa_batch(rows, &mut text_positions);
        text_positions
            .into_iter()
            .map(|pos| {
                self.fwd
                    .map_position(pos)
                    .expect("resolved SA position must be within text bounds")
            })
            .collect()
    }
}
