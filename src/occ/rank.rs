//! [`BwtRank`]: the rank/access primitives FM-index search needs from a BWT representation.
//!
//! Implemented by the plain [`OccTable`] and by the run-length
//! [`RlBwt`](crate::rlbwt::RlBwt), so the bidirectional cursor
//! ([`BidirInterval`](crate::BidirInterval)) and the run-length index's backward search are
//! written once over either backend. Every rank counts `bwt[0..i)`, i.e. excludes row `i`.

use crate::alphabet::{SymbolSet, ALPHABET_SIZE};

use super::OccTable;

/// Rank and access over a BWT of length [`bwt_len`](Self::bwt_len) on the 16-code alphabet.
///
/// All ranks count occurrences in `bwt[0..i)`; `i` may equal `bwt_len()`. Symbols absent
/// from the BWT rank 0.
pub trait BwtRank {
    /// Number of BWT rows (the text length, sentinels included).
    fn bwt_len(&self) -> u32;

    /// Occurrences of `c` in `bwt[0..i)`.
    fn rank(&self, c: u8, i: u32) -> u32;

    /// `(rank(c, lo), rank(c, hi))`.
    fn rank_pair(&self, c: u8, lo: u32, hi: u32) -> (u32, u32);

    /// `Some(((rank(c, lo), below(c, lo)), (rank(c, hi), below(c, hi))))`, where
    /// `below(c, i)` counts symbols `< c` in `bwt[0..i)`; `None` when `c` does not occur in
    /// `bwt[lo..hi)`. The one-call form of a bidirectional extension step.
    fn rank_with_below_pair(&self, c: u8, lo: u32, hi: u32) -> Option<((u32, u32), (u32, u32))>;

    /// `(rank_set(set, lo), rank_set(set, hi))`: occurrences of any member of `set`.
    fn rank_set_pair(&self, set: SymbolSet, lo: u32, hi: u32) -> (u32, u32);

    /// Rank of every code at both borders: `(rank_all(lo), rank_all(hi))`.
    fn rank_all_pair(&self, lo: u32, hi: u32) -> ([u32; ALPHABET_SIZE], [u32; ALPHABET_SIZE]);

    /// The symbol at row `i < bwt_len()`.
    fn symbol_at(&self, i: u32) -> u8;

    /// `(bwt[i], rank(bwt[i], i))` for `i < bwt_len()`: everything one LF step needs.
    fn lf_step(&self, i: u32) -> (u8, u32);
}

impl BwtRank for OccTable {
    #[inline]
    fn bwt_len(&self) -> u32 {
        self.text_len
    }

    #[inline]
    fn rank(&self, c: u8, i: u32) -> u32 {
        OccTable::rank(self, c, i)
    }

    #[inline]
    fn rank_pair(&self, c: u8, lo: u32, hi: u32) -> (u32, u32) {
        OccTable::rank_pair(self, c, lo, hi)
    }

    #[inline]
    fn rank_with_below_pair(&self, c: u8, lo: u32, hi: u32) -> Option<((u32, u32), (u32, u32))> {
        OccTable::rank_with_below_pair(self, c, lo, hi)
    }

    #[inline]
    fn rank_set_pair(&self, set: SymbolSet, lo: u32, hi: u32) -> (u32, u32) {
        OccTable::rank_set_pair(self, set, lo, hi)
    }

    #[inline]
    fn rank_all_pair(&self, lo: u32, hi: u32) -> ([u32; ALPHABET_SIZE], [u32; ALPHABET_SIZE]) {
        OccTable::rank_all_pair(self, lo, hi)
    }

    #[inline]
    fn symbol_at(&self, i: u32) -> u8 {
        OccTable::symbol_at(self, i)
    }

    #[inline]
    fn lf_step(&self, i: u32) -> (u8, u32) {
        OccTable::lf_step(self, i)
    }
}
