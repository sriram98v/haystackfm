//! [`FmHalf`]: what the bidirectional cursor needs from one half of a bidirectional index.
//!
//! Implemented by [`FmIndex`] (occ-table rank) and by
//! [`RlFmIndex`](crate::rl_fm_index::RlFmIndex) (run-length rank), so
//! [`BidirInterval`](crate::BidirInterval)'s extension, class-count and children methods run
//! unchanged on [`BidirFmIndex`](crate::BidirFmIndex) and
//! [`BidirRlFmIndex`](crate::rl_fm_index::BidirRlFmIndex). The trait is sealed: its surface
//! grows with the move-structure backends and is not meant to be implemented downstream.

use crate::alphabet::AlphabetFns;
use crate::c_array::CArray;
use crate::fm_index::FmIndex;
use crate::occ::{BwtRank, OccTable};

mod sealed {
    pub trait Sealed {}
}

pub(crate) use sealed::Sealed;

/// One half (forward or reverse) of a bidirectional FM-index, as seen by the cursor.
pub trait FmHalf: Sealed {
    /// The BWT rank backend.
    type Rank: BwtRank;

    /// Rank/access over this half's BWT.
    fn bwt_rank(&self) -> &Self::Rank;

    /// `C[c]`: number of text symbols smaller than `c`.
    fn c_array(&self) -> &CArray;

    /// Text length (sentinels included) — the number of BWT rows.
    fn half_text_len(&self) -> u32;

    /// Matching semantics used by the `*_compatible` fan-out.
    fn alphabet(&self) -> &AlphabetFns;
}

impl Sealed for FmIndex {}

impl FmHalf for FmIndex {
    type Rank = OccTable;

    #[inline]
    fn bwt_rank(&self) -> &OccTable {
        &self.occ
    }

    #[inline]
    fn c_array(&self) -> &CArray {
        &self.c_array
    }

    #[inline]
    fn half_text_len(&self) -> u32 {
        self.text_len
    }

    #[inline]
    fn alphabet(&self) -> &AlphabetFns {
        &self.alphabet_fns
    }
}
