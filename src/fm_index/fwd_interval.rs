//! Forward-only SA intervals: the half of a [`BidirInterval`] that lives in the forward
//! index, usable on its own when the reverse half is unknown (ancestor walks over the
//! LCP array, LF steps on sub-ranges of an interval, locating a row range).

use crate::error::FmIndexError;
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::seq_id::SeqId;
use crate::fm_index::FmIndex;
use crate::lcp::LCP_CAP;

/// A range of suffix-array rows of the forward index, all sharing a `len`-symbol prefix.
///
/// Unlike [`BidirInterval`] it carries no reverse interval, so it can only be extended on
/// the left (an LF step) — but it can also be *shortened* on the right through the LCP
/// array ([`parent`](Self::parent)), and it may denote an arbitrary sub-range of an
/// interval (e.g. the rows of a parent interval outside one child), which every operation
/// here accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FwdInterval {
    /// First row.
    pub lo: u32,
    /// One past the last row.
    pub hi: u32,
    /// Length of the prefix the rows share.
    pub len: u32,
}

impl FwdInterval {
    /// Every row: the interval of the empty string.
    pub fn full(text_len: u32) -> Self {
        Self {
            lo: 0,
            hi: text_len,
            len: 0,
        }
    }

    /// Number of rows.
    pub fn size(&self) -> u32 {
        self.hi.saturating_sub(self.lo)
    }

    /// True when no rows remain.
    pub fn is_empty(&self) -> bool {
        self.lo >= self.hi
    }

    /// LF step: the rows of `c·X` for every suffix `X` in this range (P → cP on the
    /// forward index only). Two rank queries; `None` if no suffix in the range is preceded
    /// by `c`. Valid on any row range, not just full intervals, since rank is monotone.
    pub fn extend_left(&self, c: u8, fwd: &FmIndex) -> Option<Self> {
        if self.is_empty() {
            return None;
        }
        let c_val = fwd.c_array.get(c);
        let (r_lo, r_hi) = fwd.occ.rank_pair(c, self.lo, self.hi);
        if r_lo >= r_hi {
            return None;
        }
        Some(Self {
            lo: c_val + r_lo,
            hi: c_val + r_hi,
            len: self.len + 1,
        })
    }

    /// The nearest proper ancestor in the suffix tree: for the maximal interval of a
    /// string `X` (`len == |X|`), the interval of `X[..d']` for the largest `d' < len` at
    /// which the interval grows, with `len = d'`. `d' = max(LCP[lo], LCP[hi])` (the longest
    /// prefix shared with a neighbouring row), widened over the LCP array to every row
    /// sharing it. `Ok(None)` for the root (`len == 0`).
    ///
    /// O(1) plus the O(log n) previous/next-smaller search of the LCP array.
    ///
    /// # Errors
    /// - [`FmIndexError::LcpNotBuilt`] if `fwd` has no LCP array.
    /// - [`FmIndexError::PatternTooLong`] if a neighbouring LCP is at the storage cap.
    /// - [`FmIndexError::InvalidContraction`] if the interval is empty or not the maximal
    ///   interval of its string (a neighbouring row shares `>= len` symbols).
    pub fn parent(&self, fwd: &FmIndex) -> Result<Option<Self>, FmIndexError> {
        let lcp = fwd.lcp.as_ref().ok_or(FmIndexError::LcpNotBuilt)?;
        if self.len == 0 {
            return Ok(None);
        }
        if self.is_empty() || self.hi > fwd.text_len {
            return Err(FmIndexError::InvalidContraction);
        }
        let left = lcp.get(self.lo);
        let right = if self.hi < fwd.text_len {
            lcp.get(self.hi)
        } else {
            0
        };
        let depth = left.max(right);
        if depth >= LCP_CAP as u32 {
            return Err(FmIndexError::PatternTooLong(depth));
        }
        if depth >= self.len {
            return Err(FmIndexError::InvalidContraction);
        }
        if depth == 0 {
            return Ok(Some(Self::full(fwd.text_len)));
        }
        let lo = lcp.psv_below(self.lo, depth);
        let hi = lcp.nsv_below(self.hi, depth);
        Ok(Some(Self { lo, hi, len: depth }))
    }

    /// `(sequence id, offset)` of every row, via the forward SA samples.
    pub fn locate(&self, fwd: &FmIndex) -> Vec<(SeqId, u32)> {
        fwd.locate_rows(self.lo, self.hi)
    }
}

impl BidirInterval {
    /// The forward half of this cursor.
    pub fn fwd(&self) -> FwdInterval {
        FwdInterval {
            lo: self.fwd_lo,
            hi: self.fwd_hi,
            len: self.len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::{encode_char, DnaSequence};
    use crate::fm_index::FmIndexConfig;

    fn index(s: &str) -> FmIndex {
        let seq = DnaSequence::from_str(s).unwrap();
        FmIndex::build_cpu(
            &[seq],
            &FmIndexConfig {
                sa_sample_rate: 1,
                use_gpu: false,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn encode(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    /// Interval of `pat` by repeated LF from the root (backward search).
    fn interval_of(fwd: &FmIndex, pat: &[u8]) -> Option<FwdInterval> {
        pat.iter()
            .rev()
            .try_fold(FwdInterval::full(fwd.text_len), |iv, &c| {
                iv.extend_left(c, fwd)
            })
    }

    #[test]
    fn extend_left_matches_count_and_backward_search() {
        let text = "ACGTACGTTTGACCAGGTAC";
        let fwd = index(text);
        for start in 0..text.len() {
            for end in start + 1..=text.len() {
                let pat = encode(&text[start..end]);
                let iv = interval_of(&fwd, &pat).unwrap();
                assert_eq!(iv.size(), fwd.count(&pat), "{}", &text[start..end]);
                assert_eq!(iv.len as usize, pat.len());
            }
        }
        assert!(interval_of(&fwd, &encode("GGG")).is_none());
    }

    #[test]
    fn extend_left_on_sub_range_matches_symbol_scan() {
        let fwd = index("ACGTACGTTTGACCAGGTACGTAC");
        let n = fwd.text_len;
        for lo in 0..n {
            for hi in lo + 1..=n {
                let piece = FwdInterval { lo, hi, len: 0 };
                for c in 0..5u8 {
                    let want: Vec<u32> = (lo..hi)
                        .filter(|&i| fwd.occ.symbol_at(i) == c)
                        .map(|i| fwd.c_array.get(c) + fwd.occ.rank(c, i))
                        .collect();
                    let got = piece.extend_left(c, &fwd);
                    match got {
                        None => assert!(want.is_empty(), "lo={lo} hi={hi} c={c}"),
                        Some(iv) => {
                            assert_eq!(iv.lo, want[0]);
                            assert_eq!(iv.hi, want[want.len() - 1] + 1);
                            assert_eq!(iv.size() as usize, want.len());
                            assert_eq!(iv.len, 1);
                        }
                    }
                }
            }
        }
    }

    /// Oracle: interval of `pat[..d']` for the largest `d' < len` where it is strictly
    /// larger than the interval of `pat`.
    fn oracle_parent(fwd: &FmIndex, pat: &[u8], iv: FwdInterval) -> Option<FwdInterval> {
        if iv.len == 0 {
            return None;
        }
        (0..iv.len as usize)
            .rev()
            .map(|d| interval_of(fwd, &pat[..d]).unwrap())
            .find(|p| p.size() > iv.size())
    }

    fn check_parent_chain(fwd: &FmIndex, pat: &[u8]) {
        let mut iv = interval_of(fwd, pat).unwrap();
        loop {
            let want = oracle_parent(fwd, pat, iv);
            let got = iv.parent(fwd).unwrap();
            assert_eq!(got, want, "pattern len {} at depth {}", pat.len(), iv.len);
            match got {
                Some(p) => iv = p,
                None => break,
            }
        }
    }

    #[test]
    fn parent_chain_matches_backward_search_oracle() {
        let text = "ACGTACGTTTGACCAGGTACGTACGAAATTTCCCGGGACGTAC";
        let fwd = index(text);
        for start in 0..text.len() {
            for end in start + 1..=text.len() {
                check_parent_chain(&fwd, &encode(&text[start..end]));
            }
        }
    }

    #[test]
    fn parent_chain_on_repeats() {
        for text in ["A".repeat(200), "ACGT".repeat(100)] {
            let fwd = index(&text);
            for start in 0..4 {
                check_parent_chain(&fwd, &encode(&text[start..]));
                check_parent_chain(&fwd, &encode(&text[start..start + 5]));
            }
        }
    }

    #[test]
    fn parent_errors_and_root() {
        let fwd = index("ACGTACGT");
        assert_eq!(FwdInterval::full(fwd.text_len).parent(&fwd).unwrap(), None);
        // Not maximal: a strict sub-range of the interval of "A".
        let a = interval_of(&fwd, &encode("A")).unwrap();
        let sub = FwdInterval {
            lo: a.lo,
            hi: a.lo + 1,
            len: 1,
        };
        assert!(matches!(
            sub.parent(&fwd),
            Err(FmIndexError::InvalidContraction)
        ));
        let seq = DnaSequence::from_str("ACGTACGT").unwrap();
        let no_lcp = FmIndex::build_cpu(
            &[seq],
            &FmIndexConfig {
                sa_sample_rate: 1,
                use_gpu: false,
                build_lcp: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(matches!(a.parent(&no_lcp), Err(FmIndexError::LcpNotBuilt)));
    }

    #[test]
    fn locate_matches_pattern_locate() {
        let fwd = index("ACGTACGTTTGACCAGGTACGTAC");
        for pat in ["AC", "ACGT", "T", "GTAC"] {
            let pat = encode(pat);
            let iv = interval_of(&fwd, &pat).unwrap();
            let mut got = iv.locate(&fwd);
            let mut want = fwd.locate(&pat);
            got.sort();
            want.sort();
            assert_eq!(got, want);
        }
    }
}
