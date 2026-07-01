use crate::alphabet::ALPHABET_SIZE;
use crate::fm_index::FmIndex;

/// A paired SA interval for bidirectional FM-index search.
///
/// Maintains two intervals simultaneously:
/// - `fwd`: interval in the forward FM-index (text T) for pattern P
/// - `rev`: interval in the reverse FM-index (text T^R) for pattern P^R
///
/// The invariant is |fwd| == |rev| at all times.
///
/// ## Extension formulae (Lam et al. 2009, Lemma 3)
///
/// **Extend right by c** (P → Pc), using the forward Occ table:
/// ```text
/// new_fwd_lo = C[c] + Occ_fwd(c, fwd_lo)
/// new_fwd_hi = C[c] + Occ_fwd(c, fwd_hi)
/// offset     = Σ_{b < c} (Occ_fwd(b, fwd_hi) − Occ_fwd(b, fwd_lo))
/// new_rev_lo = rev_lo + offset
/// new_rev_hi = rev_lo + offset + (new_fwd_hi − new_fwd_lo)
/// ```
///
/// **Extend left by c** (P → cP), using the reverse Occ table:
/// ```text
/// new_rev_lo = C[c] + Occ_rev(c, rev_lo)
/// new_rev_hi = C[c] + Occ_rev(c, rev_hi)
/// offset     = Σ_{b < c} (Occ_rev(b, rev_hi) − Occ_rev(b, rev_lo))
/// new_fwd_lo = fwd_lo + offset
/// new_fwd_hi = fwd_lo + offset + (new_rev_hi − new_rev_lo)
/// ```
///
/// The `offset` in each case counts how many occurrences of characters
/// lexicographically smaller than c appear in the current interval, thereby
/// locating the block of c-extending positions within the paired interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BidirInterval {
    /// Start of the SA interval in the forward FM-index.
    pub fwd_lo: u32,
    /// End (exclusive) of the SA interval in the forward FM-index.
    pub fwd_hi: u32,
    /// Start of the SA interval in the reverse FM-index.
    pub rev_lo: u32,
    /// End (exclusive) of the SA interval in the reverse FM-index.
    pub rev_hi: u32,
}

impl BidirInterval {
    /// The "whole text" interval representing an empty pattern match.
    ///
    /// `text_len` must match the text length of both the forward and reverse indices.
    pub fn full(text_len: u32) -> Self {
        Self {
            fwd_lo: 0,
            fwd_hi: text_len,
            rev_lo: 0,
            rev_hi: text_len,
        }
    }

    /// Number of occurrences (same for forward and reverse intervals).
    pub fn size(&self) -> u32 {
        self.fwd_hi.saturating_sub(self.fwd_lo)
    }

    /// True when no occurrences remain.
    pub fn is_empty(&self) -> bool {
        self.fwd_lo >= self.fwd_hi
    }

    /// Extend the matched pattern to the right by character `c` (P → Pc).
    ///
    /// Corresponds to a backward-search step on the **reverse** FM-index
    /// (appending `c` to `P` is the same as prepending `c` to `P^R`).
    ///
    /// - The reverse interval is updated via the standard LF-mapping on `rev`.
    /// - The forward interval narrows by counting how many characters < c
    ///   appear in the current reverse interval.
    ///
    /// Returns `None` if Pc does not occur in the text.
    pub fn extend_right(&self, c: u8, rev: &FmIndex) -> Option<Self> {
        let c_val = rev.c_array.get(c);
        let new_rev_lo = c_val + rev.occ.rank(c, self.rev_lo);
        let new_rev_hi = c_val + rev.occ.rank(c, self.rev_hi);

        if new_rev_lo >= new_rev_hi {
            return None;
        }

        // Count characters b < c in the current reverse interval.
        // This offset locates the surviving block inside the forward interval.
        let offset: u32 = count_smaller_than(c, self.rev_lo, self.rev_hi, rev);
        let new_size = new_rev_hi - new_rev_lo;

        Some(Self {
            fwd_lo: self.fwd_lo + offset,
            fwd_hi: self.fwd_lo + offset + new_size,
            rev_lo: new_rev_lo,
            rev_hi: new_rev_hi,
        })
    }

    /// Extend the matched pattern to the left by character `c` (P → cP).
    ///
    /// Corresponds to the standard backward-search step on the **forward**
    /// FM-index (BWT[i] = T[SA[i]−1] naturally adds to the left of a suffix).
    ///
    /// - The forward interval is updated via the standard LF-mapping on `fwd`.
    /// - The reverse interval narrows by counting how many characters < c
    ///   appear in the current forward interval.
    ///
    /// Returns `None` if cP does not occur in the text.
    pub fn extend_left(&self, c: u8, fwd: &FmIndex) -> Option<Self> {
        let c_val = fwd.c_array.get(c);
        let new_fwd_lo = c_val + fwd.occ.rank(c, self.fwd_lo);
        let new_fwd_hi = c_val + fwd.occ.rank(c, self.fwd_hi);

        if new_fwd_lo >= new_fwd_hi {
            return None;
        }

        let offset: u32 = count_smaller_than(c, self.fwd_lo, self.fwd_hi, fwd);
        let new_size = new_fwd_hi - new_fwd_lo;

        Some(Self {
            fwd_lo: new_fwd_lo,
            fwd_hi: new_fwd_hi,
            rev_lo: self.rev_lo + offset,
            rev_hi: self.rev_lo + offset + new_size,
        })
    }
}

/// Count the number of characters b < c that appear in BWT[lo..hi) of `index`.
///
/// This is Σ_{b=0}^{c-1} (Occ(b, hi) − Occ(b, lo)).
fn count_smaller_than(c: u8, lo: u32, hi: u32, index: &FmIndex) -> u32 {
    let c_idx = c as usize;
    // Only iterate over alphabet characters that are actually < c.
    // ALPHABET_SIZE is 6 ($ A C G T N), so c_idx is at most 5.
    (0..c_idx.min(ALPHABET_SIZE))
        .map(|b| index.occ.rank(b as u8, hi) - index.occ.rank(b as u8, lo))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::{encode_char, DnaSequence};
    use crate::fm_index::{FmIndex, FmIndexConfig};

    fn make_fwd_rev(s: &str) -> (FmIndex, FmIndex) {
        let seq = DnaSequence::from_str(s).unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            ..Default::default()
        };
        let fwd = FmIndex::build_cpu(&[seq.clone()], &config).unwrap();

        // Reverse the sequence for the reverse index
        let rev_bases: Vec<u8> = seq.as_slice().iter().rev().cloned().collect();
        let rev_seq = DnaSequence::from_encoded(rev_bases);
        let rev = FmIndex::build_cpu(&[rev_seq], &config).unwrap();
        (fwd, rev)
    }

    fn encode(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    #[test]
    fn full_interval_size_equals_text_len() {
        let (fwd, _rev) = make_fwd_rev("ACGT");
        let iv = BidirInterval::full(fwd.text_len);
        assert_eq!(iv.size(), fwd.text_len);
        assert!(!iv.is_empty());
    }

    #[test]
    fn extend_right_matches_forward_count() {
        let (fwd, rev) = make_fwd_rev("ACGTACGT");
        let iv = BidirInterval::full(fwd.text_len);

        let pattern = encode("ACGT");
        let mut cur = iv;
        // extend_right uses the REVERSE index
        for &c in &pattern {
            cur = cur.extend_right(c, &rev).expect("should extend");
        }
        // size should equal the number of occurrences of "ACGT"
        assert_eq!(cur.size(), fwd.count(&pattern));
        // reverse interval size must equal forward interval size
        assert_eq!(cur.rev_hi - cur.rev_lo, cur.fwd_hi - cur.fwd_lo);
    }

    #[test]
    fn extend_right_collapses_on_missing_pattern() {
        let (fwd, rev) = make_fwd_rev("AAAA");
        let iv = BidirInterval::full(fwd.text_len);
        // "C" does not appear in "AAAA"
        let enc_c = encode_char('C').unwrap();
        assert!(iv.extend_right(enc_c, &rev).is_none());
        let _ = fwd;
    }

    #[test]
    fn extend_left_matches_forward_count() {
        // extend_left(A) extend_left(C) extend_left(G) extend_left(T) builds interval for "ACGT"
        // (each step prepends a character: start→T→GT→CGT→ACGT)
        let (fwd, rev) = make_fwd_rev("ACGTACGT");
        let pattern = encode("ACGT");

        let mut iv = BidirInterval::full(fwd.text_len);
        // extend_left uses the FORWARD index; to match "ACGT" we prepend right-to-left: T,G,C,A
        for &c in pattern.iter().rev() {
            iv = iv.extend_left(c, &fwd).expect("should extend_left");
        }
        assert_eq!(iv.size(), fwd.count(&pattern));
        assert_eq!(iv.rev_hi - iv.rev_lo, iv.fwd_hi - iv.fwd_lo);
        let _ = rev;
    }

    #[test]
    fn size_invariant_maintained_through_extensions() {
        let (fwd, rev) = make_fwd_rev("ACGTACGTACGT");

        // Test extend_right (uses rev index)
        let mut iv = BidirInterval::full(fwd.text_len);
        for c_char in "ACGT".chars() {
            let c = encode_char(c_char).unwrap();
            if let Some(next) = iv.extend_right(c, &rev) {
                assert_eq!(
                    next.fwd_hi - next.fwd_lo,
                    next.rev_hi - next.rev_lo,
                    "size invariant broken after extend_right({})",
                    c_char
                );
                iv = next;
            } else {
                break;
            }
        }

        // Test extend_left (uses fwd index)
        iv = BidirInterval::full(fwd.text_len);
        for c_char in "TGCA".chars() {
            let c = encode_char(c_char).unwrap();
            if let Some(next) = iv.extend_left(c, &fwd) {
                assert_eq!(
                    next.fwd_hi - next.fwd_lo,
                    next.rev_hi - next.rev_lo,
                    "size invariant broken after extend_left({})",
                    c_char
                );
                iv = next;
            } else {
                break;
            }
        }
    }
}
