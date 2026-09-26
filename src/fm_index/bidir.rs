use crate::alphabet::{SymbolSet, ALPHABET_SIZE};
use crate::error::FmIndexError;
use crate::fm_index::FmIndex;
use crate::lcp::LCP_CAP;

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
///
/// ## Contraction, see [`contract_left`](Self::contract_left) / [`contract_right`](Self::contract_right)
///
/// **Contract left** (cP → P) inverts `extend_left`, using the forward index's select and
/// LCP array:
/// ```text
/// r        = select_fwd(c, fwd_lo − C[c] + 1)          // ψ(fwd_lo): row of P after its c
/// [lo, hi) = LCP-expansion of [r, r + |cP-interval|) to all rows with prefix P
/// offset   = Σ_{b < c} (Occ_fwd(b, hi) − Occ_fwd(b, lo))
/// rev_lo   = rev_lo(cP) − offset
/// ```
///
/// **Contract right** (Pc → P) inverts `extend_right` with the same steps on the reverse
/// index (Pc is c·P^R there), the roles of `fwd` and `rev` swapped. The cursor therefore
/// carries the matched pattern length in `len`.
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
    /// Length of the matched pattern P (0 for the full interval). Maintained by every
    /// extension (+1) and contraction (−1).
    pub len: u32,
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
            len: 0,
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
        // One block-record touch per border yields both the LF step (rank of `c`) and the
        // count of characters b < c in the current reverse interval; that count is the
        // offset locating the surviving block inside the forward interval. `None` means `c`
        // never occurs in the text at all.
        let ((r_lo, below_lo), (r_hi, below_hi)) =
            rev.occ.rank_with_below_pair(c, self.rev_lo, self.rev_hi)?;
        let new_rev_lo = c_val + r_lo;
        let new_rev_hi = c_val + r_hi;

        if new_rev_lo >= new_rev_hi {
            return None;
        }

        let offset = below_hi - below_lo;
        let new_size = new_rev_hi - new_rev_lo;

        Some(Self {
            fwd_lo: self.fwd_lo + offset,
            fwd_hi: self.fwd_lo + offset + new_size,
            rev_lo: new_rev_lo,
            rev_hi: new_rev_hi,
            len: self.len + 1,
        })
    }

    /// Extend the matched pattern to the left by character `c` (P → cP).
    ///
    /// Corresponds to the standard backward-search step on the **forward**
    /// FM-index (`BWT[i]` = `T[SA[i]−1]` naturally adds to the left of a suffix).
    ///
    /// - The forward interval is updated via the standard LF-mapping on `fwd`.
    /// - The reverse interval narrows by counting how many characters < c
    ///   appear in the current forward interval.
    ///
    /// Returns `None` if cP does not occur in the text.
    pub fn extend_left(&self, c: u8, fwd: &FmIndex) -> Option<Self> {
        let c_val = fwd.c_array.get(c);
        let ((r_lo, below_lo), (r_hi, below_hi)) =
            fwd.occ.rank_with_below_pair(c, self.fwd_lo, self.fwd_hi)?;
        let new_fwd_lo = c_val + r_lo;
        let new_fwd_hi = c_val + r_hi;

        if new_fwd_lo >= new_fwd_hi {
            return None;
        }

        let offset = below_hi - below_lo;
        let new_size = new_fwd_hi - new_fwd_lo;

        Some(Self {
            fwd_lo: new_fwd_lo,
            fwd_hi: new_fwd_hi,
            rev_lo: self.rev_lo + offset,
            rev_hi: self.rev_lo + offset + new_size,
            len: self.len + 1,
        })
    }

    // ── Contraction ───────────────────────────────────────────────────────────

    /// Contract the matched pattern on the left (cP → P), the inverse of
    /// [`extend_left`](Self::extend_left): `self` must be the interval of `cP` (with
    /// `len == |cP|`) and `c` the symbol it was extended by. Returns the interval of `P`.
    ///
    /// Cost: one `select` on the forward occ table, two LCP reads (plus an O(log n)
    /// previous/next-smaller search when not every occurrence of `P` is preceded by `c`)
    /// and the same `count_smaller_than` class rank `extend_left` pays — roughly two
    /// extensions, and no text access.
    ///
    /// Sentinel-preceded occurrences of `P` (P at the start of a reference) are handled
    /// like any other: they count towards the reverse offset as symbol `0 < c`, and
    /// `c == 0` (contracting `$P`) is itself allowed.
    ///
    /// # Errors
    /// - [`FmIndexError::LcpNotBuilt`] if `fwd` has no LCP array (built with
    ///   `build_lcp = false`, on the GPU, or loaded from a legacy blob).
    /// - [`FmIndexError::PatternTooLong`] if `|P| > LCP_CAP` (the LCP is stored capped).
    /// - [`FmIndexError::InvalidContraction`] if `self` is empty, has `len == 0`, or is
    ///   not the interval of `c·P` for the given `c` (detected when `fwd_lo` precedes
    ///   `C[c]` or the select runs past the last `c`).
    pub fn contract_left(&self, c: u8, fwd: &FmIndex) -> Result<Self, FmIndexError> {
        let Some(ell) = self.contracted_len(fwd)? else {
            return Ok(Self::full(fwd.text_len));
        };
        let (fwd_lo, fwd_hi, rev_lo) =
            contract_half(self.fwd_lo, self.size(), self.rev_lo, ell, c, fwd)?;
        Ok(Self {
            fwd_lo,
            fwd_hi,
            rev_lo,
            rev_hi: rev_lo + (fwd_hi - fwd_lo),
            len: ell,
        })
    }

    /// Contract the matched pattern on the right (Pc → P), the inverse of
    /// [`extend_right`](Self::extend_right): `self` must be the interval of `Pc` (with
    /// `len == |Pc|`) and `c` the symbol it was extended by. Returns the interval of `P`.
    ///
    /// Mirror image of [`contract_left`](Self::contract_left) on the **reverse** index
    /// (appending `c` to `P` prepends it to `P^R`), with the same cost: one `select` and two
    /// LCP reads on the reverse half, plus its `count_smaller_than` class rank.
    ///
    /// Sentinel-followed occurrences of `P` (P at the end of a reference) count towards the
    /// forward offset as symbol `0 < c`, and `c == 0` (contracting `P$`) is allowed.
    ///
    /// # Errors
    /// - [`FmIndexError::LcpNotBuilt`] if `rev` has no LCP array (built with
    ///   `build_lcp = false`, on the GPU, or loaded from a blob written before the reverse
    ///   half carried one).
    /// - [`FmIndexError::PatternTooLong`] if `|P| > LCP_CAP`.
    /// - [`FmIndexError::InvalidContraction`] if `self` is empty, has `len == 0`, or is
    ///   not the interval of `P·c` for the given `c`.
    pub fn contract_right(&self, c: u8, rev: &FmIndex) -> Result<Self, FmIndexError> {
        let Some(ell) = self.contracted_len(rev)? else {
            return Ok(Self::full(rev.text_len));
        };
        let (rev_lo, rev_hi, fwd_lo) =
            contract_half(self.rev_lo, self.size(), self.fwd_lo, ell, c, rev)?;
        Ok(Self {
            fwd_lo,
            fwd_hi: fwd_lo + (rev_hi - rev_lo),
            rev_lo,
            rev_hi,
            len: ell,
        })
    }

    /// Guards shared by both contractions: the length of the contracted pattern, or
    /// `None` when it is the empty pattern (the caller returns the full interval).
    fn contracted_len(&self, idx: &FmIndex) -> Result<Option<u32>, FmIndexError> {
        if !idx.has_lcp() {
            return Err(FmIndexError::LcpNotBuilt);
        }
        if self.is_empty() || self.len == 0 {
            return Err(FmIndexError::InvalidContraction);
        }
        let ell = self.len - 1;
        if ell == 0 {
            return Ok(None);
        }
        if ell > LCP_CAP as u32 {
            return Err(FmIndexError::PatternTooLong(ell));
        }
        Ok(Some(ell))
    }

    // ── Class counts ──────────────────────────────────────────────────────────

    /// Number of occurrences of the matched pattern P that are followed in the text by a
    /// symbol in `set` (P → P·x with x ∈ set).
    ///
    /// Asked of the **reverse** index: the symbol after P in T is the symbol before P^R
    /// in T^R, i.e. `BWT_rev[rev_lo..rev_hi)`. Costs two occ-block touches regardless of
    /// how many members `set` has, and never reads the text, so it works on the reverse
    /// half even though it carries no text.
    ///
    /// Equals `Σ_{x ∈ set} extend_right(x).size()`, without materialising the children.
    pub fn count_right_in(&self, set: SymbolSet, rev: &FmIndex) -> u32 {
        if self.is_empty() {
            return 0;
        }
        let (r_lo, r_hi) = rev.occ.rank_set_pair(set, self.rev_lo, self.rev_hi);
        r_hi - r_lo
    }

    /// Number of occurrences of the matched pattern P that are preceded in the text by a
    /// symbol in `set` (P → x·P with x ∈ set).
    ///
    /// Asked of the **forward** index (`BWT_fwd[fwd_lo..fwd_hi)`). Same cost model as
    /// [`count_right_in`](Self::count_right_in).
    pub fn count_left_in(&self, set: SymbolSet, fwd: &FmIndex) -> u32 {
        if self.is_empty() {
            return 0;
        }
        let (r_lo, r_hi) = fwd.occ.rank_set_pair(set, self.fwd_lo, self.fwd_hi);
        r_hi - r_lo
    }

    /// Number of occurrences followed by an ambiguity code (`N` or any degenerate IUPAC
    /// symbol, codes 5..=15) in the reference. A sentinel neighbour (occurrence at the end
    /// of a reference) is never wild.
    ///
    /// O(1) rank work independent of how many wildcard codes exist: a cursor walk can ask
    /// this at every step and only fan out over wildcard codes when it is non-zero.
    pub fn count_wild_right(&self, rev: &FmIndex) -> u32 {
        self.count_right_in(SymbolSet::WILDCARDS, rev)
    }

    /// Number of occurrences preceded by an ambiguity code (codes 5..=15) in the reference.
    /// See [`count_wild_right`](Self::count_wild_right).
    pub fn count_wild_left(&self, fwd: &FmIndex) -> u32 {
        self.count_left_in(SymbolSet::WILDCARDS, fwd)
    }

    // ── All children at once ──────────────────────────────────────────────────

    /// The child interval for every code at once: `children_right(rev)[c] ==
    /// extend_right(c, rev)` for all `c` in `0..ALPHABET_SIZE`.
    ///
    /// Two `OccTable::rank_all` calls (one per border) give every code's rank and the
    /// prefix sums the paired interval needs, so this costs about the same as a single
    /// `extend_right` rather than sixteen. Slot 0 is the sentinel child — occurrences of P
    /// sitting at the very end of a reference; it is a valid interval but extending it
    /// further would cross a sequence boundary. Child sizes sum to `self.size()`.
    pub fn children_right(&self, rev: &FmIndex) -> [Option<Self>; ALPHABET_SIZE] {
        let mut out = [None; ALPHABET_SIZE];
        if self.is_empty() {
            return out;
        }
        let (lo, hi) = rev.occ.rank_all_pair(self.rev_lo, self.rev_hi);
        let mut offset = 0u32;
        for c in 0..ALPHABET_SIZE {
            let n = hi[c] - lo[c];
            if n > 0 {
                let c_val = rev.c_array.get(c as u8);
                out[c] = Some(Self {
                    fwd_lo: self.fwd_lo + offset,
                    fwd_hi: self.fwd_lo + offset + n,
                    rev_lo: c_val + lo[c],
                    rev_hi: c_val + hi[c],
                    len: self.len + 1,
                });
            }
            offset += n;
        }
        out
    }

    /// The child interval for every code at once: `children_left(fwd)[c] ==
    /// extend_left(c, fwd)` for all `c`. Slot 0 is the sentinel child — occurrences of P
    /// at the very start of a reference. See [`children_right`](Self::children_right).
    pub fn children_left(&self, fwd: &FmIndex) -> [Option<Self>; ALPHABET_SIZE] {
        let mut out = [None; ALPHABET_SIZE];
        if self.is_empty() {
            return out;
        }
        let (lo, hi) = fwd.occ.rank_all_pair(self.fwd_lo, self.fwd_hi);
        let mut offset = 0u32;
        for c in 0..ALPHABET_SIZE {
            let n = hi[c] - lo[c];
            if n > 0 {
                let c_val = fwd.c_array.get(c as u8);
                out[c] = Some(Self {
                    fwd_lo: c_val + lo[c],
                    fwd_hi: c_val + hi[c],
                    rev_lo: self.rev_lo + offset,
                    rev_hi: self.rev_lo + offset + n,
                    len: self.len + 1,
                });
            }
            offset += n;
        }
        out
    }

    // ── Compatible-symbol fan-out ─────────────────────────────────────────────

    /// Extend right by every reference code the query code `q` matches under the index's
    /// alphabet (`AlphabetFns::compatible`), yielding one non-empty child per code in
    /// ascending code order. This is the fan-out `find_smems` / `find_mems` use internally.
    ///
    /// Costs one `extend_right` per compatible code; to first ask cheaply whether any such
    /// child exists, use [`count_right_in`](Self::count_right_in) with the same set
    /// (`rev.alphabet_fns.compatible_set(q)`).
    pub fn extend_right_compatible<'a>(
        &self,
        q: u8,
        rev: &'a FmIndex,
    ) -> impl Iterator<Item = Self> + 'a {
        let iv = *self;
        rev.alphabet_fns
            .compatible(q)
            .iter()
            .filter_map(move |c| iv.extend_right(c, rev))
    }

    /// Extend left by every reference code the query code `q` matches under the index's
    /// alphabet. See [`extend_right_compatible`](Self::extend_right_compatible).
    pub fn extend_left_compatible<'a>(
        &self,
        q: u8,
        fwd: &'a FmIndex,
    ) -> impl Iterator<Item = Self> + 'a {
        let iv = *self;
        fwd.alphabet_fns
            .compatible(q)
            .iter()
            .filter_map(move |c| iv.extend_left(c, fwd))
    }
}

/// Count the number of characters b < c that appear in BWT[lo..hi) of `index`.
///
/// This is Σ_{b=0}^{c-1} (Occ(b, hi) − Occ(b, lo)), asked as one class rank per border
/// (`OccTable::rank_set_pair`) rather than `2c` scalar ranks: the alphabet has 16 codes, so
/// extending by a high IUPAC code used to cost up to 30 extra rank calls per step.
fn count_smaller_than(c: u8, lo: u32, hi: u32, index: &FmIndex) -> u32 {
    let (r_lo, r_hi) = index.occ.rank_set_pair(SymbolSet::below(c), lo, hi);
    r_hi - r_lo
}

/// Undo a prepend of `c` on `idx`'s half of a cursor: `[lo, lo + size)` is that half's
/// interval of `c·X` and `other_lo` the paired half's start. Returns `(lo, hi, other_lo)`
/// of `X`, whose length is `ell`. Shared by `contract_left` (`idx` = forward) and
/// `contract_right` (`idx` = reverse); the caller has already checked `idx.has_lcp()`.
fn contract_half(
    lo: u32,
    size: u32,
    other_lo: u32,
    ell: u32,
    c: u8,
    idx: &FmIndex,
) -> Result<(u32, u32, u32), FmIndexError> {
    let lcp = idx.lcp.as_ref().ok_or(FmIndexError::LcpNotBuilt)?;

    // ψ(lo): the row of the suffix that follows the first occurrence's leading c. Every
    // row of cX maps into interval(X), and the `size` rows of X preceded by c start at r,
    // so [r, r + size) is a non-empty sub-range of interval(X).
    let rank_in_c = lo
        .checked_sub(idx.c_array.get(c))
        .ok_or(FmIndexError::InvalidContraction)?;
    let r = idx
        .occ
        .select(c, rank_in_c + 1)
        .ok_or(FmIndexError::InvalidContraction)?;
    if r.checked_add(size).is_none_or(|end| end > idx.text_len) {
        return Err(FmIndexError::InvalidContraction);
    }

    // Widen to the maximal run of rows sharing an `ell`-prefix. O(1) when the boundary
    // tests fail, i.e. when every occurrence of X is preceded by c.
    let mut new_lo = r;
    let mut new_hi = r + size;
    if lcp.get(new_lo) >= ell {
        new_lo = lcp.psv_below(new_lo, ell);
    }
    if new_hi < idx.text_len && lcp.get(new_hi) >= ell {
        new_hi = lcp.nsv_below(new_hi, ell);
    }

    // Invert the extension's offset: the c-block sits after every occurrence of X
    // preceded by a smaller symbol, so those are exactly the rows before it in the
    // paired half.
    let offset = count_smaller_than(c, new_lo, new_hi, idx);
    let new_other_lo = other_lo
        .checked_sub(offset)
        .ok_or(FmIndexError::InvalidContraction)?;
    Ok((new_lo, new_hi, new_other_lo))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::{encode_char, DnaSequence, SymbolSet, A, C, G, N, SENTINEL, T};
    use crate::fm_index::{FmIndex, FmIndexConfig};

    fn make_fwd_rev(s: &str) -> (FmIndex, FmIndex) {
        let seq = DnaSequence::from_str(s).unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            build_lcp: true,
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

    // ── Class counts and children ─────────────────────────────────────────────

    const IUPAC_TEXT: &str = "ACGTNRACGTYYACGTMWSACGTBDHVACGTKN";

    #[test]
    fn children_right_equal_extend_right_for_every_code() {
        let (fwd, rev) = make_fwd_rev(IUPAC_TEXT);
        let full = BidirInterval::full(fwd.text_len);
        let a = full.extend_right(A, &rev).unwrap();
        let n = full.extend_right(N, &rev).unwrap();
        let acgt = [A, C, G, T]
            .iter()
            .fold(full, |iv, &c| iv.extend_right(c, &rev).unwrap());
        for iv in [full, a, n, acgt] {
            let children = iv.children_right(&rev);
            for c in 0..ALPHABET_SIZE as u8 {
                assert_eq!(
                    children[c as usize],
                    iv.extend_right(c, &rev),
                    "children_right[{c}] != extend_right({c}) from {iv:?}"
                );
            }
            let total: u32 = children.iter().flatten().map(|c| c.size()).sum();
            assert_eq!(total, iv.size(), "children sizes must sum to parent size");
        }
    }

    #[test]
    fn children_left_equal_extend_left_for_every_code() {
        let (fwd, _rev) = make_fwd_rev(IUPAC_TEXT);
        let full = BidirInterval::full(fwd.text_len);
        let t = full.extend_left(T, &fwd).unwrap();
        let n = full.extend_left(N, &fwd).unwrap();
        let acgt = [T, G, C, A]
            .iter()
            .fold(full, |iv, &c| iv.extend_left(c, &fwd).unwrap());
        for iv in [full, t, n, acgt] {
            let children = iv.children_left(&fwd);
            for c in 0..ALPHABET_SIZE as u8 {
                assert_eq!(
                    children[c as usize],
                    iv.extend_left(c, &fwd),
                    "children_left[{c}] != extend_left({c}) from {iv:?}"
                );
            }
            let total: u32 = children.iter().flatten().map(|c| c.size()).sum();
            assert_eq!(total, iv.size(), "children sizes must sum to parent size");
        }
    }

    #[test]
    fn count_in_all_is_size_and_empty_is_zero() {
        let (fwd, rev) = make_fwd_rev(IUPAC_TEXT);
        let full = BidirInterval::full(fwd.text_len);
        let acgt = [A, C, G, T]
            .iter()
            .fold(full, |iv, &c| iv.extend_right(c, &rev).unwrap());
        for iv in [full, acgt] {
            assert_eq!(iv.count_right_in(SymbolSet::ALL, &rev), iv.size());
            assert_eq!(iv.count_left_in(SymbolSet::ALL, &fwd), iv.size());
            assert_eq!(iv.count_right_in(SymbolSet::EMPTY, &rev), 0);
            assert_eq!(iv.count_left_in(SymbolSet::EMPTY, &fwd), 0);
            // Partition: wildcards + bases + sentinel == everything.
            let sentinel = SymbolSet::single(SENTINEL);
            assert_eq!(
                iv.count_wild_right(&rev)
                    + iv.count_right_in(SymbolSet::BASES, &rev)
                    + iv.count_right_in(sentinel, &rev),
                iv.size()
            );
            assert_eq!(
                iv.count_wild_left(&fwd)
                    + iv.count_left_in(SymbolSet::BASES, &fwd)
                    + iv.count_left_in(sentinel, &fwd),
                iv.size()
            );
        }
        // "ACGT" occurs 5 times; followed by N, Y, M, B, K → all five wild on the right.
        assert_eq!(acgt.size(), 5);
        assert_eq!(acgt.count_wild_right(&rev), 5);
        // Preceded by: sequence start, R, Y, S, V → four wild on the left.
        assert_eq!(acgt.count_wild_left(&fwd), 4);
    }

    #[test]
    fn count_wild_is_zero_on_pure_acgt() {
        let (fwd, rev) = make_fwd_rev("ACGTACGTTTGCA");
        let mut iv = BidirInterval::full(fwd.text_len);
        assert_eq!(iv.count_wild_right(&rev), 0);
        assert_eq!(iv.count_wild_left(&fwd), 0);
        for &c in &[A, C, G] {
            iv = iv.extend_right(c, &rev).unwrap();
            assert_eq!(iv.count_wild_right(&rev), 0);
            assert_eq!(iv.count_wild_left(&fwd), 0);
        }
    }

    #[test]
    fn empty_interval_counts_zero_and_has_no_children() {
        let (fwd, rev) = make_fwd_rev(IUPAC_TEXT);
        let empty = BidirInterval {
            fwd_lo: 3,
            fwd_hi: 3,
            rev_lo: 3,
            rev_hi: 3,
            len: 2,
        };
        assert_eq!(empty.count_wild_right(&rev), 0);
        assert_eq!(empty.count_wild_left(&fwd), 0);
        assert_eq!(empty.count_right_in(SymbolSet::ALL, &rev), 0);
        assert!(empty.children_right(&rev).iter().all(Option::is_none));
        assert!(empty.children_left(&fwd).iter().all(Option::is_none));
        assert_eq!(empty.extend_right_compatible(N, &rev).count(), 0);
    }

    #[test]
    fn compatible_fanout_matches_count_in_compatible_set() {
        let (fwd, rev) = make_fwd_rev(IUPAC_TEXT);
        let full = BidirInterval::full(fwd.text_len);
        let acg = [A, C, G]
            .iter()
            .fold(full, |iv, &c| iv.extend_right(c, &rev).unwrap());
        for iv in [full, acg] {
            for q in 1..ALPHABET_SIZE as u8 {
                let set = rev.alphabet_fns.compatible_set(q);
                let fan: u32 = iv.extend_right_compatible(q, &rev).map(|c| c.size()).sum();
                assert_eq!(fan, iv.count_right_in(set, &rev), "right q={q}");
                let fan: u32 = iv.extend_left_compatible(q, &fwd).map(|c| c.size()).sum();
                assert_eq!(fan, iv.count_left_in(set, &fwd), "left q={q}");
            }
        }
    }

    // ── Contraction ───────────────────────────────────────────────────────────

    /// Walk `pat` right-to-left with `extend_left`, returning every prefix interval so
    /// `ivs[k]` is the interval of `pat[k..]`.
    fn left_walk(pat: &[u8], fwd: &FmIndex, rev: &FmIndex) -> Vec<BidirInterval> {
        let mut ivs = vec![BidirInterval::full(fwd.text_len)];
        for &c in pat.iter().rev() {
            let next = ivs.last().unwrap().extend_left(c, fwd).unwrap();
            ivs.push(next);
        }
        let _ = rev;
        ivs.reverse();
        ivs
    }

    #[test]
    fn len_tracks_extensions_and_contractions() {
        let (fwd, rev) = make_fwd_rev("ACGTACGTTTGACCA");
        let full = BidirInterval::full(fwd.text_len);
        assert_eq!(full.len, 0);
        let a = full.extend_right(encode("A")[0], &rev).unwrap();
        assert_eq!(a.len, 1);
        let ca = a.extend_left(encode("C")[0], &fwd).unwrap();
        assert_eq!(ca.len, 2);
        for child in ca.children_left(&fwd).iter().flatten() {
            assert_eq!(child.len, 3);
        }
        let back = ca.contract_left(encode("C")[0], &fwd).unwrap();
        assert_eq!(back, a);
        assert_eq!(back.len, 1);
    }

    #[test]
    fn contract_left_inverts_extend_left_for_every_suffix() {
        let text = "ACGTACGTTTGACCAGGTACGTACGAAATTTCCCGGG";
        let (fwd, rev) = make_fwd_rev(text);
        for start in 0..text.len() {
            for end in start + 1..=text.len() {
                let pat = encode(&text[start..end]);
                let ivs = left_walk(&pat, &fwd, &rev);
                for k in 0..pat.len() {
                    let got = ivs[k].contract_left(pat[k], &fwd).unwrap();
                    assert_eq!(got, ivs[k + 1], "pattern {} k={k}", &text[start..end]);
                }
            }
        }
    }

    #[test]
    fn contract_to_empty_pattern_gives_full_interval() {
        let (fwd, _) = make_fwd_rev("ACGTACGT");
        let full = BidirInterval::full(fwd.text_len);
        for c in encode("ACGT") {
            let iv = full.extend_left(c, &fwd).unwrap();
            assert_eq!(iv.contract_left(c, &fwd).unwrap(), full);
        }
    }

    #[test]
    fn contract_left_on_repeats_uses_lcp_expansion() {
        // A^200: the interval of A^k is not solely preceded by A (one occurrence is at
        // the sequence start, preceded by the sentinel), so [r, r+size) != interval(P).
        let text = "A".repeat(200);
        let (fwd, rev) = make_fwd_rev(&text);
        let pat = encode(&text);
        let ivs = left_walk(&pat, &fwd, &rev);
        for k in 0..pat.len() {
            assert_eq!(
                ivs[k].contract_left(pat[k], &fwd).unwrap(),
                ivs[k + 1],
                "k={k}"
            );
        }
        // Tandem repeat spanning many blocks.
        let text = "ACGT".repeat(100);
        let (fwd, rev) = make_fwd_rev(&text);
        for start in 0..8 {
            let pat = encode(&text[start..]);
            let ivs = left_walk(&pat, &fwd, &rev);
            for k in 0..pat.len() {
                assert_eq!(ivs[k].contract_left(pat[k], &fwd).unwrap(), ivs[k + 1]);
            }
        }
    }

    #[test]
    fn contract_left_with_sentinel_symbol() {
        let (fwd, rev) = make_fwd_rev("ACGTACGT");
        let pat = encode("ACG");
        let iv = left_walk(&pat, &fwd, &rev)[0];
        // "$ACG": the occurrence at the sequence start.
        let dollar = iv.extend_left(0, &fwd).unwrap();
        assert_eq!(dollar.size(), 1);
        assert_eq!(dollar.contract_left(0, &fwd).unwrap(), iv);
    }

    #[test]
    fn contract_left_errors() {
        let (fwd, _) = make_fwd_rev("ACGTACGT");
        let full = BidirInterval::full(fwd.text_len);
        assert!(matches!(
            full.contract_left(1, &fwd),
            Err(FmIndexError::InvalidContraction)
        ));
        let empty = BidirInterval {
            fwd_lo: 3,
            fwd_hi: 3,
            rev_lo: 3,
            rev_hi: 3,
            len: 2,
        };
        assert!(matches!(
            empty.contract_left(1, &fwd),
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
        assert!(!no_lcp.has_lcp());
        let iv = full.extend_left(1, &no_lcp).unwrap();
        assert!(matches!(
            iv.contract_left(1, &no_lcp),
            Err(FmIndexError::LcpNotBuilt)
        ));
    }

    // ── contract_right ────────────────────────────────────────────────────────

    /// `ivs[k]` = interval of `pat[..k]`, built left-to-right with `extend_right`.
    fn right_walk(pat: &[u8], fwd: &FmIndex, rev: &FmIndex) -> Vec<BidirInterval> {
        let mut ivs = vec![BidirInterval::full(fwd.text_len)];
        for &c in pat {
            let next = ivs.last().unwrap().extend_right(c, rev).unwrap();
            ivs.push(next);
        }
        ivs
    }

    fn assert_right_chain(pat: &[u8], fwd: &FmIndex, rev: &FmIndex, ctx: &str) {
        let ivs = right_walk(pat, fwd, rev);
        for k in 0..pat.len() {
            assert_eq!(
                ivs[k + 1].contract_right(pat[k], rev).unwrap(),
                ivs[k],
                "{ctx} k={k}"
            );
        }
    }

    #[test]
    fn len_tracks_right_contractions_too() {
        let (fwd, rev) = make_fwd_rev("ACGTACGTTTGACCA");
        let full = BidirInterval::full(fwd.text_len);
        let a = full.extend_left(encode("A")[0], &fwd).unwrap();
        let ac = a.extend_right(encode("C")[0], &rev).unwrap();
        assert_eq!(ac.len, 2);
        for child in ac.children_right(&rev).iter().flatten() {
            assert_eq!(child.len, 3);
            assert_eq!(
                child
                    .contract_right(child_code(&ac, child, &rev), &rev)
                    .unwrap(),
                ac
            );
        }
        let back = ac.contract_right(encode("C")[0], &rev).unwrap();
        assert_eq!(back, a);
        assert_eq!(back.len, 1);
        assert_eq!(back.contract_left(encode("A")[0], &fwd).unwrap(), full);
    }

    /// The code `child` was produced by, found by matching against `children_right`.
    fn child_code(parent: &BidirInterval, child: &BidirInterval, rev: &FmIndex) -> u8 {
        parent
            .children_right(rev)
            .iter()
            .position(|c| c.as_ref() == Some(child))
            .unwrap() as u8
    }

    #[test]
    fn contract_right_inverts_extend_right_for_every_prefix() {
        let text = "ACGTACGTTTGACCAGGTACGTACGAAATTTCCCGGG";
        let (fwd, rev) = make_fwd_rev(text);
        for start in 0..text.len() {
            for end in start + 1..=text.len() {
                let pat = encode(&text[start..end]);
                assert_right_chain(&pat, &fwd, &rev, &text[start..end]);
            }
        }
    }

    #[test]
    fn contract_right_to_empty_pattern_gives_full_interval() {
        let (fwd, rev) = make_fwd_rev("ACGTACGT");
        let full = BidirInterval::full(fwd.text_len);
        for c in encode("ACGT") {
            let iv = full.extend_right(c, &rev).unwrap();
            assert_eq!(iv.contract_right(c, &rev).unwrap(), full);
        }
    }

    #[test]
    fn contract_right_on_repeats_uses_lcp_expansion() {
        // A^200: one occurrence of A^k sits at the sequence end, followed by the sentinel
        // rather than A, so [r, r+size) on the reverse index != interval(P) there.
        let text = "A".repeat(200);
        let (fwd, rev) = make_fwd_rev(&text);
        assert_right_chain(&encode(&text), &fwd, &rev, "A^200");
        let text = "ACGT".repeat(100);
        let (fwd, rev) = make_fwd_rev(&text);
        for end in text.len() - 8..=text.len() {
            assert_right_chain(&encode(&text[..end]), &fwd, &rev, "ACGT^100");
        }
    }

    #[test]
    fn contract_right_with_sentinel_symbol() {
        let (fwd, rev) = make_fwd_rev("ACGTACGT");
        let pat = encode("CGT");
        let iv = *right_walk(&pat, &fwd, &rev).last().unwrap();
        assert_eq!(iv.size(), 2);
        // "CGT$": the occurrence at the sequence end.
        let dollar = iv.extend_right(0, &rev).unwrap();
        assert_eq!(dollar.size(), 1);
        assert_eq!(dollar.contract_right(0, &rev).unwrap(), iv);
    }

    #[test]
    fn contract_right_errors() {
        let (fwd, rev) = make_fwd_rev("ACGTACGT");
        let full = BidirInterval::full(fwd.text_len);
        assert!(matches!(
            full.contract_right(1, &rev),
            Err(FmIndexError::InvalidContraction)
        ));
        let empty = BidirInterval {
            fwd_lo: 3,
            fwd_hi: 3,
            rev_lo: 3,
            rev_hi: 3,
            len: 2,
        };
        assert!(matches!(
            empty.contract_right(1, &rev),
            Err(FmIndexError::InvalidContraction)
        ));
        // Wrong symbol: "AC" contracted by G — rev_lo precedes C_rev[G].
        let ac = *right_walk(&encode("AC"), &fwd, &rev).last().unwrap();
        assert!(ac.contract_right(encode("G")[0], &rev).is_err());

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
        let iv = full.extend_right(1, &no_lcp).unwrap();
        assert!(matches!(
            iv.contract_right(1, &no_lcp),
            Err(FmIndexError::LcpNotBuilt)
        ));
    }

    // ── Fused-rank extension: bit-identical to the unfused reference ─────────

    /// `extend_right` as written before `OccTable::rank_with_below_pair` existed: two scalar
    /// ranks plus one class-rank pair. Kept verbatim as the oracle for the fused version.
    fn extend_right_reference(iv: &BidirInterval, c: u8, rev: &FmIndex) -> Option<BidirInterval> {
        let c_val = rev.c_array.get(c);
        let new_rev_lo = c_val + rev.occ.rank(c, iv.rev_lo);
        let new_rev_hi = c_val + rev.occ.rank(c, iv.rev_hi);
        if new_rev_lo >= new_rev_hi {
            return None;
        }
        let (b_lo, b_hi) = rev
            .occ
            .rank_set_pair(SymbolSet::below(c), iv.rev_lo, iv.rev_hi);
        let offset = b_hi - b_lo;
        let new_size = new_rev_hi - new_rev_lo;
        Some(BidirInterval {
            fwd_lo: iv.fwd_lo + offset,
            fwd_hi: iv.fwd_lo + offset + new_size,
            rev_lo: new_rev_lo,
            rev_hi: new_rev_hi,
            len: iv.len + 1,
        })
    }

    /// Mirror of [`extend_right_reference`] on the forward index.
    fn extend_left_reference(iv: &BidirInterval, c: u8, fwd: &FmIndex) -> Option<BidirInterval> {
        let c_val = fwd.c_array.get(c);
        let new_fwd_lo = c_val + fwd.occ.rank(c, iv.fwd_lo);
        let new_fwd_hi = c_val + fwd.occ.rank(c, iv.fwd_hi);
        if new_fwd_lo >= new_fwd_hi {
            return None;
        }
        let (b_lo, b_hi) = fwd
            .occ
            .rank_set_pair(SymbolSet::below(c), iv.fwd_lo, iv.fwd_hi);
        let offset = b_hi - b_lo;
        let new_size = new_fwd_hi - new_fwd_lo;
        Some(BidirInterval {
            fwd_lo: new_fwd_lo,
            fwd_hi: new_fwd_hi,
            rev_lo: iv.rev_lo + offset,
            rev_hi: iv.rev_lo + offset + new_size,
            len: iv.len + 1,
        })
    }

    fn make_fwd_rev_with(s: &str, occ_encoding: crate::occ::OccEncoding) -> (FmIndex, FmIndex) {
        let seq = DnaSequence::from_str(s).unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            occ_encoding,
            ..Default::default()
        };
        let fwd = FmIndex::build_cpu(&[seq.clone()], &config).unwrap();
        let rev_bases: Vec<u8> = seq.as_slice().iter().rev().cloned().collect();
        let rev = FmIndex::build_cpu(&[DnaSequence::from_encoded(rev_bases)], &config).unwrap();
        (fwd, rev)
    }

    /// Breadth-first over every cursor reachable by right extension (every substring of the
    /// text up to `depth`), checking at each one that both extension directions equal the
    /// unfused reference for all 16 codes and that `children_*` agree with `extend_*`.
    fn assert_extensions_match_reference(fwd: &FmIndex, rev: &FmIndex, depth: usize) {
        let mut frontier = vec![BidirInterval::full(fwd.text_len)];
        let mut checked = 0usize;
        for _ in 0..depth {
            let mut next = Vec::new();
            for iv in &frontier {
                let kids_right = iv.children_right(rev);
                let kids_left = iv.children_left(fwd);
                for c in 0..ALPHABET_SIZE as u8 {
                    let right = iv.extend_right(c, rev);
                    assert_eq!(
                        right,
                        extend_right_reference(iv, c, rev),
                        "extend_right({c}) on {iv:?}"
                    );
                    assert_eq!(
                        kids_right[c as usize], right,
                        "children_right[{c}] on {iv:?}"
                    );
                    let left = iv.extend_left(c, fwd);
                    assert_eq!(
                        left,
                        extend_left_reference(iv, c, fwd),
                        "extend_left({c}) on {iv:?}"
                    );
                    assert_eq!(kids_left[c as usize], left, "children_left[{c}] on {iv:?}");
                    checked += 1;
                    if let Some(child) = right {
                        next.push(child);
                    }
                }
            }
            frontier = next;
        }
        assert!(checked > 16, "walk covered {checked} extensions");
    }

    #[test]
    fn extend_matches_unfused_reference_on_all_reachable_intervals() {
        // Full IUPAC table, compact ACGT table (high codes have no lane), single-symbol table.
        let texts = [
            "ACGTNACGTRYSWKMBDHVACGTNNACGTAC",
            "ACGTTGCAACGTACGTTGCAACGTGGA",
            "AAAAAAAAAAAAAAAA",
        ];
        for enc in [
            crate::occ::OccEncoding::Bitplane,
            crate::occ::OccEncoding::OneHot,
        ] {
            for text in texts {
                let (fwd, rev) = make_fwd_rev_with(text, enc);
                assert_extensions_match_reference(&fwd, &rev, 6);
            }
        }
    }
}
