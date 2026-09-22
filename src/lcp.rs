//! Longest-common-prefix array with block minima and a sparse table, supporting the
//! previous/next-smaller-value queries that
//! [`BidirInterval::contract_left`](crate::fm_index::BidirInterval::contract_left) uses to
//! widen a sub-range of an SA interval to the full interval of a shorter pattern.
//!
//! `lcp[i]` is the length of the longest common prefix of the suffixes at SA rows `i-1` and
//! `i` (`lcp[0] = 0`), built by Kasai's algorithm at index-construction time while the full
//! suffix array is still resident. Values are stored capped at [`LCP_CAP`] as `u16`: the
//! contraction only ever compares `lcp[i]` against a pattern length `ell`, so capped values
//! are exact for every `ell <= LCP_CAP`.
//!
//! Memory: 2 bytes per row for the array plus `2/64` bytes per row for the block minima
//! and `~2·log2(n/64)/64` bytes per row for the sparse table.

use serde::{Deserialize, Serialize};

/// Rows per block of the minima layer (same granularity as the occ table).
pub const LCP_BLOCK_SIZE: u32 = 64;

/// Stored LCP values saturate here; a stored `LCP_CAP` means "at least `LCP_CAP`".
pub const LCP_CAP: u16 = u16::MAX;

/// Capped LCP array over the rows of a suffix array, with O(log n) previous/next-smaller
/// queries. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LcpArray {
    /// `lcp[i] = min(LCP(SA[i-1], SA[i]), LCP_CAP)`, `lcp[0] = 0`.
    lcp: Vec<u16>,
    /// Minimum of `lcp` over each `LCP_BLOCK_SIZE`-row block.
    block_min: Vec<u16>,
    /// Sparse table over `block_min` for levels `1..=max_level`, flattened level by level:
    /// level `k` holds `nb - 2^k + 1` entries, entry `j` = `min(block_min[j..j + 2^k])`.
    /// Level 0 is `block_min` itself.
    sparse: Vec<u16>,
}

impl LcpArray {
    /// Build with Kasai's algorithm from the text (alphabet codes, sentinels included) and
    /// its full suffix array. Sentinels are ordinary symbols here, matching the order the
    /// suffix array was built with. Uses a transient `4n`-byte inverse suffix array.
    pub fn build_kasai(text: &[u8], sa: &[u32]) -> Self {
        let n = text.len();
        assert_eq!(sa.len(), n, "suffix array length must equal text length");
        let mut isa = vec![0u32; n];
        for (row, &pos) in sa.iter().enumerate() {
            isa[pos as usize] = row as u32;
        }
        let mut lcp = vec![0u16; n];
        let mut h = 0usize;
        for i in 0..n {
            let row = isa[i] as usize;
            if row == 0 {
                h = 0;
                continue;
            }
            let j = sa[row - 1] as usize;
            while i + h < n && j + h < n && text[i + h] == text[j + h] {
                h += 1;
            }
            lcp[row] = h.min(LCP_CAP as usize) as u16;
            h = h.saturating_sub(1);
        }
        drop(isa);
        Self::from_lcp(lcp)
    }

    /// Wrap an already-capped LCP array and build the minima layers.
    fn from_lcp(lcp: Vec<u16>) -> Self {
        let block_min: Vec<u16> = lcp
            .chunks(LCP_BLOCK_SIZE as usize)
            .map(|chunk| chunk.iter().copied().min().unwrap_or(LCP_CAP))
            .collect();
        let sparse = build_sparse_table(&block_min);
        Self {
            lcp,
            block_min,
            sparse,
        }
    }

    /// Number of rows.
    pub fn len(&self) -> u32 {
        self.lcp.len() as u32
    }

    /// True when the array has no rows.
    pub fn is_empty(&self) -> bool {
        self.lcp.is_empty()
    }

    /// Capped LCP between rows `i - 1` and `i` (`0` for `i == 0`).
    #[inline]
    pub fn get(&self, i: u32) -> u32 {
        self.lcp[i as usize] as u32
    }

    /// Largest row `j <= i` with `lcp[j] < ell`. Always exists for `ell >= 1` because
    /// `lcp[0] == 0`; returns `0` for `ell == 0` (no row qualifies).
    pub fn psv_below(&self, i: u32, ell: u32) -> u32 {
        let block = i / LCP_BLOCK_SIZE;
        let start = block * LCP_BLOCK_SIZE;
        if let Some(j) = self.scan_down(start, i, ell) {
            return j;
        }
        if block == 0 {
            return 0;
        }
        // Largest block m < block whose range [m, block) contains a value < ell. The
        // predicate is monotone (true for m implies true for every smaller m), and the
        // largest true m has block_min[m] < ell itself.
        let (mut lo, mut hi) = (0u32, block - 1);
        if self.range_min(lo, hi) >= ell {
            return 0;
        }
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if self.range_min(mid, block - 1) < ell {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let m_start = lo * LCP_BLOCK_SIZE;
        let m_end = (m_start + LCP_BLOCK_SIZE - 1).min(self.len() - 1);
        self.scan_down(m_start, m_end, ell).unwrap_or(0)
    }

    /// Smallest row `j >= i` with `lcp[j] < ell`, or `len()` if none.
    pub fn nsv_below(&self, i: u32, ell: u32) -> u32 {
        let n = self.len();
        if i >= n {
            return n;
        }
        let block = i / LCP_BLOCK_SIZE;
        let end = ((block + 1) * LCP_BLOCK_SIZE).min(n) - 1;
        if let Some(j) = self.scan_up(i, end, ell) {
            return j;
        }
        let nb = self.block_min.len() as u32;
        if block + 1 >= nb {
            return n;
        }
        // Smallest block m > block whose range (block, m] contains a value < ell.
        let (mut lo, mut hi) = (block + 1, nb - 1);
        if self.range_min(lo, hi) >= ell {
            return n;
        }
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.range_min(block + 1, mid) < ell {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let m_start = lo * LCP_BLOCK_SIZE;
        let m_end = (m_start + LCP_BLOCK_SIZE - 1).min(n - 1);
        self.scan_up(m_start, m_end, ell).unwrap_or(n)
    }

    /// Largest `j` in `[from, to]` with `lcp[j] < ell`.
    #[inline]
    fn scan_down(&self, from: u32, to: u32, ell: u32) -> Option<u32> {
        (from..=to).rev().find(|&j| self.get(j) < ell)
    }

    /// Smallest `j` in `[from, to]` with `lcp[j] < ell`.
    #[inline]
    fn scan_up(&self, from: u32, to: u32, ell: u32) -> Option<u32> {
        (from..=to).find(|&j| self.get(j) < ell)
    }

    /// Minimum of `block_min[a..=b]` via the sparse table (O(1)).
    #[inline]
    fn range_min(&self, a: u32, b: u32) -> u32 {
        debug_assert!(a <= b && (b as usize) < self.block_min.len());
        let span = b - a + 1;
        let k = u32::BITS - 1 - span.leading_zeros(); // floor(log2(span))
        if k == 0 {
            return self.block_min[a as usize] as u32;
        }
        let nb = self.block_min.len() as u32;
        let off = level_offset(nb, k) as usize;
        let left = self.sparse[off + a as usize];
        let right = self.sparse[off + (b + 1 - (1 << k)) as usize];
        left.min(right) as u32
    }
}

/// Offset of level `k >= 1` inside the flattened sparse table over `nb` blocks.
#[inline]
fn level_offset(nb: u32, k: u32) -> u32 {
    (1..k).map(|lvl| nb + 1 - (1 << lvl)).sum()
}

fn build_sparse_table(block_min: &[u16]) -> Vec<u16> {
    let nb = block_min.len();
    let mut sparse = Vec::new();
    let mut prev: Vec<u16> = block_min.to_vec();
    let mut k = 1usize;
    while (1usize << k) <= nb {
        let half = 1usize << (k - 1);
        let level: Vec<u16> = (0..nb + 1 - (1 << k))
            .map(|j| prev[j].min(prev[j + half]))
            .collect();
        sparse.extend_from_slice(&level);
        prev = level;
        k += 1;
    }
    sparse
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_sa(text: &[u8]) -> Vec<u32> {
        let mut sa: Vec<u32> = (0..text.len() as u32).collect();
        sa.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
        sa
    }

    fn naive_lcp(text: &[u8], sa: &[u32]) -> Vec<u32> {
        let mut out = vec![0u32; sa.len()];
        for i in 1..sa.len() {
            let a = &text[sa[i - 1] as usize..];
            let b = &text[sa[i] as usize..];
            out[i] = a.iter().zip(b).take_while(|(x, y)| x == y).count() as u32;
        }
        out
    }

    fn random_text(seed: u64, len: usize, sentinels: usize) -> Vec<u8> {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut text: Vec<u8> = (0..len).map(|_| (next() % 4) as u8 + 1).collect();
        for _ in 0..sentinels {
            let at = (next() as usize) % len;
            text[at] = 0;
        }
        let last = text.len() - 1;
        text[last] = 0;
        text
    }

    #[test]
    fn kasai_matches_naive_lcp_with_sentinels() {
        for seed in 1..20u64 {
            let text = random_text(seed, 300, 4);
            let sa = naive_sa(&text);
            let expected = naive_lcp(&text, &sa);
            let lcp = LcpArray::build_kasai(&text, &sa);
            for (i, &e) in expected.iter().enumerate() {
                assert_eq!(lcp.get(i as u32), e, "seed {seed} row {i}");
            }
        }
    }

    #[test]
    fn kasai_on_repeat_text_caps_and_spans_blocks() {
        let mut text = vec![1u8; 3000];
        *text.last_mut().unwrap() = 0;
        let sa = naive_sa(&text);
        let lcp = LcpArray::build_kasai(&text, &sa);
        // Row 0 is the sentinel suffix, row i is suffix n-1-i: lcp[i] = i-1 for i >= 2.
        assert_eq!(lcp.get(0), 0);
        assert_eq!(lcp.get(1), 0);
        assert_eq!(lcp.get(2), 1);
        assert_eq!(lcp.get(2999), 2998);
    }

    fn check_psv_nsv(lcp: &LcpArray, ells: &[u32]) {
        let n = lcp.len();
        for &ell in ells {
            for i in 0..n {
                let want_psv = (0..=i).rev().find(|&j| lcp.get(j) < ell).unwrap_or(0);
                assert_eq!(lcp.psv_below(i, ell), want_psv, "psv i={i} ell={ell}");
                let want_nsv = (i..n).find(|&j| lcp.get(j) < ell).unwrap_or(n);
                assert_eq!(lcp.nsv_below(i, ell), want_nsv, "nsv i={i} ell={ell}");
            }
        }
    }

    #[test]
    fn psv_nsv_match_linear_scan_on_repeats() {
        let mut text = vec![1u8; 3000];
        *text.last_mut().unwrap() = 0;
        let sa = naive_sa(&text);
        let lcp = LcpArray::build_kasai(&text, &sa);
        check_psv_nsv(&lcp, &[1, 2, 63, 64, 65, 500, 2000, 2999]);

        let mut tandem: Vec<u8> = [1u8, 2, 3, 4].iter().copied().cycle().take(2000).collect();
        *tandem.last_mut().unwrap() = 0;
        let sa = naive_sa(&tandem);
        let lcp = LcpArray::build_kasai(&tandem, &sa);
        check_psv_nsv(&lcp, &[1, 4, 5, 100, 1000, 1999]);
    }

    #[test]
    fn psv_nsv_match_linear_scan_on_random_text() {
        for seed in 1..6u64 {
            let text = random_text(seed, 1500, 3);
            let sa = naive_sa(&text);
            let lcp = LcpArray::build_kasai(&text, &sa);
            check_psv_nsv(&lcp, &[1, 2, 3, 5, 8]);
        }
    }

    #[test]
    fn range_min_matches_naive() {
        let text = random_text(7, 5000, 5);
        let sa = naive_sa(&text);
        let lcp = LcpArray::build_kasai(&text, &sa);
        let nb = lcp.block_min.len() as u32;
        for a in (0..nb).step_by(7) {
            for b in a..nb {
                let want = lcp.block_min[a as usize..=b as usize]
                    .iter()
                    .copied()
                    .min()
                    .unwrap() as u32;
                assert_eq!(lcp.range_min(a, b), want, "a={a} b={b}");
            }
        }
    }

    #[test]
    fn serialization_round_trip() {
        let text = random_text(3, 700, 2);
        let sa = naive_sa(&text);
        let lcp = LcpArray::build_kasai(&text, &sa);
        let bytes = bincode::serialize(&lcp).unwrap();
        let back: LcpArray = bincode::deserialize(&bytes).unwrap();
        assert_eq!(lcp, back);
    }
}
