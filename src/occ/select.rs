//! Select support for [`OccTable`]: the inverse of `rank`, locating the `r`-th occurrence
//! of a symbol in the BWT. Used by
//! [`BidirInterval::contract_left`](crate::fm_index::BidirInterval::contract_left) to
//! compute ψ (the row of the suffix one position later) without a locate.
//!
//! Layout reuse: the per-lane superblock checkpoints are monotone, so select is a binary
//! search over them, a scan of at most `SUPERBLOCK_SIZE / BLOCK_SIZE` block records, and an
//! in-word select on one lane mask. A hint array holding the superblock of every
//! [`SELECT_HINT_STRIDE`]-th occurrence bounds the binary search to a short window. The
//! hints are derived from the checkpoints, so they are rebuilt after deserialisation rather
//! than stored.

use super::{OccTable, BLOCK_SIZE, NO_LANE, SUPERBLOCK_SIZE};

/// One hint (superblock index) per this many occurrences of each lane's symbol.
pub const SELECT_HINT_STRIDE: u32 = 4096;

impl OccTable {
    #[inline]
    fn num_superblocks(&self) -> usize {
        if self.num_lanes == 0 {
            0
        } else {
            self.superblock_checkpoints.len() / self.num_lanes as usize
        }
    }

    #[inline]
    fn hints_per_lane(&self) -> usize {
        (self.text_len / SELECT_HINT_STRIDE) as usize + 1
    }

    #[inline]
    fn checkpoint(&self, sb: usize, lane: usize) -> u32 {
        self.superblock_checkpoints[sb * self.num_lanes as usize + lane]
    }

    /// (Re)build the select hint array from the superblock checkpoints:
    /// `select_hints[lane * hints_per_lane + k]` is the largest superblock whose checkpoint
    /// is below occurrence `k * SELECT_HINT_STRIDE + 1` of that lane's symbol, i.e. the
    /// superblock that occurrence lies in. Lanes with fewer occurrences are padded with the
    /// last superblock. Called from `from_parts` and after deserialisation.
    pub(crate) fn build_select_hints(&mut self) {
        let num_lanes = self.num_lanes as usize;
        let nsb = self.num_superblocks();
        let hpl = self.hints_per_lane();
        let mut hints = vec![0u32; num_lanes * hpl];
        if nsb == 0 {
            self.select_hints = hints;
            return;
        }
        for lane in 0..num_lanes {
            let row = &mut hints[lane * hpl..(lane + 1) * hpl];
            let mut k = 0usize;
            for sb in 1..nsb {
                let count = self.checkpoint(sb, lane);
                while k < hpl && count > (k as u32) * SELECT_HINT_STRIDE {
                    row[k] = (sb - 1) as u32;
                    k += 1;
                }
            }
            for slot in row.iter_mut().skip(k) {
                *slot = (nsb - 1) as u32;
            }
        }
        self.select_hints = hints;
    }

    /// Position of the `r`-th (1-based) occurrence of symbol `c` in the BWT, so that
    /// `rank(c, select(c, r)) == r - 1` and `symbol_at(select(c, r)) == c`.
    ///
    /// Returns `None` when `r == 0`, when `c` never occurs, or when `r` exceeds the number
    /// of occurrences of `c`.
    pub fn select(&self, c: u8, r: u32) -> Option<u32> {
        if r == 0 {
            return None;
        }
        let lane = self.symbol_to_lane[c as usize];
        if lane == NO_LANE {
            return None;
        }
        let lane = lane as usize;
        if r > self.rank_at_lane(lane, self.text_len) {
            return None;
        }
        let sb = self.select_superblock(lane, r);
        let (block, before) = self.select_block(sb, lane, r);
        let base = self.block_base(block);
        let mask = self.lane_mask(base, lane);
        let k = r - before;
        debug_assert!(k >= 1 && k <= mask.count_ones());
        Some(block as u32 * BLOCK_SIZE + select_in_word(mask, k))
    }

    /// Largest superblock whose checkpoint for `lane` is `< r`.
    #[inline]
    fn select_superblock(&self, lane: usize, r: u32) -> usize {
        let nsb = self.num_superblocks();
        let hpl = self.hints_per_lane();
        let k = ((r - 1) / SELECT_HINT_STRIDE) as usize;
        let (mut lo, mut hi) = if self.select_hints.len() == self.num_lanes as usize * hpl {
            let row = &self.select_hints[lane * hpl..(lane + 1) * hpl];
            let lo = row[k] as usize;
            let hi = row.get(k + 1).map_or(nsb - 1, |&h| h as usize);
            (lo, hi)
        } else {
            (0, nsb - 1)
        };
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if self.checkpoint(mid, lane) < r {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    /// Within superblock `sb`, the last block whose cumulative count for `lane` (superblock
    /// checkpoint + block delta) is `< r`, and that count.
    #[inline]
    fn select_block(&self, sb: usize, lane: usize, r: u32) -> (usize, u32) {
        let blocks_per_sb = (SUPERBLOCK_SIZE / BLOCK_SIZE) as usize;
        let num_blocks = self.text_len.div_ceil(BLOCK_SIZE) as usize;
        let first = sb * blocks_per_sb;
        let last = (first + blocks_per_sb).min(num_blocks);
        let count_before = |b: usize| {
            let base = self.block_base(b);
            self.sb_count_at(base, lane) + self.delta_at(base, lane)
        };
        let mut block = first;
        let mut before = count_before(first);
        for b in first + 1..last {
            let cnt = count_before(b);
            if cnt >= r {
                break;
            }
            block = b;
            before = cnt;
        }
        (block, before)
    }
}

/// Index of the `k`-th (1-based) set bit of `w`; `k` must not exceed `w.count_ones()`.
/// Portable (no BMI2): skips whole bytes by popcount, then clears bits within the byte.
#[inline]
fn select_in_word(w: u64, k: u32) -> u32 {
    debug_assert!(k >= 1 && k <= w.count_ones());
    let mut remaining = k;
    let mut shift = 0u32;
    loop {
        let byte = (w >> shift) & 0xFF;
        let ones = byte.count_ones();
        if ones >= remaining {
            let mut bits = byte;
            for _ in 1..remaining {
                bits &= bits - 1;
            }
            return shift + bits.trailing_zeros();
        }
        remaining -= ones;
        shift += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bwt::Bwt;
    use crate::occ::cpu::build_occ_table;
    use crate::occ::OccEncoding;

    fn random_chars(seed: u64, len: usize, wild: bool) -> Vec<u8> {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..len)
            .map(|_| {
                let v = next();
                if v % 97 == 0 {
                    0
                } else if wild && v % 13 == 0 {
                    5 + (v % 11) as u8
                } else {
                    1 + (v % 4) as u8
                }
            })
            .collect()
    }

    #[test]
    fn select_in_word_matches_bit_walk() {
        let words = [
            1u64,
            u64::MAX,
            0x8000_0000_0000_0001,
            0xF0F0_F0F0_0F0F_0F0F,
            0xDEAD_BEEF_CAFE_BABE,
        ];
        for &w in &words {
            let mut k = 0;
            for bit in 0..64 {
                if (w >> bit) & 1 == 1 {
                    k += 1;
                    assert_eq!(select_in_word(w, k), bit, "w={w:#x} k={k}");
                }
            }
        }
    }

    fn check_all_selects(chars: &[u8], encoding: OccEncoding) {
        let occ = build_occ_table(&Bwt::from_unpacked(chars.to_vec()), encoding);
        let n = chars.len() as u32;
        for c in 0..16u8 {
            let total = chars.iter().filter(|&&x| x == c).count() as u32;
            assert_eq!(occ.select(c, 0), None);
            assert_eq!(occ.select(c, total + 1), None);
            let mut expected = chars
                .iter()
                .enumerate()
                .filter(|(_, &x)| x == c)
                .map(|(i, _)| i as u32);
            for r in 1..=total {
                let pos = occ
                    .select(c, r)
                    .unwrap_or_else(|| panic!("select({c}, {r})"));
                assert_eq!(Some(pos), expected.next(), "c={c} r={r}");
                assert!(pos < n);
                assert_eq!(occ.rank(c, pos), r - 1);
                assert_eq!(occ.symbol_at(pos), c);
            }
        }
    }

    #[test]
    fn select_inverts_rank_on_long_random_bwt() {
        // > 4096 occurrences per base and > 8 superblocks so hints and both search levels
        // are exercised; both lane encodings.
        for (seed, wild) in [(1u64, false), (2, true), (3, true)] {
            let chars = random_chars(seed, 40_000, wild);
            check_all_selects(&chars, OccEncoding::Bitplane);
            check_all_selects(&chars, OccEncoding::OneHot);
        }
    }

    #[test]
    fn select_on_short_and_single_symbol_bwts() {
        check_all_selects(&[0], OccEncoding::Bitplane);
        check_all_selects(&[1, 1, 1, 0], OccEncoding::Bitplane);
        check_all_selects(&vec![2u8; 700], OccEncoding::Bitplane);
        check_all_selects(&vec![2u8; 700], OccEncoding::OneHot);
        let chars = random_chars(9, 513, true);
        check_all_selects(&chars, OccEncoding::Bitplane);
        check_all_selects(&chars, OccEncoding::OneHot);
    }

    #[test]
    fn select_without_hints_falls_back_to_full_search() {
        let chars = random_chars(4, 10_000, false);
        let mut occ = build_occ_table(&Bwt::from_unpacked(chars.clone()), OccEncoding::Bitplane);
        occ.select_hints = Vec::new();
        for c in 1..=4u8 {
            let total = chars.iter().filter(|&&x| x == c).count() as u32;
            for r in (1..=total).step_by(101) {
                let pos = occ.select(c, r).unwrap();
                assert_eq!(occ.rank(c, pos), r - 1);
                assert_eq!(occ.symbol_at(pos), c);
            }
        }
    }
}
