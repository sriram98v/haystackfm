//! CPU implementation of the Occ table (rank data structure) over the BWT.

use super::{OccTable, BLOCK_SIZE, SUPERBLOCK_SIZE};
use crate::alphabet::ALPHABET_SIZE;
use crate::bwt::Bwt;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Build the two-level Occ table from the BWT on the CPU.
///
/// Phase 1 (parallel): compute per-block bitvectors and character counts.
/// Phase 2 (sequential): prefix-sum to produce superblock u32 checkpoints
///   and per-block u16 deltas (count since last superblock).
pub fn build_occ_table(bwt: &Bwt) -> OccTable {
    let n = bwt.len() as u32;
    let num_blocks = n.div_ceil(BLOCK_SIZE) as usize;
    let num_superblocks = n.div_ceil(SUPERBLOCK_SIZE) as usize;
    let blocks_per_sb = (SUPERBLOCK_SIZE / BLOCK_SIZE) as usize;

    let chars: Vec<u8> = bwt.iter_chars().collect();

    #[cfg(not(target_arch = "wasm32"))]
    let blocks: Vec<([u64; ALPHABET_SIZE], [u32; ALPHABET_SIZE])> = (0..num_blocks)
        .into_par_iter()
        .map(|block_idx| compute_block(&chars, block_idx, n))
        .collect();

    #[cfg(target_arch = "wasm32")]
    let blocks: Vec<([u64; ALPHABET_SIZE], [u32; ALPHABET_SIZE])> = (0..num_blocks)
        .map(|block_idx| compute_block(&chars, block_idx, n))
        .collect();

    let mut superblock_checkpoints = Vec::with_capacity(num_superblocks);
    let mut block_deltas = Vec::with_capacity(num_blocks);
    let mut bitvectors = Vec::with_capacity(num_blocks);

    let mut cumulative = [0u32; ALPHABET_SIZE];
    let mut sb_base = [0u32; ALPHABET_SIZE];

    for (b, (block_bits, block_counts)) in blocks.iter().enumerate() {
        if b % blocks_per_sb == 0 {
            superblock_checkpoints.push(cumulative);
            sb_base = cumulative;
        }
        let mut delta = [0u16; ALPHABET_SIZE];
        for c in 0..ALPHABET_SIZE {
            delta[c] = (cumulative[c] - sb_base[c]) as u16;
        }
        block_deltas.push(delta);
        bitvectors.push(*block_bits);
        for c in 0..ALPHABET_SIZE {
            cumulative[c] += block_counts[c];
        }
    }

    OccTable::from_parts(superblock_checkpoints, block_deltas, bitvectors, n)
}

#[inline]
fn compute_block(
    chars: &[u8],
    block_idx: usize,
    n: u32,
) -> ([u64; ALPHABET_SIZE], [u32; ALPHABET_SIZE]) {
    let start = block_idx as u32 * BLOCK_SIZE;
    let end = std::cmp::min(start + BLOCK_SIZE, n);
    let mut block_bits = [0u64; ALPHABET_SIZE];
    let mut block_counts = [0u32; ALPHABET_SIZE];
    for pos in start..end {
        let ch = chars[pos as usize] as usize;
        let bit_pos = pos - start;
        if ch < ALPHABET_SIZE {
            block_bits[ch] |= 1u64 << bit_pos;
            block_counts[ch] += 1;
        }
    }
    (block_bits, block_counts)
}

/// Naive rank computation for testing: count occurrences of c in bwt[0..i).
pub fn naive_rank(bwt: &Bwt, c: u8, i: u32) -> u32 {
    (0..i as usize).filter(|&j| bwt.get(j) == c).count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::*;
    use crate::bwt::cpu::build_bwt;
    use crate::suffix_array::cpu::build_suffix_array;

    fn encode(s: &str) -> Vec<u8> {
        let mut v: Vec<u8> = s.chars().map(|c| encode_char(c).unwrap()).collect();
        if v.last() != Some(&SENTINEL) {
            v.push(SENTINEL);
        }
        v
    }

    #[test]
    fn test_occ_matches_naive() {
        let text = encode("ACGTACGTACGT");
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let occ = build_occ_table(&bwt);

        let n = bwt.len() as u32;
        for c in 0..ALPHABET_SIZE as u8 {
            for i in 0..=n {
                let expected = naive_rank(&bwt, c, i);
                let actual = occ.rank(c, i);
                assert_eq!(
                    actual, expected,
                    "Occ({}, {}) = {} but expected {}",
                    c, i, actual, expected
                );
            }
        }
    }

    #[test]
    fn test_occ_long_text() {
        // Text longer than one block (>64 chars)
        let s = "ACGT".repeat(20); // 80 chars + sentinel = 81
        let text = encode(&s);
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let occ = build_occ_table(&bwt);

        let n = bwt.len() as u32;
        for c in 0..ALPHABET_SIZE as u8 {
            for i in 0..=n {
                let expected = naive_rank(&bwt, c, i);
                let actual = occ.rank(c, i);
                assert_eq!(actual, expected, "Occ({}, {}) mismatch", c, i);
            }
        }
    }

    #[test]
    fn test_occ_boundary_values() {
        let text = encode("ACGTACGTACGT");
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let occ = build_occ_table(&bwt);

        // rank(c, 0) should always be 0
        for c in 0..ALPHABET_SIZE as u8 {
            assert_eq!(occ.rank(c, 0), 0);
        }

        // rank(c, n) should equal total count of c in bwt
        let n = bwt.len() as u32;
        for c in 0..ALPHABET_SIZE as u8 {
            let total = bwt.iter_chars().filter(|&ch| ch == c).count() as u32;
            assert_eq!(occ.rank(c, n), total);
        }
    }
}
