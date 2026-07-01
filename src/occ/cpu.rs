//! CPU implementation of the Occ table (rank data structure) over the BWT.

use super::{OccTable, BLOCK_SIZE, SUPERBLOCK_SIZE};
use crate::alphabet::ALPHABET_SIZE;
use crate::bwt::Bwt;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Build the two-level Occ table from the BWT on the CPU.
///
/// Phase 1 (parallel): compute per-block bitvectors and character counts (full 16-wide,
///   cheap local scratch space).
/// Phase 2: determine the effective alphabet actually present in the BWT and assign each
///   present symbol a dense lane index — absent symbols (most of the 16 IUPAC codes, for a
///   plain ACGT[+N] reference) get no storage at all.
/// Phase 3 (sequential): prefix-sum into superblock u32 checkpoints and per-block u16 deltas,
///   written directly in the compact lane layout.
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

    // Determine the effective alphabet: symbols with at least one occurrence get a lane.
    let mut totals = [0u32; ALPHABET_SIZE];
    for (_, counts) in &blocks {
        for (t, c) in totals.iter_mut().zip(counts.iter()) {
            *t += c;
        }
    }
    let mut symbol_to_lane = [u8::MAX; ALPHABET_SIZE];
    let mut num_lanes: u8 = 0;
    for (c, &total) in totals.iter().enumerate() {
        if total > 0 {
            symbol_to_lane[c] = num_lanes;
            num_lanes += 1;
        }
    }
    let num_lanes_usize = num_lanes as usize;

    // ceil(log2(num_lanes)) bitplanes encode each position's lane index directly, instead of
    // one one-hot bitvector per lane (see `occ::OccTable` docs for the memory tradeoff).
    let num_planes = if num_lanes <= 1 {
        0
    } else {
        (u8::BITS - (num_lanes - 1).leading_zeros()) as usize
    };

    let mut superblock_checkpoints = Vec::with_capacity(num_superblocks * num_lanes_usize);
    let mut block_deltas = Vec::with_capacity(num_blocks * num_lanes_usize);
    let mut planes = Vec::with_capacity(num_blocks * num_planes);

    let mut cumulative = vec![0u32; num_lanes_usize];
    let mut sb_base = vec![0u32; num_lanes_usize];

    for (b, (block_bits, block_counts)) in blocks.iter().enumerate() {
        if b % blocks_per_sb == 0 {
            superblock_checkpoints.extend_from_slice(&cumulative);
            sb_base.copy_from_slice(&cumulative);
        }

        let mut block_planes = vec![0u64; num_planes];
        for c in 0..ALPHABET_SIZE {
            let lane = symbol_to_lane[c];
            if lane == u8::MAX {
                continue;
            }
            let lane = lane as usize;
            block_deltas.push((cumulative[lane] - sb_base[lane]) as u16);
            for (p, plane) in block_planes.iter_mut().enumerate() {
                if (lane >> p) & 1 == 1 {
                    *plane |= block_bits[c];
                }
            }
            cumulative[lane] += block_counts[c];
        }
        planes.extend_from_slice(&block_planes);
    }

    OccTable::from_parts(
        num_lanes,
        symbol_to_lane,
        superblock_checkpoints,
        block_deltas,
        planes,
        n,
    )
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

    #[test]
    fn test_occ_compacts_to_effective_alphabet() {
        // "ACGTACGTACGT" + sentinel uses only 5 of the 16 IUPAC symbols ($,A,C,G,T).
        // This is the whole point of the compaction: 11 unused IUPAC lanes cost nothing.
        let text = encode("ACGTACGTACGT");
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let occ = build_occ_table(&bwt);

        assert_eq!(occ.num_lanes(), 5, "expected exactly {{$,A,C,G,T}} lanes");

        // Still bit-correct against naive rank over the full 16-symbol space, including
        // the 11 symbols that never appear (rank must be 0 everywhere for those).
        let n = bwt.len() as u32;
        for c in 0..ALPHABET_SIZE as u8 {
            for i in 0..=n {
                assert_eq!(
                    occ.rank(c, i),
                    naive_rank(&bwt, c, i),
                    "Occ({c}, {i}) mismatch"
                );
            }
        }
    }
}
