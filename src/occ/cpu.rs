//! CPU implementation of the Occ table (rank data structure) over the BWT.

use super::{OccEncoding, OccTable, BLOCK_SIZE, SUPERBLOCK_SIZE};
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
///
/// `encoding` selects the Level-3 lane storage — see [`OccEncoding`] for the memory/query-speed
/// tradeoff. Phase 1's per-block bitvectors are computed identically either way; `encoding`
/// only changes how they're packed in Phase 3 (bitplanes vs stored directly).
pub fn build_occ_table(bwt: &Bwt, encoding: OccEncoding) -> OccTable {
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

    // Bitplane: ceil(log2(num_lanes)) planes encode each position's lane index directly.
    // OneHot: one u64 bitvector per lane, stored as-is (see `occ::OccEncoding` for the
    // memory/query-speed tradeoff).
    let num_planes = if num_lanes <= 1 {
        0
    } else {
        (u8::BITS - (num_lanes - 1).leading_zeros()) as usize
    };
    let lane_data_width = match encoding {
        OccEncoding::Bitplane => num_planes,
        OccEncoding::OneHot => num_lanes_usize,
    };

    let mut superblock_checkpoints = Vec::with_capacity(num_superblocks * num_lanes_usize);
    let mut block_deltas = Vec::with_capacity(num_blocks * num_lanes_usize);
    let mut lane_data = Vec::with_capacity(num_blocks * lane_data_width);

    let mut cumulative = vec![0u32; num_lanes_usize];
    let mut sb_base = vec![0u32; num_lanes_usize];

    for (b, (block_bits, block_counts)) in blocks.iter().enumerate() {
        if b % blocks_per_sb == 0 {
            superblock_checkpoints.extend_from_slice(&cumulative);
            sb_base.copy_from_slice(&cumulative);
        }

        let mut block_lane_data = vec![0u64; lane_data_width];
        for c in 0..ALPHABET_SIZE {
            let lane = symbol_to_lane[c];
            if lane == u8::MAX {
                continue;
            }
            let lane = lane as usize;
            block_deltas.push((cumulative[lane] - sb_base[lane]) as u16);
            match encoding {
                OccEncoding::Bitplane => {
                    for (p, plane) in block_lane_data.iter_mut().enumerate() {
                        if (lane >> p) & 1 == 1 {
                            *plane |= block_bits[c];
                        }
                    }
                }
                OccEncoding::OneHot => {
                    block_lane_data[lane] = block_bits[c];
                }
            }
            cumulative[lane] += block_counts[c];
        }
        lane_data.extend_from_slice(&block_lane_data);
    }

    OccTable::from_parts(
        num_lanes,
        symbol_to_lane,
        superblock_checkpoints,
        block_deltas,
        lane_data,
        n,
        encoding,
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
        let occ = build_occ_table(&bwt, OccEncoding::Bitplane);

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
        let occ = build_occ_table(&bwt, OccEncoding::Bitplane);

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
        let occ = build_occ_table(&bwt, OccEncoding::Bitplane);

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
    fn test_lf_step_matches_symbol_at_and_rank() {
        // Longer-than-one-block text so both block-interior and superblock-boundary
        // positions get exercised.
        let s = "ACGT".repeat(50); // 200 chars + sentinel
        let text = encode(&s);
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let occ = build_occ_table(&bwt, OccEncoding::Bitplane);

        let n = bwt.len() as u32;
        for pos in 0..n {
            let (sym, rank) = occ.lf_step(pos);
            assert_eq!(sym, occ.symbol_at(pos), "symbol mismatch at pos {pos}");
            assert_eq!(rank, occ.rank(sym, pos), "rank mismatch at pos {pos}");
        }
    }

    #[test]
    fn test_occ_compacts_to_effective_alphabet() {
        // "ACGTACGTACGT" + sentinel uses only 5 of the 16 IUPAC symbols ($,A,C,G,T).
        // This is the whole point of the compaction: 11 unused IUPAC lanes cost nothing.
        let text = encode("ACGTACGTACGT");
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let occ = build_occ_table(&bwt, OccEncoding::Bitplane);

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

    #[test]
    fn test_bitplane_and_onehot_encodings_agree() {
        // Bitplane and OneHot must be observationally identical: same rank, symbol_at,
        // lf_step, and reconstruct_bwt_u32 for every position/symbol, over text spanning
        // multiple blocks and superblocks so both encodings' boundary handling is exercised.
        let s = "ACGT".repeat(50); // 200 chars + sentinel
        let text = encode(&s);
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);

        let bitplane = build_occ_table(&bwt, OccEncoding::Bitplane);
        let onehot = build_occ_table(&bwt, OccEncoding::OneHot);

        assert_eq!(bitplane.num_lanes(), onehot.num_lanes());

        let n = bwt.len() as u32;
        for c in 0..ALPHABET_SIZE as u8 {
            for i in 0..=n {
                assert_eq!(
                    bitplane.rank(c, i),
                    onehot.rank(c, i),
                    "rank({c}, {i}) mismatch between encodings"
                );
            }
        }
        for pos in 0..n {
            assert_eq!(
                bitplane.symbol_at(pos),
                onehot.symbol_at(pos),
                "symbol_at({pos}) mismatch"
            );
            assert_eq!(
                bitplane.lf_step(pos),
                onehot.lf_step(pos),
                "lf_step({pos}) mismatch"
            );
        }
        assert_eq!(
            bitplane.reconstruct_bwt_u32(),
            onehot.reconstruct_bwt_u32(),
            "reconstruct_bwt_u32 mismatch"
        );
    }

    #[test]
    fn rank_pair_matches_scalar_rank() {
        let text = encode(&"ACGTTGCAACGT".repeat(30));
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let n = bwt.len() as u32;
        for enc in [OccEncoding::Bitplane, OccEncoding::OneHot] {
            let occ = build_occ_table(&bwt, enc);
            for c in 0..ALPHABET_SIZE as u8 {
                // A spread of (lo, hi) border pairs, including lo == hi and the i == 0 edge.
                for lo in (0..=n).step_by(7) {
                    for hi in (lo..=n).step_by(11) {
                        assert_eq!(
                            occ.rank_pair(c, lo, hi),
                            (occ.rank(c, lo), occ.rank(c, hi)),
                            "rank_pair({c}, {lo}, {hi}) mismatch ({enc:?})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rank_many_matches_scalar_rank() {
        let text = encode(&"GATTACAGATTACA".repeat(20));
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let n = bwt.len() as u32;
        for enc in [OccEncoding::Bitplane, OccEncoding::OneHot] {
            let occ = build_occ_table(&bwt, enc);
            let mut queries: Vec<(u8, u32)> = Vec::new();
            for c in 0..ALPHABET_SIZE as u8 {
                for i in (0..=n).step_by(5) {
                    queries.push((c, i));
                }
            }
            let mut out = vec![0u32; queries.len()];
            occ.rank_many(&queries, &mut out);
            for (&(c, i), &got) in queries.iter().zip(&out) {
                assert_eq!(got, occ.rank(c, i), "rank_many({c}, {i}) mismatch ({enc:?})");
            }
        }
    }
}
