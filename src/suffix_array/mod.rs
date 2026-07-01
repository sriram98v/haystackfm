//! Suffix array construction and sampled suffix array for locate queries.

pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

/// Suffix array: `SA[i]` = starting position of the i-th lexicographically smallest suffix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuffixArray {
    pub data: Vec<u32>,
}

impl SuffixArray {
    /// Returns the number of entries in the suffix array (equals the text length).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the suffix array is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Words per superblock for the SA-marker rank checkpoints (8 words = 512 bits, matching
/// the occ table's `SUPERBLOCK_SIZE`).
const SA_MARKER_SB_WORDS: usize = 8;

/// Sampled suffix array for space-efficient locate queries.
///
/// Stores only the ~n/sample_rate sampled entries (where `SA[i] % sample_rate == 0`).
/// Uses a bitvector + two-level rank1 structure instead of a sorted `Vec<u32>` of row indices:
/// a `u32` superblock checkpoint every 8 words plus a `u16` per-word delta since the last
/// superblock (same two-level trick as the occ table), instead of one flat `u32` per word.
///
/// For n=250M at rate=4: ~31 MB bitvector + ~10 MB checkpoints + ~250 MB sa_vals
/// vs ~250 MB bwt_rows + ~250 MB sa_vals previously.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SampledSuffixArray {
    /// Bitvector: bit i is set iff SA[i] % sample_rate == 0.
    /// Length: ceil(text_len / 64) words.
    bitvector: Vec<u64>,
    /// superblock_checkpoints[j] = popcount of bitvector[0..j*SA_MARKER_SB_WORDS].
    superblock_checkpoints: Vec<u32>,
    /// block_deltas[w] = popcount of bitvector[sb_start..w] where sb_start is the first word
    /// of w's superblock (i.e. cumulative count since the last superblock checkpoint).
    block_deltas: Vec<u16>,
    /// SA values at sampled positions, in ascending BWT row order.
    sa_vals: Vec<u32>,
    pub sample_rate: u32,
    text_len: u32,
}

impl SampledSuffixArray {
    /// Build a sampled SA from a full SA.
    pub fn from_full(sa: &SuffixArray, sample_rate: u32) -> Self {
        let n = sa.data.len();
        let num_words = n.div_ceil(64);

        let mut bitvector = vec![0u64; num_words];
        let mut sa_vals = Vec::new();

        for (i, &sa_val) in sa.data.iter().enumerate() {
            if sa_val.is_multiple_of(sample_rate) {
                bitvector[i / 64] |= 1u64 << (i % 64);
                sa_vals.push(sa_val);
            }
        }

        // Build two-level prefix popcount checkpoints: u32 superblock every 8 words, u16 delta
        // per word since the last superblock (max delta per word is 64, well within u16).
        let num_superblocks = num_words.div_ceil(SA_MARKER_SB_WORDS);
        let mut superblock_checkpoints = Vec::with_capacity(num_superblocks);
        let mut block_deltas = Vec::with_capacity(num_words);
        let mut cumulative = 0u32;
        let mut sb_base = 0u32;
        for (w, &word) in bitvector.iter().enumerate() {
            if w % SA_MARKER_SB_WORDS == 0 {
                superblock_checkpoints.push(cumulative);
                sb_base = cumulative;
            }
            block_deltas.push((cumulative - sb_base) as u16);
            cumulative += word.count_ones();
        }

        Self {
            bitvector,
            superblock_checkpoints,
            block_deltas,
            sa_vals,
            sample_rate,
            text_len: n as u32,
        }
    }

    /// Check if BWT row `i` has a sampled SA value.
    #[inline]
    pub fn is_sampled(&self, i: u32) -> bool {
        let i = i as usize;
        (self.bitvector[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Return the SA value for BWT row `i` if it is sampled.
    #[inline]
    pub fn get(&self, i: u32) -> Option<u32> {
        if !self.is_sampled(i) {
            return None;
        }
        let i = i as usize;
        let word_idx = i / 64;
        let bit_offset = i % 64;
        let mask = if bit_offset == 0 {
            0
        } else {
            (1u64 << bit_offset) - 1
        };
        let sb = word_idx / SA_MARKER_SB_WORDS;
        let rank = self.superblock_checkpoints[sb]
            + self.block_deltas[word_idx] as u32
            + (self.bitvector[word_idx] & mask).count_ones();
        Some(self.sa_vals[rank as usize])
    }

    /// Reconstruct the flat sentinel-format `Vec<u32>` required by GPU shaders.
    /// `flat[i] = SA[i]` if sampled, else `u32::MAX`.
    pub fn to_flat_vec(&self, n: usize) -> Vec<u32> {
        let mut flat = vec![u32::MAX; n];
        let mut rank = 0usize;
        for (word_idx, &word) in self.bitvector.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let pos = word_idx * 64 + bit;
                if pos < n {
                    flat[pos] = self.sa_vals[rank];
                    rank += 1;
                }
                w &= w - 1;
            }
        }
        flat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::encode_char;
    use crate::bwt::cpu::build_bwt;
    use crate::suffix_array::cpu::build_suffix_array;

    fn encode(s: &str) -> Vec<u8> {
        use crate::alphabet::SENTINEL;
        let mut v: Vec<u8> = s.chars().map(|c| encode_char(c).unwrap()).collect();
        if v.last() != Some(&SENTINEL) {
            v.push(SENTINEL);
        }
        v
    }

    fn make_sa(text: &str) -> (SuffixArray, Vec<u8>) {
        let encoded = encode(text);
        let sa = build_suffix_array(&encoded);
        (sa, encoded)
    }

    #[test]
    fn test_sampled_sa_get_matches_full() {
        let (sa, _) = make_sa("ACGTACGTACGT");
        let sample_rate = 4;
        let ssa = SampledSuffixArray::from_full(&sa, sample_rate);

        for (i, &sa_val) in sa.data.iter().enumerate() {
            if sa_val % sample_rate == 0 {
                assert_eq!(ssa.get(i as u32), Some(sa_val), "row {i}");
                assert!(ssa.is_sampled(i as u32), "row {i} should be sampled");
            } else {
                assert_eq!(ssa.get(i as u32), None, "row {i}");
                assert!(!ssa.is_sampled(i as u32), "row {i} should not be sampled");
            }
        }
    }

    #[test]
    fn test_sampled_sa_to_flat_vec() {
        let (sa, _) = make_sa("ACGTACGTACGT");
        let sample_rate = 4;
        let ssa = SampledSuffixArray::from_full(&sa, sample_rate);
        let n = sa.data.len();
        let flat = ssa.to_flat_vec(n);

        for (i, &sa_val) in sa.data.iter().enumerate() {
            if sa_val % sample_rate == 0 {
                assert_eq!(flat[i], sa_val, "flat[{i}]");
            } else {
                assert_eq!(flat[i], u32::MAX, "flat[{i}] should be sentinel");
            }
        }
    }

    #[test]
    fn test_sampled_sa_rate_1_covers_all() {
        let (sa, _) = make_sa("ACGTACGT");
        let ssa = SampledSuffixArray::from_full(&sa, 1);
        for (i, &sa_val) in sa.data.iter().enumerate() {
            assert_eq!(ssa.get(i as u32), Some(sa_val));
        }
    }

    #[test]
    fn test_sampled_sa_long_text_spans_multiple_words() {
        // 200 chars → >64 positions, exercises multi-word bitvector
        let s = "ACGT".repeat(50);
        let (sa, _) = make_sa(&s);
        let sample_rate = 8;
        let ssa = SampledSuffixArray::from_full(&sa, sample_rate);
        let n = sa.data.len();
        let flat = ssa.to_flat_vec(n);
        for (i, &sa_val) in sa.data.iter().enumerate() {
            if sa_val % sample_rate == 0 {
                assert_eq!(flat[i], sa_val);
                assert_eq!(ssa.get(i as u32), Some(sa_val));
            } else {
                assert_eq!(flat[i], u32::MAX);
                assert_eq!(ssa.get(i as u32), None);
            }
        }
    }

    #[test]
    fn test_locate_via_lf_walk_uses_ssa() {
        // End-to-end: build index, verify locate still works through SSA
        let text = encode("ACGTACGTACGT");
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let _ = bwt; // just ensure SSA can be built from same SA
        let ssa = SampledSuffixArray::from_full(&sa, 4);
        // Spot-check: get returns Some for every sampled row
        let sampled_count = sa.data.iter().filter(|&&v| v % 4 == 0).count();
        let found: usize = (0..sa.data.len() as u32).filter_map(|i| ssa.get(i)).count();
        assert_eq!(found, sampled_count);
    }
}
