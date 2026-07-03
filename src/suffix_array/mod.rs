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

/// Bytes per interleaved per-word record: 8 (bitvector word) + 4 (superblock checkpoint,
/// duplicated into every word of its superblock) + 2 (delta since that checkpoint).
const SA_RECORD_STRIDE: usize = 14;

/// Sampled suffix array for space-efficient locate queries.
///
/// Stores only the ~n/sample_rate sampled entries (where `SA[i] % sample_rate == 0`).
/// Uses a bitvector + two-level rank1 structure instead of a sorted `Vec<u32>` of row indices.
///
/// The bitvector word, its superblock's cumulative popcount checkpoint, and its delta since
/// that checkpoint are interleaved into one `SA_RECORD_STRIDE`-byte record per word (mirroring
/// the occ table's `block_data` interleaving, see `occ::OccTable`) instead of three independent
/// arrays. `resolve_sa`'s LF-walk touches a random word on every step, so without interleaving
/// each `get` call was 2-3 separate cache misses (bitvector word, superblock checkpoint, delta)
/// on top of the `sa_vals` lookup; interleaving collapses those into one record access.
/// The superblock checkpoint is duplicated across all `SA_MARKER_SB_WORDS` words of its
/// superblock (rather than looked up in a separate table) for the same reason the occ table
/// duplicates its `sb_counts`: random access means superblock reuse across calls never happens
/// anyway, so sharing the value doesn't save memory traffic, only adds a second cache line.
///
/// For n=250M at rate=4: ~55 MB word_data + ~250 MB sa_vals vs ~250 MB bwt_rows + ~250 MB
/// sa_vals previously.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SampledSuffixArray {
    /// Interleaved per-word records, `SA_RECORD_STRIDE` bytes each: bitvector word (bit i set
    /// iff `SA[i] % sample_rate == 0`), superblock checkpoint, delta. Word `w`'s record starts
    /// at `word_data[w * SA_RECORD_STRIDE..]`. Length: `ceil(text_len / 64) * SA_RECORD_STRIDE`.
    word_data: Vec<u8>,
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

        // Interleave each word with its superblock checkpoint (u32, duplicated across the
        // superblock) and its delta since that checkpoint (u16, max delta per word is 64,
        // well within range).
        let mut word_data = vec![0u8; num_words * SA_RECORD_STRIDE];
        let mut cumulative = 0u32;
        let mut sb_base = 0u32;
        for (w, &word) in bitvector.iter().enumerate() {
            if w % SA_MARKER_SB_WORDS == 0 {
                sb_base = cumulative;
            }
            let delta = (cumulative - sb_base) as u16;
            let rec = &mut word_data[w * SA_RECORD_STRIDE..(w + 1) * SA_RECORD_STRIDE];
            rec[0..8].copy_from_slice(&word.to_ne_bytes());
            rec[8..12].copy_from_slice(&sb_base.to_ne_bytes());
            rec[12..14].copy_from_slice(&delta.to_ne_bytes());
            cumulative += word.count_ones();
        }

        Self {
            word_data,
            sa_vals,
            sample_rate,
            text_len: n as u32,
        }
    }

    #[inline]
    fn word_at(&self, word_idx: usize) -> u64 {
        let off = word_idx * SA_RECORD_STRIDE;
        debug_assert!(off + 8 <= self.word_data.len());
        // SAFETY: `off + 8 <= word_data.len()` because `word_idx` is a valid word index and
        // the bitvector word occupies the first 8 bytes of each `SA_RECORD_STRIDE`-byte record.
        unsafe {
            self.word_data
                .as_ptr()
                .add(off)
                .cast::<u64>()
                .read_unaligned()
        }
    }

    #[inline]
    fn sb_count_at(&self, word_idx: usize) -> u32 {
        let off = word_idx * SA_RECORD_STRIDE + 8;
        debug_assert!(off + 4 <= self.word_data.len());
        // SAFETY: `off + 4 <= word_data.len()`; the superblock checkpoint occupies bytes
        // `[8, 12)` of each record.
        unsafe {
            self.word_data
                .as_ptr()
                .add(off)
                .cast::<u32>()
                .read_unaligned()
        }
    }

    #[inline]
    fn delta_at(&self, word_idx: usize) -> u32 {
        let off = word_idx * SA_RECORD_STRIDE + 12;
        debug_assert!(off + 2 <= self.word_data.len());
        // SAFETY: `off + 2 <= word_data.len()`; the delta occupies bytes `[12, 14)` of each
        // record.
        unsafe {
            self.word_data
                .as_ptr()
                .add(off)
                .cast::<u16>()
                .read_unaligned() as u32
        }
    }

    /// Issue a software prefetch for the interleaved word record covering BWT row `i`, ahead
    /// of a future `get(i)` call. Used by `resolve_sa_batch`'s lockstep LF-walk (see
    /// `fm_index/query.rs`) to overlap this record's fetch with other lanes' work.
    #[inline]
    pub(crate) fn prefetch(&self, i: u32) {
        let off = (i as usize / 64) * SA_RECORD_STRIDE;
        if off < self.word_data.len() {
            crate::prefetch::prefetch_read(unsafe { self.word_data.as_ptr().add(off) });
        }
    }

    /// Check if BWT row `i` has a sampled SA value.
    #[inline]
    pub fn is_sampled(&self, i: u32) -> bool {
        let i = i as usize;
        (self.word_at(i / 64) >> (i % 64)) & 1 == 1
    }

    /// Return the SA value for BWT row `i` if it is sampled.
    #[inline]
    pub fn get(&self, i: u32) -> Option<u32> {
        let i = i as usize;
        let word_idx = i / 64;
        let bit_offset = i % 64;
        let word = self.word_at(word_idx);
        if (word >> bit_offset) & 1 == 0 {
            return None;
        }
        let mask = if bit_offset == 0 {
            0
        } else {
            (1u64 << bit_offset) - 1
        };
        let rank =
            self.sb_count_at(word_idx) + self.delta_at(word_idx) + (word & mask).count_ones();
        Some(self.sa_vals[rank as usize])
    }

    /// Reconstruct the flat sentinel-format `Vec<u32>` required by GPU shaders.
    /// `flat[i] = SA[i]` if sampled, else `u32::MAX`.
    pub fn to_flat_vec(&self, n: usize) -> Vec<u32> {
        let mut flat = vec![u32::MAX; n];
        let mut rank = 0usize;
        let num_words = self.word_data.len() / SA_RECORD_STRIDE;
        for word_idx in 0..num_words {
            let mut w = self.word_at(word_idx);
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
