use super::FmIndex;

impl FmIndex {
    /// Count occurrences of a pattern in the indexed text.
    ///
    /// Uses backward search: processes the pattern from right to left,
    /// narrowing the SA interval [lo, hi) at each step.
    ///
    /// Time: O(m) where m = pattern length.
    pub fn count(&self, pattern: &[u8]) -> u32 {
        let (lo, hi) = self.backward_search(pattern);
        hi.saturating_sub(lo)
    }

    /// Locate all occurrences of a pattern in the indexed text.
    ///
    /// Returns `(sequence_id, position)` tuples where `sequence_id` is the FASTA
    /// header of the matching sequence and `position` is 0-based within that sequence.
    ///
    /// Time: O(m + occ * k) where m = pattern length, occ = number of occurrences,
    /// k = SA sample rate.
    pub fn locate(&self, pattern: &[u8]) -> Vec<(String, u32)> {
        let (lo, hi) = self.backward_search(pattern);
        if lo >= hi {
            return vec![];
        }

        (lo..hi)
            .map(|i| {
                let text_pos = self.resolve_sa(i);
                let (seq_idx, pos_in_seq) = self
                    .map_position(text_pos)
                    .expect("resolved SA position must be within text bounds");
                (self.seq_headers[seq_idx as usize].clone(), pos_in_seq)
            })
            .collect()
    }

    /// Backward search: find the SA interval [lo, hi) for the pattern.
    fn backward_search(&self, pattern: &[u8]) -> (u32, u32) {
        if pattern.is_empty() {
            return (0, self.text_len);
        }

        let mut lo = 0u32;
        let mut hi = self.text_len;

        for &c in pattern.iter().rev() {
            let c_val = self.c_array.get(c);
            lo = c_val + self.occ.rank(c, lo);
            hi = c_val + self.occ.rank(c, hi);
            if lo >= hi {
                return (lo, hi);
            }
        }

        (lo, hi)
    }

    /// Resolve a BWT position to a text position using the sampled SA.
    ///
    /// Walk backwards through the BWT via LF-mapping until hitting a sampled position.
    pub(crate) fn resolve_sa(&self, mut i: u32) -> u32 {
        let mut steps = 0u32;
        loop {
            if let Some(sa_val) = self.sa_samples.get(i) {
                return sa_val + steps;
            }
            i = self.lf_mapping(i);
            steps += 1;
        }
    }

    /// Map a text position back to (sequence_index, position_within_sequence).
    pub fn map_position(&self, text_pos: u32) -> Option<(u32, u32)> {
        // Binary search for the sequence containing this position
        let seq_idx = self
            .seq_boundaries
            .partition_point(|&boundary| boundary <= text_pos);
        if seq_idx >= self.seq_boundaries.len() {
            return None;
        }
        let seq_start = if seq_idx == 0 {
            0
        } else {
            self.seq_boundaries[seq_idx - 1]
        };
        let pos_in_seq = text_pos - seq_start;
        Some((seq_idx as u32, pos_in_seq))
    }

    /// Locate all occurrences of multiple patterns in parallel on the GPU.
    ///
    /// Each pattern is processed by a separate GPU thread during backward search.
    /// Returns one `Vec<(sequence_header, position)>` per query, in the same order
    /// as `queries`. Positions are 0-based within each sequence.
    ///
    /// Requires the `gpu` feature.
    #[cfg(feature = "gpu")]
    pub async fn locate_gpu(
        &self,
        queries: &[impl AsRef<[u8]>],
    ) -> Result<Vec<Vec<(String, u32)>>, crate::error::FmIndexError> {
        use crate::gpu::locate::locate_batch_gpu;
        let encoded: Vec<&[u8]> = queries.iter().map(|q| q.as_ref()).collect();
        let by_idx = locate_batch_gpu(self, &encoded).await?;
        Ok(by_idx
            .into_iter()
            .map(|hits| {
                hits.into_iter()
                    .map(|(seq_idx, pos)| (self.seq_headers[seq_idx as usize].clone(), pos))
                    .collect()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::alphabet::*;
    use crate::fm_index::{FmIndex, FmIndexConfig};

    fn make_index(s: &str) -> FmIndex {
        let seq = DnaSequence::from_str(s).unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 1, // Full SA for exact testing
            use_gpu: false,
        };
        FmIndex::build_cpu(&[seq], &config).unwrap()
    }

    fn make_index_multi(seqs: &[&str]) -> FmIndex {
        let sequences: Vec<DnaSequence> = seqs
            .iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect();
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
        };
        FmIndex::build_cpu(&sequences, &config).unwrap()
    }

    fn encode_pattern(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    /// Count overlapping occurrences of pattern in text.
    fn naive_count(text: &str, pattern: &str) -> u32 {
        if pattern.is_empty() || pattern.len() > text.len() {
            return 0;
        }
        (0..=text.len() - pattern.len())
            .filter(|&i| &text[i..i + pattern.len()] == pattern)
            .count() as u32
    }

    #[test]
    fn test_count_basic() {
        let idx = make_index("ACGTACGT");
        assert_eq!(idx.count(&encode_pattern("ACGT")), 2);
        assert_eq!(idx.count(&encode_pattern("ACG")), 2);
        assert_eq!(idx.count(&encode_pattern("CGT")), 2);
        assert_eq!(idx.count(&encode_pattern("ACGTACGT")), 1);
    }

    #[test]
    fn test_count_single_char() {
        let idx = make_index("ACGTACGT");
        assert_eq!(idx.count(&encode_pattern("A")), 2);
        assert_eq!(idx.count(&encode_pattern("C")), 2);
        assert_eq!(idx.count(&encode_pattern("G")), 2);
        assert_eq!(idx.count(&encode_pattern("T")), 2);
    }

    #[test]
    fn test_count_not_found() {
        let idx = make_index("AAAA");
        assert_eq!(idx.count(&encode_pattern("C")), 0);
        assert_eq!(idx.count(&encode_pattern("AC")), 0);
    }

    #[test]
    fn test_count_matches_naive() {
        let text = "ACGTTAGCCAGTACGT";
        let idx = make_index(text);

        for pattern in &["A", "AC", "ACG", "GT", "GCC", "TAG", "ACGT", "AGTACGT"] {
            let expected = naive_count(text, pattern);
            let actual = idx.count(&encode_pattern(pattern));
            assert_eq!(
                actual, expected,
                "count('{}') = {} but expected {}",
                pattern, actual, expected
            );
        }
    }

    #[test]
    fn test_locate_basic() {
        let idx = make_index("ACGTACGT");
        let mut positions = idx.locate(&encode_pattern("ACGT"));
        positions.sort();
        assert_eq!(
            positions,
            vec![("seq_0".to_string(), 0), ("seq_0".to_string(), 4)]
        );
    }

    #[test]
    fn test_locate_single_occurrence() {
        let idx = make_index("ACGTACGT");
        let positions = idx.locate(&encode_pattern("ACGTACGT"));
        assert_eq!(positions, vec![("seq_0".to_string(), 0)]);
    }

    #[test]
    fn test_locate_not_found() {
        let idx = make_index("AAAA");
        let positions = idx.locate(&encode_pattern("C"));
        assert!(positions.is_empty());
    }

    #[test]
    fn test_locate_positions_valid() {
        let text = "ACGTTAGCCAGTACGT";
        let idx = make_index(text);
        let encoded_text: Vec<u8> = text.chars().map(|c| encode_char(c).unwrap()).collect();

        let pattern = "GT";
        let encoded_pattern = encode_pattern(pattern);
        let positions = idx.locate(&encoded_pattern);

        for (_, pos) in &positions {
            let pos = *pos as usize;
            assert_eq!(
                &encoded_text[pos..pos + pattern.len()],
                encoded_pattern.as_slice(),
                "pattern '{}' should be found at position {}",
                pattern,
                pos
            );
        }
        assert_eq!(positions.len(), naive_count(text, pattern) as usize);
    }

    #[test]
    fn test_count_with_sampled_sa() {
        let seq = DnaSequence::from_str("ACGTACGTACGT").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 4,
            use_gpu: false,
        };
        let idx = FmIndex::build_cpu(&[seq], &config).unwrap();
        // count doesn't use SA, so sampling shouldn't matter
        assert_eq!(idx.count(&encode_pattern("ACGT")), 3);
    }

    #[test]
    fn test_locate_with_sampled_sa() {
        let seq = DnaSequence::from_str("ACGTACGTACGT").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 4,
            use_gpu: false,
        };
        let idx = FmIndex::build_cpu(&[seq], &config).unwrap();
        let mut positions = idx.locate(&encode_pattern("ACGT"));
        positions.sort();
        assert_eq!(
            positions,
            vec![
                ("seq_0".to_string(), 0),
                ("seq_0".to_string(), 4),
                ("seq_0".to_string(), 8),
            ]
        );
    }

    #[test]
    fn test_multi_sequence() {
        let idx = make_index_multi(&["ACGT", "TGCA"]);
        // "ACGT" appears in first sequence
        assert_eq!(idx.count(&encode_pattern("ACGT")), 1);
        // "TGCA" appears in second sequence
        assert_eq!(idx.count(&encode_pattern("TGCA")), 1);
    }

    #[test]
    fn test_map_position() {
        let idx = make_index_multi(&["ACGT", "TGCA"]);
        // First sequence: positions 0..4, separator at 4
        // Second sequence: positions 5..9, separator at 9
        assert_eq!(idx.map_position(0), Some((0, 0)));
        assert_eq!(idx.map_position(3), Some((0, 3)));
        assert_eq!(idx.map_position(5), Some((1, 0)));
        assert_eq!(idx.map_position(8), Some((1, 3)));
    }
}
