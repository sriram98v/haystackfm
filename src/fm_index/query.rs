use super::FmIndex;

impl FmIndex {
    /// Count occurrences of a pattern in the indexed text.
    ///
    /// IUPAC ambiguity codes in the pattern and/or reference are resolved via
    /// base-set intersection: a query symbol matches a reference symbol when
    /// their IUPAC base sets share at least one nucleotide.
    pub fn count(&self, pattern: &[u8]) -> u32 {
        self.backward_search(pattern)
            .iter()
            .map(|(lo, hi)| hi - lo)
            .sum()
    }

    /// Locate all occurrences of a pattern in the indexed text.
    ///
    /// Returns `(sequence_id, position)` tuples where `sequence_id` is the FASTA
    /// header of the matching sequence and `position` is 0-based within that sequence.
    ///
    /// IUPAC ambiguity codes are resolved via base-set intersection (see `count`).
    pub fn locate(&self, pattern: &[u8]) -> Vec<(String, u32)> {
        let rows: Vec<u32> = self
            .backward_search(pattern)
            .into_iter()
            .flat_map(|(lo, hi)| lo..hi)
            .collect();
        let mut text_positions = Vec::with_capacity(rows.len());
        self.resolve_sa_batch(&rows, &mut text_positions);
        text_positions
            .into_iter()
            .map(|text_pos| {
                let (seq_idx, pos_in_seq) = self
                    .map_position(text_pos)
                    .expect("resolved SA position must be within text bounds");
                (self.seq_headers[seq_idx as usize].clone(), pos_in_seq)
            })
            .collect()
    }

    /// Locate all occurrences of a pattern, returning raw text positions.
    ///
    /// Cheaper than [`Self::locate`] when sequence header strings are not needed — avoids
    /// heap-allocating and cloning a `String` per hit. Positions are absolute offsets
    /// into the concatenated text (including sentinel bytes between sequences).
    pub fn locate_positions(&self, pattern: &[u8]) -> Vec<u32> {
        let rows: Vec<u32> = self
            .backward_search(pattern)
            .into_iter()
            .flat_map(|(lo, hi)| lo..hi)
            .collect();
        let mut out = Vec::with_capacity(rows.len());
        self.resolve_sa_batch(&rows, &mut out);
        out
    }

    /// Backward search returning a union of SA intervals covering all IUPAC-compatible matches.
    ///
    /// Each step expands the query symbol to all compatible reference symbols via
    /// base-set intersection, collects one interval per compatible symbol, then
    /// merges overlapping intervals before the next step.
    ///
    /// When a `lookup` table is present and the last `lookup.depth` symbols are all
    /// in ACGT, the search is seeded from the table (O(1)) and only the remaining
    /// prefix characters are processed character-by-character.
    pub(crate) fn backward_search(&self, pattern: &[u8]) -> Vec<(u32, u32)> {
        if pattern.is_empty() {
            return vec![(0, self.text_len)];
        }

        // Attempt to seed from the depth-k lookup table.
        let (mut intervals, start_rev_idx) = if let Some(ref lut) = self.lookup {
            let depth = lut.depth as usize;
            if pattern.len() >= depth {
                let suffix = &pattern[pattern.len() - depth..];
                if let Some(iv) = lut.get(suffix) {
                    if iv.0 >= iv.1 {
                        return vec![];
                    }
                    (vec![iv], pattern.len() - depth)
                } else {
                    (vec![(0u32, self.text_len)], pattern.len())
                }
            } else {
                (vec![(0u32, self.text_len)], pattern.len())
            }
        } else {
            (vec![(0u32, self.text_len)], pattern.len())
        };

        // Scratch buffer reused across steps to avoid per-step allocations.
        let mut next: Vec<(u32, u32)> = Vec::with_capacity(16);

        let compat_fn = self.alphabet_fns.compatible_fn;
        for &c in pattern[..start_rev_idx].iter().rev() {
            let compat = compat_fn(c);
            next.clear();
            for &(lo, hi) in &intervals {
                for &r in compat {
                    // Skip symbols absent from the text — they contribute empty intervals.
                    if self.c_array.symbol_count(r, self.text_len) == 0 {
                        continue;
                    }
                    let c_val = self.c_array.get(r);
                    let new_lo = c_val + self.occ.rank(r, lo);
                    let new_hi = c_val + self.occ.rank(r, hi);
                    if new_lo < new_hi {
                        next.push((new_lo, new_hi));
                    }
                }
            }
            // Merge only when multiple intervals exist (ACGT path: always 1).
            if next.len() > 1 {
                merge_intervals_inplace(&mut next);
            }
            if next.is_empty() {
                return vec![];
            }
            std::mem::swap(&mut intervals, &mut next);
        }

        intervals
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

    /// Resolve many BWT rows to text positions, driving all LF-walks in lockstep instead of
    /// one at a time.
    ///
    /// Each row's LF-walk is independent of every other row's — `resolve_sa` in a loop over
    /// `rows` runs them fully serially, exposing the raw pointer-chasing latency of each
    /// `sa_samples.get`/`lf_step` miss with nothing to overlap it with. Interleaving the walks
    /// (SoA: one `cur`/`steps` slot per still-active row) means the CPU's out-of-order window
    /// has many independent memory accesses in flight per round instead of one, and a software
    /// prefetch issued for every active row's next access, one full round ahead of when it's
    /// read, gives the miss extra time to resolve before it's needed. `locate`/`locate_positions`
    /// use this instead of the old `flat_map(lo..hi).map(resolve_sa)`; results are appended to
    /// `out` in `rows` order (same order the old per-row map produced).
    pub(crate) fn resolve_sa_batch(&self, rows: &[u32], out: &mut Vec<u32>) {
        // Written by index as lanes retire (out of order), so pre-size rather than push.
        out.clear();
        out.resize(rows.len(), 0);
        let mut cur: Vec<u32> = rows.to_vec();
        let mut steps: Vec<u32> = vec![0; rows.len()];
        // Indices into `cur`/`steps` still walking; shrinks as rows hit a sampled row.
        let mut active: Vec<u32> = (0..rows.len() as u32).collect();
        let mut next_active: Vec<u32> = Vec::with_capacity(active.len());

        while !active.is_empty() {
            // Phase A: issue prefetches for every active lane's current position before
            // Phase B reads any of them, so each lane's miss overlaps with the others'
            // prefetch issue + Phase B's arithmetic instead of stalling immediately.
            for &lane in &active {
                let pos = cur[lane as usize];
                self.sa_samples.prefetch(pos);
                self.occ.prefetch_block(pos);
            }

            next_active.clear();
            for &lane in &active {
                let idx = lane as usize;
                let pos = cur[idx];
                if let Some(sa_val) = self.sa_samples.get(pos) {
                    out[idx] = sa_val + steps[idx];
                } else {
                    cur[idx] = self.lf_mapping(pos);
                    steps[idx] += 1;
                    next_active.push(lane);
                }
            }
            std::mem::swap(&mut active, &mut next_active);
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
        use crate::gpu::context_cache;
        use crate::gpu::locate::locate_batch_gpu;
        let ctx = context_cache::get_or_init()?;
        let encoded: Vec<&[u8]> = queries.iter().map(|q| q.as_ref()).collect();
        let by_idx = locate_batch_gpu(&ctx, self, &encoded).await?;
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

/// Merge SA intervals in-place, combining overlapping or adjacent ones.
///
/// Only called when `ivs.len() > 1` (IUPAC multi-symbol paths).
fn merge_intervals_inplace(ivs: &mut Vec<(u32, u32)>) {
    ivs.sort_unstable_by_key(|&(lo, _)| lo);
    let mut write = 0usize;
    let (mut cur_lo, mut cur_hi) = ivs[0];
    for read in 1..ivs.len() {
        let (lo, hi) = ivs[read];
        if lo <= cur_hi {
            cur_hi = cur_hi.max(hi);
        } else {
            ivs[write] = (cur_lo, cur_hi);
            write += 1;
            cur_lo = lo;
            cur_hi = hi;
        }
    }
    ivs[write] = (cur_lo, cur_hi);
    ivs.truncate(write + 1);
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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

    #[test]
    fn test_resolve_sa_batch_matches_scalar() {
        // sa_sample_rate default (not 1) so resolve_sa's LF-walk actually takes >0 steps.
        let seq = DnaSequence::from_str("ACGTACGTACGTACGTACGTACGTACGT").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 4,
            use_gpu: false,
            ..Default::default()
        };
        let idx = FmIndex::build_cpu(&[seq], &config).unwrap();

        let rows: Vec<u32> = (0..idx.text_len).collect();
        let scalar: Vec<u32> = rows.iter().map(|&i| idx.resolve_sa(i)).collect();

        let mut batch = Vec::new();
        idx.resolve_sa_batch(&rows, &mut batch);

        assert_eq!(batch, scalar);
    }
}
