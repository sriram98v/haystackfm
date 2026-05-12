use crate::alphabet::{self, DnaSequence};
use crate::error::FmIndexError;
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::{FmIndex, FmIndexConfig};

/// A bidirectional FM-index: pairs a forward FM-index (built on text T) with a
/// reverse FM-index (built on the byte-reversal of T), enabling O(1) extension
/// of matched intervals in both the left and right directions.
///
/// # Construction
///
/// ```rust,ignore
/// let seqs = vec![DnaSequence::from_str("ACGTACGT").unwrap()];
/// let config = FmIndexConfig::default();
/// let bidir = BidirFmIndex::build_cpu(&seqs, &config)?;
/// ```
///
/// # Use
///
/// ```rust,ignore
/// let iv = bidir.full_interval();
/// let iv = bidir.extend_right(iv, alphabet::C)?;  // match "C"
/// let iv = bidir.extend_right(iv, alphabet::G)?;  // match "CG"
/// let iv = bidir.extend_left(iv, alphabet::A)?;   // match "ACG"
/// println!("occurrences: {}", iv.size());
/// let positions = bidir.locate(iv);
/// ```
#[derive(Debug, Clone)]
pub struct BidirFmIndex {
    /// FM-index of the concatenated text T.
    pub(crate) fwd: FmIndex,
    /// FM-index of the byte-reversal of T (T^R).
    pub(crate) rev: FmIndex,
}

impl BidirFmIndex {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Build a bidirectional FM-index from DNA sequences using the CPU.
    pub fn build_cpu(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }

        let (text, _) = alphabet::concatenate_sequences(sequences)?;

        // Forward index.
        let fwd = FmIndex::build_cpu(sequences, config)?;

        // Reverse index: built on the byte-reversal of the same concatenated text.
        let rev_seq = reverse_as_sequence(&text)?;
        let rev = FmIndex::build_cpu(
            &[rev_seq],
            &FmIndexConfig {
                sa_sample_rate: config.sa_sample_rate,
                use_gpu: false,
            },
        )?;

        Ok(Self { fwd, rev })
    }

    /// Build a bidirectional FM-index using GPU acceleration (async).
    #[cfg(feature = "gpu")]
    pub async fn build(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }

        let (text, _) = alphabet::concatenate_sequences(sequences)?;

        // Build forward and reverse indices concurrently via GPU.
        let rev_seq = reverse_as_sequence(&text)?;

        let rev_config = FmIndexConfig {
            sa_sample_rate: config.sa_sample_rate,
            use_gpu: true,
        };

        // Build sequentially: both paths share the GPU device pool and
        // concurrent init would contend for it.
        let fwd = FmIndex::build(sequences, config).await?;
        let rev = FmIndex::build(&[rev_seq], &rev_config).await?;

        Ok(Self { fwd, rev })
    }

    // ── Interval operations ───────────────────────────────────────────────────

    /// The "whole text" interval, corresponding to the empty pattern.
    ///
    /// All positions are valid matches; this is the starting point for all
    /// bidirectional searches.
    pub fn full_interval(&self) -> BidirInterval {
        BidirInterval::full(self.fwd.text_len)
    }

    /// Extend a bidirectional interval to the right by character `c` (P → Pc).
    ///
    /// Uses the reverse FM-index internally (right extension = left extension of P^R).
    ///
    /// Returns `None` when Pc has no occurrences in the text.
    pub fn extend_right(&self, iv: BidirInterval, c: u8) -> Option<BidirInterval> {
        iv.extend_right(c, &self.rev)
    }

    /// Extend a bidirectional interval to the left by character `c` (P → cP).
    ///
    /// Uses the forward FM-index internally (standard backward-search step).
    ///
    /// Returns `None` when cP has no occurrences in the text.
    pub fn extend_left(&self, iv: BidirInterval, c: u8) -> Option<BidirInterval> {
        iv.extend_left(c, &self.fwd)
    }

    // ── Query helpers ─────────────────────────────────────────────────────────

    /// Count occurrences of the pattern represented by `iv`.
    pub fn count_interval(&self, iv: &BidirInterval) -> u32 {
        iv.size()
    }

    /// Locate all occurrences for the pattern represented by `iv`.
    ///
    /// Returns `(sequence_id, position_within_sequence)` tuples.
    /// Uses the forward SA samples; time is O(occ × sample_rate).
    pub fn locate_interval(&self, iv: &BidirInterval) -> Vec<(String, u32)> {
        (iv.fwd_lo..iv.fwd_hi)
            .map(|i| {
                let text_pos = self.fwd.resolve_sa(i);
                let (seq_idx, pos_in_seq) = self
                    .fwd
                    .map_position(text_pos)
                    .expect("resolved SA position must be within text bounds");
                (self.fwd.seq_headers[seq_idx as usize].clone(), pos_in_seq)
            })
            .collect()
    }

    /// Total length of the indexed text (including sentinels).
    pub fn text_len(&self) -> u32 {
        self.fwd.text_len
    }

    /// Number of sequences indexed.
    pub fn num_sequences(&self) -> u32 {
        self.fwd.num_sequences
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    /// Serialize both indices to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FmIndexError> {
        let fwd_bytes = self.fwd.to_bytes()?;
        let rev_bytes = self.rev.to_bytes()?;
        // Format: [4-byte fwd_len (LE)][fwd_bytes][rev_bytes]
        let mut out = Vec::with_capacity(4 + fwd_bytes.len() + rev_bytes.len());
        out.extend_from_slice(&(fwd_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&fwd_bytes);
        out.extend_from_slice(&rev_bytes);
        Ok(out)
    }

    /// Deserialize from bytes produced by `to_bytes()`.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        if data.len() < 4 {
            return Err(FmIndexError::DeserializeError(
                "truncated bidirectional index".into(),
            ));
        }
        let fwd_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + fwd_len {
            return Err(FmIndexError::DeserializeError(
                "truncated forward index".into(),
            ));
        }
        let fwd = FmIndex::from_bytes(&data[4..4 + fwd_len])?;
        let rev = FmIndex::from_bytes(&data[4 + fwd_len..])?;
        Ok(Self { fwd, rev })
    }

    // ── GPU MEM/SMEM finding ──────────────────────────────────────────────────

    /// Find all Super-Maximal Exact Matches (SMEMs) for a batch of queries on the GPU.
    ///
    /// Each query is processed by a separate GPU thread. Returns one `Vec<Mem>` per query
    /// with `positions` empty (locate=false). Requires the `gpu` feature.
    #[cfg(feature = "gpu")]
    pub async fn find_smems_gpu(
        &self,
        queries: &[crate::alphabet::DnaSequence],
        min_len: usize,
    ) -> Result<Vec<Vec<crate::fm_index::smem::Mem>>, FmIndexError> {
        use crate::gpu::mem_find::find_smems_batch_gpu;
        let encoded: Vec<&[u8]> = queries.iter().map(|q| q.as_slice()).collect();
        let raw = find_smems_batch_gpu(self, &encoded, min_len).await?;
        Ok(raw.into_iter().map(raw_to_mems).collect())
    }

    /// Find all Maximal Exact Matches (MEMs, including non-super-maximal) for a batch
    /// of queries on the GPU.
    ///
    /// Each query is processed by a separate GPU thread. Returns one `Vec<Mem>` per query
    /// with `positions` empty (locate=false). Requires the `gpu` feature.
    #[cfg(feature = "gpu")]
    pub async fn find_mems_gpu(
        &self,
        queries: &[crate::alphabet::DnaSequence],
        min_len: usize,
    ) -> Result<Vec<Vec<crate::fm_index::smem::Mem>>, FmIndexError> {
        use crate::gpu::mem_find::find_all_mems_batch_gpu;
        let encoded: Vec<&[u8]> = queries.iter().map(|q| q.as_slice()).collect();
        let raw = find_all_mems_batch_gpu(self, &encoded, min_len).await?;
        Ok(raw.into_iter().map(raw_to_mems).collect())
    }
}

#[cfg(feature = "gpu")]
fn raw_to_mems(hits: Vec<(u32, u32, u32)>) -> Vec<crate::fm_index::smem::Mem> {
    hits.into_iter()
        .map(|(start, end, count)| crate::fm_index::smem::Mem {
            query_start: start as usize,
            query_end: end as usize,
            match_count: count,
            positions: Vec::new(),
        })
        .collect()
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Reverse a concatenated encoded text (bytes 0–4) and wrap it as a single DnaSequence.
///
/// Sentinels (0) in the middle of the text become interior characters of the reversed
/// sequence; the FM-index treats them as the lexicographically smallest character, so
/// the reverse index remains valid.
fn reverse_as_sequence(text: &[u8]) -> Result<DnaSequence, FmIndexError> {
    // Strip the trailing sentinel before reversing so we don't double-sentinel.
    let stripped = if text.last() == Some(&crate::alphabet::SENTINEL) {
        &text[..text.len() - 1]
    } else {
        text
    };
    let rev: Vec<u8> = stripped.iter().rev().cloned().collect();
    Ok(DnaSequence::from_encoded(rev))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::{encode_char, DnaSequence};

    fn encode(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    fn bidir(s: &str) -> BidirFmIndex {
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
        };
        BidirFmIndex::build_cpu(&[DnaSequence::from_str(s).unwrap()], &config).unwrap()
    }

    #[test]
    fn full_interval_covers_all() {
        let idx = bidir("ACGTACGT");
        let iv = idx.full_interval();
        assert_eq!(iv.size(), idx.text_len());
    }

    #[test]
    fn extend_right_count_matches_unidirectional() {
        let idx = bidir("ACGTACGT");
        let pattern = encode("ACGT");

        let mut iv = idx.full_interval();
        for &c in &pattern {
            iv = idx
                .extend_right(iv, c)
                .unwrap_or_else(|| panic!("extend_right failed for char {}", c));
        }
        assert_eq!(iv.size(), idx.fwd.count(&pattern));
    }

    #[test]
    fn extend_left_count_matches_unidirectional() {
        let idx = bidir("ACGTACGT");
        let pattern = encode("ACGT");

        // Build "ACGT" via extend_left: prepend T, G, C, A (right-to-left)
        let mut iv = idx.full_interval();
        for &c in pattern.iter().rev() {
            iv = idx
                .extend_left(iv, c)
                .unwrap_or_else(|| panic!("extend_left failed for char {}", c));
        }
        assert_eq!(iv.size(), idx.fwd.count(&pattern));

        // Extending left by A from the "ACGT" interval → "AACGT", absent from "ACGTACGT"
        let a = encode_char('A').unwrap();
        assert!(
            idx.extend_left(iv, a).is_none(),
            "AACGT should not appear in ACGTACGT"
        );
    }

    #[test]
    fn extend_right_and_left_combined() {
        // Text "TTACGTAA": find "ACGT" then extend in both directions.
        let idx = bidir("TTACGTAA");
        let acgt = encode("ACGT");

        let mut iv = idx.full_interval();
        for &c in &acgt {
            iv = idx
                .extend_right(iv, c)
                .unwrap_or_else(|| panic!("extend_right failed"));
        }
        assert_eq!(iv.size(), 1, "ACGT should appear once");

        // Extend left by T → "TACGT"
        let t = encode_char('T').unwrap();
        let iv2 = idx.extend_left(iv, t).expect("TACGT should be in TTACGTAA");
        assert_eq!(iv2.size(), 1);

        // Extend right by A → "TACGTA"
        let a = encode_char('A').unwrap();
        let iv3 = idx
            .extend_right(iv2, a)
            .expect("TACGTA should be in TTACGTAA");
        assert_eq!(iv3.size(), 1);
    }

    #[test]
    fn locate_interval() {
        let idx = bidir("ACGTACGT");
        let pattern = encode("ACGT");

        let mut iv = idx.full_interval();
        for &c in &pattern {
            iv = idx.extend_right(iv, c).unwrap();
        }
        let mut positions = idx.locate_interval(&iv);
        positions.sort();
        assert_eq!(
            positions,
            vec![("seq_0".to_string(), 0), ("seq_0".to_string(), 4)]
        );
    }

    #[test]
    fn serialization_roundtrip() {
        let idx = bidir("ACGTACGT");
        let bytes = idx.to_bytes().unwrap();
        let restored = BidirFmIndex::from_bytes(&bytes).unwrap();

        let pattern = encode("ACGT");
        let mut iv1 = idx.full_interval();
        let mut iv2 = restored.full_interval();
        for &c in &pattern {
            iv1 = idx.extend_right(iv1, c).unwrap();
            iv2 = restored.extend_right(iv2, c).unwrap();
        }
        assert_eq!(iv1.size(), iv2.size());
    }
}
