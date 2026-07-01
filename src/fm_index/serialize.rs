//! Serialization and deserialization of the FM-index to/from bytes via `bincode`.

use super::FmIndex;
use crate::alphabet::alphabet_fns_from_tag;
use crate::bwt::Bwt;
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::fm_index::lookup::LookupTable;
use crate::occ::OccTable;
use crate::suffix_array::SampledSuffixArray;

impl FmIndex {
    /// Serialize the FM-index to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FmIndexError> {
        let serializable = SerializableFmIndex {
            bwt: &self.bwt,
            c_array: &self.c_array,
            occ: &self.occ,
            sa_samples: &self.sa_samples,
            text_len: self.text_len,
            num_sequences: self.num_sequences,
            seq_boundaries: &self.seq_boundaries,
            seq_headers: &self.seq_headers,
            lookup: self.lookup.as_ref(),
            alphabet_tag: self.alphabet_fns.tag,
        };
        bincode::serialize(&serializable).map_err(|e| FmIndexError::SerializeError(e.to_string()))
    }

    /// Deserialize an FM-index from bytes.
    ///
    /// The alphabet tag stored in the bytes is used to reconstruct the matching
    /// semantics. Returns [`FmIndexError::DeserializeError`] for unknown tags.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        let deserialized: OwnedSerializableFmIndex = bincode::deserialize(data)
            .map_err(|e| FmIndexError::DeserializeError(e.to_string()))?;
        let alphabet_fns = alphabet_fns_from_tag(deserialized.alphabet_tag).ok_or_else(|| {
            FmIndexError::DeserializeError(format!(
                "unknown alphabet tag {} in serialized FM-index",
                deserialized.alphabet_tag
            ))
        })?;
        Ok(Self {
            bwt: deserialized.bwt,
            c_array: deserialized.c_array,
            occ: deserialized.occ,
            sa_samples: deserialized.sa_samples,
            text_len: deserialized.text_len,
            num_sequences: deserialized.num_sequences,
            seq_boundaries: deserialized.seq_boundaries,
            seq_headers: deserialized.seq_headers,
            lookup: deserialized.lookup,
            alphabet_fns,
        })
    }
}

#[derive(serde::Serialize)]
struct SerializableFmIndex<'a> {
    bwt: &'a Bwt,
    c_array: &'a CArray,
    occ: &'a OccTable,
    sa_samples: &'a SampledSuffixArray,
    text_len: u32,
    num_sequences: u32,
    seq_boundaries: &'a [u32],
    seq_headers: &'a [String],
    lookup: Option<&'a LookupTable>,
    /// Tag identifying the alphabet used for matching (0 = IupacDna, 1 = ExactDna).
    alphabet_tag: u8,
}

#[derive(serde::Deserialize)]
struct OwnedSerializableFmIndex {
    bwt: Bwt,
    c_array: CArray,
    occ: OccTable,
    sa_samples: SampledSuffixArray,
    text_len: u32,
    num_sequences: u32,
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    lookup: Option<LookupTable>,
    alphabet_tag: u8,
}

impl Default for OwnedSerializableFmIndex {
    fn default() -> Self {
        unreachable!("used only via bincode deserialization")
    }
}

#[cfg(test)]
mod tests {
    use crate::alphabet::*;
    use crate::fm_index::{FmIndex, FmIndexConfig};

    fn encode_pattern(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    #[test]
    fn test_serialize_exact_dna_roundtrip() {
        use crate::alphabet::ExactDna;
        let seq = DnaSequence::from_str("ACGTNACGT").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            ..Default::default()
        };
        let original = FmIndex::build_cpu_with::<ExactDna>(&[seq], &config).unwrap();
        let bytes = original.to_bytes().unwrap();
        let restored = FmIndex::from_bytes(&bytes).unwrap();

        // ExactDna: N should match 0.
        let n_enc = encode_pattern("N");
        assert_eq!(original.count(&n_enc), 0);
        assert_eq!(restored.count(&n_enc), 0);

        // ACGT patterns should agree.
        for pattern in &["ACG", "CGT", "ACGT"] {
            let p = encode_pattern(pattern);
            assert_eq!(
                original.count(&p),
                restored.count(&p),
                "ExactDna roundtrip count mismatch for '{}'",
                pattern
            );
        }
    }

    #[test]
    fn test_serialize_bad_tag_rejected() {
        // Manually corrupt the alphabet_tag to an unknown value and verify error.
        let seq = DnaSequence::from_str("ACGT").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 4,
            use_gpu: false,
            ..Default::default()
        };
        let original = FmIndex::build_cpu(&[seq], &config).unwrap();
        let mut bytes = original.to_bytes().unwrap();
        // Overwrite the last byte (alphabet_tag, stored last by bincode) with 0xFF.
        if let Some(last) = bytes.last_mut() {
            *last = 0xFF;
        }
        assert!(
            FmIndex::from_bytes(&bytes).is_err(),
            "should reject unknown alphabet tag"
        );
    }

    #[test]
    fn test_serialize_roundtrip() {
        let seq = DnaSequence::from_str("ACGTACGTACGT").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 4,
            use_gpu: false,
            ..Default::default()
        };
        let original = FmIndex::build_cpu(&[seq], &config).unwrap();

        let bytes = original.to_bytes().unwrap();
        let restored = FmIndex::from_bytes(&bytes).unwrap();

        // Verify the restored index produces the same results
        for pattern in &["A", "AC", "ACGT", "GT", "ACGTACGT"] {
            let p = encode_pattern(pattern);
            assert_eq!(
                original.count(&p),
                restored.count(&p),
                "count mismatch for '{}'",
                pattern
            );

            let mut orig_locs = original.locate(&p);
            let mut rest_locs = restored.locate(&p);
            orig_locs.sort_by_key(|(_, pos)| *pos);
            rest_locs.sort_by_key(|(_, pos)| *pos);
            assert_eq!(orig_locs, rest_locs, "locate mismatch for '{}'", pattern);
        }
    }

    #[test]
    fn test_serialize_multi_sequence() {
        let sequences = vec![
            DnaSequence::from_str("ACGT").unwrap(),
            DnaSequence::from_str("TGCA").unwrap(),
        ];
        let config = FmIndexConfig {
            sa_sample_rate: 2,
            use_gpu: false,
            ..Default::default()
        };
        let original = FmIndex::build_cpu(&sequences, &config).unwrap();

        let bytes = original.to_bytes().unwrap();
        let restored = FmIndex::from_bytes(&bytes).unwrap();

        assert_eq!(original.text_len(), restored.text_len());
        assert_eq!(original.num_sequences(), restored.num_sequences());
    }
}
