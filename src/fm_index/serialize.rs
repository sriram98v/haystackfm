//! Serialization and deserialization of the FM-index to/from bytes via `bincode`.
//!
//! ## Format
//!
//! Version 1 (current): `FORMAT_MAGIC` followed by the bincode encoding of
//! [`SerializableFmIndex`]. The magic identifies the format because legacy (unversioned)
//! blobs begin with the bincode encoding of `CArray::data[0]`, which is always `0u32`
//! (`C[0]` counts symbols below the sentinel), so it can never collide with the magic.
//!
//! Legacy blobs (no magic) are still accepted: they lack the LCP array, so an index loaded
//! from one reports `has_lcp() == false`.

use super::FmIndex;
use crate::alphabet::alphabet_fns_from_tag;
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::fm_index::lookup::LookupTable;
use crate::fm_index::seq_id::HeaderIndex;
use crate::lcp::LcpArray;
use crate::occ::OccTable;
use crate::suffix_array::SampledSuffixArray;

/// Leading bytes of a version-1 serialized index: `"HFM"` plus the format version.
pub const FORMAT_MAGIC: [u8; 4] = *b"HFM\x01";

impl FmIndex {
    /// Serialize the FM-index to bytes (format version 1, see the module docs).
    pub fn to_bytes(&self) -> Result<Vec<u8>, FmIndexError> {
        let serializable = SerializableFmIndex {
            c_array: &self.c_array,
            occ: &self.occ,
            sa_samples: &self.sa_samples,
            text: &self.text,
            text_len: self.text_len,
            num_sequences: self.num_sequences,
            seq_boundaries: &self.seq_boundaries,
            seq_headers: &self.seq_headers,
            lookup: self.lookup.as_ref(),
            lcp: self.lcp.as_ref(),
            alphabet_tag: self.alphabet_fns.tag,
        };
        let body = bincode::serialize(&serializable)
            .map_err(|e| FmIndexError::SerializeError(e.to_string()))?;
        let mut out = Vec::with_capacity(FORMAT_MAGIC.len() + body.len());
        out.extend_from_slice(&FORMAT_MAGIC);
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Deserialize an FM-index from bytes, accepting both the current versioned format and
    /// legacy unversioned blobs (which load without an LCP array).
    ///
    /// The alphabet tag stored in the bytes is used to reconstruct the matching
    /// semantics. Returns [`FmIndexError::DeserializeError`] for unknown tags.
    ///
    /// The header -> [`SeqId`](crate::SeqId) map is a pure function of the stored headers,
    /// so it is rebuilt here rather than serialized; likewise the occ table's select hints.
    /// Sequence ids are the stored header order and therefore identical to those of the
    /// index that was serialized.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        let deserialized: OwnedSerializableFmIndex = match data.strip_prefix(&FORMAT_MAGIC) {
            Some(body) => bincode::deserialize(body)
                .map_err(|e| FmIndexError::DeserializeError(e.to_string()))?,
            None => bincode::deserialize::<LegacyOwnedSerializableFmIndex>(data)
                .map_err(|e| FmIndexError::DeserializeError(e.to_string()))?
                .into(),
        };
        let alphabet_fns = alphabet_fns_from_tag(deserialized.alphabet_tag).ok_or_else(|| {
            FmIndexError::DeserializeError(format!(
                "unknown alphabet tag {} in serialized FM-index",
                deserialized.alphabet_tag
            ))
        })?;
        let header_index = HeaderIndex::build(&deserialized.seq_headers)?;
        let mut occ = deserialized.occ;
        occ.build_select_hints();
        Ok(Self {
            header_index,
            c_array: deserialized.c_array,
            occ,
            sa_samples: deserialized.sa_samples,
            text: deserialized.text,
            text_len: deserialized.text_len,
            num_sequences: deserialized.num_sequences,
            seq_boundaries: deserialized.seq_boundaries,
            seq_headers: deserialized.seq_headers,
            lookup: deserialized.lookup,
            alphabet_fns,
            lcp: deserialized.lcp,
        })
    }
}

#[derive(serde::Serialize)]
struct SerializableFmIndex<'a> {
    c_array: &'a CArray,
    occ: &'a OccTable,
    sa_samples: &'a SampledSuffixArray,
    /// Concatenated text in alphabet codes; backs `FmIndex::sequence`.
    text: &'a [u8],
    text_len: u32,
    num_sequences: u32,
    seq_boundaries: &'a [u32],
    seq_headers: &'a [String],
    lookup: Option<&'a LookupTable>,
    /// Present when built with `FmIndexConfig::build_lcp` (format version 1 onwards).
    lcp: Option<&'a LcpArray>,
    /// Tag identifying the alphabet used for matching (0 = IupacDna, 1 = ExactDna).
    /// Kept last: `test_serialize_bad_tag_rejected` corrupts the final byte to reach it.
    alphabet_tag: u8,
}

#[derive(serde::Deserialize)]
struct OwnedSerializableFmIndex {
    c_array: CArray,
    occ: OccTable,
    sa_samples: SampledSuffixArray,
    text: Vec<u8>,
    text_len: u32,
    num_sequences: u32,
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    lookup: Option<LookupTable>,
    lcp: Option<LcpArray>,
    alphabet_tag: u8,
}

/// Field layout of blobs written before `FORMAT_MAGIC` existed (no LCP array).
#[derive(serde::Deserialize)]
struct LegacyOwnedSerializableFmIndex {
    c_array: CArray,
    occ: OccTable,
    sa_samples: SampledSuffixArray,
    text: Vec<u8>,
    text_len: u32,
    num_sequences: u32,
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    lookup: Option<LookupTable>,
    alphabet_tag: u8,
}

impl From<LegacyOwnedSerializableFmIndex> for OwnedSerializableFmIndex {
    fn from(legacy: LegacyOwnedSerializableFmIndex) -> Self {
        Self {
            c_array: legacy.c_array,
            occ: legacy.occ,
            sa_samples: legacy.sa_samples,
            text: legacy.text,
            text_len: legacy.text_len,
            num_sequences: legacy.num_sequences,
            seq_boundaries: legacy.seq_boundaries,
            seq_headers: legacy.seq_headers,
            lookup: legacy.lookup,
            lcp: None,
            alphabet_tag: legacy.alphabet_tag,
        }
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
    fn test_versioned_bytes_start_with_magic_and_keep_lcp() {
        let seq = DnaSequence::from_str("ACGTACGTTTGA").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 2,
            use_gpu: false,
            build_lcp: true,
            ..Default::default()
        };
        let original = FmIndex::build_cpu(&[seq], &config).unwrap();
        assert!(original.has_lcp());
        let bytes = original.to_bytes().unwrap();
        assert_eq!(&bytes[..4], &super::FORMAT_MAGIC);
        let restored = FmIndex::from_bytes(&bytes).unwrap();
        assert!(restored.has_lcp());
        assert_eq!(restored.lcp, original.lcp);
        assert_eq!(restored.occ, original.occ);
    }

    #[test]
    fn test_legacy_unversioned_bytes_load_without_lcp() {
        // A legacy blob is the bincode body without magic and without the `lcp` field.
        let seq = DnaSequence::from_str("ACGTACGTTTGA").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 2,
            use_gpu: false,
            build_lcp: false,
            ..Default::default()
        };
        let original = FmIndex::build_cpu(&[seq], &config).unwrap();
        let bytes = original.to_bytes().unwrap();
        let body = &bytes[4..];
        // Strip the `Option<LcpArray>` = None discriminator (one 0 byte) before the tag.
        let mut legacy = body[..body.len() - 2].to_vec();
        legacy.push(body[body.len() - 1]);
        assert_ne!(&legacy[..4], &super::FORMAT_MAGIC);
        let restored = FmIndex::from_bytes(&legacy).unwrap();
        assert!(!restored.has_lcp());
        let p = encode_pattern("ACGT");
        assert_eq!(restored.count(&p), original.count(&p));
        assert_eq!(restored.occ, original.occ);
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
