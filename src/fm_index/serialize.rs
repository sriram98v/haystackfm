//! Serialization and deserialization of the FM-index to/from bytes via `bincode`.
//!
//! ## Format
//!
//! Version 2 (current): [`FORMAT_MAGIC`] (`"HFM\x02"`) followed by the bincode encoding of
//! [`SerializableFmIndex`], in which the large numeric vectors (occ checkpoints and block
//! records, SA samples, LCP arrays, the text) are length-prefixed little-endian byte blobs
//! (`crate::serde_raw`) that load with one copy each instead of one deserializer call per
//! element.
//!
//! Version 1 ([`FORMAT_MAGIC_V1`], `"HFM\x01"`): the same fields with those vectors as plain
//! element sequences. Still read (`OwnedSerializableFmIndexV1`), never written.
//!
//! Legacy (no magic): version-1 field encodings without the LCP array. Distinguishable
//! because such blobs begin with the bincode encoding of `CArray::data[0]`, which is always
//! `0u32` (`C[0]` counts symbols below the sentinel) and so never collides with a magic. An
//! index loaded from one reports `has_lcp() == false`.
//!
//! Any other `"HFM"` version byte is rejected with a clear error rather than misparsed.

use super::FmIndex;
use crate::alphabet::alphabet_fns_from_tag;
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::fm_index::lookup::LookupTable;
use crate::fm_index::seq_id::HeaderIndex;
use crate::lcp::{LcpArray, LcpArrayV1};
use crate::occ::{OccTable, OccTableV1};
use crate::suffix_array::{SampledSuffixArray, SampledSuffixArrayV1};

/// Leading bytes of a serialized index: `"HFM"` plus the format version (currently 2).
pub const FORMAT_MAGIC: [u8; 4] = *b"HFM\x02";
/// Magic of format version 1, still accepted by [`FmIndex::from_bytes`].
pub const FORMAT_MAGIC_V1: [u8; 4] = *b"HFM\x01";

impl FmIndex {
    /// Serialize the FM-index to bytes (format version 2, see the module docs).
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

    /// Deserialize an FM-index from bytes: the current format, format version 1, or a
    /// legacy unversioned blob (which loads without an LCP array). See the module docs.
    ///
    /// The alphabet tag stored in the bytes is used to reconstruct the matching
    /// semantics. Returns [`FmIndexError::DeserializeError`] for unknown tags and for
    /// format versions this build does not read.
    ///
    /// The header -> [`SeqId`](crate::SeqId) map is a pure function of the stored headers,
    /// so it is rebuilt here rather than serialized; likewise the occ table's select hints.
    /// Sequence ids are the stored header order and therefore identical to those of the
    /// index that was serialized.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        if data.len() < FORMAT_MAGIC.len() {
            return Err(FmIndexError::DeserializeError(format!(
                "serialized FM-index is {} bytes, shorter than its format marker",
                data.len()
            )));
        }
        let de = |e: bincode::Error| FmIndexError::DeserializeError(e.to_string());
        let deserialized: OwnedSerializableFmIndex =
            if let Some(body) = data.strip_prefix(&FORMAT_MAGIC) {
                bincode::deserialize(body).map_err(de)?
            } else if let Some(body) = data.strip_prefix(&FORMAT_MAGIC_V1) {
                bincode::deserialize::<OwnedSerializableFmIndexV1>(body)
                    .map_err(de)?
                    .into()
            } else if let [b'H', b'F', b'M', version, ..] = data {
                return Err(FmIndexError::DeserializeError(format!(
                    "unsupported serialized FM-index format version {version} \
                     (this build reads versions 1 and 2)"
                )));
            } else {
                bincode::deserialize::<LegacyOwnedSerializableFmIndex>(data)
                    .map_err(de)?
                    .into()
            };
        let alphabet_fns = alphabet_fns_from_tag(deserialized.alphabet_tag).ok_or_else(|| {
            FmIndexError::DeserializeError(format!(
                "unknown alphabet tag {} in serialized FM-index",
                deserialized.alphabet_tag
            ))
        })?;
        let header_index = HeaderIndex::build(&deserialized.seq_headers)?;
        let mut occ = deserialized.occ;
        occ.rebuild_derived();
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

/// Format version 2 write layout. Field order is the wire order.
#[derive(serde::Serialize)]
struct SerializableFmIndex<'a> {
    c_array: &'a CArray,
    occ: &'a OccTable,
    sa_samples: &'a SampledSuffixArray,
    /// Concatenated text in alphabet codes; backs `FmIndex::sequence`.
    #[serde(with = "crate::serde_raw::bytes")]
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

/// Format version 2 read layout; mirrors [`SerializableFmIndex`].
#[derive(serde::Deserialize)]
struct OwnedSerializableFmIndex {
    c_array: CArray,
    occ: OccTable,
    sa_samples: SampledSuffixArray,
    #[serde(with = "crate::serde_raw::bytes")]
    text: Vec<u8>,
    text_len: u32,
    num_sequences: u32,
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    lookup: Option<LookupTable>,
    lcp: Option<LcpArray>,
    /// Kept last (see [`SerializableFmIndex`]).
    alphabet_tag: u8,
}

/// Format version 1 read layout: the numeric vectors inside `occ`, `sa_samples` and `lcp`
/// are element sequences (`*V1` shadows); `text` is a `Vec<u8>` sequence, which is
/// wire-identical to a byte blob.
#[derive(serde::Deserialize)]
struct OwnedSerializableFmIndexV1 {
    c_array: CArray,
    occ: OccTableV1,
    sa_samples: SampledSuffixArrayV1,
    text: Vec<u8>,
    text_len: u32,
    num_sequences: u32,
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    lookup: Option<LookupTable>,
    lcp: Option<LcpArrayV1>,
    /// Kept last (see [`SerializableFmIndex`]).
    alphabet_tag: u8,
}

/// Field layout of blobs written before any magic existed: version 1 minus the LCP array.
#[derive(serde::Deserialize)]
struct LegacyOwnedSerializableFmIndex {
    c_array: CArray,
    occ: OccTableV1,
    sa_samples: SampledSuffixArrayV1,
    text: Vec<u8>,
    text_len: u32,
    num_sequences: u32,
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    lookup: Option<LookupTable>,
    /// Kept last (see [`SerializableFmIndex`]).
    alphabet_tag: u8,
}

impl From<OwnedSerializableFmIndexV1> for OwnedSerializableFmIndex {
    fn from(v1: OwnedSerializableFmIndexV1) -> Self {
        Self {
            c_array: v1.c_array,
            occ: v1.occ.into(),
            sa_samples: v1.sa_samples.into(),
            text: v1.text,
            text_len: v1.text_len,
            num_sequences: v1.num_sequences,
            seq_boundaries: v1.seq_boundaries,
            seq_headers: v1.seq_headers,
            lookup: v1.lookup,
            lcp: v1.lcp.map(Into::into),
            alphabet_tag: v1.alphabet_tag,
        }
    }
}

impl From<LegacyOwnedSerializableFmIndex> for OwnedSerializableFmIndex {
    fn from(legacy: LegacyOwnedSerializableFmIndex) -> Self {
        Self {
            c_array: legacy.c_array,
            occ: legacy.occ.into(),
            sa_samples: legacy.sa_samples.into(),
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

    // ── Frozen-format fixtures ────────────────────────────────────────────────
    //
    // `tests/fixtures/*.hfm` were written once at `9b32c9f` (format version 1) by a
    // throwaway generator from exactly the input and configs below; the legacy blob is the
    // version-1 Bitplane blob minus its magic and the `Option<LcpArray>` None byte. CPU
    // construction is deterministic, so a fresh build must serialize to the same version-2
    // bytes as the loaded fixture — i.e. every stored field survived the version upgrade.

    fn fixture_input() -> Vec<DnaSequence> {
        vec![
            DnaSequence::from_str_with_header("ACGTNACGTRYSWKMBDHVACGTACGTTTGCA", "chr1").unwrap(),
            DnaSequence::from_str_with_header("TTGACGTACGTNNACGGAAC", "plasmid").unwrap(),
            DnaSequence::from_str_with_header("GATTACA", "tiny").unwrap(),
        ]
    }

    fn onehot_fixture_config() -> FmIndexConfig {
        FmIndexConfig {
            sa_sample_rate: 2,
            use_gpu: false,
            occ_encoding: crate::occ::OccEncoding::OneHot,
            build_lcp: true,
            lookup_depth: 3,
            ..Default::default()
        }
    }

    fn bitplane_fixture_config() -> FmIndexConfig {
        FmIndexConfig {
            sa_sample_rate: 4,
            use_gpu: false,
            occ_encoding: crate::occ::OccEncoding::Bitplane,
            build_lcp: false,
            lookup_depth: 0,
            ..Default::default()
        }
    }

    fn assert_matches_fresh_build(loaded: &FmIndex, config: &FmIndexConfig) {
        let fresh = FmIndex::build_cpu(&fixture_input(), config).unwrap();
        assert_eq!(loaded.to_bytes().unwrap(), fresh.to_bytes().unwrap());
        assert_eq!(loaded.has_lcp(), config.build_lcp);
        assert_eq!(loaded.seq_headers(), fresh.seq_headers());
        for pattern in ["ACGT", "GATTACA", "NN", "CGTACGT"] {
            let p = encode_pattern(pattern);
            assert_eq!(loaded.count(&p), fresh.count(&p), "{pattern}");
            assert_eq!(loaded.locate(&p), fresh.locate(&p), "{pattern}");
        }
    }

    #[test]
    fn test_v1_onehot_fixture_loads_with_lcp_and_lookup() {
        let blob = include_bytes!("../../tests/fixtures/v1_onehot_lcp_lookup.hfm");
        assert_eq!(&blob[..4], &super::FORMAT_MAGIC_V1);
        let loaded = FmIndex::from_bytes(blob).unwrap();
        assert!(loaded.lookup.is_some());
        assert_matches_fresh_build(&loaded, &onehot_fixture_config());
    }

    #[test]
    fn test_v1_bitplane_fixture_loads_without_lcp() {
        let blob = include_bytes!("../../tests/fixtures/v1_bitplane_nolcp.hfm");
        assert_eq!(&blob[..4], &super::FORMAT_MAGIC_V1);
        let loaded = FmIndex::from_bytes(blob).unwrap();
        assert!(loaded.lookup.is_none());
        assert_matches_fresh_build(&loaded, &bitplane_fixture_config());
    }

    #[test]
    fn test_legacy_fixture_loads_without_lcp() {
        let blob = include_bytes!("../../tests/fixtures/legacy_bitplane_nolcp.hfm");
        assert_ne!(&blob[..3], b"HFM");
        let loaded = FmIndex::from_bytes(blob).unwrap();
        assert_matches_fresh_build(&loaded, &bitplane_fixture_config());
    }

    #[test]
    fn test_v2_roundtrip_every_field_on_every_config() {
        use crate::occ::OccEncoding;
        for encoding in [OccEncoding::Bitplane, OccEncoding::OneHot] {
            for build_lcp in [false, true] {
                for lookup_depth in [0, 2] {
                    let config = FmIndexConfig {
                        sa_sample_rate: 3,
                        use_gpu: false,
                        occ_encoding: encoding,
                        build_lcp,
                        lookup_depth,
                        ..Default::default()
                    };
                    let original = FmIndex::build_cpu(&fixture_input(), &config).unwrap();
                    let bytes = original.to_bytes().unwrap();
                    assert_eq!(&bytes[..4], &super::FORMAT_MAGIC);
                    let restored = FmIndex::from_bytes(&bytes).unwrap();
                    assert_eq!(restored.c_array, original.c_array);
                    assert_eq!(restored.occ, original.occ);
                    assert_eq!(restored.sa_samples, original.sa_samples);
                    assert_eq!(restored.text, original.text);
                    assert_eq!(restored.text_len, original.text_len);
                    assert_eq!(restored.num_sequences, original.num_sequences);
                    assert_eq!(restored.seq_boundaries, original.seq_boundaries);
                    assert_eq!(restored.seq_headers, original.seq_headers);
                    assert_eq!(restored.lcp, original.lcp);
                    assert_eq!(restored.lookup.is_some(), lookup_depth > 0);
                    assert_eq!(restored.alphabet_fns.tag, original.alphabet_fns.tag);
                    assert_eq!(restored.to_bytes().unwrap(), bytes);
                    assert_matches_fresh_build(&restored, &config);
                }
            }
        }
    }

    #[test]
    fn test_truncated_input_rejected_clearly() {
        let err = FmIndex::from_bytes(b"HF").unwrap_err();
        assert!(err.to_string().contains("shorter than"), "{err}");
    }

    #[test]
    fn test_unknown_format_version_rejected() {
        let seq = DnaSequence::from_str("ACGTACGTTTGA").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 2,
            use_gpu: false,
            ..Default::default()
        };
        let mut bytes = FmIndex::build_cpu(&[seq], &config)
            .unwrap()
            .to_bytes()
            .unwrap();
        bytes[3] = 3;
        let err = FmIndex::from_bytes(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("format version 3"),
            "unexpected error: {err}"
        );
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
