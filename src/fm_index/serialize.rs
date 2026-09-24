//! Serialization and deserialization of the FM-index to/from bytes via `bincode`.
//!
//! ## Format
//!
//! Version 3 (current): [`FORMAT_MAGIC`] (`"HFM\x03"`) followed by the bincode encoding of
//! [`SerializableFmIndex`]. The large numeric vectors (occ checkpoints and block records,
//! SA samples, LCP arrays, lookup table, the text) are length-prefixed little-endian byte
//! blobs (`crate::serde_raw`) that load with one copy each. The alphabet's matching tables
//! ([`AlphabetFns`]) are stored in full, so an index built with a custom alphabet loads back
//! with the semantics it was built with; the lookup table lists wildcard variants per k-mer
//! (`crate::fm_index::lookup`).
//!
//! Version 2 ([`FORMAT_MAGIC_V2`], `"HFM\x02"`): blobs as above, but only the alphabet *tag*
//! is stored (so only built-in alphabets load) and the lookup table is exact-only. Still
//! read (`OwnedSerializableFmIndexV2`), never written. A version-1/2 lookup table built for
//! an `IupacDna` index whose reference holds ambiguity codes missed matches; it is rebuilt
//! at load (same depth), otherwise it is converted in place.
//!
//! Version 1 ([`FORMAT_MAGIC_V1`], `"HFM\x01"`): version-2 fields with the vectors as plain
//! element sequences. Still read (`OwnedSerializableFmIndexV1`), never written.
//!
//! Legacy (no magic): version-1 field encodings without the LCP array. Distinguishable
//! because such blobs begin with the bincode encoding of `CArray::data[0]`, which is always
//! `0u32` (`C[0]` counts symbols below the sentinel) and so never collides with a magic. An
//! index loaded from one reports `has_lcp() == false`.
//!
//! Any other `"HFM"` version byte is rejected with a clear error rather than misparsed.

use super::FmIndex;
use crate::alphabet::{alphabet_fns_from_tag, AlphabetFns};
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::fm_index::lookup::{LookupTable, LookupTableV1};
use crate::fm_index::seq_id::HeaderIndex;
use crate::lcp::{LcpArray, LcpArrayV1};
use crate::occ::{OccTable, OccTableV1};
use crate::suffix_array::{SampledSuffixArray, SampledSuffixArrayV1};

/// Leading bytes of a serialized index: `"HFM"` plus the format version (currently 3).
pub const FORMAT_MAGIC: [u8; 4] = *b"HFM\x03";
/// Magic of format version 2, still accepted by [`FmIndex::from_bytes`].
pub const FORMAT_MAGIC_V2: [u8; 4] = *b"HFM\x02";
/// Magic of format version 1, still accepted by [`FmIndex::from_bytes`].
pub const FORMAT_MAGIC_V1: [u8; 4] = *b"HFM\x01";

impl FmIndex {
    /// Serialize the FM-index to bytes (format version 3, see the module docs).
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
            alphabet: &self.alphabet_fns,
        };
        let body = bincode::serialize(&serializable)
            .map_err(|e| FmIndexError::SerializeError(e.to_string()))?;
        let mut out = Vec::with_capacity(FORMAT_MAGIC.len() + body.len());
        out.extend_from_slice(&FORMAT_MAGIC);
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Deserialize an FM-index from bytes: the current format, format versions 1 and 2, or
    /// a legacy unversioned blob (which loads without an LCP array). See the module docs.
    ///
    /// Version 3 carries the alphabet's matching tables; older versions carry a tag from
    /// which a built-in alphabet is reconstructed. Returns
    /// [`FmIndexError::DeserializeError`] for unknown or reserved tags, for a built-in tag
    /// whose stored tables do not match the built-in alphabet, and for format versions this
    /// build does not read.
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
        let (deserialized, old_lookup) = decode(data)?;
        let alphabet_fns = deserialized.alphabet;
        if alphabet_fns.tag() < 128 {
            match alphabet_fns_from_tag(alphabet_fns.tag()) {
                Some(builtin) if builtin == alphabet_fns => {}
                Some(_) => {
                    return Err(FmIndexError::DeserializeError(format!(
                        "serialized FM-index carries built-in alphabet tag {} with tables \
                         that do not match that alphabet",
                        alphabet_fns.tag()
                    )));
                }
                None => {
                    return Err(FmIndexError::DeserializeError(format!(
                        "reserved alphabet tag {} in serialized FM-index (custom alphabets \
                         use tags >= 128)",
                        alphabet_fns.tag()
                    )));
                }
            }
        }
        let header_index = HeaderIndex::build(&deserialized.seq_headers)?;
        let mut occ = deserialized.occ;
        occ.rebuild_derived();
        // A version-1/2 table is upgraded only once the occ table can answer ranks again.
        let lookup = match (deserialized.lookup, old_lookup) {
            (Some(table), _) => Some(table),
            (None, Some(old)) => Some(old.upgrade(
                deserialized.text_len,
                &deserialized.c_array,
                &occ,
                &alphabet_fns,
            )),
            (None, None) => None,
        };
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
            lookup,
            alphabet_fns,
            lcp: deserialized.lcp,
        })
    }
}

/// Decode any accepted format into the current field layout. A version-1/2 lookup table
/// comes back separately: converting it needs a working occ table, which the caller has.
fn decode(data: &[u8]) -> Result<(OwnedSerializableFmIndex, Option<LookupTableV1>), FmIndexError> {
    let de = |e: bincode::Error| FmIndexError::DeserializeError(e.to_string());
    if let Some(body) = data.strip_prefix(&FORMAT_MAGIC) {
        let v3: OwnedSerializableFmIndex = bincode::deserialize(body).map_err(de)?;
        Ok((v3, None))
    } else if let Some(body) = data.strip_prefix(&FORMAT_MAGIC_V2) {
        let v2: OwnedSerializableFmIndexV2 = bincode::deserialize(body).map_err(de)?;
        let alphabet = alphabet_from_tag(v2.alphabet_tag)?;
        Ok((
            OwnedSerializableFmIndex {
                c_array: v2.c_array,
                occ: v2.occ,
                sa_samples: v2.sa_samples,
                text: v2.text,
                text_len: v2.text_len,
                num_sequences: v2.num_sequences,
                seq_boundaries: v2.seq_boundaries,
                seq_headers: v2.seq_headers,
                lookup: None,
                lcp: v2.lcp,
                alphabet,
            },
            v2.lookup,
        ))
    } else if let Some(body) = data.strip_prefix(&FORMAT_MAGIC_V1) {
        let v1: OwnedSerializableFmIndexV1 = bincode::deserialize(body).map_err(de)?;
        let alphabet = alphabet_from_tag(v1.alphabet_tag)?;
        Ok((
            OwnedSerializableFmIndex {
                c_array: v1.c_array,
                occ: v1.occ.into(),
                sa_samples: v1.sa_samples.into(),
                text: v1.text,
                text_len: v1.text_len,
                num_sequences: v1.num_sequences,
                seq_boundaries: v1.seq_boundaries,
                seq_headers: v1.seq_headers,
                lookup: None,
                lcp: v1.lcp.map(Into::into),
                alphabet,
            },
            v1.lookup,
        ))
    } else if let [b'H', b'F', b'M', version, ..] = data {
        Err(FmIndexError::DeserializeError(format!(
            "unsupported serialized FM-index format version {version} \
             (this build reads versions 1, 2 and 3)"
        )))
    } else {
        let legacy: LegacyOwnedSerializableFmIndex = bincode::deserialize(data).map_err(de)?;
        let alphabet = alphabet_from_tag(legacy.alphabet_tag)?;
        Ok((
            OwnedSerializableFmIndex {
                c_array: legacy.c_array,
                occ: legacy.occ.into(),
                sa_samples: legacy.sa_samples.into(),
                text: legacy.text,
                text_len: legacy.text_len,
                num_sequences: legacy.num_sequences,
                seq_boundaries: legacy.seq_boundaries,
                seq_headers: legacy.seq_headers,
                lookup: None,
                lcp: None,
                alphabet,
            },
            legacy.lookup,
        ))
    }
}

/// Formats before 3 store only the alphabet tag, so only built-in alphabets load from them.
fn alphabet_from_tag(tag: u8) -> Result<AlphabetFns, FmIndexError> {
    alphabet_fns_from_tag(tag).ok_or_else(|| {
        FmIndexError::DeserializeError(format!("unknown alphabet tag {tag} in serialized FM-index"))
    })
}

/// Format version 3 write layout. Field order is the wire order.
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
    /// The alphabet's matching tables; its tag is the blob's final byte, which
    /// `test_serialize_bad_tag_rejected` corrupts to reach it.
    alphabet: &'a AlphabetFns,
}

/// Format version 3 read layout; mirrors [`SerializableFmIndex`].
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
    alphabet: AlphabetFns,
}

/// Format version 2 read layout: version-3 blobs, an exact-only lookup table and the
/// alphabet tag alone.
#[derive(serde::Deserialize)]
struct OwnedSerializableFmIndexV2 {
    c_array: CArray,
    occ: OccTable,
    sa_samples: SampledSuffixArray,
    #[serde(with = "crate::serde_raw::bytes")]
    text: Vec<u8>,
    text_len: u32,
    num_sequences: u32,
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    lookup: Option<LookupTableV1>,
    lcp: Option<LcpArray>,
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
    lookup: Option<LookupTableV1>,
    lcp: Option<LcpArrayV1>,
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
    lookup: Option<LookupTableV1>,
    alphabet_tag: u8,
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
    // `tests/fixtures/v1_*.hfm` were written at `9b32c9f` (format version 1) by a throwaway
    // generator and `v2_*.hfm` at `7e4bff0` (format version 2) by `examples/gen_fixtures.rs`,
    // both from exactly the input and configs below; the legacy blob is the version-1
    // Bitplane blob minus its magic and the `Option<LcpArray>` None byte. CPU construction
    // is deterministic, so a fresh build must serialize to the same current-format bytes as
    // the loaded fixture — i.e. every stored field survived the version upgrade. The OneHot
    // fixtures carry a depth-3 lookup table over a reference with ambiguity codes: under
    // `IupacDna` the old exact-only table is rebuilt at load, under `ExactDna` it is
    // converted in place; either way it must equal what a fresh build produces.

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
        assert_matches_fresh_build_with::<IupacDna>(loaded, config);
    }

    fn assert_matches_fresh_build_with<A: Alphabet>(loaded: &FmIndex, config: &FmIndexConfig) {
        let fresh = FmIndex::build_cpu_with::<A>(&fixture_input(), config).unwrap();
        assert_eq!(loaded.alphabet_fns, fresh.alphabet_fns);
        assert_eq!(loaded.lookup, fresh.lookup);
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
    fn test_v2_onehot_fixture_loads_and_rebuilds_its_lookup_table() {
        let blob = include_bytes!("../../tests/fixtures/v2_onehot_lcp_lookup.hfm");
        assert_eq!(&blob[..4], &super::FORMAT_MAGIC_V2);
        let loaded = FmIndex::from_bytes(blob).unwrap();
        assert!(loaded.has_lcp());
        // The reference has N and R..V; the rebuilt table lists their variants.
        let lut = loaded.lookup.as_ref().unwrap();
        assert_eq!(lut.depth, 3);
        assert!(lut.num_intervals() > 0);
        assert_matches_fresh_build(&loaded, &onehot_fixture_config());
        // Queries whose seed window matches reference ambiguity codes agree with a full
        // (table-less) search — the old exact-only table would have missed them.
        let full = FmIndex::build_cpu(
            &fixture_input(),
            &FmIndexConfig {
                lookup_depth: 0,
                ..onehot_fixture_config()
            },
        )
        .unwrap();
        for pattern in ["TNA", "TCA", "GCA", "ACG", "NNA", "TAC"] {
            let p = encode_pattern(pattern);
            assert!(full.count(&p) > 0, "{pattern}");
            assert_eq!(loaded.count(&p), full.count(&p), "{pattern}");
            assert_eq!(loaded.locate(&p), full.locate(&p), "{pattern}");
        }
    }

    #[test]
    fn test_v2_bitplane_fixture_loads_without_lcp() {
        let blob = include_bytes!("../../tests/fixtures/v2_bitplane_nolcp.hfm");
        assert_eq!(&blob[..4], &super::FORMAT_MAGIC_V2);
        let loaded = FmIndex::from_bytes(blob).unwrap();
        assert!(loaded.lookup.is_none());
        assert_matches_fresh_build(&loaded, &bitplane_fixture_config());
    }

    #[test]
    fn test_v2_exact_fixture_converts_its_lookup_table_in_place() {
        let blob = include_bytes!("../../tests/fixtures/v2_exact_onehot_lookup.hfm");
        assert_eq!(&blob[..4], &super::FORMAT_MAGIC_V2);
        let loaded = FmIndex::from_bytes(blob).unwrap();
        assert_eq!(loaded.alphabet_fns, ExactDna::fns());
        assert_matches_fresh_build_with::<ExactDna>(&loaded, &onehot_fixture_config());
        assert_eq!(loaded.count(&encode_pattern("N")), 0);
    }

    #[test]
    fn test_v2_fixture_with_unknown_tag_rejected() {
        let mut blob = include_bytes!("../../tests/fixtures/v2_bitplane_nolcp.hfm").to_vec();
        *blob.last_mut().unwrap() = 0xFF;
        let err = FmIndex::from_bytes(&blob).unwrap_err();
        assert!(
            err.to_string().contains("unknown alphabet tag 255"),
            "{err}"
        );
    }

    #[test]
    fn test_custom_alphabet_roundtrip() {
        // A matches A or N in the reference; nothing else is ambiguous.
        struct AOrN;
        impl Alphabet for AOrN {
            fn fns() -> AlphabetFns {
                AlphabetFns::from_compatible_fn(
                    |c| match c {
                        x if x == A => &[A, N],
                        x if x == C => &[C],
                        x if x == G => &[G],
                        x if x == T => &[T],
                        _ => &[],
                    },
                    &[A, C, G, T],
                    200,
                )
            }
        }
        let seq = DnaSequence::from_str("ACGTNACGT").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            lookup_depth: 2,
            ..Default::default()
        };
        let original = FmIndex::build_cpu_with::<AOrN>(&[seq], &config).unwrap();
        let bytes = original.to_bytes().unwrap();
        assert_eq!(*bytes.last().unwrap(), 200);
        let restored = FmIndex::from_bytes(&bytes).unwrap();
        assert_eq!(restored.alphabet_fns, AOrN::fns());
        assert_eq!(restored.lookup, original.lookup);
        assert_eq!(restored.to_bytes().unwrap(), bytes);
        for pattern in ["A", "AN", "ACG", "NA", "TN"] {
            let p = encode_pattern(pattern);
            assert_eq!(restored.count(&p), original.count(&p), "{pattern}");
            assert_eq!(restored.locate(&p), original.locate(&p), "{pattern}");
        }
        // "A" matches the two A's and the N; "TA" only "TN" (position 3); N matches nothing.
        assert_eq!(restored.count(&encode_pattern("A")), 3);
        assert_eq!(restored.count(&encode_pattern("TA")), 1);
        assert_eq!(restored.locate(&encode_pattern("TA")).len(), 1);
        assert_eq!(restored.count(&encode_pattern("N")), 0);
    }

    #[test]
    fn test_v3_roundtrip_every_field_on_every_config() {
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
                    assert_eq!(restored.lookup, original.lookup);
                    assert_eq!(restored.alphabet_fns, original.alphabet_fns);
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
        bytes[3] = 4;
        let err = FmIndex::from_bytes(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("format version 4"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_serialize_bad_tag_rejected() {
        let seq = DnaSequence::from_str("ACGT").unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 4,
            use_gpu: false,
            ..Default::default()
        };
        let original = FmIndex::build_cpu(&[seq], &config).unwrap();
        let bytes = original.to_bytes().unwrap();
        // The tag is the blob's last byte. A reserved value is rejected...
        let mut reserved = bytes.clone();
        *reserved.last_mut().unwrap() = 0x7F;
        let err = FmIndex::from_bytes(&reserved).unwrap_err();
        assert!(
            err.to_string().contains("reserved alphabet tag 127"),
            "{err}"
        );
        // ...and so is a built-in tag whose stored tables belong to another alphabet.
        let mut mismatched = bytes.clone();
        *mismatched.last_mut().unwrap() = 1;
        let err = FmIndex::from_bytes(&mismatched).unwrap_err();
        assert!(err.to_string().contains("do not match"), "{err}");
        // A custom tag with the IUPAC tables is a legitimate custom alphabet.
        let mut custom = bytes;
        *custom.last_mut().unwrap() = 0xFF;
        let restored = FmIndex::from_bytes(&custom).unwrap();
        assert_eq!(restored.alphabet_fns.tag(), 0xFF);
        assert_eq!(
            restored.alphabet_fns.compatible(N),
            IupacDna::fns().compatible(N)
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
