//! Serialization and deserialization of the FM-index to/from bytes via `bincode`.
//!
//! ## Format
//!
//! Version 4 (current): [`FORMAT_MAGIC`] (`"HFM\x04"`) followed by the bincode encoding of
//! [`SerializableFmIndex`]. The large numeric vectors (occ checkpoints and block records,
//! SA samples, LCP arrays, lookup table, the text) are length-prefixed little-endian byte
//! blobs (`crate::serde_raw`) that load with one copy each. The alphabet's matching tables
//! ([`AlphabetFns`]) are stored in full, so an index built with a custom alphabet loads back
//! with the semantics it was built with; the lookup table lists wildcard variants per k-mer
//! (`crate::fm_index::lookup`). The classes of identical input sequences are stored as each
//! sequence's stored slot plus the duplicate-row weights
//! ([`seq_classes`](crate::fm_index::seq_classes)); the boundaries are those of the stored
//! sequences.
//!
//! Version 3 ([`FORMAT_MAGIC_V3`], `"HFM\x03"`): version 4 without the sequence classes
//! (written before deduplication: one stored sequence per input sequence). Still read
//! (`OwnedSerializableFmIndexV3`), never written.
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
use crate::fm_index::seq_classes::{DupWeights, SeqClasses};
use crate::fm_index::seq_id::HeaderIndex;
use crate::lcp::{LcpArray, LcpArrayV1};
use crate::occ::{OccTable, OccTableV1};
use crate::suffix_array::{SampledSuffixArray, SampledSuffixArrayV1};

/// Leading bytes of a serialized index: `"HFM"` plus the format version (currently 4).
pub const FORMAT_MAGIC: [u8; 4] = *b"HFM\x04";
/// Magic of format version 3, still accepted by [`FmIndex::from_bytes`].
pub const FORMAT_MAGIC_V3: [u8; 4] = *b"HFM\x03";
/// Magic of format version 2, still accepted by [`FmIndex::from_bytes`].
pub const FORMAT_MAGIC_V2: [u8; 4] = *b"HFM\x02";
/// Magic of format version 1, still accepted by [`FmIndex::from_bytes`].
pub const FORMAT_MAGIC_V1: [u8; 4] = *b"HFM\x01";

impl FmIndex {
    /// Serialize the FM-index to bytes (format version 4, see the module docs).
    pub fn to_bytes(&self) -> Result<Vec<u8>, FmIndexError> {
        let serializable = SerializableFmIndex {
            c_array: &self.c_array,
            occ: &self.occ,
            sa_samples: &self.sa_samples,
            text: &self.text,
            text_len: self.text_len,
            num_sequences: self.num_sequences,
            slot_boundaries: &self.slot_boundaries,
            seq_headers: &self.seq_headers,
            seq_to_slot: self.seq_classes.seq_to_slot(),
            dup_weights: &self.dup_weights,
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

    /// Deserialize an FM-index from bytes: the current format, format versions 1 to 3, or
    /// a legacy unversioned blob (which loads without an LCP array). See the module docs.
    /// Formats before 4 load with every sequence in a class of its own.
    ///
    /// Versions 3 and 4 carry the alphabet's matching tables; older versions carry a tag from
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
        check_stored_alphabet(&alphabet_fns)?;
        let header_index = HeaderIndex::build(&deserialized.seq_headers)?;
        let mut occ = deserialized.occ;
        // Rank and SA-sample reads are unchecked: a blob whose buffers disagree with its
        // lengths must be rejected here, not discovered as an out-of-bounds read later.
        let invalid =
            |e: String| FmIndexError::DeserializeError(format!("serialized FM-index: {e}"));
        occ.validate_structure().map_err(invalid)?;
        if occ.text_len != deserialized.text_len {
            return Err(invalid(
                "occ table length differs from the text length".into(),
            ));
        }
        deserialized
            .sa_samples
            .validate(deserialized.text_len)
            .map_err(invalid)?;
        let seq_classes = check_sequence_tables(
            deserialized.text_len,
            deserialized.num_sequences,
            &deserialized.slot_boundaries,
            &deserialized.seq_headers,
            deserialized.seq_to_slot,
            &deserialized.dup_weights,
        )
        .map_err(invalid)?;
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
            slot_boundaries: deserialized.slot_boundaries,
            seq_headers: deserialized.seq_headers,
            seq_classes,
            dup_weights: deserialized.dup_weights,
            lookup,
            alphabet_fns,
            lcp: deserialized.lcp,
        })
    }
}

/// Check a loaded index's per-sequence tables against each other and the text length, and
/// rebuild its [`SeqClasses`]. Shared by every index format that stores sequence classes.
pub(crate) fn check_sequence_tables(
    text_len: u32,
    num_sequences: u32,
    slot_boundaries: &[u32],
    seq_headers: &[String],
    seq_to_slot: Vec<u32>,
    dup_weights: &DupWeights,
) -> Result<SeqClasses, String> {
    if seq_headers.len() != num_sequences as usize || seq_to_slot.len() != num_sequences as usize {
        return Err(format!(
            "{num_sequences} sequences but {} headers and {} class entries",
            seq_headers.len(),
            seq_to_slot.len()
        ));
    }
    if slot_boundaries.last().copied().unwrap_or(0) != text_len
        || slot_boundaries.first() == Some(&0)
        || slot_boundaries.windows(2).any(|w| w[0] >= w[1])
    {
        return Err("sequence boundaries do not partition the text".into());
    }
    let classes = SeqClasses::from_seq_to_slot(seq_to_slot, slot_boundaries.len())?;
    dup_weights.validate(text_len, slot_boundaries, &classes)?;
    Ok(classes)
}

/// Decode any accepted format into the current field layout. A version-1/2 lookup table
/// comes back separately: converting it needs a working occ table, which the caller has.
/// Formats before 4 predate deduplication: every sequence is stored, in a class of its own.
fn decode(data: &[u8]) -> Result<(OwnedSerializableFmIndex, Option<LookupTableV1>), FmIndexError> {
    let de = |e: bincode::Error| FmIndexError::DeserializeError(e.to_string());
    // Sized by the headers actually read (bounded by the input), never by the stored
    // `num_sequences`, which a corrupt file could set to u32::MAX; `check_sequence_tables`
    // then rejects any mismatch between the two.
    let identity = |headers: &[String]| (0..headers.len() as u32).collect::<Vec<u32>>();
    if let Some(body) = data.strip_prefix(&FORMAT_MAGIC) {
        let v4: OwnedSerializableFmIndex = bincode::deserialize(body).map_err(de)?;
        Ok((v4, None))
    } else if let Some(body) = data.strip_prefix(&FORMAT_MAGIC_V3) {
        let v3: OwnedSerializableFmIndexV3 = bincode::deserialize(body).map_err(de)?;
        Ok((
            OwnedSerializableFmIndex {
                c_array: v3.c_array,
                occ: v3.occ,
                sa_samples: v3.sa_samples,
                text: v3.text,
                text_len: v3.text_len,
                num_sequences: v3.num_sequences,
                slot_boundaries: v3.seq_boundaries,
                seq_to_slot: identity(&v3.seq_headers),
                seq_headers: v3.seq_headers,
                dup_weights: DupWeights::default(),
                lookup: v3.lookup,
                lcp: v3.lcp,
                alphabet: v3.alphabet,
            },
            None,
        ))
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
                slot_boundaries: v2.seq_boundaries,
                seq_to_slot: identity(&v2.seq_headers),
                seq_headers: v2.seq_headers,
                dup_weights: DupWeights::default(),
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
                slot_boundaries: v1.seq_boundaries,
                seq_to_slot: identity(&v1.seq_headers),
                seq_headers: v1.seq_headers,
                dup_weights: DupWeights::default(),
                lookup: None,
                lcp: v1.lcp.map(Into::into),
                alphabet,
            },
            v1.lookup,
        ))
    } else if let [b'H', b'F', b'M', version, ..] = data {
        Err(FmIndexError::DeserializeError(format!(
            "unsupported serialized FM-index format version {version} \
             (this build reads versions 1 to 4)"
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
                slot_boundaries: legacy.seq_boundaries,
                seq_to_slot: identity(&legacy.seq_headers),
                seq_headers: legacy.seq_headers,
                dup_weights: DupWeights::default(),
                lookup: None,
                lcp: None,
                alphabet,
            },
            legacy.lookup,
        ))
    }
}

/// A stored alphabet with a built-in tag (`< 128`) must carry exactly that built-in's
/// tables; tags without a built-in below 128 are reserved.
pub(crate) fn check_stored_alphabet(alphabet_fns: &AlphabetFns) -> Result<(), FmIndexError> {
    if alphabet_fns.tag() >= 128 {
        return Ok(());
    }
    match alphabet_fns_from_tag(alphabet_fns.tag()) {
        Some(builtin) if builtin == *alphabet_fns => Ok(()),
        Some(_) => Err(FmIndexError::DeserializeError(format!(
            "serialized FM-index carries built-in alphabet tag {} with tables \
             that do not match that alphabet",
            alphabet_fns.tag()
        ))),
        None => Err(FmIndexError::DeserializeError(format!(
            "reserved alphabet tag {} in serialized FM-index (custom alphabets \
             use tags >= 128)",
            alphabet_fns.tag()
        ))),
    }
}

/// Formats before 3 store only the alphabet tag, so only built-in alphabets load from them.
fn alphabet_from_tag(tag: u8) -> Result<AlphabetFns, FmIndexError> {
    alphabet_fns_from_tag(tag).ok_or_else(|| {
        FmIndexError::DeserializeError(format!("unknown alphabet tag {tag} in serialized FM-index"))
    })
}

/// Format version 4 write layout. Field order is the wire order.
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
    /// Ends of the stored (distinct) sequences.
    slot_boundaries: &'a [u32],
    seq_headers: &'a [String],
    /// Stored slot of every input sequence (format version 4 onwards).
    #[serde(with = "crate::serde_raw::u32s")]
    seq_to_slot: &'a [u32],
    dup_weights: &'a DupWeights,
    lookup: Option<&'a LookupTable>,
    /// Present when built with `FmIndexConfig::build_lcp` (format version 1 onwards).
    lcp: Option<&'a LcpArray>,
    /// The alphabet's matching tables; its tag is the blob's final byte, which
    /// `test_serialize_bad_tag_rejected` corrupts to reach it.
    alphabet: &'a AlphabetFns,
}

/// Format version 4 read layout; mirrors [`SerializableFmIndex`].
#[derive(serde::Deserialize)]
struct OwnedSerializableFmIndex {
    c_array: CArray,
    occ: OccTable,
    sa_samples: SampledSuffixArray,
    #[serde(with = "crate::serde_raw::bytes")]
    text: Vec<u8>,
    text_len: u32,
    num_sequences: u32,
    slot_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_to_slot: Vec<u32>,
    dup_weights: DupWeights,
    lookup: Option<LookupTable>,
    lcp: Option<LcpArray>,
    /// Kept last (see [`SerializableFmIndex`]).
    alphabet: AlphabetFns,
}

/// Format version 3 read layout: version 4 without the sequence classes.
#[derive(serde::Deserialize)]
struct OwnedSerializableFmIndexV3 {
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
    // generator, `v2_*.hfm` at `7e4bff0` (format version 2) and `v3_*.hfm` at `1077d09`
    // (format version 3) by `examples/gen_fixtures.rs`, all from exactly the input and
    // configs below (which has no duplicate sequences); the legacy blob is the version-1
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
    fn test_v3_fixtures_load_as_identity_classes() {
        let cases: [(&[u8], FmIndexConfig, bool); 3] = [
            (
                include_bytes!("../../tests/fixtures/v3_onehot_lcp_lookup.hfm"),
                onehot_fixture_config(),
                false,
            ),
            (
                include_bytes!("../../tests/fixtures/v3_bitplane_nolcp.hfm"),
                bitplane_fixture_config(),
                false,
            ),
            (
                include_bytes!("../../tests/fixtures/v3_exact_onehot_lookup.hfm"),
                onehot_fixture_config(),
                true,
            ),
        ];
        for (blob, config, exact) in cases {
            assert_eq!(&blob[..4], &super::FORMAT_MAGIC_V3);
            let loaded = FmIndex::from_bytes(blob).unwrap();
            assert!(loaded.seq_classes.is_identity());
            assert!(loaded.dup_weights.is_empty());
            if exact {
                assert_matches_fresh_build_with::<ExactDna>(&loaded, &config);
            } else {
                assert_matches_fresh_build(&loaded, &config);
            }
        }
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
    fn test_v4_roundtrip_every_field_on_every_config() {
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
                    assert_eq!(restored.slot_boundaries, original.slot_boundaries);
                    assert_eq!(restored.seq_headers, original.seq_headers);
                    assert_eq!(restored.seq_classes, original.seq_classes);
                    assert_eq!(restored.dup_weights, original.dup_weights);
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
        bytes[3] = 5;
        let err = FmIndex::from_bytes(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("format version 5"),
            "unexpected error: {err}"
        );
    }

    fn duplicated_input() -> Vec<DnaSequence> {
        vec![
            DnaSequence::from_str_with_header("ACGTGATTACA", "a").unwrap(),
            DnaSequence::from_str_with_header("TTGCA", "b").unwrap(),
            DnaSequence::from_str_with_header("ACGTGATTACA", "a2").unwrap(),
            DnaSequence::from_str_with_header("TTGCA", "b2").unwrap(),
            DnaSequence::from_str_with_header("ACGTGATTACA", "a3").unwrap(),
        ]
    }

    #[test]
    fn test_v4_roundtrip_keeps_sequence_classes() {
        let config = FmIndexConfig {
            sa_sample_rate: 2,
            use_gpu: false,
            ..Default::default()
        };
        let original = FmIndex::build_cpu(&duplicated_input(), &config).unwrap();
        assert!(!original.seq_classes.is_identity());
        assert!(!original.dup_weights.is_empty());
        let bytes = original.to_bytes().unwrap();
        let restored = FmIndex::from_bytes(&bytes).unwrap();
        assert_eq!(restored.seq_classes, original.seq_classes);
        assert_eq!(restored.dup_weights, original.dup_weights);
        assert_eq!(restored.to_bytes().unwrap(), bytes);
        let p = encode_pattern("GATTA");
        assert_eq!(restored.count(&p), 3);
        assert_eq!(restored.locate(&p), original.locate(&p));
    }

    #[test]
    fn test_v4_rejects_inconsistent_sequence_classes() {
        let config = FmIndexConfig {
            sa_sample_rate: 2,
            use_gpu: false,
            ..Default::default()
        };
        let original = FmIndex::build_cpu(&duplicated_input(), &config).unwrap();
        // Rewrite one table at a time through the private write layout.
        let encode = |seq_to_slot: &[u32], dup_weights: &super::DupWeights| {
            let serializable = super::SerializableFmIndex {
                c_array: &original.c_array,
                occ: &original.occ,
                sa_samples: &original.sa_samples,
                text: &original.text,
                text_len: original.text_len,
                num_sequences: original.num_sequences,
                slot_boundaries: &original.slot_boundaries,
                seq_headers: &original.seq_headers,
                seq_to_slot,
                dup_weights,
                lookup: original.lookup.as_ref(),
                lcp: original.lcp.as_ref(),
                alphabet: &original.alphabet_fns,
            };
            let mut out = super::FORMAT_MAGIC.to_vec();
            out.extend(bincode::serialize(&serializable).unwrap());
            out
        };
        let good = encode(original.seq_classes.seq_to_slot(), &original.dup_weights);
        FmIndex::from_bytes(&good).unwrap();
        let bad: [&[u32]; 3] = [&[0, 1, 0, 1], &[0, 1, 0, 1, 2], &[0, 2, 0, 1, 0]];
        for bad_slots in bad {
            let err = FmIndex::from_bytes(&encode(bad_slots, &original.dup_weights)).unwrap_err();
            assert!(err.to_string().contains("serialized FM-index"), "{err}");
        }
        let err = FmIndex::from_bytes(&encode(
            original.seq_classes.seq_to_slot(),
            &super::DupWeights::default(),
        ))
        .unwrap_err();
        assert!(err.to_string().contains("class sizes"), "{err}");
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
