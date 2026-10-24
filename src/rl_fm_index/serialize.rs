//! Serialization of [`RlFmIndex`] / [`BidirRlFmIndex`] via `bincode`.
//!
//! [`FORMAT_MAGIC`] (`"HRL\x03"`) followed by the bincode encoding of the index fields; the
//! run-length BWT's vectors are little-endian byte blobs (`crate::serde_raw`). Derived state
//! (header map, bucket directory, occ select hints) is rebuilt at load, and the run
//! structure is validated before any query can touch it.
//!
//! [`BIDIR_FORMAT_MAGIC`] (`"HRB\x03"`), a little-endian `u32` byte length of the forward
//! half, then the forward and reverse halves, each a complete `"HRL"` blob.
//!
//! Version 4 stores the classes of identical input sequences (each sequence's stored slot,
//! and the duplicate-row weights; see [`seq_classes`](crate::fm_index::seq_classes)) after
//! the headers; the boundaries are those of the stored sequences. Version 3, written
//! before deduplication, is still read with every sequence in a class of its own.
//!
//! Version 3 stores the extraction rows per sentinel-separated *piece* (`piece_boundaries`
//! and `piece_end_rows`), so a sequence holding `#` record delimiters can be read back.
//! Version 2 stored one row per sequence (`seq_end_rows`); it is still read, as a file whose
//! pieces are its sequences.

use super::{BidirRlFmIndex, RlFmIndex};
use crate::alphabet::AlphabetFns;
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::fm_index::lookup::LookupTable;
use crate::fm_index::seq_classes::DupWeights;
use crate::fm_index::seq_id::HeaderIndex;
use crate::fm_index::serialize::{check_sequence_tables, check_stored_alphabet};
use crate::occ::BwtRank;
use crate::rlbwt::RlBwt;
use crate::suffix_array::SampledSuffixArray;

/// Leading bytes of a serialized [`RlFmIndex`]: `"HRL"` plus the format version.
pub const FORMAT_MAGIC: [u8; 4] = *b"HRL\x04";
/// Leading bytes of a serialized [`BidirRlFmIndex`]: `"HRB"` plus the format version.
pub const BIDIR_FORMAT_MAGIC: [u8; 4] = *b"HRB\x04";
/// Version 3 and 2 markers, still readable.
const FORMAT_MAGIC_V3: [u8; 4] = *b"HRL\x03";
const BIDIR_FORMAT_MAGIC_V3: [u8; 4] = *b"HRB\x03";
const FORMAT_MAGIC_V2: [u8; 4] = *b"HRL\x02";
const BIDIR_FORMAT_MAGIC_V2: [u8; 4] = *b"HRB\x02";

/// Write layout. Field order is the wire order.
#[derive(serde::Serialize)]
struct SerializableRlFmIndex<'a> {
    c_array: &'a CArray,
    bwt: &'a RlBwt,
    sa_samples: Option<&'a SampledSuffixArray>,
    text_len: u32,
    num_sequences: u32,
    /// Ends of the stored (distinct) sequences.
    #[serde(with = "crate::serde_raw::u32s")]
    slot_boundaries: &'a [u32],
    seq_headers: &'a [String],
    /// Stored slot of every input sequence (version 4 onwards).
    #[serde(with = "crate::serde_raw::u32s")]
    seq_to_slot: &'a [u32],
    dup_weights: &'a DupWeights,
    alphabet: &'a AlphabetFns,
    #[serde(with = "crate::serde_raw::u32s")]
    piece_boundaries: &'a [u32],
    #[serde(with = "crate::serde_raw::u32s")]
    piece_end_rows: &'a [u32],
    lookup: Option<&'a LookupTable>,
}

/// Read layout, mirroring [`SerializableRlFmIndex`].
#[derive(serde::Deserialize)]
struct OwnedSerializableRlFmIndex {
    c_array: CArray,
    bwt: RlBwt,
    sa_samples: Option<SampledSuffixArray>,
    text_len: u32,
    num_sequences: u32,
    #[serde(with = "crate::serde_raw::u32s")]
    slot_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_to_slot: Vec<u32>,
    dup_weights: DupWeights,
    alphabet: AlphabetFns,
    #[serde(with = "crate::serde_raw::u32s")]
    piece_boundaries: Vec<u32>,
    #[serde(with = "crate::serde_raw::u32s")]
    piece_end_rows: Vec<u32>,
    lookup: Option<LookupTable>,
}

/// Version 3 read layout: version 4 without the sequence classes.
#[derive(serde::Deserialize)]
struct OwnedSerializableRlFmIndexV3 {
    c_array: CArray,
    bwt: RlBwt,
    sa_samples: Option<SampledSuffixArray>,
    text_len: u32,
    num_sequences: u32,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    alphabet: AlphabetFns,
    #[serde(with = "crate::serde_raw::u32s")]
    piece_boundaries: Vec<u32>,
    #[serde(with = "crate::serde_raw::u32s")]
    piece_end_rows: Vec<u32>,
    lookup: Option<LookupTable>,
}

impl From<OwnedSerializableRlFmIndexV3> for OwnedSerializableRlFmIndex {
    /// A version 3 file predates deduplication: every sequence is stored, in its own class.
    /// The identity is sized by the headers read (bounded by the input), not by the stored
    /// `num_sequences`; the load check rejects a mismatch.
    fn from(v3: OwnedSerializableRlFmIndexV3) -> Self {
        let seq_to_slot = (0..v3.seq_headers.len() as u32).collect();
        Self {
            c_array: v3.c_array,
            bwt: v3.bwt,
            sa_samples: v3.sa_samples,
            text_len: v3.text_len,
            num_sequences: v3.num_sequences,
            slot_boundaries: v3.seq_boundaries,
            seq_headers: v3.seq_headers,
            seq_to_slot,
            dup_weights: DupWeights::default(),
            alphabet: v3.alphabet,
            piece_boundaries: v3.piece_boundaries,
            piece_end_rows: v3.piece_end_rows,
            lookup: v3.lookup,
        }
    }
}

/// Version 2 read layout: one extraction row per sequence.
#[derive(serde::Deserialize)]
struct OwnedSerializableRlFmIndexV2 {
    c_array: CArray,
    bwt: RlBwt,
    sa_samples: Option<SampledSuffixArray>,
    text_len: u32,
    num_sequences: u32,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    alphabet: AlphabetFns,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_end_rows: Vec<u32>,
    lookup: Option<LookupTable>,
}

impl From<OwnedSerializableRlFmIndexV2> for OwnedSerializableRlFmIndexV3 {
    /// A version 2 file has no `#` delimiters, so its pieces are its sequences.
    fn from(v2: OwnedSerializableRlFmIndexV2) -> Self {
        let piece_boundaries = if v2.seq_end_rows.is_empty() {
            Vec::new()
        } else {
            v2.seq_boundaries.clone()
        };
        Self {
            c_array: v2.c_array,
            bwt: v2.bwt,
            sa_samples: v2.sa_samples,
            text_len: v2.text_len,
            num_sequences: v2.num_sequences,
            seq_boundaries: v2.seq_boundaries,
            seq_headers: v2.seq_headers,
            alphabet: v2.alphabet,
            piece_boundaries,
            piece_end_rows: v2.seq_end_rows,
            lookup: v2.lookup,
        }
    }
}

fn corrupt(what: impl std::fmt::Display) -> FmIndexError {
    FmIndexError::DeserializeError(format!("serialized run-length FM-index: {what}"))
}

impl RlFmIndex {
    /// Serialize to bytes (layout: [`FORMAT_MAGIC`] then the bincode-encoded fields).
    pub fn to_bytes(&self) -> Result<Vec<u8>, FmIndexError> {
        let body = bincode::serialize(&SerializableRlFmIndex {
            c_array: &self.c_array,
            bwt: &self.bwt,
            sa_samples: self.sa_samples.as_ref(),
            text_len: self.text_len,
            num_sequences: self.num_sequences,
            slot_boundaries: &self.slot_boundaries,
            seq_headers: &self.seq_headers,
            seq_to_slot: self.seq_classes.seq_to_slot(),
            dup_weights: &self.dup_weights,
            alphabet: &self.alphabet_fns,
            piece_boundaries: &self.piece_boundaries,
            piece_end_rows: &self.piece_end_rows,
            lookup: self.lookup.as_ref(),
        })
        .map_err(|e| FmIndexError::SerializeError(e.to_string()))?;
        let mut out = Vec::with_capacity(FORMAT_MAGIC.len() + body.len());
        out.extend_from_slice(&FORMAT_MAGIC);
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Deserialize from bytes written by [`to_bytes`](Self::to_bytes).
    ///
    /// Reads version 4, 3 and 2 files.
    ///
    /// Returns [`FmIndexError::DeserializeError`] for a wrong magic or version, a truncated
    /// or malformed body, an inconsistent run structure or sequence classes, or an alphabet
    /// whose built-in tag does not match its tables.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        let d: OwnedSerializableRlFmIndex = if let Some(body) = data.strip_prefix(&FORMAT_MAGIC) {
            bincode::deserialize(body).map_err(corrupt)?
        } else if let Some(body) = data.strip_prefix(&FORMAT_MAGIC_V3) {
            bincode::deserialize::<OwnedSerializableRlFmIndexV3>(body)
                .map_err(corrupt)?
                .into()
        } else if let Some(body) = data.strip_prefix(&FORMAT_MAGIC_V2) {
            OwnedSerializableRlFmIndexV3::from(
                bincode::deserialize::<OwnedSerializableRlFmIndexV2>(body).map_err(corrupt)?,
            )
            .into()
        } else {
            return Err(corrupt("missing \"HRL\\x04\" format marker"));
        };
        check_stored_alphabet(&d.alphabet)?;
        // Deserializing the BWT already validated its runs and rebuilt the blocks.
        let bwt = d.bwt;
        if bwt.len() != d.text_len {
            return Err(corrupt("BWT length differs from the text length"));
        }
        if d.c_array.data.windows(2).any(|w| w[0] > w[1])
            || d.c_array.data[crate::alphabet::ALPHABET_SIZE - 1] > d.text_len
        {
            return Err(corrupt("C array is not a prefix sum within the text"));
        }
        for c in 0..crate::alphabet::ALPHABET_SIZE as u8 {
            if d.c_array.symbol_count(c, d.text_len) != bwt.symbol_count(c) {
                return Err(corrupt(format!(
                    "C array disagrees with the BWT on symbol {c}"
                )));
            }
        }
        if let Some(samples) = &d.sa_samples {
            samples.validate(d.text_len).map_err(corrupt)?;
        }
        // Every stored sequence holds at least its sentinel, and every input sequence names
        // one of them.
        let seq_classes = check_sequence_tables(
            d.text_len,
            d.num_sequences,
            &d.slot_boundaries,
            &d.seq_headers,
            d.seq_to_slot,
            &d.dup_weights,
        )
        .map_err(|e| corrupt(format!("sequence metadata disagrees with the text: {e}")))?;
        check_pieces(
            &d.piece_boundaries,
            &d.piece_end_rows,
            &d.slot_boundaries,
            d.text_len,
            &bwt,
        )?;
        if let Some(lookup) = &d.lookup {
            lookup.validate(d.text_len, &d.alphabet).map_err(corrupt)?;
        }
        let header_index = HeaderIndex::build(&d.seq_headers)?;
        Ok(Self {
            c_array: d.c_array,
            bwt,
            sa_samples: d.sa_samples,
            text_len: d.text_len,
            num_sequences: d.num_sequences,
            slot_boundaries: d.slot_boundaries,
            seq_headers: d.seq_headers,
            header_index,
            seq_classes,
            dup_weights: d.dup_weights,
            alphabet_fns: d.alphabet,
            piece_boundaries: d.piece_boundaries,
            piece_end_rows: d.piece_end_rows,
            lookup: d.lookup,
        })
    }
}

/// Validate the extraction metadata against the BWT and the (already checked) stored
/// sequence boundaries: empty (reverse half), or one sentinel-separated piece per BWT
/// sentinel, with strictly increasing ends finishing at the text length, every sequence end
/// among them,
/// and one distinct sentinel row (`0..pieces`) per piece. A piece's row must not hold a
/// sentinel in the BWT unless the piece is empty, or extraction would walk off its start.
fn check_pieces(
    bounds: &[u32],
    rows: &[u32],
    seq_boundaries: &[u32],
    text_len: u32,
    bwt: &RlBwt,
) -> Result<(), FmIndexError> {
    if bounds.is_empty() && rows.is_empty() {
        return Ok(());
    }
    let pieces = bounds.len();
    if rows.len() != pieces
        || pieces as u32 != bwt.symbol_count(crate::alphabet::SENTINEL)
        || bounds.last() != Some(&text_len)
        || bounds.windows(2).any(|w| w[0] >= w[1])
        || bounds.first() == Some(&0)
        || seq_boundaries
            .iter()
            .any(|b| bounds.binary_search(b).is_err())
    {
        return Err(corrupt("piece boundaries disagree with the text"));
    }
    let mut seen = vec![false; pieces];
    for (k, &row) in rows.iter().enumerate() {
        let start = k.checked_sub(1).map_or(0, |j| bounds[j]);
        let empty = bounds[k] - start == 1;
        if row as usize >= pieces
            || std::mem::replace(&mut seen[row as usize], true)
            || (bwt.symbol_at(row) == crate::alphabet::SENTINEL) != empty
        {
            return Err(corrupt("piece end rows disagree with the BWT"));
        }
    }
    Ok(())
}

impl BidirRlFmIndex {
    /// Serialize both halves (layout: [`BIDIR_FORMAT_MAGIC`], the forward half's byte length as a little-endian `u32`, then both halves).
    pub fn to_bytes(&self) -> Result<Vec<u8>, FmIndexError> {
        let fwd = self.fwd.to_bytes()?;
        let rev = self.rev.to_bytes()?;
        let fwd_len = u32::try_from(fwd.len())
            .map_err(|_| FmIndexError::SerializeError("forward half exceeds 4 GiB".into()))?;
        let mut out = Vec::with_capacity(8 + fwd.len() + rev.len());
        out.extend_from_slice(&BIDIR_FORMAT_MAGIC);
        out.extend_from_slice(&fwd_len.to_le_bytes());
        out.extend_from_slice(&fwd);
        out.extend_from_slice(&rev);
        Ok(out)
    }

    /// Deserialize from bytes written by [`to_bytes`](Self::to_bytes) (version 4, 3 or 2).
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        let rest = data
            .strip_prefix(&BIDIR_FORMAT_MAGIC)
            .or_else(|| data.strip_prefix(&BIDIR_FORMAT_MAGIC_V3))
            .or_else(|| data.strip_prefix(&BIDIR_FORMAT_MAGIC_V2))
            .ok_or_else(|| corrupt("missing \"HRB\\x04\" format marker"))?;
        let (len_bytes, rest) = rest
            .split_first_chunk::<4>()
            .ok_or_else(|| corrupt("truncated bidirectional header"))?;
        let fwd_len = u32::from_le_bytes(*len_bytes) as usize;
        if fwd_len > rest.len() {
            return Err(corrupt("forward half length exceeds the blob"));
        }
        let (fwd, rev) = rest.split_at(fwd_len);
        let fwd = RlFmIndex::from_bytes(fwd)?;
        let rev = RlFmIndex::from_bytes(rev)?;
        // Both halves index the same symbols: equal C arrays catch swapped or mixed halves,
        // which would otherwise answer queries with silently wrong intervals.
        if fwd.text_len != rev.text_len
            || fwd.alphabet_fns != rev.alphabet_fns
            || fwd.c_array != rev.c_array
        {
            return Err(corrupt("forward and reverse halves do not match"));
        }
        Ok(Self::from_halves(fwd, rev))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::DnaSequence;
    use crate::rl_fm_index::RlFmIndexConfig;

    fn index(lookup_depth: u32) -> RlFmIndex {
        let seqs: Vec<DnaSequence> = ["ACGTTGCA", "ACGATGCA", "TTTT"]
            .iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect();
        let config = RlFmIndexConfig {
            sa_sample_rate: 4,
            lookup_depth,
            ..Default::default()
        };
        RlFmIndex::build_cpu(&seqs, &config).unwrap()
    }

    fn rejects(ix: &RlFmIndex) -> bool {
        RlFmIndex::from_bytes(&ix.to_bytes().unwrap()).is_err()
    }

    #[test]
    fn round_trip_keeps_extraction_and_lookup() {
        let ix = index(3);
        let back = RlFmIndex::from_bytes(&ix.to_bytes().unwrap()).unwrap();
        assert_eq!(back.piece_boundaries, ix.piece_boundaries);
        assert_eq!(back.piece_end_rows, ix.piece_end_rows);
        assert_eq!(back.lookup, ix.lookup);
    }

    #[test]
    fn rejects_corrupted_piece_end_rows() {
        let good = index(0);
        assert!(!rejects(&good));

        let mut dup = good.clone();
        dup.piece_end_rows[1] = dup.piece_end_rows[0];
        assert!(rejects(&dup), "duplicate row");

        let mut out_of_range = good.clone();
        out_of_range.piece_end_rows[0] = good.num_sequences;
        assert!(rejects(&out_of_range), "non-sentinel row");

        let mut short = good.clone();
        short.piece_end_rows.pop();
        assert!(rejects(&short), "wrong count");

        let mut rows_only = good.clone();
        rows_only.piece_boundaries.clear();
        assert!(rejects(&rows_only), "rows without boundaries");

        // A sequence without even its sentinel would let a row reach the BWT length.
        let mut empty_first = good.clone();
        empty_first.slot_boundaries[0] = 0;
        assert!(rejects(&empty_first), "zero-length first sequence");
    }

    fn duplicated() -> RlFmIndex {
        let seqs: Vec<DnaSequence> = ["ACGT#TG", "GGCA", "ACGT#TG", "GGCA", "ACGT#TG"]
            .iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect();
        let config = RlFmIndexConfig {
            sa_sample_rate: 2,
            ..Default::default()
        };
        RlFmIndex::build_cpu(&seqs, &config).unwrap()
    }

    #[test]
    fn round_trip_keeps_sequence_classes() {
        let ix = duplicated();
        assert_eq!(ix.num_sequences(), 5);
        assert_eq!(ix.num_distinct_sequences(), 2);
        let back = RlFmIndex::from_bytes(&ix.to_bytes().unwrap()).unwrap();
        assert_eq!(back.seq_classes, ix.seq_classes);
        assert_eq!(back.dup_weights, ix.dup_weights);
        assert_eq!(back.to_bytes().unwrap(), ix.to_bytes().unwrap());
        for id in 0..5 {
            assert_eq!(
                back.sequence(crate::SeqId(id)),
                ix.sequence(crate::SeqId(id))
            );
        }
    }

    #[test]
    fn rejects_corrupted_sequence_classes() {
        let good = duplicated();
        assert!(!rejects(&good));

        let mut no_weights = good.clone();
        no_weights.dup_weights = DupWeights::default();
        assert!(rejects(&no_weights), "weights missing for duplicated slots");

        let mut extra_header = good.clone();
        extra_header.seq_headers.push("extra".into());
        assert!(rejects(&extra_header), "header count");

        let mut identity = good.clone();
        identity.seq_classes = crate::fm_index::seq_classes::SeqClasses::identity(5);
        assert!(rejects(&identity), "classes naming more slots than stored");
    }

    // `tests/fixtures/v3_rl_*` were written at `1077d09` (format version 3) by
    // `examples/gen_fixtures.rs` from the fixture input and config below.
    #[test]
    fn reads_version_3_fixtures() {
        use crate::rl_fm_index::BidirRlFmIndex;
        let blob = include_bytes!("../../tests/fixtures/v3_rl_lookup.hrl");
        assert_eq!(&blob[..4], &FORMAT_MAGIC_V3);
        let back = RlFmIndex::from_bytes(blob).unwrap();
        assert!(back.seq_classes.is_identity());
        let fresh = RlFmIndex::build_cpu(&fixture_input(), &fixture_config()).unwrap();
        assert_eq!(back.to_bytes().unwrap(), fresh.to_bytes().unwrap());

        let blob = include_bytes!("../../tests/fixtures/v3_rl_bidir_lookup.hrb");
        assert_eq!(&blob[..4], &BIDIR_FORMAT_MAGIC_V3);
        let back = BidirRlFmIndex::from_bytes(blob).unwrap();
        let fresh = BidirRlFmIndex::build_cpu(&fixture_input(), &fixture_config()).unwrap();
        assert_eq!(back.to_bytes().unwrap(), fresh.to_bytes().unwrap());
    }

    /// `#` delimiters: one piece per record, and every sequence end must be a piece end.
    #[test]
    fn rejects_corrupted_piece_boundaries() {
        let seqs: Vec<DnaSequence> = ["ACGT#TGCA#GG", "ACGATGCA", "TT#TT"]
            .iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect();
        let config = RlFmIndexConfig {
            sa_sample_rate: 4,
            ..Default::default()
        };
        let good = RlFmIndex::build_cpu(&seqs, &config).unwrap();
        assert_eq!(good.piece_boundaries, vec![5, 10, 13, 22, 25, 28]);
        assert!(!rejects(&good));

        // Merging two pieces leaves a sentinel without a piece.
        let mut merged = good.clone();
        merged.piece_boundaries.remove(0);
        merged.piece_end_rows.remove(0);
        assert!(rejects(&merged), "piece count below sentinel count");

        let mut moved = good.clone();
        moved.piece_boundaries[2] = 14;
        assert!(rejects(&moved), "sequence end not a piece end");

        let mut unsorted = good.clone();
        unsorted.piece_boundaries.swap(0, 1);
        assert!(rejects(&unsorted), "unsorted pieces");

        let mut short_text = good.clone();
        *short_text.piece_boundaries.last_mut().unwrap() = 27;
        assert!(rejects(&short_text), "last piece short of the text");
    }

    // `tests/fixtures/v2_rl_*` were written at `69749a4` (format version 2) by
    // `examples/gen_fixtures.rs` from this input and config.
    fn fixture_input() -> Vec<DnaSequence> {
        vec![
            DnaSequence::from_str_with_header("ACGTNACGTRYSWKMBDHVACGTACGTTTGCA", "chr1").unwrap(),
            DnaSequence::from_str_with_header("TTGACGTACGTNNACGGAAC", "plasmid").unwrap(),
            DnaSequence::from_str_with_header("GATTACA", "tiny").unwrap(),
        ]
    }

    fn fixture_config() -> RlFmIndexConfig {
        RlFmIndexConfig {
            sa_sample_rate: 4,
            build_threads: 1,
            lookup_depth: 3,
        }
    }

    #[test]
    fn reads_version_2_fixture() {
        let blob = include_bytes!("../../tests/fixtures/v2_rl_lookup.hrl");
        let back = RlFmIndex::from_bytes(blob).unwrap();
        let fresh = RlFmIndex::build_cpu(&fixture_input(), &fixture_config()).unwrap();
        assert_eq!(back.to_bytes().unwrap(), fresh.to_bytes().unwrap());
        assert_eq!(
            back.sequence_by_header("plasmid"),
            fresh.sequence_by_header("plasmid")
        );
    }

    #[test]
    fn reads_version_2_bidir_fixture() {
        use crate::rl_fm_index::BidirRlFmIndex;
        let blob = include_bytes!("../../tests/fixtures/v2_rl_bidir_lookup.hrb");
        let back = BidirRlFmIndex::from_bytes(blob).unwrap();
        let fresh = BidirRlFmIndex::build_cpu(&fixture_input(), &fixture_config()).unwrap();
        assert_eq!(back.to_bytes().unwrap(), fresh.to_bytes().unwrap());
    }

    #[test]
    fn bidir_rejects_halves_of_different_indexes() {
        use crate::rl_fm_index::BidirRlFmIndex;
        let build = |texts: &[&str]| {
            let seqs: Vec<DnaSequence> = texts
                .iter()
                .map(|s| DnaSequence::from_str(s).unwrap())
                .collect();
            BidirRlFmIndex::build_cpu(&seqs, &RlFmIndexConfig::default()).unwrap()
        };
        let a = build(&["ACGTACGTAA", "CCGG"]);
        let b = build(&["AAAAAAAAAA", "TTTT"]);
        assert_eq!(a.fwd.text_len, b.fwd.text_len);
        let mixed = BidirRlFmIndex::from_halves(a.fwd.clone(), b.rev.clone());
        assert!(BidirRlFmIndex::from_bytes(&a.to_bytes().unwrap()).is_ok());
        assert!(BidirRlFmIndex::from_bytes(&mixed.to_bytes().unwrap()).is_err());
    }

    #[test]
    fn rejects_corrupted_lookup_table() {
        let mut ix = index(2);
        let mut table = ix.lookup.take().unwrap();
        table.depth = 3;
        ix.lookup = Some(table);
        assert!(rejects(&ix));
    }
}
