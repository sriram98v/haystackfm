//! Serialization of [`MoveFmIndex`] (`"HMV\x02"`) and [`BMoveIndex`] (`"HBM\x02"`) via
//! `bincode`. Move tables validate themselves on load (a permutation of `[0, n)` with
//! correct row indices), the LF table is checked against the C array, and the locate
//! structures against the table sizes, so no loaded value reaches an out-of-bounds read.
//!
//! Version 2 stores the classes of identical input sequences (each sequence's stored slot,
//! and the duplicate-row weights; see [`seq_classes`](crate::fm_index::seq_classes)) after
//! the headers; the boundaries are those of the stored sequences. Version 1, written before
//! deduplication, is still read with every sequence in a class of its own.

use super::lf::MoveLf;
use super::locate::{MoveLocate, RunSamples};
use super::{BMoveIndex, MoveFmIndex};
use crate::alphabet::AlphabetFns;
use crate::error::FmIndexError;
use crate::fm_index::seq_classes::DupWeights;
use crate::fm_index::seq_id::HeaderIndex;
use crate::fm_index::serialize::{check_sequence_tables, check_stored_alphabet};

/// Leading bytes of a serialized [`MoveFmIndex`].
pub const FORMAT_MAGIC: [u8; 4] = *b"HMV\x02";
/// Leading bytes of a serialized [`BMoveIndex`].
pub const BMOVE_FORMAT_MAGIC: [u8; 4] = *b"HBM\x02";
/// Version 1 markers, still readable.
const FORMAT_MAGIC_V1: [u8; 4] = *b"HMV\x01";
const BMOVE_FORMAT_MAGIC_V1: [u8; 4] = *b"HBM\x01";

#[derive(serde::Serialize)]
struct SerializableMoveFmIndex<'a> {
    lf: &'a MoveLf,
    locate: Option<&'a MoveLocate>,
    text_len: u32,
    num_sequences: u32,
    /// Ends of the stored (distinct) sequences.
    #[serde(with = "crate::serde_raw::u32s")]
    slot_boundaries: &'a [u32],
    seq_headers: &'a [String],
    /// Stored slot of every input sequence (version 2 onwards).
    #[serde(with = "crate::serde_raw::u32s")]
    seq_to_slot: &'a [u32],
    dup_weights: &'a DupWeights,
    alphabet: &'a AlphabetFns,
}

#[derive(serde::Deserialize)]
struct OwnedSerializableMoveFmIndex {
    lf: MoveLf,
    locate: Option<MoveLocate>,
    text_len: u32,
    num_sequences: u32,
    #[serde(with = "crate::serde_raw::u32s")]
    slot_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_to_slot: Vec<u32>,
    dup_weights: DupWeights,
    alphabet: AlphabetFns,
}

/// Version 1 read layout: version 2 without the sequence classes.
#[derive(serde::Deserialize)]
struct OwnedSerializableMoveFmIndexV1 {
    lf: MoveLf,
    locate: Option<MoveLocate>,
    text_len: u32,
    num_sequences: u32,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    alphabet: AlphabetFns,
}

impl From<OwnedSerializableMoveFmIndexV1> for OwnedSerializableMoveFmIndex {
    /// A version 1 file predates deduplication: every sequence is stored, in its own class.
    /// The identity is sized by the headers read (bounded by the input), not by the stored
    /// `num_sequences`; the load check rejects a mismatch.
    fn from(v1: OwnedSerializableMoveFmIndexV1) -> Self {
        let seq_to_slot = (0..v1.seq_headers.len() as u32).collect();
        Self {
            lf: v1.lf,
            locate: v1.locate,
            text_len: v1.text_len,
            num_sequences: v1.num_sequences,
            slot_boundaries: v1.seq_boundaries,
            seq_headers: v1.seq_headers,
            seq_to_slot,
            dup_weights: DupWeights::default(),
            alphabet: v1.alphabet,
        }
    }
}

#[derive(serde::Serialize)]
struct SerializableBMove<'a> {
    rev: &'a MoveLf,
    rev_samples: Option<&'a RunSamples>,
}

#[derive(serde::Deserialize)]
struct OwnedSerializableBMove {
    rev: MoveLf,
    rev_samples: Option<RunSamples>,
}

fn corrupt(what: impl std::fmt::Display) -> FmIndexError {
    FmIndexError::DeserializeError(format!("serialized move index: {what}"))
}

fn encode<T: serde::Serialize>(magic: &[u8; 4], value: &T) -> Result<Vec<u8>, FmIndexError> {
    let body =
        bincode::serialize(value).map_err(|e| FmIndexError::SerializeError(e.to_string()))?;
    let mut out = Vec::with_capacity(magic.len() + body.len());
    out.extend_from_slice(magic);
    out.extend_from_slice(&body);
    Ok(out)
}

impl MoveFmIndex {
    /// Serialize to bytes: [`FORMAT_MAGIC`] then the bincode-encoded fields.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FmIndexError> {
        encode(
            &FORMAT_MAGIC,
            &SerializableMoveFmIndex {
                lf: &self.lf,
                locate: self.locate.as_ref(),
                text_len: self.text_len,
                num_sequences: self.num_sequences,
                slot_boundaries: &self.slot_boundaries,
                seq_headers: &self.seq_headers,
                seq_to_slot: self.seq_classes.seq_to_slot(),
                dup_weights: &self.dup_weights,
                alphabet: &self.alphabet_fns,
            },
        )
    }

    /// Deserialize from bytes written by [`to_bytes`](Self::to_bytes) (version 2 or 1);
    /// malformed or inconsistent input is a [`FmIndexError::DeserializeError`].
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        let d: OwnedSerializableMoveFmIndex = if let Some(body) = data.strip_prefix(&FORMAT_MAGIC) {
            bincode::deserialize(body).map_err(corrupt)?
        } else if let Some(body) = data.strip_prefix(&FORMAT_MAGIC_V1) {
            bincode::deserialize::<OwnedSerializableMoveFmIndexV1>(body)
                .map_err(corrupt)?
                .into()
        } else {
            return Err(corrupt("missing \"HMV\\x02\" format marker"));
        };
        Self::check(d)
    }

    fn check(d: OwnedSerializableMoveFmIndex) -> Result<Self, FmIndexError> {
        check_stored_alphabet(&d.alphabet)?;
        if d.lf.len() != d.text_len || d.text_len == 0 {
            return Err(corrupt("LF table length differs from the text length"));
        }
        if let Some(locate) = &d.locate {
            locate
                .validate(d.lf.num_rows(), d.text_len)
                .map_err(corrupt)?;
        }
        let seq_classes = check_sequence_tables(
            d.text_len,
            d.num_sequences,
            &d.slot_boundaries,
            &d.seq_headers,
            d.seq_to_slot,
            &d.dup_weights,
        )
        .map_err(|e| corrupt(format!("sequence metadata disagrees with the text: {e}")))?;
        let header_index = HeaderIndex::build(&d.seq_headers)?;
        Ok(Self {
            lf: d.lf,
            locate: d.locate,
            text_len: d.text_len,
            num_sequences: d.num_sequences,
            slot_boundaries: d.slot_boundaries,
            seq_headers: d.seq_headers,
            header_index,
            seq_classes,
            dup_weights: d.dup_weights,
            alphabet_fns: d.alphabet,
        })
    }
}

impl BMoveIndex {
    /// Serialize to bytes: [`BMOVE_FORMAT_MAGIC`], the forward half's byte length as a
    /// little-endian `u32`, the forward half (a complete `"HMV"` blob), then the reverse LF
    /// table and samples.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FmIndexError> {
        let fwd = self.fwd.to_bytes()?;
        let fwd_len = u32::try_from(fwd.len())
            .map_err(|_| FmIndexError::SerializeError("forward half exceeds 4 GiB".into()))?;
        let rev = bincode::serialize(&SerializableBMove {
            rev: &self.rev,
            rev_samples: self.rev_samples.as_ref(),
        })
        .map_err(|e| FmIndexError::SerializeError(e.to_string()))?;
        let mut out = Vec::with_capacity(8 + fwd.len() + rev.len());
        out.extend_from_slice(&BMOVE_FORMAT_MAGIC);
        out.extend_from_slice(&fwd_len.to_le_bytes());
        out.extend_from_slice(&fwd);
        out.extend_from_slice(&rev);
        Ok(out)
    }

    /// Deserialize from bytes written by [`to_bytes`](Self::to_bytes) (version 2 or 1).
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        let rest = data
            .strip_prefix(&BMOVE_FORMAT_MAGIC)
            .or_else(|| data.strip_prefix(&BMOVE_FORMAT_MAGIC_V1))
            .ok_or_else(|| corrupt("missing \"HBM\\x02\" format marker"))?;
        let (len_bytes, rest) = rest
            .split_first_chunk::<4>()
            .ok_or_else(|| corrupt("truncated b-move header"))?;
        let fwd_len = u32::from_le_bytes(*len_bytes) as usize;
        if fwd_len > rest.len() {
            return Err(corrupt("forward half length exceeds the blob"));
        }
        let (fwd, rev) = rest.split_at(fwd_len);
        let fwd = MoveFmIndex::from_bytes(fwd)?;
        let d: OwnedSerializableBMove = bincode::deserialize(rev).map_err(corrupt)?;
        if d.rev.len() != fwd.text_len {
            return Err(corrupt("reverse half covers a different text length"));
        }
        match (&d.rev_samples, fwd.can_locate()) {
            (Some(samples), true) => samples
                .validate(d.rev.num_rows(), fwd.text_len)
                .map_err(corrupt)?,
            (None, false) => {}
            _ => {
                return Err(corrupt(
                    "reverse samples present without locate, or missing",
                ))
            }
        }
        Ok(Self {
            fwd,
            rev: d.rev,
            rev_samples: d.rev_samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::DnaSequence;
    use crate::move_index::MoveFmIndexConfig;

    // `tests/fixtures/v1_move.hmv` / `v1_bmove.hbm` were written at `1077d09` (format
    // version 1) by `examples/gen_fixtures.rs` from this input and the default config.
    fn fixture_input() -> Vec<DnaSequence> {
        vec![
            DnaSequence::from_str_with_header("ACGTNACGTRYSWKMBDHVACGTACGTTTGCA", "chr1").unwrap(),
            DnaSequence::from_str_with_header("TTGACGTACGTNNACGGAAC", "plasmid").unwrap(),
            DnaSequence::from_str_with_header("GATTACA", "tiny").unwrap(),
        ]
    }

    #[test]
    fn reads_version_1_fixtures() {
        let config = MoveFmIndexConfig::default();
        let blob = include_bytes!("../../tests/fixtures/v1_move.hmv");
        assert_eq!(&blob[..4], &FORMAT_MAGIC_V1);
        let back = MoveFmIndex::from_bytes(blob).unwrap();
        assert!(back.seq_classes.is_identity());
        let fresh = MoveFmIndex::build_cpu(&fixture_input(), &config).unwrap();
        assert_eq!(back.to_bytes().unwrap(), fresh.to_bytes().unwrap());

        let blob = include_bytes!("../../tests/fixtures/v1_bmove.hbm");
        assert_eq!(&blob[..4], &BMOVE_FORMAT_MAGIC_V1);
        let back = BMoveIndex::from_bytes(blob).unwrap();
        let fresh = BMoveIndex::build_cpu(&fixture_input(), &config).unwrap();
        assert_eq!(back.to_bytes().unwrap(), fresh.to_bytes().unwrap());
    }

    fn duplicated() -> MoveFmIndex {
        let seqs: Vec<DnaSequence> = ["ACGTTG", "GGCA", "ACGTTG", "ACGTTG"]
            .iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect();
        MoveFmIndex::build_cpu(&seqs, &MoveFmIndexConfig::default()).unwrap()
    }

    #[test]
    fn round_trip_keeps_sequence_classes() {
        let ix = duplicated();
        assert_eq!(ix.num_distinct_sequences(), 2);
        let back = MoveFmIndex::from_bytes(&ix.to_bytes().unwrap()).unwrap();
        assert_eq!(back.seq_classes, ix.seq_classes);
        assert_eq!(back.dup_weights, ix.dup_weights);
        assert_eq!(back.to_bytes().unwrap(), ix.to_bytes().unwrap());
    }

    #[test]
    fn rejects_corrupted_sequence_classes() {
        let good = duplicated();
        let rejects = |ix: &MoveFmIndex| MoveFmIndex::from_bytes(&ix.to_bytes().unwrap()).is_err();
        assert!(!rejects(&good));

        let mut no_weights = good.clone();
        no_weights.dup_weights = DupWeights::default();
        assert!(rejects(&no_weights));

        let mut identity = good.clone();
        identity.seq_classes = crate::fm_index::seq_classes::SeqClasses::identity(4);
        assert!(rejects(&identity));
    }
}
