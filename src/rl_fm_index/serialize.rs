//! Serialization of [`RlFmIndex`] / [`BidirRlFmIndex`] via `bincode`.
//!
//! [`FORMAT_MAGIC`] (`"HRL\x01"`) followed by the bincode encoding of the index fields; the
//! run-length BWT's vectors are little-endian byte blobs (`crate::serde_raw`). Derived state
//! (header map, bucket directory, occ select hints) is rebuilt at load, and the run
//! structure is validated before any query can touch it.
//!
//! [`BIDIR_FORMAT_MAGIC`] (`"HRB\x01"`), a little-endian `u32` byte length of the forward
//! half, then the forward and reverse halves, each a complete `"HRL"` blob.

use super::{BidirRlFmIndex, RlFmIndex};
use crate::alphabet::AlphabetFns;
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::fm_index::lookup::LookupTable;
use crate::fm_index::seq_id::HeaderIndex;
use crate::fm_index::serialize::check_stored_alphabet;
use crate::occ::BwtRank;
use crate::rlbwt::RlBwt;
use crate::suffix_array::SampledSuffixArray;

/// Leading bytes of a serialized [`RlFmIndex`]: `"HRL"` plus the format version.
pub const FORMAT_MAGIC: [u8; 4] = *b"HRL\x01";
/// Leading bytes of a serialized [`BidirRlFmIndex`]: `"HRB"` plus the format version.
pub const BIDIR_FORMAT_MAGIC: [u8; 4] = *b"HRB\x01";

/// Write layout. Field order is the wire order.
#[derive(serde::Serialize)]
struct SerializableRlFmIndex<'a> {
    c_array: &'a CArray,
    bwt: &'a RlBwt,
    sa_samples: Option<&'a SampledSuffixArray>,
    text_len: u32,
    num_sequences: u32,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_boundaries: &'a [u32],
    seq_headers: &'a [String],
    alphabet: &'a AlphabetFns,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_end_rows: &'a [u32],
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
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    alphabet: AlphabetFns,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_end_rows: Vec<u32>,
    lookup: Option<LookupTable>,
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
            seq_boundaries: &self.seq_boundaries,
            seq_headers: &self.seq_headers,
            alphabet: &self.alphabet_fns,
            seq_end_rows: &self.seq_end_rows,
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
    /// Returns [`FmIndexError::DeserializeError`] for a wrong magic or version, a truncated
    /// or malformed body, an inconsistent run structure, or an alphabet whose built-in tag
    /// does not match its tables.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        let body = data
            .strip_prefix(&FORMAT_MAGIC)
            .ok_or_else(|| corrupt("missing \"HRL\\x01\" format marker"))?;
        let d: OwnedSerializableRlFmIndex = bincode::deserialize(body).map_err(corrupt)?;
        check_stored_alphabet(&d.alphabet)?;
        let mut bwt = d.bwt;
        bwt.validate_and_rebuild().map_err(corrupt)?;
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
        if d.seq_boundaries.len() != d.num_sequences as usize
            || d.seq_headers.len() != d.num_sequences as usize
            || d.seq_boundaries.last().copied().unwrap_or(0) != d.text_len
            || d.seq_boundaries.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(corrupt("sequence metadata disagrees with the text"));
        }
        // Empty (reverse half) or one sentinel row per sequence, each row distinct and a
        // sentinel row (`0..num_sequences`). A sequence's row must not hold a sentinel in
        // the BWT unless the sequence is empty, or extraction would walk off its start.
        if !d.seq_end_rows.is_empty() {
            if d.seq_end_rows.len() != d.num_sequences as usize {
                return Err(corrupt(
                    "sequence end rows disagree with the sequence count",
                ));
            }
            let mut seen = vec![false; d.num_sequences as usize];
            for (k, &row) in d.seq_end_rows.iter().enumerate() {
                let start = k.checked_sub(1).map_or(0, |j| d.seq_boundaries[j]);
                let empty = d.seq_boundaries.get(k).is_some_and(|&end| end - start == 1);
                if row >= d.num_sequences
                    || std::mem::replace(&mut seen[row as usize], true)
                    || (bwt.symbol_at(row) == crate::alphabet::SENTINEL) != empty
                {
                    return Err(corrupt("sequence end rows disagree with the BWT"));
                }
            }
        }
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
            seq_boundaries: d.seq_boundaries,
            seq_headers: d.seq_headers,
            header_index,
            alphabet_fns: d.alphabet,
            seq_end_rows: d.seq_end_rows,
            lookup: d.lookup,
        })
    }
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

    /// Deserialize from bytes written by [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        let rest = data
            .strip_prefix(&BIDIR_FORMAT_MAGIC)
            .ok_or_else(|| corrupt("missing \"HRB\\x01\" format marker"))?;
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
        if fwd.text_len != rev.text_len || fwd.alphabet_fns != rev.alphabet_fns {
            return Err(corrupt("forward and reverse halves do not match"));
        }
        Ok(Self { fwd, rev })
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
        assert_eq!(back.seq_end_rows, ix.seq_end_rows);
        assert_eq!(back.lookup, ix.lookup);
    }

    #[test]
    fn rejects_corrupted_sequence_end_rows() {
        let good = index(0);
        assert!(!rejects(&good));

        let mut dup = good.clone();
        dup.seq_end_rows[1] = dup.seq_end_rows[0];
        assert!(rejects(&dup), "duplicate row");

        let mut out_of_range = good.clone();
        out_of_range.seq_end_rows[0] = good.num_sequences;
        assert!(rejects(&out_of_range), "non-sentinel row");

        let mut short = good.clone();
        short.seq_end_rows.pop();
        assert!(rejects(&short), "wrong count");
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
