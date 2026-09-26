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
use crate::fm_index::seq_id::HeaderIndex;
use crate::fm_index::serialize::check_stored_alphabet;
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
