//! Serialization of [`MoveFmIndex`] (`"HMV\x01"`) and [`BMoveIndex`] (`"HBM\x01"`) via
//! `bincode`. Move tables validate themselves on load (a permutation of `[0, n)` with
//! correct row indices), the LF table is checked against the C array, and the locate
//! structures against the table sizes, so no loaded value reaches an out-of-bounds read.

use super::lf::MoveLf;
use super::locate::{MoveLocate, RunSamples};
use super::{BMoveIndex, MoveFmIndex};
use crate::alphabet::AlphabetFns;
use crate::error::FmIndexError;
use crate::fm_index::seq_id::HeaderIndex;
use crate::fm_index::serialize::check_stored_alphabet;

/// Leading bytes of a serialized [`MoveFmIndex`].
pub const FORMAT_MAGIC: [u8; 4] = *b"HMV\x01";
/// Leading bytes of a serialized [`BMoveIndex`].
pub const BMOVE_FORMAT_MAGIC: [u8; 4] = *b"HBM\x01";

#[derive(serde::Serialize)]
struct SerializableMoveFmIndex<'a> {
    lf: &'a MoveLf,
    locate: Option<&'a MoveLocate>,
    text_len: u32,
    num_sequences: u32,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_boundaries: &'a [u32],
    seq_headers: &'a [String],
    alphabet: &'a AlphabetFns,
}

#[derive(serde::Deserialize)]
struct OwnedSerializableMoveFmIndex {
    lf: MoveLf,
    locate: Option<MoveLocate>,
    text_len: u32,
    num_sequences: u32,
    #[serde(with = "crate::serde_raw::u32s")]
    seq_boundaries: Vec<u32>,
    seq_headers: Vec<String>,
    alphabet: AlphabetFns,
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
                seq_boundaries: &self.seq_boundaries,
                seq_headers: &self.seq_headers,
                alphabet: &self.alphabet_fns,
            },
        )
    }

    /// Deserialize from bytes written by [`to_bytes`](Self::to_bytes); malformed or
    /// inconsistent input is a [`FmIndexError::DeserializeError`].
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        let body = data
            .strip_prefix(&FORMAT_MAGIC)
            .ok_or_else(|| corrupt("missing \"HMV\\x01\" format marker"))?;
        Self::decode(body)
    }

    fn decode(body: &[u8]) -> Result<Self, FmIndexError> {
        let d: OwnedSerializableMoveFmIndex = bincode::deserialize(body).map_err(corrupt)?;
        check_stored_alphabet(&d.alphabet)?;
        if d.lf.len() != d.text_len || d.text_len == 0 {
            return Err(corrupt("LF table length differs from the text length"));
        }
        if let Some(locate) = &d.locate {
            locate
                .validate(d.lf.num_rows(), d.text_len)
                .map_err(corrupt)?;
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
            lf: d.lf,
            locate: d.locate,
            text_len: d.text_len,
            num_sequences: d.num_sequences,
            seq_boundaries: d.seq_boundaries,
            seq_headers: d.seq_headers,
            header_index,
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

    /// Deserialize from bytes written by [`to_bytes`](Self::to_bytes). Rebuilds the SMEM /
    /// MEM sweeps' k-mer cache (at most 40 MiB), which is not stored.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        let rest = data
            .strip_prefix(&BMOVE_FORMAT_MAGIC)
            .ok_or_else(|| corrupt("missing \"HBM\\x01\" format marker"))?;
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
        // The reversed text holds the same symbols: halves of different indexes would
        // answer with wrong intervals.
        if d.rev.c_array != fwd.lf.c_array {
            return Err(corrupt("reverse half indexes a different text"));
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
        Ok(Self::from_parts(fwd, d.rev, d.rev_samples))
    }
}
