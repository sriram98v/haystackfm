//! Stable integer identifiers for indexed reference sequences.

use std::collections::HashMap;
use std::fmt;

use crate::error::FmIndexError;

/// Identifier of a reference sequence within an index.
///
/// Assigned in build order — the first sequence passed to the builder is `SeqId(0)` — and
/// preserved across `to_bytes` / `from_bytes`, so a caller can build an `id -> label` table
/// once at load time and index it by [`index`](Self::index) thereafter.
///
/// Every query that reports a match location identifies the sequence with a `SeqId` rather
/// than its header, so no string is allocated per occurrence. Resolve one to its FASTA
/// header with `seq_header`, and go the other way with `seq_id`; both are O(1).
///
/// The wrapper exists so an id can never be confused with the offset it is paired with —
/// both are `u32`, and `(SeqId, u32)` position tuples would otherwise be trivial to swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SeqId(pub u32);

impl SeqId {
    /// Wrap a raw 0-based sequence number.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The raw 0-based sequence number.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The id as a `usize`, for indexing into `seq_headers()` or a caller's own table.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for SeqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for SeqId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<SeqId> for u32 {
    fn from(id: SeqId) -> Self {
        id.0
    }
}

impl From<SeqId> for usize {
    fn from(id: SeqId) -> Self {
        id.0 as usize
    }
}

/// Map an absolute offset in the concatenated text to `(sequence id, offset within it)`,
/// given each sequence's end (its trailing sentinel included). `None` past the text end.
pub(crate) fn map_text_position(seq_boundaries: &[u32], text_pos: u32) -> Option<(SeqId, u32)> {
    let seq_idx = seq_boundaries.partition_point(|&boundary| boundary <= text_pos);
    if seq_idx >= seq_boundaries.len() {
        return None;
    }
    let seq_start = if seq_idx == 0 {
        0
    } else {
        seq_boundaries[seq_idx - 1]
    };
    Some((SeqId::new(seq_idx as u32), text_pos - seq_start))
}

/// O(1) header -> [`SeqId`] lookup, built alongside the index's header list.
///
/// Not serialized: it is a pure function of `seq_headers`, so it is rebuilt on
/// deserialization rather than stored.
#[derive(Debug, Clone, Default)]
pub(crate) struct HeaderIndex {
    by_header: HashMap<Box<str>, u32>,
}

impl HeaderIndex {
    /// Build from the index's headers, in build order.
    ///
    /// Returns [`FmIndexError::DuplicateHeader`] when two sequences share a header, which
    /// would otherwise make the header -> id direction ambiguous. Sequences built without an
    /// explicit header are given a distinct `seq_{i}` name before this point, so this only
    /// fires on genuinely repeated names.
    pub(crate) fn build(headers: &[String]) -> Result<Self, FmIndexError> {
        let mut by_header: HashMap<Box<str>, u32> = HashMap::with_capacity(headers.len());
        for (i, header) in headers.iter().enumerate() {
            if by_header.insert(header.as_str().into(), i as u32).is_some() {
                return Err(FmIndexError::DuplicateHeader(header.clone()));
            }
        }
        Ok(Self { by_header })
    }

    /// Id for a header, or `None` when no sequence carries it. O(1).
    pub(crate) fn get(&self, header: &str) -> Option<SeqId> {
        self.by_header.get(header).copied().map(SeqId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seq_id_conversions_round_trip() {
        let id = SeqId::new(7);
        assert_eq!(id.get(), 7);
        assert_eq!(id.index(), 7usize);
        assert_eq!(u32::from(id), 7);
        assert_eq!(usize::from(id), 7usize);
        assert_eq!(SeqId::from(7u32), id);
        assert_eq!(id.to_string(), "7");
    }

    #[test]
    fn header_index_maps_headers_to_build_order() {
        let headers = vec!["chr1".to_string(), "chr2".to_string(), "chrX".to_string()];
        let index = HeaderIndex::build(&headers).unwrap();
        assert_eq!(index.get("chr1"), Some(SeqId(0)));
        assert_eq!(index.get("chr2"), Some(SeqId(1)));
        assert_eq!(index.get("chrX"), Some(SeqId(2)));
        assert_eq!(index.get("chrY"), None);
        assert_eq!(index.get(""), None);
    }

    #[test]
    fn header_index_rejects_duplicates() {
        let headers = vec!["chr1".to_string(), "chr2".to_string(), "chr1".to_string()];
        let err = HeaderIndex::build(&headers).unwrap_err();
        assert!(
            matches!(&err, FmIndexError::DuplicateHeader(h) if h == "chr1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn header_index_of_empty_header_list_is_empty() {
        let index = HeaderIndex::build(&[]).unwrap();
        assert_eq!(index.get("anything"), None);
    }
}
