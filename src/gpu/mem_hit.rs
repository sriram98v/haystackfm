/// A Maximal Exact Match with GPU-resolved reference positions.
///
/// Returned by [`crate::BidirFmIndex::find_smems_gpu`] and [`crate::BidirFmIndex::find_mems_gpu`].
/// Positions are resolved via the GPU SA resolve + reference boundary mapping passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemHit {
    /// Start position in the query (0-based, inclusive).
    pub query_start: u32,
    /// End position in the query (0-based, exclusive).
    pub query_end: u32,
    /// Number of occurrences in the reference (SA interval size).
    pub match_count: u32,
    /// Reference positions: `(ref_id, offset_within_ref)`.
    /// At most `max_hits_per_mem` entries (default 1024).
    /// `ref_id` is the 0-based index into the reference list passed at index build time.
    pub positions: Vec<(u32, u32)>,
    /// True if the hit list was truncated due to `max_hits_per_mem`.
    pub truncated: bool,
}
