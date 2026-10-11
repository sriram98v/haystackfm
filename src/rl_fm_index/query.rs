//! `count` / `locate` on the run-length FM-index: backward search with IUPAC fan-out over
//! [`BwtRank`], then LF-walks to the nearest SA sample.

use super::RlFmIndex;
use crate::fm_index::lookup::seed_backward_search;
use crate::fm_index::query::merge_intervals_inplace;
use crate::fm_index::seq_id::{map_text_position, SeqId};
use crate::occ::BwtRank;

impl RlFmIndex {
    /// Count occurrences of `pattern` (alphabet codes). Query and reference codes match
    /// under the index's alphabet, as in [`FmIndex::count`](crate::FmIndex::count).
    pub fn count(&self, pattern: &[u8]) -> u32 {
        self.backward_search(pattern)
            .iter()
            .map(|(lo, hi)| hi - lo)
            .sum()
    }

    /// Locate every occurrence of `pattern` as `(sequence id, offset in that sequence)`.
    ///
    /// # Panics
    /// If the index is count-only (see [`can_locate`](Self::can_locate)); likewise for the
    /// other `locate*` methods.
    pub fn locate(&self, pattern: &[u8]) -> Vec<(SeqId, u32)> {
        self.locate_positions(pattern)
            .into_iter()
            .map(|pos| {
                self.map_position(pos)
                    .expect("resolved SA position must be within text bounds")
            })
            .collect()
    }

    /// Locate every occurrence of `pattern` as absolute offsets into the concatenated text.
    pub fn locate_positions(&self, pattern: &[u8]) -> Vec<u32> {
        self.backward_search(pattern)
            .into_iter()
            .flat_map(|(lo, hi)| lo..hi)
            .map(|row| self.resolve_row(row))
            .collect()
    }

    /// Batched [`locate_positions`](Self::locate_positions): one result per pattern, in
    /// input order, as [`FmIndex::locate_positions_many`](crate::FmIndex::locate_positions_many).
    pub fn locate_positions_many(&self, patterns: &[&[u8]]) -> Vec<Vec<u32>> {
        patterns.iter().map(|p| self.locate_positions(p)).collect()
    }

    /// Locate the suffix-array rows `lo..hi`: `(sequence id, offset)` per row, in row order.
    pub fn locate_rows(&self, lo: u32, hi: u32) -> Vec<(SeqId, u32)> {
        (lo..hi)
            .map(|row| {
                self.map_position(self.resolve_row(row))
                    .expect("resolved SA position must be within text bounds")
            })
            .collect()
    }

    /// Map an absolute text offset to `(sequence id, offset within it)`.
    pub fn map_position(&self, text_pos: u32) -> Option<(SeqId, u32)> {
        map_text_position(&self.seq_boundaries, text_pos)
    }

    /// SA intervals of every text substring matching `pattern`, disjoint and sorted.
    pub(crate) fn backward_search(&self, pattern: &[u8]) -> Vec<(u32, u32)> {
        if pattern.is_empty() {
            return vec![(0u32, self.text_len)];
        }
        let (mut intervals, rest) = match seed_backward_search(self.lookup.as_ref(), pattern) {
            Some((seed, _)) if seed.is_empty() => return vec![],
            Some(seeded) => seeded,
            None => (vec![(0u32, self.text_len)], pattern.len()),
        };
        let mut next: Vec<(u32, u32)> = Vec::with_capacity(16);
        let present = self.bwt.present_symbols();
        for &q in pattern[..rest].iter().rev() {
            let compat = self.alphabet_fns.compatible(q) & present;
            next.clear();
            for &(lo, hi) in &intervals {
                for c in compat.iter() {
                    let c_val = self.c_array.get(c);
                    let (r_lo, r_hi) = self.bwt.rank_pair(c, lo, hi);
                    if r_lo < r_hi {
                        next.push((c_val + r_lo, c_val + r_hi));
                    }
                }
            }
            if next.len() > 1 {
                merge_intervals_inplace(&mut next);
            }
            if next.is_empty() {
                return vec![];
            }
            std::mem::swap(&mut intervals, &mut next);
        }
        intervals
    }

    /// Text position of SA row `row`: LF-walk until a sampled row. Every sequence start is
    /// sampled, so the walk never crosses a sentinel.
    pub(crate) fn resolve_row(&self, mut row: u32) -> u32 {
        let samples = self
            .sa_samples
            .as_ref()
            .expect("locate on a count-only RlFmIndex (built with sa_sample_rate = 0)");
        let mut steps = 0u32;
        loop {
            if let Some(sa) = samples.get(row) {
                return sa + steps;
            }
            let (c, rank) = self.bwt.lf_step(row);
            row = self.c_array.get(c) + rank;
            steps += 1;
            // Every LF cycle holds a sampled sequence start; only a corrupted sample set
            // could walk longer than the text.
            assert!(
                steps <= self.text_len,
                "SA samples never reached: corrupted index"
            );
        }
    }
}
