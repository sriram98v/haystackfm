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
        let rows: Vec<u32> = self
            .backward_search(pattern)
            .into_iter()
            .flat_map(|(lo, hi)| lo..hi)
            .collect();
        let mut positions = Vec::with_capacity(rows.len());
        self.resolve_rows_batch(&rows, &mut positions);
        positions
    }

    /// Batched [`locate_positions`](Self::locate_positions): one result per pattern, in
    /// input order, as [`FmIndex::locate_positions_many`](crate::FmIndex::locate_positions_many).
    pub fn locate_positions_many(&self, patterns: &[&[u8]]) -> Vec<Vec<u32>> {
        patterns.iter().map(|p| self.locate_positions(p)).collect()
    }

    /// Locate the suffix-array rows `lo..hi`: `(sequence id, offset)` per row, in row order.
    ///
    /// # Panics
    /// If `lo..hi` is not within `0..text_len` (or on a count-only index).
    pub fn locate_rows(&self, lo: u32, hi: u32) -> Vec<(SeqId, u32)> {
        assert!(
            lo <= hi && hi <= self.text_len,
            "locate_rows: rows {lo}..{hi} outside 0..{}",
            self.text_len
        );
        let rows: Vec<u32> = (lo..hi).collect();
        self.locate_row_list(&rows)
    }

    /// Locate each listed suffix-array row: `(sequence id, offset)` per row, in order.
    pub(crate) fn locate_row_list(&self, rows: &[u32]) -> Vec<(SeqId, u32)> {
        debug_assert!(rows.iter().all(|&row| row < self.text_len));
        let mut positions = Vec::with_capacity(rows.len());
        self.resolve_rows_batch(rows, &mut positions);
        positions
            .into_iter()
            .map(|pos| {
                self.map_position(pos)
                    .expect("resolved SA position must be within text bounds")
            })
            .collect()
    }

    /// Text positions of SA rows: LF-walk each row until a sampled row (every sequence start
    /// is sampled, so no walk crosses a sentinel), with the walks in lockstep (as
    /// `FmIndex::resolve_sa_batch`). A run-length LF step chases two lines (directory
    /// entry, then run block), so each round prefetches in two stages before any lane
    /// reads: every lane's sample record and directory entry, then every lane's block.
    /// Results land in `out` in `rows` order.
    pub(crate) fn resolve_rows_batch(&self, rows: &[u32], out: &mut Vec<u32>) {
        let samples = self
            .sa_samples
            .as_ref()
            .expect("locate on a count-only RlFmIndex (built with sa_sample_rate = 0)");
        // A sample sits within `sample_rate` text positions of every row (sequence starts
        // are forced), so a longer walk means a corrupted sample set.
        let max_steps = samples.sample_rate.clamp(1, self.text_len.max(1));
        out.clear();
        out.resize(rows.len(), 0);
        let mut cur: Vec<u32> = rows.to_vec();
        let mut steps: Vec<u32> = vec![0; rows.len()];
        let mut active: Vec<u32> = (0..rows.len() as u32).collect();
        let mut next_active: Vec<u32> = Vec::with_capacity(active.len());
        while !active.is_empty() {
            for &lane in &active {
                let row = cur[lane as usize];
                samples.prefetch(row);
                self.bwt.prefetch_bucket(row);
            }
            for &lane in &active {
                self.bwt.prefetch_block(cur[lane as usize]);
            }
            next_active.clear();
            for &lane in &active {
                let idx = lane as usize;
                let row = cur[idx];
                if let Some(sa) = samples.get(row) {
                    out[idx] = sa
                        .checked_add(steps[idx])
                        .expect("SA sample out of range: corrupted index");
                } else {
                    let (c, rank) = self.bwt.lf_step(row);
                    cur[idx] = self.c_array.get(c) + rank;
                    steps[idx] += 1;
                    assert!(
                        steps[idx] < max_steps,
                        "SA samples never reached: corrupted index"
                    );
                    next_active.push(lane);
                }
            }
            std::mem::swap(&mut active, &mut next_active);
        }
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
}
