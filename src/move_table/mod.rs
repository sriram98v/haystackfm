//! Move tables (Nishimoto & Tabei, ICALP 2021): a permutation of `[0, n)` that maps runs of
//! consecutive positions to runs of consecutive positions, stored as one row per run.
//!
//! Row `k` maps its input interval `[p_k, p_{k+1})` to the output interval starting at
//! `q_k`, and records `idx_k`, the row whose input interval contains `q_k`. Applying the
//! permutation to a position `i` whose row `k` is known is then
//!
//! ```text
//! i' = q_k + (i − p_k)
//! k' = idx_k, then fast-forward k' while p_{k'+1} <= i'
//! ```
//!
//! i.e. one row read plus a short forward scan instead of a rank query, and the result
//! carries its own row for the next step. The BWT's LF mapping and the r-index's `φ` /
//! `φ⁻¹` are all such permutations (see [`crate::move_index`]).
//!
//! Construction can *balance* the table (split input intervals until no output
//! interval contains `2d` or more input starts), which bounds the fast-forward to `< 2d`
//! steps at the cost of a few extra rows.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One move-table row. Row-major and 16 bytes, so a step touches one cache line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub(crate) struct MoveRow {
    /// First input position of the row.
    pub p: u32,
    /// Image of `p`.
    pub q: u32,
    /// Row whose input interval contains `q`.
    pub idx: u32,
    /// Payload symbol (the BWT run head for an LF table; unused otherwise).
    pub head: u8,
}

/// A move table over a permutation of `[0, n)`. See the [module docs](self).
///
/// Serialized as `n` plus four little-endian `u32`s per row; deserialization checks that the
/// rows describe a permutation (inputs and outputs both tile `[0, n)`, every `idx` is the
/// row containing its `q`), so a loaded table never indexes out of bounds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "MoveTableRepr", into = "MoveTableRepr")]
pub struct MoveTable {
    /// Domain size.
    n: u32,
    /// The rows plus one trailing sentinel row with `p = n`, so `rows[k + 1].p` is always
    /// readable for a real row `k`.
    rows: Vec<MoveRow>,
}

impl MoveTable {
    /// Build from runs `(p, q, head)` sorted by strictly increasing `p`, starting at `0`,
    /// where run `k` covers `[p_k, p_{k+1})` (the last run ends at `n`) and maps it onto
    /// `[q_k, q_k + len_k)`. The outputs must tile `[0, n)` too.
    ///
    /// `balance = Some(d)` (`d >= 1`) splits input intervals until every output interval
    /// holds fewer than `2d` input starts, bounding [`step`](Self::step)'s fast-forward.
    ///
    /// # Panics
    /// If the runs do not describe a permutation of `[0, n)`.
    pub(crate) fn build(runs: &[(u32, u32, u8)], n: u32, balance: Option<u32>) -> Self {
        let runs = match balance {
            Some(d) => balance_runs(runs, n, d.max(1)),
            None => runs.to_vec(),
        };
        let table = Self::from_runs_unchecked(&runs, n);
        if let Err(e) = table.validate() {
            panic!("MoveTable::build: {e}");
        }
        table
    }

    /// Rows with `idx` filled in by a sweep over the outputs sorted by `q`.
    fn from_runs_unchecked(runs: &[(u32, u32, u8)], n: u32) -> Self {
        let mut rows: Vec<MoveRow> = runs
            .iter()
            .map(|&(p, q, head)| MoveRow { p, q, idx: 0, head })
            .collect();
        let mut by_q: Vec<usize> = (0..rows.len()).collect();
        by_q.sort_unstable_by_key(|&k| rows[k].q);
        let mut k = 0usize;
        for &r in &by_q {
            let q = rows[r].q;
            while k + 1 < rows.len() && rows[k + 1].p <= q {
                k += 1;
            }
            rows[r].idx = k as u32;
        }
        rows.push(MoveRow {
            p: n,
            q: n,
            idx: rows.len().saturating_sub(1) as u32,
            head: 0,
        });
        Self { n, rows }
    }

    /// Domain size `n`.
    pub fn domain(&self) -> u32 {
        self.n
    }

    /// Number of rows (excluding the internal sentinel).
    pub fn num_rows(&self) -> u32 {
        self.rows.len() as u32 - 1
    }

    /// Heap bytes held by the rows.
    pub fn heap_size_bytes(&self) -> usize {
        self.rows.len() * std::mem::size_of::<MoveRow>()
    }

    /// Row `k` (a real row, `k < num_rows()`).
    #[inline]
    pub(crate) fn row(&self, k: usize) -> MoveRow {
        self.rows[k]
    }

    /// First input position of row `k`; `n` for `k == num_rows()`.
    #[inline]
    pub(crate) fn start(&self, k: usize) -> u32 {
        self.rows[k].p
    }

    /// Last input position of row `k`.
    #[inline]
    pub(crate) fn end(&self, k: usize) -> u32 {
        self.rows[k + 1].p - 1
    }

    /// Payload symbol of row `k`.
    #[inline]
    pub(crate) fn head(&self, k: usize) -> u8 {
        self.rows[k].head
    }

    /// Image of `i`, which lies in row `k`, without resolving the image's row.
    #[inline]
    pub(crate) fn map(&self, i: u32, k: usize) -> u32 {
        let row = self.rows[k];
        row.q + (i - row.p)
    }

    /// Apply the permutation to `i`, which lies in row `k`: returns the image and its row.
    #[inline]
    pub(crate) fn step(&self, i: u32, k: usize) -> (u32, usize) {
        debug_assert!(
            self.rows[k].p <= i && i < self.rows[k + 1].p,
            "step: {i} not in row {k}"
        );
        let row = self.rows[k];
        let image = row.q + (i - row.p);
        let mut j = row.idx as usize;
        while self.rows[j + 1].p <= image {
            j += 1;
        }
        (image, j)
    }

    /// Row containing position `i < n` (binary search).
    pub(crate) fn row_of(&self, i: u32) -> usize {
        debug_assert!(i < self.n, "row_of({i}) outside domain {}", self.n);
        let real = &self.rows[..self.rows.len() - 1];
        real.partition_point(|row| row.p <= i).saturating_sub(1)
    }

    /// Row containing `i`, searching forward from row `from` (which must start at or before
    /// `i`): a short scan, then a binary search over the rest. Stops before the sentinel
    /// row because its start `n` exceeds every `i < n`.
    #[inline]
    pub(crate) fn row_from(&self, i: u32, from: usize) -> usize {
        let mut k = from;
        for _ in 0..4 {
            if self.rows[k + 1].p > i {
                return k;
            }
            k += 1;
        }
        let real = &self.rows[k..self.rows.len() - 1];
        k + real.partition_point(|row| row.p <= i).saturating_sub(1)
    }

    /// Row containing `i`, searching backward from row `from` (which must end at or after
    /// `i`). Never passes row 0, whose start is `0 <= i` (checked by validation).
    #[inline]
    pub(crate) fn row_back_from(&self, i: u32, from: usize) -> usize {
        let mut k = from;
        for _ in 0..4 {
            if self.rows[k].p <= i {
                return k;
            }
            k -= 1;
        }
        self.rows[..=k].partition_point(|row| row.p <= i) - 1
    }

    /// Check that the rows describe a permutation of `[0, n)` with correct `idx` values.
    fn validate(&self) -> Result<(), String> {
        let rows = &self.rows;
        let Some(last) = rows.last() else {
            return Err("move table has no sentinel row".into());
        };
        if last.p != self.n {
            return Err("move table does not end at its domain size".into());
        }
        let real = &rows[..rows.len() - 1];
        if self.n > 0 && (real.is_empty() || real[0].p != 0) {
            return Err("move table does not start at position 0".into());
        }
        if self.n == 0 && !real.is_empty() {
            return Err("empty move table has rows".into());
        }
        if rows.windows(2).any(|w| w[0].p >= w[1].p) {
            return Err("move table inputs are not strictly increasing".into());
        }
        // Outputs must tile [0, n): sort by q and check contiguity.
        let mut outputs: Vec<(u32, u32)> = (0..real.len())
            .map(|k| (real[k].q, rows[k + 1].p - real[k].p))
            .collect();
        outputs.sort_unstable();
        let mut next = 0u64;
        for &(q, len) in &outputs {
            if q as u64 != next {
                return Err("move table outputs do not tile the domain".into());
            }
            next += len as u64;
        }
        if next != self.n as u64 {
            return Err("move table outputs do not cover the domain".into());
        }
        for row in real {
            if row.idx as usize >= real.len() {
                return Err("move table row index out of range".into());
            }
            let k = row.idx as usize;
            if !(rows[k].p <= row.q && row.q < rows[k + 1].p) {
                return Err("move table row index does not contain its image".into());
            }
        }
        Ok(())
    }
}

/// Split input intervals until no output interval holds `2d` or more input starts
/// (Nishimoto & Tabei's balancing). Each split of input interval `[p, p+len) → q` at offset
/// `t` puts a new input start at `p + t` (which may overload the output interval containing
/// it, so that one is rechecked) and splits the output at `q + t`.
fn balance_runs(runs: &[(u32, u32, u8)], n: u32, d: u32) -> Vec<(u32, u32, u8)> {
    // input start -> (len, q, head)
    let mut inputs: BTreeMap<u32, (u32, u32, u8)> = BTreeMap::new();
    for (k, &(p, q, head)) in runs.iter().enumerate() {
        let end = runs.get(k + 1).map_or(n, |r| r.0);
        inputs.insert(p, (end - p, q, head));
    }
    // output start -> input start
    let mut outputs: BTreeMap<u32, u32> = inputs.iter().map(|(&p, &(_, q, _))| (q, p)).collect();
    let mut work: Vec<u32> = inputs.keys().copied().collect();
    let limit = 2 * d as usize;
    while let Some(p) = work.pop() {
        let Some(&(len, q, head)) = inputs.get(&p) else {
            continue;
        };
        // Input starts inside the output interval, in order. With at least 2d of them,
        // split at the (d+1)-th so the first part keeps d; it lies strictly after q.
        let starts: Vec<u32> = inputs
            .range(q..q + len)
            .map(|(&s, _)| s)
            .take(limit)
            .collect();
        if starts.len() < limit {
            continue;
        }
        let split_at = starts[d as usize];
        let t = split_at - q;
        inputs.insert(p, (t, q, head));
        inputs.insert(p + t, (len - t, q + t, head));
        outputs.insert(q + t, p + t);
        work.push(p);
        work.push(p + t);
        // The new input start p + t may overload the output interval containing it.
        if let Some((_, &owner)) = outputs.range(..=p + t).next_back() {
            work.push(owner);
        }
    }
    inputs
        .into_iter()
        .map(|(p, (_, q, head))| (p, q, head))
        .collect()
}

/// Serialized layout of a [`MoveTable`]: the domain size and `(p, q, idx, head)` per row
/// (sentinel row excluded) as little-endian `u32`s.
#[derive(Serialize, Deserialize)]
struct MoveTableRepr {
    n: u32,
    #[serde(with = "crate::serde_raw::u32s")]
    words: Vec<u32>,
}

impl From<MoveTable> for MoveTableRepr {
    fn from(t: MoveTable) -> Self {
        let real = &t.rows[..t.rows.len() - 1];
        let mut words = Vec::with_capacity(real.len() * 4);
        for row in real {
            words.extend_from_slice(&[row.p, row.q, row.idx, row.head as u32]);
        }
        Self { n: t.n, words }
    }
}

impl TryFrom<MoveTableRepr> for MoveTable {
    type Error = String;

    fn try_from(r: MoveTableRepr) -> Result<Self, String> {
        if !r.words.len().is_multiple_of(4) {
            return Err("move table blob is not a whole number of rows".into());
        }
        let mut rows = Vec::with_capacity(r.words.len() / 4 + 1);
        for w in r.words.chunks_exact(4) {
            let head = u8::try_from(w[3]).map_err(|_| "move table head out of range")?;
            rows.push(MoveRow {
                p: w[0],
                q: w[1],
                idx: w[2],
                head,
            });
        }
        rows.push(MoveRow {
            p: r.n,
            q: r.n,
            idx: rows.len().saturating_sub(1) as u32,
            head: 0,
        });
        let table = Self { n: r.n, rows };
        table.validate()?;
        Ok(table)
    }
}

#[cfg(test)]
mod tests;
