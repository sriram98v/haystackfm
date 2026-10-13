//! SMEM / MEM finding ([`Mem`]): one sweep shared by every bidirectional index (`mod.rs`),
//! its k-mer cache and child fan-out (`fanout.rs`), row-level locate and the occ-table
//! backend (`rows.rs`).

use crate::alphabet::{AlphabetFns, ALPHABET_SIZE};
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::bidir_index::BidirFmIndex;
use crate::fm_index::seq_id::SeqId;

mod fanout;
#[cfg(test)]
mod oracle;
mod rows;
#[cfg(test)]
mod tests;

use fanout::extend_set;
pub(crate) use fanout::{Fanout, ShallowCache};
pub(crate) use rows::RowLocate;

/// A Maximal Exact Match (MEM) between a query and the indexed reference.
///
/// Maximality is judged **per occurrence**, as in MUMmer and BWA. A query interval `[s,e)`
/// is a MEM when at least one of its occurrences satisfies both:
/// - **left-maximal at that occurrence**: `s == 0`, the occurrence sits at a reference start,
///   or the preceding reference symbol is incompatible with `query[s-1]`;
/// - **right-maximal at that occurrence**: `e == query.len()`, the occurrence sits at a
///   reference end, or the following reference symbol is incompatible with `query[e]`.
///
/// A Super-Maximal Exact Match (SMEM) is a MEM whose query interval is not contained in that
/// of any other MEM. So [`BidirFmIndex::find_smems`] returns a subset of
/// [`BidirFmIndex::find_mems`].
///
/// Note the two constructors differ in what they count: `find_mems` reports only the maximal
/// occurrences of a match in `match_count` and `positions`, whereas for a `find_smems` result
/// every occurrence is maximal anyway, so the two coincide there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mem {
    /// Start position in the query (0-based, inclusive).
    pub query_start: usize,
    /// End position in the query (0-based, exclusive).
    pub query_end: usize,
    /// Number of occurrences in the reference text.
    pub match_count: u32,
    /// Reference positions (populated only when `locate = true`).
    ///
    /// Each entry is `(sequence_id, position_within_sequence)`. Resolve a [`SeqId`] to its
    /// FASTA header with [`BidirFmIndex::seq_header`] — no string is allocated per
    /// occurrence. Same shape as `MemHit::positions` on the GPU path (feature `gpu`), so
    /// CPU and GPU results can be consumed by the same code.
    pub positions: Vec<(SeqId, u32)>,
}

impl Mem {
    /// Length of the matched pattern in the query.
    pub fn len(&self) -> usize {
        self.query_end - self.query_start
    }

    pub fn is_empty(&self) -> bool {
        self.query_start >= self.query_end
    }
}

impl BidirFmIndex {
    /// Find all Super-Maximal Exact Matches (SMEMs) between `query` and the
    /// indexed reference.
    ///
    /// # Algorithm
    ///
    /// For each query position `i` (0 .. query.len()):
    /// 1. **Right extension**: start from `i`, extend right one base at a time
    ///    via [`BidirInterval::extend_right`] until the interval collapses or
    ///    the query ends.  This yields the unique right-maximal match `[i, j)`.
    /// 2. **Left-maximality check**: try extending the resulting interval one
    ///    step to the left by `query[i-1]`.  If that extension is still
    ///    non-empty, `[i, j)` can be extended to the left → not left-maximal →
    ///    skip it.
    /// 3. Accept seeds that are ≥ `min_len` and both left- and right-maximal.
    ///
    /// Complexity: O(|query|² × α) where α = [`ALPHABET_SIZE`].
    /// In practice much better: once a long SMEM is found the inner loop
    /// advances to the SMEM's right boundary.
    ///
    /// # Parameters
    ///
    /// - `query`: encoded alphabet codes (1–15; ambiguity codes fan out over the reference
    ///   codes they match under the index's alphabet; 0 = sentinel, must not appear).
    /// - `min_len`: discard matches shorter than this (must be ≥ 1).
    /// - `locate`: if `true`, populate `Mem::positions` with reference positions.
    ///
    /// # Returns
    ///
    /// SMEMs in order of increasing `query_start`.  Duplicate `(start, end)` pairs
    /// are deduplicated.
    pub fn find_smems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
        self.smems(query, min_len, locate)
    }

    /// Find all MEMs of length ≥ `min_len`, in the MUMmer / BWA sense.
    ///
    /// A query interval `[s,e)` is reported when **at least one** of its occurrences is
    /// maximal in both directions *at that occurrence* — that is, some occurrence is
    /// preceded by a symbol incompatible with `query[s-1]` (or sits at a reference start,
    /// or `s == 0`) and followed by a symbol incompatible with `query[e]` (or sits at a
    /// reference end, or `e == query.len()`).
    ///
    /// This is deliberately weaker than requiring *every* occurrence to be maximal. Under
    /// the stronger reading a single extendable occurrence in any one reference deletes the
    /// interval, which collapses the result set onto [`find_smems`](Self::find_smems) and
    /// makes per-reference match recovery systematically sparse in a database of
    /// near-identical references.
    ///
    /// The SMEMs are exactly the containment-maximal MEMs, so this result is a strict
    /// superset of [`find_smems`](Self::find_smems).
    ///
    /// `match_count` and `positions` cover **only the maximal occurrences**, not every
    /// occurrence of the matched substring.
    ///
    /// Complexity: O(|query|² × α) for the extension sweep, plus O(α × |ivs|) at each
    /// occurrence-count drop; O(|query|² × α²) worst case. The result can hold up to
    /// O(|query|²) intervals — `min_len` is the only bound on output size.
    pub fn find_mems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
        self.mems(query, min_len, locate)
    }
}

/// The search state of a matched pattern that the sweep carries: [`BidirInterval`] for the
/// occ-table and run-length indexes; the b-move cursor
/// ([`BMoveCursor`](crate::move_index::BMoveCursor)) also holds move-table rows and a toehold.
pub(crate) trait SweepCursor: Copy + std::fmt::Debug + Eq {
    /// Number of occurrences of the matched pattern.
    fn size(&self) -> u32;

    /// The SA intervals, identical across backends: the key the parity tests compare.
    #[cfg(test)]
    fn interval(&self) -> BidirInterval;
}

impl SweepCursor for BidirInterval {
    #[inline]
    fn size(&self) -> u32 {
        BidirInterval::size(self)
    }

    #[cfg(test)]
    fn interval(&self) -> BidirInterval {
        *self
    }
}

/// The SMEM/MEM sweep over any bidirectional index: [`BidirFmIndex`] (occ-table rank),
/// [`BidirRlFmIndex`](crate::rl_fm_index::BidirRlFmIndex) (run-length rank) and
/// [`BMoveIndex`](crate::move_index::BMoveIndex) (move tables). Needs only cursor extension
/// and a locate of the finished matches — never the text — so indexes that keep none get
/// the same results.
pub(crate) trait BidirSearch {
    type Cursor: SweepCursor;

    /// Whether a sweep step takes all of a cursor's children at once ([`Fanout`], from
    /// [`search_children_right`](Self::search_children_right)) rather than extending by
    /// each compatible symbol. The run-length scan ranks every code almost for free, the
    /// occ table pays per lane; per-symbol extension touches only the compatible codes, and
    /// the MEM sweep then builds a drop set's children only when it keeps one.
    const FANOUT_STEPS: bool = true;

    /// The index's alphabet: which reference codes each query code matches.
    fn search_alphabet(&self) -> &AlphabetFns;

    /// Cursor of the empty pattern (every row).
    fn search_full_cursor(&self) -> Self::Cursor;

    /// `cur` extended right by code `c` (P → Pc); `None` when Pc does not occur. Never
    /// called with the sentinel.
    fn search_extend_right(&self, cur: &Self::Cursor, c: u8) -> Option<Self::Cursor>;

    /// `cur` extended left by code `c` (P → cP); `None` when cP does not occur. Never called
    /// with the sentinel.
    fn search_extend_left(&self, cur: &Self::Cursor, c: u8) -> Option<Self::Cursor>;

    /// Every right child of `cur`, indexed by code. Slot 0 (the sentinel: occurrences at a
    /// reference end) lands in MEM drop sets, so [`search_locate`](Self::search_locate) must
    /// be able to locate it.
    fn search_children_right(&self, cur: &Self::Cursor) -> [Option<Self::Cursor>; ALPHABET_SIZE];

    /// Every left child of `cur`, indexed by code.
    fn search_children_left(&self, cur: &Self::Cursor) -> [Option<Self::Cursor>; ALPHABET_SIZE];

    /// The cursor set of `kmer` exactly as the sweeps' right extension from the full cursor
    /// builds it (same cursors, same order), when the backend caches it.
    fn shallow(&self, _kmer: &[u8]) -> Option<&[Self::Cursor]> {
        None
    }

    /// Resolve `raws` in order, locating each one's maximal occurrences (`left_ctx` says
    /// which). Panics on an index without locate support.
    fn search_locate(&self, raws: Vec<RawMem<Self::Cursor>>) -> Vec<Mem>;

    /// Into `out`: the cursor set of `query[start..=j]`, given `curr`, that of
    /// `query[start..j]`. Returns whether `fanout` now holds `curr`'s children; it does not
    /// when the set came from [`shallow`](Self::shallow).
    fn right_step(
        &self,
        query: &[u8],
        (start, j): (usize, usize),
        curr: &[Self::Cursor],
        fanout: &mut Fanout<Self::Cursor>,
        out: &mut Vec<Self::Cursor>,
    ) -> bool {
        if let Some(set) = self.shallow(&query[start..=j]) {
            out.clear();
            out.extend_from_slice(set);
            return false;
        }
        let compat = self.search_alphabet().compatible(query[j]);
        if !Self::FANOUT_STEPS {
            extend_set(curr, compat, out, |cur, c| self.search_extend_right(cur, c));
            return false;
        }
        fanout.fill(curr, |cur| self.search_children_right(cur));
        fanout.emit(compat, out);
        true
    }

    /// Into `out`: the cursors of `ivs` extended left by every symbol compatible with `c`,
    /// in [`Fanout::emit`] order.
    fn left_step(
        &self,
        ivs: &[Self::Cursor],
        c: u8,
        fanout: &mut Fanout<Self::Cursor>,
        out: &mut Vec<Self::Cursor>,
    ) {
        let compat = self.search_alphabet().compatible(c);
        if Self::FANOUT_STEPS {
            fanout.fill(ivs, |cur| self.search_children_left(cur));
            fanout.emit(compat, out);
        } else {
            extend_set(ivs, compat, out, |cur, c| self.search_extend_left(cur, c));
        }
    }

    /// SMEMs of `query`; see [`BidirFmIndex::find_smems`].
    fn smems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
        self.resolve(self.smem_raws(query, min_len), locate)
    }

    /// MEMs of `query`; see [`BidirFmIndex::find_mems`].
    fn mems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
        self.resolve(self.mem_raws(query, min_len), locate)
    }

    /// Every [`RawMem`] of one query as a [`Mem`], located when `locate` is set.
    fn resolve(&self, raws: Vec<RawMem<Self::Cursor>>, locate: bool) -> Vec<Mem> {
        if locate {
            self.search_locate(raws)
        } else {
            raws.into_iter().map(RawMem::into_unlocated).collect()
        }
    }

    /// The SMEMs of `query` as unresolved [`RawMem`]s.
    fn smem_raws(&self, query: &[u8], min_len: usize) -> Vec<RawMem<Self::Cursor>> {
        if query.is_empty() || min_len == 0 {
            return vec![];
        }

        // Single-pass BWA-MEM SMEM sweep: jump the pivot by each pivot's forward reach,
        // collecting MEM candidates (forward-collected right-maximal prefixes, each
        // backward-extended to its true left boundary and verified). Every SMEM is captured
        // by some visited pivot, so filtering the candidates to their containment-maximal
        // elements yields exactly the SMEMs. This both fixes the old pivot-advance bug
        // (which lacked backward extension, dropping SMEMs starting before the next pivot —
        // see `bug-fmidx.md`) and avoids the O(n²) re-extension of the interim fix.
        let mut raws = self.collect_smem_candidates(query, min_len);
        raws.sort_by_key(|m| (m.query_start, m.query_end));
        raws.dedup_by_key(|m| (m.query_start, m.query_end));

        let intervals: Vec<(usize, usize)> =
            raws.iter().map(|m| (m.query_start, m.query_end)).collect();

        let mut smems = Vec::new();
        for (idx, raw) in raws.into_iter().enumerate() {
            let (s, e) = (raw.query_start, raw.query_end);
            // Contained in another MEM => not super-maximal. Post-dedup no two intervals are
            // equal, so `j != idx` already excludes self; the `!=` guard is belt-and-braces.
            let contained = intervals
                .iter()
                .enumerate()
                .any(|(j, &(s2, e2))| j != idx && s2 <= s && e <= e2 && (s2, e2) != (s, e));
            if !contained {
                smems.push(raw);
            }
        }
        smems
    }

    /// The MEMs of `query` as unresolved [`RawMem`]s.
    fn mem_raws(&self, query: &[u8], min_len: usize) -> Vec<RawMem<Self::Cursor>> {
        if query.is_empty() || min_len == 0 {
            return vec![];
        }

        let mut raws = self.collect_mem_candidates(query, min_len);
        raws.sort_by_key(|m| (m.query_start, m.query_end));
        raws.dedup_by_key(|m| (m.query_start, m.query_end));
        raws
    }

    /// Collect every occurrence-level MEM candidate, without resolving positions: the
    /// pruned sweep ([`collect_mem_candidates_pruned`](Self::collect_mem_candidates_pruned)).
    fn collect_mem_candidates(&self, query: &[u8], min_len: usize) -> Vec<RawMem<Self::Cursor>> {
        self.collect_mem_candidates_pruned(query, min_len)
    }

    /// `oracle::collect_mem_candidates_reference` with the work whose output is always
    /// discarded skipped. Returns the same candidates in the same order.
    ///
    /// With `cov(i, j) = |occ(query[i..j))|`, occurrences of `query[i-1..j)` at `p` are exactly
    /// the occurrences of `query[i..j)` at `p+1` preceded by a symbol compatible with
    /// `query[i-1]` (a sentinel never is), and one extends right iff the other does. So:
    ///
    /// - **Counts by arithmetic.** The left-maximal occurrences dropping at `j` number
    ///   `(cov(i,j) − cov(i,j+1)) − (cov(i−1,j) − cov(i−1,j+1))` (`cov(i,n) − cov(i−1,n)` at
    ///   the query end): no left extension, and no drop set unless that count is positive
    ///   and `j − i >= min_len`.
    /// - **Early stop.** Once `cov(i−1,j) == cov(i,j)`, every occurrence of `query[i..j)` and
    ///   of each longer match is left-extendable: start `i` has no more candidates.
    ///
    /// `seen[j]` holds `cov(i−1, j)` exactly when start `i` reaches `j`: start `i−1` wrote it,
    /// or stopped early at some `j0 <= j` where `cov(i−2, ·) == cov(i−1, ·)` from `j0` on
    /// (so `seen[j]`, by induction `cov(i−2, j)`, is still right), or died before `j`, in
    /// which case `cov(i−1, j) = 0` and no earlier start (whose coverage is no larger) wrote
    /// `seen[j]` either.
    ///
    /// Each step takes all of a cursor's children at once ([`Fanout`]): the compatible ones
    /// continue, the rest form the drop set.
    fn collect_mem_candidates_pruned(
        &self,
        query: &[u8],
        min_len: usize,
    ) -> Vec<RawMem<Self::Cursor>> {
        let n = query.len();
        let alphabet = self.search_alphabet();
        let mut out = Vec::new();
        let mut seen = vec![0u32; n + 1];
        let mut fanout = Fanout::default();
        let mut curr: Vec<Self::Cursor> = Vec::new();
        let mut next: Vec<Self::Cursor> = Vec::new();

        for i in 0..n {
            let full = [self.search_full_cursor()];
            self.right_step(query, (i, i), &full, &mut fanout, &mut curr);
            if curr.is_empty() {
                continue;
            }
            let left_ctx = (i > 0).then(|| query[i - 1]);
            let mut cov = coverage(&curr);
            let mut j = i + 1;
            loop {
                let prev = seen[j];
                if i > 0 && prev == cov {
                    break;
                }
                seen[j] = cov;
                if j == n {
                    // Underflow only on mismatched halves (a corrupted file): no candidate.
                    let match_count = if i == 0 {
                        cov
                    } else {
                        cov.saturating_sub(prev)
                    };
                    if n - i >= min_len && match_count > 0 {
                        out.push(RawMem {
                            query_start: i,
                            query_end: n,
                            match_count,
                            ivs: std::mem::take(&mut curr),
                            left_ctx,
                        });
                    }
                    break;
                }
                let compat = alphabet.compatible(query[j]);
                let has_children = self.right_step(query, (i, j), &curr, &mut fanout, &mut next);
                let next_cov = coverage(&next);
                if next_cov != cov && j - i >= min_len {
                    let left_extendable = if i == 0 {
                        Some(0)
                    } else {
                        prev.checked_sub(seen[j + 1])
                    };
                    // `None` only on mismatched halves (a corrupted file): no candidate.
                    let match_count = left_extendable
                        .and_then(|l| (cov - next_cov).checked_sub(l))
                        .unwrap_or(0);
                    if match_count > 0 {
                        if !has_children {
                            fanout.fill(&curr, |cur| self.search_children_right(cur));
                        }
                        let mut dropped = Vec::new();
                        fanout.emit(compat.complement(), &mut dropped);
                        out.push(RawMem {
                            query_start: i,
                            query_end: j,
                            match_count,
                            ivs: dropped,
                            left_ctx,
                        });
                    }
                }
                if next.is_empty() {
                    break;
                }
                std::mem::swap(&mut curr, &mut next);
                cov = next_cov;
                j += 1;
            }
        }

        out
    }

    /// MEM candidates that contain every SMEM: the lockstep sweep
    /// ([`collect_smem_candidates_lockstep`](Self::collect_smem_candidates_lockstep)).
    fn collect_smem_candidates(&self, query: &[u8], min_len: usize) -> Vec<RawMem<Self::Cursor>> {
        self.collect_smem_candidates_lockstep(query, min_len)
    }

    /// `oracle::collect_smem_candidates_reference` (single-pass BWA-MEM pivot sweep) with
    /// the backward phase run on all of a pivot's prefixes in lockstep. Returns the same
    /// candidates in the same order: only work whose result is always rejected is skipped.
    ///
    /// Let prefixes `e_0 < e_1 < …` be the recorded right-maximal ends of a pivot `p`.
    /// Coverage does not drop strictly between `e_i` and `e_{i+1}`, so every occurrence of
    /// `query[p..e_i+1)` extends to `query[p..e_{i+1})`, and the same holds for any start
    /// `s <= p`: `|occ(query[s..e_i+1))| = |occ(query[s..e_{i+1}))|`. Hence:
    ///
    /// - **Right-maximality without extending.** `[s_i, e_i)` (`s_i` where prefix `i` stops
    ///   extending left) is right-maximal iff prefix `i+1` is already empty at `s_i`, i.e.
    ///   it stopped at a strictly larger start. The longest prefix is right-maximal by
    ///   construction (its forward extension died, or it reached the query end).
    /// - **Equal-coverage pruning.** Once prefix `i` and its next-longer neighbour have the
    ///   same coverage, every occurrence of the shorter extends to the longer and stays so
    ///   for every smaller start: the two stop together and prefix `i` can never be
    ///   right-maximal. It is dropped, and its shorter neighbour now compares against the
    ///   longer one (they stop at the same step, so its verdict is unchanged).
    fn collect_smem_candidates_lockstep(
        &self,
        query: &[u8],
        min_len: usize,
    ) -> Vec<RawMem<Self::Cursor>> {
        let n = query.len();
        let mut out = Vec::new();
        let mut buf: Vec<Self::Cursor> = Vec::new();
        let mut fanout = Fanout::default();
        let mut pivot = 0;

        while pivot < n {
            let mut curr = Vec::new();
            let full = [self.search_full_cursor()];
            self.right_step(query, (pivot, pivot), &full, &mut fanout, &mut curr);
            if curr.is_empty() {
                pivot += 1;
                continue;
            }
            let mut cov = coverage(&curr);
            let mut sets: Vec<Vec<Self::Cursor>> = Vec::new();
            let mut ends: Vec<usize> = Vec::new();
            let mut j = pivot + 1;
            loop {
                if j == n {
                    sets.push(curr);
                    ends.push(n);
                    break;
                }
                self.right_step(query, (pivot, j), &curr, &mut fanout, &mut buf);
                let ncov = coverage(&buf);
                if ncov != cov {
                    sets.push(std::mem::replace(&mut curr, std::mem::take(&mut buf)));
                    ends.push(j);
                } else {
                    std::mem::swap(&mut curr, &mut buf);
                }
                if curr.is_empty() {
                    break;
                }
                cov = ncov;
                j += 1;
            }
            let reach = *ends.last().unwrap_or(&(pivot + 1));
            let scratch = (&mut fanout, &mut buf);
            self.extend_prefixes_left(query, pivot, min_len, sets, &ends, scratch, &mut out);
            pivot = reach.max(pivot + 1);
        }

        out
    }

    /// The backward phase of
    /// [`collect_smem_candidates_lockstep`](Self::collect_smem_candidates_lockstep): extend
    /// the prefixes `query[pivot..ends[i])` (cursor sets `sets[i]`, ends ascending) left
    /// together, and push each one that ends right-maximal and long enough, in prefix order.
    #[allow(clippy::too_many_arguments)]
    fn extend_prefixes_left(
        &self,
        query: &[u8],
        pivot: usize,
        min_len: usize,
        mut sets: Vec<Vec<Self::Cursor>>,
        ends: &[usize],
        (fanout, buf): (&mut Fanout<Self::Cursor>, &mut Vec<Self::Cursor>),
        out: &mut Vec<RawMem<Self::Cursor>>,
    ) {
        let m = sets.len();
        // `chain`: prefixes still compared with a neighbour, ascending end; dropped ones
        // leave it. `stop[i]`: start where prefix `i` stopped extending (`usize::MAX` while
        // live). `covs[i]`: its current coverage.
        let mut chain: Vec<usize> = (0..m).collect();
        let mut stop = vec![usize::MAX; m];
        let mut covs: Vec<u32> = sets.iter().map(|s| coverage(s)).collect();
        let mut found: Vec<Option<RawMem<Self::Cursor>>> = (0..m).map(|_| None).collect();
        let mut s = pivot;
        let mut live = m;
        while live > 0 {
            // Longest first, so a neighbour stopping at this same start is already marked.
            for pos in (0..chain.len()).rev() {
                let i = chain[pos];
                if stop[i] != usize::MAX {
                    continue;
                }
                if s > 0 {
                    self.left_step(&sets[i], query[s - 1], fanout, buf);
                    if !buf.is_empty() {
                        std::mem::swap(&mut sets[i], buf);
                        covs[i] = coverage(&sets[i]);
                        continue;
                    }
                }
                stop[i] = s;
                live -= 1;
                let neighbour_stopped_earlier = chain
                    .get(pos + 1)
                    .is_none_or(|&longer| stop[longer] != usize::MAX && stop[longer] > s);
                if neighbour_stopped_earlier && ends[i] - s >= min_len {
                    let ivs = std::mem::take(&mut sets[i]);
                    found[i] = Some(RawMem {
                        query_start: s,
                        query_end: ends[i],
                        match_count: covs[i],
                        ivs,
                        // No occurrence extends further left.
                        left_ctx: None,
                    });
                }
            }
            if live == 0 || s == 0 {
                break;
            }
            // Drop live prefixes whose coverage equals their live longer neighbour's.
            let mut pos = chain.len() - 1;
            while pos > 0 {
                let (shorter, longer) = (chain[pos - 1], chain[pos]);
                if stop[shorter] == usize::MAX
                    && stop[longer] == usize::MAX
                    && covs[shorter] == covs[longer]
                {
                    chain.remove(pos - 1);
                    sets[shorter] = Vec::new();
                    stop[shorter] = s;
                    live -= 1;
                }
                pos -= 1;
            }
            s -= 1;
        }
        out.extend(found.into_iter().flatten());
    }
}

/// A MEM before its reference positions are resolved: query interval, occurrence count, and
/// the accepted cursors (kept so only survivors need locating).
///
/// For a candidate from the MEM sweep the cursors describe matches one symbol longer than
/// `[query_start, query_end)` — they carry the trailing symbol that failed to extend, the
/// sentinel included — except when `query_end` is the query length. Occurrence *start*
/// positions are unaffected, so locating needs no offset adjustment.
pub(crate) struct RawMem<C = BidirInterval> {
    pub(crate) query_start: usize,
    pub(crate) query_end: usize,
    pub(crate) match_count: u32,
    pub(crate) ivs: Vec<C>,
    /// `query[query_start - 1]`, against which occurrences are tested for left-maximality:
    /// only those whose preceding symbol is incompatible with it (a sentinel always is) are
    /// maximal. `None` when every occurrence in `ivs` is left-maximal by construction.
    pub(crate) left_ctx: Option<u8>,
}

impl<C> RawMem<C> {
    /// The [`Mem`] without positions.
    pub(crate) fn into_unlocated(self) -> Mem {
        self.into_mem(Vec::new())
    }

    /// The [`Mem`] with `positions`.
    pub(crate) fn into_mem(self, positions: Vec<(SeqId, u32)>) -> Mem {
        Mem {
            query_start: self.query_start,
            query_end: self.query_end,
            match_count: self.match_count,
            positions,
        }
    }
}

/// Total number of text occurrences represented by a cursor set.
fn coverage<C: SweepCursor>(ivs: &[C]) -> u32 {
    ivs.iter().map(|iv| iv.size()).sum()
}
