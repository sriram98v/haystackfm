//! SMEM / MEM finding: the [`BidirSearch`] sweep shared by every bidirectional index, its
//! k-mer cache and child fan-out ([`fanout`]), and the occ-table backend ([`rows`]).

use crate::alphabet::SymbolSet;
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::bidir_index::BidirFmIndex;
use crate::fm_index::half::FmHalf;
use crate::fm_index::seq_id::SeqId;
use crate::occ::BwtRank;

mod fanout;
mod rows;
#[cfg(test)]
mod tests;

use fanout::extend_set;
pub(crate) use fanout::{Fanout, ShallowCache};

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
    /// Complexity: O(|query|² × α) where α = [`ALPHABET_SIZE`](crate::alphabet::ALPHABET_SIZE).
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

/// The SMEM/MEM sweep over any bidirectional index whose halves implement [`FmHalf`]:
/// [`BidirFmIndex`] (occ-table rank) and
/// [`BidirRlFmIndex`](crate::rl_fm_index::BidirRlFmIndex) (run-length rank). Needs only
/// cursor extension, the forward half's BWT symbols, and forward-row locate — never the
/// text — so the run-length index, which keeps none, gets the same results.
pub(crate) trait BidirSearch {
    type Half: FmHalf;

    /// Whether a sweep step takes all of an interval's children from one rank pass
    /// ([`Fanout`]) rather than extending by each compatible symbol. The pass ranks every
    /// code at both borders, which the run-length scan does almost for free but the occ
    /// table pays per lane; per-symbol extension ranks only the compatible codes, and the
    /// MEM sweep then builds a drop set's children only when it keeps one.
    const FANOUT_STEPS: bool = true;

    /// Forward half (index of `T`); cursor `fwd_*` rows refer to its SA.
    fn fwd_half(&self) -> &Self::Half;

    /// Reverse half (index of reversed `T`).
    fn rev_half(&self) -> &Self::Half;

    /// Cursor spanning every row.
    fn search_full_interval(&self) -> BidirInterval;

    /// `(sequence id, offset)` of forward SA rows `lo..hi`, in row order.
    fn search_locate_rows(&self, lo: u32, hi: u32) -> Vec<(SeqId, u32)>;

    /// `(sequence id, offset)` of each listed forward SA row, in order.
    fn search_locate_row_list(&self, rows: &[u32]) -> Vec<(SeqId, u32)>;

    /// The interval set of `kmer` exactly as the sweeps' right extension from the full
    /// interval builds it (same intervals, same order), when the backend caches it.
    fn shallow(&self, _kmer: &[u8]) -> Option<&[BidirInterval]> {
        None
    }

    /// Into `out`: the interval set of `query[start..=j]`, given `curr`, that of
    /// `query[start..j]`. Returns whether `fanout` now holds `curr`'s children; it does not
    /// when the set came from [`shallow`](Self::shallow).
    fn right_step(
        &self,
        query: &[u8],
        (start, j): (usize, usize),
        curr: &[BidirInterval],
        fanout: &mut Fanout,
        out: &mut Vec<BidirInterval>,
    ) -> bool {
        if let Some(set) = self.shallow(&query[start..=j]) {
            out.clear();
            out.extend_from_slice(set);
            return false;
        }
        let rev = self.rev_half();
        let compat = rev.alphabet().compatible(query[j]);
        if !Self::FANOUT_STEPS {
            extend_set(curr, compat, out, |iv, c| iv.extend_right(c, rev));
            return false;
        }
        fanout.right(curr, rev);
        fanout.emit(compat, out);
        true
    }

    /// Into `out`: the intervals of `ivs` extended left by every symbol compatible with
    /// `c`, in [`Fanout::emit`] order.
    fn left_step(
        &self,
        ivs: &[BidirInterval],
        c: u8,
        fanout: &mut Fanout,
        out: &mut Vec<BidirInterval>,
    ) {
        let fwd = self.fwd_half();
        let compat = fwd.alphabet().compatible(c);
        if Self::FANOUT_STEPS {
            fanout.left(ivs, fwd);
            fanout.emit(compat, out);
        } else {
            extend_set(ivs, compat, out, |iv, c| iv.extend_left(c, fwd));
        }
    }

    /// Forward rows of `ivs` whose preceding symbol (`BWT_fwd[row]`) is not in `compat`, in
    /// order: the left-maximal occurrences of a match whose left context `compat` matches.
    fn left_maximal_rows(&self, ivs: &[BidirInterval], compat: SymbolSet) -> Vec<u32> {
        let fwd = self.fwd_half();
        ivs.iter()
            .flat_map(|iv| iv.fwd_lo..iv.fwd_hi)
            .filter(|&row| !compat.contains(fwd.bwt_rank().symbol_at(row)))
            .collect()
    }

    /// SMEMs of `query`; see [`BidirFmIndex::find_smems`].
    fn smems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
        self.locate_all(self.smem_raws(query, min_len), locate)
    }

    /// MEMs of `query`; see [`BidirFmIndex::find_mems`].
    fn mems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
        self.locate_all(self.mem_raws(query, min_len), locate)
    }

    /// Resolve every [`RawMem`] of one query, in order: one batched locate for the whole
    /// query ([`locate_all_batched`](Self::locate_all_batched)).
    fn locate_all(&self, raws: Vec<RawMem>, locate: bool) -> Vec<Mem> {
        self.locate_all_batched(raws, locate)
    }

    /// [`locate_all`](Self::locate_all) with one [`search_locate_row_list`] call for the
    /// rows of every match, so a backend that interleaves LF walks gets the whole query's
    /// rows at once instead of a few per match. Same results, same order.
    ///
    /// [`search_locate_row_list`]: Self::search_locate_row_list
    fn locate_all_batched(&self, raws: Vec<RawMem>, locate: bool) -> Vec<Mem> {
        if !locate {
            return raws
                .into_iter()
                .map(|raw| self.locate_raw(raw, false))
                .collect();
        }
        let mut rows: Vec<u32> = Vec::new();
        let mut ends: Vec<usize> = Vec::with_capacity(raws.len());
        for raw in &raws {
            if let Some(c) = raw.left_ctx {
                let compat = self.fwd_half().alphabet().compatible(c);
                rows.extend(self.left_maximal_rows(&raw.ivs, compat));
            } else {
                rows.extend(raw.ivs.iter().flat_map(|iv| iv.fwd_lo..iv.fwd_hi));
            }
            ends.push(rows.len());
        }
        let mut located = self.search_locate_row_list(&rows).into_iter();
        let mut start = 0;
        raws.into_iter()
            .zip(ends)
            .map(|(raw, end)| {
                let positions = located.by_ref().take(end - start).collect();
                start = end;
                Mem {
                    query_start: raw.query_start,
                    query_end: raw.query_end,
                    match_count: raw.match_count,
                    positions,
                }
            })
            .collect()
    }

    /// The SMEMs of `query` as unresolved [`RawMem`]s.
    fn smem_raws(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
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
    fn mem_raws(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
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
    fn collect_mem_candidates(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
        self.collect_mem_candidates_pruned(query, min_len)
    }

    /// The original MEM sweep, kept as the oracle the pruned one is tested against.
    ///
    /// For each start `i`, forward-extends the full occurrence set of `query[i..j)`. At every
    /// step where the occurrence count drops, the occurrences that failed to extend are
    /// exactly the ones right-maximal at `j`; [`drop_set`](Self::drop_set) recovers them as
    /// SA intervals. `j == query.len()` is right-maximal for the whole surviving set.
    /// [`push_mem_candidate`](Self::push_mem_candidate) then applies the per-occurrence
    /// left-maximality filter.
    ///
    /// Completeness: `curr` at step `j` is precisely the occurrence set of `query[i..j)`, so
    /// every MEM `(i,j)` has its witnessing occurrence in the drop set at `j`. Soundness:
    /// anything emitted has an occurrence maximal in both directions.
    ///
    #[cfg(test)]
    fn collect_mem_candidates_reference(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
        let n = query.len();
        let mut out: Vec<RawMem> = Vec::new();

        for i in 0..n {
            // All occurrences of a symbol compatible with `query[i]`.
            let mut curr =
                extend_multi_right(&[self.search_full_interval()], query[i], self.rev_half());
            if curr.is_empty() {
                continue;
            }
            let mut j = i + 1;
            loop {
                if j == n {
                    // Query exhausted: every surviving occurrence is right-maximal.
                    self.push_mem_candidate(&mut out, query, i, n, curr, min_len);
                    break;
                }
                let next = extend_multi_right(&curr, query[j], self.rev_half());
                if coverage(&next) != coverage(&curr) {
                    let dropped = self.drop_set(&curr, query[j]);
                    self.push_mem_candidate(&mut out, query, i, j, dropped, min_len);
                }
                if next.is_empty() {
                    break;
                }
                curr = next;
                j += 1;
            }
        }

        out
    }

    /// [`collect_mem_candidates_reference`](Self::collect_mem_candidates_reference) with the
    /// work whose output is always discarded skipped. Returns the same candidates in the same order.
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
    /// Each step takes all of an interval's children from one rank pass ([`Fanout`]): the
    /// compatible ones continue, the rest form the drop set.
    fn collect_mem_candidates_pruned(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
        let n = query.len();
        let rev = self.rev_half();
        let mut out: Vec<RawMem> = Vec::new();
        let mut seen = vec![0u32; n + 1];
        let mut fanout = Fanout::default();
        let mut curr: Vec<BidirInterval> = Vec::new();
        let mut next: Vec<BidirInterval> = Vec::new();

        for i in 0..n {
            let full = [self.search_full_interval()];
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
                let compat = rev.alphabet().compatible(query[j]);
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
                            fanout.right(&curr, rev);
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

    /// The occurrences in `ivs` that canNOT be extended right by `c`.
    ///
    /// Extends by every symbol *incompatible* with `c`. The results are disjoint (distinct
    /// following symbols), and the sentinel is incompatible with every base, so occurrences
    /// sitting at the end of a reference are captured here rather than silently lost.
    #[cfg(test)]
    fn drop_set(&self, ivs: &[BidirInterval], c: u8) -> Vec<BidirInterval> {
        let mut out = Vec::new();
        for sym in self.rev_half().alphabet().compatible(c).complement().iter() {
            for iv in ivs {
                if let Some(ext) = iv.extend_right(sym, self.rev_half()) {
                    out.push(ext);
                }
            }
        }
        out
    }

    /// Apply `min_len` and the per-occurrence left-maximality filter, then record the MEM.
    ///
    /// `ivs` holds occurrences already known to be right-maximal at `end`. `extend_multi_left`
    /// walks the *compatible* bases of `query[start-1]`, and its results are disjoint, so
    /// subtracting its coverage counts exactly the occurrences that are left-extendable —
    /// no interval subtraction and no sentinel special case (a reference start is preceded by
    /// a sentinel, which is never compatible with a base).
    #[cfg(test)]
    fn push_mem_candidate(
        &self,
        out: &mut Vec<RawMem>,
        query: &[u8],
        start: usize,
        end: usize,
        ivs: Vec<BidirInterval>,
        min_len: usize,
    ) {
        if end - start < min_len || ivs.is_empty() {
            return;
        }
        let left_ctx = if start == 0 {
            None
        } else {
            Some(query[start - 1])
        };
        let match_count = match left_ctx {
            None => coverage(&ivs),
            Some(c) => coverage(&ivs) - coverage(&extend_multi_left(&ivs, c, self.fwd_half())),
        };
        if match_count == 0 {
            return;
        }
        out.push(RawMem {
            query_start: start,
            query_end: end,
            match_count,
            ivs,
            left_ctx,
        });
    }

    /// MEM candidates that contain every SMEM: the lockstep sweep
    /// ([`collect_smem_candidates_lockstep`](Self::collect_smem_candidates_lockstep)).
    fn collect_smem_candidates(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
        self.collect_smem_candidates_lockstep(query, min_len)
    }

    /// Single-pass BWA-MEM collection of MEM candidates that contain every SMEM, kept as the
    /// oracle the lockstep sweep is tested against.
    ///
    /// Visits pivots left-to-right, advancing each time by the pivot's *forward reach* (the
    /// end of the longest match anchored at the pivot). At each pivot it forward-extends,
    /// recording every right-maximal prefix `[pivot, e)` (a right-end where the interval set's
    /// coverage drops), then backward-extends each recorded prefix to its true left boundary
    /// and keeps it if the result verifies as a MEM of length ≥ `min_len`.
    ///
    /// Correctness: max right-reach is monotonic in the start position, so any SMEM starting
    /// after a pivot must extend past that pivot's reach and therefore covers the next pivot —
    /// hence every SMEM is anchored by some visited pivot and appears here (possibly alongside
    /// non-super MEMs, which the caller's containment filter removes).
    #[cfg(test)]
    fn collect_smem_candidates_reference(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
        let n = query.len();
        let mut out: Vec<RawMem> = Vec::new();
        let mut pivot = 0;

        while pivot < n {
            // Forward extension from `pivot`, collecting right-maximal prefixes as
            // (interval set for query[pivot..end), end).
            let mut curr = extend_multi_right(
                &[self.search_full_interval()],
                query[pivot],
                self.rev_half(),
            );
            if curr.is_empty() {
                pivot += 1; // query[pivot] absent from the text
                continue;
            }
            let mut cov = coverage(&curr);
            let mut prefixes: Vec<(Vec<BidirInterval>, usize)> = Vec::new();
            let mut j = pivot + 1;
            loop {
                if j == n {
                    prefixes.push((curr, n));
                    break;
                }
                let next = extend_multi_right(&curr, query[j], self.rev_half());
                let ncov = coverage(&next);
                if ncov != cov {
                    // Some occurrences of query[pivot..j) do not extend right by query[j];
                    // [pivot, j) is a right-maximal prefix.
                    prefixes.push((curr.clone(), j));
                }
                if next.is_empty() {
                    break;
                }
                curr = next;
                cov = ncov;
                j += 1;
            }

            // The longest prefix's end is the forward reach; advance the pivot there.
            let reach = prefixes.last().map(|(_, e)| *e).unwrap_or(pivot + 1);

            // Backward-extend each right-maximal prefix to its left boundary → MEM candidate.
            for (ivs, end) in prefixes {
                // Do NOT cull by `end - pivot` here: backward extension can lengthen the match
                // well past the pivot-anchored prefix (a short prefix at a post-jump pivot can
                // extend left into a long MEM). Filter by the final `end - start` only.
                let (bivs, start) = self.extend_left_maximally(ivs, query, pivot);
                if end - start < min_len {
                    continue;
                }
                // Verify right-maximality (left-maximality is guaranteed by the backward stop):
                // the whole set must fail to extend right by query[end].
                let right_maximal =
                    end == n || extend_multi_right(&bivs, query[end], self.rev_half()).is_empty();
                if !right_maximal {
                    continue;
                }
                let match_count: u32 = bivs.iter().map(|iv| iv.size()).sum();
                out.push(RawMem {
                    query_start: start,
                    query_end: end,
                    match_count,
                    ivs: bivs,
                    // `extend_left_maximally` stopped because no occurrence extends left,
                    // so every occurrence here is already left-maximal.
                    left_ctx: None,
                });
            }

            pivot = reach.max(pivot + 1);
        }

        out
    }

    /// [`collect_smem_candidates_reference`](Self::collect_smem_candidates_reference) with the
    /// backward phase run on all of a pivot's prefixes in lockstep. Returns the same candidates in the same
    /// order: only work whose result is always rejected is skipped.
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
    fn collect_smem_candidates_lockstep(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
        let n = query.len();
        let mut out: Vec<RawMem> = Vec::new();
        let mut buf: Vec<BidirInterval> = Vec::new();
        let mut fanout = Fanout::default();
        let mut pivot = 0;

        while pivot < n {
            let mut curr = Vec::new();
            let full = [self.search_full_interval()];
            self.right_step(query, (pivot, pivot), &full, &mut fanout, &mut curr);
            if curr.is_empty() {
                pivot += 1;
                continue;
            }
            let mut cov = coverage(&curr);
            let mut sets: Vec<Vec<BidirInterval>> = Vec::new();
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
    /// the prefixes `query[pivot..ends[i])` (interval sets `sets[i]`, ends ascending) left
    /// together, and push each one that ends right-maximal and long enough, in prefix order.
    #[allow(clippy::too_many_arguments)]
    fn extend_prefixes_left(
        &self,
        query: &[u8],
        pivot: usize,
        min_len: usize,
        mut sets: Vec<Vec<BidirInterval>>,
        ends: &[usize],
        (fanout, buf): (&mut Fanout, &mut Vec<BidirInterval>),
        out: &mut Vec<RawMem>,
    ) {
        let m = sets.len();
        // `chain`: prefixes still compared with a neighbour, ascending end; dropped ones
        // leave it. `stop[i]`: start where prefix `i` stopped extending (`usize::MAX` while
        // live). `covs[i]`: its current coverage.
        let mut chain: Vec<usize> = (0..m).collect();
        let mut stop = vec![usize::MAX; m];
        let mut covs: Vec<u32> = sets.iter().map(|s| coverage(s)).collect();
        let mut found: Vec<Option<RawMem>> = (0..m).map(|_| None).collect();
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

    /// Extend an interval set as far left as possible from left boundary `from`, returning the
    /// widened set and the resulting start position. Stops when the next left extension is
    /// empty (left-maximal) or the query start is reached.
    #[cfg(test)]
    fn extend_left_maximally(
        &self,
        mut ivs: Vec<BidirInterval>,
        query: &[u8],
        from: usize,
    ) -> (Vec<BidirInterval>, usize) {
        let mut start = from;
        while start > 0 {
            let next = extend_multi_left(&ivs, query[start - 1], self.fwd_half());
            if next.is_empty() {
                break;
            }
            ivs = next;
            start -= 1;
        }
        (ivs, start)
    }

    /// Resolve a [`RawMem`] into a [`Mem`], locating its occurrences when `locate` is set.
    ///
    /// When `left_ctx` is set, occurrences that are left-extendable are dropped here so that
    /// `positions` holds exactly the maximal occurrences already counted by `match_count`.
    /// The symbol preceding the occurrence at forward row `r` is `BWT[r]`, so the test runs
    /// on the BWT before locating and needs no text. An occurrence at a reference start has
    /// a sentinel there, which no query symbol is compatible with: it is always kept.
    fn locate_raw(&self, raw: RawMem, locate: bool) -> Mem {
        let positions = if !locate {
            Vec::new()
        } else if let Some(c) = raw.left_ctx {
            let compat = self.fwd_half().alphabet().compatible(c);
            let rows = self.left_maximal_rows(&raw.ivs, compat);
            self.search_locate_row_list(&rows)
        } else {
            raw.ivs
                .iter()
                .flat_map(|iv| self.search_locate_rows(iv.fwd_lo, iv.fwd_hi))
                .collect()
        };
        Mem {
            query_start: raw.query_start,
            query_end: raw.query_end,
            match_count: raw.match_count,
            positions,
        }
    }
}

/// A MEM before its reference positions are resolved: query interval, occurrence count, and
/// the accepted bidirectional SA intervals (kept so only survivors need locating).
///
/// For a candidate from [`collect_mem_candidates`](BidirFmIndex::collect_mem_candidates) the
/// intervals describe matches one symbol longer than `[query_start, query_end)` — they carry
/// the trailing symbol that failed to extend — except when `query_end` is the query length.
/// Occurrence *start* positions are unaffected, so locating needs no offset adjustment.
pub(crate) struct RawMem {
    query_start: usize,
    query_end: usize,
    match_count: u32,
    ivs: Vec<BidirInterval>,
    /// `query[query_start - 1]`, against which occurrences are tested for left-maximality.
    /// `None` when every occurrence in `ivs` is left-maximal by construction.
    left_ctx: Option<u8>,
}

/// Total number of text occurrences represented by an interval set.
fn coverage(ivs: &[BidirInterval]) -> u32 {
    ivs.iter().map(|iv| iv.size()).sum()
}

/// Extend each interval in `ivs` right by `c`, using the index's alphabet compatibility.
#[cfg(test)]
fn extend_multi_right<H: FmHalf>(ivs: &[BidirInterval], c: u8, rev: &H) -> Vec<BidirInterval> {
    let mut result = Vec::new();
    for base in rev.alphabet().compatible(c).iter() {
        for iv in ivs {
            if let Some(ext) = iv.extend_right(base, rev) {
                result.push(ext);
            }
        }
    }
    result
}

/// Extend each interval in `ivs` left by `c`, using the index's alphabet compatibility.
#[cfg(test)]
fn extend_multi_left<H: FmHalf>(ivs: &[BidirInterval], c: u8, fwd: &H) -> Vec<BidirInterval> {
    let mut result = Vec::new();
    for base in fwd.alphabet().compatible(c).iter() {
        for iv in ivs {
            if let Some(ext) = iv.extend_left(base, fwd) {
                result.push(ext);
            }
        }
    }
    result
}
