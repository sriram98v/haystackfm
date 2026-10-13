use crate::alphabet::{SymbolSet, ALPHABET_SIZE};
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::bidir_index::BidirFmIndex;
use crate::fm_index::half::FmHalf;
use crate::fm_index::seq_id::SeqId;
use crate::fm_index::FmIndex;
use crate::occ::BwtRank;

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

impl BidirSearch for BidirFmIndex {
    type Half = FmIndex;

    const FANOUT_STEPS: bool = false;

    fn fwd_half(&self) -> &FmIndex {
        &self.fwd
    }

    fn rev_half(&self) -> &FmIndex {
        &self.rev
    }

    fn search_full_interval(&self) -> BidirInterval {
        self.full_interval()
    }

    fn search_locate_rows(&self, lo: u32, hi: u32) -> Vec<(SeqId, u32)> {
        self.fwd.locate_rows(lo, hi)
    }

    fn shallow(&self, kmer: &[u8]) -> Option<&[BidirInterval]> {
        self.shallow.get(kmer)
    }

    fn search_locate_row_list(&self, rows: &[u32]) -> Vec<(SeqId, u32)> {
        let mut text_positions = Vec::with_capacity(rows.len());
        self.fwd.resolve_sa_batch(rows, &mut text_positions);
        text_positions
            .into_iter()
            .map(|pos| {
                self.fwd
                    .map_position(pos)
                    .expect("resolved SA position must be within text bounds")
            })
            .collect()
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

/// Deepest k-mer a [`ShallowCache`] holds.
const SHALLOW_MAX_DEPTH: u32 = 8;

/// Most intervals a [`ShallowCache`] holds (~40 MB). Wildcard fan-out makes a level's size
/// depend on the text (a reference dense in ambiguity codes multiplies the variants per
/// k-mer), so the depth stops at the last level that fits.
const SHALLOW_MAX_IVS: usize = 1 << 21;

/// The interval set of every `{A,C,G,T}` k-mer up to a small depth, exactly as the sweeps'
/// right extension from the full interval produces it (built by that same extension, so
/// identical intervals in identical order, wildcard-compatible variants included). Every
/// SMEM pivot and MEM start begins with these shallow steps, whose intervals are the widest
/// and costliest to extend.
///
/// At most [`SHALLOW_MAX_IVS`] intervals: the depth shrinks for small texts and for texts
/// whose wildcard variants would overflow that budget. Derived, never serialized.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShallowCache {
    depth: usize,
    /// First set index of each depth (`level_base[d]`, `d` in `1..=depth`).
    level_base: Vec<usize>,
    /// Set `x` is `ivs[offs[x]..offs[x + 1]]`.
    offs: Vec<u32>,
    ivs: Vec<BidirInterval>,
}

impl ShallowCache {
    /// Build for a text of `text_len` symbols over the half `rev` (right extension).
    pub(crate) fn build<H: FmHalf>(full: BidirInterval, rev: &H, text_len: u32) -> Self {
        // About `log4(n)` levels: deeper ones would hold mostly singletons and misses.
        let depth = (text_len.max(1).ilog2() / 2).min(SHALLOW_MAX_DEPTH) as usize;
        let mut cache = Self {
            depth,
            level_base: vec![0; depth + 1],
            offs: vec![0],
            ivs: Vec::new(),
        };
        let mut fanout = Fanout::default();
        let mut parents: Vec<Vec<BidirInterval>> = vec![vec![full]];
        let mut child = Vec::new();
        for d in 1..=depth {
            cache.level_base[d] = cache.offs.len() - 1;
            let mut children = Vec::with_capacity(parents.len() * 4);
            // Set index within a level: the k-mer in base 4 (A = 0 .. T = 3), last symbol
            // least significant, so parent `p`'s children are `4p .. 4p + 4`.
            for parent in &parents {
                fanout.right(parent, rev);
                for c in CORE_CODES {
                    fanout.emit(rev.alphabet().compatible(c), &mut child);
                    if cache.ivs.len() + child.len() > SHALLOW_MAX_IVS {
                        // Over budget: keep only the complete levels before this one.
                        let keep = cache.level_base[d];
                        cache.ivs.truncate(cache.offs[keep] as usize);
                        cache.offs.truncate(keep + 1);
                        cache.level_base.truncate(d);
                        cache.depth = d - 1;
                        return cache;
                    }
                    cache.ivs.extend_from_slice(&child);
                    cache
                        .offs
                        .push(u32::try_from(cache.ivs.len()).expect("bounded by the budget"));
                    children.push(child.clone());
                }
            }
            parents = children;
        }
        cache
    }

    /// The cached set of `kmer`, if it is short enough and all `{A,C,G,T}`.
    #[inline]
    pub(crate) fn get(&self, kmer: &[u8]) -> Option<&[BidirInterval]> {
        if kmer.is_empty() || kmer.len() > self.depth {
            return None;
        }
        let mut x = 0usize;
        for &q in kmer {
            if !(CORE_CODES[0]..=CORE_CODES[3]).contains(&q) {
                return None;
            }
            x = x * 4 + (q - CORE_CODES[0]) as usize;
        }
        let x = self.level_base[kmer.len()] + x;
        Some(&self.ivs[self.offs[x] as usize..self.offs[x + 1] as usize])
    }

    /// Heap bytes held.
    pub(crate) fn heap_size_bytes(&self) -> usize {
        self.offs.len() * 4
            + self.ivs.len() * std::mem::size_of::<BidirInterval>()
            + self.level_base.len() * std::mem::size_of::<usize>()
    }
}

/// The codes a [`ShallowCache`] enumerates, ascending and contiguous.
const CORE_CODES: [u8; 4] = [
    crate::alphabet::A,
    crate::alphabet::C,
    crate::alphabet::G,
    crate::alphabet::T,
];

/// Every child of every interval of a set, from one rank pass per interval
/// ([`BidirInterval::children_right`] / [`children_left`](BidirInterval::children_left)),
/// then emitted for any symbol subset in the order `extend_multi_right` /
/// `extend_multi_left` produce: symbols ascending, intervals in input order. A sweep step
/// takes its compatible extension and its drop set from the same pass.
#[derive(Default)]
pub(crate) struct Fanout {
    children: Vec<[Option<BidirInterval>; ALPHABET_SIZE]>,
}

impl Fanout {
    /// Children of `ivs` extended right (the pattern grows at its end).
    fn right<H: FmHalf>(&mut self, ivs: &[BidirInterval], rev: &H) {
        self.children.clear();
        self.children
            .extend(ivs.iter().map(|iv| iv.children_right(rev)));
    }

    /// Children of `ivs` extended left (the pattern grows at its start).
    fn left<H: FmHalf>(&mut self, ivs: &[BidirInterval], fwd: &H) {
        self.children.clear();
        self.children
            .extend(ivs.iter().map(|iv| iv.children_left(fwd)));
    }

    /// The children by members of `set`, into `out` (cleared first).
    fn emit(&self, set: SymbolSet, out: &mut Vec<BidirInterval>) {
        out.clear();
        for c in set.iter() {
            out.extend(self.children.iter().filter_map(|ch| ch[c as usize]));
        }
    }
}

/// Into `out` (cleared first): `extend(iv, c)` for every member `c` of `set` and interval
/// `iv`, symbols ascending, intervals in input order (the order [`Fanout::emit`] uses).
#[inline]
fn extend_set(
    ivs: &[BidirInterval],
    set: SymbolSet,
    out: &mut Vec<BidirInterval>,
    extend: impl Fn(&BidirInterval, u8) -> Option<BidirInterval>,
) {
    out.clear();
    for c in set.iter() {
        out.extend(ivs.iter().filter_map(|iv| extend(iv, c)));
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::{encode_char, DnaSequence};
    use crate::fm_index::{FmIndex, FmIndexConfig};

    fn encode(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    fn bidir(s: &str) -> BidirFmIndex {
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            ..Default::default()
        };
        BidirFmIndex::build_cpu(&[DnaSequence::from_str(s).unwrap()], &config).unwrap()
    }

    /// Brute-force MEM finder for reference: finds all substrings of `query` that
    /// occur in `reference` and are both left- and right-maximal.
    fn brute_force_mems(reference: &str, query: &str, min_len: usize) -> Vec<(usize, usize)> {
        let n = query.len();
        let mut mems: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

        for start in 0..n {
            for end in start + min_len..=n {
                let sub = &query[start..end];
                if !reference.contains(sub) {
                    continue;
                }
                // Check right-maximal: can't extend right.
                let right_maximal = end == n || !reference.contains(&query[start..end + 1]);
                // Check left-maximal: can't extend left.
                let left_maximal = start == 0 || !reference.contains(&query[start - 1..end]);
                if right_maximal && left_maximal {
                    mems.insert((start, end));
                }
            }
        }

        let mut v: Vec<_> = mems.into_iter().collect();
        v.sort();
        v
    }

    /// True when reference symbol `t` is matched by query symbol `q` under IUPAC overlap.
    fn compat(q: u8, t: u8) -> bool {
        crate::alphabet::compatible_symbols(q).contains(&t)
    }

    /// Occurrence-level brute-force MEM enumerator — the MUMmer/BWA definition.
    ///
    /// `[s,e)` is a MEM when **some** occurrence of `query[s..e)` in some reference is
    /// maximal in both directions *at that occurrence*. Independent of the index: it walks
    /// every (query position, reference position) pair. `refs` holds the encoded bases of
    /// each reference separately, so a reference boundary counts as maximal.
    fn brute_force_mems_occ(refs: &[Vec<u8>], query: &[u8], min_len: usize) -> Vec<(usize, usize)> {
        let mut out = std::collections::BTreeSet::new();
        for t in refs {
            for i in 0..query.len() {
                for j in 0..t.len() {
                    if !compat(query[i], t[j]) {
                        continue;
                    }
                    if i > 0 && j > 0 && compat(query[i - 1], t[j - 1]) {
                        continue; // this occurrence is not left-maximal
                    }
                    // For a fixed occurrence the only right-maximal length is the full
                    // extension: every shorter prefix still extends by one more base.
                    let mut l = 0;
                    while i + l < query.len() && j + l < t.len() && compat(query[i + l], t[j + l]) {
                        l += 1;
                    }
                    if l >= min_len {
                        out.insert((i, i + l));
                    }
                }
            }
        }
        out.into_iter().collect()
    }

    #[test]
    fn no_smems_when_query_absent() {
        let idx = bidir("AAAA");
        let query = encode("CCCC");
        let smems = idx.find_smems(&query, 1, false);
        assert!(smems.is_empty());
    }

    #[test]
    fn single_smem_exact_match() {
        let idx = bidir("ACGTACGT");
        let query = encode("ACGT");
        let smems = idx.find_smems(&query, 1, false);
        assert_eq!(smems.len(), 1);
        assert_eq!(smems[0].query_start, 0);
        assert_eq!(smems[0].query_end, 4);
        assert_eq!(smems[0].match_count, 2); // "ACGT" appears twice in reference
    }

    #[test]
    fn smem_locate_returns_correct_positions() {
        let idx = bidir("ACGTACGT");
        let query = encode("ACGT");
        let smems = idx.find_smems(&query, 1, true);
        assert_eq!(smems.len(), 1);
        let mut positions = smems[0].positions.clone();
        positions.sort();
        assert_eq!(positions, vec![(SeqId::new(0), 0), (SeqId::new(0), 4)]);
    }

    #[test]
    fn min_len_filter() {
        let idx = bidir("ACGTACGT");
        let query = encode("A");
        // "A" is length 1; with min_len=2, it should be filtered out.
        let smems = idx.find_smems(&query, 2, false);
        assert!(smems.is_empty());
    }

    #[test]
    fn smems_match_brute_force() {
        let reference = "ACGTTAGCCAGTACGT";
        let query_str = "CGTTAGC";
        let idx = bidir(reference);
        let query = encode(query_str);

        let smems = idx.find_smems(&query, 1, false);
        let smem_pairs: Vec<(usize, usize)> =
            smems.iter().map(|m| (m.query_start, m.query_end)).collect();

        let expected = brute_force_mems(reference, query_str, 1);

        assert_eq!(
            smem_pairs, expected,
            "SMEMs differ from brute force.\nGot:      {:?}\nExpected: {:?}",
            smem_pairs, expected
        );
    }

    #[test]
    fn find_mems_superset_of_smems() {
        let reference = "ACGTTAGCCAGTACGT";
        let query_str = "CGTTAGC";
        let idx = bidir(reference);
        let query = encode(query_str);

        let smems = idx.find_smems(&query, 1, false);
        let mems = idx.find_mems(&query, 1, false);

        // Every SMEM should appear in the MEMs list.
        for smem in &smems {
            assert!(
                mems.iter()
                    .any(|m| m.query_start == smem.query_start && m.query_end == smem.query_end),
                "SMEM {:?} not found in MEMs list",
                smem
            );
        }
    }

    #[test]
    fn smems_all_positions_valid() {
        let reference = "ACGTTAGCCAGTACGT";
        let query_str = "AGTACGT";
        let idx = bidir(reference);
        let query_encoded = encode(query_str);

        let smems = idx.find_smems(&query_encoded, 1, true);
        for mem in &smems {
            let pattern = &query_str[mem.query_start..mem.query_end];
            for (_, pos) in &mem.positions {
                let pos = *pos as usize;
                assert!(
                    pos + pattern.len() <= reference.len(),
                    "position {} out of bounds",
                    pos
                );
                assert_eq!(
                    &reference[pos..pos + pattern.len()],
                    pattern,
                    "wrong match at pos {}: expected '{}' got '{}'",
                    pos,
                    pattern,
                    &reference[pos..pos + pattern.len()]
                );
            }
        }
    }

    #[test]
    fn smem_count_matches_unidirectional_count() {
        let reference = "ACGTACGTACGT";
        let idx = bidir(reference);
        let uni_config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            ..Default::default()
        };
        let uni =
            FmIndex::build_cpu(&[DnaSequence::from_str(reference).unwrap()], &uni_config).unwrap();

        let query = encode("ACGT");
        let smems = idx.find_smems(&query, 1, false);
        // "ACGT" occurs 3 times — should be reflected in match_count
        assert_eq!(smems[0].match_count, uni.count(&query));
    }

    #[test]
    fn empty_query_returns_empty() {
        let idx = bidir("ACGT");
        assert!(idx.find_smems(&[], 1, false).is_empty());
        assert!(idx.find_mems(&[], 1, false).is_empty());
    }

    #[test]
    fn find_mems_matches_brute_force() {
        let reference = "ACGTTAGCCAGTACGT";
        let query_str = "CGTTAGC";
        let idx = bidir(reference);
        let query = encode(query_str);

        let mems = idx.find_mems(&query, 1, false);
        let mem_pairs: Vec<(usize, usize)> =
            mems.iter().map(|m| (m.query_start, m.query_end)).collect();

        let expected = brute_force_mems_occ(&[encode(reference)], &query, 1);

        assert_eq!(
            mem_pairs, expected,
            "MEMs differ from brute force.\nGot:      {:?}\nExpected: {:?}",
            mem_pairs, expected
        );
    }

    #[test]
    fn find_mems_all_left_and_right_maximal() {
        let reference = "ACGTTAGCCAGTACGT";
        let query_str = "CGTTAGCAGT";
        let idx = bidir(reference);
        let query = encode(query_str);
        let text = encode(reference);

        let mems = idx.find_mems(&query, 1, false);
        assert!(!mems.is_empty(), "expected at least one MEM");

        for mem in &mems {
            let (s, e) = (mem.query_start, mem.query_end);
            // Occurrence-level: at least one occurrence must be maximal in both directions.
            let witnessed = (0..text.len()).any(|j| {
                if j + (e - s) > text.len() {
                    return false;
                }
                let matches = (0..e - s).all(|k| compat(query[s + k], text[j + k]));
                let left_max = s == 0 || j == 0 || !compat(query[s - 1], text[j - 1]);
                let right_max = e == query.len()
                    || j + (e - s) == text.len()
                    || !compat(query[e], text[j + (e - s)]);
                matches && left_max && right_max
            });
            assert!(
                witnessed,
                "MEM [{s},{e}) has no occurrence that is maximal in both directions"
            );
        }
    }

    /// Two references sharing a query prefix, one extending further than the other. Under
    /// whole-set maximality only `(0,10)` survives, because ref_a's occurrences veto every
    /// shorter interval; per occurrence, four more MEMs are real.
    #[test]
    fn find_mems_reports_all_occurrence_level_mems() {
        let ref_a = "ACGTACGTAC"; // holds the whole query
        let ref_b = "ACGTACT"; // holds query[0..6), flanked by T != query[6] = G
        let idx = bidir_multi(&[(ref_a, "A"), (ref_b, "B")]);
        let query = encode("ACGTACGTAC");

        let got: Vec<(usize, usize)> = idx
            .find_mems(&query, 2, false)
            .iter()
            .map(|m| (m.query_start, m.query_end))
            .collect();

        assert_eq!(got, vec![(0, 2), (0, 6), (0, 10), (4, 10), (8, 10)]);
        assert_eq!(
            got,
            brute_force_mems_occ(&[encode(ref_a), encode(ref_b)], &query, 2),
            "must agree with the independent brute-force enumerator"
        );
    }

    /// Left-maximality is per-occurrence too: `(4,10)` and `(8,10)` are real MEMs via the
    /// sequence-start occurrences, even though ref_a@4 / ref_a@8 are left-extendable.
    #[test]
    fn find_mems_left_maximality_is_per_occurrence() {
        let idx = bidir_multi(&[("ACGTACGTAC", "A"), ("ACGTACT", "B")]);
        let query = encode("ACGTACGTAC");

        let got: Vec<(usize, usize)> = idx
            .find_mems(&query, 2, false)
            .iter()
            .map(|m| (m.query_start, m.query_end))
            .collect();

        for want in [(4, 10), (8, 10)] {
            assert!(
                got.contains(&want),
                "{want:?} dropped: whole-set left-maximality over-rejects"
            );
        }
    }

    /// `find_mems` against the independent occurrence-level enumerator on randomized
    /// multi-reference corpora.
    ///
    /// The cross-check has to be against a brute-force enumerator, not against `find_smems`:
    /// under whole-set maximality the two functions return near-identical sets, so comparing
    /// them to each other cannot detect this class of defect.
    #[test]
    fn find_mems_matches_brute_force_randomized() {
        let mut state: u64 = 0x5EED_1234_ABCD_9876;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let pick = |set: &[u8], next: &mut dyn FnMut() -> u32| -> String {
            (set[(next() % set.len() as u32) as usize] as char).to_string()
        };
        let rand_seq = |n: usize, set: &[u8], next: &mut dyn FnMut() -> u32| -> String {
            (0..n).map(|_| pick(set, next)).collect()
        };
        let acgt = b"ACGT";
        let iupac = b"ACGTNRY";

        for iter in 0..60 {
            let full = rand_seq(60, acgt, &mut next);
            // Reference set: a full copy, an interior slice, and noise — plus, every other
            // iteration, IUPAC codes so extensions branch into interval sets.
            let a = full.clone();
            let b = full[10..40].to_string();
            let noise = if iter % 2 == 0 {
                rand_seq(40, iupac, &mut next)
            } else {
                rand_seq(40, acgt, &mut next)
            };
            let idx = bidir_multi(&[(&a, "A"), (&b, "B"), (&noise, "NOISE")]);

            let q = encode(&full);
            let refs = vec![encode(&a), encode(&b), encode(&noise)];
            for min_len in [1usize, 3, 8] {
                let got: Vec<(usize, usize)> = idx
                    .find_mems(&q, min_len, false)
                    .iter()
                    .map(|m| (m.query_start, m.query_end))
                    .collect();
                let expected = brute_force_mems_occ(&refs, &q, min_len);
                assert_eq!(
                    got, expected,
                    "iter={iter} min_len={min_len}\nquery={full}\nnoise={noise}"
                );
            }
        }
    }

    /// Every reported position must be an occurrence that is genuinely maximal both ways,
    /// and `match_count` must agree with the number of reported positions.
    #[test]
    fn find_mems_positions_are_maximal_occurrences() {
        let refs = [("ACGTACGTAC", "A"), ("ACGTACT", "B"), ("TTACGTACGG", "C")];
        let idx = bidir_multi(&refs);
        let query = encode("ACGTACGTAC");

        let mems = idx.find_mems(&query, 2, true);
        assert!(!mems.is_empty());

        for mem in &mems {
            let (s, e) = (mem.query_start, mem.query_end);
            assert_eq!(
                mem.positions.len() as u32,
                mem.match_count,
                "MEM [{s},{e}) match_count disagrees with positions"
            );
            for &(id, off) in &mem.positions {
                let seq = encode(refs[id.index()].0);
                let off = off as usize;
                assert!(
                    (0..e - s).all(|k| compat(query[s + k], seq[off + k])),
                    "MEM [{s},{e}) position {id:?}@{off} does not spell the match"
                );
                assert!(
                    s == 0 || off == 0 || !compat(query[s - 1], seq[off - 1]),
                    "MEM [{s},{e}) position {id:?}@{off} is left-extendable"
                );
                assert!(
                    e == query.len()
                        || off + (e - s) == seq.len()
                        || !compat(query[e], seq[off + (e - s)]),
                    "MEM [{s},{e}) position {id:?}@{off} is right-extendable"
                );
            }
        }
    }

    #[test]
    fn find_mems_min_len_filter() {
        let reference = "ACGTTAGCCAGTACGT";
        let query_str = "CGTTAGC";
        let idx = bidir(reference);
        let query = encode(query_str);

        let min_len = 3;
        let mems = idx.find_mems(&query, min_len, false);

        for mem in &mems {
            assert!(
                mem.len() >= min_len,
                "MEM [{},{}] has length {} < min_len {}",
                mem.query_start,
                mem.query_end,
                mem.len(),
                min_len
            );
        }

        // Verify that mems with min_len=1 contains strictly more entries (or equal)
        let mems_all = idx.find_mems(&query, 1, false);
        assert!(
            mems_all.len() >= mems.len(),
            "min_len=1 should return at least as many MEMs as min_len={min_len}"
        );
    }

    // ── N-wildcard tests ──────────────────────────────────────────────────────

    #[test]
    fn n_in_query_matches_any_nucleotide_smem() {
        // Reference has ACGT; query "N" should match all 4 positions.
        let idx = bidir("ACGT");
        let query = encode("N");
        let smems = idx.find_smems(&query, 1, false);
        assert_eq!(smems.len(), 1);
        assert_eq!(smems[0].query_start, 0);
        assert_eq!(smems[0].query_end, 1);
        // N matches A, C, G, T → 4 occurrences total
        assert_eq!(smems[0].match_count, 4);
    }

    #[test]
    fn n_in_query_flanked_by_exact_bases() {
        // Reference "AACAAGAAT"; query "ANT" should match "AAT" (A-N-T where N=A).
        let idx = bidir("AACAAGAAT");
        let query = encode("ANT");
        let smems = idx.find_smems(&query, 1, false);
        // "ANT" with N=A matches "AAT" in the reference (at position 6)
        assert_eq!(smems.len(), 1);
        assert_eq!(smems[0].query_start, 0);
        assert_eq!(smems[0].query_end, 3);
        assert!(smems[0].match_count >= 1);
    }

    #[test]
    fn n_wildcard_locate_returns_all_matching_positions() {
        // Reference "ACGT"; query "N" matches all 4 positions.
        let idx = bidir("ACGT");
        let query = encode("N");
        let smems = idx.find_smems(&query, 1, true);
        assert_eq!(smems.len(), 1);
        let mut positions = smems[0].positions.clone();
        positions.sort();
        assert_eq!(positions.len(), 4);
        // All offsets 0..3 must appear
        let offsets: Vec<u32> = positions.iter().map(|(_, off)| *off).collect();
        for expected in 0u32..4 {
            assert!(offsets.contains(&expected), "missing offset {expected}");
        }
    }

    #[test]
    fn n_wildcard_find_mems_superset() {
        // Every MEM found with an exact query must also appear when the
        // corresponding base is replaced with N (because N ⊇ exact base).
        let idx = bidir("ACGTACGT");
        let exact_query = encode("ACGT");
        let n_query = encode("NCGN"); // N at pos 0 and 3
        let exact_mems = idx.find_mems(&exact_query, 1, false);
        let n_mems = idx.find_mems(&n_query, 1, false);
        // N-query must find at least as many match positions as exact query.
        let exact_count: u32 = exact_mems.iter().map(|m| m.match_count).sum();
        let n_count: u32 = n_mems.iter().map(|m| m.match_count).sum();
        assert!(
            n_count >= exact_count,
            "N-wildcard count {n_count} < exact count {exact_count}"
        );
    }

    #[test]
    fn n_only_query_matches_all_positions() {
        // "NN" in query matches any 2-mer in the reference.
        let idx = bidir("ACGT");
        let query = encode("NN");
        let smems = idx.find_smems(&query, 1, false);
        // Should find exactly one SMEM of length 2 covering all 3 dinucleotides
        assert_eq!(smems.len(), 1);
        assert_eq!(smems[0].query_end - smems[0].query_start, 2);
        assert_eq!(smems[0].match_count, 3); // AC, CG, GT
    }

    #[test]
    fn n_in_reference_is_bidirectional_wildcard() {
        // N in the reference matches any query base (bidirectional wildcard).
        // Reference "ANCGT", query "AC":
        //   "AN" at ref pos 0: query A=ref A (exact), query C=ref N (wildcard) → match
        //   "NC" at ref pos 1: query A=ref N (wildcard), query C=ref C (exact) → match
        let idx = bidir("ANCGT");
        let query = encode("AC");
        let mems = idx.find_mems(&query, 2, false);
        assert_eq!(mems.len(), 1, "should find one length-2 MEM");
        assert_eq!(mems[0].query_start, 0);
        assert_eq!(mems[0].query_end, 2);
        assert_eq!(
            mems[0].match_count, 2,
            "matches 'AN' at ref pos 0 and 'NC' at ref pos 1"
        );
    }

    #[test]
    fn n_in_query_left_maximal() {
        // "NA" in AAAA: N matches A, so "NA" == "AA"; left-maximal only at pos 0
        // since all positions can extend left except the first.
        let idx = bidir("AAAA");
        let query = encode("NA");
        let smems = idx.find_smems(&query, 2, false);
        assert_eq!(smems.len(), 1);
        assert_eq!(smems[0].query_start, 0);
        assert_eq!(smems[0].query_end, 2);
    }

    // ── Regression: SMEM-drop bug (bug-fmidx.md) ──────────────────────────────

    fn bidir_multi(seqs: &[(&str, &str)]) -> BidirFmIndex {
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            ..Default::default()
        };
        let dna: Vec<DnaSequence> = seqs
            .iter()
            .map(|(s, h)| DnaSequence::from_str_with_header(s, h).unwrap())
            .collect();
        BidirFmIndex::build_cpu(&dna, &config).unwrap()
    }

    /// Verbatim reproducer from `bug-fmidx.md`: two overlapping MEMs where neither query
    /// interval contains the other (`[4,30)` and `[5,176)`) are both SMEMs. The old
    /// pivot-advance logic emitted only the left-starting one and dropped the longer seed.
    #[test]
    fn smem_drops_valid_longer_seed() {
        let query = "CGTTCTGGAAGCAATGGCTTTCCTTGAGGAATCCCACCCAGGGATCTTTGAAAACTCTTGT\
                     CTTGAAACGATGGAAGTTGTTCAGCAAACAAGAGTGGACAAACTAACTCAAGGTCGCCAGA\
                     CTTATGACTGGACATTGAATAGAAACCAACCAGCTGCAACTGCTTTGGCCAACA";
        let ref_wrong = &query[4..30]; // 26 bp  -> query[4..30)
        let ref_correct = &query[5..176]; // 171 bp -> query[5..176)

        let idx = bidir_multi(&[(ref_wrong, "REF_WRONG"), (ref_correct, "REF_CORRECT")]);
        let q = encode(query);

        let smems = idx.find_smems(&q, 19, true);
        let mems = idx.find_mems(&q, 19, true);

        let hits = |ms: &[Mem], header: &str| {
            ms.iter().any(|m| {
                m.positions
                    .iter()
                    .any(|&(id, _)| idx.seq_header(id) == Some(header))
            })
        };

        // find_mems already finds REF_CORRECT (sanity: index content is correct).
        assert!(hits(&mems, "REF_CORRECT"));
        // find_smems must now also return the 171 bp SMEM to REF_CORRECT.
        assert!(
            hits(&smems, "REF_CORRECT"),
            "find_smems dropped the valid 171 bp SMEM to REF_CORRECT"
        );
        assert!(hits(&smems, "REF_WRONG"), "find_smems dropped REF_WRONG");
    }

    /// Left/right mirror of the reproducer: the longer seed starts *earlier* and the shorter
    /// one ends later. Both remain SMEMs.
    #[test]
    fn smem_keeps_both_when_shorter_starts_later() {
        let query = "CGTTCTGGAAGCAATGGCTTTCCTTGAGGAATCCCACCCAGGGATCTTTGAAAACTCTTGT\
                     CTTGAAACGATGGAAGTTGTTCAGCAAACAAGAGTGGACAAACTAACTCAAGGTCGCCAGA\
                     CTTATGACTGGACATTGAATAGAAACCAACCAGCTGCAACTGCTTTGGCCAACA";
        // Longer seed [0..171); shorter competing seed [150..176) — neither contains the other.
        let ref_long = &query[0..171];
        let ref_short = &query[150..176];

        let idx = bidir_multi(&[(ref_long, "REF_LONG"), (ref_short, "REF_SHORT")]);
        let q = encode(query);
        let smems = idx.find_smems(&q, 19, true);

        let hits = |header: &str| {
            smems.iter().any(|m| {
                m.positions
                    .iter()
                    .any(|&(id, _)| idx.seq_header(id) == Some(header))
            })
        };
        assert!(hits("REF_LONG"), "dropped the leading long SMEM");
        assert!(hits("REF_SHORT"), "dropped the trailing short SMEM");
    }

    /// `find_smems` must equal the containment-maximal filter of `find_mems` on randomized
    /// queries with planted overlapping seeds to two references. This is the oracle that
    /// keeps the (future) single-pass SMEM algorithm honest.
    #[test]
    fn smems_equal_containment_filtered_mems_randomized() {
        // Tiny deterministic LCG for reproducibility without extra deps.
        let mut state: u64 = 0xDEADBEEFCAFEF00D;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let bases = [b'A', b'C', b'G', b'T'];
        let rand_dna = |n: usize, next: &mut dyn FnMut() -> u32| -> String {
            (0..n)
                .map(|_| bases[(next() % 4) as usize] as char)
                .collect()
        };

        // A few references may carry ambiguity codes so extensions branch into interval
        // sets (the case that makes our SMEM enumeration harder than textbook BWA).
        let iupac = [b'A', b'C', b'G', b'T', b'N', b'R', b'Y'];
        let rand_iupac = |n: usize, next: &mut dyn FnMut() -> u32| -> String {
            (0..n)
                .map(|_| iupac[(next() % iupac.len() as u32) as usize] as char)
                .collect()
        };

        for iter in 0..80 {
            let full = rand_dna(200, &mut next);
            // Two overlapping references carved from the query, plus random flank noise.
            // Every other iteration injects IUPAC/N noise to exercise branching.
            let a = &full[10..60];
            let b = &full[40..160];
            let noise = if iter % 2 == 0 {
                rand_iupac(80, &mut next)
            } else {
                rand_dna(80, &mut next)
            };
            let idx = bidir_multi(&[(a, "A"), (b, "B"), (&noise, "NOISE")]);
            let q = encode(&full);
            let min_len = 15;

            let smems = idx.find_smems(&q, min_len, false);
            let mems = idx.find_mems(&q, min_len, false);

            // Oracle: containment-maximal filter over the MEM intervals.
            let ivs: Vec<(usize, usize)> =
                mems.iter().map(|m| (m.query_start, m.query_end)).collect();
            let mut expected: Vec<(usize, usize)> = ivs
                .iter()
                .filter(|&&(s, e)| {
                    !ivs.iter()
                        .any(|&(s2, e2)| (s2, e2) != (s, e) && s2 <= s && e <= e2)
                })
                .copied()
                .collect();
            expected.sort();

            let mut got: Vec<(usize, usize)> =
                smems.iter().map(|m| (m.query_start, m.query_end)).collect();
            got.sort();

            assert_eq!(
                got, expected,
                "SMEMs != containment-maximal MEMs\nquery={full}"
            );
        }
    }

    /// The pruned/lockstep sweeps against the original algorithms they replace, compared
    /// as raw candidates (bounds, counts, intervals in order, left context) before any
    /// filtering or locating, on both backends (the uncompressed one with its shallow cache,
    /// the run-length one with its rank scan). Strain-like texts with N runs and reads with
    /// substitutions and ambiguity codes, so the early stop, the equal-coverage pruning and
    /// the drop arithmetic all fire.
    #[test]
    fn rewritten_sweeps_match_the_original_candidates() {
        use crate::rl_fm_index::BidirRlFmIndex;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        type Key = (usize, usize, u32, Vec<BidirInterval>, Option<u8>);
        fn keys(raws: Vec<RawMem>) -> Vec<Key> {
            raws.into_iter()
                .map(|r| (r.query_start, r.query_end, r.match_count, r.ivs, r.left_ctx))
                .collect()
        }

        let mut rng = StdRng::seed_from_u64(0x0a11);
        for round in 0..6 {
            let base: Vec<u8> = (0..800).map(|_| b"ACGT"[rng.random_range(0..4)]).collect();
            let texts: Vec<String> = (0..1 + round)
                .map(|_| {
                    let mut t = base.clone();
                    for _ in 0..8 {
                        let at = rng.random_range(0..t.len());
                        t[at] = b"ACGTN"[rng.random_range(0..5)];
                    }
                    String::from_utf8(t).unwrap()
                })
                .collect();
            let seqs: Vec<DnaSequence> = texts
                .iter()
                .map(|t| DnaSequence::from_str(t).unwrap())
                .collect();
            let config = FmIndexConfig {
                sa_sample_rate: 4,
                use_gpu: false,
                ..Default::default()
            };
            let fm = BidirFmIndex::build_cpu(&seqs, &config).unwrap();
            let rl = BidirRlFmIndex::from_bidir_fm_index(&fm);
            for _ in 0..25 {
                let src = texts[rng.random_range(0..texts.len())].as_bytes();
                let start = rng.random_range(0..src.len() - 120);
                let mut read = src[start..start + 120].to_vec();
                for _ in 0..rng.random_range(0..4) {
                    let at = rng.random_range(0..read.len());
                    read[at] = b"ACGTACGTNR"[rng.random_range(0..10)];
                }
                let q = encode(std::str::from_utf8(&read).unwrap());
                for min_len in [1, 6, 19] {
                    let smems = keys(fm.collect_smem_candidates_reference(&q, min_len));
                    let mems = keys(fm.collect_mem_candidates_reference(&q, min_len));
                    assert_eq!(keys(fm.collect_smem_candidates(&q, min_len)), smems);
                    assert_eq!(keys(rl.collect_smem_candidates(&q, min_len)), smems);
                    assert_eq!(keys(fm.collect_mem_candidates(&q, min_len)), mems);
                    assert_eq!(keys(rl.collect_mem_candidates(&q, min_len)), mems);
                }
            }
        }
    }

    /// A reference dense in ambiguity codes multiplies every k-mer's wildcard variants:
    /// the shallow cache must stop at its budget (shallower, never larger), and the sweeps
    /// must still match the uncompressed index exactly.
    #[test]
    fn shallow_cache_respects_its_budget_on_iupac_dense_text() {
        use crate::rl_fm_index::BidirRlFmIndex;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(0xb0d9);
        let text: String = (0..70_000)
            .map(|_| ['A', 'C', 'G', 'T', 'N', 'B', 'D', 'H', 'V'][rng.random_range(0..9)])
            .collect();
        let seqs = vec![DnaSequence::from_str(&text).unwrap()];
        let config = FmIndexConfig {
            sa_sample_rate: 8,
            use_gpu: false,
            ..Default::default()
        };
        let fm = BidirFmIndex::build_cpu(&seqs, &config).unwrap();
        let rl = BidirRlFmIndex::from_bidir_fm_index(&fm);
        assert!(rl.shallow.ivs.len() <= SHALLOW_MAX_IVS);
        assert!(
            rl.shallow.depth < SHALLOW_MAX_DEPTH as usize,
            "budget never hit"
        );
        // Few, short reads: every query position fans out over the wildcard variants.
        for _ in 0..3 {
            let at = rng.random_range(0..text.len() - 30);
            let q = encode(&text[at..at + 30]);
            let min_len = 8;
            assert_eq!(
                rl.find_smems(&q, min_len, true),
                fm.find_smems(&q, min_len, true)
            );
            assert_eq!(
                rl.find_mems(&q, min_len, true),
                fm.find_mems(&q, min_len, true)
            );
        }
    }
}
