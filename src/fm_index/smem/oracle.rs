//! The original sweeps, kept as the oracles the pruned MEM and lockstep SMEM sweeps are
//! tested against. Generic over the backend through its [`BidirSearch`] hooks only, so they
//! run unchanged on every index.

use super::{coverage, BidirSearch, RawMem};

/// Extend each cursor in `ivs` right by every symbol compatible with `c`, symbols
/// ascending, cursors in input order.
pub(super) fn extend_multi_right<S: BidirSearch + ?Sized>(
    s: &S,
    ivs: &[S::Cursor],
    c: u8,
) -> Vec<S::Cursor> {
    let mut result = Vec::new();
    for base in s.search_alphabet().compatible(c).iter() {
        result.extend(ivs.iter().filter_map(|iv| s.search_extend_right(iv, base)));
    }
    result
}

/// Extend each cursor in `ivs` left by every symbol compatible with `c`, symbols
/// ascending, cursors in input order.
pub(super) fn extend_multi_left<S: BidirSearch + ?Sized>(
    s: &S,
    ivs: &[S::Cursor],
    c: u8,
) -> Vec<S::Cursor> {
    let mut result = Vec::new();
    for base in s.search_alphabet().compatible(c).iter() {
        result.extend(ivs.iter().filter_map(|iv| s.search_extend_left(iv, base)));
    }
    result
}

/// The original MEM sweep.
///
/// For each start `i`, forward-extends the full occurrence set of `query[i..j)`. At every
/// step where the occurrence count drops, the occurrences that failed to extend are
/// exactly the ones right-maximal at `j`; [`drop_set`] recovers them as cursors.
/// `j == query.len()` is right-maximal for the whole surviving set. [`push_mem_candidate`]
/// then applies the per-occurrence left-maximality filter.
///
/// Completeness: `curr` at step `j` is precisely the occurrence set of `query[i..j)`, so
/// every MEM `(i,j)` has its witnessing occurrence in the drop set at `j`. Soundness:
/// anything emitted has an occurrence maximal in both directions.
pub(super) fn collect_mem_candidates_reference<S: BidirSearch + ?Sized>(
    s: &S,
    query: &[u8],
    min_len: usize,
) -> Vec<RawMem<S::Cursor>> {
    let n = query.len();
    let mut out = Vec::new();

    for i in 0..n {
        // All occurrences of a symbol compatible with `query[i]`.
        let mut curr = extend_multi_right(s, &[s.search_full_cursor()], query[i]);
        if curr.is_empty() {
            continue;
        }
        let mut j = i + 1;
        loop {
            if j == n {
                // Query exhausted: every surviving occurrence is right-maximal.
                push_mem_candidate(s, &mut out, query, (i, n), curr, min_len);
                break;
            }
            let next = extend_multi_right(s, &curr, query[j]);
            if coverage(&next) != coverage(&curr) {
                let dropped = drop_set(s, &curr, query[j]);
                push_mem_candidate(s, &mut out, query, (i, j), dropped, min_len);
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

/// The occurrences in `ivs` that canNOT be extended right by `c`.
///
/// The children by every symbol *incompatible* with `c`. The results are disjoint (distinct
/// following symbols), and the sentinel is incompatible with every base, so occurrences
/// sitting at the end of a reference are captured here rather than silently lost.
fn drop_set<S: BidirSearch + ?Sized>(s: &S, ivs: &[S::Cursor], c: u8) -> Vec<S::Cursor> {
    let children: Vec<_> = ivs.iter().map(|iv| s.search_children_right(iv)).collect();
    let mut out = Vec::new();
    for sym in s.search_alphabet().compatible(c).complement().iter() {
        out.extend(children.iter().filter_map(|ch| ch[sym as usize]));
    }
    out
}

/// Apply `min_len` and the per-occurrence left-maximality filter, then record the MEM.
///
/// `ivs` holds occurrences already known to be right-maximal at `end`. [`extend_multi_left`]
/// walks the *compatible* bases of `query[start-1]`, and its results are disjoint, so
/// subtracting its coverage counts exactly the occurrences that are left-extendable —
/// no interval subtraction and no sentinel special case (a reference start is preceded by
/// a sentinel, which is never compatible with a base).
fn push_mem_candidate<S: BidirSearch + ?Sized>(
    s: &S,
    out: &mut Vec<RawMem<S::Cursor>>,
    query: &[u8],
    (start, end): (usize, usize),
    ivs: Vec<S::Cursor>,
    min_len: usize,
) {
    if end - start < min_len || ivs.is_empty() {
        return;
    }
    let left_ctx = (start > 0).then(|| query[start - 1]);
    let match_count = match left_ctx {
        None => coverage(&ivs),
        Some(c) => coverage(&ivs) - coverage(&extend_multi_left(s, &ivs, c)),
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

/// Single-pass BWA-MEM collection of MEM candidates that contain every SMEM.
///
/// Visits pivots left-to-right, advancing each time by the pivot's *forward reach* (the
/// end of the longest match anchored at the pivot). At each pivot it forward-extends,
/// recording every right-maximal prefix `[pivot, e)` (a right-end where the cursor set's
/// coverage drops), then backward-extends each recorded prefix to its true left boundary
/// and keeps it if the result verifies as a MEM of length ≥ `min_len`.
///
/// Correctness: max right-reach is monotonic in the start position, so any SMEM starting
/// after a pivot must extend past that pivot's reach and therefore covers the next pivot —
/// hence every SMEM is anchored by some visited pivot and appears here (possibly alongside
/// non-super MEMs, which the caller's containment filter removes).
pub(super) fn collect_smem_candidates_reference<S: BidirSearch + ?Sized>(
    s: &S,
    query: &[u8],
    min_len: usize,
) -> Vec<RawMem<S::Cursor>> {
    let n = query.len();
    let mut out = Vec::new();
    let mut pivot = 0;

    while pivot < n {
        // Forward extension from `pivot`, collecting right-maximal prefixes as
        // (cursor set for query[pivot..end), end).
        let mut curr = extend_multi_right(s, &[s.search_full_cursor()], query[pivot]);
        if curr.is_empty() {
            pivot += 1; // query[pivot] absent from the text
            continue;
        }
        let mut cov = coverage(&curr);
        let mut prefixes: Vec<(Vec<S::Cursor>, usize)> = Vec::new();
        let mut j = pivot + 1;
        loop {
            if j == n {
                prefixes.push((curr, n));
                break;
            }
            let next = extend_multi_right(s, &curr, query[j]);
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
            let (bivs, start) = extend_left_maximally(s, ivs, query, pivot);
            if end - start < min_len {
                continue;
            }
            // Verify right-maximality (left-maximality is guaranteed by the backward stop):
            // the whole set must fail to extend right by query[end].
            let right_maximal = end == n || extend_multi_right(s, &bivs, query[end]).is_empty();
            if !right_maximal {
                continue;
            }
            out.push(RawMem {
                query_start: start,
                query_end: end,
                match_count: coverage(&bivs),
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

/// Extend a cursor set as far left as possible from left boundary `from`, returning the
/// widened set and the resulting start position. Stops when the next left extension is
/// empty (left-maximal) or the query start is reached.
fn extend_left_maximally<S: BidirSearch + ?Sized>(
    s: &S,
    mut ivs: Vec<S::Cursor>,
    query: &[u8],
    from: usize,
) -> (Vec<S::Cursor>, usize) {
    let mut start = from;
    while start > 0 {
        let next = extend_multi_left(s, &ivs, query[start - 1]);
        if next.is_empty() {
            break;
        }
        ivs = next;
        start -= 1;
    }
    (ivs, start)
}
