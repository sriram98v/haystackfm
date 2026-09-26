use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::bidir_index::BidirFmIndex;
use crate::fm_index::seq_id::SeqId;
use crate::fm_index::FmIndex;

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
        self.smem_raws(query, min_len)
            .into_iter()
            .map(|raw| self.locate_raw(raw, locate))
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
        self.mem_raws(query, min_len)
            .into_iter()
            .map(|raw| self.locate_raw(raw, locate))
            .collect()
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

    /// Collect every occurrence-level MEM candidate, without resolving positions.
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
    /// Used only by [`find_mems`](Self::find_mems); [`find_smems`](Self::find_smems) has its
    /// own generator in [`collect_smem_candidates`](Self::collect_smem_candidates).
    fn collect_mem_candidates(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
        let n = query.len();
        let mut out: Vec<RawMem> = Vec::new();

        for i in 0..n {
            // All occurrences of a symbol compatible with `query[i]`.
            let mut curr = extend_multi_right(&[self.full_interval()], query[i], &self.rev);
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
                let next = extend_multi_right(&curr, query[j], &self.rev);
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

    /// The occurrences in `ivs` that canNOT be extended right by `c`.
    ///
    /// Extends by every symbol *incompatible* with `c`. The results are disjoint (distinct
    /// following symbols), and the sentinel is incompatible with every base, so occurrences
    /// sitting at the end of a reference are captured here rather than silently lost.
    fn drop_set(&self, ivs: &[BidirInterval], c: u8) -> Vec<BidirInterval> {
        let mut out = Vec::new();
        for sym in self.rev.alphabet_fns.compatible(c).complement().iter() {
            for iv in ivs {
                if let Some(ext) = iv.extend_right(sym, &self.rev) {
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
            Some(c) => coverage(&ivs) - coverage(&extend_multi_left(&ivs, c, &self.fwd)),
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
    /// recording every right-maximal prefix `[pivot, e)` (a right-end where the interval set's
    /// coverage drops), then backward-extends each recorded prefix to its true left boundary
    /// and keeps it if the result verifies as a MEM of length ≥ `min_len`.
    ///
    /// Correctness: max right-reach is monotonic in the start position, so any SMEM starting
    /// after a pivot must extend past that pivot's reach and therefore covers the next pivot —
    /// hence every SMEM is anchored by some visited pivot and appears here (possibly alongside
    /// non-super MEMs, which the caller's containment filter removes).
    fn collect_smem_candidates(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
        let n = query.len();
        let mut out: Vec<RawMem> = Vec::new();
        let mut pivot = 0;

        while pivot < n {
            // Forward extension from `pivot`, collecting right-maximal prefixes as
            // (interval set for query[pivot..end), end).
            let mut curr = extend_multi_right(&[self.full_interval()], query[pivot], &self.rev);
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
                let next = extend_multi_right(&curr, query[j], &self.rev);
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
                    end == n || extend_multi_right(&bivs, query[end], &self.rev).is_empty();
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

    /// Extend an interval set as far left as possible from left boundary `from`, returning the
    /// widened set and the resulting start position. Stops when the next left extension is
    /// empty (left-maximal) or the query start is reached.
    fn extend_left_maximally(
        &self,
        mut ivs: Vec<BidirInterval>,
        query: &[u8],
        from: usize,
    ) -> (Vec<BidirInterval>, usize) {
        let mut start = from;
        while start > 0 {
            let next = extend_multi_left(&ivs, query[start - 1], &self.fwd);
            if next.is_empty() {
                break;
            }
            ivs = next;
            start -= 1;
        }
        (ivs, start)
    }

    /// Resolve a [`RawMem`] into a public [`Mem`], locating reference positions only when
    /// `locate` is set.
    ///
    /// When `left_ctx` is set, occurrences that are left-extendable are dropped here so that
    /// `positions` holds exactly the maximal occurrences already counted by `match_count`.
    fn locate_raw(&self, raw: RawMem, locate: bool) -> Mem {
        let positions = if locate {
            raw.ivs
                .iter()
                .flat_map(|iv| self.locate_interval(iv))
                .filter(|&(id, off)| self.is_left_maximal_at(id, off, raw.left_ctx))
                .collect()
        } else {
            Vec::new()
        };
        Mem {
            query_start: raw.query_start,
            query_end: raw.query_end,
            match_count: raw.match_count,
            positions,
        }
    }

    /// Whether the occurrence starting at `off` in reference `id` is left-maximal.
    ///
    /// `left_ctx` is `None` for matches anchored at the query start and for candidates whose
    /// intervals were already extended left as far as they go, both of which are maximal by
    /// construction. Reads the retained forward text, which a `BidirFmIndex` always keeps
    /// (only the reverse half calls `forget_text`); if it is unavailable the occurrence is
    /// kept rather than guessed away.
    fn is_left_maximal_at(&self, id: SeqId, off: u32, left_ctx: Option<u8>) -> bool {
        let Some(c) = left_ctx else { return true };
        if off == 0 {
            return true;
        }
        match self.sequence(id) {
            Some(seq) => !self
                .fwd
                .alphabet_fns
                .compatible(c)
                .contains(seq[off as usize - 1]),
            None => true,
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
struct RawMem {
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
fn extend_multi_right(ivs: &[BidirInterval], c: u8, rev: &FmIndex) -> Vec<BidirInterval> {
    let mut result = Vec::new();
    for base in rev.alphabet_fns.compatible(c).iter() {
        for iv in ivs {
            if let Some(ext) = iv.extend_right(base, rev) {
                result.push(ext);
            }
        }
    }
    result
}

/// Extend each interval in `ivs` left by `c`, using the index's alphabet compatibility.
fn extend_multi_left(ivs: &[BidirInterval], c: u8, fwd: &FmIndex) -> Vec<BidirInterval> {
    let mut result = Vec::new();
    for base in fwd.alphabet_fns.compatible(c).iter() {
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
}
