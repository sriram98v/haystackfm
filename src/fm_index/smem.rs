use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::bidir_index::BidirFmIndex;
use crate::fm_index::FmIndex;

/// A Maximal Exact Match (MEM) between a query and the indexed reference.
///
/// A MEM is a substring of the query that:
/// 1. Occurs at least once in the reference.
/// 2. Is **left-maximal**: extending one base to the left removes all occurrences.
/// 3. Is **right-maximal**: extending one base to the right removes all occurrences.
///
/// A Super-Maximal Exact Match (SMEM) additionally satisfies:
/// 4. No other MEM with the same right boundary has a larger count.
///
/// In practice `find_smems` finds all MEMs that are simultaneously left- and
/// right-maximal (i.e., SMEMs in the BWA-MEM sense).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mem {
    /// Start position in the query (0-based, inclusive).
    pub query_start: usize,
    /// End position in the query (0-based, exclusive).
    pub query_end: usize,
    /// Number of occurrences in the reference text.
    pub match_count: u32,
    /// Reference positions (populated only when `locate = true`).
    /// Each entry is `(sequence_id, position_within_sequence)`.
    pub positions: Vec<(String, u32)>,
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
    /// Complexity: O(|query|² × α) where α = ALPHABET_SIZE = 6.
    /// In practice much better: once a long SMEM is found the inner loop
    /// advances to the SMEM's right boundary.
    ///
    /// # Parameters
    ///
    /// - `query`: encoded DNA bases (values 1–4; 0 = sentinel, should not appear).
    /// - `min_len`: discard matches shorter than this (must be ≥ 1).
    /// - `locate`: if `true`, populate `Mem::positions` with reference positions.
    ///
    /// # Returns
    ///
    /// SMEMs in order of increasing `query_start`.  Duplicate `(start, end)` pairs
    /// are deduplicated.
    pub fn find_smems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
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
            let contained = intervals.iter().enumerate().any(|(j, &(s2, e2))| {
                j != idx && s2 <= s && e <= e2 && (s2, e2) != (s, e)
            });
            if !contained {
                smems.push(self.locate_raw(raw, locate));
            }
        }
        smems
    }

    /// Find all MEMs (not just super-maximal) of length ≥ `min_len`.
    ///
    /// Unlike `find_smems`, this does NOT advance past the right boundary after
    /// finding a MEM, so overlapping MEMs from different starting positions are
    /// all reported.
    ///
    /// Complexity: O(|query|² × α).
    pub fn find_mems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
        if query.is_empty() || min_len == 0 {
            return vec![];
        }

        let mut raws = self.collect_raw_mems(query, min_len);
        raws.sort_by_key(|m| (m.query_start, m.query_end));
        raws.dedup_by_key(|m| (m.query_start, m.query_end));
        raws.into_iter()
            .map(|raw| self.locate_raw(raw, locate))
            .collect()
    }

    /// Find the right-maximal, left-maximal seed starting at query position `i`.
    ///
    /// `N` bases in `query` are treated as wildcards matching any of A/C/G/T.
    ///
    /// Returns `(Some(Mem), next_i)` on success where `next_i = i + 1` (the
    /// `find_smems` outer loop may choose a larger advance).
    /// Returns `(None, i + 1)` when no valid seed exists at `i`.
    /// Collect the raw MEM anchored at every start position (one per left-maximal start),
    /// without resolving positions. Shared by [`find_mems`](Self::find_mems) and
    /// [`find_smems`](Self::find_smems).
    fn collect_raw_mems(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
        (0..query.len())
            .filter_map(|i| self.raw_mem_from(query, i, min_len))
            .collect()
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

    /// Right-maximal, left-maximal seed starting at query position `i`, *without* resolving
    /// positions. `N` bases in `query` are treated as wildcards matching any of A/C/G/T.
    ///
    /// Returns `None` when no valid seed of length ≥ `min_len` exists at `i`. The returned
    /// `query_start` is always `i`, so distinct `i` yield distinct MEMs.
    fn raw_mem_from(&self, query: &[u8], i: usize, min_len: usize) -> Option<RawMem> {
        let n = query.len();
        // Track a set of active intervals; N-wildcard may produce multiple branches.
        // `ivs` always holds the last non-empty extension, so no per-step snapshot/clone is
        // needed — on break it is exactly the accepted right-maximal interval set.
        let mut ivs: Vec<BidirInterval> = vec![self.full_interval()];
        let mut j = i;
        let mut matched = false;

        // Right extension phase: uses the reverse index.
        while j < n {
            let next = extend_multi_right(&ivs, query[j], &self.rev);
            if next.is_empty() {
                break;
            }
            ivs = next;
            j += 1;
            matched = true;
        }

        if !matched {
            return None;
        }
        let (valid_ivs, end) = (ivs, j);

        if end - i < min_len {
            return None;
        }

        // Left-maximality check: uses the forward index. Not left-maximal when ANY interval
        // in the set can be extended left.
        let left_maximal =
            i == 0 || extend_multi_left(&valid_ivs, query[i - 1], &self.fwd).is_empty();
        if !left_maximal {
            return None;
        }

        let match_count: u32 = valid_ivs.iter().map(|iv| iv.size()).sum();

        Some(RawMem {
            query_start: i,
            query_end: end,
            match_count,
            ivs: valid_ivs,
        })
    }

    /// Resolve a [`RawMem`] into a public [`Mem`], locating reference positions only when
    /// `locate` is set.
    fn locate_raw(&self, raw: RawMem, locate: bool) -> Mem {
        let positions = if locate {
            raw.ivs
                .iter()
                .flat_map(|iv| self.locate_interval(iv))
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
}

/// A MEM before its reference positions are resolved: query interval, occurrence count, and
/// the accepted bidirectional SA intervals (kept so only survivors need locating).
struct RawMem {
    query_start: usize,
    query_end: usize,
    match_count: u32,
    ivs: Vec<BidirInterval>,
}

/// Total number of text occurrences represented by an interval set.
fn coverage(ivs: &[BidirInterval]) -> u32 {
    ivs.iter().map(|iv| iv.size()).sum()
}

/// Extend each interval in `ivs` right by `c`, using the index's alphabet compatibility.
fn extend_multi_right(ivs: &[BidirInterval], c: u8, rev: &FmIndex) -> Vec<BidirInterval> {
    let bases = (rev.alphabet_fns.compatible_fn)(c);
    let mut result = Vec::new();
    for &base in bases {
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
    let bases = (fwd.alphabet_fns.compatible_fn)(c);
    let mut result = Vec::new();
    for &base in bases {
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
        assert_eq!(
            positions,
            vec![("seq_0".to_string(), 0), ("seq_0".to_string(), 4)]
        );
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

        let expected = brute_force_mems(reference, query_str, 1);

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

        let mems = idx.find_mems(&query, 1, false);
        assert!(!mems.is_empty(), "expected at least one MEM");

        for mem in &mems {
            let s = mem.query_start;
            let e = mem.query_end;
            let matched = &query_str[s..e];

            assert!(
                reference.contains(matched),
                "MEM [{s},{e}) '{matched}' not found in reference"
            );

            let right_maximal = e == query_str.len() || !reference.contains(&query_str[s..e + 1]);
            assert!(
                right_maximal,
                "MEM [{s},{e}) '{matched}' is not right-maximal: extending right still matches"
            );

            let left_maximal = s == 0 || !reference.contains(&query_str[s - 1..e]);
            assert!(
                left_maximal,
                "MEM [{s},{e}) '{matched}' is not left-maximal: extending left still matches"
            );
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
            ms.iter()
                .any(|m| m.positions.iter().any(|(h, _)| h == header))
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
            smems
                .iter()
                .any(|m| m.positions.iter().any(|(h, _)| h == header))
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
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let bases = [b'A', b'C', b'G', b'T'];
        let rand_dna = |n: usize, next: &mut dyn FnMut() -> u32| -> String {
            (0..n).map(|_| bases[(next() % 4) as usize] as char).collect()
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
                    !ivs.iter().any(|&(s2, e2)| {
                        (s2, e2) != (s, e) && s2 <= s && e <= e2
                    })
                })
                .copied()
                .collect();
            expected.sort();

            let mut got: Vec<(usize, usize)> =
                smems.iter().map(|m| (m.query_start, m.query_end)).collect();
            got.sort();

            assert_eq!(got, expected, "SMEMs != containment-maximal MEMs\nquery={full}");
        }
    }
}
