use crate::alphabet::{A, C, G, N, T};
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

        let n = query.len();
        let mut smems: Vec<Mem> = Vec::new();
        let mut i = 0;

        while i < n {
            let (mem_opt, next_i) = self.smem_from(query, i, min_len, locate);

            if let Some(mem) = mem_opt {
                // Advance past the SMEM's right boundary to avoid finding
                // redundant sub-MEMs that are dominated by this one.
                let end = mem.query_end;
                smems.push(mem);
                i = end;
            } else {
                i = next_i;
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

        let n = query.len();
        let mut mems: Vec<Mem> = Vec::new();
        // Deduplicate by (start, end) since multiple i values can produce the same MEM.
        let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

        for i in 0..n {
            let (mem_opt, _) = self.smem_from(query, i, min_len, locate);
            if let Some(mem) = mem_opt {
                let key = (mem.query_start, mem.query_end);
                if seen.insert(key) {
                    mems.push(mem);
                }
            }
        }

        mems.sort_by_key(|m| (m.query_start, m.query_end));
        mems
    }

    /// Find the right-maximal, left-maximal seed starting at query position `i`.
    ///
    /// `N` bases in `query` are treated as wildcards matching any of A/C/G/T.
    ///
    /// Returns `(Some(Mem), next_i)` on success where `next_i = i + 1` (the
    /// `find_smems` outer loop may choose a larger advance).
    /// Returns `(None, i + 1)` when no valid seed exists at `i`.
    fn smem_from(
        &self,
        query: &[u8],
        i: usize,
        min_len: usize,
        locate: bool,
    ) -> (Option<Mem>, usize) {
        let n = query.len();
        // Track a set of active intervals; N-wildcard may produce multiple branches.
        let mut ivs: Vec<BidirInterval> = vec![self.full_interval()];
        let mut j = i;
        let mut last_valid: Option<(Vec<BidirInterval>, usize)> = None;

        // Right extension phase: uses the reverse index.
        while j < n {
            let next = extend_multi_right(&ivs, query[j], &self.rev);
            if next.is_empty() {
                break;
            }
            ivs = next;
            j += 1;
            last_valid = Some((ivs.clone(), j));
        }

        let (valid_ivs, end) = match last_valid {
            Some(v) => v,
            None => return (None, i + 1),
        };

        let len = end - i;
        if len < min_len {
            return (None, i + 1);
        }

        // Left-maximality check: uses the forward index.
        // Not left-maximal when ANY interval in the set can be extended left.
        let left_maximal = if i == 0 {
            true
        } else {
            extend_multi_left(&valid_ivs, query[i - 1], &self.fwd).is_empty()
        };

        if !left_maximal {
            return (None, i + 1);
        }

        let match_count: u32 = valid_ivs.iter().map(|iv| iv.size()).sum();

        let positions = if locate {
            valid_ivs
                .iter()
                .flat_map(|iv| self.locate_interval(iv))
                .collect()
        } else {
            Vec::new()
        };

        let mem = Mem {
            query_start: i,
            query_end: end,
            match_count,
            positions,
        };

        (Some(mem), i + 1)
    }
}

/// Bases that a query byte `c` should be matched against in the reference index.
///
/// Two wildcard rules:
/// - Query `N` matches any reference base → try A, C, G, T, N.
/// - Reference `N` matches any query base → always include N in the candidate set.
fn wildcard_bases(c: u8) -> &'static [u8] {
    const ACGTN: &[u8] = &[A, C, G, T, N];
    const AN: &[u8] = &[A, N];
    const CN: &[u8] = &[C, N];
    const GN: &[u8] = &[G, N];
    const TN: &[u8] = &[T, N];
    match c {
        x if x == N => ACGTN,
        x if x == A => AN,
        x if x == C => CN,
        x if x == G => GN,
        x if x == T => TN,
        _ => &[],
    }
}

/// Extend each interval in `ivs` right by `c`, expanding N to A/C/G/T.
fn extend_multi_right(ivs: &[BidirInterval], c: u8, rev: &FmIndex) -> Vec<BidirInterval> {
    let bases = wildcard_bases(c);
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

/// Extend each interval in `ivs` left by `c`, expanding N to A/C/G/T.
fn extend_multi_left(ivs: &[BidirInterval], c: u8, fwd: &FmIndex) -> Vec<BidirInterval> {
    let bases = wildcard_bases(c);
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
}
