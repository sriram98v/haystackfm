//! Depth-k prefix lookup table for FM-index backward search.
//!
//! Stores, for every core-symbol k-mer of a fixed depth, the SA intervals of the reference
//! occurrences that match it under the index's alphabet. `backward_search` seeds from this
//! table when the last `depth` query characters are all core symbols, skipping those
//! `depth` character steps entirely (O(1) vs O(depth × rank_calls) for plain DNA queries).
//!
//! Each entry holds the **exact** k-mer's interval in slot 0 (so
//! `BidirFmIndex::lookup_interval` can seed a cursor with literal semantics) followed by the
//! intervals of the wildcard variants — reference stretches that match the k-mer only through
//! ambiguity codes such as `N`. On a pure-ACGT reference, or under `ExactDna`, there are no
//! variants and the table is one interval per entry. Where a k-mer has more than
//! [`MAX_VARIANTS_PER_ENTRY`] variants the entry is marked incomplete: slot 0 stays valid,
//! but `backward_search` ignores the entry and runs the full search instead.
//!
//! Default core symbols are ACGT (radix 4): memory = `4 × (4^depth + 1)` bytes of offsets
//! plus 8 bytes per stored interval — with no variants, `12 × 4^depth` bytes
//! (depth=10 → ~12 MB, depth=13 → ~800 MB).

use crate::alphabet::{AlphabetFns, SymbolSet};
use crate::c_array::CArray;
use crate::occ::OccTable;

/// Upper bound on the wildcard-variant intervals stored per k-mer entry. Entries that would
/// exceed it are marked incomplete and backward search falls back to a full search.
pub const MAX_VARIANTS_PER_ENTRY: usize = 16;

/// Bit 31 of an entry's start offset: the entry's variants were dropped (here or at an
/// ancestor), so only its exact interval is trustworthy.
const INCOMPLETE: u32 = 1 << 31;
const OFFSET_MASK: u32 = !INCOMPLETE;

/// A fixed-depth table mapping every core-symbol k-mer to the SA intervals matching it.
///
/// Entries are stored in CSR form: `offsets` has `radix^depth + 1` elements and entry `e`
/// owns `intervals[offsets[e] .. offsets[e + 1]]` (offsets masked by `OFFSET_MASK`). Bit 31
/// of `offsets[e]` is the entry's `INCOMPLETE` flag.
///
/// Entry invariants: an entry with no intervals matches nothing. Otherwise slot 0 is the
/// exact k-mer's interval (`(0, 0)` when the exact k-mer does not occur but a variant does),
/// and every later slot is a non-empty variant interval; all slots are pairwise disjoint.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupTable {
    /// Number of characters covered (the k in k-mer).
    pub depth: u32,
    /// The core (unambiguous) symbols, the base-`radix` digits of the k-mer index in
    /// ascending code order.
    core: SymbolSet,
    /// CSR entry starts, `radix^depth + 1` long; bit 31 flags an incomplete entry.
    #[serde(with = "crate::serde_raw::u32s")]
    offsets: Vec<u32>,
    /// The `(lo, hi)` intervals of every entry, back to back.
    #[serde(with = "crate::serde_raw::u32_pairs")]
    intervals: Vec<(u32, u32)>,
}

/// One entry of a [`LookupTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookupHit<'a> {
    /// Slot 0 is the exact k-mer's interval (possibly empty, `lo >= hi`); the rest are the
    /// non-empty intervals of its wildcard variants. Empty when nothing matches.
    pub intervals: &'a [(u32, u32)],
    /// False when variants were dropped for exceeding [`MAX_VARIANTS_PER_ENTRY`]: the exact
    /// slot is still right, but the entry must not seed a compatibility-aware search.
    pub complete: bool,
}

impl LookupTable {
    /// Build the table by BFS over core-symbol k-mers up to `depth` using the
    /// already-constructed C-array and Occ table, fanning each core symbol out over the
    /// reference codes it matches under `fns`.
    ///
    /// Cost: O(radix^depth × variants) rank operations — ~1 M for radix=4, depth=10 on a
    /// pure-ACGT reference. Deterministic, so two builds of the same index are byte-identical.
    pub fn build(
        depth: u32,
        text_len: u32,
        c_array: &CArray,
        occ: &OccTable,
        fns: &AlphabetFns,
    ) -> Self {
        assert!(depth > 0, "lookup depth must be ≥ 1");
        let core: Vec<u8> = fns.core().iter().collect();
        assert!(!core.is_empty(), "core symbols must not be empty");
        let present = c_array.present_symbols(text_len);
        // Per core symbol, the present reference codes it matches (the exact code first).
        let fanout: Vec<Vec<u8>> = core
            .iter()
            .map(|&s| fns.compatible(s).intersection(present).iter().collect())
            .collect();

        let lf = |(lo, hi): (u32, u32), r: u8| -> (u32, u32) {
            let c_val = c_array.get(r);
            let (rank_lo, rank_hi) = occ.rank_pair(r, lo, hi);
            (c_val + rank_lo, c_val + rank_hi)
        };

        // Level 0: the single empty k-mer, whose exact interval is the whole SA.
        let mut cur_offsets: Vec<u32> = vec![0, 1];
        let mut cur_intervals: Vec<(u32, u32)> = vec![(0, text_len)];

        for _level in 1..=depth as usize {
            let parents = cur_offsets.len() - 1;
            let mut next_offsets: Vec<u32> = Vec::with_capacity(parents * core.len() + 1);
            let mut next_intervals: Vec<(u32, u32)> = Vec::with_capacity(cur_intervals.len() * 2);

            for node in 0..parents {
                let (start, end, parent_flagged) = entry_bounds(&cur_offsets, node);
                let parent = &cur_intervals[start..end];

                for (digit, &s) in core.iter().enumerate() {
                    let mark = next_intervals.len() as u32;
                    // A parent with no intervals has no children; an unflagged one is a
                    // definite miss, a flagged one stays flagged so search never seeds from it.
                    if parent.is_empty() {
                        next_offsets.push(mark | if parent_flagged { INCOMPLETE } else { 0 });
                        continue;
                    }

                    let exact = normalize(lf(parent[0], s));
                    let mut variants: Vec<(u32, u32)> = Vec::new();
                    if !parent_flagged {
                        for (slot, &p) in parent.iter().enumerate() {
                            if p.0 >= p.1 {
                                continue;
                            }
                            for &r in &fanout[digit] {
                                if slot == 0 && r == s {
                                    continue;
                                }
                                let iv = lf(p, r);
                                if iv.0 < iv.1 {
                                    variants.push(iv);
                                }
                            }
                        }
                    }
                    let flagged = parent_flagged || variants.len() > MAX_VARIANTS_PER_ENTRY;
                    if flagged {
                        variants.clear();
                    }

                    next_offsets.push(mark | if flagged { INCOMPLETE } else { 0 });
                    if exact.0 < exact.1 || !variants.is_empty() {
                        next_intervals.push(exact);
                        next_intervals.extend_from_slice(&variants);
                    }
                }
            }
            next_offsets.push(next_intervals.len() as u32);
            cur_offsets = next_offsets;
            cur_intervals = next_intervals;
        }

        assert!(
            cur_intervals.len() < INCOMPLETE as usize,
            "lookup table too large: {} intervals",
            cur_intervals.len()
        );
        Self {
            depth,
            core: fns.core(),
            offsets: cur_offsets,
            intervals: cur_intervals,
        }
    }

    /// Look up the entry for a slice of codes exactly `depth` long.
    ///
    /// `codes` is ordered left-to-right (as the pattern appears). The BFS consumes the
    /// k-mer right to left (backward-search order), so the rightmost code is the most
    /// significant base-`radix` digit of the entry index and the leftmost the least.
    ///
    /// Returns `None` if `codes.len() != depth` or any symbol is not a core symbol
    /// (caller falls back to full search).
    #[inline]
    pub fn get(&self, codes: &[u8]) -> Option<LookupHit<'_>> {
        if codes.len() != self.depth as usize {
            return None;
        }
        let radix = self.core.len() as usize;
        let mut idx = 0usize;
        // Rightmost code first: it ends up in the most significant digit.
        for &c in codes.iter().rev() {
            if !self.core.contains(c) {
                return None;
            }
            let digit = self.core.intersection(SymbolSet::below(c)).len() as usize;
            idx = idx * radix + digit;
        }
        let (start, end, flagged) = entry_bounds(&self.offsets, idx);
        Some(LookupHit {
            intervals: &self.intervals[start..end],
            complete: !flagged,
        })
    }

    /// Number of stored intervals (exact slots and variants) across all entries.
    pub fn num_intervals(&self) -> usize {
        self.intervals.len()
    }
}

/// `(start, end, incomplete)` of entry `e` in a CSR offsets array.
#[inline]
fn entry_bounds(offsets: &[u32], e: usize) -> (usize, usize, bool) {
    let raw = offsets[e];
    (
        (raw & OFFSET_MASK) as usize,
        (offsets[e + 1] & OFFSET_MASK) as usize,
        raw & INCOMPLETE != 0,
    )
}

/// Empty intervals are stored canonically as `(0, 0)`.
#[inline]
fn normalize((lo, hi): (u32, u32)) -> (u32, u32) {
    if lo < hi {
        (lo, hi)
    } else {
        (0, 0)
    }
}

/// The lookup table as written by format versions 1 and 2: one exact interval per entry,
/// built without alphabet fan-out.
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct LookupTableV1 {
    pub depth: u32,
    core_symbols: Vec<u8>,
    intervals: Vec<(u32, u32)>,
}

impl LookupTableV1 {
    /// Convert to the current layout. When the old exact-only table is complete for this
    /// index (no core symbol matches another present symbol) it is converted in place;
    /// otherwise — an `IupacDna` index whose reference holds ambiguity codes — the old table
    /// missed matches, and a correct one is rebuilt at the same depth.
    pub(crate) fn upgrade(
        self,
        text_len: u32,
        c_array: &CArray,
        occ: &OccTable,
        fns: &AlphabetFns,
    ) -> LookupTable {
        let core = SymbolSet::from_codes(&self.core_symbols);
        let ascending: Vec<u8> = core.iter().collect();
        let complete = core == fns.core()
            && ascending == self.core_symbols
            && fns.exact_table_is_complete(c_array.present_symbols(text_len));
        if !complete {
            return LookupTable::build(self.depth, text_len, c_array, occ, fns);
        }
        let mut offsets = Vec::with_capacity(self.intervals.len() + 1);
        let mut intervals = Vec::with_capacity(self.intervals.len());
        for &(lo, hi) in &self.intervals {
            offsets.push(intervals.len() as u32);
            // Old tables store non-canonical empties such as `(57, 57)`.
            if lo < hi {
                intervals.push((lo, hi));
            }
        }
        offsets.push(intervals.len() as u32);
        LookupTable {
            depth: self.depth,
            core,
            offsets,
            intervals,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::{
        encode_char, Alphabet, DnaSequence, ExactDna, IupacDna, A, C, G, N, R, T,
    };
    use crate::fm_index::{FmIndex, FmIndexConfig};

    fn config(depth: u32) -> FmIndexConfig {
        FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            lookup_depth: depth,
            build_threads: 1,
            occ_encoding: Default::default(),
            build_lcp: true,
        }
    }

    fn make_index_with_lookup(s: &str, depth: u32) -> FmIndex {
        let seq = DnaSequence::from_str(s).unwrap();
        FmIndex::build_cpu(&[seq], &config(depth)).unwrap()
    }

    fn make_exact_index_with_lookup(s: &str, depth: u32) -> FmIndex {
        let seq = DnaSequence::from_str(s).unwrap();
        FmIndex::build_cpu_with::<ExactDna>(&[seq], &config(depth)).unwrap()
    }

    fn make_with<Al: Alphabet>(s: &str, depth: u32) -> FmIndex {
        let seq = DnaSequence::from_str(s).unwrap();
        FmIndex::build_cpu_with::<Al>(&[seq], &config(depth)).unwrap()
    }

    fn encode(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    fn sorted_locate(idx: &FmIndex, p: &[u8]) -> Vec<u32> {
        let mut v = idx.locate_positions(p);
        v.sort_unstable();
        v
    }

    /// Every pattern of length `len` over `alphabet`.
    fn all_patterns(alphabet: &[u8], len: usize) -> Vec<Vec<u8>> {
        let mut out = vec![Vec::new()];
        for _ in 0..len {
            out = out
                .into_iter()
                .flat_map(|p| {
                    alphabet.iter().map(move |&c| {
                        let mut q = p.clone();
                        q.push(c);
                        q
                    })
                })
                .collect();
        }
        out
    }

    #[test]
    fn lookup_count_matches_full_search() {
        let text = "ACGTTAGCCAGTACGT";
        let idx_with = make_index_with_lookup(text, 3);
        let idx_no = make_index_with_lookup(text, 0);
        for pat in &["ACG", "CGT", "GCC", "TAG", "ACGT", "TTA", "AAA"] {
            let enc = encode(pat);
            assert_eq!(
                idx_with.count(&enc),
                idx_no.count(&enc),
                "count mismatch for '{}' with lookup vs without",
                pat
            );
        }
    }

    #[test]
    fn lookup_locate_matches_full_search() {
        let text = "ACGTACGTACGT";
        let idx_with = make_index_with_lookup(text, 4);
        let idx_no = make_index_with_lookup(text, 0);
        let enc = encode("ACGT");
        assert_eq!(sorted_locate(&idx_with, &enc), sorted_locate(&idx_no, &enc));
    }

    #[test]
    fn lookup_short_pattern_falls_back() {
        // Pattern shorter than depth → falls back to full backward search.
        let idx = make_index_with_lookup("ACGTACGT", 4);
        let enc = encode("ACG"); // len 3 < depth 4
        assert_eq!(idx.count(&enc), 2);
    }

    #[test]
    fn pure_acgt_table_has_one_interval_per_occurring_kmer_and_no_variants() {
        let idx = make_index_with_lookup("ACGTTAGCCAGTACGT", 2);
        let lut = idx.lookup.as_ref().unwrap();
        for pat in all_patterns(&[A, C, G, T], 2) {
            let hit = lut.get(&pat).unwrap();
            assert!(hit.complete);
            assert!(hit.intervals.len() <= 1, "{pat:?}: {:?}", hit.intervals);
            let (lo, hi) = hit.intervals.first().copied().unwrap_or((0, 0));
            assert_eq!(hi - lo, idx.count(&pat), "{pat:?}");
        }
        // Non-core symbols and wrong-length k-mers are not tabulated.
        assert!(lut.get(&[A, N]).is_none());
        assert!(lut.get(&[R, A]).is_none());
        assert!(lut.get(&[A]).is_none());
        assert!(lut.get(&[A, C, G]).is_none());
        // Rightmost code is the most significant digit: "CA" and "AC" are distinct entries.
        assert_ne!(
            lut.get(&[C, A]).unwrap().intervals,
            lut.get(&[A, C]).unwrap().intervals
        );
    }

    // ── ExactDna alphabet tests ────────────────────────────────────────────────

    #[test]
    fn exact_dna_query_n_returns_zero_hits() {
        // Text has N in it; query N should match nothing with ExactDna.
        let text = "ACGTNACGT";
        let idx_exact = make_exact_index_with_lookup(text, 3);
        let enc_n = encode("N");
        assert_eq!(
            idx_exact.count(&enc_n),
            0,
            "ExactDna: query N should return 0 hits"
        );
        // Same query with IupacDna should find hits (N matches A,C,G,T).
        let idx_iupac = make_index_with_lookup(text, 3);
        assert!(
            idx_iupac.count(&enc_n) > 0,
            "IupacDna: query N should match something"
        );
    }

    #[test]
    fn exact_dna_acgt_counts_match_iupac() {
        // For pure ACGT queries on a pure ACGT text, ExactDna and IupacDna must agree.
        let text = "ACGTTAGCCAGTACGT";
        let idx_exact = make_exact_index_with_lookup(text, 3);
        let idx_iupac = make_index_with_lookup(text, 3);
        for pat in &["ACG", "CGT", "GCC", "TAG", "ACGT"] {
            let enc = encode(pat);
            assert_eq!(
                idx_exact.count(&enc),
                idx_iupac.count(&enc),
                "ExactDna vs IupacDna count mismatch for '{}'",
                pat
            );
        }
    }

    #[test]
    fn lookup_iupac_consistency_with_n_in_text() {
        // Text contains N. With and without lookup table, IupacDna results must agree.
        let text = "ACGTNACGT";
        let idx_with = make_index_with_lookup(text, 3);
        let idx_no = make_index_with_lookup(text, 0);
        for pat in &["ACG", "CGT", "N", "ACGT", "TNA", "GTN", "TAC", "CGTA"] {
            let enc = encode(pat);
            assert_eq!(
                idx_with.count(&enc),
                idx_no.count(&enc),
                "IupacDna lookup/no-lookup mismatch for '{}'",
                pat
            );
        }
    }

    #[test]
    fn lookup_exact_consistency_with_n_in_text() {
        // Text contains N. With and without lookup table, ExactDna results must agree.
        let text = "ACGTNACGT";
        let idx_with = make_exact_index_with_lookup(text, 3);
        let idx_no = make_exact_index_with_lookup(text, 0);
        for pat in &["ACG", "CGT", "N", "ACGT"] {
            let enc = encode(pat);
            assert_eq!(
                idx_with.count(&enc),
                idx_no.count(&enc),
                "ExactDna lookup/no-lookup mismatch for '{}'",
                pat
            );
        }
    }

    // ── Wildcards inside the seed window ──────────────────────────────────────

    /// The regression: reference `ANG` matches query `ACG` under `IupacDna` (N ⊇ C), and
    /// the whole query sits inside the depth-3 seed window.
    #[test]
    fn lookup_seeded_search_matches_reference_wildcards_in_the_seed_window() {
        let idx = make_index_with_lookup("ANG", 3);
        assert_eq!(idx.count(&encode("ACG")), 1);
        assert_eq!(sorted_locate(&idx, &encode("ACG")), vec![0]);
        let lut = idx.lookup.as_ref().unwrap();
        let hit = lut.get(&encode("ACG")).unwrap();
        assert!(hit.complete);
        // Exact slot empty, one variant.
        assert_eq!(hit.intervals.len(), 2);
        assert_eq!(hit.intervals[0], (0, 0));
        assert_eq!(hit.intervals[1].1 - hit.intervals[1].0, 1);
    }

    #[test]
    fn lookup_matches_full_search_with_wildcards_in_seed_window() {
        let texts = ["ANG", "ACGTNNNACGTRYSWKMBDHV", "GATTACA", "NNNNNNNN"];
        let query_alphabet = [A, C, G, T, N, R];
        for text in texts {
            for depth in 1..=4u32 {
                let pairs: [(FmIndex, FmIndex); 2] = [
                    (
                        make_with::<IupacDna>(text, depth),
                        make_with::<IupacDna>(text, 0),
                    ),
                    (
                        make_with::<ExactDna>(text, depth),
                        make_with::<ExactDna>(text, 0),
                    ),
                ];
                for (with, without) in &pairs {
                    for len in depth as usize..=depth as usize + 2 {
                        for pat in all_patterns(&query_alphabet, len) {
                            assert_eq!(
                                with.count(&pat),
                                without.count(&pat),
                                "count text={text} depth={depth} tag={} pat={pat:?}",
                                with.alphabet_fns.tag()
                            );
                            assert_eq!(
                                sorted_locate(with, &pat),
                                sorted_locate(without, &pat),
                                "locate text={text} depth={depth} tag={} pat={pat:?}",
                                with.alphabet_fns.tag()
                            );
                        }
                    }
                }
            }
        }
    }

    /// Exact slots always equal the literal k-mer's interval, variants or not.
    #[test]
    fn exact_slot_matches_literal_backward_search() {
        let text = "ACGTNNNACGTRYSWKMBDHVACGTACGTTTGCA";
        let idx = make_index_with_lookup(text, 3);
        let exact = make_exact_index_with_lookup(text, 0);
        let lut = idx.lookup.as_ref().unwrap();
        for pat in all_patterns(&[A, C, G, T], 3) {
            let hit = lut.get(&pat).unwrap();
            let literal = exact.count(&pat);
            let (lo, hi) = hit.intervals.first().copied().unwrap_or((0, 0));
            assert_eq!(hi.saturating_sub(lo), literal, "{pat:?}");
        }
    }

    #[test]
    fn lookup_cap_marks_entry_incomplete_and_search_stays_correct() {
        // Every distinct reference 3-gram that matches a k-mer through ambiguity codes is
        // one variant interval. "xyA" for all pairs of A-compatible wildcards gives "AAA"
        // 49 variants, far past the cap.
        let wild = ['N', 'R', 'W', 'M', 'D', 'H', 'V'];
        let mut text = String::new();
        for x in wild {
            for y in wild {
                text.push(x);
                text.push(y);
                text.push('A');
            }
        }
        text.push_str("ACGTACGTAAA");
        let text = text.as_str();
        let idx = make_index_with_lookup(text, 3);
        let full = make_index_with_lookup(text, 0);
        let lut = idx.lookup.as_ref().unwrap();
        let literal = make_exact_index_with_lookup(text, 0);
        let mut incomplete = 0;
        for pat in all_patterns(&[A, C, G, T], 3) {
            let hit = lut.get(&pat).unwrap();
            if !hit.complete {
                incomplete += 1;
                assert!(
                    hit.intervals.len() <= 1,
                    "{pat:?}: flagged entries keep only slot 0"
                );
            } else {
                assert!(hit.intervals.len() <= 1 + MAX_VARIANTS_PER_ENTRY);
            }
            let (lo, hi) = hit.intervals.first().copied().unwrap_or((0, 0));
            assert_eq!(
                hi.saturating_sub(lo),
                literal.count(&pat),
                "{pat:?} exact slot"
            );
            assert_eq!(idx.count(&pat), full.count(&pat), "{pat:?} count");
            assert_eq!(
                sorted_locate(&idx, &pat),
                sorted_locate(&full, &pat),
                "{pat:?}"
            );
        }
        assert!(incomplete > 0, "expected the cap to trigger on this text");
        let aaa = lut.get(&[A, A, A]).unwrap();
        assert!(!aaa.complete);
        // The exact slot survives: "AAA" occurs once literally.
        assert_eq!(aaa.intervals.len(), 1);
        assert_eq!(aaa.intervals[0].1 - aaa.intervals[0].0, 1);
        assert_eq!(literal.count(&[A, A, A]), 1);
    }

    #[test]
    fn v1_table_upgrade_reuses_when_complete_rebuilds_otherwise() {
        // Build v1-style exact tables by hand from the index's own components.
        fn v1_of(idx: &FmIndex, depth: u32) -> LookupTableV1 {
            let core = [A, C, G, T];
            let n = 4usize.pow(depth);
            let mut intervals = vec![(0u32, 0u32); n];
            let mut cur = vec![(0usize, 0u32, idx.text_len)];
            for level in 1..=depth as usize {
                let mut next = Vec::new();
                for &(parent, lo, hi) in &cur {
                    for (d, &s) in core.iter().enumerate() {
                        let child = parent * 4 + d;
                        let cv = idx.c_array.get(s);
                        let (nlo, nhi) = (cv + idx.occ.rank(s, lo), cv + idx.occ.rank(s, hi));
                        if level == depth as usize {
                            intervals[child] = (nlo, nhi);
                        } else if nlo < nhi {
                            next.push((child, nlo, nhi));
                        }
                    }
                }
                cur = next;
            }
            LookupTableV1 {
                depth,
                core_symbols: core.to_vec(),
                intervals,
            }
        }

        // Pure ACGT: the old table is complete and converts in place, matching a fresh build.
        let acgt = make_index_with_lookup("ACGTTAGCCAGTACGT", 3);
        let upgraded =
            v1_of(&acgt, 3).upgrade(acgt.text_len, &acgt.c_array, &acgt.occ, &acgt.alphabet_fns);
        assert_eq!(&upgraded, acgt.lookup.as_ref().unwrap());

        // Wildcards under IupacDna: the old table is incomplete and gets rebuilt.
        let iupac = make_index_with_lookup("ACGTNACGTRYS", 3);
        let upgraded = v1_of(&iupac, 3).upgrade(
            iupac.text_len,
            &iupac.c_array,
            &iupac.occ,
            &iupac.alphabet_fns,
        );
        assert_eq!(&upgraded, iupac.lookup.as_ref().unwrap());

        // The same text under ExactDna: old table complete, converted in place.
        let exact = make_exact_index_with_lookup("ACGTNACGTRYS", 3);
        let upgraded = v1_of(&exact, 3).upgrade(
            exact.text_len,
            &exact.c_array,
            &exact.occ,
            &exact.alphabet_fns,
        );
        assert_eq!(&upgraded, exact.lookup.as_ref().unwrap());
    }
}
