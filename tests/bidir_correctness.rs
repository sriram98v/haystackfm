/// Correctness tests for the bidirectional FM-index and SMEM finding.
///
/// All tests run on the CPU path and verify results against brute-force
/// string search so there are no hidden dependencies on the GPU.
use haystackfm::{BidirFmIndex, DnaSequence, FmIndex, FmIndexConfig};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn encode(s: &str) -> Vec<u8> {
    use haystackfm::alphabet::encode_char;
    s.chars().map(|c| encode_char(c).unwrap()).collect()
}

fn bidir_single(s: &str) -> BidirFmIndex {
    let config = FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: false,
        ..Default::default()
    };
    BidirFmIndex::build_cpu(&[DnaSequence::from_str(s).unwrap()], &config).unwrap()
}

fn bidir_multi(seqs: &[&str]) -> BidirFmIndex {
    let dna: Vec<DnaSequence> = seqs
        .iter()
        .map(|s| DnaSequence::from_str(s).unwrap())
        .collect();
    let config = FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: false,
        ..Default::default()
    };
    BidirFmIndex::build_cpu(&dna, &config).unwrap()
}

fn uni(s: &str) -> FmIndex {
    let config = FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: false,
        ..Default::default()
    };
    FmIndex::build_cpu(&[DnaSequence::from_str(s).unwrap()], &config).unwrap()
}

/// Count overlapping occurrences of `pattern` in `text` using plain string search.
fn naive_count(text: &str, pattern: &str) -> u32 {
    if pattern.is_empty() || pattern.len() > text.len() {
        return 0;
    }
    (0..=text.len() - pattern.len())
        .filter(|&i| &text[i..i + pattern.len()] == pattern)
        .count() as u32
}

/// Brute-force MEM finder: all substrings of `query` that occur in `reference`,
/// are left-maximal, and are right-maximal.
fn brute_force_mems(reference: &str, query: &str, min_len: usize) -> Vec<(usize, usize)> {
    let n = query.len();
    let mut set: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

    for start in 0..n {
        for end in (start + min_len)..=n {
            let sub = &query[start..end];
            if !reference.contains(sub) {
                continue;
            }
            let right_max = end == n || !reference.contains(&query[start..end + 1]);
            let left_max = start == 0 || !reference.contains(&query[start - 1..end]);
            if right_max && left_max {
                set.insert((start, end));
            }
        }
    }

    let mut v: Vec<_> = set.into_iter().collect();
    v.sort();
    v
}

// ── BidirInterval tests ───────────────────────────────────────────────────────

#[test]
fn extend_right_count_equals_forward_index() {
    for text in &[
        "ACGTACGT",
        "AAACCCGGGTTT",
        "ACGT",
        "TTTTTTTT",
        "ACGTTAGCCAGTACGT",
    ] {
        let idx = bidir_single(text);
        let uni_idx = uni(text);
        for pattern_str in &["A", "AC", "ACG", "ACGT", "TT", "GGG", "XXXXXX"] {
            // Skip patterns with invalid chars — encode handles them by filtering
            let valid: String = pattern_str
                .chars()
                .filter(|c| "ACGT".contains(*c))
                .collect();
            if valid.is_empty() {
                continue;
            }
            let pattern = encode(&valid);
            let mut iv = idx.full_interval();
            let mut ok = true;
            for &c in &pattern {
                match idx.extend_right(iv, c) {
                    Some(next) => iv = next,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            let bidir_count = if ok { iv.size() } else { 0 };
            let uni_count = uni_idx.count(&pattern);
            assert_eq!(
                bidir_count, uni_count,
                "count mismatch for pattern '{}' in text '{}'",
                valid, text
            );
        }
    }
}

#[test]
fn extend_right_size_invariant() {
    let idx = bidir_single("ACGTACGTACGT");
    let mut iv = idx.full_interval();
    assert_eq!(iv.fwd_hi - iv.fwd_lo, iv.rev_hi - iv.rev_lo);

    for c in encode("ACGT") {
        if let Some(next) = idx.extend_right(iv, c) {
            assert_eq!(
                next.fwd_hi - next.fwd_lo,
                next.rev_hi - next.rev_lo,
                "size invariant broken"
            );
            iv = next;
        }
    }
}

#[test]
fn extend_left_size_invariant() {
    let idx = bidir_single("ACGTACGTACGT");
    let mut iv = idx.full_interval();

    // Build interval for "ACGT" using extend_right first
    for c in encode("ACGT") {
        iv = idx.extend_right(iv, c).unwrap();
    }
    assert_eq!(iv.fwd_hi - iv.fwd_lo, iv.rev_hi - iv.rev_lo);

    // There's nothing to the left of the very first ACGT in this text,
    // but we can test the invariant holds when it succeeds.
    // In "ACGTACGTACGT", at positions 4 and 8, the preceding char is T.
    use haystackfm::alphabet::T;
    if let Some(next) = idx.extend_left(iv, T) {
        assert_eq!(
            next.fwd_hi - next.fwd_lo,
            next.rev_hi - next.rev_lo,
            "size invariant broken after extend_left"
        );
    }
}

#[test]
fn bidirectional_locate_positions_are_correct() {
    let text = "ACGTACGT";
    let query_str = "ACG";
    let idx = bidir_single(text);
    let pattern = encode(query_str);

    let mut iv = idx.full_interval();
    for &c in &pattern {
        iv = idx.extend_right(iv, c).unwrap();
    }

    let mut positions = idx.locate_interval(&iv);
    positions.sort();

    // Verify each position by comparing the original string (not raw bytes)
    for (_, pos) in &positions {
        let pos = *pos as usize;
        assert!(
            pos + query_str.len() <= text.len(),
            "position {} out of bounds",
            pos
        );
        assert_eq!(
            &text[pos..pos + query_str.len()],
            query_str,
            "wrong match at position {}",
            pos
        );
    }

    // Count should match naive
    assert_eq!(positions.len() as u32, naive_count(text, query_str));
}

// ── SMEM tests ────────────────────────────────────────────────────────────────

#[test]
fn smems_match_brute_force_basic() {
    let cases = [
        ("ACGTACGT", "ACGT"),
        ("ACGTTAGCCAGTACGT", "CGTTAGC"),
        ("AAAACCCCC", "AACCC"),
        ("ACGT", "TGCA"),
        ("ACGT", "ACGT"),
        ("TTTTTTTT", "TTTTT"),
    ];

    for (reference, query_str) in &cases {
        let idx = bidir_single(reference);
        let query = encode(query_str);
        let smems = idx.find_smems(&query, 1, false);
        let smem_pairs: Vec<(usize, usize)> =
            smems.iter().map(|m| (m.query_start, m.query_end)).collect();
        let expected = brute_force_mems(reference, query_str, 1);

        assert_eq!(
            smem_pairs, expected,
            "\nreference='{}' query='{}'\nGot:      {:?}\nExpected: {:?}",
            reference, query_str, smem_pairs, expected
        );
    }
}

#[test]
fn smem_count_matches_forward_index() {
    let text = "ACGTACGTACGT";
    let idx = bidir_single(text);
    let uni_idx = uni(text);

    let query = encode("ACGT");
    let smems = idx.find_smems(&query, 1, false);

    assert_eq!(smems.len(), 1);
    assert_eq!(smems[0].match_count, uni_idx.count(&query));
}

#[test]
fn smems_min_len_filtering() {
    let idx = bidir_single("AACCGGTT");
    let query = encode("AACCGGTT");

    let smems_1 = idx.find_smems(&query, 1, false);
    let smems_4 = idx.find_smems(&query, 4, false);
    let smems_100 = idx.find_smems(&query, 100, false);

    for m in &smems_4 {
        assert!(m.len() >= 4, "SMEM shorter than min_len=4: {:?}", m);
    }
    assert!(smems_100.is_empty());
    // min_len=1 should find at least as many as min_len=4
    assert!(smems_1.len() >= smems_4.len());
}

#[test]
fn smem_positions_are_valid() {
    let reference = "ACGTTAGCCAGTACGT";
    let query_str = "AGTACGT";
    let idx = bidir_single(reference);
    let query = encode(query_str);

    let smems = idx.find_smems(&query, 1, true);
    for mem in &smems {
        let pattern = &query_str[mem.query_start..mem.query_end];
        for (_, pos) in &mem.positions {
            let pos = *pos as usize;
            assert!(
                pos + pattern.len() <= reference.len(),
                "position {} out of bounds (ref len {})",
                pos,
                reference.len()
            );
            assert_eq!(
                &reference[pos..pos + pattern.len()],
                pattern,
                "wrong match at pos {}",
                pos
            );
        }
        // count must equal actual occurrence count
        assert_eq!(
            mem.match_count as usize,
            mem.positions.len(),
            "match_count != positions.len() for pattern '{}'",
            pattern
        );
    }
}

#[test]
fn smems_on_multi_sequence_index() {
    let idx = bidir_multi(&["ACGTACGT", "TGCATGCA"]);
    let query = encode("ACGT");
    let smems = idx.find_smems(&query, 1, false);

    // "ACGT" appears in the first sequence twice; TGCA is in the second sequence.
    // The bidirectional index covers both; ACGT count = 2.
    assert!(!smems.is_empty());
    for m in &smems {
        assert!(m.match_count > 0);
    }
}

#[test]
fn find_mems_contains_all_smems() {
    let reference = "ACGTTAGCCAGTACGT";
    let query_str = "CGTTAGC";
    let idx = bidir_single(reference);
    let query = encode(query_str);

    let smems = idx.find_smems(&query, 1, false);
    let mems = idx.find_mems(&query, 1, false);

    for smem in &smems {
        assert!(
            mems.iter()
                .any(|m| m.query_start == smem.query_start && m.query_end == smem.query_end),
            "SMEM ({},{}) not in MEM list",
            smem.query_start,
            smem.query_end
        );
    }
}

#[test]
fn bidir_serialization_preserves_smems() {
    let reference = "ACGTTAGCCAGTACGT";
    let original = bidir_single(reference);
    let bytes = original.to_bytes().unwrap();
    let restored = BidirFmIndex::from_bytes(&bytes).unwrap();

    let query = encode("CGTTAGC");
    let orig_smems = original.find_smems(&query, 1, false);
    let rest_smems = restored.find_smems(&query, 1, false);

    assert_eq!(orig_smems, rest_smems);
}

// ── Wildcard-aware cursor: count_wild_*, count_*_in, children_*, compatible fan-out ──
//
// Oracle: locate every occurrence of the cursor's pattern, look up the neighbouring symbol
// in the reference text, and count the ones that are ambiguity codes (>= 5). A sentinel
// neighbour (occurrence at a reference boundary) is never wild.

mod wild {
    use haystackfm::alphabet::{self, decode_char, ExactDna, SymbolSet, ALPHABET_SIZE};
    use haystackfm::{BidirFmIndex, BidirInterval, DnaSequence, FmIndexConfig, OccEncoding};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    const WILD_CODES: [u8; 11] = [
        alphabet::N,
        alphabet::R,
        alphabet::Y,
        alphabet::S,
        alphabet::W,
        alphabet::K,
        alphabet::M,
        alphabet::B,
        alphabet::D,
        alphabet::H,
        alphabet::V,
    ];
    const BASES: [u8; 4] = [alphabet::A, alphabet::C, alphabet::G, alphabet::T];

    fn config(sa_sample_rate: u32, occ_encoding: OccEncoding) -> FmIndexConfig {
        FmIndexConfig {
            sa_sample_rate,
            use_gpu: false,
            occ_encoding,
            ..Default::default()
        }
    }

    fn build(seqs: &[Vec<u8>], sa_sample_rate: u32, occ_encoding: OccEncoding) -> BidirFmIndex {
        let dna: Vec<DnaSequence> = seqs
            .iter()
            .cloned()
            .map(DnaSequence::from_encoded)
            .collect();
        BidirFmIndex::build_cpu(&dna, &config(sa_sample_rate, occ_encoding)).unwrap()
    }

    fn build_str(seqs: &[&str]) -> BidirFmIndex {
        let dna: Vec<DnaSequence> = seqs
            .iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect();
        BidirFmIndex::build_cpu(&dna, &config(1, OccEncoding::Bitplane)).unwrap()
    }

    fn is_wild(code: u8) -> bool {
        code >= alphabet::N
    }

    /// Brute-force `(count_wild_left, count_wild_right)` for a cursor matching a pattern of
    /// length `len`.
    fn oracle_wild(idx: &BidirFmIndex, iv: &BidirInterval, len: usize) -> (u32, u32) {
        let mut left = 0;
        let mut right = 0;
        for (id, pos) in idx.locate_interval(iv) {
            let seq = idx
                .sequence(id)
                .expect("sequence retained on forward index");
            let pos = pos as usize;
            if pos.checked_sub(1).map(|p| is_wild(seq[p])).unwrap_or(false) {
                left += 1;
            }
            if seq.get(pos + len).copied().map(is_wild).unwrap_or(false) {
                right += 1;
            }
        }
        (left, right)
    }

    /// Random sequence: ACGT body with `runs` injected wildcard runs of length 1..=7, one
    /// optionally forced at the start and one at the end.
    fn place_run(rng: &mut SmallRng, seq: &mut [u8], start: usize) {
        let run_len = rng.random_range(1..=7);
        for k in 0..run_len {
            if start + k < seq.len() {
                seq[start + k] = WILD_CODES[rng.random_range(0..WILD_CODES.len())];
            }
        }
    }

    fn random_seq(
        rng: &mut SmallRng,
        len: usize,
        runs: usize,
        at_start: bool,
        at_end: bool,
    ) -> Vec<u8> {
        let mut seq: Vec<u8> = (0..len).map(|_| BASES[rng.random_range(0..4)]).collect();
        for _ in 0..runs {
            let start = rng.random_range(0..len);
            place_run(rng, &mut seq, start);
        }
        if at_start {
            place_run(rng, &mut seq, 0);
        }
        if at_end {
            let run_len = rng.random_range(1..=7).min(len);
            place_run(rng, &mut seq, len - run_len);
        }
        seq
    }

    fn show(seq: &[u8]) -> String {
        seq.iter().map(|&c| decode_char(c).unwrap()).collect()
    }

    #[test]
    fn count_wild_matches_brute_force_on_random_iupac_multi_seq() {
        let mut rng = SmallRng::seed_from_u64(0x0005_7ACC_F00D_u64);
        for (rate, enc) in [
            (1, OccEncoding::Bitplane),
            (32, OccEncoding::Bitplane),
            (1, OccEncoding::OneHot),
            (32, OccEncoding::OneHot),
        ] {
            for iter in 0..40 {
                let nseq = rng.random_range(1..=4);
                let seqs: Vec<Vec<u8>> = (0..nseq)
                    .map(|k| {
                        let len = rng.random_range(5..=120);
                        let runs = rng.random_range(0..=5);
                        random_seq(&mut rng, len, runs, k % 2 == 0, k % 3 == 0)
                    })
                    .collect();
                let idx = build(&seqs, rate, enc);
                let ctx = || seqs.iter().map(|s| show(s)).collect::<Vec<_>>().join(" | ");

                for _ in 0..4 {
                    // Sample a substring of one reference (may contain wildcard codes; we
                    // extend by the exact reference code so the walk never dies early).
                    let s = &seqs[rng.random_range(0..seqs.len())];
                    let plen = rng.random_range(1..=8.min(s.len()));
                    let start = rng.random_range(0..=s.len() - plen);
                    let pat = &s[start..start + plen];

                    // Right walk.
                    let mut iv = idx.full_interval();
                    for (k, &c) in pat.iter().enumerate() {
                        iv = idx.extend_right(iv, c).expect("substring must be present");
                        let (l, r) = oracle_wild(&idx, &iv, k + 1);
                        assert_eq!(
                            idx.count_wild_right(&iv),
                            r,
                            "right walk step {k} rate={rate} {enc:?} iter={iter} pat={} refs={}",
                            show(&pat[..=k]),
                            ctx()
                        );
                        assert_eq!(
                            idx.count_wild_left(&iv),
                            l,
                            "right walk (left count) step {k} rate={rate} {enc:?} iter={iter} pat={} refs={}",
                            show(&pat[..=k]),
                            ctx()
                        );
                    }
                    // Left walk.
                    let mut iv = idx.full_interval();
                    for (k, &c) in pat.iter().rev().enumerate() {
                        iv = idx.extend_left(iv, c).expect("substring must be present");
                        let (l, r) = oracle_wild(&idx, &iv, k + 1);
                        assert_eq!(
                            idx.count_wild_left(&iv),
                            l,
                            "left walk step {k} rate={rate} {enc:?} iter={iter} refs={}",
                            ctx()
                        );
                        assert_eq!(idx.count_wild_right(&iv), r, "left walk (right count) step {k} rate={rate} {enc:?} iter={iter} refs={}", ctx());
                    }
                }
            }
        }
    }

    #[test]
    fn count_wild_at_sequence_boundaries() {
        let idx = build_str(&["RRACGT", "ACGTYY", "ACGT"]);
        let walk = |p: &str| -> BidirInterval {
            let mut iv = idx.full_interval();
            for ch in p.chars() {
                iv = idx
                    .extend_right(iv, alphabet::encode_char(ch).unwrap())
                    .unwrap();
            }
            iv
        };
        // ACGT: preceded by RR (wild), start, start; followed by end, YY (wild), end.
        let acgt = walk("ACGT");
        assert_eq!(acgt.size(), 3);
        assert_eq!(idx.count_wild_left(&acgt), 1);
        assert_eq!(idx.count_wild_right(&acgt), 1);
        assert_eq!(
            idx.count_right_in(&acgt, SymbolSet::single(alphabet::SENTINEL)),
            2
        );
        assert_eq!(
            idx.count_left_in(&acgt, SymbolSet::single(alphabet::SENTINEL)),
            2
        );
        // RR sits at a sequence start, followed by A.
        let rr = walk("RR");
        assert_eq!(idx.count_wild_left(&rr), 0);
        assert_eq!(idx.count_wild_right(&rr), 0);
        // YY sits at a sequence end, preceded by T.
        let yy = walk("YY");
        assert_eq!(idx.count_wild_left(&yy), 0);
        assert_eq!(idx.count_wild_right(&yy), 0);
        // R alone: first R is at start (not wild left), second R is preceded by R (wild).
        let r = walk("R");
        assert_eq!(r.size(), 2);
        assert_eq!(idx.count_wild_left(&r), 1);
        assert_eq!(idx.count_wild_right(&r), 1);
    }

    #[test]
    fn count_in_partitions_interval_size() {
        let mut rng = SmallRng::seed_from_u64(42);
        let seqs: Vec<Vec<u8>> = (0..3)
            .map(|_| random_seq(&mut rng, 80, 4, true, true))
            .collect();
        let idx = build(&seqs, 1, OccEncoding::Bitplane);
        let sentinel = SymbolSet::single(alphabet::SENTINEL);
        let mut iv = idx.full_interval();
        for &c in &[alphabet::A, alphabet::C] {
            let Some(next) = idx.extend_right(iv, c) else {
                break;
            };
            iv = next;
            for set in [
                SymbolSet::WILDCARDS,
                SymbolSet::BASES,
                SymbolSet::below(alphabet::G),
            ] {
                assert_eq!(
                    idx.count_right_in(&iv, set) + idx.count_right_in(&iv, set.complement()),
                    iv.size()
                );
                assert_eq!(
                    idx.count_left_in(&iv, set) + idx.count_left_in(&iv, set.complement()),
                    iv.size()
                );
            }
            assert_eq!(
                idx.count_wild_right(&iv)
                    + idx.count_right_in(&iv, SymbolSet::BASES)
                    + idx.count_right_in(&iv, sentinel),
                iv.size()
            );
            assert_eq!(idx.count_right_in(&iv, SymbolSet::ALL), iv.size());
        }
    }

    #[test]
    fn children_agree_with_extend_on_random_iupac_indexes() {
        let mut rng = SmallRng::seed_from_u64(0x000C_411D);
        for enc in [OccEncoding::Bitplane, OccEncoding::OneHot] {
            for _ in 0..20 {
                let nseq = rng.random_range(1..=3);
                let seqs: Vec<Vec<u8>> = (0..nseq)
                    .map(|_| {
                        let len = rng.random_range(5..=100);
                        random_seq(&mut rng, len, 3, true, true)
                    })
                    .collect();
                let idx = build(&seqs, 1, enc);
                let mut iv = idx.full_interval();
                loop {
                    let right = idx.children_right(&iv);
                    let left = idx.children_left(&iv);
                    let mut sum_r = 0;
                    let mut sum_l = 0;
                    for c in 0..ALPHABET_SIZE as u8 {
                        assert_eq!(
                            right[c as usize],
                            idx.extend_right(iv, c),
                            "right child {c}"
                        );
                        assert_eq!(left[c as usize], idx.extend_left(iv, c), "left child {c}");
                        sum_r += right[c as usize].map_or(0, |x| x.size());
                        sum_l += left[c as usize].map_or(0, |x| x.size());
                    }
                    assert_eq!(sum_r, iv.size());
                    assert_eq!(sum_l, iv.size());
                    // Sentinel children are occurrences at reference ends/starts.
                    assert_eq!(
                        right[0].map_or(0, |x| x.size()),
                        idx.count_right_in(&iv, SymbolSet::single(alphabet::SENTINEL))
                    );
                    assert_eq!(
                        left[0].map_or(0, |x| x.size()),
                        idx.count_left_in(&iv, SymbolSet::single(alphabet::SENTINEL))
                    );
                    // Descend into a random non-sentinel right child, if any.
                    let options: Vec<BidirInterval> =
                        right[1..].iter().flatten().copied().collect();
                    if options.is_empty() || iv.size() == 1 {
                        break;
                    }
                    iv = options[rng.random_range(0..options.len())];
                }
            }
        }
    }

    #[test]
    fn compatible_fanout_matches_count_in_compatible_set() {
        let mut rng = SmallRng::seed_from_u64(7);
        let seqs: Vec<Vec<u8>> = (0..2)
            .map(|_| random_seq(&mut rng, 90, 5, true, true))
            .collect();
        let idx = build(&seqs, 1, OccEncoding::Bitplane);
        let mut iv = idx.full_interval();
        for step in 0..3 {
            for q in 1..ALPHABET_SIZE as u8 {
                let set = idx.compatible_set(q);
                let fan_r: u32 = idx.extend_right_compatible(iv, q).map(|c| c.size()).sum();
                assert_eq!(
                    fan_r,
                    idx.count_right_in(&iv, set),
                    "right q={q} step={step}"
                );
                let fan_l: u32 = idx.extend_left_compatible(iv, q).map(|c| c.size()).sum();
                assert_eq!(fan_l, idx.count_left_in(&iv, set), "left q={q} step={step}");
                // The wildcard-only slice of the fan-out is the intended premise query.
                let wild_only: u32 = idx
                    .extend_right_compatible(iv, q)
                    .zip((idx.compatible_set(q)).iter())
                    .filter(|(_, code)| *code >= alphabet::N)
                    .map(|(c, _)| c.size())
                    .sum();
                assert_eq!(
                    wild_only,
                    idx.count_right_in(&iv, set.intersection(SymbolSet::WILDCARDS)),
                    "wild-only q={q} step={step}"
                );
            }
            let Some(next) = idx.extend_right(iv, alphabet::A) else {
                break;
            };
            iv = next;
        }
    }

    #[test]
    fn exact_dna_fanout_is_exact_but_wild_counts_still_see_reference_wildcards() {
        let dna = vec![
            DnaSequence::from_str("ACGTNNACGTRYACGT").unwrap(),
            DnaSequence::from_str("WACGTM").unwrap(),
        ];
        let idx = BidirFmIndex::build_cpu_with::<ExactDna>(&dna, &config(1, OccEncoding::Bitplane))
            .unwrap();
        assert!(idx.compatible_set(alphabet::N).is_empty());
        assert_eq!(
            idx.compatible_set(alphabet::A),
            SymbolSet::single(alphabet::A)
        );
        let mut iv = idx.full_interval();
        for &c in &[alphabet::A, alphabet::C, alphabet::G, alphabet::T] {
            iv = idx.extend_right(iv, c).unwrap();
        }
        assert_eq!(iv.size(), 4);
        // N under ExactDna matches nothing …
        assert_eq!(idx.extend_right_compatible(iv, alphabet::N).count(), 0);
        assert_eq!(idx.count_right_in(&iv, idx.compatible_set(alphabet::N)), 0);
        // … but the reference wildcards are still there to be counted.
        // ACGT followed by: N, R, end, M → 3 wild; preceded by: start, N, Y, W → 3 wild.
        assert_eq!(idx.count_wild_right(&iv), 3);
        assert_eq!(idx.count_wild_left(&iv), 3);
        let (l, r) = oracle_wild(&idx, &iv, 4);
        assert_eq!((l, r), (3, 3));
    }

    #[test]
    fn wild_counts_survive_serialization() {
        let mut rng = SmallRng::seed_from_u64(99);
        let seqs: Vec<Vec<u8>> = (0..2)
            .map(|_| random_seq(&mut rng, 60, 4, true, true))
            .collect();
        let original = build(&seqs, 4, OccEncoding::Bitplane);
        let restored = BidirFmIndex::from_bytes(&original.to_bytes().unwrap()).unwrap();
        let mut iv_o = original.full_interval();
        let mut iv_r = restored.full_interval();
        for &c in &[alphabet::A, alphabet::C, alphabet::G] {
            let (Some(o), Some(r)) = (
                original.extend_right(iv_o, c),
                restored.extend_right(iv_r, c),
            ) else {
                break;
            };
            iv_o = o;
            iv_r = r;
            assert_eq!(iv_o, iv_r);
            assert_eq!(
                original.count_wild_right(&iv_o),
                restored.count_wild_right(&iv_r)
            );
            assert_eq!(
                original.count_wild_left(&iv_o),
                restored.count_wild_left(&iv_r)
            );
            assert_eq!(
                original.children_right(&iv_o),
                restored.children_right(&iv_r)
            );
        }
    }
}
