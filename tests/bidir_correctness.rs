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

    /// Cursor contraction (`contract_left`) tests, including wildcard (IUPAC) reference
    /// symbols and sentinel-adjacent occurrences.
    mod contract {
        use super::*;
        use haystackfm::FmIndexError;

        /// `ivs[k]` = interval of `pat[k..]`, built right-to-left with `extend_left`.
        fn left_walk(idx: &BidirFmIndex, pat: &[u8]) -> Vec<BidirInterval> {
            let mut ivs = vec![idx.full_interval()];
            for &c in pat.iter().rev() {
                let next = idx.extend_left(*ivs.last().unwrap(), c).unwrap();
                ivs.push(next);
            }
            ivs.reverse();
            ivs
        }

        /// Interval of `pat` built left-to-right with `extend_right` (independent route).
        fn right_walk(idx: &BidirFmIndex, pat: &[u8]) -> BidirInterval {
            pat.iter().fold(idx.full_interval(), |iv, &c| {
                idx.extend_right(iv, c).unwrap()
            })
        }

        fn assert_round_trip(idx: &BidirFmIndex, iv: BidirInterval, c: u8, ctx: &str) {
            let ext = idx.extend_left(iv, c).unwrap();
            let back = idx
                .contract_left(&ext, c)
                .unwrap_or_else(|e| panic!("{ctx}: contract_left({c}) failed: {e}"));
            assert_eq!(back, iv, "{ctx}: contract_left({c}) != original");
        }

        #[test]
        fn contract_left_inverts_extend_left_on_random_iupac_multi_seq() {
            let mut rng = SmallRng::seed_from_u64(0x5EED_C0DE);
            let mut saw_wild = false;
            let mut saw_sentinel = false;
            for &rate in &[1u32, 32] {
                for &enc in &[OccEncoding::Bitplane, OccEncoding::OneHot] {
                    for iter in 0..30 {
                        let nseq = rng.random_range(1..=4);
                        let seqs: Vec<Vec<u8>> = (0..nseq)
                            .map(|_| {
                                let len = rng.random_range(5..=120);
                                let runs = rng.random_range(0..=3);
                                random_seq(&mut rng, len, runs, iter % 3 == 0, iter % 4 == 0)
                            })
                            .collect();
                        let idx = build(&seqs, rate, enc);
                        assert!(idx.has_lcp());
                        for seq in &seqs {
                            for _ in 0..6 {
                                let plen = rng.random_range(1..=12.min(seq.len()));
                                let start = rng.random_range(0..=seq.len() - plen);
                                let pat = &seq[start..start + plen];
                                let ctx = format!("rate={rate} enc={enc:?} pat={}", show(pat));
                                let iv = right_walk(&idx, pat);
                                assert_eq!(iv.len as usize, plen);
                                for c in 0..ALPHABET_SIZE as u8 {
                                    if idx.extend_left(iv, c).is_none() {
                                        continue;
                                    }
                                    saw_wild |= is_wild(c);
                                    saw_sentinel |= c == 0;
                                    assert_round_trip(&idx, iv, c, &ctx);
                                }
                                // Whole chain: contract every prefix symbol back down.
                                let ivs = left_walk(&idx, pat);
                                for k in 0..plen {
                                    let got = idx.contract_left(&ivs[k], pat[k]).unwrap();
                                    assert_eq!(got, ivs[k + 1], "{ctx} k={k}");
                                }
                            }
                        }
                    }
                }
            }
            assert!(saw_wild, "no wildcard contraction exercised");
            assert!(saw_sentinel, "no sentinel contraction exercised");
        }

        #[test]
        fn contract_left_across_reference_wildcards_and_inside_wild_patterns() {
            // Ambiguity codes inside the reference: the removed symbol `c` is a wildcard,
            // and P itself contains wildcards.
            let idx = build_str(&["ACGRYTACGN", "ACGTACGRYT", "NNACGTRR", "RYTACG"]);
            for pat in ["YTACG", "RYTACG", "TACG", "ACG", "NACGT", "RR", "N", "GRYT"] {
                let pat = pat
                    .chars()
                    .map(|ch| alphabet::encode_char(ch).unwrap())
                    .collect::<Vec<_>>();
                let ivs = left_walk(&idx, &pat);
                for k in 0..pat.len() {
                    let got = idx.contract_left(&ivs[k], pat[k]).unwrap();
                    assert_eq!(got, ivs[k + 1], "pat={} k={k}", show(&pat));
                }
                let iv = ivs[0];
                for c in 0..ALPHABET_SIZE as u8 {
                    if idx.extend_left(iv, c).is_some() {
                        assert_round_trip(&idx, iv, c, &show(&pat));
                    }
                }
            }
        }

        #[test]
        fn contract_left_undoes_every_child_of_compatible_fanout() {
            // Query wildcard `q` fans out into one concrete child per compatible reference
            // code; contracting any child by its code lands back on the parent.
            let mut rng = SmallRng::seed_from_u64(77);
            for iter in 0..20 {
                let seqs: Vec<Vec<u8>> = (0..rng.random_range(1..=3))
                    .map(|_| {
                        let len = rng.random_range(10..=80);
                        random_seq(&mut rng, len, 3, iter % 2 == 0, false)
                    })
                    .collect();
                let idx = build(&seqs, 1, OccEncoding::Bitplane);
                for seq in &seqs {
                    let plen = rng.random_range(1..=6.min(seq.len()));
                    let start = rng.random_range(0..=seq.len() - plen);
                    let iv = right_walk(&idx, &seq[start..start + plen]);
                    for &q in WILD_CODES.iter().chain(BASES.iter()) {
                        let codes: Vec<u8> = idx.compatible_set(q).iter().collect();
                        let via_codes: Vec<BidirInterval> = codes
                            .iter()
                            .filter_map(|&c| idx.extend_left(iv, c))
                            .collect();
                        let fanout: Vec<BidirInterval> =
                            idx.extend_left_compatible(iv, q).collect();
                        assert_eq!(fanout, via_codes, "q={q}");
                        for &c in &codes {
                            if idx.extend_left(iv, c).is_some() {
                                assert_round_trip(&idx, iv, c, &format!("q={q}"));
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn contract_left_on_repeated_and_identical_sequences() {
            // Identical copies: every occurrence of P has copies preceded by different
            // symbols (sentinel vs base), so [r, r + size) is a strict sub-range and the
            // LCP expansion must widen it.
            let idx = build_str(&["ACGTACGT", "ACGTACGT", "ACGTACGT", "TACGTACG"]);
            let text: Vec<u8> = "ACGTACGT"
                .chars()
                .map(|c| alphabet::encode_char(c).unwrap())
                .collect();
            for start in 0..text.len() {
                for end in start + 1..=text.len() {
                    let pat = &text[start..end];
                    let ivs = left_walk(&idx, pat);
                    for k in 0..pat.len() {
                        assert_eq!(idx.contract_left(&ivs[k], pat[k]).unwrap(), ivs[k + 1]);
                    }
                }
            }
            let long_a = "A".repeat(200);
            let idx = build_str(&[&long_a, &long_a, "AAAAT"]);
            let pat = vec![alphabet::A; 200];
            let ivs = left_walk(&idx, &pat);
            for k in 0..pat.len() {
                assert_eq!(
                    idx.contract_left(&ivs[k], pat[k]).unwrap(),
                    ivs[k + 1],
                    "k={k}"
                );
            }
        }

        #[test]
        fn contract_left_survives_serialization_and_reports_missing_lcp() {
            let seqs = vec![
                "ACGTRYACGTNNACGT"
                    .chars()
                    .map(|c| alphabet::encode_char(c).unwrap())
                    .collect::<Vec<_>>(),
                "TTACGTACGTAA"
                    .chars()
                    .map(|c| alphabet::encode_char(c).unwrap())
                    .collect::<Vec<_>>(),
            ];
            let idx = build(&seqs, 4, OccEncoding::Bitplane);
            let bytes = idx.to_bytes().unwrap();
            let back = BidirFmIndex::from_bytes(&bytes).unwrap();
            assert!(back.has_lcp());
            let pat = &seqs[0][3..9];
            let ivs = left_walk(&idx, pat);
            for k in 0..pat.len() {
                let a = idx.contract_left(&ivs[k], pat[k]).unwrap();
                let b = back.contract_left(&ivs[k], pat[k]).unwrap();
                assert_eq!(a, b);
                assert_eq!(a, ivs[k + 1]);
            }

            let dna: Vec<DnaSequence> = seqs
                .iter()
                .cloned()
                .map(DnaSequence::from_encoded)
                .collect();
            let no_lcp = BidirFmIndex::build_cpu(
                &dna,
                &FmIndexConfig {
                    sa_sample_rate: 1,
                    use_gpu: false,
                    build_lcp: false,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(!no_lcp.has_lcp());
            let iv = no_lcp
                .extend_left(no_lcp.full_interval(), alphabet::A)
                .unwrap();
            assert!(matches!(
                no_lcp.contract_left(&iv, alphabet::A),
                Err(FmIndexError::LcpNotBuilt)
            ));
            let iv = no_lcp
                .extend_right(no_lcp.full_interval(), alphabet::A)
                .unwrap();
            assert!(matches!(
                no_lcp.contract_right(&iv, alphabet::A),
                Err(FmIndexError::LcpNotBuilt)
            ));
        }

        // ── contract_right ────────────────────────────────────────────────────

        /// `ivs[k]` = interval of `pat[..k]`, built left-to-right with `extend_right`.
        fn right_chain(idx: &BidirFmIndex, pat: &[u8]) -> Vec<BidirInterval> {
            let mut ivs = vec![idx.full_interval()];
            for &c in pat {
                let next = idx.extend_right(*ivs.last().unwrap(), c).unwrap();
                ivs.push(next);
            }
            ivs
        }

        /// Interval of `pat` built right-to-left with `extend_left` (independent route).
        fn left_only(idx: &BidirFmIndex, pat: &[u8]) -> BidirInterval {
            pat.iter().rev().fold(idx.full_interval(), |iv, &c| {
                idx.extend_left(iv, c).unwrap()
            })
        }

        fn assert_right_round_trip(idx: &BidirFmIndex, iv: BidirInterval, c: u8, ctx: &str) {
            let ext = idx.extend_right(iv, c).unwrap();
            let back = idx
                .contract_right(&ext, c)
                .unwrap_or_else(|e| panic!("{ctx}: contract_right({c}) failed: {e}"));
            assert_eq!(back, iv, "{ctx}: contract_right({c}) != original");
        }

        fn assert_right_chain(idx: &BidirFmIndex, pat: &[u8], ctx: &str) {
            let ivs = right_chain(idx, pat);
            for k in 0..pat.len() {
                let got = idx.contract_right(&ivs[k + 1], pat[k]).unwrap();
                assert_eq!(got, ivs[k], "{ctx} k={k}");
            }
        }

        #[test]
        fn contract_right_inverts_extend_right_on_random_iupac_multi_seq() {
            let mut rng = SmallRng::seed_from_u64(0x0DD5_1DE5);
            let mut saw_wild = false;
            let mut saw_sentinel = false;
            for &rate in &[1u32, 32] {
                for &enc in &[OccEncoding::Bitplane, OccEncoding::OneHot] {
                    for iter in 0..30 {
                        let nseq = rng.random_range(1..=4);
                        let seqs: Vec<Vec<u8>> = (0..nseq)
                            .map(|_| {
                                let len = rng.random_range(5..=120);
                                let runs = rng.random_range(0..=3);
                                random_seq(&mut rng, len, runs, iter % 3 == 0, iter % 4 == 0)
                            })
                            .collect();
                        let idx = build(&seqs, rate, enc);
                        assert!(idx.has_lcp());
                        for seq in &seqs {
                            for _ in 0..6 {
                                let plen = rng.random_range(1..=12.min(seq.len()));
                                let start = rng.random_range(0..=seq.len() - plen);
                                let pat = &seq[start..start + plen];
                                let ctx = format!("rate={rate} enc={enc:?} pat={}", show(pat));
                                let iv = left_only(&idx, pat);
                                assert_eq!(iv.len as usize, plen);
                                for c in 0..ALPHABET_SIZE as u8 {
                                    if idx.extend_right(iv, c).is_none() {
                                        continue;
                                    }
                                    saw_wild |= is_wild(c);
                                    saw_sentinel |= c == 0;
                                    assert_right_round_trip(&idx, iv, c, &ctx);
                                }
                                assert_right_chain(&idx, pat, &ctx);
                            }
                        }
                    }
                }
            }
            assert!(saw_wild, "no wildcard contraction exercised");
            assert!(saw_sentinel, "no sentinel contraction exercised");
        }

        #[test]
        fn contract_right_across_reference_wildcards_and_inside_wild_patterns() {
            let idx = build_str(&["ACGRYTACGN", "ACGTACGRYT", "NNACGTRR", "RYTACG"]);
            for pat in [
                "YTACG", "RYTACG", "TACG", "ACG", "NACGT", "RR", "N", "GRYT", "ACGR",
            ] {
                let pat = pat
                    .chars()
                    .map(|ch| alphabet::encode_char(ch).unwrap())
                    .collect::<Vec<_>>();
                assert_right_chain(&idx, &pat, &show(&pat));
                let iv = left_only(&idx, &pat);
                for c in 0..ALPHABET_SIZE as u8 {
                    if idx.extend_right(iv, c).is_some() {
                        assert_right_round_trip(&idx, iv, c, &show(&pat));
                    }
                }
            }
        }

        #[test]
        fn contract_right_undoes_every_child_of_compatible_fanout() {
            let mut rng = SmallRng::seed_from_u64(78);
            for iter in 0..20 {
                let seqs: Vec<Vec<u8>> = (0..rng.random_range(1..=3))
                    .map(|_| {
                        let len = rng.random_range(10..=80);
                        random_seq(&mut rng, len, 3, iter % 2 == 0, false)
                    })
                    .collect();
                let idx = build(&seqs, 1, OccEncoding::OneHot);
                for seq in &seqs {
                    let plen = rng.random_range(1..=6.min(seq.len()));
                    let start = rng.random_range(0..=seq.len() - plen);
                    let iv = left_only(&idx, &seq[start..start + plen]);
                    for &q in WILD_CODES.iter().chain(BASES.iter()) {
                        for c in idx.compatible_set(q).iter() {
                            if idx.extend_right(iv, c).is_some() {
                                assert_right_round_trip(&idx, iv, c, &format!("q={q}"));
                            }
                        }
                    }
                    // `children_right` slot `c` is `extend_right(c)`; each contracts back.
                    for (c, child) in idx.children_right(&iv).iter().enumerate() {
                        if let Some(child) = child {
                            assert_eq!(idx.contract_right(child, c as u8).unwrap(), iv);
                        }
                    }
                }
            }
        }

        #[test]
        fn contract_right_on_repeated_and_identical_sequences() {
            // Identical copies: occurrences of P are followed by different symbols
            // (sentinel vs base), so the reverse-half sub-range must be LCP-widened.
            let idx = build_str(&["ACGTACGT", "ACGTACGT", "ACGTACGT", "TACGTACG"]);
            let text: Vec<u8> = "ACGTACGT"
                .chars()
                .map(|c| alphabet::encode_char(c).unwrap())
                .collect();
            for start in 0..text.len() {
                for end in start + 1..=text.len() {
                    assert_right_chain(&idx, &text[start..end], "copies");
                }
            }
            let long_a = "A".repeat(200);
            let idx = build_str(&[&long_a, &long_a, "TAAAA"]);
            assert_right_chain(&idx, &[alphabet::A; 200], "A^200");
        }

        #[test]
        fn contract_both_ends_interleaved_on_ambiguous_patterns() {
            // Grow a cursor by a random mix of left/right extensions over IUPAC references,
            // then undo them in reverse order; every intermediate cursor must reappear.
            let mut rng = SmallRng::seed_from_u64(0xB0D1_B0D1);
            for iter in 0..40 {
                let seqs: Vec<Vec<u8>> = (0..rng.random_range(1..=3))
                    .map(|_| {
                        let len = rng.random_range(10..=100);
                        random_seq(&mut rng, len, 3, iter % 2 == 0, iter % 3 == 0)
                    })
                    .collect();
                let idx = build(
                    &seqs,
                    if iter % 2 == 0 { 1 } else { 16 },
                    OccEncoding::Bitplane,
                );
                for seq in &seqs {
                    let plen = rng.random_range(1..=15.min(seq.len()));
                    let start = rng.random_range(0..=seq.len() - plen);
                    let pat = &seq[start..start + plen];
                    // Pick a seed position inside the pattern and grow outwards.
                    let mut l = rng.random_range(0..plen);
                    let mut r = l;
                    let mut stack: Vec<(bool, u8, BidirInterval)> = Vec::new();
                    let mut iv = idx.full_interval();
                    while l > 0 || r < plen {
                        let go_left = r == plen || (l > 0 && rng.random_bool(0.5));
                        let (c, next) = if go_left {
                            l -= 1;
                            (pat[l], idx.extend_left(iv, pat[l]).unwrap())
                        } else {
                            let c = pat[r];
                            r += 1;
                            (c, idx.extend_right(iv, c).unwrap())
                        };
                        stack.push((go_left, c, iv));
                        iv = next;
                    }
                    assert_eq!(iv.len as usize, plen);
                    while let Some((was_left, c, prev)) = stack.pop() {
                        let back = if was_left {
                            idx.contract_left(&iv, c)
                        } else {
                            idx.contract_right(&iv, c)
                        };
                        let ctx = format!("pat={} left={was_left} c={c}", show(pat));
                        iv = back.unwrap_or_else(|e| panic!("{ctx}: {e}"));
                        assert_eq!(iv, prev, "{ctx}");
                    }
                    assert_eq!(iv, idx.full_interval());
                }
            }
        }

        #[test]
        fn contract_right_survives_serialization() {
            let seqs = vec![
                "ACGTRYACGTNNACGT"
                    .chars()
                    .map(|c| alphabet::encode_char(c).unwrap())
                    .collect::<Vec<_>>(),
                "TTACGTACGTAA"
                    .chars()
                    .map(|c| alphabet::encode_char(c).unwrap())
                    .collect::<Vec<_>>(),
            ];
            let idx = build(&seqs, 4, OccEncoding::Bitplane);
            let back = BidirFmIndex::from_bytes(&idx.to_bytes().unwrap()).unwrap();
            assert!(back.has_lcp());
            assert!(back.fwd().has_lcp() && back.rev().has_lcp());
            let pat = &seqs[0][3..9];
            let ivs = right_chain(&idx, pat);
            for k in 0..pat.len() {
                let a = idx.contract_right(&ivs[k + 1], pat[k]).unwrap();
                let b = back.contract_right(&ivs[k + 1], pat[k]).unwrap();
                assert_eq!(a, b);
                assert_eq!(a, ivs[k]);
            }
        }
    }

    /// Forward-only interval operations (`FwdInterval`) and k-mer seeding.
    mod fwd {
        use super::*;
        use haystackfm::{FmIndexError, FwdInterval};

        fn encode(s: &str) -> Vec<u8> {
            s.chars()
                .map(|ch| alphabet::encode_char(ch).unwrap())
                .collect()
        }

        fn right_walk(idx: &BidirFmIndex, pat: &[u8]) -> Option<BidirInterval> {
            pat.iter()
                .try_fold(idx.full_interval(), |iv, &c| idx.extend_right(iv, c))
        }

        /// Interval of `pat` on the forward half by backward search.
        fn fwd_interval(idx: &BidirFmIndex, pat: &[u8]) -> Option<FwdInterval> {
            pat.iter()
                .rev()
                .try_fold(FwdInterval::full(idx.text_len()), |iv, &c| {
                    idx.extend_left_fwd(&iv, c)
                })
        }

        fn oracle_parent(idx: &BidirFmIndex, pat: &[u8], iv: FwdInterval) -> Option<FwdInterval> {
            (0..iv.len as usize)
                .rev()
                .map(|d| fwd_interval(idx, &pat[..d]).unwrap())
                .find(|p| p.size() > iv.size())
        }

        fn check_parent_chain(idx: &BidirFmIndex, pat: &[u8], ctx: &str) {
            let mut iv = right_walk(idx, pat).unwrap().fwd();
            assert_eq!(iv, fwd_interval(idx, pat).unwrap(), "{ctx}");
            loop {
                let want = oracle_parent(idx, pat, iv);
                let got = idx.parent_fwd(&iv).unwrap();
                assert_eq!(got, want, "{ctx}: parent at depth {}", iv.len);
                match got {
                    Some(p) => iv = p,
                    None => break,
                }
            }
        }

        #[test]
        fn fwd_extend_left_matches_bidir_forward_half() {
            let mut rng = SmallRng::seed_from_u64(11);
            for _ in 0..20 {
                let seqs: Vec<Vec<u8>> = (0..rng.random_range(1..=3))
                    .map(|_| {
                        let len = rng.random_range(8..=100);
                        random_seq(&mut rng, len, 2, false, true)
                    })
                    .collect();
                let idx = build(&seqs, 1, OccEncoding::Bitplane);
                for seq in &seqs {
                    let plen = rng.random_range(1..=8.min(seq.len()));
                    let start = rng.random_range(0..=seq.len() - plen);
                    let pat = &seq[start..start + plen];
                    let iv = right_walk(&idx, pat).unwrap();
                    for c in 0..ALPHABET_SIZE as u8 {
                        let bi = idx.extend_left(iv, c).map(|x| x.fwd());
                        let fw = idx.extend_left_fwd(&iv.fwd(), c);
                        assert_eq!(bi, fw, "pat={} c={c}", show(pat));
                    }
                }
            }
        }

        #[test]
        fn parent_chain_matches_oracle_on_random_iupac_multi_seq() {
            let mut rng = SmallRng::seed_from_u64(0xA11CE);
            for &rate in &[1u32, 32] {
                for &enc in &[OccEncoding::Bitplane, OccEncoding::OneHot] {
                    for iter in 0..25 {
                        let seqs: Vec<Vec<u8>> = (0..rng.random_range(1..=4))
                            .map(|_| {
                                let len = rng.random_range(5..=120);
                                let runs = rng.random_range(0..=3);
                                random_seq(&mut rng, len, runs, iter % 3 == 0, iter % 4 == 0)
                            })
                            .collect();
                        let idx = build(&seqs, rate, enc);
                        for seq in &seqs {
                            for _ in 0..5 {
                                let plen = rng.random_range(1..=14.min(seq.len()));
                                let start = rng.random_range(0..=seq.len() - plen);
                                let pat = &seq[start..start + plen];
                                let ctx = format!("rate={rate} enc={enc:?} pat={}", show(pat));
                                check_parent_chain(&idx, pat, &ctx);
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn parent_chain_on_repeats_and_identical_sequences() {
            let idx = build_str(&["ACGTACGT", "ACGTACGT", "ACGTACGT", "TACGTACG"]);
            let text = encode("ACGTACGT");
            for start in 0..text.len() {
                for end in start + 1..=text.len() {
                    check_parent_chain(&idx, &text[start..end], "identical");
                }
            }
            let long_a = "A".repeat(200);
            let idx = build_str(&[&long_a, &long_a, "AAAAT"]);
            check_parent_chain(&idx, &vec![alphabet::A; 200], "A200");
            check_parent_chain(&idx, &encode("AAAAT"), "AAAAT");
            let tandem = "ACGT".repeat(100);
            let idx = build_str(&[&tandem]);
            for start in 0..4 {
                check_parent_chain(&idx, &encode(&tandem[start..]), "tandem");
            }
        }

        #[test]
        fn parent_pieces_are_rows_diverging_at_parent_depth() {
            // Rows of the parent outside the child share exactly `parent.len` symbols with
            // the child's string and differ at the next one.
            let idx = build_str(&["ACGTACGTTTGACCAGGTACGTACGAAATTTCCCGGGACGTAC", "GTACGAAT"]);
            let seq0 = idx.sequence(haystackfm::SeqId::new(0)).unwrap().to_vec();
            for start in 0..seq0.len() {
                for end in start + 1..=seq0.len().min(start + 10) {
                    let pat = &seq0[start..end];
                    let child = right_walk(&idx, pat).unwrap().fwd();
                    let Some(parent) = idx.parent_fwd(&child).unwrap() else {
                        continue;
                    };
                    let d = parent.len as usize;
                    for piece in [
                        FwdInterval {
                            lo: parent.lo,
                            hi: child.lo,
                            len: parent.len,
                        },
                        FwdInterval {
                            lo: child.hi,
                            hi: parent.hi,
                            len: parent.len,
                        },
                    ] {
                        for (id, off) in idx.locate_fwd(&piece) {
                            let s = idx.sequence(id).unwrap();
                            let off = off as usize;
                            assert_eq!(&s[off..off + d], &pat[..d]);
                            assert_ne!(
                                s.get(off + d),
                                pat.get(d),
                                "piece row must diverge at depth {d}"
                            );
                        }
                    }
                }
            }
        }

        #[test]
        fn locate_rows_matches_locate_interval_and_pattern_locate() {
            // ACGT-only references: `FmIndex::locate` is IUPAC-aware and would otherwise
            // also report reference wildcards the exact cursor walk excludes.
            let seqs = vec![encode("ACGTAGACGTCCACGT"), encode("TTACGTACGTAA")];
            let idx = build(&seqs, 4, OccEncoding::Bitplane);
            for pat in ["ACGT", "A", "TA", "CGT"] {
                let pat = encode(pat);
                let iv = right_walk(&idx, &pat).unwrap();
                let mut a = idx.locate_interval(&iv);
                let mut b = idx.locate_fwd(&iv.fwd());
                let mut c = idx.fwd().locate_rows(iv.fwd_lo, iv.fwd_hi);
                let mut d = idx.fwd().locate(&pat);
                for v in [&mut a, &mut b, &mut c, &mut d] {
                    v.sort();
                }
                assert_eq!(a, b);
                assert_eq!(a, c);
                assert_eq!(a, d);
            }
        }

        fn build_lookup(seqs: &[&str], depth: u32) -> BidirFmIndex {
            let dna: Vec<DnaSequence> = seqs
                .iter()
                .map(|s| DnaSequence::from_str(s).unwrap())
                .collect();
            BidirFmIndex::build_cpu(
                &dna,
                &FmIndexConfig {
                    sa_sample_rate: 2,
                    use_gpu: false,
                    lookup_depth: depth,
                    ..Default::default()
                },
            )
            .unwrap()
        }

        fn all_kmers(k: usize) -> Vec<Vec<u8>> {
            let mut out = vec![Vec::new()];
            for _ in 0..k {
                out = out
                    .iter()
                    .flat_map(|p| {
                        BASES.iter().map(move |&b| {
                            let mut q = p.clone();
                            q.push(b);
                            q
                        })
                    })
                    .collect();
            }
            out
        }

        #[test]
        fn lookup_interval_matches_extend_right_walk() {
            let refs = [
                "ACGTACGTTTGACCAGGTACGTACGAAATTTCCCGGGACGTAC",
                "GTACGAATNNACGT",
                "RYACGT",
            ];
            for depth in [3u32, 4] {
                let idx = build_lookup(&refs, depth);
                assert_eq!(idx.lookup_depth(), depth);
                let mut hits = 0;
                for kmer in all_kmers(depth as usize) {
                    let want = right_walk(&idx, &kmer);
                    let got = idx.lookup_interval(&kmer);
                    assert_eq!(got, want, "kmer={}", show(&kmer));
                    hits += got.is_some() as u32;
                }
                assert!(hits > 0);
                // Seeded cursors extend and contract like walked ones.
                let seed = idx
                    .lookup_interval(&encode(&"ACGT"[..depth as usize]))
                    .unwrap();
                let ext = idx.extend_left(seed, alphabet::T).unwrap();
                assert_eq!(idx.contract_left(&ext, alphabet::T).unwrap(), seed);
                // Wrong length / non-core symbol.
                assert_eq!(idx.lookup_interval(&encode("AC")), None);
                assert_eq!(
                    idx.lookup_interval(&vec![alphabet::N; depth as usize]),
                    None
                );
                // Survives serialization.
                let back = BidirFmIndex::from_bytes(&idx.to_bytes().unwrap()).unwrap();
                assert_eq!(back.lookup_depth(), depth);
                assert_eq!(
                    back.lookup_interval(&encode(&"ACGT"[..depth as usize])),
                    Some(seed)
                );
            }
            let no_lookup = build_str(&refs);
            assert_eq!(no_lookup.lookup_depth(), 0);
            assert_eq!(no_lookup.lookup_interval(&encode("ACG")), None);
        }

        #[test]
        fn parent_reports_missing_lcp() {
            let dna = vec![DnaSequence::from_str("ACGTACGT").unwrap()];
            let idx = BidirFmIndex::build_cpu(
                &dna,
                &FmIndexConfig {
                    sa_sample_rate: 1,
                    use_gpu: false,
                    build_lcp: false,
                    ..Default::default()
                },
            )
            .unwrap();
            let iv = right_walk(&idx, &encode("ACG")).unwrap().fwd();
            assert!(matches!(
                idx.parent_fwd(&iv),
                Err(FmIndexError::LcpNotBuilt)
            ));
        }
    }
}
