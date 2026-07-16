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
