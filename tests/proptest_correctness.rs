/// Property-based correctness tests for the FM-index.
///
/// Inspired by genedex (https://github.com/feldroop/genedex). For every randomly
/// generated DNA text and query, the FM-index must agree with a brute-force
/// sliding-window search. Tests cover:
///
/// - `count` == `locate().len()` for all inputs
/// - `locate` positions match brute-force for a single sequence
/// - `locate` positions match brute-force across multiple sequences
/// - `locate` is exact regardless of SA sampling rate
/// - Any substring extracted from the indexed text must appear in `locate` results
use proptest::prelude::*;
use std::collections::HashSet;
use webgpu_fmidx::alphabet::encode_char;
use webgpu_fmidx::{DnaSequence, FmIndex, FmIndexConfig};

// ── Test helpers ──────────────────────────────────────────────────────────────

fn dna_string(max_len: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(
        (0usize..4).prop_map(|i| ['A', 'C', 'G', 'T'][i]),
        1..=max_len,
    )
    .prop_map(|chars| chars.into_iter().collect::<String>())
}

fn encode_pat(s: &str) -> Vec<u8> {
    s.chars().map(|c| encode_char(c).unwrap()).collect()
}

fn build_index(texts: &[String], sa_sample_rate: usize) -> FmIndex {
    let seqs: Vec<DnaSequence> = texts
        .iter()
        .map(|s| DnaSequence::from_str(s).unwrap())
        .collect();
    FmIndex::build_cpu(
        &seqs,
        &FmIndexConfig {
            sa_sample_rate: sa_sample_rate as u32,
            use_gpu: false,
            ..Default::default()
        },
    )
    .unwrap()
}

/// Brute-force positions of `pattern` in `text` (0-based, overlapping).
fn naive_positions(text: &str, pattern: &str) -> Vec<u32> {
    if pattern.is_empty() || pattern.len() > text.len() {
        return vec![];
    }
    (0..=text.len() - pattern.len())
        .filter(|&i| &text[i..i + pattern.len()] == pattern)
        .map(|i| i as u32)
        .collect()
}

/// Brute-force hits across multiple sequences → `(header, position)` set.
fn naive_hits_multi(texts: &[String], pattern: &str) -> HashSet<(String, u32)> {
    texts
        .iter()
        .enumerate()
        .flat_map(|(i, text)| {
            let header = format!("seq_{i}");
            naive_positions(text, pattern)
                .into_iter()
                .map(move |p| (header.clone(), p))
        })
        .collect()
}

// ── Property tests ────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 512,
        failure_persistence: Some(Box::new(
            prop::test_runner::FileFailurePersistence::WithSource("proptest-regressions"),
        )),
        ..Default::default()
    })]

    /// `count` must equal the number of hits returned by `locate`.
    #[test]
    fn count_equals_locate_len(
        text in dna_string(500),
        pattern in dna_string(20),
        sa_sample_rate in 1usize..=32,
    ) {
        let idx = build_index(&[text.clone()], sa_sample_rate);
        let pat = encode_pat(&pattern);
        let count = idx.count(&pat);
        let hits = idx.locate(&pat);
        prop_assert_eq!(
            count as usize,
            hits.len(),
            "count={} but locate returned {} hits | pattern='{}' text='{}'",
            count, hits.len(), pattern, text,
        );
    }

    /// `locate` positions on a single sequence must exactly match brute-force.
    #[test]
    fn locate_matches_naive_single_text(
        text in dna_string(300),
        pattern in dna_string(15),
    ) {
        let idx = build_index(&[text.clone()], 1);
        let pat = encode_pat(&pattern);

        let mut fm: Vec<u32> = idx.locate(&pat).into_iter().map(|(_, p)| p).collect();
        fm.sort_unstable();

        let mut expected = naive_positions(&text, &pattern);
        expected.sort_unstable();

        prop_assert_eq!(
            fm, expected,
            "locate mismatch | pattern='{}' text='{}'", pattern, text
        );
    }

    /// `locate` must return exactly the right `(header, position)` pairs across
    /// multiple sequences.
    #[test]
    fn locate_correct_multi_sequence(
        texts in prop::collection::vec(dna_string(200), 1..=5),
        pattern in dna_string(10),
    ) {
        let idx = build_index(&texts, 1);
        let pat = encode_pat(&pattern);

        let fm_hits: HashSet<(String, u32)> = idx.locate(&pat).into_iter().collect();
        let expected = naive_hits_multi(&texts, &pattern);

        prop_assert_eq!(
            fm_hits, expected,
            "multi-seq locate mismatch | pattern='{}'", pattern
        );
    }

    /// `locate` must return exact positions regardless of SA sampling rate.
    #[test]
    fn locate_correct_with_various_sampling_rates(
        text in dna_string(300),
        pattern in dna_string(15),
        sa_sample_rate in 1usize..=32,
    ) {
        let idx = build_index(&[text.clone()], sa_sample_rate);
        let pat = encode_pat(&pattern);

        let mut fm: Vec<u32> = idx.locate(&pat).into_iter().map(|(_, p)| p).collect();
        fm.sort_unstable();

        let mut expected = naive_positions(&text, &pattern);
        expected.sort_unstable();

        prop_assert_eq!(
            fm, expected,
            "sampling_rate={}: locate mismatch | pattern='{}' text='{}'", sa_sample_rate, pattern, text
        );
    }

    /// Any substring extracted directly from the indexed text must appear in
    /// `locate` results at a position ≤ its extraction offset.
    #[test]
    fn existing_substrings_always_found(
        text in dna_string(300),
        start_frac in 0.0f64..1.0,
        len in 1usize..=20,
        sa_sample_rate in 1usize..=16,
    ) {
        let n = text.len();
        let start = ((start_frac * n as f64) as usize).min(n.saturating_sub(1));
        let end = (start + len).min(n);
        prop_assume!(start < end);
        let pattern = text[start..end].to_string();

        let idx = build_index(&[text.clone()], sa_sample_rate);
        let pat = encode_pat(&pattern);

        let positions: HashSet<u32> = idx.locate(&pat).into_iter().map(|(_, p)| p).collect();

        prop_assert!(
            positions.contains(&(start as u32)),
            "substring '{pattern}' at offset {start} not in locate results | text='{text}'"
        );
    }
}

// ── Deterministic edge-case tests ─────────────────────────────────────────────

#[test]
fn single_char_repeated() {
    for c in ["A", "C", "G", "T"] {
        let text = c.repeat(10);
        let idx = build_index(&[text.clone()], 1);
        let pat = encode_pat(c);
        assert_eq!(idx.count(&pat), 10);
        let mut positions: Vec<u32> = idx.locate(&pat).into_iter().map(|(_, p)| p).collect();
        positions.sort_unstable();
        assert_eq!(positions, (0u32..10).collect::<Vec<_>>());
    }
}

#[test]
fn pattern_longer_than_text_returns_empty() {
    let idx = build_index(&["ACG".to_string()], 1);
    assert_eq!(idx.count(&encode_pat("ACGT")), 0);
    assert!(idx.locate(&encode_pat("ACGT")).is_empty());
}

#[test]
fn pattern_equals_text() {
    let text = "ACGTACGT".to_string();
    let idx = build_index(&[text.clone()], 1);
    let pat = encode_pat(&text);
    assert_eq!(idx.count(&pat), 1);
    let hits = idx.locate(&pat);
    assert_eq!(hits, vec![("seq_0".to_string(), 0)]);
}

#[test]
fn overlapping_pattern_count_correct() {
    // "AA" appears 3 times in "AAAA" (positions 0,1,2)
    let idx = build_index(&["AAAA".to_string()], 1);
    let pat = encode_pat("AA");
    assert_eq!(idx.count(&pat), 3);
    let mut positions: Vec<u32> = idx.locate(&pat).into_iter().map(|(_, p)| p).collect();
    positions.sort_unstable();
    assert_eq!(positions, vec![0, 1, 2]);
}

#[test]
fn multi_seq_headers_correct() {
    let texts = vec!["ACGT".to_string(), "TTTT".to_string(), "GGGG".to_string()];
    let idx = build_index(&texts, 1);

    let a_hits: HashSet<(String, u32)> = idx.locate(&encode_pat("A")).into_iter().collect();
    assert_eq!(a_hits, [("seq_0".to_string(), 0)].into_iter().collect());

    let t_hits: HashSet<(String, u32)> = idx.locate(&encode_pat("T")).into_iter().collect();
    // "ACGT" has T at 3, "TTTT" has T at 0,1,2,3
    assert!(t_hits.contains(&("seq_0".to_string(), 3)));
    for p in 0..4u32 {
        assert!(t_hits.contains(&("seq_1".to_string(), p)));
    }
    assert!(!t_hits.iter().any(|(h, _)| h == "seq_2"));
}

#[test]
fn pattern_not_in_text_returns_zero() {
    let idx = build_index(&["AAAA".to_string()], 1);
    assert_eq!(idx.count(&encode_pat("C")), 0);
    assert_eq!(idx.count(&encode_pat("AAAC")), 0);
    assert!(idx.locate(&encode_pat("G")).is_empty());
}

#[test]
fn seeded_random_correctness() {
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xDEADBEEF_CAFEBABE);
    let bases = b"ACGT";

    for _ in 0..200 {
        let text_len = rng.random_range(5usize..=200);
        let text: String = (0..text_len)
            .map(|_| bases[rng.random_range(0..4)] as char)
            .collect();

        let pat_len = rng.random_range(1usize..=15.min(text_len));
        let pattern: String = (0..pat_len)
            .map(|_| bases[rng.random_range(0..4)] as char)
            .collect();

        let sa_rate = rng.random_range(1usize..=16);
        let idx = build_index(&[text.clone()], sa_rate);
        let pat = encode_pat(&pattern);

        let count = idx.count(&pat) as usize;
        let mut positions: Vec<u32> = idx.locate(&pat).into_iter().map(|(_, p)| p).collect();
        positions.sort_unstable();

        let mut expected = naive_positions(&text, &pattern);
        expected.sort_unstable();

        assert_eq!(
            count,
            expected.len(),
            "count mismatch | pattern='{pattern}' text='{text}' rate={sa_rate}"
        );
        assert_eq!(
            positions, expected,
            "locate mismatch | pattern='{pattern}' text='{text}' rate={sa_rate}"
        );
    }
}
