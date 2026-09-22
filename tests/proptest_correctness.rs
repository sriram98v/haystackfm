use haystackfm::alphabet::encode_char;
use haystackfm::{
    BidirFmIndex, BidirInterval, DnaSequence, FmIndex, FmIndexConfig, OccEncoding, SeqId,
};
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

/// Brute-force hits across multiple sequences → `(seq_id, position)` set.
fn naive_hits_multi(texts: &[String], pattern: &str) -> HashSet<(SeqId, u32)> {
    texts
        .iter()
        .enumerate()
        .flat_map(|(i, text)| {
            let id = SeqId::new(i as u32);
            naive_positions(text, pattern)
                .into_iter()
                .map(move |p| (id, p))
        })
        .collect()
}

/// IUPAC reference text: mostly ACGT with a sprinkling of ambiguity codes, wrapped in
/// optional wildcard runs (0..=7) at both ends so sequence boundaries get covered.
fn iupac_string(max_len: usize) -> impl Strategy<Value = String> {
    const WILD: [char; 11] = ['N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V'];
    let body_char = prop_oneof![
        9 => (0usize..4).prop_map(|i| ['A', 'C', 'G', 'T'][i]),
        1 => (0usize..WILD.len()).prop_map(|i| WILD[i]),
    ];
    let wild_run = prop::collection::vec((0usize..WILD.len()).prop_map(|i| WILD[i]), 0..=7);
    (
        wild_run.clone(),
        prop::collection::vec(body_char, 1..=max_len),
        wild_run,
    )
        .prop_map(|(head, body, tail)| head.into_iter().chain(body).chain(tail).collect::<String>())
}

fn build_bidir(texts: &[String], sa_sample_rate: usize, onehot: bool) -> BidirFmIndex {
    let seqs: Vec<DnaSequence> = texts
        .iter()
        .map(|s| DnaSequence::from_str(s).unwrap())
        .collect();
    BidirFmIndex::build_cpu(
        &seqs,
        &FmIndexConfig {
            sa_sample_rate: sa_sample_rate as u32,
            use_gpu: false,
            occ_encoding: if onehot {
                OccEncoding::OneHot
            } else {
                OccEncoding::Bitplane
            },
            ..Default::default()
        },
    )
    .unwrap()
}

/// Brute-force `(count_wild_left, count_wild_right)`: locate every occurrence and inspect
/// the neighbouring reference symbol (code >= 5 is an ambiguity code; a sequence boundary
/// is never wild).
fn neighbour_wild_counts(idx: &BidirFmIndex, iv: &BidirInterval, len: usize) -> (u32, u32) {
    let (mut left, mut right) = (0, 0);
    for (id, pos) in idx.locate_interval(iv) {
        let seq = idx.sequence(id).unwrap();
        let pos = pos as usize;
        if pos > 0 && seq[pos - 1] >= 5 {
            left += 1;
        }
        if seq.get(pos + len).is_some_and(|&c| c >= 5) {
            right += 1;
        }
    }
    (left, right)
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

    /// `locate` must return exactly the right `(seq_id, position)` pairs across
    /// multiple sequences, at any SA sampling rate.
    #[test]
    fn locate_correct_multi_sequence(
        texts in prop::collection::vec(dna_string(200), 1..=5),
        pattern in dna_string(10),
        sa_sample_rate in 1usize..=32,
    ) {
        let idx = build_index(&texts, sa_sample_rate);
        let pat = encode_pat(&pattern);

        let fm_hits: HashSet<(SeqId, u32)> = idx.locate(&pat).into_iter().collect();
        let expected = naive_hits_multi(&texts, &pattern);

        prop_assert_eq!(
            fm_hits, expected,
            "multi-seq locate mismatch | pattern='{}' rate={}", pattern, sa_sample_rate
        );
    }

    /// The header <-> id accessors must be exact inverses for every indexed sequence,
    /// and every located id must be in range.
    #[test]
    fn seq_id_and_seq_header_are_inverses(
        texts in prop::collection::vec(dna_string(200), 1..=5),
        pattern in dna_string(10),
    ) {
        let idx = build_index(&texts, 1);

        for i in 0..texts.len() {
            let id = SeqId::new(i as u32);
            let header = idx.seq_header(id).expect("id in range must have a header");
            prop_assert_eq!(idx.seq_id(header), Some(id));
        }
        prop_assert_eq!(idx.seq_header(SeqId::new(texts.len() as u32)), None);

        for (id, _) in idx.locate(&encode_pat(&pattern)) {
            prop_assert!(id.index() < texts.len(), "located id {} out of range", id);
        }
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
    /// Wildcard-aware cursor counts must agree with a neighbour scan of every located
    /// occurrence, at every step of a right walk, on multi-sequence IUPAC references.
    #[test]
    fn bidir_count_wild_matches_neighbour_scan(
        texts in prop::collection::vec(iupac_string(150), 1..=4),
        pattern in dna_string(6),
        sa_sample_rate in 1usize..=32,
        onehot in any::<bool>(),
    ) {
        let idx = build_bidir(&texts, sa_sample_rate, onehot);
        let pat = encode_pat(&pattern);
        let mut iv = idx.full_interval();
        for (k, &c) in pat.iter().enumerate() {
            let Some(next) = idx.extend_right(iv, c) else { break };
            iv = next;
            let (left, right) = neighbour_wild_counts(&idx, &iv, k + 1);
            prop_assert_eq!(
                idx.count_wild_right(&iv), right,
                "count_wild_right | pattern='{}' step={} texts={:?}", pattern, k, texts
            );
            prop_assert_eq!(
                idx.count_wild_left(&iv), left,
                "count_wild_left | pattern='{}' step={} texts={:?}", pattern, k, texts
            );
        }
    }

    /// Every child from `children_*` must equal the corresponding single extension, and the
    /// children (sentinel slot included) must partition the parent interval.
    #[test]
    fn bidir_children_match_extend_and_sum_to_size(
        texts in prop::collection::vec(iupac_string(120), 1..=3),
        pattern in dna_string(5),
        onehot in any::<bool>(),
    ) {
        let idx = build_bidir(&texts, 1, onehot);
        let pat = encode_pat(&pattern);
        let mut iv = idx.full_interval();
        for &c in &pat {
            let Some(next) = idx.extend_right(iv, c) else { break };
            iv = next;
        }
        let right = idx.children_right(&iv);
        let left = idx.children_left(&iv);
        let mut sum_right = 0u32;
        let mut sum_left = 0u32;
        for code in 0..16u8 {
            prop_assert_eq!(right[code as usize], idx.extend_right(iv, code), "right child {}", code);
            prop_assert_eq!(left[code as usize], idx.extend_left(iv, code), "left child {}", code);
            sum_right += right[code as usize].map_or(0, |x| x.size());
            sum_left += left[code as usize].map_or(0, |x| x.size());
        }
        prop_assert_eq!(sum_right, iv.size());
        prop_assert_eq!(sum_left, iv.size());
    }

    /// `contract_left` is the exact inverse of `extend_left` for every code (sentinel and
    /// wildcards included) at every step of a right walk, on multi-sequence IUPAC
    /// references, at every sampling rate and both occ encodings.
    #[test]
    fn bidir_contract_left_inverts_extend_left(
        texts in prop::collection::vec(iupac_string(150), 1..=4),
        pattern in dna_string(8),
        sa_sample_rate in 1usize..=32,
        onehot in any::<bool>(),
    ) {
        let idx = build_bidir(&texts, sa_sample_rate, onehot);
        prop_assert!(idx.has_lcp());
        let pat = encode_pat(&pattern);
        let mut iv = idx.full_interval();
        for (k, &c) in pat.iter().enumerate() {
            let Some(next) = idx.extend_right(iv, c) else { break };
            iv = next;
            prop_assert_eq!(iv.len as usize, k + 1);
            for code in 0..16u8 {
                let Some(ext) = idx.extend_left(iv, code) else { continue };
                let back = idx.contract_left(&ext, code);
                prop_assert!(back.is_ok(), "contract_left({}) errored: {:?} | pattern='{}' step={} texts={:?}",
                    code, back.err(), pattern, k, texts);
                prop_assert_eq!(
                    back.unwrap(), iv,
                    "contract_left({}) | pattern='{}' step={} texts={:?}", code, pattern, k, texts
                );
            }
        }
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
    assert_eq!(hits, vec![(SeqId::new(0), 0)]);
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
fn multi_seq_ids_correct() {
    let texts = vec!["ACGT".to_string(), "TTTT".to_string(), "GGGG".to_string()];
    let idx = build_index(&texts, 1);

    for (i, expected) in ["seq_0", "seq_1", "seq_2"].iter().enumerate() {
        let id = SeqId::new(i as u32);
        assert_eq!(idx.seq_header(id), Some(*expected));
        assert_eq!(idx.seq_id(expected), Some(id));
    }

    let a_hits: HashSet<(SeqId, u32)> = idx.locate(&encode_pat("A")).into_iter().collect();
    assert_eq!(a_hits, [(SeqId::new(0), 0)].into_iter().collect());

    let t_hits: HashSet<(SeqId, u32)> = idx.locate(&encode_pat("T")).into_iter().collect();
    // "ACGT" has T at 3, "TTTT" has T at 0,1,2,3, "GGGG" has none.
    assert!(t_hits.contains(&(SeqId::new(0), 3)));
    for p in 0..4u32 {
        assert!(t_hits.contains(&(SeqId::new(1), p)));
    }
    assert!(!t_hits.iter().any(|&(id, _)| id == SeqId::new(2)));
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
