//! Move-structure correctness: `MoveFmIndex` / `BMoveIndex` must agree with the brute-force
//! oracle and with `FmIndex` / `BidirFmIndex` over the same references (identical SA
//! intervals), locate every occurrence from a toehold, and round-trip through bytes.

use haystackfm::alphabet::{encode_char, ExactDna, ALPHABET_SIZE};
use haystackfm::{
    BMoveCursor, BMoveIndex, BidirFmIndex, DnaSequence, FmIndex, FmIndexConfig, MoveFmIndex,
    MoveFmIndexConfig, SeqId, SymbolSet,
};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const WILD: [char; 11] = ['N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V'];

fn dna_string(max_len: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(
        (0usize..4).prop_map(|i| ['A', 'C', 'G', 'T'][i]),
        1..=max_len,
    )
    .prop_map(|c| c.into_iter().collect())
}

fn iupac_string(max_len: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            9 => (0usize..4).prop_map(|i| ['A', 'C', 'G', 'T'][i]),
            1 => (0usize..WILD.len()).prop_map(|i| WILD[i]),
        ],
        1..=max_len,
    )
    .prop_map(|c| c.into_iter().collect())
}

fn repetitive_string() -> impl Strategy<Value = String> {
    (dna_string(12), 2usize..12, any::<u64>()).prop_map(|(unit, copies, seed)| {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut s: Vec<char> = unit.repeat(copies).chars().collect();
        if rng.random_bool(0.5) {
            let at = rng.random_range(0..s.len());
            s[at] = ['A', 'C', 'G', 'T', 'N'][rng.random_range(0..5)];
        }
        s.into_iter().collect()
    })
}

fn texts() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(
        prop_oneof![dna_string(50), iupac_string(40), repetitive_string()],
        1..4,
    )
}

fn seqs(texts: &[String]) -> Vec<DnaSequence> {
    texts
        .iter()
        .map(|s| DnaSequence::from_str(s).unwrap())
        .collect()
}

fn encode(s: &str) -> Vec<u8> {
    s.chars().map(|c| encode_char(c).unwrap()).collect()
}

fn fm_config() -> FmIndexConfig {
    FmIndexConfig {
        sa_sample_rate: 4,
        use_gpu: false,
        ..Default::default()
    }
}

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}

fn naive_hits(texts: &[String], pattern: &str) -> Vec<(SeqId, u32)> {
    let mut out = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        if pattern.len() <= text.len() {
            for p in 0..=text.len() - pattern.len() {
                if &text[p..p + pattern.len()] == pattern {
                    out.push((SeqId::new(i as u32), p as u32));
                }
            }
        }
    }
    out.sort();
    out
}

/// The concatenated text (codes, sentinels included), for toehold checks.
fn concat(texts: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    for t in texts {
        out.extend(encode(t));
        out.push(0);
    }
    out
}

fn assert_cursor(
    fm: &BidirFmIndex,
    bm: &BMoveIndex,
    cur: &BMoveCursor,
    text: &[u8],
    pattern: &[u8],
) {
    let iv = cur.interval();
    assert_eq!(iv.len as usize, pattern.len());
    // The toehold starts an occurrence of the pattern.
    let x = cur.toehold().expect("non-sentinel cursor has a toehold") as usize;
    assert_eq!(
        &text[x..x + pattern.len()],
        pattern,
        "toehold {x} is not an occurrence"
    );
    assert_eq!(sorted(bm.locate(cur)), sorted(fm.locate_interval(&iv)));
    let fm_children = fm.children_right(&iv);
    for (c, child) in bm.children_right(cur).iter().enumerate() {
        assert_eq!(child.map(|c| c.interval()), fm_children[c]);
    }
    let fm_children = fm.children_left(&iv);
    for (c, child) in bm.children_left(cur).iter().enumerate() {
        assert_eq!(child.map(|c| c.interval()), fm_children[c]);
    }
    for set in [SymbolSet::WILDCARDS, SymbolSet::below(3), SymbolSet::ALL] {
        assert_eq!(bm.count_right_in(cur, set), fm.count_right_in(&iv, set));
        assert_eq!(bm.count_left_in(cur, set), fm.count_left_in(&iv, set));
    }
    assert_eq!(bm.count_wild_right(cur), fm.count_wild_right(&iv));
    assert_eq!(bm.count_wild_left(cur), fm.count_wild_left(&iv));
    for q in 1..ALPHABET_SIZE as u8 {
        assert!(bm
            .extend_right_compatible(cur, q)
            .map(|c| c.interval())
            .eq(fm.extend_right_compatible(iv, q)));
        assert!(bm
            .extend_left_compatible(cur, q)
            .map(|c| c.interval())
            .eq(fm.extend_left_compatible(iv, q)));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn count_and_locate_match_brute_force(
        texts in prop::collection::vec(prop_oneof![dna_string(50), dna_string(8).prop_map(|u| u.repeat(9))], 1..4),
        pattern in dna_string(6),
    ) {
        let mv = MoveFmIndex::build_cpu(&seqs(&texts), &MoveFmIndexConfig::default()).unwrap();
        let expected = naive_hits(&texts, &pattern);
        let p = encode(&pattern);
        prop_assert_eq!(mv.count(&p) as usize, expected.len());
        prop_assert_eq!(sorted(mv.locate(&p)), expected);
    }

    #[test]
    fn iupac_queries_match_fm_index(texts in texts(), pattern in iupac_string(5)) {
        let s = seqs(&texts);
        let p = encode(&pattern);
        let fm = FmIndex::build_cpu(&s, &fm_config()).unwrap();
        let mv = MoveFmIndex::build_cpu(&s, &MoveFmIndexConfig::default()).unwrap();
        prop_assert_eq!(mv.count(&p), fm.count(&p));
        prop_assert_eq!(sorted(mv.locate(&p)), sorted(fm.locate(&p)));
        prop_assert_eq!(sorted(mv.locate_positions(&p)), sorted(fm.locate_positions(&p)));

        let fm = FmIndex::build_cpu_with::<ExactDna>(&s, &fm_config()).unwrap();
        let mv = MoveFmIndex::build_cpu_with::<ExactDna>(&s, &MoveFmIndexConfig::default()).unwrap();
        prop_assert_eq!(mv.count(&p), fm.count(&p));
        prop_assert_eq!(sorted(mv.locate(&p)), sorted(fm.locate(&p)));
    }

    #[test]
    fn bmove_cursor_matches_bidir_fm_index(
        texts in texts(),
        steps in prop::collection::vec((any::<bool>(), 1u8..16), 0..10),
    ) {
        let s = seqs(&texts);
        let text = concat(&texts);
        let fm = BidirFmIndex::build_cpu(&s, &fm_config()).unwrap();
        let bm = BMoveIndex::build_cpu(&s, &MoveFmIndexConfig::default()).unwrap();
        let mut cur = bm.full_cursor();
        prop_assert_eq!(cur.interval(), fm.full_interval());
        let mut pattern: Vec<u8> = Vec::new();
        assert_cursor(&fm, &bm, &cur, &text, &pattern);
        for (right, c) in steps {
            let next = if right { bm.extend_right(&cur, c) } else { bm.extend_left(&cur, c) };
            let iv = cur.interval();
            let fm_next = if right { fm.extend_right(iv, c) } else { fm.extend_left(iv, c) };
            prop_assert_eq!(next.map(|n| n.interval()), fm_next);
            let Some(next) = next else { continue };
            cur = next;
            if right { pattern.push(c) } else { pattern.insert(0, c) }
            assert_cursor(&fm, &bm, &cur, &text, &pattern);
        }
    }

    #[test]
    fn serialization_round_trips(texts in texts(), pattern in iupac_string(4), locate in any::<bool>()) {
        let s = seqs(&texts);
        let p = encode(&pattern);
        let config = MoveFmIndexConfig { locate, ..Default::default() };
        let mv = MoveFmIndex::build_cpu(&s, &config).unwrap();
        let bytes = mv.to_bytes().unwrap();
        let back = MoveFmIndex::from_bytes(&bytes).unwrap();
        prop_assert_eq!(back.to_bytes().unwrap(), bytes);
        prop_assert_eq!(back.count(&p), mv.count(&p));
        prop_assert_eq!(back.can_locate(), locate);
        if locate {
            prop_assert_eq!(sorted(back.locate(&p)), sorted(mv.locate(&p)));
        }

        let bm = BMoveIndex::build_cpu(&s, &config).unwrap();
        let bytes = bm.to_bytes().unwrap();
        let back = BMoveIndex::from_bytes(&bytes).unwrap();
        prop_assert_eq!(back.to_bytes().unwrap(), bytes);
        let cur = back.full_cursor();
        prop_assert_eq!(
            back.children_right(&cur).map(|c| c.map(|c| c.interval())),
            bm.children_right(&bm.full_cursor()).map(|c| c.map(|c| c.interval()))
        );
    }
}

fn sample_texts() -> Vec<String> {
    vec![
        "ACGTACGTACGTNNNNACGTACGT".into(),
        "ACGTACGTACGAACGTRYACGT".into(),
        "GATTACAGATTACA".into(),
    ]
}

#[test]
fn empty_pattern_locates_every_position() {
    let texts = sample_texts();
    let mv = MoveFmIndex::build_cpu(&seqs(&texts), &MoveFmIndexConfig::default()).unwrap();
    let all = sorted(mv.locate_positions(&[]));
    assert_eq!(all, (0..mv.text_len()).collect::<Vec<_>>());
    let bm = BMoveIndex::build_cpu(&seqs(&texts), &MoveFmIndexConfig::default()).unwrap();
    assert_eq!(bm.locate(&bm.full_cursor()).len() as u32, bm.text_len());
}

#[test]
fn count_only_index_counts_but_cannot_locate() {
    let s = seqs(&sample_texts());
    let config = MoveFmIndexConfig {
        locate: false,
        ..Default::default()
    };
    let mv = MoveFmIndex::build_cpu(&s, &config).unwrap();
    let fm = FmIndex::build_cpu(&s, &fm_config()).unwrap();
    assert!(!mv.can_locate());
    assert_eq!(mv.count(&encode("ACGT")), fm.count(&encode("ACGT")));
    assert!(std::panic::catch_unwind(|| mv.locate(&encode("ACGT"))).is_err());
    let full = MoveFmIndex::build_cpu(&s, &MoveFmIndexConfig::default()).unwrap();
    assert!(mv.heap_size_bytes() < full.heap_size_bytes());
}

#[test]
fn metadata_and_empty_input() {
    let s = seqs(&sample_texts());
    let fm = FmIndex::build_cpu(&s, &fm_config()).unwrap();
    let mv = MoveFmIndex::build_cpu(&s, &MoveFmIndexConfig::default()).unwrap();
    assert_eq!(mv.text_len(), fm.text_len());
    assert_eq!(mv.num_sequences(), fm.num_sequences());
    assert_eq!(mv.seq_headers(), fm.seq_headers());
    assert_eq!(mv.seq_id("seq_1"), Some(SeqId::new(1)));
    for pos in 0..=mv.text_len() {
        assert_eq!(mv.map_position(pos), fm.map_position(pos));
    }
    assert!(MoveFmIndex::build_cpu(&[], &MoveFmIndexConfig::default()).is_err());
    assert!(BMoveIndex::build_cpu(&[], &MoveFmIndexConfig::default()).is_err());
}

#[test]
fn from_bytes_rejects_malformed_input() {
    let s = seqs(&sample_texts());
    let mv = MoveFmIndex::build_cpu(&s, &MoveFmIndexConfig::default()).unwrap();
    let bytes = mv.to_bytes().unwrap();
    assert!(MoveFmIndex::from_bytes(&[]).is_err());
    let mut wrong = bytes.clone();
    wrong[3] = 3;
    assert!(MoveFmIndex::from_bytes(&wrong).is_err());
    for cut in (0..bytes.len()).step_by(5) {
        assert!(
            MoveFmIndex::from_bytes(&bytes[..cut]).is_err(),
            "truncated at {cut}"
        );
    }
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..1000 {
        let mut c = bytes.clone();
        let at = rng.random_range(4..c.len());
        c[at] ^= 1 << rng.random_range(0..8);
        if let Ok(idx) = MoveFmIndex::from_bytes(&c) {
            // Whatever loads must still answer queries without panicking.
            let _ = idx.count(&encode("ACGT"));
            if idx.can_locate() {
                let _ = idx.locate(&encode("ACGT"));
                let _ = idx.locate(&encode("A"));
            }
        }
    }

    let bm = BMoveIndex::build_cpu(&s, &MoveFmIndexConfig::default()).unwrap();
    let bytes = bm.to_bytes().unwrap();
    assert!(BMoveIndex::from_bytes(&bytes[..6]).is_err());
    assert!(BMoveIndex::from_bytes(&mv.to_bytes().unwrap()).is_err());
    for cut in (8..bytes.len()).step_by(11) {
        assert!(
            BMoveIndex::from_bytes(&bytes[..cut]).is_err(),
            "truncated at {cut}"
        );
    }
}

#[test]
fn sentinel_extension_counts_but_is_not_locatable() {
    let s = seqs(&sample_texts());
    let fm = BidirFmIndex::build_cpu(&s, &fm_config()).unwrap();
    let bm = BMoveIndex::build_cpu(&s, &MoveFmIndexConfig::default()).unwrap();
    let a = encode("A")[0];
    let cur = bm.extend_left(&bm.full_cursor(), a).unwrap();
    let dollar = bm.extend_left(&cur, 0).unwrap();
    assert_eq!(Some(dollar.interval()), fm.extend_left(cur.interval(), 0));
    assert_eq!(dollar.toehold(), None);
    assert!(std::panic::catch_unwind(|| bm.locate(&dollar)).is_err());
    // Further extensions keep correct intervals and stay unlocatable.
    let c = encode("C")[0];
    let next = bm.extend_right(&dollar, c);
    assert_eq!(
        next.map(|n| n.interval()),
        fm.extend_right(dollar.interval(), c)
    );
    if let Some(next) = next {
        assert_eq!(next.toehold(), None);
    }

    let count_only = BMoveIndex::build_cpu(
        &s,
        &MoveFmIndexConfig {
            locate: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(count_only.full_cursor().toehold(), None);
}

#[test]
fn bmove_fuzzed_blobs_never_panic() {
    let bm = BMoveIndex::build_cpu(&seqs(&sample_texts()), &MoveFmIndexConfig::default()).unwrap();
    let bytes = bm.to_bytes().unwrap();
    let mut rng = StdRng::seed_from_u64(6);
    for _ in 0..1000 {
        let mut c = bytes.clone();
        let at = rng.random_range(8..c.len());
        c[at] ^= 1 << rng.random_range(0..8);
        if let Ok(idx) = BMoveIndex::from_bytes(&c) {
            let mut cur = idx.full_cursor();
            for (i, &code) in encode("ACGTA").iter().enumerate() {
                let next = if i % 2 == 0 {
                    idx.extend_right(&cur, code)
                } else {
                    idx.extend_left(&cur, code)
                };
                match next {
                    Some(next) => cur = next,
                    None => break,
                }
            }
            if cur.toehold().is_some() {
                let _ = idx.locate(&cur);
            }
        }
    }
}
