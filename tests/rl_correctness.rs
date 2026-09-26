//! Run-length FM-index correctness: `RlFmIndex` / `BidirRlFmIndex` must agree with the
//! brute-force oracle and with `FmIndex` / `BidirFmIndex` built over the same references
//! (same suffix arrays, so identical SA intervals), and must round-trip through bytes.

use haystackfm::alphabet::{encode_char, ExactDna, ALPHABET_SIZE};
use haystackfm::{
    BidirFmIndex, BidirInterval, BidirRlFmIndex, DnaSequence, FmIndex, FmIndexConfig, OccEncoding,
    RlBwt, RlFmIndex, RlFmIndexConfig, SeqId, SymbolSet,
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

/// Mostly ACGT with a sprinkling of ambiguity codes.
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

/// A short unit repeated with rare edits: low-run BWTs, the case run-length indexes target.
fn repetitive_string() -> impl Strategy<Value = String> {
    (dna_string(12), 2usize..12, any::<u64>()).prop_map(|(unit, copies, seed)| {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut s: Vec<char> = unit.repeat(copies).chars().collect();
        if rng.random_bool(0.5) {
            let at = rng.random_range(0..s.len());
            s[at] = ['A', 'C', 'G', 'T'][rng.random_range(0..4)];
        }
        s.into_iter().collect()
    })
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

fn fm_config(rate: u32, enc: OccEncoding) -> FmIndexConfig {
    FmIndexConfig {
        sa_sample_rate: rate,
        use_gpu: false,
        occ_encoding: enc,
        ..Default::default()
    }
}

fn rl_config(rate: u32, enc: OccEncoding) -> RlFmIndexConfig {
    RlFmIndexConfig {
        sa_sample_rate: rate,
        head_encoding: enc,
        ..Default::default()
    }
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

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}

/// Every cursor primitive on both bidirectional indexes, at `iv`.
fn assert_cursor_parity(fm: &BidirFmIndex, rl: &BidirRlFmIndex, iv: BidirInterval) {
    assert_eq!(fm.children_right(&iv), rl.children_right(&iv));
    assert_eq!(fm.children_left(&iv), rl.children_left(&iv));
    assert_eq!(fm.count_wild_right(&iv), rl.count_wild_right(&iv));
    assert_eq!(fm.count_wild_left(&iv), rl.count_wild_left(&iv));
    for set in [SymbolSet::WILDCARDS, SymbolSet::below(3), SymbolSet::ALL] {
        assert_eq!(fm.count_right_in(&iv, set), rl.count_right_in(&iv, set));
        assert_eq!(fm.count_left_in(&iv, set), rl.count_left_in(&iv, set));
    }
    for q in 0..ALPHABET_SIZE as u8 {
        assert_eq!(fm.extend_right(iv, q), rl.extend_right(iv, q));
        assert_eq!(fm.extend_left(iv, q), rl.extend_left(iv, q));
        assert!(fm
            .extend_right_compatible(iv, q)
            .eq(rl.extend_right_compatible(iv, q)));
        assert!(fm
            .extend_left_compatible(iv, q)
            .eq(rl.extend_left_compatible(iv, q)));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn count_and_locate_match_brute_force(
        texts in prop::collection::vec(prop_oneof![dna_string(60), repetitive_string()], 1..4),
        pattern in dna_string(6),
        rate in prop::sample::select(vec![1u32, 4, 32]),
    ) {
        let rl = RlFmIndex::build_cpu(&seqs(&texts), &rl_config(rate, OccEncoding::Bitplane))
            .unwrap();
        let expected = naive_hits(&texts, &pattern);
        let p = encode(&pattern);
        prop_assert_eq!(rl.count(&p) as usize, expected.len());
        prop_assert_eq!(sorted(rl.locate(&p)), expected);
    }

    #[test]
    fn iupac_queries_match_fm_index(
        texts in prop::collection::vec(iupac_string(50), 1..4),
        pattern in iupac_string(5),
        onehot in any::<bool>(),
    ) {
        let enc = if onehot { OccEncoding::OneHot } else { OccEncoding::Bitplane };
        let s = seqs(&texts);
        let p = encode(&pattern);

        let fm = FmIndex::build_cpu(&s, &fm_config(4, enc)).unwrap();
        let rl = RlFmIndex::build_cpu(&s, &rl_config(4, enc)).unwrap();
        prop_assert_eq!(rl.count(&p), fm.count(&p));
        prop_assert_eq!(sorted(rl.locate(&p)), sorted(fm.locate(&p)));
        prop_assert_eq!(sorted(rl.locate_positions(&p)), sorted(fm.locate_positions(&p)));

        let fm = FmIndex::build_cpu_with::<ExactDna>(&s, &fm_config(4, enc)).unwrap();
        let rl = RlFmIndex::build_cpu_with::<ExactDna>(&s, &rl_config(4, enc)).unwrap();
        prop_assert_eq!(rl.count(&p), fm.count(&p));
        prop_assert_eq!(sorted(rl.locate(&p)), sorted(fm.locate(&p)));
    }

    #[test]
    fn bidir_cursor_matches_bidir_fm_index(
        texts in prop::collection::vec(prop_oneof![iupac_string(40), repetitive_string()], 1..4),
        steps in prop::collection::vec((any::<bool>(), 0u8..16), 0..8),
    ) {
        let s = seqs(&texts);
        let fm = BidirFmIndex::build_cpu(&s, &fm_config(4, OccEncoding::Bitplane)).unwrap();
        let rl = BidirRlFmIndex::build_cpu(&s, &rl_config(4, OccEncoding::Bitplane)).unwrap();
        let mut iv = rl.full_interval();
        prop_assert_eq!(iv, fm.full_interval());
        assert_cursor_parity(&fm, &rl, iv);
        for (right, c) in steps {
            let next = if right { rl.extend_right(iv, c) } else { rl.extend_left(iv, c) };
            let Some(next) = next else { break };
            iv = next;
            assert_cursor_parity(&fm, &rl, iv);
            prop_assert_eq!(sorted(rl.locate_interval(&iv)), sorted(fm.locate_interval(&iv)));
        }
    }

    #[test]
    fn conversion_from_fm_index_matches_direct_build(
        texts in prop::collection::vec(iupac_string(40), 1..4),
        onehot in any::<bool>(),
    ) {
        let enc = if onehot { OccEncoding::OneHot } else { OccEncoding::Bitplane };
        let s = seqs(&texts);
        let fm = FmIndex::build_cpu(&s, &fm_config(8, enc)).unwrap();
        let direct = RlFmIndex::build_cpu(&s, &rl_config(8, enc)).unwrap();
        let converted = RlFmIndex::from(&fm);
        prop_assert_eq!(converted.bwt(), direct.bwt());
        prop_assert_eq!(converted.to_bytes().unwrap(), direct.to_bytes().unwrap());

        let fm_bidir = BidirFmIndex::build_cpu(&s, &fm_config(8, enc)).unwrap();
        let rl_bidir = BidirRlFmIndex::build_cpu(&s, &rl_config(8, enc)).unwrap();
        let converted = BidirRlFmIndex::from(&fm_bidir);
        prop_assert_eq!(converted.to_bytes().unwrap(), rl_bidir.to_bytes().unwrap());
    }

    #[test]
    fn serialization_round_trips(
        texts in prop::collection::vec(prop_oneof![iupac_string(40), repetitive_string()], 1..4),
        pattern in iupac_string(4),
        rate in prop::sample::select(vec![0u32, 1, 8]),
    ) {
        let s = seqs(&texts);
        let p = encode(&pattern);
        let rl = RlFmIndex::build_cpu(&s, &rl_config(rate, OccEncoding::Bitplane)).unwrap();
        let back = RlFmIndex::from_bytes(&rl.to_bytes().unwrap()).unwrap();
        prop_assert_eq!(back.bwt(), rl.bwt());
        prop_assert_eq!(back.count(&p), rl.count(&p));
        prop_assert_eq!(back.can_locate(), rate > 0);
        if rate > 0 {
            prop_assert_eq!(sorted(back.locate(&p)), sorted(rl.locate(&p)));
        }
        prop_assert_eq!(back.to_bytes().unwrap(), rl.to_bytes().unwrap());

        let bidir = BidirRlFmIndex::build_cpu(&s, &rl_config(rate, OccEncoding::OneHot)).unwrap();
        let back = BidirRlFmIndex::from_bytes(&bidir.to_bytes().unwrap()).unwrap();
        prop_assert_eq!(back.to_bytes().unwrap(), bidir.to_bytes().unwrap());
        let iv = back.full_interval();
        prop_assert_eq!(back.children_right(&iv), bidir.children_right(&iv));
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
fn count_only_index_counts_but_cannot_locate() {
    let rl =
        RlFmIndex::build_cpu(&seqs(&sample_texts()), &rl_config(0, OccEncoding::Bitplane)).unwrap();
    let fm =
        FmIndex::build_cpu(&seqs(&sample_texts()), &fm_config(4, OccEncoding::Bitplane)).unwrap();
    assert!(!rl.can_locate());
    assert_eq!(rl.count(&encode("ACGT")), fm.count(&encode("ACGT")));
    let located = std::panic::catch_unwind(|| rl.locate(&encode("ACGT")));
    assert!(located.is_err(), "locate on a count-only index must panic");
}

#[test]
fn metadata_accessors_match_fm_index() {
    let s = seqs(&sample_texts());
    let fm = FmIndex::build_cpu(&s, &fm_config(4, OccEncoding::Bitplane)).unwrap();
    let rl = RlFmIndex::build_cpu(&s, &rl_config(4, OccEncoding::Bitplane)).unwrap();
    assert_eq!(rl.text_len(), fm.text_len());
    assert_eq!(rl.num_sequences(), fm.num_sequences());
    assert_eq!(rl.seq_headers(), fm.seq_headers());
    assert_eq!(rl.seq_id("seq_2"), Some(SeqId::new(2)));
    assert_eq!(rl.seq_header(SeqId::new(1)), Some("seq_1"));
    for pos in 0..=rl.text_len() {
        assert_eq!(rl.map_position(pos), fm.map_position(pos));
    }
    assert_eq!(rl.bwt().len(), rl.text_len());
    assert!(RlFmIndex::build_cpu(&[], &RlFmIndexConfig::default()).is_err());
    assert!(BidirRlFmIndex::build_cpu(&[], &RlFmIndexConfig::default()).is_err());
}

#[test]
fn from_bytes_rejects_malformed_input() {
    let s = seqs(&sample_texts());
    let rl = RlFmIndex::build_cpu(&s, &rl_config(4, OccEncoding::Bitplane)).unwrap();
    let bytes = rl.to_bytes().unwrap();

    assert!(RlFmIndex::from_bytes(&[]).is_err());
    let mut wrong_version = bytes.clone();
    wrong_version[3] = 2;
    assert!(RlFmIndex::from_bytes(&wrong_version).is_err());
    let fm_blob = FmIndex::build_cpu(&s, &fm_config(4, OccEncoding::Bitplane))
        .unwrap()
        .to_bytes()
        .unwrap();
    assert!(RlFmIndex::from_bytes(&fm_blob).is_err());
    for cut in (0..bytes.len()).step_by(7) {
        assert!(
            RlFmIndex::from_bytes(&bytes[..cut]).is_err(),
            "truncated at {cut}"
        );
    }
    // Random single-byte corruption must never panic (it may still load if it hits
    // e.g. a header string).
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..500 {
        let mut corrupt = bytes.clone();
        let at = rng.random_range(4..corrupt.len());
        corrupt[at] ^= 1 << rng.random_range(0..8);
        let _ = RlFmIndex::from_bytes(&corrupt);
    }

    let bidir = BidirRlFmIndex::build_cpu(&s, &rl_config(4, OccEncoding::Bitplane)).unwrap();
    let bytes = bidir.to_bytes().unwrap();
    assert!(BidirRlFmIndex::from_bytes(&bytes[..6]).is_err());
    assert!(BidirRlFmIndex::from_bytes(&rl.to_bytes().unwrap()).is_err());
    let mut bad_len = bytes.clone();
    bad_len[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(BidirRlFmIndex::from_bytes(&bad_len).is_err());
}

#[test]
fn repetitive_collection_is_smaller_than_the_occ_table() {
    let mut rng = StdRng::seed_from_u64(42);
    let base: String = (0..4000)
        .map(|_| ['A', 'C', 'G', 'T'][rng.random_range(0..4)])
        .collect();
    let texts: Vec<String> = (0..40)
        .map(|_| {
            let mut s: Vec<char> = base.chars().collect();
            for _ in 0..4 {
                let at = rng.random_range(0..s.len());
                s[at] = ['A', 'C', 'G', 'T'][rng.random_range(0..4)];
            }
            s.into_iter().collect()
        })
        .collect();
    let s = seqs(&texts);
    let rl = RlBwt::build_cpu(&s, OccEncoding::Bitplane).unwrap();
    let fm = FmIndex::build_cpu(&s, &fm_config(32, OccEncoding::Bitplane)).unwrap();
    let occ_bytes = fm.occ().heap_size_bytes();
    assert!(rl.run_ratio() > 20.0, "n/r = {:.1}", rl.run_ratio());
    assert!(
        rl.heap_size_bytes() * 3 < occ_bytes,
        "RlBwt {} bytes vs occ {} bytes (n/r = {:.1})",
        rl.heap_size_bytes(),
        occ_bytes,
        rl.run_ratio()
    );
}
