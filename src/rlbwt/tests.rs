use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::alphabet::{A, C, G, N, SENTINEL, T};

/// Strings exercising different run structures (rank works on any string, not only BWTs).
fn corpus() -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let mut out = vec![
        vec![],
        vec![SENTINEL],
        vec![A; 300],
        (0..257).map(|i| if i % 2 == 0 { A } else { C }).collect(),
        (0..200).map(|i| (i % 16) as u8).collect(),
        vec![T, T, SENTINEL, A, A, A, N, N, G, C, C, C, C],
    ];
    // Uniform over the full IUPAC alphabet (r ≈ n).
    out.push((0..1500).map(|_| rng.random_range(0..16u8)).collect());
    // Long runs over DNA + N (r ≪ n), crossing many occ blocks and buckets.
    let mut runs = Vec::new();
    while runs.len() < 5000 {
        let sym = [SENTINEL, A, C, G, T, N][rng.random_range(0..6)];
        let len = rng.random_range(1..=200);
        runs.extend(std::iter::repeat_n(sym, len));
    }
    out.push(runs);
    // Skewed: a dense region of short runs inside long ones (uneven buckets).
    let mut skew = vec![A; 4000];
    skew.extend((0..600).map(|i| [C, G, T][i % 3]));
    skew.extend(vec![T; 4000]);
    out.push(skew);
    out
}

fn build_both(chars: &[u8], enc: OccEncoding) -> (RlBwt, OccTable, Bwt) {
    let bwt = Bwt::from_unpacked(chars.to_vec());
    (RlBwt::from_bwt(&bwt, enc), build_occ_table(&bwt, enc), bwt)
}

#[test]
fn every_primitive_matches_the_occ_table() {
    let mut rng = StdRng::seed_from_u64(7);
    for chars in corpus() {
        for enc in [OccEncoding::Bitplane, OccEncoding::OneHot] {
            let (rl, occ, _) = build_both(&chars, enc);
            let n = chars.len() as u32;
            assert_eq!(rl.len(), n);
            assert_eq!(BwtRank::bwt_len(&rl), BwtRank::bwt_len(&occ));
            for i in 0..=n {
                for c in 0..ALPHABET_SIZE as u8 {
                    assert_eq!(BwtRank::rank(&rl, c, i), occ.rank(c, i), "rank({c},{i})");
                }
                assert_eq!(
                    rl.rank_all_pair(i, i),
                    occ.rank_all_pair(i, i),
                    "rank_all {i}"
                );
                if i < n {
                    assert_eq!(
                        BwtRank::symbol_at(&rl, i),
                        occ.symbol_at(i),
                        "symbol_at {i}"
                    );
                    assert_eq!(BwtRank::lf_step(&rl, i), occ.lf_step(i), "lf_step {i}");
                }
            }
            for _ in 0..300 {
                let a = rng.random_range(0..=n);
                let b = rng.random_range(0..=n);
                let (lo, hi) = (a.min(b), a.max(b));
                let c = rng.random_range(0..16u8);
                let set: SymbolSet = (0..16u8).filter(|_| rng.random_bool(0.4)).collect();
                assert_eq!(BwtRank::rank_pair(&rl, c, lo, hi), occ.rank_pair(c, lo, hi));
                assert_eq!(
                    BwtRank::rank_with_below_pair(&rl, c, lo, hi),
                    occ.rank_with_below_pair(c, lo, hi),
                    "rank_with_below_pair({c},{lo},{hi})"
                );
                assert_eq!(
                    BwtRank::rank_set_pair(&rl, set, lo, hi),
                    occ.rank_set_pair(set, lo, hi)
                );
                assert_eq!(rl.rank_all_pair(lo, hi), occ.rank_all_pair(lo, hi));
            }
        }
    }
}

#[test]
fn runs_are_maximal_and_decode_back() {
    for chars in corpus() {
        let (rl, _, bwt) = build_both(&chars, OccEncoding::Bitplane);
        let runs: Vec<_> = rl.runs().collect();
        assert_eq!(runs, bwt.runs().collect::<Vec<_>>());
        assert!(
            runs.windows(2).all(|w| w[0].0 != w[1].0),
            "adjacent runs merged"
        );
        assert_eq!(rl.num_runs() as usize, runs.len());
        assert_eq!(rl.to_bwt(), bwt);
        for (i, &ch) in chars.iter().enumerate() {
            assert_eq!(rl.get(i as u32), ch);
        }
        for c in 0..ALPHABET_SIZE as u8 {
            let count = chars.iter().filter(|&&x| x == c).count() as u32;
            assert_eq!(rl.symbol_count(c), count);
            assert_eq!(rl.present_symbols().contains(c), count > 0);
        }
    }
}

#[test]
fn from_runs_merges_adjacent_and_skips_empty_runs() {
    let split = RlBwt::from_runs(
        [
            (A, 2),
            (A, 3),
            (C, 0),
            (G, 1),
            (C, 0),
            (G, 4),
            (SENTINEL, 1),
        ],
        OccEncoding::Bitplane,
    );
    assert_eq!(
        split.runs().collect::<Vec<_>>(),
        vec![(A, 5), (G, 5), (SENTINEL, 1)]
    );
    let chars: Vec<u8> = [vec![A; 5], vec![G; 5], vec![SENTINEL]].concat();
    let (whole, _, _) = build_both(&chars, OccEncoding::Bitplane);
    assert_eq!(split, whole);
}

#[test]
fn from_occ_matches_from_bwt() {
    for chars in corpus() {
        let (rl, occ, _) = build_both(&chars, OccEncoding::OneHot);
        assert_eq!(RlBwt::from_occ(&occ, OccEncoding::OneHot), rl);
    }
}

#[test]
fn build_cpu_run_length_encodes_the_fm_index_bwt() {
    let seqs: Vec<DnaSequence> = ["ACGTACGTACGTNNNNACGT", "ACGTACGTACGA", "GATTACA"]
        .iter()
        .map(|s| DnaSequence::from_str(s).unwrap())
        .collect();
    let rl = RlBwt::build_cpu(&seqs, OccEncoding::Bitplane).unwrap();
    let fm = crate::FmIndex::build_cpu(
        &seqs,
        &crate::FmIndexConfig {
            use_gpu: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(rl, RlBwt::from_occ(&fm.occ, OccEncoding::Bitplane));
    assert!(RlBwt::build_cpu(&[], OccEncoding::Bitplane).is_err());
}

#[test]
fn repetitive_text_compresses() {
    let mut rng = StdRng::seed_from_u64(11);
    let base: Vec<u8> = (0..2000)
        .map(|_| [A, C, G, T][rng.random_range(0..4)])
        .collect();
    let mut text = Vec::new();
    for _ in 0..50 {
        let mut copy = base.clone();
        for _ in 0..2 {
            let at = rng.random_range(0..copy.len());
            copy[at] = [A, C, G, T][rng.random_range(0..4)];
        }
        text.extend(copy);
    }
    let seq = DnaSequence::from_encoded(text);
    let rl = RlBwt::build_cpu(&[seq], OccEncoding::Bitplane).unwrap();
    assert!(rl.run_ratio() > 10.0, "n/r = {}", rl.run_ratio());
    // Well under the plain occ table's ~0.6 bytes per row.
    assert!(
        (rl.heap_size_bytes() as f64) < 0.25 * rl.len() as f64,
        "{} bytes for {} rows",
        rl.heap_size_bytes(),
        rl.len()
    );
}

#[test]
fn bucket_directory_edge_cases() {
    // r == 1 (widest buckets) and r == n (narrowest).
    for chars in [
        vec![A; 1000],
        (0..1000).map(|i| (i % 2) as u8 + 1).collect(),
    ] {
        let (rl, occ, _) = build_both(&chars, OccEncoding::Bitplane);
        for i in 0..=chars.len() as u32 {
            assert_eq!(BwtRank::rank(&rl, A, i), occ.rank(A, i));
        }
    }
}

#[test]
fn validate_rejects_corruption() {
    let chars: Vec<u8> = [vec![A; 5], vec![C; 3], vec![SENTINEL], vec![A; 2]].concat();
    let (rl, _, _) = build_both(&chars, OccEncoding::Bitplane);
    let mut ok = rl.clone();
    assert!(ok.validate_and_rebuild().is_ok());
    assert_eq!(ok, rl);

    let mut bad = rl.clone();
    bad.run_starts[1] = bad.run_starts[2];
    assert!(bad.validate_and_rebuild().is_err());

    let mut bad = rl.clone();
    bad.run_starts.pop();
    assert!(bad.validate_and_rebuild().is_err());

    let mut bad = rl.clone();
    bad.len += 1;
    assert!(bad.validate_and_rebuild().is_err());

    let mut bad = rl.clone();
    bad.cum_off[2] += 1;
    assert!(bad.validate_and_rebuild().is_err());

    let mut bad = rl;
    let last = bad.cum.len() - 1;
    bad.cum[last] += 1;
    assert!(bad.validate_and_rebuild().is_err());
}

#[test]
#[should_panic(expected = "RlBwt::get")]
fn get_past_the_end_panics() {
    let (rl, _, _) = build_both(&[A, C, SENTINEL], OccEncoding::Bitplane);
    rl.get(rl.len());
}
