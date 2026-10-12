//! Run-length index parity with the uncompressed one: MEM/SMEM finding, sequence
//! extraction, the k-mer lookup table, batched locate and forward-interval steps must give
//! exactly what `FmIndex` / `BidirFmIndex` give over the same references.

use haystackfm::alphabet::encode_char;
use haystackfm::{
    BidirFmIndex, BidirRlFmIndex, DnaSequence, FmIndex, FmIndexConfig, FwdInterval, RlFmIndex,
    RlFmIndexConfig, SeqId,
};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const WILD: [char; 11] = ['N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V'];

/// Mostly ACGT with a sprinkling of ambiguity codes.
fn iupac_string(max_len: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            12 => (0usize..4).prop_map(|i| ['A', 'C', 'G', 'T'][i]),
            1 => (0usize..WILD.len()).prop_map(|i| WILD[i]),
        ],
        1..=max_len,
    )
    .prop_map(|c| c.into_iter().collect())
}

/// Near-copies of one random unit: the strain-collection shape run-length indexes target.
fn strain_collection() -> impl Strategy<Value = Vec<String>> {
    (iupac_string(40), 1usize..6, any::<u64>()).prop_map(|(unit, copies, seed)| {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..copies)
            .map(|_| {
                let mut s: Vec<char> = unit.chars().collect();
                for _ in 0..rng.random_range(0..3) {
                    let at = rng.random_range(0..s.len());
                    s[at] = ['A', 'C', 'G', 'T'][rng.random_range(0..4)];
                }
                s.into_iter().collect()
            })
            .collect()
    })
}

fn collection() -> impl Strategy<Value = Vec<String>> {
    prop_oneof![
        prop::collection::vec(iupac_string(30), 1..5),
        strain_collection(),
    ]
}

/// A query drawn from the references (with edits and a random tail) so MEMs are long and
/// numerous, or a fully random one.
fn query_for(texts: &[String], seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    let src = &texts[rng.random_range(0..texts.len())];
    let mut q: Vec<char> = src.chars().collect();
    let start = rng.random_range(0..q.len());
    q.drain(..start);
    for _ in 0..rng.random_range(0..3) {
        if !q.is_empty() {
            let at = rng.random_range(0..q.len());
            q[at] = ['A', 'C', 'G', 'T', 'N'][rng.random_range(0..5)];
        }
    }
    for _ in 0..rng.random_range(0..6) {
        q.push(['A', 'C', 'G', 'T'][rng.random_range(0..4)]);
    }
    if q.is_empty() {
        q.push('A');
    }
    encode(&q.into_iter().collect::<String>())
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

fn fm_config(rate: u32, lookup_depth: u32) -> FmIndexConfig {
    FmIndexConfig {
        sa_sample_rate: rate,
        use_gpu: false,
        lookup_depth,
        ..Default::default()
    }
}

fn rl_config(rate: u32, lookup_depth: u32) -> RlFmIndexConfig {
    RlFmIndexConfig {
        sa_sample_rate: rate,
        lookup_depth,
        ..Default::default()
    }
}

/// MEMs with positions sorted, for comparison independent of locate order.
#[cfg(feature = "gpu")]
fn normalized(mut mems: Vec<haystackfm::Mem>) -> Vec<haystackfm::Mem> {
    for m in &mut mems {
        m.positions.sort();
    }
    mems
}

/// Identical results, positions in the same order (both locate in row order).
fn assert_mem_parity(fm: &BidirFmIndex, rl: &BidirRlFmIndex, query: &[u8]) {
    for min_len in [1, 4, 12, 19] {
        for locate in [false, true] {
            assert_eq!(
                fm.find_smems(query, min_len, locate),
                rl.find_smems(query, min_len, locate),
                "smems min_len={min_len} locate={locate}"
            );
            assert_eq!(
                fm.find_mems(query, min_len, locate),
                rl.find_mems(query, min_len, locate),
                "mems min_len={min_len} locate={locate}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn mems_and_smems_match_bidir_fm_index(
        texts in collection(),
        rate in prop_oneof![Just(1u32), Just(3), Just(32)],
        seeds in prop::collection::vec(any::<u64>(), 1..4),
    ) {
        let s = seqs(&texts);
        let fm = BidirFmIndex::build_cpu(&s, &fm_config(rate, 0)).unwrap();
        let rl = BidirRlFmIndex::build_cpu(&s, &rl_config(rate, 0)).unwrap();
        let converted = BidirRlFmIndex::from_bidir_fm_index(&fm);
        for seed in seeds {
            let q = query_for(&texts, seed);
            assert_mem_parity(&fm, &rl, &q);
            assert_mem_parity(&fm, &converted, &q);
        }
    }

    #[test]
    fn sequences_extract_from_the_bwt(
        texts in collection(),
        rate in prop_oneof![Just(0u32), Just(1), Just(5), Just(32)],
    ) {
        let s = seqs(&texts);
        let fm = FmIndex::build_cpu(&s, &fm_config(rate.max(1), 0)).unwrap();
        let rl = RlFmIndex::build_cpu(&s, &rl_config(rate, 0)).unwrap();
        let converted = RlFmIndex::from_fm_index(&fm);
        let bidir = BidirRlFmIndex::build_cpu(&s, &rl_config(rate, 0)).unwrap();
        let reloaded = RlFmIndex::from_bytes(&rl.to_bytes().unwrap()).unwrap();
        for (i, text) in texts.iter().enumerate() {
            let id = SeqId::new(i as u32);
            let expect = encode(text);
            prop_assert_eq!(fm.sequence(id), Some(expect.as_slice()));
            prop_assert_eq!(rl.sequence(id), Some(expect.clone()));
            prop_assert_eq!(converted.sequence(id), Some(expect.clone()));
            prop_assert_eq!(bidir.sequence(id), Some(expect.clone()));
            prop_assert_eq!(reloaded.sequence(id), Some(expect.clone()));
            let header = rl.seq_header(id).unwrap().to_owned();
            prop_assert_eq!(rl.sequence_by_header(&header), Some(expect.clone()));
        }
        let past_end = SeqId::new(texts.len() as u32);
        prop_assert_eq!(rl.sequence(past_end), None);
        prop_assert_eq!(bidir.sequence(past_end), None);
        prop_assert_eq!(rl.sequence_by_header("no such header"), None);
    }

    #[test]
    fn lookup_table_matches_fm_index(
        texts in collection(),
        depth in 1u32..5,
        patterns in prop::collection::vec(iupac_string(8), 1..8),
    ) {
        let s = seqs(&texts);
        let fm = BidirFmIndex::build_cpu(&s, &fm_config(2, depth)).unwrap();
        let rl = BidirRlFmIndex::build_cpu(&s, &rl_config(2, depth)).unwrap();
        let converted = BidirRlFmIndex::from_bidir_fm_index(&fm);
        let reloaded = BidirRlFmIndex::from_bytes(&rl.to_bytes().unwrap()).unwrap();
        prop_assert_eq!(rl.lookup_depth(), depth);
        prop_assert_eq!(converted.lookup_depth(), depth);
        prop_assert_eq!(reloaded.lookup_depth(), depth);
        // Every core k-mer, plus a few with ambiguity codes (not tabulated → None).
        let radix = 4u32.pow(depth);
        for idx in 0..radix {
            let kmer: Vec<u8> = (0..depth).rev().map(|d| 1 + ((idx >> (2 * d)) & 3) as u8).collect();
            let expect = fm.lookup_interval(&kmer);
            prop_assert_eq!(rl.lookup_interval(&kmer), expect);
            prop_assert_eq!(converted.lookup_interval(&kmer), expect);
            prop_assert_eq!(reloaded.lookup_interval(&kmer), expect);
        }
        // Seeded backward search answers like the unseeded uncompressed index.
        let plain = FmIndex::build_cpu(&s, &fm_config(2, 0)).unwrap();
        for p in &patterns {
            let p = encode(p);
            prop_assert_eq!(rl.count(&p), plain.count(&p));
            prop_assert_eq!(reloaded.fwd().count(&p), plain.count(&p));
            let mut got = rl.fwd().locate(&p);
            got.sort();
            let mut want = plain.locate(&p);
            want.sort();
            prop_assert_eq!(got, want);
        }
    }

    #[test]
    fn batched_locate_and_forward_steps_match_fm_index(
        texts in collection(),
        patterns in prop::collection::vec(iupac_string(6), 1..6),
    ) {
        let s = seqs(&texts);
        let fm = BidirFmIndex::build_cpu(&s, &fm_config(3, 0)).unwrap();
        let rl = BidirRlFmIndex::build_cpu(&s, &rl_config(3, 0)).unwrap();
        let encoded: Vec<Vec<u8>> = patterns.iter().map(|p| encode(p)).collect();
        let refs: Vec<&[u8]> = encoded.iter().map(Vec::as_slice).collect();
        prop_assert_eq!(
            rl.fwd().locate_positions_many(&refs),
            fm.fwd().locate_positions_many(&refs)
        );
        // Forward LF steps from the root, over whole intervals and a sub-range.
        for p in &encoded {
            let mut a = Some(FwdInterval::full(fm.text_len()));
            let mut b = a;
            for &c in p.iter().rev() {
                a = a.and_then(|iv| fm.extend_left_fwd(&iv, c));
                b = b.and_then(|iv| rl.extend_left_fwd(&iv, c));
                prop_assert_eq!(a, b);
            }
            if let Some(iv) = a {
                prop_assert_eq!(fm.locate_fwd(&iv), rl.locate_fwd(&iv));
                prop_assert_eq!(fm.count_interval(&fm.full_interval()), rl.count_interval(&rl.full_interval()));
                let half = FwdInterval { hi: iv.lo + iv.size().div_ceil(2), ..iv };
                for c in 1..5u8 {
                    prop_assert_eq!(fm.extend_left_fwd(&half, c), rl.extend_left_fwd(&half, c));
                }
            }
        }
    }
}

#[test]
fn mems_on_count_only_index_without_locate() {
    let texts = vec!["ACGTACGTTTGACCA".to_string(), "ACGTACGATTGACCA".to_string()];
    let s = seqs(&texts);
    let fm = BidirFmIndex::build_cpu(&s, &fm_config(1, 0)).unwrap();
    let rl = BidirRlFmIndex::build_cpu(&s, &rl_config(0, 0)).unwrap();
    let q = encode("TTACGTACGTTTGACCAGG");
    assert_eq!(fm.find_smems(&q, 3, false), rl.find_smems(&q, 3, false));
    assert_eq!(fm.find_mems(&q, 3, false), rl.find_mems(&q, 3, false));
    assert_eq!(rl.sequence(SeqId::new(1)), Some(encode(&texts[1])));
}

#[test]
fn single_base_sequences_extract() {
    let texts: Vec<String> = ["A", "C", "GT", "N", "T"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let rl = RlFmIndex::build_cpu(&seqs(&texts), &rl_config(0, 0)).unwrap();
    for (i, t) in texts.iter().enumerate() {
        assert_eq!(rl.sequence(SeqId::new(i as u32)), Some(encode(t)));
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_build_matches_cpu_build() {
    use pollster::FutureExt as _;
    let texts: Vec<String> = ["ACGTACGTTTGACCAN", "ACGTACGATTGACCA", "GGGTTTACGTACG"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let s = seqs(&texts);
    for rate in [0, 3] {
        let config = rl_config(rate, 3);
        let cpu = BidirRlFmIndex::build_cpu(&s, &config).unwrap();
        let gpu = BidirRlFmIndex::build(&s, &config).block_on().unwrap();
        assert_eq!(gpu.fwd().can_locate(), rate > 0);
        assert_eq!(gpu.lookup_depth(), 3);
        let q = encode("TTACGTACGTTTGACCAGG");
        assert_eq!(
            normalized(cpu.find_smems(&q, 3, rate > 0)),
            normalized(gpu.find_smems(&q, 3, rate > 0))
        );
        let single = RlFmIndex::build(&s, &config).block_on().unwrap();
        for (i, t) in texts.iter().enumerate() {
            let id = SeqId::new(i as u32);
            assert_eq!(gpu.sequence(id), Some(encode(t)));
            assert_eq!(single.sequence(id), Some(encode(t)));
        }
    }
}

/// Converting the reverse half of a bidirectional index directly (its one "sequence" holds
/// every other sentinel, where LF is ambiguous) must terminate; it just cannot extract.
#[test]
fn converting_a_reverse_half_terminates_without_extraction() {
    let texts: Vec<String> = [
        "TCCAAGACGACCCAMGG",
        "TCCAAGACGACCCAMGG",
        "TCCAAGACGACCCGMGG",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let s = seqs(&texts);
    for rate in [1, 32] {
        let fm = BidirFmIndex::build_cpu(&s, &fm_config(rate, 0)).unwrap();
        let rev = RlFmIndex::from_fm_index(fm.rev());
        assert_eq!(rev.count(&encode("CCA")), fm.rev().count(&encode("CCA")));
        let fwd = RlFmIndex::from_fm_index(fm.fwd());
        assert_eq!(fwd.sequence(SeqId::new(2)), Some(encode(&texts[2])));
    }
}

/// Benchmark-shaped parity: a strain collection (near-copies with SNPs and N runs, so
/// n/r is small and many occurrences share contexts), 150 bp reads with substitutions and
/// the occasional ambiguity code, `min_len` 19, both locate modes. Exercises the run-length
/// fast paths (narrow intervals inside one run, wide ones past the scan limit) and the
/// pruned MEM sweep at the depths the small proptest cases never reach.
#[test]
fn mems_match_on_a_strain_collection_with_long_reads() {
    let mut rng = StdRng::seed_from_u64(0x57a1);
    let base: Vec<u8> = (0..6000).map(|_| b"ACGT"[rng.random_range(0..4)]).collect();
    let texts: Vec<String> = (0..12)
        .map(|_| {
            let mut s = base.clone();
            for _ in 0..30 {
                let at = rng.random_range(0..s.len());
                s[at] = b"ACGT"[rng.random_range(0..4)];
            }
            if rng.random_bool(0.5) {
                let at = rng.random_range(0..s.len() - 40);
                s[at..at + rng.random_range(1..40)].fill(b'N');
            }
            String::from_utf8(s).unwrap()
        })
        .collect();
    let refs = seqs(&texts);
    let fm = BidirFmIndex::build_cpu(&refs, &fm_config(32, 0)).unwrap();
    let rl = BidirRlFmIndex::from_bidir_fm_index(&fm);
    for read_no in 0..60 {
        let src = texts[rng.random_range(0..texts.len())].as_bytes();
        let start = rng.random_range(0..src.len() - 150);
        let mut read = src[start..start + 150].to_vec();
        for _ in 0..rng.random_range(0..4) {
            let at = rng.random_range(0..150);
            read[at] = b"ACGTACGTN"[rng.random_range(0..9)];
        }
        let query = encode(std::str::from_utf8(&read).unwrap());
        for locate in [false, true] {
            for min_len in [5, 19] {
                assert_eq!(
                    fm.find_smems(&query, min_len, locate),
                    rl.find_smems(&query, min_len, locate),
                    "smems read {read_no} min_len={min_len} locate={locate}"
                );
                assert_eq!(
                    fm.find_mems(&query, min_len, locate),
                    rl.find_mems(&query, min_len, locate),
                    "mems read {read_no} min_len={min_len} locate={locate}"
                );
            }
        }
    }
}
