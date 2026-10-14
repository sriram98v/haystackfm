//! b-move SMEM / MEM finding: `BMoveIndex::find_smems` / `find_mems` must equal
//! `BidirFmIndex`'s over the same references — the same matches and counts, and the same
//! positions as a set (b-move locates from toeholds, not in row order).

use haystackfm::alphabet::{encode_char, ExactDna};
use haystackfm::{
    BMoveIndex, BidirFmIndex, DnaSequence, FmIndexConfig, Mem, MoveFmIndexConfig, SeqId,
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
        prop_oneof![dna_string(60), iupac_string(40), repetitive_string()],
        1..5,
    )
}

fn seqs(texts: &[impl AsRef<str>]) -> Vec<DnaSequence> {
    texts
        .iter()
        .map(|s| DnaSequence::from_str(s.as_ref()).unwrap())
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

/// Each match's positions sorted.
fn normalized(mut mems: Vec<Mem>) -> Vec<Mem> {
    for m in &mut mems {
        m.positions.sort();
    }
    mems
}

fn assert_mems_match(fm: &BidirFmIndex, bm: &BMoveIndex, query: &[u8], locate: bool) {
    for min_len in [1, 4, 12, 19] {
        assert_eq!(
            normalized(bm.find_smems(query, min_len, locate)),
            normalized(fm.find_smems(query, min_len, locate)),
            "smems, min_len {min_len}, locate {locate}, query {query:?}"
        );
        assert_eq!(
            normalized(bm.find_mems(query, min_len, locate)),
            normalized(fm.find_mems(query, min_len, locate)),
            "mems, min_len {min_len}, locate {locate}, query {query:?}"
        );
    }
}

/// A window of the references concatenated without separators (so it may span a sequence
/// boundary), with up to three substitutions, some by ambiguity codes.
fn sample_query(rng: &mut StdRng, texts: &[String]) -> Vec<u8> {
    let joined: String = texts.concat();
    let start = rng.random_range(0..joined.len());
    let len = rng.random_range(1..=(joined.len() - start).min(80));
    let mut q: Vec<char> = joined[start..start + len].chars().collect();
    for _ in 0..rng.random_range(0..4) {
        let at = rng.random_range(0..q.len());
        q[at] = ['A', 'C', 'G', 'T', 'N', 'R', 'Y'][rng.random_range(0..7)];
    }
    encode(&q.into_iter().collect::<String>())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn mems_and_smems_match_bidir_fm_index(texts in texts(), seed in any::<u64>()) {
        let s = seqs(&texts);
        let fm = BidirFmIndex::build_cpu(&s, &fm_config()).unwrap();
        let bm = BMoveIndex::build_cpu(&s, &MoveFmIndexConfig::default()).unwrap();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..6 {
            let q = sample_query(&mut rng, &texts);
            assert_mems_match(&fm, &bm, &q, true);
        }
        let random: Vec<u8> = (0..rng.random_range(1..40))
            .map(|_| rng.random_range(1..16))
            .collect();
        assert_mems_match(&fm, &bm, &random, true);
    }
}

/// Matches running into reference ends and out of reference starts: the sentinel child of
/// the MEM drop set, sentinel-preceded occurrences, the first and the last reference (whose
/// end is the text's closing sentinel), and a query equal to whole references.
#[test]
fn matches_at_reference_boundaries() {
    let texts = ["ACGTTGCA", "GGATCCAA", "ACGTTGCA", "TTGCAGGA", "TTT"];
    let s = seqs(&texts);
    for (fm, bm) in [
        (
            BidirFmIndex::build_cpu(&s, &fm_config()).unwrap(),
            BMoveIndex::build_cpu(&s, &MoveFmIndexConfig::default()).unwrap(),
        ),
        (
            BidirFmIndex::build_cpu_with::<ExactDna>(&s, &fm_config()).unwrap(),
            BMoveIndex::build_cpu_with::<ExactDna>(&s, &MoveFmIndexConfig::default()).unwrap(),
        ),
    ] {
        for q in [
            "ACGTTGCA",
            "ACGTTGCAGGATCCAA",
            "TGCAGGAT",
            "CCAAACGT",
            "TTT",
            "GGATTT",
            "TTGCAGGATTT",
            "ACGTTGCATTGCAGGATTT",
            "CANGGA",
            "ACGTNNCA",
        ] {
            assert_mems_match(&fm, &bm, &encode(q), true);
            assert_mems_match(&fm, &bm, &encode(q), false);
        }
    }
}

#[test]
fn count_only_index_finds_but_cannot_locate() {
    let texts = ["ACGTTGCA", "GGATCCAA", "TTT"];
    let s = seqs(&texts);
    let fm = BidirFmIndex::build_cpu(&s, &fm_config()).unwrap();
    let bm = BMoveIndex::build_cpu(
        &s,
        &MoveFmIndexConfig {
            locate: false,
            ..Default::default()
        },
    )
    .unwrap();
    for q in ["ACGTTGCAGG", "CCAATTT", "GATC"] {
        assert_mems_match(&fm, &bm, &encode(q), false);
    }
    let q = encode("GATC");
    assert!(std::panic::catch_unwind(|| bm.find_smems(&q, 1, true)).is_err());
    assert!(std::panic::catch_unwind(|| bm.find_mems(&q, 1, true)).is_err());
    // Even with nothing to locate.
    let absent = encode("CCCCCCCC");
    assert!(std::panic::catch_unwind(|| bm.find_mems(&absent, 8, true)).is_err());
}

#[test]
fn round_tripped_index_finds_the_same_matches() {
    let texts = ["ACGTTGCANNACGT", "GGATCCAA", "ACGTRGCA", "TTT"];
    let s = seqs(&texts);
    let fm = BidirFmIndex::build_cpu(&s, &fm_config()).unwrap();
    let bm = BMoveIndex::build_cpu(&s, &MoveFmIndexConfig::default()).unwrap();
    let back = BMoveIndex::from_bytes(&bm.to_bytes().unwrap()).unwrap();
    assert_eq!(back.heap_size_bytes(), bm.heap_size_bytes());
    for q in ["ACGTTGCAGGATCC", "ACGTAGCATTT", "GGNTCC"] {
        assert_mems_match(&fm, &back, &encode(q), true);
    }
}

#[test]
fn halves_of_different_indexes_are_rejected() {
    let a = BMoveIndex::build_cpu(&seqs(&["AAAACCCC"]), &MoveFmIndexConfig::default()).unwrap();
    let b = BMoveIndex::build_cpu(&seqs(&["ACGTACGT"]), &MoveFmIndexConfig::default()).unwrap();
    let (a, b) = (a.to_bytes().unwrap(), b.to_bytes().unwrap());
    let fwd_len = |blob: &[u8]| u32::from_le_bytes(blob[4..8].try_into().unwrap()) as usize;
    let mut spliced = a[..8 + fwd_len(&a)].to_vec();
    spliced.extend_from_slice(&b[8 + fwd_len(&b)..]);
    assert!(BMoveIndex::from_bytes(&spliced).is_err());
}

/// Near-identical strains with N runs, and long reads with substitutions and N: wide
/// cursor sets, many MEMs per read, every left-maximality case.
#[test]
fn strain_collection_with_long_reads() {
    let mut rng = StdRng::seed_from_u64(0x57a1);
    let base: Vec<u8> = (0..5000).map(|_| b"ACGT"[rng.random_range(0..4)]).collect();
    let strains: Vec<String> = (0..20)
        .map(|_| {
            let mut t = base.clone();
            for _ in 0..25 {
                let at = rng.random_range(0..t.len());
                t[at] = b"ACGT"[rng.random_range(0..4)];
            }
            if rng.random_bool(0.3) {
                let at = rng.random_range(0..t.len() - 40);
                t[at..at + rng.random_range(5..40)].fill(b'N');
            }
            String::from_utf8(t).unwrap()
        })
        .collect();
    let s = seqs(&strains);
    let fm = BidirFmIndex::build_cpu(&s, &fm_config()).unwrap();
    let bm = BMoveIndex::build_cpu(&s, &MoveFmIndexConfig::default()).unwrap();
    for _ in 0..4 {
        let src = strains[rng.random_range(0..strains.len())].as_bytes();
        let at = rng.random_range(0..src.len() - 2000);
        let mut read = src[at..at + 2000].to_vec();
        for _ in 0..12 {
            let i = rng.random_range(0..read.len());
            read[i] = b"ACGTN"[rng.random_range(0..5)];
        }
        let q = encode(std::str::from_utf8(&read).unwrap());
        for min_len in [5, 19] {
            assert_eq!(
                normalized(bm.find_mems(&q, min_len, true)),
                normalized(fm.find_mems(&q, min_len, true))
            );
            assert_eq!(
                normalized(bm.find_smems(&q, min_len, true)),
                normalized(fm.find_smems(&q, min_len, true))
            );
        }
    }
    // A read running off one strain's end.
    let tail = &strains[3][strains[3].len() - 300..];
    let q = encode(tail);
    assert_eq!(
        normalized(bm.find_mems(&q, 19, true)),
        normalized(fm.find_mems(&q, 19, true))
    );
    let hits: Vec<(SeqId, u32)> = bm
        .find_smems(&q, 19, true)
        .into_iter()
        .flat_map(|m| m.positions)
        .collect();
    assert!(hits.contains(&(SeqId::new(3), 4700)));
}
