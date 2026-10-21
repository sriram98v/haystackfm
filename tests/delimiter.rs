//! The `#` record delimiter: records joined with `#` index as one logical sequence.
//!
//! A `#` is stored as the sentinel code, so the text of a corpus whose records are joined
//! with `#` is byte-for-byte the text of the same records indexed as separate sequences.
//! Every query must therefore agree with the split index once the split index's
//! `(record, offset)` hits are remapped to `(logical sequence, offset in the joined string)`.
//! That parity, plus a brute-force search over the joined strings, is the oracle here.

mod common;

use common::encode_pattern;
use haystackfm::alphabet::{decode_sequence, DnaSequence, SENTINEL};
use haystackfm::{
    BMoveIndex, BidirFmIndex, BidirRlFmIndex, FmIndex, FmIndexConfig, Mem, MoveFmIndex,
    MoveFmIndexConfig, RlFmIndex, RlFmIndexConfig, SeqId,
};
use proptest::prelude::*;
use std::collections::BTreeSet;

/// Logical sequences, each a list of records. "GATTACA" recurs within and across records,
/// and several record joins would spell it (or other patterns) if the delimiter matched.
const CORPUS: &[(&str, &[&str])] = &[
    ("genome_a", &["ACGTGATTA", "CAGGATTACAT", "TTGCA"]),
    ("plasmid", &["CCGATTACATNNGATTACA"]),
    ("genome_b", &["AAGGAT", "TACA", "GATTACAGATTACA", "CG"]),
];

const PATTERNS: &[&str] = &[
    "GATTACA",
    "GATTA",
    "TACA",
    "A",
    "C",
    "G",
    "T",
    "CA",
    "AC",
    "TTGCA",
    "CGAT",
    "AT",
    "GG",
    "ATCA",
    "AGAT",
    "CAG",
    "GATTACAGATTACA",
    "TTTTTTTT",
];

fn joined_str(records: &[&str]) -> String {
    records.join("#")
}

fn joined() -> Vec<DnaSequence> {
    CORPUS
        .iter()
        .map(|(h, recs)| DnaSequence::from_str_with_header(&joined_str(recs), h).unwrap())
        .collect()
}

fn split() -> Vec<DnaSequence> {
    CORPUS
        .iter()
        .flat_map(|(_, recs)| recs.iter().map(|r| DnaSequence::from_str(r).unwrap()))
        .collect()
}

/// `(logical id, offset of the record in the joined string)` for each split record.
fn remap_table() -> Vec<(SeqId, u32)> {
    let mut table = Vec::new();
    for (id, (_, recs)) in CORPUS.iter().enumerate() {
        let mut base = 0u32;
        for r in recs.iter() {
            table.push((SeqId::from(id as u32), base));
            base += r.len() as u32 + 1;
        }
    }
    table
}

fn remap(hits: Vec<(SeqId, u32)>, table: &[(SeqId, u32)]) -> BTreeSet<(SeqId, u32)> {
    hits.into_iter()
        .map(|(s, off)| {
            let (id, base) = table[s.index()];
            (id, base + off)
        })
        .collect()
}

fn set(hits: Vec<(SeqId, u32)>) -> BTreeSet<(SeqId, u32)> {
    let n = hits.len();
    let s: BTreeSet<_> = hits.into_iter().collect();
    assert_eq!(s.len(), n, "duplicate hits");
    s
}

/// Brute-force hits of an ACGT-only pattern in the joined strings under IUPAC matching (a
/// reference `N` matches any base); `#` matches nothing.
fn naive(pattern: &str) -> BTreeSet<(SeqId, u32)> {
    naive_with(pattern, b'N')
}

/// [`naive`] where only an exact base matches (cursor extension by a single symbol).
fn naive_exact(pattern: &str) -> BTreeSet<(SeqId, u32)> {
    naive_with(pattern, 0)
}

/// `wild`: the reference byte that matches any query base (0: none).
fn naive_with(pattern: &str, wild: u8) -> BTreeSet<(SeqId, u32)> {
    let p = pattern.as_bytes();
    let mut out = BTreeSet::new();
    for (id, (_, recs)) in CORPUS.iter().enumerate() {
        let text = joined_str(recs);
        let t = text.as_bytes();
        for start in 0..=t.len().saturating_sub(p.len()) {
            let hit = p
                .iter()
                .zip(&t[start..])
                .all(|(&q, &r)| r == q || r == wild);
            if hit && start + p.len() <= t.len() {
                out.insert((SeqId::from(id as u32), start as u32));
            }
        }
    }
    out
}

fn fm_config(sa_sample_rate: u32, lookup_depth: u32) -> FmIndexConfig {
    FmIndexConfig {
        sa_sample_rate,
        use_gpu: false,
        lookup_depth,
        ..Default::default()
    }
}

fn rl_config(sa_sample_rate: u32, lookup_depth: u32) -> RlFmIndexConfig {
    RlFmIndexConfig {
        sa_sample_rate,
        build_threads: 1,
        lookup_depth,
    }
}

const RATES: &[u32] = &[1, 2, 3, 4, 7, 8, 32];

#[test]
fn fm_count_and_locate_span_records() {
    let table = remap_table();
    for &rate in RATES {
        for depth in [0, 3] {
            let j = FmIndex::build_cpu(&joined(), &fm_config(rate, depth)).unwrap();
            let s = FmIndex::build_cpu(&split(), &fm_config(rate, depth)).unwrap();
            assert_eq!(j.num_sequences(), CORPUS.len() as u32);
            for p in PATTERNS {
                let pat = encode_pattern(p);
                let ctx = format!("pattern {p} rate {rate} depth {depth}");
                assert_eq!(j.count(&pat), s.count(&pat), "{ctx}");
                assert_eq!(set(j.locate(&pat)), naive(p), "{ctx}");
                assert_eq!(set(j.locate(&pat)), remap(s.locate(&pat), &table), "{ctx}");
            }
        }
    }
}

#[test]
fn rl_count_and_locate_span_records() {
    let table = remap_table();
    for &rate in RATES {
        for depth in [0, 3] {
            let j = RlFmIndex::build_cpu(&joined(), &rl_config(rate, depth)).unwrap();
            let s = RlFmIndex::build_cpu(&split(), &rl_config(rate, depth)).unwrap();
            for p in PATTERNS {
                let pat = encode_pattern(p);
                let ctx = format!("pattern {p} rate {rate} depth {depth}");
                assert_eq!(j.count(&pat), s.count(&pat), "{ctx}");
                assert_eq!(set(j.locate(&pat)), naive(p), "{ctx}");
                assert_eq!(set(j.locate(&pat)), remap(s.locate(&pat), &table), "{ctx}");
            }
        }
    }
}

#[test]
fn move_and_bmove_locate_span_records() {
    let mv = MoveFmIndex::build_cpu(&joined(), &MoveFmIndexConfig::default()).unwrap();
    let bm = BMoveIndex::build_cpu(&joined(), &MoveFmIndexConfig::default()).unwrap();
    for p in PATTERNS {
        let pat = encode_pattern(p);
        assert_eq!(set(mv.locate(&pat)), naive(p), "move, pattern {p}");
        let mut cur = Some(bm.full_cursor());
        for &c in pat.iter().rev() {
            cur = cur.and_then(|cur| bm.extend_left(&cur, c));
        }
        let got = cur.map_or_else(Vec::new, |cur| bm.locate(&cur));
        assert_eq!(set(got), naive_exact(p), "b-move, pattern {p}");
    }
}

#[test]
fn delimiter_matches_nothing() {
    let j = joined();
    let fm = FmIndex::build_cpu(&j, &fm_config(2, 3)).unwrap();
    let rl = RlFmIndex::build_cpu(&j, &rl_config(2, 3)).unwrap();
    let mv = MoveFmIndex::build_cpu(&j, &MoveFmIndexConfig::default()).unwrap();
    // Each would occur across record joins ("GATTA#CAGG…", "…ACAT#TTGCA",
    // "AAGGAT#TACA#GATT…") if the delimiter were transparent.
    for p in ["TTACAGGA", "ACATTTGC", "GGATTACAGATT"] {
        assert!(naive(p).is_empty(), "{p} occurs inside a record");
    }
    for p in [
        "TTACAGGA",
        "ACATTTGC",
        "GGATTACAGATT",
        "TA#CA",
        "#",
        "A#",
        "#C",
        "TTA#CAG",
    ] {
        let pat = encode_pattern(p);
        assert_eq!(fm.count(&pat), 0, "fm {p}");
        assert_eq!(rl.count(&pat), 0, "rl {p}");
        assert_eq!(mv.count(&pat), 0, "move {p}");
    }
}

#[test]
fn sequence_extraction_round_trips_delimiters() {
    let j = joined();
    let fm = FmIndex::build_cpu(&j, &fm_config(4, 0)).unwrap();
    let bidir = BidirFmIndex::build_cpu(&j, &fm_config(4, 0)).unwrap();
    let rl = RlFmIndex::build_cpu(&j, &rl_config(4, 0)).unwrap();
    let rl_count_only = RlFmIndex::build_cpu(&j, &rl_config(0, 0)).unwrap();
    let rl_conv = RlFmIndex::from_fm_index(&fm);
    let bidir_rl = BidirRlFmIndex::build_cpu(&j, &rl_config(4, 0)).unwrap();
    for (id, (h, recs)) in CORPUS.iter().enumerate() {
        let id = SeqId::from(id as u32);
        let want = joined_str(recs);
        let enc = DnaSequence::from_str(&want).unwrap();
        assert_eq!(fm.sequence(id), Some(enc.as_slice()), "fm {h}");
        assert_eq!(decode_sequence(fm.sequence(id).unwrap()), want, "fm {h}");
        assert_eq!(bidir.sequence(id), Some(enc.as_slice()), "bidir {h}");
        for (name, idx) in [
            ("rl", &rl),
            ("rl count-only", &rl_count_only),
            ("rl converted", &rl_conv),
        ] {
            assert_eq!(
                idx.sequence(id).as_deref(),
                Some(enc.as_slice()),
                "{name} {h}"
            );
            assert_eq!(
                idx.sequence_by_header(h).as_deref(),
                Some(enc.as_slice()),
                "{name} {h}"
            );
        }
        assert_eq!(
            bidir_rl.sequence(id).as_deref(),
            Some(enc.as_slice()),
            "bidir rl {h}"
        );
    }
}

#[test]
fn rl_serialization_keeps_pieces() {
    let j = joined();
    let table = remap_table();
    let s = RlFmIndex::build_cpu(&split(), &rl_config(3, 2)).unwrap();
    for rate in [0, 3] {
        let rl = RlFmIndex::build_cpu(&j, &rl_config(rate, 2)).unwrap();
        let bytes = rl.to_bytes().unwrap();
        let back = RlFmIndex::from_bytes(&bytes).unwrap();
        assert_eq!(back.to_bytes().unwrap(), bytes, "rate {rate}");
        for id in 0..CORPUS.len() as u32 {
            let id = SeqId::from(id);
            assert_eq!(back.sequence(id), rl.sequence(id), "rate {rate}");
        }
        if rate > 0 {
            for p in PATTERNS {
                let pat = encode_pattern(p);
                assert_eq!(set(back.locate(&pat)), remap(s.locate(&pat), &table), "{p}");
            }
        }
    }
    let brl = BidirRlFmIndex::build_cpu(&j, &rl_config(3, 2)).unwrap();
    let bytes = brl.to_bytes().unwrap();
    let back = BidirRlFmIndex::from_bytes(&bytes).unwrap();
    assert_eq!(back.to_bytes().unwrap(), bytes);
    assert_eq!(back.sequence(SeqId::from(2)), brl.sequence(SeqId::from(2)));
}

/// Queries that straddle record joins (and one carrying `#` itself).
const QUERIES: &[&str] = &[
    "ACGTGATTACAGGATTACAT",
    "TTACATTGCA",
    "AAGGATTACAGATTACAGATTACACG",
    "GATTA#CAGGATT",
    "CCGATTACATNNGATTACATACA",
    "TACAGATTACAGGAT",
];

fn sorted(
    mut mems: Vec<Mem>,
    table: Option<&[(SeqId, u32)]>,
) -> Vec<(usize, usize, u32, Vec<(SeqId, u32)>)> {
    mems.sort_by_key(|m| (m.query_start, m.query_end));
    mems.into_iter()
        .map(|m| {
            let pos: Vec<_> = match table {
                Some(t) => remap(m.positions, t).into_iter().collect(),
                None => set(m.positions).into_iter().collect(),
            };
            (m.query_start, m.query_end, m.match_count, pos)
        })
        .collect()
}

#[test]
fn smems_and_mems_stop_at_record_joins() {
    let table = remap_table();
    let jf = BidirFmIndex::build_cpu(&joined(), &fm_config(3, 3)).unwrap();
    let sf = BidirFmIndex::build_cpu(&split(), &fm_config(3, 3)).unwrap();
    let jr = BidirRlFmIndex::build_cpu(&joined(), &rl_config(3, 3)).unwrap();
    for q in QUERIES {
        let query = encode_pattern(q);
        for min_len in [1, 4] {
            let want_smem = sorted(sf.find_smems(&query, min_len, true), Some(&table));
            let want_mem = sorted(sf.find_mems(&query, min_len, true), Some(&table));
            assert_eq!(
                sorted(jf.find_smems(&query, min_len, true), None),
                want_smem,
                "fm smems {q}"
            );
            assert_eq!(
                sorted(jf.find_mems(&query, min_len, true), None),
                want_mem,
                "fm mems {q}"
            );
            assert_eq!(
                sorted(jr.find_smems(&query, min_len, true), None),
                want_smem,
                "rl smems {q}"
            );
            assert_eq!(
                sorted(jr.find_mems(&query, min_len, true), None),
                want_mem,
                "rl mems {q}"
            );
        }
        // No MEM covers a `#` in the query, and none is longer than the longest record.
        let longest_record = CORPUS
            .iter()
            .flat_map(|(_, recs)| recs.iter().map(|r| r.len()))
            .max()
            .unwrap();
        for m in jf.find_mems(&query, 1, false) {
            assert!(
                !query[m.query_start..m.query_end].contains(&SENTINEL),
                "{q}"
            );
            assert!(m.query_end - m.query_start <= longest_record, "{q}: {m:?}");
        }
    }
}

// ── Property: random records, random grouping ────────────────────────────────

fn record() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop::sample::select(vec!['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T', 'N']),
        1..40,
    )
    .prop_map(|v| v.into_iter().collect())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        failure_persistence: None,
        ..Default::default()
    })]

    #[test]
    fn joined_locate_equals_split_remapped(
        groups in prop::collection::vec(prop::collection::vec(record(), 1..5), 1..5),
        rate in 1u32..=12,
        pattern in prop::collection::vec(
            prop::sample::select(vec!['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T', 'N', 'R', '#']),
            1..5,
        ),
    ) {
        let joined: Vec<DnaSequence> = groups
            .iter()
            .map(|g| DnaSequence::from_str(&g.join("#")).unwrap())
            .collect();
        let split: Vec<DnaSequence> = groups
            .iter()
            .flatten()
            .map(|r| DnaSequence::from_str(r).unwrap())
            .collect();
        let mut table = Vec::new();
        for (id, g) in groups.iter().enumerate() {
            let mut base = 0u32;
            for r in g {
                table.push((SeqId::from(id as u32), base));
                base += r.len() as u32 + 1;
            }
        }
        let pat = encode_pattern(&pattern.into_iter().collect::<String>());

        let jf = FmIndex::build_cpu(&joined, &fm_config(rate, 0)).unwrap();
        let sf = FmIndex::build_cpu(&split, &fm_config(rate, 0)).unwrap();
        let want = remap(sf.locate(&pat), &table);
        prop_assert_eq!(set(jf.locate(&pat)), want.clone());
        if pat.contains(&SENTINEL) {
            prop_assert!(want.is_empty());
        }

        let jr = RlFmIndex::build_cpu(&joined, &rl_config(rate, 0)).unwrap();
        prop_assert_eq!(set(jr.locate(&pat)), want);
        for (id, seq) in joined.iter().enumerate() {
            let id = SeqId::from(id as u32);
            prop_assert_eq!(jf.sequence(id), Some(seq.as_slice()));
            let got = jr.sequence(id);
            prop_assert_eq!(got.as_deref(), Some(seq.as_slice()));
        }
    }
}
