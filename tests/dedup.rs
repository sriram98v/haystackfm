//! Deduplication of identical input sequences: every index stores one representative of
//! each class of byte-identical sequences, and reports a match in it once per class member.
//!
//! The oracle is brute force over the **original** input, duplicates included: every
//! count, every flat locate and every SMEM/MEM must equal what an index over all copies
//! would report, and the grouped variants must report exactly the hits on representatives.

use haystackfm::alphabet::encode_char;
use haystackfm::{
    BMoveIndex, BidirFmIndex, BidirRlFmIndex, DnaSequence, FmIndex, FmIndexConfig, FmIndexError,
    Mem, MoveFmIndex, MoveFmIndexConfig, RlFmIndex, RlFmIndexConfig, SeqId,
};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeMap, BTreeSet};

/// `(header, bases)`; `None` headers get the default `seq_{i}`. Classes: {0, 2, 6},
/// {1, 9} (with an `N`), {3} (a prefix of 0, not grouped with it), {4, 5} (with `#`),
/// {7, 8}.
const CORPUS: &[(Option<&str>, &str)] = &[
    (Some("a"), "ACGTGATTACA"),
    (Some("b"), "TTGCAGATTACANCG"),
    (Some("a2"), "ACGTGATTACA"),
    (Some("c"), "ACGTGATTAC"),
    (Some("d"), "ACGT#GATTACA"),
    (Some("d2"), "ACGT#GATTACA"),
    (Some("a3"), "ACGTGATTACA"),
    (None, "GGGTTTACG"),
    (None, "GGGTTTACG"),
    (Some("b2"), "TTGCAGATTACANCG"),
];

const CLASSES: &[&[u32]] = &[&[0, 2, 6], &[1, 9], &[3], &[4, 5], &[7, 8]];

const PATTERNS: &[&str] = &[
    "GATTACA",
    "GATTA",
    "TACA",
    "A",
    "C",
    "G",
    "T",
    "ACGT",
    "ACGTG",
    "GTGA",
    "CAG",
    "TTT",
    "GGGTTTACG",
    "TTAC",
    "ACGTGATTACA",
    "ACGTGATTAC",
    "CGTA",
    "AAAA",
    "NCG",
    "ACG",
    "CA",
];

fn texts() -> Vec<&'static str> {
    CORPUS.iter().map(|(_, s)| *s).collect()
}

fn corpus() -> Vec<DnaSequence> {
    CORPUS
        .iter()
        .map(|(h, s)| match h {
            Some(h) => DnaSequence::from_str_with_header(s, h).unwrap(),
            None => DnaSequence::from_str(s).unwrap(),
        })
        .collect()
}

fn encode(s: &str) -> Vec<u8> {
    s.chars().map(|c| encode_char(c).unwrap()).collect()
}

fn rep_of(id: u32) -> u32 {
    CLASSES
        .iter()
        .find(|c| c.contains(&id))
        .map(|c| c[0])
        .unwrap()
}

fn is_rep(id: SeqId) -> bool {
    rep_of(id.get()) == id.get()
}

/// Does reference byte `r` match query symbol `q`? An `N` on either side matches any base;
/// `#` (on either side) matches nothing. `exact`: only the identical symbol (single-symbol
/// cursor extension).
fn compat(q: u8, r: u8, exact: bool) -> bool {
    q != b'#' && r != b'#' && (r == q || (!exact && (r == b'N' || q == b'N')))
}

fn naive_with(refs: &[&str], pattern: &str, exact: bool) -> BTreeSet<(SeqId, u32)> {
    let p = pattern.as_bytes();
    let mut out = BTreeSet::new();
    for (id, text) in refs.iter().enumerate() {
        let t = text.as_bytes();
        if p.len() > t.len() {
            continue;
        }
        for start in 0..=t.len() - p.len() {
            if p.iter()
                .zip(&t[start..])
                .all(|(&q, &r)| compat(q, r, exact))
            {
                out.insert((SeqId::new(id as u32), start as u32));
            }
        }
    }
    out
}

fn naive(pattern: &str) -> BTreeSet<(SeqId, u32)> {
    naive_with(&texts(), pattern, false)
}

fn naive_exact(pattern: &str) -> BTreeSet<(SeqId, u32)> {
    naive_with(&texts(), pattern, true)
}

fn reps_only(hits: &BTreeSet<(SeqId, u32)>) -> BTreeSet<(SeqId, u32)> {
    hits.iter().copied().filter(|&(id, _)| is_rep(id)).collect()
}

fn set(hits: Vec<(SeqId, u32)>) -> BTreeSet<(SeqId, u32)> {
    let n = hits.len();
    let s: BTreeSet<_> = hits.into_iter().collect();
    assert_eq!(s.len(), n, "duplicate hits");
    s
}

fn original_len() -> u32 {
    CORPUS.iter().map(|(_, s)| s.len() as u32 + 1).sum()
}

fn distinct_len() -> u32 {
    CLASSES
        .iter()
        .map(|c| CORPUS[c[0] as usize].1.len() as u32 + 1)
        .sum()
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

/// Metadata every index type exposes identically.
macro_rules! assert_classes {
    ($idx:expr) => {{
        let idx = &$idx;
        assert_eq!(idx.num_sequences(), CORPUS.len() as u32);
        assert_eq!(idx.num_distinct_sequences(), CLASSES.len() as u32);
        assert_eq!(idx.text_len(), distinct_len());
        assert_eq!(idx.seq_classes().num_classes(), CLASSES.len() as u32);
        assert!(!idx.seq_classes().is_identity());
        let headers: Vec<String> = CORPUS
            .iter()
            .enumerate()
            .map(|(i, (h, _))| h.map_or_else(|| format!("seq_{i}"), str::to_string))
            .collect();
        assert_eq!(idx.seq_headers(), headers.as_slice());
        for (i, h) in headers.iter().enumerate() {
            assert_eq!(idx.seq_id(h), Some(SeqId::new(i as u32)), "{h}");
        }
        for class in CLASSES {
            let members: Vec<SeqId> = class.iter().map(|&i| SeqId::new(i)).collect();
            for &m in &members {
                assert_eq!(idx.equivalence_class(m), members.as_slice());
                assert_eq!(idx.representative(m), Some(members[0]));
            }
        }
        let past = SeqId::new(CORPUS.len() as u32);
        assert!(idx.equivalence_class(past).is_empty());
        assert_eq!(idx.representative(past), None);
    }};
}

#[test]
fn every_index_reports_the_classes() {
    let fm = FmIndex::build_cpu(&corpus(), &fm_config(4, 0)).unwrap();
    assert_classes!(fm);
    assert_eq!(fm.seq_boundaries().len(), CLASSES.len());
    assert_classes!(BidirFmIndex::build_cpu(&corpus(), &fm_config(4, 0)).unwrap());
    assert_classes!(RlFmIndex::build_cpu(&corpus(), &rl_config(4, 0)).unwrap());
    assert_classes!(BidirRlFmIndex::build_cpu(&corpus(), &rl_config(4, 0)).unwrap());
    let config = MoveFmIndexConfig::default();
    assert_classes!(MoveFmIndex::build_cpu(&corpus(), &config).unwrap());
    assert_classes!(BMoveIndex::build_cpu(&corpus(), &config).unwrap());
}

#[test]
fn sequence_returns_every_members_bases() {
    let fm = FmIndex::build_cpu(&corpus(), &fm_config(4, 0)).unwrap();
    let bidir = BidirFmIndex::build_cpu(&corpus(), &fm_config(4, 0)).unwrap();
    let rl = RlFmIndex::build_cpu(&corpus(), &rl_config(4, 0)).unwrap();
    let bidir_rl = BidirRlFmIndex::build_cpu(&corpus(), &rl_config(4, 0)).unwrap();
    for (i, seq) in corpus().iter().enumerate() {
        let id = SeqId::new(i as u32);
        assert_eq!(fm.sequence(id), Some(seq.as_slice()), "fm {i}");
        assert_eq!(bidir.sequence(id), Some(seq.as_slice()), "bidir {i}");
        assert_eq!(rl.sequence(id).as_deref(), Some(seq.as_slice()), "rl {i}");
        assert_eq!(
            bidir_rl.sequence(id).as_deref(),
            Some(seq.as_slice()),
            "rl bidir {i}"
        );
    }
    let past = SeqId::new(CORPUS.len() as u32);
    assert_eq!(fm.sequence(past), None);
    assert_eq!(rl.sequence(past), None);
}

#[test]
fn map_position_names_representatives() {
    let fm = FmIndex::build_cpu(&corpus(), &fm_config(1, 0)).unwrap();
    let mut start = 0;
    for (slot, &end) in fm.seq_boundaries().iter().enumerate() {
        let rep = SeqId::new(CLASSES[slot][0]);
        for pos in start..end {
            assert_eq!(fm.map_position(pos), Some((rep, pos - start)));
        }
        start = end;
    }
    assert_eq!(fm.map_position(fm.text_len()), None);
}

#[test]
fn fm_count_and_locate_cover_every_member() {
    for rate in [1, 2, 3, 32] {
        for depth in [0, 3] {
            let idx = FmIndex::build_cpu(&corpus(), &fm_config(rate, depth)).unwrap();
            assert_eq!(idx.count(&[]), original_len());
            for p in PATTERNS {
                let pat = encode(p);
                let want = naive(p);
                let ctx = format!("pattern {p} rate {rate} depth {depth}");
                assert_eq!(idx.count(&pat), want.len() as u32, "{ctx}");
                assert_eq!(set(idx.locate(&pat)), want, "{ctx}");
                assert_eq!(set(idx.locate_grouped(&pat)), reps_only(&want), "{ctx}");
                let raw = idx.locate_positions(&pat);
                assert_eq!(raw.len(), reps_only(&want).len(), "{ctx}");
            }
        }
    }
}

#[test]
fn rl_count_and_locate_cover_every_member() {
    for rate in [0, 1, 3, 32] {
        for depth in [0, 3] {
            let idx = RlFmIndex::build_cpu(&corpus(), &rl_config(rate, depth)).unwrap();
            assert_eq!(idx.count(&[]), original_len());
            for p in PATTERNS {
                let pat = encode(p);
                let want = naive(p);
                let ctx = format!("pattern {p} rate {rate} depth {depth}");
                assert_eq!(idx.count(&pat), want.len() as u32, "{ctx}");
                if idx.can_locate() {
                    assert_eq!(set(idx.locate(&pat)), want, "{ctx}");
                    assert_eq!(set(idx.locate_grouped(&pat)), reps_only(&want), "{ctx}");
                }
            }
        }
    }
}

#[test]
fn move_count_and_locate_cover_every_member() {
    for locate in [true, false] {
        let config = MoveFmIndexConfig {
            locate,
            ..Default::default()
        };
        let idx = MoveFmIndex::build_cpu(&corpus(), &config).unwrap();
        assert_eq!(idx.count(&[]), original_len());
        for p in PATTERNS {
            let pat = encode(p);
            let want = naive(p);
            assert_eq!(idx.count(&pat), want.len() as u32, "pattern {p}");
            if locate {
                assert_eq!(set(idx.locate(&pat)), want, "pattern {p}");
                assert_eq!(
                    set(idx.locate_grouped(&pat)),
                    reps_only(&want),
                    "pattern {p}"
                );
            }
        }
    }
}

#[test]
fn cursors_count_and_locate_cover_every_member() {
    let bidir = BidirFmIndex::build_cpu(&corpus(), &fm_config(3, 0)).unwrap();
    let bidir_rl = BidirRlFmIndex::build_cpu(&corpus(), &rl_config(3, 0)).unwrap();
    let bmove = BMoveIndex::build_cpu(&corpus(), &MoveFmIndexConfig::default()).unwrap();
    for p in PATTERNS {
        let pat = encode(p);
        let want = naive_exact(p);
        let n = want.len() as u32;

        let mut iv = Some(bidir.full_interval());
        for &c in pat.iter().rev() {
            iv = iv.and_then(|iv| bidir.extend_left(iv, c));
        }
        let (count, hits, grouped) = iv.map_or((0, vec![], vec![]), |iv| {
            (
                bidir.count_interval(&iv),
                bidir.locate_interval(&iv),
                bidir.locate_interval_grouped(&iv),
            )
        });
        assert_eq!(count, n, "bidir {p}");
        assert_eq!(set(hits), want, "bidir {p}");
        assert_eq!(set(grouped), reps_only(&want), "bidir {p}");

        let mut iv = Some(bidir_rl.full_interval());
        for &c in pat.iter().rev() {
            iv = iv.and_then(|iv| bidir_rl.extend_left(iv, c));
        }
        let (count, hits, grouped) = iv.map_or((0, vec![], vec![]), |iv| {
            (
                bidir_rl.count_interval(&iv),
                bidir_rl.locate_interval(&iv),
                bidir_rl.locate_interval_grouped(&iv),
            )
        });
        assert_eq!(count, n, "rl bidir {p}");
        assert_eq!(set(hits), want, "rl bidir {p}");
        assert_eq!(set(grouped), reps_only(&want), "rl bidir {p}");

        let mut cur = Some(bmove.full_cursor());
        for &c in pat.iter().rev() {
            cur = cur.and_then(|cur| bmove.extend_left(&cur, c));
        }
        let (count, hits, grouped) = cur.map_or((0, vec![], vec![]), |cur| {
            (
                bmove.count_cursor(&cur),
                bmove.locate(&cur),
                bmove.locate_grouped(&cur),
            )
        });
        assert_eq!(count, n, "b-move {p}");
        assert_eq!(set(hits), want, "b-move {p}");
        assert_eq!(set(grouped), reps_only(&want), "b-move {p}");
    }
}

// ── SMEMs / MEMs ──────────────────────────────────────────────────────────────

type MemTable = BTreeMap<(usize, usize), BTreeSet<(SeqId, u32)>>;

/// Every MEM of `q` (length ≥ `min_len`) over the original references, with its maximal
/// occurrences: an occurrence is maximal when it cannot be extended left nor right by a
/// compatible reference symbol (reference ends, `#` and mismatches stop it).
fn brute_mems(refs: &[&str], q: &[u8], min_len: usize) -> MemTable {
    let n = q.len();
    let mut out: MemTable = BTreeMap::new();
    for s in 0..n {
        for e in s + min_len.max(1)..=n {
            let len = e - s;
            for (id, text) in refs.iter().enumerate() {
                let t = text.as_bytes();
                if len > t.len() {
                    continue;
                }
                for p in 0..=t.len() - len {
                    if !(0..len).all(|k| compat(q[s + k], t[p + k], false)) {
                        continue;
                    }
                    let left = s == 0 || p == 0 || !compat(q[s - 1], t[p - 1], false);
                    let right = e == n || p + len == t.len() || !compat(q[e], t[p + len], false);
                    if left && right {
                        out.entry((s, e))
                            .or_default()
                            .insert((SeqId::new(id as u32), p as u32));
                    }
                }
            }
        }
    }
    out
}

/// The MEMs not contained in another MEM's query interval. Every occurrence of an SMEM is
/// maximal, so its occurrences are exactly the MEM's.
fn brute_smems(mems: &MemTable) -> MemTable {
    mems.iter()
        .filter(|((s, e), _)| {
            !mems
                .keys()
                .any(|&(s2, e2)| (s2, e2) != (*s, *e) && s2 <= *s && *e <= e2)
        })
        .map(|(k, v)| (*k, v.clone()))
        .collect()
}

fn ascii(q: &[u8]) -> String {
    q.iter().map(|&b| b as char).collect()
}

fn check_mems(
    got: Vec<Mem>,
    want: &MemTable,
    is_rep: &dyn Fn(SeqId) -> bool,
    mode: &str,
    ctx: &str,
) {
    let mut seen = BTreeSet::new();
    for m in &got {
        let key = (m.query_start, m.query_end);
        assert!(seen.insert(key), "{ctx} {mode}: duplicate {key:?}");
        let occ = want
            .get(&key)
            .unwrap_or_else(|| panic!("{ctx} {mode}: unexpected MEM {key:?}"));
        assert_eq!(
            m.match_count,
            occ.len() as u32,
            "{ctx} {mode}: count of {key:?}"
        );
        match mode {
            "off" => assert!(m.positions.is_empty(), "{ctx}"),
            "flat" => {
                assert_eq!(m.positions.len(), occ.len(), "{ctx} flat {key:?}");
                assert_eq!(&set(m.positions.clone()), occ, "{ctx} flat {key:?}");
            }
            _ => {
                let reps: BTreeSet<_> = occ.iter().copied().filter(|&(id, _)| is_rep(id)).collect();
                assert_eq!(set(m.positions.clone()), reps, "{ctx} grouped {key:?}")
            }
        }
    }
    let want_keys: BTreeSet<_> = want.keys().copied().collect();
    assert_eq!(seen, want_keys, "{ctx} {mode}: MEM intervals");
}

/// Queries built from the references: substrings, joins of two, and point mutations
/// (including to `N`, which matches any base, and `#`, which matches nothing).
fn mem_queries() -> Vec<Vec<u8>> {
    let refs: Vec<String> = texts()
        .iter()
        .flat_map(|t| t.split('#').map(str::to_string))
        .collect();
    let mut rng = StdRng::seed_from_u64(7);
    let mut out: Vec<Vec<u8>> = vec![
        b"ACGTGATTACA".to_vec(),
        b"TTACGTGATTACAGG".to_vec(),
        b"GGGTTTACGTGATTAC".to_vec(),
        b"CAGATTACAACG".to_vec(),
        b"ACGTGATTACAACGTGATTAC".to_vec(),
        b"TTGCAGATTACANCG".to_vec(),
        b"GATTACAGCGT".to_vec(),
        b"CAGNTTACANC".to_vec(),
        b"ACGT#GATTACA".to_vec(),
        b"NNGATTACANN".to_vec(),
    ];
    for _ in 0..40 {
        let a = &refs[rng.random_range(0..refs.len())];
        let b = &refs[rng.random_range(0..refs.len())];
        let i = rng.random_range(0..a.len());
        let j = rng.random_range(0..b.len());
        let mut q: Vec<u8> = format!("{}{}", &a[i..], &b[..=j]).into_bytes();
        if rng.random_bool(0.5) {
            let k = rng.random_range(0..q.len());
            q[k] = b"ACGTN#"[rng.random_range(0..6)];
        }
        out.push(q);
    }
    out
}

/// Every SMEM/MEM entry point of a bidirectional index (off, flat, grouped) against brute
/// force over `refs`.
macro_rules! check_sweeps {
    ($idx:expr, $refs:expr, $is_rep:expr, $q:expr, $min_len:expr, $ctx:expr) => {{
        let (idx, q, min_len, ctx) = (&$idx, $q, $min_len, $ctx);
        let enc = encode(&ascii(q));
        let mems = brute_mems($refs, q, min_len);
        let smems = brute_smems(&mems);
        let is_rep: &dyn Fn(SeqId) -> bool = $is_rep;
        check_mems(
            idx.find_mems(&enc, min_len, false),
            &mems,
            is_rep,
            "off",
            ctx,
        );
        check_mems(
            idx.find_mems(&enc, min_len, true),
            &mems,
            is_rep,
            "flat",
            ctx,
        );
        check_mems(
            idx.find_mems_grouped(&enc, min_len),
            &mems,
            is_rep,
            "grouped",
            ctx,
        );
        check_mems(
            idx.find_smems(&enc, min_len, false),
            &smems,
            is_rep,
            "off",
            ctx,
        );
        check_mems(
            idx.find_smems(&enc, min_len, true),
            &smems,
            is_rep,
            "flat",
            ctx,
        );
        check_mems(
            idx.find_smems_grouped(&enc, min_len),
            &smems,
            is_rep,
            "grouped",
            ctx,
        );
    }};
}

#[test]
fn smems_and_mems_cover_every_member() {
    let bidir = BidirFmIndex::build_cpu(&corpus(), &fm_config(3, 0)).unwrap();
    let bidir_rl = BidirRlFmIndex::build_cpu(&corpus(), &rl_config(3, 0)).unwrap();
    for q in mem_queries() {
        for min_len in [1, 3, 6] {
            let ctx = format!("query {} min_len {min_len}", ascii(&q));
            check_sweeps!(bidir, &texts(), &is_rep, &q, min_len, &ctx);
            check_sweeps!(
                bidir_rl,
                &texts(),
                &is_rep,
                &q,
                min_len,
                &format!("rl {ctx}")
            );
        }
    }
}

// ── Serialization and conversion ──────────────────────────────────────────────

#[test]
fn every_index_round_trips_its_classes() {
    let probe = encode("GATTACA");
    let want = naive("GATTACA");

    let fm = FmIndex::build_cpu(&corpus(), &fm_config(3, 3)).unwrap();
    let back = FmIndex::from_bytes(&fm.to_bytes().unwrap()).unwrap();
    assert_eq!(back.seq_classes(), fm.seq_classes());
    assert_eq!(back.to_bytes().unwrap(), fm.to_bytes().unwrap());
    assert_eq!(set(back.locate(&probe)), want);

    let bidir = BidirFmIndex::build_cpu(&corpus(), &fm_config(3, 3)).unwrap();
    let back = BidirFmIndex::from_bytes(&bidir.to_bytes().unwrap()).unwrap();
    assert_eq!(back.seq_classes(), bidir.seq_classes());
    assert_eq!(back.to_bytes().unwrap(), bidir.to_bytes().unwrap());

    let rl = RlFmIndex::build_cpu(&corpus(), &rl_config(3, 3)).unwrap();
    let back = RlFmIndex::from_bytes(&rl.to_bytes().unwrap()).unwrap();
    assert_eq!(back.seq_classes(), rl.seq_classes());
    assert_eq!(set(back.locate(&probe)), want);
    assert_eq!(back.count(&probe), want.len() as u32);

    let bidir_rl = BidirRlFmIndex::build_cpu(&corpus(), &rl_config(3, 3)).unwrap();
    let back = BidirRlFmIndex::from_bytes(&bidir_rl.to_bytes().unwrap()).unwrap();
    assert_eq!(back.to_bytes().unwrap(), bidir_rl.to_bytes().unwrap());

    let config = MoveFmIndexConfig::default();
    let mv = MoveFmIndex::build_cpu(&corpus(), &config).unwrap();
    let back = MoveFmIndex::from_bytes(&mv.to_bytes().unwrap()).unwrap();
    assert_eq!(back.seq_classes(), mv.seq_classes());
    assert_eq!(set(back.locate(&probe)), want);

    let bm = BMoveIndex::build_cpu(&corpus(), &config).unwrap();
    let back = BMoveIndex::from_bytes(&bm.to_bytes().unwrap()).unwrap();
    assert_eq!(back.seq_classes(), bm.seq_classes());
    assert_eq!(back.to_bytes().unwrap(), bm.to_bytes().unwrap());
}

#[test]
fn conversions_keep_the_classes() {
    let fm = FmIndex::build_cpu(&corpus(), &fm_config(3, 0)).unwrap();
    let rl = RlFmIndex::from_fm_index(&fm);
    assert_eq!(rl.seq_classes(), fm.seq_classes());
    let bidir = BidirFmIndex::build_cpu(&corpus(), &fm_config(3, 0)).unwrap();
    let bidir_rl = BidirRlFmIndex::from_bidir_fm_index(&bidir);
    assert_eq!(bidir_rl.seq_classes(), bidir.seq_classes());
    for p in PATTERNS {
        let pat = encode(p);
        let want = naive(p);
        assert_eq!(rl.count(&pat), want.len() as u32, "{p}");
        assert_eq!(set(rl.locate(&pat)), want, "{p}");
        assert_eq!(bidir_rl.count(&pat), want.len() as u32, "{p}");
    }
    for (i, seq) in corpus().iter().enumerate() {
        let id = SeqId::new(i as u32);
        assert_eq!(rl.sequence(id).as_deref(), Some(seq.as_slice()), "{i}");
    }
}

#[test]
fn identical_sequences_still_need_distinct_headers() {
    let seqs = vec![
        DnaSequence::from_str_with_header("ACGT", "x").unwrap(),
        DnaSequence::from_str_with_header("ACGT", "x").unwrap(),
    ];
    let err = FmIndex::build_cpu(&seqs, &fm_config(4, 0)).unwrap_err();
    assert!(matches!(err, FmIndexError::DuplicateHeader(h) if h == "x"));
}

#[test]
fn distinct_input_has_identity_classes() {
    let seqs: Vec<DnaSequence> = ["ACGT", "ACG", "CGT", "ACGT#A"]
        .iter()
        .map(|s| DnaSequence::from_str(s).unwrap())
        .collect();
    let fm = FmIndex::build_cpu(&seqs, &fm_config(2, 0)).unwrap();
    assert!(fm.seq_classes().is_identity());
    assert_eq!(fm.num_distinct_sequences(), 4);
    assert_eq!(fm.seq_boundaries(), &[5, 9, 13, 20]);
}

// ── Property test ─────────────────────────────────────────────────────────────

/// Random distinct-or-not references (duplicates drawn from the same pool) over ACGT, `N`
/// and `#`-joined records, checked against brute force over the full input.
fn corpus_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec("[ACGTN]{1,8}(#[ACGTN]{1,6})?", 1..5).prop_flat_map(|pool| {
        let n = pool.len();
        prop::collection::vec(0..n, 1..9)
            .prop_map(move |picks| picks.into_iter().map(|k| pool[k].clone()).collect())
    })
}

/// Representative test for a random corpus: the first occurrence of its bases.
fn first_occurrence(refs: &[String]) -> impl Fn(SeqId) -> bool + '_ {
    move |id: SeqId| refs.iter().position(|r| *r == refs[id.index()]) == Some(id.index())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_duplicates_match_brute_force(
        refs in corpus_strategy(),
        rate in 1u32..5,
        depth in 0u32..4,
        patterns in prop::collection::vec("[ACGTN]{1,4}", 1..6),
    ) {
        let seqs: Vec<DnaSequence> =
            refs.iter().map(|s| DnaSequence::from_str(s).unwrap()).collect();
        let strs: Vec<&str> = refs.iter().map(String::as_str).collect();
        let fm = FmIndex::build_cpu(&seqs, &fm_config(rate, depth)).unwrap();
        let rl = RlFmIndex::build_cpu(&seqs, &rl_config(rate, depth)).unwrap();
        let mv = MoveFmIndex::build_cpu(&seqs, &MoveFmIndexConfig::default()).unwrap();
        let distinct: BTreeSet<&str> = strs.iter().copied().collect();
        prop_assert_eq!(fm.num_distinct_sequences() as usize, distinct.len());
        prop_assert_eq!(fm.seq_classes().is_identity(), distinct.len() == strs.len());
        let total: u32 = strs.iter().map(|s| s.len() as u32 + 1).sum();
        prop_assert_eq!(fm.count(&[]), total);
        for p in &patterns {
            let pat = encode(p);
            let want = naive_with(&strs, p, false);
            prop_assert_eq!(fm.count(&pat), want.len() as u32);
            prop_assert_eq!(rl.count(&pat), want.len() as u32);
            prop_assert_eq!(mv.count(&pat), want.len() as u32);
            prop_assert_eq!(set(fm.locate(&pat)), want.clone());
            prop_assert_eq!(set(rl.locate(&pat)), want.clone());
            prop_assert_eq!(set(mv.locate(&pat)), want);
        }
        for (i, s) in refs.iter().enumerate() {
            let id = SeqId::new(i as u32);
            let bases = encode(s);
            prop_assert_eq!(fm.sequence(id), Some(bases.as_slice()));
            prop_assert_eq!(rl.sequence(id), Some(bases));
        }
    }

    #[test]
    fn random_duplicates_sweeps_match_brute_force(
        refs in corpus_strategy(),
        query in "[ACGTN]{1,6}(#?[ACGTN]{1,6})?",
        min_len in 1usize..4,
    ) {
        let seqs: Vec<DnaSequence> =
            refs.iter().map(|s| DnaSequence::from_str(s).unwrap()).collect();
        let strs: Vec<&str> = refs.iter().map(String::as_str).collect();
        let is_rep = first_occurrence(&refs);
        let bidir = BidirFmIndex::build_cpu(&seqs, &fm_config(2, 2)).unwrap();
        let bidir_rl = BidirRlFmIndex::build_cpu(&seqs, &rl_config(2, 2)).unwrap();
        let q = query.as_bytes();
        let ctx = format!("refs {refs:?} query {query} min_len {min_len}");
        check_sweeps!(bidir, &strs, &is_rep, q, min_len, &ctx);
        check_sweeps!(bidir_rl, &strs, &is_rep, q, min_len, &format!("rl {ctx}"));
    }
}
