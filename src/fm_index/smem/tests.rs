use super::fanout::SHALLOW_MAX_DEPTH;
use super::oracle;
use super::*;
use crate::alphabet::{encode_char, DnaSequence};
use crate::fm_index::{FmIndex, FmIndexConfig};

fn encode(s: &str) -> Vec<u8> {
    s.chars().map(|c| encode_char(c).unwrap()).collect()
}

fn bidir(s: &str) -> BidirFmIndex {
    let config = FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: false,
        ..Default::default()
    };
    BidirFmIndex::build_cpu(&[DnaSequence::from_str(s).unwrap()], &config).unwrap()
}

/// Brute-force MEM finder for reference: finds all substrings of `query` that
/// occur in `reference` and are both left- and right-maximal.
fn brute_force_mems(reference: &str, query: &str, min_len: usize) -> Vec<(usize, usize)> {
    let n = query.len();
    let mut mems: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

    for start in 0..n {
        for end in start + min_len..=n {
            let sub = &query[start..end];
            if !reference.contains(sub) {
                continue;
            }
            // Check right-maximal: can't extend right.
            let right_maximal = end == n || !reference.contains(&query[start..end + 1]);
            // Check left-maximal: can't extend left.
            let left_maximal = start == 0 || !reference.contains(&query[start - 1..end]);
            if right_maximal && left_maximal {
                mems.insert((start, end));
            }
        }
    }

    let mut v: Vec<_> = mems.into_iter().collect();
    v.sort();
    v
}

/// True when reference symbol `t` is matched by query symbol `q` under IUPAC overlap.
fn compat(q: u8, t: u8) -> bool {
    crate::alphabet::compatible_symbols(q).contains(&t)
}

/// Occurrence-level brute-force MEM enumerator — the MUMmer/BWA definition.
///
/// `[s,e)` is a MEM when **some** occurrence of `query[s..e)` in some reference is
/// maximal in both directions *at that occurrence*. Independent of the index: it walks
/// every (query position, reference position) pair. `refs` holds the encoded bases of
/// each reference separately, so a reference boundary counts as maximal.
fn brute_force_mems_occ(refs: &[Vec<u8>], query: &[u8], min_len: usize) -> Vec<(usize, usize)> {
    let mut out = std::collections::BTreeSet::new();
    for t in refs {
        for i in 0..query.len() {
            for j in 0..t.len() {
                if !compat(query[i], t[j]) {
                    continue;
                }
                if i > 0 && j > 0 && compat(query[i - 1], t[j - 1]) {
                    continue; // this occurrence is not left-maximal
                }
                // For a fixed occurrence the only right-maximal length is the full
                // extension: every shorter prefix still extends by one more base.
                let mut l = 0;
                while i + l < query.len() && j + l < t.len() && compat(query[i + l], t[j + l]) {
                    l += 1;
                }
                if l >= min_len {
                    out.insert((i, i + l));
                }
            }
        }
    }
    out.into_iter().collect()
}

#[test]
fn no_smems_when_query_absent() {
    let idx = bidir("AAAA");
    let query = encode("CCCC");
    let smems = idx.find_smems(&query, 1, false);
    assert!(smems.is_empty());
}

#[test]
fn single_smem_exact_match() {
    let idx = bidir("ACGTACGT");
    let query = encode("ACGT");
    let smems = idx.find_smems(&query, 1, false);
    assert_eq!(smems.len(), 1);
    assert_eq!(smems[0].query_start, 0);
    assert_eq!(smems[0].query_end, 4);
    assert_eq!(smems[0].match_count, 2); // "ACGT" appears twice in reference
}

#[test]
fn smem_locate_returns_correct_positions() {
    let idx = bidir("ACGTACGT");
    let query = encode("ACGT");
    let smems = idx.find_smems(&query, 1, true);
    assert_eq!(smems.len(), 1);
    let mut positions = smems[0].positions.clone();
    positions.sort();
    assert_eq!(positions, vec![(SeqId::new(0), 0), (SeqId::new(0), 4)]);
}

#[test]
fn min_len_filter() {
    let idx = bidir("ACGTACGT");
    let query = encode("A");
    // "A" is length 1; with min_len=2, it should be filtered out.
    let smems = idx.find_smems(&query, 2, false);
    assert!(smems.is_empty());
}

#[test]
fn smems_match_brute_force() {
    let reference = "ACGTTAGCCAGTACGT";
    let query_str = "CGTTAGC";
    let idx = bidir(reference);
    let query = encode(query_str);

    let smems = idx.find_smems(&query, 1, false);
    let smem_pairs: Vec<(usize, usize)> =
        smems.iter().map(|m| (m.query_start, m.query_end)).collect();

    let expected = brute_force_mems(reference, query_str, 1);

    assert_eq!(
        smem_pairs, expected,
        "SMEMs differ from brute force.\nGot:      {:?}\nExpected: {:?}",
        smem_pairs, expected
    );
}

#[test]
fn find_mems_superset_of_smems() {
    let reference = "ACGTTAGCCAGTACGT";
    let query_str = "CGTTAGC";
    let idx = bidir(reference);
    let query = encode(query_str);

    let smems = idx.find_smems(&query, 1, false);
    let mems = idx.find_mems(&query, 1, false);

    // Every SMEM should appear in the MEMs list.
    for smem in &smems {
        assert!(
            mems.iter()
                .any(|m| m.query_start == smem.query_start && m.query_end == smem.query_end),
            "SMEM {:?} not found in MEMs list",
            smem
        );
    }
}

#[test]
fn smems_all_positions_valid() {
    let reference = "ACGTTAGCCAGTACGT";
    let query_str = "AGTACGT";
    let idx = bidir(reference);
    let query_encoded = encode(query_str);

    let smems = idx.find_smems(&query_encoded, 1, true);
    for mem in &smems {
        let pattern = &query_str[mem.query_start..mem.query_end];
        for (_, pos) in &mem.positions {
            let pos = *pos as usize;
            assert!(
                pos + pattern.len() <= reference.len(),
                "position {} out of bounds",
                pos
            );
            assert_eq!(
                &reference[pos..pos + pattern.len()],
                pattern,
                "wrong match at pos {}: expected '{}' got '{}'",
                pos,
                pattern,
                &reference[pos..pos + pattern.len()]
            );
        }
    }
}

#[test]
fn smem_count_matches_unidirectional_count() {
    let reference = "ACGTACGTACGT";
    let idx = bidir(reference);
    let uni_config = FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: false,
        ..Default::default()
    };
    let uni =
        FmIndex::build_cpu(&[DnaSequence::from_str(reference).unwrap()], &uni_config).unwrap();

    let query = encode("ACGT");
    let smems = idx.find_smems(&query, 1, false);
    // "ACGT" occurs 3 times — should be reflected in match_count
    assert_eq!(smems[0].match_count, uni.count(&query));
}

#[test]
fn empty_query_returns_empty() {
    let idx = bidir("ACGT");
    assert!(idx.find_smems(&[], 1, false).is_empty());
    assert!(idx.find_mems(&[], 1, false).is_empty());
}

#[test]
fn find_mems_matches_brute_force() {
    let reference = "ACGTTAGCCAGTACGT";
    let query_str = "CGTTAGC";
    let idx = bidir(reference);
    let query = encode(query_str);

    let mems = idx.find_mems(&query, 1, false);
    let mem_pairs: Vec<(usize, usize)> =
        mems.iter().map(|m| (m.query_start, m.query_end)).collect();

    let expected = brute_force_mems_occ(&[encode(reference)], &query, 1);

    assert_eq!(
        mem_pairs, expected,
        "MEMs differ from brute force.\nGot:      {:?}\nExpected: {:?}",
        mem_pairs, expected
    );
}

#[test]
fn find_mems_all_left_and_right_maximal() {
    let reference = "ACGTTAGCCAGTACGT";
    let query_str = "CGTTAGCAGT";
    let idx = bidir(reference);
    let query = encode(query_str);
    let text = encode(reference);

    let mems = idx.find_mems(&query, 1, false);
    assert!(!mems.is_empty(), "expected at least one MEM");

    for mem in &mems {
        let (s, e) = (mem.query_start, mem.query_end);
        // Occurrence-level: at least one occurrence must be maximal in both directions.
        let witnessed = (0..text.len()).any(|j| {
            if j + (e - s) > text.len() {
                return false;
            }
            let matches = (0..e - s).all(|k| compat(query[s + k], text[j + k]));
            let left_max = s == 0 || j == 0 || !compat(query[s - 1], text[j - 1]);
            let right_max = e == query.len()
                || j + (e - s) == text.len()
                || !compat(query[e], text[j + (e - s)]);
            matches && left_max && right_max
        });
        assert!(
            witnessed,
            "MEM [{s},{e}) has no occurrence that is maximal in both directions"
        );
    }
}

/// Two references sharing a query prefix, one extending further than the other. Under
/// whole-set maximality only `(0,10)` survives, because ref_a's occurrences veto every
/// shorter interval; per occurrence, four more MEMs are real.
#[test]
fn find_mems_reports_all_occurrence_level_mems() {
    let ref_a = "ACGTACGTAC"; // holds the whole query
    let ref_b = "ACGTACT"; // holds query[0..6), flanked by T != query[6] = G
    let idx = bidir_multi(&[(ref_a, "A"), (ref_b, "B")]);
    let query = encode("ACGTACGTAC");

    let got: Vec<(usize, usize)> = idx
        .find_mems(&query, 2, false)
        .iter()
        .map(|m| (m.query_start, m.query_end))
        .collect();

    assert_eq!(got, vec![(0, 2), (0, 6), (0, 10), (4, 10), (8, 10)]);
    assert_eq!(
        got,
        brute_force_mems_occ(&[encode(ref_a), encode(ref_b)], &query, 2),
        "must agree with the independent brute-force enumerator"
    );
}

/// Left-maximality is per-occurrence too: `(4,10)` and `(8,10)` are real MEMs via the
/// sequence-start occurrences, even though ref_a@4 / ref_a@8 are left-extendable.
#[test]
fn find_mems_left_maximality_is_per_occurrence() {
    let idx = bidir_multi(&[("ACGTACGTAC", "A"), ("ACGTACT", "B")]);
    let query = encode("ACGTACGTAC");

    let got: Vec<(usize, usize)> = idx
        .find_mems(&query, 2, false)
        .iter()
        .map(|m| (m.query_start, m.query_end))
        .collect();

    for want in [(4, 10), (8, 10)] {
        assert!(
            got.contains(&want),
            "{want:?} dropped: whole-set left-maximality over-rejects"
        );
    }
}

/// `find_mems` against the independent occurrence-level enumerator on randomized
/// multi-reference corpora.
///
/// The cross-check has to be against a brute-force enumerator, not against `find_smems`:
/// under whole-set maximality the two functions return near-identical sets, so comparing
/// them to each other cannot detect this class of defect.
#[test]
fn find_mems_matches_brute_force_randomized() {
    let mut state: u64 = 0x5EED_1234_ABCD_9876;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let pick = |set: &[u8], next: &mut dyn FnMut() -> u32| -> String {
        (set[(next() % set.len() as u32) as usize] as char).to_string()
    };
    let rand_seq = |n: usize, set: &[u8], next: &mut dyn FnMut() -> u32| -> String {
        (0..n).map(|_| pick(set, next)).collect()
    };
    let acgt = b"ACGT";
    let iupac = b"ACGTNRY";

    for iter in 0..60 {
        let full = rand_seq(60, acgt, &mut next);
        // Reference set: a full copy, an interior slice, and noise — plus, every other
        // iteration, IUPAC codes so extensions branch into interval sets.
        let a = full.clone();
        let b = full[10..40].to_string();
        let noise = if iter % 2 == 0 {
            rand_seq(40, iupac, &mut next)
        } else {
            rand_seq(40, acgt, &mut next)
        };
        let idx = bidir_multi(&[(&a, "A"), (&b, "B"), (&noise, "NOISE")]);

        let q = encode(&full);
        let refs = vec![encode(&a), encode(&b), encode(&noise)];
        for min_len in [1usize, 3, 8] {
            let got: Vec<(usize, usize)> = idx
                .find_mems(&q, min_len, false)
                .iter()
                .map(|m| (m.query_start, m.query_end))
                .collect();
            let expected = brute_force_mems_occ(&refs, &q, min_len);
            assert_eq!(
                got, expected,
                "iter={iter} min_len={min_len}\nquery={full}\nnoise={noise}"
            );
        }
    }
}

/// Every reported position must be an occurrence that is genuinely maximal both ways,
/// and `match_count` must agree with the number of reported positions.
#[test]
fn find_mems_positions_are_maximal_occurrences() {
    let refs = [("ACGTACGTAC", "A"), ("ACGTACT", "B"), ("TTACGTACGG", "C")];
    let idx = bidir_multi(&refs);
    let query = encode("ACGTACGTAC");

    let mems = idx.find_mems(&query, 2, true);
    assert!(!mems.is_empty());

    for mem in &mems {
        let (s, e) = (mem.query_start, mem.query_end);
        assert_eq!(
            mem.positions.len() as u32,
            mem.match_count,
            "MEM [{s},{e}) match_count disagrees with positions"
        );
        for &(id, off) in &mem.positions {
            let seq = encode(refs[id.index()].0);
            let off = off as usize;
            assert!(
                (0..e - s).all(|k| compat(query[s + k], seq[off + k])),
                "MEM [{s},{e}) position {id:?}@{off} does not spell the match"
            );
            assert!(
                s == 0 || off == 0 || !compat(query[s - 1], seq[off - 1]),
                "MEM [{s},{e}) position {id:?}@{off} is left-extendable"
            );
            assert!(
                e == query.len()
                    || off + (e - s) == seq.len()
                    || !compat(query[e], seq[off + (e - s)]),
                "MEM [{s},{e}) position {id:?}@{off} is right-extendable"
            );
        }
    }
}

#[test]
fn find_mems_min_len_filter() {
    let reference = "ACGTTAGCCAGTACGT";
    let query_str = "CGTTAGC";
    let idx = bidir(reference);
    let query = encode(query_str);

    let min_len = 3;
    let mems = idx.find_mems(&query, min_len, false);

    for mem in &mems {
        assert!(
            mem.len() >= min_len,
            "MEM [{},{}] has length {} < min_len {}",
            mem.query_start,
            mem.query_end,
            mem.len(),
            min_len
        );
    }

    // Verify that mems with min_len=1 contains strictly more entries (or equal)
    let mems_all = idx.find_mems(&query, 1, false);
    assert!(
        mems_all.len() >= mems.len(),
        "min_len=1 should return at least as many MEMs as min_len={min_len}"
    );
}

// ── N-wildcard tests ──────────────────────────────────────────────────────

#[test]
fn n_in_query_matches_any_nucleotide_smem() {
    // Reference has ACGT; query "N" should match all 4 positions.
    let idx = bidir("ACGT");
    let query = encode("N");
    let smems = idx.find_smems(&query, 1, false);
    assert_eq!(smems.len(), 1);
    assert_eq!(smems[0].query_start, 0);
    assert_eq!(smems[0].query_end, 1);
    // N matches A, C, G, T → 4 occurrences total
    assert_eq!(smems[0].match_count, 4);
}

#[test]
fn n_in_query_flanked_by_exact_bases() {
    // Reference "AACAAGAAT"; query "ANT" should match "AAT" (A-N-T where N=A).
    let idx = bidir("AACAAGAAT");
    let query = encode("ANT");
    let smems = idx.find_smems(&query, 1, false);
    // "ANT" with N=A matches "AAT" in the reference (at position 6)
    assert_eq!(smems.len(), 1);
    assert_eq!(smems[0].query_start, 0);
    assert_eq!(smems[0].query_end, 3);
    assert!(smems[0].match_count >= 1);
}

#[test]
fn n_wildcard_locate_returns_all_matching_positions() {
    // Reference "ACGT"; query "N" matches all 4 positions.
    let idx = bidir("ACGT");
    let query = encode("N");
    let smems = idx.find_smems(&query, 1, true);
    assert_eq!(smems.len(), 1);
    let mut positions = smems[0].positions.clone();
    positions.sort();
    assert_eq!(positions.len(), 4);
    // All offsets 0..3 must appear
    let offsets: Vec<u32> = positions.iter().map(|(_, off)| *off).collect();
    for expected in 0u32..4 {
        assert!(offsets.contains(&expected), "missing offset {expected}");
    }
}

#[test]
fn n_wildcard_find_mems_superset() {
    // Every MEM found with an exact query must also appear when the
    // corresponding base is replaced with N (because N ⊇ exact base).
    let idx = bidir("ACGTACGT");
    let exact_query = encode("ACGT");
    let n_query = encode("NCGN"); // N at pos 0 and 3
    let exact_mems = idx.find_mems(&exact_query, 1, false);
    let n_mems = idx.find_mems(&n_query, 1, false);
    // N-query must find at least as many match positions as exact query.
    let exact_count: u32 = exact_mems.iter().map(|m| m.match_count).sum();
    let n_count: u32 = n_mems.iter().map(|m| m.match_count).sum();
    assert!(
        n_count >= exact_count,
        "N-wildcard count {n_count} < exact count {exact_count}"
    );
}

#[test]
fn n_only_query_matches_all_positions() {
    // "NN" in query matches any 2-mer in the reference.
    let idx = bidir("ACGT");
    let query = encode("NN");
    let smems = idx.find_smems(&query, 1, false);
    // Should find exactly one SMEM of length 2 covering all 3 dinucleotides
    assert_eq!(smems.len(), 1);
    assert_eq!(smems[0].query_end - smems[0].query_start, 2);
    assert_eq!(smems[0].match_count, 3); // AC, CG, GT
}

#[test]
fn n_in_reference_is_bidirectional_wildcard() {
    // N in the reference matches any query base (bidirectional wildcard).
    // Reference "ANCGT", query "AC":
    //   "AN" at ref pos 0: query A=ref A (exact), query C=ref N (wildcard) → match
    //   "NC" at ref pos 1: query A=ref N (wildcard), query C=ref C (exact) → match
    let idx = bidir("ANCGT");
    let query = encode("AC");
    let mems = idx.find_mems(&query, 2, false);
    assert_eq!(mems.len(), 1, "should find one length-2 MEM");
    assert_eq!(mems[0].query_start, 0);
    assert_eq!(mems[0].query_end, 2);
    assert_eq!(
        mems[0].match_count, 2,
        "matches 'AN' at ref pos 0 and 'NC' at ref pos 1"
    );
}

#[test]
fn n_in_query_left_maximal() {
    // "NA" in AAAA: N matches A, so "NA" == "AA"; left-maximal only at pos 0
    // since all positions can extend left except the first.
    let idx = bidir("AAAA");
    let query = encode("NA");
    let smems = idx.find_smems(&query, 2, false);
    assert_eq!(smems.len(), 1);
    assert_eq!(smems[0].query_start, 0);
    assert_eq!(smems[0].query_end, 2);
}

// ── Regression: SMEM-drop bug (bug-fmidx.md) ──────────────────────────────

fn bidir_multi(seqs: &[(&str, &str)]) -> BidirFmIndex {
    let config = FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: false,
        ..Default::default()
    };
    let dna: Vec<DnaSequence> = seqs
        .iter()
        .map(|(s, h)| DnaSequence::from_str_with_header(s, h).unwrap())
        .collect();
    BidirFmIndex::build_cpu(&dna, &config).unwrap()
}

/// Verbatim reproducer from `bug-fmidx.md`: two overlapping MEMs where neither query
/// interval contains the other (`[4,30)` and `[5,176)`) are both SMEMs. The old
/// pivot-advance logic emitted only the left-starting one and dropped the longer seed.
#[test]
fn smem_drops_valid_longer_seed() {
    let query = "CGTTCTGGAAGCAATGGCTTTCCTTGAGGAATCCCACCCAGGGATCTTTGAAAACTCTTGT\
                 CTTGAAACGATGGAAGTTGTTCAGCAAACAAGAGTGGACAAACTAACTCAAGGTCGCCAGA\
                 CTTATGACTGGACATTGAATAGAAACCAACCAGCTGCAACTGCTTTGGCCAACA";
    let ref_wrong = &query[4..30]; // 26 bp  -> query[4..30)
    let ref_correct = &query[5..176]; // 171 bp -> query[5..176)

    let idx = bidir_multi(&[(ref_wrong, "REF_WRONG"), (ref_correct, "REF_CORRECT")]);
    let q = encode(query);

    let smems = idx.find_smems(&q, 19, true);
    let mems = idx.find_mems(&q, 19, true);

    let hits = |ms: &[Mem], header: &str| {
        ms.iter().any(|m| {
            m.positions
                .iter()
                .any(|&(id, _)| idx.seq_header(id) == Some(header))
        })
    };

    // find_mems already finds REF_CORRECT (sanity: index content is correct).
    assert!(hits(&mems, "REF_CORRECT"));
    // find_smems must now also return the 171 bp SMEM to REF_CORRECT.
    assert!(
        hits(&smems, "REF_CORRECT"),
        "find_smems dropped the valid 171 bp SMEM to REF_CORRECT"
    );
    assert!(hits(&smems, "REF_WRONG"), "find_smems dropped REF_WRONG");
}

/// Left/right mirror of the reproducer: the longer seed starts *earlier* and the shorter
/// one ends later. Both remain SMEMs.
#[test]
fn smem_keeps_both_when_shorter_starts_later() {
    let query = "CGTTCTGGAAGCAATGGCTTTCCTTGAGGAATCCCACCCAGGGATCTTTGAAAACTCTTGT\
                 CTTGAAACGATGGAAGTTGTTCAGCAAACAAGAGTGGACAAACTAACTCAAGGTCGCCAGA\
                 CTTATGACTGGACATTGAATAGAAACCAACCAGCTGCAACTGCTTTGGCCAACA";
    // Longer seed [0..171); shorter competing seed [150..176) — neither contains the other.
    let ref_long = &query[0..171];
    let ref_short = &query[150..176];

    let idx = bidir_multi(&[(ref_long, "REF_LONG"), (ref_short, "REF_SHORT")]);
    let q = encode(query);
    let smems = idx.find_smems(&q, 19, true);

    let hits = |header: &str| {
        smems.iter().any(|m| {
            m.positions
                .iter()
                .any(|&(id, _)| idx.seq_header(id) == Some(header))
        })
    };
    assert!(hits("REF_LONG"), "dropped the leading long SMEM");
    assert!(hits("REF_SHORT"), "dropped the trailing short SMEM");
}

/// `find_smems` must equal the containment-maximal filter of `find_mems` on randomized
/// queries with planted overlapping seeds to two references. This is the oracle that
/// keeps the (future) single-pass SMEM algorithm honest.
#[test]
fn smems_equal_containment_filtered_mems_randomized() {
    // Tiny deterministic LCG for reproducibility without extra deps.
    let mut state: u64 = 0xDEADBEEFCAFEF00D;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let bases = [b'A', b'C', b'G', b'T'];
    let rand_dna = |n: usize, next: &mut dyn FnMut() -> u32| -> String {
        (0..n)
            .map(|_| bases[(next() % 4) as usize] as char)
            .collect()
    };

    // A few references may carry ambiguity codes so extensions branch into interval
    // sets (the case that makes our SMEM enumeration harder than textbook BWA).
    let iupac = [b'A', b'C', b'G', b'T', b'N', b'R', b'Y'];
    let rand_iupac = |n: usize, next: &mut dyn FnMut() -> u32| -> String {
        (0..n)
            .map(|_| iupac[(next() % iupac.len() as u32) as usize] as char)
            .collect()
    };

    for iter in 0..80 {
        let full = rand_dna(200, &mut next);
        // Two overlapping references carved from the query, plus random flank noise.
        // Every other iteration injects IUPAC/N noise to exercise branching.
        let a = &full[10..60];
        let b = &full[40..160];
        let noise = if iter % 2 == 0 {
            rand_iupac(80, &mut next)
        } else {
            rand_dna(80, &mut next)
        };
        let idx = bidir_multi(&[(a, "A"), (b, "B"), (&noise, "NOISE")]);
        let q = encode(&full);
        let min_len = 15;

        let smems = idx.find_smems(&q, min_len, false);
        let mems = idx.find_mems(&q, min_len, false);

        // Oracle: containment-maximal filter over the MEM intervals.
        let ivs: Vec<(usize, usize)> = mems.iter().map(|m| (m.query_start, m.query_end)).collect();
        let mut expected: Vec<(usize, usize)> = ivs
            .iter()
            .filter(|&&(s, e)| {
                !ivs.iter()
                    .any(|&(s2, e2)| (s2, e2) != (s, e) && s2 <= s && e <= e2)
            })
            .copied()
            .collect();
        expected.sort();

        let mut got: Vec<(usize, usize)> =
            smems.iter().map(|m| (m.query_start, m.query_end)).collect();
        got.sort();

        assert_eq!(
            got, expected,
            "SMEMs != containment-maximal MEMs\nquery={full}"
        );
    }
}

/// The pruned/lockstep sweeps against the original algorithms they replace, compared
/// as raw candidates (bounds, counts, intervals in order, left context) before any
/// filtering or locating, on every backend (the uncompressed one with its shallow cache,
/// the run-length one with its rank scan, b-move with its move tables; the oracles also run
/// on b-move's own hooks). Strain-like texts with N runs and reads with
/// substitutions and ambiguity codes, so the early stop, the equal-coverage pruning and
/// the drop arithmetic all fire.
#[test]
fn rewritten_sweeps_match_the_original_candidates() {
    use crate::move_index::{BMoveIndex, MoveFmIndexConfig};
    use crate::rl_fm_index::BidirRlFmIndex;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    type Key = (usize, usize, u32, Vec<BidirInterval>, Option<u8>);
    fn keys<C: SweepCursor>(raws: Vec<RawMem<C>>) -> Vec<Key> {
        raws.into_iter()
            .map(|r| {
                let ivs = r.ivs.iter().map(C::interval).collect();
                (r.query_start, r.query_end, r.match_count, ivs, r.left_ctx)
            })
            .collect()
    }

    let mut rng = StdRng::seed_from_u64(0x0a11);
    for round in 0..6 {
        let base: Vec<u8> = (0..800).map(|_| b"ACGT"[rng.random_range(0..4)]).collect();
        let texts: Vec<String> = (0..1 + round)
            .map(|_| {
                let mut t = base.clone();
                for _ in 0..8 {
                    let at = rng.random_range(0..t.len());
                    t[at] = b"ACGTN"[rng.random_range(0..5)];
                }
                String::from_utf8(t).unwrap()
            })
            .collect();
        let seqs: Vec<DnaSequence> = texts
            .iter()
            .map(|t| DnaSequence::from_str(t).unwrap())
            .collect();
        let config = FmIndexConfig {
            sa_sample_rate: 4,
            use_gpu: false,
            ..Default::default()
        };
        let fm = BidirFmIndex::build_cpu(&seqs, &config).unwrap();
        let rl = BidirRlFmIndex::from_bidir_fm_index(&fm);
        let bm = BMoveIndex::build_cpu(&seqs, &MoveFmIndexConfig::default()).unwrap();
        for _ in 0..25 {
            let src = texts[rng.random_range(0..texts.len())].as_bytes();
            let start = rng.random_range(0..src.len() - 120);
            let mut read = src[start..start + 120].to_vec();
            for _ in 0..rng.random_range(0..4) {
                let at = rng.random_range(0..read.len());
                read[at] = b"ACGTACGTNR"[rng.random_range(0..10)];
            }
            let q = encode(std::str::from_utf8(&read).unwrap());
            for min_len in [1, 6, 19] {
                let smems = keys(oracle::collect_smem_candidates_reference(&fm, &q, min_len));
                let mems = keys(oracle::collect_mem_candidates_reference(&fm, &q, min_len));
                assert_eq!(keys(fm.collect_smem_candidates(&q, min_len)), smems);
                assert_eq!(keys(rl.collect_smem_candidates(&q, min_len)), smems);
                assert_eq!(keys(fm.collect_mem_candidates(&q, min_len)), mems);
                assert_eq!(keys(rl.collect_mem_candidates(&q, min_len)), mems);
                assert_eq!(keys(bm.collect_smem_candidates(&q, min_len)), smems);
                assert_eq!(keys(bm.collect_mem_candidates(&q, min_len)), mems);
                let bm_smems = oracle::collect_smem_candidates_reference(&bm, &q, min_len);
                let bm_mems = oracle::collect_mem_candidates_reference(&bm, &q, min_len);
                assert_eq!(keys(bm_smems), smems);
                assert_eq!(keys(bm_mems), mems);
            }
        }
    }
}

/// A reference dense in ambiguity codes multiplies every k-mer's wildcard variants:
/// the shallow cache must stop at its budget (shallower, never larger), and the sweeps
/// must still match the uncompressed index exactly (b-move: positions as a set).
#[test]
fn shallow_cache_respects_its_budget_on_iupac_dense_text() {
    use crate::move_index::{BMoveCursor, BMoveIndex, MoveFmIndexConfig};
    use crate::rl_fm_index::BidirRlFmIndex;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(0xb0d9);
    let text: String = (0..70_000)
        .map(|_| ['A', 'C', 'G', 'T', 'N', 'B', 'D', 'H', 'V'][rng.random_range(0..9)])
        .collect();
    let seqs = vec![DnaSequence::from_str(&text).unwrap()];
    let config = FmIndexConfig {
        sa_sample_rate: 8,
        use_gpu: false,
        ..Default::default()
    };
    let fm = BidirFmIndex::build_cpu(&seqs, &config).unwrap();
    let rl = BidirRlFmIndex::from_bidir_fm_index(&fm);
    assert!(rl.shallow.ivs.len() <= ShallowCache::<BidirInterval>::MAX_ENTRIES);
    assert!(
        rl.shallow.depth < SHALLOW_MAX_DEPTH as usize,
        "budget never hit"
    );
    let bm = BMoveIndex::build_cpu(&seqs, &MoveFmIndexConfig::default()).unwrap();
    assert!(bm.shallow.ivs.len() <= ShallowCache::<BMoveCursor>::MAX_ENTRIES);
    assert!(bm.shallow.depth <= rl.shallow.depth);
    let sorted = |mut mems: Vec<Mem>| {
        mems.iter_mut().for_each(|m| m.positions.sort());
        mems
    };
    // Few, short reads: every query position fans out over the wildcard variants.
    for _ in 0..3 {
        let at = rng.random_range(0..text.len() - 30);
        let q = encode(&text[at..at + 30]);
        let min_len = 8;
        assert_eq!(
            rl.find_smems(&q, min_len, true),
            fm.find_smems(&q, min_len, true)
        );
        assert_eq!(
            rl.find_mems(&q, min_len, true),
            fm.find_mems(&q, min_len, true)
        );
        assert_eq!(
            sorted(bm.find_smems(&q, min_len, true)),
            sorted(fm.find_smems(&q, min_len, true))
        );
        assert_eq!(
            sorted(bm.find_mems(&q, min_len, true)),
            sorted(fm.find_mems(&q, min_len, true))
        );
    }
}
