//! GPU paths on an index with identical input sequences: GPU construction must store the
//! same classes as the CPU, and `locate_gpu` / `find_smems_gpu` must expand hits to every
//! class member and count them like the CPU paths.

#[cfg(feature = "gpu")]
mod tests {
    use haystackfm::alphabet::encode_char;
    use haystackfm::{BidirFmIndex, DnaSequence, FmIndex, FmIndexConfig, SeqId};
    use pollster::FutureExt as _;
    use std::collections::BTreeSet;

    /// Classes {0, 2, 3}, {1}, {4, 5}.
    const CORPUS: &[&str] = &[
        "ACGTGATTACAGG",
        "TTGCAGATTACA",
        "ACGTGATTACAGG",
        "ACGTGATTACAGG",
        "CC#GATTACA",
        "CC#GATTACA",
    ];

    const PATTERNS: &[&str] = &["GATTACA", "TACA", "A", "ACGTG", "CC", "TTGCA"];

    fn encode(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    fn sequences() -> Vec<DnaSequence> {
        CORPUS
            .iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect()
    }

    fn config(sa_sample_rate: u32, use_gpu: bool) -> FmIndexConfig {
        FmIndexConfig {
            sa_sample_rate,
            use_gpu,
            ..Default::default()
        }
    }

    fn hit_set(hits: impl IntoIterator<Item = (SeqId, u32)>) -> BTreeSet<(SeqId, u32)> {
        hits.into_iter().collect()
    }

    #[test]
    fn gpu_build_and_locate_expand_classes() {
        let patterns: Vec<Vec<u8>> = PATTERNS.iter().map(|p| encode(p)).collect();
        for rate in [1, 4] {
            let cpu = FmIndex::build_cpu(&sequences(), &config(rate, false)).unwrap();
            let gpu_built = FmIndex::build(&sequences(), &config(rate, true))
                .block_on()
                .unwrap();
            assert_eq!(gpu_built.seq_classes(), cpu.seq_classes());
            assert_eq!(gpu_built.num_distinct_sequences(), 3);
            assert_eq!(gpu_built.count(&[]), cpu.count(&[]));
            let located = cpu.locate_gpu(&patterns).block_on().unwrap();
            let gpu_located = gpu_built.locate_gpu(&patterns).block_on().unwrap();
            for (k, pat) in patterns.iter().enumerate() {
                let want = cpu.locate(pat);
                let ctx = format!("{} rate {rate}", PATTERNS[k]);
                assert_eq!(want.len() as u32, cpu.count(pat), "{ctx}");
                assert_eq!(gpu_built.count(pat), cpu.count(pat), "{ctx}");
                assert_eq!(
                    hit_set(gpu_built.locate(pat)),
                    hit_set(want.clone()),
                    "{ctx}"
                );
                assert_eq!(located[k].len(), want.len(), "locate_gpu, {ctx}");
                assert_eq!(hit_set(located[k].clone()), hit_set(want.clone()), "{ctx}");
                assert_eq!(hit_set(gpu_located[k].clone()), hit_set(want), "{ctx}");
            }
        }
    }

    /// GPU construction of every GPU-buildable index type stores the CPU build's classes
    /// and answers counts / locate like it.
    #[test]
    fn gpu_builds_of_every_index_keep_the_classes() {
        use haystackfm::{BidirRlFmIndex, RlFmIndex, RlFmIndexConfig};
        let rl_config = RlFmIndexConfig {
            sa_sample_rate: 2,
            build_threads: 1,
            lookup_depth: 2,
        };
        let fm = FmIndex::build_cpu(&sequences(), &config(2, false)).unwrap();
        let bidir = BidirFmIndex::build(&sequences(), &config(2, true))
            .block_on()
            .unwrap();
        let rl = RlFmIndex::build(&sequences(), &rl_config)
            .block_on()
            .unwrap();
        let bidir_rl = BidirRlFmIndex::build(&sequences(), &rl_config)
            .block_on()
            .unwrap();
        assert_eq!(bidir.seq_classes(), fm.seq_classes());
        assert_eq!(rl.seq_classes(), fm.seq_classes());
        assert_eq!(bidir_rl.seq_classes(), fm.seq_classes());
        assert_eq!(bidir.text_len(), fm.text_len());
        for p in PATTERNS {
            let pat = encode(p);
            let want = hit_set(fm.locate(&pat));
            assert_eq!(rl.count(&pat), fm.count(&pat), "rl {p}");
            assert_eq!(bidir_rl.count(&pat), fm.count(&pat), "rl bidir {p}");
            assert_eq!(hit_set(rl.locate(&pat)), want, "rl {p}");
            let mut iv = Some(bidir.full_interval());
            for &c in pat.iter().rev() {
                iv = iv.and_then(|iv| bidir.extend_left(iv, c));
            }
            if let Some(iv) = iv {
                assert_eq!(bidir.count_interval(&iv), fm.count(&pat), "bidir {p}");
                assert_eq!(hit_set(bidir.locate_interval(&iv)), want, "bidir {p}");
            }
        }
        for (i, seq) in sequences().iter().enumerate() {
            let id = SeqId::new(i as u32);
            assert_eq!(rl.sequence(id).as_deref(), Some(seq.as_slice()), "rl {i}");
            assert_eq!(bidir.sequence(id), Some(seq.as_slice()), "bidir {i}");
        }
    }

    fn queries() -> Vec<DnaSequence> {
        ["ACGTGATTACAGG", "TTGCAGATTACAGG", "CCGATTACAT"]
            .iter()
            .map(|q| DnaSequence::from_str(q).unwrap())
            .collect()
    }

    #[test]
    fn gpu_smems_expand_to_every_member_and_respect_the_cap() {
        let idx = BidirFmIndex::build_cpu(&sequences(), &config(2, false)).unwrap();
        let bounds = idx.seq_boundaries().to_vec();
        assert_eq!(bounds.len(), 3);
        let qs = queries();
        let full = idx
            .find_smems_gpu(&qs, 3, &bounds, 1024)
            .block_on()
            .unwrap();
        let counts_only = idx.find_smems_gpu(&qs, 3, &[], 1024).block_on().unwrap();
        let capped = idx.find_smems_gpu(&qs, 3, &bounds, 1).block_on().unwrap();
        let custom = idx
            .find_smems_gpu(&qs, 3, &[idx.text_len()], 1024)
            .block_on()
            .unwrap();
        for (q, query) in qs.iter().enumerate() {
            for (k, hit) in full[q].iter().enumerate() {
                let ctx = format!("query {q} hit {k}");
                // Expanded: one position per occurrence in every member, ids in range.
                assert!(!hit.truncated, "{ctx}");
                assert_eq!(hit.positions.len() as u32, hit.match_count, "{ctx}");
                assert_eq!(
                    hit_set(hit.positions.clone()).len(),
                    hit.positions.len(),
                    "{ctx}"
                );
                let want = &query.as_slice()[hit.query_start as usize..hit.query_end as usize];
                for &(id, off) in &hit.positions {
                    let at = off as usize;
                    let got = &CORPUS[id.index()][at..at + want.len()];
                    assert_eq!(encode(got), want, "{ctx}");
                }
                let counted = &counts_only[q][k];
                assert_eq!(counted.match_count, hit.match_count, "{ctx}");
                assert!(counted.positions.is_empty(), "{ctx}");
                let one = &capped[q][k];
                assert_eq!(one.positions.len(), 1, "{ctx}");
                assert_eq!(one.truncated, hit.match_count > 1, "{ctx}");
                // A caller's boundaries: one position per stored occurrence.
                let raw = &custom[q][k];
                assert_eq!(raw.match_count, hit.match_count, "{ctx}");
                assert!(raw.positions.len() <= hit.positions.len(), "{ctx}");
                assert!(
                    raw.positions.iter().all(|&(id, _)| id == SeqId::new(0)),
                    "{ctx}"
                );
            }
        }
        // Some SMEM hits a duplicated class, so expansion is exercised.
        assert!(full
            .iter()
            .flatten()
            .any(|h| h.positions.iter().any(|&(id, _)| id == SeqId::new(3))));
    }
}
