//! GPU locate parity tests: assert locate_gpu returns the same multiset of
//! (seq_header, pos) pairs as the CPU locate for random corpora and queries.

#[cfg(feature = "gpu")]
mod tests {
    use pollster::FutureExt as _;
    use webgpu_fmidx::alphabet::encode_char;
    use webgpu_fmidx::{DnaSequence, FmIndex, FmIndexConfig};

    fn cpu_config() -> FmIndexConfig {
        FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            ..Default::default()
        }
    }

    fn encode(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    fn sorted_hits(mut v: Vec<(String, u32)>) -> Vec<(String, u32)> {
        v.sort();
        v
    }

    fn build(seqs: &[&str]) -> FmIndex {
        let dna: Vec<DnaSequence> = seqs
            .iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect();
        FmIndex::build_cpu(&dna, &cpu_config()).unwrap()
    }

    fn locate_gpu_sync(idx: &FmIndex, queries: &[Vec<u8>]) -> Vec<Vec<(String, u32)>> {
        let refs: Vec<&[u8]> = queries.iter().map(|q| q.as_slice()).collect();
        idx.locate_gpu(&refs).block_on().unwrap()
    }

    #[test]
    fn single_seq_single_query() {
        let idx = build(&["ACGTACGT"]);
        let pat = encode("ACGT");
        let cpu = idx.locate(&pat);
        let gpu = locate_gpu_sync(&idx, &[pat])[0].clone();
        assert_eq!(sorted_hits(cpu), sorted_hits(gpu));
    }

    #[test]
    fn no_match() {
        let idx = build(&["ACGTACGT"]);
        let pat = encode("TTTT");
        let cpu = idx.locate(&pat);
        let gpu = locate_gpu_sync(&idx, &[pat])[0].clone();
        assert!(cpu.is_empty());
        assert!(gpu.is_empty());
    }

    #[test]
    fn empty_pattern_matches_all() {
        let idx = build(&["ACGT"]);
        let cpu = idx.locate(&[]);
        let gpu = locate_gpu_sync(&idx, &[vec![]])[0].clone();
        assert_eq!(sorted_hits(cpu), sorted_hits(gpu));
    }

    #[test]
    fn single_char_repeated() {
        let idx = build(&["AAAACCCGGG"]);
        for ch in ["A", "C", "G"] {
            let pat = encode(ch);
            let cpu = idx.locate(&pat);
            let gpu = locate_gpu_sync(&idx, &[pat.clone()])[0].clone();
            assert_eq!(sorted_hits(cpu), sorted_hits(gpu), "char={ch}");
        }
    }

    #[test]
    fn multi_seq_single_query() {
        let idx = build(&["ACGT", "TGCA"]);
        let pat = encode("A");
        let cpu = idx.locate(&pat);
        let gpu = locate_gpu_sync(&idx, &[pat])[0].clone();
        assert_eq!(sorted_hits(cpu), sorted_hits(gpu));
    }

    #[test]
    fn batch_multiple_queries() {
        let idx = build(&["ACGTACGT", "NNNACGTNNN"]);
        let patterns_valid = vec![encode("ACG"), encode("ACGT")];
        let cpu_results: Vec<Vec<(String, u32)>> =
            patterns_valid.iter().map(|p| idx.locate(p)).collect();
        let gpu_results = locate_gpu_sync(&idx, &patterns_valid);
        for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            assert_eq!(
                sorted_hits(cpu.clone()),
                sorted_hits(gpu.clone()),
                "query {i}"
            );
        }
    }

    #[test]
    fn pattern_with_n() {
        let idx = build(&["ACNGT"]);
        let pat = encode("N");
        let cpu = idx.locate(&pat);
        let gpu = locate_gpu_sync(&idx, &[pat])[0].clone();
        assert_eq!(sorted_hits(cpu), sorted_hits(gpu));
    }

    #[test]
    fn many_sequences() {
        let seqs: Vec<&str> = vec!["ACGT", "TTTT", "CCCC", "GGGG", "ACGTACGT"];
        let idx = build(&seqs);
        let pat = encode("ACGT");
        let cpu = idx.locate(&pat);
        let gpu = locate_gpu_sync(&idx, &[pat])[0].clone();
        assert_eq!(sorted_hits(cpu), sorted_hits(gpu));
    }

    #[test]
    fn large_batch() {
        let idx = build(&["ACGTACGTACGTACGT"]);
        let patterns: Vec<Vec<u8>> = vec![
            encode("A"),
            encode("C"),
            encode("G"),
            encode("T"),
            encode("AC"),
            encode("CG"),
            encode("GT"),
            encode("ACG"),
        ];
        let cpu_results: Vec<Vec<(String, u32)>> = patterns.iter().map(|p| idx.locate(p)).collect();
        let gpu_results = locate_gpu_sync(&idx, &patterns);
        for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            assert_eq!(
                sorted_hits(cpu.clone()),
                sorted_hits(gpu.clone()),
                "query {i}"
            );
        }
    }
}
