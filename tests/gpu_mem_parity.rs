//! GPU MEM/SMEM parity tests: assert GPU find_smems_gpu / find_mems_gpu
//! return the same multiset of (query_start, query_end, match_count) as
//! the CPU find_smems / find_mems for random corpora and queries.

#[cfg(feature = "gpu")]
mod tests {
    use pollster::FutureExt as _;
    use webgpu_fmidx::alphabet::DnaSequence;
    use webgpu_fmidx::fm_index::smem::Mem;
    use webgpu_fmidx::{BidirFmIndex, FmIndexConfig};

    fn cpu_config() -> FmIndexConfig {
        FmIndexConfig { sa_sample_rate: 1, use_gpu: false }
    }

    fn build(seqs: &[&str]) -> BidirFmIndex {
        let dna: Vec<DnaSequence> = seqs.iter().map(|s| DnaSequence::from_str(s).unwrap()).collect();
        BidirFmIndex::build_cpu(&dna, &cpu_config()).unwrap()
    }

    fn seq(s: &str) -> DnaSequence {
        DnaSequence::from_str(s).unwrap()
    }

    fn mem_key(m: &Mem) -> (usize, usize, u32) {
        (m.query_start, m.query_end, m.match_count)
    }

    fn sorted_keys(mems: &[Mem]) -> Vec<(usize, usize, u32)> {
        let mut keys: Vec<_> = mems.iter().map(mem_key).collect();
        keys.sort();
        keys
    }

    fn smems_gpu_sync(idx: &BidirFmIndex, queries: &[DnaSequence], min_len: usize) -> Vec<Vec<Mem>> {
        idx.find_smems_gpu(queries, min_len).block_on().unwrap()
    }

    fn mems_gpu_sync(idx: &BidirFmIndex, queries: &[DnaSequence], min_len: usize) -> Vec<Vec<Mem>> {
        idx.find_mems_gpu(queries, min_len).block_on().unwrap()
    }

    // ── SMEM parity tests ─────────────────────────────────────────────────────

    #[test]
    fn smem_single_query() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("ACGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let gpu = smems_gpu_sync(&idx, &[q], 1);
        assert_eq!(sorted_keys(&cpu), sorted_keys(&gpu[0]));
    }

    #[test]
    fn smem_no_match() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("TTTT");
        let cpu = idx.find_smems(q.as_slice(), 4, false);
        let gpu = smems_gpu_sync(&idx, &[q], 4);
        assert!(cpu.is_empty());
        assert!(gpu[0].is_empty());
    }

    #[test]
    fn smem_min_len_filter() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("AC");
        let cpu = idx.find_smems(q.as_slice(), 3, false);
        let gpu = smems_gpu_sync(&idx, &[q], 3);
        assert!(cpu.is_empty());
        assert!(gpu[0].is_empty());
    }

    #[test]
    fn smem_single_char() {
        let idx = build(&["AAAACCCGGG"]);
        for ch in ["A", "C", "G"] {
            let q = seq(ch);
            let cpu = idx.find_smems(q.as_slice(), 1, false);
            let gpu = smems_gpu_sync(&idx, &[q.clone()], 1);
            assert_eq!(sorted_keys(&cpu), sorted_keys(&gpu[0]), "char={ch}");
        }
    }

    #[test]
    fn smem_multi_seq() {
        let idx = build(&["ACGT", "TGCA"]);
        let q = seq("ACGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let gpu = smems_gpu_sync(&idx, &[q], 1);
        assert_eq!(sorted_keys(&cpu), sorted_keys(&gpu[0]));
    }

    #[test]
    fn smem_batch() {
        let idx = build(&["ACGTACGT", "NNNACGTNNN"]);
        let queries = vec![seq("ACG"), seq("ACGT"), seq("TTT")];
        let cpu: Vec<Vec<Mem>> = queries.iter().map(|q| idx.find_smems(q.as_slice(), 1, false)).collect();
        let gpu = smems_gpu_sync(&idx, &queries, 1);
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(sorted_keys(c), sorted_keys(g), "query {i}");
        }
    }

    #[test]
    fn smem_with_n() {
        let idx = build(&["ACNGT"]);
        let q = seq("CN");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let gpu = smems_gpu_sync(&idx, &[q], 1);
        assert_eq!(sorted_keys(&cpu), sorted_keys(&gpu[0]));
    }

    #[test]
    fn smem_repeated_pattern() {
        let idx = build(&["ACGTACGTACGT"]);
        let q = seq("ACGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let gpu = smems_gpu_sync(&idx, &[q], 1);
        assert_eq!(sorted_keys(&cpu), sorted_keys(&gpu[0]));
    }

    #[test]
    fn smem_longer_query() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("ACGTACGT");
        let cpu = idx.find_smems(q.as_slice(), 2, false);
        let gpu = smems_gpu_sync(&idx, &[q], 2);
        assert_eq!(sorted_keys(&cpu), sorted_keys(&gpu[0]));
    }

    // ── MEM parity tests ──────────────────────────────────────────────────────

    #[test]
    fn mem_single_query() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("ACGT");
        let cpu = idx.find_mems(q.as_slice(), 1, false);
        let gpu = mems_gpu_sync(&idx, &[q], 1);
        assert_eq!(sorted_keys(&cpu), sorted_keys(&gpu[0]));
    }

    #[test]
    fn mem_no_match() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("TTTT");
        let cpu = idx.find_mems(q.as_slice(), 4, false);
        let gpu = mems_gpu_sync(&idx, &[q], 4);
        assert!(cpu.is_empty());
        assert!(gpu[0].is_empty());
    }

    #[test]
    fn mem_batch() {
        let idx = build(&["ACGTACGT"]);
        let queries = vec![seq("A"), seq("AC"), seq("ACG"), seq("ACGT")];
        let cpu: Vec<Vec<Mem>> = queries.iter().map(|q| idx.find_mems(q.as_slice(), 1, false)).collect();
        let gpu = mems_gpu_sync(&idx, &queries, 1);
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(sorted_keys(c), sorted_keys(g), "query {i}");
        }
    }

    #[test]
    fn mem_multi_seq() {
        let idx = build(&["ACGT", "TGCA", "AAAA"]);
        let q = seq("ACGTA");
        let cpu = idx.find_mems(q.as_slice(), 1, false);
        let gpu = mems_gpu_sync(&idx, &[q], 1);
        assert_eq!(sorted_keys(&cpu), sorted_keys(&gpu[0]));
    }
}
