//! GPU MEM/SMEM parity tests: assert GPU find_smems_gpu / find_mems_gpu
//! return the same multiset of (query_start, query_end, match_count) as
//! the CPU find_smems / find_mems for the same corpora and queries.

#[cfg(feature = "gpu")]
mod tests {
    use pollster::FutureExt as _;
    use haystackfm::alphabet::DnaSequence;
    use haystackfm::fm_index::smem::Mem;
    use haystackfm::{BidirFmIndex, FmIndexConfig, MemHit};

    fn cpu_config() -> FmIndexConfig {
        FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            ..Default::default()
        }
    }

    fn build(seqs: &[&str]) -> BidirFmIndex {
        let dna: Vec<DnaSequence> = seqs
            .iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect();
        BidirFmIndex::build_cpu(&dna, &cpu_config()).unwrap()
    }

    fn seq(s: &str) -> DnaSequence {
        DnaSequence::from_str(s).unwrap()
    }

    fn cpu_key(m: &Mem) -> (usize, usize, u32) {
        (m.query_start, m.query_end, m.match_count)
    }

    fn gpu_key(m: &MemHit) -> (usize, usize, u32) {
        (m.query_start as usize, m.query_end as usize, m.match_count)
    }

    fn cpu_sorted(mems: &[Mem]) -> Vec<(usize, usize, u32)> {
        let mut keys: Vec<_> = mems.iter().map(cpu_key).collect();
        keys.sort();
        keys
    }

    fn gpu_sorted(mems: &[MemHit]) -> Vec<(usize, usize, u32)> {
        let mut keys: Vec<_> = mems.iter().map(gpu_key).collect();
        keys.sort();
        keys
    }

    // Pass &[] for ref_boundaries — skips position resolution, tests MEM spans only.
    fn smems_gpu_sync(
        idx: &BidirFmIndex,
        queries: &[DnaSequence],
        min_len: usize,
    ) -> Vec<Vec<MemHit>> {
        idx.find_smems_gpu(queries, min_len, &[], 1024)
            .block_on()
            .unwrap()
    }

    fn mems_gpu_sync(
        idx: &BidirFmIndex,
        queries: &[DnaSequence],
        min_len: usize,
    ) -> Vec<Vec<MemHit>> {
        idx.find_mems_gpu(queries, min_len, &[], 1024)
            .block_on()
            .unwrap()
    }

    // ── SMEM parity tests ─────────────────────────────────────────────────────

    #[test]
    fn smem_single_query() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("ACGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let gpu = smems_gpu_sync(&idx, &[q], 1);
        assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]));
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
            assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]), "char={ch}");
        }
    }

    #[test]
    fn smem_multi_seq() {
        let idx = build(&["ACGT", "TGCA"]);
        let q = seq("ACGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let gpu = smems_gpu_sync(&idx, &[q], 1);
        assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]));
    }

    #[test]
    fn smem_repeated_pattern() {
        let idx = build(&["ACGTACGTACGT"]);
        let q = seq("ACGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let gpu = smems_gpu_sync(&idx, &[q], 1);
        assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]));
    }

    #[test]
    fn smem_longer_query() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("ACGTACGT");
        let cpu = idx.find_smems(q.as_slice(), 2, false);
        let gpu = smems_gpu_sync(&idx, &[q], 2);
        assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]));
    }

    // ── MEM parity tests ──────────────────────────────────────────────────────

    #[test]
    fn mem_single_query() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("ACGT");
        let cpu = idx.find_mems(q.as_slice(), 1, false);
        let gpu = mems_gpu_sync(&idx, &[q], 1);
        assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]));
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
        let cpu: Vec<Vec<Mem>> = queries
            .iter()
            .map(|q| idx.find_mems(q.as_slice(), 1, false))
            .collect();
        let gpu = mems_gpu_sync(&idx, &queries, 1);
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(cpu_sorted(c), gpu_sorted(g), "query {i}");
        }
    }

    #[test]
    fn mem_multi_seq() {
        let idx = build(&["ACGT", "TGCA", "AAAA"]);
        let q = seq("ACGTA");
        let cpu = idx.find_mems(q.as_slice(), 1, false);
        let gpu = mems_gpu_sync(&idx, &[q], 1);
        assert_eq!(cpu_sorted(&cpu), gpu_sorted(&gpu[0]));
    }
}
