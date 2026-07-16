//! IUPAC parity tests for GPU MEM/SMEM finding: assert GPU find_smems_gpu /
//! find_mems_gpu match CPU find_smems / find_mems for IUPAC ambiguity queries.

#[cfg(feature = "gpu")]
mod tests {
    use pollster::FutureExt as _;
    use haystackfm::alphabet::DnaSequence;
    use haystackfm::error::FmIndexError;
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

    /// Merge MemHit entries with the same (query_start, query_end) by summing match_counts.
    /// GPU emits one MemHit per SA interval; CPU emits one Mem per span with total count.
    fn merge_gpu_hits(mems: &[MemHit]) -> Vec<(usize, usize, u32)> {
        use std::collections::HashMap;
        let mut map: HashMap<(usize, usize), u32> = HashMap::new();
        for h in mems {
            *map.entry((h.query_start as usize, h.query_end as usize))
                .or_insert(0) += h.match_count;
        }
        let mut keys: Vec<_> = map.into_iter().map(|((qs, qe), mc)| (qs, qe, mc)).collect();
        keys.sort();
        keys
    }

    /// Returns `None` when no GPU adapter is available (caller should skip).
    fn try_smems_gpu(
        idx: &BidirFmIndex,
        queries: &[DnaSequence],
        min_len: usize,
    ) -> Option<Vec<Vec<MemHit>>> {
        match idx.find_smems_gpu(queries, min_len, &[], 1024).block_on() {
            Ok(r) => Some(r),
            Err(FmIndexError::GpuError(_)) => None,
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    /// Returns `None` when no GPU adapter is available (caller should skip).
    fn try_mems_gpu(
        idx: &BidirFmIndex,
        queries: &[DnaSequence],
        min_len: usize,
    ) -> Option<Vec<Vec<MemHit>>> {
        match idx.find_mems_gpu(queries, min_len, &[], 1024).block_on() {
            Ok(r) => Some(r),
            Err(FmIndexError::GpuError(_)) => None,
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // ── SMEM IUPAC parity ─────────────────────────────────────────────────────

    #[test]
    fn smem_iupac_n_in_query() {
        // N is universal — SMEM spans should match CPU
        let idx = build(&["ACGTACGT"]);
        let q = seq("NACGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let Some(gpu) = try_smems_gpu(&idx, &[q], 1) else {
            return;
        };
        assert_eq!(cpu_sorted(&cpu), merge_gpu_hits(&gpu[0]));
    }

    #[test]
    fn smem_iupac_r_in_query() {
        // R = A or G
        let idx = build(&["ACGTACGT"]);
        let q = seq("RCGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let Some(gpu) = try_smems_gpu(&idx, &[q], 1) else {
            return;
        };
        assert_eq!(cpu_sorted(&cpu), merge_gpu_hits(&gpu[0]));
    }

    #[test]
    fn smem_iupac_y_in_query() {
        // Y = C or T
        let idx = build(&["ACGTACGT"]);
        let q = seq("AYGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let Some(gpu) = try_smems_gpu(&idx, &[q], 1) else {
            return;
        };
        assert_eq!(cpu_sorted(&cpu), merge_gpu_hits(&gpu[0]));
    }

    #[test]
    fn smem_iupac_n_only_query() {
        // All-N query — every position is maximal
        let idx = build(&["ACGT"]);
        let q = seq("NN");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let Some(gpu) = try_smems_gpu(&idx, &[q], 1) else {
            return;
        };
        assert_eq!(cpu_sorted(&cpu), merge_gpu_hits(&gpu[0]));
    }

    #[test]
    fn smem_iupac_multi_seq_corpus() {
        let idx = build(&["ACGT", "TGCA", "GGGG"]);
        let q = seq("RNGT");
        let cpu = idx.find_smems(q.as_slice(), 1, false);
        let Some(gpu) = try_smems_gpu(&idx, &[q], 1) else {
            return;
        };
        assert_eq!(cpu_sorted(&cpu), merge_gpu_hits(&gpu[0]));
    }

    #[test]
    fn smem_iupac_min_len_filter() {
        // Short match suppressed by min_len
        let idx = build(&["ACGT"]);
        let q = seq("NA");
        let cpu = idx.find_smems(q.as_slice(), 3, false);
        let Some(gpu) = try_smems_gpu(&idx, &[q], 3) else {
            return;
        };
        assert_eq!(cpu_sorted(&cpu), merge_gpu_hits(&gpu[0]));
    }

    #[test]
    fn smem_iupac_batch_queries() {
        let idx = build(&["ACGTACGT"]);
        let queries = vec![seq("NACGT"), seq("RCGT"), seq("ACYN")];
        let cpu: Vec<Vec<Mem>> = queries
            .iter()
            .map(|q| idx.find_smems(q.as_slice(), 1, false))
            .collect();
        let Some(gpu) = try_smems_gpu(&idx, &queries, 1) else {
            return;
        };
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(cpu_sorted(c), merge_gpu_hits(g), "smem batch query {i}");
        }
    }

    // ── MEM IUPAC parity ──────────────────────────────────────────────────────

    #[test]
    fn mem_iupac_n_in_query() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("NACGT");
        let cpu = idx.find_mems(q.as_slice(), 1, false);
        let Some(gpu) = try_mems_gpu(&idx, &[q], 1) else {
            return;
        };
        assert_eq!(cpu_sorted(&cpu), merge_gpu_hits(&gpu[0]));
    }

    #[test]
    fn mem_iupac_r_in_query() {
        let idx = build(&["ACGTACGT"]);
        let q = seq("RCGT");
        let cpu = idx.find_mems(q.as_slice(), 1, false);
        let Some(gpu) = try_mems_gpu(&idx, &[q], 1) else {
            return;
        };
        assert_eq!(cpu_sorted(&cpu), merge_gpu_hits(&gpu[0]));
    }

    #[test]
    fn mem_iupac_mixed_query() {
        // Mix of exact and ambiguous bases
        let idx = build(&["ACGTACGT"]);
        let q = seq("ANGT");
        let cpu = idx.find_mems(q.as_slice(), 1, false);
        let Some(gpu) = try_mems_gpu(&idx, &[q], 1) else {
            return;
        };
        assert_eq!(cpu_sorted(&cpu), merge_gpu_hits(&gpu[0]));
    }

    #[test]
    fn mem_iupac_all_ambiguous() {
        let idx = build(&["ACGT"]);
        let q = seq("NNN");
        let cpu = idx.find_mems(q.as_slice(), 1, false);
        let Some(gpu) = try_mems_gpu(&idx, &[q], 1) else {
            return;
        };
        assert_eq!(cpu_sorted(&cpu), merge_gpu_hits(&gpu[0]));
    }

    #[test]
    fn mem_iupac_multi_seq_corpus() {
        let idx = build(&["ACGT", "TGCA", "GGGG"]);
        let q = seq("RNGT");
        let cpu = idx.find_mems(q.as_slice(), 1, false);
        let Some(gpu) = try_mems_gpu(&idx, &[q], 1) else {
            return;
        };
        assert_eq!(cpu_sorted(&cpu), merge_gpu_hits(&gpu[0]));
    }

    #[test]
    fn mem_iupac_no_match() {
        // S = G or C; all-A reference has no match
        let idx = build(&["AAAA"]);
        let q = seq("SSS");
        let cpu = idx.find_mems(q.as_slice(), 1, false);
        let Some(gpu) = try_mems_gpu(&idx, &[q], 1) else {
            return;
        };
        assert!(cpu.is_empty());
        assert!(merge_gpu_hits(&gpu[0]).is_empty());
    }

    #[test]
    fn mem_iupac_batch_queries() {
        let idx = build(&["ACGTACGT"]);
        let queries = vec![seq("NACGT"), seq("RCGT"), seq("ACYN")];
        let cpu: Vec<Vec<Mem>> = queries
            .iter()
            .map(|q| idx.find_mems(q.as_slice(), 1, false))
            .collect();
        let Some(gpu) = try_mems_gpu(&idx, &queries, 1) else {
            return;
        };
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(cpu_sorted(c), merge_gpu_hits(g), "mem batch query {i}");
        }
    }
}
