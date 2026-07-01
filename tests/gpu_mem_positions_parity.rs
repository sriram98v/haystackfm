//! Position-resolve parity tests: GPU find_smems_gpu / find_mems_gpu must return
//! the same (ref_id, offset) position sets as the CPU oracle (find_smems + locate).

#[cfg(feature = "gpu")]
mod tests {
    use pollster::FutureExt as _;
    use std::collections::HashSet;
    use webgpu_fmidx::alphabet::DnaSequence;
    use webgpu_fmidx::{BidirFmIndex, FmIndexConfig, MemHit};

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

    /// Parse "seq_N" → N.
    fn seq_name_to_id(name: &str) -> u32 {
        name.strip_prefix("seq_")
            .and_then(|n| n.parse().ok())
            .unwrap_or(0)
    }

    /// CPU oracle: find MEMs then locate each one. Returns sorted vec of
    /// (query_start, query_end, sorted_positions).
    fn cpu_mems_with_positions(
        idx: &BidirFmIndex,
        query: &[u8],
        min_len: usize,
    ) -> Vec<(usize, usize, HashSet<(u32, u32)>)> {
        idx.find_mems(query, min_len, true)
            .into_iter()
            .map(|m| {
                let positions: HashSet<(u32, u32)> = m
                    .positions
                    .iter()
                    .map(|(name, off)| (seq_name_to_id(name), *off))
                    .collect();
                (m.query_start, m.query_end, positions)
            })
            .collect()
    }

    fn cpu_smems_with_positions(
        idx: &BidirFmIndex,
        query: &[u8],
        min_len: usize,
    ) -> Vec<(usize, usize, HashSet<(u32, u32)>)> {
        idx.find_smems(query, min_len, true)
            .into_iter()
            .map(|m| {
                let positions: HashSet<(u32, u32)> = m
                    .positions
                    .iter()
                    .map(|(name, off)| (seq_name_to_id(name), *off))
                    .collect();
                (m.query_start, m.query_end, positions)
            })
            .collect()
    }

    /// GPU results as comparable structure.
    fn gpu_mems_with_positions(hits: &[MemHit]) -> Vec<(usize, usize, HashSet<(u32, u32)>)> {
        hits.iter()
            .map(|h| {
                let positions: HashSet<(u32, u32)> = h.positions.iter().copied().collect();
                (h.query_start as usize, h.query_end as usize, positions)
            })
            .collect()
    }

    fn sort_mem_tuples(v: &mut Vec<(usize, usize, HashSet<(u32, u32)>)>) {
        v.sort_by_key(|(qs, qe, _)| (*qs, *qe));
    }

    // ── SMEM position tests ───────────────────────────────────────────────────

    #[test]
    fn smem_positions_single_ref() {
        let idx = build(&["ACGTACGT"]);
        let boundaries = idx.seq_boundaries().to_vec();
        let q = seq("ACGT");

        let mut cpu = cpu_smems_with_positions(&idx, q.as_slice(), 1);
        let gpu_raw = idx
            .find_smems_gpu(&[q], 1, &boundaries, 1024)
            .block_on()
            .unwrap();
        let mut gpu = gpu_mems_with_positions(&gpu_raw[0]);

        sort_mem_tuples(&mut cpu);
        sort_mem_tuples(&mut gpu);

        assert_eq!(cpu.len(), gpu.len(), "MEM count differs");
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(c.0, g.0, "MEM {i}: query_start");
            assert_eq!(c.1, g.1, "MEM {i}: query_end");
            assert_eq!(
                c.2, g.2,
                "MEM {i}: positions mismatch\ncpu={:?}\ngpu={:?}",
                c.2, g.2
            );
        }
    }

    #[test]
    fn smem_positions_multi_ref() {
        let idx = build(&["ACGTACGT", "ACGT", "TTTTACGT"]);
        let boundaries = idx.seq_boundaries().to_vec();
        let q = seq("ACGT");

        let mut cpu = cpu_smems_with_positions(&idx, q.as_slice(), 1);
        let gpu_raw = idx
            .find_smems_gpu(&[q], 1, &boundaries, 1024)
            .block_on()
            .unwrap();
        let mut gpu = gpu_mems_with_positions(&gpu_raw[0]);

        sort_mem_tuples(&mut cpu);
        sort_mem_tuples(&mut gpu);

        assert_eq!(cpu.len(), gpu.len());
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(c.2, g.2, "MEM {i}: positions\ncpu={:?}\ngpu={:?}", c.2, g.2);
        }
    }

    // ── MEM position tests ────────────────────────────────────────────────────

    #[test]
    fn mem_positions_single_ref() {
        let idx = build(&["ACGTACGT"]);
        let boundaries = idx.seq_boundaries().to_vec();
        let q = seq("ACGT");

        let mut cpu = cpu_mems_with_positions(&idx, q.as_slice(), 1);
        let gpu_raw = idx
            .find_mems_gpu(&[q], 1, &boundaries, 1024)
            .block_on()
            .unwrap();
        let mut gpu = gpu_mems_with_positions(&gpu_raw[0]);

        sort_mem_tuples(&mut cpu);
        sort_mem_tuples(&mut gpu);

        assert_eq!(cpu.len(), gpu.len());
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(c.2, g.2, "MEM {i}: positions\ncpu={:?}\ngpu={:?}", c.2, g.2);
        }
    }

    #[test]
    fn mem_positions_multi_ref_batch() {
        let idx = build(&["ACGTACGT", "CGTTAGCC", "AAACGT"]);
        let boundaries = idx.seq_boundaries().to_vec();
        let queries = vec![seq("ACGT"), seq("CGT"), seq("AAA")];

        let cpu: Vec<_> = queries
            .iter()
            .map(|q| {
                let mut v = cpu_mems_with_positions(&idx, q.as_slice(), 1);
                sort_mem_tuples(&mut v);
                v
            })
            .collect();

        let gpu_raw = idx
            .find_mems_gpu(&queries, 1, &boundaries, 1024)
            .block_on()
            .unwrap();

        for (qi, (cpu_q, gpu_q)) in cpu.iter().zip(gpu_raw.iter()).enumerate() {
            let mut gpu_v = gpu_mems_with_positions(gpu_q);
            sort_mem_tuples(&mut gpu_v);
            assert_eq!(cpu_q.len(), gpu_v.len(), "query {qi}: MEM count");
            for (i, (c, g)) in cpu_q.iter().zip(gpu_v.iter()).enumerate() {
                assert_eq!(
                    c.2, g.2,
                    "query {qi} MEM {i}: positions\ncpu={:?}\ngpu={:?}",
                    c.2, g.2
                );
            }
        }
    }

    #[test]
    fn mem_positions_empty_query() {
        let idx = build(&["ACGTACGT"]);
        let boundaries = idx.seq_boundaries().to_vec();
        let q = seq("A"); // length 1, min_len=4 → no hits
        let gpu_raw = idx
            .find_mems_gpu(&[q], 4, &boundaries, 1024)
            .block_on()
            .unwrap();
        assert!(gpu_raw[0].is_empty());
    }

    #[test]
    fn mem_positions_no_ref_boundaries_skips_resolve() {
        // Passing &[] skips SA resolve — positions should be empty, no panic.
        let idx = build(&["ACGTACGT"]);
        let q = seq("ACGT");
        let gpu_raw = idx.find_mems_gpu(&[q], 1, &[], 1024).block_on().unwrap();
        // MEMs found but positions empty (resolve skipped).
        assert!(!gpu_raw[0].is_empty());
        for hit in &gpu_raw[0] {
            assert!(hit.positions.is_empty());
        }
    }

    #[test]
    fn mem_truncation_flag_set_when_capped() {
        // Build an index where a short pattern has many hits.
        // Use max_hits_per_mem=1 to force truncation on any MEM with >1 hit.
        let idx = build(&["ACGTACGTACGTACGT"]); // "ACGT" appears 4 times
        let boundaries = idx.seq_boundaries().to_vec();
        let q = seq("ACGT");
        let gpu_raw = idx
            .find_mems_gpu(&[q], 1, &boundaries, 1)
            .block_on()
            .unwrap();
        // The ACGT MEM has 4 hits but cap=1 → truncated=true, positions.len()=1.
        let hit = &gpu_raw[0][0];
        assert_eq!(hit.positions.len(), 1);
        assert!(hit.truncated);
    }
}
