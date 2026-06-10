//! IUPAC parity tests for GPU locate/count: assert GPU matches CPU across all
//! IUPAC ambiguity code query × reference combinations.

#[cfg(feature = "gpu")]
mod tests {
    use pollster::FutureExt as _;
    use webgpu_fmidx::alphabet::encode_char;
    use webgpu_fmidx::error::FmIndexError;
    use webgpu_fmidx::{DnaSequence, FmIndex, FmIndexConfig};

    fn cpu_config() -> FmIndexConfig {
        FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
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

    /// Returns `None` when no GPU adapter is available (caller should skip).
    fn try_locate_gpu(
        idx: &FmIndex,
        queries: &[Vec<u8>],
    ) -> Option<Vec<Vec<(String, u32)>>> {
        let refs: Vec<&[u8]> = queries.iter().map(|q| q.as_slice()).collect();
        match idx.locate_gpu(&refs).block_on() {
            Ok(r) => Some(r),
            Err(FmIndexError::GpuError(_)) => None,
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    fn assert_parity(idx: &FmIndex, pat: Vec<u8>, label: &str) {
        let cpu = idx.locate(&pat);
        let Some(gpu_all) = try_locate_gpu(idx, &[pat]) else {
            return;
        };
        assert_eq!(
            sorted_hits(cpu),
            sorted_hits(gpu_all.into_iter().next().unwrap()),
            "parity failed for query '{label}'"
        );
    }

    // ── Single-character IUPAC ambiguity codes ────────────────────────────────

    #[test]
    fn iupac_n_matches_all_bases() {
        // N should match A, C, G, T — one hit each in "ACGT"
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("N"), "N");
    }

    #[test]
    fn iupac_r_matches_a_and_g() {
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("R"), "R");
    }

    #[test]
    fn iupac_y_matches_c_and_t() {
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("Y"), "Y");
    }

    #[test]
    fn iupac_s_matches_g_and_c() {
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("S"), "S");
    }

    #[test]
    fn iupac_w_matches_a_and_t() {
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("W"), "W");
    }

    #[test]
    fn iupac_k_matches_g_and_t() {
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("K"), "K");
    }

    #[test]
    fn iupac_m_matches_a_and_c() {
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("M"), "M");
    }

    #[test]
    fn iupac_b_excludes_a() {
        // B = C or G or T
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("B"), "B");
    }

    #[test]
    fn iupac_d_excludes_c() {
        // D = A or G or T
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("D"), "D");
    }

    #[test]
    fn iupac_h_excludes_g() {
        // H = A or C or T
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("H"), "H");
    }

    #[test]
    fn iupac_v_excludes_t() {
        // V = A or C or G
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("V"), "V");
    }

    // ── Multi-character IUPAC patterns ────────────────────────────────────────

    #[test]
    fn iupac_rr_in_homopolymer_context() {
        // RR should match AA, AG, GA, GG
        let idx = build(&["AAAGGG"]);
        assert_parity(&idx, encode("RR"), "RR");
    }

    #[test]
    fn iupac_nn_matches_all_dinucleotides() {
        let idx = build(&["ACGT"]);
        assert_parity(&idx, encode("NN"), "NN");
    }

    #[test]
    fn iupac_nnn_in_longer_sequence() {
        let idx = build(&["ACGTACGT"]);
        assert_parity(&idx, encode("NNN"), "NNN");
    }

    #[test]
    fn iupac_exact_prefix_ambiguous_suffix() {
        // AC followed by R (A or G) — should match ACG and ACA positions
        let idx = build(&["ACGACA"]);
        assert_parity(&idx, encode("ACR"), "ACR");
    }

    #[test]
    fn iupac_ambiguous_prefix_exact_suffix() {
        let idx = build(&["ACGTACGT"]);
        assert_parity(&idx, encode("RCG"), "RCG");
    }

    // ── Multi-sequence corpus ─────────────────────────────────────────────────

    #[test]
    fn iupac_n_multi_seq() {
        let idx = build(&["ACGT", "TGCA"]);
        assert_parity(&idx, encode("N"), "N multi-seq");
    }

    #[test]
    fn iupac_r_multi_seq() {
        let idx = build(&["AAGG", "CCTT"]);
        assert_parity(&idx, encode("R"), "R multi-seq");
    }

    // ── Batch queries mixing exact and IUPAC ─────────────────────────────────

    #[test]
    fn batch_exact_and_iupac() {
        let idx = build(&["ACGTACGT"]);
        let patterns = vec![encode("ACG"), encode("N"), encode("RY"), encode("ACGT")];
        let cpu_results: Vec<Vec<(String, u32)>> =
            patterns.iter().map(|p| idx.locate(p)).collect();
        let Some(gpu_results) = try_locate_gpu(&idx, &patterns) else {
            return;
        };
        for (i, (cpu, gpu)) in cpu_results.into_iter().zip(gpu_results).enumerate() {
            assert_eq!(sorted_hits(cpu), sorted_hits(gpu), "batch query {i}");
        }
    }

    // ── No-match cases ────────────────────────────────────────────────────────

    #[test]
    fn iupac_no_match_exact_sequence() {
        // Query "S" (G or C) against an A-only reference — no hits
        let idx = build(&["AAAA"]);
        assert_parity(&idx, encode("S"), "S no-match");
    }
}
