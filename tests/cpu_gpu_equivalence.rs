//! Property tests verifying that CPU and GPU construction produce identical FM-indexes.
//!
//! The invariant: for any input, `FmIndex::build_cpu()` and `FmIndex::build()` must produce
//! structurally equivalent indexes that give identical query results.

#![cfg(feature = "gpu")]

mod common;

use haystackfm::alphabet::{encode_char, DnaSequence};
use haystackfm::fm_index::{FmIndex, FmIndexConfig};
use haystackfm::gpu::GpuContext;

fn get_gpu_context() -> Option<GpuContext> {
    pollster::block_on(GpuContext::new()).ok()
}

fn config() -> FmIndexConfig {
    FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: true,
        ..Default::default()
    }
}

fn encode_pattern(s: &str) -> Vec<u8> {
    s.chars().map(|c| encode_char(c).unwrap()).collect()
}

/// Build both CPU and GPU indexes for the given DNA string, return them.
fn both_indexes(dna: &str) -> (FmIndex, FmIndex) {
    let seq = DnaSequence::from_str(dna).unwrap();
    let cfg = config();
    let cpu = FmIndex::build_cpu(&[seq.clone()], &cfg).unwrap();
    let gpu = pollster::block_on(FmIndex::build(&[seq], &cfg)).unwrap();
    (cpu, gpu)
}

/// Assert that two FM-indexes produce identical results for a set of patterns.
fn assert_query_equivalent(cpu: &FmIndex, gpu: &FmIndex, patterns: &[&str]) {
    for pattern in patterns {
        let p = encode_pattern(pattern);
        assert_eq!(
            cpu.count(&p),
            gpu.count(&p),
            "count({:?}) mismatch",
            pattern
        );
        let mut cpu_locs = cpu.locate(&p);
        let mut gpu_locs = gpu.locate(&p);
        cpu_locs.sort();
        gpu_locs.sort();
        assert_eq!(cpu_locs, gpu_locs, "locate({:?}) mismatch", pattern);
    }
}

// ===================== Count + locate equivalence =====================

#[test]
fn cpu_gpu_queries_equal_basic() {
    let Some(_) = get_gpu_context() else {
        return;
    };
    let (cpu, gpu) = both_indexes("ACGTACGTACGT");
    assert_query_equivalent(
        &cpu,
        &gpu,
        &["A", "C", "G", "T", "AC", "ACG", "ACGT", "AAA"],
    );
}

#[test]
fn cpu_gpu_queries_equal_repetitive() {
    let Some(_) = get_gpu_context() else {
        return;
    };
    let dna = "AAACCCGGGTTT".repeat(3);
    let (cpu, gpu) = both_indexes(&dna);
    assert_query_equivalent(&cpu, &gpu, &["AAA", "CCC", "GGG", "TTT", "AC", "AAACCC"]);
}

#[test]
fn cpu_gpu_queries_equal_multi_block() {
    let Some(_) = get_gpu_context() else {
        return;
    };
    let dna = "ACGT".repeat(20); // 80 chars → 2 Occ blocks
    let (cpu, gpu) = both_indexes(&dna);
    assert_query_equivalent(&cpu, &gpu, &["ACG", "ACGT", "CGTA", "T"]);
}

#[test]
fn cpu_gpu_absent_pattern_returns_zero() {
    let Some(_) = get_gpu_context() else {
        return;
    };
    let (cpu, gpu) = both_indexes("ACGT");
    let p = encode_pattern("TTTT");
    assert_eq!(cpu.count(&p), 0);
    assert_eq!(gpu.count(&p), 0);
    assert!(cpu.locate(&p).is_empty());
    assert!(gpu.locate(&p).is_empty());
}

// ===================== Serialization round-trip =====================

#[test]
fn cpu_gpu_serialize_roundtrip() {
    let Some(_) = get_gpu_context() else {
        return;
    };
    let (cpu, gpu) = both_indexes("ACGTACGTACGT");

    let cpu_bytes = cpu.to_bytes().unwrap();
    let gpu_bytes = gpu.to_bytes().unwrap();

    // Both should deserialize cleanly
    let cpu2 = FmIndex::from_bytes(&cpu_bytes).unwrap();
    let gpu2 = FmIndex::from_bytes(&gpu_bytes).unwrap();

    // And produce the same query results
    let p = encode_pattern("ACGT");
    assert_eq!(cpu2.count(&p), gpu2.count(&p));

    let mut cpu_locs = cpu2.locate(&p);
    let mut gpu_locs = gpu2.locate(&p);
    cpu_locs.sort();
    gpu_locs.sort();
    assert_eq!(cpu_locs, gpu_locs);
}

// ===================== Multi-sequence equivalence =====================

#[test]
fn cpu_gpu_multi_sequence_equal() {
    let Some(_) = get_gpu_context() else {
        return;
    };
    let seqs: Vec<DnaSequence> = ["ACGT", "TGCA", "AACCGGTT"]
        .iter()
        .map(|s| DnaSequence::from_str(s).unwrap())
        .collect();
    let cfg = config();
    let cpu = FmIndex::build_cpu(&seqs, &cfg).unwrap();
    let gpu = pollster::block_on(FmIndex::build(&seqs, &cfg)).unwrap();

    for pattern in &["ACG", "TGC", "AACCGG"] {
        let p = encode_pattern(pattern);
        assert_eq!(
            cpu.count(&p),
            gpu.count(&p),
            "multi-seq count({:?}) mismatch",
            pattern
        );
        let mut cpu_locs = cpu.locate(&p);
        let mut gpu_locs = gpu.locate(&p);
        cpu_locs.sort();
        gpu_locs.sort();
        assert_eq!(
            cpu_locs, gpu_locs,
            "multi-seq locate({:?}) mismatch",
            pattern
        );
    }
}
