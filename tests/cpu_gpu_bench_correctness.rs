//! Per-stage correctness tests: verifies that each GPU construction stage
//! produces output identical to its CPU counterpart.
//!
//! These tests complement the full-pipeline equivalence tests in
//! `cpu_gpu_equivalence.rs` by isolating individual stages so that failures
//! point directly to the broken stage.
//!
//! Run with:
//! ```bash
//! cargo test --test cpu_gpu_bench_correctness --features gpu
//! ```

#![cfg(feature = "gpu")]

use webgpu_fmidx::alphabet::{concatenate_sequences, DnaSequence, ALPHABET_SIZE};
use webgpu_fmidx::bwt::{cpu::build_bwt as cpu_bwt, gpu::BwtPipelines};
use webgpu_fmidx::gpu::GpuContext;
use webgpu_fmidx::occ::{cpu::build_occ_table as cpu_occ, gpu::OccPipelines, OccEncoding};
use webgpu_fmidx::suffix_array::{cpu::build_suffix_array as cpu_sa, gpu::SaPipelines};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn get_ctx() -> Option<GpuContext> {
    pollster::block_on(GpuContext::new()).ok()
}

fn encode(dna: &str) -> Vec<u8> {
    let seq = DnaSequence::from_str(dna).unwrap();
    let (text, _) = concatenate_sequences(&[seq]).unwrap();
    text
}

fn random_dna(len: usize) -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let bases = ['A', 'C', 'G', 'T'];
    (0..len).map(|_| bases[rng.random_range(0..4)]).collect()
}

// ── Suffix Array ──────────────────────────────────────────────────────────────

#[test]
fn gpu_sa_matches_cpu_small() {
    let Some(ctx) = get_ctx() else {
        eprintln!("Skipping: GPU not available");
        return;
    };
    let text = encode("ACGTACGTACGTACGT");
    let cpu = cpu_sa(&text);
    let gpu = pollster::block_on(SaPipelines::new(&ctx).build_suffix_array(&ctx, &text));
    assert_eq!(cpu.data, gpu.data, "SA mismatch on small input");
}

#[test]
fn gpu_sa_matches_cpu_1k() {
    let Some(ctx) = get_ctx() else {
        eprintln!("Skipping: GPU not available");
        return;
    };
    let text = encode(&random_dna(1_000));
    let cpu = cpu_sa(&text);
    let gpu = pollster::block_on(SaPipelines::new(&ctx).build_suffix_array(&ctx, &text));
    assert_eq!(cpu.data, gpu.data, "SA mismatch on 1 K input");
}

// ── BWT ───────────────────────────────────────────────────────────────────────

#[test]
fn gpu_bwt_matches_cpu_small() {
    let Some(ctx) = get_ctx() else {
        eprintln!("Skipping: GPU not available");
        return;
    };
    let text = encode("ACGTACGTACGTACGT");
    let sa = cpu_sa(&text);
    let cpu = cpu_bwt(&text, &sa);
    let gpu = pollster::block_on(BwtPipelines::new(&ctx).build_bwt(&ctx, &text, &sa));
    assert_eq!(cpu, gpu, "BWT mismatch on small input");
}

#[test]
fn gpu_bwt_matches_cpu_1k() {
    let Some(ctx) = get_ctx() else {
        eprintln!("Skipping: GPU not available");
        return;
    };
    let text = encode(&random_dna(1_000));
    let sa = cpu_sa(&text);
    let cpu = cpu_bwt(&text, &sa);
    let gpu = pollster::block_on(BwtPipelines::new(&ctx).build_bwt(&ctx, &text, &sa));
    assert_eq!(cpu, gpu, "BWT mismatch on 1 K input");
}

// ── OCC table ────────────────────────────────────────────────────────────────

/// Compare rank(c, i) for every character and every position.
fn assert_occ_rank_equal(
    cpu: &webgpu_fmidx::occ::OccTable,
    gpu: &webgpu_fmidx::occ::OccTable,
    n: u32,
    label: &str,
) {
    // For large n, sample ~200 positions to keep the test fast.
    let step = (n / 200).max(1) as usize;
    for i in (0..=n).step_by(step) {
        for c in 0..ALPHABET_SIZE as u8 {
            let cpu_rank = cpu.rank(c, i);
            let gpu_rank = gpu.rank(c, i);
            assert_eq!(
                cpu_rank, gpu_rank,
                "{label}: rank mismatch at char={c}, pos={i}"
            );
        }
    }
}

#[test]
fn gpu_occ_matches_cpu_small() {
    let Some(ctx) = get_ctx() else {
        eprintln!("Skipping: GPU not available");
        return;
    };
    let text = encode("ACGTACGTACGTACGT");
    let sa = cpu_sa(&text);
    let bwt = cpu_bwt(&text, &sa);
    let cpu = cpu_occ(&bwt, OccEncoding::Bitplane);
    let gpu = pollster::block_on(OccPipelines::new(&ctx).build_occ_table(&ctx, &bwt));
    // Small input: check every position exhaustively
    for i in 0..=bwt.len() as u32 {
        for c in 0..ALPHABET_SIZE as u8 {
            assert_eq!(
                cpu.rank(c, i),
                gpu.rank(c, i),
                "OCC rank mismatch at char={c}, pos={i}"
            );
        }
    }
}

#[test]
fn gpu_occ_matches_cpu_1k() {
    let Some(ctx) = get_ctx() else {
        eprintln!("Skipping: GPU not available");
        return;
    };
    let text = encode(&random_dna(1_000));
    let sa = cpu_sa(&text);
    let bwt = cpu_bwt(&text, &sa);
    let n = bwt.len() as u32;
    let cpu = cpu_occ(&bwt, OccEncoding::Bitplane);
    let gpu = pollster::block_on(OccPipelines::new(&ctx).build_occ_table(&ctx, &bwt));
    assert_occ_rank_equal(&cpu, &gpu, n, "1 K OCC");
}
