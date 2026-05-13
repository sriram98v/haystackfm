//! Crossover benchmark: CPU (find_mems + locate) vs GPU (find_mems_gpu with positions)
//! across reference-count sweep K and query-batch-size sweep B.
//!
//! Each reference is ~2kbp of random DNA. K references → K×2kbp concatenated index.
//!
//! Run:
//!   cargo bench --features gpu --bench mem_positions_bench
//!
//! For large-K runs (K=10_000, K=50_000) which require more RAM/time, add:
//!   cargo bench --features gpu --bench mem_positions_bench -- "positions/K=10000"

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pollster::FutureExt as _;
use webgpu_fmidx::alphabet::DnaSequence;
use webgpu_fmidx::{BidirFmIndex, FmIndexConfig};

const REF_LEN: usize = 2_000;
const QUERY_LEN: usize = 50;
const MIN_LEN: usize = 10;
// Default batch sizes for crossover sweep.
const BATCH_SIZES: &[usize] = &[1, 10, 100, 1_000, 10_000];

fn random_dna(len: usize, seed: u64) -> String {
    let bases = [b'A', b'C', b'G', b'T'];
    // Simple LCG — no external dep required.
    let mut state = seed ^ 0xdeadbeef_cafebabe;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            bases[((state >> 33) & 3) as usize] as char
        })
        .collect()
}

fn build_index(num_refs: usize) -> (BidirFmIndex, Vec<u32>) {
    let config = FmIndexConfig { sa_sample_rate: 4, use_gpu: false };
    let seqs: Vec<DnaSequence> = (0..num_refs)
        .map(|i| DnaSequence::from_str(&random_dna(REF_LEN, i as u64 + 1)).unwrap())
        .collect();
    let idx = BidirFmIndex::build_cpu(&seqs, &config).unwrap();
    let boundaries = idx.seq_boundaries().to_vec();
    (idx, boundaries)
}

fn make_queries(_idx: &BidirFmIndex, n: usize, boundaries: &[u32]) -> Vec<DnaSequence> {
    // Sample queries from the concatenated text at even intervals.
    let total_len = *boundaries.last().unwrap_or(&0) as usize;
    let effective = total_len.saturating_sub(QUERY_LEN).max(1);
    let step = effective / n.max(1);
    // Use the fwd index's BWT implicitly via find_mems — but we need raw text.
    // Instead, rebuild from random_dna with matching seeds.
    let num_refs = boundaries.len();
    (0..n)
        .map(|i| {
            let ref_idx = (i * num_refs) / n.max(1);
            let ref_text = random_dna(REF_LEN, ref_idx as u64 + 1);
            let start = (i * step) % ref_text.len().saturating_sub(QUERY_LEN).max(1);
            let end = (start + QUERY_LEN).min(ref_text.len());
            DnaSequence::from_str(&ref_text[start..end]).unwrap()
        })
        .collect()
}

fn bench_positions(c: &mut Criterion, num_refs: usize) {
    // Build index once outside the timed region.
    let (idx, boundaries) = build_index(num_refs);
    let group_name = format!("positions/K={num_refs}");
    let mut group = c.benchmark_group(&group_name);
    // Fewer samples for large indices to keep bench time reasonable.
    if num_refs >= 1_000 {
        group.sample_size(10);
    }

    for &batch in BATCH_SIZES {
        let queries = make_queries(&idx, batch, &boundaries);

        // CPU: find_mems + locate (positions included).
        group.bench_with_input(
            BenchmarkId::new("cpu", batch),
            &queries,
            |b, qs| {
                b.iter(|| {
                    for q in qs {
                        let _ = idx.find_mems(q.as_slice(), MIN_LEN, true);
                    }
                })
            },
        );

        // GPU: find_mems_gpu with full position resolution.
        group.bench_with_input(
            BenchmarkId::new("gpu", batch),
            &queries,
            |b, qs| {
                b.iter(|| {
                    let _ = idx
                        .find_mems_gpu(qs, MIN_LEN, &boundaries, 1024)
                        .block_on()
                        .unwrap();
                })
            },
        );
    }

    group.finish();
}

fn bench_k1(c: &mut Criterion)    { bench_positions(c, 1); }
fn bench_k10(c: &mut Criterion)   { bench_positions(c, 10); }
fn bench_k100(c: &mut Criterion)  { bench_positions(c, 100); }
fn bench_k1000(c: &mut Criterion) { bench_positions(c, 1_000); }

criterion_group!(benches, bench_k1, bench_k10, bench_k100, bench_k1000);
criterion_main!(benches);
