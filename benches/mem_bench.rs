#[path = "bench_utils.rs"]
mod bench_utils;

use bench_utils::{gpu_available, measure_ms};
use criterion::{BenchmarkId, Criterion};
use pollster::FutureExt as _;
use haystackfm::alphabet::DnaSequence;
use haystackfm::{BidirFmIndex, FmIndexConfig};

fn random_dna(len: usize, seed: u64) -> String {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let bases = ['A', 'C', 'G', 'T'];
    (0..len).map(|_| bases[rng.random_range(0..4)]).collect()
}

fn build_bidir(corpus: &str) -> BidirFmIndex {
    let seq = DnaSequence::from_str(corpus).unwrap();
    let config = FmIndexConfig {
        sa_sample_rate: 32,
        use_gpu: false,
    };
    BidirFmIndex::build_cpu(&[seq], &config).unwrap()
}

fn make_queries(corpus: &str, qlen: usize, n: usize) -> Vec<DnaSequence> {
    let step = corpus.len().saturating_sub(qlen) / n.max(1);
    (0..n)
        .map(|i| DnaSequence::from_str(&corpus[i * step..i * step + qlen]).unwrap())
        .collect()
}

const MIN_LEN: usize = 10;

fn bench_smem_cpu(c: &mut Criterion) {
    let corpus = random_dna(10_000, 42);
    let idx = build_bidir(&corpus);

    let mut group = c.benchmark_group("smem/cpu");
    for &n in &[1usize, 8, 64] {
        let queries = make_queries(&corpus, 50, n);
        group.bench_with_input(BenchmarkId::new("batch", n), &queries, |b, qs| {
            b.iter(|| {
                for q in qs {
                    let _ = idx.find_smems(q.as_slice(), MIN_LEN, false);
                }
            })
        });
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_smem_gpu(c: &mut Criterion) {
    let corpus = random_dna(10_000, 42);
    let idx = build_bidir(&corpus);

    let mut group = c.benchmark_group("smem/gpu");
    for &n in &[1usize, 8, 64] {
        let queries = make_queries(&corpus, 50, n);
        group.bench_with_input(BenchmarkId::new("batch", n), &queries, |b, qs| {
            b.iter(|| {
                let _ = idx
                    .find_smems_gpu(qs, MIN_LEN, &[], 1024)
                    .block_on()
                    .unwrap();
            })
        });
    }
    group.finish();
}

fn bench_mem_cpu(c: &mut Criterion) {
    let corpus = random_dna(10_000, 42);
    let idx = build_bidir(&corpus);

    let mut group = c.benchmark_group("mem/cpu");
    for &n in &[1usize, 8, 64] {
        let queries = make_queries(&corpus, 50, n);
        group.bench_with_input(BenchmarkId::new("batch", n), &queries, |b, qs| {
            b.iter(|| {
                for q in qs {
                    let _ = idx.find_mems(q.as_slice(), MIN_LEN, false);
                }
            })
        });
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_mem_gpu(c: &mut Criterion) {
    let corpus = random_dna(10_000, 42);
    let idx = build_bidir(&corpus);

    let mut group = c.benchmark_group("mem/gpu");
    for &n in &[1usize, 8, 64] {
        let queries = make_queries(&corpus, 50, n);
        group.bench_with_input(BenchmarkId::new("batch", n), &queries, |b, qs| {
            b.iter(|| {
                let _ = idx
                    .find_mems_gpu(qs, MIN_LEN, &[], 1024)
                    .block_on()
                    .unwrap();
            })
        });
    }
    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_smem_gpu(_c: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_mem_gpu(_c: &mut Criterion) {}

// ── Speedup summary ───────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn print_mem_speedup_table() {
    const BATCH_SIZES: &[usize] = &[1, 8, 64];
    const WARMUP: usize = 1;
    const ITERS: usize = 3;

    let corpus = random_dna(10_000, 42);
    let idx = build_bidir(&corpus);

    let sep = "─".repeat(66);
    let dbl = "═".repeat(66);
    eprintln!("\n{dbl}");
    eprintln!("  MEM/SMEM CPU vs GPU Speedup  (warmup={WARMUP}, iters={ITERS})");
    eprintln!("{dbl}");
    eprintln!(
        "  {:<12} {:>6} {:>10}  {:>10}  {:>8}",
        "Stage", "Batch", "CPU (ms)", "GPU (ms)", "Speedup"
    );
    eprintln!("  {sep}");

    for &n in BATCH_SIZES {
        let queries = make_queries(&corpus, 50, n);

        // SMEM
        let cpu_ms = measure_ms(
            || {
                for q in &queries {
                    let _ = idx.find_smems(q.as_slice(), MIN_LEN, false);
                }
            },
            WARMUP,
            ITERS,
        );
        let gpu_ms = measure_ms(
            || {
                let _ = idx
                    .find_smems_gpu(&queries, MIN_LEN, &[], 1024)
                    .block_on()
                    .unwrap();
            },
            WARMUP,
            ITERS,
        );
        let speedup = cpu_ms / gpu_ms;
        let marker = if speedup >= 1.0 { "▲" } else { "▼" };
        eprintln!(
            "  {:<12} {:>6} {:>10.3}  {:>10.3}  {:>6.2}x {marker}",
            "SMEM", n, cpu_ms, gpu_ms, speedup
        );

        // MEM
        let cpu_ms = measure_ms(
            || {
                for q in &queries {
                    let _ = idx.find_mems(q.as_slice(), MIN_LEN, false);
                }
            },
            WARMUP,
            ITERS,
        );
        let gpu_ms = measure_ms(
            || {
                let _ = idx
                    .find_mems_gpu(&queries, MIN_LEN, &[], 1024)
                    .block_on()
                    .unwrap();
            },
            WARMUP,
            ITERS,
        );
        let speedup = cpu_ms / gpu_ms;
        let marker = if speedup >= 1.0 { "▲" } else { "▼" };
        eprintln!(
            "  {:<12} {:>6} {:>10.3}  {:>10.3}  {:>6.2}x {marker}",
            "MEM", n, cpu_ms, gpu_ms, speedup
        );
    }
    eprintln!("  {sep}");
    eprintln!("{dbl}");
    eprintln!("  ▲ = GPU faster   ▼ = CPU faster");
    eprintln!("{dbl}\n");
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    #[cfg(feature = "gpu")]
    {
        if gpu_available() {
            print_mem_speedup_table();
        } else {
            eprintln!("Note: GPU not available – running CPU-only benchmarks.");
        }
    }

    let mut c = Criterion::default().configure_from_args();
    bench_smem_cpu(&mut c);
    bench_smem_gpu(&mut c);
    bench_mem_cpu(&mut c);
    bench_mem_gpu(&mut c);
    c.final_summary();
}
