#[path = "bench_utils.rs"]
mod bench_utils;

use bench_utils::{gpu_available, measure_ms};
use criterion::{BenchmarkId, Criterion};
use pollster::FutureExt as _;
use webgpu_fmidx::alphabet::{encode_char, DnaSequence};
use webgpu_fmidx::fm_index::{FmIndex, FmIndexConfig};

fn random_dna(len: usize, seed: u64) -> String {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let bases = ['A', 'C', 'G', 'T'];
    (0..len).map(|_| bases[rng.random_range(0..4)]).collect()
}

fn encode(s: &str) -> Vec<u8> {
    s.chars().map(|c| encode_char(c).unwrap()).collect()
}

fn build_index(corpus: &str) -> FmIndex {
    let seq = DnaSequence::from_str(corpus).unwrap();
    let config = FmIndexConfig {
        sa_sample_rate: 32,
        use_gpu: false,
        ..Default::default()
    };
    FmIndex::build_cpu(&[seq], &config).unwrap()
}

fn make_patterns(corpus: &str, pat_len: usize, n: usize) -> Vec<Vec<u8>> {
    let step = corpus.len().saturating_sub(pat_len) / n.max(1);
    (0..n)
        .map(|i| encode(&corpus[i * step..i * step + pat_len]))
        .collect()
}

const BATCH_SIZES: &[usize] = &[1, 64, 512, 1024, 2048, 4096, 8192, 16384, 32768];

fn bench_locate_cpu(c: &mut Criterion) {
    let corpus = random_dna(10_000, 42);
    let idx = build_index(&corpus);

    let mut group = c.benchmark_group("locate/cpu");
    for &n in BATCH_SIZES {
        let patterns = make_patterns(&corpus, 8, n);
        group.bench_with_input(BenchmarkId::new("batch", n), &patterns, |b, pats| {
            b.iter(|| {
                for p in pats {
                    let _ = idx.locate(p);
                }
            })
        });
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_locate_gpu(c: &mut Criterion) {
    let corpus = random_dna(10_000, 42);
    let idx = build_index(&corpus);

    let mut group = c.benchmark_group("locate/gpu");
    for &n in BATCH_SIZES {
        let patterns = make_patterns(&corpus, 8, n);
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
        group.bench_with_input(BenchmarkId::new("batch", n), &pattern_refs, |b, pats| {
            b.iter(|| {
                let _ = idx.locate_gpu(pats).block_on().unwrap();
            })
        });
    }
    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_locate_gpu(_c: &mut Criterion) {}

// ── Speedup summary ───────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn print_locate_speedup_table() {
    const WARMUP: usize = 1;
    const ITERS: usize = 3;

    let corpus = random_dna(10_000, 42);
    let idx = build_index(&corpus);

    let sep = "─".repeat(74);
    let dbl = "═".repeat(74);
    eprintln!("\n{dbl}");
    eprintln!("  Locate CPU vs GPU Speedup  (warmup={WARMUP}, iters={ITERS})");
    eprintln!("{dbl}");
    eprintln!(
        "  {:<10} {:>10}  {:>10}  {:>8}",
        "Batch", "CPU (ms)", "GPU (ms)", "Speedup"
    );
    eprintln!("  {sep}");

    let mut prev_speedup: f64 = 0.0;
    for &n in BATCH_SIZES {
        let patterns = make_patterns(&corpus, 8, n);
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();

        let cpu_ms = measure_ms(
            || {
                for p in &patterns {
                    let _ = idx.locate(p);
                }
            },
            WARMUP,
            ITERS,
        );
        let gpu_ms = measure_ms(
            || {
                let _ = idx.locate_gpu(&pattern_refs).block_on().unwrap();
            },
            WARMUP,
            ITERS,
        );

        let speedup = cpu_ms / gpu_ms;
        let marker = if speedup >= 1.0 { "▲" } else { "▼" };
        let crossover = if speedup >= 1.0 && prev_speedup < 1.0 {
            "  ← crossover"
        } else {
            ""
        };
        eprintln!(
            "  {:<10} {:>10.3}  {:>10.3}  {:>6.2}x {marker}{crossover}",
            n, cpu_ms, gpu_ms, speedup
        );
        prev_speedup = speedup;
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
            print_locate_speedup_table();
        } else {
            eprintln!("Note: GPU not available – running CPU-only benchmarks.");
        }
    }

    let mut c = Criterion::default().configure_from_args();
    bench_locate_cpu(&mut c);
    bench_locate_gpu(&mut c);
    c.final_summary();
}
