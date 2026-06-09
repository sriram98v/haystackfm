use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pollster::FutureExt as _;
use webgpu_fmidx::alphabet::DnaSequence;
use webgpu_fmidx::{BidirFmIndex, FmIndexConfig};

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

criterion_group!(
    benches,
    bench_smem_cpu,
    bench_smem_gpu,
    bench_mem_cpu,
    bench_mem_gpu
);
criterion_main!(benches);
