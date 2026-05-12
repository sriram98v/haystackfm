use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pollster::FutureExt as _;
use webgpu_fmidx::alphabet::{encode_char, DnaSequence};
use webgpu_fmidx::fm_index::{FmIndex, FmIndexConfig};

fn random_dna(len: usize, seed: u64) -> String {
    use rand::SeedableRng;
    use rand::Rng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let bases = ['A', 'C', 'G', 'T'];
    (0..len).map(|_| bases[rng.random_range(0..4)]).collect()
}

fn encode(s: &str) -> Vec<u8> {
    s.chars().map(|c| encode_char(c).unwrap()).collect()
}

fn build_index(corpus: &str) -> FmIndex {
    let seq = DnaSequence::from_str(corpus).unwrap();
    let config = FmIndexConfig { sa_sample_rate: 32, use_gpu: false };
    FmIndex::build_cpu(&[seq], &config).unwrap()
}

fn make_patterns(corpus: &str, pat_len: usize, n: usize) -> Vec<Vec<u8>> {
    let step = corpus.len().saturating_sub(pat_len) / n.max(1);
    (0..n)
        .map(|i| encode(&corpus[i * step..i * step + pat_len]))
        .collect()
}

fn bench_locate_cpu(c: &mut Criterion) {
    let corpus = random_dna(10_000, 42);
    let idx = build_index(&corpus);

    let mut group = c.benchmark_group("locate/cpu");
    for &n in &[1usize, 8, 64, 256] {
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
    for &n in &[1usize, 8, 64, 256] {
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

criterion_group!(benches, bench_locate_cpu, bench_locate_gpu);
criterion_main!(benches);
