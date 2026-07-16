use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use haystackfm::alphabet::DnaSequence;
use haystackfm::fm_index::{FmIndex, FmIndexConfig};

fn random_dna(len: usize) -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let bases = ['A', 'C', 'G', 'T'];
    (0..len).map(|_| bases[rng.random_range(0..4)]).collect()
}

fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        use haystackfm::gpu::GpuContext;
        pollster::block_on(GpuContext::new()).is_ok()
    }
    #[cfg(not(feature = "gpu"))]
    false
}

// ---- CPU construction benchmarks ----

fn bench_build_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_cpu");
    for size in [1_000usize, 10_000, 100_000, 1_000_000] {
        let dna = random_dna(size);
        let seq = DnaSequence::from_str(&dna).unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 32,
            use_gpu: false,
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &seq, |b, s| {
            b.iter(|| FmIndex::build_cpu(&[s.clone()], &config).unwrap())
        });
    }
    group.finish();
}

// ---- GPU construction benchmarks ----

fn bench_build_gpu(c: &mut Criterion) {
    if !gpu_available() {
        eprintln!("Skipping GPU benchmarks: no GPU available");
        return;
    }

    let mut group = c.benchmark_group("build_gpu");
    for size in [10_000usize, 100_000, 500_000] {
        let dna = random_dna(size);
        let seq = DnaSequence::from_str(&dna).unwrap();
        #[cfg(feature = "gpu")]
        {
            let config = FmIndexConfig {
                sa_sample_rate: 32,
                use_gpu: true,
            };
            group.bench_with_input(BenchmarkId::from_parameter(size), &seq, |b, s| {
                b.iter(|| pollster::block_on(FmIndex::build(&[s.clone()], &config)).unwrap())
            });
        }
        let _ = seq; // suppress unused warning when gpu feature is off
    }
    group.finish();
}

// ---- Side-by-side CPU vs GPU at 100K ----

fn bench_cpu_vs_gpu(c: &mut Criterion) {
    let size = 100_000;
    let dna = random_dna(size);
    let seq = DnaSequence::from_str(&dna).unwrap();

    let mut group = c.benchmark_group("cpu_vs_gpu_100k");

    let cpu_config = FmIndexConfig {
        sa_sample_rate: 32,
        use_gpu: false,
    };
    group.bench_with_input("cpu", &seq, |b, s| {
        b.iter(|| FmIndex::build_cpu(&[s.clone()], &cpu_config).unwrap())
    });

    if gpu_available() {
        #[cfg(feature = "gpu")]
        {
            let gpu_config = FmIndexConfig {
                sa_sample_rate: 32,
                use_gpu: true,
            };
            group.bench_with_input("gpu", &seq, |b, s| {
                b.iter(|| pollster::block_on(FmIndex::build(&[s.clone()], &gpu_config)).unwrap())
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_build_cpu, bench_build_gpu, bench_cpu_vs_gpu);
criterion_main!(benches);
