use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use haystackfm::alphabet::{encode_char, DnaSequence};
use haystackfm::fm_index::{FmIndex, FmIndexConfig};

fn random_dna(len: usize) -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let bases = ['A', 'C', 'G', 'T'];
    (0..len).map(|_| bases[rng.random_range(0..4)]).collect()
}

fn bench_count(c: &mut Criterion) {
    let dna = random_dna(100_000);
    let seq = DnaSequence::from_str(&dna).unwrap();
    let config = FmIndexConfig {
        sa_sample_rate: 32,
        use_gpu: false,
    };
    let idx = FmIndex::build_cpu(&[seq], &config).unwrap();

    let mut group = c.benchmark_group("count");
    for pattern_len in [4, 8, 16, 32] {
        let pattern: Vec<u8> = dna[..pattern_len]
            .chars()
            .map(|c| encode_char(c).unwrap())
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(pattern_len),
            &pattern,
            |b, p| b.iter(|| idx.count(p)),
        );
    }
    group.finish();
}

fn bench_locate(c: &mut Criterion) {
    let dna = random_dna(100_000);
    let seq = DnaSequence::from_str(&dna).unwrap();
    let config = FmIndexConfig {
        sa_sample_rate: 32,
        use_gpu: false,
    };
    let idx = FmIndex::build_cpu(&[seq], &config).unwrap();

    let mut group = c.benchmark_group("locate");
    for pattern_len in [4, 8, 16] {
        let pattern: Vec<u8> = dna[..pattern_len]
            .chars()
            .map(|c| encode_char(c).unwrap())
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(pattern_len),
            &pattern,
            |b, p| b.iter(|| idx.locate(p)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_count, bench_locate);
criterion_main!(benches);
