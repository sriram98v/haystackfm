use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use haystackfm::alphabet::{encode_char, DnaSequence};
use haystackfm::fm_index::{FmIndex, FmIndexConfig};
use haystackfm::{BidirFmIndex, BidirInterval};

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
        ..Default::default()
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
        ..Default::default()
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

/// Random DNA with `runs` injected IUPAC wildcard runs (length 1..=7), mirroring a
/// reference database that carries ambiguity codes.
fn random_dna_with_wildcards(len: usize, runs: usize) -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let mut text: Vec<char> = random_dna(len).chars().collect();
    let wild = ['N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V'];
    for _ in 0..runs {
        let start = rng.random_range(0..len);
        let run_len = rng.random_range(1..=7);
        for k in 0..run_len {
            if start + k < len {
                text[start + k] = wild[rng.random_range(0..wild.len())];
            }
        }
    }
    text.into_iter().collect()
}

/// Cost of asking "does any occurrence continue into a wildcard" per cursor step:
/// the class count (`count_wild_right`) vs. today's fan-out over all 11 wildcard codes
/// vs. materialising every child. The first must stay flat regardless of how many wildcard
/// codes exist in the alphabet.
fn bench_wild_counts(c: &mut Criterion) {
    let dna = random_dna_with_wildcards(100_000, 500);
    let seq = DnaSequence::from_str(&dna).unwrap();
    let config = FmIndexConfig {
        sa_sample_rate: 32,
        use_gpu: false,
        ..Default::default()
    };
    let idx = BidirFmIndex::build_cpu(&[seq], &config).unwrap();

    // 64 cursors: random 12-mers of the text walked by extend_right (skip any that die on
    // a wildcard-containing seed).
    let mut cursors: Vec<BidirInterval> = Vec::new();
    let mut start = 0usize;
    while cursors.len() < 64 && start + 12 <= dna.len() {
        let mut iv = idx.full_interval();
        let mut alive = true;
        for ch in dna[start..start + 12].chars() {
            match idx.extend_right(iv, encode_char(ch).unwrap()) {
                Some(next) => iv = next,
                None => {
                    alive = false;
                    break;
                }
            }
        }
        if alive {
            cursors.push(iv);
        }
        start += 1_543;
    }

    let mut group = c.benchmark_group("wild_counts");
    group.bench_function("count_wild_right", |b| {
        b.iter(|| {
            cursors
                .iter()
                .map(|iv| idx.count_wild_right(iv))
                .sum::<u32>()
        })
    });
    group.bench_function("fanout_11_extend_right", |b| {
        b.iter(|| {
            cursors
                .iter()
                .map(|&iv| {
                    (5u8..16)
                        .filter_map(|code| idx.extend_right(iv, code))
                        .map(|child| child.size())
                        .sum::<u32>()
                })
                .sum::<u32>()
        })
    });
    group.bench_function("children_right", |b| {
        b.iter(|| {
            cursors
                .iter()
                .map(|iv| {
                    idx.children_right(iv)[5..]
                        .iter()
                        .flatten()
                        .map(|child| child.size())
                        .sum::<u32>()
                })
                .sum::<u32>()
        })
    });
    group.bench_function("extend_right_single_base", |b| {
        b.iter(|| {
            cursors
                .iter()
                .filter_map(|&iv| idx.extend_right(iv, haystackfm::alphabet::A))
                .map(|child| child.size())
                .sum::<u32>()
        })
    });
    group.finish();
}

criterion_group!(benches, bench_count, bench_locate, bench_wild_counts);
criterion_main!(benches);
