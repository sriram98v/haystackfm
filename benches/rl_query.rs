//! Run-length vs plain FM-index: `count`, `locate` and cursor extension on a random
//! reference (r ≈ n, the run-length worst case) and a repetitive collection (r ≪ n).
//! Inputs are seeded so runs are comparable.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use haystackfm::alphabet::DnaSequence;
use haystackfm::{
    BidirFmIndex, BidirRlFmIndex, FmIndex, FmIndexConfig, RlFmIndex, RlFmIndexConfig,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BASES: [u8; 4] = [1, 2, 3, 4];

fn random_codes(rng: &mut StdRng, len: usize) -> Vec<u8> {
    (0..len).map(|_| BASES[rng.random_range(0..4)]).collect()
}

/// `(name, references)`: 400 kbp random, and 40 copies of a 10 kbp genome with 0.1% SNPs.
fn inputs() -> Vec<(&'static str, Vec<DnaSequence>)> {
    let mut rng = StdRng::seed_from_u64(0xbe7c);
    let random = vec![DnaSequence::from_encoded(random_codes(&mut rng, 400_000))];
    let base = random_codes(&mut rng, 10_000);
    let repetitive = (0..40)
        .map(|_| {
            let mut copy = base.clone();
            for _ in 0..10 {
                let at = rng.random_range(0..copy.len());
                copy[at] = BASES[rng.random_range(0..4)];
            }
            DnaSequence::from_encoded(copy)
        })
        .collect();
    vec![("random", random), ("repetitive", repetitive)]
}

/// Patterns sampled from the first reference, so every one occurs.
fn patterns(seqs: &[DnaSequence], len: usize, count: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(len as u64);
    let text = seqs[0].as_slice();
    (0..count)
        .map(|_| {
            let at = rng.random_range(0..text.len() - len);
            text[at..at + len].to_vec()
        })
        .collect()
}

fn bench_rl(c: &mut Criterion) {
    for (name, seqs) in inputs() {
        let fm = FmIndex::build_cpu(
            &seqs,
            &FmIndexConfig {
                use_gpu: false,
                ..Default::default()
            },
        )
        .unwrap();
        let rl = RlFmIndex::build_cpu(&seqs, &RlFmIndexConfig::default()).unwrap();
        eprintln!(
            "{name}: n = {}, r = {}, n/r = {:.1}, occ = {} B, RlBwt = {} B",
            rl.text_len(),
            rl.bwt().num_runs(),
            rl.bwt().run_ratio(),
            fm.occ().heap_size_bytes(),
            rl.bwt().heap_size_bytes()
        );

        let mut group = c.benchmark_group(format!("rl_count/{name}"));
        for len in [8, 32] {
            let pats = patterns(&seqs, len, 64);
            group.bench_with_input(BenchmarkId::new("fm", len), &pats, |b, pats| {
                b.iter(|| pats.iter().map(|p| fm.count(p)).sum::<u32>())
            });
            group.bench_with_input(BenchmarkId::new("rl", len), &pats, |b, pats| {
                b.iter(|| pats.iter().map(|p| rl.count(p)).sum::<u32>())
            });
        }
        group.finish();

        let mut group = c.benchmark_group(format!("rl_locate/{name}"));
        let pats = patterns(&seqs, 16, 16);
        group.bench_function("fm", |b| {
            b.iter(|| pats.iter().map(|p| fm.locate(p).len()).sum::<usize>())
        });
        group.bench_function("rl", |b| {
            b.iter(|| pats.iter().map(|p| rl.locate(p).len()).sum::<usize>())
        });
        group.finish();

        let fm_bidir = BidirFmIndex::build_cpu(
            &seqs,
            &FmIndexConfig {
                use_gpu: false,
                ..Default::default()
            },
        )
        .unwrap();
        let rl_bidir = BidirRlFmIndex::build_cpu(&seqs, &RlFmIndexConfig::default()).unwrap();
        let pats = patterns(&seqs, 24, 64);
        let mut group = c.benchmark_group(format!("rl_extend_right/{name}"));
        group.bench_function("fm", |b| {
            b.iter(|| {
                pats.iter()
                    .filter_map(|p| {
                        p.iter().try_fold(fm_bidir.full_interval(), |iv, &c| {
                            fm_bidir.extend_right(iv, c)
                        })
                    })
                    .count()
            })
        });
        group.bench_function("rl", |b| {
            b.iter(|| {
                pats.iter()
                    .filter_map(|p| {
                        p.iter().try_fold(rl_bidir.full_interval(), |iv, &c| {
                            rl_bidir.extend_right(iv, c)
                        })
                    })
                    .count()
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench_rl);
criterion_main!(benches);
