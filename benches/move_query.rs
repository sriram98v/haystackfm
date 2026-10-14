//! Plain vs run-length vs move-structure indexes: `count`, `locate`, cursor extension and
//! located SMEM / MEM finding on a random reference (r ≈ n) and a repetitive collection
//! (r ≪ n). Inputs are seeded.

use criterion::{criterion_group, criterion_main, Criterion};
use haystackfm::alphabet::DnaSequence;
use haystackfm::{
    BMoveIndex, BidirFmIndex, BidirRlFmIndex, FmIndex, FmIndexConfig, Mem, MoveFmIndex,
    MoveFmIndexConfig, RlFmIndex, RlFmIndexConfig,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BASES: [u8; 4] = [1, 2, 3, 4];

fn random_codes(rng: &mut StdRng, len: usize) -> Vec<u8> {
    (0..len).map(|_| BASES[rng.random_range(0..4)]).collect()
}

/// 400 kbp random, and 40 copies of a 10 kbp genome with 0.1% SNPs.
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

/// `count` reads of `len` bases with up to three substitutions each.
fn reads(seqs: &[DnaSequence], len: usize, count: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(0x4ead);
    patterns(seqs, len, count)
        .into_iter()
        .map(|mut read| {
            for _ in 0..rng.random_range(0..4) {
                let at = rng.random_range(0..len);
                read[at] = BASES[rng.random_range(0..4)];
            }
            read
        })
        .collect()
}

/// Total positions over `reads`.
fn positions(reads: &[Vec<u8>], find: impl Fn(&[u8]) -> Vec<Mem>) -> usize {
    reads
        .iter()
        .flat_map(|r| find(r))
        .map(|m| m.positions.len())
        .sum()
}

fn fm_config() -> FmIndexConfig {
    FmIndexConfig {
        use_gpu: false,
        ..Default::default()
    }
}

fn bench_move(c: &mut Criterion) {
    for (name, seqs) in inputs() {
        let fm = FmIndex::build_cpu(&seqs, &fm_config()).unwrap();
        let rl = RlFmIndex::build_cpu(&seqs, &RlFmIndexConfig::default()).unwrap();
        let mv = MoveFmIndex::build_cpu(&seqs, &MoveFmIndexConfig::default()).unwrap();
        let n = fm.text_len() as f64;
        eprintln!(
            "{name}: n = {}, r = {}, bytes/base: fm occ {:.3} + samples, rl {:.3}, move {:.3} (LF {:.3})",
            fm.text_len(),
            rl.bwt().num_runs(),
            fm.occ().heap_size_bytes() as f64 / n,
            rl.heap_size_bytes() as f64 / n,
            mv.heap_size_bytes() as f64 / n,
            mv.lf().heap_size_bytes() as f64 / n,
        );

        let mut group = c.benchmark_group(format!("move_count/{name}"));
        let pats = patterns(&seqs, 32, 64);
        group.bench_function("fm", |b| {
            b.iter(|| pats.iter().map(|p| fm.count(p)).sum::<u32>())
        });
        group.bench_function("rl", |b| {
            b.iter(|| pats.iter().map(|p| rl.count(p)).sum::<u32>())
        });
        group.bench_function("move", |b| {
            b.iter(|| pats.iter().map(|p| mv.count(p)).sum::<u32>())
        });
        group.finish();

        let mut group = c.benchmark_group(format!("move_locate/{name}"));
        let pats = patterns(&seqs, 16, 16);
        group.bench_function("fm", |b| {
            b.iter(|| pats.iter().map(|p| fm.locate(p).len()).sum::<usize>())
        });
        group.bench_function("rl", |b| {
            b.iter(|| pats.iter().map(|p| rl.locate(p).len()).sum::<usize>())
        });
        group.bench_function("move", |b| {
            b.iter(|| pats.iter().map(|p| mv.locate(p).len()).sum::<usize>())
        });
        group.finish();

        let fm_bidir = BidirFmIndex::build_cpu(&seqs, &fm_config()).unwrap();
        let rl_bidir = BidirRlFmIndex::build_cpu(&seqs, &RlFmIndexConfig::default()).unwrap();
        let bm = BMoveIndex::build_cpu(&seqs, &MoveFmIndexConfig::default()).unwrap();
        let pats = patterns(&seqs, 24, 64);
        let mut group = c.benchmark_group(format!("move_extend_right/{name}"));
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
        group.bench_function("bmove", |b| {
            b.iter(|| {
                pats.iter()
                    .filter_map(|p| {
                        p.iter()
                            .try_fold(bm.full_cursor(), |cur, &c| bm.extend_right(&cur, c))
                    })
                    .count()
            })
        });
        group.finish();

        let rs = reads(&seqs, 150, 16);
        for (op, mems) in [("smems", false), ("mems", true)] {
            let mut group = c.benchmark_group(format!("move_{op}/{name}"));
            let run = |find: &dyn Fn(&[u8]) -> Vec<Mem>| positions(&rs, find);
            group.bench_function("fm", |b| {
                b.iter(|| {
                    run(&|r| match mems {
                        true => fm_bidir.find_mems(r, 19, true),
                        false => fm_bidir.find_smems(r, 19, true),
                    })
                })
            });
            group.bench_function("rl", |b| {
                b.iter(|| {
                    run(&|r| match mems {
                        true => rl_bidir.find_mems(r, 19, true),
                        false => rl_bidir.find_smems(r, 19, true),
                    })
                })
            });
            group.bench_function("bmove", |b| {
                b.iter(|| {
                    run(&|r| match mems {
                        true => bm.find_mems(r, 19, true),
                        false => bm.find_smems(r, 19, true),
                    })
                })
            });
            group.finish();
        }
    }
}

criterion_group!(benches, bench_move);
criterion_main!(benches);
