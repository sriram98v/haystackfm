//! `to_bytes` / `from_bytes` on a query-time-shaped index (dense SA samples, one-hot occ,
//! no LCP): the single-threaded startup cost a consumer pays before any query runs.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use haystackfm::alphabet::DnaSequence;
use haystackfm::occ::OccEncoding;
use haystackfm::{BidirFmIndex, FmIndexConfig};

mod bench_utils;

fn bench_serialize(c: &mut Criterion) {
    let dna = bench_utils::random_dna(2_000_000);
    let seq = DnaSequence::from_str(&dna).unwrap();
    let config = FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: false,
        occ_encoding: OccEncoding::OneHot,
        build_lcp: false,
        ..Default::default()
    };
    let index = BidirFmIndex::build_cpu(&[seq], &config).unwrap();
    let bytes = index.to_bytes().unwrap();

    let mut group = c.benchmark_group("serialize");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("to_bytes", |b| {
        b.iter(|| black_box(index.to_bytes().unwrap()))
    });
    group.bench_function("from_bytes", |b| {
        b.iter(|| black_box(BidirFmIndex::from_bytes(&bytes).unwrap()))
    });
    group.finish();
}

criterion_group!(benches, bench_serialize);
criterion_main!(benches);
