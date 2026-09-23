//! Startup-cost harness: how long `BidirFmIndex::to_bytes` / `from_bytes` take on an index
//! shaped like a real query-time deployment (dense SA samples, one-hot occ, no LCP).
//!
//! Run:   cargo run --release --example serde_profile [ref_len_bp] [from_bytes_reps]
//! Default: 10 Mbp, 5 reps. Reports blob size, per-call time and throughput.

use std::time::Instant;

use haystackfm::alphabet::DnaSequence;
use haystackfm::occ::OccEncoding;
use haystackfm::{BidirFmIndex, FmIndexConfig};

// Tiny deterministic xorshift RNG so runs are reproducible without extra deps.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn dna(&mut self, len: usize) -> String {
        const B: [u8; 4] = *b"ACGT";
        (0..len)
            .map(|_| B[(self.next() % 4) as usize] as char)
            .collect()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ref_len: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    let mut rng = Rng(0x9E3779B97F4A7C15);
    let reference = rng.dna(ref_len);
    let ref_seq = DnaSequence::from_str(&reference).unwrap();

    let cfg = FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: false,
        occ_encoding: OccEncoding::OneHot,
        build_lcp: false,
        ..Default::default()
    };

    eprintln!("building bidirectional index over {ref_len} bp ...");
    let t = Instant::now();
    let index = BidirFmIndex::build_cpu(&[ref_seq], &cfg).unwrap();
    eprintln!("build: {:?}", t.elapsed());

    let t = Instant::now();
    let bytes = index.to_bytes().unwrap();
    let ser = t.elapsed();
    let mb = bytes.len() as f64 / 1e6;
    eprintln!(
        "to_bytes: {:.1} MB ({:.2} B/base) in {:?} ({:.0} MB/s)",
        mb,
        bytes.len() as f64 / ref_len as f64,
        ser,
        mb / ser.as_secs_f64()
    );

    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let restored = BidirFmIndex::from_bytes(&bytes).unwrap();
        times.push(t.elapsed());
        assert_eq!(restored.text_len(), index.text_len());
    }
    times.sort();
    let min = times[0];
    let median = times[times.len() / 2];
    eprintln!(
        "from_bytes x{reps}: min {:?} ({:.0} MB/s, {:.2} ns/byte), median {:?}",
        min,
        mb / min.as_secs_f64(),
        min.as_nanos() as f64 / bytes.len() as f64,
        median
    );
}
