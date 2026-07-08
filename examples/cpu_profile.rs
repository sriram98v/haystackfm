//! CPU profiling harness for locate / find_smems / find_mems hot-path analysis.
//!
//! Build:  cargo build --release --example cpu_profile
//! Perf:   perf record -g --call-graph dwarf ./target/release/examples/cpu_profile <mode> <ref_len> <n_queries> <query_len>
//!         perf report --stdio
//! Modes:  locate | smems | mems | all

use std::str::FromStr;
use std::time::Instant;

use webgpu_fmidx::alphabet::DnaSequence;
use webgpu_fmidx::fm_index::bidir_index::BidirFmIndex;
use webgpu_fmidx::fm_index::{FmIndex, FmIndexConfig};

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
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let ref_len: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    let n_queries: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let query_len: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(100);

    let mut rng = Rng(0x9E3779B97F4A7C15);
    let reference = rng.dna(ref_len);
    let ref_seq = DnaSequence::from_str(&reference).unwrap();

    let cfg = FmIndexConfig {
        sa_sample_rate: 32,
        use_gpu: false,
        ..Default::default()
    };

    eprintln!("building indices over {ref_len} bp ...");
    let fwd = FmIndex::build_cpu(&[ref_seq.clone()], &cfg).unwrap();
    let bidir = BidirFmIndex::build_cpu(&[ref_seq], &cfg).unwrap();

    // Queries drawn as random substrings of the reference so they actually match
    // (exercises the deep LF-walks / long SMEMs that dominate real workloads).
    let ref_bytes = reference.as_bytes();
    let queries: Vec<Vec<u8>> = (0..n_queries)
        .map(|_| {
            let start = (rng.next() as usize) % (ref_len - query_len);
            // encode to alphabet values 1..4
            ref_bytes[start..start + query_len]
                .iter()
                .map(|&b| match b {
                    b'A' => 1u8,
                    b'C' => 2,
                    b'G' => 3,
                    _ => 4,
                })
                .collect()
        })
        .collect();
    // ASCII form for locate (backward_search takes encoded already via count/locate? use encoded)
    let run_locate = |label: &str| {
        let t = Instant::now();
        let mut total = 0usize;
        for q in &queries {
            total += fwd.locate_positions(q).len();
        }
        eprintln!("{label}: {} hits in {:?}", total, t.elapsed());
    };
    let run_locate_batch = |label: &str| {
        let refs: Vec<&[u8]> = queries.iter().map(|q| q.as_slice()).collect();
        let t = Instant::now();
        let res = fwd.locate_positions_many(&refs);
        let total: usize = res.iter().map(|v| v.len()).sum();
        eprintln!("{label}: {} hits in {:?}", total, t.elapsed());
    };
    let run_smems = |label: &str| {
        let t = Instant::now();
        let mut total = 0usize;
        for q in &queries {
            total += bidir.find_smems(q, 20, false).len();
        }
        eprintln!("{label}: {} smems in {:?}", total, t.elapsed());
    };
    let run_mems = |label: &str| {
        let t = Instant::now();
        let mut total = 0usize;
        for q in &queries {
            total += bidir.find_mems(q, 20, false).len();
        }
        eprintln!("{label}: {} mems in {:?}", total, t.elapsed());
    };

    match mode {
        "locate" => run_locate("locate"),
        "locate_batch" => run_locate_batch("locate_batch"),
        "locate_both" => {
            run_locate("locate");
            run_locate_batch("locate_batch");
        }
        "smems" => run_smems("smems"),
        "mems" => run_mems("mems"),
        _ => {
            run_locate("locate");
            run_smems("smems");
            run_mems("mems");
        }
    }
}
