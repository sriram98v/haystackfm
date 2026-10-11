//! Compare the uncompressed and run-length bidirectional indexes on a FASTA file: build a
//! `BidirFmIndex`, convert it to a `BidirRlFmIndex` (same suffix arrays, so identical query
//! results), and report serialized and resident sizes. Then sanity-check that sampled reads
//! give the same SMEMs on both, and that sequences extract identically, with timings.
//!
//! ```text
//! cargo run --release --example size_compare -- <fasta> [--max-bases N] [--threads T]
//!     [--sample-rate S] [--reads R]
//! ```
//!
//! `--max-bases` keeps whole sequences until the cap is reached (at least one).

use std::io::{BufRead, BufReader};
use std::time::Instant;

use haystackfm::{BidirFmIndex, BidirRlFmIndex, DnaSequence, FmIndexConfig, SeqId};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

struct Args {
    fasta: String,
    max_bases: u64,
    threads: u16,
    sample_rate: u32,
    reads: usize,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let mut args = Args {
        fasta: String::new(),
        max_bases: u64::MAX,
        threads: 1,
        sample_rate: 32,
        reads: 200,
    };
    while let Some(a) = it.next() {
        let mut value = || it.next().expect("flag needs a value");
        match a.as_str() {
            "--max-bases" => args.max_bases = value().parse().expect("--max-bases"),
            "--threads" => args.threads = value().parse().expect("--threads"),
            "--sample-rate" => args.sample_rate = value().parse().expect("--sample-rate"),
            "--reads" => args.reads = value().parse().expect("--reads"),
            _ => args.fasta = a,
        }
    }
    assert!(
        !args.fasta.is_empty(),
        "usage: size_compare <fasta> [flags]"
    );
    args
}

/// Whole records until `max_bases` is reached. Bases are upper-cased; anything outside the
/// IUPAC alphabet aborts with the offending record's header.
fn read_fasta(path: &str, max_bases: u64) -> Vec<DnaSequence> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut out = Vec::new();
    let mut total = 0u64;
    let mut header: Option<String> = None;
    let mut bases = String::new();
    for line in BufReader::with_capacity(1 << 20, file).lines() {
        let line = line.expect("read line");
        if let Some(h) = line.strip_prefix('>') {
            if let Some(prev) = header.take() {
                total += bases.len() as u64;
                out.push(record(&prev, &bases));
                bases.clear();
                if total >= max_bases {
                    return out;
                }
            }
            header = Some(h.split_whitespace().next().unwrap_or("").to_string());
        } else {
            bases.push_str(&line.trim_end().to_ascii_uppercase());
        }
    }
    if let Some(prev) = header {
        out.push(record(&prev, &bases));
    }
    out
}

fn record(header: &str, bases: &str) -> DnaSequence {
    DnaSequence::from_str_with_header(bases, header)
        .unwrap_or_else(|e| panic!("record {header}: {e}"))
}

fn mb(bytes: usize) -> String {
    format!("{:.1} MB", bytes as f64 / 1e6)
}

fn bpb(bytes: usize, n: u32) -> String {
    format!("{:.3} B/base", bytes as f64 / n as f64)
}

/// `count` reads of length `len` drawn from the references, each with a few substitutions.
fn sample_reads(rl: &BidirRlFmIndex, count: usize, len: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(7);
    let mut reads = Vec::with_capacity(count);
    let n = rl.num_sequences();
    while reads.len() < count {
        let seq = rl.sequence(SeqId::new(rng.random_range(0..n))).unwrap();
        if seq.len() < len {
            continue;
        }
        let start = rng.random_range(0..=seq.len() - len);
        let mut read = seq[start..start + len].to_vec();
        for _ in 0..rng.random_range(0..4) {
            let at = rng.random_range(0..len);
            read[at] = rng.random_range(1..=4);
        }
        reads.push(read);
    }
    reads
}

fn main() {
    let args = parse_args();
    let t = Instant::now();
    let seqs = read_fasta(&args.fasta, args.max_bases);
    let bases: usize = seqs.iter().map(DnaSequence::len).sum();
    println!(
        "input: {} sequences, {} bases ({:.1}s to read)",
        seqs.len(),
        bases,
        t.elapsed().as_secs_f64()
    );

    let config = FmIndexConfig {
        sa_sample_rate: args.sample_rate,
        use_gpu: false,
        build_threads: args.threads,
        ..Default::default()
    };
    let t = Instant::now();
    let fm = BidirFmIndex::build_cpu(&seqs, &config).expect("build BidirFmIndex");
    println!("BidirFmIndex built in {:.1}s", t.elapsed().as_secs_f64());
    drop(seqs);

    let t = Instant::now();
    let rl = BidirRlFmIndex::from_bidir_fm_index(&fm);
    println!(
        "BidirRlFmIndex converted in {:.1}s",
        t.elapsed().as_secs_f64()
    );

    let n = fm.text_len();
    let (fwd_runs, rev_runs) = (rl.fwd().bwt().num_runs(), rl.rev().bwt().num_runs());
    println!(
        "\nn = {n}, r(fwd) = {fwd_runs} (n/r = {:.1}), r(rev) = {rev_runs} (n/r = {:.1})",
        n as f64 / fwd_runs as f64,
        n as f64 / rev_runs as f64
    );

    let fm_fwd = fm.fwd().to_bytes().expect("serialize").len();
    let fm_bidir = fm.to_bytes().expect("serialize").len();
    let rl_fwd = rl.fwd().to_bytes().expect("serialize").len();
    // `"HRB\x01"` + u32 length + both halves; summed per half to keep one blob live at a time.
    let rl_bidir = 8 + rl_fwd + rl.rev().to_bytes().expect("serialize").len();
    println!("\nserialized size (sa_sample_rate = {}):", args.sample_rate);
    println!("  {:<26}{:>14}{:>18}", "", "uncompressed", "run-length");
    println!(
        "  {:<26}{:>14}{:>18}",
        "forward index",
        mb(fm_fwd),
        mb(rl_fwd)
    );
    println!(
        "  {:<26}{:>14}{:>18}",
        "bidirectional index",
        mb(fm_bidir),
        mb(rl_bidir)
    );
    println!(
        "  {:<26}{:>14}{:>18}",
        "bidirectional per base",
        bpb(fm_bidir, n),
        bpb(rl_bidir, n)
    );
    println!(
        "  ratio (bidirectional): {:.2}x",
        fm_bidir as f64 / rl_bidir as f64
    );

    // Resident breakdown: rank structures are where the two differ.
    let occ = fm.fwd().occ().heap_size_bytes() + fm.rev().occ().heap_size_bytes();
    let rlbwt = rl.fwd().bwt().heap_size_bytes() + rl.rev().bwt().heap_size_bytes();
    println!("\nresident rank structures (both halves):");
    println!("  occ tables:      {} ({})", mb(occ), bpb(occ, n));
    println!("  run-length BWTs: {} ({})", mb(rlbwt), bpb(rlbwt, n));
    println!(
        "  run-length index total heap: {} ({})",
        mb(rl.heap_size_bytes()),
        bpb(rl.heap_size_bytes(), n)
    );
    println!("  uncompressed retained text: {}", mb(n as usize));

    // Sequence extraction: identical bases, timed.
    let ids: Vec<SeqId> = (0..fm.num_sequences().min(50)).map(SeqId::new).collect();
    let t = Instant::now();
    let mut extracted = 0usize;
    for &id in &ids {
        let got = rl.sequence(id).expect("extract");
        assert_eq!(
            Some(got.as_slice()),
            fm.sequence(id),
            "sequence {id} differs"
        );
        extracted += got.len();
    }
    println!(
        "\nextracted {} sequences ({} bases) in {:.2}s, all identical",
        ids.len(),
        extracted,
        t.elapsed().as_secs_f64()
    );

    // SMEMs on sampled reads: identical results, timed.
    let reads = sample_reads(&rl, args.reads, 150);
    for locate in [false, true] {
        let t = Instant::now();
        let a: Vec<_> = reads.iter().map(|r| fm.find_smems(r, 19, locate)).collect();
        let fm_time = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let b: Vec<_> = reads.iter().map(|r| rl.find_smems(r, 19, locate)).collect();
        let rl_time = t.elapsed().as_secs_f64();
        let mems: usize = a.iter().map(Vec::len).sum();
        let mut same = true;
        for (mut x, mut y) in a.into_iter().zip(b) {
            x.iter_mut().for_each(|m| m.positions.sort());
            y.iter_mut().for_each(|m| m.positions.sort());
            same &= x == y;
        }
        assert!(same, "SMEMs differ between the indexes");
        println!(
            "find_smems x{} (min_len 19, locate={locate}): {mems} SMEMs, uncompressed {:.3}s, run-length {:.3}s, identical",
            reads.len(),
            fm_time,
            rl_time
        );
    }
}
