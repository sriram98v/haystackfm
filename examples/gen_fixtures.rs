//! Freeze on-disk format fixtures for the compatibility tests in
//! `src/fm_index/serialize.rs`, `src/fm_index/bidir_index.rs` and
//! `src/rl_fm_index/serialize.rs`.
//!
//! Run this on the commit whose format you want to freeze (before bumping its
//! `FORMAT_MAGIC`); it writes `tests/fixtures/v{N}_*.hfm` (FM-index) and
//! `tests/fixtures/v{N}_rl_*.hrl` / `.hrb` (run-length) from exactly the input and configs
//! those tests rebuild. CPU construction is deterministic, so a later reader must load these
//! bytes into an index that serializes identically to a fresh build. Pass `fm` or `rl` to
//! write only that family.
//!
//! ```text
//! cargo run --example gen_fixtures [fm|rl]
//! ```

use haystackfm::alphabet::{DnaSequence, ExactDna};
use haystackfm::fm_index::bidir_index::BidirFmIndex;
use haystackfm::fm_index::serialize::FORMAT_MAGIC;
use haystackfm::fm_index::{FmIndex, FmIndexConfig};
use haystackfm::occ::OccEncoding;
use haystackfm::rl_fm_index::{BidirRlFmIndex, RlFmIndex, RlFmIndexConfig};
use std::path::PathBuf;

fn fixture_input() -> Vec<DnaSequence> {
    vec![
        DnaSequence::from_str_with_header("ACGTNACGTRYSWKMBDHVACGTACGTTTGCA", "chr1").unwrap(),
        DnaSequence::from_str_with_header("TTGACGTACGTNNACGGAAC", "plasmid").unwrap(),
        DnaSequence::from_str_with_header("GATTACA", "tiny").unwrap(),
    ]
}

fn onehot_config() -> FmIndexConfig {
    FmIndexConfig {
        sa_sample_rate: 2,
        use_gpu: false,
        occ_encoding: OccEncoding::OneHot,
        build_lcp: true,
        lookup_depth: 3,
        ..Default::default()
    }
}

fn bitplane_config() -> FmIndexConfig {
    FmIndexConfig {
        sa_sample_rate: 4,
        use_gpu: false,
        occ_encoding: OccEncoding::Bitplane,
        build_lcp: false,
        lookup_depth: 0,
        ..Default::default()
    }
}

fn rl_config() -> RlFmIndexConfig {
    RlFmIndexConfig {
        sa_sample_rate: 4,
        build_threads: 1,
        lookup_depth: 3,
    }
}

fn main() {
    let family = std::env::args().nth(1);
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    if family.as_deref() != Some("rl") {
        write_fm(&dir);
    }
    if family.as_deref() != Some("fm") {
        write_rl(&dir);
    }
}

fn write_rl(dir: &std::path::Path) {
    let version = haystackfm::rl_fm_index::FORMAT_MAGIC[3];
    let input = fixture_input();
    let blobs: [(&str, Vec<u8>); 2] = [
        (
            "rl_lookup.hrl",
            RlFmIndex::build_cpu(&input, &rl_config())
                .unwrap()
                .to_bytes()
                .unwrap(),
        ),
        (
            "rl_bidir_lookup.hrb",
            BidirRlFmIndex::build_cpu(&input, &rl_config())
                .unwrap()
                .to_bytes()
                .unwrap(),
        ),
    ];
    for (name, bytes) in blobs {
        let path = dir.join(format!("v{version}_{name}"));
        std::fs::write(&path, &bytes).unwrap();
        println!("{} ({} bytes)", path.display(), bytes.len());
    }
}

fn write_fm(dir: &std::path::Path) {
    let version = FORMAT_MAGIC[3];
    let input = fixture_input();

    let blobs: [(&str, Vec<u8>); 4] = [
        (
            "onehot_lcp_lookup",
            FmIndex::build_cpu(&input, &onehot_config())
                .unwrap()
                .to_bytes()
                .unwrap(),
        ),
        (
            "bitplane_nolcp",
            FmIndex::build_cpu(&input, &bitplane_config())
                .unwrap()
                .to_bytes()
                .unwrap(),
        ),
        (
            "bidir_onehot_lcp_lookup",
            BidirFmIndex::build_cpu(&input, &onehot_config())
                .unwrap()
                .to_bytes()
                .unwrap(),
        ),
        (
            "exact_onehot_lookup",
            FmIndex::build_cpu_with::<ExactDna>(&input, &onehot_config())
                .unwrap()
                .to_bytes()
                .unwrap(),
        ),
    ];

    for (name, bytes) in blobs {
        let path = dir.join(format!("v{version}_{name}.hfm"));
        std::fs::write(&path, &bytes).unwrap();
        println!("{} ({} bytes)", path.display(), bytes.len());
    }
}
