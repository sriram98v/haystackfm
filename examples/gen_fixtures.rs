//! Freeze on-disk format fixtures for the compatibility tests in
//! `src/fm_index/serialize.rs` and `src/fm_index/bidir_index.rs`.
//!
//! Run this on the commit whose format you want to freeze (before bumping
//! `FORMAT_MAGIC`); it writes `tests/fixtures/v{N}_*.hfm` from exactly the input and configs
//! those tests rebuild. CPU construction is deterministic, so a later reader must load these
//! bytes into an index that serializes identically to a fresh build.
//!
//! ```text
//! cargo run --example gen_fixtures
//! ```

use haystackfm::alphabet::{DnaSequence, ExactDna};
use haystackfm::fm_index::bidir_index::BidirFmIndex;
use haystackfm::fm_index::serialize::FORMAT_MAGIC;
use haystackfm::fm_index::{FmIndex, FmIndexConfig};
use haystackfm::occ::OccEncoding;
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

fn main() {
    let version = FORMAT_MAGIC[3];
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
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
