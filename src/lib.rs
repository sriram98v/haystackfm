//! # webgpu-fmidx
//!
//! A GPU-accelerated FM-index library for DNA sequence alignment.
//!
//! This crate provides an FM-index data structure built on the Burrows-Wheeler
//! Transform (BWT) for efficient exact-match and approximate-match queries over
//! DNA sequences (A, C, G, T). Construction can run on the CPU or, when the
//! `gpu` feature is enabled, be accelerated via WebGPU compute shaders.
//!
//! ## Features
//!
//! - `cpu` *(default)*: CPU-only BWT, suffix-array, and Occ-table construction.
//! - `gpu`: GPU-accelerated construction via `wgpu` (requires a compatible adapter).
//! - `wasm`: WebAssembly bindings exposing the index to JavaScript/TypeScript.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use webgpu_fmidx::{DnaSequence, FmIndex, FmIndexConfig};
//!
//! let seq = DnaSequence::from_str("ACGTACGT").unwrap();
//! let config = FmIndexConfig { sa_sample_rate: 4, use_gpu: false };
//! let index = FmIndex::build_cpu(&[seq], &config).unwrap();
//!
//! let pattern = [1u8, 2, 3, 4]; // A C G T (encoded)
//! assert_eq!(index.count(&pattern), 2);
//! ```
//!
//! ## Bidirectional FM-index and SMEM Finding
//!
//! For sequence alignment workloads, [`BidirFmIndex`] supports efficient
//! Super-Maximal Exact Match (SMEM) finding via the Lam et al. 2009 algorithm:
//!
//! ```rust,ignore
//! use webgpu_fmidx::{DnaSequence, BidirFmIndex, FmIndexConfig};
//!
//! let seq = DnaSequence::from_str("ACGTACGT").unwrap();
//! let config = FmIndexConfig { sa_sample_rate: 4, use_gpu: false };
//! let bidir = BidirFmIndex::build_cpu(&[seq], &config).unwrap();
//! let query = DnaSequence::from_str("ACGT").unwrap();
//! let smems = bidir.find_smems(query.as_slice(), 1);
//! ```

pub mod alphabet;
pub mod bwt;
pub mod c_array;
pub mod error;
pub mod fm_index;
pub mod occ;
pub mod suffix_array;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use alphabet::DnaSequence;
pub use error::FmIndexError;
pub use fm_index::bidir::BidirInterval;
pub use fm_index::bidir_index::BidirFmIndex;
pub use fm_index::smem::Mem;
#[cfg(feature = "gpu")]
pub use gpu::MemHit;
pub use fm_index::{FmIndex, FmIndexConfig};
