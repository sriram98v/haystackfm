//! Run-length FM-index ([`RlFmIndex`]) and its bidirectional form ([`BidirRlFmIndex`]).
//!
//! The same queries as [`FmIndex`] / [`BidirFmIndex`](crate::BidirFmIndex) — `count`,
//! `locate`, and the [`BidirInterval`](crate::BidirInterval) cursor — over a run-length
//! BWT ([`RlBwt`]), whose size grows with the number of BWT runs `r` rather than the text
//! length `n`. On repetitive collections (many genomes of one species) `n / r` reaches the
//! tens to hundreds, and the rank structure shrinks accordingly.
//!
//! Locate uses the regular sampled suffix array (every `sa_sample_rate`-th text position
//! plus each sequence start) and LF-walks over the run-length rank, so its space is still
//! `O(n / sa_sample_rate)`. `r`-sized SA sampling (toehold + φ) arrives with the move
//! structure. The text itself is not retained.
//!
//! Not yet supported compared with [`FmIndex`]: the k-mer lookup table, LCP-backed cursor
//! contraction, SMEM/MEM finding, and GPU/WASM paths.

mod bidir;
mod query;
mod serialize;

pub use bidir::BidirRlFmIndex;
pub use serialize::{BIDIR_FORMAT_MAGIC, FORMAT_MAGIC};

use crate::alphabet::{Alphabet, AlphabetFns, DnaSequence, IupacDna};
use crate::bwt::cpu::build_bwt;
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::fm_index::half::{FmHalf, Sealed};
use crate::fm_index::prepare::{in_build_pool, prepare_cpu, Prepared};
use crate::fm_index::seq_id::{HeaderIndex, SeqId};
use crate::fm_index::FmIndex;
use crate::occ::OccEncoding;
use crate::rlbwt::RlBwt;
use crate::suffix_array::SampledSuffixArray;

/// Configuration for run-length FM-index construction.
#[derive(Debug, Clone)]
pub struct RlFmIndexConfig {
    /// SA sampling rate for locate: every `sa_sample_rate`-th text position (and every
    /// sequence start) keeps its SA value. Higher = less memory, slower locate. `0` builds
    /// a count-only index without SA samples (the sample bitvector alone costs ~0.22
    /// bytes per text position, often more than the run-length rank structure), on which
    /// the `locate*` methods panic. Default: 32.
    pub sa_sample_rate: u32,
    /// Worker threads for construction (native targets). Default: 1.
    pub build_threads: u16,
    /// Lane encoding of the run-head rank table (see [`OccEncoding`]). Default: `Bitplane`.
    pub head_encoding: OccEncoding,
}

impl Default for RlFmIndexConfig {
    fn default() -> Self {
        Self {
            sa_sample_rate: 32,
            build_threads: 1,
            head_encoding: OccEncoding::Bitplane,
        }
    }
}

/// Run-length FM-index, ready for queries. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct RlFmIndex {
    pub(crate) c_array: CArray,
    pub(crate) bwt: RlBwt,
    /// `None` for a count-only index (`sa_sample_rate == 0`, or the reverse half of a
    /// [`BidirRlFmIndex`], which never locates).
    pub(crate) sa_samples: Option<SampledSuffixArray>,
    pub(crate) text_len: u32,
    pub(crate) num_sequences: u32,
    /// `seq_boundaries[k]` is the end of sequence `k` including its trailing sentinel.
    pub(crate) seq_boundaries: Vec<u32>,
    pub(crate) seq_headers: Vec<String>,
    /// Derived from `seq_headers`; not serialized.
    pub(crate) header_index: HeaderIndex,
    pub(crate) alphabet_fns: AlphabetFns,
}

impl RlFmIndex {
    /// Build on the CPU with the [`IupacDna`] alphabet (full IUPAC ambiguity matching).
    pub fn build_cpu(
        sequences: &[DnaSequence],
        config: &RlFmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        Self::build_cpu_with::<IupacDna>(sequences, config)
    }

    /// Build on the CPU with a custom [`Alphabet`] for match semantics.
    pub fn build_cpu_with<A: Alphabet>(
        sequences: &[DnaSequence],
        config: &RlFmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }
        in_build_pool(config.build_threads, || {
            let prepared = prepare_cpu(sequences)?;
            Ok(Self::from_prepared(prepared, A::fns(), config))
        })
    }

    fn from_prepared(p: Prepared, alphabet_fns: AlphabetFns, config: &RlFmIndexConfig) -> Self {
        let sa_samples = (config.sa_sample_rate > 0)
            .then(|| SampledSuffixArray::from_full(&p.sa, config.sa_sample_rate, &p.seq_starts));
        let bwt = build_bwt(&p.text, &p.sa);
        drop(p.sa);
        drop(p.text);
        let bwt = RlBwt::from_bwt(&bwt, config.head_encoding);
        Self {
            c_array: p.c_array,
            bwt,
            sa_samples,
            text_len: p.text_len,
            num_sequences: p.num_sequences,
            seq_boundaries: p.seq_boundaries,
            seq_headers: p.seq_headers,
            header_index: p.header_index,
            alphabet_fns,
        }
    }

    /// Convert an existing [`FmIndex`] without rebuilding its suffix array: its BWT is
    /// run-length encoded (head table in the same [`OccEncoding`]) and its SA samples,
    /// metadata and alphabet are copied. Lookup table, LCP and text are not carried over.
    pub fn from_fm_index(fm: &FmIndex) -> Self {
        Self {
            c_array: fm.c_array.clone(),
            bwt: RlBwt::from_occ(&fm.occ, fm.occ.encoding()),
            sa_samples: Some(fm.sa_samples.clone()),
            text_len: fm.text_len,
            num_sequences: fm.num_sequences,
            seq_boundaries: fm.seq_boundaries.clone(),
            seq_headers: fm.seq_headers.clone(),
            header_index: fm.header_index.clone(),
            alphabet_fns: fm.alphabet_fns,
        }
    }

    /// Total text length (sentinels included).
    pub fn text_len(&self) -> u32 {
        self.text_len
    }

    /// Number of sequences indexed.
    pub fn num_sequences(&self) -> u32 {
        self.num_sequences
    }

    /// FASTA headers in build order; a header's position is its [`SeqId`].
    pub fn seq_headers(&self) -> &[String] {
        &self.seq_headers
    }

    /// Header for a sequence id, or `None` when out of range.
    pub fn seq_header(&self, id: SeqId) -> Option<&str> {
        self.seq_headers.get(id.index()).map(String::as_str)
    }

    /// Id for a header, or `None` when no reference carries it.
    pub fn seq_id(&self, header: &str) -> Option<SeqId> {
        self.header_index.get(header)
    }

    /// Exclusive end of each reference (its sentinel included) in the concatenated text.
    pub fn seq_boundaries(&self) -> &[u32] {
        &self.seq_boundaries
    }

    /// True when the index carries SA samples, i.e. supports the `locate*` methods.
    pub fn can_locate(&self) -> bool {
        self.sa_samples.is_some()
    }

    /// The run-length BWT.
    pub fn bwt(&self) -> &RlBwt {
        &self.bwt
    }

    /// The matching semantics this index was built with.
    pub fn alphabet_fns(&self) -> &AlphabetFns {
        &self.alphabet_fns
    }

    /// Heap bytes of the rank structure plus the SA samples and metadata.
    pub fn heap_size_bytes(&self) -> usize {
        self.bwt.heap_size_bytes()
            + self
                .sa_samples
                .as_ref()
                .map_or(0, SampledSuffixArray::heap_size_bytes)
            + self.seq_boundaries.len() * 4
            + self.seq_headers.iter().map(String::len).sum::<usize>()
    }
}

impl From<&FmIndex> for RlFmIndex {
    fn from(fm: &FmIndex) -> Self {
        Self::from_fm_index(fm)
    }
}

impl Sealed for RlFmIndex {}

impl FmHalf for RlFmIndex {
    type Rank = RlBwt;

    #[inline]
    fn bwt_rank(&self) -> &RlBwt {
        &self.bwt
    }

    #[inline]
    fn c_array(&self) -> &CArray {
        &self.c_array
    }

    #[inline]
    fn half_text_len(&self) -> u32 {
        self.text_len
    }

    #[inline]
    fn alphabet(&self) -> &AlphabetFns {
        &self.alphabet_fns
    }
}
