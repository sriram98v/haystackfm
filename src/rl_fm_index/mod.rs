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
//! structure. The text itself is not retained: [`RlFmIndex::sequence`] reads a reference
//! back off the BWT by LF-walking from its sentinel row (one stored row per
//! sentinel-separated piece: per sequence, plus per `#` record delimiter inside one).
//!
//! Also shared with the uncompressed index: the k-mer lookup table (seeding backward search
//! and `lookup_interval`), SMEM/MEM finding (the same sweep, generic over the index half),
//! and GPU construction (an uncompressed GPU build, then converted).
//!
//! Not yet supported compared with [`FmIndex`]: LCP-backed cursor contraction and
//! `parent_fwd` (an `O(r)` LCP is future work), GPU queries, and WASM bindings.

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
use crate::fm_index::lookup::LookupTable;
use crate::fm_index::prepare::{in_build_pool, prepare_cpu, Prepared};
use crate::fm_index::seq_id::{HeaderIndex, SeqId};
use crate::fm_index::FmIndex;
use crate::occ::BwtRank;
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
    /// Depth of the k-mer lookup table that seeds backward search, as
    /// [`FmIndexConfig::lookup_depth`](crate::FmIndexConfig::lookup_depth). Its size does
    /// not shrink with the run count. `0` disables it. Default: 0.
    pub lookup_depth: u32,
}

impl RlFmIndexConfig {
    /// The uncompressed-build config GPU construction converts from.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_fm_config(&self) -> crate::FmIndexConfig {
        crate::FmIndexConfig {
            sa_sample_rate: if self.sa_sample_rate == 0 {
                32
            } else {
                self.sa_sample_rate
            },
            use_gpu: true,
            lookup_depth: self.lookup_depth,
            build_threads: self.build_threads,
            occ_encoding: crate::occ::OccEncoding::Bitplane,
            build_lcp: false,
        }
    }
}

impl Default for RlFmIndexConfig {
    fn default() -> Self {
        Self {
            sa_sample_rate: 32,
            build_threads: 1,
            lookup_depth: 0,
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
    /// Exclusive end (sentinel included) of every sentinel-separated piece of the text: one
    /// per sequence, plus one per `#` record delimiter inside a sequence. Every
    /// `seq_boundaries` entry is one of them. Empty when `piece_end_rows` is.
    pub(crate) piece_boundaries: Vec<u32>,
    /// `piece_end_rows[k]` is the BWT row whose suffix starts at piece `k`'s sentinel, the
    /// starting point of [`sequence`](Self::sequence)'s LF walk over that piece. Empty on
    /// the reverse half of a [`BidirRlFmIndex`], which never extracts.
    pub(crate) piece_end_rows: Vec<u32>,
    /// Optional depth-k lookup table seeding backward search.
    pub(crate) lookup: Option<LookupTable>,
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
            Ok(Self::from_prepared(prepared, A::fns(), config, true))
        })
    }

    /// `extract`: record the sentinel rows [`sequence`](Self::sequence) starts from. Off for
    /// the reverse half of a bidirectional index, whose single "sequence" holds every other
    /// sentinel inside it.
    pub(crate) fn from_prepared(
        p: Prepared,
        alphabet_fns: AlphabetFns,
        config: &RlFmIndexConfig,
        extract: bool,
    ) -> Self {
        let sa_samples = (config.sa_sample_rate > 0)
            .then(|| SampledSuffixArray::from_full(&p.sa, config.sa_sample_rate, &p.seq_starts));
        let (piece_boundaries, piece_end_rows) = if extract {
            let bounds = crate::alphabet::piece_boundaries(&p.text);
            let rows = piece_end_rows_from(&bounds, |row| Some(p.sa.data[row as usize]))
                .expect("sentinel rows of a fresh suffix array");
            (bounds, rows)
        } else {
            (Vec::new(), Vec::new())
        };
        let bwt = build_bwt(&p.text, &p.sa);
        drop(p.sa);
        drop(p.text);
        let bwt = RlBwt::from_bwt(&bwt);
        let lookup = (config.lookup_depth > 0).then(|| {
            LookupTable::build(
                config.lookup_depth,
                p.text_len,
                &p.c_array,
                &bwt,
                &alphabet_fns,
            )
        });
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
            piece_boundaries,
            piece_end_rows,
            lookup,
        }
    }

    /// Build with GPU acceleration: an [`FmIndex`] is built on the GPU and converted (see
    /// [`from_fm_index`](Self::from_fm_index)), so peak memory is the uncompressed index's.
    /// `sa_sample_rate = 0` still builds samples to resolve the sentinel rows, then drops
    /// them. The lookup table is built after conversion, on the run-length rank.
    #[cfg(feature = "gpu")]
    pub async fn build(
        sequences: &[DnaSequence],
        config: &RlFmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        let fm = FmIndex::build(sequences, &config.gpu_fm_config()).await?;
        let mut rl = Self::from_fm_index(&fm);
        if config.sa_sample_rate == 0 {
            rl.sa_samples = None;
        }
        rl.ensure_lookup(config.lookup_depth);
        Ok(rl)
    }

    /// Build the lookup table on the run-length rank when `depth > 0` and none was carried
    /// over (GPU-built `FmIndex`es have none).
    #[cfg(feature = "gpu")]
    pub(crate) fn ensure_lookup(&mut self, depth: u32) {
        if depth > 0 && self.lookup.is_none() {
            self.lookup = Some(LookupTable::build(
                depth,
                self.text_len,
                &self.c_array,
                &self.bwt,
                &self.alphabet_fns,
            ));
        }
    }

    /// Convert an existing [`FmIndex`] without rebuilding its suffix array: its BWT is
    /// run-length encoded and its SA samples,
    /// metadata, alphabet and lookup table are copied. LCP and text are not carried over;
    /// [`sequence`](Self::sequence) extracts from the BWT instead.
    pub fn from_fm_index(fm: &FmIndex) -> Self {
        Self::convert(fm, true)
    }

    /// [`from_fm_index`](Self::from_fm_index); `extract` as in `from_prepared`. Resolving
    /// the sentinel rows of a reverse half would LF-walk across its inner sentinels, where
    /// LF is ambiguous and the walk need not reach a sample.
    pub(crate) fn convert(fm: &FmIndex, extract: bool) -> Self {
        let pieces = if extract {
            // A forward `FmIndex` keeps its text; without it, assume one piece per sequence
            // (the row mapping below rejects that guess if the text held `#` delimiters).
            let bounds = if fm.text.is_empty() {
                fm.seq_boundaries.clone()
            } else {
                crate::alphabet::piece_boundaries(&fm.text)
            };
            sentinel_rows_of(fm, &bounds).map(|rows| (bounds, rows))
        } else {
            None
        };
        Self::convert_with_rows(fm, pieces.unwrap_or_default())
    }

    fn convert_with_rows(
        fm: &FmIndex,
        (piece_boundaries, piece_end_rows): (Vec<u32>, Vec<u32>),
    ) -> Self {
        Self {
            c_array: fm.c_array.clone(),
            bwt: RlBwt::from_occ(&fm.occ),
            sa_samples: Some(fm.sa_samples.clone()),
            text_len: fm.text_len,
            num_sequences: fm.num_sequences,
            seq_boundaries: fm.seq_boundaries.clone(),
            seq_headers: fm.seq_headers.clone(),
            header_index: fm.header_index.clone(),
            alphabet_fns: fm.alphabet_fns,
            piece_boundaries,
            piece_end_rows,
            lookup: fm.lookup.clone(),
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

    /// Bases of sequence `id` as alphabet codes (see
    /// [`FmIndex::sequence`](crate::FmIndex::sequence)), or `None` when out of range or the
    /// index cannot extract (the reverse half of a bidirectional index, or one converted
    /// from an `FmIndex` whose SA samples miss sequence starts).
    ///
    /// The text is not stored: the bases are read back off the BWT by LF-walking from each
    /// piece's sentinel row to the preceding sentinel, one rank per base. A sequence with
    /// `#` record delimiters is walked piece by piece (LF is ambiguous at a sentinel), with
    /// [`SENTINEL`](crate::alphabet::SENTINEL) written at each delimiter, as
    /// `FmIndex::sequence` returns it. Unlike `FmIndex::sequence` this returns an owned
    /// buffer.
    pub fn sequence(&self, id: SeqId) -> Option<Vec<u8>> {
        use crate::alphabet::SENTINEL;
        if self.piece_end_rows.is_empty() {
            return None;
        }
        let end = *self.seq_boundaries.get(id.index())?;
        let start = id
            .index()
            .checked_sub(1)
            .map_or(0, |k| self.seq_boundaries[k]);
        let mut out = vec![SENTINEL; (end - start - 1) as usize];
        // Pieces whose boundary falls in `(start, end]`: this sequence's records.
        let first = self.piece_boundaries.partition_point(|&b| b <= start);
        let last = self.piece_boundaries.partition_point(|&b| b <= end);
        for k in first..last {
            let piece_start = k.checked_sub(1).map_or(0, |j| self.piece_boundaries[j]);
            let lo = (piece_start - start) as usize;
            let hi = (self.piece_boundaries[k] - 1 - start) as usize;
            let mut row = self.piece_end_rows[k];
            for slot in out[lo..hi].iter_mut().rev() {
                let (c, rank) = self.bwt.lf_step(row);
                if c == SENTINEL {
                    // Only corrupted piece metadata can end the walk early.
                    return None;
                }
                *slot = c;
                row = self.c_array.get(c) + rank;
            }
        }
        Some(out)
    }

    /// Bases of the sequence carrying `header`; see [`sequence`](Self::sequence).
    pub fn sequence_by_header(&self, header: &str) -> Option<Vec<u8>> {
        self.seq_id(header).and_then(|id| self.sequence(id))
    }

    /// Depth of the k-mer lookup table, or 0 when none was built.
    pub fn lookup_depth(&self) -> u32 {
        self.lookup.as_ref().map_or(0, |t| t.depth)
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
            + self.piece_boundaries.len() * 4
            + self.piece_end_rows.len() * 4
            + self.lookup.as_ref().map_or(0, LookupTable::heap_size_bytes)
            + self.seq_headers.iter().map(String::len).sum::<usize>()
    }
}

/// Sentinel row of each piece: rows `0..piece_boundaries.len()` hold the sentinel suffixes
/// (one sentinel per piece), and `text_pos_of(row)` gives the text position of that row's
/// suffix. `None` when a row is unresolved or does not map one-to-one onto the piece ends.
fn piece_end_rows_from(
    piece_boundaries: &[u32],
    text_pos_of: impl Fn(u32) -> Option<u32>,
) -> Option<Vec<u32>> {
    let mut rows = vec![u32::MAX; piece_boundaries.len()];
    for row in 0..piece_boundaries.len() as u32 {
        let pos = text_pos_of(row)?;
        // A sentinel at `pos` ends the piece whose exclusive boundary is `pos + 1`.
        let k = piece_boundaries.partition_point(|&b| b <= pos);
        if piece_boundaries.get(k) != Some(&(pos + 1)) || rows[k] != u32::MAX {
            return None;
        }
        rows[k] = row;
    }
    Some(rows)
}

/// Sentinel rows of an existing index, by LF-walking each sentinel row back to an SA
/// sample. A walk stops (unresolved) at the piece's first base, so it never crosses a
/// sentinel, where LF is ambiguous, and takes at most one piece length. Freshly built
/// indexes sample every piece start, so the walk always resolves. It may not for an older
/// file, or for the reverse half of a bidirectional index, whose one "sequence" holds the
/// other sentinels; extraction is then unavailable (`None`).
fn sentinel_rows_of(fm: &FmIndex, piece_boundaries: &[u32]) -> Option<Vec<u32>> {
    // Rows `0..pieces` must be exactly the sentinel rows.
    if fm
        .c_array
        .symbol_count(crate::alphabet::SENTINEL, fm.text_len)
        != piece_boundaries.len() as u32
    {
        return None;
    }
    let resolve = |mut row: u32| -> Option<u32> {
        let mut steps = 0u32;
        loop {
            if let Some(sa) = fm.sa_samples.get(row) {
                return sa.checked_add(steps);
            }
            let (c, rank) = fm.occ.lf_step(row);
            if c == crate::alphabet::SENTINEL || steps >= fm.text_len {
                return None;
            }
            row = fm.c_array.get(c) + rank;
            steps += 1;
        }
    };
    piece_end_rows_from(piece_boundaries, resolve)
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
