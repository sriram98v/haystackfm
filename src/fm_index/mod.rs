pub mod bidir;
pub mod bidir_index;
pub mod fwd_interval;
pub mod half;
pub mod lookup;
pub(crate) mod prepare;
pub mod query;
pub mod seq_classes;
pub mod seq_id;
pub mod serialize;
pub mod smem;

use crate::alphabet::{Alphabet, AlphabetFns, DnaSequence, IupacDna};
use crate::bwt::cpu::build_bwt;
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::fm_index::lookup::LookupTable;
use crate::fm_index::seq_classes::{DupWeights, SeqClasses};
use crate::fm_index::seq_id::{HeaderIndex, SeqId};
use crate::lcp::LcpArray;
use crate::occ::cpu::build_occ_table;
use crate::occ::{OccEncoding, OccTable};
use crate::suffix_array::SampledSuffixArray;

/// Configuration for FM-index construction.
#[derive(Debug, Clone)]
pub struct FmIndexConfig {
    /// SA sampling rate for locate queries. Higher = less memory, slower locate.
    /// Default: 32. Set to 1 for full SA (fastest locate, most memory).
    pub sa_sample_rate: u32,
    /// Whether to use GPU acceleration. Falls back to CPU if GPU unavailable.
    pub use_gpu: bool,
    /// Depth of the ACGT prefix lookup table for seeding backward search.
    /// 0 disables the table. Depth k uses about 4^k × 12 bytes on a pure-ACGT reference
    /// (k=10 → ~12 MB, k=13 → ~800 MB), plus 8 bytes per reference stretch that matches a
    /// k-mer through ambiguity codes (see [`crate::fm_index::lookup`]).
    /// Default: 0 (disabled).
    pub lookup_depth: u32,
    /// Number of threads for CPU index construction.  Rayon thread pool is
    /// sized to this value during `build_cpu`.  0 or 1 → single-threaded.
    /// Note: suffix array construction (psacak) is always single-threaded.
    pub build_threads: u16,
    /// Occ table Level-3 lane encoding (CPU construction only — GPU construction always uses
    /// `Bitplane`). `Bitplane` (default) is more memory-compact; `OneHot` skips the bitplane
    /// AND/XOR reconstruction on every `rank`/`lf_step` call at the cost of more resident
    /// memory. See [`crate::occ::OccEncoding`].
    pub occ_encoding: OccEncoding,
    /// Build the LCP array (CPU construction only) that the cursor contraction and
    /// ancestor operations need —
    /// [`BidirInterval::contract_left`](crate::fm_index::BidirInterval::contract_left),
    /// [`contract_right`](crate::fm_index::BidirInterval::contract_right) and
    /// [`FwdInterval::parent`](crate::fm_index::FwdInterval::parent). Those return
    /// [`FmIndexError::LcpNotBuilt`] without it. Opt-in: costs ~2.7 bytes per base of
    /// resident and serialized size on top of the index, and a `BidirFmIndex` builds one
    /// per half. Default: `false`. GPU construction never builds it.
    pub build_lcp: bool,
}

impl Default for FmIndexConfig {
    fn default() -> Self {
        Self {
            sa_sample_rate: 32,
            use_gpu: true,
            lookup_depth: 0,
            build_threads: 1,
            occ_encoding: OccEncoding::Bitplane,
            build_lcp: false,
        }
    }
}

/// The FM-index, ready for queries.
#[derive(Debug, Clone)]
pub struct FmIndex {
    pub(crate) c_array: CArray,
    pub(crate) occ: OccTable,
    pub(crate) sa_samples: SampledSuffixArray,
    /// The concatenated, sentinel-separated text in alphabet codes.
    ///
    /// Retained so [`sequence`](Self::sequence) can hand out a borrowed slice in O(1).
    /// An FM-index can otherwise only recover text by LF-walking backwards one symbol at a
    /// time, which is far too slow for callers that need random-access substrings (e.g. an
    /// aligner rescoring a read against a candidate diagonal). Stored unpacked: the alphabet
    /// fits in a nibble, but a packed representation could not be borrowed without unpacking
    /// into a fresh allocation on every call, which is the cost this field exists to avoid.
    pub(crate) text: Vec<u8>,
    pub(crate) text_len: u32,
    /// Number of input sequences, duplicates included.
    pub(crate) num_sequences: u32,
    /// Cumulative lengths of the stored (distinct) sequences, for mapping positions back.
    /// `slot_boundaries[s]` is the end of stored sequence `s` *including* its trailing
    /// sentinel. See [`seq_classes`](crate::fm_index::seq_classes).
    pub(crate) slot_boundaries: Vec<u32>,
    /// FASTA headers of every input sequence. A sequence's position here is its [`SeqId`].
    pub(crate) seq_headers: Vec<String>,
    /// O(1) reverse lookup for `seq_headers`. Derived, not serialized.
    pub(crate) header_index: HeaderIndex,
    /// Classes of identical input sequences; one representative of each is stored.
    pub(crate) seq_classes: SeqClasses,
    /// Extra multiplicity of SA rows inside duplicated representatives, for counts.
    pub(crate) dup_weights: DupWeights,
    /// Optional depth-k prefix lookup table for seeding backward search.
    pub(crate) lookup: Option<LookupTable>,
    /// Alphabet matching semantics (compatible symbols + core symbols for lookup BFS).
    pub(crate) alphabet_fns: AlphabetFns,
    /// Capped LCP array over the SA rows, present when built with
    /// `FmIndexConfig::build_lcp` (CPU only). Backs cursor contraction.
    pub(crate) lcp: Option<LcpArray>,
}

impl FmIndex {
    /// Build an FM-index from a set of DNA sequences using CPU with [`IupacDna`] alphabet
    /// (full IUPAC ambiguity-code matching — the default).
    ///
    /// To use a different matching alphabet, call [`build_cpu_with`].
    ///
    /// [`build_cpu_with`]: FmIndex::build_cpu_with
    pub fn build_cpu(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        Self::build_cpu_with::<IupacDna>(sequences, config)
    }

    /// Build an FM-index using CPU with a custom [`Alphabet`] for match semantics.
    ///
    /// # Example
    /// ```rust,ignore
    /// use haystackfm::{FmIndex, FmIndexConfig, ExactDna};
    /// let index = FmIndex::build_cpu_with::<ExactDna>(&seqs, &config)?;
    /// ```
    pub fn build_cpu_with<A: Alphabet>(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }

        prepare::in_build_pool(config.build_threads, || {
            Self::build_cpu_inner::<A>(sequences, config, false)
        })
    }

    /// The reverse half of a [`BidirFmIndex`](crate::BidirFmIndex), built on the one
    /// sequence [`reverse_as_sequence`](crate::fm_index::bidir_index::reverse_as_sequence)
    /// returns. That half never locates, so only text offset 0 is force-sampled rather than
    /// every piece start behind its interior sentinels (which also keeps it byte-identical
    /// to halves written before `#` delimiters existed).
    pub(crate) fn build_cpu_reverse_half_with<A: Alphabet>(
        reversed: &DnaSequence,
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        prepare::in_build_pool(config.build_threads, || {
            Self::build_cpu_inner::<A>(std::slice::from_ref(reversed), config, true)
        })
    }

    fn build_cpu_inner<A: Alphabet>(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
        reverse_half: bool,
    ) -> Result<Self, FmIndexError> {
        let alphabet_fns = A::fns();
        let prepare::Prepared {
            text,
            text_len,
            num_sequences,
            slot_boundaries,
            seq_headers,
            header_index,
            seq_classes,
            dup_weights,
            c_array,
            sa,
            seq_starts,
        } = prepare::prepare_cpu(sequences)?;
        let seq_starts = if reverse_half { vec![0] } else { seq_starts };

        // Build BWT from SA. The text is retained (not dropped) to back `sequence()`.
        let bwt = build_bwt(&text, &sa);

        // LCP via Kasai while the full SA is still resident (transient 4n-byte ISA).
        let lcp = config
            .build_lcp
            .then(|| LcpArray::build_kasai(&text, &sa.data));

        // Sample SA then free it before building Occ (saves ~4n bytes of peak memory).
        // Sequence starts are force-sampled (see `Prepared::seq_starts`).
        let sa_samples = SampledSuffixArray::from_full(&sa, config.sa_sample_rate, &seq_starts);
        drop(sa);

        // Build Occ table from BWT (parallelises internally on non-WASM targets), then drop the
        // BWT — its one-hot presence bitvectors already encode every symbol, so keeping a
        // separate packed `Bwt` around would duplicate ~0.5 bytes/base of resident memory.
        let occ = build_occ_table(&bwt, config.occ_encoding);
        drop(bwt);

        // Build depth-k lookup table if requested (using alphabet's core symbols as radix)
        let lookup = if config.lookup_depth > 0 {
            Some(LookupTable::build(
                config.lookup_depth,
                text_len,
                &c_array,
                &occ,
                &alphabet_fns,
            ))
        } else {
            None
        };

        Ok(Self {
            c_array,
            occ,
            sa_samples,
            text,
            text_len,
            num_sequences,
            slot_boundaries,
            seq_headers,
            header_index,
            seq_classes,
            dup_weights,
            lookup,
            alphabet_fns,
            lcp,
        })
    }

    /// Total text length (including sentinels).
    pub fn text_len(&self) -> u32 {
        self.text_len
    }

    /// The occ table (rank over the BWT), e.g. for [`OccTable::heap_size_bytes`] or
    /// [`RlBwt::from_occ`](crate::rlbwt::RlBwt::from_occ).
    pub fn occ(&self) -> &OccTable {
        &self.occ
    }

    /// Number of sequences indexed: every input sequence, duplicates included.
    pub fn num_sequences(&self) -> u32 {
        self.num_sequences
    }

    /// Number of distinct sequences stored in the text (one per
    /// [equivalence class](crate::fm_index::seq_classes)).
    pub fn num_distinct_sequences(&self) -> u32 {
        self.slot_boundaries.len() as u32
    }

    /// End offset (trailing sentinel included) of every **stored** sequence in the indexed
    /// text, in text order. Stored sequences are the class representatives, so this is
    /// indexed by stored slot, not by [`SeqId`]; with no duplicate input sequences the two
    /// coincide. Raw text positions (`locate_positions`, [`map_position`](Self::map_position))
    /// live in this coordinate space.
    pub fn seq_boundaries(&self) -> &[u32] {
        &self.slot_boundaries
    }

    /// Classes of identical input sequences. See [`crate::fm_index::seq_classes`].
    pub fn seq_classes(&self) -> &SeqClasses {
        &self.seq_classes
    }

    /// Every input sequence identical to `id` (itself included), ascending; empty when `id`
    /// is out of range.
    pub fn equivalence_class(&self, id: SeqId) -> &[SeqId] {
        self.seq_classes.equivalence_class(id)
    }

    /// The member of `id`'s class that is stored in the index, or `None` when `id` is out
    /// of range.
    pub fn representative(&self, id: SeqId) -> Option<SeqId> {
        self.seq_classes.representative(id)
    }

    /// FASTA headers of every indexed reference, in build order.
    ///
    /// A header's position in this slice is its [`SeqId`]. Ids are preserved by
    /// [`to_bytes`](Self::to_bytes) / [`from_bytes`](Self::from_bytes), so a caller can
    /// build an `id -> label` table from this slice once at load time and index it by
    /// [`SeqId::index`] thereafter.
    pub fn seq_headers(&self) -> &[String] {
        &self.seq_headers
    }

    /// Header for a sequence id, or `None` when the id is out of range. O(1).
    pub fn seq_header(&self, id: SeqId) -> Option<&str> {
        self.seq_headers.get(id.index()).map(String::as_str)
    }

    /// Id for a header, or `None` when no reference carries it. O(1).
    ///
    /// Headers are unique — [`FmIndexError::DuplicateHeader`] is raised at build time
    /// otherwise — so this is an exact inverse of [`seq_header`](Self::seq_header).
    pub fn seq_id(&self, header: &str) -> Option<SeqId> {
        self.header_index.get(header)
    }

    /// Bases of one indexed sequence, or `None` when the id is out of range.
    ///
    /// O(1) — a borrowed slice of the retained text, with the trailing sentinel excluded.
    ///
    /// The bases are **alphabet codes**, not ASCII: `A = 1`, `C = 2`, `G = 3`, `T = 4`,
    /// `N = 5`, and so on (see [`crate::alphabet`]). Use [`crate::alphabet::decode_char`]
    /// to render them, or [`crate::alphabet::encode_byte`] to bring a query into the same
    /// space before comparing. A sequence built with `#` record delimiters carries
    /// [`SENTINEL`](crate::alphabet::SENTINEL) at each one;
    /// [`crate::alphabet::decode_sequence`] renders those as `#`.
    pub fn sequence(&self, id: SeqId) -> Option<&[u8]> {
        // Empty text means it was released by `forget_text` (the reverse half of a
        // `BidirFmIndex`); a built index always has a non-empty text otherwise.
        if self.text.is_empty() {
            return None;
        }
        // A duplicate's bases are its representative's. `slot_boundaries[s]` is the end of
        // stored sequence s *including* its sentinel, so the bases run from the previous
        // boundary up to (but not including) that sentinel.
        let slot = self.seq_classes.slot_of(id)? as usize;
        let end = self.slot_boundaries[slot] as usize - 1;
        let start = match slot {
            0 => 0,
            s => self.slot_boundaries[s - 1] as usize,
        };
        Some(&self.text[start..end])
    }

    /// Bases of the sequence carrying `header`, or `None` when no reference carries it.
    ///
    /// Equivalent to `seq_id(header).and_then(|id| self.sequence(id))`; see
    /// [`sequence`](Self::sequence) for the encoding of the returned bases.
    pub fn sequence_by_header(&self, header: &str) -> Option<&[u8]> {
        self.seq_id(header).and_then(|id| self.sequence(id))
    }

    /// Release the retained text, giving up [`sequence`](Self::sequence) (which then
    /// returns `None`) in exchange for ~n bytes of resident and serialized size.
    ///
    /// Used for the reverse half of a [`BidirFmIndex`], whose text is a redundant
    /// reversal of the forward half's and is never served to callers.
    pub(crate) fn forget_text(&mut self) {
        self.text = Vec::new();
    }

    /// True when this index carries an LCP array (built with `FmIndexConfig::build_lcp` on
    /// the CPU), which cursor contraction requires.
    pub fn has_lcp(&self) -> bool {
        self.lcp.is_some()
    }

    /// Build an FM-index from a set of DNA sequences using GPU acceleration.
    #[cfg(feature = "gpu")]
    pub async fn build(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        Self::build_gpu_inner(sequences, config, false).await
    }

    /// GPU counterpart of [`build_cpu_reverse_half_with`](Self::build_cpu_reverse_half_with):
    /// only text offset 0 is force-sampled.
    #[cfg(feature = "gpu")]
    pub(crate) async fn build_reverse_half(
        reversed: &DnaSequence,
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        Self::build_gpu_inner(std::slice::from_ref(reversed), config, true).await
    }

    #[cfg(feature = "gpu")]
    async fn build_gpu_inner(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
        reverse_half: bool,
    ) -> Result<Self, FmIndexError> {
        use crate::bwt::gpu::BwtPipelines;
        use crate::gpu::GpuContext;
        use crate::occ::gpu::OccPipelines;
        use crate::suffix_array::gpu::SaPipelines;

        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }

        let prepare::PreparedText {
            text,
            text_len,
            num_sequences,
            slot_boundaries,
            seq_headers,
            header_index,
            seq_classes,
        } = prepare::prepare_text(sequences)?;

        let ctx = GpuContext::new().await?;
        let sa_pipelines = SaPipelines::new(&ctx);
        let bwt_pipelines = BwtPipelines::new(&ctx);
        let occ_pipelines = OccPipelines::new(&ctx);

        // Build suffix array on GPU
        let sa = sa_pipelines.build_suffix_array(&ctx, &text).await;

        // Build BWT on GPU
        let bwt = bwt_pipelines.build_bwt(&ctx, &text, &sa).await;

        // Build C array on CPU (trivial from BWT character counts)
        let c_array = CArray::from_bwt(&bwt);

        // Build Occ table on GPU
        let occ = occ_pipelines.build_occ_table(&ctx, &bwt).await;

        // Sample the suffix array
        // Every piece start (sequence or `#`-delimited record) must be sampled so
        // `resolve_sa`'s LF-walk never crosses a sentinel (all sentinels share one byte
        // value → ambiguous LF).
        let seq_starts = if reverse_half {
            vec![0]
        } else {
            crate::alphabet::piece_starts(&text)
        };
        let sa_samples = SampledSuffixArray::from_full(&sa, config.sa_sample_rate, &seq_starts);
        let dup_weights = DupWeights::build(&sa.data, &slot_boundaries, &seq_classes);

        drop(bwt);

        Ok(Self {
            c_array,
            occ,
            sa_samples,
            text,
            text_len,
            num_sequences,
            slot_boundaries,
            seq_headers,
            header_index,
            seq_classes,
            dup_weights,
            lookup: None,
            // GPU construction is IUPAC-only (shaders hard-code the 16-symbol COMPAT table).
            alphabet_fns: IupacDna::fns(),
            // LCP construction is CPU-only; GPU-built indexes cannot contract cursors.
            lcp: None,
        })
    }

    /// LF-mapping: given a position in the BWT, return the position of the
    /// same character in the first column.
    fn lf_mapping(&self, i: u32) -> u32 {
        let (c, rank) = self.occ.lf_step(i);
        self.c_array.get(c) + rank
    }
}
