pub mod bidir;
pub mod bidir_index;
pub mod lookup;
pub mod query;
pub mod serialize;
pub mod smem;

use crate::alphabet::{self, Alphabet, AlphabetFns, DnaSequence, IupacDna};
use crate::bwt::cpu::build_bwt;
use crate::bwt::Bwt;
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::fm_index::lookup::LookupTable;
use crate::occ::cpu::build_occ_table;
use crate::occ::OccTable;
use crate::suffix_array::cpu::build_suffix_array;
use crate::suffix_array::SampledSuffixArray;

#[cfg(not(target_arch = "wasm32"))]
use rayon;

/// Configuration for FM-index construction.
#[derive(Debug, Clone)]
pub struct FmIndexConfig {
    /// SA sampling rate for locate queries. Higher = less memory, slower locate.
    /// Default: 32. Set to 1 for full SA (fastest locate, most memory).
    pub sa_sample_rate: u32,
    /// Whether to use GPU acceleration. Falls back to CPU if GPU unavailable.
    pub use_gpu: bool,
    /// Depth of the ACGT prefix lookup table for seeding backward search.
    /// 0 disables the table. Depth k uses 4^k × 8 bytes (k=10 → ~8 MB, k=13 → ~537 MB).
    /// Default: 0 (disabled).
    pub lookup_depth: u32,
    /// Number of threads for CPU index construction.  Rayon thread pool is
    /// sized to this value during `build_cpu`.  0 or 1 → single-threaded.
    /// Note: suffix array construction (psacak) is always single-threaded.
    pub build_threads: u16,
}

impl Default for FmIndexConfig {
    fn default() -> Self {
        Self {
            sa_sample_rate: 32,
            use_gpu: true,
            lookup_depth: 0,
            build_threads: 1,
        }
    }
}

/// The FM-index, ready for queries.
#[derive(Debug, Clone)]
pub struct FmIndex {
    pub(crate) bwt: Bwt,
    pub(crate) c_array: CArray,
    pub(crate) occ: OccTable,
    pub(crate) sa_samples: SampledSuffixArray,
    pub(crate) text_len: u32,
    pub(crate) num_sequences: u32,
    /// Cumulative sequence lengths for mapping positions back to sequences.
    pub(crate) seq_boundaries: Vec<u32>,
    /// FASTA headers for each sequence (index-parallel with seq_boundaries).
    pub(crate) seq_headers: Vec<String>,
    /// Optional depth-k prefix lookup table for seeding backward search.
    pub(crate) lookup: Option<LookupTable>,
    /// Alphabet matching semantics (compatible symbols + core symbols for lookup BFS).
    pub(crate) alphabet_fns: AlphabetFns,
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
    /// use webgpu_fmidx::{FmIndex, FmIndexConfig, ExactDna};
    /// let index = FmIndex::build_cpu_with::<ExactDna>(&seqs, &config)?;
    /// ```
    pub fn build_cpu_with<A: Alphabet>(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }

        // Configure rayon thread pool when multi-threading is requested.
        // We build inside a closure so the pool is scoped to construction.
        #[cfg(not(target_arch = "wasm32"))]
        if config.build_threads > 1 {
            let threads = config.build_threads as usize;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());
            return pool.install(|| Self::build_cpu_inner::<A>(sequences, config));
        }
        Self::build_cpu_inner::<A>(sequences, config)
    }

    fn build_cpu_inner<A: Alphabet>(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        let alphabet_fns = A::fns();
        let (text, seq_boundaries) = alphabet::concatenate_sequences(sequences)?;
        let text_len = text.len() as u32;
        let num_sequences = sequences.len() as u32;

        let seq_headers: Vec<String> = sequences
            .iter()
            .enumerate()
            .map(|(i, seq)| {
                let h = seq.header();
                if h.is_empty() {
                    format!("seq_{}", i)
                } else {
                    h.to_string()
                }
            })
            .collect();

        // C array from text before SA construction — BWT is a permutation of text,
        // so character frequencies are identical. Avoids a second n-byte scan of BWT.
        let c_array = CArray::from_text(&text);

        // Build suffix array (single-threaded; psacak has no parallel API)
        let sa = build_suffix_array(&text);

        // Build BWT from SA, then free text (~n bytes peak reduction)
        let bwt = build_bwt(&text, &sa);
        drop(text);

        // Sample SA then free it before building Occ (saves ~4n bytes of peak memory)
        let sa_samples = SampledSuffixArray::from_full(&sa, config.sa_sample_rate);
        drop(sa);

        // Build Occ table from BWT (parallelises internally on non-WASM targets)
        let occ = build_occ_table(&bwt);

        // Build depth-k lookup table if requested (using alphabet's core symbols as radix)
        let lookup = if config.lookup_depth > 0 {
            Some(LookupTable::build(
                config.lookup_depth,
                text_len,
                &c_array,
                &occ,
                alphabet_fns.core_symbols,
            ))
        } else {
            None
        };

        Ok(Self {
            bwt,
            c_array,
            occ,
            sa_samples,
            text_len,
            num_sequences,
            seq_boundaries,
            seq_headers,
            lookup,
            alphabet_fns,
        })
    }

    /// Total text length (including sentinels).
    pub fn text_len(&self) -> u32 {
        self.text_len
    }

    /// Number of sequences indexed.
    pub fn num_sequences(&self) -> u32 {
        self.num_sequences
    }

    /// Build an FM-index from a set of DNA sequences using GPU acceleration.
    #[cfg(feature = "gpu")]
    pub async fn build(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        use crate::bwt::gpu::BwtPipelines;
        use crate::gpu::GpuContext;
        use crate::occ::gpu::OccPipelines;
        use crate::suffix_array::gpu::SaPipelines;

        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }

        let (text, seq_boundaries) = alphabet::concatenate_sequences(sequences)?;
        let text_len = text.len() as u32;
        let num_sequences = sequences.len() as u32;

        let seq_headers: Vec<String> = sequences
            .iter()
            .enumerate()
            .map(|(i, seq)| {
                let h = seq.header();
                if h.is_empty() {
                    format!("seq_{}", i)
                } else {
                    h.to_string()
                }
            })
            .collect();

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
        let sa_samples = SampledSuffixArray::from_full(&sa, config.sa_sample_rate);

        Ok(Self {
            bwt,
            c_array,
            occ,
            sa_samples,
            text_len,
            num_sequences,
            seq_boundaries,
            seq_headers,
            lookup: None,
            // GPU construction is IUPAC-only (shaders hard-code the 16-symbol COMPAT table).
            alphabet_fns: IupacDna::fns(),
        })
    }

    /// LF-mapping: given a position in the BWT, return the position of the
    /// same character in the first column.
    fn lf_mapping(&self, i: u32) -> u32 {
        let c = self.bwt.get(i as usize);
        self.c_array.get(c) + self.occ.rank(c, i)
    }
}
