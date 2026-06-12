pub mod bidir;
pub mod bidir_index;
pub mod query;
pub mod serialize;
pub mod smem;

use crate::alphabet::{self, DnaSequence};
use crate::bwt::cpu::build_bwt;
use crate::bwt::Bwt;
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::occ::cpu::build_occ_table;
use crate::occ::OccTable;
use crate::suffix_array::cpu::build_suffix_array;
use crate::suffix_array::SampledSuffixArray;

/// Configuration for FM-index construction.
#[derive(Debug, Clone)]
pub struct FmIndexConfig {
    /// SA sampling rate for locate queries. Higher = less memory, slower locate.
    /// Default: 32. Set to 1 for full SA (fastest locate, most memory).
    pub sa_sample_rate: u32,
    /// Whether to use GPU acceleration. Falls back to CPU if GPU unavailable.
    pub use_gpu: bool,
}

impl Default for FmIndexConfig {
    fn default() -> Self {
        Self {
            sa_sample_rate: 32,
            use_gpu: true,
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
}

impl FmIndex {
    /// Build an FM-index from a set of DNA sequences using CPU.
    pub fn build_cpu(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
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

        // Build suffix array
        let sa = build_suffix_array(&text);

        // Build BWT from SA
        let bwt = build_bwt(&text, &sa);

        // Sample SA then free it before building Occ (saves ~4n bytes of peak memory)
        let sa_samples = SampledSuffixArray::from_full(&sa, config.sa_sample_rate);
        drop(sa);

        // Build C array from BWT
        let c_array = CArray::from_text(&bwt.data);

        // Build Occ table from BWT
        let occ = build_occ_table(&bwt);

        Ok(Self {
            bwt,
            c_array,
            occ,
            sa_samples,
            text_len,
            num_sequences,
            seq_boundaries,
            seq_headers,
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
        let c_array = CArray::from_text(&bwt.data);

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
        })
    }

    /// LF-mapping: given a position in the BWT, return the position of the
    /// same character in the first column.
    fn lf_mapping(&self, i: u32) -> u32 {
        let c = self.bwt.data[i as usize];
        self.c_array.get(c) + self.occ.rank(c, i)
    }
}
