//! Construction prefix shared by [`FmIndex`](super::FmIndex) and
//! [`RlFmIndex`](crate::rl_fm_index::RlFmIndex): concatenate the references, validate headers,
//! count symbols, build the suffix array and derive the SA-sample anchors. Each index then
//! turns the suffix array and BWT into its own rank structure.

use crate::alphabet::{self, DnaSequence};
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::fm_index::seq_id::HeaderIndex;
use crate::suffix_array::cpu::build_suffix_array;
use crate::suffix_array::SuffixArray;

/// Text, metadata and suffix array of a set of references, before any BWT-derived structure.
pub(crate) struct Prepared {
    /// Concatenated, sentinel-separated text in alphabet codes.
    pub text: Vec<u8>,
    pub text_len: u32,
    pub num_sequences: u32,
    /// `seq_boundaries[k]` is the end of sequence `k` including its trailing sentinel.
    pub seq_boundaries: Vec<u32>,
    pub seq_headers: Vec<String>,
    pub header_index: HeaderIndex,
    pub c_array: CArray,
    pub sa: SuffixArray,
    /// Text offsets every SA sampling must keep: the start of each sentinel-separated piece
    /// (each sequence, and each `#`-delimited record inside one), so an LF-walk never
    /// crosses a sentinel (all sentinels share one byte value, which makes LF ambiguous
    /// there). See [`alphabet::piece_starts`].
    pub seq_starts: Vec<u32>,
}

/// Run the shared construction prefix on the CPU. The caller has already rejected an empty
/// `sequences` slice.
pub(crate) fn prepare_cpu(sequences: &[DnaSequence]) -> Result<Prepared, FmIndexError> {
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
    // Built before the suffix array so a duplicate header fails fast rather than after
    // the expensive construction passes.
    let header_index = HeaderIndex::build(&seq_headers)?;

    // C array from text before SA construction — BWT is a permutation of text,
    // so character frequencies are identical. Avoids a second n-byte scan of BWT.
    let c_array = CArray::from_text(&text);

    // Build suffix array (single-threaded; psacak has no parallel API)
    let sa = build_suffix_array(&text);

    let seq_starts = alphabet::piece_starts(&text);

    Ok(Prepared {
        text,
        text_len,
        num_sequences,
        seq_boundaries,
        seq_headers,
        header_index,
        c_array,
        sa,
        seq_starts,
    })
}

/// Run `f` inside a scoped rayon pool of `threads` workers when `threads > 1` (native
/// targets only); otherwise run it on the calling thread's default pool.
pub(crate) fn in_build_pool<R: Send>(threads: u16, f: impl FnOnce() -> R + Send) -> R {
    #[cfg(not(target_arch = "wasm32"))]
    if threads > 1 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .build()
            .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());
        return pool.install(f);
    }
    #[cfg(target_arch = "wasm32")]
    let _ = threads;
    f()
}
