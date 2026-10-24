//! Construction prefix shared by [`FmIndex`](super::FmIndex) and
//! [`RlFmIndex`](crate::rl_fm_index::RlFmIndex): concatenate the references, validate headers,
//! count symbols, build the suffix array and derive the SA-sample anchors. Each index then
//! turns the suffix array and BWT into its own rank structure.

use crate::alphabet::{self, DnaSequence};
use crate::c_array::CArray;
use crate::error::FmIndexError;
use crate::fm_index::seq_classes::{DupWeights, SeqClasses};
use crate::fm_index::seq_id::HeaderIndex;
use crate::suffix_array::cpu::build_suffix_array;
use crate::suffix_array::SuffixArray;

/// The deduplicated text of a set of references and their metadata (see
/// [`seq_classes`](crate::fm_index::seq_classes)).
pub(crate) struct PreparedText {
    /// Concatenated, sentinel-separated text of the stored (distinct) sequences, in
    /// alphabet codes.
    pub text: Vec<u8>,
    pub text_len: u32,
    /// Number of input sequences, duplicates included.
    pub num_sequences: u32,
    /// `slot_boundaries[s]` is the end of stored sequence `s` including its trailing
    /// sentinel.
    pub slot_boundaries: Vec<u32>,
    /// Header of every input sequence, indexed by [`SeqId`](crate::SeqId).
    pub seq_headers: Vec<String>,
    pub header_index: HeaderIndex,
    pub seq_classes: SeqClasses,
}

/// Text, metadata and suffix array of a set of references, before any BWT-derived structure.
pub(crate) struct Prepared {
    /// Concatenated, sentinel-separated text in alphabet codes.
    pub text: Vec<u8>,
    pub text_len: u32,
    pub num_sequences: u32,
    /// `slot_boundaries[s]` is the end of stored sequence `s` including its trailing
    /// sentinel.
    pub slot_boundaries: Vec<u32>,
    pub seq_headers: Vec<String>,
    pub header_index: HeaderIndex,
    pub seq_classes: SeqClasses,
    pub dup_weights: DupWeights,
    pub c_array: CArray,
    pub sa: SuffixArray,
    /// Text offsets every SA sampling must keep: the start of each sentinel-separated piece
    /// (each sequence, and each `#`-delimited record inside one), so an LF-walk never
    /// crosses a sentinel (all sentinels share one byte value, which makes LF ambiguous
    /// there). See [`alphabet::piece_starts`].
    pub seq_starts: Vec<u32>,
}

/// Name every input sequence, group identical ones and concatenate one representative per
/// class. The caller has already rejected an empty `sequences` slice.
pub(crate) fn prepare_text(sequences: &[DnaSequence]) -> Result<PreparedText, FmIndexError> {
    // Counts over the deduplicated index include every class member, so the original
    // (undeduplicated) text must still fit the u32 they are reported in.
    let original_len: usize = sequences.iter().map(|s| s.len() + 1).sum();
    if original_len > u32::MAX as usize {
        return Err(FmIndexError::TextTooLarge(original_len));
    }

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

    let (seq_classes, reps) = SeqClasses::classify(sequences);
    let parts: Vec<&[u8]> = reps.iter().map(|&i| sequences[i].as_slice()).collect();
    let (text, slot_boundaries) = alphabet::concatenate_parts(&parts)?;

    Ok(PreparedText {
        text_len: text.len() as u32,
        text,
        num_sequences: sequences.len() as u32,
        slot_boundaries,
        seq_headers,
        header_index,
        seq_classes,
    })
}

/// Run the shared construction prefix on the CPU. The caller has already rejected an empty
/// `sequences` slice.
pub(crate) fn prepare_cpu(sequences: &[DnaSequence]) -> Result<Prepared, FmIndexError> {
    let PreparedText {
        text,
        text_len,
        num_sequences,
        slot_boundaries,
        seq_headers,
        header_index,
        seq_classes,
    } = prepare_text(sequences)?;

    // C array from text before SA construction — BWT is a permutation of text,
    // so character frequencies are identical. Avoids a second n-byte scan of BWT.
    let c_array = CArray::from_text(&text);

    // Build suffix array (single-threaded; psacak has no parallel API)
    let sa = build_suffix_array(&text);

    let dup_weights = DupWeights::build(&sa.data, &slot_boundaries, &seq_classes);

    let seq_starts = alphabet::piece_starts(&text);

    Ok(Prepared {
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
