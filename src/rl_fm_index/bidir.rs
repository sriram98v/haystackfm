//! [`BidirRlFmIndex`]: a forward and a reverse run-length FM-index driving the shared
//! [`BidirInterval`] cursor.

use super::{RlFmIndex, RlFmIndexConfig};
use crate::alphabet::{Alphabet, DnaSequence, IupacDna, SymbolSet, ALPHABET_SIZE};
use crate::error::FmIndexError;
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::bidir_index::{reverse_as_sequence, BidirFmIndex};
use crate::fm_index::prepare::{in_build_pool, prepare_cpu};
use crate::fm_index::seq_id::SeqId;

/// Bidirectional run-length FM-index: [`RlFmIndex`] halves over the text `T` (`fwd`) and
/// its reversal `T^R` (`rev`), the run-length counterpart of [`BidirFmIndex`].
///
/// Cursor steps are the same [`BidirInterval`] methods (`extend_*`, `children_*`,
/// `count_*_in`, `*_compatible`) running on run-length rank. Intervals are identical to a
/// [`BidirFmIndex`] over the same references, since both halves index the same suffix
/// arrays. Locate uses the forward half's SA samples; the reverse half carries none.
///
/// Not yet supported compared with [`BidirFmIndex`]: cursor contraction (needs the LCP),
/// k-mer `lookup_interval`, and SMEM/MEM finding.
#[derive(Debug, Clone)]
pub struct BidirRlFmIndex {
    pub(crate) fwd: RlFmIndex,
    pub(crate) rev: RlFmIndex,
}

impl BidirRlFmIndex {
    /// Build on the CPU with the [`IupacDna`] alphabet.
    pub fn build_cpu(
        sequences: &[DnaSequence],
        config: &RlFmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        Self::build_cpu_with::<IupacDna>(sequences, config)
    }

    /// Build on the CPU with a custom [`Alphabet`].
    pub fn build_cpu_with<A: Alphabet>(
        sequences: &[DnaSequence],
        config: &RlFmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }
        in_build_pool(config.build_threads, || {
            let prepared = prepare_cpu(sequences)?;
            // Reverse half: the byte-reversal of the same concatenated text, as one
            // sequence (see `BidirFmIndex::build_cpu_with`). It never locates.
            let rev_seq = reverse_as_sequence(&prepared.text)?;
            let fwd = RlFmIndex::from_prepared(prepared, A::fns(), config);
            let rev_config = RlFmIndexConfig {
                sa_sample_rate: 0,
                ..config.clone()
            };
            let rev = RlFmIndex::from_prepared(prepare_cpu(&[rev_seq])?, A::fns(), &rev_config);
            Ok(Self { fwd, rev })
        })
    }

    /// Convert an existing [`BidirFmIndex`] without rebuilding its suffix arrays. The
    /// reverse half's SA samples are dropped.
    pub fn from_bidir_fm_index(bidir: &BidirFmIndex) -> Self {
        let mut rev = RlFmIndex::from_fm_index(bidir.rev());
        rev.sa_samples = None;
        Self {
            fwd: RlFmIndex::from_fm_index(bidir.fwd()),
            rev,
        }
    }

    /// The forward half (index of `T`); `BidirInterval::fwd_*` rows refer to its SA.
    pub fn fwd(&self) -> &RlFmIndex {
        &self.fwd
    }

    /// The reverse half (index of `T^R`), count-only.
    pub fn rev(&self) -> &RlFmIndex {
        &self.rev
    }

    /// The interval of the empty pattern.
    pub fn full_interval(&self) -> BidirInterval {
        BidirInterval::full(self.fwd.text_len)
    }

    /// Extend right by `c` (P → Pc); `None` when Pc does not occur.
    pub fn extend_right(&self, iv: BidirInterval, c: u8) -> Option<BidirInterval> {
        iv.extend_right(c, &self.rev)
    }

    /// Extend left by `c` (P → cP); `None` when cP does not occur.
    pub fn extend_left(&self, iv: BidirInterval, c: u8) -> Option<BidirInterval> {
        iv.extend_left(c, &self.fwd)
    }

    /// Occurrences of `iv`'s pattern followed by a symbol in `set`.
    pub fn count_right_in(&self, iv: &BidirInterval, set: SymbolSet) -> u32 {
        iv.count_right_in(set, &self.rev)
    }

    /// Occurrences of `iv`'s pattern preceded by a symbol in `set`.
    pub fn count_left_in(&self, iv: &BidirInterval, set: SymbolSet) -> u32 {
        iv.count_left_in(set, &self.fwd)
    }

    /// Occurrences followed by an ambiguity code (codes 5..=15).
    pub fn count_wild_right(&self, iv: &BidirInterval) -> u32 {
        iv.count_wild_right(&self.rev)
    }

    /// Occurrences preceded by an ambiguity code (codes 5..=15).
    pub fn count_wild_left(&self, iv: &BidirInterval) -> u32 {
        iv.count_wild_left(&self.fwd)
    }

    /// Every right child at once: `children_right(iv)[c] == extend_right(iv, c)`.
    pub fn children_right(&self, iv: &BidirInterval) -> [Option<BidirInterval>; ALPHABET_SIZE] {
        iv.children_right(&self.rev)
    }

    /// Every left child at once: `children_left(iv)[c] == extend_left(iv, c)`.
    pub fn children_left(&self, iv: &BidirInterval) -> [Option<BidirInterval>; ALPHABET_SIZE] {
        iv.children_left(&self.fwd)
    }

    /// Extend right by every reference code query code `q` matches, yielding the
    /// non-empty children.
    pub fn extend_right_compatible(
        &self,
        iv: BidirInterval,
        q: u8,
    ) -> impl Iterator<Item = BidirInterval> + '_ {
        iv.extend_right_compatible(q, &self.rev)
    }

    /// Extend left by every reference code query code `q` matches.
    pub fn extend_left_compatible(
        &self,
        iv: BidirInterval,
        q: u8,
    ) -> impl Iterator<Item = BidirInterval> + '_ {
        iv.extend_left_compatible(q, &self.fwd)
    }

    /// The reference codes query code `q` matches under this index's alphabet.
    pub fn compatible_set(&self, q: u8) -> SymbolSet {
        self.fwd.alphabet_fns.compatible_set(q)
    }

    /// Count occurrences of `pattern` (backward search on the forward half).
    pub fn count(&self, pattern: &[u8]) -> u32 {
        self.fwd.count(pattern)
    }

    /// Locate every occurrence of `iv`'s pattern via the forward SA samples.
    ///
    /// # Panics
    /// If the forward half is count-only (built with `sa_sample_rate == 0`).
    pub fn locate_interval(&self, iv: &BidirInterval) -> Vec<(SeqId, u32)> {
        self.fwd.locate_rows(iv.fwd_lo, iv.fwd_hi)
    }

    /// Total text length (sentinels included).
    pub fn text_len(&self) -> u32 {
        self.fwd.text_len
    }

    /// Number of sequences indexed.
    pub fn num_sequences(&self) -> u32 {
        self.fwd.num_sequences
    }

    /// FASTA headers in build order.
    pub fn seq_headers(&self) -> &[String] {
        self.fwd.seq_headers()
    }

    /// Header for a sequence id.
    pub fn seq_header(&self, id: SeqId) -> Option<&str> {
        self.fwd.seq_header(id)
    }

    /// Id for a header.
    pub fn seq_id(&self, header: &str) -> Option<SeqId> {
        self.fwd.seq_id(header)
    }

    /// Heap bytes of both halves.
    pub fn heap_size_bytes(&self) -> usize {
        self.fwd.heap_size_bytes() + self.rev.heap_size_bytes()
    }
}

impl From<&BidirFmIndex> for BidirRlFmIndex {
    fn from(bidir: &BidirFmIndex) -> Self {
        Self::from_bidir_fm_index(bidir)
    }
}
