//! [`BidirRlFmIndex`]: a forward and a reverse run-length FM-index driving the shared
//! [`BidirInterval`] cursor.

use super::{RlFmIndex, RlFmIndexConfig};
use crate::alphabet::{Alphabet, DnaSequence, IupacDna, SymbolSet, ALPHABET_SIZE};
use crate::error::FmIndexError;
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::bidir_index::{reverse_as_sequence, BidirFmIndex};
use crate::fm_index::fwd_interval::FwdInterval;
use crate::fm_index::prepare::{in_build_pool, prepare_cpu};
use crate::fm_index::seq_id::SeqId;
use crate::fm_index::smem::{BidirSearch, Mem, RawMem, ShallowCache};

/// Bidirectional run-length FM-index: [`RlFmIndex`] halves over the text `T` (`fwd`) and
/// its reversal `T^R` (`rev`), the run-length counterpart of [`BidirFmIndex`].
///
/// Cursor steps are the same [`BidirInterval`] methods (`extend_*`, `children_*`,
/// `count_*_in`, `*_compatible`) running on run-length rank. Intervals are identical to a
/// [`BidirFmIndex`] over the same references, since both halves index the same suffix
/// arrays. Locate uses the forward half's SA samples; the reverse half carries none.
///
/// Not yet supported compared with [`BidirFmIndex`]: cursor contraction and `parent_fwd`
/// (both need the LCP), and the GPU MEM/SMEM kernels.
#[derive(Debug, Clone)]
pub struct BidirRlFmIndex {
    pub(crate) fwd: RlFmIndex,
    pub(crate) rev: RlFmIndex,
    /// Shallow k-mer interval sets seeding the SMEM/MEM sweeps. Derived from the halves.
    pub(crate) shallow: ShallowCache,
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
            let fwd = RlFmIndex::from_prepared(prepared, A::fns(), config, true);
            let rev_config = RlFmIndexConfig {
                sa_sample_rate: 0,
                ..config.clone()
            };
            // All sequence accessors delegate to the forward half.
            let rev =
                RlFmIndex::from_prepared(prepare_cpu(&[rev_seq])?, A::fns(), &rev_config, false);
            Ok(Self::from_halves(fwd, rev))
        })
    }

    /// Build with GPU acceleration: a [`BidirFmIndex`] is built on the GPU and converted
    /// (see [`from_bidir_fm_index`](Self::from_bidir_fm_index)).
    #[cfg(feature = "gpu")]
    pub async fn build(
        sequences: &[DnaSequence],
        config: &RlFmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        let fm = BidirFmIndex::build(sequences, &config.gpu_fm_config()).await?;
        let mut rl = Self::from_bidir_fm_index(&fm);
        if config.sa_sample_rate == 0 {
            rl.fwd.sa_samples = None;
        }
        rl.fwd.ensure_lookup(config.lookup_depth);
        rl.rev.ensure_lookup(config.lookup_depth);
        Ok(rl)
    }

    /// Convert an existing [`BidirFmIndex`] without rebuilding its suffix arrays. The
    /// reverse half's SA samples are dropped.
    pub fn from_bidir_fm_index(bidir: &BidirFmIndex) -> Self {
        let mut rev = RlFmIndex::convert(bidir.rev(), false);
        rev.sa_samples = None;
        Self::from_halves(RlFmIndex::from_fm_index(bidir.fwd()), rev)
    }

    /// Pair two halves and derive the sweep cache.
    pub(crate) fn from_halves(fwd: RlFmIndex, rev: RlFmIndex) -> Self {
        let shallow = ShallowCache::build(BidirInterval::full(fwd.text_len), &rev, fwd.text_len);
        Self { fwd, rev, shallow }
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

    /// Occurrences of `iv`'s pattern (its size).
    pub fn count_interval(&self, iv: &BidirInterval) -> u32 {
        iv.size()
    }

    /// LF step on a forward row range (any sub-range of an interval): see
    /// [`FwdInterval::extend_left`].
    pub fn extend_left_fwd(&self, iv: &FwdInterval, c: u8) -> Option<FwdInterval> {
        iv.extend_left(c, &self.fwd)
    }

    /// Locate every row of a forward interval via the forward SA samples.
    ///
    /// # Panics
    /// If the forward half is count-only.
    pub fn locate_fwd(&self, iv: &FwdInterval) -> Vec<(SeqId, u32)> {
        self.fwd.locate_rows(iv.lo, iv.hi)
    }

    /// Depth of the k-mer lookup tables (`RlFmIndexConfig::lookup_depth`), or 0 when the
    /// halves carry none (or disagree).
    pub fn lookup_depth(&self) -> u32 {
        match (&self.fwd.lookup, &self.rev.lookup) {
            (Some(f), Some(r)) if f.depth == r.depth => f.depth,
            _ => 0,
        }
    }

    /// The cursor of `kmer` straight from the lookup tables: see
    /// [`BidirFmIndex::lookup_interval`].
    pub fn lookup_interval(&self, kmer: &[u8]) -> Option<BidirInterval> {
        let depth = self.lookup_depth();
        if depth == 0 || kmer.len() != depth as usize {
            return None;
        }
        let &(fwd_lo, fwd_hi) = self.fwd.lookup.as_ref()?.get(kmer)?.intervals.first()?;
        if fwd_lo >= fwd_hi {
            return None;
        }
        let reversed: Vec<u8> = kmer.iter().rev().copied().collect();
        let &(rev_lo, rev_hi) = self
            .rev
            .lookup
            .as_ref()?
            .get(&reversed)?
            .intervals
            .first()?;
        // Tables from one build always agree; mismatched ones (a corrupted file) give no
        // cursor rather than one of inconsistent widths.
        if fwd_hi - fwd_lo != rev_hi.saturating_sub(rev_lo) {
            return None;
        }
        Some(BidirInterval {
            fwd_lo,
            fwd_hi,
            rev_lo,
            rev_hi,
            len: depth,
        })
    }

    /// SMEMs of `query`: see [`BidirFmIndex::find_smems`]. Identical results, run on
    /// run-length rank.
    ///
    /// # Panics
    /// With `locate = true` on a count-only index.
    pub fn find_smems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
        self.smems(query, min_len, locate)
    }

    /// MEMs of `query`: see [`BidirFmIndex::find_mems`]. Identical results, run on
    /// run-length rank.
    ///
    /// # Panics
    /// With `locate = true` on a count-only index.
    pub fn find_mems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
        self.mems(query, min_len, locate)
    }

    /// Bases of sequence `id`, extracted from the forward BWT: see
    /// [`RlFmIndex::sequence`].
    pub fn sequence(&self, id: SeqId) -> Option<Vec<u8>> {
        self.fwd.sequence(id)
    }

    /// Bases of the sequence carrying `header`.
    pub fn sequence_by_header(&self, header: &str) -> Option<Vec<u8>> {
        self.fwd.sequence_by_header(header)
    }

    /// Exclusive end of each reference (its sentinel included) in the concatenated text.
    pub fn seq_boundaries(&self) -> &[u32] {
        self.fwd.seq_boundaries()
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
        self.fwd.heap_size_bytes() + self.rev.heap_size_bytes() + self.shallow.heap_size_bytes()
    }
}

impl BidirSearch for BidirRlFmIndex {
    type Half = RlFmIndex;

    fn fwd_half(&self) -> &RlFmIndex {
        &self.fwd
    }

    fn rev_half(&self) -> &RlFmIndex {
        &self.rev
    }

    fn search_full_interval(&self) -> BidirInterval {
        self.full_interval()
    }

    fn search_locate_rows(&self, lo: u32, hi: u32) -> Vec<(SeqId, u32)> {
        self.fwd.locate_rows(lo, hi)
    }

    fn shallow(&self, kmer: &[u8]) -> Option<&[BidirInterval]> {
        self.shallow.get(kmer)
    }

    fn locate_all(&self, raws: Vec<RawMem>, locate: bool) -> Vec<Mem> {
        self.locate_all_batched(raws, locate)
    }

    fn collect_smem_candidates(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
        self.collect_smem_candidates_lockstep(query, min_len)
    }

    fn collect_mem_candidates(&self, query: &[u8], min_len: usize) -> Vec<RawMem> {
        self.collect_mem_candidates_pruned(query, min_len)
    }

    fn search_locate_row_list(&self, rows: &[u32]) -> Vec<(SeqId, u32)> {
        self.fwd.locate_row_list(rows)
    }

    fn left_maximal_rows(&self, ivs: &[BidirInterval], compat: SymbolSet) -> Vec<u32> {
        let mut rows = Vec::new();
        for iv in ivs {
            self.fwd
                .bwt
                .rows_not_in(compat, iv.fwd_lo, iv.fwd_hi, &mut rows);
        }
        rows
    }
}

impl From<&BidirFmIndex> for BidirRlFmIndex {
    fn from(bidir: &BidirFmIndex) -> Self {
        Self::from_bidir_fm_index(bidir)
    }
}
