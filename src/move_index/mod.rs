//! Move-structure indexes: [`MoveFmIndex`] (an r-index on move tables) and [`BMoveIndex`]
//! (b-move, its bidirectional form).
//!
//! **LF as a table step.** The BWT's LF mapping sends each run to a contiguous range, so a
//! move table ([`crate::move_table`]) with one row per run answers LF with a row read and a
//! short fast-forward, and every search position carries the index of its row. Backward
//! search looks up the first and last row of the extension symbol inside the interval
//! (a short walk, falling back to rank/select over the run heads) and steps both.
//!
//! **Locate in `O(r)` space.** SA values are stored only at the first and last row of each
//! run. Every extension keeps a *toehold* — the text position of one occurrence of the
//! current pattern: taken from a sampled row holding the extension symbol inside the
//! interval (minus one), or, when the interval lies inside a single run, carried from the
//! previous toehold. The toehold may sit anywhere in the final interval (always so after
//! bidirectional search), so locate walks `φ` (the previous row's SA value) while the
//! permuted LCP says the previous suffix still starts with the pattern, and `φ⁻¹` likewise
//! downwards. `φ`, `φ⁻¹` and the permuted LCP are piecewise linear over `O(r)` pieces and
//! are stored as balanced move tables plus one LCP value per piece (Gagie, Navarro &
//! Prezza's r-index; Arakawa et al.'s bidirectional toehold; Depuydt et al.'s b-move).
//!
//! **Balancing.** `φ` / `φ⁻¹` tables are balanced, which bounds each locate step. LF tables
//! keep one row per run (as in b-move and Movi, where balancing them showed no gain), so an
//! LF step's fast-forward is short in practice but not bounded on adversarial inputs.
//!
//! **Sentinels.** All sequence ends share one sentinel code, which makes LF ambiguous at
//! the wrap-around from text position 0. Toeholds only step through non-sentinel symbols,
//! and every sentinel row is its own move-table row (hence sampled), which keeps `φ` and
//! the permuted LCP piecewise linear around sequence boundaries.

mod bmove;
mod lf;
mod locate;
mod serialize;

pub use bmove::{BMoveCursor, BMoveIndex};
pub use lf::MoveLf;
pub use serialize::{BMOVE_FORMAT_MAGIC, FORMAT_MAGIC};

use lf::Span;
use locate::MoveLocate;

use crate::alphabet::{Alphabet, AlphabetFns, DnaSequence, IupacDna};
use crate::bwt::cpu::build_bwt;
use crate::error::FmIndexError;
use crate::fm_index::prepare::{in_build_pool, prepare_cpu, Prepared};
use crate::fm_index::seq_classes::{DupWeights, SeqClasses};
use crate::fm_index::seq_id::{map_text_position, HeaderIndex, SeqId};

/// Toehold value meaning "no known occurrence" (count-only index, sentinel extension, or
/// a corrupted index); locate enumerates nothing from it.
pub(crate) const UNKNOWN_TOEHOLD: u32 = u32::MAX;

/// Configuration for move-structure index construction.
#[derive(Debug, Clone)]
pub struct MoveFmIndexConfig {
    /// Build the locate structures (run-boundary SA samples, `φ` / `φ⁻¹` tables, permuted
    /// LCP samples). Without them the index counts only and `locate*` panic. Default: true.
    pub locate: bool,
    /// Worker threads for construction (native targets). Default: 1.
    pub build_threads: u16,
}

impl Default for MoveFmIndexConfig {
    fn default() -> Self {
        Self {
            locate: true,
            build_threads: 1,
        }
    }
}

/// An r-index on move tables: `count` and `locate` in space proportional to the number of
/// BWT runs. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct MoveFmIndex {
    pub(crate) lf: MoveLf,
    pub(crate) locate: Option<MoveLocate>,
    pub(crate) text_len: u32,
    /// Number of input sequences, duplicates included.
    pub(crate) num_sequences: u32,
    /// Ends of the stored (distinct) sequences, sentinel included. See
    /// [`seq_classes`](crate::fm_index::seq_classes).
    pub(crate) slot_boundaries: Vec<u32>,
    /// Header of every input sequence, indexed by [`SeqId`].
    pub(crate) seq_headers: Vec<String>,
    pub(crate) header_index: HeaderIndex,
    /// Classes of identical input sequences; one representative of each is stored.
    pub(crate) seq_classes: SeqClasses,
    /// Extra multiplicity of rows inside duplicated representatives, for counts.
    pub(crate) dup_weights: DupWeights,
    pub(crate) alphabet_fns: AlphabetFns,
}

impl MoveFmIndex {
    /// Build on the CPU with the [`IupacDna`] alphabet.
    pub fn build_cpu(
        sequences: &[DnaSequence],
        config: &MoveFmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        Self::build_cpu_with::<IupacDna>(sequences, config)
    }

    /// Build on the CPU with a custom [`Alphabet`].
    pub fn build_cpu_with<A: Alphabet>(
        sequences: &[DnaSequence],
        config: &MoveFmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }
        in_build_pool(config.build_threads, || {
            let prepared = prepare_cpu(sequences)?;
            Ok(Self::from_prepared(&prepared, A::fns(), config.locate))
        })
    }

    pub(crate) fn from_prepared(p: &Prepared, alphabet_fns: AlphabetFns, locate: bool) -> Self {
        let bwt = build_bwt(&p.text, &p.sa);
        let lf = MoveLf::from_bwt(&bwt, &p.c_array);
        drop(bwt);
        let locate = locate.then(|| MoveLocate::build(&lf.table, &p.text, &p.sa.data));
        Self {
            lf,
            locate,
            text_len: p.text_len,
            num_sequences: p.num_sequences,
            slot_boundaries: p.slot_boundaries.clone(),
            seq_headers: p.seq_headers.clone(),
            header_index: p.header_index.clone(),
            seq_classes: p.seq_classes.clone(),
            dup_weights: p.dup_weights.clone(),
            alphabet_fns,
        }
    }

    /// Count occurrences of `pattern` (alphabet codes), matching codes under the index's
    /// alphabet as [`FmIndex::count`](crate::FmIndex::count) does, in every member of a
    /// class of identical input sequences.
    pub fn count(&self, pattern: &[u8]) -> u32 {
        self.search(pattern)
            .iter()
            .map(|(span, _)| self.dup_weights.weighted_size(span.s, span.e + 1))
            .sum()
    }

    /// Locate every occurrence of `pattern` as `(sequence id, offset in that sequence)`,
    /// in no particular order. An occurrence in a class of identical sequences is reported
    /// once per member, consecutively.
    ///
    /// # Panics
    /// If the index was built without locate support ([`can_locate`](Self::can_locate)).
    pub fn locate(&self, pattern: &[u8]) -> Vec<(SeqId, u32)> {
        self.seq_classes.expand(self.locate_grouped(pattern))
    }

    /// [`locate`](Self::locate) with one hit per occurrence in the stored text, naming the
    /// representative of a class of identical sequences.
    ///
    /// # Panics
    /// If the index was built without locate support.
    pub fn locate_grouped(&self, pattern: &[u8]) -> Vec<(SeqId, u32)> {
        self.locate_positions(pattern)
            .into_iter()
            .map(|pos| {
                self.map_position(pos)
                    .expect("located position must be within text bounds")
            })
            .collect()
    }

    /// Locate every occurrence of `pattern` as absolute offsets into the concatenated
    /// text, in no particular order: positions in the stored sequences (class
    /// representatives) only.
    ///
    /// # Panics
    /// If the index was built without locate support.
    pub fn locate_positions(&self, pattern: &[u8]) -> Vec<u32> {
        let locate = self.locate_support();
        let mut out = Vec::new();
        for (span, toehold) in self.search(pattern) {
            locate.enumerate(toehold, pattern.len() as u32, span.size(), &mut out);
        }
        out
    }

    /// Backward search with IUPAC fan-out: one `(span, toehold)` per distinct matching
    /// reference string. Spans are not merged — each keeps the toehold of its own string.
    fn search(&self, pattern: &[u8]) -> Vec<(Span, u32)> {
        let Some(full) = self.lf.full_span() else {
            return vec![];
        };
        let first = self
            .locate
            .as_ref()
            .map_or(UNKNOWN_TOEHOLD, MoveLocate::sa_first);
        let mut states = vec![(full, first)];
        let mut next = Vec::new();
        let present = self.lf.present();
        for &q in pattern.iter().rev() {
            let compat = self.alphabet_fns.compatible(q) & present;
            next.clear();
            for (span, x) in &states {
                for c in compat.iter() {
                    if let Some((span, toehold)) = self.lf.extend(c, span) {
                        next.push((span, self.next_toehold(toehold, *x)));
                    }
                }
            }
            std::mem::swap(&mut states, &mut next);
            if states.is_empty() {
                break;
            }
        }
        states
    }

    /// Toehold after a backward step: a sampled row's SA value minus one, or the carried
    /// toehold minus one (both `>= 1` for a non-sentinel step on a valid index), else
    /// [`UNKNOWN_TOEHOLD`] — also on an index without locate support.
    #[inline]
    fn next_toehold(&self, toehold: lf::Toehold, x: u32) -> u32 {
        let Some(locate) = &self.locate else {
            return UNKNOWN_TOEHOLD;
        };
        locate
            .samples
            .get(toehold)
            .unwrap_or(x)
            .checked_sub(1)
            .unwrap_or(UNKNOWN_TOEHOLD)
    }

    fn locate_support(&self) -> &MoveLocate {
        self.locate
            .as_ref()
            .expect("locate on a MoveFmIndex built with locate = false")
    }

    /// True when the index carries the locate structures.
    pub fn can_locate(&self) -> bool {
        self.locate.is_some()
    }

    /// Map an absolute text offset to `(sequence id, offset within it)`; the id is that of
    /// the stored sequence, the representative of its class.
    pub fn map_position(&self, text_pos: u32) -> Option<(SeqId, u32)> {
        map_text_position(&self.slot_boundaries, &self.seq_classes, text_pos)
    }

    /// Total text length (sentinels included).
    pub fn text_len(&self) -> u32 {
        self.text_len
    }

    /// Number of sequences indexed: every input sequence, duplicates included.
    pub fn num_sequences(&self) -> u32 {
        self.num_sequences
    }

    /// Number of distinct sequences stored in the text.
    pub fn num_distinct_sequences(&self) -> u32 {
        self.slot_boundaries.len() as u32
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

    /// FASTA headers in build order.
    pub fn seq_headers(&self) -> &[String] {
        &self.seq_headers
    }

    /// Header for a sequence id.
    pub fn seq_header(&self, id: SeqId) -> Option<&str> {
        self.seq_headers.get(id.index()).map(String::as_str)
    }

    /// Id for a header.
    pub fn seq_id(&self, header: &str) -> Option<SeqId> {
        self.header_index.get(header)
    }

    /// Exclusive end of each **stored** reference (sentinel included) in the concatenated
    /// text, indexed by stored slot (see [`FmIndex::seq_boundaries`](crate::FmIndex::seq_boundaries)).
    pub fn seq_boundaries(&self) -> &[u32] {
        &self.slot_boundaries
    }

    /// The matching semantics this index was built with.
    pub fn alphabet_fns(&self) -> &AlphabetFns {
        &self.alphabet_fns
    }

    /// The LF move structure.
    pub fn lf(&self) -> &MoveLf {
        &self.lf
    }

    /// Heap bytes of the LF structure, the locate structures and the metadata.
    pub fn heap_size_bytes(&self) -> usize {
        self.lf.heap_size_bytes()
            + self.locate.as_ref().map_or(0, MoveLocate::heap_size_bytes)
            + self.slot_boundaries.len() * 4
            + self.seq_classes.heap_size_bytes()
            + self.dup_weights.heap_size_bytes()
            + self.seq_headers.iter().map(String::len).sum::<usize>()
    }
}
