//! Equivalence classes of identical input sequences.
//!
//! Every index builder groups its input sequences whose encoded bases are byte-identical
//! (`#` record delimiters included, headers ignored) into one class, and stores only the
//! first member of each class, its **representative**, in the indexed text. A sequence
//! that is merely a prefix or substring of another stays in a class of its own.
//!
//! [`SeqId`]s keep their meaning, the position of a sequence in the builder's input, for
//! every input sequence, stored or not. Query results that name a sequence report a match
//! in a representative once per class member (`SeqClasses::expand`), and counts include
//! every member (`DupWeights`); the `*_grouped` query variants report representatives only.
//!
//! Terminology used across the crate:
//! - a **slot** is one stored sequence, numbered in text order; slot `s` spans
//!   `slot_boundaries[s - 1] .. slot_boundaries[s]` (sentinel included) of the indexed text;
//! - slots are numbered in order of first occurrence, so slot order is `SeqId` order of the
//!   representatives, and a class's representative is its smallest member.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::alphabet::DnaSequence;
use crate::fm_index::seq_id::SeqId;

/// The partition of an index's input sequences into classes of identical sequences.
///
/// With no duplicates in the input every class is a singleton
/// ([`is_identity`](Self::is_identity)) and results are exactly those of an index over the
/// same sequences without deduplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqClasses {
    /// `seq_to_slot[id]` is the stored slot of sequence `id`. The only serialized field.
    seq_to_slot: Vec<u32>,
    /// `slot_rep[s]` is the representative (smallest member) of slot `s`.
    slot_rep: Vec<SeqId>,
    /// CSR offsets into `members`, `num_slots + 1` long.
    member_offsets: Vec<u32>,
    /// Members of every slot, ascending within a slot, slots in order.
    members: Vec<SeqId>,
}

impl SeqClasses {
    /// Every one of `n` sequences in a class of its own.
    #[cfg(test)]
    pub(crate) fn identity(n: u32) -> Self {
        Self::from_canonical((0..n).collect(), n as usize)
    }

    /// Group `sequences` by their encoded bases. Returns the classes and the input index of
    /// each slot's representative, in slot order (the order to concatenate them in).
    pub(crate) fn classify(sequences: &[DnaSequence]) -> (Self, Vec<usize>) {
        let mut slot_by_bases: HashMap<&[u8], u32> = HashMap::with_capacity(sequences.len());
        let mut reps = Vec::new();
        let seq_to_slot = sequences
            .iter()
            .enumerate()
            .map(|(i, seq)| {
                let next = reps.len() as u32;
                let slot = *slot_by_bases.entry(seq.as_slice()).or_insert(next);
                if slot == next {
                    reps.push(i);
                }
                slot
            })
            .collect();
        let num_slots = reps.len();
        (Self::from_canonical(seq_to_slot, num_slots), reps)
    }

    /// Rebuild from the serialized `seq_to_slot` table of an index storing `num_slots`
    /// sequences, checking it is a canonical numbering: slots appear in order of first
    /// occurrence and every slot has at least one member.
    pub(crate) fn from_seq_to_slot(
        seq_to_slot: Vec<u32>,
        num_slots: usize,
    ) -> Result<Self, String> {
        let mut next = 0u32;
        for (id, &slot) in seq_to_slot.iter().enumerate() {
            if slot > next {
                return Err(format!(
                    "sequence {id} is in slot {slot} before slot {next} has a member"
                ));
            }
            if slot == next {
                next += 1;
            }
        }
        if next as usize != num_slots {
            return Err(format!(
                "sequence classes name {next} stored sequences, the index stores {num_slots}"
            ));
        }
        Ok(Self::from_canonical(seq_to_slot, num_slots))
    }

    /// Derive the lookup tables from a canonical `seq_to_slot`.
    fn from_canonical(seq_to_slot: Vec<u32>, num_slots: usize) -> Self {
        let mut member_offsets = vec![0u32; num_slots + 1];
        for &slot in &seq_to_slot {
            member_offsets[slot as usize + 1] += 1;
        }
        for s in 0..num_slots {
            member_offsets[s + 1] += member_offsets[s];
        }
        let mut fill = member_offsets[..num_slots].to_vec();
        let mut members = vec![SeqId::new(0); seq_to_slot.len()];
        for (id, &slot) in seq_to_slot.iter().enumerate() {
            members[fill[slot as usize] as usize] = SeqId::new(id as u32);
            fill[slot as usize] += 1;
        }
        let slot_rep = member_offsets[..num_slots]
            .iter()
            .map(|&off| members[off as usize])
            .collect();
        Self {
            seq_to_slot,
            slot_rep,
            member_offsets,
            members,
        }
    }

    /// The serialized form: the stored slot of every sequence.
    pub(crate) fn seq_to_slot(&self) -> &[u32] {
        &self.seq_to_slot
    }

    /// True when no two input sequences were identical, so every class is a singleton.
    pub fn is_identity(&self) -> bool {
        self.slot_rep.len() == self.seq_to_slot.len()
    }

    /// Number of input sequences (every class member).
    pub fn num_sequences(&self) -> u32 {
        self.seq_to_slot.len() as u32
    }

    /// Number of classes, i.e. distinct sequences stored in the index.
    pub fn num_classes(&self) -> u32 {
        self.slot_rep.len() as u32
    }

    /// Stored slot of sequence `id`, or `None` when out of range.
    pub(crate) fn slot_of(&self, id: SeqId) -> Option<u32> {
        self.seq_to_slot.get(id.index()).copied()
    }

    /// Representative of stored slot `slot`. Panics when out of range.
    pub(crate) fn rep_of_slot(&self, slot: u32) -> SeqId {
        self.slot_rep[slot as usize]
    }

    /// Members of stored slot `slot`, ascending. Panics when out of range.
    pub(crate) fn members_of_slot(&self, slot: u32) -> &[SeqId] {
        let s = slot as usize;
        &self.members[self.member_offsets[s] as usize..self.member_offsets[s + 1] as usize]
    }

    /// Number of members of stored slot `slot`. Panics when out of range.
    pub(crate) fn slot_size(&self, slot: u32) -> u32 {
        let s = slot as usize;
        self.member_offsets[s + 1] - self.member_offsets[s]
    }

    /// Every sequence identical to `id` (itself included), ascending; empty when `id` is
    /// out of range.
    pub fn equivalence_class(&self, id: SeqId) -> &[SeqId] {
        match self.slot_of(id) {
            Some(slot) => self.members_of_slot(slot),
            None => &[],
        }
    }

    /// The member of `id`'s class that is stored in the index (its smallest id), or `None`
    /// when `id` is out of range.
    pub fn representative(&self, id: SeqId) -> Option<SeqId> {
        self.slot_of(id).map(|slot| self.rep_of_slot(slot))
    }

    /// Replace every hit `(rep, offset)` on a representative by one hit per member of its
    /// class, members ascending, in place of the original hit; the order of hits is
    /// otherwise kept. Returns `hits` untouched when every class is a singleton; a hit
    /// naming an out-of-range id is kept as is.
    pub(crate) fn expand(&self, hits: Vec<(SeqId, u32)>) -> Vec<(SeqId, u32)> {
        if self.is_identity() {
            return hits;
        }
        let mut out = Vec::with_capacity(hits.len());
        for (rep, off) in hits {
            match self.slot_of(rep) {
                Some(slot) => out.extend(self.members_of_slot(slot).iter().map(|&m| (m, off))),
                None => out.push((rep, off)),
            }
        }
        out
    }

    /// [`expand`](Self::expand) for hits naming stored slots (as the GPU reference map
    /// reports them): each in-range slot's hit becomes one hit per member of its class; an
    /// out-of-range slot (an unresolvable row) is kept as is, never expanded.
    pub(crate) fn expand_slots(&self, hits: Vec<(u32, u32)>) -> Vec<(SeqId, u32)> {
        if self.is_identity() {
            return hits
                .into_iter()
                .map(|(s, off)| (SeqId::new(s), off))
                .collect();
        }
        let mut out = Vec::with_capacity(hits.len());
        for (slot, off) in hits {
            if slot < self.num_classes() {
                out.extend(self.members_of_slot(slot).iter().map(|&m| (m, off)));
            } else {
                out.push((SeqId::new(slot), off));
            }
        }
        out
    }

    /// Resident heap bytes.
    pub(crate) fn heap_size_bytes(&self) -> usize {
        (self.seq_to_slot.capacity()
            + self.slot_rep.capacity()
            + self.member_offsets.capacity()
            + self.members.capacity())
            * std::mem::size_of::<u32>()
    }
}

/// Extra multiplicity of the suffix-array rows that fall inside a duplicated
/// representative, so counts can include every class member without a locate.
///
/// A forward SA row stands for `k` occurrences when its suffix starts in a stored sequence
/// (trailing sentinel included) whose class has `k` members, and for one otherwise. Only
/// rows with `k > 1` are listed, grouped by `k`, so an index over distinct sequences carries
/// an empty table.
///
/// Costs 4 bytes per base of duplicated representatives, resident and serialized, in every
/// index type: for the run-length and move indexes, whose rank structures shrink with the
/// run count, this table does not, and on a collection with large duplicated sequences it
/// can dominate their size. A count costs two binary searches per distinct class size.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) struct DupWeights {
    /// One list per class size `k >= 2` present, `k` ascending.
    classes: Vec<DupClass>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DupClass {
    /// Class size: every listed row stands for `k` occurrences.
    k: u32,
    /// Forward SA rows, ascending.
    #[serde(with = "crate::serde_raw::u32s")]
    rows: Vec<u32>,
}

impl DupWeights {
    /// Weights for the suffix array `sa` of a text whose stored sequences end at
    /// `slot_boundaries` and are grouped by `classes`. Empty when `classes` is the identity.
    pub(crate) fn build(sa: &[u32], slot_boundaries: &[u32], classes: &SeqClasses) -> Self {
        if classes.is_identity() {
            return Self::default();
        }
        let sizes: Vec<u32> = (0..slot_boundaries.len() as u32)
            .map(|s| classes.slot_size(s))
            .collect();
        let mut by_k: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (row, &pos) in sa.iter().enumerate() {
            let slot = slot_boundaries.partition_point(|&b| b <= pos);
            let k = sizes[slot];
            if k > 1 {
                by_k.entry(k).or_default().push(row as u32);
            }
        }
        Self {
            classes: by_k
                .into_iter()
                .map(|(k, rows)| DupClass { k, rows })
                .collect(),
        }
    }

    /// True when no row carries extra weight.
    pub(crate) fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// Occurrences that the forward rows `lo..hi` stand for: `hi - lo` plus the extra
    /// copies of rows inside duplicated representatives.
    #[inline]
    pub(crate) fn weighted_size(&self, lo: u32, hi: u32) -> u32 {
        if self.classes.is_empty() || hi <= lo {
            return hi.saturating_sub(lo);
        }
        let mut total = u64::from(hi - lo);
        for class in &self.classes {
            let n =
                class.rows.partition_point(|&r| r < hi) - class.rows.partition_point(|&r| r < lo);
            total += u64::from(class.k - 1) * n as u64;
        }
        // The original (undeduplicated) text fits in u32 (checked at build and by
        // `validate`), so any count over it does too.
        total as u32
    }

    /// Check the table against the index it was loaded with. Which rows are listed cannot be
    /// verified without the suffix array; a corrupt list yields wrong counts, never a panic.
    pub(crate) fn validate(
        &self,
        text_len: u32,
        slot_boundaries: &[u32],
        classes: &SeqClasses,
    ) -> Result<(), String> {
        // Expected number of rows per class size: the stored length of its slots. The
        // original text (every member) must fit in u32, as at build time, so counts cannot
        // overflow.
        let mut expected: BTreeMap<u32, u64> = BTreeMap::new();
        let mut original_len = 0u64;
        let mut start = 0u32;
        for (s, &end) in slot_boundaries.iter().enumerate() {
            let k = classes.slot_size(s as u32);
            let len = u64::from(end - start);
            original_len += u64::from(k) * len;
            if k > 1 {
                *expected.entry(k).or_default() += len;
            }
            start = end;
        }
        if original_len > u64::from(u32::MAX) {
            return Err(format!(
                "the sequences the index stands for span {original_len} positions, over u32::MAX"
            ));
        }
        if self.classes.len() != expected.len() {
            return Err(format!(
                "duplicate weights list {} class sizes, the sequence classes have {}",
                self.classes.len(),
                expected.len()
            ));
        }
        let mut all_rows: Vec<u32> = Vec::new();
        for (class, (&k, &len)) in self.classes.iter().zip(&expected) {
            if class.k != k {
                return Err(format!(
                    "duplicate weights for class size {}, expected {k}",
                    class.k
                ));
            }
            if class.rows.len() as u64 != len {
                return Err(format!(
                    "{} duplicate-weight rows for class size {k}, expected {len}",
                    class.rows.len()
                ));
            }
            if !class.rows.windows(2).all(|w| w[0] < w[1]) {
                return Err(format!(
                    "duplicate-weight rows for class size {k} are not strictly increasing"
                ));
            }
            if class.rows.last().is_some_and(|&r| r >= text_len) {
                return Err(format!(
                    "duplicate-weight row for class size {k} is past the text end {text_len}"
                ));
            }
            all_rows.extend_from_slice(&class.rows);
        }
        all_rows.sort_unstable();
        if !all_rows.windows(2).all(|w| w[0] < w[1]) {
            return Err("a row is listed under two class sizes".to_string());
        }
        Ok(())
    }

    /// Resident heap bytes.
    pub(crate) fn heap_size_bytes(&self) -> usize {
        self.classes.capacity() * std::mem::size_of::<DupClass>()
            + self
                .classes
                .iter()
                .map(|c| c.rows.capacity() * std::mem::size_of::<u32>())
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::concatenate_sequences;
    use crate::suffix_array::cpu::build_suffix_array;

    fn seqs(strs: &[&str]) -> Vec<DnaSequence> {
        strs.iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect()
    }

    fn ids(v: &[u32]) -> Vec<SeqId> {
        v.iter().copied().map(SeqId::new).collect()
    }

    #[test]
    fn classify_groups_identical_sequences_only() {
        // 0 = 2 = 5, 1 alone, 3 is a prefix of 0 (own class), 4 = 6 (with `#`).
        let input = seqs(&[
            "ACGTGA", "TTGCA", "ACGTGA", "ACGTG", "AC#GT", "ACGTGA", "AC#GT",
        ]);
        let (classes, reps) = SeqClasses::classify(&input);
        assert_eq!(reps, vec![0, 1, 3, 4]);
        assert_eq!(classes.seq_to_slot(), &[0, 1, 0, 2, 3, 0, 3]);
        assert!(!classes.is_identity());
        assert_eq!(classes.num_sequences(), 7);
        assert_eq!(classes.num_classes(), 4);
        assert_eq!(
            classes.equivalence_class(SeqId(5)),
            ids(&[0, 2, 5]).as_slice()
        );
        assert_eq!(classes.equivalence_class(SeqId(3)), ids(&[3]).as_slice());
        assert_eq!(classes.equivalence_class(SeqId(6)), ids(&[4, 6]).as_slice());
        assert_eq!(classes.equivalence_class(SeqId(7)), &[] as &[SeqId]);
        assert_eq!(classes.representative(SeqId(2)), Some(SeqId(0)));
        assert_eq!(classes.representative(SeqId(6)), Some(SeqId(4)));
        assert_eq!(classes.representative(SeqId(3)), Some(SeqId(3)));
        assert_eq!(classes.representative(SeqId(9)), None);
    }

    #[test]
    fn classify_distinguishes_delimiter_layouts() {
        let input = seqs(&["AC#GT", "ACG#T", "ACGT"]);
        let (classes, _) = SeqClasses::classify(&input);
        assert!(classes.is_identity());
    }

    #[test]
    fn distinct_input_is_identity() {
        let (classes, reps) = SeqClasses::classify(&seqs(&["A", "C", "G"]));
        assert!(classes.is_identity());
        assert_eq!(reps, vec![0, 1, 2]);
        assert_eq!(classes, SeqClasses::identity(3));
        let hits = vec![(SeqId(2), 5), (SeqId(0), 1)];
        assert_eq!(classes.expand(hits.clone()), hits);
    }

    #[test]
    fn expand_replaces_each_hit_in_place() {
        let (classes, _) = SeqClasses::classify(&seqs(&["AC", "GT", "AC", "AC"]));
        let hits = vec![(SeqId(1), 4), (SeqId(0), 2), (SeqId(1), 0)];
        assert_eq!(
            classes.expand(hits),
            vec![
                (SeqId(1), 4),
                (SeqId(0), 2),
                (SeqId(2), 2),
                (SeqId(3), 2),
                (SeqId(1), 0)
            ]
        );
    }

    #[test]
    fn expand_slots_names_members_and_keeps_unresolvable_slots() {
        let (classes, _) = SeqClasses::classify(&seqs(&["AC", "GT", "AC"]));
        let hits = vec![(0, 4), (1, 0), (9, 2)];
        assert_eq!(
            classes.expand_slots(hits),
            vec![(SeqId(0), 4), (SeqId(2), 4), (SeqId(1), 0), (SeqId(9), 2)]
        );
        let identity = SeqClasses::identity(2);
        assert_eq!(
            identity.expand_slots(vec![(1, 3), (7, 0)]),
            vec![(SeqId(1), 3), (SeqId(7), 0)]
        );
    }

    #[test]
    fn validate_rejects_counts_past_u32() {
        // Two copies of one 3-billion-position sequence: counts would wrap.
        let (classes, _) = SeqClasses::classify(&seqs(&["A", "A"]));
        let err = DupWeights::default()
            .validate(3_000_000_000, &[3_000_000_000], &classes)
            .unwrap_err();
        assert!(err.contains("u32::MAX"), "{err}");
    }

    #[test]
    fn from_seq_to_slot_round_trips_and_rejects_non_canonical() {
        let (classes, _) = SeqClasses::classify(&seqs(&["AC", "GT", "AC", "T"]));
        let back = SeqClasses::from_seq_to_slot(classes.seq_to_slot().to_vec(), 3).unwrap();
        assert_eq!(back, classes);
        // Slot 1 used before slot 0.
        assert!(SeqClasses::from_seq_to_slot(vec![1, 0], 2).is_err());
        // Slot 2 skipped.
        assert!(SeqClasses::from_seq_to_slot(vec![0, 1, 3], 4).is_err());
        // Wrong slot count.
        assert!(SeqClasses::from_seq_to_slot(vec![0, 1, 0], 3).is_err());
        assert!(SeqClasses::from_seq_to_slot(vec![0, 1, 0], 1).is_err());
    }

    /// Deduplicated text, slot boundaries, classes and suffix array of `strs`.
    fn dedup(strs: &[&str]) -> (Vec<u8>, Vec<u32>, SeqClasses, Vec<u32>) {
        let input = seqs(strs);
        let (classes, reps) = SeqClasses::classify(&input);
        let stored: Vec<DnaSequence> = reps.iter().map(|&i| input[i].clone()).collect();
        let (text, bounds) = concatenate_sequences(&stored).unwrap();
        let sa = build_suffix_array(&text).data;
        (text, bounds, classes, sa)
    }

    fn naive_weight(sa: &[u32], bounds: &[u32], classes: &SeqClasses, lo: u32, hi: u32) -> u32 {
        (lo..hi)
            .map(|r| {
                let slot = bounds.partition_point(|&b| b <= sa[r as usize]);
                classes.slot_size(slot as u32)
            })
            .sum()
    }

    #[test]
    fn weights_match_naive_per_row_sum() {
        let (text, bounds, classes, sa) =
            dedup(&["ACGTA", "GG", "ACGTA", "T#GA", "ACGTA", "T#GA", "C"]);
        let w = DupWeights::build(&sa, &bounds, &classes);
        assert!(!w.is_empty());
        let n = text.len() as u32;
        for lo in 0..=n {
            for hi in lo..=n {
                assert_eq!(
                    w.weighted_size(lo, hi),
                    naive_weight(&sa, &bounds, &classes, lo, hi),
                    "rows {lo}..{hi}"
                );
            }
        }
        // All rows: the original concatenated length.
        assert_eq!(w.weighted_size(0, n), 6 * 3 + 3 + 5 * 2 + 2);
        w.validate(n, &bounds, &classes).unwrap();
    }

    #[test]
    fn weights_are_empty_without_duplicates() {
        let (text, bounds, classes, sa) = dedup(&["ACGTA", "GG", "C"]);
        let w = DupWeights::build(&sa, &bounds, &classes);
        assert!(w.is_empty());
        assert_eq!(w.weighted_size(1, 4), 3);
        w.validate(text.len() as u32, &bounds, &classes).unwrap();
    }

    #[test]
    fn validate_rejects_corrupt_weights() {
        let (text, bounds, classes, sa) = dedup(&["ACGTA", "GG", "ACGTA", "GG", "GG"]);
        let n = text.len() as u32;
        let good = DupWeights::build(&sa, &bounds, &classes);
        good.validate(n, &bounds, &classes).unwrap();

        let mut wrong_k = good.clone();
        wrong_k.classes[0].k = 7;
        assert!(wrong_k.validate(n, &bounds, &classes).is_err());

        let mut short = good.clone();
        short.classes[0].rows.pop();
        assert!(short.validate(n, &bounds, &classes).is_err());

        let mut unsorted = good.clone();
        unsorted.classes[0].rows.swap(0, 1);
        assert!(unsorted.validate(n, &bounds, &classes).is_err());

        let mut overlap = good.clone();
        let r = overlap.classes[0].rows[0];
        overlap.classes[1].rows[0] = r;
        overlap.classes[1].rows.sort_unstable();
        assert!(overlap.validate(n, &bounds, &classes).is_err());

        let mut out_of_range = good.clone();
        *out_of_range.classes[1].rows.last_mut().unwrap() = n;
        assert!(out_of_range.validate(n, &bounds, &classes).is_err());

        assert!(DupWeights::default()
            .validate(n, &bounds, &classes)
            .is_err());
    }
}
