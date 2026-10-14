//! SMEM / MEM finding on [`BMoveIndex`]: the sweep shared with the other bidirectional
//! indexes ([`BidirSearch`]) over b-move cursors, located from their toeholds.
//!
//! **Locate.** A finished match is a set of cursors, each with a toehold, and `φ` / `φ⁻¹`
//! enumerate a cursor's occurrences from it. A MEM candidate with a left context `q` keeps
//! only its occurrences preceded by a symbol incompatible with `q`. Rows cannot be located
//! one by one, so they are enumerated by preceding symbol instead: for each incompatible
//! symbol `a` in the cursor's forward span, one forward LF step gives the cursor of `aP`
//! and a toehold for it, whose occurrences shifted right by one are exactly the
//! occurrences of `P` preceded by `a`. Occurrences preceded by a sentinel (a reference
//! start) sit at sentinel rows, each its own sampled move-table row, so their positions are
//! read directly.

use super::bmove::{BMoveCursor, BMoveIndex};
use super::locate::MoveLocate;
use super::UNKNOWN_TOEHOLD;
use crate::alphabet::{AlphabetFns, SymbolSet, ALPHABET_SIZE, SENTINEL};
#[cfg(test)]
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::smem::{BidirSearch, Mem, RawMem, SweepCursor};

impl SweepCursor for BMoveCursor {
    #[inline]
    fn size(&self) -> u32 {
        BMoveCursor::size(self)
    }

    #[cfg(test)]
    fn interval(&self) -> BidirInterval {
        BMoveCursor::interval(self)
    }
}

impl BMoveIndex {
    /// SMEMs of `query`: see [`BidirFmIndex::find_smems`](crate::BidirFmIndex::find_smems).
    /// The same matches and counts; each match's positions as a set, in no particular
    /// order.
    ///
    /// # Panics
    /// With `locate = true` on a count-only index.
    pub fn find_smems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
        self.smems(query, min_len, locate)
    }

    /// MEMs of `query`: see [`BidirFmIndex::find_mems`](crate::BidirFmIndex::find_mems).
    /// The same matches and counts; each match's positions as a set, in no particular
    /// order.
    ///
    /// # Panics
    /// With `locate = true` on a count-only index.
    pub fn find_mems(&self, query: &[u8], min_len: usize, locate: bool) -> Vec<Mem> {
        self.mems(query, min_len, locate)
    }

    /// Into `out`: the text positions of the maximal occurrences of `raw` (all of them, or
    /// with a left context those preceded by an incompatible symbol or a sentinel).
    fn push_maximal(&self, locate: &MoveLocate, raw: &RawMem<BMoveCursor>, out: &mut Vec<u32>) {
        let Some(q) = raw.left_ctx else {
            for cur in &raw.ivs {
                locate.enumerate(cur.toehold, cur.len, cur.size(), out);
            }
            return;
        };
        let lf = &self.fwd.lf;
        let incompatible = self
            .fwd
            .alphabet_fns
            .compatible(q)
            .complement()
            .difference(SymbolSet::single(SENTINEL));
        for cur in &raw.ivs {
            lf.rows_with(SENTINEL, &cur.fwd, |k| out.push(locate.samples.start[k]));
            for a in lf.symbols_in(&cur.fwd).intersection(incompatible).iter() {
                let Some((span, toehold)) = lf.extend(a, &cur.fwd) else {
                    continue;
                };
                // An occurrence of aP: a sampled row of a run of `a` in the span, else (the
                // span inside one run) the cursor's own toehold, moved one left.
                let x = match locate.samples.get(toehold) {
                    Some(v) => v.checked_sub(1),
                    None if cur.toehold != UNKNOWN_TOEHOLD => cur.toehold.checked_sub(1),
                    None => None,
                }
                .unwrap_or(UNKNOWN_TOEHOLD);
                let from = out.len();
                locate.enumerate(x, cur.len + 1, span.size(), out);
                out[from..].iter_mut().for_each(|p| *p += 1);
            }
        }
    }
}

impl BidirSearch for BMoveIndex {
    type Cursor = BMoveCursor;

    /// A child costs an LF step whether it is wanted or not, so steps extend by the
    /// compatible symbols only (~15% faster than fanning out on 200 Mbp of strains).
    const FANOUT_STEPS: bool = false;

    fn search_alphabet(&self) -> &AlphabetFns {
        &self.fwd.alphabet_fns
    }

    fn search_full_cursor(&self) -> BMoveCursor {
        self.full_cursor()
    }

    #[inline]
    fn search_extend_right(&self, cur: &BMoveCursor, c: u8) -> Option<BMoveCursor> {
        debug_assert_ne!(c, SENTINEL, "the sweep never extends by the sentinel");
        self.extend_right(cur, c)
    }

    #[inline]
    fn search_extend_left(&self, cur: &BMoveCursor, c: u8) -> Option<BMoveCursor> {
        debug_assert_ne!(c, SENTINEL, "the sweep never extends by the sentinel");
        self.extend_left(cur, c)
    }

    #[inline]
    fn search_children_right(&self, cur: &BMoveCursor) -> [Option<BMoveCursor>; ALPHABET_SIZE] {
        self.children_right_sweep(cur)
    }

    #[inline]
    fn search_children_left(&self, cur: &BMoveCursor) -> [Option<BMoveCursor>; ALPHABET_SIZE] {
        self.children_left(cur)
    }

    fn shallow(&self, kmer: &[u8]) -> Option<&[BMoveCursor]> {
        self.shallow.get(kmer)
    }

    fn search_locate(&self, raws: Vec<RawMem<BMoveCursor>>) -> Vec<Mem> {
        let locate = self.fwd.locate_support();
        let mut text = Vec::new();
        raws.into_iter()
            .map(|raw| {
                text.clear();
                self.push_maximal(locate, &raw, &mut text);
                debug_assert_eq!(text.len(), raw.match_count as usize);
                let positions = text
                    .iter()
                    .map(|&p| {
                        self.fwd
                            .map_position(p)
                            .expect("located position must be within text bounds")
                    })
                    .collect();
                raw.into_mem(positions)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::DnaSequence;
    use crate::move_index::MoveFmIndexConfig;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Random multi-reference texts, some sharing whole references, and their concatenation
    /// with sentinels.
    fn cases() -> Vec<(BMoveIndex, Vec<u8>)> {
        let mut rng = StdRng::seed_from_u64(0xb40e);
        let mut out = Vec::new();
        for round in 0..10 {
            let mut refs: Vec<Vec<u8>> = (0..rng.random_range(1..6))
                .map(|_| {
                    let top = if round % 3 == 0 { 6 } else { 5 };
                    (0..rng.random_range(1..30))
                        .map(|_| rng.random_range(1..top))
                        .collect()
                })
                .collect();
            refs.push(refs[0].clone());
            let seqs: Vec<DnaSequence> = refs
                .iter()
                .cloned()
                .map(DnaSequence::from_encoded)
                .collect();
            let bm = BMoveIndex::build_cpu(&seqs, &MoveFmIndexConfig::default()).unwrap();
            let text = refs
                .iter()
                .flat_map(|r| r.iter().copied().chain([0]))
                .collect();
            out.push((bm, text));
        }
        out
    }

    /// Every cursor reachable by right extension from the full one, to depth 5, with its
    /// pattern.
    fn cursors(bm: &BMoveIndex) -> Vec<(BMoveCursor, Vec<u8>)> {
        let mut out = vec![(bm.full_cursor(), Vec::new())];
        let mut level = out.clone();
        for _ in 0..5 {
            let mut next = Vec::new();
            for (cur, p) in &level {
                for c in 1..ALPHABET_SIZE as u8 {
                    if let Some(child) = bm.extend_right(cur, c) {
                        next.push((child, [p.as_slice(), &[c]].concat()));
                    }
                }
            }
            out.extend(next.iter().cloned());
            level = next;
        }
        out
    }

    #[test]
    fn sweep_children_equal_extensions_off_the_sentinel() {
        for (bm, _) in cases() {
            for (cur, _) in cursors(&bm) {
                let children = bm.children_right_sweep(&cur);
                for c in 1..ALPHABET_SIZE as u8 {
                    assert_eq!(children[c as usize], bm.extend_right(&cur, c), "code {c}");
                }
                assert_eq!(
                    children[0].map(|ch| ch.interval()),
                    bm.extend_right(&cur, SENTINEL).map(|ch| ch.interval())
                );
            }
        }
    }

    /// The sentinel child's toehold starts an occurrence of `P$`, and locating the child
    /// yields exactly the occurrences of `P` at a reference end, the last reference (whose
    /// end is the text's closing sentinel) included.
    #[test]
    fn sentinel_children_locate_the_occurrences_at_reference_ends() {
        for (bm, text) in cases() {
            for (cur, p) in cursors(&bm).into_iter().skip(1) {
                let Some(child) = bm.children_right_sweep(&cur)[0] else {
                    continue;
                };
                let t = child.toehold().expect("sweep sentinel child has a toehold") as usize;
                assert_eq!(&text[t..t + p.len()], p.as_slice());
                assert_eq!(text[t + p.len()], 0);
                let mut got: Vec<u32> = Vec::new();
                bm.fwd
                    .locate_support()
                    .enumerate(child.toehold, child.len, child.size(), &mut got);
                got.sort();
                let want: Vec<u32> = (0..text.len() - p.len())
                    .filter(|&i| text[i..i + p.len()] == p[..] && text[i + p.len()] == 0)
                    .map(|i| i as u32)
                    .collect();
                assert_eq!(got, want, "pattern {p:?}");
            }
        }
    }
}
