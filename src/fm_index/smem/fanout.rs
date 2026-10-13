use super::SweepCursor;
use crate::alphabet::{AlphabetFns, SymbolSet, ALPHABET_SIZE};
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::half::FmHalf;

/// Deepest k-mer a [`ShallowCache`] holds.
pub(super) const SHALLOW_MAX_DEPTH: u32 = 8;

/// Most bytes of cursors a [`ShallowCache`] holds (40 MiB: 2^21 [`BidirInterval`]s).
/// Wildcard fan-out makes a level's size depend on the text (a reference dense in ambiguity
/// codes multiplies the variants per k-mer), so the depth stops at the last level that fits.
const SHALLOW_MAX_BYTES: usize = 40 << 20;

const _: () = assert!(std::mem::size_of::<BidirInterval>() == 20);

/// The cursor set of every `{A,C,G,T}` k-mer up to a small depth, exactly as the sweeps'
/// right extension from the full cursor produces it (built from the same children, so
/// identical cursors in identical order, wildcard-compatible variants included). Every
/// SMEM pivot and MEM start begins with these shallow steps, whose cursors are the widest
/// and costliest to extend.
///
/// At most [`MAX_ENTRIES`](Self::MAX_ENTRIES) cursors: the depth shrinks for small texts
/// and for texts whose wildcard variants would overflow that budget. Derived, never
/// serialized.
#[derive(Debug, Clone)]
pub(crate) struct ShallowCache<C = BidirInterval> {
    pub(super) depth: usize,
    /// First set index of each depth (`level_base[d]`, `d` in `1..=depth`).
    level_base: Vec<usize>,
    /// Set `x` is `ivs[offs[x]..offs[x + 1]]`.
    offs: Vec<u32>,
    pub(super) ivs: Vec<C>,
}

impl<C> Default for ShallowCache<C> {
    fn default() -> Self {
        Self {
            depth: 0,
            level_base: vec![0],
            offs: vec![0],
            ivs: Vec::new(),
        }
    }
}

impl ShallowCache {
    /// Build for a text of `text_len` symbols over the half `rev` (right extension).
    pub(crate) fn build<H: FmHalf>(full: BidirInterval, rev: &H, text_len: u32) -> Self {
        Self::build_with(full, text_len, rev.alphabet(), |iv| iv.children_right(rev))
    }
}

impl<C: SweepCursor> ShallowCache<C> {
    /// Most cursors held: [`SHALLOW_MAX_BYTES`] of them.
    pub(crate) const MAX_ENTRIES: usize = SHALLOW_MAX_BYTES / std::mem::size_of::<C>();

    /// Build for a text of `text_len` symbols from the cursor `full` of the empty pattern,
    /// with `children_right` the backend's right fan-out.
    pub(crate) fn build_with(
        full: C,
        text_len: u32,
        alphabet: &AlphabetFns,
        children_right: impl Fn(&C) -> [Option<C>; ALPHABET_SIZE],
    ) -> Self {
        // About `log4(n)` levels: deeper ones would hold mostly singletons and misses.
        let depth = (text_len.max(1).ilog2() / 2).min(SHALLOW_MAX_DEPTH) as usize;
        let mut cache = Self {
            depth,
            level_base: vec![0; depth + 1],
            offs: vec![0],
            ivs: Vec::new(),
        };
        let mut fanout = Fanout::default();
        let mut parents: Vec<Vec<C>> = vec![vec![full]];
        let mut child = Vec::new();
        for d in 1..=depth {
            cache.level_base[d] = cache.offs.len() - 1;
            let mut children = Vec::with_capacity(parents.len() * 4);
            // Set index within a level: the k-mer in base 4 (A = 0 .. T = 3), last symbol
            // least significant, so parent `p`'s children are `4p .. 4p + 4`.
            for parent in &parents {
                fanout.fill(parent, &children_right);
                for c in CORE_CODES {
                    fanout.emit(alphabet.compatible(c), &mut child);
                    if cache.ivs.len() + child.len() > Self::MAX_ENTRIES {
                        // Over budget: keep only the complete levels before this one.
                        let keep = cache.level_base[d];
                        cache.ivs.truncate(cache.offs[keep] as usize);
                        cache.offs.truncate(keep + 1);
                        cache.level_base.truncate(d);
                        cache.depth = d - 1;
                        return cache;
                    }
                    cache.ivs.extend_from_slice(&child);
                    cache
                        .offs
                        .push(u32::try_from(cache.ivs.len()).expect("bounded by the budget"));
                    children.push(child.clone());
                }
            }
            parents = children;
        }
        cache
    }

    /// The cached set of `kmer`, if it is short enough and all `{A,C,G,T}`.
    #[inline]
    pub(crate) fn get(&self, kmer: &[u8]) -> Option<&[C]> {
        if kmer.is_empty() || kmer.len() > self.depth {
            return None;
        }
        let mut x = 0usize;
        for &q in kmer {
            if !(CORE_CODES[0]..=CORE_CODES[3]).contains(&q) {
                return None;
            }
            x = x * 4 + (q - CORE_CODES[0]) as usize;
        }
        let x = self.level_base[kmer.len()] + x;
        Some(&self.ivs[self.offs[x] as usize..self.offs[x + 1] as usize])
    }

    /// Heap bytes held.
    pub(crate) fn heap_size_bytes(&self) -> usize {
        self.offs.len() * 4
            + self.ivs.len() * std::mem::size_of::<C>()
            + self.level_base.len() * std::mem::size_of::<usize>()
    }
}

/// The codes a [`ShallowCache`] enumerates, ascending and contiguous.
const CORE_CODES: [u8; 4] = [
    crate::alphabet::A,
    crate::alphabet::C,
    crate::alphabet::G,
    crate::alphabet::T,
];

/// Every child of every cursor of a set, from the backend's fan-out
/// ([`search_children_right`](super::BidirSearch::search_children_right) /
/// [`search_children_left`](super::BidirSearch::search_children_left)), then emitted for any
/// symbol subset in the order per-symbol extension produces: symbols ascending, cursors in
/// input order. A sweep step takes its compatible extension and its drop set from the same
/// fan-out.
pub(crate) struct Fanout<C> {
    children: Vec<[Option<C>; ALPHABET_SIZE]>,
}

impl<C> Default for Fanout<C> {
    fn default() -> Self {
        Self {
            children: Vec::new(),
        }
    }
}

impl<C: Copy> Fanout<C> {
    /// The children of every cursor of `ivs`.
    #[inline]
    pub(super) fn fill(&mut self, ivs: &[C], children: impl Fn(&C) -> [Option<C>; ALPHABET_SIZE]) {
        self.children.clear();
        self.children.extend(ivs.iter().map(children));
    }

    /// The children by members of `set`, into `out` (cleared first).
    pub(super) fn emit(&self, set: SymbolSet, out: &mut Vec<C>) {
        out.clear();
        for c in set.iter() {
            out.extend(self.children.iter().filter_map(|ch| ch[c as usize]));
        }
    }
}

/// Into `out` (cleared first): `extend(iv, c)` for every member `c` of `set` and cursor
/// `iv`, symbols ascending, cursors in input order (the order [`Fanout::emit`] uses).
#[inline]
pub(super) fn extend_set<C>(
    ivs: &[C],
    set: SymbolSet,
    out: &mut Vec<C>,
    extend: impl Fn(&C, u8) -> Option<C>,
) {
    out.clear();
    for c in set.iter() {
        out.extend(ivs.iter().filter_map(|iv| extend(iv, c)));
    }
}
