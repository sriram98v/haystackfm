use crate::alphabet::{SymbolSet, ALPHABET_SIZE};
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::half::FmHalf;

/// Deepest k-mer a [`ShallowCache`] holds.
pub(super) const SHALLOW_MAX_DEPTH: u32 = 8;

/// Most intervals a [`ShallowCache`] holds (~40 MB). Wildcard fan-out makes a level's size
/// depend on the text (a reference dense in ambiguity codes multiplies the variants per
/// k-mer), so the depth stops at the last level that fits.
pub(super) const SHALLOW_MAX_IVS: usize = 1 << 21;

/// The interval set of every `{A,C,G,T}` k-mer up to a small depth, exactly as the sweeps'
/// right extension from the full interval produces it (built by that same extension, so
/// identical intervals in identical order, wildcard-compatible variants included). Every
/// SMEM pivot and MEM start begins with these shallow steps, whose intervals are the widest
/// and costliest to extend.
///
/// At most [`SHALLOW_MAX_IVS`] intervals: the depth shrinks for small texts and for texts
/// whose wildcard variants would overflow that budget. Derived, never serialized.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShallowCache {
    pub(super) depth: usize,
    /// First set index of each depth (`level_base[d]`, `d` in `1..=depth`).
    level_base: Vec<usize>,
    /// Set `x` is `ivs[offs[x]..offs[x + 1]]`.
    offs: Vec<u32>,
    pub(super) ivs: Vec<BidirInterval>,
}

impl ShallowCache {
    /// Build for a text of `text_len` symbols over the half `rev` (right extension).
    pub(crate) fn build<H: FmHalf>(full: BidirInterval, rev: &H, text_len: u32) -> Self {
        // About `log4(n)` levels: deeper ones would hold mostly singletons and misses.
        let depth = (text_len.max(1).ilog2() / 2).min(SHALLOW_MAX_DEPTH) as usize;
        let mut cache = Self {
            depth,
            level_base: vec![0; depth + 1],
            offs: vec![0],
            ivs: Vec::new(),
        };
        let mut fanout = Fanout::default();
        let mut parents: Vec<Vec<BidirInterval>> = vec![vec![full]];
        let mut child = Vec::new();
        for d in 1..=depth {
            cache.level_base[d] = cache.offs.len() - 1;
            let mut children = Vec::with_capacity(parents.len() * 4);
            // Set index within a level: the k-mer in base 4 (A = 0 .. T = 3), last symbol
            // least significant, so parent `p`'s children are `4p .. 4p + 4`.
            for parent in &parents {
                fanout.right(parent, rev);
                for c in CORE_CODES {
                    fanout.emit(rev.alphabet().compatible(c), &mut child);
                    if cache.ivs.len() + child.len() > SHALLOW_MAX_IVS {
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
    pub(crate) fn get(&self, kmer: &[u8]) -> Option<&[BidirInterval]> {
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
            + self.ivs.len() * std::mem::size_of::<BidirInterval>()
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

/// Every child of every interval of a set, from one rank pass per interval
/// ([`BidirInterval::children_right`] / [`children_left`](BidirInterval::children_left)),
/// then emitted for any symbol subset in the order `extend_multi_right` /
/// `extend_multi_left` produce: symbols ascending, intervals in input order. A sweep step
/// takes its compatible extension and its drop set from the same pass.
#[derive(Default)]
pub(crate) struct Fanout {
    children: Vec<[Option<BidirInterval>; ALPHABET_SIZE]>,
}

impl Fanout {
    /// Children of `ivs` extended right (the pattern grows at its end).
    pub(super) fn right<H: FmHalf>(&mut self, ivs: &[BidirInterval], rev: &H) {
        self.children.clear();
        self.children
            .extend(ivs.iter().map(|iv| iv.children_right(rev)));
    }

    /// Children of `ivs` extended left (the pattern grows at its start).
    pub(super) fn left<H: FmHalf>(&mut self, ivs: &[BidirInterval], fwd: &H) {
        self.children.clear();
        self.children
            .extend(ivs.iter().map(|iv| iv.children_left(fwd)));
    }

    /// The children by members of `set`, into `out` (cleared first).
    pub(super) fn emit(&self, set: SymbolSet, out: &mut Vec<BidirInterval>) {
        out.clear();
        for c in set.iter() {
            out.extend(self.children.iter().filter_map(|ch| ch[c as usize]));
        }
    }
}

/// Into `out` (cleared first): `extend(iv, c)` for every member `c` of `set` and interval
/// `iv`, symbols ascending, intervals in input order (the order [`Fanout::emit`] uses).
#[inline]
pub(super) fn extend_set(
    ivs: &[BidirInterval],
    set: SymbolSet,
    out: &mut Vec<BidirInterval>,
    extend: impl Fn(&BidirInterval, u8) -> Option<BidirInterval>,
) {
    out.clear();
    for c in set.iter() {
        out.extend(ivs.iter().filter_map(|iv| extend(iv, c)));
    }
}
