//! IUPAC nucleotide alphabet encoding and DNA sequence types.
//!
//! The alphabet has 16 symbols (codes 0–15):
//!
//! | Code | Symbol | Bases |
//! |------|--------|-------|
//! | 0 | `$` | sentinel |
//! | 1–4 | A C G T | exact bases |
//! | 5 | N | A C G T (any) |
//! | 6–15 | R Y S W K M B D H V | degenerate IUPAC |
//!
//! [`compatible_symbols`] returns the set of codes whose base sets overlap with
//! a given code. An index carries that relation as [`AlphabetFns`], one [`SymbolSet`] per
//! query code, so the CPU query path tests compatibility with a mask intersection. The GPU
//! WGSL shaders hard-code the same IUPAC table; a unit test parses the shader sources and
//! checks them against [`IupacDna`].

use crate::error::FmIndexError;

/// Alphabet size: $, A, C, G, T, N, R, Y, S, W, K, M, B, D, H, V
pub const ALPHABET_SIZE: usize = 16;

/// Sentinel character (lexicographically smallest)
pub const SENTINEL: u8 = 0;
/// Encoded value for adenine (A).
pub const A: u8 = 1;
/// Encoded value for cytosine (C).
pub const C: u8 = 2;
/// Encoded value for guanine (G).
pub const G: u8 = 3;
/// Encoded value for thymine (T).
pub const T: u8 = 4;
/// N = A, C, G, or T (any base).
pub const N: u8 = 5;
/// R = A or G (purines).
pub const R: u8 = 6;
/// Y = C or T (pyrimidines).
pub const Y: u8 = 7;
/// S = G or C (strong).
pub const S: u8 = 8;
/// W = A or T (weak).
pub const W: u8 = 9;
/// K = G or T (keto).
pub const K: u8 = 10;
/// M = A or C (amino).
pub const M: u8 = 11;
/// B = C, G, or T (not A).
pub const B: u8 = 12;
/// D = A, G, or T (not C).
pub const D: u8 = 13;
/// H = A, C, or T (not G).
pub const H: u8 = 14;
/// V = A, C, or G (not T).
pub const V: u8 = 15;

const fn build_encode_lut() -> [i8; 256] {
    let mut lut = [-1i8; 256];
    lut[b'$' as usize] = SENTINEL as i8;
    lut[b'A' as usize] = A as i8;
    lut[b'a' as usize] = A as i8;
    lut[b'C' as usize] = C as i8;
    lut[b'c' as usize] = C as i8;
    lut[b'G' as usize] = G as i8;
    lut[b'g' as usize] = G as i8;
    lut[b'T' as usize] = T as i8;
    lut[b't' as usize] = T as i8;
    lut[b'U' as usize] = T as i8;
    lut[b'u' as usize] = T as i8;
    lut[b'N' as usize] = N as i8;
    lut[b'n' as usize] = N as i8;
    lut[b'R' as usize] = R as i8;
    lut[b'r' as usize] = R as i8;
    lut[b'Y' as usize] = Y as i8;
    lut[b'y' as usize] = Y as i8;
    lut[b'S' as usize] = S as i8;
    lut[b's' as usize] = S as i8;
    lut[b'W' as usize] = W as i8;
    lut[b'w' as usize] = W as i8;
    lut[b'K' as usize] = K as i8;
    lut[b'k' as usize] = K as i8;
    lut[b'M' as usize] = M as i8;
    lut[b'm' as usize] = M as i8;
    lut[b'B' as usize] = B as i8;
    lut[b'b' as usize] = B as i8;
    lut[b'D' as usize] = D as i8;
    lut[b'd' as usize] = D as i8;
    lut[b'H' as usize] = H as i8;
    lut[b'h' as usize] = H as i8;
    lut[b'V' as usize] = V as i8;
    lut[b'v' as usize] = V as i8;
    lut
}

/// 256-entry lookup table from raw ASCII byte to alphabet index (-1 = invalid).
/// Built at compile time so `encode_byte` is a single branchless array load.
static ENCODE_LUT: [i8; 256] = build_encode_lut();

/// Encode a single raw ASCII IUPAC nucleotide byte to its alphabet index.
///
/// O(1) table lookup — the preferred entry point when working with `&[u8]` text
/// (FASTA/FASTQ bytes), matching the convention used by crates like `bio`.
/// U/u is treated as T (RNA → DNA). Gap bytes `-` and `.` return `None`.
#[inline]
pub fn encode_byte(b: u8) -> Option<u8> {
    let v = ENCODE_LUT[b as usize];
    if v < 0 {
        None
    } else {
        Some(v as u8)
    }
}

/// Encode a single IUPAC nucleotide character to its alphabet index.
///
/// U/u is treated as T (RNA → DNA). Gap characters `-` and `.` return `None`.
/// Non-ASCII characters always return `None`. Prefer [`encode_byte`] when the
/// input is already `&[u8]` to avoid the `char` conversion.
#[inline]
pub fn encode_char(ch: char) -> Option<u8> {
    if ch.is_ascii() {
        encode_byte(ch as u8)
    } else {
        None
    }
}

/// Decode an alphabet index back to its IUPAC ASCII character.
pub fn decode_char(code: u8) -> Option<char> {
    match code {
        SENTINEL => Some('$'),
        A => Some('A'),
        C => Some('C'),
        G => Some('G'),
        T => Some('T'),
        N => Some('N'),
        R => Some('R'),
        Y => Some('Y'),
        S => Some('S'),
        W => Some('W'),
        K => Some('K'),
        M => Some('M'),
        B => Some('B'),
        D => Some('D'),
        H => Some('H'),
        V => Some('V'),
        _ => None,
    }
}

/// Returns the {A, C, G, T} base codes that an IUPAC symbol represents.
pub fn iupac_bases(code: u8) -> &'static [u8] {
    match code {
        x if x == A => &[A],
        x if x == C => &[C],
        x if x == G => &[G],
        x if x == T => &[T],
        x if x == N => &[A, C, G, T],
        x if x == R => &[A, G],
        x if x == Y => &[C, T],
        x if x == S => &[G, C],
        x if x == W => &[A, T],
        x if x == K => &[G, T],
        x if x == M => &[A, C],
        x if x == B => &[C, G, T],
        x if x == D => &[A, G, T],
        x if x == H => &[A, C, T],
        x if x == V => &[A, C, G],
        _ => &[],
    }
}

/// Returns all alphabet symbols (codes 1–15) whose base set overlaps with `code`'s base set.
///
/// Two IUPAC symbols match when their base sets share at least one nucleotide.
/// This drives both backward search and bidirectional MEM/SMEM extension.
pub fn compatible_symbols(code: u8) -> &'static [u8] {
    match code {
        x if x == A => &[A, N, R, W, M, D, H, V],
        x if x == C => &[C, N, Y, S, M, B, H, V],
        x if x == G => &[G, N, R, S, K, B, D, V],
        x if x == T => &[T, N, Y, W, K, B, D, H],
        x if x == N => &[A, C, G, T, N, R, Y, S, W, K, M, B, D, H, V],
        x if x == R => &[A, G, N, R, S, W, K, M, B, D, H, V],
        x if x == Y => &[C, T, N, Y, S, W, K, M, B, D, H, V],
        x if x == S => &[C, G, N, R, Y, S, K, M, B, D, H, V],
        x if x == W => &[A, T, N, R, Y, W, K, M, B, D, H, V],
        x if x == K => &[G, T, N, R, Y, S, W, K, B, D, H, V],
        x if x == M => &[A, C, N, R, Y, S, W, M, B, D, H, V],
        x if x == B => &[C, G, T, N, R, Y, S, W, K, M, B, D, H, V],
        x if x == D => &[A, G, T, N, R, Y, S, W, K, M, B, D, H, V],
        x if x == H => &[A, C, T, N, R, Y, S, W, K, M, B, D, H, V],
        x if x == V => &[A, C, G, N, R, Y, S, W, K, M, B, D, H, V],
        _ => &[],
    }
}

// ── SymbolSet ────────────────────────────────────────────────────────────────

/// A set of alphabet codes, stored as a 16-bit mask (bit `c` set ⇔ code `c` is a member).
///
/// Used to ask the occurrence table for the count of *any* symbol in a class with a single
/// query (see `OccTable::rank_set`), instead of one rank per member. The bidirectional cursor
/// builds on it for wildcard-aware extension: [`BidirInterval::count_wild_right`] is
/// `count_right_in(SymbolSet::WILDCARDS)`.
///
/// [`BidirInterval::count_wild_right`]: crate::fm_index::bidir::BidirInterval::count_wild_right
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, Debug, Default, serde::Serialize, serde::Deserialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct SymbolSet(u16);

impl SymbolSet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);
    /// Every code `0..ALPHABET_SIZE`, sentinel included.
    pub const ALL: Self = Self(u16::MAX);
    /// The four exact bases `{A, C, G, T}`.
    pub const BASES: Self = Self((1 << A) | (1 << C) | (1 << G) | (1 << T));
    /// Every ambiguity code, `N` and the ten degenerate IUPAC symbols: codes `5..=15`.
    pub const WILDCARDS: Self = Self(u16::MAX << N);
    /// Every code except the sentinel: `BASES ∪ WILDCARDS`.
    pub const NON_SENTINEL: Self = Self(u16::MAX << A);

    /// The set containing only `c`. Panics in debug builds if `c >= ALPHABET_SIZE`.
    #[inline]
    pub const fn single(c: u8) -> Self {
        debug_assert!((c as usize) < ALPHABET_SIZE);
        Self(1 << c)
    }

    /// Every code strictly smaller than `c`: `below(0)` is empty, `below(16)` is `ALL`.
    #[inline]
    pub const fn below(c: u8) -> Self {
        if c as usize >= ALPHABET_SIZE {
            Self::ALL
        } else {
            Self((1u32 << c).wrapping_sub(1) as u16)
        }
    }

    /// Build a set from a slice of codes (codes `>= ALPHABET_SIZE` are ignored).
    pub fn from_codes(codes: &[u8]) -> Self {
        codes
            .iter()
            .filter(|&&c| (c as usize) < ALPHABET_SIZE)
            .fold(Self::EMPTY, |acc, &c| acc.union(Self::single(c)))
    }

    /// True if `c` is a member.
    #[inline]
    pub const fn contains(self, c: u8) -> bool {
        (c as usize) < ALPHABET_SIZE && (self.0 >> c) & 1 == 1
    }

    /// Set union.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Set intersection.
    #[inline]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Members of `self` that are not in `other`.
    #[inline]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Complement within `ALL` (the sentinel is a code like any other).
    #[inline]
    pub const fn complement(self) -> Self {
        Self(!self.0)
    }

    /// Number of members.
    #[inline]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// True if the set has no members.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The raw 16-bit mask.
    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Iterate the member codes in ascending order.
    #[inline]
    pub fn iter(self) -> impl Iterator<Item = u8> {
        let mut bits = self.0;
        std::iter::from_fn(move || {
            if bits == 0 {
                None
            } else {
                let c = bits.trailing_zeros() as u8;
                bits &= bits - 1;
                Some(c)
            }
        })
    }
}

impl std::ops::BitOr for SymbolSet {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl std::ops::BitAnd for SymbolSet {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        self.intersection(rhs)
    }
}

impl std::ops::Not for SymbolSet {
    type Output = Self;
    fn not(self) -> Self {
        self.complement()
    }
}

impl FromIterator<u8> for SymbolSet {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        iter.into_iter()
            .filter(|&c| (c as usize) < ALPHABET_SIZE)
            .fold(Self::EMPTY, |acc, c| acc.union(Self::single(c)))
    }
}

// ── Alphabet trait and built-in implementations ──────────────────────────────

/// The matching semantics of an alphabet, as data: one [`SymbolSet`] of compatible reference
/// codes per query code, the core (unambiguous) symbols the lookup table is built over, and
/// a tag.
///
/// Stored inside [`FmIndex`] so queries test compatibility with a mask intersection, without
/// generic parameters on the struct itself, and written into the serialized index (format
/// version 3) so a custom alphabet loads back exactly as it was built.
///
/// [`FmIndex`]: crate::fm_index::FmIndex
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub struct AlphabetFns {
    /// `compat[q]` = the reference codes query code `q` matches.
    compat: [SymbolSet; ALPHABET_SIZE],
    /// The "core" (unambiguous) symbols used as the radix of the depth-k lookup table.
    core: SymbolSet,
    /// Small integer identifying the alphabet: 0 = [`IupacDna`], 1 = [`ExactDna`],
    /// 2..=127 reserved, ≥ 128 for custom alphabets. Serialized last.
    tag: u8,
}

impl AlphabetFns {
    /// Build from explicit tables. `core` is the set of unambiguous symbols the lookup
    /// table enumerates (`{A, C, G, T}` for both built-ins).
    pub const fn new(compat: [SymbolSet; ALPHABET_SIZE], core: SymbolSet, tag: u8) -> Self {
        Self { compat, core, tag }
    }

    /// Build from a compatibility function, evaluated once per code. Membership is all
    /// that is kept: the fan-out order of query-time expansion is ascending code order,
    /// whatever order `f` lists its codes in.
    pub fn from_compatible_fn(f: fn(u8) -> &'static [u8], core_symbols: &[u8], tag: u8) -> Self {
        let mut compat = [SymbolSet::EMPTY; ALPHABET_SIZE];
        for (q, set) in compat.iter_mut().enumerate() {
            *set = SymbolSet::from_codes(f(q as u8));
        }
        Self::new(compat, SymbolSet::from_codes(core_symbols), tag)
    }

    /// The reference codes that query code `q` matches under this alphabet. Empty for
    /// `q >= ALPHABET_SIZE`, and e.g. empty for `N` under [`ExactDna`]. Backward search,
    /// MEM extension and the cursor fan-out all expand a query code through this set.
    #[inline]
    pub fn compatible(&self, q: u8) -> SymbolSet {
        if (q as usize) < ALPHABET_SIZE {
            self.compat[q as usize]
        } else {
            SymbolSet::EMPTY
        }
    }

    /// Alias of [`compatible`](Self::compatible).
    #[inline]
    pub fn compatible_set(&self, q: u8) -> SymbolSet {
        self.compatible(q)
    }

    /// The core (unambiguous) symbols.
    #[inline]
    pub fn core(&self) -> SymbolSet {
        self.core
    }

    /// The alphabet tag.
    #[inline]
    pub fn tag(&self) -> u8 {
        self.tag
    }

    /// True if a lookup table built by exact core-symbol steps is complete for a text whose
    /// present symbols are `present`: no core symbol matches any *other* present symbol.
    /// (A core symbol absent from the text is fine: its entries are empty either way.)
    pub(crate) fn exact_table_is_complete(&self, present: SymbolSet) -> bool {
        self.core.iter().all(|s| {
            self.compat[s as usize]
                .intersection(present)
                .difference(SymbolSet::single(s))
                .is_empty()
        })
    }
}

/// Trait for DNA alphabet matching semantics.
///
/// Implement this to define custom symbol sets and match rules for use with
/// [`FmIndex::build_cpu_with`].
///
/// # Contract
/// * [`fns`] must return an equal value every time it is called.
/// * [`tag`] must be ≥ 128 and unique across all impls in use within a program; values
///   2..=127 are reserved and rejected when an index is loaded.
///
/// [`FmIndex::build_cpu_with`]: crate::fm_index::FmIndex::build_cpu_with
/// [`fns`]: Alphabet::fns
/// [`tag`]: AlphabetFns::tag
pub trait Alphabet: Send + Sync + 'static {
    /// Return the matching tables for this alphabet.
    fn fns() -> AlphabetFns;
}

/// Reconstruct a built-in [`AlphabetFns`] from its tag.
///
/// Returns `None` for unrecognized tags.  Built-in tags: 0 = [`IupacDna`], 1 = [`ExactDna`].
/// Format-1 and format-2 indices store only the tag, so only built-in alphabets load from
/// them; format 3 stores the tables themselves.
pub fn alphabet_fns_from_tag(tag: u8) -> Option<AlphabetFns> {
    match tag {
        0 => Some(IupacDna::fns()),
        1 => Some(ExactDna::fns()),
        _ => None,
    }
}

/// Full IUPAC 16-symbol DNA alphabet with ambiguity-code matching (default).
///
/// Query symbol `N` matches any base; other ambiguity codes match via base-set overlap.
/// This is the default alphabet used by [`FmIndex::build_cpu`].
///
/// [`FmIndex::build_cpu`]: crate::fm_index::FmIndex::build_cpu
pub struct IupacDna;

impl Alphabet for IupacDna {
    fn fns() -> AlphabetFns {
        AlphabetFns::from_compatible_fn(compatible_symbols, &[A, C, G, T], 0)
    }
}

/// Exact-match ACGT alphabet: query N / ambiguity codes produce zero hits.
///
/// Only the four canonical bases (A, C, G, T) match themselves; any other
/// query code returns an empty compatible set. Use this with
/// [`FmIndex::build_cpu_with::<ExactDna>`] for peer-comparable benchmarks where
/// ambiguity-code expansion is undesirable.
///
/// [`FmIndex::build_cpu_with::<ExactDna>`]: crate::fm_index::FmIndex::build_cpu_with
pub struct ExactDna;

impl ExactDna {
    /// Compatible-symbols function for [`ExactDna`]: ACGT → self, everything else → empty.
    pub fn compatible(code: u8) -> &'static [u8] {
        match code {
            x if x == A => &[A],
            x if x == C => &[C],
            x if x == G => &[G],
            x if x == T => &[T],
            _ => &[],
        }
    }
}

impl Alphabet for ExactDna {
    fn fns() -> AlphabetFns {
        AlphabetFns::from_compatible_fn(ExactDna::compatible, &[A, C, G, T], 1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// A DNA/RNA sequence with full IUPAC ambiguity code support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DnaSequence {
    bases: Vec<u8>,
    /// FASTA header (without leading `>`). Empty string if not provided.
    header: String,
}

impl DnaSequence {
    /// Parse from a string of IUPAC nucleotide characters. Returns Err on invalid characters.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, FmIndexError> {
        if s.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }
        let mut bases = Vec::with_capacity(s.len());
        for (i, b) in s.bytes().enumerate() {
            match encode_byte(b) {
                Some(SENTINEL) | None => {
                    let ch = s[i..].chars().next().unwrap_or(b as char);
                    return Err(FmIndexError::InvalidCharacter(ch, i));
                }
                Some(code) => bases.push(code),
            }
        }
        Ok(Self {
            bases,
            header: String::new(),
        })
    }

    /// Parse from a string of IUPAC nucleotide characters with a FASTA header.
    pub fn from_str_with_header(s: &str, header: &str) -> Result<Self, FmIndexError> {
        let mut seq = Self::from_str(s)?;
        seq.header = header.to_string();
        Ok(seq)
    }

    /// Create from pre-encoded bases (no validation).
    pub fn from_encoded(bases: Vec<u8>) -> Self {
        Self {
            bases,
            header: String::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.bases.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }

    pub fn header(&self) -> &str {
        &self.header
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bases
    }
}

/// Concatenate multiple DNA sequences with $ separators into a single encoded text.
/// Result: s1 $ s2 $ ... $ sn $
/// Returns the concatenated text and the cumulative lengths (for mapping positions back).
pub fn concatenate_sequences(
    sequences: &[DnaSequence],
) -> Result<(Vec<u8>, Vec<u32>), FmIndexError> {
    let total_len: usize = sequences.iter().map(|s| s.len() + 1).sum();
    if total_len > u32::MAX as usize {
        return Err(FmIndexError::TextTooLarge(total_len));
    }

    let mut text = Vec::with_capacity(total_len);
    let mut cumulative_lengths = Vec::with_capacity(sequences.len());

    for seq in sequences {
        text.extend_from_slice(seq.as_slice());
        text.push(SENTINEL);
        cumulative_lengths.push(text.len() as u32);
    }

    Ok((text, cumulative_lengths))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let pairs = [
            ('A', A),
            ('C', C),
            ('G', G),
            ('T', T),
            ('N', N),
            ('R', R),
            ('Y', Y),
            ('S', S),
            ('W', W),
            ('K', K),
            ('M', M),
            ('B', B),
            ('D', D),
            ('H', H),
            ('V', V),
        ];
        for (ch, code) in pairs {
            assert_eq!(encode_char(ch), Some(code), "encode {ch}");
            assert_eq!(decode_char(code), Some(ch), "decode {code}");
        }
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(encode_char('a'), Some(A));
        assert_eq!(encode_char('r'), Some(R));
        assert_eq!(encode_char('y'), Some(Y));
        assert_eq!(encode_char('n'), Some(N));
    }

    #[test]
    fn test_u_maps_to_t() {
        assert_eq!(encode_char('U'), Some(T));
        assert_eq!(encode_char('u'), Some(T));
    }

    #[test]
    fn test_gap_chars_invalid() {
        assert!(encode_char('-').is_none());
        assert!(encode_char('.').is_none());
    }

    #[test]
    fn test_invalid_char() {
        assert!(encode_char('X').is_none());
        assert!(encode_char('Z').is_none());
    }

    #[test]
    fn test_dna_sequence_acgt() {
        let seq = DnaSequence::from_str("ACGT").unwrap();
        assert_eq!(seq.as_slice(), &[A, C, G, T]);
    }

    #[test]
    fn test_dna_sequence_iupac() {
        let seq = DnaSequence::from_str("ACGTRYNSWKMBDHV").unwrap();
        assert_eq!(
            seq.as_slice(),
            &[A, C, G, T, R, Y, N, S, W, K, M, B, D, H, V]
        );
    }

    #[test]
    fn test_dna_sequence_invalid() {
        assert!(DnaSequence::from_str("ACXGT").is_err());
        assert!(DnaSequence::from_str("AC-GT").is_err());
    }

    #[test]
    fn test_dna_sequence_empty() {
        assert!(DnaSequence::from_str("").is_err());
    }

    #[test]
    fn test_iupac_bases_correctness() {
        assert_eq!(iupac_bases(A), &[A]);
        assert_eq!(iupac_bases(N), &[A, C, G, T]);
        assert_eq!(iupac_bases(R), &[A, G]);
        assert_eq!(iupac_bases(Y), &[C, T]);
        assert_eq!(iupac_bases(B), &[C, G, T]);
        assert_eq!(iupac_bases(V), &[A, C, G]);
    }

    #[test]
    fn test_compatible_symbols_a() {
        let compat = compatible_symbols(A);
        // A is compatible with everything that includes A: A, N, R, W, M, D, H, V
        assert!(compat.contains(&A));
        assert!(compat.contains(&N));
        assert!(compat.contains(&R));
        assert!(compat.contains(&W));
        assert!(compat.contains(&M));
        assert!(compat.contains(&D));
        assert!(compat.contains(&H));
        assert!(compat.contains(&V));
        // Not compatible with C-only, G-only, T-only, or codes with no A
        assert!(!compat.contains(&C));
        assert!(!compat.contains(&G));
        assert!(!compat.contains(&T));
        assert!(!compat.contains(&Y)); // C,T
        assert!(!compat.contains(&S)); // G,C
        assert!(!compat.contains(&K)); // G,T
        assert!(!compat.contains(&B)); // C,G,T
    }

    #[test]
    fn test_compatible_symbols_n_is_universal() {
        let compat = compatible_symbols(N);
        for code in 1u8..=15 {
            assert!(
                compat.contains(&code),
                "N should be compatible with code {code}"
            );
        }
    }

    #[test]
    fn test_compatible_symbols_symmetric() {
        // Compatibility must be symmetric: if a ∈ compatible(b) then b ∈ compatible(a)
        for a in 1u8..=15 {
            for &b in compatible_symbols(a) {
                assert!(
                    compatible_symbols(b).contains(&a),
                    "compatible_symbols not symmetric: {a} ∈ compatible({b}) but {b} ∉ compatible({a})"
                );
            }
        }
    }

    #[test]
    fn test_concatenate() {
        let s1 = DnaSequence::from_str("ACG").unwrap();
        let s2 = DnaSequence::from_str("TT").unwrap();
        let (text, cum) = concatenate_sequences(&[s1, s2]).unwrap();
        assert_eq!(text, vec![A, C, G, SENTINEL, T, T, SENTINEL]);
        assert_eq!(cum, vec![4, 7]);
    }

    // ── SymbolSet ─────────────────────────────────────────────────────────────

    #[test]
    fn symbol_set_below_equals_from_codes_range() {
        for c in 0..=16u8 {
            let codes: Vec<u8> = (0..c.min(16)).collect();
            assert_eq!(
                SymbolSet::below(c),
                SymbolSet::from_codes(&codes),
                "below({c})"
            );
            assert_eq!(SymbolSet::below(c).len(), c.min(16) as u32);
        }
        assert_eq!(SymbolSet::below(0), SymbolSet::EMPTY);
        assert_eq!(SymbolSet::below(16), SymbolSet::ALL);
    }

    #[test]
    fn symbol_set_complement_partitions_all() {
        let mut sets = vec![
            SymbolSet::EMPTY,
            SymbolSet::ALL,
            SymbolSet::BASES,
            SymbolSet::WILDCARDS,
        ];
        for c in 0..16u8 {
            sets.push(SymbolSet::single(c));
            sets.push(SymbolSet::below(c));
        }
        for s in sets {
            let comp = s.complement();
            assert_eq!(s.union(comp), SymbolSet::ALL);
            assert!(s.intersection(comp).is_empty());
            assert_eq!(s.len() + comp.len(), 16);
            assert_eq!(!s, comp);
            assert_eq!(s | comp, SymbolSet::ALL);
            assert_eq!(s & comp, SymbolSet::EMPTY);
            assert_eq!(s.difference(comp), s);
        }
    }

    #[test]
    fn symbol_set_named_constants() {
        assert_eq!(SymbolSet::ALL.len(), 16);
        assert_eq!(
            SymbolSet::WILDCARDS,
            SymbolSet::from_codes(&[N, R, Y, S, W, K, M, B, D, H, V])
        );
        assert_eq!(
            SymbolSet::WILDCARDS,
            SymbolSet::from_codes(&(5..16).collect::<Vec<u8>>())
        );
        assert_eq!(SymbolSet::BASES, SymbolSet::from_codes(&[A, C, G, T]));
        assert_eq!(
            SymbolSet::NON_SENTINEL,
            SymbolSet::BASES.union(SymbolSet::WILDCARDS)
        );
        assert_eq!(
            SymbolSet::NON_SENTINEL,
            SymbolSet::single(SENTINEL).complement()
        );
        assert!(!SymbolSet::WILDCARDS.contains(SENTINEL));
        assert!(!SymbolSet::WILDCARDS.contains(T));
        assert!(SymbolSet::WILDCARDS.contains(N));
        assert!(SymbolSet::WILDCARDS.contains(V));
        assert!(!SymbolSet::ALL.contains(16));
    }

    #[test]
    fn symbol_set_iter_ascending_matches_contains() {
        let s = SymbolSet::from_codes(&[V, A, N, SENTINEL, K]);
        let got: Vec<u8> = s.iter().collect();
        assert_eq!(got, vec![SENTINEL, A, N, K, V]);
        for c in 0..16u8 {
            assert_eq!(s.contains(c), got.contains(&c));
        }
        assert_eq!(SymbolSet::EMPTY.iter().count(), 0);
        assert_eq!(SymbolSet::ALL.iter().count(), 16);
        let collected: SymbolSet = got.iter().copied().collect();
        assert_eq!(collected, s);
        // Out-of-range codes are ignored rather than wrapping.
        assert_eq!(SymbolSet::from_codes(&[A, 16, 255]), SymbolSet::single(A));
    }

    #[test]
    fn compatible_matches_compatible_fn_for_iupac_and_exact() {
        for (fns, f) in [
            (
                IupacDna::fns(),
                compatible_symbols as fn(u8) -> &'static [u8],
            ),
            (ExactDna::fns(), ExactDna::compatible),
        ] {
            for q in 0..16u8 {
                let expect = SymbolSet::from_codes(f(q));
                assert_eq!(fns.compatible(q), expect, "tag {} code {q}", fns.tag());
                assert_eq!(fns.compatible_set(q), expect);
            }
            assert_eq!(fns.core(), SymbolSet::BASES);
            // Codes past the alphabet match nothing rather than indexing out of range.
            assert!(fns.compatible(16).is_empty());
            assert!(fns.compatible(255).is_empty());
        }
        assert_eq!(IupacDna::fns().tag(), 0);
        assert_eq!(ExactDna::fns().tag(), 1);
        assert_eq!(IupacDna::fns().compatible(N), SymbolSet::NON_SENTINEL);
        assert!(ExactDna::fns().compatible(N).is_empty());
        assert_eq!(ExactDna::fns().compatible(A), SymbolSet::single(A));
        assert_eq!(
            IupacDna::fns()
                .compatible(A)
                .intersection(SymbolSet::WILDCARDS),
            SymbolSet::from_codes(&[N, R, W, M, D, H, V])
        );
    }

    #[test]
    fn alphabet_fns_round_trip_through_serde_and_tag_lookup() {
        for fns in [IupacDna::fns(), ExactDna::fns()] {
            assert_eq!(alphabet_fns_from_tag(fns.tag()), Some(fns));
            let bytes = bincode::serialize(&fns).unwrap();
            // 16 masks × 2 bytes + core mask + tag.
            assert_eq!(bytes.len(), ALPHABET_SIZE * 2 + 2 + 1);
            assert_eq!(*bytes.last().unwrap(), fns.tag());
            let back: AlphabetFns = bincode::deserialize(&bytes).unwrap();
            assert_eq!(back, fns);
        }
        assert_eq!(alphabet_fns_from_tag(2), None);
        assert_eq!(alphabet_fns_from_tag(200), None);
    }

    #[test]
    fn exact_table_is_complete_only_when_core_symbols_match_nothing_else_present() {
        let iupac = IupacDna::fns();
        let exact = ExactDna::fns();
        let acgt = SymbolSet::BASES.union(SymbolSet::single(SENTINEL));
        let with_n = acgt.union(SymbolSet::single(N));
        assert!(iupac.exact_table_is_complete(acgt));
        assert!(!iupac.exact_table_is_complete(with_n));
        // A wildcard in the text is harmless when the query side never matches it.
        assert!(exact.exact_table_is_complete(with_n));
        // A core symbol absent from the text does not make the table incomplete.
        assert!(iupac.exact_table_is_complete(SymbolSet::from_codes(&[SENTINEL, A, C])));
        // ...but a present wildcard still does.
        assert!(!iupac.exact_table_is_complete(SymbolSet::from_codes(&[SENTINEL, A, R])));
    }

    /// Parse a `const NAME: array<u32, N> = array<u32, N>( ... );` literal out of WGSL
    /// source: strip `//` comments, split the parenthesised body on commas, drop the `u`
    /// suffix.
    fn parse_wgsl_u32_array(src: &str, name: &str) -> Vec<u32> {
        let decl = format!("const {name}:");
        let start = src
            .find(&decl)
            .unwrap_or_else(|| panic!("no `{decl}` in shader"));
        let body_start = src[start..].find('(').unwrap() + start + 1;
        let body_end = src[body_start..].find(");").unwrap() + body_start;
        src[body_start..body_end]
            .lines()
            .map(|line| line.split("//").next().unwrap())
            .flat_map(|line| line.split(','))
            .map(str::trim)
            .filter(|tok| !tok.is_empty())
            .map(|tok| {
                tok.strip_suffix('u')
                    .unwrap_or(tok)
                    .parse::<u32>()
                    .unwrap_or_else(|_| panic!("bad u32 literal {tok:?} in {name}"))
            })
            .collect()
    }

    // The GPU shaders hard-code the IUPAC compatibility relation as `COMPAT_LEN` /
    // `COMPAT` constants. Parse them from the shader sources and check them against
    // `IupacDna::fns()`, so a change to either side fails here rather than in a GPU-only
    // parity test.
    #[test]
    fn wgsl_compat_tables_match_iupac_dna() {
        let fns = IupacDna::fns();
        for (path, src) in [
            (
                "shaders/locate_search.wgsl",
                include_str!("../shaders/locate_search.wgsl"),
            ),
            (
                "shaders/mem_find.wgsl",
                include_str!("../shaders/mem_find.wgsl"),
            ),
        ] {
            let lens = parse_wgsl_u32_array(src, "COMPAT_LEN");
            let compat = parse_wgsl_u32_array(src, "COMPAT");
            assert_eq!(lens.len(), ALPHABET_SIZE, "{path}: COMPAT_LEN size");
            assert_eq!(compat.len(), ALPHABET_SIZE * 16, "{path}: COMPAT size");
            for code in 0..ALPHABET_SIZE {
                let expect: Vec<u32> = fns.compatible(code as u8).iter().map(u32::from).collect();
                let row = &compat[code * 16..(code + 1) * 16];
                assert_eq!(
                    lens[code] as usize,
                    expect.len(),
                    "{path}: COMPAT_LEN[{code}]"
                );
                assert_eq!(
                    &row[..expect.len()],
                    &expect[..],
                    "{path}: COMPAT row {code}"
                );
                assert!(
                    row[expect.len()..].iter().all(|&x| x == 0),
                    "{path}: COMPAT row {code} padding"
                );
            }
        }
    }
}
