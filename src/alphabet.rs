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
//! a given code. Both the CPU query path and the GPU WGSL shaders use this table;
//! the WGSL `COMPAT` array is parity-tested against it in CI.

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

/// Encode a single IUPAC nucleotide character to its alphabet index.
///
/// U/u is treated as T (RNA → DNA). Gap characters `-` and `.` return `None`.
pub fn encode_char(ch: char) -> Option<u8> {
    match ch {
        '$' => Some(SENTINEL),
        'A' | 'a' => Some(A),
        'C' | 'c' => Some(C),
        'G' | 'g' => Some(G),
        'T' | 't' | 'U' | 'u' => Some(T),
        'N' | 'n' => Some(N),
        'R' | 'r' => Some(R),
        'Y' | 'y' => Some(Y),
        'S' | 's' => Some(S),
        'W' | 'w' => Some(W),
        'K' | 'k' => Some(K),
        'M' | 'm' => Some(M),
        'B' | 'b' => Some(B),
        'D' | 'd' => Some(D),
        'H' | 'h' => Some(H),
        'V' | 'v' => Some(V),
        _ => None,
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
        for (i, ch) in s.chars().enumerate() {
            match encode_char(ch) {
                Some(SENTINEL) | None => return Err(FmIndexError::InvalidCharacter(ch, i)),
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

    // Verifies that the WGSL COMPAT / COMPAT_LEN constants in
    // shaders/locate_search.wgsl and shaders/mem_find.wgsl exactly match
    // the compatible_symbols function above.  If this test fails, update the
    // shader constants to match.
    #[test]
    fn wgsl_compat_table_matches_compatible_symbols() {
        // Expected COMPAT_LEN (one entry per code 0..16)
        let expected_len: [u8; 16] = [0, 8, 8, 8, 8, 15, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14];

        // Expected COMPAT flat table (code * 16 + k → symbol, 0 = padding)
        #[rustfmt::skip]
        let expected_compat: [[u8; 16]; 16] = [
            // code  0 ($)
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            // code  1 (A): A N R W M D H V
            [1,  5,  6,  9, 11, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0],
            // code  2 (C): C N Y S M B H V
            [2,  5,  7,  8, 11, 12, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0],
            // code  3 (G): G N R S K B D V
            [3,  5,  6,  8, 10, 12, 13, 15,  0,  0,  0,  0,  0,  0,  0,  0],
            // code  4 (T): T N Y W K B D H
            [4,  5,  7,  9, 10, 12, 13, 14,  0,  0,  0,  0,  0,  0,  0,  0],
            // code  5 (N): all 15 non-sentinel codes
            [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0],
            // code  6 (R=A|G)
            [1,  3,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0],
            // code  7 (Y=C|T)
            [2,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0],
            // code  8 (S=G|C)
            [2,  3,  5,  6,  7,  8, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0],
            // code  9 (W=A|T)
            [1,  4,  5,  6,  7,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0],
            // code 10 (K=G|T)
            [3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,  0],
            // code 11 (M=A|C)
            [1,  2,  5,  6,  7,  8,  9, 11, 12, 13, 14, 15,  0,  0,  0,  0],
            // code 12 (B=C|G|T)
            [2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0],
            // code 13 (D=A|G|T)
            [1,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0],
            // code 14 (H=A|C|T)
            [1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0],
            // code 15 (V=A|C|G)
            [1,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0],
        ];

        for code in 0u8..16 {
            let syms = compatible_symbols(code);
            let len = syms.len() as u8;
            assert_eq!(
                len, expected_len[code as usize],
                "COMPAT_LEN mismatch for code {code}"
            );
            // Check each compatible symbol matches the expected slot
            for (k, &sym) in syms.iter().enumerate() {
                assert_eq!(
                    sym, expected_compat[code as usize][k],
                    "COMPAT mismatch for code {code} slot {k}: got {sym}, expected {}",
                    expected_compat[code as usize][k]
                );
            }
            // Padding slots must be 0
            for k in syms.len()..16 {
                assert_eq!(
                    expected_compat[code as usize][k], 0,
                    "COMPAT padding non-zero for code {code} slot {k}"
                );
            }
        }
    }
}
