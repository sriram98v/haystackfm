//! CPU implementation of suffix array construction (SACA-K via psacak).

use super::SuffixArray;

/// Build a suffix array using the pSACAK algorithm.
///
/// Time: O(n), Space: O(n).
/// Requires `text` to end with a unique minimum sentinel byte (0).
pub fn build_suffix_array(text: &[u8]) -> SuffixArray {
    if text.is_empty() {
        return SuffixArray { data: vec![] };
    }
    SuffixArray {
        data: psacak::psacak(text),
    }
}

/// Simple O(n log n) SA construction using standard library sort.
/// Used as a reference for testing.
pub fn build_suffix_array_naive(text: &[u8]) -> SuffixArray {
    let n = text.len();
    let mut sa: Vec<u32> = (0..n as u32).collect();
    sa.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
    SuffixArray { data: sa }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::*;

    fn encode(s: &str) -> Vec<u8> {
        let mut v: Vec<u8> = s.chars().map(|c| encode_char(c).unwrap()).collect();
        // Append sentinel if not present
        if v.last() != Some(&SENTINEL) {
            v.push(SENTINEL);
        }
        v
    }

    #[test]
    fn test_banana() {
        // "banana$" using our alphabet: b->1(A), a->1(A), n->3(G)...
        // Let's use a DNA-like string instead
        // Use "ACAC$"
        let text = encode("ACAC");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(
            sa.data, naive.data,
            "prefix doubling SA must match naive SA"
        );
    }

    #[test]
    fn test_single_char() {
        let text = encode("A");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(sa.data, naive.data);
    }

    #[test]
    fn test_all_same() {
        let text = encode("AAAA");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(sa.data, naive.data);
    }

    #[test]
    fn test_all_different() {
        let text = encode("ACGT");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(sa.data, naive.data);
    }

    #[test]
    fn test_longer_sequence() {
        let text = encode("ACGTACGTACGT");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(sa.data, naive.data);
    }

    #[test]
    fn test_sa_is_permutation() {
        let text = encode("ACGTTAGCCA");
        let sa = build_suffix_array(&text);
        let n = sa.len();
        let mut sorted = sa.data.clone();
        sorted.sort();
        assert_eq!(sorted, (0..n as u32).collect::<Vec<_>>());
    }

    #[test]
    fn test_sa_is_sorted() {
        let text = encode("ACGTTAGCCA");
        let sa = build_suffix_array(&text);
        for i in 1..sa.len() {
            let a = sa.data[i - 1] as usize;
            let b = sa.data[i] as usize;
            assert!(
                text[a..] < text[b..],
                "suffix at {} should be < suffix at {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_repetitive() {
        let text = encode("ACACACACAC");
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(sa.data, naive.data);
    }
}
