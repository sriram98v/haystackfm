//! CPU implementation of BWT construction and inverse BWT.

use super::Bwt;
use crate::suffix_array::SuffixArray;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Derive BWT from text and suffix array.
///
/// BWT[i] = text[(SA[i] - 1 + n) % n]
///
/// Time: O(n), embarrassingly parallel.
pub fn build_bwt(text: &[u8], sa: &SuffixArray) -> Bwt {
    let n = text.len();
    #[cfg(not(target_arch = "wasm32"))]
    let data: Vec<u8> = sa
        .data
        .par_iter()
        .map(|&sa_i| {
            let pos = if sa_i == 0 { n - 1 } else { sa_i as usize - 1 };
            text[pos]
        })
        .collect();
    #[cfg(target_arch = "wasm32")]
    let data: Vec<u8> = sa
        .data
        .iter()
        .map(|&sa_i| {
            let pos = if sa_i == 0 { n - 1 } else { sa_i as usize - 1 };
            text[pos]
        })
        .collect();
    Bwt::from_unpacked(data)
}

/// Reconstruct the original text from BWT using inverse BWT (LF-mapping).
pub fn inverse_bwt(bwt: &Bwt) -> Vec<u8> {
    let n = bwt.len();
    if n == 0 {
        return vec![];
    }

    // Build C array (cumulative character counts)
    let mut counts = [0u32; 256];
    for ch in bwt.iter_chars() {
        counts[ch as usize] += 1;
    }
    let mut c = [0u32; 256];
    let mut sum = 0u32;
    for i in 0..256 {
        c[i] = sum;
        sum += counts[i];
    }

    // Build LF mapping: LF[i] = C[BWT[i]] + rank(BWT[i], i)
    let mut occ = [0u32; 256];
    let mut lf = vec![0u32; n];
    for (i, byte) in bwt.iter_chars().enumerate() {
        let ch = byte as usize;
        lf[i] = c[ch] + occ[ch];
        occ[ch] += 1;
    }

    // Start at row 0 (F[0] = $ since sentinel sorts first).
    // Track the first-column character: F[LF[row]] = BWT[row].
    let mut result = vec![0u8; n];
    let mut row = 0usize;
    let mut first_col = 0u8; // F[0] = sentinel
    for i in (0..n).rev() {
        result[i] = first_col;
        first_col = bwt.get(row);
        row = lf[row] as usize;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::*;
    use crate::suffix_array::cpu::build_suffix_array;

    fn encode(s: &str) -> Vec<u8> {
        let mut v: Vec<u8> = s.chars().map(|c| encode_char(c).unwrap()).collect();
        if v.last() != Some(&SENTINEL) {
            v.push(SENTINEL);
        }
        v
    }

    #[test]
    fn test_bwt_basic() {
        let text = encode("ACGT");
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);

        // BWT should be a permutation of the text
        let mut bwt_sorted: Vec<u8> = bwt.iter_chars().collect();
        bwt_sorted.sort();
        let mut text_sorted = text.clone();
        text_sorted.sort();
        assert_eq!(bwt_sorted, text_sorted);
    }

    #[test]
    fn test_inverse_bwt() {
        let text = encode("ACGTACGT");
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let recovered = inverse_bwt(&bwt);
        assert_eq!(recovered, text, "inverse BWT should recover original text");
    }

    #[test]
    fn test_inverse_bwt_all_same() {
        let text = encode("AAAA");
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let recovered = inverse_bwt(&bwt);
        assert_eq!(recovered, text);
    }

    #[test]
    fn test_inverse_bwt_longer() {
        let text = encode("ACGTTAGCCAGTACGT");
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let recovered = inverse_bwt(&bwt);
        assert_eq!(recovered, text);
    }
}
