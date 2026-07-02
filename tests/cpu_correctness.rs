mod common;

use webgpu_fmidx::alphabet::*;
use webgpu_fmidx::bwt::cpu::{build_bwt, inverse_bwt};
use webgpu_fmidx::c_array::CArray;
use webgpu_fmidx::occ::cpu::{build_occ_table, naive_rank};
use webgpu_fmidx::occ::OccEncoding;
use webgpu_fmidx::suffix_array::cpu::{build_suffix_array, build_suffix_array_naive};

use common::encode;

#[test]
fn sa_matches_naive_for_various_inputs() {
    let inputs = [
        "A",
        "ACGT",
        "AAAA",
        "ACGTACGT",
        "TGCATGCA",
        "ACGTTAGCCAGTACGT",
        "ACACACACAC",
        "GCGCGCGCGC",
    ];

    for input in &inputs {
        let text = encode(input);
        let sa = build_suffix_array(&text);
        let naive = build_suffix_array_naive(&text);
        assert_eq!(sa.data, naive.data, "SA mismatch for input '{}'", input);
    }
}

#[test]
fn sa_is_valid_permutation() {
    let inputs = ["ACGT", "ACGTACGTACGT", "AAACCCGGGTTT", "TTTAAACCCGGG"];

    for input in &inputs {
        let text = encode(input);
        let sa = build_suffix_array(&text);
        let n = sa.len();
        let mut sorted = sa.data.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            (0..n as u32).collect::<Vec<_>>(),
            "SA is not a permutation for '{}'",
            input
        );
    }
}

#[test]
fn sa_produces_sorted_suffixes() {
    let inputs = ["ACGTTAGCCAGTACGT", "ACACACACAC", "GCGCGCGCGC"];

    for input in &inputs {
        let text = encode(input);
        let sa = build_suffix_array(&text);
        for i in 1..sa.len() {
            let a = sa.data[i - 1] as usize;
            let b = sa.data[i] as usize;
            assert!(
                text[a..] < text[b..],
                "SA not sorted at position {} for '{}'",
                i,
                input
            );
        }
    }
}

#[test]
fn bwt_is_permutation_of_text() {
    let text = encode("ACGTACGT");
    let sa = build_suffix_array(&text);
    let bwt = build_bwt(&text, &sa);

    let mut bwt_sorted: Vec<u8> = bwt.iter_chars().collect();
    bwt_sorted.sort();
    let mut text_sorted = text.clone();
    text_sorted.sort();
    assert_eq!(bwt_sorted, text_sorted);
}

#[test]
fn inverse_bwt_recovers_text() {
    let inputs = [
        "A",
        "ACGT",
        "ACGTACGT",
        "AAAA",
        "ACGTTAGCCAGTACGT",
        "GCGCGCGCGC",
    ];

    for input in &inputs {
        let text = encode(input);
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let recovered = inverse_bwt(&bwt);
        assert_eq!(recovered, text, "inverse BWT failed for '{}'", input);
    }
}

#[test]
fn c_array_values_correct() {
    let text = encode("ACGTACGT");
    let sa = build_suffix_array(&text);
    let bwt = build_bwt(&text, &sa);
    let c_array = CArray::from_bwt(&bwt);

    // Count each character in BWT (which is a permutation of text)
    let mut freq = [0u32; ALPHABET_SIZE];
    for ch in bwt.iter_chars() {
        freq[ch as usize] += 1;
    }

    // C[i] should be sum of freq[0..i)
    let mut expected_c = [0u32; ALPHABET_SIZE];
    let mut sum = 0u32;
    for i in 0..ALPHABET_SIZE {
        expected_c[i] = sum;
        sum += freq[i];
    }

    assert_eq!(c_array.data, expected_c);
}

#[test]
fn occ_table_matches_naive_for_all_positions() {
    let inputs = ["ACGT", "ACGTACGT", "AAACCCGGGTTT"];

    for input in &inputs {
        let text = encode(input);
        let sa = build_suffix_array(&text);
        let bwt = build_bwt(&text, &sa);
        let occ = build_occ_table(&bwt, OccEncoding::Bitplane);

        let n = bwt.len() as u32;
        for c in 0..ALPHABET_SIZE as u8 {
            for i in 0..=n {
                let expected = naive_rank(&bwt, c, i);
                let actual = occ.rank(c, i);
                assert_eq!(
                    actual, expected,
                    "Occ({}, {}) mismatch for '{}': got {} expected {}",
                    c, i, input, actual, expected
                );
            }
        }
    }
}

#[test]
fn occ_table_multi_block() {
    // Text longer than 64 chars to test multi-block behavior
    let input = "ACGT".repeat(25); // 100 chars
    let text = encode(&input);
    let sa = build_suffix_array(&text);
    let bwt = build_bwt(&text, &sa);
    let occ = build_occ_table(&bwt, OccEncoding::Bitplane);

    let n = bwt.len() as u32;
    for c in 0..ALPHABET_SIZE as u8 {
        // Test at block boundaries and random positions
        for i in [0, 1, 63, 64, 65, 100, n / 2, n - 1, n] {
            if i > n {
                continue;
            }
            let expected = naive_rank(&bwt, c, i);
            let actual = occ.rank(c, i);
            assert_eq!(
                actual, expected,
                "Occ({}, {}) mismatch in multi-block test",
                c, i
            );
        }
    }
}
