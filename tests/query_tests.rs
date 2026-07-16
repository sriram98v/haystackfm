mod common;

use common::{encode_pattern, make_index, naive_count, naive_locate, test_vectors};

#[test]
fn count_matches_test_vectors() {
    for (text, pattern, expected) in test_vectors() {
        let idx = make_index(text);
        let actual = idx.count(&encode_pattern(pattern));
        assert_eq!(
            actual, expected,
            "count('{}' in '{}') = {} expected {}",
            pattern, text, actual, expected
        );
    }
}

#[test]
fn locate_positions_are_valid() {
    let texts = [
        "ACGT",
        "ACGTACGT",
        "ACGTTAGCCAGTACGT",
        "AAACCCGGGTTT",
        "ACACACACAC",
    ];
    let patterns = ["A", "C", "G", "T", "AC", "GT", "ACGT", "AA", "CC"];

    for text in &texts {
        let idx = make_index(text);
        let encoded_text: Vec<u8> = text
            .chars()
            .map(|c| haystackfm::alphabet::encode_char(c).unwrap())
            .collect();

        for pattern in &patterns {
            let p = encode_pattern(pattern);
            let positions = idx.locate(&p);

            // Every returned position should be valid
            for (_, pos) in &positions {
                let pos = *pos as usize;
                if pos + pattern.len() <= encoded_text.len() {
                    assert_eq!(
                        &encoded_text[pos..pos + pattern.len()],
                        p.as_slice(),
                        "'{}' at position {} in '{}'",
                        pattern,
                        pos,
                        text
                    );
                }
            }

            // Should find the right number
            let expected_count = naive_count(text, pattern);
            assert_eq!(
                positions.len() as u32,
                expected_count,
                "locate count mismatch for '{}' in '{}'",
                pattern,
                text
            );
        }
    }
}

#[test]
fn locate_matches_naive_positions() {
    let texts = ["ACGTACGTACGT", "ACGTTAGCCAGTACGT"];

    for text in &texts {
        let idx = make_index(text);

        for pattern in &["ACGT", "GT", "A", "GCC"] {
            let p = encode_pattern(pattern);
            let mut fm_positions: Vec<usize> = idx
                .locate(&p)
                .into_iter()
                .map(|(_, pos)| pos as usize)
                .collect();
            fm_positions.sort();

            let naive_positions = naive_locate(text, pattern);

            assert_eq!(
                fm_positions, naive_positions,
                "locate('{}' in '{}') mismatch",
                pattern, text
            );
        }
    }
}

#[test]
fn empty_pattern_count() {
    let idx = make_index("ACGT");
    // Empty pattern matches at every position (text_len includes sentinel)
    let count = idx.count(&[]);
    assert_eq!(count, idx.text_len());
}

#[test]
fn count_empty_result() {
    let idx = make_index("AAAA");
    assert_eq!(idx.count(&encode_pattern("G")), 0);
    assert_eq!(idx.count(&encode_pattern("ACGT")), 0);
}

#[test]
fn locate_empty_result() {
    let idx = make_index("AAAA");
    assert!(idx.locate(&encode_pattern("G")).is_empty());
}

#[test]
fn multi_sequence_queries() {
    let seqs = vec![
        haystackfm::DnaSequence::from_str("ACGTACGT").unwrap(),
        haystackfm::DnaSequence::from_str("TGCATGCA").unwrap(),
        haystackfm::DnaSequence::from_str("AAACCC").unwrap(),
    ];
    let config = haystackfm::FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: false,
        ..Default::default()
    };
    let idx = haystackfm::FmIndex::build_cpu(&seqs, &config).unwrap();

    // "ACGT" appears twice (in first sequence at pos 0 and 4)
    assert_eq!(idx.count(&encode_pattern("ACGT")), 2);

    // "TGCA" appears twice (in second sequence at pos 0 and 4)
    assert_eq!(idx.count(&encode_pattern("TGCA")), 2);

    // "AAA" appears once
    assert_eq!(idx.count(&encode_pattern("AAA")), 1);

    // "CCC" appears once
    assert_eq!(idx.count(&encode_pattern("CCC")), 1);

    assert_eq!(idx.num_sequences(), 3);
}
