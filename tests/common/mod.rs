use haystackfm::alphabet::{encode_char, DnaSequence, SENTINEL};

/// Helper: encode a DNA string with sentinel appended.
pub fn encode(s: &str) -> Vec<u8> {
    let mut v: Vec<u8> = s.chars().map(|c| encode_char(c).unwrap()).collect();
    if v.last() != Some(&SENTINEL) {
        v.push(SENTINEL);
    }
    v
}

/// Helper: encode a pattern (no sentinel).
pub fn encode_pattern(s: &str) -> Vec<u8> {
    s.chars().map(|c| encode_char(c).unwrap()).collect()
}

/// Helper: create a single-sequence FM-index with full SA.
pub fn make_index(s: &str) -> haystackfm::FmIndex {
    let seq = DnaSequence::from_str(s).unwrap();
    let config = haystackfm::FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: false,
        ..Default::default()
    };
    haystackfm::FmIndex::build_cpu(&[seq], &config).unwrap()
}

/// Helper: count overlapping occurrences via naive string matching.
pub fn naive_count(text: &str, pattern: &str) -> u32 {
    if pattern.is_empty() {
        return text.len() as u32 + 1;
    }
    naive_locate(text, pattern).len() as u32
}

/// Helper: find all (overlapping) positions of pattern in text via naive search.
pub fn naive_locate(text: &str, pattern: &str) -> Vec<usize> {
    if pattern.is_empty() || pattern.len() > text.len() {
        return vec![];
    }
    (0..=text.len() - pattern.len())
        .filter(|&i| &text[i..i + pattern.len()] == pattern)
        .collect()
}

/// Known test vectors: (text, pattern, expected_count).
pub fn test_vectors() -> Vec<(&'static str, &'static str, u32)> {
    vec![
        ("ACGT", "A", 1),
        ("ACGT", "C", 1),
        ("ACGT", "G", 1),
        ("ACGT", "T", 1),
        ("ACGT", "AC", 1),
        ("ACGT", "ACGT", 1),
        ("ACGTACGT", "ACGT", 2),
        ("ACGTACGT", "A", 2),
        ("AAAA", "A", 4),
        ("AAAA", "AA", 3),
        ("AAAA", "AAA", 2),
        ("AAAA", "AAAA", 1),
        ("AAAA", "C", 0),
        ("ACGTTAGCCAGTACGT", "GT", 3),
        ("ACGTTAGCCAGTACGT", "AGC", 1),
        ("ACGTTAGCCAGTACGT", "TAG", 1),
    ]
}
