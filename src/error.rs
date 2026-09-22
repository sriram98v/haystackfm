use thiserror::Error;

/// Errors returned by FM-index construction and query operations.
#[derive(Debug, Error)]
pub enum FmIndexError {
    /// A character outside the DNA alphabet (A, C, G, T) was encountered.
    #[error("invalid DNA character '{0}' at position {1}")]
    InvalidCharacter(char, usize),

    /// An empty sequence was provided where a non-empty sequence is required.
    #[error("empty sequence")]
    EmptySequence,

    /// Two sequences share a FASTA header, making the header -> id lookup ambiguous.
    #[error("duplicate sequence header '{0}': headers must be unique")]
    DuplicateHeader(String),

    /// The combined text length exceeds the `u32::MAX` limit.
    #[error("text too large: {0} bytes exceeds u32::MAX")]
    TextTooLarge(usize),

    /// Deserialization from bytes failed.
    #[error("deserialization failed: {0}")]
    DeserializeError(String),

    /// Serialization to bytes failed.
    #[error("serialization failed: {0}")]
    SerializeError(String),

    /// A cursor contraction was requested on an index built without an LCP array
    /// (`FmIndexConfig::build_lcp = false`, a GPU-built index, or a legacy serialized one).
    #[error("index has no LCP array; build with FmIndexConfig::build_lcp = true on the CPU")]
    LcpNotBuilt,

    /// The pattern is too long for the capped LCP array to decide a contraction.
    #[error("pattern length {0} exceeds the LCP cap (65535)")]
    PatternTooLong(u32),

    /// The interval / symbol pair passed to a contraction is not a valid `cP` interval
    /// (empty interval, zero length, or `c` is not the symbol the interval was extended by).
    #[error("invalid cursor contraction: interval is not the interval of c·P")]
    InvalidContraction,

    /// A GPU operation failed (only available with the `gpu` feature).
    #[cfg(feature = "gpu")]
    #[error("GPU error: {0}")]
    GpuError(String),
}
