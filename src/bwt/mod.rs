//! Burrows-Wheeler Transform (BWT) construction and representation.
//!
//! The BWT is a reversible permutation of the input text that groups similar
//! characters together, enabling efficient rank queries via the Occ table.

pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

/// Burrows-Wheeler Transform: a permutation of the input text.
///
/// Stored in 4-bit packed form (2 chars per byte) since IUPAC values fit in `[0,15]`.
/// Saves ~n/2 bytes vs `Vec<u8>` for large genomes.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Bwt {
    /// 4-bit packed characters: byte k holds chars at positions 2k (low nibble) and 2k+1 (high nibble).
    packed: Vec<u8>,
    /// Original length in characters (needed because last byte may hold only one valid nibble).
    text_len: usize,
}

impl Bwt {
    /// Build from a flat unpacked byte slice (values 0–15).
    pub(crate) fn from_unpacked(data: Vec<u8>) -> Self {
        let n = data.len();
        let mut packed = vec![0u8; n.div_ceil(2)];
        for (i, &b) in data.iter().enumerate() {
            if i.is_multiple_of(2) {
                packed[i / 2] = b & 0xF;
            } else {
                packed[i / 2] |= (b & 0xF) << 4;
            }
        }
        Self {
            packed,
            text_len: n,
        }
    }

    /// Return the character at BWT position `i`.
    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        if i.is_multiple_of(2) {
            self.packed[i / 2] & 0xF
        } else {
            (self.packed[i / 2] >> 4) & 0xF
        }
    }

    /// Iterate over all characters in BWT order.
    pub fn iter_chars(&self) -> impl Iterator<Item = u8> + '_ {
        (0..self.text_len).map(move |i| self.get(i))
    }

    /// Materialize as a `Vec<u32>` for GPU upload (one u32 per BWT char).
    pub fn to_u32_vec(&self) -> Vec<u32> {
        self.iter_chars().map(|b| b as u32).collect()
    }

    /// Returns the length of the BWT (equals the length of the original text).
    pub fn len(&self) -> usize {
        self.text_len
    }

    /// Returns `true` if the BWT is empty.
    pub fn is_empty(&self) -> bool {
        self.text_len == 0
    }
}
