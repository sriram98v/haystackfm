use crate::alphabet::ALPHABET_SIZE;
use crate::bwt::Bwt;

/// C array: C[c] = number of characters in the text that are lexicographically smaller than c.
///
/// Covers the full IUPAC alphabet {$=0,A=1,C=2,G=3,T=4,N=5,R=6,Y=7,S=8,W=9,K=10,M=11,B=12,D=13,H=14,V=15}.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CArray {
    pub data: [u32; ALPHABET_SIZE],
}

impl CArray {
    /// Build C array from BWT (or any text) by counting character frequencies.
    pub fn from_text(text: &[u8]) -> Self {
        let mut freq = [0u32; ALPHABET_SIZE];
        for &ch in text {
            if (ch as usize) < ALPHABET_SIZE {
                freq[ch as usize] += 1;
            }
        }

        // Exclusive prefix sum
        let mut data = [0u32; ALPHABET_SIZE];
        let mut sum = 0u32;
        for i in 0..ALPHABET_SIZE {
            data[i] = sum;
            sum += freq[i];
        }

        Self { data }
    }

    /// Build C array from a packed BWT (4-bit encoded).
    pub fn from_bwt(bwt: &Bwt) -> Self {
        let mut freq = [0u32; ALPHABET_SIZE];
        for ch in bwt.iter_chars() {
            if (ch as usize) < ALPHABET_SIZE {
                freq[ch as usize] += 1;
            }
        }
        let mut data = [0u32; ALPHABET_SIZE];
        let mut sum = 0u32;
        for i in 0..ALPHABET_SIZE {
            data[i] = sum;
            sum += freq[i];
        }
        Self { data }
    }

    /// Get C[c]: number of characters smaller than c in the text.
    pub fn get(&self, c: u8) -> u32 {
        self.data[c as usize]
    }

    /// Number of occurrences of symbol `c` in the text.
    ///
    /// Used to skip absent symbols in backward search without calling `rank`.
    #[inline]
    pub fn symbol_count(&self, c: u8, text_len: u32) -> u32 {
        let next = (c as usize) + 1;
        let upper = if next < ALPHABET_SIZE {
            self.data[next]
        } else {
            text_len
        };
        upper - self.data[c as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::*;

    #[test]
    fn test_c_array_basic() {
        // text: A C G T $  => freq: $=1, A=1, C=1, G=1, T=1, rest=0
        let text = vec![A, C, G, T, SENTINEL];
        let c = CArray::from_text(&text);
        assert_eq!(c.data, [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn test_c_array_repeated() {
        // text: A A C C $  => freq: $=1, A=2, C=2, rest=0
        let text = vec![A, A, C, C, SENTINEL];
        let c = CArray::from_text(&text);
        assert_eq!(c.data, [0, 1, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn test_c_array_empty() {
        let text: Vec<u8> = vec![];
        let c = CArray::from_text(&text);
        assert_eq!(c.data, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_c_array_with_n() {
        // text: A N N $  => freq: $=1, A=1, N=2, rest=0
        // prefix sums: idx 5(N) ends at 2+0=2, idx 6(R) = 2+2 = 4
        let text = vec![A, N, N, SENTINEL];
        let c = CArray::from_text(&text);
        assert_eq!(c.data, [0, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]);
    }

    #[test]
    fn test_c_array_with_iupac() {
        // text: A R $  => freq: $=1, A=1, R=1
        let text = vec![A, R, SENTINEL];
        let c = CArray::from_text(&text);
        // prefix: [0=0, A=1, C=2, G=2, T=2, N=2, R=2, Y=3, ...]
        assert_eq!(c.data[0], 0); // $
        assert_eq!(c.data[A as usize], 1);
        assert_eq!(c.data[C as usize], 2); // +A
        assert_eq!(c.data[N as usize], 2); // no C,G,T
        assert_eq!(c.data[R as usize], 2); // +N=0
        assert_eq!(c.data[Y as usize], 3); // +R=1
    }
}
