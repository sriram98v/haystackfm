//! Suffix array construction and sampled suffix array for locate queries.

pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

/// Suffix array: SA[i] = starting position of the i-th lexicographically smallest suffix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuffixArray {
    pub data: Vec<u32>,
}

impl SuffixArray {
    /// Returns the number of entries in the suffix array (equals the text length).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the suffix array is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Sampled suffix array for space-efficient locate queries.
///
/// Stores only the ~n/sample_rate sampled entries (where SA[i] % sample_rate == 0)
/// as two parallel sorted Vecs instead of a flat n-element array.
/// Memory: 8n/sample_rate bytes vs the previous 4n bytes.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SampledSuffixArray {
    /// BWT row indices i where SA[i] % sample_rate == 0, sorted ascending.
    bwt_rows: Vec<u32>,
    /// SA value at each entry in bwt_rows (parallel array).
    sa_vals: Vec<u32>,
    pub sample_rate: u32,
}

impl SampledSuffixArray {
    /// Build a sampled SA from a full SA.
    pub fn from_full(sa: &SuffixArray, sample_rate: u32) -> Self {
        let mut bwt_rows = Vec::new();
        let mut sa_vals = Vec::new();
        for (i, &sa_val) in sa.data.iter().enumerate() {
            if sa_val.is_multiple_of(sample_rate) {
                bwt_rows.push(i as u32);
                sa_vals.push(sa_val);
            }
        }
        Self {
            bwt_rows,
            sa_vals,
            sample_rate,
        }
    }

    /// Check if BWT row `i` has a sampled SA value.
    pub fn is_sampled(&self, i: u32) -> bool {
        self.bwt_rows.binary_search(&i).is_ok()
    }

    /// Return the SA value for BWT row `i` if it is sampled.
    pub fn get(&self, i: u32) -> Option<u32> {
        self.bwt_rows
            .binary_search(&i)
            .ok()
            .map(|idx| self.sa_vals[idx])
    }

    /// Reconstruct the flat sentinel-format Vec<u32> required by GPU shaders.
    /// flat[i] = SA[i] if sampled, else u32::MAX.
    pub fn to_flat_vec(&self, n: usize) -> Vec<u32> {
        let mut flat = vec![u32::MAX; n];
        for (j, &row) in self.bwt_rows.iter().enumerate() {
            flat[row as usize] = self.sa_vals[j];
        }
        flat
    }
}
