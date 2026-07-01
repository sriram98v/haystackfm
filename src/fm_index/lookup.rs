//! Depth-k prefix lookup table for FM-index backward search.
//!
//! Stores one SA interval `(lo, hi)` for every core-symbol k-mer of a fixed depth.
//! `backward_search` seeds from this table when the last `depth` query characters
//! are all core symbols, skipping those `depth` character steps entirely (O(1)
//! vs O(depth × rank_calls) for plain DNA queries).
//!
//! Default core symbols are ACGT (radix 4): memory = 4^depth × 8 bytes.
//! depth=10 → ~8 MB, depth=13 → ~537 MB.

use crate::c_array::CArray;
use crate::occ::OccTable;

/// A fixed-depth table mapping every core-symbol k-mer to its SA interval.
///
/// The radix (number of core symbols) is stored so that `get` can decode the
/// base-`radix` index without hard-coding ACGT.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LookupTable {
    /// Number of characters covered (the k in k-mer).
    pub depth: u32,
    /// The core (unambiguous) symbols used as the BFS radix, stored for `get`.
    core_symbols: Vec<u8>,
    /// Flat array of size `radix^depth` indexed by base-`radix` k-mer code.
    /// Entry `(0, 0)` means empty interval.
    intervals: Vec<(u32, u32)>,
}

impl LookupTable {
    /// Build the lookup table by BFS over core-symbol k-mers up to `depth` using the
    /// already-constructed C-array and Occ table.
    ///
    /// `core_symbols` is the alphabet's unambiguous symbol set (e.g. `[A,C,G,T]`).
    /// The radix equals `core_symbols.len()`.
    ///
    /// Cost: O(radix^depth) rank operations — ~1 M for radix=4, depth=10.
    pub fn build(
        depth: u32,
        text_len: u32,
        c_array: &CArray,
        occ: &OccTable,
        core_symbols: &[u8],
    ) -> Self {
        assert!(depth > 0, "lookup depth must be ≥ 1");
        assert!(!core_symbols.is_empty(), "core_symbols must not be empty");
        let radix = core_symbols.len();
        let table_size = radix.pow(depth);
        // Initialise all entries to empty (lo == hi == 0).
        let mut intervals = vec![(0u32, 0u32); table_size];

        // BFS: start with the full SA interval and extend one core symbol at a time
        // from right to left (backward search direction).
        //
        // "index_so_far" is the base-`radix` encoding of the characters consumed so far
        // (most-recent character = least significant digit).

        let mut current: Vec<(usize, u32, u32)> = vec![(0, 0, text_len)];

        for level in 1..=depth as usize {
            let mut next: Vec<(usize, u32, u32)> = Vec::with_capacity(current.len() * radix);
            let is_last = level == depth as usize;

            for &(parent_idx, lo, hi) in &current {
                for (digit, &sym) in core_symbols.iter().enumerate() {
                    let child_idx = parent_idx * radix + digit;
                    if c_array.symbol_count(sym, text_len) == 0 {
                        // Symbol absent from text → empty interval; already (0,0).
                        continue;
                    }
                    let cv = c_array.get(sym);
                    let new_lo = cv + occ.rank(sym, lo);
                    let new_hi = cv + occ.rank(sym, hi);
                    if is_last {
                        intervals[child_idx] = (new_lo, new_hi);
                    } else if new_lo < new_hi {
                        next.push((child_idx, new_lo, new_hi));
                    }
                    // If not last level and interval is empty, descendants stay (0,0).
                }
            }
            if !is_last {
                current = next;
            }
        }

        Self {
            depth,
            core_symbols: core_symbols.to_vec(),
            intervals,
        }
    }

    /// Look up the SA interval for a slice of codes exactly `depth` long.
    ///
    /// `codes` is ordered left-to-right (as the pattern appears). The rightmost
    /// code maps to the LSB of the base-`radix` index (matching BFS construction).
    ///
    /// Returns `None` if any symbol is not in `core_symbols` (caller falls back to
    /// full search). Returns `Some((lo, hi))`; `lo == hi` means no match.
    #[inline]
    pub fn get(&self, codes: &[u8]) -> Option<(u32, u32)> {
        debug_assert_eq!(codes.len(), self.depth as usize);
        let radix = self.core_symbols.len();
        let mut idx = 0usize;
        // Reverse iteration: rightmost code → LSB.
        for &c in codes.iter().rev() {
            let d = self.core_symbols.iter().position(|&s| s == c)?;
            idx = idx * radix + d;
        }
        Some(self.intervals[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::{DnaSequence, encode_char};
    use crate::fm_index::{FmIndex, FmIndexConfig};
    use crate::alphabet::ExactDna;

    fn make_index_with_lookup(s: &str, depth: u32) -> FmIndex {
        let seq = DnaSequence::from_str(s).unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            lookup_depth: depth,
            build_threads: 1,
        };
        FmIndex::build_cpu(&[seq], &config).unwrap()
    }

    fn make_exact_index_with_lookup(s: &str, depth: u32) -> FmIndex {
        let seq = DnaSequence::from_str(s).unwrap();
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            lookup_depth: depth,
            build_threads: 1,
        };
        FmIndex::build_cpu_with::<ExactDna>(&[seq], &config).unwrap()
    }

    fn encode(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    #[test]
    fn lookup_count_matches_full_search() {
        let text = "ACGTTAGCCAGTACGT";
        let idx_with = make_index_with_lookup(text, 3);
        let idx_no = {
            let seq = DnaSequence::from_str(text).unwrap();
            let cfg = FmIndexConfig {
                sa_sample_rate: 1,
                use_gpu: false,
                lookup_depth: 0,
                build_threads: 1,
            };
            FmIndex::build_cpu(&[seq], &cfg).unwrap()
        };

        for pat in &["ACG", "CGT", "GCC", "TAG", "ACGT", "TTA", "AAA"] {
            let enc = encode(pat);
            assert_eq!(
                idx_with.count(&enc),
                idx_no.count(&enc),
                "count mismatch for '{}' with lookup vs without",
                pat
            );
        }
    }

    #[test]
    fn lookup_locate_matches_full_search() {
        let text = "ACGTACGTACGT";
        let idx_with = make_index_with_lookup(text, 4);
        let idx_no = {
            let seq = DnaSequence::from_str(text).unwrap();
            let cfg = FmIndexConfig {
                sa_sample_rate: 1,
                use_gpu: false,
                lookup_depth: 0,
                build_threads: 1,
            };
            FmIndex::build_cpu(&[seq], &cfg).unwrap()
        };

        let enc = encode("ACGT");
        let mut pos_with = idx_with.locate_positions(&enc);
        let mut pos_no = idx_no.locate_positions(&enc);
        pos_with.sort();
        pos_no.sort();
        assert_eq!(pos_with, pos_no);
    }

    #[test]
    fn lookup_short_pattern_falls_back() {
        // Pattern shorter than depth → falls back to full backward search.
        let idx = make_index_with_lookup("ACGTACGT", 4);
        let enc = encode("ACG"); // len 3 < depth 4
        assert_eq!(idx.count(&enc), 2);
    }

    // ── ExactDna alphabet tests ────────────────────────────────────────────────

    #[test]
    fn exact_dna_query_n_returns_zero_hits() {
        // Text has N in it; query N should match nothing with ExactDna.
        let text = "ACGTNACGT";
        let idx_exact = make_exact_index_with_lookup(text, 3);
        let enc_n = encode("N");
        assert_eq!(
            idx_exact.count(&enc_n),
            0,
            "ExactDna: query N should return 0 hits"
        );
        // Same query with IupacDna should find hits (N matches A,C,G,T).
        let idx_iupac = make_index_with_lookup(text, 3);
        assert!(
            idx_iupac.count(&enc_n) > 0,
            "IupacDna: query N should match something"
        );
    }

    #[test]
    fn exact_dna_acgt_counts_match_iupac() {
        // For pure ACGT queries, ExactDna and IupacDna must agree.
        let text = "ACGTTAGCCAGTACGT";
        let idx_exact = make_exact_index_with_lookup(text, 3);
        let idx_iupac = make_index_with_lookup(text, 3);
        for pat in &["ACG", "CGT", "GCC", "TAG", "ACGT"] {
            let enc = encode(pat);
            assert_eq!(
                idx_exact.count(&enc),
                idx_iupac.count(&enc),
                "ExactDna vs IupacDna count mismatch for '{}'",
                pat
            );
        }
    }

    #[test]
    fn lookup_iupac_consistency_with_n_in_text() {
        // Text contains N. With and without lookup table, IupacDna results must agree.
        let text = "ACGTNACGT";
        let idx_with = make_index_with_lookup(text, 3);
        let idx_no = {
            let seq = DnaSequence::from_str(text).unwrap();
            FmIndex::build_cpu(
                &[seq],
                &FmIndexConfig {
                    sa_sample_rate: 1,
                    use_gpu: false,
                    lookup_depth: 0,
                    build_threads: 1,
                },
            )
            .unwrap()
        };
        for pat in &["ACG", "CGT", "N", "ACGT"] {
            let enc = encode(pat);
            assert_eq!(
                idx_with.count(&enc),
                idx_no.count(&enc),
                "IupacDna lookup/no-lookup mismatch for '{}'",
                pat
            );
        }
    }

    #[test]
    fn lookup_exact_consistency_with_n_in_text() {
        // Text contains N. With and without lookup table, ExactDna results must agree.
        let text = "ACGTNACGT";
        let idx_with = make_exact_index_with_lookup(text, 3);
        let idx_no = {
            let seq = DnaSequence::from_str(text).unwrap();
            FmIndex::build_cpu_with::<ExactDna>(
                &[seq],
                &FmIndexConfig {
                    sa_sample_rate: 1,
                    use_gpu: false,
                    lookup_depth: 0,
                    build_threads: 1,
                },
            )
            .unwrap()
        };
        for pat in &["ACG", "CGT", "N", "ACGT"] {
            let enc = encode(pat);
            assert_eq!(
                idx_with.count(&enc),
                idx_no.count(&enc),
                "ExactDna lookup/no-lookup mismatch for '{}'",
                pat
            );
        }
    }
}
