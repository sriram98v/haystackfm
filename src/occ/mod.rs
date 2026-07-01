pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

use crate::alphabet::ALPHABET_SIZE;

/// Granularity of bitvectors (must be 64 — matches u64 popcount).
pub const BLOCK_SIZE: u32 = 64;

/// Spacing of u32 superblock checkpoints (must be a multiple of BLOCK_SIZE).
/// Blocks within a superblock store u16 deltas; max delta = SUPERBLOCK_SIZE ≤ 65535.
pub const SUPERBLOCK_SIZE: u32 = 512;

/// Occ table: O(1) rank queries over the BWT via a two-level checkpoint structure.
///
/// Level 1 — superblock checkpoints (every SUPERBLOCK_SIZE positions): u32 cumulative counts.
/// Level 2 — block deltas (every BLOCK_SIZE positions): u16 count since last superblock.
/// Level 3 — bitvectors: u64 presence bit per position within the block.
///
/// rank(c, i) = superblock[i/S][c] + delta[i/B][c] + popcount(bitvec[i/B][c] & mask)
///
/// Memory vs single-level (n=250M, ALPHA=16):
///   old: n/64 × (16×4 + 16×8)  = 3n bytes   = 750 MB
///   new: n/512×16×4 + n/64×16×2 + n/64×16×8 = 2.625n bytes ≈ 656 MB  (−94 MB)
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OccTable {
    /// superblock_checkpoints[sb][c] = cumulative count of c in bwt[0..sb*SUPERBLOCK_SIZE)
    superblock_checkpoints: Vec<[u32; ALPHABET_SIZE]>,
    /// block_deltas[b][c] = count of c in bwt[sb_start..b*BLOCK_SIZE), where sb = b/(S/B)
    block_deltas: Vec<[u16; ALPHABET_SIZE]>,
    /// bitvectors[b][c] = 64-bit vector: bit j set iff bwt[b*BLOCK_SIZE + j] == c
    pub bitvectors: Vec<[u64; ALPHABET_SIZE]>,
    pub text_len: u32,
}

impl OccTable {
    /// Construct from raw components (used by CPU and GPU builders).
    pub fn from_parts(
        superblock_checkpoints: Vec<[u32; ALPHABET_SIZE]>,
        block_deltas: Vec<[u16; ALPHABET_SIZE]>,
        bitvectors: Vec<[u64; ALPHABET_SIZE]>,
        text_len: u32,
    ) -> Self {
        Self { superblock_checkpoints, block_deltas, bitvectors, text_len }
    }

    /// Rank query: count of character `c` in bwt[0..i).
    pub fn rank(&self, c: u8, i: u32) -> u32 {
        if i == 0 {
            return 0;
        }
        let c_idx = c as usize;
        let pos = i - 1;
        let block = (pos / BLOCK_SIZE) as usize;
        let offset = pos % BLOCK_SIZE;
        let sb = (pos / SUPERBLOCK_SIZE) as usize;

        let sb_count = self.superblock_checkpoints[sb][c_idx];
        let delta = self.block_deltas[block][c_idx] as u32;
        let bitvec = self.bitvectors[block][c_idx];

        let mask = if offset == 63 { u64::MAX } else { (1u64 << (offset + 1)) - 1 };

        sb_count + delta + (bitvec & mask).count_ones()
    }

    /// Reconstruct full per-block u32 checkpoints for GPU upload.
    ///
    /// The GPU shader expects flat cumulative u32 counts, one per (block, char).
    /// This materializes them from the compact two-level representation.
    pub fn flat_block_checkpoints(&self) -> Vec<[u32; ALPHABET_SIZE]> {
        let blocks_per_sb = (SUPERBLOCK_SIZE / BLOCK_SIZE) as usize;
        self.block_deltas
            .iter()
            .enumerate()
            .map(|(b, deltas)| {
                let sb = b / blocks_per_sb;
                let sb_counts = self.superblock_checkpoints[sb];
                let mut combined = [0u32; ALPHABET_SIZE];
                for c in 0..ALPHABET_SIZE {
                    combined[c] = sb_counts[c] + deltas[c] as u32;
                }
                combined
            })
            .collect()
    }
}
