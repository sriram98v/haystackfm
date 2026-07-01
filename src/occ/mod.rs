pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

use crate::alphabet::ALPHABET_SIZE;

/// Granularity of bitplanes (must be 64 — matches u64 popcount).
pub const BLOCK_SIZE: u32 = 64;

/// Spacing of u32 superblock checkpoints (must be a multiple of BLOCK_SIZE).
/// Blocks within a superblock store u16 deltas; max delta = SUPERBLOCK_SIZE ≤ 65535.
pub const SUPERBLOCK_SIZE: u32 = 512;

/// Occ table: O(1) rank queries over the BWT via a two-level checkpoint structure,
/// compacted to the symbol alphabet actually present in the BWT.
///
/// Level 1 — superblock checkpoints (every SUPERBLOCK_SIZE positions): u32 cumulative counts,
///   one per lane.
/// Level 2 — block deltas (every BLOCK_SIZE positions): u16 count since last superblock, one
///   per lane.
/// Level 3 — bitplanes: rather than one one-hot `u64` bitvector per lane (which wastes
///   `num_lanes - 1` bits per position — only one lane can ever be set), each block stores
///   `ceil(log2(num_lanes))` `u64` planes encoding the *binary representation* of each
///   position's lane index (bit `p` of the lane index lives in plane `p`). Recovering "is
///   this position lane L" is an AND/XOR of `num_planes` planes against L's bit pattern
///   followed by one popcount — more ALU per query than a direct load, but the FM-index rank
///   path is memory-bound, and this roughly halves resident memory for a 5-16 lane alphabet
///   (`ceil(log2(6))=3` vs `6` planes; `ceil(log2(16))=4` vs `16` for GPU-built tables).
///
/// `rank(c, i) = superblock[i/S][lane(c)] + delta[i/B][lane(c)] + popcount(lane_mask(i/B, lane(c)) & window)`
///
/// Reference text is IUPAC-encoded (`ALPHABET_SIZE = 16`), but a DNA/RNA reference BWT only
/// ever contains a handful of distinct symbols (typically `{$,A,C,G,T}`, maybe `N`). Ambiguity
/// matching (`N` etc.) is resolved at query time by expanding to compatible core symbols
/// (`src/fm_index/query.rs`), so absent symbols never need their own occ lane —
/// `symbol_to_lane` maps each of the 16 symbol codes down to a dense `[0, num_lanes)` lane index,
/// or `u8::MAX` if the symbol never appears in this BWT.
///
/// Memory vs fixed 16-lane one-hot layout (n=250M, effective alphabet = 6 lanes):
///   fixed 16 one-hot:  n/512×16×4 + n/64×16×2 + n/64×16×8            = 2.625n bytes ≈ 656 MB
///   compact 6 one-hot: n/512×6×4  + n/64×6×2  + n/64×6×8             = 0.984n bytes ≈ 246 MB
///   compact 6 planes:  n/512×6×4  + n/64×6×2  + n/64×3×8 (3 planes)  = 0.609n bytes ≈ 152 MB
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OccTable {
    /// Number of distinct symbols with a dedicated lane (`<= ALPHABET_SIZE`).
    num_lanes: u8,
    /// Number of bitplanes needed to encode a lane index: `ceil(log2(num_lanes))`, 0 if
    /// `num_lanes <= 1` (every occupied position trivially has the single lane).
    num_planes: u8,
    /// symbol code (0..ALPHABET_SIZE) -> dense lane index, or `u8::MAX` if absent from the BWT.
    symbol_to_lane: [u8; ALPHABET_SIZE],
    /// lane index -> symbol code; reverse of `symbol_to_lane`, precomputed once so
    /// `symbol_at`/`reconstruct_bwt_u32` (which the BWT is reconstructed from, since we no
    /// longer keep a separate resident `Bwt`) don't re-derive it on every call.
    lane_to_symbol: [u8; ALPHABET_SIZE],
    /// superblock_checkpoints[sb*num_lanes + lane] = cumulative count of lane's symbol
    /// in bwt[0..sb*SUPERBLOCK_SIZE)
    superblock_checkpoints: Vec<u32>,
    /// block_deltas[b*num_lanes + lane] = count since the block's superblock
    block_deltas: Vec<u16>,
    /// planes[b*num_planes + p] = bit `p` of the lane index at each position within block `b`
    /// (bit j set iff bit p of `lane_at(b, j)` is 1). See the struct doc for how a lane's
    /// one-hot bitvector is recovered from these.
    planes: Vec<u64>,
    pub text_len: u32,
}

/// Sentinel lane value meaning "symbol never appears in this BWT".
const NO_LANE: u8 = u8::MAX;

/// Number of bits needed to represent lane indices `[0, num_lanes)`.
fn num_planes_for(num_lanes: u8) -> u8 {
    if num_lanes <= 1 {
        0
    } else {
        (u8::BITS - (num_lanes - 1).leading_zeros()) as u8
    }
}

impl OccTable {
    /// Construct from raw compacted components (used by CPU and GPU builders).
    ///
    /// `planes` holds `ceil(log2(num_lanes))` `u64` bitplanes per block (see struct docs).
    pub fn from_parts(
        num_lanes: u8,
        symbol_to_lane: [u8; ALPHABET_SIZE],
        superblock_checkpoints: Vec<u32>,
        block_deltas: Vec<u16>,
        planes: Vec<u64>,
        text_len: u32,
    ) -> Self {
        let mut lane_to_symbol = [0u8; ALPHABET_SIZE];
        for (c, &lane) in symbol_to_lane.iter().enumerate() {
            if lane != NO_LANE {
                lane_to_symbol[lane as usize] = c as u8;
            }
        }
        Self {
            num_lanes,
            num_planes: num_planes_for(num_lanes),
            symbol_to_lane,
            lane_to_symbol,
            superblock_checkpoints,
            block_deltas,
            planes,
            text_len,
        }
    }

    /// Number of distinct symbols with a dedicated occ lane (`<= ALPHABET_SIZE`).
    /// A plain ACGT[+N][+$] reference compacts down to 5-6; GPU-built tables always use 16.
    pub fn num_lanes(&self) -> u8 {
        self.num_lanes
    }

    /// One-hot bitvector for `lane` within block `block`: bit j set iff position j of the
    /// block has that lane. Recovered from the bitplanes via AND/XOR against the lane's bit
    /// pattern — this is what a resident per-lane bitvector would have stored directly.
    #[inline]
    fn lane_mask(&self, block: usize, lane: usize) -> u64 {
        let num_planes = self.num_planes as usize;
        if num_planes == 0 {
            // 0 or 1 lanes total: every occupied position trivially belongs to lane 0.
            return u64::MAX;
        }
        let mut mask = u64::MAX;
        for p in 0..num_planes {
            let plane_val = self.planes[block * num_planes + p];
            mask &= if (lane >> p) & 1 == 1 {
                plane_val
            } else {
                !plane_val
            };
        }
        mask
    }

    /// Lane index at position `offset` within `block`, recovered from the bitplanes.
    #[inline]
    fn lane_at(&self, block: usize, offset: u32) -> u8 {
        let num_planes = self.num_planes as usize;
        if num_planes == 0 {
            return 0;
        }
        let mut lane = 0u8;
        for p in 0..num_planes {
            let bit = (self.planes[block * num_planes + p] >> offset) & 1;
            lane |= (bit as u8) << p;
        }
        lane
    }

    /// Rank query: count of character `c` in bwt[0..i). Symbols absent from this BWT
    /// (no dedicated lane) always have rank 0 — callers should already skip them via
    /// `CArray::symbol_count`, but this is a safe fallback.
    pub fn rank(&self, c: u8, i: u32) -> u32 {
        if i == 0 {
            return 0;
        }
        let lane = self.symbol_to_lane[c as usize];
        if lane == NO_LANE {
            return 0;
        }
        let lane = lane as usize;
        let num_lanes = self.num_lanes as usize;
        let pos = i - 1;
        let block = (pos / BLOCK_SIZE) as usize;
        let offset = pos % BLOCK_SIZE;
        let sb = (pos / SUPERBLOCK_SIZE) as usize;

        let sb_count = self.superblock_checkpoints[sb * num_lanes + lane];
        let delta = self.block_deltas[block * num_lanes + lane] as u32;
        let lane_bits = self.lane_mask(block, lane);

        let mask = if offset == 63 {
            u64::MAX
        } else {
            (1u64 << (offset + 1)) - 1
        };

        sb_count + delta + (lane_bits & mask).count_ones()
    }

    /// Symbol at BWT position `pos`, recovered from the occ bitplanes.
    ///
    /// We no longer keep a separate resident `Bwt` (it duplicated exactly what the occ
    /// bitplanes already encode — see `FmIndex::lf_mapping`), so this is the only way to
    /// recover the character at a given row.
    pub fn symbol_at(&self, pos: u32) -> u8 {
        let block = (pos / BLOCK_SIZE) as usize;
        let offset = pos % BLOCK_SIZE;
        self.lane_to_symbol[self.lane_at(block, offset) as usize]
    }

    /// Fused LF-step primitive: returns `(symbol_at(pos), rank(symbol_at(pos), pos))` in a
    /// single pass over `pos`'s block planes, instead of the two independent calls this
    /// replaces (`symbol_at` + `rank`), which each re-read and re-derive the same block's
    /// lane mask. Used by [`crate::fm_index::FmIndex::lf_mapping`], the hottest loop in
    /// `locate` (one call per LF step during `resolve_sa`).
    #[inline]
    pub fn lf_step(&self, pos: u32) -> (u8, u32) {
        let block = (pos / BLOCK_SIZE) as usize;
        let offset = pos % BLOCK_SIZE;
        let lane = self.lane_at(block, offset) as usize;
        let symbol = self.lane_to_symbol[lane];

        let num_lanes = self.num_lanes as usize;
        let sb = (pos / SUPERBLOCK_SIZE) as usize;
        let sb_count = self.superblock_checkpoints[sb * num_lanes + lane];
        let delta = self.block_deltas[block * num_lanes + lane] as u32;
        let lane_bits = self.lane_mask(block, lane);

        // rank(symbol, pos) counts occurrences in bwt[0..pos), i.e. bits strictly below
        // `offset` within this block (mirrors `rank`'s `(1 << (offset+1)) - 1` mask for
        // `i = pos + 1`, minus the bit at `offset` itself).
        let mask = (1u64 << offset) - 1; // offset in [0, 63]; offset==0 => mask==0
        let rank = sb_count + delta + (lane_bits & mask).count_ones();

        (symbol, rank)
    }

    /// Reconstruct the full BWT as one `u32` per position, for GPU upload.
    ///
    /// O(n); GPU query paths (`locate`, `mem_resolve`) call this once per index instead of
    /// keeping a resident packed `Bwt` around, trading a one-time reconstruction pass for
    /// resident CPU memory.
    pub fn reconstruct_bwt_u32(&self) -> Vec<u32> {
        let num_planes = self.num_planes.max(1) as usize;
        let num_blocks = if self.num_planes == 0 {
            // Degenerate (<=1 lane) case: fall back to block_deltas/superblock layout to size
            // num_blocks, since there are no planes to divide by.
            self.block_deltas.len() / self.num_lanes.max(1) as usize
        } else {
            self.planes.len() / num_planes
        };
        let mut out = Vec::with_capacity(num_blocks * BLOCK_SIZE as usize);
        for b in 0..num_blocks {
            for offset in 0..BLOCK_SIZE {
                let sym = self.lane_to_symbol[self.lane_at(b, offset) as usize];
                out.push(sym as u32);
            }
        }
        out.truncate(self.text_len as usize);
        out
    }

    /// Reconstruct full 16-lane per-block u32 checkpoints for GPU upload.
    ///
    /// GPU shaders operate over the fixed 16-symbol IUPAC alphabet regardless of which
    /// symbols are present in any given reference, so this expands the compact lane layout
    /// back out, filling absent symbols with 0.
    #[cfg(feature = "gpu")]
    pub fn flat_block_checkpoints(&self) -> Vec<[u32; ALPHABET_SIZE]> {
        let blocks_per_sb = (SUPERBLOCK_SIZE / BLOCK_SIZE) as usize;
        let num_blocks = self.block_deltas.len() / self.num_lanes.max(1) as usize;
        let num_lanes = self.num_lanes as usize;
        (0..num_blocks)
            .map(|b| {
                let sb = b / blocks_per_sb;
                let mut combined = [0u32; ALPHABET_SIZE];
                for c in 0..ALPHABET_SIZE {
                    let lane = self.symbol_to_lane[c];
                    if lane == NO_LANE {
                        continue;
                    }
                    let lane = lane as usize;
                    combined[c] = self.superblock_checkpoints[sb * num_lanes + lane]
                        + self.block_deltas[b * num_lanes + lane] as u32;
                }
                combined
            })
            .collect()
    }

    /// Reconstruct full 16-lane per-block bitvectors for GPU upload (see
    /// [`Self::flat_block_checkpoints`] for why GPU always uses the full alphabet width).
    #[cfg(feature = "gpu")]
    pub fn bitvectors_full16(&self) -> Vec<[u64; ALPHABET_SIZE]> {
        let num_blocks = self.block_deltas.len() / self.num_lanes.max(1) as usize;
        (0..num_blocks)
            .map(|b| {
                let mut bv = [0u64; ALPHABET_SIZE];
                for c in 0..ALPHABET_SIZE {
                    let lane = self.symbol_to_lane[c];
                    if lane == NO_LANE {
                        continue;
                    }
                    bv[c] = self.lane_mask(b, lane as usize);
                }
                bv
            })
            .collect()
    }
}
