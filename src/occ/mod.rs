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
    /// in bwt[0..sb*SUPERBLOCK_SIZE). Touched once per SUPERBLOCK_SIZE (512) positions,
    /// so it stays a separate array rather than bloating the per-block record.
    superblock_checkpoints: Vec<u32>,
    /// Per-block record, interleaved so one `rank`/`lf_step` call touches a single
    /// contiguous slice instead of 3 independent arrays (was `block_deltas: Vec<u16>` +
    /// `planes: Vec<u64>` in separate `Vec`s — each `rank` call was 2-3 cache misses on a
    /// large index; profiling showed ~58% of `backward_search` self-time was these reads).
    /// Layout per block: `[deltas: num_lanes x u16][planes-or-bitvecs: (num_planes|num_lanes) x u64]`,
    /// depending on `encoding` (see [`OccEncoding`]).
    /// `block_stride` = `num_lanes*2 + (num_planes or num_lanes)*8` bytes; block `b`'s record
    /// starts at `block_data[b*block_stride..]`.
    block_data: Vec<u8>,
    /// Byte stride of one block's record in `block_data`.
    block_stride: usize,
    /// Which Level-3 lane encoding `block_data`'s trailing region uses.
    /// `#[serde(default)]` so indices serialized before this field existed deserialize as
    /// `Bitplane` (the only encoding that ever existed on disk).
    #[serde(default)]
    encoding: OccEncoding,
    pub text_len: u32,
}

/// Sentinel lane value meaning "symbol never appears in this BWT".
const NO_LANE: u8 = u8::MAX;

/// Per-block lane encoding used by the occ table's Level 3 storage.
///
/// `Bitplane` (default) stores `ceil(log2(num_lanes))` `u64` planes per block — smaller
/// resident memory, but `rank`/`lf_step` reconstruct a one-hot mask via an AND/XOR loop over
/// the planes before popcounting. `OneHot` instead stores one `u64` bitvector per lane per
/// block directly — larger (up to 16x for large alphabets, but the alphabet is already
/// compacted to the symbols present in the BWT so this is closer to a `num_lanes / num_planes`
/// factor in practice), but `rank`/`lf_step` skip the reconstruction loop entirely (single
/// load + popcount). Choose `OneHot` when query latency matters more than resident memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum OccEncoding {
    #[default]
    Bitplane,
    OneHot,
}

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
    /// `lane_data` holds, per block, either `ceil(log2(num_lanes))` `u64` bitplanes
    /// (`encoding == Bitplane`) or `num_lanes` `u64` one-hot bitvectors (`encoding == OneHot`)
    /// — see [`OccEncoding`] and struct docs.
    pub fn from_parts(
        num_lanes: u8,
        symbol_to_lane: [u8; ALPHABET_SIZE],
        superblock_checkpoints: Vec<u32>,
        block_deltas: Vec<u16>,
        lane_data: Vec<u64>,
        text_len: u32,
        encoding: OccEncoding,
    ) -> Self {
        let mut lane_to_symbol = [0u8; ALPHABET_SIZE];
        for (c, &lane) in symbol_to_lane.iter().enumerate() {
            if lane != NO_LANE {
                lane_to_symbol[lane as usize] = c as u8;
            }
        }
        let num_planes = num_planes_for(num_lanes);
        let num_lanes_usize = num_lanes as usize;
        let num_planes_usize = num_planes as usize;
        // Level-3 record width: `num_planes` u64s for Bitplane, `num_lanes` u64s for OneHot.
        let lane_data_width = match encoding {
            OccEncoding::Bitplane => num_planes_usize,
            OccEncoding::OneHot => num_lanes_usize,
        };
        // sb_counts region duplicates each block's superblock checkpoint (normally read from
        // the separate `superblock_checkpoints` array) so `rank`/`lf_step` touch one cache
        // line instead of two: LF-walk/backward-search positions are effectively random, so
        // superblock reuse across consecutive blocks never happens in practice — the
        // superblock array access was a fresh miss on every call regardless of how rarely it
        // changes value. Costs `num_lanes*4` bytes/block (duplicated storage) for one fewer
        // miss per query step.
        let block_stride = num_lanes_usize * 4 + num_lanes_usize * 2 + lane_data_width * 8;
        let num_blocks = block_deltas.len().checked_div(num_lanes_usize).unwrap_or(0);
        let blocks_per_sb = (SUPERBLOCK_SIZE / BLOCK_SIZE) as usize;
        // Interleave the builders' separate `superblock_checkpoints`/`block_deltas`/`lane_data`
        // arrays into one contiguous, cache-line-sized record per block (see `block_data`
        // docs): a single `rank`/`lf_step` call then touches one slice instead of striding
        // through three independently-indexed arrays.
        let mut block_data = vec![0u8; num_blocks * block_stride];
        for b in 0..num_blocks {
            let sb = b / blocks_per_sb.max(1);
            let rec = &mut block_data[b * block_stride..(b + 1) * block_stride];
            for lane in 0..num_lanes_usize {
                let sb_count = superblock_checkpoints[sb * num_lanes_usize + lane];
                rec[lane * 4..lane * 4 + 4].copy_from_slice(&sb_count.to_ne_bytes());
            }
            let deltas_off = num_lanes_usize * 4;
            for lane in 0..num_lanes_usize {
                let delta = block_deltas[b * num_lanes_usize + lane];
                rec[deltas_off + lane * 2..deltas_off + lane * 2 + 2]
                    .copy_from_slice(&delta.to_ne_bytes());
            }
            let lane_data_off = deltas_off + num_lanes_usize * 2;
            for w in 0..lane_data_width {
                let word = lane_data[b * lane_data_width + w];
                rec[lane_data_off + w * 8..lane_data_off + w * 8 + 8]
                    .copy_from_slice(&word.to_ne_bytes());
            }
        }
        Self {
            num_lanes,
            num_planes,
            symbol_to_lane,
            lane_to_symbol,
            superblock_checkpoints,
            block_data,
            block_stride,
            encoding,
            text_len,
        }
    }

    // Unaligned pointer reads instead of `slice[..].try_into().unwrap()`: the safe form
    // compiles to an extra bounds check plus an array-literal copy per call. Since these run
    // on every `rank`/`lf_step` call (the hottest loop in the crate), that overhead alone ate
    // the entire win from interleaving sb_counts+deltas+planes into one cache line.
    //
    // Each takes the record's byte `base` (`block * block_stride`, computed once by the
    // caller) rather than `block` itself: `block_stride` is a runtime field, not a compile-time
    // constant, so `block * block_stride` is a real multiply — recomputing it independently in
    // every accessor call (3 per `rank`, plus once per plane in `lane_mask`'s loop) measurably
    // added instructions back on top of the miss-count win from interleaving.
    #[inline]
    fn block_base(&self, block: usize) -> usize {
        block * self.block_stride
    }

    #[inline]
    fn sb_count_at(&self, base: usize, lane: usize) -> u32 {
        let off = base + lane * 4;
        debug_assert!(off + 4 <= self.block_data.len());
        // SAFETY: `off + 4 <= block_data.len()` because `base` is a valid block's offset and
        // `lane < num_lanes`; the sb_counts region occupies `[0, num_lanes*4)` of each
        // block's `block_stride`-byte record (see `from_parts`/struct docs).
        unsafe {
            self.block_data
                .as_ptr()
                .add(off)
                .cast::<u32>()
                .read_unaligned()
        }
    }

    #[inline]
    fn delta_at(&self, base: usize, lane: usize) -> u32 {
        let num_lanes = self.num_lanes as usize;
        let off = base + num_lanes * 4 + lane * 2;
        debug_assert!(off + 2 <= self.block_data.len());
        // SAFETY: `off + 2 <= block_data.len()` because `base` is a valid block's offset and
        // `lane < num_lanes`; the deltas region occupies
        // `[num_lanes*4, num_lanes*4 + num_lanes*2)` of each block's record.
        unsafe {
            self.block_data
                .as_ptr()
                .add(off)
                .cast::<u16>()
                .read_unaligned() as u32
        }
    }

    #[inline]
    /// Reads word `w` of the trailing Level-3 region: the `w`-th bitplane under
    /// `OccEncoding::Bitplane`, or lane `w`'s one-hot bitvector under `OccEncoding::OneHot`.
    fn word_at(&self, base: usize, w: usize) -> u64 {
        let num_lanes = self.num_lanes as usize;
        let off = base + num_lanes * 4 + num_lanes * 2 + w * 8;
        debug_assert!(off + 8 <= self.block_data.len());
        // SAFETY: `off + 8 <= block_data.len()` because `base` is a valid block's offset and
        // `w` is in range for the current encoding (`num_planes` for Bitplane, `num_lanes` for
        // OneHot); the trailing region occupies `[num_lanes*6, num_lanes*6 + width*8)` of each
        // block's record.
        unsafe {
            self.block_data
                .as_ptr()
                .add(off)
                .cast::<u64>()
                .read_unaligned()
        }
    }

    /// Number of distinct symbols with a dedicated occ lane (`<= ALPHABET_SIZE`).
    /// A plain ACGT[+N][+$] reference compacts down to 5-6; GPU-built tables always use 16.
    pub fn num_lanes(&self) -> u8 {
        self.num_lanes
    }

    /// Issue a software prefetch for the block record containing BWT position `pos`, ahead
    /// of a future `lf_step(pos)` call. Used by `resolve_sa_batch`'s lockstep LF-walk to hide
    /// miss latency: while one lane's current-round `lf_step` result is consumed, the next
    /// round's block for another lane is already in flight.
    #[inline]
    pub(crate) fn prefetch_block(&self, pos: u32) {
        let block = (pos / BLOCK_SIZE) as usize;
        let base = self.block_base(block);
        if base < self.block_data.len() {
            crate::prefetch::prefetch_read(unsafe { self.block_data.as_ptr().add(base) });
        }
    }

    /// One-hot bitvector for `lane` within block `block`: bit j set iff position j of the
    /// block has that lane. Under `OccEncoding::OneHot` this is a direct load; under
    /// `OccEncoding::Bitplane` it's recovered via AND/XOR against the lane's bit pattern.
    #[inline]
    fn lane_mask(&self, base: usize, lane: usize) -> u64 {
        if self.encoding == OccEncoding::OneHot {
            if self.num_lanes as usize == 0 {
                return 0;
            }
            return self.word_at(base, lane);
        }
        let num_planes = self.num_planes as usize;
        if num_planes == 0 {
            // 0 or 1 lanes total: every occupied position trivially belongs to lane 0.
            return u64::MAX;
        }
        let mut mask = u64::MAX;
        for p in 0..num_planes {
            let plane_val = self.word_at(base, p);
            mask &= if (lane >> p) & 1 == 1 {
                plane_val
            } else {
                !plane_val
            };
        }
        mask
    }

    /// Lane index at position `offset` within the block at byte `base`.
    /// Under `OccEncoding::OneHot`, at most one lane's bitvector has the bit set at any given
    /// position, so this scans lanes for the set bit (O(num_lanes), vs O(num_planes) for
    /// Bitplane's direct decode) — the cost `OneHot` trades away resident memory for.
    #[inline]
    fn lane_at(&self, base: usize, offset: u32) -> u8 {
        if self.encoding == OccEncoding::OneHot {
            let num_lanes = self.num_lanes as usize;
            for lane in 0..num_lanes {
                if (self.word_at(base, lane) >> offset) & 1 == 1 {
                    return lane as u8;
                }
            }
            return 0;
        }
        let num_planes = self.num_planes as usize;
        if num_planes == 0 {
            return 0;
        }
        let mut lane = 0u8;
        for p in 0..num_planes {
            let bit = (self.word_at(base, p) >> offset) & 1;
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
        let pos = i - 1;
        let block = (pos / BLOCK_SIZE) as usize;
        let offset = pos % BLOCK_SIZE;
        let base = self.block_base(block);

        let sb_count = self.sb_count_at(base, lane);
        let delta = self.delta_at(base, lane);
        let lane_bits = self.lane_mask(base, lane);

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
        let base = self.block_base(block);
        self.lane_to_symbol[self.lane_at(base, offset) as usize]
    }

    /// Fused decode of `(lane, lane_mask)` at `offset` in one sweep over the block's planes,
    /// for `OccEncoding::Bitplane` only. `lane_at` + `lane_mask(base, lane)` each independently
    /// loop `0..num_planes` re-reading the same plane words — `lane_at` to extract bit `p` at
    /// `offset`, `lane_mask` to AND/XOR the full plane against that same bit. Since the bit
    /// `lane_mask` tests per plane (`(lane >> p) & 1`) is exactly the bit `lane_at` just
    /// extracted from that same plane at `offset`, both can be computed from a single load per
    /// plane instead of two. Halves the plane reads in `lf_step`'s LF-walk (the hottest loop in
    /// `locate`'s `resolve_sa`).
    #[inline]
    fn lane_and_mask_bitplane(&self, base: usize, offset: u32) -> (u8, u64) {
        let num_planes = self.num_planes as usize;
        if num_planes == 0 {
            return (0, u64::MAX);
        }
        let mut lane = 0u8;
        let mut mask = u64::MAX;
        for p in 0..num_planes {
            let plane_val = self.word_at(base, p);
            let bit = (plane_val >> offset) & 1;
            lane |= (bit as u8) << p;
            mask &= if bit == 1 { plane_val } else { !plane_val };
        }
        (lane, mask)
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
        let base = self.block_base(block);

        let (lane, lane_bits) = if self.encoding == OccEncoding::Bitplane {
            self.lane_and_mask_bitplane(base, offset)
        } else {
            let lane = self.lane_at(base, offset);
            (lane, self.lane_mask(base, lane as usize))
        };
        let lane = lane as usize;
        let symbol = self.lane_to_symbol[lane];

        let sb_count = self.sb_count_at(base, lane);
        let delta = self.delta_at(base, lane);

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
        let num_blocks = self.block_data.len() / self.block_stride.max(1);
        let mut out = Vec::with_capacity(num_blocks * BLOCK_SIZE as usize);
        for b in 0..num_blocks {
            let base = self.block_base(b);
            for offset in 0..BLOCK_SIZE {
                let sym = self.lane_to_symbol[self.lane_at(base, offset) as usize];
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
        let num_blocks = self.block_data.len() / self.block_stride.max(1);
        let num_lanes = self.num_lanes as usize;
        (0..num_blocks)
            .map(|b| {
                let sb = b / blocks_per_sb;
                let base = self.block_base(b);
                let mut combined = [0u32; ALPHABET_SIZE];
                for c in 0..ALPHABET_SIZE {
                    let lane = self.symbol_to_lane[c];
                    if lane == NO_LANE {
                        continue;
                    }
                    let lane = lane as usize;
                    combined[c] = self.superblock_checkpoints[sb * num_lanes + lane]
                        + self.delta_at(base, lane);
                }
                combined
            })
            .collect()
    }

    /// Reconstruct full 16-lane per-block bitvectors for GPU upload (see
    /// [`Self::flat_block_checkpoints`] for why GPU always uses the full alphabet width).
    #[cfg(feature = "gpu")]
    pub fn bitvectors_full16(&self) -> Vec<[u64; ALPHABET_SIZE]> {
        let num_blocks = self.block_data.len() / self.block_stride.max(1);
        (0..num_blocks)
            .map(|b| {
                let base = self.block_base(b);
                let mut bv = [0u64; ALPHABET_SIZE];
                for c in 0..ALPHABET_SIZE {
                    let lane = self.symbol_to_lane[c];
                    if lane == NO_LANE {
                        continue;
                    }
                    bv[c] = self.lane_mask(base, lane as usize);
                }
                bv
            })
            .collect()
    }
}
