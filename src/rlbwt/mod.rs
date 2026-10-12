//! Run-length encoded Burrows–Wheeler transform.
//!
//! A BWT of length `n` over a repetitive text consists of few maximal runs `r ≪ n` of equal
//! symbols. [`RlBwt`] stores those runs and answers the FM-index primitives ([`BwtRank`]) in
//! `O(r)` space, in the spirit of Mäkinen & Navarro's run-length FM-index, laid out so a
//! rank costs two cache lines instead of a chain of four lookups:
//!
//! - **run blocks**: the runs in groups of [`RUNS_PER_BLOCK`], one 64-byte-aligned block
//!   each (two when more than [`INLINE_LANES`] symbols occur) holding the group's run start
//!   rows plus the next group's first (so every run's end is on its own line), its run-head
//!   symbols (nibbles) and, per symbol present in the BWT, its rank at the group's first
//!   row;
//! - **bucket directory**: for every `2^bucket_shift` rows, the run holding the bucket's
//!   first row, narrowing the predecessor search to a few runs.
//!
//! ```text
//! k      = run containing row i               (directory + search over block starts)
//! b, t   = k / 8, k % 8
//! rank_c = sample_c[b] + Σ_{u<t, H[u]=c} len(u) + (H[t] == c ? i − start[t] : 0)
//! ```
//!
//! Every quantity a rank, an LF step or a bidirectional extension needs sits in the block
//! of `k` (and, for an interval, the next blocks). Only the runs are serialized; blocks and
//! directory are rebuilt and validated at load.
//!
//! This is the representation the move structure and its bidirectional variant (b-move)
//! build on: both index the same runs, adding per-run LF/φ tables.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::alphabet::{DnaSequence, SymbolSet, ALPHABET_SIZE};
use crate::bwt::cpu::build_bwt;
use crate::bwt::Bwt;
use crate::error::FmIndexError;
use crate::occ::{BwtRank, OccTable};

/// Average runs per bucket of the predecessor directory: a bucket costs 4 bytes, and the
/// search inside one touches one or two run blocks.
const RUNS_PER_BUCKET: u64 = 4;

/// Runs per block. A block's starts and heads fill words `0..=8`.
pub(crate) const RUNS_PER_BLOCK: usize = 8;

/// Present symbols whose rank samples fit in a single 64-byte block (words `10..16`). With
/// more, every block takes two lines.
const INLINE_LANES: usize = 6;

/// Word of a block's head nibbles. Words `0..=8` are the starts of its eight runs and of
/// the next block's first run, so a run's end never needs another line.
const HEAD_WORD: usize = RUNS_PER_BLOCK + 1;

/// Word of the first rank sample in a block.
const SAMPLE_WORD: usize = RUNS_PER_BLOCK + 2;

/// Runs [`BwtRank::extend_ranks`] walks before switching to two independent ranks.
const MAX_SCAN_RUNS: usize = 8;

/// No lane: the symbol does not occur.
const NO_LANE: u8 = u8::MAX;

/// One cache line of a run block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(64))]
struct Line([u32; 16]);

/// Run-length encoded BWT with rank/access support. See the [module docs](self).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlBwt {
    /// Number of BWT rows `n`.
    len: u32,
    /// Number of maximal runs `r`.
    num_runs: u32,
    /// Run blocks, `lines_per_block` lines each: `⌊r / 8⌋ + 1` blocks, so the start of run
    /// `r` (= `len`) is always stored. Starts past the last run are `len`, their heads 0.
    blocks: Vec<Line>,
    /// 1, or 2 when more than [`INLINE_LANES`] symbols occur.
    lines_per_block: usize,
    /// Rank-sample lane of each symbol, [`NO_LANE`] when absent.
    lane_of: [u8; ALPHABET_SIZE],
    /// Occurrences of each symbol in the whole BWT.
    totals: [u32; ALPHABET_SIZE],
    /// Symbols with at least one run.
    present: SymbolSet,
    /// log2 of the rows covered by one `buckets` entry.
    bucket_shift: u32,
    /// `buckets[b]` = run containing row `min(b << bucket_shift, len − 1)`.
    buckets: Vec<u32>,
}

/// On-disk form: the runs only. Run `k` has head `heads[k / 2] >> (4 * (k % 2)) & 15` and
/// covers rows `starts[k]..starts[k + 1]` (`len` for the last).
#[derive(Serialize, Deserialize)]
struct RlBwtWire {
    len: u32,
    #[serde(with = "crate::serde_raw::bytes")]
    heads: Vec<u8>,
    #[serde(with = "crate::serde_raw::u32s")]
    starts: Vec<u32>,
}

impl Serialize for RlBwt {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let r = self.num_runs as usize;
        let mut heads = vec![0u8; r.div_ceil(2)];
        for k in 0..r {
            heads[k / 2] |= self.head(k) << (4 * (k % 2));
        }
        let starts = (0..r).map(|k| self.start(k)).collect();
        RlBwtWire {
            len: self.len,
            heads,
            starts,
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for RlBwt {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = RlBwtWire::deserialize(d)?;
        Self::from_wire(wire).map_err(serde::de::Error::custom)
    }
}

impl RlBwt {
    /// Build from `(symbol, length)` runs in BWT order. Adjacent runs of the same symbol are
    /// merged and zero-length runs skipped, so any split of the BWT into runs is accepted —
    /// including one run per row.
    ///
    /// # Panics
    /// If a symbol is `>= ALPHABET_SIZE` or the total length exceeds `u32::MAX`.
    pub fn from_runs<I>(runs: I) -> Self
    where
        I: IntoIterator<Item = (u8, u32)>,
    {
        let mut heads: Vec<u8> = Vec::new();
        let mut starts: Vec<u32> = Vec::new();
        let mut len: u32 = 0;
        for (sym, run_len) in runs {
            assert!(
                (sym as usize) < ALPHABET_SIZE,
                "RlBwt: symbol {sym} out of range"
            );
            if run_len == 0 {
                continue;
            }
            if heads.last() != Some(&sym) {
                heads.push(sym);
                starts.push(len);
            }
            len = len
                .checked_add(run_len)
                .expect("RlBwt: total length exceeds u32::MAX");
        }
        Self::from_maximal_runs(len, &heads, &starts)
    }

    /// Lay out maximal runs (`heads[k]` covering `starts[k]..starts[k + 1]`, the last ending
    /// at `len`), already validated: starts strictly increasing from 0 and below `len`,
    /// adjacent heads distinct, symbols in range.
    fn from_maximal_runs(len: u32, heads: &[u8], starts: &[u32]) -> Self {
        let r = heads.len();
        let mut totals = [0u32; ALPHABET_SIZE];
        for (k, &h) in heads.iter().enumerate() {
            let end = starts.get(k + 1).copied().unwrap_or(len);
            totals[h as usize] += end - starts[k];
        }
        let present: SymbolSet = (0..ALPHABET_SIZE as u8)
            .filter(|&c| totals[c as usize] > 0)
            .collect();
        let mut lane_of = [NO_LANE; ALPHABET_SIZE];
        for (lane, c) in present.iter().enumerate() {
            lane_of[c as usize] = lane as u8;
        }
        let lines_per_block = if present.len() as usize <= INLINE_LANES {
            1
        } else {
            2
        };

        let num_blocks = r / RUNS_PER_BLOCK + 1;
        let mut blocks = vec![Line([0; 16]); num_blocks * lines_per_block];
        let mut running = [0u32; ALPHABET_SIZE];
        for b in 0..num_blocks {
            let base = b * lines_per_block;
            for (c, lane) in lane_of.iter().enumerate() {
                if *lane != NO_LANE {
                    let w = SAMPLE_WORD + *lane as usize;
                    blocks[base + w / 16].0[w % 16] = running[c];
                }
            }
            let mut nibbles = 0u32;
            for t in 0..RUNS_PER_BLOCK {
                let k = b * RUNS_PER_BLOCK + t;
                let (start, head) = if k < r {
                    let end = starts.get(k + 1).copied().unwrap_or(len);
                    running[heads[k] as usize] += end - starts[k];
                    (starts[k], heads[k])
                } else {
                    (len, 0)
                };
                blocks[base].0[t] = start;
                nibbles |= (head as u32) << (4 * t);
            }
            let next = (b + 1) * RUNS_PER_BLOCK;
            blocks[base].0[RUNS_PER_BLOCK] = starts.get(next).copied().unwrap_or(len);
            blocks[base].0[HEAD_WORD] = nibbles;
        }

        let mut rl = Self {
            len,
            num_runs: r as u32,
            blocks,
            lines_per_block,
            lane_of,
            totals,
            present,
            bucket_shift: 0,
            buckets: Vec::new(),
        };
        rl.build_directory();
        rl
    }

    /// Check the on-disk runs and lay them out. Returns a description of the first
    /// violation.
    fn from_wire(w: RlBwtWire) -> Result<Self, String> {
        let r = w.starts.len();
        if w.heads.len() != r.div_ceil(2) {
            return Err(format!("{} head bytes for {r} runs", w.heads.len()));
        }
        if r % 2 == 1 && w.heads[r / 2] >> 4 != 0 {
            return Err("unused head nibble is not zero".into());
        }
        if (r == 0) != (w.len == 0) {
            return Err(format!("{r} runs for {} rows", w.len));
        }
        if r > 0 && w.starts[0] != 0 {
            return Err("first run does not start at row 0".into());
        }
        if w.starts.windows(2).any(|p| p[0] >= p[1]) {
            return Err("run starts are not strictly increasing".into());
        }
        if w.starts.last().is_some_and(|&s| s >= w.len) {
            return Err("last run starts at or past the BWT length".into());
        }
        let heads: Vec<u8> = (0..r)
            .map(|k| (w.heads[k / 2] >> (4 * (k % 2))) & 15)
            .collect();
        if heads.windows(2).any(|p| p[0] == p[1]) {
            return Err("adjacent runs share a symbol (runs are not maximal)".into());
        }
        Ok(Self::from_maximal_runs(w.len, &heads, &w.starts))
    }

    /// Run-length encode a packed [`Bwt`].
    pub fn from_bwt(bwt: &Bwt) -> Self {
        Self::from_runs(bwt.runs())
    }

    /// Run-length encode the BWT held by an [`OccTable`] (e.g. an existing
    /// [`FmIndex`](crate::FmIndex)'s), without rebuilding the suffix array.
    pub fn from_occ(occ: &OccTable) -> Self {
        Self::from_runs((0..occ.text_len).map(|i| (occ.symbol_at(i), 1)))
    }

    /// Build the run-length BWT of a set of references on the CPU: concatenate them with
    /// sentinels (as [`FmIndex`](crate::FmIndex) does), build the suffix array and BWT, and
    /// run-length encode it.
    pub fn build_cpu(sequences: &[DnaSequence]) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }
        let prepared = crate::fm_index::prepare::prepare_cpu(sequences)?;
        let bwt = build_bwt(&prepared.text, &prepared.sa);
        Ok(Self::from_bwt(&bwt))
    }

    /// Number of BWT rows `n`.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// True for the empty BWT.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of maximal runs `r`.
    pub fn num_runs(&self) -> u32 {
        self.num_runs
    }

    /// `n / r`: average run length, the compression lever of every run-length structure.
    pub fn run_ratio(&self) -> f64 {
        if self.num_runs == 0 {
            0.0
        } else {
            self.len as f64 / self.num_runs as f64
        }
    }

    /// The maximal runs as `(symbol, length)`, in BWT order.
    pub fn runs(&self) -> impl Iterator<Item = (u8, u32)> + '_ {
        (0..self.num_runs as usize).map(|k| (self.head(k), self.start(k + 1) - self.start(k)))
    }

    /// Symbol at row `i`.
    ///
    /// # Panics
    /// If `i >= len()`.
    pub fn get(&self, i: u32) -> u8 {
        assert!(i < self.len, "RlBwt::get row {i} >= len {}", self.len);
        BwtRank::symbol_at(self, i)
    }

    /// Decode into a packed [`Bwt`].
    pub fn to_bwt(&self) -> Bwt {
        let mut out = Vec::with_capacity(self.len as usize);
        for (sym, run_len) in self.runs() {
            out.extend(std::iter::repeat_n(sym, run_len as usize));
        }
        Bwt::from_unpacked(out)
    }

    /// The set of symbols occurring in the BWT.
    pub fn present_symbols(&self) -> SymbolSet {
        self.present
    }

    /// Occurrences of `c` in the whole BWT.
    pub fn symbol_count(&self, c: u8) -> u32 {
        self.totals.get(c as usize).copied().unwrap_or(0)
    }

    /// Heap bytes held by this structure (run blocks and bucket directory).
    pub fn heap_size_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Line>() + self.buckets.len() * 4
    }

    /// Recompute the bucket directory from the run starts.
    fn build_directory(&mut self) {
        let r = self.num_runs;
        self.bucket_shift = if r == 0 {
            0
        } else {
            // About `RUNS_PER_BUCKET` runs per bucket: 2^shift >= RUNS_PER_BUCKET * n / r.
            (self.len as u64 * RUNS_PER_BUCKET)
                .div_ceil(r as u64)
                .next_power_of_two()
                .trailing_zeros()
                .min(31)
        };
        self.buckets.clear();
        if self.len == 0 {
            return;
        }
        let last_row = self.len - 1;
        let num_buckets = (last_row >> self.bucket_shift) as usize + 2;
        self.buckets.reserve_exact(num_buckets);
        let mut k = 0usize;
        for b in 0..num_buckets {
            let row = ((b as u64) << self.bucket_shift).min(last_row as u64) as u32;
            while self.start(k + 1) <= row {
                k += 1;
            }
            self.buckets.push(k as u32);
        }
    }

    /// Start row of run `k` (`len` for `k >= r`, up to the last block's end).
    #[inline(always)]
    fn start(&self, k: usize) -> u32 {
        self.blocks[(k / RUNS_PER_BLOCK) * self.lines_per_block].0[k % RUNS_PER_BLOCK]
    }

    /// End row of run `k` (`start(k + 1)`), read from run `k`'s own block.
    #[inline(always)]
    fn end(&self, k: usize) -> u32 {
        self.blocks[(k / RUNS_PER_BLOCK) * self.lines_per_block].0[k % RUNS_PER_BLOCK + 1]
    }

    /// Head symbol of run `k`.
    #[inline(always)]
    fn head(&self, k: usize) -> u8 {
        let nibbles = self.blocks[(k / RUNS_PER_BLOCK) * self.lines_per_block].0[HEAD_WORD];
        ((nibbles >> (4 * (k % RUNS_PER_BLOCK))) & 15) as u8
    }

    /// Rank of the symbol of `lane` at the first row of block `b`.
    #[inline(always)]
    fn sample(&self, b: usize, lane: u8) -> u32 {
        let w = SAMPLE_WORD + lane as usize;
        self.blocks[b * self.lines_per_block + w / 16].0[w % 16]
    }

    /// Prefetch the directory entry for row `i`: stage one of a batched LF walk.
    #[inline(always)]
    pub(crate) fn prefetch_bucket(&self, i: u32) {
        if let Some(entry) = self.buckets.get((i >> self.bucket_shift) as usize) {
            crate::prefetch::prefetch_read(entry);
        }
    }

    /// Prefetch the run block the directory points row `i` to: stage two, once the entry
    /// [`prefetch_bucket`](Self::prefetch_bucket) requested has arrived.
    #[inline(always)]
    pub(crate) fn prefetch_block(&self, i: u32) {
        if let Some(&k) = self.buckets.get((i >> self.bucket_shift) as usize) {
            let line = (k as usize / RUNS_PER_BLOCK) * self.lines_per_block;
            if let Some(block) = self.blocks.get(line) {
                crate::prefetch::prefetch_read(block);
            }
        }
    }

    /// Rows of `lo..hi` whose BWT symbol is not in `set`, in row order: one predecessor
    /// search, then whole runs.
    pub(crate) fn rows_not_in(&self, set: SymbolSet, lo: u32, hi: u32, out: &mut Vec<u32>) {
        if lo >= hi {
            return;
        }
        let mut k = self.run_of(lo);
        while self.start(k) < hi {
            if !set.contains(self.head(k)) {
                out.extend(self.start(k).max(lo)..self.end(k).min(hi));
            }
            k += 1;
        }
    }

    /// Index of the run containing row `i`, or `r` for `i == len`.
    #[inline(always)]
    fn run_of(&self, i: u32) -> usize {
        if i >= self.len {
            return self.num_runs as usize;
        }
        let b = (i >> self.bucket_shift) as usize;
        let mut base = self.buckets[b] as usize;
        let top = self.buckets[b + 1] as usize;
        // The run holding `i` is the last of `[base, top]` starting at or before `i`. Keep
        // the search on `base`'s block when `i` ends there (its end is stored in-line);
        // otherwise the answer starts at or after the next block.
        let block_first = base / RUNS_PER_BLOCK * RUNS_PER_BLOCK;
        let mut n = if i < self.end(block_first + RUNS_PER_BLOCK - 1) {
            top.min(block_first + RUNS_PER_BLOCK - 1) - base
        } else {
            base = block_first + RUNS_PER_BLOCK;
            top - base
        };
        while n > 0 {
            let half = n / 2;
            let mid = base + half + 1;
            if self.start(mid) <= i {
                base = mid;
                n -= half + 1;
            } else {
                n = half;
            }
        }
        base
    }

    /// `rank(c, i)` for a present symbol `c` (lane `lane`) given `k = run_of(i)`: the block
    /// sample plus the `c`-runs of the block before `k`, plus the part of run `k` before `i`.
    #[inline(always)]
    fn rank_at(&self, c: u8, lane: u8, i: u32, k: usize) -> u32 {
        let b = k / RUNS_PER_BLOCK;
        let t = k % RUNS_PER_BLOCK;
        let line = &self.blocks[b * self.lines_per_block].0;
        let nibbles = line[HEAD_WORD];
        let mut rank = self.sample(b, lane);
        for u in 0..t {
            if (nibbles >> (4 * u)) & 15 == c as u32 {
                rank += line[u + 1] - line[u];
            }
        }
        if (nibbles >> (4 * t)) & 15 == c as u32 {
            rank += i - line[t];
        }
        rank
    }

    /// Occurrences of members of `set` (all present) in `bwt[0..i)` given `k = run_of(i)`.
    #[inline(always)]
    fn rank_set_at(&self, set: SymbolSet, i: u32, k: usize) -> u32 {
        let b = k / RUNS_PER_BLOCK;
        let t = k % RUNS_PER_BLOCK;
        let line = &self.blocks[b * self.lines_per_block].0;
        let nibbles = line[HEAD_WORD];
        let mut rank: u32 = set
            .iter()
            .map(|c| self.sample(b, self.lane_of[c as usize]))
            .sum();
        for u in 0..t {
            if set.contains(((nibbles >> (4 * u)) & 15) as u8) {
                rank += line[u + 1] - line[u];
            }
        }
        if set.contains(((nibbles >> (4 * t)) & 15) as u8) {
            rank += i - line[t];
        }
        rank
    }

    /// `rank_all(i)` given `k = run_of(i)`.
    fn rank_all_at(&self, i: u32, k: usize) -> [u32; ALPHABET_SIZE] {
        let b = k / RUNS_PER_BLOCK;
        let t = k % RUNS_PER_BLOCK;
        let line = &self.blocks[b * self.lines_per_block].0;
        let nibbles = line[HEAD_WORD];
        let mut out = [0u32; ALPHABET_SIZE];
        for c in self.present.iter() {
            out[c as usize] = self.sample(b, self.lane_of[c as usize]);
        }
        for u in 0..t {
            out[((nibbles >> (4 * u)) & 15) as usize] += line[u + 1] - line[u];
        }
        if k < self.num_runs as usize {
            out[((nibbles >> (4 * t)) & 15) as usize] += i - line[t];
        }
        out
    }

    /// [`BwtRank::extend_ranks`] for an interval spanning more than [`MAX_SCAN_RUNS`] runs,
    /// given `k_lo = run_of(lo)`: two ranks and two class ranks.
    #[cold]
    fn extend_ranks_wide(
        &self,
        c: u8,
        lane: u8,
        lo: u32,
        hi: u32,
        k_lo: usize,
    ) -> Option<(u32, u32, u32)> {
        let k_hi = self.run_of(hi);
        let r_lo = self.rank_at(c, lane, lo, k_lo);
        let r_hi = self.rank_at(c, lane, hi, k_hi);
        if r_lo >= r_hi {
            return None;
        }
        let below = SymbolSet::below(c) & self.present;
        let b_lo = self.rank_set_at(below, lo, k_lo);
        let b_hi = self.rank_set_at(below, hi, k_hi);
        Some((r_lo, r_hi, b_hi - b_lo))
    }
}

impl BwtRank for RlBwt {
    #[inline]
    fn bwt_len(&self) -> u32 {
        self.len
    }

    #[inline]
    fn rank(&self, c: u8, i: u32) -> u32 {
        // Absent symbols (most IUPAC codes in a fan-out) cost nothing, as for the occ table.
        match self.lane_of.get(c as usize) {
            Some(&lane) if lane != NO_LANE => self.rank_at(c, lane, i, self.run_of(i)),
            _ => 0,
        }
    }

    #[inline]
    fn rank_pair(&self, c: u8, lo: u32, hi: u32) -> (u32, u32) {
        match self.lane_of.get(c as usize) {
            Some(&lane) if lane != NO_LANE => (
                self.rank_at(c, lane, lo, self.run_of(lo)),
                self.rank_at(c, lane, hi, self.run_of(hi)),
            ),
            _ => (0, 0),
        }
    }

    fn rank_with_below_pair(&self, c: u8, lo: u32, hi: u32) -> Option<((u32, u32), (u32, u32))> {
        let lane = *self.lane_of.get(c as usize)?;
        if lane == NO_LANE {
            return None;
        }
        let (k_lo, k_hi) = (self.run_of(lo), self.run_of(hi));
        let r_lo = self.rank_at(c, lane, lo, k_lo);
        let r_hi = self.rank_at(c, lane, hi, k_hi);
        if r_lo >= r_hi {
            return None;
        }
        let below = SymbolSet::below(c) & self.present;
        Some((
            (r_lo, self.rank_set_at(below, lo, k_lo)),
            (r_hi, self.rank_set_at(below, hi, k_hi)),
        ))
    }

    fn extend_ranks(&self, c: u8, lo: u32, hi: u32) -> Option<(u32, u32, u32)> {
        let lane = *self.lane_of.get(c as usize)?;
        if lane == NO_LANE || lo >= hi {
            return None;
        }
        let k_lo = self.run_of(lo);
        // Narrow intervals (the bulk of a deep search) span a few runs, all in the block of
        // `k_lo` or the next: walk them once, summing the overlap of `c`-runs and of runs
        // below `c`. No second predecessor search and no class ranks.
        let mut k = k_lo;
        let mut count = 0u32;
        let mut below = 0u32;
        while self.start(k) < hi {
            if k - k_lo == MAX_SCAN_RUNS {
                return self.extend_ranks_wide(c, lane, lo, hi, k_lo);
            }
            let head = self.head(k);
            if head <= c {
                let overlap = self.end(k).min(hi) - self.start(k).max(lo);
                if head == c {
                    count += overlap;
                } else {
                    below += overlap;
                }
            }
            k += 1;
        }
        if count == 0 {
            return None;
        }
        let r_lo = self.rank_at(c, lane, lo, k_lo);
        Some((r_lo, r_lo + count, below))
    }

    fn rank_set_pair(&self, set: SymbolSet, lo: u32, hi: u32) -> (u32, u32) {
        let set = set & self.present;
        if set.is_empty() {
            return (0, 0);
        }
        (
            self.rank_set_at(set, lo, self.run_of(lo)),
            self.rank_set_at(set, hi, self.run_of(hi)),
        )
    }

    fn rank_all_pair(&self, lo: u32, hi: u32) -> ([u32; ALPHABET_SIZE], [u32; ALPHABET_SIZE]) {
        let k_lo = self.run_of(lo);
        let at_lo = self.rank_all_at(lo, k_lo);
        // A narrow range: `hi`'s ranks are `lo`'s plus the per-symbol overlap of the few runs
        // in between, read off the same blocks (the fused children step of a sweep).
        let mut at_hi = at_lo;
        let mut k = k_lo;
        while self.start(k) < hi {
            if k - k_lo == MAX_SCAN_RUNS {
                return (at_lo, self.rank_all_at(hi, self.run_of(hi)));
            }
            at_hi[self.head(k) as usize] += self.end(k).min(hi) - self.start(k).max(lo);
            k += 1;
        }
        (at_lo, at_hi)
    }

    #[inline]
    fn symbol_at(&self, i: u32) -> u8 {
        assert!(i < self.len, "symbol_at row {i} >= len {}", self.len);
        self.head(self.run_of(i))
    }

    #[inline]
    fn lf_step(&self, i: u32) -> (u8, u32) {
        assert!(i < self.len, "lf_step row {i} >= len {}", self.len);
        let k = self.run_of(i);
        let head = self.head(k);
        (head, self.rank_at(head, self.lane_of[head as usize], i, k))
    }
}

#[cfg(test)]
mod tests;
