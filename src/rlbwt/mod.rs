//! Run-length encoded Burrows–Wheeler transform.
//!
//! A BWT of length `n` over a repetitive text consists of few maximal runs `r ≪ n` of equal
//! symbols. [`RlBwt`] stores those runs and answers the FM-index primitives ([`BwtRank`]) in
//! `O(r)` space, following Mäkinen & Navarro's run-length FM-index:
//!
//! - `H`, the run-head string (`H[k]` = symbol of run `k`, length `r`), with an [`OccTable`]
//!   over it for rank/access;
//! - the run start rows (`r + 1` values, the last one `n`), with a bucket directory for
//!   predecessor search;
//! - per symbol, prefix sums of the lengths of that symbol's runs.
//!
//! ```text
//! k      = run containing row i                       (predecessor over run starts)
//! j      = rank_H(c, k)                               (c-runs before run k)
//! rank_c = cum_c[j] + (H[k] == c ? i − start[k] : 0)
//! ```
//!
//! This is the representation the move structure and its bidirectional variant (b-move)
//! build on: both index the same runs, adding per-run LF/φ tables.

use serde::{Deserialize, Serialize};

use crate::alphabet::{DnaSequence, SymbolSet, ALPHABET_SIZE};
use crate::bwt::cpu::build_bwt;
use crate::bwt::Bwt;
use crate::error::FmIndexError;
use crate::occ::cpu::build_occ_table;
use crate::occ::{BwtRank, OccEncoding, OccTable};

/// Average runs per bucket of the predecessor directory: a bucket costs 4 bytes, and the
/// binary search inside one touches the same cache line or two of `run_starts`.
const RUNS_PER_BUCKET: u64 = 4;

/// Run-length encoded BWT with rank/access support. See the [module docs](self).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RlBwt {
    /// Number of BWT rows `n`.
    len: u32,
    /// Rank/access over the run-head string `H` (one symbol per run).
    heads: OccTable,
    /// Start row of every run, plus `len` as a final sentinel entry (`r + 1` values,
    /// strictly increasing, first `0`).
    #[serde(with = "crate::serde_raw::u32s")]
    run_starts: Vec<u32>,
    /// Per-symbol run-length prefix sums, concatenated: `cum[cum_off[c] + j]` is the total
    /// length of the first `j` runs of symbol `c`. Symbol `c` owns `runs_c + 1` entries.
    #[serde(with = "crate::serde_raw::u32s")]
    cum: Vec<u32>,
    /// Section offsets into `cum`; `cum_off[ALPHABET_SIZE] == cum.len()`.
    cum_off: [u32; ALPHABET_SIZE + 1],
    /// log2 of the rows covered by one `buckets` entry. Derived.
    #[serde(skip)]
    bucket_shift: u32,
    /// `buckets[b]` = run containing row `min(b << bucket_shift, len − 1)`. Derived from
    /// `run_starts`, so skipped on disk and rebuilt by [`rebuild_derived`](Self::rebuild_derived).
    #[serde(skip)]
    buckets: Vec<u32>,
    /// Symbols with at least one run. Derived from `cum_off`.
    #[serde(skip)]
    present: SymbolSet,
}

impl RlBwt {
    /// Build from `(symbol, length)` runs in BWT order. Adjacent runs of the same symbol are
    /// merged and zero-length runs skipped, so any split of the BWT into runs is accepted —
    /// including one run per row.
    ///
    /// `head_encoding` selects the [`OccEncoding`] of the run-head rank table.
    ///
    /// # Panics
    /// If a symbol is `>= ALPHABET_SIZE` or the total length exceeds `u32::MAX`.
    pub fn from_runs<I>(runs: I, head_encoding: OccEncoding) -> Self
    where
        I: IntoIterator<Item = (u8, u32)>,
    {
        let mut heads: Vec<u8> = Vec::new();
        let mut run_starts: Vec<u32> = Vec::new();
        let mut lengths: Vec<u32> = Vec::new();
        let mut len: u32 = 0;
        for (sym, run_len) in runs {
            assert!(
                (sym as usize) < ALPHABET_SIZE,
                "RlBwt: symbol {sym} out of range"
            );
            if run_len == 0 {
                continue;
            }
            match heads.last() {
                Some(&last) if last == sym => *lengths.last_mut().unwrap() += run_len,
                _ => {
                    heads.push(sym);
                    run_starts.push(len);
                    lengths.push(run_len);
                }
            }
            len = len
                .checked_add(run_len)
                .expect("RlBwt: total length exceeds u32::MAX");
        }
        run_starts.push(len);

        // Per-symbol prefix sums of run lengths, in run order.
        let mut per_symbol: [Vec<u32>; ALPHABET_SIZE] = std::array::from_fn(|_| vec![0]);
        for (&sym, &run_len) in heads.iter().zip(&lengths) {
            let section = &mut per_symbol[sym as usize];
            let total = *section.last().unwrap();
            section.push(total + run_len);
        }
        let mut cum_off = [0u32; ALPHABET_SIZE + 1];
        let mut cum = Vec::with_capacity(heads.len() + ALPHABET_SIZE);
        for (c, section) in per_symbol.iter().enumerate() {
            cum_off[c] = u32::try_from(cum.len()).expect("RlBwt: too many runs");
            cum.extend_from_slice(section);
        }
        cum_off[ALPHABET_SIZE] = u32::try_from(cum.len()).expect("RlBwt: too many runs");

        let heads = build_occ_table(&Bwt::from_unpacked(heads), head_encoding);
        let mut rl = Self {
            len,
            heads,
            run_starts,
            cum,
            cum_off,
            bucket_shift: 0,
            buckets: Vec::new(),
            present: SymbolSet::EMPTY,
        };
        rl.rebuild_derived();
        rl
    }

    /// Run-length encode a packed [`Bwt`].
    pub fn from_bwt(bwt: &Bwt, head_encoding: OccEncoding) -> Self {
        Self::from_runs(bwt.runs(), head_encoding)
    }

    /// Run-length encode the BWT held by an [`OccTable`] (e.g. an existing
    /// [`FmIndex`](crate::FmIndex)'s), without rebuilding the suffix array.
    pub fn from_occ(occ: &OccTable, head_encoding: OccEncoding) -> Self {
        Self::from_runs(
            (0..occ.text_len).map(|i| (occ.symbol_at(i), 1)),
            head_encoding,
        )
    }

    /// Build the run-length BWT of a set of references on the CPU: concatenate them with
    /// sentinels (as [`FmIndex`](crate::FmIndex) does), build the suffix array and BWT, and
    /// run-length encode it.
    pub fn build_cpu(
        sequences: &[DnaSequence],
        head_encoding: OccEncoding,
    ) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }
        let prepared = crate::fm_index::prepare::prepare_cpu(sequences)?;
        let bwt = build_bwt(&prepared.text, &prepared.sa);
        Ok(Self::from_bwt(&bwt, head_encoding))
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
        self.heads.text_len
    }

    /// `n / r`: average run length, the compression lever of every run-length structure.
    pub fn run_ratio(&self) -> f64 {
        if self.num_runs() == 0 {
            0.0
        } else {
            self.len as f64 / self.num_runs() as f64
        }
    }

    /// The maximal runs as `(symbol, length)`, in BWT order.
    pub fn runs(&self) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.run_starts
            .windows(2)
            .enumerate()
            .map(|(k, w)| (self.heads.symbol_at(k as u32), w[1] - w[0]))
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
        self.cum_at(c, self.symbol_runs(c))
    }

    /// Heap bytes held by this structure (run heads, run starts, prefix sums, buckets).
    pub fn heap_size_bytes(&self) -> usize {
        self.heads.heap_size_bytes()
            + self.run_starts.len() * 4
            + self.cum.len() * 4
            + self.buckets.len() * 4
    }

    /// Recompute the bucket directory from `run_starts`. Called on construction and after
    /// deserialization.
    pub(crate) fn rebuild_derived(&mut self) {
        self.present = (0..ALPHABET_SIZE as u8)
            .filter(|&c| self.symbol_runs(c) > 0)
            .collect();
        let r = self.num_runs();
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
            while self.run_starts[k + 1] <= row {
                k += 1;
            }
            self.buckets.push(k as u32);
        }
    }

    /// Check the invariants a deserialized value must satisfy before any query touches it,
    /// then rebuild the derived state. Returns a description of the first violation.
    pub(crate) fn validate_and_rebuild(&mut self) -> Result<(), String> {
        // The head table must be structurally sound before any read, then exactly the
        // table its own symbols rebuild to (so its counts can be trusted as indexes).
        self.heads.validate_structure()?;
        let r = self.heads.text_len as usize;
        let decoded: Vec<u8> = (0..r as u32).map(|k| self.heads.symbol_at(k)).collect();
        let rebuilt = build_occ_table(&Bwt::from_unpacked(decoded), self.heads.encoding());
        self.heads.rebuild_derived();
        if rebuilt != self.heads {
            return Err("run-head table is not the table of its own symbols".into());
        }
        if self.run_starts.len() != r + 1 {
            return Err(format!("{} run starts for {r} runs", self.run_starts.len()));
        }
        if self.run_starts[0] != 0 && r > 0 {
            return Err("first run does not start at row 0".into());
        }
        if self.run_starts[r] != self.len {
            return Err("run starts do not end at the BWT length".into());
        }
        if self.run_starts.windows(2).any(|w| w[0] >= w[1]) {
            return Err("run starts are not strictly increasing".into());
        }
        if self.cum_off[0] != 0 || self.cum_off[ALPHABET_SIZE] as usize != self.cum.len() {
            return Err("prefix-sum offsets do not cover the prefix sums".into());
        }
        let mut total = 0u64;
        for c in 0..ALPHABET_SIZE {
            let (lo, hi) = (self.cum_off[c] as usize, self.cum_off[c + 1] as usize);
            let runs_c = self.heads.rank(c as u8, r as u32) as usize;
            if hi < lo || hi - lo != runs_c + 1 {
                return Err(format!("symbol {c}: prefix sums disagree with run heads"));
            }
            let section = &self.cum[lo..hi];
            if section[0] != 0 || section.windows(2).any(|w| w[0] >= w[1]) {
                return Err(format!("symbol {c}: prefix sums not strictly increasing"));
            }
            total += *section.last().unwrap() as u64;
        }
        if total != self.len as u64 {
            return Err("run lengths do not sum to the BWT length".into());
        }
        self.rebuild_derived();
        Ok(())
    }

    /// Number of runs of symbol `c`.
    #[inline]
    fn symbol_runs(&self, c: u8) -> u32 {
        self.cum_off[c as usize + 1] - self.cum_off[c as usize] - 1
    }

    /// Total length of the first `j` runs of symbol `c`.
    #[inline]
    fn cum_at(&self, c: u8, j: u32) -> u32 {
        self.cum[(self.cum_off[c as usize] + j) as usize]
    }

    /// Index of the run containing row `i`, or `r` for `i == len`.
    #[inline]
    fn run_of(&self, i: u32) -> usize {
        if i >= self.len {
            return self.num_runs() as usize;
        }
        let b = (i >> self.bucket_shift) as usize;
        let lo = self.buckets[b] as usize;
        let hi = self.buckets[b + 1] as usize;
        // The run holding `i` lies in `[lo, hi]`: count the starts in `(lo, hi]` that are <= i.
        lo + self.run_starts[lo + 1..=hi].partition_point(|&s| s <= i)
    }

    /// `rank(c, i)` given `k = run_of(i)`.
    #[inline]
    fn rank_in_run(&self, c: u8, i: u32, k: usize) -> u32 {
        if k < self.num_runs() as usize {
            let (head, j) = self.heads.lf_step(k as u32);
            if head == c {
                return self.cum_at(c, j) + (i - self.run_starts[k]);
            }
        }
        self.cum_at(c, self.heads.rank(c, k as u32))
    }

    /// Occurrences of members of `set` in `bwt[0..i)`, given `k = run_of(i)`: one head-table
    /// `rank_all` plus one prefix-sum read per present member.
    #[inline]
    fn rank_set_in_run(&self, set: SymbolSet, i: u32, k: usize) -> u32 {
        let set = set & self.present;
        if set.is_empty() {
            return 0;
        }
        let runs_before = self.heads.rank_all(k as u32);
        let mut total: u32 = set
            .iter()
            .map(|c| self.cum_at(c, runs_before[c as usize]))
            .sum();
        if k < self.num_runs() as usize && set.contains(self.heads.symbol_at(k as u32)) {
            total += i - self.run_starts[k];
        }
        total
    }

    /// `rank_all(i)` given `k = run_of(i)`.
    #[inline]
    fn rank_all_in_run(&self, i: u32, k: usize) -> [u32; ALPHABET_SIZE] {
        let runs_before = self.heads.rank_all(k as u32);
        let mut out = [0u32; ALPHABET_SIZE];
        for c in 0..ALPHABET_SIZE {
            out[c] = self.cum_at(c as u8, runs_before[c]);
        }
        if k < self.num_runs() as usize {
            let head = self.heads.symbol_at(k as u32);
            out[head as usize] += i - self.run_starts[k];
        }
        out
    }
}

impl BwtRank for RlBwt {
    #[inline]
    fn bwt_len(&self) -> u32 {
        self.len
    }

    #[inline]
    fn rank(&self, c: u8, i: u32) -> u32 {
        self.rank_in_run(c, i, self.run_of(i))
    }

    #[inline]
    fn rank_pair(&self, c: u8, lo: u32, hi: u32) -> (u32, u32) {
        (self.rank(c, lo), self.rank(c, hi))
    }

    fn rank_with_below_pair(&self, c: u8, lo: u32, hi: u32) -> Option<((u32, u32), (u32, u32))> {
        let (k_lo, k_hi) = (self.run_of(lo), self.run_of(hi));
        let r_lo = self.rank_in_run(c, lo, k_lo);
        let r_hi = self.rank_in_run(c, hi, k_hi);
        if r_lo >= r_hi {
            return None;
        }
        let below = SymbolSet::below(c);
        Some((
            (r_lo, self.rank_set_in_run(below, lo, k_lo)),
            (r_hi, self.rank_set_in_run(below, hi, k_hi)),
        ))
    }

    fn rank_set_pair(&self, set: SymbolSet, lo: u32, hi: u32) -> (u32, u32) {
        (
            self.rank_set_in_run(set, lo, self.run_of(lo)),
            self.rank_set_in_run(set, hi, self.run_of(hi)),
        )
    }

    fn rank_all_pair(&self, lo: u32, hi: u32) -> ([u32; ALPHABET_SIZE], [u32; ALPHABET_SIZE]) {
        (
            self.rank_all_in_run(lo, self.run_of(lo)),
            self.rank_all_in_run(hi, self.run_of(hi)),
        )
    }

    #[inline]
    fn symbol_at(&self, i: u32) -> u8 {
        debug_assert!(i < self.len, "symbol_at row {i} >= len {}", self.len);
        self.heads.symbol_at(self.run_of(i) as u32)
    }

    #[inline]
    fn lf_step(&self, i: u32) -> (u8, u32) {
        debug_assert!(i < self.len, "lf_step row {i} >= len {}", self.len);
        let k = self.run_of(i);
        let (head, j) = self.heads.lf_step(k as u32);
        (head, self.cum_at(head, j) + (i - self.run_starts[k]))
    }
}

#[cfg(test)]
mod tests;
