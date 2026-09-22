use crate::alphabet::{self, Alphabet, DnaSequence, IupacDna, SymbolSet, ALPHABET_SIZE};
use crate::error::FmIndexError;
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::seq_id::SeqId;
use crate::fm_index::{FmIndex, FmIndexConfig};

/// A bidirectional FM-index: pairs a forward FM-index (built on text T) with a
/// reverse FM-index (built on the byte-reversal of T), enabling O(1) extension
/// of matched intervals in both the left and right directions.
///
/// # Construction
///
/// ```rust,ignore
/// let seqs = vec![DnaSequence::from_str("ACGTACGT").unwrap()];
/// let config = FmIndexConfig::default();
/// let bidir = BidirFmIndex::build_cpu(&seqs, &config)?;
/// ```
///
/// # Use
///
/// ```rust,ignore
/// let iv = bidir.full_interval();
/// let iv = bidir.extend_right(iv, alphabet::C)?;  // match "C"
/// let iv = bidir.extend_right(iv, alphabet::G)?;  // match "CG"
/// let iv = bidir.extend_left(iv, alphabet::A)?;   // match "ACG"
/// println!("occurrences: {}", iv.size());
/// let positions = bidir.locate_interval(&iv);
/// ```
#[derive(Debug, Clone)]
pub struct BidirFmIndex {
    /// FM-index of the concatenated text T.
    pub(crate) fwd: FmIndex,
    /// FM-index of the byte-reversal of T (T^R).
    pub(crate) rev: FmIndex,
}

impl BidirFmIndex {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Build a bidirectional FM-index from DNA sequences using the CPU with [`IupacDna`]
    /// alphabet (full IUPAC ambiguity-code matching — the default).
    ///
    /// To use a different alphabet, call [`build_cpu_with`].
    ///
    /// [`build_cpu_with`]: BidirFmIndex::build_cpu_with
    pub fn build_cpu(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        Self::build_cpu_with::<IupacDna>(sequences, config)
    }

    /// Build a bidirectional FM-index using CPU with a custom [`Alphabet`].
    pub fn build_cpu_with<A: Alphabet>(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }

        let (text, _) = alphabet::concatenate_sequences(sequences)?;

        // Forward index.
        let fwd = FmIndex::build_cpu_with::<A>(sequences, config)?;

        // Reverse index: built on the byte-reversal of the same concatenated text.
        let rev_seq = reverse_as_sequence(&text)?;
        let mut rev = FmIndex::build_cpu_with::<A>(
            &[rev_seq],
            &FmIndexConfig {
                sa_sample_rate: config.sa_sample_rate,
                use_gpu: false,
                lookup_depth: 0,
                build_threads: config.build_threads,
                occ_encoding: config.occ_encoding,
                // Only `contract_left` exists today and it needs the forward LCP; the
                // reverse half's LCP would serve a future `contract_right`.
                build_lcp: false,
            },
        )?;
        // The reverse index's text is just the reversal of the forward one and is never
        // served through `sequence()` (all accessors delegate to `fwd`). Free it rather
        // than carry — and serialize — a second copy of the whole text.
        rev.forget_text();

        Ok(Self { fwd, rev })
    }

    /// Exclusive end positions of each reference in the concatenated text.
    /// `seq_boundaries()[i]` is the position just past the last base of reference `i`.
    /// Pass this slice as `ref_boundaries` to `find_smems_gpu` / `find_mems_gpu`.
    pub fn seq_boundaries(&self) -> &[u32] {
        &self.fwd.seq_boundaries
    }

    /// Build a bidirectional FM-index using GPU acceleration (async).
    #[cfg(feature = "gpu")]
    pub async fn build(
        sequences: &[DnaSequence],
        config: &FmIndexConfig,
    ) -> Result<Self, FmIndexError> {
        if sequences.is_empty() {
            return Err(FmIndexError::EmptySequence);
        }

        let (text, _) = alphabet::concatenate_sequences(sequences)?;

        // Build forward and reverse indices concurrently via GPU.
        let rev_seq = reverse_as_sequence(&text)?;

        let rev_config = FmIndexConfig {
            sa_sample_rate: config.sa_sample_rate,
            use_gpu: true,
            ..Default::default()
        };

        // Build sequentially: both paths share the GPU device pool and
        // concurrent init would contend for it.
        let fwd = FmIndex::build(sequences, config).await?;
        let mut rev = FmIndex::build(&[rev_seq], &rev_config).await?;
        // See `build_cpu_with`: the reverse text is redundant and never served.
        rev.forget_text();

        Ok(Self { fwd, rev })
    }

    // ── Interval operations ───────────────────────────────────────────────────

    /// The "whole text" interval, corresponding to the empty pattern.
    ///
    /// All positions are valid matches; this is the starting point for all
    /// bidirectional searches.
    pub fn full_interval(&self) -> BidirInterval {
        BidirInterval::full(self.fwd.text_len)
    }

    /// Extend a bidirectional interval to the right by character `c` (P → Pc).
    ///
    /// Uses the reverse FM-index internally (right extension = left extension of P^R).
    ///
    /// Returns `None` when Pc has no occurrences in the text.
    pub fn extend_right(&self, iv: BidirInterval, c: u8) -> Option<BidirInterval> {
        iv.extend_right(c, &self.rev)
    }

    /// Extend a bidirectional interval to the left by character `c` (P → cP).
    ///
    /// Uses the forward FM-index internally (standard backward-search step).
    ///
    /// Returns `None` when cP has no occurrences in the text.
    pub fn extend_left(&self, iv: BidirInterval, c: u8) -> Option<BidirInterval> {
        iv.extend_left(c, &self.fwd)
    }

    /// Contract the pattern on the left (cP → P): the inverse of
    /// [`extend_left`](Self::extend_left). `iv` must be the interval of `cP` and `c` the
    /// symbol it was extended by. Requires the forward half's LCP array
    /// (`FmIndexConfig::build_lcp`, CPU construction); see
    /// [`BidirInterval::contract_left`] for cost and errors.
    pub fn contract_left(&self, iv: &BidirInterval, c: u8) -> Result<BidirInterval, FmIndexError> {
        iv.contract_left(c, &self.fwd)
    }

    /// True when cursor contraction is available (the forward half carries an LCP array).
    pub fn has_lcp(&self) -> bool {
        self.fwd.has_lcp()
    }

    // ── Wildcard-aware cursor operations ──────────────────────────────────────

    /// Number of occurrences of `iv`'s pattern followed in the reference by an ambiguity
    /// code (`N` or a degenerate IUPAC symbol, codes 5..=15).
    ///
    /// Two occ-block touches regardless of how many wildcard codes exist, and no text
    /// access, so a cursor walk can ask at every step and fan out over wildcard codes
    /// (e.g. via [`children_right`](Self::children_right)) only when this is non-zero.
    /// Sentinel neighbours — occurrences at the end of a reference — are never wild.
    pub fn count_wild_right(&self, iv: &BidirInterval) -> u32 {
        iv.count_wild_right(&self.rev)
    }

    /// Number of occurrences of `iv`'s pattern preceded in the reference by an ambiguity
    /// code (codes 5..=15). See [`count_wild_right`](Self::count_wild_right).
    pub fn count_wild_left(&self, iv: &BidirInterval) -> u32 {
        iv.count_wild_left(&self.fwd)
    }

    /// Number of occurrences of `iv`'s pattern followed in the reference by a symbol in
    /// `set`; equals `Σ_{x ∈ set} extend_right(iv, x).size()` at the cost of a single
    /// extension. For "which wildcards could this read base still match", pass
    /// `self.compatible_set(base).intersection(SymbolSet::WILDCARDS)`.
    pub fn count_right_in(&self, iv: &BidirInterval, set: SymbolSet) -> u32 {
        iv.count_right_in(set, &self.rev)
    }

    /// Number of occurrences of `iv`'s pattern preceded in the reference by a symbol in
    /// `set`. See [`count_right_in`](Self::count_right_in).
    pub fn count_left_in(&self, iv: &BidirInterval, set: SymbolSet) -> u32 {
        iv.count_left_in(set, &self.fwd)
    }

    /// Every right child of `iv` at once: `children_right(iv)[c] == extend_right(iv, c)`
    /// for all 16 codes, for roughly the cost of one extension. Slot 0 is the sentinel
    /// child (occurrences at the end of a reference); do not extend it further.
    pub fn children_right(&self, iv: &BidirInterval) -> [Option<BidirInterval>; ALPHABET_SIZE] {
        iv.children_right(&self.rev)
    }

    /// Every left child of `iv` at once: `children_left(iv)[c] == extend_left(iv, c)`
    /// for all 16 codes. Slot 0 is the sentinel child (occurrences at the start of a
    /// reference). See [`children_right`](Self::children_right).
    pub fn children_left(&self, iv: &BidirInterval) -> [Option<BidirInterval>; ALPHABET_SIZE] {
        iv.children_left(&self.fwd)
    }

    /// Extend `iv` right by every reference code that query code `q` matches under this
    /// index's alphabet, yielding the non-empty children. This is the fan-out
    /// `find_smems` / `find_mems` perform per query symbol.
    pub fn extend_right_compatible(
        &self,
        iv: BidirInterval,
        q: u8,
    ) -> impl Iterator<Item = BidirInterval> + '_ {
        iv.extend_right_compatible(q, &self.rev)
    }

    /// Extend `iv` left by every reference code that query code `q` matches under this
    /// index's alphabet. See [`extend_right_compatible`](Self::extend_right_compatible).
    pub fn extend_left_compatible(
        &self,
        iv: BidirInterval,
        q: u8,
    ) -> impl Iterator<Item = BidirInterval> + '_ {
        iv.extend_left_compatible(q, &self.fwd)
    }

    /// The reference codes that query code `q` matches under this index's alphabet
    /// (empty for `N` on an [`ExactDna`](crate::alphabet::ExactDna) index).
    pub fn compatible_set(&self, q: u8) -> SymbolSet {
        self.fwd.alphabet_fns.compatible_set(q)
    }

    // ── Query helpers ─────────────────────────────────────────────────────────

    /// Count occurrences of the pattern represented by `iv`.
    pub fn count_interval(&self, iv: &BidirInterval) -> u32 {
        iv.size()
    }

    /// Locate all occurrences for the pattern represented by `iv`.
    ///
    /// Returns `(sequence_id, position_within_sequence)` tuples; resolve a [`SeqId`] to its
    /// FASTA header with [`seq_header`](Self::seq_header).
    /// Uses the forward SA samples; time is O(occ × sample_rate).
    pub fn locate_interval(&self, iv: &BidirInterval) -> Vec<(SeqId, u32)> {
        (iv.fwd_lo..iv.fwd_hi)
            .map(|i| {
                let text_pos = self.fwd.resolve_sa(i);
                self.fwd
                    .map_position(text_pos)
                    .expect("resolved SA position must be within text bounds")
            })
            .collect()
    }

    /// Total length of the indexed text (including sentinels).
    pub fn text_len(&self) -> u32 {
        self.fwd.text_len
    }

    /// Number of sequences indexed.
    pub fn num_sequences(&self) -> u32 {
        self.fwd.num_sequences
    }

    /// FASTA headers of every indexed reference, in build order.
    ///
    /// A header's position in this slice is its [`SeqId`] — the id reported by
    /// [`locate_interval`](Self::locate_interval), [`find_smems`](Self::find_smems) and
    /// [`find_mems`](Self::find_mems). Ids are preserved across serialization, so a caller
    /// can build an `id -> label` table from this slice once at load time and index it by
    /// [`SeqId::index`] thereafter instead of hashing a header per occurrence.
    pub fn seq_headers(&self) -> &[String] {
        self.fwd.seq_headers()
    }

    /// Header for a sequence id, or `None` when the id is out of range. O(1).
    pub fn seq_header(&self, id: SeqId) -> Option<&str> {
        self.fwd.seq_header(id)
    }

    /// Id for a header, or `None` when no reference carries it. O(1).
    ///
    /// Headers are unique — [`FmIndexError::DuplicateHeader`] is raised at build time
    /// otherwise — so this is an exact inverse of [`seq_header`](Self::seq_header).
    pub fn seq_id(&self, header: &str) -> Option<SeqId> {
        self.fwd.seq_id(header)
    }

    /// Bases of one indexed sequence, or `None` when the id is out of range.
    ///
    /// O(1) — a borrowed slice of the retained text, with the trailing sentinel excluded.
    /// An aligner rescoring a read against a candidate diagonal can take its substring
    /// from here rather than keeping a second copy of every reference alongside the index.
    ///
    /// The bases are **alphabet codes**, not ASCII: `A = 1`, `C = 2`, `G = 3`, `T = 4`,
    /// `N = 5`, and so on (see [`crate::alphabet`]). Use [`crate::alphabet::decode_char`]
    /// to render them, or [`crate::alphabet::encode_byte`] to bring a query into the same
    /// space before comparing.
    pub fn sequence(&self, id: SeqId) -> Option<&[u8]> {
        self.fwd.sequence(id)
    }

    /// Bases of the sequence carrying `header`, or `None` when no reference carries it.
    ///
    /// Equivalent to `seq_id(header).and_then(|id| self.sequence(id))`; see
    /// [`sequence`](Self::sequence) for the encoding of the returned bases.
    pub fn sequence_by_header(&self, header: &str) -> Option<&[u8]> {
        self.fwd.sequence_by_header(header)
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    /// Serialize both indices to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FmIndexError> {
        let fwd_bytes = self.fwd.to_bytes()?;
        let rev_bytes = self.rev.to_bytes()?;
        // Format: [4-byte fwd_len (LE)][fwd_bytes][rev_bytes]
        let mut out = Vec::with_capacity(4 + fwd_bytes.len() + rev_bytes.len());
        out.extend_from_slice(&(fwd_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&fwd_bytes);
        out.extend_from_slice(&rev_bytes);
        Ok(out)
    }

    /// Deserialize from bytes produced by `to_bytes()`.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FmIndexError> {
        if data.len() < 4 {
            return Err(FmIndexError::DeserializeError(
                "truncated bidirectional index".into(),
            ));
        }
        let fwd_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + fwd_len {
            return Err(FmIndexError::DeserializeError(
                "truncated forward index".into(),
            ));
        }
        let fwd = FmIndex::from_bytes(&data[4..4 + fwd_len])?;
        let rev = FmIndex::from_bytes(&data[4 + fwd_len..])?;
        Ok(Self { fwd, rev })
    }

    // ── GPU MEM/SMEM finding ──────────────────────────────────────────────────

    /// Find all Super-Maximal Exact Matches (SMEMs) for a batch of queries on the GPU.
    ///
    /// Find all Super-Maximal Exact Matches (SMEMs) for a batch of queries on the GPU,
    /// resolving each match to `(ref_id, offset_within_ref)` positions.
    ///
    /// `ref_boundaries[i]` is the exclusive end position of reference `i` in the
    /// concatenated text (same order as passed to [`BidirFmIndex::build_cpu`]).
    /// Pass `&[]` to skip position resolution (positions will be empty).
    ///
    /// Hits per MEM are capped at `max_hits_per_mem`. Pass `1024` for the default.
    #[cfg(feature = "gpu")]
    pub async fn find_smems_gpu(
        &self,
        queries: &[crate::alphabet::DnaSequence],
        min_len: usize,
        ref_boundaries: &[u32],
        max_hits_per_mem: u32,
    ) -> Result<Vec<Vec<crate::gpu::MemHit>>, FmIndexError> {
        use crate::gpu::{context_cache, mem_find::MODE_SMEM};
        let ctx = context_cache::get_or_init()?;
        let encoded: Vec<&[u8]> = queries.iter().map(|q| q.as_slice()).collect();
        resolve_mem_hits_gpu(
            &ctx,
            self,
            &encoded,
            min_len,
            MODE_SMEM,
            ref_boundaries,
            max_hits_per_mem,
        )
        .await
    }

    /// Find Maximal Exact Matches (MEMs) for a batch of queries on the GPU,
    /// resolving each match to `(ref_id, offset_within_ref)` positions.
    ///
    /// # Not equivalent to [`find_mems`](Self::find_mems)
    ///
    /// The shader has not been ported to occurrence-level MEMs. It applies whole-set
    /// maximality and reports only the longest match per query start position, so its result
    /// is a **strict subset** of [`find_mems`](Self::find_mems). Use the CPU path when you
    /// need every MEM. Tracked as issue 1 in `KNOWN-ISSUES.md`.
    ///
    /// [`find_smems_gpu`](Self::find_smems_gpu) is unaffected by this.
    ///
    /// `ref_boundaries[i]` is the exclusive end position of reference `i` in the
    /// concatenated text.
    /// Pass `&[]` to skip position resolution (positions will be empty).
    ///
    /// Hits per MEM are capped at `max_hits_per_mem`. Pass `1024` for the default.
    #[cfg(feature = "gpu")]
    pub async fn find_mems_gpu(
        &self,
        queries: &[crate::alphabet::DnaSequence],
        min_len: usize,
        ref_boundaries: &[u32],
        max_hits_per_mem: u32,
    ) -> Result<Vec<Vec<crate::gpu::MemHit>>, FmIndexError> {
        use crate::gpu::{context_cache, mem_find::MODE_MEM};
        let ctx = context_cache::get_or_init()?;
        let encoded: Vec<&[u8]> = queries.iter().map(|q| q.as_slice()).collect();
        resolve_mem_hits_gpu(
            &ctx,
            self,
            &encoded,
            min_len,
            MODE_MEM,
            ref_boundaries,
            max_hits_per_mem,
        )
        .await
    }
}

// ── Resolve batching helpers ──────────────────────────────────────────────────

/// A contiguous SA sub-range derived from one MEM interval, carrying enough
/// context to scatter results back into `output[q][m].positions[dest_start..]`.
#[cfg(feature = "gpu")]
struct SubInterval {
    fwd_lo: u32,
    fwd_hi: u32, // exclusive; `fwd_hi - fwd_lo` <= budget
    q: usize,
    m: usize,
    dest_start: usize,
}

/// One GPU resolve dispatch: a batch of sub-intervals whose total hit count fits
/// within the per-batch budget.
#[cfg(feature = "gpu")]
struct ResolveBatch {
    intervals_flat: Vec<u32>,   // stride-2: [fwd_lo, fwd_hi] per sub-interval
    position_offsets: Vec<u32>, // exclusive prefix-sum (len = n_subs + 1)
    total_pos: u32,
    slot_map: Vec<(usize, usize, usize)>, // (q, m, dest_start) per sub-interval
}

/// Split `flat_intervals` into batches whose summed hit count <= `budget`.
///
/// Intervals whose raw hit count exceeds `budget` (after capping at
/// `max_hits_per_mem`) are split into disjoint SA sub-ranges so no data is lost.
#[cfg(feature = "gpu")]
fn plan_resolve_batches(
    flat_intervals: &[crate::gpu::RawMemInterval],
    index_map: &[(usize, usize)],
    max_hits_per_mem: u32,
    budget: u32,
) -> Vec<ResolveBatch> {
    // Expand each interval into (possibly many) sub-intervals.
    let mut subs: Vec<SubInterval> = Vec::new();
    for (k, iv) in flat_intervals.iter().enumerate() {
        let (q, m) = index_map[k];
        let raw = iv.fwd_hi.saturating_sub(iv.fwd_lo);
        let effective = raw.min(max_hits_per_mem);
        if effective == 0 {
            continue;
        }
        let mut dest_start = 0usize;
        let mut lo = iv.fwd_lo;
        let mut remaining = effective;
        while remaining > 0 {
            let chunk = remaining.min(budget);
            subs.push(SubInterval {
                fwd_lo: lo,
                fwd_hi: lo + chunk,
                q,
                m,
                dest_start,
            });
            lo += chunk;
            dest_start += chunk as usize;
            remaining -= chunk;
        }
    }

    // Greedily pack sub-intervals into batches.
    let mut batches: Vec<ResolveBatch> = Vec::new();
    let mut current_subs: Vec<SubInterval> = Vec::new();
    let mut current_total: u32 = 0;

    for sub in subs {
        let hits = sub.fwd_hi - sub.fwd_lo;
        if !current_subs.is_empty() && current_total + hits > budget {
            batches.push(make_resolve_batch(current_subs));
            current_subs = Vec::new();
            current_total = 0;
        }
        current_total += hits;
        current_subs.push(sub);
    }
    if !current_subs.is_empty() {
        batches.push(make_resolve_batch(current_subs));
    }
    batches
}

#[cfg(feature = "gpu")]
fn make_resolve_batch(subs: Vec<SubInterval>) -> ResolveBatch {
    let mut intervals_flat = Vec::with_capacity(subs.len() * 2);
    let mut position_offsets = Vec::with_capacity(subs.len() + 1);
    let mut slot_map = Vec::with_capacity(subs.len());
    position_offsets.push(0u32);
    for sub in &subs {
        intervals_flat.push(sub.fwd_lo);
        intervals_flat.push(sub.fwd_hi);
        let hits = sub.fwd_hi - sub.fwd_lo;
        let prev = *position_offsets.last().unwrap();
        position_offsets.push(prev + hits);
        slot_map.push((sub.q, sub.m, sub.dest_start));
    }
    let total_pos = *position_offsets.last().unwrap();
    ResolveBatch {
        intervals_flat,
        position_offsets,
        total_pos,
        slot_map,
    }
}

// ── Orchestrator ──────────────────────────────────────────────────────────────

/// Shared GPU pipeline: MEM find → SA resolve → ref boundary map → MemHit assembly.
///
/// Uploads index buffers once; batches both the find and resolve phases against
/// `ctx.output_budget_u32()` to avoid `max_storage_buffer_binding_size` overflow.
#[cfg(feature = "gpu")]
async fn resolve_mem_hits_gpu(
    ctx: &crate::gpu::GpuContext,
    bidir: &BidirFmIndex,
    encoded: &[&[u8]],
    min_len: usize,
    mode: u32,
    ref_boundaries: &[u32],
    max_hits_per_mem: u32,
) -> Result<Vec<Vec<crate::gpu::MemHit>>, FmIndexError> {
    use crate::gpu::ref_map::map_positions_to_refs;
    use crate::gpu::MemHit;
    use crate::gpu::{
        find_mem_intervals_for_batch, resolve_intervals_batch, FindIndexBuffers,
        ResolveIndexBuffers,
    };

    let budget = ctx.output_budget_u32();

    // Upload index buffers once.
    let find_idx = FindIndexBuffers::new(ctx, bidir);
    let resolve_idx = ResolveIndexBuffers::new(ctx, &bidir.fwd)?;

    // ── Find phase ────────────────────────────────────────────────────────────
    // Chunk queries so queries_flat fits within budget.
    let mut per_query_intervals: Vec<Vec<crate::gpu::RawMemInterval>> =
        vec![Vec::new(); encoded.len()];

    let mut chunk_start = 0usize;
    while chunk_start < encoded.len() {
        // Accumulate queries until adding the next would exceed budget.
        let mut flat_len: u32 = 0;
        let mut chunk_end = chunk_start;
        while chunk_end < encoded.len() {
            let q_len = encoded[chunk_end].len() as u32;
            if chunk_end > chunk_start && flat_len + q_len > budget {
                break;
            }
            flat_len += q_len;
            chunk_end += 1;
        }

        let chunk = &encoded[chunk_start..chunk_end];
        let chunk_ivs = find_mem_intervals_for_batch(ctx, &find_idx, chunk, min_len, mode).await?;

        for (i, ivs) in chunk_ivs.into_iter().enumerate() {
            per_query_intervals[chunk_start + i] = ivs;
        }
        chunk_start = chunk_end;
    }

    // ── Flatten intervals + build index_map ──────────────────────────────────
    let mut flat_intervals: Vec<crate::gpu::RawMemInterval> = Vec::new();
    let mut index_map: Vec<(usize, usize)> = Vec::new();
    for (q, mems) in per_query_intervals.iter().enumerate() {
        for (m, iv) in mems.iter().enumerate() {
            flat_intervals.push(*iv);
            index_map.push((q, m));
        }
    }

    // ── Build output skeleton with empty positions ────────────────────────────
    let mut output: Vec<Vec<MemHit>> = per_query_intervals
        .iter()
        .map(|mems| {
            mems.iter()
                .map(|iv| {
                    let raw = iv.fwd_hi.saturating_sub(iv.fwd_lo);
                    MemHit {
                        query_start: iv.query_start,
                        query_end: iv.query_end,
                        match_count: raw,
                        positions: Vec::new(),
                        truncated: raw > max_hits_per_mem,
                    }
                })
                .collect()
        })
        .collect();

    if flat_intervals.is_empty() || ref_boundaries.is_empty() {
        return Ok(output);
    }

    // Pre-allocate position slots now that we know we'll resolve.
    for (k, iv) in flat_intervals.iter().enumerate() {
        let (q, m) = index_map[k];
        let raw = iv.fwd_hi.saturating_sub(iv.fwd_lo);
        let effective = raw.min(max_hits_per_mem) as usize;
        output[q][m].positions = vec![(SeqId::new(0), 0u32); effective];
    }

    // ── Resolve phase (batched) ───────────────────────────────────────────────
    let batches = plan_resolve_batches(&flat_intervals, &index_map, max_hits_per_mem, budget);

    for batch in batches {
        if batch.total_pos == 0 {
            continue;
        }
        let positions_flat = resolve_intervals_batch(
            ctx,
            &resolve_idx,
            &batch.intervals_flat,
            &batch.position_offsets,
            batch.total_pos,
        )
        .await;

        let (ref_ids, ref_offs) =
            map_positions_to_refs(ctx, &positions_flat, ref_boundaries).await?;

        // Scatter into pre-allocated slots.
        for (i, &(q, m, dest_start)) in batch.slot_map.iter().enumerate() {
            let start = batch.position_offsets[i] as usize;
            let end = batch.position_offsets[i + 1] as usize;
            let hits = end - start;
            let dst = &mut output[q][m].positions[dest_start..dest_start + hits];
            for (j, slot) in dst.iter_mut().enumerate() {
                *slot = (SeqId::new(ref_ids[start + j]), ref_offs[start + j]);
            }
        }
    }

    Ok(output)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Reverse a concatenated encoded text (bytes 0–4) and wrap it as a single DnaSequence.
///
/// Sentinels (0) in the middle of the text become interior characters of the reversed
/// sequence; the FM-index treats them as the lexicographically smallest character, so
/// the reverse index remains valid.
fn reverse_as_sequence(text: &[u8]) -> Result<DnaSequence, FmIndexError> {
    // Strip the trailing sentinel before reversing so we don't double-sentinel.
    let stripped = if text.last() == Some(&crate::alphabet::SENTINEL) {
        &text[..text.len() - 1]
    } else {
        text
    };
    let rev: Vec<u8> = stripped.iter().rev().cloned().collect();
    Ok(DnaSequence::from_encoded(rev))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::{encode_char, DnaSequence};

    fn encode(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    fn bidir(s: &str) -> BidirFmIndex {
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            ..Default::default()
        };
        BidirFmIndex::build_cpu(&[DnaSequence::from_str(s).unwrap()], &config).unwrap()
    }

    fn bidir_multi(seqs: &[(&str, &str)]) -> BidirFmIndex {
        let config = FmIndexConfig {
            sa_sample_rate: 1,
            use_gpu: false,
            ..Default::default()
        };
        let sequences: Vec<DnaSequence> = seqs
            .iter()
            .map(|(s, h)| DnaSequence::from_str_with_header(s, h).unwrap())
            .collect();
        BidirFmIndex::build_cpu(&sequences, &config).unwrap()
    }

    #[test]
    fn full_interval_covers_all() {
        let idx = bidir("ACGTACGT");
        let iv = idx.full_interval();
        assert_eq!(iv.size(), idx.text_len());
    }

    #[test]
    fn sequence_round_trips_every_reference() {
        // Deliberately different lengths so a boundary off-by-one cannot pass by luck.
        let refs = [
            ("ACGTACGTAC", "chr1"),
            ("TTGCA", "chr2"),
            ("GGGGNNNNCCCCAAAA", "chr3"),
        ];
        let idx = bidir_multi(&refs);

        for (i, (bases, _)) in refs.iter().enumerate() {
            let got = idx.sequence(SeqId::new(i as u32)).expect("id is in range");
            assert_eq!(got, encode(bases).as_slice(), "sequence {i} mismatch");
        }
    }

    #[test]
    fn sequence_excludes_the_sentinel() {
        let idx = bidir_multi(&[("ACGT", "a"), ("TTTT", "b")]);
        for i in 0..idx.num_sequences() {
            let seq = idx.sequence(SeqId::new(i)).unwrap();
            assert!(
                !seq.contains(&crate::alphabet::SENTINEL),
                "sequence {i} leaked its separator"
            );
        }
    }

    #[test]
    fn sequence_out_of_range_is_none() {
        let idx = bidir_multi(&[("ACGT", "a"), ("TTTT", "b")]);
        assert!(idx.sequence(SeqId::new(2)).is_none());
        assert!(idx.sequence(SeqId::new(9999)).is_none());
    }

    #[test]
    fn sequence_by_header_agrees_with_sequence_by_id() {
        let idx = bidir_multi(&[("ACGTA", "chr1"), ("TTGCA", "chr2")]);
        for header in ["chr1", "chr2"] {
            let id = idx.seq_id(header).expect("header is indexed");
            assert_eq!(idx.sequence_by_header(header), idx.sequence(id));
        }
        assert!(idx.sequence_by_header("absent").is_none());
    }

    #[test]
    fn sequence_survives_serialization() {
        let refs = [("ACGTACGTAC", "chr1"), ("TTGCA", "chr2")];
        let original = bidir_multi(&refs);
        let restored = BidirFmIndex::from_bytes(&original.to_bytes().unwrap()).unwrap();

        for i in 0..original.num_sequences() {
            let id = SeqId::new(i);
            assert_eq!(
                original.sequence(id),
                restored.sequence(id),
                "sequence {i} changed across serialization"
            );
        }
    }

    #[test]
    fn reverse_half_carries_no_duplicate_text() {
        // The reverse index's text is a redundant reversal of the forward one; dropping it
        // is what keeps the serialized size at ~n rather than ~2n.
        let idx = bidir_multi(&[("ACGTACGTAC", "chr1")]);
        assert!(idx.rev.text.is_empty(), "reverse half retained its text");
        assert!(
            idx.sequence(SeqId::new(0)).is_some(),
            "forward half lost its text"
        );
    }

    #[test]
    fn extend_right_count_matches_unidirectional() {
        let idx = bidir("ACGTACGT");
        let pattern = encode("ACGT");

        let mut iv = idx.full_interval();
        for &c in &pattern {
            iv = idx
                .extend_right(iv, c)
                .unwrap_or_else(|| panic!("extend_right failed for char {}", c));
        }
        assert_eq!(iv.size(), idx.fwd.count(&pattern));
    }

    #[test]
    fn extend_left_count_matches_unidirectional() {
        let idx = bidir("ACGTACGT");
        let pattern = encode("ACGT");

        // Build "ACGT" via extend_left: prepend T, G, C, A (right-to-left)
        let mut iv = idx.full_interval();
        for &c in pattern.iter().rev() {
            iv = idx
                .extend_left(iv, c)
                .unwrap_or_else(|| panic!("extend_left failed for char {}", c));
        }
        assert_eq!(iv.size(), idx.fwd.count(&pattern));

        // Extending left by A from the "ACGT" interval → "AACGT", absent from "ACGTACGT"
        let a = encode_char('A').unwrap();
        assert!(
            idx.extend_left(iv, a).is_none(),
            "AACGT should not appear in ACGTACGT"
        );
    }

    #[test]
    fn extend_right_and_left_combined() {
        // Text "TTACGTAA": find "ACGT" then extend in both directions.
        let idx = bidir("TTACGTAA");
        let acgt = encode("ACGT");

        let mut iv = idx.full_interval();
        for &c in &acgt {
            iv = idx
                .extend_right(iv, c)
                .unwrap_or_else(|| panic!("extend_right failed"));
        }
        assert_eq!(iv.size(), 1, "ACGT should appear once");

        // Extend left by T → "TACGT"
        let t = encode_char('T').unwrap();
        let iv2 = idx.extend_left(iv, t).expect("TACGT should be in TTACGTAA");
        assert_eq!(iv2.size(), 1);

        // Extend right by A → "TACGTA"
        let a = encode_char('A').unwrap();
        let iv3 = idx
            .extend_right(iv2, a)
            .expect("TACGTA should be in TTACGTAA");
        assert_eq!(iv3.size(), 1);
    }

    #[test]
    fn locate_interval() {
        let idx = bidir("ACGTACGT");
        let pattern = encode("ACGT");

        let mut iv = idx.full_interval();
        for &c in &pattern {
            iv = idx.extend_right(iv, c).unwrap();
        }
        let mut positions = idx.locate_interval(&iv);
        positions.sort();
        assert_eq!(positions, vec![(SeqId::new(0), 0), (SeqId::new(0), 4)]);
    }

    #[test]
    fn serialization_roundtrip() {
        let idx = bidir("ACGTACGT");
        let bytes = idx.to_bytes().unwrap();
        let restored = BidirFmIndex::from_bytes(&bytes).unwrap();

        let pattern = encode("ACGT");
        let mut iv1 = idx.full_interval();
        let mut iv2 = restored.full_interval();
        for &c in &pattern {
            iv1 = idx.extend_right(iv1, c).unwrap();
            iv2 = restored.extend_right(iv2, c).unwrap();
        }
        assert_eq!(iv1.size(), iv2.size());
    }
}
