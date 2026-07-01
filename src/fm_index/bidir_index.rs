use crate::alphabet::{self, Alphabet, DnaSequence, IupacDna};
use crate::error::FmIndexError;
use crate::fm_index::bidir::BidirInterval;
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
/// let positions = bidir.locate(iv);
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
        let rev = FmIndex::build_cpu_with::<A>(
            &[rev_seq],
            &FmIndexConfig {
                sa_sample_rate: config.sa_sample_rate,
                use_gpu: false,
                lookup_depth: 0,
                build_threads: config.build_threads,
            },
        )?;

        Ok(Self { fwd, rev })
    }

    /// Exclusive end positions of each reference in the concatenated text.
    /// `seq_boundaries()[i]` is the position just past the last base of reference `i`.
    /// Pass this slice as `ref_boundaries` to [`find_smems_gpu`] / [`find_mems_gpu`].
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
        let rev = FmIndex::build(&[rev_seq], &rev_config).await?;

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

    // ── Query helpers ─────────────────────────────────────────────────────────

    /// Count occurrences of the pattern represented by `iv`.
    pub fn count_interval(&self, iv: &BidirInterval) -> u32 {
        iv.size()
    }

    /// Locate all occurrences for the pattern represented by `iv`.
    ///
    /// Returns `(sequence_id, position_within_sequence)` tuples.
    /// Uses the forward SA samples; time is O(occ × sample_rate).
    pub fn locate_interval(&self, iv: &BidirInterval) -> Vec<(String, u32)> {
        (iv.fwd_lo..iv.fwd_hi)
            .map(|i| {
                let text_pos = self.fwd.resolve_sa(i);
                let (seq_idx, pos_in_seq) = self
                    .fwd
                    .map_position(text_pos)
                    .expect("resolved SA position must be within text bounds");
                (self.fwd.seq_headers[seq_idx as usize].clone(), pos_in_seq)
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

    /// Find all Maximal Exact Matches (MEMs) for a batch of queries on the GPU,
    /// resolving each match to `(ref_id, offset_within_ref)` positions.
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
    intervals_flat: Vec<u32>,    // stride-2: [fwd_lo, fwd_hi] per sub-interval
    position_offsets: Vec<u32>,  // exclusive prefix-sum (len = n_subs + 1)
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
    ResolveBatch { intervals_flat, position_offsets, total_pos, slot_map }
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
    use crate::gpu::{FindIndexBuffers, ResolveIndexBuffers, find_mem_intervals_for_batch, resolve_intervals_batch};
    use crate::gpu::ref_map::map_positions_to_refs;
    use crate::gpu::MemHit;

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
        output[q][m].positions = vec![(0u32, 0u32); effective];
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
                *slot = (ref_ids[start + j], ref_offs[start + j]);
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

    #[test]
    fn full_interval_covers_all() {
        let idx = bidir("ACGTACGT");
        let iv = idx.full_interval();
        assert_eq!(iv.size(), idx.text_len());
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
        assert_eq!(
            positions,
            vec![("seq_0".to_string(), 0), ("seq_0".to_string(), 4)]
        );
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
