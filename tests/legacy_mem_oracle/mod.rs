//! Reimplementation of the pre-fix `find_mems` algorithm, for GPU parity tests.
//!
//! `BidirFmIndex::find_mems` now enumerates MEMs in the MUMmer / BWA sense — maximality is
//! judged per occurrence. The `shaders/mem_find.wgsl` MODE_MEM path still runs the older
//! algorithm: whole-set maximality, and only the longest match per query start position.
//! That divergence is tracked as issue 1 in `KNOWN-ISSUES.md`.
//!
//! Until the shader is ported, GPU MEM parity tests assert against this oracle rather than
//! against `find_mems`. That keeps them testing the shader's actual documented behavior
//! instead of failing on a known, tracked gap — and it makes the port's before/after
//! explicit: when MODE_MEM is fixed, these assertions move back to `find_mems` and this file
//! is deleted.
//!
//! Assumes the `IupacDna` alphabet, which every GPU parity corpus builds with.

#![allow(dead_code)]

use haystackfm::alphabet::compatible_symbols;
use haystackfm::{BidirFmIndex, BidirInterval, SeqId};

/// A match found by the legacy algorithm. Mirrors `haystackfm::Mem`.
pub struct LegacyMem {
    pub query_start: usize,
    pub query_end: usize,
    pub match_count: u32,
    pub positions: Vec<(SeqId, u32)>,
}

fn ext_right(idx: &BidirFmIndex, ivs: &[BidirInterval], c: u8) -> Vec<BidirInterval> {
    let mut out = Vec::new();
    for &base in compatible_symbols(c) {
        for iv in ivs {
            if let Some(ext) = idx.extend_right(*iv, base) {
                out.push(ext);
            }
        }
    }
    out
}

fn ext_left(idx: &BidirFmIndex, ivs: &[BidirInterval], c: u8) -> Vec<BidirInterval> {
    let mut out = Vec::new();
    for &base in compatible_symbols(c) {
        for iv in ivs {
            if let Some(ext) = idx.extend_left(*iv, base) {
                out.push(ext);
            }
        }
    }
    out
}

/// The longest match anchored at each query start position that is left- and right-maximal
/// **as a whole occurrence set** — one result per start, at most.
pub fn legacy_mems(
    idx: &BidirFmIndex,
    query: &[u8],
    min_len: usize,
    locate: bool,
) -> Vec<LegacyMem> {
    if query.is_empty() || min_len == 0 {
        return vec![];
    }
    let n = query.len();
    let mut out = Vec::new();

    for i in 0..n {
        // Right-extend until the occurrence set goes empty; keep only that final set.
        let mut ivs = vec![idx.full_interval()];
        let mut j = i;
        let mut matched = false;
        while j < n {
            let next = ext_right(idx, &ivs, query[j]);
            if next.is_empty() {
                break;
            }
            ivs = next;
            j += 1;
            matched = true;
        }
        if !matched || j - i < min_len {
            continue;
        }
        // Whole-set left-maximality: rejected if ANY occurrence can extend left.
        if i > 0 && !ext_left(idx, &ivs, query[i - 1]).is_empty() {
            continue;
        }
        out.push(LegacyMem {
            query_start: i,
            query_end: j,
            match_count: ivs.iter().map(|iv| idx.count_interval(iv)).sum(),
            positions: if locate {
                ivs.iter().flat_map(|iv| idx.locate_interval(iv)).collect()
            } else {
                Vec::new()
            },
        });
    }

    out.sort_by_key(|m| (m.query_start, m.query_end));
    out.dedup_by_key(|m| (m.query_start, m.query_end));
    out
}
