use super::BidirSearch;
use crate::alphabet::SymbolSet;
use crate::fm_index::bidir::BidirInterval;
use crate::fm_index::bidir_index::BidirFmIndex;
use crate::fm_index::seq_id::SeqId;
use crate::fm_index::FmIndex;

impl BidirSearch for BidirFmIndex {
    type Half = FmIndex;

    const FANOUT_STEPS: bool = false;

    fn fwd_half(&self) -> &FmIndex {
        &self.fwd
    }

    fn rev_half(&self) -> &FmIndex {
        &self.rev
    }

    fn search_full_interval(&self) -> BidirInterval {
        self.full_interval()
    }

    fn search_locate_rows(&self, lo: u32, hi: u32) -> Vec<(SeqId, u32)> {
        self.fwd.locate_rows(lo, hi)
    }

    fn shallow(&self, kmer: &[u8]) -> Option<&[BidirInterval]> {
        self.shallow.get(kmer)
    }

    fn left_maximal_rows(&self, ivs: &[BidirInterval], compat: SymbolSet) -> Vec<u32> {
        let mut rows = Vec::new();
        for iv in ivs {
            self.fwd
                .occ
                .rows_not_in(compat, iv.fwd_lo, iv.fwd_hi, &mut rows);
        }
        rows
    }

    fn search_locate_row_list(&self, rows: &[u32]) -> Vec<(SeqId, u32)> {
        let mut text_positions = Vec::with_capacity(rows.len());
        self.fwd.resolve_sa_batch(rows, &mut text_positions);
        text_positions
            .into_iter()
            .map(|pos| {
                self.fwd
                    .map_position(pos)
                    .expect("resolved SA position must be within text bounds")
            })
            .collect()
    }
}
