//! r-index locate on move tables: SA samples at run boundaries give a toehold (one SA value
//! of an interval) during search, and `φ` / `φ⁻¹` move tables with permuted-LCP samples
//! enumerate the rest of the interval from it — wherever in the interval the toehold sits.

use serde::{Deserialize, Serialize};

use super::lf::Toehold;
use crate::move_table::MoveTable;

/// Balancing parameter of the `φ` / `φ⁻¹` tables: a step fast-forwards fewer than `2d` rows.
const PHI_BALANCE: u32 = 4;

/// SA values at the first and last row of every LF move-table row (BWT run).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RunSamples {
    #[serde(with = "crate::serde_raw::u32s")]
    pub start: Vec<u32>,
    #[serde(with = "crate::serde_raw::u32s")]
    pub end: Vec<u32>,
}

impl RunSamples {
    /// Samples for the LF table `lf` of a text whose suffix array is `sa`.
    pub fn build(lf: &MoveTable, sa: &[u32]) -> Self {
        let rows = lf.num_rows() as usize;
        Self {
            start: (0..rows).map(|k| sa[lf.start(k) as usize]).collect(),
            end: (0..rows).map(|k| sa[lf.end(k) as usize]).collect(),
        }
    }

    /// The SA value named by `toehold` (`None` for [`Toehold::Carry`]).
    #[inline]
    pub fn get(&self, toehold: Toehold) -> Option<u32> {
        match toehold {
            Toehold::RunStart(k) => Some(self.start[k as usize]),
            Toehold::RunEnd(k) => Some(self.end[k as usize]),
            Toehold::Carry => None,
        }
    }

    pub fn validate(&self, rows: u32, n: u32) -> Result<(), String> {
        if self.start.len() != rows as usize || self.end.len() != rows as usize {
            return Err("run samples do not match the LF rows".into());
        }
        if self.start.iter().chain(&self.end).any(|&v| v >= n) {
            return Err("run sample past the text end".into());
        }
        Ok(())
    }

    pub fn heap_size_bytes(&self) -> usize {
        (self.start.len() + self.end.len()) * 4
    }
}

/// Everything locate needs beyond the LF table. See the [module docs](self).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct MoveLocate {
    pub samples: RunSamples,
    /// `φ(x) = SA[ISA[x] − 1]` (cyclically `SA[n − 1]` for row 0).
    phi: MoveTable,
    /// `φ⁻¹(x) = SA[ISA[x] + 1]` (cyclically `SA[0]` for row `n − 1`).
    phi_inv: MoveTable,
    /// `PLCP[p_k]` for every `φ` row `k`; inside the row it decreases by one per position.
    #[serde(with = "crate::serde_raw::u32s")]
    plcp: Vec<u32>,
    /// `lcp(T[x..], T[φ⁻¹(x)..])` at every `φ⁻¹` row start; decreases likewise.
    #[serde(with = "crate::serde_raw::u32s")]
    plcp_inv: Vec<u32>,
}

impl MoveLocate {
    /// Build from the LF table, the text and its suffix array.
    pub fn build(lf: &MoveTable, text: &[u8], sa: &[u32]) -> Self {
        let n = sa.len() as u32;
        let rows = lf.num_rows() as usize;
        let samples = RunSamples::build(lf, sa);
        // φ input intervals start at the text positions of run starts, φ⁻¹ ones at run
        // ends; within an interval both maps (and the PLCP) move by one per position.
        let mut phi_runs: Vec<(u32, u32, u8)> = (0..rows)
            .map(|k| {
                let p = lf.start(k);
                let above = if p == 0 { n - 1 } else { p - 1 };
                (sa[p as usize], sa[above as usize], 0)
            })
            .collect();
        let mut phi_inv_runs: Vec<(u32, u32, u8)> = (0..rows)
            .map(|k| {
                let e = lf.end(k);
                let below = if e == n - 1 { 0 } else { e + 1 };
                (sa[e as usize], sa[below as usize], 0)
            })
            .collect();
        phi_runs.sort_unstable();
        phi_inv_runs.sort_unstable();
        let phi = MoveTable::build(&phi_runs, n, Some(PHI_BALANCE));
        let phi_inv = MoveTable::build(&phi_inv_runs, n, Some(PHI_BALANCE));

        let full = crate::lcp::plcp_u32(text, sa);
        let plcp = (0..phi.num_rows() as usize)
            .map(|k| full[phi.start(k) as usize])
            .collect();
        // lcp(T[x..], T[φ⁻¹(x)..]) = PLCP[φ⁻¹(x)].
        let plcp_inv = (0..phi_inv.num_rows() as usize)
            .map(|k| full[phi_inv.row(k).q as usize])
            .collect();
        Self {
            samples,
            phi,
            phi_inv,
            plcp,
            plcp_inv,
        }
    }

    /// Check a deserialized value against the LF table's row count and the text length.
    pub fn validate(&self, lf_rows: u32, n: u32) -> Result<(), String> {
        self.samples.validate(lf_rows, n)?;
        if self.phi.domain() != n || self.phi_inv.domain() != n {
            return Err("φ tables cover a different text length".into());
        }
        for (table, plcp) in [(&self.phi, &self.plcp), (&self.phi_inv, &self.plcp_inv)] {
            if plcp.len() != table.num_rows() as usize {
                return Err("PLCP samples do not match the φ rows".into());
            }
            // PLCP never goes negative inside a row.
            for (k, &v) in plcp.iter().enumerate() {
                if v < table.end(k) - table.start(k) {
                    return Err("PLCP sample too small for its row".into());
                }
            }
        }
        Ok(())
    }

    pub fn heap_size_bytes(&self) -> usize {
        self.samples.heap_size_bytes()
            + self.phi.heap_size_bytes()
            + self.phi_inv.heap_size_bytes()
            + (self.plcp.len() + self.plcp_inv.len()) * 4
    }

    /// SA value of row 0 (the start of every full interval).
    pub fn sa_first(&self) -> u32 {
        self.samples.start[0]
    }

    /// Append to `out` the text positions of all `size` occurrences of a pattern of length
    /// `m`, given `x`, the position of one of them. Walks `φ` while the previous suffix
    /// still shares `m` symbols and `φ⁻¹` while the next one does. An `x` outside the text
    /// (an unknown toehold) yields nothing.
    pub fn enumerate(&self, x: u32, m: u32, size: u32, out: &mut Vec<u32>) {
        let target = out.len() + size as usize;
        if m == 0 {
            // The empty pattern's interval is every row: walk down from row 0.
            let (mut y, mut k) = (self.sa_first(), self.phi_inv.row_of(self.sa_first()));
            out.push(y);
            while out.len() < target {
                (y, k) = self.phi_inv.step(y, k);
                out.push(y);
            }
            return;
        }
        if x >= self.phi.domain() {
            return;
        }
        out.push(x);
        let (mut y, mut k) = (x, self.phi.row_of(x));
        while out.len() < target && Self::lcp_at(&self.phi, &self.plcp, y, k) >= m {
            (y, k) = self.phi.step(y, k);
            out.push(y);
        }
        let (mut y, mut k) = (x, self.phi_inv.row_of(x));
        while out.len() < target && Self::lcp_at(&self.phi_inv, &self.plcp_inv, y, k) >= m {
            (y, k) = self.phi_inv.step(y, k);
            out.push(y);
        }
    }

    #[inline]
    fn lcp_at(table: &MoveTable, plcp: &[u32], y: u32, k: usize) -> u32 {
        plcp[k] - (y - table.start(k))
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::alphabet::DnaSequence;
    use crate::bwt::cpu::build_bwt;
    use crate::fm_index::prepare::prepare_cpu;
    use crate::move_index::lf::MoveLf;

    fn texts() -> Vec<Vec<&'static str>> {
        vec![
            vec!["A"],
            vec!["AAAAAAAA"],
            vec!["ACGTACGTACGT", "ACGTACGA", "GATTACA"],
            vec!["NNNNACGTRYNN", "A", "AA", "ACGTNNNN"],
        ]
    }

    fn check(seqs: &[DnaSequence]) {
        let p = prepare_cpu(seqs).unwrap();
        let sa = &p.sa.data;
        let n = sa.len();
        let bwt = build_bwt(&p.text, &p.sa);
        let lf = MoveLf::from_bwt(&bwt, &p.c_array);
        let loc = MoveLocate::build(&lf.table, &p.text, sa);
        assert!(loc.validate(lf.num_rows(), n as u32).is_ok());
        let plcp = crate::lcp::plcp_u32(&p.text, sa);
        let mut isa = vec![0usize; n];
        for (row, &x) in sa.iter().enumerate() {
            isa[x as usize] = row;
        }
        for x in 0..n as u32 {
            let row = isa[x as usize];
            let phi = sa[if row == 0 { n - 1 } else { row - 1 }];
            let phi_inv = sa[if row == n - 1 { 0 } else { row + 1 }];
            let k = loc.phi.row_of(x);
            assert_eq!(loc.phi.step(x, k).0, phi, "φ({x})");
            assert_eq!(
                MoveLocate::lcp_at(&loc.phi, &loc.plcp, x, k),
                plcp[x as usize],
                "PLCP[{x}]"
            );
            let k = loc.phi_inv.row_of(x);
            assert_eq!(loc.phi_inv.step(x, k).0, phi_inv, "φ⁻¹({x})");
            assert_eq!(
                MoveLocate::lcp_at(&loc.phi_inv, &loc.plcp_inv, x, k),
                plcp[phi_inv as usize],
                "PLCP⁻[{x}]"
            );
        }
    }

    #[test]
    fn phi_tables_and_plcp_samples_match_the_suffix_array() {
        for set in texts() {
            let seqs: Vec<DnaSequence> = set
                .iter()
                .map(|s| DnaSequence::from_str(s).unwrap())
                .collect();
            check(&seqs);
        }
        let mut rng = StdRng::seed_from_u64(9);
        for round in 0..20 {
            let unit: Vec<u8> = (0..rng.random_range(1..30))
                .map(|_| rng.random_range(1..6))
                .collect();
            let seqs: Vec<DnaSequence> = (0..rng.random_range(1..5))
                .map(|_| {
                    let mut s = unit.repeat(rng.random_range(1..6));
                    if round % 2 == 0 {
                        let at = rng.random_range(0..s.len());
                        s[at] = rng.random_range(1..16);
                    }
                    DnaSequence::from_encoded(s)
                })
                .collect();
            check(&seqs);
        }
    }
}
