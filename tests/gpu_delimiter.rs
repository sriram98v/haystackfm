//! GPU paths on an index whose sequences hold `#` record delimiters: GPU construction,
//! `locate_gpu` and `find_smems_gpu` must report the same logical `(SeqId, offset)` hits as
//! the CPU, at sample rates where a locate walk would otherwise cross a delimiter.

#[cfg(feature = "gpu")]
mod tests {
    use haystackfm::alphabet::encode_char;
    use haystackfm::{BidirFmIndex, DnaSequence, FmIndex, FmIndexConfig, SeqId};
    use pollster::FutureExt as _;
    use std::collections::BTreeSet;

    /// Two logical sequences of `#`-joined records; "GATTACA" recurs across records.
    const JOINED: &[&str] = &[
        "ACGTGATTA#CAGGATTACAT#TTGCA",
        "AAGGAT#TACA#GATTACAGATTACA#CG",
    ];

    const PATTERNS: &[&str] = &["GATTACA", "TACA", "A", "CA", "AT", "GGAT", "TTACAGGA"];

    fn encode(s: &str) -> Vec<u8> {
        s.chars().map(|c| encode_char(c).unwrap()).collect()
    }

    fn sequences() -> Vec<DnaSequence> {
        JOINED
            .iter()
            .map(|s| DnaSequence::from_str(s).unwrap())
            .collect()
    }

    fn config(sa_sample_rate: u32, use_gpu: bool) -> FmIndexConfig {
        FmIndexConfig {
            sa_sample_rate,
            use_gpu,
            ..Default::default()
        }
    }

    fn hit_set(hits: impl IntoIterator<Item = (SeqId, u32)>) -> BTreeSet<(SeqId, u32)> {
        hits.into_iter().collect()
    }

    #[test]
    fn gpu_build_and_locate_match_cpu() {
        let patterns: Vec<Vec<u8>> = PATTERNS.iter().map(|p| encode(p)).collect();
        for rate in [1, 2, 4, 8] {
            let cpu = FmIndex::build_cpu(&sequences(), &config(rate, false)).unwrap();
            let gpu_built = FmIndex::build(&sequences(), &config(rate, true))
                .block_on()
                .unwrap();
            let cpu_gpu_locate = cpu.locate_gpu(&patterns).block_on().unwrap();
            let gpu_gpu_locate = gpu_built.locate_gpu(&patterns).block_on().unwrap();
            for (k, pat) in patterns.iter().enumerate() {
                let want = hit_set(cpu.locate(pat));
                let ctx = format!("{} rate {rate}", PATTERNS[k]);
                assert_eq!(hit_set(gpu_built.locate(pat)), want, "gpu build, {ctx}");
                assert_eq!(
                    hit_set(cpu_gpu_locate[k].clone()),
                    want,
                    "locate_gpu, {ctx}"
                );
                assert_eq!(hit_set(gpu_gpu_locate[k].clone()), want, "gpu both, {ctx}");
            }
        }
    }

    type Smem = (u32, u32, BTreeSet<(SeqId, u32)>);

    fn gpu_smems(
        seqs: &[DnaSequence],
        queries: &[DnaSequence],
        remap: impl Fn((SeqId, u32)) -> (SeqId, u32),
    ) -> Vec<Vec<Smem>> {
        let idx = BidirFmIndex::build_cpu(seqs, &config(4, false)).unwrap();
        let bounds = idx.seq_boundaries().to_vec();
        let hits = idx
            .find_smems_gpu(queries, 3, &bounds, 1024)
            .block_on()
            .unwrap();
        hits.into_iter()
            .map(|per_query| {
                let mut v: Vec<Smem> = per_query
                    .into_iter()
                    .map(|h| {
                        let pos = hit_set(h.positions.into_iter().map(&remap));
                        (h.query_start, h.query_end, pos)
                    })
                    .collect();
                v.sort_by_key(|m| (m.0, m.1));
                v
            })
            .collect()
    }

    /// The joined text is byte-identical to the split one (`#` and `$` are both the
    /// sentinel), so GPU SMEMs must agree once split hits are remapped to logical offsets.
    /// (Compared GPU-to-GPU: GPU and CPU SMEM sets can differ on whole-record matches
    /// independently of `#`.)
    #[test]
    fn gpu_smems_joined_equal_split_remapped() {
        let mut table = Vec::new();
        let mut split = Vec::new();
        for (id, s) in JOINED.iter().enumerate() {
            let mut base = 0u32;
            for rec in s.split('#') {
                table.push((SeqId::from(id as u32), base));
                split.push(DnaSequence::from_str(rec).unwrap());
                base += rec.len() as u32 + 1;
            }
        }
        let queries: Vec<DnaSequence> = [
            "ACGTGATTACAGGATTACAT",
            "AAGGATTACAGATTACAGATTACACG",
            "GATTA#CAGGATT",
        ]
        .iter()
        .map(|q| DnaSequence::from_str(q).unwrap())
        .collect();
        let joined = gpu_smems(&sequences(), &queries, |h| h);
        let split = gpu_smems(&split, &queries, |(s, off)| {
            let (id, base) = table[s.index()];
            (id, base + off)
        });
        assert_eq!(joined, split);
        assert!(joined.iter().all(|q| !q.is_empty()));
    }
}
