use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::*;

/// A random permutation of `[0, n)` made of `runs` consecutive-to-consecutive pieces:
/// returns the runs `(p, q, head)` and the full permutation.
fn random_runs(rng: &mut StdRng, n: u32, runs: usize) -> (Vec<(u32, u32, u8)>, Vec<u32>) {
    let mut cuts: Vec<u32> = (1..n).collect();
    cuts.shuffle(rng);
    let mut starts: Vec<u32> = cuts.into_iter().take(runs.saturating_sub(1)).collect();
    starts.push(0);
    starts.sort_unstable();
    let lens: Vec<u32> = (0..starts.len())
        .map(|k| starts.get(k + 1).copied().unwrap_or(n) - starts[k])
        .collect();
    // Lay the outputs out in a shuffled order of the runs.
    let mut order: Vec<usize> = (0..starts.len()).collect();
    order.shuffle(rng);
    let mut q = vec![0u32; starts.len()];
    let mut next = 0;
    for &k in &order {
        q[k] = next;
        next += lens[k];
    }
    let mut perm = vec![0u32; n as usize];
    for k in 0..starts.len() {
        for off in 0..lens[k] {
            perm[(starts[k] + off) as usize] = q[k] + off;
        }
    }
    let rows = (0..starts.len())
        .map(|k| (starts[k], q[k], rng.random_range(0..16u8)))
        .collect();
    (rows, perm)
}

#[test]
fn step_follows_the_permutation_balanced_and_not() {
    let mut rng = StdRng::seed_from_u64(1);
    for &(n, runs) in &[
        (1u32, 1usize),
        (2, 2),
        (50, 1),
        (50, 50),
        (500, 37),
        (3000, 400),
    ] {
        let (rows, perm) = random_runs(&mut rng, n, runs);
        for balance in [None, Some(1), Some(2), Some(4)] {
            let t = MoveTable::build(&rows, n, balance);
            assert_eq!(t.domain(), n);
            if balance.is_none() {
                assert_eq!(t.num_rows() as usize, rows.len());
            }
            for i in 0..n {
                let k = t.row_of(i);
                assert!(t.start(k) <= i && i <= t.end(k));
                let (img, k2) = t.step(i, k);
                assert_eq!(img, perm[i as usize], "n={n} runs={runs} i={i}");
                assert_eq!(k2, t.row_of(img));
                assert_eq!(t.map(i, k), img);
                if let Some(d) = balance {
                    let from = t.row(k).idx as usize;
                    assert!(
                        k2 - from < 2 * d as usize,
                        "fast-forward {} >= 2d",
                        k2 - from
                    );
                }
            }
        }
    }
}

#[test]
fn balancing_keeps_payloads_and_bounds_rows() {
    // A permutation whose single output interval swallows many input starts: identity
    // pieces of length 1 followed by one long run mapped to the front.
    let n = 1000u32;
    let mut rows: Vec<(u32, u32, u8)> = (0..100).map(|i| (i, 900 + i, 7)).collect();
    rows.push((100, 0, 9));
    let t = MoveTable::build(&rows, n, Some(2));
    assert!(t.num_rows() > rows.len() as u32);
    assert!(t.num_rows() < 3 * rows.len() as u32);
    for k in 0..t.num_rows() as usize {
        let expected = if t.start(k) < 100 { 7 } else { 9 };
        assert_eq!(t.head(k), expected);
    }
}

#[test]
fn row_searches_from_a_hint() {
    let mut rng = StdRng::seed_from_u64(2);
    let (rows, _) = random_runs(&mut rng, 2000, 300);
    let t = MoveTable::build(&rows, 2000, None);
    for _ in 0..2000 {
        let i = rng.random_range(0..2000);
        let k = t.row_of(i);
        let from = rng.random_range(0..=k);
        assert_eq!(t.row_from(i, from), k);
        let back = rng.random_range(k..t.num_rows() as usize);
        assert_eq!(t.row_back_from(i, back), k);
    }
}

#[test]
fn serde_round_trips_and_rejects_non_permutations() {
    let mut rng = StdRng::seed_from_u64(3);
    let (rows, _) = random_runs(&mut rng, 300, 40);
    let t = MoveTable::build(&rows, 300, Some(2));
    let bytes = bincode::serialize(&t).unwrap();
    let back: MoveTable = bincode::deserialize(&bytes).unwrap();
    assert_eq!(back, t);

    let repr: MoveTableRepr = t.clone().into();
    let corrupt = |f: &dyn Fn(&mut MoveTableRepr)| {
        let mut r = MoveTableRepr {
            n: repr.n,
            words: repr.words.clone(),
        };
        f(&mut r);
        MoveTable::try_from(r)
    };
    assert!(corrupt(&|r| r.words[5] += 1).is_err()); // q shifted: outputs overlap
    assert!(corrupt(&|r| r.words[4] = r.words[0]).is_err()); // p not increasing
    assert!(corrupt(&|r| r.words[2] = 10_000).is_err()); // idx out of range
    assert!(corrupt(&|r| r.words[2] ^= 1).is_err() || t.num_rows() == 1); // wrong idx
    assert!(corrupt(&|r| r.n += 1).is_err());
    assert!(corrupt(&|r| {
        r.words.pop();
    })
    .is_err());
    assert!(corrupt(&|r| r.words[3] = 300).is_err()); // head out of range
    let empty = MoveTable::try_from(MoveTableRepr {
        n: 0,
        words: vec![],
    })
    .unwrap();
    assert_eq!(empty.num_rows(), 0);
}
