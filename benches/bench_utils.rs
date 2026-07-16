//! Shared benchmark utilities used by the cpu_vs_gpu benchmark suite.

use haystackfm::alphabet::DnaSequence;

/// Input sizes for the per-stage and full-pipeline benchmarks.
pub const BENCH_SIZES: &[usize] = &[1_000, 10_000, 100_000, 500_000];

/// Generate a random DNA string of the given length.
pub fn random_dna(len: usize) -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let bases = ['A', 'C', 'G', 'T'];
    (0..len).map(|_| bases[rng.random_range(0..4)]).collect()
}

/// Generate a random, encoded `DnaSequence` of the given length.
pub fn random_dna_seq(len: usize) -> DnaSequence {
    DnaSequence::from_str(&random_dna(len)).unwrap()
}

/// Returns `true` if a GPU adapter can be initialised on this machine.
pub fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        use haystackfm::gpu::GpuContext;
        pollster::block_on(GpuContext::new()).is_ok()
    }
    #[cfg(not(feature = "gpu"))]
    false
}

/// Run `f` for `warmup` throw-away iterations, then measure `iters` timed
/// iterations and return the mean elapsed time in milliseconds.
pub fn measure_ms<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}
