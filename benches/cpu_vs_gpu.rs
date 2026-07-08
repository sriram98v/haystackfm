//! CPU vs GPU benchmark suite for each FM-index construction stage.
//!
//! Benchmarks suffix-array (SA), BWT, OCC-table, and full-pipeline construction
//! for both CPU and GPU implementations across five input sizes.
//!
//! # Running
//!
//! ```bash
//! # Full statistical benchmarks + HTML reports
//! cargo bench --bench cpu_vs_gpu --features gpu
//!
//! # Quick smoke-run (no timing output, verifies compilation)
//! cargo bench --bench cpu_vs_gpu --features gpu --no-run
//! ```
//!
//! The speedup summary table is printed to stderr before Criterion starts.

#[path = "bench_utils.rs"]
mod bench_utils;

use bench_utils::{gpu_available, measure_ms, random_dna_seq, BENCH_SIZES};
use criterion::{BenchmarkId, Criterion};
use webgpu_fmidx::alphabet::concatenate_sequences;
use webgpu_fmidx::bwt::cpu::build_bwt as cpu_build_bwt;
use webgpu_fmidx::fm_index::{FmIndex, FmIndexConfig};
use webgpu_fmidx::occ::cpu::build_occ_table as cpu_build_occ;
use webgpu_fmidx::occ::OccEncoding;
use webgpu_fmidx::suffix_array::cpu::build_suffix_array as cpu_build_sa;

#[cfg(feature = "gpu")]
use {
    webgpu_fmidx::bwt::gpu::BwtPipelines, webgpu_fmidx::gpu::GpuContext,
    webgpu_fmidx::occ::gpu::OccPipelines, webgpu_fmidx::suffix_array::gpu::SaPipelines,
};

// ── Speedup summary ───────────────────────────────────────────────────────────

/// Print a human-readable CPU vs GPU speedup table to stderr.
///
/// Uses wall-clock measurements (1 warmup + 3 timed iterations) – less
/// precise than Criterion but sufficient for a quick "is GPU faster?" answer.
#[cfg(feature = "gpu")]
fn print_speedup_table() {
    let ctx = match pollster::block_on(GpuContext::new()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Speedup table skipped: {e}");
            return;
        }
    };
    let sa_pipelines = SaPipelines::new(&ctx);
    let bwt_pipelines = BwtPipelines::new(&ctx);
    let occ_pipelines = OccPipelines::new(&ctx);

    const WARMUP: usize = 1;
    const ITERS: usize = 3;

    let sep = "─".repeat(68);
    let dbl = "═".repeat(68);
    eprintln!("\n{dbl}");
    eprintln!("  CPU vs GPU Speedup Summary  (warmup={WARMUP}, iters={ITERS})");
    eprintln!("{dbl}");
    eprintln!(
        "  {:<18} {:>8}  {:>10}  {:>10}  {:>8}",
        "Stage", "Size", "CPU (ms)", "GPU (ms)", "Speedup"
    );
    eprintln!("  {sep}");

    for &size in BENCH_SIZES {
        let seq = random_dna_seq(size);
        let (text, _) = concatenate_sequences(&[seq.clone()]).unwrap();

        // Pre-build intermediate artifacts (not measured)
        let sa = cpu_build_sa(&text);
        let bwt = cpu_build_bwt(&text, &sa);

        // SA
        let cpu_sa_ms = measure_ms(|| drop(cpu_build_sa(&text)), WARMUP, ITERS);
        let gpu_sa_ms = measure_ms(
            || {
                drop(pollster::block_on(
                    sa_pipelines.build_suffix_array(&ctx, &text),
                ))
            },
            WARMUP,
            ITERS,
        );

        // BWT (SA pre-built via CPU for both)
        let cpu_bwt_ms = measure_ms(|| drop(cpu_build_bwt(&text, &sa)), WARMUP, ITERS);
        let gpu_bwt_ms = measure_ms(
            || {
                drop(pollster::block_on(
                    bwt_pipelines.build_bwt(&ctx, &text, &sa),
                ))
            },
            WARMUP,
            ITERS,
        );

        // OCC (SA + BWT pre-built via CPU for both)
        let cpu_occ_ms =
            measure_ms(|| drop(cpu_build_occ(&bwt, OccEncoding::Bitplane)), WARMUP, ITERS);
        let gpu_occ_ms = measure_ms(
            || {
                drop(pollster::block_on(
                    occ_pipelines.build_occ_table(&ctx, &bwt),
                ))
            },
            WARMUP,
            ITERS,
        );

        // Full pipeline
        let cpu_full_ms = measure_ms(
            || {
                let cfg = FmIndexConfig {
                    sa_sample_rate: 32,
                    use_gpu: false,
                    ..Default::default()
                };
                drop(FmIndex::build_cpu(&[seq.clone()], &cfg).unwrap());
            },
            WARMUP,
            ITERS,
        );
        let gpu_full_ms = measure_ms(
            || {
                let cfg = FmIndexConfig {
                    sa_sample_rate: 32,
                    use_gpu: true,
                    ..Default::default()
                };
                drop(pollster::block_on(FmIndex::build(&[seq.clone()], &cfg)).unwrap());
            },
            WARMUP,
            ITERS,
        );

        for (stage, cpu_ms, gpu_ms) in [
            ("SA construct", cpu_sa_ms, gpu_sa_ms),
            ("BWT construct", cpu_bwt_ms, gpu_bwt_ms),
            ("OCC construct", cpu_occ_ms, gpu_occ_ms),
            ("Full pipeline", cpu_full_ms, gpu_full_ms),
        ] {
            let speedup = cpu_ms / gpu_ms;
            let marker = if speedup >= 1.0 { "▲" } else { "▼" };
            eprintln!(
                "  {:<18} {:>8}  {:>10.3}  {:>10.3}  {:>6.2}x {marker}",
                stage, size, cpu_ms, gpu_ms, speedup
            );
        }
        eprintln!("  {sep}");
    }
    eprintln!("{dbl}");
    eprintln!("  ▲ = GPU faster   ▼ = CPU faster (GPU overhead dominates at small sizes)");
    eprintln!("{dbl}\n");
}

// ── SA construction ───────────────────────────────────────────────────────────

fn bench_sa_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("sa_construction");

    #[cfg(feature = "gpu")]
    let gpu_state = if gpu_available() {
        let ctx = pollster::block_on(GpuContext::new()).unwrap();
        let pipelines = SaPipelines::new(&ctx);
        Some((ctx, pipelines))
    } else {
        None
    };

    for &size in BENCH_SIZES {
        let seq = random_dna_seq(size);
        let (text, _) = concatenate_sequences(&[seq]).unwrap();

        group.bench_function(BenchmarkId::new("cpu", size), |b| {
            b.iter(|| cpu_build_sa(&text))
        });

        #[cfg(feature = "gpu")]
        if let Some((ref ctx, ref pipelines)) = gpu_state {
            group.bench_function(BenchmarkId::new("gpu", size), |b| {
                b.iter(|| pollster::block_on(pipelines.build_suffix_array(ctx, &text)))
            });
        }
    }
    group.finish();
}

// ── BWT construction ──────────────────────────────────────────────────────────

fn bench_bwt_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("bwt_construction");

    #[cfg(feature = "gpu")]
    let gpu_state = if gpu_available() {
        let ctx = pollster::block_on(GpuContext::new()).unwrap();
        let pipelines = BwtPipelines::new(&ctx);
        Some((ctx, pipelines))
    } else {
        None
    };

    for &size in BENCH_SIZES {
        let seq = random_dna_seq(size);
        let (text, _) = concatenate_sequences(&[seq]).unwrap();
        // SA is pre-built via CPU so we measure only BWT cost
        let sa = cpu_build_sa(&text);

        group.bench_function(BenchmarkId::new("cpu", size), |b| {
            b.iter(|| cpu_build_bwt(&text, &sa))
        });

        #[cfg(feature = "gpu")]
        if let Some((ref ctx, ref pipelines)) = gpu_state {
            group.bench_function(BenchmarkId::new("gpu", size), |b| {
                b.iter(|| pollster::block_on(pipelines.build_bwt(ctx, &text, &sa)))
            });
        }
    }
    group.finish();
}

// ── OCC construction ──────────────────────────────────────────────────────────

fn bench_occ_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("occ_construction");

    #[cfg(feature = "gpu")]
    let gpu_state = if gpu_available() {
        let ctx = pollster::block_on(GpuContext::new()).unwrap();
        let pipelines = OccPipelines::new(&ctx);
        Some((ctx, pipelines))
    } else {
        None
    };

    for &size in BENCH_SIZES {
        let seq = random_dna_seq(size);
        let (text, _) = concatenate_sequences(&[seq]).unwrap();
        // SA + BWT pre-built via CPU so we measure only OCC cost
        let sa = cpu_build_sa(&text);
        let bwt = cpu_build_bwt(&text, &sa);

        group.bench_function(BenchmarkId::new("cpu", size), |b| {
            b.iter(|| cpu_build_occ(&bwt, OccEncoding::Bitplane))
        });

        #[cfg(feature = "gpu")]
        if let Some((ref ctx, ref pipelines)) = gpu_state {
            group.bench_function(BenchmarkId::new("gpu", size), |b| {
                b.iter(|| pollster::block_on(pipelines.build_occ_table(ctx, &bwt)))
            });
        }
    }
    group.finish();
}

// ── Full pipeline ─────────────────────────────────────────────────────────────

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    for &size in BENCH_SIZES {
        let seq = random_dna_seq(size);
        let cpu_cfg = FmIndexConfig {
            sa_sample_rate: 32,
            use_gpu: false,
            ..Default::default()
        };

        group.bench_function(BenchmarkId::new("cpu", size), |b| {
            b.iter(|| FmIndex::build_cpu(&[seq.clone()], &cpu_cfg).unwrap())
        });

        #[cfg(feature = "gpu")]
        if gpu_available() {
            let gpu_cfg = FmIndexConfig {
                sa_sample_rate: 32,
                use_gpu: true,
                ..Default::default()
            };
            group.bench_function(BenchmarkId::new("gpu", size), |b| {
                b.iter(|| pollster::block_on(FmIndex::build(&[seq.clone()], &gpu_cfg)).unwrap())
            });
        }
    }
    group.finish();
}

// ── GPU context initialisation overhead ───────────────────────────────────────

#[cfg(feature = "gpu")]
fn bench_gpu_init_overhead(c: &mut Criterion) {
    if !gpu_available() {
        eprintln!("Skipping gpu_init benchmark: GPU unavailable");
        return;
    }
    let mut group = c.benchmark_group("gpu_init");
    group.bench_function("context_creation", |b| {
        b.iter(|| pollster::block_on(GpuContext::new()).unwrap())
    });
    group.finish();
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    // Print speedup table first (wall-clock, not Criterion stats)
    #[cfg(feature = "gpu")]
    {
        if gpu_available() {
            print_speedup_table();
        } else {
            eprintln!("Note: GPU not available – running CPU-only benchmarks.");
        }
    }

    let mut c = Criterion::default().configure_from_args();
    bench_sa_construction(&mut c);
    bench_bwt_construction(&mut c);
    bench_occ_construction(&mut c);
    bench_full_pipeline(&mut c);
    #[cfg(feature = "gpu")]
    bench_gpu_init_overhead(&mut c);
    c.final_summary();
}
