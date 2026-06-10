# Plan: Add webgpu-fmidx CPU build to `rust-fmindex-benchmark`

## Scope

Add only the **CPU build** of `webgpu-fmidx` to the benchmark. The GPU build is excluded because:

- It requires async execution (`FmIndex::build()` is `async`) — the trait is synchronous
- Results would be hardware-dependent (GPU model, VRAM) — breaks the "algorithm comparison" premise
- `wgpu` has no CPU fallback on all platforms — would break CI on bare-metal runners

## Architecture mapping

The benchmark's `BenchmarkFmIndex` trait maps to `webgpu_fmidx` as follows:

| Trait method                                   | `webgpu_fmidx` call                  | Notes                         |
| ---------------------------------------------- | ------------------------------------ | ----------------------------- |
| `construct_for_benchmark(config, texts)`       | `FmIndex::build_cpu(&seqs, &config)` | Sync, single-threaded         |
| `count_for_benchmark(index, query)`            | `index.count(&encoded)`              | Encoded query bytes           |
| `count_via_locate_for_benchmark(index, query)` | `index.locate(&encoded)`             | Returns positions, count them |
| `write_to_file_for_benchmark(index, path)`     | `index.to_bytes()` + `fs::write`     | Bincode serialization         |
| `load_from_file_for_benchmark(path)`           | `FmIndex::from_bytes(&bytes)`        | Bincode deserialization       |

## Files to create / modify

### 1. New file: `src/webgpu_fmidx_bench.rs`

A thin wrapper implementing `BenchmarkFmIndex` for `webgpu_fmidx::FmIndex`.

Key conversion logic:

**Text encoding** (in `construct_for_benchmark`):

- Input: `Vec<Vec<u8>>` from FASTA files (raw bytes like `b"ACGT..."`)
- Need to: lowercase → uppercase, validate only A/C/G/T, append sentinel `$` (encoded as `0`)
- Convert to `Vec<DnaSequence>` using `DnaSequence::from_slice()`
- Set `sa_sample_rate = config.suffix_array_sampling_rate`

**Query encoding** (in `count_for_benchmark` / `count_via_locate_for_benchmark`):

- Input: `&[u8]` raw bytes like `b"ACGT"`
- Map: `A→1, C→2, G→3, T→4` (your library's encoding)
- Pass as `&[u8]` to `index.count()` / `index.locate()`

**File I/O**:

- `write`: `index.to_bytes()?` → `fs::write(path, bytes)?`
- `load`: `fs::read(path)?` → `FmIndex::from_bytes(&bytes)?`

### 2. Modify: `src/main.rs`

Add one variant to the `Library` enum:

```rust
WebgpuFmidx,
```

Add the match arm in `run_benchmark_for_index_type`:

```rust
Library::WebgpuFmidx => webgpu_fmidx_bench::WebgpuFmidx::run_benchmark(config),
```

Add `mod webgpu_fmidx_bench;` at the top.

### 3. Modify: `Cargo.toml`

Add dependency:

```toml
webgpu-fmidx = { version = "0.1", features = ["cpu"] }
```

### 4. Modify: `plots/main.py`

Add to `library_name_to_info`:

```python
"WebgpuFmidx": ("webgpu (cpu)", "darkorchid", "darkorchid"),
```

## Design decisions

### 1. Single-threaded construction

- Your CPU build is single-threaded
- The benchmark runs with `--build-thread-count 1` by default
- No need for a multithreaded variant — mark multithreaded support as ❌

### 2. Memory usage

- Prefix-doubling SA construction is O(n log² n) work with high constant
- Expect peak memory to be **10-20×** input size (worse than genedex's 5-10×)
- This is a fair comparison — the benchmark measures actual memory, not just "good"

### 3. Sampling rate

- Benchmark uses `sa_sample_rate = 4` for all libraries
- Map directly: `config.suffix_array_sampling_rate` → `FmIndexConfig.sa_sample_rate`

### 4. Lookup table

- Benchmark supports configurable lookup table depth
- `webgpu_fmidx` does not currently have a lookup table feature
- Mark this as unsupported; the benchmark will use default (no lookup table)

### 5. Multiple texts

- Your library supports `Vec<DnaSequence>` — multiple texts natively
- Mark as ✅

### 6. mmap support

- Your library uses bincode serialization to a `Vec<u8>` — no mmap
- Mark as ❌

## Implementation estimate

| Step                               | Effort                  |
| ---------------------------------- | ----------------------- |
| `webgpu_fmidx_bench.rs` wrapper    | ~3-4 hours              |
| `main.rs` integration              | ~30 min                 |
| `Cargo.toml` dependency            | ~5 min                  |
| `plots/main.py` color registration | ~5 min                  |
| Local testing with small dataset   | ~1-2 hours              |
| Testing with hg38 dataset          | ~2-4 hours (build time) |
| PR + review                        | ~1 hour                 |

**Total: ~8-12 hours** (mostly testing with real data)

## Risks and mitigations

1. **Your library's CPU build might be slow** — prefix-doubling on hg38 could take 30+ minutes. Mitigation: test with i32 dataset first.

2. **Memory usage might exceed the server's limits** — hg38 is 3.3 GB, double-hg38 is 6.6 GB. If peak memory exceeds 10×, that's 33-66 GB. Mitigation: the benchmark server has 1 TB RAM, so this is fine. But the chromosome/i32 datasets should work on modest machines.

3. **Your library's encoding might not match raw FASTA bytes** — need to verify the lowercase→uppercase + ACGT validation step works correctly with the benchmark's input format.

4. **Your library's `sa_sample_rate` field name** — verify it's exactly `sa_sample_rate` in `FmIndexConfig`.

## PR structure

Split into two PRs for cleaner review:

1. **PR 1: `webgpu-fmidx` side** — Add any missing public API methods needed by the benchmark (e.g., ensure `build_cpu` is public, verify `from_bytes` is public, add any needed error type conversions).

2. **PR 2: `rust-fmindex-benchmark` side** — The wrapper implementation + integration.

Actually, on second thought, since your library already has all the needed APIs (`build_cpu`, `count`, `locate`, `to_bytes`, `from_bytes`), a single PR to `rust-fmindex-benchmark` should suffice.

## Testing checklist

- [ ] `cargo build` succeeds with the new dependency
- [ ] Benchmark runs on `i32` dataset (fast, ~2 GB)
- [ ] Benchmark runs on `hg38` dataset (~3.3 GB)
- [ ] Results appear in JSON output
- [ ] Plots generate correctly with the new library
- [ ] Verify results are deterministic (run twice, compare)
- [ ] Verify locate results match count results (sanity check)
