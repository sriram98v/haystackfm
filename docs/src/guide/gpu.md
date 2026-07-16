# GPU Acceleration

With the `gpu` feature, construction and the query hot loops offload to the GPU via
[`wgpu`](https://github.com/gfx-rs/wgpu). On native targets that means Vulkan, Metal, or
DX12; in the browser it means WebGPU (see [WebAssembly](./wasm.md)).

## Building on the GPU

```rust
let index = FmIndex::build(&seqs, &config).await?; // async; returns a normal FmIndex
```

The suffix array, BWT, and Occ table are all built with WGSL compute shaders. The result is
the same `FmIndex` you'd get from `build_cpu` — you can query it on either the CPU or the GPU.

## `GpuContext` caching

Adapter and device initialization is not free, so haystackfm caches a process-wide
`GpuContext` in an `OnceLock` (`src/gpu/context_cache.rs`). GPU functions accept an optional
pre-initialized context: pass `None` on the first call and reuse the cached context
afterward. This is what keeps benchmarks and tests from paying init overhead on every call.

## GPU query paths

- **Batch locate** — `locate_batch_gpu` runs backward search and SA resolution on the GPU for
  a batch of patterns. See [Count & Locate](./count-locate.md).
- **MEM / SMEM** — `find_mems_gpu` / `find_smems_gpu` run the bidirectional-extension
  pipeline on the GPU. See [MEM / SMEM Finding](./mem-smem.md).

Every GPU path is validated against the CPU implementation, which remains the correctness
oracle. The multi-pass shader pipelines are described in
[Architecture](../reference/architecture.md).

## When the GPU helps

The GPU wins on **batches** of queries over **large** indices, where there's enough
data-parallel work to amortize dispatch and transfer costs. For a single short pattern on a
small index, the CPU path is typically faster — the query itself is cheaper than moving data
to the device. Measure with your own workload; the benchmarks in the repo
(`cargo bench --features gpu`) are a good starting template.
