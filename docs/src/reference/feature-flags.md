# Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `cpu` | yes | CPU construction and queries |
| `gpu` | no | WebGPU-accelerated construction and queries via `wgpu` |
| `wasm` | no | `wasm-bindgen` JS/TS bindings |

## `cpu`

The default. Everything builds and queries on the CPU with no GPU dependency. This is the
correctness ground truth for every GPU path.

## `gpu`

Adds `wgpu` and the GPU construction + query paths (`FmIndex::build`, `locate_batch_gpu`,
`find_mems_gpu` / `find_smems_gpu`). Required for all GPU benchmarks and tests:

```bash
cargo test --features gpu
cargo bench --features gpu
```

## `wasm`

Adds the `wasm-bindgen` bindings for browser use and implies `gpu` (browser WebGPU). Build
the package with `wasm-pack build --target web --features wasm`. See
[WebAssembly](../guide/wasm.md).
