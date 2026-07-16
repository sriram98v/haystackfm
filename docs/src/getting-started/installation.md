# Installation

haystackfm is a Rust crate published on [crates.io](https://crates.io/crates/haystackfm).
Its minimum supported Rust version (MSRV) is **1.87**.

## CPU only (default)

```bash
cargo add haystackfm
```

The default `cpu` feature builds and queries indices entirely on the CPU — no GPU driver or
`wgpu` dependency required.

## With GPU acceleration

```bash
cargo add haystackfm --features gpu
```

The `gpu` feature pulls in `wgpu` and enables GPU-accelerated construction plus the batched
GPU query paths. On native targets it uses Vulkan, Metal, or DX12 depending on your platform.

## For the browser (WebAssembly)

```bash
cargo add haystackfm --features wasm
```

Build the WASM package with [`wasm-pack`](https://rustwasm.github.io/wasm-pack/):

```bash
wasm-pack build --target web --features wasm
```

This emits a `pkg/` directory with `haystackfm.js` + `haystackfm_bg.wasm` and TypeScript
type definitions. See [WebAssembly](../guide/wasm.md) for the JS/TS API.

## Feature flags at a glance

| Flag | Default | Description |
|------|---------|-------------|
| `cpu` | yes | CPU construction and queries |
| `gpu` | no | WebGPU-accelerated construction and queries via `wgpu` |
| `wasm` | no | `wasm-bindgen` JS/TS bindings (implies `gpu`) |

See [Feature Flags](../reference/feature-flags.md) for details.
