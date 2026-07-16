# haystackfm

[![Crates.io](https://img.shields.io/crates/v/haystackfm.svg)](https://crates.io/crates/haystackfm)
[![Docs.rs](https://docs.rs/haystackfm/badge.svg)](https://docs.rs/haystackfm)
[![CI](https://github.com/sriram98v/haystackfm/actions/workflows/ci.yml/badge.svg)](https://github.com/sriram98v/haystackfm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/sriram98v/haystackfm/blob/main/LICENSE)
![MSRV](https://img.shields.io/badge/MSRV-1.87-blue.svg)

A **GPU-accelerated FM-index** for DNA sequence alignment. It runs on native targets
(Vulkan / Metal / DX12) through [`wgpu`](https://github.com/gfx-rs/wgpu), and compiles to
**WebAssembly** for in-browser WebGPU use — the same index code, on the desktop or in a tab.

haystackfm supports the full **16-symbol IUPAC ambiguity alphabet**
(A C G T N R Y S W K M B D H V) in both its CPU and GPU query paths, with a pluggable
`Alphabet` trait for swapping in custom matching semantics.

> 🔎 **Try it in your browser:** the [live demo](./demo.md) builds an index and runs
> `count` / `locate` queries entirely client-side via WebGPU — no server, no upload.

## What is an FM-index?

An FM-index is a compressed full-text substring index built on the Burrows–Wheeler
Transform (BWT). It answers two core questions over a text of length *n* without keeping the
text uncompressed:

- **`count(P)`** — how many times pattern *P* occurs, in *O(|P|)* time.
- **`locate(P)`** — *where* each occurrence is, in *O(|P| + occ·k)* time.

That makes it the workhorse behind short-read aligners (BWA, Bowtie) and other
bioinformatics tooling. haystackfm pushes the construction and the query hot loops onto the
GPU while keeping a CPU implementation as the correctness ground truth.

## Feature overview

| Feature | Description |
|---------|-------------|
| CPU construction | BWT, suffix array, and Occ table built on CPU |
| GPU construction | All three steps GPU-accelerated via WGSL compute shaders |
| `count` / `locate` | Exact-match and position queries |
| GPU batch locate | IUPAC-aware backward search + SA resolve on GPU |
| MEM / SMEM finding | Bidirectional FM-index; CPU and GPU paths |
| IUPAC ambiguity | Full 16-symbol alphabet in query and reference; GPU parity-tested |
| Pluggable `Alphabet` | Swap matching semantics (`IupacDna` default, `ExactDna` ACGT-only) |
| Lookup-table seeding | Optional depth-*k* prefix table skips the first *k* search steps |
| WASM bindings | `wasm-bindgen` JS/TS API for browser use |
| Serialization | `to_bytes` / `from_bytes` for index persistence |

## Where to go next

- New here? Start with [Installation](./getting-started/installation.md) and the
  [Quick Start](./getting-started/quick-start.md).
- Want the mental model? Read [Concepts](./guide/concepts.md) and
  [Architecture](./reference/architecture.md).
- Looking for the type-level API? It lives on [docs.rs](https://docs.rs/haystackfm).
