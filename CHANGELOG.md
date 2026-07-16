# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First release under the `haystackfm` name. The project was previously published as
`webgpu-fmidx` (versions 0.1.0–0.5.1); version numbering restarts at 0.1.0 under the new
name, and its earlier history is not carried over here.

### Added
- GPU-accelerated FM-index construction (suffix array, BWT, Occ table) via WebGPU compute
  shaders, alongside CPU construction.
- `count` / `locate` queries; bidirectional index with MEM / SMEM finding (CPU and GPU paths).
- Full 16-symbol IUPAC ambiguity alphabet with a pluggable `Alphabet` trait
  (`IupacDna` default, `ExactDna` for exact ACGT matching).
- WASM bindings for in-browser WebGPU use; index serialization (`to_bytes` / `from_bytes`).
- Community health files, CI (fmt / clippy / build / test on `--all-features`), and Dependabot.

### Changed
- Licensed under Apache-2.0.

[Unreleased]: https://github.com/sriram98v/haystackfm/commits/main
