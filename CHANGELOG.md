# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR
  templates, `CODEOWNERS`, and Dependabot configuration.
- README badges (crates.io, docs.rs, CI, license, MSRV) and crate metadata
  (`keywords`, `categories`, `rust-version`, `documentation`, `authors`).

### Changed
- CI now lints and builds with `--all-features` so GPU/WASM code is checked.
- Corrected license metadata: the crate is licensed under **Apache-2.0** (previously
  mislabeled `MIT` in `Cargo.toml` and the README while the `LICENSE` file was Apache-2.0).

## [0.6.0]

### Changed
- Restricted `SampledSuffixArray::from_full` to crate-internal visibility (`pub(crate)`).
  This is an API-breaking change, hence the minor bump.

## [0.5.1]

- Documentation and API-accuracy fixes.

## [0.3.0]

- Earlier tagged release.

[Unreleased]: https://github.com/sriram98v/haystackfm/compare/v0.5.1...HEAD
[0.6.0]: https://github.com/sriram98v/haystackfm/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/sriram98v/haystackfm/compare/v0.3.0...v0.5.1
[0.3.0]: https://github.com/sriram98v/haystackfm/releases/tag/v0.3.0
