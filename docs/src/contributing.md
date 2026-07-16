# Contributing

Contributions are welcome. The authoritative guides live in the repository:

- [CONTRIBUTING.md](https://github.com/sriram98v/haystackfm/blob/main/CONTRIBUTING.md) —
  build / test / lint workflow and PR conventions.
- [CODE_OF_CONDUCT.md](https://github.com/sriram98v/haystackfm/blob/main/CODE_OF_CONDUCT.md) —
  community expectations.
- [SECURITY.md](https://github.com/sriram98v/haystackfm/blob/main/SECURITY.md) — how to
  report a security issue (please don't open a public issue for vulnerabilities).

## Local checks

Before opening a PR, make sure these pass:

```bash
cargo fmt --check
cargo clippy --all-features -- -D warnings
cargo test --features gpu
```

GPU tests and benchmarks require the `gpu` feature and a working GPU/driver.

## License

haystackfm is licensed under the
[Apache License, Version 2.0](https://github.com/sriram98v/haystackfm/blob/main/LICENSE).
Unless you state otherwise, any contribution you submit for inclusion is licensed as above,
without additional terms or conditions.
