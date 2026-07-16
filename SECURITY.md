# Security Policy

## Supported Versions

Security fixes are applied to the latest released `0.x` minor version. Older versions are not
maintained — please upgrade to the newest release before reporting.

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately using GitHub's [private vulnerability reporting][gh-advisory] ("Report a
vulnerability" under the repository's **Security** tab), or email **sriram.98v@gmail.com** with:

- A description of the issue and its impact
- Steps to reproduce (a minimal test case or input is ideal)
- Affected version(s) and platform (native backend / browser WebGPU)

You can expect an acknowledgement within **7 days**. Once a fix is ready we will coordinate a
release and, if you wish, credit you in the release notes.

[gh-advisory]: https://github.com/sriram98v/haystackfm/security/advisories/new

## Scope

This is a data-structure/algorithm library operating on caller-supplied sequence data. Of
particular interest are memory-safety issues (out-of-bounds access in index construction/query,
`unsafe` blocks, GPU buffer sizing), panics on malformed input, and deserialization of untrusted
`to_bytes`/`from_bytes` payloads.
