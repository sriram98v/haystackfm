# Serialization

A built index can be serialized to bytes and reloaded, so you can construct once and cache
the result rather than rebuilding on every run.

## Rust

```rust
let bytes = index.to_bytes()?;              // Vec<u8>
let restored = FmIndex::from_bytes(&bytes)?; // FmIndex
```

`BidirFmIndex` exposes the same pair. The serialized form records the alphabet's
serialization tag, so a reloaded index keeps the matching semantics it was built with (see
[Custom Alphabets](./alphabets.md)).

## JavaScript / TypeScript

```typescript
const bytes = handle.to_bytes();             // Uint8Array
const restored = FmIndexHandle.from_bytes(bytes);
```

## Trusting input

Deserialization reconstructs an index from a byte buffer. Treat `from_bytes` input as you
would any deserialized data — only load payloads you produced or otherwise trust. See the
project [security policy](https://github.com/sriram98v/haystackfm/blob/main/SECURITY.md) for
the threat model.
