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

Bytes start with a 4-byte format marker (`"HFM"` + version). Blobs written before the
marker existed still load, but they carry no LCP array, so `has_lcp()` is `false` and
`contract_left` / `contract_right` return `FmIndexError::LcpNotBuilt`; rebuild and
re-serialize to get it. Bidirectional blobs written when only the forward half carried an
LCP array load the same way: `contract_left` works, `contract_right` does not, and
`has_lcp()` (which requires both halves) is `false`.

## Sequence ids are stable

The headers are stored in build order, so a
[`SeqId`](./concepts.md#sequence-ids-vs-headers) means the same reference before and after a
round trip — you can record ids against a serialized index and reuse them on reload. (The
header -> id map is derived from the header list rather than stored, so `from_bytes` rebuilds
it; this is invisible to callers.)

Ids are only meaningful *within* one index. A differently-built index — sequences added,
removed, or reordered — renumbers them; match on headers when crossing that boundary.

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
