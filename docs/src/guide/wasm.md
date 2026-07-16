# WebAssembly

The same index runs in the browser. With the `wasm` feature, haystackfm exposes a
`wasm-bindgen` JS/TS API and uses the browser's WebGPU for GPU construction and queries.

## Building the package

```bash
cargo add haystackfm --features wasm
wasm-pack build --target web --features wasm
```

This produces a `pkg/` directory containing `haystackfm.js`, `haystackfm_bg.wasm`, and
`.d.ts` type definitions.

## JS/TS API

```typescript
import init, { FmIndexBuilder, FmIndexHandle } from "./pkg/haystackfm.js";

await init();

const builder = new FmIndexBuilder(/*sa_sample_rate=*/32);
builder.add_fasta(`>seq1\nACGTACGT\n>seq2\nTGCATGCA`);

const handle = await builder.build_gpu();    // or builder.build_cpu()

console.log(handle.count("ACGT"));           // number of occurrences
console.log(handle.locate("ACGT"));          // Array of [seqId, offset] pairs
console.log(handle.text_len());              // total text length
console.log(handle.num_sequences());         // number of indexed sequences

const bytes = handle.to_bytes();             // Uint8Array — serialize
const restored = FmIndexHandle.from_bytes(bytes);
```

## Requirements

GPU builds and queries require a browser with WebGPU. CPU builds (`build_cpu`) work anywhere
WASM runs. See [Browser Support](../reference/browser-support.md), and try the
[live demo](../demo.md) to see it end-to-end.
