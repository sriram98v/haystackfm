# Live Demo

haystackfm runs entirely in your browser. The demo builds an FM-index (on the CPU **or** the
GPU via WebGPU), runs `count` / `locate` queries, and serializes the index — all client-side,
with nothing uploaded to a server.

<div class="demo-cta">
  <a class="demo-button" href="demo/index.html">▶ Open the interactive demo</a>
</div>

## What you can do

1. **Input sequences** — paste plain ACGT lines or FASTA, or load the bundled example.
2. **Build** — construct the index on the CPU or, if your browser supports WebGPU, on the GPU.
3. **Search** — enter a pattern and see the occurrence count and located positions.
4. **Serialize** — export the built index to bytes and reload it, exercising
   `to_bytes` / `from_bytes`.

## Requirements

The GPU build path needs a browser with **WebGPU** (Chrome/Edge 113+, Safari 18+, Firefox
behind a flag — see [Browser Support](./reference/browser-support.md)). The CPU build works in
any browser that runs WebAssembly; the demo detects WebGPU and enables the GPU button only
when it's available.
