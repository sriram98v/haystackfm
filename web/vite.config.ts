import { defineConfig } from "vite";

export default defineConfig({
  // Allow serving the WASM file with correct MIME type
  assetsInclude: ["**/*.wasm"],
  // In CI the workflow sets VITE_BASE_URL to /<repo-name>/ so that
  // GitHub Pages serves assets from the correct path. Locally defaults to /.
  base: process.env.VITE_BASE_URL ?? "/",
  server: {
    headers: {
      // Required for SharedArrayBuffer / WebGPU in some browsers
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  build: {
    // Top-level await (used by wasm-bindgen init + WebGPU calls) requires
    // a modern target. esnext lets esbuild pass it through unchanged.
    target: "esnext",
  },
  optimizeDeps: {
    // Don't bundle the WASM package — it is loaded dynamically
    exclude: ["haystackfm"],
  },
});
