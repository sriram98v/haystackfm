# Browser Support

GPU construction and queries in the browser require **WebGPU**.

| Browser | Support |
|---------|---------|
| Chrome 113+ | Stable |
| Edge 113+ | Stable |
| Safari 18+ | Stable |
| Firefox | Behind a flag |

CPU builds (`build_cpu`) work in any browser that runs WebAssembly — WebGPU is only needed
for the GPU build path (`build_gpu`).

## Checking at runtime

```javascript
if ("gpu" in navigator) {
  // WebGPU available — build_gpu will work
} else {
  // Fall back to build_cpu
}
```

The [live demo](../demo.md) does exactly this: it detects WebGPU, shows a badge, and enables
or disables the GPU build button accordingly.
