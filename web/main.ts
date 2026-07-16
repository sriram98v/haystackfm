// Import the wasm-pack generated module.
// Run `npm run build-wasm` first to generate ./pkg/
import init, { FmIndexBuilder, FmIndexHandle } from "./pkg/haystackfm.js";

// ── Initialise WASM ──────────────────────────────────────────────────────────

await init();

// ── DOM helpers ──────────────────────────────────────────────────────────────

function $(id: string): HTMLElement {
  return document.getElementById(id)!;
}

function setStatus(
  el: HTMLElement,
  kind: "info" | "success" | "error",
  msg: string
): void {
  el.className = `status ${kind}`;
  el.textContent = msg;
  el.classList.remove("hidden");
}

function show(el: HTMLElement): void {
  el.classList.remove("hidden");
}

function hide(el: HTMLElement): void {
  el.classList.add("hidden");
}

// ── GPU availability badge ────────────────────────────────────────────────────

const gpuBadge = $("gpu-badge");
if ("gpu" in navigator) {
  const adapter = await (navigator as unknown as { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter();
  if (adapter) {
    gpuBadge.textContent = "WebGPU available";
    gpuBadge.className = "available";
  } else {
    gpuBadge.textContent = "WebGPU unavailable";
    gpuBadge.className = "unavailable";
  }
} else {
  gpuBadge.textContent = "No WebGPU";
  gpuBadge.className = "unavailable";
}

// ── State ────────────────────────────────────────────────────────────────────

let currentHandle: FmIndexHandle | null = null;
let exportedBytes: Uint8Array | null = null;

// ── Example sequences ─────────────────────────────────────────────────────────

const EXAMPLE_FASTA = `>human_brca1_fragment
ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAA
ATCTTGGAGTGTCCAATGAAACTTTGTCATGAGTGGCGAAACCCGAAAATGATTTATCTG
>e_coli_recA_fragment
ATGGCTATCGACGAAAACAAACAGAAAGCGTTGGCGGCAGCACTGGGCCAGATTGAGAAA
ACAGAAGGCTTTCTGTGTGAAACAGAAATCGAAGACATTGAACGTGGTGGCGGTCTGGCT
>arabidopsis_thaliana_fragment
ATGAAGACGATCATCGCCTTGGCGACGCTTGCGTTCACCTTCATCTTCCTCTTTCTTCTT
ATCATCATCGTCGCCGCCGTTAACGCCGCCATCGTCGCCGCTGCCATCATCATCATCACC`;

$("load-example-btn").addEventListener("click", () => {
  ($("sequences-input") as HTMLTextAreaElement).value = EXAMPLE_FASTA;
});

// ── Build ────────────────────────────────────────────────────────────────────

async function buildIndex(useGpu: boolean): Promise<void> {
  const input = ($("sequences-input") as HTMLTextAreaElement).value.trim();
  if (!input) {
    setStatus($("build-status"), "error", "Please enter at least one sequence.");
    return;
  }

  const sampleRateStr = ($("sample-rate") as HTMLInputElement).value.trim();
  const sampleRate = parseInt(sampleRateStr, 10);
  if (isNaN(sampleRate) || sampleRate < 1) {
    setStatus($("build-status"), "error", "SA sample rate must be a positive integer.");
    return;
  }

  // Disable buttons while building
  ($("build-cpu-btn") as HTMLButtonElement).disabled = true;
  ($("build-gpu-btn") as HTMLButtonElement).disabled = true;
  setStatus($("build-status"), "info", useGpu ? "Building on GPU…" : "Building on CPU…");

  const builder = new FmIndexBuilder(sampleRate);

  try {
    builder.add_fasta(input);
  } catch (e) {
    setStatus($("build-status"), "error", `Input error: ${e}`);
    ($("build-cpu-btn") as HTMLButtonElement).disabled = false;
    ($("build-gpu-btn") as HTMLButtonElement).disabled = false;
    builder.free();
    return;
  }

  const seqCount = builder.sequence_count();
  const t0 = performance.now();

  try {
    currentHandle?.free();
    currentHandle = useGpu ? await builder.build_gpu() : builder.build_cpu();
    exportedBytes = null;
    ($("import-btn") as HTMLButtonElement).disabled = true;
  } catch (e) {
    setStatus($("build-status"), "error", `Build failed: ${e}`);
    ($("build-cpu-btn") as HTMLButtonElement).disabled = false;
    ($("build-gpu-btn") as HTMLButtonElement).disabled = false;
    builder.free();
    return;
  }

  const elapsed = performance.now() - t0;
  builder.free();

  setStatus(
    $("build-status"),
    "success",
    `Built in ${elapsed.toFixed(1)} ms using ${useGpu ? "GPU" : "CPU"}.`
  );

  // Show stats
  $("index-metrics").innerHTML = `
    <div class="metric"><span>${seqCount}</span><small>sequences</small></div>
    <div class="metric"><span>${currentHandle!.text_len().toLocaleString()}</span><small>text length (incl. sentinels)</small></div>
    <div class="metric"><span>${sampleRate}x</span><small>SA sample rate</small></div>
    <div class="metric"><span>${elapsed.toFixed(1)} ms</span><small>build time (${useGpu ? "GPU" : "CPU"})</small></div>
  `;
  show($("index-stats-card"));
  show($("query-card"));
  show($("serialize-card"));

  ($("build-cpu-btn") as HTMLButtonElement).disabled = false;
  ($("build-gpu-btn") as HTMLButtonElement).disabled = false;
}

$("build-cpu-btn").addEventListener("click", () => buildIndex(false));
$("build-gpu-btn").addEventListener("click", () => buildIndex(true));

// ── Search ────────────────────────────────────────────────────────────────────

$("search-btn").addEventListener("click", () => {
  if (!currentHandle) {
    setStatus($("search-status"), "error", "Build an index first.");
    return;
  }

  const pattern = ($("pattern-input") as HTMLInputElement).value.trim().toUpperCase();
  if (!pattern) {
    setStatus($("search-status"), "error", "Enter a pattern to search.");
    return;
  }

  const t0 = performance.now();
  let count: number;
  let positions: Uint32Array;

  try {
    count = currentHandle.count(pattern);
    positions = currentHandle.locate(pattern) as unknown as Uint32Array;
  } catch (e) {
    setStatus($("search-status"), "error", `Query error: ${e}`);
    hide($("search-results"));
    return;
  }

  const elapsed = performance.now() - t0;

  setStatus(
    $("search-status"),
    count > 0 ? "success" : "info",
    `Found ${count} occurrence${count !== 1 ? "s" : ""} in ${elapsed.toFixed(2)} ms.`
  );

  if (count === 0) {
    hide($("search-results"));
    return;
  }

  const sorted = Array.from(positions).sort((a, b) => a - b);
  const rows = sorted
    .slice(0, 200) // cap display at 200 rows
    .map((pos) => `<tr><td>${pos}</td></tr>`)
    .join("");

  const truncated = sorted.length > 200
    ? `<p style="font-size:0.8rem;color:#8b949e">Showing first 200 of ${sorted.length} positions.</p>`
    : "";

  $("search-results").innerHTML = `
    ${truncated}
    <table>
      <thead><tr><th>Position (0-based)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
  show($("search-results"));
});

// ── Serialize / Deserialize ───────────────────────────────────────────────────

$("export-btn").addEventListener("click", () => {
  if (!currentHandle) {
    setStatus($("serialize-status"), "error", "Build an index first.");
    return;
  }

  try {
    exportedBytes = currentHandle.to_bytes() as unknown as Uint8Array;
  } catch (e) {
    setStatus($("serialize-status"), "error", `Export failed: ${e}`);
    return;
  }

  const kb = (exportedBytes.byteLength / 1024).toFixed(1);
  setStatus(
    $("serialize-status"),
    "success",
    `Exported ${exportedBytes.byteLength.toLocaleString()} bytes (${kb} KB). Click "Reload from bytes" to deserialize.`
  );
  ($("import-btn") as HTMLButtonElement).disabled = false;
});

$("import-btn").addEventListener("click", () => {
  if (!exportedBytes) return;

  try {
    currentHandle?.free();
    currentHandle = FmIndexHandle.from_bytes(exportedBytes);
  } catch (e) {
    setStatus($("serialize-status"), "error", `Import failed: ${e}`);
    return;
  }

  setStatus(
    $("serialize-status"),
    "success",
    `Deserialized successfully. Index is ready (${currentHandle!.num_sequences()} sequences, text length ${currentHandle!.text_len().toLocaleString()}).`
  );
});
