use crate::alphabet::ALPHABET_SIZE;
use crate::error::FmIndexError;
use crate::fm_index::bidir_index::BidirFmIndex;
use crate::gpu::GpuContext;
use wgpu;

const MEM_FIND_SHADER: &str = include_str!("../../shaders/mem_find.wgsl");

pub(crate) const MODE_SMEM: u32 = 0;
pub(crate) const MODE_MEM: u32 = 1;

/// Raw MEM interval as emitted by the GPU finder: query span + forward SA interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawMemInterval {
    pub query_start: u32,
    pub query_end:   u32,
    pub fwd_lo:      u32,
    pub fwd_hi:      u32,
}

// 20 × u32 = 80 bytes (multiple of 16 — satisfies WGSL uniform alignment).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MemFindParams {
    n_queries:      u32,
    min_len:        u32,
    fwd_text_len:   u32,
    rev_text_len:   u32,
    fwd_num_blocks: u32,
    rev_num_blocks: u32,
    mode:           u32,
    total_mems:     u32,
    fwd_c:          [u32; 6],
    rev_c:          [u32; 6],
}

/// Private core: runs the two GPU MEM-finding passes.
///
/// Returns `None` when no MEMs were found (all counts zero).
/// Otherwise returns `(results_flat, mem_counts, mem_offsets)` where
/// `results_flat` has stride 4: `[query_start, query_end, fwd_lo, fwd_hi]` per MEM.
async fn run_mem_find_gpu(
    ctx: &GpuContext,
    bidir: &BidirFmIndex,
    queries: &[&[u8]],
    min_len: usize,
    mode: u32,
) -> Result<Option<(Vec<u32>, Vec<u32>, Vec<u32>)>, FmIndexError> {
    if queries.is_empty() {
        return Ok(None);
    }

    let n_queries      = queries.len() as u32;
    let block_size: u32 = 64;
    let alpha           = ALPHABET_SIZE as u32;

    let fwd_text_len   = bidir.fwd.text_len;
    let rev_text_len   = bidir.rev.text_len;
    let fwd_num_blocks = (fwd_text_len + block_size - 1) / block_size;
    let rev_num_blocks = (rev_text_len + block_size - 1) / block_size;

    let fwd_c: [u32; 6] = bidir.fwd.c_array.data;
    let rev_c: [u32; 6] = bidir.rev.c_array.data;

    // Flatten fwd then rev checkpoints into a single buffer.
    let mut all_checkpoints: Vec<u32> =
        Vec::with_capacity(((fwd_num_blocks + rev_num_blocks) * alpha) as usize);
    for block in &bidir.fwd.occ.checkpoints {
        all_checkpoints.extend_from_slice(block);
    }
    for block in &bidir.rev.occ.checkpoints {
        all_checkpoints.extend_from_slice(block);
    }

    // Flatten fwd then rev bitvectors (each u64 split lo/hi).
    let mut all_bitvectors: Vec<u32> =
        Vec::with_capacity(((fwd_num_blocks + rev_num_blocks) * alpha * 2) as usize);
    for block in &bidir.fwd.occ.bitvectors {
        for &bv64 in block.iter() {
            all_bitvectors.push(bv64 as u32);
            all_bitvectors.push((bv64 >> 32) as u32);
        }
    }
    for block in &bidir.rev.occ.bitvectors {
        for &bv64 in block.iter() {
            all_bitvectors.push(bv64 as u32);
            all_bitvectors.push((bv64 >> 32) as u32);
        }
    }

    // Encode queries flat + per-query offsets.
    let mut queries_flat: Vec<u32> = Vec::new();
    let mut query_offsets: Vec<u32> = Vec::with_capacity(queries.len() + 1);
    query_offsets.push(0);
    for &q in queries {
        for &b in q {
            queries_flat.push(b as u32);
        }
        query_offsets.push(queries_flat.len() as u32);
    }
    if queries_flat.is_empty() {
        queries_flat.push(0); // wgpu requires non-zero-size buffers
    }

    let chk_buf   = ctx.create_buffer_init("mem_chk",   &all_checkpoints);
    let bv_buf    = ctx.create_buffer_init("mem_bv",    &all_bitvectors);
    let qflat_buf = ctx.create_buffer_init("mem_qflat", &queries_flat);
    let qoff_buf  = ctx.create_buffer_init("mem_qoff",  &query_offsets);

    // pass_buf_a: pass 1 output (mem_counts), pass 2 input (mem_offsets)
    let pass_buf_a = ctx.create_buffer_empty("mem_pass_a", n_queries + 1);
    // dummy binding 5 for pass 1 (mems_out not written in count pass)
    let dummy_buf  = ctx.create_buffer_empty("mem_dummy", 1);

    // ── Pass 1: count MEMs per query ─────────────────────────────────────────
    let count_params = MemFindParams {
        n_queries,
        min_len: min_len as u32,
        fwd_text_len,
        rev_text_len,
        fwd_num_blocks,
        rev_num_blocks,
        mode,
        total_mems: 0,
        fwd_c,
        rev_c,
    };
    let count_params_buf = ctx.create_uniform_buffer("mem_count_params", &count_params);

    let count_pipeline = ctx.create_compute_pipeline("mem_count", MEM_FIND_SHADER, "count_mems");
    let count_bg = ctx.create_bind_group(
        &count_pipeline,
        0,
        &[
            wgpu::BindGroupEntry { binding: 0, resource: qflat_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: qoff_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: chk_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bv_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pass_buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: dummy_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: count_params_buf.as_entire_binding() },
        ],
    );

    let wg_size: u32 = 64;
    ctx.dispatch(
        &count_pipeline,
        &count_bg,
        ((n_queries + wg_size - 1) / wg_size, 1, 1),
    );

    let mem_counts = ctx.download_buffer(&pass_buf_a, n_queries).await;

    // Compute total MEMs and exclusive prefix sum → mem_offsets.
    let total_mems: u32 = mem_counts.iter().sum();
    if total_mems == 0 {
        return Ok(None);
    }

    let mut mem_offsets: Vec<u32> = Vec::with_capacity(queries.len() + 1);
    mem_offsets.push(0);
    for &c in &mem_counts {
        let prev = *mem_offsets.last().unwrap();
        mem_offsets.push(prev + c);
    }

    // ── Pass 2: write MEM results ─────────────────────────────────────────────
    let offsets_buf = ctx.create_buffer_init("mem_offsets", &mem_offsets);
    let mems_out_buf = ctx.create_buffer_empty("mem_results", total_mems * 4);

    let write_params = MemFindParams {
        total_mems,
        ..count_params
    };
    let write_params_buf = ctx.create_uniform_buffer("mem_write_params", &write_params);

    let write_pipeline = ctx.create_compute_pipeline("mem_write", MEM_FIND_SHADER, "write_mems");
    let write_bg = ctx.create_bind_group(
        &write_pipeline,
        0,
        &[
            wgpu::BindGroupEntry { binding: 0, resource: qflat_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: qoff_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: chk_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bv_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: offsets_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: mems_out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: write_params_buf.as_entire_binding() },
        ],
    );

    ctx.dispatch(
        &write_pipeline,
        &write_bg,
        ((n_queries + wg_size - 1) / wg_size, 1, 1),
    );

    let results_flat = ctx.download_buffer(&mems_out_buf, total_mems * 4).await;
    Ok(Some((results_flat, mem_counts, mem_offsets)))
}

/// GPU-accelerated batch MEM/SMEM finding.
///
/// Returns one `Vec<(query_start, query_end, match_count)>` per query.
/// `mode` selects SMEM (`MODE_SMEM=0`) or MEM (`MODE_MEM=1`) behaviour.
pub async fn find_mems_batch_gpu(
    ctx: &GpuContext,
    bidir: &BidirFmIndex,
    queries: &[&[u8]],
    min_len: usize,
    mode: u32,
) -> Result<Vec<Vec<(u32, u32, u32)>>, FmIndexError> {
    let n = queries.len();
    let Some((flat, counts, offsets)) =
        run_mem_find_gpu(ctx, bidir, queries, min_len, mode).await?
    else {
        return Ok(vec![vec![]; n]);
    };
    let mut output: Vec<Vec<(u32, u32, u32)>> = vec![vec![]; n];
    for (q, &count) in counts.iter().enumerate() {
        let off = offsets[q] as usize;
        let mut hits = Vec::with_capacity(count as usize);
        for k in 0..count as usize {
            let base = (off + k) * 4;
            // slot: [query_start, query_end, fwd_lo, fwd_hi]
            // shim: match_count = fwd_hi - fwd_lo
            hits.push((flat[base], flat[base + 1], flat[base + 3] - flat[base + 2]));
        }
        output[q] = hits;
    }
    Ok(output)
}

/// Returns raw SA intervals per MEM — used by the GPU position-resolve pipeline.
pub(crate) async fn find_mem_intervals_batch_gpu(
    ctx: &GpuContext,
    bidir: &BidirFmIndex,
    queries: &[&[u8]],
    min_len: usize,
    mode: u32,
) -> Result<Vec<Vec<RawMemInterval>>, FmIndexError> {
    let n = queries.len();
    let Some((flat, counts, offsets)) =
        run_mem_find_gpu(ctx, bidir, queries, min_len, mode).await?
    else {
        return Ok(vec![vec![]; n]);
    };
    let mut output: Vec<Vec<RawMemInterval>> = vec![vec![]; n];
    for (q, &count) in counts.iter().enumerate() {
        let off = offsets[q] as usize;
        let mut hits = Vec::with_capacity(count as usize);
        for k in 0..count as usize {
            let base = (off + k) * 4;
            hits.push(RawMemInterval {
                query_start: flat[base],
                query_end:   flat[base + 1],
                fwd_lo:      flat[base + 2],
                fwd_hi:      flat[base + 3],
            });
        }
        output[q] = hits;
    }
    Ok(output)
}

/// GPU batch SMEM finding. Returns `(query_start, query_end, match_count)` per MEM.
pub async fn find_smems_batch_gpu(
    ctx: &GpuContext,
    bidir: &BidirFmIndex,
    queries: &[&[u8]],
    min_len: usize,
) -> Result<Vec<Vec<(u32, u32, u32)>>, FmIndexError> {
    find_mems_batch_gpu(ctx, bidir, queries, min_len, MODE_SMEM).await
}

/// GPU batch MEM finding (all maximal matches). Returns `(query_start, query_end, match_count)`.
pub async fn find_all_mems_batch_gpu(
    ctx: &GpuContext,
    bidir: &BidirFmIndex,
    queries: &[&[u8]],
    min_len: usize,
) -> Result<Vec<Vec<(u32, u32, u32)>>, FmIndexError> {
    find_mems_batch_gpu(ctx, bidir, queries, min_len, MODE_MEM).await
}
