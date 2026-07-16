use crate::alphabet::ALPHABET_SIZE;
use crate::error::FmIndexError;
use crate::fm_index::FmIndex;
use crate::gpu::mem_find::RawMemInterval;
use crate::gpu::GpuContext;
use wgpu;

const MEM_RESOLVE_SHADER: &str = include_str!("../../shaders/mem_resolve.wgsl");

// 24 × u32 = 96 bytes (multiple of 16 — satisfies WGSL uniform alignment).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MemResolveParams {
    num_mems: u32,
    text_len: u32,
    num_blocks: u32,
    sample_rate: u32,
    total_pos: u32,
    _pad0: u32,
    c: [u32; 16],
    _pad1: [u32; 2],
}

// ── Cached resolve index buffers ─────────────────────────────────────────────

/// GPU buffers for the resolve-pass index (BWT, checkpoints, bitvectors, SA).
///
/// Build once per `FmIndex` and reuse across resolve batches.
pub(crate) struct ResolveIndexBuffers {
    pub bwt_buf: wgpu::Buffer,
    pub chk_buf: wgpu::Buffer,
    pub bv_buf: wgpu::Buffer,
    pub sa_buf: wgpu::Buffer,
    pub text_len: u32,
    pub num_blocks: u32,
    pub sample_rate: u32,
    pub c: [u32; 16],
}

impl ResolveIndexBuffers {
    /// Upload BWT, checkpoints, bitvectors, and SA to the GPU.
    ///
    /// Returns `Err` if SA buffer exceeds `max_storage_buffer_binding_size`.
    pub fn new(ctx: &GpuContext, index: &FmIndex) -> Result<Self, FmIndexError> {
        let sa_bytes = (index.text_len as u64) * 4;
        let max_binding = ctx.device.limits().max_storage_buffer_binding_size as u64;
        if sa_bytes > max_binding {
            return Err(FmIndexError::GpuError(format!(
                "SA buffer ({sa_bytes} bytes) exceeds device max_storage_buffer_binding_size ({max_binding} bytes)"
            )));
        }

        let block_size: u32 = 64;
        let alpha = ALPHABET_SIZE as u32;
        let text_len = index.text_len;
        let num_blocks = text_len.div_ceil(block_size);

        let mut checkpoints_flat: Vec<u32> = Vec::with_capacity((num_blocks * alpha) as usize);
        for block in index.occ.flat_block_checkpoints() {
            checkpoints_flat.extend_from_slice(&block);
        }

        let mut bitvectors_flat: Vec<u32> = Vec::with_capacity((num_blocks * alpha * 2) as usize);
        for block in &index.occ.bitvectors_full16() {
            for &bv64 in block.iter() {
                bitvectors_flat.push(bv64 as u32);
                bitvectors_flat.push((bv64 >> 32) as u32);
            }
        }

        let bwt_u32 = index.occ.reconstruct_bwt_u32();
        let sa_flat = index.sa_samples.to_flat_vec(index.text_len as usize);

        Ok(Self {
            bwt_buf: ctx.create_buffer_init("mr_bwt", &bwt_u32),
            chk_buf: ctx.create_buffer_init("mr_chk", &checkpoints_flat),
            bv_buf: ctx.create_buffer_init("mr_bv", &bitvectors_flat),
            sa_buf: ctx.create_buffer_init("mr_sa", &sa_flat),
            text_len,
            num_blocks,
            sample_rate: index.sa_samples.sample_rate,
            c: index.c_array.data,
        })
    }
}

// ── Batched resolve dispatch ──────────────────────────────────────────────────

/// Resolve a pre-sized batch of intervals using cached index buffers.
///
/// `intervals_flat` is stride-2: `[fwd_lo, fwd_hi]` per interval (fwd_hi already
/// capped by the caller). `position_offsets` is the exclusive prefix-sum (len =
/// intervals_flat.len()/2 + 1). Returns `positions_flat` of length `total_pos`.
pub(crate) async fn resolve_intervals_batch(
    ctx: &GpuContext,
    idx: &ResolveIndexBuffers,
    intervals_flat: &[u32],
    position_offsets: &[u32],
    total_pos: u32,
) -> Vec<u32> {
    let num_mems = (intervals_flat.len() / 2) as u32;

    let ivs_buf = ctx.create_buffer_init("mr_ivs", intervals_flat);
    let offsets_buf = ctx.create_buffer_init("mr_offsets", position_offsets);
    let pos_out_buf = ctx.create_buffer_empty("mr_pos_out", total_pos);

    let params = MemResolveParams {
        num_mems,
        text_len: idx.text_len,
        num_blocks: idx.num_blocks,
        sample_rate: idx.sample_rate,
        total_pos,
        _pad0: 0,
        c: idx.c,
        _pad1: [0, 0],
    };
    let params_buf = ctx.create_uniform_buffer("mr_params", &params);

    let pipeline =
        ctx.create_compute_pipeline("mem_resolve", MEM_RESOLVE_SHADER, "resolve_positions");
    let bg = ctx.create_bind_group(
        &pipeline,
        0,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: idx.bwt_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: idx.chk_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: idx.bv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: idx.sa_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: ivs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: offsets_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: pos_out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: params_buf.as_entire_binding(),
            },
        ],
    );

    let wg_size: u32 = 64;
    ctx.dispatch(&pipeline, &bg, (total_pos.div_ceil(wg_size), 1, 1));

    ctx.download_buffer(&pos_out_buf, total_pos).await
}

// ── Legacy single-shot wrapper ────────────────────────────────────────────────

/// Resolves raw SA positions for a flat list of MEM intervals (single-shot wrapper).
///
/// Returns `None` when all intervals are empty (total_positions == 0).
/// Otherwise returns `(positions_flat, position_offsets)`:
///   - `positions_flat[i]` is the raw text position for the i-th match across all MEMs
///   - `position_offsets[m]` is the start index in `positions_flat` for MEM `m`
///   - `position_offsets[m+1] - position_offsets[m]` = number of hits for MEM `m`
///
/// Positions are capped at `max_hits_per_mem` per MEM interval to prevent
/// output buffer explosion on repetitive references.
#[allow(dead_code)] // per-call variant; resident path is used instead. kept for reference/reuse.
pub(crate) async fn resolve_mem_intervals_gpu(
    ctx: &GpuContext,
    index: &FmIndex,
    intervals: &[RawMemInterval],
    max_hits_per_mem: u32,
) -> Result<Option<(Vec<u32>, Vec<u32>)>, FmIndexError> {
    if intervals.is_empty() {
        return Ok(None);
    }

    let idx = ResolveIndexBuffers::new(ctx, index)?;

    // Compute per-MEM hit counts (capped) and prefix sums.
    let mut position_offsets: Vec<u32> = Vec::with_capacity(intervals.len() + 1);
    position_offsets.push(0);
    for iv in intervals {
        let raw_count = iv.fwd_hi.saturating_sub(iv.fwd_lo);
        let capped = raw_count.min(max_hits_per_mem);
        let prev = *position_offsets.last().unwrap();
        position_offsets.push(prev + capped);
    }
    let total_pos: u32 = *position_offsets.last().unwrap();
    if total_pos == 0 {
        return Ok(None);
    }

    // Build flat [fwd_lo, fwd_hi] intervals buffer (using capped fwd_hi).
    let mut intervals_flat: Vec<u32> = Vec::with_capacity(intervals.len() * 2);
    for (i, iv) in intervals.iter().enumerate() {
        let count = position_offsets[i + 1] - position_offsets[i];
        intervals_flat.push(iv.fwd_lo);
        intervals_flat.push(iv.fwd_lo + count); // capped hi
    }

    let positions_flat =
        resolve_intervals_batch(ctx, &idx, &intervals_flat, &position_offsets, total_pos).await;
    Ok(Some((positions_flat, position_offsets)))
}
