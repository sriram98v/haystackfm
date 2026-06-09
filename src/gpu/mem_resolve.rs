use crate::alphabet::ALPHABET_SIZE;
use crate::error::FmIndexError;
use crate::fm_index::FmIndex;
use crate::gpu::mem_find::RawMemInterval;
use crate::gpu::GpuContext;
use wgpu;

const MEM_RESOLVE_SHADER: &str = include_str!("../../shaders/mem_resolve.wgsl");

// 16 × u32 = 64 bytes (multiple of 16 — satisfies WGSL uniform alignment).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MemResolveParams {
    num_mems: u32,
    text_len: u32,
    num_blocks: u32,
    sample_rate: u32,
    total_pos: u32,
    _pad0: u32,
    c: [u32; 6],
    _pad1: [u32; 2],
}

/// Resolves raw SA positions for a flat list of MEM intervals.
///
/// Returns `None` when all intervals are empty (total_positions == 0).
/// Otherwise returns `(positions_flat, position_offsets)`:
///   - `positions_flat[i]` is the raw text position for the i-th match across all MEMs
///   - `position_offsets[m]` is the start index in `positions_flat` for MEM `m`
///   - `position_offsets[m+1] - position_offsets[m]` = number of hits for MEM `m`
///
/// Positions are capped at `max_hits_per_mem` per MEM interval to prevent
/// output buffer explosion on repetitive references.
pub(crate) async fn resolve_mem_intervals_gpu(
    ctx: &GpuContext,
    index: &FmIndex,
    intervals: &[RawMemInterval],
    max_hits_per_mem: u32,
) -> Result<Option<(Vec<u32>, Vec<u32>)>, FmIndexError> {
    if intervals.is_empty() {
        return Ok(None);
    }

    let num_mems = intervals.len() as u32;
    let block_size: u32 = 64;
    let alpha = ALPHABET_SIZE as u32;
    let text_len = index.text_len;
    let num_blocks = (text_len + block_size - 1) / block_size;
    let sample_rate = index.sa_samples.sample_rate;
    let c_arr: [u32; 6] = index.c_array.data;

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

    // Pre-flight: check SA buffer fits within device limits.
    let sa_bytes = (index.sa_samples.samples.len() as u64) * 4;
    let max_binding = ctx.device.limits().max_storage_buffer_binding_size as u64;
    if sa_bytes > max_binding {
        return Err(FmIndexError::GpuError(format!(
            "SA buffer ({sa_bytes} bytes) exceeds device max_storage_buffer_binding_size ({max_binding} bytes)"
        )));
    }

    // Build flat [fwd_lo, fwd_hi] intervals buffer (using capped fwd_hi).
    let mut intervals_flat: Vec<u32> = Vec::with_capacity(intervals.len() * 2);
    for (i, iv) in intervals.iter().enumerate() {
        let count = position_offsets[i + 1] - position_offsets[i];
        intervals_flat.push(iv.fwd_lo);
        intervals_flat.push(iv.fwd_lo + count); // capped hi
    }

    // Flatten Occ checkpoints and bitvectors (fwd index only).
    let mut checkpoints_flat: Vec<u32> = Vec::with_capacity((num_blocks * alpha) as usize);
    for block in &index.occ.checkpoints {
        checkpoints_flat.extend_from_slice(block);
    }
    let mut bitvectors_flat: Vec<u32> = Vec::with_capacity((num_blocks * alpha * 2) as usize);
    for block in &index.occ.bitvectors {
        for &bv64 in block.iter() {
            bitvectors_flat.push(bv64 as u32);
            bitvectors_flat.push((bv64 >> 32) as u32);
        }
    }

    let bwt_u32: Vec<u32> = index.bwt.data.iter().map(|&b| b as u32).collect();

    let bwt_buf = ctx.create_buffer_init("mr_bwt", &bwt_u32);
    let chk_buf = ctx.create_buffer_init("mr_chk", &checkpoints_flat);
    let bv_buf = ctx.create_buffer_init("mr_bv", &bitvectors_flat);
    let sa_buf = ctx.create_buffer_init("mr_sa", &index.sa_samples.samples);
    let ivs_buf = ctx.create_buffer_init("mr_ivs", &intervals_flat);
    let offsets_buf = ctx.create_buffer_init("mr_offsets", &position_offsets);
    let pos_out_buf = ctx.create_buffer_empty("mr_pos_out", total_pos);

    let params = MemResolveParams {
        num_mems,
        text_len,
        num_blocks,
        sample_rate,
        total_pos,
        _pad0: 0,
        c: c_arr,
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
                resource: bwt_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: chk_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: bv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sa_buf.as_entire_binding(),
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
    ctx.dispatch(&pipeline, &bg, ((total_pos + wg_size - 1) / wg_size, 1, 1));

    let positions_flat = ctx.download_buffer(&pos_out_buf, total_pos).await;
    Ok(Some((positions_flat, position_offsets)))
}
