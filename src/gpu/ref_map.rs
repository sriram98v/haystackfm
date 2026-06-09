use crate::error::FmIndexError;
use crate::gpu::GpuContext;
use wgpu;

const REF_MAP_SHADER: &str = include_str!("../../shaders/ref_map.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RefMapParams {
    total_pos: u32,
    num_refs: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Maps flat raw text positions to (ref_id, offset_within_ref) pairs.
///
/// `boundaries[i]` = exclusive end position of reference `i` in the
/// concatenated text. Length = number of references.
///
/// Returns `(ref_ids, offsets)` — parallel arrays, one entry per input position.
/// Positions with value `u32::MAX` (SA walk-back failure) yield `(u32::MAX, u32::MAX)`.
pub(crate) async fn map_positions_to_refs(
    ctx: &GpuContext,
    positions: &[u32],
    boundaries: &[u32],
) -> Result<(Vec<u32>, Vec<u32>), FmIndexError> {
    if positions.is_empty() {
        return Ok((vec![], vec![]));
    }

    let total_pos = positions.len() as u32;
    let num_refs = boundaries.len() as u32;

    let pos_buf = ctx.create_buffer_init("rm_pos", positions);
    let bnd_buf = ctx.create_buffer_init("rm_bnd", boundaries);
    let rid_buf = ctx.create_buffer_empty("rm_rid", total_pos);
    let off_buf = ctx.create_buffer_empty("rm_off", total_pos);

    let params = RefMapParams {
        total_pos,
        num_refs,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = ctx.create_uniform_buffer("rm_params", &params);

    let pipeline = ctx.create_compute_pipeline("ref_map", REF_MAP_SHADER, "map_positions");
    let bg = ctx.create_bind_group(
        &pipeline,
        0,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pos_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bnd_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rid_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: off_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    );

    let wg_size: u32 = 64;
    ctx.dispatch(&pipeline, &bg, ((total_pos + wg_size - 1) / wg_size, 1, 1));

    let ref_ids = ctx.download_buffer(&rid_buf, total_pos).await;
    let offsets = ctx.download_buffer(&off_buf, total_pos).await;
    Ok((ref_ids, offsets))
}
