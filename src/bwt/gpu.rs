use crate::bwt::Bwt;
use crate::gpu::GpuContext;
use crate::suffix_array::SuffixArray;

const SHADER: &str = include_str!("../../shaders/bwt_gather.wgsl");
const WG_SIZE: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
}

/// Cached pipeline for GPU BWT construction.
pub struct BwtPipelines {
    gather_pipeline: wgpu::ComputePipeline,
}

impl BwtPipelines {
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            gather_pipeline: ctx.create_compute_pipeline("bwt_gather", SHADER, "bwt_gather"),
        }
    }

    /// Build the BWT on the GPU: `BWT[i] = text[(SA[i] - 1 + n) % n]`.
    pub async fn build_bwt(&self, ctx: &GpuContext, text: &[u8], sa: &SuffixArray) -> Bwt {
        let n = text.len() as u32;

        let text_u32: Vec<u32> = text.iter().map(|&b| b as u32).collect();
        let text_buf = ctx.create_buffer_init("bwt_text", &text_u32);
        let sa_buf = ctx.create_buffer_init("bwt_sa", &sa.data);
        let bwt_buf = ctx.create_buffer_empty("bwt_out", n);

        let params = Params { n };
        let params_buf = ctx.create_uniform_buffer("bwt_params", &params);

        let bg = ctx.create_bind_group(
            &self.gather_pipeline,
            0,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: text_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sa_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bwt_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        );

        ctx.dispatch(
            &self.gather_pipeline,
            &bg,
            (GpuContext::workgroup_count(n, WG_SIZE), 1, 1),
        );

        let bwt_u32 = ctx.download_buffer(&bwt_buf, n).await;
        let data: Vec<u8> = bwt_u32.iter().map(|&v| v as u8).collect();
        Bwt::from_unpacked(data)
    }
}
