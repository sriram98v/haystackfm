use super::prefix_sum::PrefixSumPipelines;
use super::GpuContext;

const SHADER_SOURCE: &str = include_str!("../../shaders/radix_sort.wgsl");
const WORKGROUP_SIZE: u32 = 256;
const ITEMS_PER_THREAD: u32 = 1;
const TILE_SIZE: u32 = WORKGROUP_SIZE * ITEMS_PER_THREAD; // 256
const RADIX: u32 = 256;

/// Parameters uniform for radix sort shaders.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    digit_shift: u32,
    num_workgroups: u32,
    _pad: u32,
}

/// Cached compute pipelines for radix sort.
pub struct RadixSortPipelines {
    count_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
}

impl RadixSortPipelines {
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            count_pipeline: ctx.create_compute_pipeline(
                "radix_sort_count",
                SHADER_SOURCE,
                "count_digits",
            ),
            scatter_pipeline: ctx.create_compute_pipeline(
                "radix_sort_scatter",
                SHADER_SOURCE,
                "scatter",
            ),
        }
    }

    /// Sort key-value pairs by key using LSD radix sort (4 passes for 32-bit keys).
    ///
    /// After this call, `keys_a` and `vals_a` contain the sorted result if the number
    /// of passes is even (4 passes → even → result in a/b depends on pass direction).
    /// We always ensure the final result is in `keys_a` / `vals_a` by copying if needed.
    ///
    /// Returns (keys_buffer, vals_buffer) containing the sorted data.
    #[allow(clippy::too_many_arguments)]
    pub fn sort(
        &self,
        ctx: &GpuContext,
        prefix_sum: &PrefixSumPipelines,
        keys_a: &wgpu::Buffer,
        vals_a: &wgpu::Buffer,
        keys_b: &wgpu::Buffer,
        vals_b: &wgpu::Buffer,
        n: u32,
    ) -> SortResult {
        if n <= 1 {
            return SortResult::InA;
        }

        let num_wg = GpuContext::workgroup_count(n, TILE_SIZE);
        let hist_size = RADIX * num_wg;

        let histogram_buf = ctx.create_buffer_empty("radix_histograms", hist_size);

        let mut in_a = true;

        // 4 passes: shift by 0, 8, 16, 24
        for pass in 0..4u32 {
            let digit_shift = pass * 8;
            let params = Params {
                n,
                digit_shift,
                num_workgroups: num_wg,
                _pad: 0,
            };
            let params_buf = ctx.create_uniform_buffer("radix_params", &params);

            let (keys_in, vals_in, keys_out, vals_out) = if in_a {
                (keys_a, vals_a, keys_b, vals_b)
            } else {
                (keys_b, vals_b, keys_a, vals_a)
            };

            // Zero the histogram buffer
            ctx.upload_to_buffer(&histogram_buf, &vec![0u32; hist_size as usize]);

            // Count pass
            let count_bg = ctx.create_bind_group(
                &self.count_pipeline,
                0,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: keys_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: histogram_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            );
            ctx.dispatch(&self.count_pipeline, &count_bg, (num_wg, 1, 1));

            // Prefix sum on histograms (makes them global offsets)
            prefix_sum.exclusive_prefix_sum(ctx, &histogram_buf, hist_size);

            // Scatter pass
            let scatter_bg = ctx.create_bind_group(
                &self.scatter_pipeline,
                0,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: keys_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vals_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keys_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: vals_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: histogram_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            );
            ctx.dispatch(&self.scatter_pipeline, &scatter_bg, (num_wg, 1, 1));

            in_a = !in_a;
        }

        // After 4 passes (even), result is back in the original (a) buffers
        if in_a {
            SortResult::InA
        } else {
            SortResult::InB
        }
    }
}

/// Indicates which buffer pair contains the sorted result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SortResult {
    InA,
    InB,
}
