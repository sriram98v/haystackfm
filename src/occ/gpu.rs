use super::{OccTable, BLOCK_SIZE, SUPERBLOCK_SIZE};
use crate::alphabet::ALPHABET_SIZE;
use crate::bwt::Bwt;
use crate::gpu::GpuContext;

const SHADER: &str = include_str!("../../shaders/occ_scan.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    num_blocks: u32,
}

/// Cached pipeline for GPU Occ table construction.
pub struct OccPipelines {
    block_pipeline: wgpu::ComputePipeline,
}

impl OccPipelines {
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            block_pipeline: ctx.create_compute_pipeline("occ_block", SHADER, "occ_block"),
        }
    }

    /// Build the Occ table on the GPU.
    ///
    /// Each GPU workgroup processes one block of 64 BWT characters:
    ///   - Counts occurrences per character in the block
    ///   - Builds 64-bit presence bitvectors per character
    ///
    /// The CPU then prefix-sums block_counts to produce the checkpoint array.
    pub async fn build_occ_table(&self, ctx: &GpuContext, bwt: &Bwt) -> OccTable {
        let n = bwt.len() as u32;
        let num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let alpha = ALPHABET_SIZE as u32;

        // Upload BWT as u32 array
        let bwt_u32: Vec<u32> = bwt.to_u32_vec();
        let bwt_buf = ctx.create_buffer_init("occ_bwt", &bwt_u32);

        // Allocate output buffers
        // block_counts[num_blocks * ALPHA]: count of each char in each block
        let block_counts_buf = ctx.create_buffer_empty("occ_block_counts", num_blocks * alpha);
        // bitvectors[num_blocks * ALPHA * 2]: lo and hi u32 halves of each 64-bit bitvector
        let bitvectors_buf = ctx.create_buffer_empty("occ_bitvectors", num_blocks * alpha * 2);

        let params = Params { n, num_blocks };
        let params_buf = ctx.create_uniform_buffer("occ_params", &params);

        let bg = ctx.create_bind_group(
            &self.block_pipeline,
            0,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bwt_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: block_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bitvectors_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        );

        // One workgroup per block (workgroup_size = 64 = BLOCK_SIZE)
        ctx.dispatch(&self.block_pipeline, &bg, (num_blocks, 1, 1));

        // Download results
        let block_counts = ctx
            .download_buffer(&block_counts_buf, num_blocks * alpha)
            .await;
        let bitvec_flat = ctx
            .download_buffer(&bitvectors_buf, num_blocks * alpha * 2)
            .await;

        // Assemble two-level OccTable on CPU from GPU block counts and bitvectors.
        // block_counts layout: [block0_c0, block0_c1, ..., block0_cN, block1_c0, ...]
        // bitvec_flat layout:  [block0_c0_lo, block0_c0_hi, block0_c1_lo, block0_c1_hi, ...]
        let num_blocks_usize = num_blocks as usize;
        let num_superblocks = n.div_ceil(SUPERBLOCK_SIZE) as usize;
        let blocks_per_sb = (SUPERBLOCK_SIZE / BLOCK_SIZE) as usize;
        let alpha_usize = ALPHABET_SIZE;

        let mut superblock_checkpoints: Vec<[u32; ALPHABET_SIZE]> =
            Vec::with_capacity(num_superblocks);
        let mut block_deltas: Vec<[u16; ALPHABET_SIZE]> = Vec::with_capacity(num_blocks_usize);
        let mut bitvectors: Vec<[u64; ALPHABET_SIZE]> = Vec::with_capacity(num_blocks_usize);

        let mut cumulative = [0u32; ALPHABET_SIZE];
        let mut sb_base = [0u32; ALPHABET_SIZE];

        for b in 0..num_blocks_usize {
            if b % blocks_per_sb == 0 {
                superblock_checkpoints.push(cumulative);
                sb_base = cumulative;
            }

            let mut delta = [0u16; ALPHABET_SIZE];
            for c in 0..alpha_usize {
                delta[c] = (cumulative[c] - sb_base[c]) as u16;
            }
            block_deltas.push(delta);

            let mut bv = [0u64; ALPHABET_SIZE];
            for c in 0..alpha_usize {
                let lo = bitvec_flat[(b * alpha_usize + c) * 2];
                let hi = bitvec_flat[(b * alpha_usize + c) * 2 + 1];
                bv[c] = (hi as u64) << 32 | lo as u64;
                cumulative[c] += block_counts[b * alpha_usize + c];
            }
            bitvectors.push(bv);
        }

        OccTable::from_parts(superblock_checkpoints, block_deltas, bitvectors, n)
    }
}
