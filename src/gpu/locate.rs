use crate::alphabet::ALPHABET_SIZE;
use crate::error::FmIndexError;
use crate::fm_index::FmIndex;
use crate::gpu::GpuContext;
use wgpu;

const LOCATE_SEARCH_SHADER: &str = include_str!("../../shaders/locate_search.wgsl");
const LOCATE_RESOLVE_SHADER: &str = include_str!("../../shaders/locate_resolve.wgsl");

// Uniform struct for locate_search.wgsl.
// 20 × u32 = 80 bytes (multiple of 16, satisfies WGSL uniform alignment).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SearchParams {
    num_queries: u32,
    text_len: u32,
    num_blocks: u32,
    _pad: u32,
    // C-array values embedded to stay within max_storage_buffers_per_shader_stage=8
    c: [u32; 16],
}

// Uniform struct for locate_resolve.wgsl.
// 24 × u32 = 96 bytes (multiple of 16, satisfies WGSL uniform alignment).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveParams {
    num_queries: u32,
    text_len: u32,
    num_blocks: u32,
    num_seqs: u32,
    sample_rate: u32,
    total_matches: u32,
    // C-array values embedded to stay within max_storage_buffers_per_shader_stage=8
    c: [u32; 16],
    _pad2: [u32; 2],
}

/// GPU-accelerated batch locate (2-pass pipeline).
///
/// **Pass 1** (`locate_search.wgsl`): each thread runs backward search over its
/// query. IUPAC ambiguity codes expand into up to `MAX_IVS` parallel SA intervals
/// via the WGSL `COMPAT` table. Emits a flat interval buffer and per-query match
/// counts/offsets.
///
/// **Pass 2** (`locate_resolve.wgsl`): each thread walks one match position via
/// LF-mapping to the nearest sampled SA entry, then maps the text offset to a
/// reference sequence and within-sequence position.
///
/// Returns one `Vec<(seq_idx, pos_in_seq)>` per query.
/// `seq_idx` is a 0-based index into the original sequence list.
pub async fn locate_batch_gpu(
    ctx: &GpuContext,
    index: &FmIndex,
    queries: &[&[u8]],
) -> Result<Vec<Vec<(u32, u32)>>, FmIndexError> {
    if queries.is_empty() {
        return Ok(vec![]);
    }

    let num_queries = queries.len() as u32;
    let text_len = index.text_len;
    let block_size: u32 = 64;
    let num_blocks = (text_len + block_size - 1) / block_size;
    let num_seqs = index.num_sequences;
    let sample_rate = index.sa_samples.sample_rate;
    let alpha = ALPHABET_SIZE as u32;

    let c_arr: [u32; 16] = index.c_array.data;

    // Flatten Occ checkpoints: [block * ALPHA + c]
    let mut checkpoints_flat: Vec<u32> = Vec::with_capacity((num_blocks * alpha) as usize);
    for block in &index.occ.checkpoints {
        checkpoints_flat.extend_from_slice(block);
    }

    // Flatten Occ bitvectors: split each u64 into (lo_u32, hi_u32)
    let mut bitvectors_flat: Vec<u32> = Vec::with_capacity((num_blocks * alpha * 2) as usize);
    for block in &index.occ.bitvectors {
        for &bv64 in block.iter() {
            bitvectors_flat.push(bv64 as u32);
            bitvectors_flat.push((bv64 >> 32) as u32);
        }
    }

    // Encode queries: flat bytes + per-query offsets
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

    let bwt_u32: Vec<u32> = index.bwt.data.iter().map(|&b| b as u32).collect();
    let sa_samples_data: Vec<u32> = index.sa_samples.samples.clone();
    let seq_bounds_data: Vec<u32> = index.seq_boundaries.clone();

    let bwt_buf = ctx.create_buffer_init("locate_bwt", &bwt_u32);
    let chk_buf = ctx.create_buffer_init("locate_chk", &checkpoints_flat);
    let bv_buf = ctx.create_buffer_init("locate_bv", &bitvectors_flat);
    let sa_buf = ctx.create_buffer_init("locate_sa", &sa_samples_data);
    let seqb_buf = ctx.create_buffer_init("locate_seqb", &seq_bounds_data);
    let qflat_buf = ctx.create_buffer_init("locate_qflat", &queries_flat);
    let qoff_buf = ctx.create_buffer_init("locate_qoff", &query_offsets);
    let intervals_buf = ctx.create_buffer_empty("locate_intervals", num_queries * 16 * 2);

    // ── Phase A: backward search (one thread per query) ─────────────────────
    let search_params = SearchParams {
        num_queries,
        text_len,
        num_blocks,
        _pad: 0,
        c: c_arr,
    };
    let search_params_buf = ctx.create_uniform_buffer("locate_search_params", &search_params);

    let search_pipeline =
        ctx.create_compute_pipeline("locate_search", LOCATE_SEARCH_SHADER, "locate_search");
    let search_bg = ctx.create_bind_group(
        &search_pipeline,
        0,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: qflat_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: qoff_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: chk_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bv_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: intervals_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: search_params_buf.as_entire_binding(),
            },
        ],
    );

    let wg_size: u32 = 64;
    ctx.dispatch(
        &search_pipeline,
        &search_bg,
        ((num_queries + wg_size - 1) / wg_size, 1, 1),
    );

    let intervals = ctx.download_buffer(&intervals_buf, num_queries * 16 * 2).await;

    // Compute per-query match counts: sum (hi-lo) over up to 16 interval slots.
    let match_counts: Vec<u32> = (0..num_queries as usize)
        .map(|q| {
            (0..16usize)
                .map(|i| {
                    let lo = intervals[q * 32 + i * 2];
                    let hi = intervals[q * 32 + i * 2 + 1];
                    hi.saturating_sub(lo)
                })
                .sum()
        })
        .collect();

    let total_matches: u32 = match_counts.iter().sum();

    if total_matches == 0 {
        return Ok(vec![vec![]; queries.len()]);
    }

    // Exclusive prefix sum → match_offsets (length num_queries + 1)
    let mut match_offsets: Vec<u32> = Vec::with_capacity(queries.len() + 1);
    match_offsets.push(0);
    for &c in &match_counts {
        let prev = *match_offsets.last().unwrap();
        match_offsets.push(prev + c);
    }

    // ── Phase B: resolve SA positions (one thread per match) ────────────────
    let results_buf = ctx.create_buffer_empty("locate_results", total_matches * 2);
    let match_off_buf = ctx.create_buffer_init("locate_match_offsets", &match_offsets);

    let resolve_params = ResolveParams {
        num_queries,
        text_len,
        num_blocks,
        num_seqs,
        sample_rate,
        total_matches,
        c: c_arr,
        _pad2: [0, 0],
    };
    let resolve_params_buf = ctx.create_uniform_buffer("locate_resolve_params", &resolve_params);

    let resolve_pipeline =
        ctx.create_compute_pipeline("locate_resolve", LOCATE_RESOLVE_SHADER, "locate_resolve");
    let resolve_bg = ctx.create_bind_group(
        &resolve_pipeline,
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
                resource: seqb_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: intervals_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: match_off_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: results_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: resolve_params_buf.as_entire_binding(),
            },
        ],
    );

    ctx.dispatch(
        &resolve_pipeline,
        &resolve_bg,
        ((total_matches + wg_size - 1) / wg_size, 1, 1),
    );

    let results_flat = ctx.download_buffer(&results_buf, total_matches * 2).await;

    // Assemble per-query results
    let mut output: Vec<Vec<(u32, u32)>> = vec![vec![]; queries.len()];
    for (q, &count) in match_counts.iter().enumerate() {
        let off = match_offsets[q] as usize;
        let mut hits = Vec::with_capacity(count as usize);
        for k in 0..count as usize {
            let seq_id = results_flat[(off + k) * 2];
            let pos = results_flat[(off + k) * 2 + 1];
            hits.push((seq_id, pos));
        }
        output[q] = hits;
    }

    Ok(output)
}
