// GPU reference boundary mapping.
//
// One thread per raw text position.
// Each thread binary-searches `boundaries` (cumulative reference end positions)
// to find (ref_id, offset_within_ref) for its position.
//
// `boundaries[i]` = exclusive end of reference i in the concatenated text.
// So reference i occupies [boundaries[i-1], boundaries[i]) with boundaries[-1] = 0.
//
// Binding layout (4 storage + 1 uniform):
//   0: positions_in  — raw text positions [total_pos]
//   1: boundaries    — cumulative ref end positions [num_refs]
//   2: ref_ids_out   — output ref_id per position [total_pos]
//   3: offsets_out   — output offset_within_ref per position [total_pos]
//   4: params        — uniform

struct Params {
    total_pos: u32,
    num_refs:  u32,
    _pad0:     u32,
    _pad1:     u32,
}
// 4 × u32 = 16 bytes.

@group(0) @binding(0) var<storage, read>       positions_in: array<u32>;
@group(0) @binding(1) var<storage, read>       boundaries:   array<u32>;
@group(0) @binding(2) var<storage, read_write> ref_ids_out:  array<u32>;
@group(0) @binding(3) var<storage, read_write> offsets_out:  array<u32>;
@group(0) @binding(4) var<uniform>             params:        Params;

@compute @workgroup_size(64)
fn map_positions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if tid >= params.total_pos { return; }

    let pos = positions_in[tid];

    // Binary search: find first ref index where boundaries[ref_id] > pos.
    var lo = 0u;
    var hi = params.num_refs;
    loop {
        if lo >= hi { break; }
        let mid = (lo + hi) / 2u;
        if boundaries[mid] <= pos {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let ref_id = lo;

    var ref_start = 0u;
    if ref_id > 0u {
        ref_start = boundaries[ref_id - 1u];
    }

    ref_ids_out[tid] = ref_id;
    offsets_out[tid] = pos - ref_start;
}
