use crate::error::FmIndexError;
use crate::gpu::GpuContext;
use std::sync::{Arc, OnceLock};

static GPU_CONTEXT: OnceLock<Arc<GpuContext>> = OnceLock::new();

/// Return the process-wide cached `GpuContext`, initializing it on the first call.
///
/// Subsequent calls are essentially free (Arc clone). The ~220 ms wgpu adapter/device
/// initialization is paid exactly once per process.
pub fn get_or_init() -> Result<Arc<GpuContext>, FmIndexError> {
    if let Some(ctx) = GPU_CONTEXT.get() {
        return Ok(Arc::clone(ctx));
    }
    let ctx = Arc::new(pollster::block_on(GpuContext::new())?);
    // Two racing first callers: one wins the set, both return the same context.
    let _ = GPU_CONTEXT.set(Arc::clone(&ctx));
    Ok(Arc::clone(GPU_CONTEXT.get().unwrap()))
}
