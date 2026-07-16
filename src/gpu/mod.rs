pub mod buffers;
pub mod context_cache;
pub mod locate;
pub mod mem_find;
pub mod mem_hit;
pub mod mem_resolve;
pub mod ref_map;
pub use mem_find::RawMemInterval;
pub(crate) use mem_find::{find_mem_intervals_for_batch, FindIndexBuffers};
pub use mem_hit::MemHit;
pub(crate) use mem_resolve::{resolve_intervals_batch, ResolveIndexBuffers};
pub mod pipeline;
pub mod prefix_sum;
pub mod radix_sort;

use crate::error::FmIndexError;

/// GPU context: owns the adapter, device, and queue.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a new GPU context. Requests an adapter and device with required limits.
    pub async fn new() -> Result<Self, FmIndexError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| FmIndexError::GpuError("no suitable GPU adapter found".into()))?;

        let mut required_limits = wgpu::Limits::default();
        // Request the maximum the adapter supports — large SA buffers (up to ~400MB
        // for 100M-base combined references) require raising both limits beyond the
        // wgpu defaults of 256MB / 128MB.
        let adapter_limits = adapter.limits();
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;
        required_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_compute_workgroups_per_dimension =
            adapter_limits.max_compute_workgroups_per_dimension;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("haystackfm"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| FmIndexError::GpuError(format!("failed to request device: {e}")))?;

        Ok(Self { device, queue })
    }

    /// Maximum number of u32 elements that fit in a single storage buffer.
    pub fn max_buffer_elements(&self) -> u32 {
        (self.device.limits().max_buffer_size / 4) as u32
    }

    /// Safe per-batch u32 budget for a single output storage binding.
    ///
    /// Uses 90% of `max_storage_buffer_binding_size` to leave headroom for
    /// alignment padding and other bookkeeping.  All auto-sized batches cap
    /// their output buffers at this value.
    pub fn output_budget_u32(&self) -> u32 {
        let binding = self.device.limits().max_storage_buffer_binding_size as u64;
        let elems = binding / 4; // bytes → u32 slots
        (elems * 9 / 10).min(u32::MAX as u64) as u32
    }
}
