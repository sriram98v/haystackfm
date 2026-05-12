pub mod buffers;
pub mod locate;
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
        // Request generous buffer sizes for large genomic data
        let adapter_limits = adapter.limits();
        required_limits.max_buffer_size = adapter_limits.max_buffer_size.min(256 * 1024 * 1024);
        required_limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .min(128 * 1024 * 1024);
        required_limits.max_compute_workgroups_per_dimension =
            adapter_limits.max_compute_workgroups_per_dimension;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("webgpu-fmidx"),
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
}
