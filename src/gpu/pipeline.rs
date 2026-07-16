use super::GpuContext;

impl GpuContext {
    /// Create a compute pipeline from WGSL source.
    pub fn create_compute_pipeline(
        &self,
        label: &str,
        shader_source: &str,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // Auto-layout from shader
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    /// Create a bind group from a pipeline's layout and buffer entries.
    pub fn create_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        group_index: u32,
        entries: &[wgpu::BindGroupEntry],
    ) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(group_index);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries,
        })
    }

    /// Dispatch a compute pass: encode a single pipeline dispatch into a command buffer and submit.
    pub fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dispatch_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bind_group), &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch multiple passes in a single command encoder submission.
    /// Each entry: (pipeline, bind_group, workgroups).
    pub fn dispatch_multi(
        &self,
        passes: &[(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32, u32))],
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("multi_dispatch_encoder"),
            });
        for &(pipeline, bind_group, workgroups) in passes {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bind_group), &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Compute the number of workgroups needed for n elements with given workgroup size.
    pub fn workgroup_count(n: u32, workgroup_size: u32) -> u32 {
        n.div_ceil(workgroup_size)
    }
}
