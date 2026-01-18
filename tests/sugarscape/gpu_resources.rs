#![cfg(feature = "gpu")]

use abm_framework::gpu::{GPUResource, GPUBindingDesc, GPUContext};
use abm_framework::engine::error::ECSResult;

use wgpu::util::DeviceExt;

pub struct SugarGrid {
    pub w: u32,
    pub h: u32,
    capacity_cpu: Vec<f32>,

    sugar: Option<wgpu::Buffer>,
    capacity: Option<wgpu::Buffer>,
    occupancy: Option<wgpu::Buffer>,
    grid_info: Option<wgpu::Buffer>,
}

impl SugarGrid {
    pub fn new(w: u32, h: u32, cap: Vec<f32>) -> Self {
        Self {
            w,
            h,
            capacity_cpu: cap,
            sugar: None,
            capacity: None,
            occupancy: None,
            grid_info: None,
        }
    }
}

impl GPUResource for SugarGrid {
    fn name(&self) -> &'static str { "SugarGrid" }

    fn create_gpu(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        let sugar = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sugar_grid.sugar"),
            contents: bytemuck::cast_slice(&self.capacity_cpu),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let capacity = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sugar_grid.capacity"),
            contents: bytemuck::cast_slice(&self.capacity_cpu),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let zeros = vec![0u32; (self.w * self.h) as usize];
        let occupancy = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sugar_grid.occupancy"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // NEW: {w, h, 0, 0} as a STORAGE read-only buffer
        let info = [self.w, self.h, 0u32, 0u32];
        let grid_info = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sugar_grid.grid_info"),
            contents: bytemuck::cast_slice(&info),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        self.sugar = Some(sugar);
        self.capacity = Some(capacity);
        self.occupancy = Some(occupancy);
        self.grid_info = Some(grid_info);

        Ok(())
    }

    fn upload(&mut self, _: &GPUContext) -> ECSResult<()> { Ok(()) }
    fn download(&mut self, _: &GPUContext) -> ECSResult<()> { Ok(()) }

    fn bindings(&self) -> &[GPUBindingDesc] {
        static B: [GPUBindingDesc; 4] = [
            GPUBindingDesc { read_only: false }, // sugar
            GPUBindingDesc { read_only: true  }, // capacity
            GPUBindingDesc { read_only: false }, // occupancy
            GPUBindingDesc { read_only: true  }, // grid_info
        ];
        &B
    }

    fn encode_bind_group_entries<'a>(
        &'a self,
        base: u32,
        out: &mut Vec<wgpu::BindGroupEntry<'a>>,
    ) -> ECSResult<()> {
        out.push(wgpu::BindGroupEntry {
            binding: base + 0,
            resource: self.sugar.as_ref().unwrap().as_entire_binding(),
        });
        out.push(wgpu::BindGroupEntry {
            binding: base + 1,
            resource: self.capacity.as_ref().unwrap().as_entire_binding(),
        });
        out.push(wgpu::BindGroupEntry {
            binding: base + 2,
            resource: self.occupancy.as_ref().unwrap().as_entire_binding(),
        });
        out.push(wgpu::BindGroupEntry {
            binding: base + 3,
            resource: self.grid_info.as_ref().unwrap().as_entire_binding(),
        });
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

pub struct AgentIntentBuffers {
    capacity: usize,

    // GPU
    agent_target: Option<wgpu::Buffer>,
    agent_score: Option<wgpu::Buffer>,

    // CPU mirrors
    pub target_cpu: Vec<u32>,
    pub score_cpu: Vec<f32>,
}

impl AgentIntentBuffers {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            agent_target: None,
            agent_score: None,
            target_cpu: vec![0xffffffffu32; capacity],
            score_cpu: vec![-1.0f32; capacity],
        }
    }

    #[inline]
    pub fn len(&self) -> usize { self.capacity }
}

fn read_buffer_u32(ctx: &GPUContext, buf: &wgpu::Buffer, out: &mut [u32]) -> ECSResult<()> {
    use std::sync::mpsc;
    let bytes_len = (out.len() * std::mem::size_of::<u32>()) as u64;

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_u32"),
        size: bytes_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder_u32"),
    });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes_len);
    let submission = ctx.queue.submit(Some(encoder.finish()));

    ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None })
        .map_err(|e| abm_framework::engine::error::ECSError::from(
            abm_framework::engine::error::ExecutionError::GpuDispatchFailed {
                message: format!("poll failed: {e:?}").into(),
            }
        ))?;

    let slice = staging.slice(0..bytes_len);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();

    rx.recv()
    .unwrap()
    .map_err(|e| {
        abm_framework::engine::error::ECSError::from(
            abm_framework::engine::error::ExecutionError::GpuDispatchFailed {
                message: format!("map_async (u32) failed: {e:?}").into(),
            },
        )
    })?;

    let data = slice.get_mapped_range();
    let src: &[u8] = &data;
    let words: &[u32] = bytemuck::cast_slice(src);
    out.copy_from_slice(&words[..out.len()]);
    drop(data);
    staging.unmap();
    Ok(())
}

fn read_buffer_f32(ctx: &GPUContext, buf: &wgpu::Buffer, out: &mut [f32]) -> ECSResult<()> {
    // identical to read_buffer_u32 except types
    use std::sync::mpsc;
    let bytes_len = (out.len() * std::mem::size_of::<f32>()) as u64;

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_f32"),
        size: bytes_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder_f32"),
    });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes_len);
    let submission = ctx.queue.submit(Some(encoder.finish()));

    ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None })
        .map_err(|e| abm_framework::engine::error::ECSError::from(
            abm_framework::engine::error::ExecutionError::GpuDispatchFailed {
                message: format!("poll failed: {e:?}").into(),
            }
        ))?;

    let slice = staging.slice(0..bytes_len);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();

    rx.recv()
    .unwrap()
    .map_err(|e| {
        abm_framework::engine::error::ECSError::from(
            abm_framework::engine::error::ExecutionError::GpuDispatchFailed {
                message: format!("map_async (u32) failed: {e:?}").into(),
            },
        )
    })?;

    let data = slice.get_mapped_range();
    let src: &[u8] = &data;
    let words: &[f32] = bytemuck::cast_slice(src);
    out.copy_from_slice(&words[..out.len()]);
    drop(data);
    staging.unmap();
    Ok(())
}

fn write_buffer_u32(ctx: &GPUContext, buf: &wgpu::Buffer, data: &[u32]) {
    ctx.queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
}

impl GPUResource for AgentIntentBuffers {
    fn name(&self) -> &'static str { "AgentIntentBuffers" }

    fn create_gpu(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        let zeros_u32 = vec![0xffffffffu32; self.capacity];
        let zeros_f32 = vec![-1.0f32; self.capacity];

        self.agent_target = Some(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("intent.agent_target"),
            contents: bytemuck::cast_slice(&zeros_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        }));

        self.agent_score = Some(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("intent.agent_score"),
            contents: bytemuck::cast_slice(&zeros_f32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        }));

        Ok(())
    }

    fn upload(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        let tgt = self.agent_target.as_ref().unwrap();
        write_buffer_u32(ctx, tgt, &self.target_cpu);
        Ok(())
    }

    fn download(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        read_buffer_u32(ctx, self.agent_target.as_ref().unwrap(), &mut self.target_cpu)?;
        read_buffer_f32(ctx, self.agent_score.as_ref().unwrap(), &mut self.score_cpu)?;
        Ok(())
    }

    fn bindings(&self) -> &[GPUBindingDesc] {
        static B: [GPUBindingDesc; 2] = [
            GPUBindingDesc { read_only: false }, // target
            GPUBindingDesc { read_only: false }, // score
        ];
        &B
    }

    fn encode_bind_group_entries<'a>(&'a self, base: u32, out: &mut Vec<wgpu::BindGroupEntry<'a>>) -> ECSResult<()> {
        out.push(wgpu::BindGroupEntry {
            binding: base + 0,
            resource: self.agent_target.as_ref().unwrap().as_entire_binding(),
        });
        out.push(wgpu::BindGroupEntry {
            binding: base + 1,
            resource: self.agent_score.as_ref().unwrap().as_entire_binding(),
        });
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
