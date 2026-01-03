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

    // NEW: grid metadata buffer (storage, read)
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
        // UPDATED: 4 bindings now
        static B: [GPUBindingDesc; 4] = [
            GPUBindingDesc { read_only: false }, // sugar
            GPUBindingDesc { read_only: true  }, // capacity
            GPUBindingDesc { read_only: false }, // occupancy
            GPUBindingDesc { read_only: true  }, // grid_info (read-only storage)
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
        // NEW
        out.push(wgpu::BindGroupEntry {
            binding: base + 3,
            resource: self.grid_info.as_ref().unwrap().as_entire_binding(),
        });
        Ok(())
    }
}

pub struct AgentIntentBuffers {
    capacity: usize,
    agent_target: Option<wgpu::Buffer>,
}

impl AgentIntentBuffers {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            agent_target: None,
        }
    }
}

impl GPUResource for AgentIntentBuffers {
    fn name(&self) -> &'static str { "AgentIntentBuffers" }

    fn create_gpu(&mut self, ctx: &GPUContext) -> ECSResult<()> {
        let zeros = vec![0u32; self.capacity];
        self.agent_target = Some(
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("intent.agent_target"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }),
        );
        Ok(())
    }

    fn upload(&mut self, _: &GPUContext) -> ECSResult<()> { Ok(()) }
    fn download(&mut self, _: &GPUContext) -> ECSResult<()> { Ok(()) }

    fn bindings(&self) -> &[GPUBindingDesc] {
        static B: [GPUBindingDesc; 1] = [
            GPUBindingDesc { read_only: false },
        ];
        &B
    }

    fn encode_bind_group_entries<'a>(
        &'a self,
        base: u32,
        out: &mut Vec<wgpu::BindGroupEntry<'a>>,
    ) -> ECSResult<()> {
        out.push(wgpu::BindGroupEntry {
            binding: base,
            resource: self.agent_target.as_ref().unwrap().as_entire_binding(),
        });
        Ok(())
    }
}
