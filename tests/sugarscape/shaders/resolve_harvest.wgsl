struct Params {
    entity_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct GridInfo {
    w: u32,
    h: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> alive : array<u32>;
@group(0) @binding(1) var<storage, read_write> pos : array<vec2<i32>>;
@group(0) @binding(2) var<storage, read_write> sugar_agent : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

// SugarGrid resource layout:
@group(1) @binding(0) var<storage, read_write> sugar_grid : array<f32>;
@group(1) @binding(1) var<storage, read> capacity : array<f32>;
@group(1) @binding(2) var<storage, read_write> occupancy : array<atomic<u32>>;

// NEW
@group(1) @binding(3) var<storage, read> grid : GridInfo;

// Intent shifts
@group(1) @binding(4) var<storage, read_write> agent_target : array<u32>;

const INVALID : u32 = 0xffffffffu;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_len || alive[i] == 0u) {
        return;
    }

    let tgt_u = agent_target[i];
    if (tgt_u == INVALID) {
        return;
    }

    let w = i32(grid.w);
    let h = i32(grid.h);

    let idx = i32(tgt_u);
    let n = w * h;
    if (idx < 0 || idx >= n) {
        return;
    }

    let x = idx % w;
    let y = idx / w;

    pos[i] = vec2<i32>(x, y);

    let harvested = sugar_grid[idx];
    sugar_grid[idx] = 0.0;
    sugar_agent[i] = sugar_agent[i] + harvested;
}
