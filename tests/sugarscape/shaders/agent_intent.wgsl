// Params

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

// ECS component buffers (group 0)

@group(0) @binding(0) var<storage, read> vision: array<i32>;
@group(0) @binding(1) var<storage, read> pos: array<vec2<i32>>;
@group(0) @binding(2) var<storage, read> alive: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

// SugarGrid resource (group 1)

@group(1) @binding(0) var<storage, read_write> sugar: array<f32>;
@group(1) @binding(1) var<storage, read> capacity : array<f32>;
@group(1) @binding(2) var<storage, read_write> occupancy : array<atomic<u32>>;
@group(1) @binding(3) var<storage, read> grid : GridInfo;

// Intent buffers

@group(1) @binding(4) var<storage, read_write> agent_target : array<u32>;
@group(1) @binding(5) var<storage, read_write> agent_score : array<f32>;

const INVALID : u32 = 0xffffffffu;

// Kernel

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.entity_len || alive[i] == 0u) {
        return;
    }

    let w = i32(grid.w);
    let h = i32(grid.h);

    let p = pos[i];
    let v = max(vision[i], 0);

    agent_target[i] = INVALID;
    agent_score[i] = -1.0;

    if (p.x < 0 || p.y < 0 || p.x >= w || p.y >= h) {
        return;
    }

    var best_idx   : i32 = -1;
    var best_score : f32 = -1.0;

    for (var dx = -v; dx <= v; dx = dx + 1) {
        for (var dy = -v; dy <= v; dy = dy + 1) {
            let nx = p.x + dx;
            let ny = p.y + dy;

            if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
                continue;
            }

            let idx = ny * w + nx;
            let s = sugar[idx];

            if (s > best_score) {
                best_score = s;
                best_idx = idx;
            }
        }
    }

    if (best_idx >= 0) {
        agent_target[i] = u32(best_idx);
        agent_score[i]  = best_score;
    }
}
