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

@group(0) @binding(0) var<uniform> params : Params;

// SugarGrid resource layout:
@group(1) @binding(2) var<storage, read_write> occupancy : array<atomic<u32>>;

// NEW
@group(1) @binding(3) var<storage, read> grid : GridInfo;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = grid.w * grid.h;
    if (idx >= n) { return; }
    atomicStore(&occupancy[idx], 0u);
}
