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

@group(0) @binding(0) var<storage, read> vision : array<i32>;
@group(0) @binding(1) var<storage, read> pos    : array<vec2<i32>>;
@group(0) @binding(2) var<storage, read> alive  : array<u32>;
@group(0) @binding(3) var<uniform> params : Params;

// SugarGrid resource layout
@group(1) @binding(0) var<storage, read_write> sugar     : array<f32>;
@group(1) @binding(1) var<storage, read>       capacity : array<f32>;
@group(1) @binding(2) var<storage, read_write> occupancy : array<atomic<u32>>;
@group(1) @binding(3) var<storage, read>        grid     : GridInfo;

// Intent buffer
@group(1) @binding(4) var<storage, read_write> agent_target : array<u32>;

const INVALID : u32 = 0xffffffffu;

//
// Deterministic hash for tie-breaking
//
fn hash_u32(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}

fn jitter(agent: u32, cell: u32) -> f32 {
    let h = hash_u32(agent ^ (cell * 747796405u));
    return f32(h & 0x00ffffffu) / 16777216.0; // [0,1)
}

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

    if (p.x < 0 || p.y < 0 || p.x >= w || p.y >= h) {
        return;
    }

    var best_idx   : i32 = -1;
    var best_score : f32 = -1.0;

    // Vision scan
    for (var dx = -v; dx <= v; dx = dx + 1) {
        for (var dy = -v; dy <= v; dy = dy + 1) {
            let nx = p.x + dx;
            let ny = p.y + dy;

            if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
                continue;
            }

            let idx_i = ny * w + nx;
            let idx   = u32(idx_i);

            // Skip already claimed cells
            if (atomicLoad(&occupancy[idx]) != 0u) {
                continue;
            }

            let s = sugar[idx];

            // CPU-equivalent scoring with tiny deterministic tie-break
            let score = s + 0.001 * jitter(i, idx);

            if (score > best_score) {
                best_score = score;
                best_idx   = idx_i;
            }
        }
    }

    if (best_idx < 0) {
        return;
    }

    let tgt_idx = u32(best_idx);

    // Weak CAS with retry to avoid spurious failure
    var claimed : bool = false;
    for (var k: i32 = 0; k < 8; k = k + 1) {
        let res = atomicCompareExchangeWeak(&occupancy[tgt_idx], 0u, 1u);
        if (res.exchanged) {
            claimed = true;
            break;
        }
        if (res.old_value != 0u) {
            break; // genuinely occupied
        }
    }

    agent_target[i] = select(INVALID, tgt_idx, claimed);
}
