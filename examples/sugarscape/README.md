# Sugarscape Example

A large-scale implementation of the classic Sugarscape agent-based model (Epstein & Axtell, 1996) built on the Syren ABM Framework. This example demonstrates the framework's core capabilities — ECS-based agent storage, deterministic scheduling with dependency ordering, optional GPU-accelerated systems, and profiling support — while simulating up to one million agents on a toroidal sugar landscape.

## Background

Sugarscape is one of the foundational models in agent-based computational economics. Agents inhabit a 2D grid where each cell contains a renewable resource ("sugar"). Agents move, harvest, metabolise, and eventually die, producing emergent wealth distributions that mirror real-world inequality patterns. This implementation follows the Axtell-style variant with agent replacement, meaning the population stays constant as dead agents are immediately replaced by fresh ones.

## Model Rules

Each simulation tick executes five systems in strict dependency order:

1. **Growback** — Every cell regenerates sugar by `growth_rate` per tick, capped at its natural capacity. Sugar capacity is determined by distance to two landscape peaks located at roughly (30%, 30%) and (70%, 70%) of the grid, producing a two-peaked geography with rich centres and barren peripheries.

2. **Move & Harvest** — Agents are processed in a shuffled random order. Each agent scans the four cardinal directions up to its `vision` range, selects the nearest unoccupied cell with the highest sugar (ties broken randomly), moves there, harvests all sugar on the destination cell, and ages by one tick.

3. **Metabolism** — Each agent's wealth is reduced by its `metabolism` rate. This system has two backends: a standard CPU path, and an optional GPU path that runs a WGSL compute shader to process all agents in parallel on the GPU.

4. **Replacement** — Agents that have starved (wealth reaches zero) or exceeded their `max_age` are removed and replaced with a new randomly-initialised agent at an unoccupied cell. This keeps the total population constant across ticks.

5. **Stats** — Collects and prints per-tick statistics to stdout in CSV format: agent count, average sugar per tile, and wealth distribution metrics (mean, median, p90, p99, max, and Gini coefficient).

## Agent Properties

Each agent carries the following attributes, all assigned randomly at creation:

| Property     | Range   | Description                                        |
|--------------|---------|----------------------------------------------------|
| `wealth`     | 5–25    | Initial sugar endowment                            |
| `metabolism` | 1–4     | Sugar consumed per tick                            |
| `vision`     | 1–6     | How many cells the agent can see in each direction |
| `max_age`    | 60–100  | Age (in ticks) at which the agent dies             |

## Running the Example

Since the example file lives in `examples/sugarscape/sugarscape_v2.rs` rather than the standard `examples/` root, you need to point Cargo at it explicitly. You can do this by adding an `[[example]]` entry to your `Cargo.toml`:

```toml
[[example]]
name = "sugarscape_v2"
path = "examples/sugarscape/sugarscape_v2.rs"
```

Then run with:

```bash
# CPU-only, default settings (1M agents, 4096x4096 grid, 20 ticks)
cargo run --release --example sugarscape_v2

# With GPU metabolism acceleration
cargo run --release --features gpu --example sugarscape_v2

# With profiling (writes a Chrome Trace JSON file)
cargo run --release --features profiling --example sugarscape_v2

# Both GPU and profiling
cargo run --release --features "gpu profiling" --example sugarscape_v2
```

## Command-Line Options

All parameters have sensible defaults, so the example runs out of the box with no arguments.

| Flag              | Default       | Description                                                  |
|-------------------|---------------|--------------------------------------------------------------|
| `--agents N`      | `1000000`     | Number of agents in the simulation                           |
| `--width W`       | `4096`        | Grid width in cells                                          |
| `--height H`      | `4096`        | Grid height in cells                                         |
| `--ticks N`       | `20`          | Number of simulation ticks to run                            |
| `--seed N`        | `0xA57E11`    | RNG seed (decimal or `0x`-prefixed hex)                      |
| `--growth-rate N` | `1`           | Sugar regrown per cell per tick                              |
| `--cpu`           | off           | Force CPU metabolism even when GPU is available              |
| `--profile PATH`  | `profile/sugarscape_axtell_large.json` | Chrome Trace output path (requires `profiling` feature) |
| `--no-profile`    | —             | Disable profiling trace output                               |
| `-h`, `--help`    | —             | Print usage information                                      |

### Example Configurations

```bash
# Small test run
cargo run --release --example sugarscape_v2 -- --agents 1000 --width 128 --height 128 --ticks 5

# Large-scale benchmark
cargo run --release --features gpu --example sugarscape_v2 -- --agents 2000000 --width 8192 --height 8192 --ticks 50

# Reproducible run with a specific seed
cargo run --release --example sugarscape_v2 -- --seed 0xDEADBEEF
```

## Output Format

Progress information is printed to **stderr** (setup time, per-tick timing, total elapsed time). Statistical output is printed to **stdout** as CSV with the following columns:

```
tick,agents,avg_sugar_per_tile,wealth_mean,wealth_p50,wealth_p90,wealth_p99,wealth_max,wealth_gini
```

This makes it straightforward to redirect data to a file for analysis:

```bash
cargo run --release --example sugarscape_v2 -- --ticks 100 > results.csv
```

## Framework Features Demonstrated

This example exercises several key features of the Syren ABM Framework:

- **Archetype-based ECS** — Agents are stored as `SugarAgent` components registered through the `ComponentRegistry`, enabling cache-friendly iteration over millions of entities.
- **Entity sharding** — `EntityShards` distributes entities across multiple shards (one per worker thread) for parallel access.
- **Deterministic scheduling** — Systems declare `AccessSets` with `produces`/`consumes` dependencies, and the `Scheduler` enforces a strict execution order (Growback → Move → Metabolism → Replacement → Stats).
- **GPU compute** — The metabolism system can optionally run as a WGSL compute shader via `wgpu`, demonstrating the framework's `GpuSystem` trait and `GPUPod` marker for GPU-safe data types.
- **Profiling** — All systems emit tracing spans via `abm_framework::span()`, producing Chrome Trace JSON output when compiled with the `profiling` feature.
- **Deferred commands** — Agent spawning uses `Command::Spawn` with deferred application, showing the framework's structural mutation pattern.

## Notes

- The grid uses **toroidal (wrapping) boundaries** — agents moving past one edge appear on the opposite side.
- The RNG is a simple xorshift64 generator, chosen for speed and determinism rather than statistical quality. This is sufficient for the model's randomisation needs and ensures reproducibility given the same seed.
- Population count must not exceed the total number of grid cells (`width × height`), and both dimensions must fit within `u32` indexing.
