# Macroeconomy model documentation

An implementation of the agent-based macroeconomic model in

> Wiese, S., et al. (2024), *Forecasting macroeconomic dynamics with an
> agent-based model* — appendix equations A.1–A.142

which extends Poledna et al., *Economic forecasting with an agent-based model*
(IIASA WP-20-001). It runs on synthetic data rather than the OECD / HFCS /
Compustat microdata the authors use, so it reproduces the model's *mechanics*,
not their Austrian trajectories.

| Document | Contents |
|---|---|
| [model.md](model.md) | Agents, tick sequence, and the equation set block by block |
| [parameters.md](parameters.md) | Every parameter, its value, and its source |
| [synthetic-data.md](synthetic-data.md) | How the initial population is solved |
| [deviations.md](deviations.md) | The three deliberate departures from the paper |
| [limitations.md](limitations.md) | Known defects, inert subsystems, open work |

## Running it

```bash
cargo run --release --features "model messaging" --example macroeconomy -- --fixture tiny --ticks 40 --seed 42 --firms-per-sector 33
```

| Flag | Meaning |
|---|---|
| `--firms-per-sector N` | Population scale. 33 ≈ Austria at the paper's 1:1000. |
| `--config <path> --scenario <name>` | Apply a block from `config.yaml`. |
| `--ticks N`, `--seed N` | Quarters to run; model seed. |
| `--trace` | Write `trace_aggregates.csv` and `trace_firms.csv`. |
| `--debug-firm <id>` | Per-quarter dump of one firm's internals. |

Set `RAYON_NUM_THREADS=1` for reproducible output — see
[limitations.md](limitations.md).

Tests: `cargo test --release --features "model messaging" --test macroeconomy`
