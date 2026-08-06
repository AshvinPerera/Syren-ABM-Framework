# Macroeconomy example

A multi-market macroeconomic agent-based model on the Syren ABM Framework,
implementing appendix equations A.1–A.142 of Wiese et al. It runs on
**synthetic data only**: the model's mechanics are reproduced, the authors'
Austrian trajectories are not.

## Sources

- Wiese, S., Chmieliauskaite, K., Arroyo, J., Kaszowska-Mojsa, J., Moran, J.,
  Farmer, J. D. et al. "Forecasting Macroeconomic Dynamics Using a Data-Driven
  Agent-Based Model." arXiv:2409.18760, INET Oxford Working Paper 2024-25;
  *Journal of Economic Dynamics and Control* 173, 105076 (2025), DOI
  `10.1016/j.jedc.2025.105076`. The appendix is the equation index for
  A.1–A.142.
- Wiese, S. "Dynamic interactions in economics: from micro-level games to
  macroeconomic agent-based models." University of Oxford DPhil thesis (2024),
  DOI `10.5287/ora-5reg8nv9g`. Used where it states model structure,
  parameters, or initialisation detail more explicitly than the paper.
- Poledna, S., Miess, M. G., Hommes, C., Rabitsch, K. "Economic forecasting
  with an agent-based model." IIASA WP-20-001; *European Economic Review* 151,
  104306 (2023). The model Wiese et al. extend. Supplies the goods-market
  search-and-matching algorithm (Online Appendix A.1.1), the Austrian parameter
  table, and several terms Wiese cite rather than restate.
- Baptista, R., Farmer, J. D., Hinterschweiger, M., Low, K., Tang, D., Uluc, A.
  "Macroprudential policy in an agent-based model of the UK housing market."
  Bank of England Staff Working Paper 619 (2016). The source Wiese cite for the
  housing block.

Equation numbers are cited in comments throughout `equations.rs` and
`systems.rs`.

## Scope

One or more OECD-style national economies trading with the rest of the world.
One tick is one quarter; one simulated agent stands for `1000` real ones.

| Agent | Count at `--firms-per-sector 33` | Role |
|---|---|---|
| Firm | 595, across 18 NACE Rev. 2 sectors | Produces one sectoral good from labour, intermediates and capital |
| Individual | 9,571 | Employed, unemployed, or not economically active |
| Household | 2,393 | Groups individuals; consumes, invests, borrows, owns property |
| Bank | 2 | Takes deposits, extends credit under the A.25–A.32 screens |
| Central bank | 1 | Sets the policy rate by Taylor rule |
| Government entity | 149 | Buys on the goods market |
| Government account | 1 | Collects taxes, pays benefits, carries the debt |
| Property | ~2,400 | Owner-occupied, rented, or listed |
| Rest of world | 1 | Exports and imports |

Four markets clear each quarter: goods (firms, households, government entities
and the rest of the world), labour, credit (firm loans, household consumption
loans, mortgages), and housing (sales and rentals).

## Tick sequence

Ten systems run at fixed scheduler priorities. The order is load-bearing:
credit clears before firms buy inputs, and goods before profits are realised.

| Priority | System | Equations |
|---|---|---|
| 10 | `aggregate_previous_state` | A.1–A.15 |
| 20 | `refit_expectations` | A.16–A.21 |
| 30 | `firm_individual_targets` | A.59–A.68, A.129–A.132 |
| 40 | `labour_market` | A.141–A.142 |
| 50 | `planning_and_production` | A.45, A.72–A.82, A.95–A.106, A.134–A.139 |
| 60 | `housing_preclear` | A.107–A.116 |
| 70 | `credit_market` | A.25–A.39, A.117–A.118 |
| 80 | `housing_completion` | A.112–A.116 |
| 90 | `goods_market` | A.88, A.140 |
| 100 | `realised_accounting` | A.40–A.44, A.85–A.100, A.119–A.127 |

Each block is described in [docs/model.md](docs/model.md).

## Initial state

`FixtureDataProvider` *solves* an initial state that satisfies the model's own
equations rather than fitting one to data: a social accounting matrix supplies
every sectoral weight, saving and investment rates come from A.101/A.102, firm
stocks and prices from A.50–A.58, and bank reserves from A.23. Nothing is tuned
to make the simulation behave. See [docs/synthetic-data.md](docs/synthetic-data.md)
for the construction order and what remains synthetic.

`RealDataProvider` is a stub. It fails fast naming the datasets the papers'
pipeline requires — OECD ICIO and national accounts, IMF and World Bank series,
BIS policy rates, ECB HFCS microdata, Compustat firm and bank microdata, ESRB
macroprudential measures — and ingesting them is out of scope here.

## Deviations

Four departures from the paper as printed, each argued: the work-effort factor
applied to a base wage rather than compounded (A.69), Poledna's dividend term
restored to A.80, A.30's loan-to-income base read as annual income, and A.109's
annuity denominator taken with the conventional negative exponent.
[docs/deviations.md](docs/deviations.md) states the case for each; `config.yaml`
carries a scenario reverting the first three.

## Running

```bash
cargo run --release --features "model messaging" --example macroeconomy -- --fixture tiny --ticks 40 --seed 42 --firms-per-sector 33
```

| Flag | Meaning |
|---|---|
| `--fixture tiny` | Synthetic population (the default mode). |
| `--firms-per-sector N` | Population scale. 33 ≈ Austria at the 1:1000 factor. |
| `--ticks N`, `--seed N` | Quarters to run; model seed. |
| `--config <path> --scenario <name>` | Apply a block from `config.yaml`. |
| `--trace <dir>` | Write `trace_aggregates.csv` and `trace_firms.csv` into `<dir>`, which is created if needed. |
| `--profile <path>` | Write a Chrome Trace profile. Needs `--features profiling`. |
| `--debug-firm <id>` | Per-quarter dump of one firm's internals. |
| `--data-dir <path> --country <code> --initialisation <yyyy-Qn>` | Real-data mode. |

Per-tick aggregates go to stdout as CSV, beginning:

```text
tick,production,ppi,cpi,hpi,rpi,total_loans,gdp_gap,blocked_mortgages,excess_demand,...
```

with credit, labour, goods-market and cost breakdowns following. The housing
and mortgage summaries print to stderr on completion.

Output is reproducible at any thread count. What makes that hold is described
in [docs/limitations.md](docs/limitations.md), alongside the measured scaling
and the subsystems that the published calibration leaves inert.

## Tests

```bash
cargo test --release --features "model messaging" --test macroeconomy
```

The suite covers scheduler and market ordering, thesis parameter defaults and
the initialisation recipe, AR(1) / Taylor / ARDL helper shapes, named equation
arithmetic, configuration overrides, shard sizing, and byte-identical
trajectories across seeds and thread counts.

## Documentation

| Document | Contents |
|---|---|
| [docs/model.md](docs/model.md) | Agents, tick sequence, and the equation set block by block |
| [docs/parameters.md](docs/parameters.md) | Every parameter, its value, and its source |
| [docs/synthetic-data.md](docs/synthetic-data.md) | How the initial population is solved |
| [docs/deviations.md](docs/deviations.md) | The deliberate departures from the paper |
| [docs/limitations.md](docs/limitations.md) | Determinism, scale, inert subsystems, open work |
