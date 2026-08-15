# Known limitations and open work

Measured facts about the current build.

## Determinism

Runs are reproducible at any thread count. Verified over 40 quarters at 595
firms: 1, 2, 4 and 8 threads produce an identical hash.

Two things make that true, and both are load-bearing:

- `collect_rows_by` sorts rows by model id. Rows are staged per worker and
  concatenated, and which worker sees which rows depends on work stealing, so
  the collected order is otherwise unstable. Every system that iterates a
  collected row set inherits it.
- Systems that accumulate inside parallel iteration keep a partial per worker
  and add them back in worker order (`ParallelSumF64`). Float addition is not
  associative, so summing in completion order would drift with thread count.
  Maxima use `AtomicMaxF64`, which is order-independent by construction.

Per-agent random draws are keyed on the agent (`rng_for_agent`), so a draw does
not depend on the position at which the agent is visited. Draws that shuffle a
whole collection — the labour market's hiring order, the credit market's
arrival order, the goods market's seller sampling — still come from one
sequential stream per system, which is what those equations describe.

The model seed reaches those draws through `RunContext::simulation_seed`, which
`ModelBuilder::with_seed(config.seed)` sets on the root scheduler and every
shared sub-scheduler. `DetRng::from_context` keys each stream on
`(simulation_seed, tick, system_id, salt)`, so distinct seeds diverge and a
fixed seed reproduces exactly at any thread count.

## Scale

Three ticks, one process, all cores:

| Firms | Individuals | Elapsed | Ratio |
|---|---|---|---|
| 1,189 | 19,142 | 0.74 s | — |
| 2,377 | 38,284 | 1.07 s | 1.45× |
| 4,753 | 76,568 | 2.01 s | 1.88× |
| 9,505 | 153,270 | 4.14 s | 2.06× |

Fitted exponent **0.84** — linear within measurement noise.

Composition at `--firms-per-sector 33` is 12,712 agents: 595 firms, 9,571
individuals, 2,393 households, 149 government entities, 2 banks, a central bank
and the rest of the world. Two million total agents is 157× that, i.e.
`--firms-per-sector 5192`, giving ~93,600 firms and ~1.5M individuals.
Extrapolating the ladder puts that at roughly **half a minute for three ticks
and six minutes for forty quarters**, single process.

That extrapolation is 10× beyond the largest point measured and has not been
run. Memory is uninstrumented.

Where the time goes at 9,505 firms (`--profile`, Chrome Trace):

| | |
|---|---|
| goods market | 1.15 s |
| collect + write back | 1.14 s |
| planning and production | 0.75 s |
| realised accounting | 0.47 s |
| credit market | 0.40 s |

The goods market is per-transaction settlement — search, match, settle, emit.
Reducing it means changing what A.1 does, not how it is written.

The collect and write-back remain because four systems need a global view by
construction: the labour market matches firms to job seekers, the credit market
clears in the randomised arrival order A.36 specifies, the goods market depletes
sellers as it goes, and the housing market draws households one at a time
against remaining listings. None can be expressed as per-row iteration. The
cost is repetition rather than any single collect: `Household` is materialised
six times a tick, `Firm` eight, `Property` five.

The systems that *are* per-row — target setting, planning, accounting — iterate
ECS columns in place and never materialise a row set.

## Inert subsystems

### No relative prices

Under Table 4's Austrian calibration `φ^DP = φ^CP = 0`, so A.73 reduces to
`P_f(t) = (1 + π̄^PPI) P_f(t−1)` — the same multiplier for every firm — and A.52
starts every price at 1. All firm and sector prices stay identical forever, so
every price index coincides (CPI ≡ PPI exactly).

This is a property of the published calibration, not a defect, but it means the
model has **no relative price movement at all**. State it whenever reporting
results.

It is also the likely reason a persistent excess-demand gap never clears: the
two price channels that would close it are switched off. Not confirmed.

### House prices barely move

HPI runs 1.000002 to 1.000538 over 40 quarters against 874 completed sales —
five basis points in ten years. Wiese's housing block has exactly two sale-price
rules, both implemented: an ask marked up by *predicted* HPI inflation (A.112)
and a random reduction while unsold (A.113). There is no competitive bid-up
anywhere in A.107–A.116.

Rents are not stuck the same way: the A.116 CPI indexation of tenanted
properties is an external driver the sale side has no counterpart to, and RPI
runs 0.9992 to 1.0430.

The loop is closed: ask = (1 + π̄^HPI) × value → sale = ask → value = sale →
π̄^HPI is an AR(1) on realised HPI. Once HPI settles at 1.0 the markup goes to
zero, and A.113 can only push down.

Carro's random markup and double-auction bid-up would break the loop, but they
are **not in Wiese** — adding them would be a fourth deviation, and unlike the
other three it would not be restoring a dropped term. Left as specified.

### Credit market switches itself off

Firm debt decays to almost nothing within ten quarters and interest costs go to
zero. In a converged steady state firms retain roughly 1,700 a quarter after
dividend and tax with no investment to fund, so they never hit A.81's financing
gap and the A.25–A.27 screens stop binding on anything.

Bank reserves are the A.43 residual, so with deposits rising and loans flat they
accumulate. Arithmetic downstream, not an independent defect.

## Government balance deteriorates after t15

| Tick | 1 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|
| Revenue / GDP | 44.0% | 43.3% | 42.8% | 45.2% | 46.6% | 47.5% |
| Deficit / GDP | −1.5% | −6.4% | −6.1% | 11.5% | 12.4% | 12.5% |
| Debt / GDP | 0.78× | 0.47× | 0.06× | 0.73× | 1.91× | 3.10× |

Revenue lands close to Austria's actual ~49% of GDP, which is an independent
check nothing was fitted to. The balance is Austria-plausible through t10 —
covering the paper's 12-quarter forecast horizon — then deteriorates.

Structural: A.97 grows `sb^O` and A.95 grows government consumption at predicted
growth, against a revenue base that tracks a converging wage bill.

## Framework surface not used

The example does not exercise everything the framework offers:

- **`reduce_read` / `reduce_read2`** — parallel folds that never materialise a
  row set. `aggregate_previous_state` collects seven components purely to sum
  them, which is exactly the shape these are for. Not converted:
  `compute_aggregates` would have to be restructured into folds for a system
  that costs 0.08 s.
- **`Bucket` and `Targeted` message specialisations** — all eleven message types
  are `BruteForce`, so every consumer scans the whole buffer. Several are
  addressed to one agent and would suit `Bucket`, keyed on the recipient.
- **`space`, `network`, `gpu`, `batching`** — no spatial dimension, no explicit
  agent graph, and no GPU work attempted.

## Fidelity gaps not yet closed

- Linear sum assignment matchings (bank–firm, bank–household, firm–employee,
  household–property) are simple assignments here.
- `SIM_SCALE_FACTOR` is carried on the environment and settable from
  `config.yaml`, but no equation reads it.
- Household consumption weights are not differentiated by income quintile
  (Wiese A.7.5.1).
- 18 sectors rather than 64 NACE-2 industries.

## What the model does do

40 quarters at 595 firms (`--seed 42`): zero bankruptcies, output converging to
a steady state, full employment of the active population from t4, PPI 1.000 →
1.112 (~1.06%/yr), profits positive throughout, and GDP by output and by
expenditure agreeing within ~1% at steady state — the main stock-flow-consistency
check. Housing clears 854 sales against 27 blocked mortgages. Robust across seeds
1, 7, 42, 99, 2024.

| Tick | 1 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|
| Production | 18,810 | 21,104 | 25,777 | 29,961 | 30,061 | 30,061 |
| PPI | 1.000 | 1.013 | 1.029 | 1.059 | 1.086 | 1.112 |
