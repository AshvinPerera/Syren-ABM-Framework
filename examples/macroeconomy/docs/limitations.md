# Known limitations and open work

Measured facts about the current build.

## Nondeterminism

**Two identical invocations with the same seed produce different output at 595
firms.** Set `RAYON_NUM_THREADS=1` for reproducible runs.

| Threads | Firms | Reproducible |
|---|---|---|
| 1 | 595 | yes |
| 4 | 595 | **no** |
| 8 | 595 | **no** |
| 4 | 19 | yes |

Ticks 1–2 are bit-identical; tick 3 diverges by 1.2e-5 relative. That is far too
large to be accumulated rounding, so it is generated *within* tick 3 by an
ordering difference.

**Mechanism.** Messages are staged per-thread and concatenated by `drain_into`,
so the order workers are visited is the message order every system observes.
Downstream the markets do:

```rust
let mut block: Vec<GoodsDemand> = demands.iter().filter(..).collect();
rng.shuffle(&mut block);
```

A Fisher–Yates shuffle with a fixed stream applies a fixed permutation *to
positions*. Applied to two differently-ordered inputs it gives two different
orders — the shuffle preserves the nondeterminism rather than removing it.
Clearing order then decides which buyer reaches which seller, and A.1's
search-and-matching is order-sensitive by construction.

Worker registration order has been made stable (workers now register under
`worker_id()` rather than in first-emit order), but that is **not sufficient**:
Rayon's work-stealing decides which worker handles which agents, so a given
worker's buffer does not hold the same messages twice.

The 19-firm case passes only because the workload never splits. Both
reproducibility tests run the tiny fixture, which is why they pass while the
model is broken.

**Fix**: give the drained buffer a canonical order independent of which thread
produced what. See the strategy note in the framework discussion — the
`Bucket`/`Targeted` message specialisations make this cheaper than a global
sort.

## Scale

Runtime is super-quadratic and worsening. Three ticks per point:

| Firms | Individuals | Elapsed | Ratio |
|---|---|---|---|
| 595 | 9,571 | 2.5 s | — |
| 1,189 | 19,008 | 8.7 s | 3.5× |
| 2,377 | 38,016 | 42.5 s | 4.9× |
| 4,753 | 76,032 | 347 s | 8.2× |
| 9,506 | 152,064 | timed out (>9 min) | — |

Implied exponents 1.81, 2.29, 3.03 — the signature of an O(n²) algorithm falling
out of cache.

**Cause**: `systems.rs` contains 16 `iter().find(..)` / `iter().position(..)`
sites, several inside per-message loops — e.g. `firms.iter().find(..)` per credit
application, `properties.iter().position(..)` per housing purchase. At 595 firms
a scan is free; at 78,000 it is ~6e9 comparisons per tick.

The deeper cause is that all eleven message types are declared
`BruteForceMessage` even though most are addressed to one agent. The framework
already provides `BucketMessage` and `TargetedMessage` for exactly this.

**Projection to 2M agents**: ~78,700 firms, ~107 hours for 3 ticks, ~59 days for
40 quarters — for one Monte Carlo draw. The paper averages 500. Not a hardware
problem.

Note the 595-firm run is already Austria at the paper's 1:1000 scale, so the
paper's own scale is the *smallest* row above.

## Inert subsystems

### No relative prices

Under Table 4's Austrian calibration `φ^DP = φ^CP = 0`, so A.73 reduces to
`P_f(t) = (1 + π̄^PPI) P_f(t−1)` — the same multiplier for every firm — and A.52
starts every price at 1. All firm and sector prices stay identical forever, so
every price index coincides (CPI ≡ PPI exactly).

This is a property of the published calibration, not a defect, but it means the
model has **no relative price movement at all**. State it whenever reporting
results.

It is also the likely reason a ~20% excess demand gap never clears: the two
price channels that would close it are switched off. Not confirmed.

### Housing prices do not move

HPI is 1.0000 for all 40 quarters despite ~860 transactions. Wiese's housing
block has exactly two price rules, both implemented: an ask marked up by
*predicted* HPI inflation (A.112) and a random reduction while unsold (A.113).
There is no competitive bid-up anywhere in A.107–A.116.

The loop is therefore closed: ask = (1 + π̄^HPI) × value → sale = ask → value =
sale → π̄^HPI is an AR(1) on realised HPI. Once HPI settles at 1.0 the markup
goes to zero. A.113 can only push down.

Carro's Eq. (6) random markup and double-auction bid-up would break the loop,
but they are **not in Wiese** — adding them would be a fourth deviation, and
unlike the other three it would not be restoring a dropped term. Left as
specified.

### Credit market switches itself off

Firm debt decays from 8,844 to ~1.6 by t9 and never recovers; interest costs go
to zero. In a converged steady state firms retain ~1,700/quarter after dividend
and tax with no investment to fund, so they never hit A.81's financing gap. The
A.25–A.27 screens stop binding on anything.

Bank reserves are the A.43 residual, so with deposits rising and loans flat they
balloon. Arithmetic downstream, not an independent defect.

## Government balance deteriorates after t15

| Tick | 1 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|
| Revenue / GDP | 44.0% | 43.3% | 42.8% | 45.3% | 46.6% | 47.5% |
| Deficit / GDP | −1.5% | −6.3% | −5.7% | 11.6% | 12.5% | 12.5% |
| Debt / GDP | 0.78× | 0.47× | 0.07× | 0.77× | 1.95× | 3.14× |

Austria-plausible through t10, which covers the paper's 12-quarter forecast
horizon, then deteriorating. Structural: A.97 grows `sb^O` and A.95 grows
government consumption at predicted growth, against a revenue base that tracks a
converging wage bill.

## Fidelity gaps not yet closed

- Linear sum assignment matchings (bank–firm, bank–household, firm–employee,
  household–property) are simple assignments here.
- `SIM_SCALE_FACTOR` is declared and unused.
- Household consumption weights are not differentiated by income quintile
  (Wiese A.7.5.1).
- 18 sectors rather than 64 NACE-2 industries.

## What the model does do

40 quarters, 595 firms, zero bankruptcies, output converging to a steady state,
full employment of the active population from t4, PPI ~1.06%/yr, profits
positive throughout, and GDP by output and by expenditure agreeing within ~1% —
the main stock-flow-consistency check. Robust across seeds 1, 7, 42, 99, 2024.
