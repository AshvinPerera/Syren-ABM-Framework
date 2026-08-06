# Known limitations and open work

Measured facts about the current build.

## Nondeterminism — fixed

Runs are reproducible at any thread count. Verified over 40 quarters at 595
firms: 1, 4 and 8 threads all produce an identical hash.

The dominant cause was **not** the message drain order. `collect_rows` pushed
rows into a shared `Vec` from a parallel `for_each`, so row order was
work-stealing order, and every system that iterated a collected row set
inherited it. Sorting rows by model id after collection removes it.

Message drain order is still work-stealing dependent, and worker slots now
register in `worker_id` order rather than first-emit order. That remains
theoretically observable if a system consumes messages without going through a
sorted row set; no such path is currently reachable, which is why the runs
agree. Ordering within a message bucket is the remaining hardening.

## Scale — still open

`write_rows` was O(n^2): a `rows.iter().find()` per slot, for every component
type in every system. It now builds a dense id index once, so it is O(n).

That did **not** make the model scale. Measured, 3 ticks per point:

| Firms | Individuals | Elapsed |
|---|---|---|
| 595 | 9,571 | 2.2 s |
| 1,189 | 19,142 | 7.1 s |
| 2,377 | 38,284 | 85.5 s |
| 4,753 | 76,568 | 494 s |

Still super-quadratic. Two reasons the fix did not land:

1. **16 `iter().find(..)` / `iter().position(..)` sites remain** in the market
   loops -- borrower lookup per credit application, property lookup per housing
   purchase. These are the inverse of a message lookup ("given a message, find
   its agent") so message specialisation does not address them; they need the
   same dense index `write_rows` now uses.
2. **The determinism sort costs.** `collect_rows_by` sorts on every call, 42
   calls a tick, O(n log n) each. Correct, but it should come from the
   collection order rather than a sort -- which needs `engine::workers::worker_id`
   exported so the example can stage per worker like `space` does.

Note these figures are not comparable with any measured before the inactive
population was added: that change nearly doubled the agent count at every
`--firms-per-sector` value.

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
