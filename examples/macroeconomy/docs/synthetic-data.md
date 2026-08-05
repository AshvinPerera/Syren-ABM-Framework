# The synthetic population

The paper initialises from OECD input-output tables, HFCS household microdata,
Compustat firm microdata, and BIS/IMF/World Bank series. This implementation has
none of those, so `data.rs` *solves* an initial state that satisfies the model's
own equations instead of fitting one to data.

The distinction matters: nothing here is tuned to make the simulation behave.
Every quantity is derived from an accounting identity or from an initial
condition the paper states.

## Why the initial state must be solved, not guessed

Several of the paper's equations are simultaneously binding at `t = 0`:

```
A.25:  L ≤ ρ^DtE · K                     (with ρ^DtE = 1, so L ≤ K)
A.26:  K − L + D ≥ Π̄ / ρ^RoE            (capital large relative to profit)
A.27:  D + S + M + K ≤ Π̄ / ρ^RoA        (assets small relative to profit)
```

An initial state that violates any of them refuses all firms credit on tick 1.
`capital_per_output` is solved from A.27 and `debt_to_capital` from A.26, so all
three hold jointly at initialisation.

## Construction order

1. **Social accounting matrix.** `solve_sam()` closes
   `Y_s = intermediates + capital + C + I + G + X − M` per sector using national
   accounting shares. Everything downstream that needs a sectoral weight —
   `cpi_weights`, government consumption weights, investment weights, trade
   weights — is a normalised column of this matrix, never a uniform `1/18`.

2. **Saving and investment rates.** `ψ` and `ψ^H` are solved from A.101/A.102 so
   that initial consumption and investment match the SAM's final-demand columns.

3. **Firms.** Sizes are a Pareto draw normalised within each sector to hit the
   target headcount, giving employment heterogeneity (CV 0.63 at 33 firms per
   sector). Initial production, prices, demand, inventory, and input stocks come
   from A.50–A.56. Initial costs and profits are computed from A.57/A.58 — not
   assumed.

4. **Individuals and households.** Employed, unemployed at 7%, and inactive at
   46.6% (Poledna Table 5). Households group four individuals. The inactive
   supply no labour and earn no individual income; the household's `sb^O`
   supports them.

5. **Banks.** Reserves from A.23; equity proportional to loans granted.
   Household deposits are topped up so that A.23 reserves are non-negative,
   which they otherwise are not.

6. **Government.** Consumption weights and the `sb^O` level from the SAM and
   from OECD social-expenditure shares. Both the history *and* the current
   aggregate are seeded — see below.

7. **History.** 52 quarters, the window A.2's AR(1) fits over, derived from
   `quarters_between(first_real_data_quarter, initialisation)` rather than
   chosen. Carries a trend and a small wobble: a flat history makes every AR(1)
   forecast zero change forever.

## AR(1) series need an initial condition

Six series are seeded (production, sectoral production, PPI, CPI, HPI, RPI) and
so is government consumption. That last one was originally omitted on the
reasoning that "the model produces this series itself, so the AR(1) needs no
external input".

That reasoning is wrong, and the failure mode is worth recording. An AR(1)
fitted on an empty history with a zero current level forecasts zero. The
forecast sets the target, the target determines the realised flow, the realised
flow feeds the history. It is a self-sustaining zero — **government demand,
about 18% of final demand, was absent from every tick**.

Endogenous propagation still needs an initial condition, exactly as A.2's
production AR(1) does.

## What remains synthetic

- Sector count is 18 rather than the paper's 64 NACE-2 industries.
- The input-output, capital-compensation and net-fixed-asset matrices are banded
  synthetic matrices, not OECD ICIO tables.
- Household heterogeneity is limited to size and the Pareto firm draw; there is
  no HFCS income/wealth distribution, and consumption weights are not
  differentiated by income quintile as in Wiese A.7.5.1.
- Bank-firm and bank-household matching is by simple assignment rather than the
  linear sum assignment problem the paper solves.
