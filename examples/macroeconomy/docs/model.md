# The model

## Agents

| Agent | Count at `--firms-per-sector 33` | Role |
|---|---|---|
| Firm | 595, across 18 NACE-2 sectors | Produces one sectoral good from labour, intermediates, capital |
| Individual | 9,571 | Employed, unemployed, or not economically active (`I^N`) |
| Household | ~2,393 | Groups individuals; consumes, invests, borrows, owns property |
| Bank | 2 | Takes deposits, extends credit subject to A.25–A.32 |
| Central bank | 1 | Sets the policy rate by Taylor rule (A.45) |
| Government entity | ~149, i.e. 25% of firms | Consumes on the goods market |
| Rest of world | 1 | Exports and imports (A.136–A.139) |
| Property | ~2,400 | Owner-occupied, rented, or for sale |

One tick is one quarter.

## Sequence of events

Ten systems run at fixed scheduler priorities (`systems.rs`). The ordering is
load-bearing: credit must clear before firms buy inputs, and goods before
profits are realised.

| Priority | System | Does | Equations |
|---|---|---|---|
| 10 | `aggregate_previous_state` | Snapshot last quarter, roll history buffers | A.133–A.135 |
| 20 | `refit_expectations` | AR(1) refits: growth, PPI/CPI/HPI/RPI, government consumption | A.2, A.16–A.21 |
| 30 | `firm_individual_targets` | Predicted growth, demand, profits; target production and wages | A.59–A.62, A.69–A.71 |
| 40 | `labour_market` | Fire to target, post vacancies, unemployed search | A.128–A.131 |
| 50 | `planning_and_production` | Work effort, Leontief production, prices, input demand | A.63–A.68, A.72–A.79 |
| 60 | `housing_preclear` | List properties, form bids, size mortgage demand | A.107–A.118 |
| 70 | `credit_market` | Firm and household applications against the lending screens | A.24–A.39 |
| 80 | `housing_completion` | Settle purchases and rentals conditional on mortgage grants | A.112–A.116 |
| 90 | `goods_market` | Search-and-matching clearing, firms served first | A.1, A.88 |
| 100 | `realised_accounting` | Profits, deposits, equity, taxes, bankruptcy, aggregates | A.89–A.100, A.119–A.127 |

## Firms

### Expectations

**Idiosyncratic growth** (A.59) is `Q_f(t−1) / (Y_f(t−1) + S_f(t−2)) − 1` when
the firm faced excess demand while priced above its sector average, or excess
supply while priced below; otherwise zero.

**Predicted demand** (A.60): `Q̄_f = (1 + γ̄_s)(1 + φ^Q_F γ̄_f) Q_f(t−1)`.

**Predicted profit** (A.61): `Π̄_f = (1 + π̄^PPI)(1 + γ̄_f) Π_f(t−1)`.

> A.61 is multiplicative in *last quarter's realised profit*, and both factors
> are positive. The sign of profit is therefore **absorbing**: one negative
> quarter makes every subsequent prediction negative, so A.27 refuses that firm
> credit permanently, it cannot restock, and A.72 caps its output at zero. This
> is the mechanism behind the collapse documented in
> [deviations.md](deviations.md).

### Production

**Target production** (A.62) is the minimum of four terms: predicted demand plus
the inventory gap `φ^StY Y_f(t−1) − S_f(t−1)`, and predicted demand adjusted
toward each of the labour, intermediate and capital ceilings by `χ^H`, `χ^M`,
`χ^K` respectively.

**Realised production** (A.72) is Leontief:
`Y_f = min(Ŷ_f, H_f, M_f, K_f)`.

- `M_f` (A.63) and `K_f` (A.64) are the minima over input sectors of stock
  divided by the technical coefficient.
- Labour input `H_f` (A.65) is the work-effort factor times summed individual
  labour supply.
- **Work effort** (A.66–A.67): `φ^WE = min(h^max, min(M_f, K_f) / (h_f(0) Σ H_i))`,
  then `h_f(t) = φ^WE h_f(0)`. Capped at `h^max = 1.5`.

### Prices and wages

**Prices** (A.73) compound predicted PPI inflation with demand-pull (A.74–A.75)
and cost-push (A.76–A.77) terms. Under the Austrian calibration
`φ^DP = φ^CP = 0`, so **both terms are inert** and every firm's price follows an
identical path — see [limitations.md](limitations.md).

**Wages** (A.69): `w_i(t) = (1 + π̄^PPI)(1 + μ^WN) φ^WE w_i(t−1)`. The labour
market tightness markup `μ^WN` (A.70) is zero for Austria (`φ^WN = 0`). This
equation is one of the three [deviations](deviations.md).

### Finance and accounting

Predicted deposit change (A.80) drives short-term (A.81) and long-term (A.82)
loan demand. Credit that is applied for but not granted scales back input
purchases (A.83–A.84). Inventory and stocks update in A.85–A.87; realised demand
is sales plus unmet demand (A.88).

Costs (A.89) are wages, intermediate and capital purchases, deposit and loan
interest, and production taxes. Profits (A.90) are sales plus inventory change
less costs. Deposits (A.91), debt (A.92) and equity (A.93) close the quarter. A
firm that is both cash-flow insolvent (`D_f < 0`) and balance-sheet insolvent
(`E_f < 0`) is replaced by an entrant in the same sector (A.94).

## Banks

Credit to a firm is bounded by three borrower screens and one supply
constraint. All four bind independently; the tightest wins.

| Eq | Screen | Form |
|---|---|---|
| A.25 | Debt-to-equity | `V_l ≤ ρ^DtE Σ_s P_s K_fs − L_f(t−1) + [D_f(t−1)]^− + …` |
| A.26 | Return-on-equity | `V_l ≤ Σ_s P_s K_fs + D_f(t−1) − L_f(t−1) − Π̄_f/ρ^RoE` |
| A.27 | Return-on-assets | grant only if `Π̄_f / (L_f(t−1) + E_f(t−1)) ≥ ρ^RoA` |
| A.32 | Supply | `V_b^max = E_b(t−1)/ρ^CAR − Σ_l V_l` |

Households face A.28 (consumption loan-to-income) and A.29–A.31 (mortgage
loan-to-value, loan-to-income, debt-service-to-income). Supply is allocated
across loan classes by non-performing-loan ratios (A.33–A.36).

Rates come from a single-equation ARDL error-correction model (A.24). Bank
profits (A.40), equity (A.41), liabilities (A.42) and reserves (A.43) follow. A
bank whose solvency ratio falls below `ρ^SR` is bailed in by the others (A.44).

## Households and individuals

**Income** (A.104) sums individual incomes, other social benefits `sb^O`, rental
income, and income from financial assets. Predicted income (A.103) is the same
sum on predicted terms.

**Consumption** (A.105) is the maximum of three terms: a floor based on
unemployment benefit, a fraction `1 − φ^SR` of predicted income, and a smoothed
average of the last `T^CO` quarters. **Investment** (A.106) is a fixed rate
`φ^IR` on predicted income.

**Housing** (A.107–A.118) depends on tenure. A household forms a maximum
affordable price from predicted income (A.107), compares the annual cost of
renting (A.108) with that of buying (A.109), and buys with a logistic
probability in the difference (A.110). Owners list at
`(1 + π̄^HPI) V_p` (A.112) and cut the price randomly while unsold (A.113).
Mortgage demand is price less full financial wealth (A.118).

**Wealth** identities close in A.119–A.127: real and financial assets, deposits,
other financial assets, total and net wealth, and insolvency.

**Individuals** are employed, unemployed, or not economically active. Only the
unemployed search (A.128–A.131). The inactive earn no individual income and are
supported through their household's `sb^O`.

## Government and central bank

Consumption is an AR(1) on the realised series, distributed evenly across
entities (A.95). Unemployment benefits rise when a downturn is predicted (A.96);
other benefits grow with predicted growth (A.97).

**Revenue** (A.98) collects social contributions and taxes on labour income,
rents, corporate profits, value added, capital formation, production and
exports. **Deficit** (A.99) is social benefits plus consumption plus interest
less revenue; **debt** (A.100) accumulates it.

The policy rate is a growth-form Taylor rule with no output gap (A.45), whose
`r*`, `ξ^π`, `ξ^γ` are recovered from an OLS regression at initialisation
(A.46–A.49).

## Rest of world

Exports and imports are indexed to their initial levels scaled by the production
index and the price index (A.136–A.139), with sector weights taken from the
input-output table. Net exports accumulate once per quarter.

## Accounting identities

GDP is computed three ways and cross-checked each tick (A.140–A.142): output,
income, and expenditure. In the current build output and expenditure agree to
within ~1% over 40 quarters, which is the model's main stock-flow-consistency
check.
