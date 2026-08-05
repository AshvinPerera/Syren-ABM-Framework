# Parameters and their sources

Defaults live in `state.rs` and `components.rs`, beside the equation each one
serves. `config.yaml` overrides them; its `defaults:` block is deliberately
empty so there is only one place for a value to disagree with the paper.

## Tax and contribution rates

Wiese A.6.2: "The income tax rate, corporate tax rate, export taxes,
value-added tax rate, and social insurance rates are taken directly from the
OECD database." Values are Poledna Table 1 for Austria.

| Symbol | Field | Value | Note |
|---|---|---|---|
| `τ^VAT` | `vat_rate` | 0.1529 | |
| `τ^INC` | `income_tax_rate` | 0.2134 | |
| `τ^SIW` | `social_insurance_worker_rate` | 0.1711 | Employees' |
| `τ^SIF` | `social_insurance_firm_rate` | 0.2122 | Employers' |
| `τ^CF` | `capital_tax_rate` | 0.2521 | |
| `τ^CORP` | `corporate_tax_rate` | 0.0779 | |
| `τ^EXPORT` | `export_tax_rate` | 0.003 | |
| `τ^PROD` | `production_tax_by_sector` | 0.02 | Synthetic stand-in; the paper takes these from IO tables |

## Bank regulatory requirements

Wiese A.3.2, verbatim.

| Symbol | Field | Value | Source |
|---|---|---|---|
| `ρ^CAR` | `car` | 0.08 | Basel III |
| `ρ^SR` | `solvency_ratio` | 0.10 | |
| `ρ^DtE` | `debt_to_equity` | 1.0 | |
| `ρ^RoE` | `return_on_equity` | 0.15 | |
| `ρ^RoA` | `return_on_assets` | 0.05 | |
| `ρ^LTI-C` | `consumption_lti` | 0.36 | |
| `ρ^LTV`, `ρ^LTI-M`, `ρ^DSTI` | mortgage caps | 0.80, 4.5, 0.35 | ESRB *Overview of national macroprudential measures* |
| `φ^CS` | `credit_supply_phi` | 2.0 | |

Loan maturities: firm short-term 1 quarter, long-term 2 years, household
consumption 1 quarter, mortgages 25 years (100 quarters).

## Firm parameters (Table 4, Austria / NPE)

| Symbol | Field | Value | Note |
|---|---|---|---|
| `h^max` | `work_effort_max` | 1.5 | Poledna: 150% of a full position |
| `φ^StY` | target inventory fraction | 0.10 | |
| `χ^H`, `χ^M`, `χ^K` | `chi_h`, `chi_m`, `chi_k` | 0.53, 0.03, 0.18 | Influence of each input ceiling on A.62 |
| `φ^Q_F` | demand adjustment | 0.0 | Firm-specific growth does not enter A.60 |
| `φ^DP`, `φ^CP` | price channels | 0.0 | **Both inert** — see [limitations.md](limitations.md) |
| `φ^WN` | wage markup | 0.0 | A.70 deliberately inert |
| `ω^M`, `ω^K` | utilisation rates | 0.85 | |
| `φ^M`, `φ^K` | stock influence | 1.0 | |
| `φ^FM`, `φ^FK` | financial friction on targets | 0.0 | Left to future research by the authors |
| `T^KD` | capital acquisition delay | 1 | Short horizons |
| `δ_s` | inventory depreciation | 0.0 | Short horizons |

## Housing (Carro et al. 2023, via Wiese A.7.2)

| Symbol | Value | | Symbol | Value |
|---|---|---|---|---|
| `φ^HP` | 42.9036 | | `β^HP` | 0.7892 |
| `μ^HP` | −0.0177 | | `σ^HP` | 0.1684 |
| `μ^PS` | 0.4 | | `φ^B` | 0.001 |
| `φ^HR` | 17.2166 | | `β^HR` | 0.3464 |
| `p^RS` | 7/8 | | `p^OS` | 79/80 |
| `p^PM` | 0.1964 | | `μ^PM`, `σ^PM` | 1.4531, 0.4889 |
| `p^RM` | 0.2848 | | `μ^RM`, `σ^RM` | 1.6559, 0.7855 |
| `φ^PIR` | 1 | | `T^PIR` | 1 |

## Household and government

| Symbol | Value | Source |
|---|---|---|
| `θ^DIV` | 0.7953 | Poledna Table 1. A [deviation](deviations.md) — Wiese omits the term |
| `θ^UB` | 0.3586 | Unemployment benefit replacement rate |
| `π*` | 0.02 | CB inflation target (A.4.1) |
| `d^RA` | 0.05 | Depreciation of other real assets |
| `T^CO` | 12 | Consumption smoothing window |
| `sb^O` | 0.415 × wage bill | Austria ESA D.62 ~21% of GDP against compensation ~47%, less the ~1.5 points A.99 handles as `\|I^U\| w^U` |

## Population scale

| Field | Value | Source |
|---|---|---|
| `inactive_rate` | 0.4661 | Poledna Table 5: H^inact 4,130,385 vs H^act 4,729,215 |
| `unemployment_rate` | 0.07 | Synthetic |
| `individuals_per_household` | 4 | Synthetic |
| government entities | 25% of firms | Wiese A.6.1, following Poledna |

At `--firms-per-sector 33` the population is 595 firms and 9,571 individuals,
which is Austria at the paper's stated 1:1000 scale (Poledna Table 5 gives 8.86M
people and ~611k firms).

## Replication blockers

Values the paper takes from data this implementation does not have, and which
are therefore synthetic:

- ARDL coefficients for A.24 (estimated per country from ECB/BIS series)
- Taylor rule `ρ`, `r*`, `ξ^π`, `ξ^γ` (A.46–A.49 OLS on BIS/IMF series)
- Sectoral production tax rates `τ^PROD_s` (IO tables)
- `n^LF`, `n^LH` labour and housing search intensities
- Voluntary job-exit probability
- The split of household saving between deposits and other financial assets
