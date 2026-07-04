# Macroeconomy Example

This example implements the data-driven macroeconomic agent-based model 
described by Wiese et al on the Syren ABM Framework. It is intended as a 
paper-aligned example of a multi-market economy built with the Syren ABM 
Framework.

The executable fixture mode is self-contained and uses synthetic data. Real-data
replication is deliberately fail-fast: the example names the required external
datasets.

## Sources

Primary model sources:

- Wiese, S., Chmieliauskaite, K., Arroyo, J., Kaszowska-Mojsa, J., Moran, J.,
  Farmer, J. D. et al. "Forecasting Macroeconomic Dynamics Using a Data-Driven
  Agent-Based Model." arXiv:2409.18760, INET Oxford Working Paper 2024-25,
  later published in Journal of Economic Dynamics and Control 173, 105076
  (2025), DOI `10.1016/j.jedc.2025.105076`.
- Wiese, S. "Dynamic interactions in economics: from micro-level games to
  macroeconomic agent-based models." University of Oxford DPhil thesis (2024),
  DOI `10.5287/ora-5reg8nv9g`.

The paper remains the equation index for Appendix A.1-A.142. The thesis is used
as the stronger source where it gives more explicit model structure, parameters,
initialization detail.

## Model Scope

The model represents one or more OECD-style national economies interacting with
the rest of the world. One simulation tick is one quarter. One simulated agent
represents `1000` real agents.

Agent categories:

- firms, operating in 18 NACE Rev. 2 level-1 sectors;
- individuals, classified as employed, unemployed, or inactive;
- households, containing individuals and holding income, wealth, debt, housing,
  consumption, and investment state;
- banks, supplying firm loans, household consumption loans, and mortgages;
- central banks, setting policy rates with a Taylor-rule structure;
- government accounts and government goods-market entities;
- properties, used in the housing sale and rental markets;
- rest-of-world agents, representing imports and exports.

Markets:

- goods market, matching firms, households, government entities, and rest of
  world buyers and sellers;
- labour market, matching firms with individuals;
- credit market, matching firms and households with banks;
- housing market, matching households with properties for sale or rent.

## Scheduler

Each quarterly tick runs the macroeconomy systems in a fixed order:

1. `aggregate_previous_state` computes previous-quarter aggregates and GDP
   identity inputs.
2. `refit_expectations` updates common macro expectations.
3. `target_setting` updates firm, individual, household, and government targets.
4. `labour_market` fires before hiring and updates employment.
5. `planning_and_production` sets policy rates, wages, production, prices, and
   preliminary goods/credit demand.
6. `housing_preclear` prepares tentative housing sale and rental matches.
7. `credit_market` clears firm loans, household consumption loans, and mortgages.
8. `housing_completion` executes housing transfers only when required mortgages
   are granted.
9. `goods_market` clears goods demand against firm supply.
10. `realised_accounting` updates stocks, balance sheets, GDP identities, debt,
    defaults, government accounts, and history.

The scheduler uses Syren `ModelBuilder`, `AgentTemplate`, environment state, and
message buffers. GPU support is not required for this example.

## Core Rules

Expectations: deterministic AR(1) models with
lag one are fit on log levels using data through `t-1`. The fixture uses synthetic
history; real-data mode is structured to use histories beginning at `2000-Q1`.
Sparse-history and missing-data edge cases remain a reported exact-replication
gap because the author policy is not specified.

Firms forecast demand and profits, set target production, determine labour,
intermediate, and capital needs, set wages and prices, request short- and
long-term loans when deposits are insufficient, produce with a Leontief
technology, sell goods, update inventory and stocks, and may go bankrupt.

Households receive individual income, social transfers, rental income, and
financial-asset income. They set consumption and investment demand, decide
whether to buy or rent housing, request consumption loans or mortgages when
needed, update wealth and debt, and may go bankrupt.

Banks hold reserves, deposits, loans, equity, and liabilities. They set deposit
and overdraft rates, supply credit subject to capital and borrower constraints,
allocate credit away from higher non-performing-loan categories, receive interest,
write off bad debt, and may become insolvent. The thesis ARDL/error-correction
interest-rate form is represented, but exact lag-grid and preprocessing choices
remain unresolved without author code or config.

The central bank uses the Taylor-rule structure from the paper and thesis. The
example exposes the transformed Taylor coefficients and keeps the policy-rate
state on the central-bank agent.

Government entities buy goods. The government account collects labour, corporate,
VAT, capital-formation, production, and export taxes; pays unemployment and other
benefits; and updates deficit and debt.

The rest of the world provides import supply and export demand using fixed
exchange-rate assumptions and the ROW indexing rules from the paper and thesis.

## Parameters Used

The default fixture parameters include specified values where available:

- `phi_GM = 2` for goods-market seller priority;
- `gamma_F = gamma_H = 1` for labour-market firing and hiring speeds;
- `rho_CAR = 0.08`, `rho_SR = 0.1`;
- firm credit ratios `rho_DtE = 1.0`, `rho_RoE = 0.15`,
  `rho_RoA = 0.05`;
- household consumption loan-to-income ratio `rho_LTI_C = 0.36`;
- `phi_CS = 2.0` for credit-supply allocation away from NPL categories;
- `omega_M = omega_K = 0.85`, represented through fixture stock initialization;
- capital installation delay `T_KD = 1`;
- `h_max = 1.5`;
- government entities equal 25 percent of domestic firms, rounded to at least
  one entity in the tiny fixture;
- housing parameters from thesis pp. 197-201, including `phi_HP = 42.90`,
  `beta_HP = 0.79`, `mu_HP = -0.018`, `sigma_HP = 0.17`, `mu_PS = 0.4`,
  `phi_B = 0.001`, `phi_HR = 17.22`, `beta_HR = 0.35`, `p_RS = 7/8`,
  `p_OS = 79/80`, `p_PM = 0.1964`, `mu_PM = 1.4531`, `sigma_PM = 0.4889`,
  `p_RM = 0.2848`, `mu_RM = 1.6559`, `sigma_RM = 0.7855`, and full CPI rent
  indexation with one-quarter lag.

## Initialization

Fixture mode constructs a tiny synthetic economy with the same agent categories,
state variables, loan book, histories, and market connections needed to exercise
the model.

Real-data mode is structured around the papers initialization pipeline:

- sample firms with replacement from Compustat and rescale to OECD sector and
  balance-sheet aggregates;
- sample banks with replacement from Compustat where available and rescale to
  IMF/OECD aggregates;
- draw households from ECB HFCS using survey weights and map income, assets,
  liabilities, tenure, rent, and consumption fields;
- link individuals to sampled households through HFCS ids and adjust employment
  status/sector to aggregate unemployment, vacancies, and employment by sector;
- generate properties from HFCS tenure and property-value fields;
- use linear sum assignment for firm-employee, firm-bank, household-bank, and
  household-property matching.

The required real-data sources are OECD ICIO and national accounts, IMF
quarterly macro and financial series, World Bank fiscal/unemployment/NPL series,
BIS policy rates, ECB HFCS microdata, Compustat firm and bank microdata, and ESRB
macroprudential mortgage measures.

## Exactness and Gap Analysis

Where the paper or thesis gives a unique equation or rule, this example is intended
to implement that equation or rule directly. The scheduler order, agent categories,
market sequence, explicit accounting equations, literal housing formulas, thesis
fixed parameters, log-level AR(1) expectation form, credit-market clearing order,
ascending-rate bank visits, and goods-market seller-priority formula are
implemented from the paper/thesis sources. The goods-market clearing pseudocode
uses the inherited Poledna et al. Online Appendix A.1.1 search-and-matching
algorithm because the Wiese model is based on that ABM lineage.

The example should not yet be described as an author-exact numerical replica of
the paper's forecasts. The public paper and thesis do not fully specify every
procedure needed for bit-for-bit replication of the authors' trajectories,
country-level initial microstates, or posterior forecast distributions. Those
unknowns are recorded as exact-replication gaps. They are not hidden deviations:
strict mode refuses to run while any gap remains, and normal fixture runs can
print the gap report.

Resolved or user-specified exactness policies:

| Item | Resolution | Exactness note |
| --- | --- | --- |
| `goods-flow-preservation-pseudocode` | Use the Poledna et al. Online Appendix A.1.1 rule: consumers are randomly ordered, visit domestic or foreign firms selling the requested good, seller probability averages normalized `exp(-phi_GM * price)` and normalized firm size, buyers move to remaining sellers when the preferred seller is short, and unmet demand becomes involuntary saving/excess demand. | Treated as source-resolved for this example because Wiese builds on the Poledna ABM. |
| `trajectory-exact-randomness` | Use a deterministic Python-like MT19937 stream with 53-bit uniform draws, unbiased Fisher-Yates shuffles, and seeded random/Bernoulli tie policy where exact ties remain after ranking. | This is a user-specified reproducibility policy, not an author-provided RNG convention. It should reproduce this implementation exactly, but it is not claimed to reproduce the authors' private random streams. |

| Gap | Affects | Missing detail | Why it matters | What would close it |
| --- | --- | --- | --- | --- |
| `ar1-missing-data-policy` | A.16-A.21 expectations | The thesis specifies deterministic AR(1) on log levels through `t-1`, but not how to handle missing, zero, negative, revised, or too-short histories. | Expectations drive production, prices, consumption, benefits, investment, and housing decisions. Different edge-case policies can alter paths for countries with sparse or problematic historical series. | User-approved edge-case policy, or author preprocessing code. |
| `ardl-lag-grid-and-preprocessing` | A.24 bank rates | The thesis gives the ARDL-derived error-correction equation and AIC selection, but not the candidate lag grid, transformations, residual handling, missing-data policy, or rate caps/floors. | Bank rates affect credit approval, debt service, mortgage affordability, firm financing, defaults, bank equity, and downstream production/housing outcomes. | Author ARDL estimation script, coefficient files, or a full per-loan-type estimation specification. |
| `credit-visit-limits` | A.12 credit market | The thesis confirms applicants sample random subsets of `nLF`/`nLH` banks and visit them by ascending offered rate, but does not give numeric values for `nLF` or `nLH`. | Visit limits change credit approval probability, rate competition, bank concentration, and the amount of unmet credit demand. | Author default config or parameter table giving `nLF` and `nLH`. |
| `author-initialization-tie-breaks` | initial microstate | The thesis describes sampling, rescaling, and linear-sum-assignment initialization, but not exact weighted-sampling implementation, solver choices, cost scaling, or tie-breaking. | Different microstates can preserve identical country aggregates while changing firm-worker, household-bank, bank-loan, and household-property networks. These networks influence later dynamics. | Author initialization scripts, initialized baseline microstates, or exact assignment/sampling/tie rules. |

Separately, real-data ingestion and posterior training require licensed/source
datasets and author-equivalent preprocessing. Those are data and fitting work,
not additional published-model equations.

## Running

Fixture run:

```bash
cargo run --release --features "model messaging" --example macroeconomy -- --fixture tiny --ticks 8 --seed 42
```

Fixture run with no final gap report:

```bash
cargo run --release --features "model messaging" --example macroeconomy -- --fixture tiny --ticks 8 --seed 42 --gap-report none
```

Fixture run with machine-readable gap output:

```bash
cargo run --release --features "model messaging" --example macroeconomy -- --fixture tiny --ticks 8 --seed 42 --gap-report json
```

Config-file run:

```bash
cargo run --release --features "model messaging" --example macroeconomy -- --fixture tiny --ticks 8 --seed 42 --config path/to/macroeconomy.cfg
```

Real-data mode currently fails fast unless the external assets and future provider
implementation are supplied:

```bash
cargo run --release --features "model messaging" --example macroeconomy -- --data-dir path/to/data --country AUT --initialisation 2013-Q1
```

## Output

Per-tick output is printed to stdout as CSV:

```text
tick,production,ppi,cpi,hpi,rpi,total_loans,gdp_gap,blocked_mortgages,excess_demand
```

The final summary and gap report are printed to stderr. The gap report can be
`text`, `json`, or `none`.

## Tests

Run the macroeconomy integration tests:

```bash
cargo test --features "model messaging" --test macroeconomy
```

Run all tests that are enabled by the model and messaging features:

```bash
cargo test --features "model messaging" --tests
```

The tests check paper equation coverage, thesis-informed parameters, scheduler
order, accounting identities, market ordering, AR(1) and ARDL helper shape,
configuration overrides, real-data fail-fast errors, and remaining-gap reporting.
