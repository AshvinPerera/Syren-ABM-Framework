# Macroeconomy Example

A multi-market macroeconomic agent-based model on the Syren ABM Framework,
structured after Wiese et al. It runs on **synthetic data only**. It is not a
replication of the authors' results and does not attempt to be: there is no
empirical calibration target, and the equation-coverage/gap-report machinery
that once tracked replication fidelity has been retired.

> **Status: under active reconstruction.** The scheduler, agent types, market
> structure, and equation library are in place, but several behaviours the
> source model specifies are missing or wrong, and the current output is
> economically degenerate (frozen HPI, zero RPI, no new credit). See
> [Known gaps](#known-gaps) below. The reconstruction plan targets ~2M agents.

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

The paper is the equation index for Appendix A.1-A.142. The thesis is used as
the stronger source where it gives more explicit model structure, parameters,
or initialization detail. Equation numbers are cited in comments throughout
`equations.rs` and `systems.rs`.

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
allocate credit away from higher non-performing-loan categories, and receive
interest. **Not yet implemented:** bad-debt write-off, the insolvency bail-in,
and the ARDL rate pass-through (the helpers exist in `forecasting.rs` but are
never called, so bank lending rates never move).

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

## Known gaps

The scheduler order, agent categories, market sequence, accounting equations,
thesis fixed parameters, log-level AR(1) expectations, credit-market clearing
order, ascending-rate bank visits, and the goods-market seller-priority formula
are implemented from the sources. Goods-market clearing uses the inherited
Poledna et al. Online Appendix A.1.1 search-and-matching algorithm, since the
Wiese model builds on that ABM lineage.

These behaviours are **specified by the source but not implemented**, and each
one is visible in the output:

| Gap | Consequence in a live run |
| --- | --- |
| No bad-debt write-off | Loans of bankrupt borrowers accrue interest forever while the borrower's debt fields are zeroed. Bank equity and NPL buckets are wrong; stock-flow consistency is violated. |
| No firm replacement after bankruptcy | Bankrupt firms become permanent zombies; the productive sector drains over a long run. |
| No bank bail-in (eq 6.45) | Insolvency sets a flag and nothing else happens. |
| ARDL pass-through never invoked | Bank lending rates are frozen `Bank::default` constants, so the Taylor rule has no transmission channel at all. |
| Property revaluation only on sale | `property.value` is written at exactly one site. Untraded properties never revalue, so HPI is frozen. |
| Housing price/rent reduction formula | Both `HousingReductionPolicy` branches are wrong. The default applies a ~98% haircut per reduction, which is why RPI reads `0.000000`. |

These parameters are wired but left at zero, which switches the corresponding
channel off: capital depreciation, wage tightness sensitivity, `phi_dp`/`phi_cp`
price markups, and the input-output / net-fixed-assets matrices (diagonal only,
so there is no cross-sector linkage across the 18 sectors).

The fixture is also not internally consistent: every firm in it fails the
model's own ROA credit screen (`borrower_credit_cap`), so no firm can ever
borrow. That is why `total_loans` declines monotonically.

Real-data ingestion and posterior calibration are out of scope. `RealDataProvider`
still fails fast naming the datasets it would need, and will be removed when the
synthetic population generator lands.

## Running

Fixture run:

```bash
cargo run --release --features "model messaging" --example macroeconomy -- --fixture tiny --ticks 8 --seed 42
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

The completion summary is printed to stderr.

## Tests

Run the macroeconomy integration tests:

```bash
cargo test --features "model messaging" --test macroeconomy
```

Run all tests that are enabled by the model and messaging features:

```bash
cargo test --features "model messaging" --tests
```

The tests check thesis-informed parameter defaults, scheduler order, market
ordering, AR(1)/Taylor/ARDL helper shapes, named equation arithmetic,
configuration overrides, and reproducibility across thread counts.
