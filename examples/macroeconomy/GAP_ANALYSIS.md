# Macroeconomy Example: Gap Analysis vs. Wiese (2024)

This document is an implementation specification for closing every divergence between
`examples/macroeconomy/` and Wiese, S. (2024), *Dynamic Interactions in Economics:
From Micro-Level Games to Macroeconomic Agent-Based Models*, University of Oxford
DPhil thesis (Chapter 6, equations 6.1–6.146; Chapter 7 calibration & forecasting).
The companion arXiv preprint (Wiese et al. 2024) renumbers Chapter 6's equations
as Appendix A.1–A.142.

It is written for an autonomous code agent. Every gap is given a stable identifier,
the source of the specification (with thesis equation number and page or paper
appendix label), the location in the implementation, the concrete change required,
and a solvability classification. Gaps are ordered by recommended implementation
sequence: bug-shaped defects first, structural omissions second, data and
calibration assets last.

## How to use this document

- Each gap has a **gap-id** of the form `WIESE-NNN`. Reference these IDs in commit
  messages, PRs, and `coverage.rs::blocker_log` entries.
- **Solvability** is one of:
  - **C** — solvable by editing files in `examples/macroeconomy/`; no external
    asset required.
  - **P** — solvable when a parameter table or single number is provided.
  - **D** — solvable when an external dataset is ingested.
  - **A** — solvable when the authors' calibration code or trained networks are
    obtained.
  - **U** — not solvable from publicly available sources.
- Every gap also lists **affects** (downstream behavior altered) and
  **acceptance test** (a check the agent should add or update).
- File paths are repository-relative. Line numbers reference the state of the
  code as analyzed; treat them as anchors and re-confirm before editing.

## Sequencing

Implement in this order. Later gaps assume earlier ones are closed.

1. Phase I — Defect-shaped fixes (A-block): WIESE-001, 002, 003, 004, 005.
2. Phase II — Coverage and self-report integrity: WIESE-006, 007, 008.
3. Phase III — Behavioral simplifications: WIESE-009 through WIESE-020.
4. Phase IV — Equation factoring & test scaffolding: WIESE-021 through WIESE-030.
5. Phase V — Parameter and policy completeness: WIESE-031 through WIESE-040.
6. Phase VI — Real-data ingestion and calibration: WIESE-041 through WIESE-050.

---

## Phase I — Defect-shaped fixes

These are direct contradictions between the paper and the implementation. Fix
these first; they are bugs, not deliberate simplifications.

### WIESE-001 — Unemployment-benefit update has a wrong floor (eq 6.97)

- **Source**: Thesis §6.4.4.3, equation 6.97:
  `wU(t) = wU(t-1) / (1 + γ(t))`. The benefit shrinks when growth is positive
  and grows when growth is negative; there is no floor in the paper rule.
- **Code**: `examples/macroeconomy/systems.rs` ~line 793 in
  `planning_and_production_system`:
  ```rust
  account.unemployment_benefit *=
      (1.0 / (1.0 + state.forecast.predicted_growth)).max(1.0);
  ```
  The `.max(1.0)` clip prevents the multiplier from ever being below 1, breaking
  the rule whenever growth is positive.
- **Fix**: Remove the `.max(1.0)` clamp:
  ```rust
  account.unemployment_benefit *= 1.0 / (1.0 + state.forecast.predicted_growth);
  ```
- **Solvability**: **C**.
- **Affects**: government social-spending trajectory, household income for
  unemployed individuals, demand path, GDP identity.
- **Acceptance test**: Add a unit test in `tests/macroeconomy.rs` that runs one
  tick with `predicted_growth = 0.05` and asserts
  `unemployment_benefit ≈ unemployment_benefit_prev / 1.05` (not equality).

### WIESE-002 — No firm replacement after bankruptcy

- **Source**: Thesis §6.4.3 introduction:
  *"firms can go bankrupt and exit the market; for simplicity, a new firm
  immediately replaces the bankrupt one."*
- **Code**: `examples/macroeconomy/systems.rs` ~line 1652 sets
  `firm.bankrupt = true` and zeroes `deposits`, `short_debt`, `long_debt`,
  `overdraft`. The firm's row is never reseeded; subsequent ticks treat it as a
  zombie agent producing nothing.
- **Fix**:
  1. Add a helper `fn respawn_firm(firm: &mut Firm, params: &CountryParameters,
     rng: &mut PythonLikeRng, sector: u8, country: u16)` in `systems.rs` that
     resets every Firm field to a fresh-firm state. In fixture mode this should
     mirror the values used in `data::tiny_fixture`. In real-data mode it should
     redraw from the (yet-to-be-implemented, see WIESE-041) Compustat sample,
     conditional on the same sector.
  2. After the bankruptcy block, immediately call `respawn_firm` and clear
     `bankrupt = false`. Keep `firm.id` stable so downstream messaging still
     resolves the entity.
  3. Update the matching `Loan` rows in `state.loan_book` for the bankrupt firm
     (see WIESE-004).
- **Solvability**: **C** in fixture mode; **C+A** in real-data mode (needs the
  authors' Compustat resample script for replication-grade behavior).
- **Affects**: aggregate production after bankruptcies, sector composition,
  bank loan books, employment dynamics.
- **Acceptance test**: Force one firm into bankruptcy via large negative
  productivity shock, advance one tick, assert `firm.production > 0` again and
  `firm.bankrupt == false`.

### WIESE-003 — No bank bailin loop (eq 6.45)

- **Source**: Thesis §6.4.1.3 "Insolvency", equation 6.45: an insolvent bank's
  liabilities are absorbed by other banks until its equity equals the average
  equity of solvent banks.
- **Code**: `examples/macroeconomy/systems.rs:1779` only sets
  `bank.insolvent = bank.equity / (bank.liabilities + positive_part(bank.reserves)) < ρ_SR`.
  No equity transfer happens. The `Bank` struct
  (`components.rs:238-290`) has no `equity_injection` field or analogue, and
  the profit equation 6.41 is missing the `I_b(t)` injection term.
- **Fix**:
  1. Add `equity_injection: f64` to `Bank` and reset to 0 each tick.
  2. After the existing solvency loop in `realised_accounting_system`, run:
     ```rust
     let solvent: Vec<usize> = banks.iter().enumerate()
         .filter(|(_, b)| !b.insolvent).map(|(i, _)| i).collect();
     if !solvent.is_empty() {
         let avg_equity = solvent.iter().map(|&i| banks[i].equity).sum::<f64>()
             / solvent.len() as f64;
         for i in 0..banks.len() {
             if banks[i].insolvent {
                 let needed = (avg_equity - banks[i].equity).max(0.0);
                 if needed > 0.0 {
                     let per_solvent = needed / solvent.len() as f64;
                     for &j in &solvent {
                         banks[j].equity -= per_solvent;
                         banks[j].equity_injection += per_solvent;
                     }
                     banks[i].equity = avg_equity;
                     banks[i].insolvent = false;
                 }
             }
         }
     }
     ```
  3. Subtract `equity_injection` from each bank's profit when computing eq 6.41
     (paper's `I_b(t)` term).
- **Solvability**: **C**.
- **Affects**: bank equity distribution, system-wide credit supply after stress,
  contagion dynamics.
- **Acceptance test**: Construct a fixture with two banks, force one to negative
  equity, advance one tick, assert both have equal equity afterwards and the
  formerly insolvent bank has `insolvent == false`.

### WIESE-004 — No bad-debt write-off and dual-bookkeeping divergence

- **Source**: Paper bank rules (§6.4.1) require banks to recognize losses on
  loans whose borrowers go bankrupt. README claims this happens; no code
  implements it.
- **Code**: `grep "write_off\|bad debt" systems.rs` returns nothing.
  `settle_loan_book` (`systems.rs:2287-2365`) only amortizes by maturity;
  it does not consult borrower bankruptcy state. When firm or household
  bankruptcy zeroes the borrower's debt fields (`systems.rs:1652, 1703`), the
  matching `Loan` rows in `state.loan_book.loans` continue to accrue interest
  forever.
- **Fix**:
  1. Before `settle_loan_book`, walk `state.loan_book.loans` and for each loan
     whose borrower is now bankrupt:
     - Record the outstanding amount in the bank's NPL bucket
       (`bank.npl_firm_by_sector[sector] += outstanding` or
       `npl_consumption`/`npl_mortgage` as appropriate).
     - Deduct `outstanding` from `bank.equity` (loss recognition).
     - Set `loan.outstanding = 0`.
  2. After the walk, the existing `retain` filter in `settle_loan_book` will
     remove zeroed loans.
  3. Reset bank loan-volume aggregates after this pass so eq 6.42 / 6.43 use
     post-write-off numbers.
- **Solvability**: **C**.
- **Affects**: bank equity, NPL ratios that feed eq 6.37 credit-supply
  allocation, future credit availability, downstream production.
- **Acceptance test**: Force one firm bankrupt with outstanding `Loan` of
  principal 100, run one tick, assert `bank.equity` decreased by ~100 and the
  loan is removed from `loan_book`.

### WIESE-005 — Bank lending rates never update at runtime

- **Source**: Thesis §6.4.1.3 specifies an ARDL/error-correction pass-through
  from the policy rate to bank loan rates. Helper functions
  `ardl_error_correction_delta_rate` and `select_ardl_lag_by_aic` exist in
  `forecasting.rs:9-150`.
- **Code**: `grep "ardl\|ARDL" systems.rs` returns nothing. The four bank rate
  fields (`short_firm_rate`, `long_firm_rate`, `household_rate`,
  `mortgage_rate`) are initialized in `Bank::default` and never modified. They
  flow into `bank_credit_supply` and credit-market clearing as constants.
  `coverage.rs:85-91` self-reports this as `ardl-lag-grid-and-preprocessing`,
  but understates the problem: it is not just an unspecified lag grid; the
  ARDL update is never invoked.
- **Fix**:
  1. After the Taylor rule block in `planning_and_production_system` (~line
     520), iterate over banks. For each bank and each rate type, call
     `ardl_error_correction_delta_rate` with:
     - `current_rate` = the bank's current rate for that loan class,
     - `policy_rate` = the freshly updated central-bank policy rate,
     - `phi_ec, phi_lr, alpha_p, alpha_q` = coefficients sourced from the
       `CalibrationParameters` (add a per-loan-type coefficient block to
       `calibration.rs`),
     - `lag_history` = a deque of past policy-rate observations stored in the
       `Bank` struct (add `policy_rate_lag_buffer: [f64; 8]`).
  2. Apply rate floor at zero and a configurable cap.
  3. Add a `RatePassThroughCoefficients` config struct in
     `state.rs::CountryParameters` so users can override defaults from
     `*.cfg` files (use `apply_config_str`).
  4. Until author coefficients are obtained, ship a documented placeholder set
     (e.g. `phi_ec = -0.3, phi_lr = 1.0`) and gate strict mode on the user
     supplying a non-placeholder set.
- **Solvability**: **C** for the wiring; **P+A** for the empirical coefficients.
- **Affects**: credit approval probabilities (eqs 6.32–6.35), debt service
  costs, household and firm cash flow, defaults, downstream production and
  housing decisions.
- **Acceptance test**: Initialize fixture with `policy_rate = 0.01`, advance
  three ticks while gradually moving `policy_rate` to 0.03, assert
  `bank.short_firm_rate` increased monotonically and the steady-state pass-
  through approaches `phi_lr * (0.03 - 0.01)`.

---

## Phase II — Coverage and self-report integrity

### WIESE-006 — Coverage tracker stops at A.11; A.12 (credit) and A.13 (housing) have no equations mapped

- **Source**: Thesis runs 6.1–6.146; paper renumbers as A.1–A.142.
- **Code**: `coverage.rs:243-258` `appendix_for_equation` only matches up to
  `141..=142 => "A.11 labour market"`. `appendix_coverage()` separately lists
  `A.12 credit market` and `A.13 housing market` as `Implemented`, but no
  equation in the 1..=142 range is mapped to them. The credit-market
  equations (6.31–6.45 in §6.4.1.3 / §6.5.3) currently live in the `A.3 banks`
  bucket; the housing-market equations (6.108–6.131) currently live in the
  `A.7 households` bucket.
- **Fix**:
  1. Audit which thesis equations are credit-market behavior vs. bank balance
     sheet, and which are housing-market clearing vs. household state.
  2. Reassign appendix mappings in `appendix_for_equation` to populate A.12 and
     A.13 properly. The likely correct split is: credit-market clearing rules
     under A.12, housing decision/clearing under A.13, with banks (A.3) and
     households (A.7) keeping only the agent-state equations.
  3. Update `note_for_equation` and `source_for_equation` accordingly.
  4. Update `tests/macroeconomy.rs` coverage assertions.
- **Solvability**: **C**.
- **Affects**: gap report accuracy, paper-equation traceability.
- **Acceptance test**: `EquationCoverage::missing_equations()` returns empty
  AND each of the 13 appendix labels has at least one equation mapped to it.

### WIESE-007 — Equations 6.143–6.146 not represented in coverage map

- **Source**: Thesis equations 6.143 (net-export identity), 6.144 (goods-market
  seller priority), 6.145 (firing speed γ_F), 6.146 (hiring speed γ_H).
- **Code**: `coverage.rs::paper_aligned()` builds entries 1..=142 only. The
  underlying behavior is implemented (GDP identity in `systems.rs:1825`,
  goods market in goods_market_system, labour market in
  `systems.rs:386-460`), just not tracked.
- **Fix**: Either extend the coverage map to 1..=146 with the thesis numbering
  preserved, or add a parallel `thesis_appendix_for_equation` that runs in
  thesis numbering. Document the paper↔thesis offset table in the README's
  exactness section.
- **Solvability**: **C**.
- **Affects**: traceability only.
- **Acceptance test**: New test asserts every thesis equation 6.1–6.146 maps
  to a non-empty appendix label and a non-default note.

### WIESE-008 — Replication-blocker list is incomplete

- **Source**: This document.
- **Code**: `coverage.rs::blocker_log()` lists only four blockers
  (ar1-missing-data-policy, ardl-lag-grid-and-preprocessing,
  credit-visit-limits, author-initialization-tie-breaks).
- **Fix**: Add `ReplicationBlocker` entries for at least:
  - `firm-replacement-rule` (until WIESE-002 lands)
  - `bank-bailin-rule` (until WIESE-003 lands)
  - `loan-write-off-rule` (until WIESE-004 lands)
  - `bank-rate-update-rule` (until WIESE-005 lands)
  - `government-benefit-floor` (until WIESE-001 lands)
  - `firm-replacement-resample-source` (kept post-002 if real-data mode is
    not yet author-equivalent)
  - `linear-sum-assignment-solver` (until WIESE-046 lands)
  - `npe-posterior-table` (until WIESE-049 lands)
  Each blocker entry should reference the corresponding `WIESE-NNN` id in this
  document.
- **Solvability**: **C**.
- **Affects**: gap report transparency.
- **Acceptance test**: Once WIESE-001..005 are closed, the corresponding
  blockers are removed; remaining blockers correspond 1:1 to outstanding
  WIESE ids.

---

## Phase III — Behavioral simplifications

These diverge from the paper but are not necessarily defects; they are silent
simplifications that should either be fixed or surfaced in `blocker_log`.

### WIESE-009 — Property-level housing forecasts shared across all households

- **Source**: Thesis §6.4.5.3 eqs 6.108–6.111: each household evaluates the
  rent vs. buy decision using its own predicted income trajectory, financial
  assets, and house-price expectation.
- **Code**: `Property` (`components.rs:374-418`) carries
  `predicted_annual_buy_price`, `predicted_annual_rent_price`,
  `predicted_rental_yield` as a single value, applied identically to every
  household considering that property.
- **Fix**: Move forecast calculation into `housing_preclear_system` as a
  per-(household, candidate property) pair. The `Property` struct can keep its
  fields for diagnostic display, but the values used in
  `buy_probability_a110(...)` should be computed inside the household loop
  using the household's own `predicted_income`, `deposits`, and
  `state.forecast.predicted_hpi_inflation`.
- **Solvability**: **C**.
- **Affects**: housing market clearing distribution, mortgage demand, HPI/RPI.
- **Acceptance test**: Two households differing only in `predicted_income`
  facing identical properties produce different buy probabilities.

### WIESE-010 — Government consumption is scalar, not sectoral

- **Source**: Thesis §6.2.4 and §6.4.4: government consumption is allocated
  across NACE Rev. 2 sectors via the `government_consumption_weights[SECTORS]`
  vector (paper uses COICOP→ISIC mapping).
- **Code**: `GovernmentEntity` (`components.rs:292-299`) has scalar
  `target_consumption` and `realised_consumption`. Sector weight is consulted
  only when emitting goods-market demand messages, not stored on the entity.
- **Fix**: Add `target_consumption_by_sector: [f64; SECTORS]` and
  `realised_consumption_by_sector: [f64; SECTORS]`. Compute targets in
  `target_setting_system` using `state.params.government_consumption_weights`
  and the GDP target. Emit one `GoodsDemand` message per sector.
- **Solvability**: **C**.
- **Affects**: goods market sectoral allocation, sector PPI, sectoral
  production paths.
- **Acceptance test**: Sum of `target_consumption_by_sector` equals
  `target_consumption`; weighted distribution matches
  `government_consumption_weights`.

### WIESE-011 — ROW prices and weights are static

- **Source**: Thesis §6.4.7.3 eqs 6.138–6.142: ROW import/export sector
  weights index to a global aggregate price; prices update each tick.
- **Code**: `RestOfWorld.sector_prices` and `*_weights`
  (`components.rs:430-451`) are set in `Default` and never updated. The
  `planning_and_production_system` ROW block consumes them as constants.
- **Fix**:
  1. After computing `state.aggregates.ppi` and per-sector PPI, update each
     ROW agent's `sector_prices[s]` from a configurable indexation rule
     (e.g. equal to domestic sector PPI, or a configurable convex combination
     of own-country PPI and foreign anchor).
  2. Update `import_weights` and `export_weights` only at initialization from
     OECD data (real-data mode); leave them static within a run.
- **Solvability**: **C** for indexation rule; **D** for empirical weights.
- **Affects**: import-side and export-side prices, GDP deflator, trade
  balance.
- **Acceptance test**: After three ticks of positive PPI inflation, the
  ROW agent's sector prices have moved by approximately the same factor.

### WIESE-012 — Mortgage maturity hard-coded; not surfaced in config

- **Source**: Thesis §6.4.1.2 fixes mortgage maturity at 25 years (100
  quarters). Country-specific overrides are mentioned in the calibration
  discussion.
- **Code**: `state.rs:290` sets `mortgage_maturity_quarters: 100`, correct as
  default. Not currently overridable through `apply_config_file`.
- **Fix**: Add a `mortgage_maturity_quarters` parser key to
  `config.rs::apply_config_str` and round-trip via `tests/macroeconomy.rs`.
- **Solvability**: **C**.
- **Affects**: mortgage DSTI computation (eq 6.35), monthly payment
  amortization, household debt service.
- **Acceptance test**: `apply_config_str` with `mortgage_maturity_quarters=80`
  yields `state.params.mortgage_maturity_quarters == 80`.

### WIESE-013 — Capital depreciation rate defaults to 0.0

- **Source**: Thesis §6.4.3.1 "Stocks": capital depreciates each quarter at a
  sector-specific rate sourced from OECD net fixed assets.
- **Code**: `state.rs:257`:
  `capital_depreciation_rate_by_sector: [0.0; SECTORS]`. With zero, the
  capital evolution equation 6.86 reduces to pure accumulation; capital never
  depreciates.
- **Fix**:
  1. Set fixture default to a sector-uniform `0.025` (per-quarter, ≈10%/yr).
  2. Document in README that this is a placeholder until OECD net-fixed-asset
     data is ingested (see WIESE-043).
  3. Wire `capital_depreciation_rate_by_sector` into `apply_config_str`.
- **Solvability**: **C** for placeholder; **D** for empirical sectoral rates.
- **Affects**: firm capital stocks, target investment, GDP/investment ratio.
- **Acceptance test**: Capital decays geometrically when target investment is
  zero.

### WIESE-014 — Wage tightness sensitivity defaults to 0.0

- **Source**: Thesis §6.4.3.3 "Wage Adjustments": wages respond to local
  labour-market tightness via parameter `phi_W`.
- **Code**: `state.rs:269` sets `wage_tightness_sensitivity: 0.0`. With zero
  the wage rule degenerates to pure indexation.
- **Fix**: Set fixture default to a non-zero placeholder (e.g. 0.5) and
  surface in `config.rs`.
- **Solvability**: **P**.
- **Affects**: wage dynamics, household income, household consumption.
- **Acceptance test**: A firm with `target_labour > current_labour` raises its
  wage proportionally to the gap.

### WIESE-015 — Pricing markup parameters φ_DP = φ_CP = 0 in default Austria calibration

- **Source**: Thesis Table 7.1 priors and §7.2 NPE posteriors give non-zero
  values for `phi_dp` and `phi_cp`.
- **Code**: `calibration.rs::austria_npe_table4()` returns
  `phi_dp = 0.0, phi_cp = 0.0`. With both zero, eq 6.73 prices follow only
  inflation forecast.
- **Fix**: Replace the placeholder zeros with the paper's posterior means for
  Austria when published; document the source. Until then, mark this entry
  as a `ReplicationBlocker`.
- **Solvability**: **P+A**.
- **Affects**: firm price paths, PPI, CPI, real production.
- **Acceptance test**: A firm facing rising demand and rising costs raises
  prices when `phi_dp, phi_cp > 0`.

### WIESE-016 — Linear-sum-assignment solver not implemented

- **Source**: Thesis §6.4.1.1, §6.4.3.2, §6.4.5.4, §6.4.6.4 specify LSA
  matching for firm↔employee, firm↔bank, household↔bank, household↔property.
- **Code**: No call site for any LSA solver anywhere in the repo. Real-data
  initialisation is a stub (see WIESE-041).
- **Fix**:
  1. Add `lapjv = "..."` (or `pathfinding::kuhn_munkres`) to `Cargo.toml`
     under a `real_data` feature so fixture builds stay light.
  2. Implement `fn lsa_assign(cost: &[Vec<f64>]) -> Vec<usize>` in
     `data.rs`.
  3. Use it in the four matching contexts when real-data initialization is
     wired in.
- **Solvability**: **C**.
- **Affects**: initial microstate networks; downstream credit, employment,
  housing dynamics depend on these networks.
- **Acceptance test**: A unit test feeds a small cost matrix and checks the
  solver returns the optimal assignment.

### WIESE-017 — Sectoral input-output matrix is identity-with-noise

- **Source**: Thesis §6.4.3.1 "Sector-specific Weights": `m_{ss'}` taken from
  OECD Inter-country Input-Output (ICIO) tables.
- **Code**: `state.rs:248-252` sets diagonal entries to 0.10 and off-diagonal
  to 0. The result is no real cross-sector linkage.
- **Fix**: Provide a fixture-only `synthetic_io_matrix(seed)` that produces a
  mildly off-diagonal stochastic matrix (e.g. row-stochastic with 0.5 on
  diagonal, 0.5 spread across sectors). For real-data mode, wire OECD ICIO
  ingestion (see WIESE-043).
- **Solvability**: **C** for fixture; **D** for real-data.
- **Affects**: firm intermediate-input demand, sector cross-spillovers,
  goods-market clearing.
- **Acceptance test**: For non-fixture seed, the IO matrix has at least one
  non-zero off-diagonal entry per row.

### WIESE-018 — Net-fixed-assets matrix is identity-with-noise

- **Source**: Same as above; `k_{ss'}` from OECD net-fixed-assets data.
- **Code**: `state.rs` initializes `net_fixed_assets_matrix` and
  `capital_compensation_matrix` with diagonal 0.03 only.
- **Fix**: Same approach as WIESE-017.
- **Solvability**: **C** + **D**.
- **Affects**: firm capital demand by sector, investment composition.
- **Acceptance test**: Mirror of WIESE-017 for the capital matrix.

### WIESE-019 — RNG family is non-configurable

- **Source**: Authors' code uses NumPy's default PCG64; this implementation
  uses MT19937 with Python-style 53-bit doubles.
- **Code**: `state.rs:79-144` defines `PythonLikeRng` (MT19937).
  `MacroEnvironment.rng` is hard-typed to it.
- **Fix**:
  1. Define a trait `MacroRng` with the methods used elsewhere (`next_u32`,
     `unit_f64`, `below`, `bernoulli`, `shuffle`, `normal_f64`).
  2. Implement it for both `PythonLikeRng` and a `Pcg64Rng` (from `rand`'s
     `rand_pcg`).
  3. Make `MacroEnvironment.rng` `Box<dyn MacroRng>` or a generic.
  4. Add a config key to choose the RNG family.
- **Solvability**: **P** (after C wiring).
- **Affects**: every random draw — labour market shuffles, goods market
  ordering, housing reductions, AR(1) error draws if any. Determinism is
  preserved within a chosen family.
- **Acceptance test**: Same seed, same family, two runs produce identical
  trajectories; different families produce different but reproducible
  trajectories.

### WIESE-020 — Goods-market clearing inherits Poledna pseudocode rather than a Wiese rule

- **Source**: Thesis §6.5.1 references the Poledna et al. (2023) ABM lineage
  but does not exhaustively specify a search-and-match algorithm. The README
  endorses the Poledna A.1.1 algorithm as the resolution.
- **Code**: `goods_market_system` implements the Poledna algorithm.
- **Fix**: Document this divergence explicitly in `coverage.rs::blocker_log`
  as `goods-flow-preservation-pseudocode` (the README already mentions this
  as resolved). Promote it from prose to a tracked `ReplicationBlocker` so it
  appears in the gap report. Optional: add a `GoodsClearingPolicy` variant
  for an alternative author-specified rule, behind a feature flag.
- **Solvability**: **C** for tracking; **U** for an alternative rule unless
  the authors publish one.
- **Affects**: goods-market matching distribution, excess-demand statistics.
- **Acceptance test**: Gap report includes
  `goods-flow-preservation-pseudocode`.

---

## Phase IV — Equation factoring & test scaffolding

### WIESE-021 — Refactor inline equations into `equations.rs`

- **Source**: This document, motivated by testability.
- **Code**: `equations.rs` currently exposes ~30 `// A.x` functions. Many
  paper equations are inlined in `systems.rs` (Taylor 6.46, wage rules
  6.69–6.72, firm accounting 6.86–6.94, government accounting 6.99–6.101,
  household accounting 6.116–6.131, individual accounting 6.132–6.137, ROW
  6.138–6.142, market rules 6.144–6.146).
- **Fix**: For each inlined equation, extract a `// 6.x` (and `// A.y` if it
  has a paper number) tagged function in `equations.rs`. Each function takes
  primitive inputs and returns a primitive output; no agent struct
  references. Replace the inline expression in `systems.rs` with a call to
  the extracted function.
- **Solvability**: **C**.
- **Affects**: testability, traceability, regression isolation.
- **Acceptance test**: Add a property-style unit test per extracted
  equation in `tests/macroeconomy.rs`.

### WIESE-022 — Add per-equation paper-form regression tests

- **Source**: This document.
- **Code**: `tests/macroeconomy.rs` exists but does not exercise every
  equation against a known input/output pair.
- **Fix**: For each equation extracted in WIESE-021, add a test with a
  hand-computed expected value derived from the thesis form. Include edge
  cases: zero, negative, very large, NaN.
- **Solvability**: **C**.
- **Affects**: regression safety as later phases are implemented.

### WIESE-023 — Add scheduler-order test

- **Source**: Thesis §6.2.3 prescribes the phase ordering.
- **Code**: Unclear whether the existing tests exercise phase ordering
  beyond the `MarketAudit::phase_log` field.
- **Fix**: Add a test that runs one tick and asserts
  `state.audit.phase_log == ["aggregate_previous_state", "refit_expectations",
  "target_setting", "labour_market", "planning_and_production",
  "housing_preclear", "credit_market", "housing_completion", "goods_market",
  "realised_accounting"]`.
- **Solvability**: **C**.

### WIESE-024 — Add accounting identity invariant

- **Source**: Thesis eq 6.19 GDP triple identity (output = expenditure =
  income).
- **Code**: `accounting.rs::GdpIdentity::held()` checks within tolerance but
  is only invoked in tests, not as a runtime invariant.
- **Fix**: After `realised_accounting_system` writes
  `state.aggregates.gdp`, assert `state.aggregates.gdp.held(epsilon)` in
  debug builds. In release builds, log a warning when it fails. Add a
  `MarketAudit::gdp_identity_violations: u32` counter.
- **Solvability**: **C**.
- **Affects**: confidence that subsequent edits do not break stock-flow
  consistency.
- **Acceptance test**: A test that intentionally introduces a leak (e.g.
  forgets to subtract imports in expenditure) makes the assertion fire.

### WIESE-025 — Verify firm income aggregation includes all five sources (eq 6.108)

- **Source**: Thesis eq 6.108: household income = wage + unemployment benefits
  + other benefits + rental income + other financial-asset income.
- **Code**: Trace through `planning_and_production_system` and
  `realised_accounting_system`; the agent's wide survey reports this is not
  fully verified.
- **Fix**: Confirm each of the five income components is summed into
  `Household.income`. If any term is missing, add it. Specifically check that
  rental income for landlord households (those owning rented properties) is
  computed and added.
- **Solvability**: **C**.
- **Affects**: household consumption, savings, mortgage affordability.
- **Acceptance test**: A landlord household's income strictly exceeds an
  identical non-landlord household's income, all else equal.

### WIESE-026 — Verify consumption smoothing (eqs 6.104–6.106)

- **Source**: Thesis eqs 6.104–6.106 specify habit/floor formulation with
  `phi_consumption_history`.
- **Code**: `state.rs:271` sets `phi_consumption_history: 1.0`. Trace
  through `planning_and_production_system` for the consumption-target
  computation.
- **Fix**: Confirm the smoothing parameter is consumed and the floor is
  enforced. If the floor is missing, add it.
- **Solvability**: **C**.

### WIESE-027 — Verify credit-rationing feedback into consumption/mortgage targets (eqs 6.117–6.118)

- **Source**: Thesis eqs 6.117–6.118: when granted credit < desired credit,
  the household scales down consumption and investment proportionally.
- **Code**: Trace through `housing_completion_system` and
  `goods_market_system` to confirm the feedback exists. The
  `Household.consumption_gap_after_financial_assets` field (`components.rs:189`)
  suggests partial implementation.
- **Fix**: Confirm scaling matches eqs 6.117–6.118 quantitatively. If only
  binary clipping is in place, replace with proportional scaling.
- **Solvability**: **C**.

### WIESE-028 — Verify firm idiosyncratic-growth condition (eq 6.59)

- **Source**: Thesis eq 6.59 applies idiosyncratic growth `γ_f(t)` only when
  the firm "faces an inventory overhang"; the precise condition is in prose.
- **Code**: `equations.rs::idiosyncratic_growth_a59` takes a boolean
  `applies` parameter; its caller (in `target_setting_system`) decides the
  condition.
- **Fix**: Confirm the caller's condition is "inventory > previous demand"
  or whatever the thesis prose specifies. Document the exact threshold in a
  comment with a thesis page reference.
- **Solvability**: **C**.

### WIESE-029 — Verify ROW indexation actually fires

- **Source**: Thesis §6.4.7 and eqs 6.140–6.142.
- **Code**: `RestOfWorld.adjustment_speed` is in the struct; trace through
  `planning_and_production_system`'s ROW block to confirm it is consumed.
- **Fix**: If the ROW block does not actually update `target_exports` /
  `target_imports` using `adjustment_speed`, fix it.
- **Solvability**: **C**.
- **Affects**: trade balance, GDP composition.

### WIESE-030 — Verify rent-indexation lag handling

- **Source**: Thesis §6.4.5.3 specifies CPI rent indexation with one-quarter
  lag.
- **Code**: `systems.rs:877` does
  `property.rent *= 1.0 + state.params.rent_partial_indexation_phi * cpi_lag;`
  where `cpi_lag` comes from `systems.rs:2067-2074`.
- **Fix**: Confirm `cpi_lag` is the ratio of CPI at `t-lag` to CPI at
  `t-lag-1` minus 1, not a simple level. Add a unit test.
- **Solvability**: **C**.
- **Affects**: rental price index, household disposable income for renters.

---

## Phase V — Parameter and policy completeness

### WIESE-031 — Country-specific tax rates from OECD

- **Source**: Thesis §6.4.4.2: VAT, income tax, corporate tax, social
  insurance rates pulled from OECD per country.
- **Code**: `GovernmentAccount::default()` uses single set of rates for all
  countries.
- **Fix**: Add a `country_tax_table.toml` (or .json) with per-country
  overrides; load in `data.rs::tiny_fixture` or `RealDataProvider::load`.
- **Solvability**: **D**.

### WIESE-032 — Country-specific ESRB macroprudential mortgage parameters

- **Source**: ESRB macroprudential mortgage measures referenced in
  `data.rs:108` for `mortgage_ltv`, `mortgage_lti`, `mortgage_dsti`.
- **Code**: `state.rs:284-286` ships single defaults.
- **Fix**: Add ESRB ingestion under the `real_data` feature; populate
  `CountryParameters` per country.
- **Solvability**: **D**.

### WIESE-033 — Country-specific Taylor rule coefficients

- **Source**: Thesis eqs 6.46–6.50: ρ, ξ_π, ξ_γ, r* estimated per country
  via OLS on BIS/IMF data.
- **Code**: `CentralBank::default` sets `rho=0.8, xi_pi=1.5, xi_gamma=0.5,
  natural_rate=0.01` for all countries.
- **Fix**: Add Taylor estimation in `forecasting.rs` (the helper
  `transform_taylor_rule` exists). Wire BIS/IMF data ingestion. Use
  estimated coefficients per country during real-data initialization.
- **Solvability**: **C** for estimation logic; **D** for inputs.

### WIESE-034 — Country-specific inflation targets

- **Source**: Thesis §6.4.2.2: inflation target is the central bank's
  declared target (often 2% but not universally).
- **Code**: `inflation_target: 0.02` for all countries.
- **Fix**: Add per-country override in country parameters table.
- **Solvability**: **P**.

### WIESE-035 — Credit-visit limits as paper values when known

- **Source**: Thesis §6.5.3 confirms the random-subset, ascending-rate visit
  rule but does not give numeric `n_LF`, `n_LH`. README acknowledges this
  as `credit-visit-limits`.
- **Code**: `state.rs:165-166` defaults both to 2.
- **Fix**: Once authors publish or respond to a query with values, replace
  the defaults. Until then, this remains a tracked blocker.
- **Solvability**: **P** or **A**.

### WIESE-036 — AR(1) edge-case policy

- **Source**: Thesis §6.2.5 specifies AR(1) but not edge-case handling.
- **Code**: `forecasting.rs:9-35` falls back when fewer than 3 positive
  observations exist; this is a project-defined policy.
- **Fix**: Either obtain the authors' preprocessing code (then implement
  exactly), or freeze the current policy and document it as the
  exact-replication policy for this implementation. Surface choice in
  `ReplicationPolicy`.
- **Solvability**: **C** for freeze; **A** for author-equivalent.

### WIESE-037 — Initial individual employment status mapping rule

- **Source**: Thesis §6.4.6.4 maps HFCS labour status to one of three model
  states. Mapping table not fully published.
- **Code**: Fixture uses synthetic; real-data mode is stubbed.
- **Fix**: Implement the HFCS code mapping documented in the thesis (Tables
  6.13/6.14). For statuses not directly mappable, define a deterministic
  rule (e.g. always assign 'inactive') and document.
- **Solvability**: **C** for fallback; **A** for author-equivalent.

### WIESE-038 — Initial income/wealth rescaling rules

- **Source**: Thesis §§6.4.3.2, 6.4.5.4, 6.4.1.1 describe rescaling drawn
  agents to match OECD/IMF aggregates.
- **Code**: Real-data path absent.
- **Fix**: Implement multiplicative-rescaling helper that takes a vector of
  drawn values and a target aggregate, returns the scaled vector.
- **Solvability**: **C** for the helper; **D** for the aggregate inputs.

### WIESE-039 — Document `T_RW = 8` (wage anchor window) provenance

- **Source**: Thesis §6.4.6: reservation wage anchor uses 8-quarter window.
- **Code**: `Individual.wage_history: [f64; 8]` is sized correctly. Not
  configurable.
- **Fix**: Add a comment citing thesis page; expose
  `wage_history_quarters` as compile-time generic if a country needs a
  different window. Likely no action needed.
- **Solvability**: **C** if change is desired; otherwise no-op.

### WIESE-040 — `T_KD = 1` documentation

- **Source**: Thesis §6.4.3.1: capital installation delay 1 quarter for all
  sectors.
- **Code**: `state.rs:258` sets all entries to 1.
- **Fix**: No change needed; just add a paper citation in the comment.
- **Solvability**: N/A.

---

## Phase VI — Real-data ingestion and calibration

### WIESE-041 — Implement Compustat firm/bank ingestion

- **Source**: Thesis §6.4.3.2 (firms), §6.4.1.1 (banks).
- **Code**: `data.rs::RealDataProvider::load` returns
  `Err(DataError::MissingAssets)`. The `InitialisationRecipe` struct holds
  prose, not code.
- **Fix**:
  1. Add a `compustat` module that parses Compustat CSV/Parquet exports.
  2. Sample firms with replacement, weighted to match OECD sector aggregates.
  3. Rescale `employees`, `deposits`, `debt`, `equity` to OECD totals.
  4. Same for banks against IMF/OECD financial aggregates.
  5. Keep the entire module behind a `real_data` Cargo feature so fixture
     builds remain dependency-light.
- **Solvability**: **C** for parsing/sampling; **D** for the data;
  **A** for the author's exact sampling/tie-break order.

### WIESE-042 — Implement HFCS household and individual ingestion

- **Source**: Thesis §6.4.5.4 (households), §6.4.6.4 (individuals).
- **Code**: Same stub as WIESE-041.
- **Fix**: Add a `hfcs` module that:
  1. Reads ECB HFCS microdata (likely `.csv` or `.dta`).
  2. Draws households using the survey's `hweight` field.
  3. Maps the HFCS variable codes in thesis Tables 6.10–6.14 to model state.
  4. Links individuals to households by HFCS id; derives employment status
     and sector.
  5. Generates property entities from HFCS tenure and value fields.
- **Solvability**: **C** for parsing; **D** for the licensed microdata.

### WIESE-043 — Implement OECD ICIO + national-accounts ingestion

- **Source**: Thesis §6.4.3.1, §6.4.4, §6.4.7.
- **Code**: Stub.
- **Fix**: Add an `oecd_icio` module that reads ICIO tables and produces
  `io_matrix`, `net_fixed_assets_matrix`, `capital_compensation_matrix`,
  `government_consumption_weights`, `cpi_weights`, `row_export_weights`,
  `row_import_weights`. Add an `oecd_national_accounts` module for the
  rescaling targets.
- **Solvability**: **C+D**.

### WIESE-044 — Implement IMF / World Bank / BIS / ESRB readers

- **Source**: Thesis Table 6.2 lists the data sources.
- **Code**: Stub.
- **Fix**: One module per data source. Each loads time series at quarterly
  frequency, indexed by country and indicator code. Use these for the
  Taylor rule estimation, ARDL estimation, government debt initialization,
  and ESRB macroprudential parameters.
- **Solvability**: **C+D**.

### WIESE-045 — Implement firm replacement using Compustat resampling

- **Source**: WIESE-002 specifies behavior; this gap covers real-data variant.
- **Code**: After WIESE-002 lands, fixture mode replaces with stored initial
  template. Real-data mode should redraw from the Compustat sample.
- **Fix**: Cache the Compustat sample on `MacroEnvironment` at
  initialization; on bankruptcy in real-data mode, draw a fresh firm from
  the cached sample matching the bankrupt firm's sector.
- **Solvability**: **C** assuming WIESE-041 is done.

### WIESE-046 — Wire LSA solver into the four matching steps

- **Source**: Same as WIESE-016.
- **Code**: After WIESE-016 lands, `lsa_assign` exists but is not called.
- **Fix**: Use it in:
  - firm↔employee matching (initial sectoral employment),
  - firm↔bank matching (deposit/loan relationships),
  - household↔bank matching (consumption loans, mortgages),
  - household↔property matching (initial residence).
  Cost matrices use thesis-specified distance functions (e.g. wage gap,
  collateral gap, geographic proximity).
- **Solvability**: **C+A** (cost-function definitions need authors' specs to
  be exactly equivalent).

### WIESE-047 — ARDL coefficient estimation pipeline

- **Source**: Thesis §6.4.1.3.
- **Code**: Helpers in `forecasting.rs` exist but are not called.
- **Fix**:
  1. Add `fn estimate_ardl(history: &[(f64, f64)], lag_grid: &[(usize, usize)])
     -> ArdlEstimate` that runs OLS for each lag combination, picks the lag
     pair minimizing AIC, returns coefficients.
  2. Run at initialization with policy-rate and bank-rate histories per
     country and per loan type.
  3. Cache estimates in `CalibrationParameters` so the runtime ARDL update
     (WIESE-005) consumes them.
- **Solvability**: **C+D** for the estimation; **A** for exact author lag
  grid and preprocessing.

### WIESE-048 — Taylor rule estimation pipeline

- **Source**: Thesis eqs 6.47–6.50.
- **Code**: `forecasting.rs::transform_taylor_rule` exists; estimation step
  is not wired in.
- **Fix**: At real-data initialization, fit eq 6.47 by OLS on BIS/IMF/IMF
  series, transform via eqs 6.48–6.50, populate `CentralBank` fields per
  country.
- **Solvability**: **C+D**.

### WIESE-049 — Neural Posterior Estimation infrastructure

- **Source**: Thesis Chapter 7 §7.2.
- **Code**: `calibration.rs::NeuralPosteriorConfig` defines hyperparameters
  but no training loop or sampler exists.
- **Fix**: Out of scope for a pure-Rust port. Two paths:
  - **Path A (recommended)**: keep calibration external (Python `sbi`
    library), serialize posterior samples to a `posterior_*.toml` file per
    country, load in `calibration.rs`. This is the lightest-touch fix.
  - **Path B**: integrate `tch-rs` (libtorch bindings) and reimplement NPE
    in Rust. Significantly larger project.
- **Solvability**: **A** in either path; the asset that closes this is the
  authors' trained NPE / posterior samples for each country.

### WIESE-050 — Bayes factor pipeline

- **Source**: Thesis §7.5.
- **Code**: `BayesFactorConfig` defines architecture only.
- **Fix**: Same two paths as WIESE-049. Recommend external (Python) Bayes-
  factor estimation feeding a `bayes_factors.toml`.
- **Solvability**: **A**.

---

## Cross-cutting recommendations

- **Add `coverage.rs` blockers as each gap is closed.** Whenever a
  `WIESE-NNN` is implemented and shipped, remove the corresponding entry
  from `blocker_log()`. The gap report becomes a runtime ground-truth.
- **Add a `make audit` target** that runs the full test suite and prints the
  resulting gap report. Fail CI when an unresolved blocker is added without
  matching documentation.
- **Add a `--strict` runtime mode** that requires `blocker_log()` to be empty.
  Already supported by `validate_replication_policy` in `lib.rs:168-185`;
  extend the strict-mode default to include any of the blockers introduced
  by this document.
- **Cross-reference paper and thesis equation numbers everywhere.** Each
  comment in `equations.rs` and each `// ...` comment in `systems.rs` should
  use the form `// thesis 6.X / paper A.Y` so future readers can navigate
  both numbering schemes.

## Solvability summary

| Class | Count | Examples | Notes |
|---|---|---|---|
| C (code-only) | ~30 | WIESE-001..010, 012, 014, 016..030, 045..048 | No external assets needed |
| P (parameter) | ~6 | WIESE-014, 015, 019, 034, 035, 036 | Need a value table |
| D (external data) | ~10 | WIESE-013, 017, 018, 031..033, 041..044 | Need Compustat / HFCS / OECD / IMF / BIS / ESRB |
| A (author code/posteriors) | ~6 | WIESE-005, 015, 035, 037, 047, 049, 050 | Need authors' calibration assets |
| U (unsolvable from public sources) | 0 | — | Bit-exact trajectory reproduction is the only candidate, and it is downgraded once the RNG family is matched |

Every gap is solvable in principle. ~30 are pure code edits. ~10 need
licensed datasets (Compustat is the binding one). ~6 need the authors'
calibration scripts or trained networks. None require modifications to the
Syren framework itself.
