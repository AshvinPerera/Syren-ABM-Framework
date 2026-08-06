# Deliberate deviations from the Wiese et al. paper

Four departures from the appendix as printed. Each is stated with the case for
it; the first three are reversible from `config.yaml`.

## 1. Wage rule — `wage_effort_on_base` (default `true`)

**Wiese A.69** applies the work-effort factor `φ^WE` as a *growth factor* to the
previous wage:

```
w_i(t) = (1 + π̄^PPI)(1 + μ^WN) · φ^WE · w_i(t−1)
```

**Poledna A.26**, cited by Wiese as the source, applies the identical factor to
a *base* wage:

```
w_i(t) = w̄_i · min(1.5, min(Q^s, βM, κK)/(N_i ᾱ_i))
```

`φ^WE` is a level — input capacity over labour, capped at `h^max`. Applying a
level as a growth rate compounds it. Ten quarters of 10% overtime leaves the
wage at 1.10× base under Poledna, and 1.10¹⁰ = 2.6× under Wiese.

Revert with `--scenario wiese_exact` or `wage_effort_on_base: false`.

## 2. Dividend — `theta_dividend` (default `0.7953`)

Wiese A.80 omits Poledna's dividend term
`− θ^DIV (1 − τ^FIRM) max(0, Π)`. Without it, firm profits accumulate in
deposits and never return to households as income.

Set to Poledna Table 1's Austrian value, **0.7953**. Both legs are present:
firms pay out (A.33) and households receive it as investor income (A.53).

Revert with `theta_dividend: 0.0`.

## 3. A.30 income base — annualised

A.30's loan-to-income cap is printed against `½(Y_h(t−2) + Y_h(t−1))`, the mean
of two *quarterly* incomes.

Baptista et al. (2016) Eq. (13) — the source Wiese cites for the housing block —
reads `q ≤ Φ_i y` with `y` the household's gross **annual** income, and
`ρ^LTI-M = 4.5` is the ESRB annual multiple.

## 4. A.109 annuity exponent

A.109's annual mortgage repayment carries the annuity term

```
4 · r*(P_h − W^FA_h) / (1 − (1 + r*)^{m_l})
```

with a **positive** exponent on `m_l`. For any `r* > 0` and a 100-quarter
maturity, `(1 + r*)^{m_l} ≫ 1`, so the denominator is large and negative and the
interest leg of the cost of buying is negative — the higher mortgage rates go,
the cheaper buying looks, and A.110 tips households into the housing market
exactly when credit is dearest.

Implemented with the standard annuity denominator `1 − (1 + r*)^{−m_l}`, which
is what the surrounding text describes ("the second term corresponds to
interest"). `equations.rs::purchase_cost_a109`.

Not switchable: there is no reading of the printed form that is a coherent
alternative model.

## Scenarios

`config.yaml` carries each as a named block:

| Scenario | Effect |
|---|---|
| `wiese_exact` | Both deviations reverted — reproduces the collapse |
| `wiese_wage_only` | Wage rule reverted, dividend kept |
| `no_dividend` | Dividend removed, wage rule kept |
| `no_overtime` | `h^max = 1.0` — diagnostic for the work-effort channel |
| `roa_quarterly` | Quarterly A.27 threshold — records the units test |
