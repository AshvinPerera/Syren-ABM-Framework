# Deliberate deviations from the Wiese. et al. paper

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

Restored at Poledna Table 1's Austrian value, **0.7953**. Both legs were already
implemented: firms pay out (A.33), households receive it as investor income
(A.53).

Revert with `theta_dividend: 0.0`.

## 3. A.30 income base — annualised

A.30's loan-to-income cap is printed against `½(Y_h(t−2) + Y_h(t−1))`, the mean
of two *quarterly* incomes.

Baptista et al. (2016) Eq. (13) — the source Wiese cites for the housing block —
reads `q ≤ Φ_i y` with `y` the household's gross **annual** income, and
`ρ^LTI-M = 4.5` is the ESRB annual multiple.

## Scenarios

`config.yaml` carries each as a named block:

| Scenario | Effect |
|---|---|
| `wiese_exact` | Both deviations reverted — reproduces the collapse |
| `wiese_wage_only` | Wage rule reverted, dividend kept |
| `no_dividend` | Dividend removed, wage rule kept |
| `no_overtime` | `h^max = 1.0` — diagnostic for the work-effort channel |
| `roa_quarterly` | Quarterly A.27 threshold — records the units test |
