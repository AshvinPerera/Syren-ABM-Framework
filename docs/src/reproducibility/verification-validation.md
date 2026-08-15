# Verification and validation

A model raises two separate questions. This chapter keeps them apart.

## Verification: does the model do what the specification says?

Verification asks whether the implementation realises the intended model — the
equations, rules, and accounting. It is answered with the code and tests, not
with data. Syren supports several techniques:

- **Named equation tests.** Test each equation or rule against a hand-computed
  expected value. The macroeconomy example tests its named equations this way.
- **Invariants and accounting identities.** Assert conservation laws each tick —
  for example that stocks and flows balance, or that GDP by output and by
  expenditure agree within tolerance.
- **Ordering checks.** Assert that systems run in the intended phase order.
- **Determinism checks.** Assert thread-count invariance and seed divergence, so a
  refactor cannot silently introduce order sensitivity.

The test suite is built for verification, and the reproducibility guarantees make
its checks trustworthy.

## Validation: does the model match reality?

Validation asks whether the model's behaviour matches the empirical system it
represents. It is answered with data and domain judgement, and is **outside** what
the framework can establish. Syren makes model outputs reproducible and records
which parameters and inputs produced them, which is a precondition for credible
validation. Whether a calibration is correct, or a result is empirically
meaningful, is the modeller's responsibility.

## What to claim

State which question a result answers. "The accounting identities hold each tick"
is a verification claim the tests can back. "The model reproduces the observed
distribution of firm sizes" is a validation claim that needs data and belongs to
the study, not the framework. Keep the two separate in documentation and papers.
