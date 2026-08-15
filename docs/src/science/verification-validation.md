# Verification and validation

Two different questions run through any model built with Syren, and it helps to
keep them apart.

## Verification: does the model do what the specification says?

Verification asks whether the implementation faithfully realises the model you
intended — the equations, rules, and accounting. It is answered with the code and
tests, not with data. Techniques Syren supports:

- **Named equation tests.** Test each equation or rule against a hand-computed
  expected value. The macroeconomy example tests its named equations this way.
- **Invariants and accounting identities.** Assert conservation laws each tick —
  for example that stocks and flows balance, or that GDP by output and by
  expenditure agree within tolerance.
- **Ordering checks.** Assert that systems run in the intended phase order.
- **Determinism checks.** Assert thread-count invariance and seed divergence, so a
  refactor cannot silently introduce order sensitivity.

Verification is the framework's home turf: it is what the test suite is for, and
what the reproducibility guarantees make trustworthy.

## Validation: does the model match reality?

Validation asks whether the model's behaviour matches the empirical system it
represents. It is answered with data and domain judgement, and it is **outside**
what the framework can establish. Syren makes model outputs reproducible and
lets you record exactly which parameters and inputs produced them, which is a
precondition for credible validation — but whether a calibration is right, or a
result is empirically meaningful, is the modeller's responsibility.

## What to claim

Be precise about which question a result answers. "The accounting identities hold
each tick" is a verification claim the tests can back. "The model reproduces the
observed distribution of firm sizes" is a validation claim that needs data and
belongs to the study, not the framework. Keep the two separate in documentation
and papers.
