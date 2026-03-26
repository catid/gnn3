# Cascade Prototype Defer Follow-up

## Setup

This follow-up tested whether the current prototype family could preserve the
ultra-low-coverage strength of `prototype_hybrid` while still borrowing the
broader matched-band gains from `prototype_agree_mix_hybrid`.

The new cascade head uses three stages:

- a shared-projection prototype anchor, like `prototype_hybrid`
- the agreement-gated shared/dual geometry from `prototype_agree_mix_hybrid`
- a second tiny gate that can only add a **nonnegative lift** above the shared
  anchor

That makes the architecture intentionally one-sided:

- it can keep the sharp shared score
- it can add a broader geometry lift
- it cannot rewrite the shared anchor downward

Variants:

- `prototype_cascade`: lift-only cascade without the risk branch
- `prototype_cascade_hybrid`: lift-only cascade plus the tiny margin/regime
  risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

`prototype_cascade` is dead. `prototype_cascade_hybrid` is useful only as a
negative calibration result and is closed.

### `prototype_cascade`

This variant is fully closed.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- never moved the hard near-tie slice off baseline
- mostly selected broad-safe controls with no Tier-1 value

Best point (`2.0%` nominal budget):

- overall coverage: `2.00%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `0.0000`

### `prototype_cascade_hybrid`

This variant is not dead, but it is still closed.

At `0.75%` nominal budget:

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `50%`
- overall defer precision: `3.0%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall target match: `96.51% -> 96.69%`
- overall mean delta regret: `-0.0110`

At `1.50%` nominal budget:

- overall coverage: `1.52%`
- stable-positive-v2 recovery: `50%`
- overall defer precision: `1.5%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall target match: `96.51% -> 96.72%`
- overall mean delta regret: `-0.0122`

At `2.00%` nominal budget:

- overall coverage: `2.00%`
- stable-positive-v2 recovery: `50%`
- overall defer precision: `1.1%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall target match: `96.51% -> 96.73%`
- overall mean delta regret: `-0.0125`

Large-gap controls stayed clean at those higher-budget points:

- large-gap target match: `99.79% -> 99.90%`
- large-gap mean delta regret: `-0.0065`

## Comparison against current leads

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

`prototype_cascade_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0110`

So the cascade preserves aggregate caution well, but it gives back too much of
the real sparse-positive frontier to replace `prototype_hybrid`.

Best coverage-efficient matched-band follow-up:

`prototype_agree_mix_hybrid @ 1.50%`

- overall coverage: `1.05%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0137`

`prototype_cascade_hybrid @ 1.50%`

- overall coverage: `1.52%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0122`

So the cascade also fails to beat the current matched-band lead.

## Interpretation

This follow-up makes the current frontier sharper:

- a shared anchor plus a nonnegative lift is too conservative
- it does improve broad overall regret by selecting safe teacher wins
- but it under-recovers the true sparse-positive family

Current read:

- the sparse-positive frontier needs some states to move farther away from the
  shared anchor than a lift-only cascade allows
- preserving large-gap and aggregate behavior is easy
- matching the true Tier-1 pack still requires the freer agreement-mixture
  geometry

The key failure mode is now explicit:

- shared-anchor cascades mostly harvest broad-safe non-Tier-1 cases
- once coverage rises, overall precision falls quickly
- but held-out stable-positive recovery still caps at `50%`

## Decision

Keep:

- `prototype_hybrid` as the best ultra-low-coverage architecture lead
- `prototype_agree_mix_hybrid` as the best coverage-efficient matched-band
  follow-up

Close:

- `prototype_cascade`
- `prototype_cascade_hybrid`

## Artifacts

- `scripts/run_prototype_cascade_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_cascade_defer_summary.csv`
- `reports/plots/prototype_cascade_defer_decisions.csv`
- `reports/plots/prototype_cascade_defer_summary.png`
