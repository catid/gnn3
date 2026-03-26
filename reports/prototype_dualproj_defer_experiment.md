# Dual-Projection Prototype Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` lead was limited by
using the same query projection for both positive and negative prototype banks.

The new head changes the prototype geometry itself:

- one query projection for matching positive prototypes
- a separate query projection for matching negative prototypes
- the defer score is the positive-bank evidence minus the negative-bank
  evidence in their own projected spaces

Variants:

- `prototype_dualproj`: dual-projection geometry only
- `prototype_dualproj_hybrid`: the same geometry plus the tiny margin/regime
  risk branch used by the live `prototype_hybrid` lead

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

The family is closed.

### `prototype_dualproj`

This variant is effectively dead.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- never moved held-out hard near-tie above baseline
- only found a tiny overall-risk niche with negligible value

Best point (`2.0%` nominal budget):

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: `90.53% -> 90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall target match: `96.51% -> 96.51%`
- overall mean delta regret: `-0.0003`

### `prototype_dualproj_hybrid`

This variant is the only useful member of the family, but it is still closed.

Best low-coverage point (`0.10%` nominal budget):

- overall coverage: `0.10%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall target match: `96.51% -> 96.57%`
- overall mean delta regret: `-0.0047`

Best overall-regret point (`0.75%` nominal budget):

- overall coverage: `0.43%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall target match: `96.51% -> 96.60%`
- overall mean delta regret: `-0.0069`

So the geometry change is not inert, but it still leaves most of the real
source family untouched:

- stable-positive-v2 stayed capped at `25%`
- hard near-tie stayed capped at `90.53% -> 90.60%`
- no budget beat the live `prototype_hybrid` band

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Best dual-projection point (`prototype_dualproj_hybrid @ 0.75%`)

- overall coverage: `0.43%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall mean delta regret: `-0.0069`

The asymmetric geometry does help a little at tiny coverage, but it does not
recover enough of the sparse-positive family to replace the live lead.

## Interpretation

The sparse positive family does not appear to be bottlenecked mainly by using a
shared query projection for positive and negative prototype evidence.

Current read:

- changing the bank geometry itself is better than many of the recent residual
  tweaks
- but the gain is still too small and too partial
- the live `prototype_hybrid` lead still gets much more of the useful sparse
  family

This closes another plausible architecture path:

- asymmetric positive-vs-negative query spaces are not enough on their own
- a geometry change without stronger sparse-positive recall is not promotable

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `prototype_dualproj`
- `prototype_dualproj_hybrid`

Do not reopen dual-projection prototype defer in this form.

## Artifacts

- `scripts/run_dual_projection_prototype_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_dualproj_defer_summary.csv`
- `reports/plots/prototype_dualproj_defer_decisions.csv`
- `reports/plots/prototype_dualproj_defer_summary.png`
