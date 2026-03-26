# Prototype Specialist-Bank Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` lead was losing
signal because a **single positive prototype bank** had to represent two
different source families at once.

The training positives split cleanly into:

- a larger `high_headroom_near_tie` subgroup
- a smaller residual `baseline_error_near_tie` subgroup outside high-headroom

So this experiment replaced the single positive bank with two specialists:

- a headroom specialist bank
- a residual-error specialist bank
- one shared negative bank

Variants:

- `prototype_specialist_max`: take the max specialist score against the shared
  negative bank
- `prototype_specialist_gate`: risk-conditioned mixture over the two specialist
  scores

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

## Main result

The family is closed.

### `prototype_specialist_max`

This was the better specialist variant, but it still missed the live lead.

Best working point (`1.5%` nominal budget):

- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0099`

That looks superficially respectable, but it still under-recovers the target
subset and needs far more coverage than `prototype_hybrid` to get there.

### `prototype_specialist_gate`

This variant is also closed.

- only recovered `25%` of held-out `stable_positive_v2`
- never improved hard near-tie beyond the weak `90.53% -> 90.60%` band
- mostly behaved like another conservative overall-risk branch

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Best specialist point (`prototype_specialist_max @ 1.5%`)

- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0099`

So the specialist split did not sharpen the true sparse-positive correction
problem. It mostly traded recall for more coverage.

## Interpretation

The live prototype-memory geometry does **not** seem bottlenecked by mixing the
headroom and residual-error positives in one bank.

Current read:

- the single-bank `prototype_hybrid` already captures the useful shared
  structure
- forcing a bank split makes the smaller residual subgroup too weak
- specialist routing does not recover the missing held-out positives

So the next architecture gain is unlikely to come from simply partitioning the
same positive family into separate banks.

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `prototype_specialist_max`
- `prototype_specialist_gate`

Do not reopen specialist-bank prototype defer in this form.

## Artifacts

- `scripts/run_prototype_specialist_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_specialist_defer_summary.csv`
- `reports/plots/prototype_specialist_defer_decisions.csv`
- `reports/plots/prototype_specialist_defer_summary.png`
