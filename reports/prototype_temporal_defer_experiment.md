# Prototype Temporal-Context Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` architecture still
missed a small amount of **outer-step temporal context** that was already
present in the cached refinement traces.

The experiment reused the round-twelve cached split:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`
- budgets: `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

The prototype bank stayed the same as the live branch:

- prototype memory still uses `decision_augmented_features(...)`
- only the auxiliary risk branch changed

Temporal context features included:

- per-step top-2 selection margins across the three outer steps
- margin span, margin standard deviation, and final-minus-initial margin drift
- top-1 candidate switch count across outer steps
- optional probe-feature drift norms and cosine similarity across steps

Variants:

- `temporal_scalar_hybrid`
- `temporal_probe_hybrid`

## Main result

The family is closed.

### `temporal_scalar_hybrid`

This variant was the only live member, but it still did **not** beat the
existing `prototype_hybrid` lead on the actual Tier-1 target.

Best temporal point (`1.0%` nominal budget, `0.53%` actual overall coverage):

- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall target match: `96.51% -> 96.74%`
- overall mean delta regret: `-0.0129`

So temporal scalar context made the branch more conservative and slightly
cleaner overall, but it only recovered **half** of the held-out
stable-positive-v2 pack.

### `temporal_probe_hybrid`

This branch is dead.

- recovered only `25%` of held-out stable-positive-v2
- never moved held-out hard near-tie target match off the weaker
  `90.53% -> 90.60%` band
- added probe-drift complexity without improving the actual sparse positive
  family

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Best temporal point (`temporal_scalar_hybrid @ 1.0%`)

- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0129`

So the temporal branch wins a little on aggregate caution, but it loses on the
actual objective:

- lower stable-positive recovery
- weaker hard near-tie lift
- no improvement over the live architecture lead on Tier-1

## Interpretation

Temporal outer-step context appears to help **risk calibration**, not sparse
positive correction.

The current read is:

- prototype-memory geometry still carries the useful local correction signal
- extra temporal drift features mostly make the gate more conservative
- that improves broad aggregate regret slightly
- but it dilutes the exact stable-positive family we are trying to recover

So the remaining architecture opportunity still looks like a better
prototype-memory correction head, not a temporal-context branch layered on top
of it.

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `temporal_scalar_hybrid`
- `temporal_probe_hybrid`

Do not reopen prototype temporal-context defer in this form.

## Artifacts

- `scripts/run_prototype_temporal_defer.py`
- `reports/plots/prototype_temporal_defer_summary.csv`
- `reports/plots/prototype_temporal_defer_decisions.csv`
- `reports/plots/prototype_temporal_defer_summary.png`
