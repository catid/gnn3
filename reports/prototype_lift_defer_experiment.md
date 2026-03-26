# Prototype Lift Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` lead was being hurt
because its residual branch could suppress useful prototype evidence.

The new head keeps the same positive-vs-negative prototype score, but restricts
the extra branch to a **nonnegative lift** only:

- pooled prototype score
- top positive prototype similarity
- top negative prototype similarity
- top-gap between the two

Variants:

- `prototype_lift_hybrid`: positive-only lift conditioned on
  `risk_features + prototype_summary`
- `prototype_lift_gated`: the same lift, but with an extra learned gate over
  the prototype summary

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

The family is closed.

### `prototype_lift_hybrid`

This variant is effectively dead.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- never moved held-out hard near-tie above baseline
- only found a tiny broad-risk niche with no Tier-1 value

Best point (`2.0%` nominal budget):

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: `90.53% -> 90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall target match: `96.51% -> 96.53%`
- overall mean delta regret: `-0.0013`

### `prototype_lift_gated`

This variant is also closed.

It found a broader caution niche and recovered one held-out positive, but it
still missed the real target.

Best overall point (`2.0%` nominal budget):

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `25%`
- stable-positive-v2 precision: `75%` inside the target slice
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0023`
- overall target match: `96.51% -> 96.71%`
- overall mean delta regret: `-0.0098`

That is not enough on Tier-1:

- stable-positive-v2 stayed capped at `25%`
- hard near-tie stayed capped at `90.53% -> 90.60%`
- it still clearly loses to the live `prototype_hybrid` lead

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Best lift point (`prototype_lift_gated @ 2.0%`)

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0023`
- overall mean delta regret: `-0.0098`

So the positive-only lift constraint does not preserve the live sparse-positive
retrieval and does not create a better overall-risk defer policy either.

## Interpretation

The live prototype family is not bottlenecked by the residual branch being too
free to suppress evidence.

Current read:

- forcing the residual to be monotone-positive removes too much flexibility
- the held-out sparse-positive family still is not recovered
- the live gain appears to require a bidirectional calibration around the base
  prototype score, not just an always-positive lift

This closes another plausible architecture path:

- positive-only residual lifts do not improve the live `prototype_hybrid` lead
- the plain lift mostly collapses to baseline behavior
- the gated lift finds some broad caution, but still misses most of the sparse
  positive family

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `prototype_lift_hybrid`
- `prototype_lift_gated`

Do not reopen positive-only prototype lift residuals in this form.

## Artifacts

- `scripts/run_prototype_lift_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_lift_defer_summary.csv`
- `reports/plots/prototype_lift_defer_decisions.csv`
- `reports/plots/prototype_lift_defer_summary.png`
