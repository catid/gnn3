# Positive-Lift Evidence Agreement Follow-up

## Setup

This follow-up tested whether the live `prototype_agree_mix_hybrid` /
`prototype_evidence_agree_hybrid` line was missing a one-sided correction path.

The new head keeps the same shared and dual prototype geometry as the existing
agreement-mixture family, but changes the control rule:

- keep the existing agreement gate as the base path
- add a separate positive-lift gate driven by evidence advantages
- only let that extra lift spend the remaining dual blend when the dual branch
  has stronger positive evidence or cleaner negative evidence than the shared
  branch
- never use the lift path to make the dual branch pull the logit downward

Variants:

- `prototype_positive_lift_agree`: one-sided positive-lift agreement without
  the risk branch
- `prototype_positive_lift_agree_hybrid`: one-sided positive-lift agreement
  plus the tiny margin/regime risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

This family is closed.

- `prototype_positive_lift_agree` is dead.
- `prototype_positive_lift_agree_hybrid` found broad-safe aggregate regret
  improvements, but it still failed the real target.

### `prototype_positive_lift_agree`

This variant is dead.

At every budget:

- held-out `stable_positive_v2` recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`

It selected only non-target states:

- overall false-positive rate: `100%`
- large-gap controls stayed clean
- but there was no Tier-1 value anywhere on the held-out panel

### `prototype_positive_lift_agree_hybrid`

This variant is still closed.

It found a broad-safe niche, but it did not recover the real sparse-positive
frontier.

At `0.75%` nominal budget:

- overall coverage: `0.76%`
- held-out `stable_positive_v2` recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall target match: `96.51% -> 96.69%`
- overall mean delta regret: `-0.0106`

At `1.50%` nominal budget:

- overall coverage: `1.23%`
- held-out `stable_positive_v2` recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall target match: `96.51% -> 96.74%`
- overall mean delta regret: `-0.0123`

At `2.00%` nominal budget:

- overall coverage: `1.47%`
- held-out `stable_positive_v2` recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall target match: `96.51% -> 96.75%`
- overall mean delta regret: `-0.0127`

Large-gap controls stayed clean at those points:

- large-gap target match: `99.79% -> 99.90%`
- large-gap mean delta regret: `-0.0065`

But that is still the weaker hard-slice band, and it only recovers `25%` of
held-out `stable_positive_v2`.

## Comparison against current leads

`prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- held-out `stable_positive_v2` recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

`prototype_agree_mix_hybrid @ 1.5%`

- overall coverage: `1.05%`
- held-out `stable_positive_v2` recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0137`

`prototype_evidence_agree_hybrid @ 2.0%`

- overall coverage: `2.00%`
- held-out `stable_positive_v2` recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0148`

`prototype_positive_lift_agree_hybrid @ 2.0%`

- overall coverage: `1.47%`
- held-out `stable_positive_v2` recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall mean delta regret: `-0.0127`

So the one-sided lift does improve aggregate regret relative to the older
`prototype_hybrid` baseline, but it is still clearly behind both live
matched-band leads on the actual sparse-positive target.

## Interpretation

This is another useful negative result.

The one-sided positive-lift idea did what it was designed to do:

- it found safe aggregate improvements
- it preserved large-gap control behavior
- it avoided reopening broad harmful churn

But it still did not solve the real problem:

- it mostly spent coverage on non-target broad-safe corrections
- it recovered only `25%` of held-out `stable_positive_v2`
- it never moved past the weaker `90.53% -> 90.60%` hard near-tie band

Current read:

- the remaining sparse-positive family is not just missing a positive override
- the live agreement-mixture/evidence-agreement heads already capture most of
  the recoverable one-sided lift signal
- the remaining gap still looks more like precise target selection than
  directional bias

## Decision

Keep:

- `prototype_hybrid` as the best ultra-low-coverage architecture lead
- `prototype_agree_mix_hybrid` as the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` as the best aggregate-quality matched-band
  follow-up

Close:

- `prototype_positive_lift_agree`
- `prototype_positive_lift_agree_hybrid`

## Artifacts

- `scripts/run_prototype_positive_lift_agreement_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_positive_lift_agreement_defer.json`
- `reports/plots/prototype_positive_lift_agreement_defer_summary.csv`
- `reports/plots/prototype_positive_lift_agreement_defer_decisions.csv`
- `reports/plots/prototype_positive_lift_agreement_defer_summary.png`
