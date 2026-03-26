# Agreement-Gated Prototype Mixture Follow-up

## Setup

This follow-up tested whether the new geometry-mixture lead could be made more
coverage-efficient by anchoring on the shared-projection prototype score and
only letting the dual-projection branch pull the logit when the two geometry
branches agree enough to trust it.

The new head keeps the same two geometry branches as `prototype_mixture_hybrid`:

- shared-projection prototype score
- dual-projection prototype score

But instead of one global mixture weight, it learns a tiny per-state gate over:

- shared score
- dual score
- absolute score gap
- score product

Variants:

- `prototype_agree_mix`: agreement-gated mixture without the risk branch
- `prototype_agree_mix_hybrid`: agreement-gated mixture plus the tiny
  margin/regime risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

`prototype_agree_mix` is closed. `prototype_agree_mix_hybrid` is alive and is
now the best coverage-efficient matched-band follow-up in the prototype family.

### `prototype_agree_mix`

This variant is closed.

- recovered at most `25%` of held-out `stable_positive_v2`
- never moved beyond the weak `90.53% -> 90.60%` hard near-tie band
- started selecting harmful states at larger budgets

Best point (`0.5%` nominal budget):

- overall coverage: `0.51%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0019`
- overall target match: `96.51% -> 96.53%`
- overall mean delta regret: `-0.0008`

### `prototype_agree_mix_hybrid`

This variant is a real positive follow-up.

At `1.0%` nominal budget:

- overall coverage: `0.79%`
- stable-positive-v2 recovery: `50%`
- stable-positive-v2 precision: `100%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall target match: `96.51% -> 96.73%`
- overall mean delta regret: `-0.0124`

At `1.5%` nominal budget:

- overall coverage: `1.05%`
- stable-positive-v2 recovery: `75%`
- stable-positive-v2 precision: `100%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall target match: `96.51% -> 96.76%`
- overall mean delta regret: `-0.0137`

At `2.0%` nominal budget:

- overall coverage: `1.29%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0137`

Large-gap controls stayed clean at those matched-band points:

- large-gap target match: `99.79% -> 99.90%`
- large-gap mean delta regret: `-0.0065`

## Comparison against current leads

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

`prototype_agree_mix_hybrid @ 1.0%`

- overall coverage: `0.79%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0124`

So it does not replace `prototype_hybrid` as the ultra-low-coverage leader.

But it becomes the best coverage-efficient matched-band follow-up:

`prototype_mixture_hybrid @ 2.0%`

- overall coverage: `1.84%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0138`

`prototype_agree_mix_hybrid @ 1.5%`

- overall coverage: `1.05%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0137`

So the agreement-gated version reaches the same real frontier band and the same
held-out sparse-positive recovery at materially lower overall coverage.

## Interpretation

This is the clearest architecture result so far inside the prototype family:

- changing geometry helped
- mixing geometries helped more
- and agreement-gated mixing is the most coverage-efficient version yet

Current read:

- the useful sparse-positive family is not captured by one geometry alone
- shared and dual geometry are complementary
- per-state agreement gating is a better control mechanism than a single global
  mixture weight

What it does **not** show yet:

- a clean reason to change the default exploit policy
- a stronger ultra-low-coverage leader than `prototype_hybrid`

## Decision

Keep:

- `prototype_hybrid` as the best ultra-low-coverage architecture lead
- `prototype_agree_mix_hybrid` as the best coverage-efficient matched-band
  follow-up inside the prototype family

Close:

- `prototype_agree_mix`

The ungated `prototype_mixture_hybrid` is now mainly a reference point rather
than the top follow-up target.

## Artifacts

- `scripts/run_prototype_agreement_mixture_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_agreement_mixture_defer_summary.csv`
- `reports/plots/prototype_agreement_mixture_defer_decisions.csv`
- `reports/plots/prototype_agreement_mixture_defer_summary.png`
