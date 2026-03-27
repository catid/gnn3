# Prototype Branch-Strength Negative-Cleanup Max Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch improves if the
newer branch-strength sharp negative-tail cleanup is combined with the older
fixed negative-tail cleanup via a **max-style union** instead of a learned lift.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the improved branch-strength sharp cleanup as one branch
- preserve the fixed negative-tail cleanup as the older high-recall branch
- take a hard nonnegative union between them
- recover more of the fixed branch's `100%` held-out sparse-positive recall
  lane around `1%` coverage without losing the improved matched-band behavior

This is the direct max-style counterpart to the branch-strength lift experiment.

## Implementation

- New head:
  `BranchStrengthNegativeCleanupMaxSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_branch_strength_negative_cleanup_max_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_branch_strength_negative_cleanup_max_support_agree_mix`
  - `prototype_branch_strength_negative_cleanup_max_support_agree_mix_hybrid`

Relative to the branch-strength lift follow-up:

- the same branch-strength sharp cleanup is kept
- the same fixed negative-tail branch is kept
- but the final score is now `max(sharp_branch, fixed_branch)`
- there is no learned lift gate

So this is a very narrow test of whether the failed lift mechanism was the
problem, or whether the fusion itself is wrong.

## Held-Out Result

### `prototype_branch_strength_negative_cleanup_max_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall target match unchanged at `96.51%`
- overall mean delta regret `0.0000`

So the plain max branch collapses completely to baseline.

### `prototype_branch_strength_negative_cleanup_max_support_agree_mix_hybrid`

Closed positive, but still dominated.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.57%`
- overall mean delta regret `-0.0049`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.74%`
- overall mean delta regret `-0.0127`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery still `50%`
- hard near-tie target match still `90.53% -> 90.66%`
- hard near-tie mean delta regret still `-0.0071`
- overall target match `96.83%`
- overall mean delta regret `-0.0155`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.84%`
- overall mean delta regret `-0.0158`

Large-gap controls stayed clean:

- large-gap target match stayed in the `99.84% -> 99.90%` range
- mean delta regret remained non-positive
- no harmful large-gap miss pattern appeared

## Comparison against nearby branches

### Versus the branch-strength lift variant

`prototype_branch_strength_negative_cleanup_lift_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `25%`
- hard near-tie `90.53% -> 90.60%`
- overall mean delta regret `-0.0128`

`prototype_branch_strength_negative_cleanup_max_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `50%`
- hard near-tie `90.53% -> 90.66%`
- overall mean delta regret `-0.0127`

So hard max is meaningfully better than the learned lift in the `1%` region.

### Versus the branch-strength sharp base

`prototype_branch_strength_sharp_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `0.92%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie `90.53% -> 90.66%`
- overall mean delta regret `-0.0149`

`prototype_branch_strength_negative_cleanup_max_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.01%`
- same `50%` / `90.66%`
- weaker overall mean delta regret `-0.0127`

So the hard max union still makes the improved sharp base worse at matched
coverage.

### Versus the live sharp-negative branch below `1%`

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_branch_strength_negative_cleanup_max_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- only `25%` held-out `stable_positive_v2` recovery
- only `90.53% -> 90.60%`
- overall mean delta regret `-0.0109`

So the max union still loses badly to the live low-coverage lane.

### Versus the fixed negative-tail high-recall branch around `1%`

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_branch_strength_negative_cleanup_max_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery only `50%`
- hard near-tie only `90.53% -> 90.66%`
- overall mean delta regret `-0.0127`

So the hard max union still does not preserve the older high-recall lane.

### Versus the older higher-budget matched-band reference

`prototype_support_weighted_agree_mix_hybrid @ 2.00%`

- overall mean delta regret `-0.0165`

`prototype_branch_strength_negative_cleanup_max_support_agree_mix_hybrid @ 2.00%`

- overall mean delta regret `-0.0158`

So even after the union finally reaches the full `75%` / `90.73%` frontier at
`2%`, it still trails the older higher-budget reference slightly.

## Interpretation

This is a real improvement over the failed branch-strength lift, but still not
enough to create a new live lane.

Current read:

- hard max is a better fusion mechanism than the learned lift
- it partially restores the old `50%` / `90.66%` middle lane by `1%`
- but it still does not preserve either live frontier:
  - not the sharp-negative branch's low-coverage `75%` / `90.73%` lane
  - not the fixed negative-tail branch's `100%` / `90.80%` recall lane
- and once it reaches the full frontier at `2%`, it still trails the older
  support-weighted agreement-mixture reference slightly

So even with the improved branch-strength sharp base, max-style fusion remains a
dominated compromise.

## Decision

Close:

- `prototype_branch_strength_negative_cleanup_max_support_agree_mix`
- `prototype_branch_strength_negative_cleanup_max_support_agree_mix_hybrid`

Keep the live shortlist unchanged:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best
  sub-`1%` full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out
  recall around `1%`
- `prototype_support_weighted_agree_mix_hybrid` as the higher-budget
  matched-band reference
