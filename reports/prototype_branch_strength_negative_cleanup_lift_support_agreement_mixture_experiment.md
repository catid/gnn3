# Prototype Branch-Strength Negative-Cleanup Lift Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch improves if the
newer branch-strength sharp negative-tail cleanup is used as the **base** score
and a gated nonnegative lift from the older fixed negative-tail cleanup branch
is added on top.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the stronger branch-specific sharp cleanup as the default retrieval view
- add only a positive fixed-cleanup lift when the older high-recall branch looks
  distinctly stronger
- preserve the sharp branch below `1%` coverage while recovering more of the
  fixed branch's `100%` held-out sparse-positive recall lane around `1%`

This is the direct follow-up to the branch-strength sharp experiment.

## Implementation

- New head:
  `BranchStrengthNegativeCleanupLiftSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_branch_strength_negative_cleanup_lift_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_branch_strength_negative_cleanup_lift_support_agree_mix`
  - `prototype_branch_strength_negative_cleanup_lift_support_agree_mix_hybrid`

Relative to the earlier lift branch:

- the sharp base now uses separate shared and dual cleanup amplitudes
- the fixed branch is only allowed to add a gated positive lift
- the final branch still cannot go below the sharp-base score

So this is still a narrow fusion test, not a new broad architecture family.

## Held-Out Result

### `prototype_branch_strength_negative_cleanup_lift_support_agree_mix`

Closed, effectively dead.

Best point:

- budget `1.50–2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall target match only `96.54–96.56%`
- overall mean delta regret only `-0.0019` to `-0.0023`

So the plain branch only found a tiny broad-safe control fix and never moved the
real target slice.

### `prototype_branch_strength_negative_cleanup_lift_support_agree_mix_hybrid`

Closed positive, but still dominated.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.59%`
- overall mean delta regret `-0.0058`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie target match only `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.71%`
- overall mean delta regret `-0.0114`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie target match only `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.74%`
- overall mean delta regret `-0.0128`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.81%`
- overall mean delta regret `-0.0151`

At `2.00%` nominal budget:

- overall coverage `1.99%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.86%`
- overall mean delta regret `-0.0162`

Large-gap controls stayed clean:

- large-gap target match stayed in the `99.84% -> 99.90%` range
- mean delta regret remained non-positive
- no harmful large-gap miss pattern appeared

## Comparison against live leads

### Versus the branch-strength sharp base below `1%`

`prototype_branch_strength_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- overall mean delta regret `-0.0138`

`prototype_branch_strength_negative_cleanup_lift_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- only `25%` held-out `stable_positive_v2` recovery
- only `90.53% -> 90.60%`
- overall mean delta regret `-0.0114`

So adding the fixed-branch lift actually makes the branch-strength sharp base
worse below `1%`.

### Versus the live sharp-negative branch below `1%`

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

The new lift branch is far below that live low-coverage lane.

### Versus the high-recall fixed negative-tail branch around `1%`

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_branch_strength_negative_cleanup_lift_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie target match only `90.53% -> 90.60%`
- overall mean delta regret `-0.0128`

So the fixed-branch lift does not recover the older high-recall lane at all.

### Versus the older higher-budget matched-band reference

`prototype_support_weighted_agree_mix_hybrid @ 2.00%`

- overall mean delta regret `-0.0165`

`prototype_branch_strength_negative_cleanup_lift_support_agree_mix_hybrid @ 2.00%`

- overall mean delta regret `-0.0162`

So even once the lift finally reaches the full `75%` / `90.73%` frontier, it
still trails the older support-weighted agreement-mixture reference slightly on
aggregate regret.

## Interpretation

This is real signal, but it only appears once coverage rises into the old
higher-budget matched band.

Current read:

- branch-strength sharp cleanup is a better base than the older global sharp
  branch
- but adding a fixed-branch lift still collapses the low-coverage frontier
- the branch only becomes useful again once coverage rises to about `1.5–2.0%`
- and by then it still does not beat the older higher-budget matched-band
  reference

So the fixed-cleanup lift remains the wrong fusion mechanism for this family,
even when the sharp base is improved with branch-specific cleanup amplitude.

## Decision

Close:

- `prototype_branch_strength_negative_cleanup_lift_support_agree_mix`
- `prototype_branch_strength_negative_cleanup_lift_support_agree_mix_hybrid`

Keep the live shortlist unchanged:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best
  sub-`1%` full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out
  recall around `1%`
- `prototype_support_weighted_agree_mix_hybrid` as the higher-budget
  matched-band reference
