# Prototype Ramped Rescue-Weighted Anchored Dual-Lift Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the rescue-weighted anchored dual-lift variant improves if it
transitions from rescue-weighted behavior at low anchor scores toward the
broader anchored dual lift as the branchwise-max anchor score strengthens.

The design goal was:

- keep the accepted branchwise-max mixed score as the anchor
- keep the rescue-weighted micro-budget improvement
- recover some of the broader anchored-lift behavior at `1.0–1.5%`
- avoid the fixed-rescue-only over-tightening

This is the direct follow-up to the new rescue-weighted anchored dual-lift
micro-budget companion.

## Implementation

- New head:
  `RampedRescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_ramped_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_ramped_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_ramped_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the rescue-weighted anchored dual lift:

- the same anchored dual-lift gate is kept
- the same rescue weight is kept
- but the effective lift weight becomes
  `rescue_weight + anchor_ramp * (1 - rescue_weight)`
- so low-anchor states behave like rescue-weighted lift
- stronger-anchor states move back toward the broader anchored dual lift

This is a narrow interpolation test, not a new family.

## Held-Out Result

### `prototype_ramped_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`

Closed, effectively dead.

At `0.10–1.50%` nominal budgets:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall mean delta regret only `0.0000` to `-0.0013`

At `2.00%` it becomes slightly harmful:

- overall target match dips to `96.50%`
- overall mean delta regret becomes `+0.0007`

So the plain branch is dead and slightly harmful at the high end.

### `prototype_ramped_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed positive, but dominated.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.58%`
- overall mean delta regret `-0.0055`

At `0.25%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.63%`
- overall mean delta regret `-0.0081`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- hard near-tie mean delta regret `-0.0069`
- overall target match `96.76%`
- overall mean delta regret `-0.0136`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.82%`
- overall mean delta regret `-0.0153`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.84%`
- overall mean delta regret `-0.0159`

Large-gap controls stayed clean:

- large-gap target match stayed at `99.89%`
- mean delta regret stayed non-positive
- no harmful large-gap miss pattern appeared

## Comparison against live references

### Versus the rescue-weighted micro-budget companion

At `0.10%`:

- rescue-weighted anchored lift: overall mean delta regret `-0.0064`
- ramped rescue-weighted anchored lift: overall mean delta regret `-0.0055`

At `0.25%`:

- rescue-weighted anchored lift: overall mean delta regret `-0.0088`
- ramped rescue-weighted anchored lift: overall mean delta regret `-0.0081`

Both branches keep the same `50%` held-out `stable_positive_v2` and the same
`90.45%` hard near-tie point, so the ramped variant is strictly worse in the
micro-budget lane it was meant to preserve.

### Versus the accepted branchwise-max reference

At `1.50%`:

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie `90.39% -> 90.60%`
- overall mean delta regret `-0.0159`

`prototype_ramped_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- the same `83.3%` / `90.60%` target-slice point
- weaker overall mean delta regret `-0.0153`

At `2.00%`:

- accepted branchwise-max: `100%` held-out `stable_positive_v2`, hard near-tie
  `90.39% -> 90.68%`, overall mean delta regret `-0.0167`
- ramped rescue-weighted anchored lift: only `83.3%`, only `90.39% -> 90.60%`,
  overall mean delta regret `-0.0159`

So the accepted branchwise-max branch still dominates the ramped variant at the
higher-budget frontier.

## Interpretation

This closes the interpolation hypothesis cleanly.

Current read:

- rescue weighting by itself really did create a micro-budget improvement
- broad anchored lift really did help recover some of the middle-budget target
  lane
- but a simple anchor-score ramp between the two just averages the two failure
  modes instead of combining the wins
- below `0.25%` it is weaker than the rescue-weighted micro-budget branch
- above `1.0%` it is still weaker than accepted branchwise-max

So this is a dominated compromise, not a new operating point.

## Decision

Close:

- `prototype_ramped_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_ramped_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Keep the live interpretation unchanged:

- accepted branchwise-max remains the main robust prototype correction
  reference
- rescue-weighted anchored dual lift remains the only live micro-budget
  companion

The lesson is:

- the remaining lift problem is not a simple interpolation problem between the
  rescue-weighted and broader anchored regimes
