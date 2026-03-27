# Prototype Fixed-Rescue Anchored Dual-Lift Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the anchored dual-lift follow-up can be repaired by only allowing
extra lift when the dual branch is winning specifically because its
fixed-cleanup path beat its sharp-cleanup path.

The design goal was:

- keep the accepted branchwise-max mixed score as the anchor
- keep the anchored-lift idea narrow
- only add lift through `relu(fixed_dual - sharp_dual)`
- avoid boosting ordinary dual wins that already come from the sharper path

This is the direct follow-up to the closed anchored dual-lift branchwise-max
experiment.

## Implementation

- New head:
  `FixedRescueAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_fixed_rescue_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_fixed_rescue_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_fixed_rescue_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the earlier anchored dual-lift head:

- the accepted branchwise-max mixed score is still the anchor
- but the extra lift is capped by the dual branch's fixed-over-sharp rescue
  amount
- so the branch can no longer amplify dual wins that do not depend on the fixed
  cleanup path

This is the narrowest remaining positive-lift test inside the accepted
branchwise-max family.

## Held-Out Result

### `prototype_fixed_rescue_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match unchanged at `96.51%`
- overall mean delta regret `0.0000`

So the plain branch collapses completely to baseline.

### `prototype_fixed_rescue_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed positive, but not promotable.

At `0.25%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- hard near-tie mean delta regret `-0.0064`
- overall target match `96.59%`
- overall mean delta regret `-0.0049`

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.65%`
- overall mean delta regret `-0.0069`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.72%`
- overall mean delta regret `-0.0106`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.74%`
- overall mean delta regret `-0.0119`

At `1.50–2.00%` nominal budgets it saturates:

- overall coverage only `1.29–1.53%`
- held-out `stable_positive_v2` recovery remains `83.3%`
- hard near-tie stays at `90.60%`
- overall mean delta regret stays at only `-0.0123`

Large-gap controls stayed clean:

- large-gap target match stayed at `99.89%`
- mean delta regret stayed non-positive
- no harmful large-gap miss pattern appeared

## Comparison against the accepted branchwise-max reference

### Ultra-low and low coverage

At `0.50%` nominal budget:

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- held-out `stable_positive_v2` recovery `50%`
- hard near-tie `90.39% -> 90.45%`
- overall mean delta regret `-0.0111`

`prototype_fixed_rescue_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie `90.39% -> 90.60%`
- overall mean delta regret `-0.0069`

So the fixed-rescue restriction does sharpen the narrow target slice early, but
it pays for that by giving back a large amount of overall regret.

### At `1.00%`

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie `90.39% -> 90.60%`
- overall mean delta regret `-0.0145`

`prototype_fixed_rescue_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- the same `83.3%` / `90.60%` target-slice point
- much weaker overall mean delta regret `-0.0119`

So by `1.00%`, the rescue-only rule is already dominated.

### At `2.00%`

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- held-out `stable_positive_v2` recovery `100%`
- hard near-tie `90.39% -> 90.68%`
- overall mean delta regret `-0.0167`

`prototype_fixed_rescue_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- held-out `stable_positive_v2` recovery only `83.3%`
- hard near-tie only `90.39% -> 90.60%`
- overall mean delta regret only `-0.0123`

So the rescue-only restriction also destroys the accepted higher-budget
frontier.

## Comparison against the earlier anchored dual-lift follow-up

The earlier anchored dual-lift branch reached:

- `100%` held-out `stable_positive_v2` recovery at `1.50%`
- hard near-tie `90.39% -> 90.68%`
- overall mean delta regret `-0.0154`

The fixed-rescue-only restriction reaches only:

- `83.3%` held-out `stable_positive_v2` recovery at `1.50%`
- hard near-tie `90.39% -> 90.60%`
- overall mean delta regret `-0.0123`

So the fixed-rescue restriction removes exactly the part of the anchored lift
that was buying the stronger `1.5%` target-slice improvement.

## Interpretation

This result closes the cleanest remaining “fixed rescue only” explanation for
the anchored-lift signal.

Current read:

- the sparse target-family improvement from the anchored lift was not coming
  only from cases where `fixed_dual > sharp_dual`
- restricting the lift to fixed-over-sharp rescue does sharpen the earliest
  target slice, but it immediately gives up too much broad-safe helpfulness
- once the budget reaches `1.0%`, the accepted branchwise-max reference already
  dominates it
- and by `1.5–2.0%`, it is clearly worse than both the accepted branchwise-max
  branch and the earlier anchored dual-lift follow-up

So this is a useful mechanism negative, not a new live lane.

## Decision

Close:

- `prototype_fixed_rescue_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_fixed_rescue_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Keep the accepted live reference unchanged:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

The lesson is:

- a rescue-only positive lift is too narrow
- any future lift variant would need to preserve the broader branchwise-max
  helpfulness without reopening the aggregate-regret loss from the earlier
  anchored lift
