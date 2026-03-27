# Prototype Rescue-Weighted Anchored Dual-Lift Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the anchored dual-lift branchwise-max follow-up improves if the
lift is still anchored to the accepted branchwise-max score but is
multiplicatively weighted by structured dual fixed-rescue strength.

The design goal was:

- keep the accepted branchwise-max mixed score as the anchor
- keep the anchored dual-lift geometry
- avoid the fixed-rescue-only over-tightening
- shrink lift when the dual branch is not actually being helped by the fixed
  cleanup path
- preserve the anchored lift's micro-budget gains without its higher-budget
  regret loss

This is the midpoint between the closed anchored dual lift and the closed
fixed-rescue-only anchored lift.

## Implementation

- New head:
  `RescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the earlier anchored dual lift:

- the accepted branchwise-max mixed score still acts as the anchor
- the same learned dual-lift gate is kept
- but the lift is scaled by `sigmoid(scale * (fixed_dual - sharp_dual) + bias)`
- so extra dual lift is still possible even when fixed rescue is modest
- but it is suppressed when the dual branch win is not actually tied to fixed
  rescue

So this is still a narrow additive change inside the accepted branchwise-max
family.

## Held-Out Result

### `prototype_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match unchanged at `96.51%`
- overall mean delta regret `0.0000`

So the plain branch again collapses to baseline.

### `prototype_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Alive as a narrow micro-budget positive, but not a broad replacement lead.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.59%`
- overall mean delta regret `-0.0064`

At `0.25%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.64%`
- overall mean delta regret `-0.0088`

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.67%`
- overall mean delta regret `-0.0105`

At `1.00%` nominal budget:

- overall coverage `0.85%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- hard near-tie mean delta regret `-0.0069`
- overall target match `96.78%`
- overall mean delta regret `-0.0141`

At `1.50%` nominal budget:

- overall coverage `1.11%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- hard near-tie mean delta regret `-0.0069`
- overall target match `96.81%`
- overall mean delta regret `-0.0150`

At `2.00%` nominal budget:

- overall coverage `1.35%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.86%`
- overall mean delta regret `-0.0162`

Large-gap controls stayed clean:

- large-gap target match stayed at `99.89%`
- mean delta regret stayed non-positive
- no harmful large-gap miss pattern appeared

## Comparison against the accepted branchwise-max reference

### At `0.10–0.25%`

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- `0.10%`: overall mean delta regret `-0.0050`
- `0.25%`: overall mean delta regret `-0.0074`
- same `50%` held-out `stable_positive_v2`
- same hard near-tie `90.39% -> 90.45%`

`prototype_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- `0.10%`: overall mean delta regret `-0.0064`
- `0.25%`: overall mean delta regret `-0.0088`
- same `50%` held-out `stable_positive_v2`
- same hard near-tie `90.39% -> 90.45%`

So this is a real micro-budget improvement over the accepted branchwise-max
reference.

### Above `0.25%`

At `0.50%`:

- accepted branchwise-max: overall mean delta regret `-0.0111`
- rescue-weighted anchored lift: overall mean delta regret `-0.0105`

At `1.00%`:

- accepted branchwise-max: `83.3%` held-out `stable_positive_v2`, hard near-tie
  `90.39% -> 90.60%`, overall mean delta regret `-0.0145`
- rescue-weighted anchored lift: only `66.7%`, only `90.39% -> 90.53%`,
  overall mean delta regret `-0.0141`

At `2.00%`:

- accepted branchwise-max: `100%` held-out `stable_positive_v2`, hard near-tie
  `90.39% -> 90.68%`, overall mean delta regret `-0.0167`
- rescue-weighted anchored lift: only `83.3%`, only `90.39% -> 90.60%`,
  overall mean delta regret `-0.0162`

So once coverage rises beyond the tiny micro-budget band, the accepted
branchwise-max reference clearly dominates again.

## Comparison against the earlier anchored dual-lift follow-up

The earlier anchored dual-lift branch:

- improved the narrow `1.5%` lane to `100%` held-out `stable_positive_v2`
- but lost aggregate overall regret at `1.5–2.0%`

The rescue-weighted version instead:

- fixes the micro-budget `0.10–0.25%` point
- but gives up the anchored lift's `1.5%` full-recall gain entirely

So rescue weighting is a cleaner low-budget fix, not a repair of the
higher-budget anchored-lift lane.

## Interpretation

This is the first positive-lift variant that actually improves the accepted
branchwise-max reference on the full held-out promotion surface, but only in a
very narrow budget band.

Current read:

- unweighted anchored lift was too broad
- fixed-rescue-only lift was too narrow
- rescue weighting finds a useful middle ground at `0.10–0.25%`
- but it still cannot preserve the accepted branchwise-max frontier once the
  budget rises above that micro-budget range

So the right conclusion is not “replace branchwise-max.” It is:

- keep accepted branchwise-max as the main live reference
- add rescue-weighted anchored lift as a new micro-budget companion for the
  `0.10–0.25%` regime only

## Decision

Close:

- `prototype_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`

Keep alive:

- `prototype_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Updated interpretation:

- accepted branchwise-max remains the main robust prototype correction
  reference
- rescue-weighted anchored lift is now the best branchwise-max-family
  micro-budget companion at `0.10–0.25%`
- do not use it above `0.25%`, where accepted branchwise-max dominates again
