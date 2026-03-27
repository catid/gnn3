# Prototype Anchored Dual-Lift Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the accepted branchwise-max family improves if the existing
branchwise-max mixed score stays as the anchor and only receives a bounded
positive lift when the dual branch clearly exceeds the shared branch.

The design goal was:

- keep the accepted branchwise-max fusion intact
- avoid reopening a generic soft learned lift
- let the dual branch add signal only when it has clear extra evidence
- preserve the accepted `1.0%` operating point while expanding the `1.5%`
  stable-positive recall lane

This is the direct follow-up to the closed global lift, branchwise lift, and
branchwise margin-max variants.

## Implementation

- New head:
  `AnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the accepted branchwise-max head:

- shared and dual fixed/sharp cleanup scores are unchanged
- branch-local `max(fixed, sharp)` remains the anchor inside each branch
- the outer agreement mixture remains unchanged
- only after that does a bounded gate add `relu(branch_dual - branch_shared)`

So this is a narrow fusion tweak, not a new retrieval family.

## Held-Out Result

### `prototype_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`

Closed, effectively dead.

At `0.10–1.00%` nominal budgets:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match unchanged at `96.51%`
- overall mean delta regret stayed at `0.0000`

At `1.50–2.00%` nominal budgets it only wakes up slightly:

- held-out `stable_positive_v2` recovery only `16.7–33.3%`
- hard near-tie only `90.47%`
- overall mean delta regret only `-0.0005` to `-0.0013`

So the plain branch remains dead.

### `prototype_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed positive, but not promotable.

At `0.25%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.61%`
- overall mean delta regret `-0.0073`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- hard near-tie mean delta regret `-0.0069`
- overall target match `96.73%`
- overall mean delta regret `-0.0127`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.79%`
- overall mean delta regret `-0.0145`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.39% -> 90.68%`
- hard near-tie mean delta regret `-0.0102`
- overall target match `96.82%`
- overall mean delta regret `-0.0154`

At `2.00%` nominal budget:

- overall coverage `1.91%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.39% -> 90.68%`
- hard near-tie mean delta regret `-0.0102`
- overall target match `96.83%`
- overall mean delta regret `-0.0156`

Large-gap controls stayed clean:

- large-gap target match stayed at `99.89%`
- mean delta regret stayed non-positive
- no harmful large-gap miss pattern appeared

## Comparison against the accepted branchwise-max reference

### At `1.00%`

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie `90.39% -> 90.60%`
- overall mean delta regret `-0.0145`

`prototype_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- the same `83.3%` / `90.60%` point
- the same overall mean delta regret `-0.0145`

So the anchored lift preserves the accepted `1.00%` point, but does not improve
it.

### At `1.50%`

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 1.50%`

- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie `90.39% -> 90.60%`
- overall mean delta regret `-0.0159`

`prototype_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 1.50%`

- held-out `stable_positive_v2` recovery `100%`
- hard near-tie `90.39% -> 90.68%`
- overall mean delta regret `-0.0154`

So the anchored lift does improve the narrow target pack at `1.50%`, but it
still loses aggregate overall regret at matched coverage.

### At `2.00%`

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 2.00%`

- held-out `stable_positive_v2` recovery `100%`
- hard near-tie `90.39% -> 90.68%`
- overall mean delta regret `-0.0167`

`prototype_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 2.00%`

- the same `100%` / `90.68%` target-slice point
- weaker overall mean delta regret `-0.0156`
- slightly lower overall coverage `1.91%` instead of `2.00%`

So once the accepted branchwise-max reference reaches full held-out recall, the
anchored lift is strictly worse on aggregate quality.

## Interpretation

This is the first soft lift variant on top of branchwise-max that does not
immediately collapse the accepted frontier. That part is real.

Current read:

- keeping the accepted branchwise-max score as the anchor is much safer than
  replacing it with a generic learned lift
- the bounded dual lift can tighten the branch around `1.5%` and recover the
  last held-out stable-positive cases earlier
- but the gain comes from spending less broad non-target coverage, and that
  gives back aggregate overall regret
- by `2.0%`, the accepted branchwise-max branch catches up on the target pack
  and clearly wins on overall regret

So this is a valid mechanism datapoint, but not a new live operating lane.

## Decision

Close:

- `prototype_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Keep the accepted live reference unchanged:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

The lesson is narrower:

- anchoring to branchwise-max is necessary if any softer positive lift is
  reopened
- but even this anchored version still does not beat the accepted frontier on
  the full promotion surface, because it gives back aggregate quality once the
  budget rises above `1%`
