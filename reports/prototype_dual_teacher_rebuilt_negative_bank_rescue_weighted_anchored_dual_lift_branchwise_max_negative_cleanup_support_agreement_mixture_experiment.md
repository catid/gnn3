# Prototype Dual Teacher-Rebuilt Negative-Bank Rescue-Weighted Anchored Dual-Lift Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the strongest teacher-guided bank edit becomes useful once it is
applied to the rescue-weighted anchored dual-lift branchwise-max family rather
than to raw branchwise-max.

The design goal was:

- keep the accepted branchwise-max score as the outer anchor
- keep the rescue-weighted anchored dual lift that already wins the
  `0.10–0.25%` micro-budget lane
- rebuild only the dual negative bank from teacher-marked harmful states after
  fitting
- leave the shared negative bank learned
- see whether the dual-only teacher rebuild can extend the rescue-weighted gain
  into the `~0.50%` regime without reopening the higher-budget regressions

This is the direct combination of the live rescue-weighted lift and the
strongest simple teacher-guided bank-rebuild result.

## Implementation

- New head:
  `DualTeacherRebuiltNegativeBankRescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the earlier rescue-weighted anchored lift:

- the same rescue-weighted anchored dual-lift geometry is fit first
- then only the dual negative bank is rebuilt from teacher-marked harmful
  states at fixed cardinality
- the rebuilt dual bank keeps explicit support weights
- the shared negative bank stays exactly as learned

So this is a narrow additive test inside the accepted branchwise-max family:
teacher-guided dual-bank editing on top of the one positive lift variant.

## Held-Out Result

### `prototype_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match unchanged at `96.51%`
- overall mean delta regret `0.0000`

So the plain branch again collapses to baseline.

### `prototype_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Real positive result, but only as a narrow target-heavy low-mid-budget lane.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.58%`
- overall mean delta regret `-0.0058`

At `0.25%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.60%`
- overall mean delta regret `-0.0071`

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- hard near-tie mean delta regret `-0.0069`
- overall target match `96.69%`
- overall mean delta regret `-0.0113`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- hard near-tie mean delta regret `-0.0069`
- overall target match `96.74%`
- overall mean delta regret `-0.0129`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- hard near-tie mean delta regret `-0.0069`
- overall target match `96.79%`
- overall mean delta regret `-0.0145`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.83%`
- overall mean delta regret `-0.0155`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.86%`
- overall mean delta regret `-0.0162`

Large-gap controls stayed clean:

- large-gap target match stayed at `99.89%`
- mean delta regret stayed non-positive
- no harmful large-gap miss pattern appeared

## Comparison against rescue-weighted anchored lift

At `0.10–0.25%`, the combined branch is weaker than rescue-weighted alone:

- rescue-weighted keeps the same `50%` held-out `stable_positive_v2`
  and the same `90.45%` hard near-tie point
- but rescue-weighted gets better overall mean delta regret:
  - `-0.0064` vs `-0.0058` at `0.10%`
  - `-0.0088` vs `-0.0071` at `0.25%`

At `0.50%`, the combined branch becomes clearly better:

- rescue-weighted anchored lift:
  `50%` held-out `stable_positive_v2`, `90.39% -> 90.45%`, overall mean delta
  regret `-0.0105`
- dual-teacher-rebuilt rescue-weighted lift:
  `66.7%`, `90.39% -> 90.53%`, overall mean delta regret `-0.0113`

At `1.00–2.00%`, the combined branch stays better than rescue-weighted:

- `1.00%`: same broad aggregate band, but slightly stronger overall mean delta
  regret `-0.0145` versus `-0.0141` with the same `66.7%` / `90.53%`
  target slice
- `1.50–2.00%`: combined branch reaches `83.3%` / `90.60%`, while
  rescue-weighted alone only reaches that level at the very top end

So the dual-only teacher rebuild does become useful once it is layered on top
of rescue weighting.

## Comparison against the accepted branchwise-max reference

At `0.50%`, this creates a real new tradeoff:

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- overall mean delta regret `-0.0111`

`prototype_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- overall mean delta regret `-0.0113`

So at `~0.50%`, the new branch trades only a tiny aggregate-regret loss for a
materially stronger target slice.

But by `0.75%`, accepted branchwise-max already dominates again:

- same `66.7%` held-out `stable_positive_v2`
- same `90.39% -> 90.53%` hard near-tie
- better overall mean delta regret `-0.0137` vs `-0.0129`

And above `1.00%`, the accepted branch stays clearly better:

- `1.00%`: accepted `83.3%` / `90.60%` / `-0.0145`; new branch only
  `66.7%` / `90.53%` / `-0.0145`
- `2.00%`: accepted `100%` / `90.68%` / `-0.0167`; new branch only
  `83.3%` / `90.60%` / `-0.0162`

So this does not replace accepted branchwise-max as the main live reference.

## Interpretation

This is the first teacher-guided bank edit that combines positively with the
rescue-weighted anchored lift.

Current read:

- dual-only teacher-guided negative-bank reconstruction is not useful by itself
  as a replacement branch
- but it does become useful once it is layered on top of the rescue-weighted
  anchored lift
- the resulting value is narrow:
  - rescue-weighted alone still owns `0.10–0.25%`
  - dual-teacher-rebuilt rescue-weighted lift creates a target-heavy
    `~0.50%` operating point
  - accepted branchwise-max dominates again from `0.75%` upward

So the right conclusion is not “replace the live references.” It is:

- keep rescue-weighted anchored lift as the micro-budget companion
- keep accepted branchwise-max as the main robust reference
- add the dual-teacher-rebuilt rescue-weighted branch only as a narrow
  `~0.50%` target-heavy companion

## Decision

Close:

- `prototype_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`

Keep alive:

- `prototype_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Updated live interpretation:

- rescue-weighted anchored dual lift remains the best branchwise-max-family
  micro-budget companion at `0.10–0.25%`
- dual-teacher-rebuilt rescue-weighted anchored dual lift adds a new
  target-heavy `~0.50%` operating point
- accepted branchwise-max remains the main higher-budget reference from
  `0.75%` upward
