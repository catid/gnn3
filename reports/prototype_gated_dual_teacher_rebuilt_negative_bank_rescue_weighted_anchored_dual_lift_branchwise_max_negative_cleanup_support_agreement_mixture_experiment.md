# Prototype Gated Dual Teacher-Rebuilt Negative-Bank Rescue-Weighted Anchored Dual-Lift Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the positive hard-swap dual teacher-rebuild can be made safer by
blending between the learned and rebuilt dual negative banks with a
rescue-sensitive gate, instead of replacing the learned dual bank outright.

The design goal was:

- keep the accepted branchwise-max score as the outer anchor
- keep the rescue-weighted anchored dual lift that already wins the
  `0.10–0.25%` micro-budget lane
- keep the dual-only teacher-guided harmful-state rebuild that created the
  target-heavy `~0.50%` lane in the previous hard-swap follow-up
- but blend the rebuilt dual bank in only when a learned rescue-sensitive gate
  says it should matter

This is the direct “conditional bank edit” follow-up to the positive hard-swap
dual-teacher rescue-weighted result.

## Implementation

- New head:
  `GatedDualTeacherRebuiltNegativeBankRescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_gated_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_gated_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_gated_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the earlier hard-swap dual-teacher rescue-weighted lift:

- the same rescue-weighted anchored dual-lift base is fit first
- the same dual-only teacher-guided harmful-state rebuild is constructed
- but the dual branch now computes scores against both the learned and rebuilt
  negative banks
- and a learned scalar gate blends between those score families before the
  outer rescue-weighted anchored lift is applied
- after attaching the rebuilt bank, only the new blend parameters are refit on
  the training pack

So this is a narrow attempt to keep the `0.50%` hard-swap gain while
suppressing the rebuild when it is not needed.

## Held-Out Result

### `prototype_gated_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match unchanged at `96.51%`
- overall mean delta regret `0.0000`

So the plain branch again collapses to baseline.

### `prototype_gated_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed positive, but fully dominated.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.59%`
- overall mean delta regret `-0.0063`

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

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- hard near-tie mean delta regret `-0.0069`
- overall target match `96.73%`
- overall mean delta regret `-0.0126`

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
- overall target match `96.82%`
- overall mean delta regret `-0.0154`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.87%`
- overall mean delta regret `-0.0164`

Large-gap controls stayed clean:

- large-gap target match stayed at `99.89%`
- mean delta regret stayed non-positive
- no harmful large-gap miss pattern appeared

## Comparison against rescue-weighted anchored lift

At `0.10–0.25%`, this just recreates the rescue-weighted micro-budget lane:

- `0.10%`: same `50%` held-out `stable_positive_v2`, same `90.45%` hard
  near-tie, and only a tiny aggregate change (`-0.0064 -> -0.0063`)
- `0.25%`: same `50%`, same `90.45%`, and effectively identical aggregate
  regret (`-0.0088`)

At `0.50%`, it gives back the hard-swap improvement entirely:

- rescue-weighted anchored lift:
  `50%` held-out `stable_positive_v2`, `90.39% -> 90.45%`, overall mean delta
  regret `-0.0105`
- gated dual-teacher rebuild:
  the same `50%` / `90.45%` / `-0.0105`

So the learned blend suppresses the rebuilt dual bank so strongly that the
`0.50%` hard-swap gain disappears.

## Comparison against the hard-swap dual-teacher rescue-weighted branch

The earlier hard-swap branch reached a real new target-heavy point at `0.50%`:

- `66.7%` held-out `stable_positive_v2`
- hard near-tie `90.39% -> 90.53%`
- overall mean delta regret `-0.0113`

The gated version falls back to:

- only `50%`
- only `90.39% -> 90.45%`
- overall mean delta regret `-0.0105`

So the conditional blend is not a safer version of the hard swap. It simply
removes the one new lane that the hard swap created.

## Comparison against the accepted branchwise-max reference

At `0.10–0.25%`, the accepted branchwise-max reference is still beaten by the
older rescue-weighted micro-budget companion, and this gated follow-up only
reproduces that same rescue-weighted lane.

At `0.50%`, accepted branchwise-max remains:

- `50%` held-out `stable_positive_v2`
- hard near-tie `90.39% -> 90.45%`
- overall mean delta regret `-0.0111`

The gated follow-up reaches the same target slice but weaker aggregate regret:

- `50%`
- `90.39% -> 90.45%`
- `-0.0105`

And above `0.75%`, accepted branchwise-max dominates again:

- `0.75%`: same `66.7%` / `90.53%`, but accepted gets better overall mean
  delta regret `-0.0137` vs `-0.0126`
- `1.00%`: accepted `83.3%` / `90.60%` / `-0.0145`; gated branch only
  `66.7%` / `90.53%` / `-0.0145`
- `2.00%`: accepted `100%` / `90.68%` / `-0.0167`; gated branch only
  `83.3%` / `90.60%` / `-0.0164`

So this never beats the accepted branchwise-max frontier at any matched
budget.

## Blend-Gate Diagnostics

After the post-attach gate refit, the learned blend stayed weak:

- plain: `dual_rebuild_blend_scale = 0.120`, `dual_rebuild_blend_bias = -0.881`
- hybrid: `dual_rebuild_blend_scale = 0.139`, `dual_rebuild_blend_bias = -0.863`

That is the clearest mechanism result from this experiment: the model learned
to keep the rebuilt dual bank mostly suppressed. So the soft conditional blend
does not preserve the positive hard-swap target lift; it retracts back toward
the older rescue-weighted curve.

## Interpretation

This closes the “soft conditional teacher-bank blend” hypothesis.

Current read:

- hard dual-bank replacement was useful only because it committed to a real
  target-heavy edit around `0.50%`
- once that edit is softened into a rescue-sensitive blend, the model learns to
  keep the rebuilt bank mostly off
- that removes the `0.50%` gain while still never improving the higher-budget
  accepted branchwise-max frontier

So the right conclusion is not “soften the bank edit.” It is:

- keep rescue-weighted anchored lift for the `0.10–0.25%` micro-budget lane
- keep the hard-swap dual-teacher rescue-weighted branch as the narrow
  target-heavy `~0.50%` companion if that tradeoff is desired
- do not reopen rescue-sensitive learned blending between the learned and
  rebuilt dual negative banks

## Decision

Close:

- `prototype_gated_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_gated_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`
