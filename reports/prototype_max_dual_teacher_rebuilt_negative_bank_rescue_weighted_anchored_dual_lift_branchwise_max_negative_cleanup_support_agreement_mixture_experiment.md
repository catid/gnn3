# Prototype Max Dual Teacher-Rebuilt Negative-Bank Rescue-Weighted Anchored Dual-Lift Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the positive dual-teacher rescue-weighted branch should combine
the learned and rebuilt dual negative-bank score families with a hard max,
instead of either replacing the learned dual bank outright or blending it back
softly.

The design goal was:

- keep the accepted branchwise-max score as the outer anchor
- keep the rescue-weighted anchored dual lift that already wins the
  `0.10–0.25%` micro-budget lane
- keep the dual-only teacher-guided harmful-state rebuild that created the
  target-heavy `~0.50%` lane in the hard-swap follow-up
- but replace the failed soft blend with a hard per-state union between the
  learned-bank and rebuilt-bank dual branch scores

This is the direct hard-max follow-up to the closed gated dual-bank blend.

## Implementation

- New head:
  `MaxDualTeacherRebuiltNegativeBankRescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_max_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_max_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_max_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the earlier hard-swap and gated variants:

- the same rescue-weighted anchored dual-lift base is fit first
- the same dual-only teacher-guided harmful-state rebuild is constructed
- the dual branch computes scores against both the learned and rebuilt dual
  negative banks
- fixed and sharp dual scores each take a hard max between the learned-bank and
  rebuilt-bank versions
- the outer rescue-weighted anchored lift then operates on that maxed dual
  branch

So this is the bank-level analogue of the broader branchwise-max win: keep the
hard union, but move it down to the learned-vs-rebuilt dual-bank choice.

## Held-Out Result

### `prototype_max_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match unchanged at `96.51%`
- overall mean delta regret `0.0000`

So the plain branch again collapses to baseline.

### `prototype_max_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Real positive result. This branch is alive.

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
- overall target match `96.63%`
- overall mean delta regret `-0.0082`

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.67%`
- overall mean delta regret `-0.0104`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.76%`
- overall mean delta regret `-0.0137`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.80%`
- overall mean delta regret `-0.0148`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.84%`
- overall mean delta regret `-0.0159`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- hard near-tie mean delta regret `-0.0090`
- overall target match `96.88%`
- overall mean delta regret `-0.0165`

Large-gap controls stayed clean:

- large-gap target match stayed at `99.89%`
- mean delta regret stayed non-positive
- no harmful large-gap miss pattern appeared

## Comparison against rescue-weighted anchored lift

At `0.10%`, the new branch essentially preserves the rescue-weighted
micro-budget win:

- same `50%` held-out `stable_positive_v2`
- same `90.39% -> 90.45%` hard near-tie
- overall mean delta regret `-0.0064` versus `-0.0064`

At `0.25–0.50%`, it is slightly weaker than rescue-weighted alone:

- `0.25%`: `-0.0082` versus `-0.0088`
- `0.50%`: `-0.0104` versus `-0.0105`

So the hard bankwise max does not preserve the earlier hard-swap `0.50%`
target-heavy lane.

But from `0.75%` upward it becomes much stronger:

- `0.75%`: `83.3%` / `90.60%` / `-0.0137`, versus rescue-weighted alone
  only `66.7%` / `90.53%` / `-0.0129`
- `1.00–2.00%`: it consistently stays above the rescue-weighted curve on the
  target slice and on overall regret

So hard bankwise max does not help at `0.50%`, but it does stabilize the
teacher-guided rebuild into the stronger matched-band regime.

## Comparison against the hard-swap dual-teacher rescue-weighted branch

The earlier hard-swap branch created the narrow `0.50%` target-heavy lane:

- `66.7%` held-out `stable_positive_v2`
- hard near-tie `90.39% -> 90.53%`
- overall mean delta regret `-0.0113`

The hard max version gives that back at `0.50%`, but improves materially once
coverage rises:

- `0.75%`: hard swap only `66.7%` / `90.53%` / `-0.0129`, hard max reaches
  `83.3%` / `90.60%` / `-0.0137`
- `1.00%`: hard swap only `66.7%` / `90.53%` / `-0.0145`, hard max reaches
  `83.3%` / `90.60%` / `-0.0148`
- `1.50%`: hard swap `83.3%` / `90.60%` / `-0.0155`, hard max improves the
  same target slice to `-0.0159`

So hard max is not a safer version of the `0.50%` lane. It is a stronger
`0.75–1.5%` matched-band continuation of the same teacher-guided idea.

## Comparison against the accepted branchwise-max reference

At `0.75%`, the accepted branchwise-max reference had:

- `66.7%` held-out `stable_positive_v2`
- hard near-tie `90.39% -> 90.53%`
- overall mean delta regret `-0.0137`

The hard max dual-teacher branch improves that to:

- `83.3%`
- `90.39% -> 90.60%`
- `-0.0137`

At `1.00%`, accepted branchwise-max had:

- `83.3%`
- `90.39% -> 90.60%`
- `-0.0145`

The hard max dual-teacher branch keeps the same target slice and improves
overall mean delta regret to `-0.0148`.

At `1.50%`, accepted branchwise-max had:

- `83.3%`
- `90.39% -> 90.60%`
- `-0.0159`

The hard max dual-teacher branch again keeps the same target slice and edges
aggregate regret to `-0.0159` (`-0.01594` versus `-0.01586`).

At `2.00%`, however, accepted branchwise-max still wins:

- accepted: `100%` held-out `stable_positive_v2`, hard near-tie
  `90.39% -> 90.68%`, overall mean delta regret `-0.0167`
- hard max dual-teacher: only `83.3%` / `90.60%` / `-0.0165`

So this is not a full replacement for accepted branchwise-max. It is a new
leader only in the `0.75–1.5%` matched-band range.

## Interpretation

This is the first bank-level hard union that survives in the teacher-guided
branchwise-max family.

Current read:

- soft conditional blending was too timid and collapsed back to the
  rescue-weighted curve
- hard swapping was too aggressive and only created a narrow `0.50%` lane
- hard max between the learned-bank and rebuilt-bank dual score families finds
  the right middle ground
  - it preserves the rescue-weighted micro-budget point at `0.10%`
  - it gives back the hard-swap `0.50%` lane
  - but it becomes the strongest branchwise-max-family matched-band contender
    from `0.75%` through `1.5%`
  - it still does not replace the accepted `2.0%` high-recall endpoint

So the updated live interpretation is:

- rescue-weighted anchored dual lift for `0.10–0.25%`
- hard-swap dual-teacher rescue-weighted branch for the narrow target-heavy
  `~0.50%` lane
- hard-max dual-teacher rescue-weighted branch for the strongest
  `0.75–1.5%` matched-band lane
- accepted branchwise-max for the `2.0%` high-recall endpoint

## Decision

Close:

- `prototype_max_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`

Keep alive:

- `prototype_max_dual_teacher_rebuilt_negative_bank_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`
