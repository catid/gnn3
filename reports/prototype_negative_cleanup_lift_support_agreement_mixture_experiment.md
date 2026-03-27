# Prototype Negative-Cleanup Lift Support Agreement-Mixture Experiment

## Question

Test whether the fixed and sharp negative-cleanup lanes can be combined more
selectively than the weak cleanup-blend and cleanup-max follow-ups:

- keep the sharp negative-tail branch as the default score
- allow only a nonnegative fixed-branch lift
- gate that lift so it only appears when the fixed branch is distinctly stronger

The design goal was:

- preserve the sharp branch's strong aggregate-quality behavior below `1%`
  coverage
- recover some of the fixed branch's sparse-positive recall around `1%`
- avoid both the blend branch's weak-middle collapse and the max branch's broad
  late union

## Implementation

- New head: `NegativeCleanupLiftSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_negative_cleanup_lift_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_negative_cleanup_lift_support_agree_mix`
  - `prototype_negative_cleanup_lift_support_agree_mix_hybrid`

Relative to the live negative-cleanup heads:

- the model computes one score with fixed negative-tail cleanup
- it computes a second score with sharp negative-tail cleanup
- it uses the sharp score as the base path
- it adds only a gated positive lift from the fixed score when `fixed > sharp`

So this is a one-sided retrieval lift, not a learned average and not a hard
union.

## Held-Out Result

### `prototype_negative_cleanup_lift_support_agree_mix`

Dead.

- no held-out `stable_positive_v2` recovery at any budget
- hard near-tie unchanged at `90.53%`
- overall mean delta regret `0.0000`

### `prototype_negative_cleanup_lift_support_agree_mix_hybrid`

Closed weak positive.

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0023`
- overall target match `96.51% -> 96.63%`
- overall mean delta regret `-0.0068`

At `1.50%` nominal budget:

- overall coverage `1.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0042`
- overall target match `96.51% -> 96.71%`
- overall mean delta regret `-0.0095`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0042`
- overall target match `96.51% -> 96.73%`
- overall mean delta regret `-0.0104`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret `-0.0065`
- large-gap mean delta miss `0.0000`

## Comparison against live leads

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_negative_cleanup_max_support_agree_mix_hybrid @ 1.50%`

- overall coverage `1.51%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0118`

So the lift branch does not preserve either live lane:

- it never reaches the sharp branch's `75%` / `90.73%` frontier
- it never reaches the fixed branch's `100%` / `90.80%` recall lane
- it is weaker than even the cleanup-max follow-up once coverage rises

## Interpretation

This confirms that using the sharp branch as the base and allowing only a
positive fixed-branch lift is still too restrictive to recover the real sparse
correction family:

- below `1%` coverage it only finds a weak `25%` / `90.60%` niche
- by `1.5–2.0%` coverage it only reaches `50%` / `90.66%`
- the gate never learns a useful selective rescue of the fixed-branch positives

So the fixed-branch recall lane is not recoverable here by simply adding a
one-sided lift on top of the sharp branch.

## Decision

Close:

- `prototype_negative_cleanup_lift_support_agree_mix`
- `prototype_negative_cleanup_lift_support_agree_mix_hybrid`

Keep the live shortlist unchanged:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best sub-`1%`
  full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out recall
  around `1%`
- `prototype_support_weighted_agree_mix_hybrid` as the higher-budget
  matched-band reference
