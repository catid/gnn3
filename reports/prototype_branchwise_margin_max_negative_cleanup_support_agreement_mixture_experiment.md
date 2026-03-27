# Prototype Branchwise-Margin-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the positive branchwise-max result improves if the fixed-cleanup
path must beat the branch-strength sharp path by a learned **shared/dual
branch margin** before taking over inside each branch.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the branch-strength sharp negative-tail cleanup as the base
- keep the older fixed negative-tail cleanup as the secondary source
- keep the successful branch-local fusion level from branchwise max
- but require `fixed > sharp + margin` before the fixed path can win inside a
  branch
- preserve the sharp low-coverage lane while still retaining the higher-budget
  recall gain

This is the direct follow-up to the positive branchwise-max result and the
closed branchwise-lift result.

## Implementation

- New head:
  `BranchwiseMarginMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_branchwise_margin_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_branchwise_margin_max_negative_cleanup_support_agree_mix`
  - `prototype_branchwise_margin_max_negative_cleanup_support_agree_mix_hybrid`

Relative to branchwise max:

- the same fixed and branch-strength sharp branch scores are computed
- but the branchwise union becomes `max(sharp, fixed - margin)` with separate
  learned shared and dual margins
- then the usual agreement mixture combines those branchwise fused scores

So this isolates one question: was hard branchwise max correct only because it
was branch-local, or because it was also margin-free?

## Held-Out Result

### `prototype_branchwise_margin_max_negative_cleanup_support_agree_mix`

Closed, effectively dead.

At `0.10%` through `1.50%` nominal budgets:

- held-out `stable_positive_v2` recovery stayed at `0%`
- hard near-tie target match stayed unchanged at `90.53%`
- overall mean delta regret stayed `0.0000`

At `2.00%` nominal budget it finally woke up slightly:

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall mean delta regret only `-0.0008`

So the plain branch is not useful.

### `prototype_branchwise_margin_max_negative_cleanup_support_agree_mix_hybrid`

Closed positive, but still not a new lead.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall mean delta regret `-0.0057`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall mean delta regret `-0.0131`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.81%`
- overall mean delta regret `-0.0152`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.82%`
- overall mean delta regret `-0.0154`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.87%`
- overall mean delta regret `-0.0164`

So the hybrid does restore the full `75%` / `90.73%` matched-band frontier by
`1.0%` coverage, but it never reaches the fixed negative-tail branch's `100%`
held-out recall lane and still trails the best live references.

## Comparison against nearby variants

### Versus the closed branchwise lift

`prototype_branchwise_lift_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `50%`
- hard near-tie `90.53% -> 90.66%`
- overall mean delta regret `-0.0149`

`prototype_branchwise_margin_max_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0152`

So adding a branch margin on top of branch-local hard fusion clearly works
better than replacing branchwise max with a softer branchwise lift.

### Versus the live sharp-negative branch below `1%`

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_branchwise_margin_max_negative_cleanup_support_agree_mix_hybrid @ 0.75%`

- only `50%` held-out `stable_positive_v2`
- only `90.53% -> 90.66%`
- overall mean delta regret `-0.0131`

So the branch margin is still not enough to preserve the live sharp-negative
lane below `1%` coverage.

At `1.00%` nominal budget:

- both reach the full `75%` / `90.73%` frontier
- but the branch-margin head only reaches `-0.0152`, effectively matching but
  not beating the live sharp-negative branch

So it does not become the new best `~1%` point.

### Versus branchwise max

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0145`

`prototype_branchwise_margin_max_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- same `75%` / `90.73%`
- slightly better overall mean delta regret `-0.0152`

So the margin does improve the `1.0%` point slightly.

But at `1.50%` and `2.00%`:

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- `1.50%`: overall mean delta regret `-0.0159`
- `2.00%`: `100%` held-out `stable_positive_v2`, `90.53% -> 90.80%`,
  overall mean delta regret `-0.0167`

`prototype_branchwise_margin_max_negative_cleanup_support_agree_mix_hybrid`

- `1.50%`: overall mean delta regret `-0.0154`
- `2.00%`: still only `75%` held-out `stable_positive_v2`,
  still only `90.53% -> 90.73%`, overall mean delta regret `-0.0164`

So the learned margin helps at `1.0%`, but it suppresses too much of the
higher-budget recall upside that made branchwise max valuable.

### Versus the older higher-budget matched-band reference

`prototype_support_weighted_agree_mix_hybrid @ 1.50%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0158`

`prototype_branchwise_margin_max_negative_cleanup_support_agree_mix_hybrid @ 1.50%`

- same `75%` / `90.73%`
- weaker overall mean delta regret `-0.0154`

So it still does not replace the older higher-budget reference, let alone the
newer branchwise-max branch.

## Interpretation

The positive branchwise-max result was not just about adding a little
thresholding before fixed cleanup can win. It needed the full branchwise hard
union to keep its higher-budget recall upside.

Current read:

- fusion really does belong inside the shared and dual branches
- adding a branch margin is better than a softer learned branchwise lift
- but it still does not preserve the sharp-negative low-coverage lane
- and it still gives back too much of the branchwise-max higher-budget recall
  upside

So the right conclusion is:

- branch-local fusion was the important structural correction
- but hard branchwise max is still the better live choice than a margin-gated
  branchwise max

## Decision

Close:

- `prototype_branchwise_margin_max_negative_cleanup_support_agree_mix`
- `prototype_branchwise_margin_max_negative_cleanup_support_agree_mix_hybrid`

Live shortlist remains:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best
  sub-`1%` full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out
  recall around `1%`
- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` for the
  higher-budget matched-band and higher-budget max-recall lane
