# Prototype Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch improves if the
newer branch-strength sharp negative-tail cleanup and the older fixed
negative-tail cleanup are fused **inside each branch** before the outer
agreement mixture, instead of taking a max only after the final mixed score.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the improved branch-strength sharp cleanup
- keep the older fixed negative-tail cleanup
- take `max(fixed, sharp)` separately in the shared branch and in the dual
  branch
- then let the outer agreement mixture combine those branchwise fused scores
- preserve more of the fixed branch's recall without giving up the sharper
  branch's low-coverage quality

This is the direct follow-up to the failed global max fusion.

## Implementation

- New head:
  `BranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the older branch-strength global max:

- the same fixed and branch-strength sharp branch scores are used
- but the max happens inside the shared and dual branches separately
- only after that does the agreement gate mix shared and dual

So this is still a narrow fusion test, but it changes the fusion level in the
one place that was still ambiguous.

## Held-Out Result

### `prototype_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall target match unchanged at `96.51%`
- overall mean delta regret `0.0000`

So the plain branch fully collapses to baseline.

### `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Real positive result. This branch is alive.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.57%`
- overall mean delta regret `-0.0050`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0066`
- overall target match `96.76%`
- overall mean delta regret `-0.0137`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.79%`
- overall mean delta regret `-0.0145`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.84%`
- overall mean delta regret `-0.0159`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- hard near-tie mean delta regret `-0.0100`
- overall target match `96.89%`
- overall mean delta regret `-0.0167`

Large-gap controls stayed clean:

- large-gap target match stayed at `99.90%`
- mean delta regret remained non-positive
- no harmful large-gap miss pattern appeared

## Comparison against live leads

### Versus the older global max fusion

`prototype_branch_strength_negative_cleanup_max_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `50%`
- hard near-tie `90.53% -> 90.66%`
- overall mean delta regret `-0.0127`

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0145`

So moving the max inside the shared and dual branches is the key difference.

### Versus the live sharp-negative branch below `1%`

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0152`

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- same `75%` / `90.73%` frontier
- slightly weaker overall mean delta regret `-0.0145`

So this does not replace the live sharp-negative branch as the best `~1%`
coverage point.

### Versus the older higher-budget matched-band reference

`prototype_support_weighted_agree_mix_hybrid @ 1.50%`

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0158`

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 1.50%`

- same `1.52%` coverage
- same `75%` / `90.73%` frontier
- slightly better overall mean delta regret `-0.0159`

So this does supersede the older support-weighted agreement-mixture branch as
the higher-budget matched-band reference.

### Versus the fixed negative-tail high-recall branch

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 2.00%`

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie `90.53% -> 90.80%`
- overall mean delta regret `-0.0167`

So the fixed negative-tail branch still owns the lighter `~1%` high-recall
lane, but the new branch becomes the stronger higher-budget high-recall branch.

## Interpretation

This is the first fusion of the improved sharp branch and the fixed negative
tail that actually survives.

Current read:

- global max after the final mixed score was the wrong fusion level
- branchwise max before agreement mixing preserves more of the useful
  complementary signal
- below `1%`, the older sharp-negative branch still remains best
- around `1%`, this branch now matches the full `75%` / `90.73%` frontier but
  is still slightly weaker than the sharp-negative branch on aggregate regret
- at `1.5–2.0%`, this becomes the strongest higher-budget branch:
  - it edges the older support-weighted agreement-mixture matched-band
    reference at `1.5%`
  - and at `2.0%` it reaches `100%` held-out sparse-positive recovery and
    `90.80%` hard near-tie with the best overall regret seen in this family

So the right conclusion is not “replace the live low-coverage lanes.” It is:

- keep the current low-coverage leaders
- add this as the new higher-budget matched-band and max-recall leader

## Decision

Close:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix`

Keep alive:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Updated live shortlist:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best
  sub-`1%` full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out
  recall around `1%`
- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` as the
  new higher-budget matched-band and higher-budget max-recall leader
- keep `prototype_support_weighted_agree_mix_hybrid` only as the older
  higher-budget reference behind the new branchwise-max result
