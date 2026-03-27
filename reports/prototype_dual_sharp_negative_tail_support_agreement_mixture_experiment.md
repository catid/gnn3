# Prototype Dual-Sharp Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the live negative-tail and sharp-negative-tail gains can be
combined by applying sharpness-gated negative cleanup only on the **dual**
branch while keeping the **shared** branch on the older fixed negative-tail
cleanup.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the shared branch on fixed negative cleanup
- move adaptive sharpness-gated negative cleanup only to the dual branch
- preserve more of the high-recall lane while improving aggregate quality

This is the direct complement to the failed shared-only sharp-cleanup branch.

## Implementation

- New head: `DualSharpNegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_dual_sharp_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_dual_sharp_negative_tail_support_agree_mix`
  - `prototype_dual_sharp_negative_tail_support_agree_mix_hybrid`

Relative to the sharp-negative-tail branch:

- the shared branch keeps fixed negative-tail cleanup
- the dual branch uses sharpness-gated negative cleanup

So this is the opposite branch split from the dead shared-only variant.

## Held-Out Result

### `prototype_dual_sharp_negative_tail_support_agree_mix`

Closed weak positive.

Best point:

- budget `0.10%`
- overall coverage `0.10%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0010`
- overall mean delta regret `-0.0005`

So the plain branch finds a tiny real niche, but it is not competitive.

### `prototype_dual_sharp_negative_tail_support_agree_mix_hybrid`

Closed weak positive.

Best point:

- budget `0.25%`
- overall coverage `0.18%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0019`
- overall mean delta regret `-0.0033`

It holds the same weak niche through every higher nominal budget, so it never
expands into a real frontier branch.

## Comparison against live leads

`prototype_memory_agree_blend_hybrid @ 0.25%`

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- overall mean delta regret `-0.0049`

`prototype_dual_sharp_negative_tail_support_agree_mix_hybrid @ 0.25%`

- lower overall coverage `0.18%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie only `90.53% -> 90.60%`
- overall mean delta regret only `-0.0033`

So even the micro-budget memory-agreement follow-up beats it.

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_dual_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage only `0.18%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie only `90.53% -> 90.60%`
- overall mean delta regret only `-0.0033`

So dual-only adaptive cleanup does not recover the real low-coverage frontier.

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_dual_sharp_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage only `0.18%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie only `90.53% -> 90.60%`
- overall mean delta regret `-0.0033`

So it also does not preserve the high-recall lane.

## Interpretation

This is not a live architecture direction.

Current read:

- shared-only sharp cleanup failed
- dual-only sharp cleanup also fails to recover either live lane
- the useful sharp-negative-tail behavior appears to require coordinated
  cleanup across both branches rather than branch-splitting it

So the shortlist remains unchanged:

- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` remains the best
  sub-`1%` full-band and coverage-efficient matched-band branch
- `prototype_negative_tail_support_agree_mix_hybrid` remains the best
  high-recall branch around `1%`
- `prototype_support_weighted_agree_mix_hybrid` remains the higher-budget
  matched-band reference

## Decision

Close:

- `prototype_dual_sharp_negative_tail_support_agree_mix`
- `prototype_dual_sharp_negative_tail_support_agree_mix_hybrid`
