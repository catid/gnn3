# Prototype Shared-Sharp Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the live sharp-negative-tail and negative-tail gains can be
combined by applying sharpness-gated negative cleanup only on the **shared**
branch while keeping the **dual** branch on the older fixed negative-tail
cleanup.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the dual branch on fixed negative cleanup to preserve the recall lane
- apply adaptive sharpness-gated cleanup only on the shared branch
- improve aggregate quality without giving up the older negative-tail branch's
  recall behavior

This is the direct follow-up to the positive sharp-negative-tail result.

## Implementation

- New head: `SharedSharpNegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_shared_sharp_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_shared_sharp_negative_tail_support_agree_mix`
  - `prototype_shared_sharp_negative_tail_support_agree_mix_hybrid`

Relative to the sharp-negative-tail branch:

- the shared branch uses sharpness-gated negative-bank cleanup
- the dual branch keeps the older fixed negative-tail cleanup

So this is a branch-asymmetric version of the negative-bank cleanup idea.

## Held-Out Result

### `prototype_shared_sharp_negative_tail_support_agree_mix`

Closed weak positive.

Best point:

- budget `0.75%`
- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0029`
- overall mean delta regret `-0.0014`

So the plain branch found some retrieval signal, but it stayed far below the
live support-weighted family.

### `prototype_shared_sharp_negative_tail_support_agree_mix_hybrid`

Dead.

At `1.00%` nominal budget:

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie unchanged at `90.53%`
- overall mean delta regret `-0.0012`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie unchanged at `90.53%`
- overall mean delta regret `-0.0027`

So the hybrid fully collapsed on the real target slice.

## Comparison against live leads

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_shared_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie unchanged at `90.53%`
- overall mean delta regret `0.0000`

So moving sharpness-gated cleanup to the shared branch only destroys the live
low-coverage frontier.

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_shared_sharp_negative_tail_support_agree_mix_hybrid @ 1.00%`

- same overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie unchanged
- overall mean delta regret only `-0.0012`

So leaving the dual branch fixed is not enough to preserve the high-recall
lane.

## Interpretation

This is not a live architecture direction.

Current read:

- the sharp-negative-tail gain is not coming from shared-only cleanup
- the fixed negative-tail recall lane is also not preserved if only the dual
  branch stays fixed while the shared branch changes
- the successful negative-bank cleanup branches appear to rely on coordinated
  branch behavior, not this shared-only asymmetric split

So the shortlist is unchanged:

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

- `prototype_shared_sharp_negative_tail_support_agree_mix`
- `prototype_shared_sharp_negative_tail_support_agree_mix_hybrid`
