# Prototype Floor+Sharp Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the live sharp-negative-tail branch improves if it keeps a fixed
negative-tail cleanup floor and then adds extra sharpness-gated cleanup only
when the negative bank is diffuse.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- preserve the older negative-tail branch's high-recall floor
- add the newer sharpness-gated cleanup as an extra term
- try to combine the recall of the negative-tail branch with the stronger
  aggregate regret of the sharp-negative-tail branch

This is the direct follow-up to the positive sharp-negative-tail result.

## Implementation

- New head: `FloorSharpNegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_floor_sharp_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_floor_sharp_negative_tail_support_agree_mix`
  - `prototype_floor_sharp_negative_tail_support_agree_mix_hybrid`

Relative to the sharp-negative-tail branch:

- negative-bank tail cleanup now has a fixed floor term
- it also has a separate adaptive term
- the adaptive term is scaled by negative-bank sharpness

So this is a two-part negative-bank cleanup:

- always-on cleanup
- plus extra cleanup only when the bank is diffuse

## Held-Out Result

### `prototype_floor_sharp_negative_tail_support_agree_mix`

Dead.

Best point:

- all budgets
- overall coverage `0%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie unchanged at `90.53%`
- overall mean delta regret `0.0000`

So the plain branch collapsed fully to baseline.

### `prototype_floor_sharp_negative_tail_support_agree_mix_hybrid`

Closed negative.

At `1.00%` nominal budget:

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`
- overall mean delta regret `-0.0034`

At `1.50%` nominal budget:

- overall coverage `1.51%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match only `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0023`
- overall mean delta regret `-0.0084`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match only `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0023`
- overall mean delta regret `-0.0096`

So the hybrid lost essentially all of the real frontier signal. It only found a
late weak niche at high coverage and never approached either live negative-bank
branch.

## Comparison against live leads

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_floor_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie unchanged at `90.53%`
- overall mean delta regret `-0.0031`

So adding the fixed floor destroyed the strong low-coverage frontier.

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_floor_sharp_negative_tail_support_agree_mix_hybrid @ 1.00%`

- same overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie unchanged
- overall mean delta regret only `-0.0034`

So it also completely failed at the high-recall lane it was supposed to
preserve.

## Interpretation

This is not a live architecture direction.

Current read:

- fixed negative-tail cleanup and sharpness-gated cleanup are each useful alone
- but stacking them additively over-suppresses the bank
- the floor term removes too much supporting mass before the adaptive term can
  provide any benefit

So the shortlist is unchanged:

- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` remains the best
  sub-`1%` full-band and coverage-efficient matched-band branch
- `prototype_negative_tail_support_agree_mix_hybrid` remains the best
  high-recall branch around `1%` coverage
- `prototype_support_weighted_agree_mix_hybrid` remains the higher-budget
  matched-band reference

## Decision

Close:

- `prototype_floor_sharp_negative_tail_support_agree_mix`
- `prototype_floor_sharp_negative_tail_support_agree_mix_hybrid`
