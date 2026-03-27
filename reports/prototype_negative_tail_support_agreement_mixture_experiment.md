# Prototype Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the soft-tail gain in the live support-weighted
agreement-mixture family comes mainly from suppressing **diffuse negative-bank
overlap**, while leaving positive-bank pooling untouched.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep positive-bank pooling unchanged
- apply soft tail suppression only to the shared and dual negative banks
- see whether that preserves positive supporting mass while sharpening the real
  sparse-positive frontier

This is the direct follow-up to the broader soft-tail result.

## Implementation

- New head: `NegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_negative_tail_support_agree_mix`
  - `prototype_negative_tail_support_agree_mix_hybrid`

Relative to the full soft-tail branch:

- only the shared negative bank gets a tail penalty
- only the dual negative bank gets a tail penalty
- positive banks keep the original support-weighted full-bank pooling

## Held-Out Result

### `prototype_negative_tail_support_agree_mix`

Dead.

Best point:

- budget `2.0%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match slightly down at `90.53% -> 90.46%`
- overall mean delta regret `-0.0025`

So negative-tail cleanup without the risk branch is still inert.

### `prototype_negative_tail_support_agree_mix_hybrid`

Real positive result.

At `0.25%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0066`
- overall target match `96.51% -> 96.59%`
- overall mean delta regret `-0.0055`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0066`
- overall target match `96.51% -> 96.71%`
- overall mean delta regret `-0.0111`

At `1.0%` nominal budget:

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- hard near-tie mean delta regret `-0.0100`
- overall target match `96.51% -> 96.76%`
- overall mean delta regret `-0.0131`

At `1.5%` nominal budget:

- overall coverage `1.51%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- hard near-tie mean delta regret `-0.0100`
- overall target match `96.51% -> 96.78%`
- overall mean delta regret `-0.0138`

At `2.0%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- hard near-tie mean delta regret `-0.0100`
- overall target match `96.51% -> 96.78%`
- overall mean delta regret `-0.0138`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret `-0.0065`

## Comparison against live leads

`prototype_soft_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0104`

`prototype_negative_tail_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- held-out `stable_positive_v2` recovery only `50%`
- hard near-tie only `90.53% -> 90.66%`
- overall mean delta regret `-0.0111`

So the full soft-tail branch remains the better sub-`1%` full-band lead.

`prototype_support_weighted_agree_mix_hybrid @ 1.0%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

`prototype_negative_tail_support_agree_mix_hybrid @ 1.0%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

So the older support-weighted agreement-mixture branch still wins on aggregate
matched-band regret, but the new negative-tail branch is stronger on held-out
stable-positive recall and hard-slice quality.

## Interpretation

This is another real architecture improvement, but in a different direction
from the full soft-tail result.

Current read:

- broad soft-tail cleanup is best for reaching the full frontier band below
  `1%` coverage
- negative-bank-only cleanup is best once the goal becomes **maximum held-out
  sparse-positive recall** around `1%` coverage
- the soft-tail gain does appear to be driven substantially by suppressing
  diffuse negative overlap

So the shortlist expands:

- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_soft_tail_support_agree_mix_hybrid` remains the best sub-`1%`
  full-band lead
- `prototype_negative_tail_support_agree_mix_hybrid` becomes the best
  high-recall branch around `1%` coverage
- `prototype_support_weighted_agree_mix_hybrid` remains the best matched-band
  branch overall on aggregate regret

## Decision

Close:

- `prototype_negative_tail_support_agree_mix`

Keep alive:

- `prototype_negative_tail_support_agree_mix_hybrid`
