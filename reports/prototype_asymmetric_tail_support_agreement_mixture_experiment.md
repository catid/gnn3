# Prototype Asymmetric-Tail Support Agreement-Mixture Experiment

## Question

Test whether the high-recall negative-tail branch improves if it adds back a
smaller positive-bank soft-tail cleanup instead of leaving positive-bank
pooling fully untouched.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep stronger negative-bank tail suppression
- add only a smaller positive-bank tail penalty
- preserve the negative-tail branch's `100%` held-out sparse-positive recall
  around `1%` coverage while improving the sub-`1%` frontier

This is the direct follow-up to the positive negative-tail-only result.

## Implementation

- New head: `AsymmetricTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_asymmetric_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_asymmetric_tail_support_agree_mix`
  - `prototype_asymmetric_tail_support_agree_mix_hybrid`

Relative to the negative-tail branch:

- shared and dual negative banks keep the stronger soft-tail penalty
- shared and dual positive banks now get a smaller soft-tail penalty
- all banks still use support-weighted full-bank `logsumexp` pooling

So this tests whether a lighter positive cleanup can sharpen the bank without
giving back the recovered sparse-positive frontier.

## Held-Out Result

### `prototype_asymmetric_tail_support_agree_mix`

Dead.

Best point:

- budget `0.75%`
- overall coverage `0.74%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall mean delta regret `0.0000`

So the non-hybrid branch is inert.

### `prototype_asymmetric_tail_support_agree_mix_hybrid`

Closed weak positive.

At `0.10%` nominal budget:

- overall coverage `0.10%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.51% -> 96.58%`
- overall mean delta regret `-0.0055`

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0066`
- overall target match `96.51% -> 96.64%`
- overall mean delta regret `-0.0077`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.66%`
- overall mean delta regret `-0.0085`

At `1.00%` nominal budget:

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.71%`
- overall mean delta regret `-0.0105`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.81%`
- overall mean delta regret `-0.0147`

So the hybrid does preserve the full `75%` / `90.73%` frontier band, but it
does not improve either live lane.

## Comparison against live leads

`prototype_soft_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0104`

`prototype_asymmetric_tail_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- the same `75%` held-out stable-positive-v2 recovery
- the same `90.53% -> 90.73%` hard near-tie band
- weaker overall mean delta regret at only `-0.0085`

So the older full soft-tail branch remains the better sub-`1%` full-band
contender.

`prototype_support_weighted_agree_mix_hybrid @ 1.00%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

`prototype_asymmetric_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0105`

So the older support-weighted agreement-mixture branch still wins on aggregate
matched-band quality.

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_asymmetric_tail_support_agree_mix_hybrid @ 1.00%`

- same overall coverage `1.00%`
- held-out `stable_positive_v2` recovery only `75%`
- hard near-tie only `90.53% -> 90.73%`
- overall mean delta regret only `-0.0105`

So adding even a smaller positive-tail penalty gives back the negative-tail
branch's strongest recall and hard-slice advantage.

## Interpretation

This is not a new live architecture lead.

Current read:

- negative-bank cleanup is real
- broad symmetric soft-tail cleanup is also real
- but mixing the two asymmetrically just drifts back toward the middle
- the positive-bank tail penalty appears to remove useful support mass before it
  buys enough extra precision to matter

So the shortlist stays:

- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_soft_tail_support_agree_mix_hybrid` remains the best sub-`1%`
  full-band lead
- `prototype_negative_tail_support_agree_mix_hybrid` remains the best
  high-recall branch around `1%` coverage
- `prototype_support_weighted_agree_mix_hybrid` remains the best matched-band
  branch overall on aggregate regret

## Decision

Close:

- `prototype_asymmetric_tail_support_agree_mix`
- `prototype_asymmetric_tail_support_agree_mix_hybrid`
