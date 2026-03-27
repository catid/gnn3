# Prototype Sharp Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the successful negative-tail branch improves if negative-bank soft
tail suppression is applied **selectively** based on how diffuse the negative
bank currently is.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep negative-bank-only tail cleanup
- make that cleanup stronger when the negative bank is diffuse
- make it weaker when the negative bank is already sharp
- preserve the recovered sparse-positive frontier while improving aggregate
  regret below about `1%` coverage

This is the direct follow-up to the positive but more brute-force
negative-tail-only branch.

## Implementation

- New head: `SharpNegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_sharp_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_sharp_negative_tail_support_agree_mix`
  - `prototype_sharp_negative_tail_support_agree_mix_hybrid`

Relative to the older negative-tail branch:

- shared and dual negative banks still receive soft tail suppression
- the suppression strength is now multiplied by a sharpness gate
- that gate is high when the negative bank top-1 vs top-2 gap is small
- that gate is low when the negative bank is already sharp

So this is still a retrieval-side cleanup, but it is now conditioned on
internal bank geometry rather than using one fixed penalty everywhere.

## Held-Out Result

### `prototype_sharp_negative_tail_support_agree_mix`

Closed weak positive.

Best point:

- budget `0.75%`
- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0029`
- overall mean delta regret `-0.0029`

So the plain branch did surface some retrieval signal, but it is still far too
weak to matter.

### `prototype_sharp_negative_tail_support_agree_mix_hybrid`

Real positive result.

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.76%`
- overall mean delta regret `-0.0133`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.81%`
- overall mean delta regret `-0.0144`

At `1.00%` nominal budget:

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.82%`
- overall mean delta regret `-0.0152`

At `1.50%` nominal budget:

- overall coverage `1.51%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.83%`
- overall mean delta regret `-0.0153`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.83%`
- overall mean delta regret `-0.0153`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret `-0.0065`
- large-gap mean delta miss `0.0000`

## Comparison against live leads

`prototype_soft_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0104`

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- the same `75%` held-out stable-positive-v2 recovery
- the same `90.53% -> 90.73%` hard near-tie band
- materially better overall mean delta regret at `-0.0144`

So this cleanly supersedes the older soft-tail branch as the best sub-`1%`
full-band contender.

`prototype_support_weighted_agree_mix_hybrid @ 1.00%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- the same `75%` held-out stable-positive-v2 recovery
- the same `90.53% -> 90.73%` hard near-tie band
- slightly better overall mean delta regret at `-0.0152`

So this is now the strongest coverage-efficient matched-band branch around the
`0.75–1.0%` regime.

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0152`

So the older negative-tail branch still remains the best **high-recall**
contender around `1%`, but the new sharpness-gated branch is better on
aggregate matched-band quality.

## Interpretation

This is a real architecture improvement.

Current read:

- bank-internal negative cleanup is definitely a real direction
- using internal bank sharpness is better than applying the same cleanup
  strength everywhere
- the extra adaptivity keeps the full frontier band at low coverage while
  materially improving aggregate regret
- but it does not replace the pure negative-tail branch when the specific goal
  is maximum held-out sparse-positive recall

So the shortlist updates to:

- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` becomes the best
  sub-`1%` full-band contender and the strongest coverage-efficient
  matched-band branch around `0.75–1.0%`
- `prototype_negative_tail_support_agree_mix_hybrid` remains the best
  high-recall branch around `1%` coverage
- `prototype_support_weighted_agree_mix_hybrid` remains the best higher-budget
  matched-band branch once coverage can rise toward `1.5–2.0%`

## Decision

Close:

- `prototype_sharp_negative_tail_support_agree_mix`

Keep alive:

- `prototype_sharp_negative_tail_support_agree_mix_hybrid`
