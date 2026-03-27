# Prototype Soft-Tail Blend Support Agreement-Mixture Experiment

## Question

Test whether blending the live support-weighted agreement-mixture score with the
new soft-tail score can keep the sub-`1%` frontier gain while recovering the
stronger matched-band aggregate quality of the original branch.

The design goal was:

- keep the same support-weighted agreement-mixture bank
- compute both the original full-bank score and the soft-tail score from that
  same bank
- learn a small blend gate between those two scores before the risk branch

So this is a very narrow same-bank interpolation follow-up, not a new family.

## Implementation

- New head: `SoftTailBlendSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_soft_tail_blend_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_soft_tail_blend_support_agree_mix`
  - `prototype_soft_tail_blend_support_agree_mix_hybrid`

The head:

- computes the live support-weighted agreement-mixture score
- computes the soft-tail score from the same prototypes and support weights
- learns a tiny gate over `[base, soft, |delta|, base * soft]`
- interpolates between the two scores before the optional risk branch

## Held-Out Result

### `prototype_soft_tail_blend_support_agree_mix`

Dead.

Best point:

- budget `2.0%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `-0.0016`
- overall mean delta regret `-0.0037`

So the plain blend does not help.

### `prototype_soft_tail_blend_support_agree_mix_hybrid`

Closed weak positive.

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
- overall target match `96.51% -> 96.61%`
- overall mean delta regret `-0.0066`

At `2.0%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0066`
- overall target match `96.51% -> 96.65%`
- overall mean delta regret `-0.0085`

Large-gap controls stayed safe, but the blend never recovered the full
`75%` / `90.73%` frontier band from the soft-tail branch.

## Comparison against live branches

`prototype_soft_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0104`

`prototype_soft_tail_blend_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- held-out `stable_positive_v2` recovery only `50%`
- hard near-tie only `90.53% -> 90.66%`
- overall mean delta regret only `-0.0066`

So the blend gives back the actual frontier gain.

`prototype_support_weighted_agree_mix_hybrid @ 1.0%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

The blend also does not challenge the older matched-band branch.

## Interpretation

This sets a useful boundary on the new soft-tail result.

Current read:

- soft tail suppression is helpful as a direct retrieval cleanup
- but learned interpolation between the original and soft-tail scores just
  regresses toward the weaker middle
- the gain seems to depend on committing to the soft-tail retrieval view at the
  actual frontier point, not averaging it back toward the old score

So the shortlist does not change:

- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_soft_tail_support_agree_mix_hybrid` remains the best sub-`1%`
  full-band lead
- `prototype_support_weighted_agree_mix_hybrid` remains the best matched-band
  branch overall

## Decision

Close:

- `prototype_soft_tail_blend_support_agree_mix`
- `prototype_soft_tail_blend_support_agree_mix_hybrid`
