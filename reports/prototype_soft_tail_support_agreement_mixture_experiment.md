# Prototype Soft-Tail Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch improves if
low-ranked prototype logits are **softly** suppressed relative to the per-bank
winner instead of being hard-truncated by top-k pooling.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep full-bank pooling and the existing support weights
- softly penalize only logits that fall behind the bank winner beyond a small
  margin
- preserve useful tail mass while reducing diffuse bank pollution

This is the direct follow-up to the dead hard top-k truncation branch.

## Implementation

- New head: `SoftTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_soft_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_soft_tail_support_agree_mix`
  - `prototype_soft_tail_support_agree_mix_hybrid`

The new pooling change is:

- keep the support-weighted similarity logits for each bank
- compute the bank winner per row
- subtract a bounded penalty from logits that trail the winner by more than a
  small margin
- keep full-bank `logsumexp` after that soft suppression

So this is still a retrieval-side cleanup, not a new outer gate.

## Held-Out Result

### `prototype_soft_tail_support_agree_mix`

Dead.

Best point:

- budget `2.0%`
- overall coverage `0.07%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall mean delta regret `0.0000`

So soft tail suppression without the risk branch is inert.

### `prototype_soft_tail_support_agree_mix_hybrid`

Real positive result.

At `0.10%` nominal budget:

- overall coverage `0.10%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0023`
- overall target match `96.51% -> 96.59%`
- overall mean delta regret `-0.0060`

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.67%`
- overall mean delta regret `-0.0094`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.69%`
- overall mean delta regret `-0.0104`

At `1.0%` nominal budget:

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.69%`
- overall mean delta regret `-0.0104`

At `2.0%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.76%`
- overall mean delta regret `-0.0129`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret `-0.0065`

## Comparison against current leads

`prototype_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall target match `96.51% -> 96.68%`
- overall mean delta regret `-0.0097`

`prototype_soft_tail_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- the same `75%` held-out stable-positive-v2 recovery
- the same `90.53% -> 90.73%` hard near-tie band
- slightly better overall target match at `96.51% -> 96.69%`
- slightly better overall mean delta regret at `-0.0104`

So this is the first follow-up that actually edges past `prototype_hybrid` at
the same ultra-low-coverage frontier point.

`prototype_memory_agree_blend_hybrid @ 0.25%`

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- overall mean delta regret `-0.0049`

`prototype_soft_tail_support_agree_mix_hybrid @ 0.10%`

- overall coverage `0.10%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- overall mean delta regret `-0.0060`

So the memory-agreement blend remains the better pure micro-budget companion
below about `0.25–0.50%`.

`prototype_support_weighted_agree_mix_hybrid @ 1.0%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

`prototype_soft_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0104`

So the older support-weighted agreement-mixture branch still wins on aggregate
matched-band quality once coverage reaches `~1%`, but the new soft-tail branch
reaches the full frontier band materially earlier.

## Interpretation

This is a real architecture improvement.

Current read:

- the live agreement-mixture gain still comes from bank cleanup
- hard truncation was too aggressive
- but a soft tail penalty preserves enough supporting mass to keep the frontier
  intact while moving the full-band point below `1%` overall coverage

So the shortlist changes again:

- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up below about `0.25–0.50%`
- `prototype_soft_tail_support_agree_mix_hybrid` becomes the best sub-`1%`
  full-band architecture lead
- `prototype_support_weighted_agree_mix_hybrid` remains the best matched-band
  branch overall once coverage can rise to about `1%` or more

## Decision

Close:

- `prototype_soft_tail_support_agree_mix`

Keep alive:

- `prototype_soft_tail_support_agree_mix_hybrid`
