# Prototype Split-Scale Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch improves if the
shared and dual prototype banks can use separate temperatures for their
positive and negative members instead of a single scale per branch.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the same agreement gate and static support weighting
- split each branch into separate positive and negative temperatures
- see whether asymmetric sharpening can preserve the sparse-positive frontier
  while cleaning up matched-band coverage

## Implementation

- New head: `SplitScaleSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_split_scale_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_split_scale_support_agree_mix`
  - `prototype_split_scale_support_agree_mix_hybrid`

Relative to the live support-weighted agreement-mixture head, this adds four
learned temperature deltas:

- shared positive scale delta
- shared negative scale delta
- dual positive scale delta
- dual negative scale delta

Those deltas are exponentiated, clamped, and regularized, then applied on top
of the existing shared and dual bank temperatures before support-weighted
`logsumexp` pooling.

## Held-Out Result

### `prototype_split_scale_support_agree_mix`

Dead.

Best point:

- budget `2.0%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall mean delta regret `-0.0025`

So asymmetric temperatures alone do not rescue the plain branch.

### `prototype_split_scale_support_agree_mix_hybrid`

Closed weak positive.

At `1.5%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.51% -> 96.53%`
- overall mean delta regret `-0.0021`

At `2.0%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.51% -> 96.53%`
- overall mean delta regret `-0.0021`

Below `1.5%`, the hybrid recovered `0%` of held-out `stable_positive_v2` and
left hard near-tie unchanged.

## Comparison against current leads

`prototype_support_weighted_agree_mix_hybrid @ 1.0%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

`prototype_split_scale_support_agree_mix_hybrid @ 1.5%`

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- overall mean delta regret `-0.0021`

So splitting positive and negative temperatures gives back almost all of the
live matched-band gain.

`prototype_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0097`

`prototype_memory_agree_blend_hybrid @ 0.50%`

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- overall mean delta regret `-0.0093`

The split-scale branch does not beat either the ultra-low-coverage lead or the
micro-budget companion.

## Interpretation

This is a clear boundary on the current support-weighted agreement-mixture
family.

Current read:

- bank-internal support weighting is a real architecture gain
- but the gain does not come from simply giving the positive and negative banks
  more temperature freedom
- asymmetric scale splitting weakens the sparse-positive frontier and collapses
  matched-band quality back toward the old weak `25%` / `90.60%` pattern

So the live shortlist does not change:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_support_weighted_agree_mix_hybrid` remains the best matched-band
  branch overall

## Decision

Close:

- `prototype_split_scale_support_agree_mix`
- `prototype_split_scale_support_agree_mix_hybrid`
