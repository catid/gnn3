# Prototype Sharp-Mass Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch improves if
adaptive negative-tail cleanup is gated by **either** local top-1 vs top-2
sharpness **or** broader total negative tail mass behind the lead prototype,
instead of either signal alone.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep negative-bank-only cleanup
- preserve the sharp branch's sub-`1%` quality
- recover some of the fixed branch's higher recall
- do that by changing the bank-internal gate rather than blending final scores

This is the direct follow-up to the single-signal sharp-negative and
mass-negative cleanup branches.

## Implementation

- New head:
  `SharpMassNegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_sharp_mass_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_sharp_mass_negative_tail_support_agree_mix`
  - `prototype_sharp_mass_negative_tail_support_agree_mix_hybrid`

Relative to the older sharp-negative branch:

- shared and dual negative banks still receive adaptive soft tail suppression
- one gate still tracks local negative-bank sharpness via the top-1 vs top-2 gap
- a second gate tracks broader negative-bank diffuseness via lead-vs-tail-mass
  gap
- the final cleanup gate is a smooth OR over those two signals

So this is still a retrieval-side cleanup change, but it now reacts to either a
strong runner-up or a strong broader tail.

## Held-Out Result

### `prototype_sharp_mass_negative_tail_support_agree_mix`

Closed, effectively dead.

Best point:

- budget `0.10–2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`
- overall mean delta regret `0.0000`

### `prototype_sharp_mass_negative_tail_support_agree_mix_hybrid`

Closed positive, but not a new live lead.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.51% -> 96.61%`
- overall mean delta regret `-0.0064`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.74%`
- overall mean delta regret `-0.0130`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.79%`
- overall mean delta regret `-0.0145`

At `2.00%` nominal budget:

- overall coverage `1.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.87%`
- overall mean delta regret `-0.0164`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret improved
- no harmful large-gap miss pattern appeared

## Comparison against live leads

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_sharp_mass_negative_tail_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- only `50%` held-out `stable_positive_v2` recovery
- only `90.53% -> 90.66%`
- overall mean delta regret `-0.0130`

So the combined gate clearly loses to the live sharp-negative branch below
`1%`.

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_sharp_mass_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery only `50%`
- hard near-tie target match only `90.53% -> 90.66%`
- overall mean delta regret `-0.0145`

So the combined gate also fails to preserve the fixed branch's high-recall
lane.

`prototype_support_weighted_agree_mix_hybrid @ 2.00%`

- overall coverage about `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0165`

`prototype_sharp_mass_negative_tail_support_agree_mix_hybrid @ 2.00%`

- overall coverage `1.76%`
- the same `75%` / `90.73%` frontier band
- overall mean delta regret `-0.0164`

So the new branch does eventually recover the higher-budget matched band, but
it still slightly trails the existing support-weighted agreement-mixture
reference on aggregate quality while getting there later than the sharp branch.

## Interpretation

This is a real positive, but not a new winner.

Current read:

- combining sharpness and tail-mass signals is better than using tail mass
  alone
- but the combined gate still softens cleanup too much at low coverage
- it only recovers the full `75%` / `90.73%` frontier once coverage grows
  toward `2%`
- by that point it is still not clearly better than the live higher-budget
  reference

So this is another useful constraint:

- bank-internal gate design matters
- but smooth OR-combining the sharp and mass signals is still not the way to
  bridge the sharp-quality and fixed-recall lanes

## Decision

Close:

- `prototype_sharp_mass_negative_tail_support_agree_mix`
- `prototype_sharp_mass_negative_tail_support_agree_mix_hybrid`

Keep the live shortlist unchanged:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best
  sub-`1%` full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out
  recall around `1%`
- `prototype_support_weighted_agree_mix_hybrid` as the higher-budget
  matched-band reference
