# Prototype Top2 Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch improves if
negative-tail cleanup preserves the strongest two negative matches and only
suppresses the broader tail beneath them.

The design goal was:

- keep the current support-weighted agreement-mixture geometry
- keep negative-bank-only cleanup
- avoid suppressing the runner-up negative evidence
- only remove diffuse lower-ranked negative clutter
- preserve more recall than the sharp-negative branch without giving back too
  much aggregate quality

This is a bank-level follow-up rather than another final-score composition of
the fixed and sharp cleanup branches.

## Implementation

- New head: `Top2NegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_top2_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_top2_negative_tail_support_agree_mix`
  - `prototype_top2_negative_tail_support_agree_mix_hybrid`

Relative to the older fixed negative-tail branch:

- negative-bank cleanup still uses a soft tail penalty
- but the penalty is now referenced to the runner-up negative logit
- that means the top-2 negative matches are preserved
- only lower-ranked negative-bank members get suppressed

So this is a softer tail-cleanup rule that tries to keep the strongest
competing negative evidence intact.

## Held-Out Result

### `prototype_top2_negative_tail_support_agree_mix`

Closed weak positive, effectively dead.

Best point:

- budget `1.00–2.00%`
- overall coverage `1.00–2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`
- overall mean delta regret only `-0.0005`

### `prototype_top2_negative_tail_support_agree_mix_hybrid`

Closed weak positive.

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0023`
- overall target match `96.51% -> 96.58%`
- overall mean delta regret `-0.0031`

At `1.00%` nominal budget:

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0023`
- overall target match `96.51% -> 96.61%`
- overall mean delta regret `-0.0049`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0042`
- overall target match `96.51% -> 96.66%`
- overall mean delta regret `-0.0065`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.84%`
- large-gap mean delta regret `-0.0036`
- large-gap mean delta miss `0.0000`

## Comparison against live leads

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_support_weighted_agree_mix_hybrid @ 1.00%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

So the top2-negative variant is clearly weaker than every live lane:

- it never reaches the `75%` / `90.73%` frontier band below `1%`
- it never reaches the fixed negative-tail branch's `100%` / `90.80%` recall
  lane
- even at `2.0%` coverage it only reaches `50%` / `90.66%`

## Interpretation

Preserving the runner-up negative match weakens cleanup too much. The live
negative-tail gains appear to require broader suppression of diffuse negative
overlap, not just removal of logits below the top-2 set.

This is another useful constraint:

- the bank-level location is reasonable
- but the cleanup cannot be this conservative

## Decision

Close:

- `prototype_top2_negative_tail_support_agree_mix`
- `prototype_top2_negative_tail_support_agree_mix_hybrid`

Keep the live shortlist unchanged:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best sub-`1%`
  full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out recall
  around `1%`
- `prototype_support_weighted_agree_mix_hybrid` as the higher-budget
  matched-band reference
