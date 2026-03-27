# Prototype Mass-Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch improves if
sharp negative-tail cleanup is driven by the **full negative tail mass** behind
the lead prototype instead of the top-1 vs top-2 gap alone.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep negative-bank-only cleanup
- keep the cleanup adaptive rather than fixed
- react to broader diffuse negative overlap instead of a single runner-up
- preserve more of the live negative-cleanup frontier than the top2-preserving
  variant without giving back aggregate quality

This is a bank-internal retrieval follow-up rather than another outer-score
blend.

## Implementation

- New head:
  `MassAwareSharpNegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_mass_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_mass_negative_tail_support_agree_mix`
  - `prototype_mass_negative_tail_support_agree_mix_hybrid`

Relative to the older sharp-negative branch:

- shared and dual negative banks still receive soft tail suppression
- the suppression strength is still gated per state
- but the gate now uses lead-vs-tail-mass gap
- tail mass is computed as the `logsumexp` of every non-leading negative logit

So this is still adaptive negative cleanup, but it now responds to the whole
negative tail rather than only the strongest runner-up match.

## Held-Out Result

### `prototype_mass_negative_tail_support_agree_mix`

Closed weak positive, effectively dead.

Best point:

- budget `2.00%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`
- overall mean delta regret only `0.0027`

### `prototype_mass_negative_tail_support_agree_mix_hybrid`

Closed positive, but not a live lead.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.51% -> 96.61%`
- overall mean delta regret `-0.0049`

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.74%`
- overall mean delta regret `-0.0109`

At `1.00%` nominal budget:

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.79%`
- overall mean delta regret `-0.0132`

At `2.00%` nominal budget:

- overall coverage `1.77%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.84%`
- overall mean delta regret `-0.0157`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret improved
- no harmful large-gap miss pattern appeared

## Comparison against live leads

`prototype_memory_agree_blend_hybrid @ 0.25%`

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`

`prototype_mass_negative_tail_support_agree_mix_hybrid @ 0.50%`

- overall coverage `0.51%`
- the same `50%` held-out recovery
- the same `90.53% -> 90.66%` hard near-tie band
- better aggregate regret, but at roughly double the coverage

So the new branch recreates the old middle lane, but it does not replace the
memory-agreement blend as the more coverage-efficient micro-budget option.

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

`prototype_mass_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery only `50%`
- hard near-tie target match only `90.53% -> 90.66%`
- overall mean delta regret `-0.0132`

So the new mass-aware gate does not preserve either live negative-cleanup
frontier:

- it does not preserve the sharp branch's `75%` / `90.73%` matched-band lane
- it does not preserve the fixed branch's `100%` / `90.80%` recall lane

## Interpretation

This is a real bank-level signal, but not a new lead.

Current read:

- using full negative tail mass is better than preserving the runner-up
  negative match directly
- it cleanly recovers the older `50%` / `90.66%` middle lane inside the
  stronger support-weighted agreement-mixture family
- but it still weakens cleanup too much to preserve either live
  negative-cleanup frontier

So the tail-mass gate is a useful constraint on the design space, not a new
architecture winner.

## Decision

Close:

- `prototype_mass_negative_tail_support_agree_mix`
- `prototype_mass_negative_tail_support_agree_mix_hybrid`

Keep the live shortlist unchanged:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best
  sub-`1%` full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out
  recall around `1%`
- `prototype_support_weighted_agree_mix_hybrid` as the higher-budget
  matched-band reference
