# Prototype Support-Weighted Evidence-Agreement Experiment

## Question

Test whether the older evidence-agreement family was also limited by **bank
pollution**, not just by its gate design.

The design goal was:

- keep the live evidence-agreement geometry
- keep the same evidence-aware blend gate
- add bounded per-prototype support logits inside the shared and dual banks
- let the model clean up prototype evidence before the gate reads top positive
  and negative matches

This is the direct support-weighting follow-up to the older
`prototype_evidence_agree_hybrid` branch.

## Implementation

- New head: `SupportWeightedEvidenceAgreementPrototypeDeferHead`
- New runner:
  `scripts/run_prototype_support_weighted_evidence_agreement_defer.py`
- Variants:
  - `prototype_support_weighted_evidence_agree`
  - `prototype_support_weighted_evidence_agree_hybrid`

The support mechanism matches the other successful support-weighted follow-ups:

- each positive and negative prototype gets a bounded support logit
- support logits are mean-centered and passed through `tanh`
- those support terms are added to the bank logits before top-match extraction
- a small support regularizer discourages saturation

## Held-Out Result

### `prototype_support_weighted_evidence_agree`

Dead.

Best point:

- budget `2.0%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall mean delta regret `0.0000`

So support weighting without the risk branch is still fully inert here.

### `prototype_support_weighted_evidence_agree_hybrid`

Weak positive, but closed.

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match only `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0023`
- overall target match `96.51% -> 96.61%`
- overall mean delta regret `-0.0069`

At `1.0%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.66%`
- overall mean delta regret `-0.0089`

At `1.5%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.68%`
- overall mean delta regret `-0.0097`

At `2.0%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.71%`
- overall mean delta regret `-0.0104`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret `-0.0065`

## Comparison against current leads

`prototype_evidence_agree_hybrid @ 1.5%`

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0140`

`prototype_support_weighted_evidence_agree_hybrid @ 1.5%`

- same overall coverage `1.52%`
- the same `75%` / `90.73%` hard-slice band
- materially worse overall mean delta regret at `-0.0097`

So support weighting hurts this family relative to the older evidence-agreement
baseline instead of helping it.

`prototype_support_weighted_agree_mix_hybrid @ 1.0%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

`prototype_support_weighted_evidence_agree_hybrid @ 2.0%`

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret only `-0.0104`

So the support-weighted evidence gate is clearly dominated by the live
support-weighted agreement-mixture branch.

It also loses to the original ultra-low-coverage lead:

`prototype_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0097`

`prototype_support_weighted_evidence_agree_hybrid @ 0.75%`

- same overall coverage `0.76%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie target match only `90.53% -> 90.60%`
- overall mean delta regret `-0.0069`

## Interpretation

This closes another branch cleanly.

Current read:

- support weighting is not a universal improvement for every structured
  geometry
- inside the evidence-agreement family, support weighting weakens the useful
  aggregate-quality behavior instead of sharpening it
- the evidence-aware gate seems to rely on the raw top-match structure rather
  than benefiting from the same bank cleanup that helped the score-only
  agreement and memory-agreement families

So this branch is closed:

- worse than the older `prototype_evidence_agree_hybrid`
- worse than `prototype_support_weighted_agree_mix_hybrid`
- worse than `prototype_hybrid` on the ultra-low-coverage frontier

## Decision

Close:

- `prototype_support_weighted_evidence_agree`
- `prototype_support_weighted_evidence_agree_hybrid`

Keep the shortlist unchanged:

- `prototype_hybrid` for ultra-low coverage
- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_support_weighted_agree_mix_hybrid` as the primary matched-band
  branch
