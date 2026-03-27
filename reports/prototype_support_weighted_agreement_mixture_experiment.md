# Prototype Support-Weighted Agreement-Mixture Experiment

## Question

Test whether the live agreement-mixture branch is also limited by **bank
pollution**, not just by branch-level gating.

The design goal was:

- keep the live shared/dual agreement-mixture geometry
- keep the same agreement gate
- add bounded per-prototype support logits inside the shared and dual banks
- let the model suppress diffuse prototype members before `logsumexp` pooling

This is the direct follow-up to the support-weighted memory blend result, but
applied to the more coverage-efficient agreement-mixture branch.

## Implementation

- New head: `SupportWeightedAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_support_weighted_agreement_mixture_defer.py`
- Variants:
  - `prototype_support_weighted_agree_mix`
  - `prototype_support_weighted_agree_mix_hybrid`

The support mechanism is the same as the memory-blend follow-up:

- each positive and negative prototype gets a bounded support logit
- support logits are mean-centered and passed through `tanh`
- those support terms are added to the prototype similarity logits before
  `logsumexp`
- a small support regularizer discourages saturation

## Held-Out Result

### `prototype_support_weighted_agree_mix`

Dead.

Best point:

- budget `2.0%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall mean delta regret `0.0000`

So support weighting without the risk branch does not help here either.

### `prototype_support_weighted_agree_mix_hybrid`

Real positive result.

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.71%`
- overall mean delta regret `-0.0114`

At `1.0%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.80%`
- overall mean delta regret `-0.0148`

At `1.5%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.84%`
- overall mean delta regret `-0.0158`

At `2.0%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.88%`
- overall mean delta regret `-0.0165`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret `-0.0065`

## Comparison against current leads

`prototype_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0097`

`prototype_support_weighted_agree_mix_hybrid @ 1.0%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

So it does not replace `prototype_hybrid` as the ultra-low-coverage leader, but
it reaches the same real frontier band at only slightly higher coverage and with
materially better overall regret.

`prototype_memory_agree_blend_hybrid @ 0.50%`

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- overall mean delta regret `-0.0093`

`prototype_support_weighted_agree_mix_hybrid @ 0.50%`

- same overall coverage `0.51%`
- the same `50%` held-out stable-positive-v2 recovery
- the same `90.53% -> 90.66%` weaker band
- stronger overall mean delta regret at `-0.0114`

So it beats the micro-budget memory-agreement companion at the same `0.5%`
point, although that memory branch still remains the cleaner sub-`0.5%`
specialist.

`prototype_agree_mix_hybrid @ 1.5%`

- overall coverage `1.05%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0137`

`prototype_support_weighted_agree_mix_hybrid @ 1.0%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

So support weighting now dominates the old coverage-efficient agreement-mixture
lead even before the `1.5%` point.

`prototype_support_weighted_memory_blend_hybrid @ 1.5%`

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0157`

`prototype_support_weighted_agree_mix_hybrid @ 1.5%`

- same overall coverage `1.52%`
- the same `75%` / `90.73%` hard-slice band
- slightly better overall mean delta regret at `-0.0158`

So it also edges past the support-weighted memory branch on aggregate quality.

It also clears the remembered round-eleven reference:

`round11 margin_regime @ 2.0%`

- hard near-tie mean delta regret `-0.0089`
- overall mean delta regret `-0.0151`

`prototype_support_weighted_agree_mix_hybrid @ 1.5%`

- hard near-tie mean delta regret `-0.0089`
- overall mean delta regret `-0.0158`

## Interpretation

This is the strongest matched-band architecture result in the prototype family
so far.

Current read:

- bank-internal support weighting is not specific to the memory-anchor geometry
- the useful gain survives in the more coverage-efficient agreement-mixture
  family
- support weighting plus the tiny risk branch is now the best matched-band
  combination we have

So the shortlist changes:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best sub-`0.5%`
  micro-budget Tier-1 follow-up
- `prototype_support_weighted_agree_mix_hybrid` becomes the best matched-band
  branch overall

That means it supersedes:

- `prototype_agree_mix_hybrid` as the coverage-efficient matched-band leader
- `prototype_support_weighted_memory_blend_hybrid` as the aggregate-quality
  matched-band leader
- `prototype_evidence_agree_hybrid` as the older aggregate-quality
  matched-band reference

## Decision

Close:

- `prototype_support_weighted_agree_mix`

Keep alive:

- `prototype_support_weighted_agree_mix_hybrid`
