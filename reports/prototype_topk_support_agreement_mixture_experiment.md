# Prototype Top-K Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch improves if
each shared and dual prototype bank only pools over the top-k
support-weighted logits instead of all prototypes.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the same support weights and agreement gate
- replace full-bank `logsumexp` pooling with capped top-k pooling per bank
- see whether explicit tail suppression sharpens the sparse-positive frontier

## Implementation

- New head: `TopKSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_topk_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_topk_support_agree_mix`
  - `prototype_topk_support_agree_mix_hybrid`
- Pooling:
  - bounded support logits are still added before pooling
  - each positive and negative bank then keeps only the top `k=4` logits
  - those top-k logits are reduced with `logsumexp`

## Held-Out Result

### `prototype_topk_support_agree_mix`

Closed weak positive.

At `1.5%` nominal budget:

- overall coverage `1.51%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0019`
- overall target match `96.51% -> 96.58%`
- overall mean delta regret `-0.0029`

At `2.0%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0029`
- overall target match `96.51% -> 96.59%`
- overall mean delta regret `-0.0030`

So hard top-k pooling does surface some signal in the plain branch, but only at
high coverage and still below every live lead.

### `prototype_topk_support_agree_mix_hybrid`

Dead.

Best point:

- budget `0.10%`
- overall coverage `0.10%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall mean delta regret `-0.0017`

The risk branch plus hard top-k pooling collapses almost entirely to a tiny
large-gap control fix.

## Comparison against current leads

`prototype_support_weighted_agree_mix_hybrid @ 1.0%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

`prototype_topk_support_agree_mix @ 2.0%`

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- overall mean delta regret `-0.0030`

So capped pooling is clearly worse than the live support-weighted
agreement-mixture branch.

`prototype_memory_agree_blend_hybrid @ 0.50%`

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- overall mean delta regret `-0.0093`

The top-k plain branch only reaches the same `50%` / `90.66%` weak band at
much higher coverage and much worse aggregate regret.

`prototype_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0097`

So top-k pooling does not challenge the ultra-low-coverage frontier either.

## Interpretation

This sets another boundary on the live support-weighted agreement-mixture
family.

Current read:

- bank-internal support weighting is a real gain
- but the gain still depends on soft full-bank pooling over the cleaned bank
- hard top-k truncation throws away useful supporting mass
- the risk-branch variant collapses almost completely under that truncation

So the shortlist does not change:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_support_weighted_agree_mix_hybrid` remains the best matched-band
  branch overall

## Decision

Close:

- `prototype_topk_support_agree_mix`
- `prototype_topk_support_agree_mix_hybrid`
