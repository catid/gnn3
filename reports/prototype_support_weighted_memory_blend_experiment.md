# Prototype Support-Weighted Memory Blend Experiment

## Question

Test whether the live memory-agreement blend is limited by **bank pollution**
rather than another missing outer gate.

The design goal was:

- keep the live memory-agreement blend geometry
- keep the same memory anchor plus one-sided agreement lift
- add bounded per-prototype support logits inside the memory, shared, and dual
  banks
- let the head suppress diffuse prototype members before branch-level blending

This is different from the earlier follow-ups because it changes retrieval
*inside* the bank instead of only reweighting whole branch scores after the
fact.

## Implementation

- New head: `SupportWeightedMemoryAgreementBlendPrototypeDeferHead`
- New runner:
  `scripts/run_prototype_support_weighted_memory_blend_defer.py`
- Variants:
  - `prototype_support_weighted_memory_blend`
  - `prototype_support_weighted_memory_blend_hybrid`

The new mechanism is simple:

- every positive and negative prototype gets a bounded support logit
- support logits are mean-centered and passed through `tanh`
- those support terms are added directly to the prototype similarity logits
  before `logsumexp` pooling
- a small support regularizer keeps the bank weighting from saturating

## Held-Out Result

### `prototype_support_weighted_memory_blend`

Dead.

Best point:

- budget `2.0%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall mean delta regret `0.0000`

So bank weighting alone does not help.

### `prototype_support_weighted_memory_blend_hybrid`

Real positive result.

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.74%`
- overall mean delta regret `-0.0130`

At `1.5%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.83%`
- overall mean delta regret `-0.0157`

At `2.0%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.86%`
- overall mean delta regret `-0.0162`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret `-0.0065`

## Comparison against current leads

`prototype_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0097`

`prototype_support_weighted_memory_blend_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- overall mean delta regret `-0.0130`

So it does **not** replace `prototype_hybrid` as the ultra-low-coverage leader.

`prototype_memory_agree_blend_hybrid @ 0.50%`

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- overall mean delta regret `-0.0093`

`prototype_support_weighted_memory_blend_hybrid @ 0.75%`

- same `50%` held-out recovery and the same weaker hard-slice band
- but stronger overall mean delta regret at `-0.0130`

So the support-weighted bank improves the same broad band, but at a larger
budget than the micro-budget memory-agreement lead.

`prototype_agree_mix_hybrid @ 1.5%`

- overall coverage `1.05%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0137`

`prototype_evidence_agree_hybrid @ 1.5%`

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0140`

`prototype_support_weighted_memory_blend_hybrid @ 1.5%`

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0157`

So it is **not** the most coverage-efficient matched-band branch, but it
becomes the best aggregate-quality matched-band branch so far.

It also edges past the remembered round-eleven reference:

`round11 margin_regime @ 2.0%`

- hard near-tie mean delta regret `-0.0089`
- overall mean delta regret `-0.0151`

`prototype_support_weighted_memory_blend_hybrid @ 1.5%`

- hard near-tie mean delta regret `-0.0089`
- overall mean delta regret `-0.0157`

## Interpretation

This is the first clear sign that **bank-internal prototype weighting** matters
inside the live prototype family.

Current read:

- the memory-agreement geometry was not exhausted
- the missing piece was not another outer gate
- bank-level support weighting can improve matched-band quality materially
- the gain shows up only when the risk branch is still present

So this is now a live architecture result:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band
  follow-up
- `prototype_support_weighted_memory_blend_hybrid` becomes the best
  aggregate-quality matched-band follow-up

That means `prototype_evidence_agree_hybrid` is now superseded as the
aggregate-quality matched-band leader.

## Decision

Close:

- `prototype_support_weighted_memory_blend`

Keep alive:

- `prototype_support_weighted_memory_blend_hybrid`

Keep the shortlist ordered as:

- `prototype_hybrid` for ultra-low coverage
- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_agree_mix_hybrid` for coverage-efficient matched-band deployment
- `prototype_support_weighted_memory_blend_hybrid` for aggregate-quality
  matched-band deployment
