# Prototype Support-Weighted Prototype-Memory Experiment

## Question

Test whether the original `prototype_hybrid` geometry is also limited by **bank
pollution**, not just by missing outer structure.

The design goal was:

- keep the original prototype-memory retrieval geometry
- keep the same tiny risk branch
- add bounded per-prototype support logits inside the positive and negative
  prototype banks
- let the model suppress diffuse prototype members before `logsumexp` pooling

This is the simplest possible bank-internal support-weighting follow-up: apply
the successful weighting idea directly to the original ultra-low-coverage lead
before adding any agreement or anchor structure.

## Implementation

- New head: `SupportWeightedPrototypeMemoryDeferHead`
- New runner: `scripts/run_prototype_support_weighted_memory_defer.py`
- Variants:
  - `prototype_support_weighted_memory`
  - `prototype_support_weighted_hybrid`

The mechanism matches the later support-weighted branches:

- each positive and negative prototype gets a bounded support logit
- support logits are mean-centered and passed through `tanh`
- those support terms are added to the prototype similarity logits before
  `logsumexp`
- a small support regularizer discourages saturation

## Held-Out Result

### `prototype_support_weighted_memory`

Dead.

Best point:

- budget `2.0%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match degraded slightly to `90.53% -> 90.40%`
- overall mean delta regret only `-0.0024`

So support weighting without the risk branch is still dead here.

### `prototype_support_weighted_hybrid`

Weak positive, but closed.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.51% -> 96.58%`
- overall mean delta regret `-0.0057`

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match still only `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.51% -> 96.71%`
- overall mean delta regret `-0.0116`

At `1.0%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.76%`
- overall mean delta regret `-0.0138`

At `1.5%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match still only `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.78%`
- overall mean delta regret `-0.0142`

At `2.0%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match still only `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.78%`
- overall mean delta regret `-0.0142`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret `-0.0065`

## Comparison against current leads

`prototype_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0097`

`prototype_support_weighted_hybrid @ 1.0%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match only `90.53% -> 90.66%`
- overall mean delta regret `-0.0138`

So direct support weighting improves broad regret, but it does not preserve the
real ultra-low-coverage frontier.

`prototype_memory_agree_blend_hybrid @ 0.25%`

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`

`prototype_support_weighted_hybrid @ 0.10%`

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie target match only `90.53% -> 90.60%`

So it does not replace the micro-budget memory-agreement branch either.

`prototype_support_weighted_memory_blend_hybrid @ 1.5%`

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0157`

`prototype_support_weighted_agree_mix_hybrid @ 1.0%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

`prototype_support_weighted_hybrid @ 1.5%`

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match only `90.53% -> 90.66%`
- overall mean delta regret `-0.0142`

So bank-internal support weighting is only powerful once the geometry already
contains the stronger agreement/anchor structure.

## Interpretation

This closes an important ambiguity.

Bank-internal support weighting is real, but it is **not** enough on the raw
prototype-memory geometry.

Current read:

- support weighting can sharpen retrieval inside the more structured
  memory-agreement and agreement-mixture families
- the original prototype-memory head still needs that extra structure to reach
  the full sparse-positive frontier
- direct support weighting on the raw prototype-memory bank mostly broadens
  safe non-target fixes instead of preserving the full held-out target pack

So this branch is a closed weak positive:

- better broad regret than the old `prototype_hybrid`
- clearly worse Tier-1 recovery than the live ultra-low-coverage lead
- clearly worse matched-band behavior than the newer support-weighted
  agreement-mixture and memory-agreement branches

## Decision

Close:

- `prototype_support_weighted_memory`
- `prototype_support_weighted_hybrid`

Keep the shortlist unchanged:

- `prototype_hybrid` for ultra-low coverage
- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_support_weighted_agree_mix_hybrid` as the primary matched-band
  branch
