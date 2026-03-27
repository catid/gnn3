# Prototype Branchwise-Lift Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the successful branchwise-max fusion should be softened into
separate **positive-only fixed-cleanup lifts inside the shared and dual
branches** before the outer agreement mixture.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the stronger branch-strength sharp negative-tail cleanup as the base
- keep the older fixed negative-tail cleanup as a secondary source
- replace hard branchwise `max(fixed, sharp)` with learned branchwise positive
  lifts from fixed into sharp
- preserve the sharp branch's lower-coverage quality while borrowing fixed-path
  recall more selectively

This is the direct follow-up to the positive branchwise-max result.

## Implementation

- New head:
  `BranchwiseLiftNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_branchwise_lift_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_branchwise_lift_negative_cleanup_support_agree_mix`
  - `prototype_branchwise_lift_negative_cleanup_support_agree_mix_hybrid`

Relative to branchwise max:

- the same fixed and branch-strength sharp branch scores are computed
- but each branch learns a separate gate for a **positive-only**
  `relu(fixed - sharp)` lift
- those branchwise lifted scores are then mixed by the usual agreement gate

So the experiment isolates one question: is the right fusion still branch-local,
but softer than a hard max?

## Held-Out Result

### `prototype_branchwise_lift_negative_cleanup_support_agree_mix`

Closed, dead-to-harmful.

At every budget:

- held-out `stable_positive_v2` recovery stayed at `0%`
- the branch was inert through `0.50%` coverage
- beyond `0.75%` it started to **degrade** hard near-tie target match:
  - `90.53% -> 90.46%` at `0.75%`
  - `90.53% -> 90.40%` at `1.00–2.00%`
- overall mean delta regret only reached `-0.0031` at `2.00%`

So the plain branch is not just dead. Once it becomes active, it is slightly
miscalibrated on the real target slice.

### `prototype_branchwise_lift_negative_cleanup_support_agree_mix_hybrid`

Closed positive, but not a live lead.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall mean delta regret `-0.0050`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall mean delta regret `-0.0127`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall mean delta regret `-0.0149`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall mean delta regret `-0.0150`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall mean delta regret `-0.0144`

So the hybrid does recover the full matched-band frontier, but only once
coverage rises to about `1.5%`, and it never reaches the fixed negative-tail
branch's `100%` held-out recall lane.

## Comparison against nearby variants

### Versus the older global lift

`prototype_branch_strength_negative_cleanup_lift_support_agree_mix_hybrid`
showed that fixed-cleanup lift was the wrong fusion mechanism when applied only
after the final mixed score:

- through `1.0%` coverage it only recovered `25%` of held-out
  `stable_positive_v2`
- and it stayed capped at the weaker `90.53% -> 90.60%` hard near-tie band

`prototype_branchwise_lift_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- improves that to `50%` held-out `stable_positive_v2`
- improves hard near-tie to `90.53% -> 90.66%`
- and reaches overall mean delta regret `-0.0149`

So localizing the lift to the shared and dual branches clearly helps.

### Versus the live sharp-negative branch below `1%`

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_branchwise_lift_negative_cleanup_support_agree_mix_hybrid @ 0.75%`

- same `0.76%` coverage
- only `50%` held-out `stable_positive_v2`
- only `90.53% -> 90.66%`
- overall mean delta regret `-0.0127`

So this does not preserve the live low-coverage sharp-negative lane.

### Versus branchwise max

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0145`

`prototype_branchwise_lift_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `50%`
- hard near-tie `90.53% -> 90.66%`
- overall mean delta regret `-0.0149`

So branchwise lift buys a little broad-safe regret, but it gives back too much
of the actual target frontier.

At `1.50%` nominal budget:

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- keeps `75%` / `90.73%`
- overall mean delta regret `-0.0159`

`prototype_branchwise_lift_negative_cleanup_support_agree_mix_hybrid`

- keeps the same `75%` / `90.73%`
- but only reaches overall mean delta regret `-0.0150`

So once coverage rises enough for both to hit the same target band, hard
branchwise max is still clearly better.

### Versus the older higher-budget reference

`prototype_support_weighted_agree_mix_hybrid @ 1.50%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0158`

`prototype_branchwise_lift_negative_cleanup_support_agree_mix_hybrid @ 1.50%`

- same `75%` / `90.73%`
- weaker overall mean delta regret `-0.0150`

So this does not even beat the older higher-budget matched-band reference,
let alone the newer branchwise-max branch.

## Interpretation

The useful part of the branchwise-max result is not just “move the fusion inside
the shared and dual branches.” It is also that the fusion must stay **hard**
enough to preserve the fixed branch's recall signal.

Current read:

- localizing the lift to branches is better than applying one lift only after
  the final mixed score
- but a learned positive-only lift is still too soft
- it recreates the familiar weak middle:
  - better broad-safe aggregate regret than the older failed lift
  - but weaker held-out sparse-positive recovery and weaker hard-slice quality
    than the live sharp-negative and branchwise-max lanes

So the right conclusion is:

- branchwise fusion was the correct structural insight
- but in this family, hard branchwise max still works better than branchwise
  learned lift

## Decision

Close:

- `prototype_branchwise_lift_negative_cleanup_support_agree_mix`
- `prototype_branchwise_lift_negative_cleanup_support_agree_mix_hybrid`

Live shortlist remains:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best
  sub-`1%` full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out
  recall around `1%`
- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` for the
  higher-budget matched-band and higher-budget max-recall lane
