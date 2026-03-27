# Prototype Floor-Gated Sharp Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the live sharp-negative-tail branch improves if its sharpness gate
has a nonzero floor instead of ranging from zero to one.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the sharpness-gated negative cleanup form
- ensure cleanup never turns fully off by adding a minimum gate floor
- recover some of the fixed negative-tail branch's recall without repeating the
  additive over-suppression from the floor-plus-sharp experiment

This is the direct follow-up to the positive sharp-negative-tail result.

## Implementation

- New head: `FloorGatedSharpNegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_floor_gated_sharp_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_floor_gated_sharp_negative_tail_support_agree_mix`
  - `prototype_floor_gated_sharp_negative_tail_support_agree_mix_hybrid`

Relative to the sharp-negative-tail branch:

- the sharpness gate is unchanged in shape
- but it is lifted by a nonzero floor
- cleanup strength is therefore never fully turned off

So this is a multiplicative floor on the existing adaptive gate, not an
additive second penalty term.

## Held-Out Result

### `prototype_floor_gated_sharp_negative_tail_support_agree_mix`

Closed weak positive.

Best point:

- budget `2.00%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0029`
- overall mean delta regret `-0.0005`

So the plain branch remained too weak to matter.

### `prototype_floor_gated_sharp_negative_tail_support_agree_mix_hybrid`

Closed weak positive.

Best point:

- budget `0.25%`
- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0023`
- overall mean delta regret `-0.0027`

At higher budgets it only scales broad-safe regret a bit:

- `1.00%` overall coverage still only `25%` held-out recovery
- hard near-tie remains stuck at `90.53% -> 90.60%`
- overall mean delta regret improves to only `-0.0061`

So the floor does not recover the real sparse-positive frontier.

## Comparison against live leads

`prototype_memory_agree_blend_hybrid @ 0.25%`

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- overall mean delta regret `-0.0049`

`prototype_floor_gated_sharp_negative_tail_support_agree_mix_hybrid @ 0.25%`

- same overall coverage `0.25%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie only `90.53% -> 90.60%`
- overall mean delta regret only `-0.0027`

So even the micro-budget memory-agreement branch still beats it.

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_floor_gated_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie only `90.53% -> 90.60%`
- overall mean delta regret only `-0.0052`

So adding a floor breaks the sharp-negative-tail branch's real advantage.

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_floor_gated_sharp_negative_tail_support_agree_mix_hybrid @ 1.00%`

- same overall coverage `1.00%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie only `90.53% -> 90.60%`
- overall mean delta regret only `-0.0061`

So it does not recover the high-recall lane either.

## Interpretation

This is not a live architecture direction.

Current read:

- the sharp-negative-tail gain depends on letting the gate truly turn down
- forcing a minimum cleanup floor drifts the branch back into the weak
  `25%` / `90.60%` niche
- the additive floor-plus-sharp branch failed by over-suppressing
- this multiplicative floor-gated branch fails by blurring away the useful
  adaptive contrast

So the shortlist remains unchanged:

- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` remains the best
  sub-`1%` full-band and coverage-efficient matched-band branch
- `prototype_negative_tail_support_agree_mix_hybrid` remains the best
  high-recall branch around `1%`
- `prototype_support_weighted_agree_mix_hybrid` remains the higher-budget
  matched-band reference

## Decision

Close:

- `prototype_floor_gated_sharp_negative_tail_support_agree_mix`
- `prototype_floor_gated_sharp_negative_tail_support_agree_mix_hybrid`
