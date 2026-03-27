# Prototype Branch-Strength Sharp Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the live sharp-negative-tail support-weighted agreement-mixture
branch improves if the shared and dual negative banks keep the same
sharpness-triggered cleanup gate but learn **separate cleanup amplitudes**
instead of sharing one global tail-strength ceiling.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep negative-bank-only cleanup
- keep the successful sharpness gate unchanged
- let the shared and dual negative banks use different cleanup strength ranges
- preserve the live sharp-negative branch below `1%` coverage while improving
  recall or aggregate matched-band regret above that point

This is the cleanest test of whether the remaining useful flexibility is in the
gate shape or in the cleanup amplitude.

## Implementation

- New head:
  `BranchStrengthSharpNegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_branch_strength_sharp_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_branch_strength_sharp_negative_tail_support_agree_mix`
  - `prototype_branch_strength_sharp_negative_tail_support_agree_mix_hybrid`

Relative to the live sharp-negative branch:

- the sharpness gate is unchanged
- the shared negative bank gets its own learned cleanup scale
- the dual negative bank gets its own learned cleanup scale
- cleanup strength is still bounded and regularized

So this is a very narrow follow-up: separate branch cleanup amplitude, same
successful retrieval geometry and same successful gate shape.

## Held-Out Result

### `prototype_branch_strength_sharp_negative_tail_support_agree_mix`

Closed, effectively dead.

Best point:

- budget `0.10%`
- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall target match regressed slightly to `96.73% -> 96.70%`
- overall mean delta regret `+0.0017`

So the plain branch is not just inert; it is slightly harmful.

### `prototype_branch_strength_sharp_negative_tail_support_agree_mix_hybrid`

Closed positive, but not a new live lead.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.58%`
- overall mean delta regret `-0.0058`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.76%`
- overall mean delta regret `-0.0138`

At `1.00%` nominal budget:

- overall coverage `0.92%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.80%`
- overall mean delta regret `-0.0149`

At `1.50%` nominal budget:

- overall coverage `1.17%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.82%`
- overall mean delta regret `-0.0154`

At `2.00%` nominal budget:

- overall coverage `1.19%`
- same held-out result as `1.50%`
- same overall mean delta regret `-0.0154`

Large-gap controls stayed clean throughout:

- large-gap target match stayed at `99.90%`
- mean delta regret remained non-positive
- no harmful large-gap miss pattern appeared

## Comparison against live leads

### Versus the live sharp-negative branch below `1%`

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_branch_strength_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- only `50%` held-out `stable_positive_v2` recovery
- only `90.53% -> 90.66%`
- overall mean delta regret `-0.0138`

So separate branch cleanup amplitudes still lose clearly to the live
sharp-negative branch below `1%`.

### Versus the high-recall fixed negative-tail branch around `1%`

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_branch_strength_sharp_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `0.92%`
- held-out `stable_positive_v2` recovery only `50%`
- hard near-tie target match only `90.53% -> 90.66%`
- overall mean delta regret `-0.0149`

So this branch also does not preserve the fixed negative-tail branch's
high-recall lane.

### Versus recent post-sharp branch-specific follow-ups

`prototype_branch_calibrated_sharp_negative_tail_support_agree_mix_hybrid`

- only reached the full `75%` / `90.73%` frontier by about `1.52%` coverage

`prototype_learned_gate_negative_tail_support_agree_mix_hybrid`

- only reached the full `75%` / `90.73%` frontier by about `1.46%` coverage

`prototype_branch_strength_sharp_negative_tail_support_agree_mix_hybrid`

- reaches the same full `75%` / `90.73%` frontier by about `1.17%` coverage

So separate branch cleanup amplitude is a more useful degree of freedom than
separate branch gate calibration or learned summary gating.

### Versus the older matched-band reference

`prototype_support_weighted_agree_mix_hybrid @ 1.00%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

`prototype_branch_strength_sharp_negative_tail_support_agree_mix_hybrid @ 1.50%`

- overall coverage `1.17%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0154`

This is a real bridge between the low-coverage sharp-negative lane and the
older higher-budget matched-band reference, but it does not produce a clean new
deployment lane of its own.

## Interpretation

This is the strongest result among the recent post-sharp branch-specific
follow-ups, but it is still dominated.

Current read:

- separating branch cleanup amplitude is real signal
- it is more useful than changing branch gate shape
- it helps the branch reach the full `75%` / `90.73%` frontier earlier than the
  older branch-calibrated and learned-gate follow-ups
- but it still loses to the live sharp-negative branch below `1%`
- and it still does not clearly replace the older support-weighted
  agreement-mixture reference once coverage can rise into the higher-budget
  matched band

So branch-specific cleanup amplitude is a valid explanatory direction, but not a
new live architecture lane.

## Decision

Close:

- `prototype_branch_strength_sharp_negative_tail_support_agree_mix`
- `prototype_branch_strength_sharp_negative_tail_support_agree_mix_hybrid`

Keep the live shortlist unchanged:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best
  sub-`1%` full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out
  recall around `1%`
- `prototype_support_weighted_agree_mix_hybrid` as the higher-budget
  matched-band reference
