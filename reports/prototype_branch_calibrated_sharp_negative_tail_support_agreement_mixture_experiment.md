# Prototype Branch-Calibrated Sharp Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch improves if the
shared and dual negative banks use **separate learned sharpness-gate centers and
slopes** instead of one global sharpness schedule.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep negative-bank-only cleanup
- preserve the sharp-negative branch's strong sub-`1%` quality
- recover more recall at `1–2%` coverage by letting shared and dual banks
  calibrate independently

This is a bank-internal gate-calibration follow-up, not another final-score
blend.

## Implementation

- New head:
  `BranchCalibratedSharpNegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_branch_calibrated_sharp_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_branch_calibrated_sharp_negative_tail_support_agree_mix`
  - `prototype_branch_calibrated_sharp_negative_tail_support_agree_mix_hybrid`

Relative to the older sharp-negative branch:

- negative cleanup still uses the top-1 vs top-2 sharpness gate
- but the shared bank now has its own learned center and slope
- and the dual bank also has its own learned center and slope

So the core idea is to learn whether the two geometries want different
aggressiveness schedules without changing the final agreement-mixture logic.

## Held-Out Result

### `prototype_branch_calibrated_sharp_negative_tail_support_agree_mix`

Closed, effectively dead.

Best point:

- budget `2.00%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`
- overall mean delta regret only `0.0009`

### `prototype_branch_calibrated_sharp_negative_tail_support_agree_mix_hybrid`

Closed positive, but not a new live lead.

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.51% -> 96.70%`
- overall mean delta regret `-0.0097`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.79%`
- overall mean delta regret `-0.0145`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.82%`
- overall mean delta regret `-0.0154`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.86%`
- overall mean delta regret `-0.0162`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret improved
- no harmful large-gap miss pattern appeared

## Comparison against live leads

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_branch_calibrated_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- only `25%` held-out `stable_positive_v2` recovery
- only `90.53% -> 90.60%`
- overall mean delta regret `-0.0114`

So the branch-calibrated gate clearly loses to the live sharp-negative branch
below `1%`.

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_branch_calibrated_sharp_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery only `50%`
- hard near-tie target match only `90.53% -> 90.66%`
- overall mean delta regret `-0.0145`

So the new branch also does not preserve the fixed negative-tail branch's
high-recall lane.

`prototype_support_weighted_agree_mix_hybrid @ 1.50%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0158`

`prototype_branch_calibrated_sharp_negative_tail_support_agree_mix_hybrid @ 1.50%`

- the same `75%` / `90.73%` frontier band
- overall mean delta regret `-0.0154`

At `2.00%`, the branch-calibrated variant still trails the higher-budget live
reference slightly:

- `prototype_support_weighted_agree_mix_hybrid`: `-0.0165`
- `prototype_branch_calibrated_sharp_negative_tail_support_agree_mix_hybrid`:
  `-0.0162`

## Interpretation

This is a real positive, but still a dominated one.

Current read:

- shared and dual banks do benefit from more flexible calibration than the pure
  sharp-negative baseline
- but that gain shows up too late
- the branch-calibrated gate only reaches the full `75%` / `90.73%` frontier
  by `1.5%`
- by that point it is still slightly weaker than the existing higher-budget
  support-weighted agreement-mixture reference

So this does not change the live deployment shortlist.

## Decision

Close:

- `prototype_branch_calibrated_sharp_negative_tail_support_agree_mix`
- `prototype_branch_calibrated_sharp_negative_tail_support_agree_mix_hybrid`

Keep the live shortlist unchanged:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best
  sub-`1%` full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out
  recall around `1%`
- `prototype_support_weighted_agree_mix_hybrid` as the higher-budget
  matched-band reference
