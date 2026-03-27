# Prototype Negative-Cleanup Blend Support Agreement-Mixture Experiment

## Question

Test whether the two live negative-cleanup lanes can be combined with a
learned per-state blend:

- the fixed negative-tail cleanup path, which is the strongest high-recall
  branch around `1%` coverage
- the sharp negative-tail cleanup path, which is the strongest aggregate-quality
  branch around `0.75–1.0%` coverage

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep negative-bank-only cleanup
- let the model interpolate per state between fixed and sharp cleanup
- recover more of the fixed branch's held-out sparse-positive recall without
  giving back the sharp branch's aggregate regret

## Implementation

- New head: `NegativeCleanupBlendSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_negative_cleanup_blend_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_negative_cleanup_blend_support_agree_mix`
  - `prototype_negative_cleanup_blend_support_agree_mix_hybrid`

Relative to the live negative-cleanup branches:

- the model computes one score using the older fixed negative-tail cleanup
- it computes a second score using the newer sharpness-gated negative cleanup
- it learns a small per-state blend gate over those two retrieval views
- the risk branch remains optional and only appears in the `_hybrid` variant

So this is not another tail-strength tweak. It is an attempt to route between
the two already-positive cleanup geometries inside the same support-weighted
agreement-mixture family.

## Held-Out Result

### `prototype_negative_cleanup_blend_support_agree_mix`

Closed weak positive.

Best point:

- budget `2.00%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0026`
- overall target match `96.51% -> 96.57%`
- overall mean delta regret `-0.0023`

The plain branch stayed mostly inert until high coverage, then only recovered
the weaker middle behavior that many already-closed branches found earlier.

### `prototype_negative_cleanup_blend_support_agree_mix_hybrid`

Closed weak positive.

At `0.10%` nominal budget:

- overall coverage `0.10%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.51% -> 96.55%`
- overall mean delta regret `-0.0035`

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.51% -> 96.57%`
- overall mean delta regret `-0.0048`

That behavior then saturated. Budgets `0.50%` through `2.00%` all produced the
same held-out sparse-positive recovery and the same hard-slice band.

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.84%`
- large-gap mean delta regret `-0.0036`
- large-gap mean delta miss `0.0000`

## Comparison against live leads

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

`prototype_memory_agree_blend_hybrid @ 0.25%`

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- overall mean delta regret `-0.0062`

So the new cleanup-blend branch does not preserve either live negative-cleanup
lane:

- it does not keep the sharp branch's `75%` / `90.73%` full-frontier band
- it does not keep the fixed negative-tail branch's `100%` / `90.80%`
  high-recall lane
- it does not even beat the older micro-budget memory-agreement blend on the
  weaker `90.60%` niche

## Interpretation

The result is much weaker than the earlier fixed-floor and additive follow-ups,
because the hybrid collapses very quickly into a single small niche:

- `25%` held-out sparse-positive recovery
- the weaker `90.53% -> 90.60%` hard near-tie band
- modest broad overall regret gains only

That suggests the two live negative-cleanup lanes are not usefully combined by a
simple learned interpolation over the final retrieval scores. The model does not
learn to preserve the recall lane on one subset and the sharp-quality lane on
another. It just drifts back toward the old weak middle.

## Decision

Close:

- `prototype_negative_cleanup_blend_support_agree_mix`
- `prototype_negative_cleanup_blend_support_agree_mix_hybrid`

Keep the live shortlist unchanged:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best sub-`1%`
  full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out recall
  around `1%`
- `prototype_support_weighted_agree_mix_hybrid` as the higher-budget
  matched-band reference
