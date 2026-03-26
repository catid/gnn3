# Prototype Regime-Split Memory Blend Experiment

## Question

Test whether the live memory-anchor blend improves if the one-sided positive lift is split into two specialists:

- a headroom-style lift
- a residual baseline-error-style lift

The intended use was narrow:

- keep the current memory-anchor geometry
- keep the current one-sided agreement lift
- replace one shared outer lift gate with regime-specific lift gates
- learn a tiny regime classifier on the stable-positive-v2 source family

The goal was to beat `prototype_memory_agree_blend_hybrid` on the micro-budget frontier or at least recover a broader matched-band gain without broad regression.

## Implementation

- New head: `RegimeSplitMemoryBlendPrototypeDeferHead`
- New runner: `scripts/run_prototype_regime_split_memory_blend_defer.py`
- Variants:
  - `prototype_regime_split_memory`
  - `prototype_regime_split_memory_hybrid`
- Regime targets on train seed `314` stable-positive-v2 rows:
  - headroom: `31`
  - residual baseline-error: `11`
  - other: `0`

The hybrid variant keeps the same risk branch contract as the other memory-anchor follow-ups.

## Held-Out Result

This branch is closed.

### `prototype_regime_split_memory`

Dead on the target.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- hard near-tie target match stayed at `90.53%`
- hard near-tie mean delta regret stayed at `0.0000`

### `prototype_regime_split_memory_hybrid`

Weak positive signal, but still below the live micro-budget lead.

Best micro-budget point:

- budget `0.10%`
- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall mean delta regret `-0.0021`

Best overall-regret point:

- budget `2.0%`
- overall coverage `1.17%`
- held-out `stable_positive_v2` recovery still only `25%`
- hard near-tie target match still only `90.53% -> 90.60%`
- overall mean delta regret `-0.0038`

Large-gap control stayed clean, but the actual frontier result did not move enough to matter.

## Interpretation

The regime split did find a real held-out positive case, so the source-family decomposition is not meaningless. But it still underperformed the existing live branches:

- worse than `prototype_memory_agree_blend_hybrid` on the micro-budget frontier
- worse than `prototype_hybrid` on sparse-positive recall
- far worse than the matched-band `prototype_agree_mix_hybrid` and `prototype_evidence_agree_hybrid` follow-ups

The failure mode is clear:

- the regime classifier is learning a broad allocation pattern
- but the specialist lift heads are not producing a sharper correction frontier than the single shared memory-agreement lift

## Decision

Close `prototype_regime_split_memory` and `prototype_regime_split_memory_hybrid`.

Keep the shortlist unchanged:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1 follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band follow-up
- `prototype_evidence_agree_hybrid` remains the best aggregate-quality matched-band follow-up
