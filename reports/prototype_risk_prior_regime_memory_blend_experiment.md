# Prototype Risk-Prior Regime Memory Blend Experiment

## Question

Test whether the weak regime-split memory result can be improved by anchoring the specialist mixture with explicit risk priors.

The design goal was:

- keep the live memory-anchor geometry
- keep the one-sided agreement lift
- keep separate headroom and residual specialist lift gates
- add an explicit risk-prior branch into the regime logits so the split does not have to rediscover headroom / residual cues from the prototype features alone

## Implementation

- New head: `RiskPriorRegimeMemoryBlendPrototypeDeferHead`
- New runner: `scripts/run_prototype_risk_prior_regime_memory_blend_defer.py`
- Variants:
  - `prototype_risk_prior_regime_memory`
  - `prototype_risk_prior_regime_memory_hybrid`

The risk-prior branch reads the same normalized risk features already used elsewhere:

- margin-only features
- margin/regime features
- continuation-gap headroom

These risk priors are added directly into the regime logits before the specialist lift mixture is applied.

## Held-Out Result

This branch is closed.

### `prototype_risk_prior_regime_memory`

Dead on the target and slightly harmful overall at larger budgets.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- hard near-tie target match stayed at `90.53%`
- hard near-tie mean delta regret stayed at `0.0000`
- by `0.50%+` it began introducing mild overall regressions from false-positive non-target selections

### `prototype_risk_prior_regime_memory_hybrid`

Broad-safe but still target-dead.

Best aggregate point:

- budget `2.0%`
- overall coverage `0.99%`
- overall target match `96.51% -> 96.59%`
- overall mean delta regret `-0.00449`

But the real target never moved:

- held-out `stable_positive_v2` recovery `0%` at every budget
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`
- large-gap control unchanged

## Interpretation

The explicit risk prior made the regime split *broader*, not sharper.

The learned regime allocations on held-out seeds were dominated by the residual specialist:

- plain model: residual probability mean about `0.62` to `0.70`
- hybrid model: residual probability mean about `0.68` to `0.78`

That did not recover the rare correction family. It simply redirected more coverage into obvious non-target states.

So this is worse than both nearby baselines:

- worse than `prototype_regime_split_memory_hybrid`, which at least surfaced a weak `25%` held-out stable-positive niche
- much worse than `prototype_memory_agree_blend_hybrid`, which remains the real micro-budget lead

## Decision

Close `prototype_risk_prior_regime_memory` and `prototype_risk_prior_regime_memory_hybrid`.

Keep the shortlist unchanged:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1 follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band follow-up
- `prototype_evidence_agree_hybrid` remains the best aggregate-quality matched-band follow-up
