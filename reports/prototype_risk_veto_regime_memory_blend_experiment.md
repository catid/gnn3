# Prototype Risk-Veto Regime Memory Blend Experiment

## Question

Test whether the failed risk-prior regime split can be rescued by using risk
features only as suppressive vetoes instead of additive routing signals.

The design goal was:

- keep the live memory-anchor geometry
- keep the one-sided agreement lift
- keep separate headroom and residual specialist lift gates
- use risk features only to veto specialist lifts when the local state looks
  unsafe for that regime

## Implementation

- New head: `RiskVetoRegimeMemoryBlendPrototypeDeferHead`
- New runner: `scripts/run_prototype_risk_veto_regime_memory_blend_defer.py`
- Variants:
  - `prototype_risk_veto_regime_memory`
  - `prototype_risk_veto_regime_memory_hybrid`

The veto branch reads the same normalized risk features already used elsewhere:

- margin-only features
- margin/regime features
- continuation-gap headroom

Unlike the prior branch, these features do not add mass into the regime logits.
They only gate the headroom and residual specialist lifts downward.

## Held-Out Result

This branch is closed.

### `prototype_risk_veto_regime_memory`

Dead on the real target.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- hard near-tie target match stayed at `90.53%`
- hard near-tie mean delta regret stayed at `0.0000`
- the only visible effect was a tiny broad-safe overall gain from one non-target
  selection:
  - best overall point was already saturated by `0.50%` nominal budget
  - overall coverage `0.25%`
  - overall target match `96.51% -> 96.53%`
  - overall mean delta regret `-0.00217`

### `prototype_risk_veto_regime_memory_hybrid`

Also dead on the target and even more collapsed.

Best overall point:

- budget `2.0%`
- overall coverage `0.03%`
- overall target match `96.51% -> 96.53%`
- overall mean delta regret `-0.00204`
- overall mean delta miss `-0.00012`

But the real target never moved:

- held-out `stable_positive_v2` recovery `0%` at every budget
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`
- large-gap control moved only through a single harmless control correction:
  - target match `99.79% -> 99.84%`
  - mean delta regret `-0.00359`

## Interpretation

The suppressive veto over-corrected.

The previous additive risk-prior branch broadened the regime split into
non-target states. This veto version did the opposite: it collapsed the already
weak specialist signal almost completely back to baseline behavior.

So this is worse than both nearby baselines:

- worse than `prototype_regime_split_memory_hybrid`, which at least surfaced a
  weak `25%` held-out stable-positive niche
- worse than `prototype_risk_prior_regime_memory_hybrid`, which was too broad
  but still moved broad-safe aggregate regret a bit more
- much worse than `prototype_memory_agree_blend_hybrid`, which remains the real
  micro-budget lead

## Decision

Close `prototype_risk_veto_regime_memory` and
`prototype_risk_veto_regime_memory_hybrid`.

Keep the shortlist unchanged:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` remains the best aggregate-quality
  matched-band follow-up
