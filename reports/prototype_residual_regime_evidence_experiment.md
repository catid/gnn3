# Prototype Residual-Regime Evidence Experiment

## Question

Test whether the evidence-agreement family can use the two known positive source
families without the diffuse false-positive behavior from the full
regime-specialized bank split.

The design goal was:

- keep the live evidence-agreement score as the anchor
- train headroom and baseline-error specialist bank families
- allow specialists to add only bounded positive lift above the anchor
- use a small regime head to decide how much headroom vs residual lift to take

## Implementation

- New head: `ResidualRegimeEvidenceAgreementPrototypeDeferHead`
- New runner: `scripts/run_prototype_residual_regime_evidence_defer.py`
- Variants:
  - `prototype_residual_regime_evidence`
  - `prototype_residual_regime_evidence_hybrid`

This differs from the closed full regime split in one important way:

- specialists cannot replace the anchor score
- they can only add nonnegative lift above the base evidence-agreement score

## Held-Out Result

This branch is closed.

### `prototype_residual_regime_evidence`

Weak positive, but still not competitive.

Best point:

- budget `0.75%`
- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `25%`
- overall target match `96.51% -> 96.52%`
- overall mean delta regret `-0.00017`

But the real target stayed weak:

- hard near-tie target match only `90.53% -> 90.60%`
- hard near-tie mean delta regret only `-0.00102`

### `prototype_residual_regime_evidence_hybrid`

Also weak and not promotable.

Best aggregate point:

- budget `2.0%`
- overall coverage `1.07%`
- held-out `stable_positive_v2` recovery `25%`
- overall target match `96.51% -> 96.69%`
- overall mean delta regret `-0.00955`

But the Tier-1/Tier-2 trade is still too weak:

- hard near-tie target match only `90.53% -> 90.60%`
- hard near-tie mean delta regret only `-0.00234`
- large-gap control coverage rose to `1.15%`
- high-headroom and baseline-error slices still carried many false positives

## Interpretation

Anchoring the specialist regimes helped relative to the full regime split, but
not enough.

The branch recovered one held-out stable-positive-v2 case, but it never moved
beyond the weaker `90.53% -> 90.60%` hard-slice band and still spent too much
coverage away from the real sparse-positive frontier.

So it still loses to every live lead:

- worse than `prototype_hybrid` on the ultra-low-coverage frontier
- worse than `prototype_memory_agree_blend_hybrid` on the micro-budget Tier-1
  point
- worse than `prototype_agree_mix_hybrid` on coverage-efficient matched-band
  quality
- worse than `prototype_evidence_agree_hybrid` on aggregate matched-band
  quality

## Decision

Close `prototype_residual_regime_evidence` and
`prototype_residual_regime_evidence_hybrid`.

Keep the shortlist unchanged:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` remains the best aggregate-quality
  matched-band follow-up
