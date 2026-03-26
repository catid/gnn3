# Prototype Supported Residual-Regime Evidence Experiment

## Question

Test whether the anchored residual-regime evidence family can be sharpened by
adding explicit positive support gates for the headroom and baseline-error
specialist lifts.

The design goal was:

- keep the live evidence-agreement score as the anchor
- keep the bounded nonnegative headroom and residual lifts from the previous
  residual-regime branch
- add support gates that only allow specialist lift when the anchor and the
  specialist both show the right positive evidence pattern
- reduce the broad false-positive behavior from the residual-regime family
  without losing the one held-out positive it already found

## Implementation

- New head: `SupportedResidualRegimeEvidenceAgreementPrototypeDeferHead`
- New runner:
  `scripts/run_prototype_supported_residual_regime_evidence_defer.py`
- Variants:
  - `prototype_supported_residual_regime_evidence`
  - `prototype_supported_residual_regime_evidence_hybrid`

The new piece is a small support gate per specialist. Each gate sees:

- anchor score
- specialist score
- positive lift above the anchor
- margin lift above the anchor
- absolute score gap
- anchor margin
- specialist margin
- score interaction

So the specialist can only add lift when the local evidence pattern looks
supportive, not just when the specialist score happens to be larger.

## Held-Out Result

This branch is closed.

### `prototype_supported_residual_regime_evidence`

Effectively dead.

Best point:

- budget `2.0%`
- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `0%`
- overall target match `96.51% -> 96.52%`
- overall mean delta regret `-0.00032`

But the real frontier did not move:

- hard near-tie target match stayed `90.53%`
- hard near-tie mean delta regret stayed `0.0000`

### `prototype_supported_residual_regime_evidence_hybrid`

Weak positive, but still not promotable.

Best efficient point:

- budget `0.75%`
- overall coverage `0.35%`
- held-out `stable_positive_v2` recovery `25%`
- overall target match `96.51% -> 96.66%`
- overall mean delta regret `-0.00934`

The hard slice is still capped at the weaker band:

- hard near-tie target match only `90.53% -> 90.60%`
- hard near-tie mean delta regret only `-0.00476`

And the supposed sharpening did not actually remove diffuse selection:

- high-headroom near-tie coverage `20.0%`, defer precision `25%`,
  false-positive rate `75%`
- baseline-error near-tie coverage `9.72%`, defer precision `24.65%`,
  false-positive rate `75.35%`

Relative to the previous residual-regime hybrid, this is a small coverage
efficiency improvement, but not a real frontier improvement.

## Interpretation

The support gates preserved the one held-out positive from the residual-regime
family and made the hybrid cheaper in coverage, but they did not actually
sharpen the source-family selection. The branch still spends most of its
targeted coverage on false positives, still recovers only `1 / 4` held-out
stable-positive-v2 cases, and still cannot move beyond the weaker
`90.53% -> 90.60%` hard-slice band.

So this is another closed weak positive:

- slightly better than the previous residual-regime hybrid on
  coverage-efficiency
- still worse than `prototype_hybrid` on the ultra-low-coverage frontier
- still worse than `prototype_memory_agree_blend_hybrid` on the micro-budget
  Tier-1 point
- still worse than `prototype_agree_mix_hybrid` on coverage-efficient
  matched-band quality
- still worse than `prototype_evidence_agree_hybrid` on aggregate
  matched-band quality

## Decision

Close `prototype_supported_residual_regime_evidence` and
`prototype_supported_residual_regime_evidence_hybrid`.

Keep the shortlist unchanged:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` remains the best aggregate-quality
  matched-band follow-up
