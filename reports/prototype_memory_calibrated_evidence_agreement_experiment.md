# Prototype Memory-Calibrated Evidence Agreement Experiment

## Question

Test whether the matched-band evidence-agreement family can be improved by
showing its gate the memory-anchor signal without forcing a memory-anchored
final score.

The design goal was:

- keep the existing shared-vs-dual evidence-agreement score path
- add memory score and memory prototype evidence only as calibration inputs
  to the agreement gate
- avoid the hard memory anchor that already failed in the direct
  memory-evidence blend family

## Implementation

- New head: `MemoryCalibratedEvidenceAgreementPrototypeDeferHead`
- New runner: `scripts/run_prototype_memory_calibrated_evidence_agreement_defer.py`
- Variants:
  - `prototype_memory_calibrated_evidence`
  - `prototype_memory_calibrated_evidence_hybrid`

The new gate sees:

- shared score and dual score
- shared / dual positive and negative top evidence
- shared / dual margins
- memory score
- memory positive and negative top evidence
- shared-vs-memory and dual-vs-memory margin deltas

The final score remains the same evidence-agreement interpolation:

- `shared_score + gate * (dual_score - shared_score)`

So memory is a calibrator here, not the output anchor.

## Held-Out Result

This branch is closed.

### `prototype_memory_calibrated_evidence`

Dead on the target and inert overall.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- hard near-tie target match stayed at `90.53%`
- hard near-tie mean delta regret stayed at `0.0000`
- by `2.0%` nominal budget it spent about `1.00%` overall coverage almost
  entirely on non-target states, but still produced no aggregate gain

The only visible motion was harmless control churn:

- large-gap control coverage grew to `1.67%`
- large-gap control target match stayed at `99.79%`
- large-gap control mean delta regret stayed at `0.0000`

### `prototype_memory_calibrated_evidence_hybrid`

Still target-dead.

Best aggregate point:

- budget `0.10%` nominal, already saturated through `2.0%`
- overall coverage `0.24%`
- overall target match `96.51% -> 96.52%`
- overall mean delta regret `-0.00125`
- overall mean delta miss `-0.00012`

But the real target never moved:

- held-out `stable_positive_v2` recovery `0%` at every budget
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`
- hard near-tie coverage rose to `0.94%` with zero useful selections

## Interpretation

Direct memory-context injection into the current evidence-agreement gate does
not sharpen the sparse-positive frontier.

The matched-band evidence-agreement family already appears to have the useful
local evidence it needs. Adding memory score and memory top-match context did
not improve ranking of the rare positive cases. It only redirected a small
amount of coverage into broad-safe non-target states.

So this is worse than every live nearby baseline:

- much worse than `prototype_hybrid`, which still recovers real held-out
  stable-positive states
- much worse than `prototype_memory_agree_blend_hybrid`, which is still the
  only real micro-budget memory-led follow-up
- much worse than `prototype_agree_mix_hybrid` and
  `prototype_evidence_agree_hybrid`, which both still reach the `90.73%`
  hard-slice band

## Decision

Close `prototype_memory_calibrated_evidence` and
`prototype_memory_calibrated_evidence_hybrid`.

Keep the shortlist unchanged:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` remains the best aggregate-quality
  matched-band follow-up
