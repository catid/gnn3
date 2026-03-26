# Prototype Teacher-Signal Evidence Agreement Experiment

## Question

Test whether the live evidence-agreement prototype family can be sharpened by
adding auxiliary teacher-signal heads during training.

The design goal was:

- keep the current shared-vs-dual evidence-agreement score path
- keep the same inference-time inputs
- add small auxiliary predictions for teacher-bank committee support and safe
  teacher gain
- use those targets only to shape the gate trunk, not to directly anchor the
  final score

## Implementation

- New head: `TeacherSignalEvidenceAgreementPrototypeDeferHead`
- New runner: `scripts/run_prototype_teacher_signal_evidence_agreement_defer.py`
- Variants:
  - `prototype_teacher_signal_evidence`
  - `prototype_teacher_signal_evidence_hybrid`

Auxiliary targets were attached from the existing teacher-bank artifact:

- normalized `committee_support`
- normalized clipped `best_safe_teacher_gain`

The main defer objective stayed unchanged: sparse-positive BCE with the same
harmful-row weighting and the same prototype regularization used by the live
evidence-agreement family.

## Held-Out Result

This branch is closed.

### `prototype_teacher_signal_evidence`

Target-dead.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- hard near-tie target match stayed at `90.53%`
- hard near-tie mean delta regret stayed at `0.0000`

Best aggregate point:

- budget `2.0%`
- overall coverage `1.00%`
- overall target match `96.51% -> 96.56%`
- overall mean delta regret `-0.00216`

That gain came entirely from non-target control behavior:

- large-gap control target match `99.79% -> 99.84%`
- large-gap control mean delta regret `-0.00359`

### `prototype_teacher_signal_evidence_hybrid`

Also target-dead and weaker overall.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- hard near-tie target match stayed at `90.53%`
- hard near-tie mean delta regret stayed at `0.0000`

Best aggregate point:

- budget `1.5%` and `2.0%` both saturated the same overall result
- overall coverage `0.91%` to `1.15%`
- overall target match `96.51% -> 96.52%`
- overall mean delta regret `-0.00042`

The extra risk branch only redirected more coverage into false-positive
non-target selections without recovering the true sparse correction subset.

## Interpretation

Auxiliary teacher-signal prediction did not sharpen the live evidence-agreement
family.

The gate trunk did learn something, but it learned the wrong thing for this
promotion surface: broad-safe control cleanup instead of sparse-positive
frontier recovery.

So this is still worse than every live lead:

- worse than `prototype_hybrid` on the ultra-low-coverage frontier
- worse than `prototype_memory_agree_blend_hybrid` on the micro-budget Tier-1
  point
- worse than `prototype_agree_mix_hybrid` on coverage-efficient matched-band
  quality
- worse than `prototype_evidence_agree_hybrid` on aggregate matched-band
  quality

## Decision

Close `prototype_teacher_signal_evidence` and
`prototype_teacher_signal_evidence_hybrid`.

Keep the shortlist unchanged:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` remains the best aggregate-quality
  matched-band follow-up
