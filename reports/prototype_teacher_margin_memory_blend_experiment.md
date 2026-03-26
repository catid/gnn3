# Prototype Teacher-Margin Memory Blend Experiment

## Question

Test whether the live `prototype_memory_agree_blend` geometry can be improved by adding an auxiliary teacher-gain head.

The design constraint was strict:

- keep the current memory-anchor branch intact
- keep the current score-only agreement lift intact
- add only a bounded one-sided lift calibrated by predicted teacher gain
- supervise that gain prediction from `best_safe_teacher_gain` on `stable_positive_v2_case` rows only

The target was to beat `prototype_memory_agree_blend_hybrid` on the micro-budget frontier or at least improve the matched-band follow-up without losing the stable-positive-v2 pack.

## Implementation

- New head: `TeacherMarginMemoryBlendPrototypeDeferHead`
- New runner: `scripts/run_prototype_teacher_margin_memory_blend_defer.py`
- Variants:
  - `prototype_teacher_margin_memory`
  - `prototype_teacher_margin_memory_hybrid`
- Auxiliary target:
  - `log1p(best_safe_teacher_gain)`
  - Smooth L1 loss
  - applied only on `stable_positive_v2_case`
  - loss weight `0.10`
- The gain branch does not create a new routing path. It only scales the existing positive lift:
  - base path: `memory_score + outer_gate * positive_lift`
  - added path: bounded extra lift from predicted gain

Held-out evaluation reused the standard round-12 teacher-bank artifact and the same train/eval split as the other prototype follow-ups:

- train: seed `314`
- held-out eval: seeds `315` and `316`

## Held-Out Result

This branch is closed.

### `prototype_teacher_margin_memory`

Dead on the real target.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- hard near-tie target match stayed at `90.53%`
- hard near-tie mean delta regret stayed at `0.0000`
- all selected states were false positives

### `prototype_teacher_margin_memory_hybrid`

Broad-safe but not target-aware.

Best aggregate point:

- budget `1.5%` to `2.0%`
- overall coverage `0.78%` to `1.03%`
- overall target match `96.51% -> 96.58%`
- overall mean delta regret `-0.00417`

But the target slice stayed dead:

- held-out `stable_positive_v2` recovery `0%` at every budget
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`
- large-gap control stayed unchanged

## Interpretation

Explicit teacher-gain supervision did not sharpen the rare correction family. Instead, it pushed the hybrid toward broad-safe non-target states that already admit harmless improvements.

That makes this branch worse than every live shortlist member for the actual decision problem:

- worse than `prototype_hybrid` on the ultra-low-coverage stable-positive frontier
- worse than `prototype_memory_agree_blend_hybrid` on the micro-budget frontier
- worse than `prototype_agree_mix_hybrid` and `prototype_evidence_agree_hybrid` on the matched-band frontier

The main qualitative result is useful:

- direct teacher-gain regression on the current memory-anchor geometry does not recover the sparse stable-positive family
- the gain signal appears to calibrate broad-safe helpfulness rather than the rare correction source family

## Decision

Close `prototype_teacher_margin_memory` and `prototype_teacher_margin_memory_hybrid`.

Keep the prototype shortlist unchanged:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1 follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band follow-up
- `prototype_evidence_agree_hybrid` remains the best aggregate-quality matched-band follow-up
