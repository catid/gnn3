# Prototype Shared Teacher-Rebuilt Negative-Bank Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the full teacher-rebuilt negative-bank failure was caused mainly by
over-editing both branches at once.

The design goal was:

- keep the accepted branchwise-max scoring geometry unchanged
- rebuild only the shared negative bank from teacher-marked harmful states
- leave the dual negative bank learned and effectively untouched
- preserve the accepted mid-budget branch behavior while only sharpening the
  micro-budget lane

So this is the narrowest branch-local version of teacher-guided bank
reconstruction.

## Implementation

- New head:
  `SharedTeacherRebuiltNegativeBankBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_shared_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_shared_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_shared_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the earlier full teacher rebuild:

- fit the same accepted branchwise-max head first
- rebuild only the shared negative bank from harmful teacher-bank states
- keep the full `8/8` shared negative capacity
- keep the dual negative bank learned
- keep the rest of the model unchanged

So this isolates whether teacher guidance belongs only in the shared branch.

## Rebuild summary

This was again a real bank edit:

- harmful shared-bank candidates: `983`
- rebuilt shared negatives: `8/8`
- shared negative padding from learned bank: `0`
- shared negative average stable-positive overlap:
  - plain `0.388`
  - hybrid `0.425`
- shared negative support std:
  - plain `1.51`
  - hybrid `2.33`

The dual bank stayed effectively learned:

- dual negative count `8`
- dual negative support std stayed near zero:
  - plain `0.038`
  - hybrid `0.039`

So this really was a shared-only reconstruction, not another null edit.

## Held-Out Result

### `prototype_shared_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix`

Closed, dead to slightly harmful.

At `0.10–1.50%` nominal budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match unchanged at `96.51%`
- overall mean delta regret `0.0000`

At `2.00%` nominal budget:

- still `0%` held-out `stable_positive_v2`
- hard near-tie slips slightly to `90.31%`
- overall mean delta regret turns slightly positive

So the plain branch again collapses to baseline.

### `prototype_shared_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed positive, but dominated.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.59%`
- overall mean delta regret `-0.0061`

At `0.25%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.63%`
- overall mean delta regret `-0.0081`

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match still only `90.39% -> 90.45%`
- overall mean delta regret `-0.0105`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- overall mean delta regret `-0.0141`

At `1.50–2.00%` nominal budget:

- overall coverage `1.52–2.00%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- overall mean delta regret `-0.0157`

Large-gap controls stayed clean:

- no large-gap miss regression
- overall regret stayed non-positive in the live hybrid branch

## Comparison against the accepted branchwise-max reference

At `0.10–0.25%`:

- accepted branchwise-max is weaker on overall mean delta regret
- shared-only rebuild improves:
  - `0.10%`: `-0.0050 -> -0.0061`
  - `0.25%`: `-0.0074 -> -0.0081`
- while preserving the same `50%` held-out `stable_positive_v2` recovery and
  the same `90.45%` hard near-tie point

So this is a real micro-budget positive over the accepted branchwise-max
reference.

Above `0.25%`, accepted branchwise-max dominates again:

- `0.50%`: accepted branchwise-max still has stronger overall mean delta regret
- `1.00%`: accepted branchwise-max reaches `83.3%` held-out
  `stable_positive_v2` and `90.60%`, while shared-only rebuild only reaches
  `66.7%` and `90.53%`
- `2.00%`: accepted branchwise-max still keeps the stronger higher-budget
  frontier with full held-out recall, while shared-only rebuild stalls at
  `83.3%` / `90.60%`

So the shared-only rebuild does not replace the main accepted branch.

## Comparison against rescue-weighted anchored dual lift

This is the key portfolio comparison.

At `0.10–0.25%`, rescue-weighted anchored dual lift still wins the micro-budget
lane:

- rescue-weighted:
  - `0.10%`: overall mean delta regret `-0.0064`
  - `0.25%`: overall mean delta regret `-0.0088`
- shared-only teacher rebuild:
  - `0.10%`: `-0.0061`
  - `0.25%`: `-0.0081`

They share the same held-out `stable_positive_v2` recovery and hard near-tie
point there.

So the shared-only rebuild is a real micro-budget positive, but it is still
strictly weaker than the already-accepted rescue-weighted micro-budget
companion.

## Interpretation

This closes the simple “teacher guidance only belongs in the shared branch”
hypothesis.

What happened:

- rebuilding only the shared negative bank is clearly safer than rebuilding
  both branches
- unlike the full teacher rebuild, it preserves a useful micro-budget lane
- but it still does not create a new operating region
- rescue-weighted anchored dual lift remains the stronger micro-budget answer
- accepted branchwise-max still dominates from `0.50%` upward

So this is not a promotion. It is a branchwise confirmation that teacher-guided
shared-bank reconstruction is better than full bank reconstruction, but still
not the right stabilization mechanism.

## Decision

Close:

- `prototype_shared_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_shared_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Updated interpretation:

- accepted branchwise-max remains the main live reference
- rescue-weighted anchored dual lift remains the best micro-budget companion
- do not reopen simple shared-only teacher-guided harmful-state negative-bank
  reconstruction, because it only recreated a weaker version of the same
  micro-budget lane and never improved the `0.50%+` frontier
