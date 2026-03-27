# Prototype Dual Teacher-Rebuilt Negative-Bank Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the full teacher-rebuilt failure and the weaker shared-only result
both happened because teacher guidance belongs mainly in the dual branch.

The design goal was:

- keep the accepted branchwise-max scoring geometry unchanged
- rebuild only the dual negative bank from teacher-marked harmful states
- leave the shared negative bank learned
- preserve the accepted higher-budget branch behavior while still allowing
  teacher guidance to sharpen the dual branch

So this is the branch-symmetric counterpart to the closed shared-only rebuild.

## Implementation

- New head:
  `DualTeacherRebuiltNegativeBankBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_dual_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_dual_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_dual_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the accepted branchwise-max reference:

- fit the same branchwise-max prototype head first
- rebuild only the dual negative bank from harmful teacher-bank states
- keep the full `8/8` dual negative capacity
- keep the shared negative bank learned
- keep the rest of the model unchanged

So this isolates whether teacher-guided reconstruction helps only when applied
to the dual branch.

## Rebuild summary

This was a real dual-bank edit:

- harmful dual-bank candidates: `983`
- rebuilt dual negatives: `8/8`
- dual negative padding from learned bank: `0`
- dual negative average stable-positive overlap:
  - plain `0.819`
  - hybrid `0.653`
- dual negative support std:
  - plain `2.13`
  - hybrid `2.02`

The shared bank stayed effectively learned:

- shared negative count `8`
- shared negative support std stayed near zero:
  - plain `0.030`
  - hybrid `0.055`

So this was again a real branch-local reconstruction, not another null edit.

## Held-Out Result

### `prototype_dual_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match unchanged at `96.51%`
- overall mean delta regret `0.0000`

So the plain branch again collapses completely to baseline.

### `prototype_dual_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed positive, but still dominated.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.57%`
- overall mean delta regret `-0.0040`

At `0.25%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- overall mean delta regret `-0.0060`

At `0.50–0.75%` nominal budgets:

- overall coverage `0.51–0.76%`
- held-out `stable_positive_v2` recovery stays at `50%`
- hard near-tie target match stays at `90.39% -> 90.45%`
- overall mean delta regret improves to `-0.0095` and `-0.0111`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.39% -> 90.53%`
- overall mean delta regret `-0.0126`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.39% -> 90.60%`
- overall mean delta regret `-0.0153`

At `2.00%` nominal budget:

- overall coverage `1.82%`
- held-out `stable_positive_v2` recovery still `83.3%`
- hard near-tie target match still `90.39% -> 90.60%`
- overall target match `96.84%`
- overall mean delta regret `-0.0158`

Large-gap controls stayed clean:

- no large-gap miss regression
- overall regret stayed non-positive in the live hybrid branch

## Comparison against the accepted branchwise-max reference

At `0.10–0.25%`:

- accepted branchwise-max remains better:
  - `0.10%`: `-0.0050` vs dual-rebuilt `-0.0040`
  - `0.25%`: `-0.0074` vs dual-rebuilt `-0.0060`
- both have the same `50%` held-out `stable_positive_v2` recovery and the same
  `90.45%` hard near-tie point

So dual-only rebuild does not even win the micro-budget lane.

At `0.50–0.75%`:

- accepted branchwise-max still has stronger overall regret
- dual-only rebuild improves clearly over the full and shared-only rebuilds
- but it still keeps only the weaker `50%` / `90.45%` lane

At `1.00%`:

- accepted branchwise-max: `83.3%` held-out `stable_positive_v2`, hard near-tie
  `90.39% -> 90.60%`, overall mean delta regret `-0.0145`
- dual-only rebuild: only `66.7%`, only `90.39% -> 90.53%`, overall mean delta
  regret `-0.0126`

At `1.50–2.00%`:

- accepted branchwise-max reaches the stronger higher-budget frontier
- dual-only rebuild improves to `83.3%` / `90.60%`
- but still gives back the accepted `100%` / `90.68%` lane and aggregate regret

So accepted branchwise-max still dominates at every matched budget.

## Comparison against the other teacher-rebuild variants

This is the useful structural comparison.

Against the full teacher rebuild:

- full rebuild collapsed to the weak `50%` / `90.45%` lane from `0.50%`
  upward, with overall mean delta regret only `-0.0035`
- dual-only rebuild is much stronger above `0.50%`, reaching `83.3%` /
  `90.60%` and `-0.0158`

Against the shared-only teacher rebuild:

- shared-only rebuild owned the micro-budget teacher-rebuild lane:
  - `0.10%`: `-0.0061`
  - `0.25%`: `-0.0081`
- dual-only rebuild is worse there:
  - `0.10%`: `-0.0040`
  - `0.25%`: `-0.0060`
- but dual-only rebuild is stronger above `0.50%`

So teacher-guided reconstruction is much more viable in the dual branch than in
the full or shared-only forms, but it still does not challenge the accepted
branchwise-max frontier.

## Interpretation

This closes the simple “teacher guidance belongs in the dual branch” hypothesis.

What happened:

- dual-only rebuild is clearly the best of the three simple teacher-rebuild
  variants
- full rebuild over-edited both branches and collapsed
- shared-only rebuild only recreated a weaker micro-budget companion
- dual-only rebuild recovers a decent mid/high-budget branchwise-max-family lane
- but it still does not beat the accepted branchwise-max reference at any
  matched budget

So the right conclusion is not promotion. It is:

- if teacher-guided bank reconstruction reopens, the dual branch is the only
  reasonable place to start
- but simple fixed-cardinality harmful-state centroid replacement is still not
  enough

## Decision

Close:

- `prototype_dual_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_dual_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Updated interpretation:

- accepted branchwise-max remains the main live reference
- rescue-weighted anchored dual lift remains the best micro-budget companion
- do not reopen simple dual-only teacher-guided harmful-state negative-bank
  reconstruction, because although it is the strongest teacher-rebuild variant,
  it still stays weaker than accepted branchwise-max at every matched budget
