# Prototype Teacher-Rebuilt Negative-Bank Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the accepted branchwise-max family improves if the shared and dual
negative prototype banks are rebuilt at full cardinality from
teacher-marked harmful states instead of being softly pruned, hard-masked, or
retuned with extra scalar margins.

The design goal was:

- keep the accepted branchwise-max scoring geometry unchanged
- keep the full `8/8` shared and dual negative-bank capacity
- replace the learned negative prototypes with diverse weighted centroids from
  teacher-marked harmful states
- rebuild the negative support logits from the resulting cluster masses
- avoid the earlier failure mode where hard masking simply deleted too much bank
  capacity

So this is the simple teacher-guided bank-reconstruction path that remained open
in the round-13 handoff notes.

## Implementation

- New head:
  `TeacherRebuiltNegativeBankBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the accepted branchwise-max reference:

- fit the same branchwise-max prototype head first
- collect the train harmful teacher-bank states
- encode them into the shared negative and dual negative spaces separately
- choose `8` diverse seeds in each negative space with a weighted novelty rule
- assign all harmful states back to those seeds and rebuild fixed-cardinality
  weighted centroids
- set the rebuilt negative supports from the resulting cluster masses

So the branch capacity stays fixed, but the negative-bank contents are replaced
completely.

## Rebuild summary

This was a real bank edit, not a null edit:

- harmful candidates per bank: `983`
- rebuilt shared negatives: `8/8`
- rebuilt dual negatives: `8/8`
- padding from learned bank: `0`
- shared negative support std:
  - plain `2.10`
  - hybrid `1.84`
- dual negative support std:
  - plain `2.06`
  - hybrid `2.35`

The average stable-positive overlap of the rebuilt harmful centroids was still
high:

- shared negative overlap:
  - plain `0.47`
  - hybrid `0.57`
- dual negative overlap:
  - plain `0.56`
  - hybrid `0.75`

So the rebuild really changed the banks, but it also pulled them too close to
the stable-positive source family.

## Held-Out Result

### `prototype_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match unchanged at `96.51%`
- overall mean delta regret `0.0000`

So the plain branch collapses completely to baseline.

### `prototype_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed weak positive, but far below the accepted frontier.

At `0.10–0.25%` nominal budget:

- overall coverage only `0.07–0.14%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall mean delta regret only `-0.0026`

At `0.50%` nominal budget:

- overall coverage `0.26%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.54%`
- overall mean delta regret only `-0.0035`

At `0.75–2.00%` nominal budget:

- overall coverage only `0.39–1.01%`
- held-out `stable_positive_v2` recovery stays stuck at `50%`
- hard near-tie target match stays stuck at `90.39% -> 90.45%`
- overall mean delta regret stays stuck at `-0.0035`

Large-gap controls stayed clean:

- no large-gap miss regression
- overall regret stayed non-positive

But this never approaches the accepted branchwise-max correction frontier.

## Comparison against the accepted branchwise-max reference

At `0.10%`:

- accepted branchwise-max: `50%` held-out `stable_positive_v2`, hard near-tie
  `90.39% -> 90.45%`, overall mean delta regret `-0.0050`
- teacher-rebuilt hybrid: `0%`, unchanged hard near-tie, overall mean delta
  regret only `-0.0026`

At `0.50%`:

- accepted branchwise-max: `50%`, hard near-tie `90.39% -> 90.45%`, overall
  mean delta regret `-0.0111`
- teacher-rebuilt hybrid: same `50%` and same `90.45%`, but overall mean delta
  regret only `-0.0035`

At `1.00%`:

- accepted branchwise-max: `83.3%` held-out `stable_positive_v2`, hard near-tie
  `90.39% -> 90.60%`, overall mean delta regret `-0.0145`
- teacher-rebuilt hybrid: only `50%`, only `90.39% -> 90.45%`, overall mean
  delta regret only `-0.0035`

At `2.00%`:

- accepted branchwise-max: `100%` held-out `stable_positive_v2`, hard near-tie
  `90.39% -> 90.68%`, overall mean delta regret `-0.0167`
- teacher-rebuilt hybrid: only `50%`, only `90.39% -> 90.45%`, overall mean
  delta regret only `-0.0035`

So the accepted branchwise-max reference dominates cleanly at every matched
budget.

## Interpretation

This closes the simple teacher-guided bank-reconstruction path.

What happened:

- keeping full bank cardinality was not enough
- replacing all shared and dual negative prototypes with harmful-state centroids
  over-corrected the negative bank
- the rebuilt centroids stayed too close to the stable-positive source family
- the resulting policy became conservative and almost stopped expanding beyond a
  tiny weak `50%` / `90.45%` lane

So the problem with the earlier masking experiments was not only “too few
negative prototypes.” The full harmful-state reconstruction itself is also the
wrong bank-editing rule in this family.

## Decision

Close:

- `prototype_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Updated interpretation:

- accepted branchwise-max remains the main live reference
- rescue-weighted anchored dual lift remains the only positive micro-budget
  companion
- do not reopen simple full teacher-guided harmful-state negative-bank
  reconstruction, because it materially edited the banks and still collapsed the
  held-out frontier
