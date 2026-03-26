# Prototype Budget-Conditioned Evidence Agreement Experiment

## Question

Test whether the live evidence-agreement prototype family can improve
coverage-specific calibration by conditioning its gate directly on the intended
defer budget.

The design goal was:

- keep the current shared-vs-dual evidence-agreement score path
- add an explicit budget-conditioning input to the gate
- train across the round budget grid with harsher negative penalties at lower
  budgets so the model has a reason to use the budget input
- avoid reopening the already-closed budget-conditioned outer gating on the
  memory-anchor family

## Implementation

- New head: `BudgetConditionedEvidenceAgreementPrototypeDeferHead`
- New runner: `scripts/run_budget_conditioned_evidence_agreement_defer.py`
- Variants:
  - `prototype_budget_evidence_agree`
  - `prototype_budget_evidence_agree_hybrid`

Training duplicated the cached seed314 rows across the standard
`0.10% -> 2.00%` budget grid and reweighted negatives more aggressively at
lower budgets.

## Held-Out Result

This branch is closed.

### `prototype_budget_evidence_agree`

Dead on the target and inert overall.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- hard near-tie target match stayed at `90.53%`
- hard near-tie mean delta regret stayed at `0.0000`
- as budget increased it simply spent more coverage on non-target states:
  - overall coverage reached `2.00%`
  - large-gap control coverage reached `3.75%`
  - overall and large-gap target match still stayed unchanged

### `prototype_budget_evidence_agree_hybrid`

Also target-dead.

Best aggregate point:

- budget `0.10%`, already saturated through `2.00%`
- overall coverage `0.01%`
- overall target match `96.51% -> 96.52%`
- overall mean delta regret `-0.00079`

But that gain came only from a tiny large-gap control fix:

- large-gap control coverage `0.05%`
- large-gap target match `99.79% -> 99.84%`
- large-gap mean delta regret `-0.00359`

The real target never moved:

- held-out `stable_positive_v2` recovery `0%` at every budget
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`

## Interpretation

Budget-conditioning did not rescue the live evidence-agreement family.

The explicit budget input and budget-specific training weights did change the
selection pattern, but only on broad-safe control states. The model still never
found the rare positive correction subset.

So this is worse than every live lead:

- worse than `prototype_hybrid` on the ultra-low-coverage frontier
- worse than `prototype_memory_agree_blend_hybrid` on the micro-budget Tier-1
  point
- worse than `prototype_agree_mix_hybrid` on coverage-efficient matched-band
  quality
- worse than `prototype_evidence_agree_hybrid` on aggregate matched-band
  quality

## Decision

Close `prototype_budget_evidence_agree` and
`prototype_budget_evidence_agree_hybrid`.

Keep the shortlist unchanged:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` remains the best aggregate-quality
  matched-band follow-up
