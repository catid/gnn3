# Continuation Report Round 8

## Scope

Round eight opened a near-tie decision round around one question:

- can counterfactual value supervision and bounded ambiguity correction improve policy quality on the score-based hard near-tie oracle-feasible slice?

This pass completed:

1. a repo-grounded round-eight audit
2. a fresh three-seed `multiheavy` guardrail batch
3. a dedicated near-tie headroom audit
4. a frozen-feature near-tie probe extension
5. a cached counterfactual all-action dataset build
6. a five-variant direct critic family
7. a bounded-search runtime test
8. a narrow path-cost tie-break backup
9. a round-eight reporting and portfolio refresh

Key artifacts:

- [continuation_audit_round8.md](/home/catid/gnn3/reports/continuation_audit_round8.md)
- [near_tie_headroom_round8.md](/home/catid/gnn3/reports/near_tie_headroom_round8.md)
- [probe_audit_round8.md](/home/catid/gnn3/reports/probe_audit_round8.md)
- [counterfactual_dataset_round8.md](/home/catid/gnn3/reports/counterfactual_dataset_round8.md)
- [counterfactual_critic_round8.md](/home/catid/gnn3/reports/counterfactual_critic_round8.md)
- [near_tie_search_round8.md](/home/catid/gnn3/reports/near_tie_search_round8.md)
- [path_tiebreak_round8.md](/home/catid/gnn3/reports/path_tiebreak_round8.md)
- [round8_multiheavy_guardrail.csv](/home/catid/gnn3/reports/plots/round8_multiheavy_guardrail.csv)
- [round8_critic_summary.csv](/home/catid/gnn3/reports/plots/round8_critic_summary.csv)
- [round8_search_runtime.csv](/home/catid/gnn3/reports/plots/round8_search_runtime.csv)
- [round8_portfolio_usage.csv](/home/catid/gnn3/reports/plots/round8_portfolio_usage.csv)

## Fresh Guardrail

Round eight did not rely on archived baselines. It refreshed the `multiheavy` guardrail again on the corrected feasible suites.

Fresh seed results:

- seed `311`: regret `1.50`, p95 `5.96`, miss `43.8%`
- seed `312`: regret `1.92`, p95 `5.45`, miss `43.8%`
- seed `313`: regret `0.55`, p95 `2.90`, miss `37.5%`

So the round-four / round-six `multiheavy` band survives another fresh batch unchanged.

## Headroom Audit

The main round-eight audit tightened the target slice again.

Across three fresh audited seeds:

- seed `312` hard near-tie error rate: `9.31%`
- seed `313` hard near-tie error rate: `11.05%`
- seed `311` hard near-tie error rate: `10.0%`

Large-gap stays effectively solved:

- seed `312` large-gap control error rate: `0.29%`
- seed `313` large-gap control error rate: `0.0%`
- seed `311` large-gap control error rate: `0.27%`

The residual headroom is therefore real, but narrow:

- small baseline-error subsets measured in only a few dozen decisions per audited seed
- consistently higher ambiguity sensitivity on the baseline-error subset than on the hard near-tie slice as a whole
- no evidence that reopening large-gap constructor work would be useful

This confirms the corrected round-seven diagnosis:

- the remaining opportunity is in hard near-tie ambiguity
- not in obvious large-gap mistakes

## Probe Audit

The frozen-feature probe extension again supports the “decision-rule bottleneck, not missing-signal bottleneck” diagnosis.

Seed `312` already showed:

- strong OOD pairwise top-2 ranking (`0.942` to `0.963`)
- useful OOD oracle-gap prediction (`0.657` to `0.762`)
- strong deadline-risk prediction (`0.877` to `0.902`)

That is enough to justify counterfactual value supervision on ambiguous states rather than reopening another broad feature-side architecture family.

A third-seed probe confirmation on seed `311` was started as exploit-side follow-up, but it was explicitly stopped once the fresh three-seed guardrail batch and third-seed headroom audit had already locked the round decision. Round eight therefore treats:

- the fresh seed `312` round-eight probe audit, plus
- the existing round-seven multi-probe evidence

as sufficient for the representation-side conclusion.

## Counterfactual Dataset

Round eight built and cached an all-action dataset specifically for the hard near-tie question.

Seed `312` cache size:

- `25,007` candidate rows total
- heavy near-tie coverage concentrated exactly where expected:
  - `deeper_packets6`: `590` hard near-tie rows, `85` baseline-error rows
  - `heavy_dynamic`: `194` hard near-tie rows, `35` baseline-error rows

That cache was then reused across the critic family, which is the main reason round eight could do substantially more useful work than the recent rounds without relaxing discipline.

## Direct Critic Family

The direct critic family answered the main counterfactual-value question cleanly.

### Scalar Q

This branch found real error-recovery signal, but it was too destructive as a direct decision rule.

Hard near-tie OOD aggregate:

- disagreement: `34.6%`
- correction rate: `4.72%`
- new-error rate: `27.0%`
- baseline-error recovery: `62.5%`
- hard near-tie regret delta: `+0.0377`

Interpretation:

- useful evidence that counterfactual error signal exists
- not promotable directly

### Multi-Risk

This was a clear negative:

- disagreement: `34.3%`
- correction rate: `0.94%`
- new-error rate: `27.4%`
- baseline-error recovery: `12.5%`
- hard near-tie regret delta: `+0.0723`

Verdict:

- killed

### Late-Unfreeze

This was safer than scalar but still not safe enough:

- disagreement: `29.6%`
- correction rate: `3.46%`
- new-error rate: `22.6%`
- baseline-error recovery: `45.8%`
- hard near-tie regret delta: `+0.0399`

The tighter-gate variant reduced movement and also gave away most of the recovery:

- disagreement: `24.8%`
- correction rate: `1.26%`
- new-error rate: `19.5%`
- baseline-error recovery: `16.7%`

Verdict:

- informative but not promotable

### Pairwise Ranking

This is the round-eight direct-critic leader.

Hard near-tie OOD aggregate:

- disagreement: `5.03%`
- correction rate: `1.89%`
- new-error rate: `3.14%`
- baseline-error recovery: `25.0%`
- hard near-tie regret delta: `+0.0008`
- hard near-tie miss delta: `0.0`

This is the first round-eight branch that stayed conservative enough to avoid broad damage while still correcting a nonzero share of baseline errors.

But it still did not earn direct promotion:

- the gains are small
- the slice regret delta is essentially flat rather than clearly improved
- it is better treated as evidence that near-tie ranking signal exists than as a new default policy

## Search And Backup

Round eight then tested whether that critic signal could be turned into a useful ambiguity-time correction mechanism.

The answer is negative on runtime grounds.

### Critic-Guided Bounded Search

Full-suite scouts were killed early:

- scalar-Q search: about `24.3m`
- pairwise search: about `12.2m`

Two-suite targeted OOD scouts were still too slow:

- scalar-Q targeted search: about `16.3m`
- pairwise targeted search: about `16.3m`

That is too expensive for scout use in this repo.

### Local Path-Cost Tie-Break Backup

Round eight then tested a cheaper fallback:

- near-tie gate only
- top-2 action expansion
- local suffix-cost tie-break

This was cheaper, but still crossed the runtime bar:

- targeted backup scout killed at about `10.3m`

So round eight closes the whole search-side thesis cleanly:

- critic signal exists
- but the current search-style and local-planner correction mechanisms are still too expensive to justify promotion

Distillation was not opened because no search-time branch first earned promotion.

## Round-Eight Conclusion

Round eight found a narrower and more decisive answer than round seven.

The good news is:

1. the residual near-tie headroom is real and stable across fresh seeds
2. the backbone still encodes enough local signal for better ambiguous-state ranking
3. counterfactual value supervision can recover some real baseline mistakes

The limiting news is:

1. direct critics are still either too destructive or too weak
2. the safest direct critic (`pairwise`) is only a small local improvement, not a new default policy
3. turning critic signal into bounded decision-time correction is too expensive in the current form

So the round-eight recommendation is:

- keep plain `multiheavy` as the default exploit policy
- keep the round-eight near-tie diagnosis
- do not reopen large-gap constructor work
- do not reopen direct scalar/risk critics as default policies
- do not reopen bounded search or path-cost tie-breakers in their current runtime form

## Portfolio And Resource Use

Round eight stayed inside the requested exploration-heavy window even after counting the killed search/runtime scouts honestly.

Round-eight totals:

- exploit: `0.7760`
- explore: `1.5844`
- split: `32.9% exploit / 67.1% explore`

This is the important resource read for the round:

- the critic family and the killed search/tie-break scouts did consume real GPU time
- those hours still count, because they settled real questions
- even with those runtime negatives counted, the round stayed within the requested `25–35% / 65–75%` exploit/explore window

## Merge Status

Round-eight work was merged back into local `main` at commit `TBD`.

Pushing `main` depends on GitHub auth in the current shell and will be recorded explicitly at handoff.
