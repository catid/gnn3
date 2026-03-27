# Round 13 Hierarchical Defer / Dispatcher

## Scope

Wave D tested whether the live prototype family works better as a composite frontier than as separate operating points.

Artifacts:

- `reports/plots/round13_hierarchical_dispatcher_summary.csv`
- `reports/plots/round13_hierarchical_dispatcher_comparison.csv`
- `reports/plots/round13_hierarchical_dispatcher_static_ladder.csv`
- `reports/plots/round13_hierarchical_dispatcher_family_mix.csv`
- `reports/plots/round13_hierarchical_dispatcher_comparison.png`

The analysis used three sources:

- archived shortlist exports
- rerun 1 shortlist exports
- rerun 2 shortlist exports

## D1. Static Budget Ladder

The static ladder was defined by the matched-budget frontier leaders.

Archived ladder:

- `0.10%`: branchwise-max
- `0.25%` to `0.50%`: memory-agree
- `0.75%`: sharp-negative
- `1.00%+`: negative-tail

Fresh reruns did not preserve that ladder:

- rerun 1: branchwise-max leads every budget
- rerun 2: branchwise-max leads every budget

Interpretation:

- the archived piecewise ladder is not robust enough to act as a promoted deployment policy
- once rerun stability is required, the ladder collapses to one branch

## D2. Score-Band Dispatcher

I tested a monotone score-band dispatcher that:

- keeps branch-specific reference bands
  - memory-agree `0.25%`
  - sharp-negative `0.75%`
  - negative-tail `1.00%`
  - branchwise-max `1.50%`
- computes a per-state margin above each family’s reference threshold
- applies family-priority ordering so lower-budget lanes win only when they are clearly active
- uses a single monotone score for matched-budget evaluation

## D3. Composite vs Best Single

The dispatcher does **not** beat the best single branch.

### Archived

At low budgets it only ties or slightly helps aggregate regret:

- `0.10%`: effectively tied on Tier 1 and Tier 2
- `0.25%` and `0.50%`: ties Tier 1 and Tier 2 but is slightly worse on overall mean delta regret

At the important archived frontier bands it is worse:

- `0.75%`
  - dispatcher improves recall `75.0% -> 83.3%`
  - but hard near-tie drops `90.73% -> 90.60%`
  - overall mean delta regret worsens from `-0.0144` to `-0.0117`
- `1.00%`
  - dispatcher drops recall `100.0% -> 83.3%`
  - hard near-tie drops `90.80% -> 90.60%`
  - overall mean delta regret only slightly improves
- `1.25%+`
  - dispatcher never restores the archived negative-tail hard-slice target match

### Rerun 1

The dispatcher still fails promotion, even though it picks up some low-budget recall:

- `0.25%` to `0.75%`
  - it improves recall from `50.0%` to `66.7%`
  - it improves hard near-tie target match from `90.45%` to `90.53%`
  - but it is still worse on overall mean delta regret by `0.0013` to `0.0030`
- `1.00%`
  - it only ties Tier 1 and Tier 2
  - it is still worse on overall mean delta regret by `0.0042`
- `1.25%` to `2.50%`
  - it falls from `83.3%` recall to `66.7%`
  - it gives up `0.0008` hard near-tie target match
  - it remains worse on overall mean delta regret at every budget

So rerun 1 still rejects the dispatcher on matched-budget quality.

### Rerun 2

The dispatcher again loses:

- `0.10%` to `0.25%`
  - it collapses from `50.0%` recall to `0%`
  - it gives up `0.0006` hard near-tie target match
- `0.50%`
  - it improves recall from `50.0%` to `66.7%`
  - it improves hard near-tie target match from `90.45%` to `90.53%`
  - but it is still worse on overall mean delta regret by `0.0016`
- `0.75%` to `1.00%`
  - it only ties Tier 1 and Tier 2
  - it is still worse on overall mean delta regret by `0.0033` to `0.0037`
- `1.25%` to `2.50%`
  - it ties Tier 1 and Tier 2
  - it is still worse on overall mean delta regret at every matched budget

## D4. Family Mix

The dispatcher family mix shows why it fails.

Archived:

- the selected set is dominated by negative-tail plus sharp-negative through the middle budgets
- branchwise-max only becomes a large share at higher coverage
- the policy blends families instead of preserving the exact strongest single-branch operating point

Rerun 1 and rerun 2:

- the same score-band logic over-allocates to fragile archived lanes
- the robust branchwise-max lane is diluted instead of preserved

## Promotion Decision

Wave D is a **no-go** for a composite policy.

Keep:

- the family as separate operating points for analysis
- branchwise-max as the robust single-branch production candidate

Do not keep:

- a static budget ladder as a promoted policy
- a score-band dispatcher as a promoted policy
- any unified composite that depends on archived-only operating regions

## Round-13 Dispatcher Conclusion

The prototype family does **not** become stronger when unified.

The correct round-13 conclusion is:

- keep separate mechanistic branches for analysis
- keep only `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` as the robust promoted branch
- do not promote a hierarchical dispatcher
