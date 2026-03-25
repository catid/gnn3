# Counterfactual Critic Round 8

## Scope

Round eight tested cached all-action critics on the fresh seed `312` counterfactual dataset.

Variants run:

1. frozen-trunk scalar Q
2. frozen-trunk multi-risk critic
3. late-unfreeze adapter critic
4. late-unfreeze adapter critic with a tighter gate
5. pairwise near-tie ranking critic

Artifacts:

- [round8_critic_scalar_q_seed312](/home/catid/gnn3/artifacts/round8_critic_scalar_q_seed312)
- [round8_critic_risk_multi_seed312](/home/catid/gnn3/artifacts/round8_critic_risk_multi_seed312)
- [round8_critic_late_unfreeze_seed312](/home/catid/gnn3/artifacts/round8_critic_late_unfreeze_seed312)
- [round8_critic_late_unfreeze_gate15_seed312](/home/catid/gnn3/artifacts/round8_critic_late_unfreeze_gate15_seed312)
- [round8_critic_pairwise_seed312](/home/catid/gnn3/artifacts/round8_critic_pairwise_seed312)

## Main Result

The critic family did find real decision signal on the hard near-tie error subset, but most direct critics were too destructive when used as immediate policy replacements.

The main split is:

- scalar and late-unfreeze critics do correct many baseline errors on `deeper_packets6`
- but they also introduce too many new errors on hard near-tie states
- the multi-risk critic is clearly worse than scalar
- the pairwise critic is the first variant with low disagreement, low harm, and nonzero error recovery

That makes pairwise ranking the direct-critic leader, and it makes bounded search on top of critic signal worth testing.

## Variant Summary

### Scalar Q

This is the highest-recall critic on the error subset, but it is too aggressive as a direct replacement.

Aggregate hard near-tie slice:

- disagreement: `30.7%`
- candidate target-match: `0.735`
- correction rate: `4.19%`
- new-error rate: `24.0%`
- mean regret delta: `+0.033`

Hard near-tie error subset:

- recovery: `62.5%`
- error regret delta: `+0.036`

Suite behavior:

- `deeper_packets6` hard near-tie errors: `70.6%` recovery, but large new-error spillover
- `heavy_dynamic` hard near-tie errors: `42.9%` recovery, but positive regret and miss deltas

Verdict:

- not promotable as a direct decision rule
- still useful as evidence that counterfactual error signal exists

### Multi-Risk

This branch is a clear negative.

Aggregate hard near-tie slice:

- disagreement: `30.4%`
- candidate target-match: `0.698`
- correction rate: `0.84%`
- new-error rate: `24.3%`
- mean regret delta: `+0.064`

Hard near-tie error subset:

- recovery: `12.5%`
- error regret delta: `+0.159`

Verdict:

- killed

### Late-Unfreeze Adapter

This was safer than scalar but still too harmful overall.

Aggregate hard near-tie slice:

- disagreement: `26.3%`
- candidate target-match: `0.763`
- correction rate: `3.07%`
- new-error rate: `20.1%`
- mean regret delta: `+0.035`

Hard near-tie error subset:

- recovery: `45.8%`
- error regret delta: `+0.071`

On `deeper_packets6` specifically:

- hard near-tie disagreement: `36.8%`
- hard error recovery: `64.7%`

Verdict:

- interesting signal, but still not safe enough as a direct policy

### Late-Unfreeze With Tighter Gate

Tightening the gate reduced harm but also gave away most of the recovery.

Aggregate hard near-tie slice:

- disagreement: `22.1%`
- candidate target-match: `0.771`
- correction rate: `1.12%`
- new-error rate: `17.3%`
- mean regret delta: `+0.035`

Hard near-tie error subset:

- recovery: `16.7%`
- error regret delta: `+0.059`

Verdict:

- not enough
- tighter gating alone does not rescue the late-unfreeze critic

### Pairwise Ranking

This is the current direct-critic leader.

Aggregate hard near-tie slice:

- disagreement: `4.47%`
- candidate target-match: `0.922`
- correction rate: `1.68%`
- new-error rate: `2.79%`
- mean regret delta: `+0.0007`

Hard near-tie error subset:

- recovery: `25.0%`
- error regret delta: `-0.0156`

Suite-level highlights:

- `deeper_packets6` hard near-tie errors: `17.6%` recovery, negative error regret delta
- `heavy_dynamic` hard near-tie errors: `42.9%` recovery, negative error regret delta
- `heavy_dynamic` hard near-tie aggregate slice: target-match improves from `0.934` to `0.953`

Verdict:

- first direct critic with low disagreement and low harm
- small but real near-tie decision improvement
- worth using as one of the bounded-search bases

## Decision

The direct-critic family is not the final constructor.

But it did settle two important questions:

1. counterfactual value supervision can produce real correction signal on the hard near-tie error subset
2. the best direct path so far is a conservative pairwise critic, not a broad risk head

That is enough to justify round-eight bounded-search scouts built on:

- scalar Q, to test whether search can harvest its high-recall signal more safely
- pairwise ranking, to test whether a safer direct critic becomes genuinely helpful with bounded expansion

