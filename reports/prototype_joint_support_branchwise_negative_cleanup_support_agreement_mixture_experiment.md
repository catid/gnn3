# Prototype Joint-Support Branchwise Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the downside of branchwise max comes from fixed-cleanup takeovers
that happen on only one branch at a time.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the branch-strength sharp negative-tail cleanup as the base
- keep the older fixed negative-tail cleanup as the secondary source
- keep fusion inside the shared and dual branches
- but only allow the fixed-cleanup gain that is **jointly supported by both
  branches**
- suppress one-branch takeovers below `1%` coverage while preserving the
  higher-budget branchwise-max upside

This is the direct follow-up to the positive branchwise-max result and the
closed branchwise-margin result.

## Implementation

- New head:
  `JointSupportBranchwiseNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_joint_support_branchwise_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_joint_support_branchwise_negative_cleanup_support_agree_mix`
  - `prototype_joint_support_branchwise_negative_cleanup_support_agree_mix_hybrid`

Relative to branchwise max:

- fixed and branch-strength sharp scores are computed separately in the shared
  and dual branches
- each branch gain is `relu(fixed - sharp)`
- the actual branch lift is only the shared common gain:
  `min(shared_gain, dual_gain)`
- then the usual agreement mixture combines the lifted branch scores

So this is the strictest “both branches must support the fixed cleanup” version
of the branchwise fusion idea.

## Held-Out Result

### `prototype_joint_support_branchwise_negative_cleanup_support_agree_mix`

Closed, dead.

At every budget through `1.50%` nominal:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie unchanged at `90.53%`
- overall mean delta regret `0.0000`

At `2.00%` nominal budget it only found a tiny niche:

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall mean delta regret `-0.0008`

So the plain branch is inert.

### `prototype_joint_support_branchwise_negative_cleanup_support_agree_mix_hybrid`

Closed weak positive.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall mean delta regret `-0.0053`

At `0.25%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- overall mean delta regret `-0.0066`

At `0.50%` nominal budget:

- overall coverage `0.43%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- overall mean delta regret `-0.0077`

At `0.75%` nominal budget:

- overall coverage `0.55%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- overall mean delta regret `-0.0077`

At `1.00%` through `2.00%` nominal budgets:

- overall coverage saturates at `0.68%`
- held-out `stable_positive_v2` recovery stays at `25%`
- hard near-tie stays at `90.53% -> 90.60%`
- overall mean delta regret stays at `-0.0077`

So the hybrid finds only the tiny ultra-low-coverage niche and then fully
saturates. It never approaches the real matched-band frontier.

## Comparison against nearby variants

### Versus the live sharp-negative branch

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_joint_support_branchwise_negative_cleanup_support_agree_mix_hybrid @ 0.75%`

- held-out `stable_positive_v2` recovery `25%`
- hard near-tie `90.53% -> 90.60%`
- overall mean delta regret `-0.0077`

So joint-support fusion is far too strict to preserve the sharp-negative lane.

### Versus branchwise max

`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0145`

`prototype_joint_support_branchwise_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- overall coverage only `0.68%`
- held-out `stable_positive_v2` recovery only `25%`
- hard near-tie only `90.53% -> 90.60%`
- overall mean delta regret only `-0.0077`

So requiring a shared common gain from both branches removes almost all of the
useful branchwise-max signal.

### Versus branchwise margin-max

`prototype_branchwise_margin_max_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- held-out `stable_positive_v2` recovery `75%`
- hard near-tie `90.53% -> 90.73%`
- overall mean delta regret `-0.0152`

`prototype_joint_support_branchwise_negative_cleanup_support_agree_mix_hybrid @ 1.00%`

- only `25%` held-out `stable_positive_v2`
- only `90.53% -> 90.60%`
- overall mean delta regret `-0.0077`

So even the already-closed branchwise-margin follow-up is much better than this
intersection-style joint-support fusion.

## Interpretation

This closes the strongest remaining “maybe branchwise max is only failing
because one branch fires alone” hypothesis.

Current read:

- yes, branch-local fusion was the important structural correction
- but requiring both branches to contribute the same fixed-cleanup gain is too
  conservative
- the useful fixed-cleanup recall signal is not restricted to states where both
  branches improve together by the same amount

So the correct lesson from the branchwise-max result is:

- branch-local fusion is valuable
- but full joint-support intersection throws away too much of the fixed branch
- hard branchwise max still remains the best live fusion for this family

## Decision

Close:

- `prototype_joint_support_branchwise_negative_cleanup_support_agree_mix`
- `prototype_joint_support_branchwise_negative_cleanup_support_agree_mix_hybrid`

Live shortlist remains:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best
  sub-`1%` full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out
  recall around `1%`
- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` for the
  higher-budget matched-band and higher-budget max-recall lane
