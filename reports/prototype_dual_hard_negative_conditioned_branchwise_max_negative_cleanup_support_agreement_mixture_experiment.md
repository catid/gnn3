# Prototype Dual-Only Hard-Negative-Conditioned Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the accepted branchwise-max negative-cleanup prototype family
improves if hard-negative-conditioned bank editing is restricted to the dual
negative bank while the shared negative bank stays fully intact.

This is the direct follow-up to the failed full hard-negative-conditioned bank
editing experiment, which always collapsed the shared negative bank to `2/8`
kept prototypes and then failed badly on held-out.

The design goal was:

- keep the accepted support-weighted agreement-mixture geometry
- keep the accepted shared plus dual branch decomposition
- keep the accepted branch-local fixed-vs-sharp cleanup
- keep the shared negative bank untouched
- search only the dual negative keep count using train-pack utility against
  stable positives and harmful negatives

## Implementation

- New runner:
  `scripts/run_prototype_dual_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Reused head:
  `HardNegativeConditionedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- Variants:
  - `prototype_dual_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_dual_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Search procedure:

- train the accepted branchwise-max weights once
- freeze weights
- keep the shared negative bank at `8 / 8`
- search only the dual negative keep count from `2` through `8`

## Selected keep counts

The search did **not** choose any bank edit.

### Plain branch

- shared negatives kept: `8 / 8`
- dual negatives kept: `8 / 8`

### Hybrid branch

- shared negatives kept: `8 / 8`
- dual negatives kept: `8 / 8`

So the selected policy is just the all-keep baseline.

## Held-Out Result

### `prototype_dual_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix`

Closed weak positive, but not meaningful as a new architecture.

Because the search selected the all-keep baseline, the plain branch is just a
fresh rerun of the accepted plain geometry. It recovered only `16.7%` of
held-out `stable_positive_v2` and only reached the weaker `90.47%` hard
near-tie band.

### `prototype_dual_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed as a null-edit control, not a promoted new branch.

Held-out curve:

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.45%`
- overall target match `96.58%`
- overall mean delta regret `-0.0051`

At `0.50%` nominal budget:

- overall coverage `0.36%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.53%`
- overall target match `96.65%`
- overall mean delta regret `-0.0082`

At `0.75%` nominal budget:

- overall coverage `0.48%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.60%`
- overall target match `96.69%`
- overall mean delta regret `-0.0098`

At `1.00%` nominal budget:

- overall coverage `0.61%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.68%`
- overall target match `96.74%`
- overall mean delta regret `-0.0115`

At `1.50%` nominal budget:

- overall coverage `0.86%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match still `90.68%`
- overall target match `96.83%`
- overall mean delta regret `-0.0150`

At `2.00%` nominal budget:

- overall coverage `0.98%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match still `90.68%`
- overall target match `96.86%`
- overall mean delta regret `-0.0158`

Those are real held-out numbers, but the selected keep count is `8 / 8`, so
they are not evidence for the proposed dual-only bank edit. They are just a
fresh rerun of the accepted branchwise-max family that happened to land on a
stronger point.

## Interpretation

This experiment does **not** validate dual-only hard-negative-conditioned bank
editing.

It validates a narrower statement instead:

- when the shared bank is forced to stay intact
- and the dual bank is allowed to be edited
- the search chooses **not** to edit the dual bank either

So the actual conclusion is:

- the searched dual-only hard-negative-conditioned edit is a no-op
- any strength in the held-out curve should be attributed to branchwise-max
  rerun variance, not to the proposed bank-editing architecture

This is still useful, because it closes the remaining simple train-pack
utility-ranked bank-editing path:

- full bank editing overprunes the shared bank and fails
- dual-only bank editing declines to change the bank at all

If bank editing reopens again, require:

- a selected mask materially different from the all-keep baseline
- and rerun-stable gains after that edit

Otherwise it is just a disguised robustness rerun of the accepted branch.

## Decision

Close:

- `prototype_dual_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_dual_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Do **not** promote this as a new architecture lead.

Keep the accepted live reference unchanged:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Treat the stronger held-out point only as a robustness datapoint for the
existing branchwise-max family, not as evidence for dual-only bank editing.
