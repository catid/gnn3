# Prototype Pruned Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the accepted branchwise-max negative-cleanup prototype family
becomes more robust if each shared and dual prototype bank gets a
suppression-only keep mask before `logsumexp` pooling.

The design goal was:

- keep the accepted support-weighted agreement-mixture geometry
- keep the accepted branch-local fixed-vs-sharp negative cleanup
- keep the accepted hard branchwise max before the outer agreement mix
- add only one new mechanism:
  - per-bank learned keep probabilities
  - implemented as additive `log(keep)` penalties, so they can only suppress
    prototypes and never amplify them
- encourage pruning lightly with a mean-keep regularizer instead of another
  outer routing or lift path

This is the direct round-13 follow-up to the accepted “prototype pruning /
deduplication inside branchwise-max” recommendation.

## Implementation

- New head:
  `PrunedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_pruned_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_pruned_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_pruned_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

The runner was kept intentionally close to the live branchwise-max runner.
The only new knob that mattered was the keep regularization weight; the final
checked-in run uses `0.01`.

## Held-Out Result

### `prototype_pruned_branchwise_max_negative_cleanup_support_agree_mix`

Closed. Effectively dead.

At every budget through `1.5%`:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall mean delta regret `0.0000`

At `2.0%` it became slightly harmful:

- overall target match `96.51% -> 96.50%`
- overall mean delta regret `+0.0010`

So the plain branch should be treated as closed.

### `prototype_pruned_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed weak positive, but not promotable.

Official held-out run:

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.45%`
- overall target match `96.59%`
- overall mean delta regret `-0.0064`

At `0.25%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match still `90.45%`
- overall target match `96.63%`
- overall mean delta regret `-0.0083`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.53%`
- overall target match `96.74%`
- overall mean delta regret `-0.0131`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.53%`
- overall target match `96.78%`
- overall mean delta regret `-0.0141`

At `1.50–2.00%` nominal budget:

- coverage saturates early at only `1.38–1.62%`
- held-out `stable_positive_v2` recovery stays stuck at `66.7%`
- hard near-tie target match stays stuck at `90.53%`
- overall mean delta regret plateaus at `-0.0145`

So the official run shows only a tiny ultra-low-coverage aggregate-quality
improvement. It gives back the accepted branchwise-max frontier from `1.0%`
upward.

## What actually pruned

The learned keep masks barely moved.

Official hybrid keep means:

- shared positive `0.979`
- shared negative `0.980`
- dual positive `0.981`
- dual negative `0.982`

So this did not produce a meaningful bank simplification. It mostly learned a
tiny global attenuation.

## Same-config rerun

The stronger keep-regularization setting was rerun once more at the same
configuration before closing the branch.

That rerun briefly looked better:

- at `1.00%` it recovered `83.3%` of held-out `stable_positive_v2`
- matched the accepted hard near-tie band at `90.60%`
- improved overall mean delta regret to `-0.0152`

But that improvement did not reproduce on the clean rerun above, which fell
back to:

- only `66.7%` held-out `stable_positive_v2` recovery
- only `90.53%` hard near-tie target match
- overall mean delta regret `-0.0141`

That makes the branch unstable even before it is competitive.

## Comparison against the accepted branchwise-max reference

Accepted reference:
`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

At `0.25%`:

- reference overall mean delta regret `-0.0074`
- pruned variant overall mean delta regret `-0.0083`
- same held-out `stable_positive_v2` recovery `50%`
- same hard near-tie target match `90.45%`

So there is a tiny micro-budget aggregate gain.

At `1.00%`:

- reference held-out `stable_positive_v2` recovery `83.3%`
- pruned variant `66.7%`
- reference hard near-tie target match `90.60%`
- pruned variant `90.53%`
- reference overall mean delta regret `-0.0145`
- pruned variant `-0.0141`

At `2.00%`:

- reference held-out `stable_positive_v2` recovery `100%`
- pruned variant `66.7%`
- reference hard near-tie target match `90.68%`
- pruned variant `90.53%`
- reference overall mean delta regret `-0.0167`
- pruned variant `-0.0145`

So the pruning path fails exactly where the accepted branch is valuable:
budget growth no longer buys back sparse-positive recall or hard near-tie
quality.

## Interpretation

The useful conclusion is narrower than “pruning failed.”

What failed was:

- soft, suppression-only keep masks
- applied globally to the existing banks
- with no actual hard prototype removal or deduplication step

That mechanism mostly learned keep probabilities around `0.98`, so it behaved
like a tiny score attenuation. The single stronger rerun that briefly looked
promising was not reproducible.

So the accepted round-13 recommendation should be refined:

- do not spend another cycle on soft keep-mask pruning
- if pruning reopens, it should be explicit bank surgery:
  - hard prototype deduplication
  - explicit prototype dropping
  - or a bank reconstruction step that materially changes which prototypes
    exist

## Decision

Close:

- `prototype_pruned_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_pruned_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Keep the accepted live prototype reference unchanged:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

No promotion.
