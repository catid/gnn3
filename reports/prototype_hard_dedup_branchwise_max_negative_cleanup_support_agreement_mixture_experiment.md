# Prototype Hard-Dedup Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the accepted branchwise-max negative-cleanup prototype family
improves if the negative banks are explicitly edited before pooling rather than
softly reweighted.

The design goal was:

- keep the accepted support-weighted agreement-mixture geometry
- keep the accepted fixed-vs-sharp branch-local negative cleanup
- keep the accepted hard branchwise max before the outer agreement mix
- replace soft pruning with explicit bank surgery:
  - rank negative prototypes by learned support
  - greedily keep only support-ranked unique negatives
  - hard-mask the dropped negatives before `logsumexp`

This is the direct follow-up to the failed soft keep-mask pruning experiment.

## Implementation

- New head:
  `HardDedupBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_hard_dedup_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_hard_dedup_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_hard_dedup_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

The final checked-in run uses a negative-bank cosine dedup threshold of `0.2`.

## Threshold sanity check

The first threshold pass mattered for interpretation:

- `0.5` kept all `8/8` negatives in both shared and dual banks
- so that setting was effectively just a rerun of the accepted reference
- `0.2` was the first setting that actually edited the banks:
  - plain branch: shared `5/8`, dual `4/8`
  - hybrid branch: shared `5/8`, dual `5/8`

So the official result below is the `0.2` run, because it is the actual
hard-dedup test.

## Held-Out Result

### `prototype_hard_dedup_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall mean delta regret `0.0000`

So the plain branch is closed immediately.

### `prototype_hard_dedup_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed weak positive, but not promotable.

Official held-out run with active dedup:

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
- overall target match `96.61%`
- overall mean delta regret `-0.0077`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.53%`
- overall target match `96.75%`
- overall mean delta regret `-0.0133`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery only `66.7%`
- hard near-tie target match only `90.53%`
- overall target match `96.80%`
- overall mean delta regret `-0.0149`

At `1.50%` nominal budget:

- overall coverage `1.52%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.60%`
- overall target match `96.83%`
- overall mean delta regret `-0.0157`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery only `83.3%`
- hard near-tie target match only `90.60%`
- overall target match `96.86%`
- overall mean delta regret `-0.0162`

So the branch does get a tiny aggregate-quality gain at the very lowest budget,
but it gives back the accepted `1.0%` and `2.0%` frontier positions.

## Comparison against the accepted branchwise-max reference

Accepted reference:
`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

At `0.25%`:

- same held-out `stable_positive_v2` recovery `50%`
- same hard near-tie target match `90.45%`
- reference overall mean delta regret `-0.0074`
- hard-dedup `-0.0077`

So there is a tiny micro-budget aggregate gain.

At `1.00%`:

- reference held-out `stable_positive_v2` recovery `83.3%`
- hard-dedup `66.7%`
- reference hard near-tie target match `90.60%`
- hard-dedup `90.53%`
- reference overall mean delta regret `-0.0145`
- hard-dedup `-0.0149`

So the aggregate regret is slightly better, but only by giving back the actual
target-slice frontier.

At `2.00%`:

- reference held-out `stable_positive_v2` recovery `100%`
- hard-dedup `83.3%`
- reference hard near-tie target match `90.68%`
- hard-dedup `90.60%`
- reference overall mean delta regret `-0.0167`
- hard-dedup `-0.0162`

So the accepted branch still clearly wins the higher-budget operating region.

## Interpretation

This experiment answers the open pruning question cleanly.

What changed:

- bank surgery did activate
- the shared and dual negative banks were really reduced from `8` negatives to
  about `5`

What that bought:

- a tiny micro-budget aggregate-regret gain

What it cost:

- the `1.0%` matched-band frontier
- the `2.0%` high-recall frontier

So the conclusion is:

- explicit hard negative-bank deduplication is more meaningful than the older
  soft keep-mask pruning
- but static support-ranked cosine thresholding is still the wrong bank-editing
  rule for the live branchwise-max family

If bank editing reopens again, it should not be another static similarity
threshold. It should be:

- offline bank reconstruction
- teacher-guided prototype rebuild
- or deduplication conditioned on the hard-negative pack rather than only bank
  geometry

## Decision

Close:

- `prototype_hard_dedup_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_hard_dedup_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Keep the accepted live reference unchanged:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

No promotion.
