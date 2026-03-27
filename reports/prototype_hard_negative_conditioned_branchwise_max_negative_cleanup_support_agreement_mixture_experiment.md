# Prototype Hard-Negative-Conditioned Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the accepted branchwise-max negative-cleanup prototype family
improves if the negative banks are edited using train-pack utility against
stable positives and harmful negatives rather than soft masks, cosine dedup, or
scalar margin tuning.

This is the direct follow-up to the failed soft pruning, hard dedup, learned
margin, and searched fixed-margin experiments.

The design goal was:

- keep the accepted support-weighted agreement-mixture geometry
- keep the accepted branch-local fixed-vs-sharp cleanup
- keep the accepted hard branchwise max before the outer agreement mix
- train the accepted branchwise-max weights exactly as before
- then edit only the negative banks:
  - rank negative prototypes by
    `harmful-negative logit support - stable-positive logit suppression`
  - search shared and dual keep counts on the training pack
  - apply the selected hard masks at inference

## Implementation

- New head:
  `HardNegativeConditionedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Search procedure:

- train the accepted branchwise-max weights once
- freeze weights
- compute per-prototype utility on the training pack
- search shared and dual kept-negative counts from `2` through `8`
- rank search candidates on:
  - stable-positive recall
  - hard near-tie target match
  - hard near-tie mean delta regret
  - overall mean delta regret
  - harmful and false-positive rates

## Selected keep counts

The search always selected a very aggressive shared-bank cut:

### Plain branch

- shared negatives kept: `2 / 8`
- dual negatives kept: `8 / 8`

### Hybrid branch

- shared negatives kept: `2 / 8`
- dual negatives kept: `8 / 8`

So the search conclusion was effectively:

- heavily collapse the shared negative bank
- leave the dual negative bank untouched

## Held-Out Result

### `prototype_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall mean delta regret `0.0000`

So the plain branch closes immediately.

### `prototype_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed weak positive, but clearly not promotable.

At `0.10%` nominal budget:

- overall coverage `0.07%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match `96.53%`
- overall mean delta regret `-0.0020`

At `0.50%` nominal budget:

- overall coverage `0.27%`
- held-out `stable_positive_v2` recovery `16.7%`
- hard near-tie target match `90.47%`
- overall target match `96.66%`
- overall mean delta regret `-0.0085`

At `1.00%` nominal budget:

- overall coverage `0.52%`
- held-out `stable_positive_v2` recovery still only `16.7%`
- hard near-tie target match still only `90.47%`
- overall target match `96.74%`
- overall mean delta regret `-0.0123`

At `1.50%` nominal budget:

- overall coverage `0.77%`
- held-out `stable_positive_v2` recovery still only `16.7%`
- hard near-tie target match still `90.47%`
- overall target match `96.78%`
- overall mean delta regret `-0.0131`

At `2.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery only `33.3%`
- hard near-tie target match only `90.54%`
- overall target match `96.82%`
- overall mean delta regret `-0.0143`

So the hybrid is not just weaker than the accepted reference. It collapses to a
low-coverage, low-recall partial fix.

## Comparison against the accepted branchwise-max reference

Accepted reference:
`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

At `1.00%`:

- reference held-out `stable_positive_v2` recovery `83.3%`
- hard-negative-conditioned `16.7%`
- reference hard near-tie target match `90.60%`
- hard-negative-conditioned `90.47%`
- reference overall mean delta regret `-0.0145`
- hard-negative-conditioned `-0.0123`

At `2.00%`:

- reference held-out `stable_positive_v2` recovery `100%`
- hard-negative-conditioned `33.3%`
- reference hard near-tie target match `90.68%`
- hard-negative-conditioned `90.54%`
- reference overall mean delta regret `-0.0167`
- hard-negative-conditioned `-0.0143`

So the accepted branch still wins comfortably on both the stable-positive pack
and the broad hard near-tie frontier.

## Interpretation

This closes the simple hard-negative-conditioned bank-editing idea.

What the training search learned:

- collapse the shared negative bank from `8` to `2`
- leave the dual negative bank alone

What that means:

- the shared bank utility score is overfitting the tiny train positive/harmful
  packs
- the search is rewarding a bank edit that looks good on the narrow train
  metric stack but does not survive held-out evaluation

So the conclusion is:

- train-pack utility-ranked hard bank editing is not a real improvement path
  for the live branchwise-max family
- it is a stronger overfit failure than the scalar-margin variants
- the accepted branchwise-max family still appears to need its broad shared
  negative bank for held-out robustness

If bank editing reopens again, it should not be:

- static geometry dedup
- soft keep masks
- scalar margin tuning
- or simple train-pack utility-ranked hard masking

It would need something richer, such as:

- teacher-guided offline bank rebuild
- rerun-stable outer validation across seeds
- or explicit pack-aware reconstruction rather than train-only ranking

## Decision

Close:

- `prototype_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Keep the accepted live reference unchanged:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

No promotion.
