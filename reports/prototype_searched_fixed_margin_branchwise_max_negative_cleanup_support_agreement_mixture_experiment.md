# Prototype Searched Fixed-Margin Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the accepted branchwise-max negative-cleanup prototype family
improves if the shared/dual fixed/sharp cleanup margins are chosen explicitly by
offline search on the training stable-positive and harmful-negative packs
instead of being shared globally or learned end-to-end.

This is the direct follow-up to the failed learned tail-margin calibration
experiment.

The design goal was:

- keep the accepted support-weighted agreement-mixture geometry
- keep the accepted shared plus dual branch decomposition
- keep the accepted hard branchwise max before the outer agreement mix
- keep the accepted trained prototype banks unchanged
- only replace the single global cleanup margin with four fixed searched values:
  - shared fixed margin
  - shared sharp margin
  - dual fixed margin
  - dual sharp margin

## Implementation

- New head:
  `FixedTailMarginBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_searched_fixed_margin_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_searched_fixed_margin_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_searched_fixed_margin_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Search procedure:

- train the accepted branchwise-max weights once
- freeze weights
- sweep a `5^4 = 625` fixed-margin grid over:
  - `0.2`
  - `0.35`
  - `0.5`
  - `0.65`
  - `0.8`
- rank each configuration on the training pack using the accepted narrow and
  broad surfaces:
  - stable-positive recall
  - hard near-tie target match
  - hard near-tie mean delta regret
  - overall mean delta regret
  - harmful and false-positive rates

## Selected margins

The search chose:

### Plain branch

- shared fixed margin `0.2`
- shared sharp margin `0.2`
- dual fixed margin `0.65`
- dual sharp margin `0.2`

### Hybrid branch

- shared fixed margin `0.2`
- shared sharp margin `0.2`
- dual fixed margin `0.5`
- dual sharp margin `0.2`

So the training search pushed the shared cleanup margins all the way to the
most aggressive grid floor.

## Held-Out Result

### `prototype_searched_fixed_margin_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall mean delta regret `0.0000`

So the plain branch closes immediately.

### `prototype_searched_fixed_margin_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed negative, worse than the accepted reference at every meaningful budget.

At `0.10%` nominal budget:

- overall coverage `0.06%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match `96.56%`
- overall mean delta regret `-0.0034`

At `0.50%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery still `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match `96.59%`
- overall mean delta regret `-0.0052`

At `1.00%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery only `33.3%`
- hard near-tie target match only `90.54%`
- overall target match `96.65%`
- overall mean delta regret `-0.0072`

At `1.50%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery still only `33.3%`
- hard near-tie target match still only `90.54%`
- overall target match `96.66%`
- overall mean delta regret `-0.0078`

At `2.00%` nominal budget:

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery still only `33.3%`
- hard near-tie target match still only `90.54%`
- overall target match `96.68%`
- overall mean delta regret `-0.0087`

So the searched fixed-margin hybrid is not a weak positive. It is a clear
frontier regression.

## Comparison against the accepted branchwise-max reference

Accepted reference:
`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

At `1.00%`:

- reference held-out `stable_positive_v2` recovery `83.3%`
- searched fixed-margin `33.3%`
- reference hard near-tie target match `90.60%`
- searched fixed-margin `90.54%`
- reference overall mean delta regret `-0.0145`
- searched fixed-margin `-0.0072`

At `2.00%`:

- reference held-out `stable_positive_v2` recovery `100%`
- searched fixed-margin `33.3%`
- reference hard near-tie target match `90.68%`
- searched fixed-margin `90.54%`
- reference overall mean delta regret `-0.0167`
- searched fixed-margin `-0.0087`

So the accepted branch still wins by a wide margin on both the Tier-1 stable
positive pack and the Tier-2 hard near-tie frontier.

## Interpretation

This closes the simple fixed-threshold search path cleanly.

What happened on train:

- the search found apparently strong aggressive shared margins
- top train configurations recovered more than `82%` stable-positive recall on
  the training pack

What happened on held-out:

- those searched thresholds collapsed to only `33.3%` held-out recall
- hard near-tie quality fell back toward the weaker `90.53%` band
- aggregate regret also degraded sharply

So the search did not discover a real operating point. It overfit the tiny
training positive pack and the narrow hard-negative pack.

The conclusion is:

- simple offline fixed branch/path-specific margin search is not a valid
  improvement path for the live branchwise-max family
- learned free margin deltas were too weak and inert
- offline searched fixed margins are the opposite failure mode: too aggressive
  on train and not stable on held-out

If cleanup tuning reopens again, it should not be another scalar-margin path.
It would need:

- explicit rerun-stable outer validation
- hard-negative-conditioned calibration beyond four scalar margins
- or a richer offline bank rebuild rather than threshold search

## Decision

Close:

- `prototype_searched_fixed_margin_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_searched_fixed_margin_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Keep the accepted live reference unchanged:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

No promotion.
