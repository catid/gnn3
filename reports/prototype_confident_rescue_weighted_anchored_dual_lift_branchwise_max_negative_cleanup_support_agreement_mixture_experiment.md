# Prototype Confident Rescue-Weighted Anchored Dual-Lift Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the rescue-weighted anchored dual-lift follow-up improves if the
extra lift is additionally gated by anchor confidence.

The design goal was:

- keep the accepted branchwise-max score as the anchor
- keep the rescue-weighted dual fixed-rescue scaling
- preserve the micro-budget `0.10–0.25%` win from the rescue-weighted variant
- force the extra lift to decay automatically before the broader `0.50%+`
  regime where accepted branchwise-max dominates

So this is a narrow control on top of the only positive lift variant that had
survived so far.

## Implementation

- New head:
  `ConfidentRescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_confident_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_confident_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_confident_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Relative to the earlier rescue-weighted anchored dual lift:

- keep the accepted branchwise-max mixed score as the anchor
- keep the learned rescue weight on the dual fixed-over-sharp advantage
- add a second multiplicative confidence gate driven by the anchored mixed score
- only allow the extra dual lift to stay large when the anchor itself is highly
  confident

So the intended effect was “same micro-budget lane, less mid-budget spill.”

## Held-Out Result

### `prototype_confident_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`

Closed, dead to slightly harmful.

At `0.10–0.75%` nominal budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall target match unchanged at `96.78%`
- overall mean delta regret `0.0000`

At `1.00–2.00%` nominal budget:

- held-out `stable_positive_v2` recovery still `0%`
- hard near-tie target match unchanged or slightly worse
- overall mean delta regret only `-0.0008` to `-0.0012`

So the plain branch again collapses to baseline.

### `prototype_confident_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed positive, but fully dominated.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.59%`
- overall mean delta regret `-0.0064`

At `0.25%` nominal budget:

- overall coverage `0.25%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.39% -> 90.45%`
- hard near-tie mean delta regret `-0.0043`
- overall target match `96.64%`
- overall mean delta regret `-0.0088`

At `0.50%` nominal budget:

- overall coverage `0.51%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match still only `90.39% -> 90.45%`
- overall mean delta regret `-0.0101`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery only `50%`
- hard near-tie target match still only `90.39% -> 90.45%`
- overall target match `96.78%`
- overall mean delta regret `-0.0141`

At `1.50–2.00%` nominal budget:

- overall coverage `1.35–1.59%`
- held-out `stable_positive_v2` recovery only `66.7%`
- hard near-tie target match only `90.39% -> 90.53%`
- overall mean delta regret plateaued at `-0.0149`

Large-gap controls stayed clean:

- large-gap target match stayed at `99.89%`
- mean delta regret stayed non-positive
- no harmful large-gap miss pattern appeared

## Comparison against rescue-weighted anchored dual lift

At `0.10–0.25%`, the confidence-gated follow-up reproduces the earlier
rescue-weighted curve exactly:

- `0.10%`: same `50%` held-out `stable_positive_v2`, same hard near-tie
  `90.39% -> 90.45%`, same overall mean delta regret `-0.0064`
- `0.25%`: same `50%`, same `90.39% -> 90.45%`, same overall mean delta
  regret `-0.0088`

So the extra confidence gate does not create a better micro-budget operating
point.

Above `0.25%`, it is worse:

- the rescue-weighted branch had already reached `66.7%` held-out
  `stable_positive_v2` by `1.00%`
- the confidence-gated branch stays stuck at `50%` there
- the rescue-weighted branch reached `83.3%` and `90.60%` hard near-tie by
  `2.00%`
- the confidence-gated branch only reaches `66.7%` and `90.53%`

So the confidence gate only suppresses the broader rescue-weighted lane without
improving the low-budget lane.

## Comparison against the accepted branchwise-max reference

At `0.10–0.25%`:

- accepted branchwise-max is weaker on overall mean delta regret
- but the confidence-gated branch is only tying the already-accepted
  rescue-weighted micro-budget companion, not improving it

At `0.50%`:

- accepted branchwise-max: overall mean delta regret `-0.0111`
- confidence-gated follow-up: `-0.0101`
- same held-out `stable_positive_v2` recovery `50%`
- same hard near-tie `90.39% -> 90.45%`

At `1.00%`:

- accepted branchwise-max: `83.3%` held-out `stable_positive_v2`, hard near-tie
  `90.39% -> 90.60%`, overall mean delta regret `-0.0145`
- confidence-gated follow-up: only `50%`, only `90.39% -> 90.45%`, overall
  mean delta regret `-0.0141`

At `2.00%`:

- accepted branchwise-max: `100%` held-out `stable_positive_v2`, hard near-tie
  `90.39% -> 90.68%`, overall mean delta regret `-0.0167`
- confidence-gated follow-up: only `66.7%`, only `90.39% -> 90.53%`, overall
  mean delta regret `-0.0149`

So accepted branchwise-max still dominates clearly from `0.50%` upward.

## Interpretation

The extra confidence gate did exactly the wrong thing:

- it did not sharpen the rescue-weighted micro-budget regime beyond the already
  accepted rescue-weighted variant
- it did suppress the broader `0.50–2.00%` lane

That means the rescue-weighted follow-up was already using the right amount of
implicit anchoring. Adding explicit anchor-confidence gating just duplicates the
micro-budget point and then over-suppresses the mid-budget recovery path.

So this closes the “extra anchor confidence on top of rescue weighting”
hypothesis.

## Decision

Close:

- `prototype_confident_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_confident_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Updated interpretation:

- accepted branchwise-max remains the main live reference
- rescue-weighted anchored dual lift remains the only positive micro-budget lift
  companion at `0.10–0.25%`
- do not add a separate anchor-confidence gate on top of rescue weighting,
  because it exactly reproduces the same micro-budget win and then loses the
  broader `0.50%+` lane
