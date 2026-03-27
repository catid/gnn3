# Prototype Tail-Margin-Calibrated Branchwise-Max Negative-Cleanup Support Agreement-Mixture Experiment

## Question

Test whether the accepted branchwise-max negative-cleanup prototype family
improves if the fixed and sharp cleanup paths inside the shared and dual
branches learn separate bounded tail margins instead of sharing one global
`tail_margin`.

This is the direct follow-up to the round-thirteen note that if cleanup tuning
reopens, it should be tested against the `71` useful hard negatives rather than
through more generic bank-editing sweeps.

The design goal was:

- keep the accepted support-weighted agreement-mixture geometry
- keep the accepted shared plus dual branch decomposition
- keep the accepted branch-local fixed-vs-sharp negative cleanup
- keep the accepted hard branchwise max before the outer agreement mix
- only add a small calibration degree of freedom:
  - shared fixed margin
  - shared sharp margin
  - dual fixed margin
  - dual sharp margin

## Implementation

- New head:
  `TailMarginCalibratedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_tail_margin_calibrated_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_tail_margin_calibrated_branchwise_max_negative_cleanup_support_agree_mix`
  - `prototype_tail_margin_calibrated_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

The calibration stays intentionally conservative:

- each margin is `tail_margin + margin_delta_scale * tanh(raw_delta)`
- `margin_delta_scale = 0.4`
- small L2 regularization is applied to the raw deltas

## Held-Out Result

### `prototype_tail_margin_calibrated_branchwise_max_negative_cleanup_support_agree_mix`

Closed, fully dead.

At every budget:

- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.39%`
- overall mean delta regret `0.0000`

So the plain branch closes immediately.

### `prototype_tail_margin_calibrated_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Closed weak positive, but not promotable.

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
- overall mean delta regret `-0.0086`

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `66.7%`
- hard near-tie target match `90.53%`
- overall target match `96.75%`
- overall mean delta regret `-0.0134`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery only `66.7%`
- hard near-tie target match only `90.53%`
- overall target match `96.79%`
- overall mean delta regret `-0.0145`

At `1.50%` nominal budget:

- overall coverage `1.38%`
- held-out `stable_positive_v2` recovery `83.3%`
- hard near-tie target match `90.60%`
- overall target match `96.83%`
- overall mean delta regret `-0.0157`

At `2.00%` nominal budget:

- overall coverage `1.62%`
- held-out `stable_positive_v2` recovery only `83.3%`
- hard near-tie target match only `90.60%`
- overall target match `96.87%`
- overall mean delta regret `-0.0164`

So the branch finds only a tiny micro-budget aggregate-quality gain, then gives
back the accepted `1.0%` and `2.0%` frontier positions.

## Margin sanity check

The learned margins explain the failure clearly.

Hybrid margin summary:

- shared fixed margin `0.4982`
- shared sharp margin `0.5000`
- dual fixed margin `0.4985`
- dual sharp margin `0.5000`

Plain margin summary:

- shared fixed margin `0.4896`
- shared sharp margin `0.5000`
- dual fixed margin `0.4988`
- dual sharp margin `0.5000`

So the new calibration almost never moved away from the original global
`tail_margin = 0.5`. The extra flexibility was effectively inert.

## Comparison against the accepted branchwise-max reference

Accepted reference:
`prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

At `0.25%`:

- same held-out `stable_positive_v2` recovery `50%`
- same hard near-tie target match `90.45%`
- reference overall mean delta regret `-0.0074`
- tail-margin-calibrated `-0.0086`

So there is a small micro-budget aggregate gain.

At `1.00%`:

- reference held-out `stable_positive_v2` recovery `83.3%`
- tail-margin-calibrated `66.7%`
- reference hard near-tie target match `90.60%`
- tail-margin-calibrated `90.53%`
- reference overall mean delta regret `-0.0145`
- tail-margin-calibrated `-0.0145`

So it gives back the actual target-slice frontier for no meaningful aggregate
benefit.

At `2.00%`:

- reference held-out `stable_positive_v2` recovery `100%`
- tail-margin-calibrated `83.3%`
- reference hard near-tie target match `90.68%`
- tail-margin-calibrated `90.60%`
- reference overall mean delta regret `-0.0167`
- tail-margin-calibrated `-0.0164`

So the accepted branch still clearly wins the higher-budget operating region.

## Interpretation

This closes the first direct cleanup-threshold-tuning follow-up.

What changed:

- the model was allowed to tune four separate cleanup margins

What actually happened:

- the learned margins stayed pinned almost exactly at the original `0.5`
- the added flexibility did not create a new operating region

What it bought:

- a tiny micro-budget aggregate-regret gain around `0.1–0.25%`

What it cost:

- the accepted `1.0%` matched-band point
- the accepted `2.0%` high-recall point

So the conclusion is:

- branch/path-specific free tail-margin calibration is not the right cleanup
  tuning mechanism for the live branchwise-max family
- if cleanup tuning reopens, it should be tied to the hard-negative pack or an
  offline search/calibration pass, not tiny differentiable margin deltas

## Decision

Close:

- `prototype_tail_margin_calibrated_branchwise_max_negative_cleanup_support_agree_mix`
- `prototype_tail_margin_calibrated_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Keep the accepted live reference unchanged:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

No promotion.
