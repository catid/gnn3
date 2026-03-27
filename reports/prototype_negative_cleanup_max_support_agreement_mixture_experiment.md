# Prototype Negative-Cleanup Max Support Agreement-Mixture Experiment

## Question

Test whether the two live negative-cleanup lanes can be combined without
averaging them together:

- keep the sharp negative-tail branch as the default retrieval view
- allow a nonnegative max-style lift from the fixed negative-tail branch
- union in any extra sparse-positive states that only the fixed branch scores
  strongly

The design goal was:

- preserve the sharp branch's stronger aggregate regret below `1%` coverage
- preserve more of the fixed branch's high-recall behavior around `1%`
- avoid the weak middle that appeared in the learned cleanup-blend follow-up

## Implementation

- New head: `NegativeCleanupMaxSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_negative_cleanup_max_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_negative_cleanup_max_support_agree_mix`
  - `prototype_negative_cleanup_max_support_agree_mix_hybrid`

Relative to the live negative-cleanup heads:

- the model computes one score with fixed negative-tail cleanup
- it computes a second score with sharpness-gated negative-tail cleanup
- it takes the elementwise max of those two final retrieval scores
- the risk branch remains optional and is only added in the `_hybrid` variant

So this is a hard union of the two live cleanup views rather than a learned
interpolation between them.

## Held-Out Result

### `prototype_negative_cleanup_max_support_agree_mix`

Dead.

- no selected states at any budget
- `0%` held-out `stable_positive_v2` recovery
- hard near-tie unchanged at `90.53%`
- overall mean delta regret `0.0000`

The plain max head fully collapsed to baseline.

### `prototype_negative_cleanup_max_support_agree_mix_hybrid`

Positive, but clearly dominated by existing live leads.

At `0.75%` nominal budget:

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.69%`
- overall mean delta regret `-0.0103`

At `1.50%` nominal budget:

- overall coverage `1.51%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.74%`
- overall mean delta regret `-0.0118`

At `2.00%` nominal budget:

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.78%`
- overall mean delta regret `-0.0132`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret `-0.0065`
- large-gap mean delta miss `0.0000`

## Comparison against live leads

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_support_weighted_agree_mix_hybrid @ 1.00%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

So the cleanup-max branch does recover real signal, but it does not survive:

- below `1%` coverage it still only reaches the weaker `50%` / `90.66%` band
- by the time it reaches the full `75%` / `90.73%` frontier band, it needs
  `1.5–2.0%` coverage
- it is still weaker there than both the sharp negative-tail and original
  support-weighted matched-band leads
- it also gives back the fixed negative-tail branch's `100%` held-out recall

## Interpretation

This confirms that the fixed and sharp negative-cleanup lanes do contain
compatible signal, but a hard union over their final scores is still too blunt:

- it is stronger than the learned cleanup blend
- it eventually recovers the same `75%` / `90.73%` band as the live sharp branch
- but it reaches that band too late and too expensively

So the issue is not just preserving both signals somewhere inside the model. It
is preserving them at the right coverage scale without reopening too much broad
selection.

## Decision

Close:

- `prototype_negative_cleanup_max_support_agree_mix`
- `prototype_negative_cleanup_max_support_agree_mix_hybrid`

Keep the live shortlist unchanged:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best sub-`1%`
  full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out recall
  around `1%`
- `prototype_support_weighted_agree_mix_hybrid` as the higher-budget
  matched-band reference
