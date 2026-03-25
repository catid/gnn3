# Round 9 Triggered Continuation

## Goal

Test whether ambiguity-triggered extra continuation can keep the useful part of
the `compute5` signal while avoiding the unconditional damage of always running
to the final step.

Evaluated on:

- `a1_multiheavy_ood_deeper_packets6_round9_eval_seed314`
- hard near-tie intersection
- baseline-error near-tie subset
- large-gap control slice

## Trigger Variants

### `margin_gate_050`

- hard near-tie disagreement: `21.18%`
- hard near-tie recovery: `8.10%`
- hard near-tie new-error: `13.08%`
- hard near-tie target-match: `81.31%`
- large-gap control target-match: `59.60%`
- average selected step: `3.90`

### `margin_gate_100`

- hard near-tie disagreement: `21.18%`
- hard near-tie recovery: `8.10%`
- hard near-tie new-error: `13.08%`
- hard near-tie target-match: `81.31%`
- large-gap control target-match: `67.86%`
- average selected step: `3.98`

Interpretation for both margin gates:

- the gates do trigger on the right frontier region
- but they still spill too much damage onto large-gap controls
- the lower compute cost does not justify the control regression

### `risk_gate_tight`

- hard near-tie disagreement: `17.13%`
- hard near-tie recovery: `8.10%`
- hard near-tie new-error: `9.03%`
- hard near-tie target-match: `85.36%`
- large-gap control target-match: `97.32%`
- average selected step: `4.26`

Interpretation:

- on the frontier slice, this collapses to the same behavior as `fixed_final`
- it therefore preserves the same strengths and the same weaknesses
- it does not supply a better selective-compute trade than the unconditional
  final-step policy

## Verdict

Triggered continuation is negative in the tested form.

What we learned:

- simple margin triggers are too blunt
- the risk trigger does not improve the target slice beyond the unconditional
  final-step policy
- none of the tested triggers achieved the needed combination of:
  - baseline-error recovery
  - low new-error rate
  - preserved large-gap control behavior
  - meaningful compute savings

So no triggered-continuation policy is promoted from round nine.
