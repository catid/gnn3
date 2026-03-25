# Round 9 Adaptive Halting

## Goal

Test whether the seed314 `compute5` signal can be compressed into a better
depth-selection rule on the hard near-tie frontier, rather than paying fixed
extra compute everywhere.

The audited target surface here is:

- `a1_multiheavy_ood_deeper_packets6_round9_eval_seed314`
- hard near-tie intersection
- baseline-error near-tie subset
- large-gap control slice

All reported disagreement is against the seed314 `multiheavy` baseline from the
frontier pack.

## Baseline Context

On this suite, the baseline-error near-tie subset contains `44` errors out of
`321` hard near-tie decisions, so baseline target-match on that slice is
`86.29%`.

## Halting / Step-Choice Variants

### `fixed_middle`

- hard near-tie disagreement: `28.97%`
- hard near-tie recovery: `7.48%`
- hard near-tie new-error: `21.50%`
- hard near-tie target-match: `72.27%`
- large-gap control target-match: `31.47%`

Interpretation:

- middle-step policies do change decisions a lot
- they are strongly anti-guardrail and are not viable

### `fixed_final`

- hard near-tie disagreement: `17.13%`
- hard near-tie recovery: `8.10%`
- hard near-tie new-error: `9.03%`
- hard near-tie target-match: `85.36%`
- baseline-error near-tie recovery: `59.09%`
- large-gap control target-match: `97.32%`

Interpretation:

- deeper compute really does recover many baseline hard near-tie mistakes
- but it still introduces nearly as many new mistakes as it fixes
- net hard near-tie target-match remains slightly below the baseline surface

### `learned_gate`

- hard near-tie disagreement: `22.12%`
- hard near-tie recovery: `7.48%`
- hard near-tie new-error: `14.64%`
- hard near-tie target-match: `79.13%`
- large-gap control target-match: `85.49%`

Interpretation:

- the learned gate did not localize the useful extra-compute changes
- it regressed both the frontier slice and the control slice

## Verdict

Adaptive halting is negative in the tested form.

The real lesson is not that extra compute is useless. It is:

- extra compute changes some of the right hard near-tie decisions
- but the tested halting rules cannot preserve those good changes without
  causing too many new errors elsewhere

So no adaptive-halting branch is promoted from this round.
