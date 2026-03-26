# Round 12 Ultra-Low-Coverage Defer

## Setup

Round twelve re-ran defer-to-teacher, but narrowed the question further:

> can a very small coverage budget beat the round-eleven `margin_regime`
> reference once the richer teacher bank is available?

Families:

- `linear`
- `mlp`
- `margin_regime`

Coverage budgets:

- `0.10%`
- `0.25%`
- `0.50%`
- `0.75%`
- `1.00%`
- `1.50%`
- `2.00%`

Teacher target at selection time stayed the audited `compute5` correction.

## Main result

The result is the same as round eleven:

- `linear`: dead
- `mlp`: dead
- `margin_regime`: only surviving learned gate

And even `margin_regime` did **not** beat the round-eleven reference once
compared fairly on held-out seeds.

## Held-out operating points

Aggregate held-out behavior for `margin_regime`:

At `0.10%` budget:

- overall coverage: `0.115%`
- stable-positive-v2 recovery: `25%`
- hard near-tie mean delta regret: `-0.0048`
- overall mean delta regret: `-0.0057`

At `0.50%` budget:

- overall coverage: `0.507%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0075`

At `1.00%` to `2.00%` budget:

- stable-positive-v2 recovery stays at `50%`
- hard near-tie mean delta regret stays flat at `-0.0071`
- overall mean delta regret also stays flat at about `-0.0075`

So the round-twelve gate does not improve as more budget is spent. It saturates
early and then stops moving the real frontier.

## Comparison to the round-eleven reference

Round-eleven `margin_regime` reference:

- at `1%`: hard near-tie delta regret `-0.0071`, overall `-0.0134`
- at `2%`: hard near-tie delta regret `-0.0089`, overall `-0.0151`

Round-twelve `margin_regime`:

- at `0.50%`: hard near-tie delta regret `-0.0071`, overall `-0.0075`
- at `1–2%`: still hard near-tie `-0.0071`, overall `-0.0075`

So round twelve recovered the same narrow positive behavior, but it did not
improve the combined system enough to beat the round-eleven reference.

## Decision

Do not promote round-twelve `margin_regime`.

What survives:

- it is still the only live learned defer family
- the usable band is still very low coverage

Why it is not promoted:

- no meaningful gain over the round-eleven reference
- the stable-positive-v2 pack is still only `4` held-out cases
- linear and MLP gates again fail to contribute anything

## Artifacts

- `reports/plots/round12_ultralow_defer_summary.csv`
- `reports/plots/round12_ultralow_defer_decisions.csv`
- `reports/plots/round12_ultralow_defer_summary.png`
