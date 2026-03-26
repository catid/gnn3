# Round 11 Subset-Only Distillation

## Setup

Round eleven reopened distillation only under the new narrow thesis:

- train on the stable-positive correction pack only
- keep a hard baseline anchor outside the pack
- evaluate on held-out seed315 and seed316

Families:

- `pairwise`
- `kl`
- `residual`
- `gated_pairwise`

This was intentionally narrower than round ten. The goal was to see whether
subset-only supervision can preserve the positive teacher signal without the
old broad near-tie regressions.

## Main result

Subset-only distillation is also closed.

The family reproduced the same split as round ten:

- aggressive students recovered more positive states, but broke too many solved
  cases
- conservative students stayed safer, but still did not improve the held-out
  hard near-tie frontier enough

## Held-out aggregate behavior

`pairwise`:

- stable-positive pack target match: `100%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `+0.0009`
- large-gap control target match: `99.79% -> 91.98%`

So it recovered the tiny Tier-1 pack, but only by breaking too many solved
large-gap cases. Not deployable.

`kl`:

- stable-positive pack target match: `50%`
- hard near-tie target match: `90.53% -> 86.23%`
- hard near-tie mean delta regret: `+0.1072`
- large-gap control target match: `99.79% -> 96.56%`

This is clearly too destructive.

`residual`:

- stable-positive pack target match: `0%`
- hard near-tie target match: `90.53% -> 78.44%`
- hard near-tie mean delta regret: `+0.3678`

This reproduces the old aggressive failure mode directly and stays closed.

`gated_pairwise`:

- stable-positive pack target match: `50%`
- hard near-tie target match: `90.53% -> 90.46%`
- hard near-tie mean delta regret: `+0.0036`
- large-gap control target match: `99.79% -> 99.90%`
- overall target match: `96.51% -> 96.60%`

This is the safest student again, but it still misses the real gate. It stays
too close to baseline on seed315 and slightly regresses aggregate hard near-tie
on the held-out pair.

## Seed split

Seed315:

- `pairwise` and `kl` recovered the single stable-positive case
- both still regressed solved cases too much, especially large-gap control
- `gated_pairwise` collapsed to baseline and missed the single positive case

Seed316:

- `pairwise` and `gated_pairwise` both recovered `2 / 3` stable-positive cases
- `pairwise` and especially `residual` paid too many new errors on solved slices
- `gated_pairwise` stayed globally safe, but its hard near-tie gain was too
  small and inconsistent

## Decision

Subset-only distillation does not clear the round-eleven bar.

The stable-positive source family is too sparse to support a deployable student
under the current teacher bank and current frozen representation.

The only branch worth remembering is still `gated_pairwise`, but only as a
reference conservative student, not a contender.

## Artifacts

- `reports/plots/round11_subset_distill_heldout_summary.csv`
- `reports/plots/round11_subset_distill_heldout.json`
- `reports/plots/round11_subset_distill_heldout_summary.png`
