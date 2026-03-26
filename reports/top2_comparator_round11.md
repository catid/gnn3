# Round 11 Top-2 Comparator With Abstain

## Setup

Round eleven tested a top-2 correction policy aimed at the actual residual
structure of the problem:

- baseline top-1 vs baseline top-2 candidate
- keep / flip / abstain objective
- tiny coverage budgets
- held-out evaluation on seed315 and seed316

Families:

- `frozen`
- `candidate_conditioned`

Scopes:

- `broad`
- `narrow`

The intended promotion surface was still:

- Tier 1: stable-positive pack
- Tier 2: full near-tie frontier pack plus large-gap control

## Main result

The comparator family is closed.

`frozen` comparators were essentially inert:

- no stable-positive recovery on held-out seed315
- no meaningful hard near-tie movement on seed315
- on seed316 they changed almost nothing, while still drifting slightly off the
  baseline policy

`candidate_conditioned:broad` was also effectively inert.

`candidate_conditioned:narrow` was worse than inert: it flipped solved cases
without finding the stable-positive pack.

## Held-out verdict

Seed315:

- every comparator variant had `0%` stable-positive recovery
- hard near-tie coverage stayed at `0%` across the useful budgets
- no branch produced meaningful disagreement in the right places

Seed316:

- `candidate_conditioned:narrow` was actively harmful
- at `1%` budget on hard near-tie:
  - coverage `2.89%`
  - correction rate `0.0%`
  - new-error rate `2.89%`
  - hard near-tie target match fell to `86.15%`
- at `5%` budget on hard near-tie:
  - coverage `12.02%`
  - correction rate `0.15%`
  - new-error rate `11.87%`
  - hard near-tie target match collapsed to `77.32%`
- large-gap control also regressed materially:
  - target match fell to `97.22%` at `5%`

The broad and frozen comparators stayed closer to baseline, but none of them
recovered the stable-positive pack at useful low coverage.

## Decision

The comparator-with-abstain family does not move the true frontier in a safe
way.

It fails for two separate reasons:

- the conservative variants are almost policy-identical
- the candidate-conditioned narrow variant moves, but moves in the wrong places

Round eleven therefore closes the top-2 comparator family without promotion.

## Artifacts

- `reports/plots/round11_top2_comparator_heldout_summary.csv`
- `reports/plots/round11_top2_comparator_heldout.json`
- `reports/plots/round11_top2_comparator_heldout_summary.png`
