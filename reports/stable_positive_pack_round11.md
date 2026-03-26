# Round 11 Stable-Positive Pack

## Definition

Round eleven defines the stable-positive correction pack as the intersection of:

- hard near-tie
- stable near-tie under perturbation
- teacher action changed
- teacher was labeled helpful
- material teacher gain:
  - regret gain at least `0.10`, or
  - miss improvement, or
  - explicit baseline-error recovery
- high-headroom near-tie or baseline-error near-tie

This is the narrow Tier-1 promotion surface for round eleven.

The canonical artifact is:

- `reports/plots/round11_teacher_bank_stable_positive_manifest.csv`

The broad Tier-2 guard remains the full near-tie frontier pack from round nine.

## Pack size

Across the three audited seeds:

- hard near-tie decisions: `2173`
- stable-positive pack decisions: `46`
- share of hard near-tie: `2.12%`

That is small, but still materially larger than zero and large enough to test
precision-first correction logic.

## Pack quality

The pack is extremely clean by construction:

- all stable-positive cases are action-changed teacher wins
- target match moves from `0.0%` to `100.0%`
- mean regret gain: `2.7310`
- p95 regret gain floor: about `0.6505`
- mean miss gain: `0.0435`

So the issue is not per-case value. The issue is coverage and transfer.

## Source families inside the pack

The pack mostly sits inside the two round-ten positive corners:

- high-headroom near-tie
- baseline-error near-tie

Aggregate rates:

- high-headroom near-tie stable-positive rate: `28.46%`
- baseline-error near-tie stable-positive rate: `18.85%`

That confirms the accepted round-ten recommendation: correction work should be
limited to the narrow high-value subset, not the full near-tie regime.

## Robustness warning

The pack is not stable across seeds in any strong sense.

By seed:

- seed314: `42` stable-positive decisions
- seed315: `1`
- seed316: `3`

Signature overlap across seeds is `0.0` for every pair at the current
source-family granularity.

Interpretation:

- the stable-positive pack is real
- but it is extremely concentrated and fragile
- the burden of proof for any deployable round-eleven policy is therefore high

This makes low-coverage, high-precision defer/correct logic the only sensible
continuation path.

## What counts as a win now

A round-eleven branch is interesting only if it:

- recovers a meaningful fraction of this stable-positive pack
- keeps false-positive corrections very low
- stays neutral-to-positive on the full near-tie frontier pack
- leaves large-gap controls effectively solved

If a branch needs broad coverage to recover the pack, it fails the deployment
goal.

## Artifacts

- `reports/plots/round11_teacher_bank_stable_positive_manifest.csv`
- `reports/plots/round11_teacher_bank_harmful_manifest.csv`
- `reports/plots/round11_teacher_bank_seed_summary.csv`
- `reports/plots/round11_teacher_bank_seed_overlap.csv`
