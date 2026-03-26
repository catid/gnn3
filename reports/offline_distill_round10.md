# Round 10 Offline Distillation

## Setup

Round 10 trained six offline student families from the seed314 helpful-compute cache and evaluated them on
held-out seed315 and seed316 caches:

- `pairwise`
- `kl`
- `residual`
- `gated_pairwise`
- `gated_kl`
- `gated_residual`

The promotion surface remains the held-out near-tie frontier pack, not baseline-error recovery by itself.

## Main result

No offline student earned promotion.

The family split cleanly into:

- aggressive students that recover many audited baseline errors but break too many solved cases
- conservative students that stay safe but do not improve the full hard near-tie frontier enough

## Best conservative student

`gated_pairwise` was the only variant that stayed globally safe enough to remain interesting.

Aggregate held-out behavior:

- overall target match: `96.50%` vs baseline `96.43%`
- overall mean delta regret: `-0.0027`
- large-gap control target match: `99.96%` vs baseline `99.82%`
- large-gap control mean delta regret: `-0.0083`

But it still missed the real promotion gate:

- hard near-tie target match: `91.01%` vs baseline `91.16%`
- hard near-tie mean delta regret: `+0.0110`
- hard near-tie correction rate: `0.73%`
- hard near-tie new-error rate: `0.88%`

It does help on the narrow source families:

- high-headroom near-tie correction rate: `9.02%`
- high-headroom mean delta regret: `-0.2858`
- baseline-error near-tie correction rate: `6.98%`
- baseline-error near-tie mean delta regret: `-0.1590`

That makes it the best conservative reference from this round, but not a contender.

## Near-identity student

`kl` was almost policy-identical and slightly safer on the full hard near-tie slice:

- hard near-tie disagreement: `0.66%`
- hard near-tie target match: `91.20%` vs baseline `91.16%`
- hard near-tie mean delta regret: `+0.0045`

That is too little movement to matter. It is effectively a no-op.

## Aggressive students

`residual`, `gated_residual`, and `gated_kl` recovered much more of the audited baseline-error slice, but the
trade was unacceptable:

- `residual` baseline-error correction: `84.60%`, but hard near-tie target match collapsed to `80.91%`
- `gated_kl` baseline-error correction: `36.56%`, but hard near-tie target match fell to `87.82%`
- `gated_residual` baseline-error correction: `35.78%`, but hard near-tie target match fell to `87.20%`

So the family reproduced the expected failure mode: more recovery only by paying too many new errors.

## Decision

Pure offline distillation is closed for this round.

The strongest thing it demonstrated is narrower than the original thesis:

- there is a small conservative correction band that a gated pairwise student can imitate
- that band is not large enough to produce a clean near-tie frontier-pack win

If this family is ever reopened, `gated_pairwise` is the only sensible starting point, and only under a much
stricter abstention or precision-first objective.

## Artifacts

- `reports/plots/round10_offline_distill_summary.csv`
- `reports/plots/round10_offline_distill.json`
- `reports/plots/round10_offline_distill_summary.png`
- `reports/plots/round10_offline_distill_checkpoints/`
