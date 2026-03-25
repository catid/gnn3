# Round 9 Continuation Report

## Current Status

Round nine is testing the remaining compute-and-state thesis:

> extra conditional compute and/or explicit delayed state may improve the
> hard near-tie frontier without breaking the robust `multiheavy` baseline.

The round is running on branch `round9/compute-state-near-tie` and will merge
back into local `main` once the queued experiments finish and the repo is green.

## Accepted Baseline

The accepted default policy is still plain `multiheavy`.

The anchor remains the corrected round-four comparison against fresh `E3`:

- mean regret `1.32` vs `2.25`
- mean p95 regret `4.77` vs `10.48`
- mean deadline miss `41.7%` vs `54.2%`

Round six reproduced the same baseline band, and round eight showed that the
remaining mistakes concentrate in the hard near-tie regime rather than the old
large-gap slice.

## Frontier Pack

The fresh round-nine frontier pack is now live at:

- `reports/plots/round9_frontier_pack_seed314_decisions.csv`
- `reports/plots/round9_frontier_pack_seed314.json`

Current seed314 frontier facts:

- hard near-tie intersection: `777` decisions, baseline error rate `15.96%`
- stable near-tie: `738` decisions, baseline error rate `14.09%`
- high-headroom near-tie: `62` decisions, baseline error rate `100%`
- baseline-error intersection: `124` decisions, mean headroom `1.832`,
  p95 headroom `4.048`
- large-gap control: `1216` decisions, baseline error rate `0.33%`

The practical target surface is now:

- `hard_near_tie_intersection`
- `stable_near_tie`
- `high_headroom_near_tie`
- `baseline_error_intersection`

and the primary guardrail remains `large_gap_control`.

## First Positive Signal

The first extra-compute scout is positive.

`e3_memory_hubs_rsm_round9_compute5_seed314` improved over the matched fresh
seed314 `multiheavy` baseline:

- baseline seed314 rollout: regret `1.898`, p95 `8.140`, miss `43.75%`
- compute5 seed314 rollout: regret `1.302`, p95 `5.277`, miss `31.25%`

This is the first round-nine result that clearly justifies promotion beyond a
single seed. The matched seed315 confirmation is now running, along with a
deeper `compute7` headroom scout and the seed314 frontier guard.

## Active Queue

Running or queued at this stage:

- seed315 matched `compute5` confirmation
- seed314 deeper `compute7` headroom scout
- seed314 frontier guard: `multiheavy` vs `compute5`
- seed314 mailbox scouts
- seed316 matched contender pair if seed315 confirms the compute5 signal

## Closed Doors Still Closed

Round nine does not reopen:

- old rerankers
- selector-only tuning
- generic history-memory banks
- prior poly/self-improve/specialist-teacher families
- large-gap targeting

Any promoted branch must still win on the hard near-tie frontier and keep the
large-gap controls effectively solved.
