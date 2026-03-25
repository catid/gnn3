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

## Fixed Extra Compute Verdict

The first extra-compute scout was positive on seed314:

- baseline seed314 rollout: regret `1.898`, p95 `8.140`, miss `43.75%`
- compute5 seed314 rollout: regret `1.302`, p95 `5.277`, miss `31.25%`

But the matched follow-ups did not replicate:

- baseline seed315 rollout: regret `2.107`, p95 `6.164`, miss `56.25%`
- compute5 seed315 rollout: regret `2.107`, p95 `6.164`, miss `56.25%`
- baseline seed316 rollout: regret `1.508`, p95 `7.307`, miss `50.00%`
- compute5 seed316 rollout: regret `1.508`, p95 `7.307`, miss `50.00%`

So the fixed `compute5` family is no longer a contender. The current read is
that extra fixed compute can help on individual seeds, but it is not a robust
constructor upgrade in its plain form.

## Deeper Fixed Compute

The first deeper fixed-compute headroom scout (`compute7` on seed314) failed
its early gate immediately:

- epoch-1 rollout regret `8893.68`
- epoch-1 p95 regret `24061.70`
- epoch-1 miss `100%`

That branch was killed before spending full scout budget. The current compute
thesis remains focused on `compute5`, not unboundedly increasing outer rounds.

## Delay Mailbox Early Read

The first two minimal delay-mailbox scouts both failed their early gates on the
same seed314 corrected benchmark:

- `mailbox_monitor12_seed314`
  - epoch-2 rollout: regret `6.676`, p95 `20.028`, miss `68.75%`
  - epoch-3 rollout: regret `7.542`, p95 `23.491`, miss `68.75%`
- `mailbox_hubmonitor124_seed314`
  - epoch-2 rollout: regret `7.029`, p95 `21.113`, miss `56.25%`

Both remained far worse than the matched seed314 `multiheavy` baseline, so the
plain mailbox family is currently tracking as a negative unless a later combo
branch changes the picture materially.

## Active Queue

Running or queued at this stage:

- seed314 compute-policy sweep on the positive `compute5` checkpoint
- seed314 outer-step headroom audit on the same checkpoint
- offline branch-teacher grid after the audit pair clears
- report and portfolio refresh once the long-running audits land

## Closed Doors Still Closed

Round nine does not reopen:

- old rerankers
- selector-only tuning
- generic history-memory banks
- prior poly/self-improve/specialist-teacher families
- large-gap targeting

Any promoted branch must still win on the hard near-tie frontier and keep the
large-gap controls effectively solved.
