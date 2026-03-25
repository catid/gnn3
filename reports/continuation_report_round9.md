# Round 9 Continuation Report

## Round Objective

Round nine tested the remaining compute-and-state thesis:

> extra conditional compute and/or explicit delayed state may improve the
> hard near-tie frontier without breaking the robust `multiheavy` baseline.

The round ran on branch `round9/compute-state-near-tie` and will merge back
into local `main` once the final validation pass is green.

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

## Compute Compression Verdict

Round nine then tested whether the seed314 `compute5` signal could be
compressed into cheaper depth-selection rules on the main frontier suite
`deeper_packets6`.

Key context:

- `321` hard near-tie decisions
- `44` baseline hard near-tie errors
- baseline hard near-tie target-match `86.29%`

Best direct compute policy:

- `fixed_final`
  - hard near-tie disagreement `17.13%`
  - baseline-error near-tie recovery `59.09%`
  - hard near-tie new-error `9.03%`
  - hard near-tie target-match `85.36%`
  - large-gap control target-match `97.32%`

Other tested policies:

- `risk_gate_tight`
  - effectively matched `fixed_final` on the frontier and control slices, but
    did not improve the tradeoff
- `margin_gate_050`
  - hard near-tie target-match `81.31%`
  - large-gap control target-match `59.60%`
- `margin_gate_100`
  - hard near-tie target-match `81.31%`
  - large-gap control target-match `67.86%`
- `learned_gate`
  - hard near-tie target-match `79.13%`
  - large-gap control target-match `85.49%`
- `fixed_middle`
  - catastrophically bad on large-gap controls

Verdict:

- extra compute really does change some of the right hard near-tie decisions
- none of the tested compression rules preserve a net win on the frontier
- no adaptive-halting or triggered-continuation policy is promoted

## Offline Branch Teacher Verdict

The offline branch-refinement teacher family was tested directly on the
seed314 `compute5` checkpoint over the same `deeper_packets6` hard near-tie
slice.

Teacher grid:

- `top_k=2`, horizon `1`
  - disagreement `12.5%`, recovery `0.0%`, new-error `12.5%`
- `top_k=2`, horizon `2`
  - disagreement `6.25%`, recovery `0.0%`, new-error `6.25%`
- `top_k=3`, horizon `1`
  - disagreement `25.0%`, recovery `0.0%`, new-error `25.0%`
- `top_k=3`, horizon `2`
  - disagreement `18.75%`, recovery `0.0%`, new-error `18.75%`

Verdict:

- branching and refining does move decisions
- every tested teacher variant was net harmful
- no teacher-for-compute distillation branch is justified from this family

## Delay Mailbox Verdict

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

## Route Persistence Verdict

The route-persistence audit on the hard near-tie frontier was also negative.

On `deeper_packets6`:

- oracle hub defined rate `81.25%`
- oracle stable rate only `53.85%` at horizons `1` and `2`
- oracle stable rate `38.46%` at horizons `3` and `4`
- model unnecessary flip rate `0.0%` at every measured horizon

So the remaining frontier does not currently look like gratuitous short-lived
route flipping. Route-option work stays closed.

## Round-Nine Verdict

Round nine closes the compute-and-state campaign in its tested forms:

- fixed extra compute is real but non-robust
- adaptive halting / triggered continuation did not localize that gain well
- offline branch teachers were decisively anti-helpful
- plain delay mailboxes were early negatives
- route persistence did not look like the missing mechanism

The default policy therefore stays plain `multiheavy`.

## Closed Doors Still Closed

Round nine does not reopen:

- old rerankers
- selector-only tuning
- generic history-memory banks
- prior poly/self-improve/specialist-teacher families
- large-gap targeting

Any promoted branch must still win on the hard near-tie frontier and keep the
large-gap controls effectively solved.
