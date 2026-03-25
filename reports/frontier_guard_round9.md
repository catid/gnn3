# Round 9 Frontier Guard

## Purpose

Round nine does not promote branches on broad average metrics alone. Every
candidate is judged first on the corrected hard near-tie frontier, then on
control slices that must remain stable.

The machine-readable source of truth for this guard is the seed314 frontier
pack:

- `reports/plots/round9_frontier_pack_seed314_decisions.csv`
- `reports/plots/round9_frontier_pack_seed314.json`

## Frontier Slices

Primary promotion surface:

- `hard_near_tie_intersection`
- `stable_near_tie`
- `high_headroom_near_tie`
- `baseline_error_intersection`

Primary control surface:

- `large_gap_control`

Suite emphasis for promotion:

- `a1_multiheavy_mixed_compute_round9_eval_seed314`
- `a1_multiheavy_ood_deeper_packets6_round9_eval_seed314`
- `a1_multiheavy_ood_heavy_dynamic_round9_eval_seed314`

The corrected base feasible suite remains a guardrail, not the main source of
opportunity, because the frontier pack shows zero hard near-tie decisions there.

## Seed314 Frontier Facts

- hard near-tie intersection: `777` decisions, baseline error rate `15.96%`
- stable near-tie: `738` decisions, baseline error rate `14.09%`
- high-headroom near-tie: `62` decisions, baseline error rate `100%`
- baseline-error intersection: `124` decisions, mean headroom `1.832`,
  p95 headroom `4.048`
- large-gap control: `1216` decisions, baseline error rate `0.33%`

Source concentration on the target slice:

- mixed-compute: `264`
- `deeper_packets6`: `321`
- `heavy_dynamic`: `190`
- `branching3`: `2`

That concentration is why round-nine promotion decisions should be dominated by
`deeper_packets6`, `heavy_dynamic`, and mixed-compute behavior rather than by
the plain corrected feasible suite.

## Required Per-Branch Metrics

Every new branch must report, at minimum:

- disagreement vs `multiheavy` on `hard_near_tie_intersection`
- recovery rate on `baseline_error_intersection`
- new-error rate on baseline-correct hard near-tie states
- target-match / error-rate change on `high_headroom_near_tie`
- behavior on `large_gap_control`
- broad feasible-suite regression check

When compute varies, also report:

- average selected outer steps
- regret at fixed or matched average compute where possible
- wall-clock and throughput proxy

## Promotion Gates

Promote a scout only if at least one of these is true on the corrected hard
near-tie frontier:

- disagreement vs `multiheavy` >= `6%` and not strongly anti-oracle
- baseline-error recovery >= `30%`
- absolute hard near-tie regret drop >= `0.10`
- materially better regret-at-fixed-average-compute on the mixed-compute suite

Additional requirements:

- `large_gap_control` must remain effectively solved
- no broad feasible-suite regression beyond tolerance
- runtime overhead must be acceptable unless the branch is explicitly
  teacher-only

## Kill Rules

Kill immediately if:

- hard near-tie disagreement is effectively zero
- moved decisions are strongly anti-oracle
- gains come only from large compute blowup without acceptable payoff
- `large_gap_control` regresses materially
- broad feasible-suite regret regresses beyond tolerance
- the branch recreates a previously closed family in disguise

## Current Guardrail Read

The current accepted guardrail conclusions entering the remaining round-nine
batches are:

- fixed `compute5` is not a robust upgrade
  - positive on seed314
  - exact baseline match on seeds 315 and 316
- deeper fixed compute (`compute7`) is a hard negative
- plain delay-mailbox scouts are early negatives

So the only remaining admissible promotions are ones that beat the frontier
surface more selectively:

- compute-policy compression
- outer-step / continuation policies
- offline branch-teacher signals that could support cheaper continuation or
  distillation
- minimal delayed state only if it changes the hard near-tie frontier without
  reintroducing the negative mailbox behavior

## Current Frontier Guard Read

The later round-nine compute-state probes now add concrete guardrail numbers on
the main frontier suite `deeper_packets6`.

Best direct continuation candidate:

- `fixed_final`
  - hard near-tie disagreement `17.13%`
  - baseline-error near-tie recovery `59.09%`
  - hard near-tie new-error `9.03%`
  - hard near-tie target-match `85.36%`
  - large-gap control target-match `97.32%`

Negative reference points:

- `fixed_middle`
  - hard near-tie target-match `72.27%`
  - large-gap control target-match `31.47%`
- `margin_gate_050`
  - hard near-tie target-match `81.31%`
  - large-gap control target-match `59.60%`
- `learned_gate`
  - hard near-tie target-match `79.13%`
  - large-gap control target-match `85.49%`

So the guardrail verdict is:

- extra compute can correct some audited frontier mistakes
- none of the tested direct compute policies preserve a net frontier win while
  keeping control slices healthy

Operational note:

- the monolithic `run_round9_frontier_guard.py` rollup was attempted and then
  stopped for throughput reasons
- round nine uses the completed decision-slice audit artifacts as the concrete
  frontier guard instead of waiting on that slower wrapper
