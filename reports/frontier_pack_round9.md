# Round 9 Frontier Pack

## Summary

Round nine uses the machine-readable frontier pack generated at
`reports/plots/round9_frontier_pack_seed314_decisions.csv` and
`reports/plots/round9_frontier_pack_seed314.json`.

The corrected target surface is the score-based hard near-tie slice, not the
old large-gap control.

Key slice stats from the fresh seed314 frontier pack:

- `hard_feasible`: `2819` decisions, baseline error rate `4.93%`
- `hard_near_tie_intersection`: `777` decisions, baseline error rate `15.96%`,
  mean headroom `0.292`, p95 headroom `2.285`
- `stable_near_tie`: `738` decisions, baseline error rate `14.09%`, effective
  tie rate `0.0%`
- `high_headroom_near_tie`: `62` decisions, baseline error rate `100%`, mean
  headroom `2.993`, p95 headroom `7.671`
- `baseline_error_intersection`: `124` decisions, mean headroom `1.832`,
  p95 headroom `4.048`, effective tie rate `16.13%`
- `large_gap_control`: `1216` decisions, baseline error rate `0.33%`

This reproduces the round-seven/round-eight diagnosis with a more operational
surface:

- large-gap controls remain essentially solved
- the real frontier is the hard near-tie slice
- the highest-value subset is the high-headroom near-tie subset
- most hard near-tie states are stable under perturbation, so the remaining
  problem is not just simulation noise

## Thresholds

The frontier pack thresholds used for round nine are:

- large-gap threshold: `4.7261`
- large-gap ratio threshold: `0.2916`
- near-tie gap threshold: `4.2455`
- model-margin threshold: `4.0494`
- high-headroom near-tie threshold: `1.4241`

These thresholds are stored in
`reports/plots/round9_frontier_pack_seed314.json` and should be reused for
matched seed314 comparisons in this round.

## Suite Distribution

The hard near-tie target slice is concentrated in stress suites rather than the
plain corrected feasible seed314 baseline:

- base corrected suite: `0` hard near-tie decisions
- mixed-compute suite: `264`
- `deeper_packets6`: `321`
- `heavy_dynamic`: `190`
- `branching3`: `2`

That means promotion should be decided primarily on:

- mixed-compute
- `deeper_packets6`
- `heavy_dynamic`

The base feasible suite remains a guardrail, not the main opportunity source.

## Operational Use

The frontier pack supplies these reusable manifests:

- `hard_near_tie_intersection`
- `stable_near_tie`
- `high_headroom_near_tie`
- `baseline_error_intersection`
- `large_gap_control`
- `decodable_near_tie`
- `weakly_decodable_near_tie`

All round-nine promotions should report:

- disagreement on `hard_near_tie_intersection`
- recovery on `baseline_error_intersection`
- behavior on `high_headroom_near_tie`
- guardrail behavior on `large_gap_control`
