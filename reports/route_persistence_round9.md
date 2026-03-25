# Round 9 Route Persistence Audit

## Scope

This audit asks whether short-lived route commitment is still a plausible
round-nine follow-up on the hard near-tie frontier.

It was run on the seed314 `multiheavy` checkpoint using:

- `a1_multiheavy_ood_deeper_packets6_round9_eval_seed314`
- `a1_multiheavy_ood_heavy_dynamic_round9_eval_seed314`
- target slice: `hard_near_tie_intersection_case`
- max persistence horizon: `4` hops

Artifacts:

- `reports/plots/round9_route_persistence_seed314_frontier_summary.csv`
- `reports/plots/round9_route_persistence_seed314_frontier_decisions.csv`
- `reports/plots/round9_route_persistence_seed314_frontier.json`

## Result

This audit is currently negative for route-option work.

### `deeper_packets6`

- target decisions: `16`
- oracle hub defined rate: `81.25%`
- model hub defined rate: `50.0%`
- oracle stable for 1 hop: `53.85%`
- oracle stable for 2 hops: `53.85%`
- oracle stable for 3 hops: `38.46%`
- oracle stable for 4 hops: `38.46%`
- model flip given oracle stable: `0.0%` at all measured horizons
- model unnecessary flip rate: `0.0%` at all measured horizons

### `heavy_dynamic`

- target decisions: `1`
- oracle hub defined rate: `0.0%`
- model hub defined rate: `0.0%`

## Interpretation

The remaining hard near-tie frontier does not currently look like a problem of
the model repeatedly abandoning a stable route preference.

What the audit says:

- some hard near-tie decisions do admit an oracle hub preference
- that preference is only moderately persistent even in `deeper_packets6`
- the current model is not showing gratuitous hub-flip behavior when the oracle
  route preference stays stable

So the main gap is still local ambiguous decision quality, not obviously missing
short-lived route commitment.

## Verdict

Do not open a route-option branch at this stage.

A route-persistence branch would need new evidence that:

- the frontier slice expands materially beyond this small audit sample, and
- model flip behavior becomes a real error source rather than near zero

Until then, route options stay behind:

- conditional compute
- offline branch-teacher supervision
- adaptive continuation / compute distillation
