# Hard-Feasible Action-Gap Audit Round 7

## Scope

- Baseline checkpoint:
  - [e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml)
  - [best.pt](/home/catid/gnn3/artifacts/experiments/e3_memory_hubs_rsm_round7_multiheavy_seed312/checkpoints/best.pt)
- Audited suites:
  - [e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml)
  - [a1_multiheavy_ood_branching3_round7_eval.yaml](/home/catid/gnn3/configs/experiments/a1_multiheavy_ood_branching3_round7_eval.yaml)
  - [a1_multiheavy_ood_deeper_packets6_round7_eval.yaml](/home/catid/gnn3/configs/experiments/a1_multiheavy_ood_deeper_packets6_round7_eval.yaml)
  - [a1_multiheavy_ood_heavy_dynamic_round7_eval.yaml](/home/catid/gnn3/configs/experiments/a1_multiheavy_ood_heavy_dynamic_round7_eval.yaml)
- Artifacts:
  - [round7_hard_feasible_action_gap.json](/home/catid/gnn3/reports/plots/round7_hard_feasible_action_gap.json)
  - [round7_hard_feasible_action_gap_summary.csv](/home/catid/gnn3/reports/plots/round7_hard_feasible_action_gap_summary.csv)
  - [round7_hard_feasible_action_gap_summary.png](/home/catid/gnn3/reports/plots/round7_hard_feasible_action_gap_summary.png)
  - [round7_hard_feasible_action_gap_decisions.csv](/home/catid/gnn3/reports/plots/round7_hard_feasible_action_gap_decisions.csv)
  - [round7_hard_feasible_action_gap_large_gap_manifest.csv](/home/catid/gnn3/reports/plots/round7_hard_feasible_action_gap_large_gap_manifest.csv)

## Main Finding

The original round-seven target was too strict and pointed at the wrong opportunity.

The first four-way hard-feasible definition:

- oracle-feasible
- critical or very-tight slack
- `5+` packets
- high load
- depth `4`

produced an effectively empty constructor gate on the corrected suites. After replacing it with a score-based hard slice, the audit became informative:

- `hard_feasible_case = oracle-feasible and at least 2 of {critical-or-very-tight slack, 5+ packets, depth 4, high load}`
- hard-feasible decisions: `1490`
- hard-feasible episodes: `85`
- thresholded large-gap hard-feasible decisions: `677`
- thresholded large-gap hard-feasible episodes: `35`

That corrected slice closes the key round-seven question:

- plain `multiheavy` is **not** making many mistakes on large-gap hard-feasible states
- its remaining mistakes on the hard slice are concentrated in **near-tie** decisions instead

So the audited constructor opportunity is not “find a branch that overturns obvious large-gap hard-feasible mistakes.” That opportunity is mostly absent in the current suites.

## Where The Baseline Actually Fails

On the corrected score-based hard slice:

- hard-feasible error rate: `2.68%` (`40 / 1490`)
- thresholded large-gap hard-feasible error rate: `0.30%` (`2 / 677`)

By gap bucket inside the hard-feasible slice:

- near-tie: `674` decisions, `37` mistakes, `5.49%` error
- medium-gap: `456` decisions, `3` mistakes, `0.66%` error
- large-gap: `360` decisions, `0` mistakes, `0.00%` error

This is the central round-seven audit result. The hard slice is real, but it is not a large-gap mistake slice. The remaining constructor opportunity is concentrated in ambiguous or near-tie decisions under pressure.

## Where The Hard Slice Lives

The score-based hard-feasible slice is almost entirely in the OOD suites:

- [a1_multiheavy_ood_deeper_packets6_round7_eval.yaml](/home/catid/gnn3/configs/experiments/a1_multiheavy_ood_deeper_packets6_round7_eval.yaml): `735` hard-feasible decisions, baseline accuracy `98.23%`
- [a1_multiheavy_ood_heavy_dynamic_round7_eval.yaml](/home/catid/gnn3/configs/experiments/a1_multiheavy_ood_heavy_dynamic_round7_eval.yaml): `747` hard-feasible decisions, baseline accuracy `96.39%`
- [a1_multiheavy_ood_branching3_round7_eval.yaml](/home/catid/gnn3/configs/experiments/a1_multiheavy_ood_branching3_round7_eval.yaml): `7` hard-feasible decisions, baseline accuracy `100%`
- [e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml): `1` hard-feasible decision, baseline accuracy `100%`

The thresholded large-gap hard-feasible slice also lives almost entirely in the two harder OOD suites:

- `deeper_packets6`: `346` decisions, baseline accuracy `100%`
- `heavy_dynamic`: `331` decisions, baseline accuracy `99.40%`

That is why base-suite-only disagreement compares were not sufficient for round-seven kill decisions.

## Regime Structure Inside The Hard Slice

The corrected summary tables confirm that the hard slice still aligns with the round-six failure regimes:

- `5+` packets carries the biggest large-gap hard-feasible mass: `47.3%`
- `high_load` carries `28.4%`
- depth `4` carries `22.9%`
- `hard_condition_count >= 2` carries `45.4%`

But those conditions do not imply large-gap mistakes. They mostly define where the hard states live, not where the baseline is obviously wrong.

The slack tables make the same point:

- `critical` slack has the highest overall decision-level error rate at `24.4%`
- but those errors are not large-gap; they sit near the knife-edge slice instead

## Implication For Constructor Search

This audit materially changes the round-seven hypothesis.

The working hypothesis before the audit was:

- a better constructor might disagree with `multiheavy` on large-gap hard-feasible states and fix clear baseline errors there

The audit-supported hypothesis after the audit is:

- large-gap hard-feasible states are mostly already solved correctly by `multiheavy`
- the remaining policy opportunity is in hard **near-tie** states where the action gap is small but the constraint pressure is high

That means a constructor branch should only be promoted if it can do one of the following:

1. change policy on the hard near-tie slice and reduce regret or miss there, or
2. expose a new representation that separates suboptimality and feasibility even when the oracle gap is small

It is **not** enough to claim that a branch targets “large-gap hard-feasible” errors, because this audit shows that slice is nearly exhausted already.

## Recommendation

- Keep the score-based hard-feasible slice for round-seven reporting.
- Treat “large-gap hard-feasible disagreement” as a strong but likely sparse diagnostic, not the only gate.
- Use the round-seven constructor kills and the probe audit to decide whether the current plateau is:
  - a representation bottleneck on hard near-tie states, or
  - a constructor/training-dynamic bottleneck where the right signals already exist but the policy still does not move.
