# Oracle Deadline Audit Round 4

## Scope

This audit was run before any round-four architecture work to answer a basic question: are the corrected baseline and hard OOD suites actually on-time feasible under the benchmark's own oracle and cost/deadline contract?

Artifacts:

- original-suite summary: [oracle_deadline_audit_round4_summary.csv](/home/catid/gnn3/reports/plots/oracle_deadline_audit_round4_summary.csv)
- original-suite slack histograms: [oracle_deadline_audit_round4_slack_hist.png](/home/catid/gnn3/reports/plots/oracle_deadline_audit_round4_slack_hist.png)
- original-suite traffic breakdown: [oracle_deadline_audit_round4_traffic_breakdown.csv](/home/catid/gnn3/reports/plots/oracle_deadline_audit_round4_traffic_breakdown.csv)
- original-suite depth breakdown: [oracle_deadline_audit_round4_depth_breakdown.csv](/home/catid/gnn3/reports/plots/oracle_deadline_audit_round4_depth_breakdown.csv)
- rebalanced-suite summary: [oracle_deadline_audit_round4_rebalanced_summary.csv](/home/catid/gnn3/reports/plots/oracle_deadline_audit_round4_rebalanced_summary.csv)
- rebalanced-suite slack histograms: [oracle_deadline_audit_round4_rebalanced_slack_hist.png](/home/catid/gnn3/reports/plots/oracle_deadline_audit_round4_rebalanced_slack_hist.png)
- rebalanced-suite traffic breakdown: [oracle_deadline_audit_round4_rebalanced_traffic_breakdown.csv](/home/catid/gnn3/reports/plots/oracle_deadline_audit_round4_rebalanced_traffic_breakdown.csv)
- rebalanced-suite depth breakdown: [oracle_deadline_audit_round4_rebalanced_depth_breakdown.csv](/home/catid/gnn3/reports/plots/oracle_deadline_audit_round4_rebalanced_depth_breakdown.csv)

## Result

The original corrected suites are not suitable targets for deadline-robustness architecture work.

On the original round-three baseline and hard OOD suites:

- `e3_memory_hubs_rsm_round3_seed211`
  - oracle on-time feasible fraction: `0.00`
  - oracle deadline miss rate: `1.00`
  - median oracle initial slack: `-20.46`
- `a1_e3_ood_branching3_eval`
  - oracle on-time feasible fraction: `0.00`
  - oracle deadline miss rate: `1.00`
  - median oracle initial slack: `-28.72`
- `a1_e3_ood_deeper_packets6_eval`
  - oracle on-time feasible fraction: `0.00`
  - oracle deadline miss rate: `1.00`
  - median oracle initial slack: `-40.14`
- `a1_e3_ood_heavy_dynamic_eval`
  - oracle on-time feasible fraction: `0.00`
  - oracle deadline miss rate: `1.00`
  - median oracle initial slack: `-56.01`

Interpretation:

- the deadline problem in the original corrected suites is dominated by benchmark infeasibility, not model failure
- deadline miss rate from those suites is not a useful model-selection signal
- proceeding directly to richer heads or extra refinement losses on those suites would be cargo-culting against an impossible target

## Controlled Fix

I added an opt-in `deadline_mode: oracle_calibrated` benchmark mode in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py). It preserves the old benchmark behavior by default and only affects new round-four configs.

Calibration rule:

- sample the graph and packet identities as before
- compute a large-budget oracle reference cost in the current dynamic graph
- assign each packet a tight positive slack budget using configurable ratio and absolute slack ranges
- update the working graph packet-by-packet so later deadlines reflect traffic pressure instead of static optimistic costs

This was then used to define new round-four baseline and hard OOD suites:

- [e3_memory_hubs_rsm_round4_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round4_seed311.yaml)
- [a0_e3_ood_branching3_round4_eval.yaml](/home/catid/gnn3/configs/experiments/a0_e3_ood_branching3_round4_eval.yaml)
- [a0_e3_ood_deeper_packets6_round4_eval.yaml](/home/catid/gnn3/configs/experiments/a0_e3_ood_deeper_packets6_round4_eval.yaml)
- [a0_e3_ood_heavy_dynamic_round4_eval.yaml](/home/catid/gnn3/configs/experiments/a0_e3_ood_heavy_dynamic_round4_eval.yaml)

## Rebalanced Suite Status

The new round-four suites are feasible but still tight:

- `e3_memory_hubs_rsm_round4_seed311`
  - oracle on-time feasible fraction: `0.976`
  - oracle deadline miss rate: `0.157`
  - fully feasible episode fraction: `0.958`
  - median oracle initial slack: `1.44`
- `a0_e3_ood_branching3_round4_eval`
  - oracle on-time feasible fraction: `0.945`
  - oracle deadline miss rate: `0.236`
  - fully feasible episode fraction: `0.812`
  - median oracle initial slack: `1.54`
- `a0_e3_ood_deeper_packets6_round4_eval`
  - oracle on-time feasible fraction: `0.932`
  - oracle deadline miss rate: `0.247`
  - fully feasible episode fraction: `0.797`
  - median oracle initial slack: `2.09`
- `a0_e3_ood_heavy_dynamic_round4_eval`
  - oracle on-time feasible fraction: `0.901`
  - oracle deadline miss rate: `0.340`
  - fully feasible episode fraction: `0.578`
  - median oracle initial slack: `2.12`

Interpretation:

- the rebalanced suites now distinguish between feasible-but-hard and truly missed cases
- the heavy dynamic suite remains the strongest deadline-stress target, which is desirable for round four
- these suites are now suitable for:
  - fresh `E3` baseline reruns
  - deadline/slack-aware heads
  - verifier-backed refinement losses
  - a small hazard-memory scout

## Decision

Round four should not use the original corrected suites for deadline-robustness claims.

From this point on:

1. baseline and exploit-side comparisons should use the rebalanced `oracle_calibrated` deadline configs
2. old-suite deadline metrics should be treated as diagnostic evidence of benchmark infeasibility, not model ranking evidence
3. any new architecture or training-contract change must be evaluated first on the rebalanced baseline and hard OOD suites
