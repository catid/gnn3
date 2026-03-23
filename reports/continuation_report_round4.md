# Continuation Report Round 4

## Scope

Round four stops treating communication novelty as the main lever. The architecture base is still `E3`, and the goal in this pass was to improve deadline / slack / regret robustness on top of the corrected split discipline from round three.

Key artifact bundle:

- [continuation_audit_round4.md](/home/catid/gnn3/reports/continuation_audit_round4.md)
- [oracle_deadline_audit_round4.md](/home/catid/gnn3/reports/oracle_deadline_audit_round4.md)
- [round4_e3_matched_baseline.csv](/home/catid/gnn3/reports/plots/round4_e3_matched_baseline.csv)
- [round4_e3_matched_baseline.png](/home/catid/gnn3/reports/plots/round4_e3_matched_baseline.png)
- [round4_calibration_curves.csv](/home/catid/gnn3/reports/plots/round4_calibration_curves.csv)
- [round4_calibration_curves.png](/home/catid/gnn3/reports/plots/round4_calibration_curves.png)
- [round4_seed311_variant_compare.csv](/home/catid/gnn3/reports/plots/round4_seed311_variant_compare.csv)
- [round4_seed311_variant_compare.png](/home/catid/gnn3/reports/plots/round4_seed311_variant_compare.png)
- [round4_deadline_p95_compare.csv](/home/catid/gnn3/reports/plots/round4_deadline_p95_compare.csv)
- [round4_deadline_p95_compare.png](/home/catid/gnn3/reports/plots/round4_deadline_p95_compare.png)
- [round4_b1_vs_e3_matched.csv](/home/catid/gnn3/reports/plots/round4_b1_vs_e3_matched.csv)
- [round4_b1_vs_e3_matched.png](/home/catid/gnn3/reports/plots/round4_b1_vs_e3_matched.png)
- [portfolio_usage_round4.csv](/home/catid/gnn3/reports/plots/portfolio_usage_round4.csv)
- [portfolio_usage_round4.png](/home/catid/gnn3/reports/plots/portfolio_usage_round4.png)

## Benchmark Contract Correction

The first round-four result is about the benchmark, not the model.

The original corrected deadline suites from round three were still effectively unusable for deadline-robustness ranking:

- original baseline suite oracle feasible fraction: `0.00`
- original branching OOD oracle feasible fraction: `0.00`
- original deeper-packets OOD oracle feasible fraction: `0.00`
- original heavy-dynamic OOD oracle feasible fraction: `0.00`

That means the `100%` deadline miss rates from round three were dominated by benchmark infeasibility, not model choice. Round four therefore added an opt-in `deadline_mode: oracle_calibrated` path in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py) and reran the exploit work on rebalanced suites.

The rebalanced suites are tight but usable:

- baseline suite oracle feasible fraction: `0.976`
- branching OOD oracle feasible fraction: `0.945`
- deeper-packets OOD oracle feasible fraction: `0.932`
- heavy-dynamic OOD oracle feasible fraction: `0.901`

Decision:

- deadline-robustness claims should use the rebalanced `oracle_calibrated` suites only
- the old corrected suites remain useful only as evidence that the prior deadline contract was too harsh

## Fresh `E3` Baseline

Artifacts:

- [round4_e3_matched_baseline.csv](/home/catid/gnn3/reports/plots/round4_e3_matched_baseline.csv)
- [round4_e3_matched_baseline.png](/home/catid/gnn3/reports/plots/round4_e3_matched_baseline.png)

Fresh three-seed `E3` baseline on the rebalanced suites:

- mean test next-hop accuracy: `95.57%`
- mean regret: `2.25`
- mean p95 regret: `10.48`
- mean deadline miss rate: `54.2%`

Per-seed spread:

- seed `311`: `95.64%` next-hop, `1.26` regret, `7.50` p95, `50.0%` miss
- seed `312`: `94.87%` next-hop, `3.22` regret, `13.31` p95, `50.0%` miss
- seed `313`: `96.18%` next-hop, `2.27` regret, `10.63` p95, `62.5%` miss

Interpretation:

- `E3` remains the correct lead baseline
- the problem is no longer saturated solved-rate
- the real target is tail quality and miss-rate reduction on feasible-but-tight suites

## Exploit Variants

### `A2` Distributional Slack / Miss Head

Artifacts:

- [a2_e3_deadline_head_round4_seed311](/home/catid/gnn3/artifacts/experiments/a2_e3_deadline_head_round4_seed311)
- [round4_calibration_curves.csv](/home/catid/gnn3/reports/plots/round4_calibration_curves.csv)
- [round4_calibration_curves.png](/home/catid/gnn3/reports/plots/round4_calibration_curves.png)
- [round4_seed311_variant_compare.csv](/home/catid/gnn3/reports/plots/round4_seed311_variant_compare.csv)

`A2` added:

- candidate on-time probability
- candidate slack prediction
- candidate cost quantiles
- risk-aware selection scoring

What moved:

- calibration improved materially on the shared seed-311 suite
  - `A2` slack MAE: `2.91`
  - `A2` on-time Brier: `0.070`
  - `A2` median-quantile MAE: `2.995`
- training also exposed a better transient validation checkpoint at epoch `4`
  - validation rollout regret `0.50`
  - validation p95 regret `2.87`
  - validation deadline miss `37.5%`

What did not move:

- the fair common-suite best-checkpoint rollout matched `E3` exactly on seed `311`
  - regret `1.2639`
  - p95 regret `7.504`
  - deadline miss `50.0%`

Take:

- explicit deadline/slack modeling is the strongest exploit-side signal from round four
- it improved calibration but not yet final rollout quality
- the next leverage point is training selection or training contract, not a larger architecture change

### `A4` Verifier-Backed Outer Refinement

Artifacts:

- [a4_e3_verifier_refine_round4_seed311](/home/catid/gnn3/artifacts/experiments/a4_e3_verifier_refine_round4_seed311)
- [round4_deadline_p95_compare.csv](/home/catid/gnn3/reports/plots/round4_deadline_p95_compare.csv)
- [round4_deadline_p95_compare.png](/home/catid/gnn3/reports/plots/round4_deadline_p95_compare.png)

`A4` kept detached warm-up and added bounded verifier-derived auxiliary losses on the last outer steps.

Scout result on its own small split looked positive:

- regret `1.04`
- p95 regret `4.78`
- deadline miss `50.0%`

But the fair shared-suite comparison did not survive:

- common-suite best-checkpoint rollout matched `E3` exactly on seed `311`
  - regret `1.2639`
  - p95 regret `7.504`
  - deadline miss `50.0%`

Decision:

- `A4` is implemented and validated
- it is not promoted from this round

## Exploration Scout

### `B1` Targeted Hazard Memory

Artifacts:

- [b1_e3_hazard_memory_round4_seed311](/home/catid/gnn3/artifacts/experiments/b1_e3_hazard_memory_round4_seed311)
- [b1_e3_hazard_memory_round4_seed312](/home/catid/gnn3/artifacts/experiments/b1_e3_hazard_memory_round4_seed312)
- [b1_e3_hazard_memory_round4_seed313](/home/catid/gnn3/artifacts/experiments/b1_e3_hazard_memory_round4_seed313)
- [round4_b1_vs_e3_matched.csv](/home/catid/gnn3/reports/plots/round4_b1_vs_e3_matched.csv)
- [round4_b1_vs_e3_matched.png](/home/catid/gnn3/reports/plots/round4_b1_vs_e3_matched.png)

This was the only exploration branch in round four. It added a narrow structured hazard summary to the slow / hub pathway instead of another generic communication mechanism.

Standalone scout summaries looked promising enough to justify the 3-seed check:

- seed `311`: `1.04` regret, `4.78` p95, `50.0%` miss
- seed `312`: `3.65` regret, `13.58` p95, `58.3%` miss
- seed `313`: `1.71` regret, `6.78` p95, `58.3%` miss

But the fair matched best-checkpoint comparison against `E3` on the shared baseline suites was flat:

- seed `311`: identical to `E3`
- seed `312`: identical to `E3`
- seed `313`: identical to `E3`

Decision:

- `B1` stays a scoped scout, not a promoted branch
- it is the right level of exploratory complexity if exploration resumes, but there is no evidence to spend more architecture budget on it now

## Candidate-Path Reranking Status

The candidate-path reranker was not implemented in this round. The verifier-backed refinement branch (`A4`) satisfied the round requirement to evaluate at least one bounded path-level or verifier-level extension, and the stronger immediate bottleneck turned out to be benchmark feasibility plus exploit-side calibration conversion.

The reranker remains open as a follow-up because the unresolved failure mode is still path-level deadline choice under load.

## Recommendation

1. Keep `E3` as the lead baseline and keep the rebalanced `oracle_calibrated` suites as the only valid deadline-robustness ranking target.
2. Keep `A2` as the main exploit continuation bet, but do not call it an improvement yet. The evidence supports calibration gains, not rollout gains.
3. If exploit work continues next, target the conversion problem directly:
   - checkpoint selection on miss-rate / p95-aware metrics
   - final-step feasibility/slack objectives
   - path-level reranking rather than more communication
4. Do not promote `A4` or `B1` from round four.
5. Keep `detach_warmup` untouched.

## Portfolio

- round-four exploit GPU-hours: `0.5628`
- round-four explore GPU-hours: `0.2646`
- round-four split: `68.0% exploit / 32.0% explore`
- this is inside the requested `70/30` operating band

## Final Status

Definition-of-done checklist:

- repo revalidated on corrected manifests: complete
- oracle deadline-feasibility audit: complete
- fresh matched `E3` baseline: complete
- distributional slack / miss head implemented and evaluated: complete
- verifier-backed refinement implemented and evaluated: complete
- single exploration scout implemented and cleanly not promoted: complete
- exploit / explore balance kept within target: complete

Round-four conclusion:

- the main new knowledge is that the benchmark deadline contract needed recalibration before model work could be trusted
- after that fix, `E3` remained the lead baseline
- the best remaining architecture-adjacent signal is deadline/slack calibration (`A2`), but it still needs a training / selection contract that converts calibration into lower miss rate and lower tail regret
