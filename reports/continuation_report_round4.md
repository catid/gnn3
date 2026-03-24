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
- [round4_multiheavy_vs_e3.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_vs_e3.csv)
- [round4_multiheavy_vs_e3.png](/home/catid/gnn3/reports/plots/round4_multiheavy_vs_e3.png)
- [round4_path_reranker_vs_e3.csv](/home/catid/gnn3/reports/plots/round4_path_reranker_vs_e3.csv)
- [round4_path_reranker_vs_e3.png](/home/catid/gnn3/reports/plots/round4_path_reranker_vs_e3.png)
- [round4_multiheavy_path_reranker_vs_e3.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_vs_e3.csv)
- [round4_multiheavy_path_reranker_vs_e3.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_vs_e3.png)
- [round4_multiheavy_path_reranker_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_vs_multiheavy.png)
- [round4_combined_deadline_head_vs_combined.csv](/home/catid/gnn3/reports/plots/round4_combined_deadline_head_vs_combined.csv)
- [round4_combined_deadline_head_vs_combined.png](/home/catid/gnn3/reports/plots/round4_combined_deadline_head_vs_combined.png)
- [round4_multiheavy_path_reranker_ood_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_ood_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_ood_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_ood_vs_multiheavy.png)
- [round4_multiheavy_deadline_head_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_deadline_head_vs_multiheavy.csv)
- [round4_multiheavy_deadline_head_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_deadline_head_vs_multiheavy.png)
- [round4_multiheavy_path_reranker_gated_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_gated_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_gated_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_gated_vs_multiheavy.png)
- [round4_multiheavy_path_reranker_gated_ood_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_gated_ood_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_gated_ood_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_gated_ood_vs_multiheavy.png)
- [round4_multiheavy_path_reranker_verifier_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_verifier_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_vs_multiheavy.png)
- [round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy.png)
- [round4_multiheavy_path_reranker_verifier_vs_multiheavy_seed312.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_vs_multiheavy_seed312.csv)
- [round4_multiheavy_path_reranker_verifier_vs_multiheavy_seed312.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_vs_multiheavy_seed312.png)
- [round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy_seed312.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy_seed312.csv)
- [round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy_seed312.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy_seed312.png)
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

## Post-Round Exploit Follow-Up

After the original round-four report was written, I finished the two remaining open exploit-side tasks on the same rebalanced suite family.

### Multiheavy Follow-Up

Artifacts:

- [round4_multiheavy_vs_e3.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_vs_e3.csv)
- [round4_multiheavy_vs_e3.png](/home/catid/gnn3/reports/plots/round4_multiheavy_vs_e3.png)

This closed the older open question about heavier multi-packet curriculum pressure using a real 3-seed batch on the rebalanced suites.

Matched result versus fresh `E3`:

- `E3` mean test next-hop accuracy: `95.57%`
- multiheavy mean test next-hop accuracy: `95.82%`
- `E3` mean regret: `2.25`
- multiheavy mean regret: `1.32`
- `E3` mean p95 regret: `10.48`
- multiheavy mean p95 regret: `4.77`
- `E3` mean deadline miss rate: `54.2%`
- multiheavy mean deadline miss rate: `41.7%`

Per-seed direction:

- seed `311`: regret worsened slightly (`1.50` vs `1.26`) but p95 improved (`5.96` vs `7.50`) and miss rate improved (`43.8%` vs `50.0%`)
- seed `312`: strong win (`1.92` vs `3.22`, `5.45` vs `13.31`, `43.8%` vs `50.0%`)
- seed `313`: strong win (`0.55` vs `2.27`, `2.90` vs `10.63`, `37.5%` vs `62.5%`)

Decision:

- the rebalanced multiheavy curriculum is now the current lead exploit training recipe
- the earlier “mixed at best” multiheavy interpretation from round three does not survive the better deadline contract

### Candidate-Path Reranker

Artifacts:

- [round4_path_reranker_vs_e3.csv](/home/catid/gnn3/reports/plots/round4_path_reranker_vs_e3.csv)
- [round4_path_reranker_vs_e3.png](/home/catid/gnn3/reports/plots/round4_path_reranker_vs_e3.png)

I implemented the bounded path-level extension that was deferred from the first round-four pass. It uses one candidate path per feasible next hop from the existing shortest-path machinery, aggregates path structure into the current `E3` readout, and reranks next hops additively rather than replacing the model family.

Two matched scout seeds against fresh `E3`:

- seed `311`:
  - `E3`: `1.26` regret, `7.50` p95, `50.0%` miss
  - reranker: `1.10` regret, `7.50` p95, `43.8%` miss
- seed `312`:
  - `E3`: `3.22` regret, `13.31` p95, `50.0%` miss
  - reranker: `2.48` regret, `11.25` p95, `43.8%` miss

Two-seed mean:

- matched `E3` mean regret: `2.24`
- reranker mean regret: `1.79`
- matched `E3` mean p95 regret: `10.41`
- reranker mean p95 regret: `9.38`
- matched `E3` mean deadline miss rate: `50.0%`
- reranker mean deadline miss rate: `43.8%`

Decision:

- the reranker implementation is valid and does help in some seeds
- it did not earn standalone promotion
- the third matched seed was a clean negative and was killed early after plateauing at `2.47` regret, `9.20` p95, and `68.8%` miss

### Combined Multiheavy + Candidate-Path Reranker

Artifacts:

- [round4_multiheavy_path_reranker_vs_e3.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_vs_e3.csv)
- [round4_multiheavy_path_reranker_vs_e3.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_vs_e3.png)
- [round4_multiheavy_path_reranker_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_vs_multiheavy.png)

I then ran the actual additive test: keep the rebalanced multiheavy curriculum and add the lightweight reranker on top.

Matched result versus fresh `E3`:

- `E3` mean test next-hop accuracy: `95.57%`
- combined mean test next-hop accuracy: `95.87%`
- `E3` mean regret: `2.25`
- combined mean regret: `1.23`
- `E3` mean p95 regret: `10.48`
- combined mean p95 regret: `4.69`
- `E3` mean deadline miss rate: `54.2%`
- combined mean deadline miss rate: `39.6%`

Direct comparison versus plain multiheavy:

- multiheavy mean test next-hop accuracy: `95.82%`
- combined mean test next-hop accuracy: `95.87%`
- multiheavy mean regret: `1.32`
- combined mean regret: `1.23`
- multiheavy mean p95 regret: `4.77`
- combined mean p95 regret: `4.69`
- multiheavy mean deadline miss rate: `41.7%`
- combined mean deadline miss rate: `39.6%`

Per-seed direction against plain multiheavy:

- seed `311`: combined improved regret (`1.37` vs `1.50`), p95 (`5.71` vs `5.96`), and miss (`37.5%` vs `43.8%`)
- seed `312`: combined improved regret (`1.78` vs `1.92`) while matching p95 (`5.45`) and miss (`43.8%`)
- seed `313`: combined matched multiheavy exactly on rollout (`0.55` regret, `2.90` p95, `37.5%` miss)

Decision:

- the combined multiheavy plus reranker recipe is the lead **in-distribution** exploit contender in the repo
- its gain over plain multiheavy is modest but consistent on the matched baseline suites
- the standalone reranker signal is too unstable to prioritize separately, but the add-on is still worth stress-testing before promotion

### Combined Deadline-Head Add-On

Artifacts:

- [round4_combined_deadline_head_vs_combined.csv](/home/catid/gnn3/reports/plots/round4_combined_deadline_head_vs_combined.csv)
- [round4_combined_deadline_head_vs_combined.png](/home/catid/gnn3/reports/plots/round4_combined_deadline_head_vs_combined.png)

I ran the smallest calibration-style follow-up on top of the combined leader: keep multiheavy plus reranking intact, then add the candidate-level on-time / slack / quantile heads with the existing risk-aware scoring path.

Seed `311` result against the combined baseline:

- combined baseline:
  - `1.37` regret
  - `5.71` p95
  - `37.5%` miss
- combined + deadline head:
  - `1.37` regret
  - `5.71` p95
  - `37.5%` miss

What changed:

- auxiliary metrics improved and the training curve found a better mid-training validation rollout (`1.19` regret, `4.28` p95) than the plain combined model
- the selected test-time checkpoint did **not** improve the actual held-out rollout

Decision:

- this add-on is **not promoted**
- it remains a calibration-positive but rollout-flat scout on the real test split
- if auxiliary heads are revisited again, they should be tested against the plain multiheavy baseline or under stronger checkpoint-selection discipline

### OOD Stress Follow-Up

Artifacts:

- [round4_multiheavy_path_reranker_ood_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_ood_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_ood_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_ood_vs_multiheavy.png)

I then stress-tested the combined model against plain multiheavy on the rebalanced OOD suites.

Mean OOD result across seeds `311/312/313` and suites:

- multiheavy:
  - mean regret: `6.45`
  - mean p95 regret: `17.81`
  - mean deadline miss rate: `95.8%`
  - mean rollout next-hop accuracy: `93.4%`
- combined:
  - mean regret: `74.34`
  - mean p95 regret: `300.64`
  - mean deadline miss rate: `93.8%`
  - mean rollout next-hop accuracy: `91.7%`

Suite-wise mean comparison:

- branching3:
  - multiheavy: `6.89` regret, `15.54` p95
  - combined: `5.04` regret, `12.85` p95
- deeper_packets6:
  - multiheavy: `3.77` regret, `10.03` p95
  - combined: `204.19` regret, `820.68` p95
- heavy_dynamic:
  - multiheavy: `8.69` regret, `27.85` p95
  - combined: `13.80` regret, `68.38` p95

Interpretation:

- the reranker helps on the easier branching OOD suite
- it is catastrophically unstable on deeper-packets OOD for seeds `311` and `312`
- it is also worse than plain multiheavy on the heavy-dynamic suite on average

Decision:

- the combined reranker recipe is **not** the robust exploit default
- plain multiheavy remains the correct exploit baseline for hard OOD work
- the combined model should now be treated as an in-distribution contender that requires OOD stabilization before any broader promotion

### Plain Multiheavy Deadline-Head Recheck

Artifacts:

- [round4_multiheavy_deadline_head_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_deadline_head_vs_multiheavy.csv)
- [round4_multiheavy_deadline_head_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_deadline_head_vs_multiheavy.png)

Before revisiting the plain multiheavy deadline-head path, I fixed a trainer contract bug in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py): the trainer was writing `best.pt` using the selection metric, but the final `summary.json` test rollout was still being reported from the last epoch model rather than from the selected checkpoint. The fix now reloads `best.pt` before final evaluation and records `selected_epoch` plus `selected_selection_score` in the summary.

I then reran the deadline/slack/quantile auxiliary head on the **plain multiheavy** baseline under that corrected best-checkpoint evaluation path instead of the unstable combined reranker path.

Seed `311` result against the robust exploit default:

- multiheavy:
  - `1.50` regret
  - `5.96` p95
  - `43.8%` miss
- multiheavy + deadline head:
  - `1.50` regret
  - `5.96` p95
  - `43.8%` miss

What changed:

- the auxiliary calibration outputs were populated cleanly on the robust baseline
  - on-time Brier: `0.064`
  - slack MAE: `2.91`
  - median-quantile MAE: `2.39`
- the held-out rollout did **not** move at all

Decision:

- this exploit follow-up is a clean negative
- fixing best-checkpoint evaluation did not rescue the auxiliary-head branch
- auxiliary deadline heads are still not worth promoting on the current shortlist unless checkpoint selection or policy coupling changes materially

### Traffic-Gated Reranker Follow-Up

Artifacts:

- [round4_multiheavy_path_reranker_gated_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_gated_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_gated_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_gated_vs_multiheavy.png)
- [round4_multiheavy_path_reranker_gated_ood_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_gated_ood_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_gated_ood_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_gated_ood_vs_multiheavy.png)

I then tested the smallest plausible stabilization for the reranker branch: bound the reranker residual and damp it deterministically under long, queued, low-capacity, high-packet-pressure paths.

Matched baseline-suite result versus plain multiheavy:

- multiheavy mean test next-hop accuracy: `95.82%`
- gated reranker mean test next-hop accuracy: `96.10%`
- multiheavy mean regret: `1.32`
- gated reranker mean regret: `1.32`
- multiheavy mean p95 regret: `4.77`
- gated reranker mean p95 regret: `4.77`
- multiheavy mean deadline miss rate: `41.7%`
- gated reranker mean deadline miss rate: `41.7%`

Per-seed rollout direction:

- seed `311`: exact match to multiheavy
- seed `312`: exact match to multiheavy
- seed `313`: exact match to multiheavy

That means the gate succeeded at making the reranker safe **in-distribution**, but it also removed the original baseline-suite gain.

OOD result across the two historical failure seeds (`311` and `312`):

- multiheavy overall mean regret: `6.56`
- gated reranker overall mean regret: `202.26`
- multiheavy overall mean p95 regret: `17.68`
- gated reranker overall mean p95 regret: `807.64`
- multiheavy overall mean deadline miss rate: `95.8%`
- gated reranker overall mean deadline miss rate: `51.0%`

The mean is dominated by one surviving catastrophic failure:

- seed `311` improved on all three OOD suites
  - branching3 regret: `6.95` -> `3.36`
  - deeper_packets6 regret: `3.61` -> `2.77`
  - heavy_dynamic regret: `6.62` -> `3.82`
- seed `312` improved on branching3 and heavy_dynamic
  - branching3 regret: `6.84` -> `1.98`
  - heavy_dynamic regret: `11.10` -> `2.42`
- but seed `312` still catastrophically failed on deeper_packets6
  - regret: `4.28` -> `1199.23`
  - p95 regret: `11.71` -> `4797.20`
  - solved rate: `1.00` -> `0.9375`

Decision:

- this stabilization is **not** enough to promote any reranker recipe
- it is a useful diagnostic guardrail because it preserved the base-suite behavior and fixed one historical OOD failure seed
- but it still fails the actual acceptance bar because deeper_packets6 seed `312` remains a hard catastrophic case

### Verifier-Filtered Reranker Rescue

Artifacts:

- [round4_multiheavy_path_reranker_verifier_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_verifier_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_vs_multiheavy.png)
- [round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy.csv)
- [round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy.png)
- [round4_multiheavy_path_reranker_verifier_vs_multiheavy_seed312.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_vs_multiheavy_seed312.csv)
- [round4_multiheavy_path_reranker_verifier_vs_multiheavy_seed312.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_vs_multiheavy_seed312.png)
- [round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy_seed312.csv](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy_seed312.csv)
- [round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy_seed312.png](/home/catid/gnn3/reports/plots/round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy_seed312.png)

The next targeted reranker rescue replaced the scalar traffic gate with an explicit verifier-backed feasibility filter. Candidate-path slack and on-time feasibility signals are now threaded into the reranker readout in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py) and [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py), so infeasible candidates can be penalized or hard-masked instead of merely damped.

Matched seed312 in-distribution result versus plain multiheavy:

- multiheavy:
  - `94.73%` test next-hop accuracy
  - `1.92` regret
  - `5.45` p95
  - `43.8%` miss
- verifier-filter reranker:
  - `95.47%` test next-hop accuracy
  - `1.92` regret
  - `5.45` p95
  - `43.8%` miss

That seed312 run was the right in-distribution behavior for the rescue pass: it preserved the robust multiheavy rollout instead of inventing new baseline-suite movement.

Seed312 rebalanced OOD result versus plain multiheavy:

- overall mean:
  - multiheavy: `7.41` regret, `18.90` p95, `95.8%` miss
  - verifier-filter reranker: `2.50` regret, `10.41` p95, `54.2%` miss
- branching3:
  - regret `6.84` -> `1.98`
  - p95 `15.54` -> `7.95`
  - miss `100.0%` -> `56.2%`
- deeper_packets6:
  - regret `4.28` -> `3.09`
  - p95 `11.71` -> `12.00`
  - miss `87.5%` -> `75.0%`
- heavy_dynamic:
  - regret `11.10` -> `2.42`
  - p95 `29.44` -> `11.27`
  - miss `100.0%` -> `31.2%`

Most importantly, this removed the catastrophic `deeper_packets6` seed312 failure that survived the traffic-gated reranker path.

I then expanded the verifier-filter reranker to the remaining matched seeds `311` and `313`.

Three-seed matched result versus plain multiheavy:

- multiheavy mean:
  - `95.82%` test next-hop accuracy
  - `1.32` regret
  - `4.77` p95
  - `41.7%` miss
- verifier-filter reranker mean:
  - `96.06%` test next-hop accuracy
  - `1.60` regret
  - `6.31` p95
  - `43.8%` miss

Per-seed direction:

- seed `311`: exact rollout match to multiheavy
- seed `312`: exact rollout match to multiheavy
- seed `313`: negative in-distribution regression (`1.39` vs `0.55` regret, `7.52` vs `2.90` p95, `43.8%` vs `37.5%` miss)

Three-seed rebalanced OOD result versus plain multiheavy:

- overall mean:
  - multiheavy: `6.45` regret, `17.81` p95, `95.8%` miss
  - verifier-filter reranker: `3.15` regret, `11.88` p95, `50.0%` miss
- branching3 mean:
  - regret `6.89` -> `2.50`
  - p95 `15.54` -> `8.02`
  - miss `100.0%` -> `56.2%`
- deeper_packets6 mean:
  - regret `3.77` -> `3.64`
  - p95 `10.03` -> `15.24`
  - miss `87.5%` -> `68.8%`
- heavy_dynamic mean:
  - regret `8.69` -> `3.31`
  - p95 `27.85` -> `12.38`
  - miss `100.0%` -> `25.0%`

Interpretation:

- the verifier filter is the first reranker variant that holds up as a **broad OOD improvement** across all three confirmation seeds
- it is not a clean default replacement for plain multiheavy because the seed313 matched rollout regressed materially
- the branch now looks like an OOD-specialized path-level module, not a general in-distribution upgrade

Decision:

- this is the first reranker stabilization that materially improves the targeted seed312 OOD failure and still shows strong three-seed OOD gains
- it is **not** promoted over plain multiheavy as the default exploit recipe because the matched three-seed in-distribution mean got worse
- it becomes the highest-value OOD-specialized path-level branch in the repo and the only reranker variant worth carrying forward

## Recommendation

1. Keep the rebalanced `oracle_calibrated` suites as the only valid deadline-robustness ranking target.
2. Keep plain multiheavy as the **robust exploit default**. It remains clearly better than fresh `E3` and it does not show the reranker’s deep-OOD blow-up.
3. Do not promote the original combined reranker recipe or the traffic-gated reranker recipe. Both still fail the hard OOD bar.
4. Do not promote the verifier-filter reranker as the new default either. Its three-seed matched in-distribution mean is worse than plain multiheavy.
5. Keep the verifier-filter reranker as the only reranker direction worth carrying forward. It cuts three-seed OOD mean regret from `6.45` to `3.15` and mean miss rate from `95.8%` to `50.0%`.
6. Do not promote the plain multiheavy deadline-head add-on. It remains flat even after fixing best-checkpoint evaluation in the trainer.
7. If reranking is revisited, use explicit path-feasibility or verifier-backed filtering for deeper/heavier traffic and couple it to conditional deployment or better checkpoint selection so the seed313 in-distribution regression does not become the default behavior.
8. Demote `A2`, `A4`, and `B1` behind the current exploit winners. If auxiliary heads are revisited, do so against plain multiheavy and require rollout movement, not calibration-only improvement.
9. Keep `detach_warmup` untouched.

## Portfolio

- round-four exploit GPU-hours before follow-up closeout: `0.5628`
- round-four explore GPU-hours: `0.2646`
- round-four-plus-follow-up exploit GPU-hours: `1.8310`
- round-four-plus-follow-up explore GPU-hours: `0.7770`
- round-four-plus-follow-up split: `70.2% exploit / 29.8% explore`

This still leans exploit-heavy, but the final closeout moved back toward exploration because the reranker stabilization task required additional multi-seed OOD evidence.

## Final Status

Definition-of-done checklist:

- repo revalidated on corrected manifests: complete
- oracle deadline-feasibility audit: complete
- fresh matched `E3` baseline: complete
- distributional slack / miss head implemented and evaluated: complete
- verifier-backed refinement implemented and evaluated: complete
- single exploration scout implemented and cleanly not promoted: complete
- multiheavy exploit follow-up completed across 3 matched seeds: complete
- standalone candidate-path reranker checked through the negative third seed: complete
- combined multiheavy plus reranker contender completed across 3 matched seeds: complete
- combined deadline-head add-on checked on the lead in-distribution recipe: complete
- combined versus multiheavy OOD stress batch completed across 3 matched seeds: complete
- plain multiheavy deadline-head recheck completed and not promoted: complete
- bounded traffic-gated reranker stabilization completed and not promoted: complete
- trainer best-checkpoint evaluation bug fixed and validated: complete
- verifier-filter reranker three-seed confirmation completed: complete

Round-four conclusion:

- the main new knowledge is that the benchmark deadline contract needed recalibration before model work could be trusted
- after that fix, plain multiheavy became the robust exploit default and it remains the lead recipe after every follow-up in this pass
- auxiliary deadline heads are still calibration-positive but rollout-flat on the shortlist
- path-level reranking is still not promoted as a default, but explicit verifier-backed feasibility filtering is now the first reranker variant that produces strong three-seed OOD gains; it should be treated as an OOD-specialized branch until its seed313 in-distribution regression is explained or eliminated
