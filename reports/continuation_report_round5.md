# Continuation Report Round 5

## Scope

This pass tested six bounded non-reranker exploit changes on top of the robust plain `multiheavy` recipe:

1. risk-biased checkpoint selection
2. tighter training-only oracle-calibrated deadlines with fixed validation and test manifests
3. deadline-aware soft action targets using existing candidate cost and on-time labels
4. deadline-aware pairwise ranking loss using the same oracle candidate labels
5. feasible-first hard target supervision when an on-time candidate exists
6. slack-critical weighting on the main next-hop CE

Key artifacts:

- [round5_multiheavy_tail_select_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_tail_select_vs_multiheavy.csv)
- [round5_multiheavy_tail_select_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_tail_select_vs_multiheavy.png)
- [round5_multiheavy_tighttrain_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_tighttrain_vs_multiheavy.csv)
- [round5_multiheavy_tighttrain_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_tighttrain_vs_multiheavy.png)
- [round5_multiheavy_softtargets_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_softtargets_vs_multiheavy.csv)
- [round5_multiheavy_softtargets_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_softtargets_vs_multiheavy.png)
- [round5_multiheavy_pairwise_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_pairwise_vs_multiheavy.csv)
- [round5_multiheavy_pairwise_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_pairwise_vs_multiheavy.png)
- [round5_multiheavy_feasible_target_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_feasible_target_vs_multiheavy.csv)
- [round5_multiheavy_feasible_target_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_feasible_target_vs_multiheavy.png)
- [round5_multiheavy_slack_weight_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_slack_weight_vs_multiheavy.csv)
- [round5_multiheavy_slack_weight_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_slack_weight_vs_multiheavy.png)
- [experiment_summary.csv](/home/catid/gnn3/reports/plots/experiment_summary.csv)

Implementation changes:

- configurable checkpoint-selection weights in [config.py](/home/catid/gnn3/src/gnn3/train/config.py)
- selection-score wiring in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py)
- split-specific train/val/test hidden-corridor overrides in [config.py](/home/catid/gnn3/src/gnn3/train/config.py)
- deadline-aware soft-target selection loss in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
- deadline-aware pairwise ranking loss in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
- feasible-first hard-target selection loss in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
- slack-critical weighting on the main selection loss in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
- matched round-five configs in [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed313.yaml)
- matched tighter-train configs in [e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed313.yaml)
- matched soft-target configs in [e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed313.yaml)
- matched pairwise configs in [e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed313.yaml)
- matched feasible-target configs in [e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed313.yaml)
- matched slack-weight configs in [e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed313.yaml)
- selection regression tests in [test_trainer_selection.py](/home/catid/gnn3/tests/test_trainer_selection.py)
- split-override regression coverage in [test_config_split_overrides.py](/home/catid/gnn3/tests/test_config_split_overrides.py)
- soft-target loss regression coverage in [test_loss_coupling.py](/home/catid/gnn3/tests/test_loss_coupling.py)

## Matched Result

Matched three-seed comparison against the current plain `multiheavy` baseline:

- multiheavy mean next-hop accuracy: `95.82%`
- tail-select mean next-hop accuracy: `96.01%`
- multiheavy mean regret: `1.32`
- tail-select mean regret: `1.32`
- multiheavy mean p95 regret: `4.77`
- tail-select mean p95 regret: `4.77`
- multiheavy mean deadline miss rate: `41.7%`
- tail-select mean deadline miss rate: `41.7%`

Per-seed rollout comparison:

- seed `311`: exact match on regret, p95, and miss rate
- seed `312`: exact match on regret, p95, and miss rate
- seed `313`: exact match on regret, p95, and miss rate

What did change:

- the risk-biased selector chose earlier checkpoints on all three seeds: `3 / 2 / 2`
- selected validation rollout looked better than the default selector on some seeds, especially seed `311`

What did not change:

- the held-out test rollout stayed identical to the existing `multiheavy` baseline on all three matched seeds

## Tighter Training-Deadline Scout

Matched three-seed comparison against the same plain `multiheavy` baseline, but with tighter `oracle_calibrated` deadlines applied only to the training split:

- multiheavy mean next-hop accuracy: `95.82%`
- tight-train mean next-hop accuracy: `96.06%`
- multiheavy mean rollout next-hop accuracy: `95.52%`
- tight-train mean rollout next-hop accuracy: `95.52%`
- multiheavy mean regret: `1.32`
- tight-train mean regret: `1.32`
- multiheavy mean p95 regret: `4.77`
- tight-train mean p95 regret: `4.77`
- multiheavy mean deadline miss rate: `41.7%`
- tight-train mean deadline miss rate: `41.7%`

What changed:

- train-manifest hashes changed on every seed, proving the training distribution really was different
- validation/test manifest hashes stayed fixed, preserving the shared comparison contract
- seed `311` found a stronger validation checkpoint on regret and p95 than the baseline

What did not change:

- the held-out test rollout stayed identical to the existing `multiheavy` baseline on all three matched seeds

Take:

- making training deadlines tighter without changing the evaluation manifests is not enough to move the learned policy on held-out rollout
- the next exploit lever must be stronger than train-distribution tightening alone

## Deadline-Aware Soft-Target Scout

Matched three-seed comparison against the same plain `multiheavy` baseline, but with a bounded soft action-target loss built from the existing candidate cost-to-go and on-time labels:

- multiheavy mean next-hop accuracy: `95.82%`
- soft-target mean next-hop accuracy: `96.10%`
- multiheavy mean rollout next-hop accuracy: `95.52%`
- soft-target mean rollout next-hop accuracy: `95.52%`
- multiheavy mean regret: `1.32`
- soft-target mean regret: `1.32`
- multiheavy mean p95 regret: `4.77`
- soft-target mean p95 regret: `4.77`
- multiheavy mean deadline miss rate: `41.7%`
- soft-target mean deadline miss rate: `41.7%`

What changed:

- the trainer now carries an explicit `selection_soft_target_loss` through train, validation, and test
- selected epochs moved to `4 / 2 / 2`
- the auxiliary loss stayed finite on every seed at test time: `0.141`, `0.135`, `0.151`

What did not change:

- the held-out test rollout stayed identical to the existing `multiheavy` baseline on all three matched seeds
- no matched-seed gain appeared in regret, p95 regret, deadline miss rate, or rollout next-hop accuracy

## Deadline-Aware Pairwise Ranking Scout

Matched three-seed comparison against the same plain `multiheavy` baseline, but with a bounded pairwise ranking loss built from the existing candidate cost-to-go, on-time, and slack labels:

- multiheavy mean next-hop accuracy: `95.82%`
- pairwise mean next-hop accuracy: `96.10%`
- multiheavy mean rollout next-hop accuracy: `95.52%`
- pairwise mean rollout next-hop accuracy: `95.52%`
- multiheavy mean regret: `1.32`
- pairwise mean regret: `1.32`
- multiheavy mean p95 regret: `4.77`
- pairwise mean p95 regret: `4.77`
- multiheavy mean deadline miss rate: `41.7%`
- pairwise mean deadline miss rate: `41.7%`

What changed:

- the trainer now carries an explicit `selection_pairwise_loss` through train, validation, and test
- selected epochs moved to `2 / 2 / 3`
- the auxiliary pairwise loss stayed finite on every seed at test time: `0.299`, `0.292`, `0.336`
- seed `313` had a rough first epoch before recovering to the same held-out rollout as baseline

What did not change:

- the held-out test rollout stayed identical to the existing `multiheavy` baseline on all three matched seeds
- no matched-seed gain appeared in regret, p95 regret, deadline miss rate, or rollout next-hop accuracy

## Feasible-First Hard-Target Scout

Matched three-seed comparison against the same plain `multiheavy` baseline, but with an extra hard-target CE term that switches to the lowest-cost on-time candidate whenever one exists:

- multiheavy mean next-hop accuracy: `95.82%`
- feasible-target mean next-hop accuracy: `95.78%`
- multiheavy mean rollout next-hop accuracy: `95.52%`
- feasible-target mean rollout next-hop accuracy: `95.23%`
- multiheavy mean regret: `1.32`
- feasible-target mean regret: `1.39`
- multiheavy mean p95 regret: `4.77`
- feasible-target mean p95 regret: `5.35`
- multiheavy mean deadline miss rate: `41.7%`
- feasible-target mean deadline miss rate: `43.8%`

What changed:

- the trainer now carries an explicit `selection_feasible_target_loss` through train, validation, and test
- selected epochs moved to `3 / 5 / 2`
- seed `311` briefly looked better mid-training, but the final selected checkpoint regressed on held-out rollout

What did not change:

- no matched-seed gain appeared in deadline miss rate, p95 regret, or regret
- seed `312` and seed `313` stayed at their baseline rollout, while seed `311` got worse

## Slack-Critical Weighting Scout

Matched three-seed comparison against the same plain `multiheavy` baseline, but with bounded slack-critical weighting on the main next-hop CE so low-slack decisions get upweighted during training:

- multiheavy mean next-hop accuracy: `95.82%`
- slack-weight mean next-hop accuracy: `96.10%`
- multiheavy mean rollout next-hop accuracy: `95.52%`
- slack-weight mean rollout next-hop accuracy: `95.52%`
- multiheavy mean regret: `1.32`
- slack-weight mean regret: `1.32`
- multiheavy mean p95 regret: `4.77`
- slack-weight mean p95 regret: `4.77`
- multiheavy mean deadline miss rate: `41.7%`
- slack-weight mean deadline miss rate: `41.7%`

What changed:

- the main CE term put more pressure on low-slack decisions during training
- seed `311` and seed `313` both started with very poor early rollout checkpoints before converging back
- selected epochs changed to `4 / 2 / 2`

What did not change:

- the held-out test rollout stayed identical to the existing `multiheavy` baseline on all three matched seeds
- no matched-seed gain appeared in regret, p95 regret, deadline miss rate, or rollout next-hop accuracy

## Decision

All six round-five exploit changes are clean negatives.

Checkpoint-selection policy alone is not the next leverage point for this repo. Tightening the training-only deadline contract also is not enough on its own. Both changes moved internal training behavior, but neither changed the actual held-out rollout once the current plain `multiheavy` model had trained.
Deadline-aware soft action targets also are not enough on their own. They changed selected checkpoints and introduced a stable auxiliary loss, but they still did not move held-out rollout quality.
Deadline-aware pairwise ranking also is not enough on its own. It pushed the policy with a stronger relative-ordering objective than soft targets, but it still converged to the same held-out rollout as plain `multiheavy`.
Feasible-first hard-target supervision is also not enough on its own. It changed the supervised action target directly when an on-time candidate existed, but the matched held-out rollout still did not improve and slightly worsened overall.
Slack-critical CE weighting is also not enough on its own. It emphasized deadline-sensitive decisions directly in the main loss, but the matched held-out rollout still converged back to plain `multiheavy`.

Updated recommendation:

1. Keep plain `multiheavy` as the exploit default.
2. Do not spend more time on rerankers or selector-only tuning for now.
3. Do not spend another cycle on train-only deadline tightening by itself; it changed train manifests but not held-out rollout.
4. Do not spend another cycle on soft candidate distillation alone; it changed internal losses but not the held-out rollout.
5. Do not spend another cycle on pairwise action-ranking loss alone; it altered the ordering objective directly, but the matched held-out rollout still stayed flat.
6. Do not spend another cycle on feasible-first hard targets alone; they changed the supervised choice directly but still failed to improve the matched held-out rollout.
7. Do not spend another cycle on slack-critical CE weighting alone; it changed decision pressure inside the main loss, but the matched held-out rollout still stayed flat.
8. If exploit work continues, the next lever must change the learned policy more materially than checkpoint ranking, train-distribution tightening, soft candidate coupling, pairwise ranking, feasible-first hard targets, or slack-critical CE weighting alone.
