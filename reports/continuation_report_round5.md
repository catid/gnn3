# Continuation Report Round 5

## Scope

This pass tested ten bounded exploit changes on top of the robust plain `multiheavy` recipe: nine non-reranker tweaks plus one integrated candidate-path scout that was killed after two seeds.

1. risk-biased checkpoint selection
2. tighter training-only oracle-calibrated deadlines with fixed validation and test manifests
3. train-only `packets_max=6` curriculum with fixed validation and test manifests
4. train-only critical-decision oversampling toward low-slack and multi-packet decisions
5. deadline-aware soft action targets using existing candidate cost and on-time labels
6. deadline-aware pairwise ranking loss using the same oracle candidate labels
7. feasible-first hard target supervision when an on-time candidate exists
8. slack-critical weighting on the main next-hop CE
9. bounded DAgger-style state-refresh finetuning on oracle-relabeled model-visited train states
10. supervised traffic-gated candidate-path scoring on the existing integrated path head

Key artifacts:

- [round5_multiheavy_tail_select_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_tail_select_vs_multiheavy.csv)
- [round5_multiheavy_tail_select_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_tail_select_vs_multiheavy.png)
- [round5_multiheavy_tighttrain_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_tighttrain_vs_multiheavy.csv)
- [round5_multiheavy_tighttrain_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_tighttrain_vs_multiheavy.png)
- [round5_multiheavy_packets6_train_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_packets6_train_vs_multiheavy.csv)
- [round5_multiheavy_packets6_train_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_packets6_train_vs_multiheavy.png)
- [round5_multiheavy_critical_sampling_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_critical_sampling_vs_multiheavy.csv)
- [round5_multiheavy_critical_sampling_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_critical_sampling_vs_multiheavy.png)
- [round5_multiheavy_softtargets_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_softtargets_vs_multiheavy.csv)
- [round5_multiheavy_softtargets_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_softtargets_vs_multiheavy.png)
- [round5_multiheavy_pairwise_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_pairwise_vs_multiheavy.csv)
- [round5_multiheavy_pairwise_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_pairwise_vs_multiheavy.png)
- [round5_multiheavy_feasible_target_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_feasible_target_vs_multiheavy.csv)
- [round5_multiheavy_feasible_target_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_feasible_target_vs_multiheavy.png)
- [round5_multiheavy_slack_weight_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_slack_weight_vs_multiheavy.csv)
- [round5_multiheavy_slack_weight_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_slack_weight_vs_multiheavy.png)
- [round5_multiheavy_dagger_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_dagger_vs_multiheavy.csv)
- [round5_multiheavy_dagger_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_dagger_vs_multiheavy.png)
- [round5_multiheavy_pathhead_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_pathhead_vs_multiheavy.csv)
- [round5_multiheavy_pathhead_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_pathhead_vs_multiheavy.png)
- [experiment_summary.csv](/home/catid/gnn3/reports/plots/experiment_summary.csv)

Implementation changes:

- configurable checkpoint-selection weights in [config.py](/home/catid/gnn3/src/gnn3/train/config.py)
- selection-score wiring in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py)
- split-specific train/val/test hidden-corridor overrides in [config.py](/home/catid/gnn3/src/gnn3/train/config.py)
- deadline-aware soft-target selection loss in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
- deadline-aware pairwise ranking loss in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
- feasible-first hard-target selection loss in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
- slack-critical weighting on the main selection loss in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
- bounded DAgger refresh knobs in [config.py](/home/catid/gnn3/src/gnn3/train/config.py)
- oracle relabel collection for model-visited train states in [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py)
- bounded DAgger finetune loop in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py)
- supervised path-head soft-target loss in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
- matched round-five configs in [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed313.yaml)
- matched tighter-train configs in [e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed313.yaml)
- matched packets6-train configs in [e3_memory_hubs_rsm_round5_multiheavy_packets6_train_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_packets6_train_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_packets6_train_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_packets6_train_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_packets6_train_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_packets6_train_seed313.yaml)
- matched critical-sampling configs in [e3_memory_hubs_rsm_round5_multiheavy_critical_sampling_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_critical_sampling_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_critical_sampling_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_critical_sampling_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_critical_sampling_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_critical_sampling_seed313.yaml)
- matched soft-target configs in [e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_softtargets_seed313.yaml)
- matched pairwise configs in [e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_pairwise_seed313.yaml)
- matched feasible-target configs in [e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_feasible_target_seed313.yaml)
- matched slack-weight configs in [e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_slack_weight_seed313.yaml)
- matched DAgger-refresh configs in [e3_memory_hubs_rsm_round5_multiheavy_dagger_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_dagger_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_dagger_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_dagger_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_dagger_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_dagger_seed313.yaml)
- killed-early path-head scout configs in [e3_memory_hubs_rsm_round5_multiheavy_pathhead_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_pathhead_seed311.yaml) and [e3_memory_hubs_rsm_round5_multiheavy_pathhead_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_pathhead_seed312.yaml)
- selection regression tests in [test_trainer_selection.py](/home/catid/gnn3/tests/test_trainer_selection.py)
- split-override regression coverage in [test_config_split_overrides.py](/home/catid/gnn3/tests/test_config_split_overrides.py)
- critical-sampling and DAgger-refresh regression coverage in [test_train_sampling.py](/home/catid/gnn3/tests/test_train_sampling.py)
- soft-target and path-head loss regression coverage in [test_loss_coupling.py](/home/catid/gnn3/tests/test_loss_coupling.py)

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

## Packets6 Train-Only Curriculum Scout

Matched three-seed comparison against the same plain `multiheavy` baseline, but with the training split packet cap increased from `4` to `6` while validation and test manifests stayed fixed:

- multiheavy mean next-hop accuracy: `95.82%`
- packets6-train mean next-hop accuracy: `96.10%`
- multiheavy mean rollout next-hop accuracy: `95.52%`
- packets6-train mean rollout next-hop accuracy: `95.52%`
- multiheavy mean regret: `1.32`
- packets6-train mean regret: `1.32`
- multiheavy mean p95 regret: `4.77`
- packets6-train mean p95 regret: `4.77`
- multiheavy mean deadline miss rate: `41.7%`
- packets6-train mean deadline miss rate: `41.7%`

What changed:

- the train-manifest hashes changed on every seed, proving the train split really did include heavier packet-count episodes
- seed `311` and seed `313` both spent early epochs in visibly worse regions before snapping back to the shared baseline rollout
- mean test next-hop accuracy on the static held-out decision set rose slightly from `95.82%` to `96.10%`

What did not change:

- the held-out test rollout stayed identical to the existing `multiheavy` baseline on all three matched seeds
- no matched-seed gain appeared in regret, p95 regret, deadline miss rate, or rollout next-hop accuracy

Take:

- harder train-only packet-count pressure is not enough on its own to move the learned rollout policy under the fixed shared evaluation manifests
- this closes off another pure-curriculum lever unless it is paired with a stronger objective or policy change

## Critical-Decision Oversampling Scout

Matched three-seed comparison against the same plain `multiheavy` baseline, but with train-only decision-level oversampling toward low-slack, infeasible, and higher-packet-count decisions:

- multiheavy mean next-hop accuracy: `95.82%`
- critical-sampling mean next-hop accuracy: `96.10%`
- multiheavy mean rollout next-hop accuracy: `95.52%`
- critical-sampling mean rollout next-hop accuracy: `95.52%`
- multiheavy mean regret: `1.32`
- critical-sampling mean regret: `1.32`
- multiheavy mean p95 regret: `4.77`
- critical-sampling mean p95 regret: `4.77`
- multiheavy mean deadline miss rate: `41.7%`
- critical-sampling mean deadline miss rate: `41.7%`

What changed:

- the train sampler materially changed minibatch pressure instead of just changing scalar loss weights
- mean train sample weight was about `1.83x`, with weights clipped up to `4.0x`
- seed `311` and seed `312` both found stronger validation rollout checkpoints than their plain-multiheavy baselines before snapping back on held-out test rollout

What did not change:

- the held-out test rollout stayed identical to the existing `multiheavy` baseline on all three matched seeds
- no matched-seed gain appeared in regret, p95 regret, deadline miss rate, or rollout next-hop accuracy

Take:

- even materially harder replay of low-slack and multi-packet decisions was not enough to move the final learned policy under the current training contract
- this is a stronger negative than the earlier loss-only tweaks because the training distribution inside each epoch actually changed

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

## DAgger State-Refresh Scout

Matched three-seed comparison against the same plain `multiheavy` baseline, but with a bounded DAgger-style finetune that relabels model-visited train states with oracle next-hop targets and then runs two short refresh epochs:

- multiheavy mean next-hop accuracy: `95.82%`
- dagger mean next-hop accuracy: `96.10%`
- multiheavy mean rollout next-hop accuracy: `95.52%`
- dagger mean rollout next-hop accuracy: `95.52%`
- multiheavy mean regret: `1.32`
- dagger mean regret: `1.32`
- multiheavy mean p95 regret: `4.77`
- dagger mean p95 regret: `4.77`
- multiheavy mean deadline miss rate: `41.7%`
- dagger mean deadline miss rate: `41.7%`

What changed:

- the trainer collected and mixed back `338 / 365 / 324` oracle-relabeled model-visited decisions, `1,027` total
- each seed ran an explicit DAgger finetune phase with separate `phase: dagger` metrics and refresh-decision counts in the artifact summaries
- the selected checkpoints stayed in the base phase at `3 / 2 / 3`, showing the DAgger pass never beat the best pre-refresh checkpoint

What did not change:

- the held-out test rollout stayed identical to the existing `multiheavy` baseline on all three matched seeds
- no matched-seed gain appeared in regret, p95 regret, deadline miss rate, or rollout next-hop accuracy
- the extra state-refresh supervision raised static test next-hop accuracy slightly, but that did not translate into a different rollout policy

## Integrated Candidate-Path Head Scout

Killed-after-two-seeds comparison against the same plain `multiheavy` baseline, using the existing traffic-gated `path_reranker_head` inside the model plus a direct soft-target loss on `path_scores`:

- multiheavy mean next-hop accuracy on seeds `311/312`: `94.98%`
- path-head mean next-hop accuracy on seeds `311/312`: `95.40%`
- multiheavy mean rollout next-hop accuracy on seeds `311/312`: `94.85%`
- path-head mean rollout next-hop accuracy on seeds `311/312`: `94.63%`
- multiheavy mean regret on seeds `311/312`: `1.71`
- path-head mean regret on seeds `311/312`: `1.80`
- multiheavy mean p95 regret on seeds `311/312`: `5.70`
- path-head mean p95 regret on seeds `311/312`: `6.58`
- multiheavy mean deadline miss rate on seeds `311/312`: `43.8%`
- path-head mean deadline miss rate on seeds `311/312`: `46.9%`

What changed:

- the model carried a live `path_soft_target_loss` on both seeds, around `0.61` to `0.65` at test time
- seed `312` stayed exactly on the plain `multiheavy` held-out rollout while slightly improving static test next-hop accuracy
- seed `311` regressed on held-out rollout: regret `1.69` vs `1.50`, p95 `7.71` vs `5.96`, miss `50.0%` vs `43.8%`

What did not change:

- the scout did not beat plain `multiheavy` on any matched rollout metric across the first two seeds
- the stronger path-level supervision still did not produce a trustworthy exploit gain, so the branch was killed before seed `313`

## Decision

All ten round-five exploit changes are negatives. Nine reached the full matched three-seed bar, and the integrated path-head scout was killed early after two seeds because it already trailed plain `multiheavy` on held-out rollout.

Checkpoint-selection policy alone is not the next leverage point for this repo. Tightening the training-only deadline contract also is not enough on its own. Both changes moved internal training behavior, but neither changed the actual held-out rollout once the current plain `multiheavy` model had trained.
Deadline-aware soft action targets also are not enough on their own. They changed selected checkpoints and introduced a stable auxiliary loss, but they still did not move held-out rollout quality.
Deadline-aware pairwise ranking also is not enough on its own. It pushed the policy with a stronger relative-ordering objective than soft targets, but it still converged to the same held-out rollout as plain `multiheavy`.
Feasible-first hard-target supervision is also not enough on its own. It changed the supervised action target directly when an on-time candidate existed, but the matched held-out rollout still did not improve and slightly worsened overall.
Slack-critical CE weighting is also not enough on its own. It emphasized deadline-sensitive decisions directly in the main loss, but the matched held-out rollout still converged back to plain `multiheavy`.
Critical-decision oversampling is also not enough on its own. It changed train-time replay pressure much more strongly than the loss-only tweaks, but the matched held-out rollout still converged back to plain `multiheavy`.
Bounded DAgger state refresh is also not enough on its own. It changed the supervised state distribution much more directly than the earlier train-only tweaks, but even after `1,027` oracle-relabeled model-visited refresh decisions, the matched held-out rollout still converged back to plain `multiheavy`.
Supervised path-head coupling is also not enough on its own. It moved path-level supervision into the existing integrated candidate-path head, but the first two matched seeds still underperformed plain `multiheavy` on held-out rollout, so the branch was killed before a third seed.

## Feasible-First Oracle-Policy Note

I scoped out a separate train-only feasible-first oracle-policy branch without spending GPU time. Under the current benchmark contract, on-time feasibility is defined by the same cumulative `_edge_cost` that [shortest_path](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py) already minimizes. That makes a train-only “choose the cheapest on-time path when one exists” oracle rollout algebraically equivalent to the existing cost-first oracle except for exact ties, so it is not a distinct exploit lever in this repo.

Updated recommendation:

1. Keep plain `multiheavy` as the exploit default.
2. Do not spend more time on rerankers or selector-only tuning for now.
3. Do not spend another cycle on train-only deadline tightening by itself; it changed train manifests but not held-out rollout.
4. Do not spend another cycle on train-only packet-cap widening alone; it changed train manifests and early optimization behavior, but the held-out rollout still stayed flat.
5. Do not spend another cycle on train-only critical-decision oversampling alone; it materially changed minibatch pressure, but the held-out rollout still stayed flat.
6. Do not spend another cycle on soft candidate distillation alone; it changed internal losses but not the held-out rollout.
7. Do not spend another cycle on pairwise action-ranking loss alone; it altered the ordering objective directly, but the matched held-out rollout still stayed flat.
8. Do not spend another cycle on feasible-first hard targets alone; they changed the supervised choice directly but still failed to improve the matched held-out rollout.
9. Do not spend another cycle on slack-critical CE weighting alone; it changed decision pressure inside the main loss, but the matched held-out rollout still stayed flat.
10. Do not spend another cycle on bounded DAgger refresh alone; it changed the supervised state distribution directly, but the matched held-out rollout still stayed flat.
11. Do not spend another cycle on supervised path-head coupling alone; the first two matched seeds already trailed plain `multiheavy` on held-out rollout.
12. Do not open a separate train-only feasible-first oracle-policy branch under the current cost/deadline contract; it is equivalent to the existing oracle except for tie cases.
13. If exploit work continues, the next lever must change the learned policy more materially than checkpoint ranking, train-distribution tightening, packet-cap widening, critical-decision oversampling, soft candidate coupling, pairwise ranking, feasible-first hard targets, slack-critical CE weighting, bounded DAgger refresh, or supervised path-head coupling alone.
