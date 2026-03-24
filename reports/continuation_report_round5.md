# Continuation Report Round 5

## Scope

This pass tested two bounded non-reranker exploit changes on top of the robust plain `multiheavy` recipe:

1. risk-biased checkpoint selection
2. tighter training-only oracle-calibrated deadlines with fixed validation and test manifests

Key artifacts:

- [round5_multiheavy_tail_select_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_tail_select_vs_multiheavy.csv)
- [round5_multiheavy_tail_select_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_tail_select_vs_multiheavy.png)
- [round5_multiheavy_tighttrain_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_tighttrain_vs_multiheavy.csv)
- [round5_multiheavy_tighttrain_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_tighttrain_vs_multiheavy.png)
- [experiment_summary.csv](/home/catid/gnn3/reports/plots/experiment_summary.csv)

Implementation changes:

- configurable checkpoint-selection weights in [config.py](/home/catid/gnn3/src/gnn3/train/config.py)
- selection-score wiring in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py)
- split-specific train/val/test hidden-corridor overrides in [config.py](/home/catid/gnn3/src/gnn3/train/config.py)
- matched round-five configs in [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed313.yaml)
- matched tighter-train configs in [e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tighttrain_seed313.yaml)
- selection regression tests in [test_trainer_selection.py](/home/catid/gnn3/tests/test_trainer_selection.py)
- split-override regression coverage in [test_config_split_overrides.py](/home/catid/gnn3/tests/test_config_split_overrides.py)

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

## Decision

Both round-five exploit changes are clean negatives.

Checkpoint-selection policy alone is not the next leverage point for this repo. Tightening the training-only deadline contract also is not enough on its own. Both changes moved internal training behavior, but neither changed the actual held-out rollout once the current plain `multiheavy` model had trained.

Updated recommendation:

1. Keep plain `multiheavy` as the exploit default.
2. Do not spend more time on rerankers or selector-only tuning for now.
3. Do not spend another cycle on train-only deadline tightening by itself; it changed train manifests but not held-out rollout.
4. If exploit work continues, the next lever must change the learned policy or loss coupling more directly than checkpoint ranking or train-distribution tightening alone.
