# Continuation Report Round 5

## Scope

This pass tested the smallest non-reranker exploit change suggested by round four: keep the robust plain `multiheavy` recipe fixed, but change checkpoint selection to weight tail regret and deadline behavior more heavily than solved rate or local decision accuracy.

Key artifacts:

- [round5_multiheavy_tail_select_vs_multiheavy.csv](/home/catid/gnn3/reports/plots/round5_multiheavy_tail_select_vs_multiheavy.csv)
- [round5_multiheavy_tail_select_vs_multiheavy.png](/home/catid/gnn3/reports/plots/round5_multiheavy_tail_select_vs_multiheavy.png)
- [experiment_summary.csv](/home/catid/gnn3/reports/plots/experiment_summary.csv)

Implementation changes:

- configurable checkpoint-selection weights in [config.py](/home/catid/gnn3/src/gnn3/train/config.py)
- selection-score wiring in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py)
- matched round-five configs in [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed311.yaml), [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed312.yaml), and [e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed313.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed313.yaml)
- selection regression tests in [test_trainer_selection.py](/home/catid/gnn3/tests/test_trainer_selection.py)

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

## Decision

This is a clean negative result.

Checkpoint-selection policy alone is not the next leverage point for this repo. It can move `selected_epoch`, but it does not move the actual held-out rollout once the current plain `multiheavy` model has trained.

Updated recommendation:

1. Keep plain `multiheavy` as the exploit default.
2. Do not spend more time on rerankers or selector-only tuning for now.
3. If exploit work continues, the next lever must change the learned policy or the training signal itself, not only checkpoint ranking.
