# Continuation Audit Round 3

## What Exists Now

- The repo is still driven directly by config dataclasses, not a separate model registry.
  - Experiment registration is effectively `ExperimentConfig` in [config.py](/home/catid/gnn3/src/gnn3/train/config.py).
  - Variant selection is via `PacketMambaConfig` in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py).
- Benchmark and dataset generation live in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py).
  - Episodes are generated on the fly from `HiddenCorridorConfig.seed`.
  - Round three now persists split manifests to `dataset_manifests.json` and records manifest hashes in `metadata.json` and `summary.json`.
  - Split-specific dataset seeds are now resolved in [config.py](/home/catid/gnn3/src/gnn3/train/config.py) so train, val, and test no longer alias the same generator stream.
  - Oracle supervision is the shortest-path / rollout path in `shortest_path()`, `make_decision_record()`, and `oracle_rollout()`.
- Training and logging live in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py).
  - Current outputs already record git commit, git branch, stage, notes, hardware, and device placement.
  - Best checkpoint selection now leans more toward regret and deadline behavior than it did in earlier rounds, but the final summary remains an end-of-training evaluation rather than a best-checkpoint eval.
- Existing launchers are thin:
  - [run_train.py](/home/catid/gnn3/scripts/run_train.py)
  - [run_eval.py](/home/catid/gnn3/scripts/run_eval.py)
  - [run_eval_sweep.py](/home/catid/gnn3/scripts/run_eval_sweep.py)
  - [run_dual_gpu_cycle.sh](/home/catid/gnn3/scripts/run_dual_gpu_cycle.sh)
  - [run_train_ddp.sh](/home/catid/gnn3/scripts/run_train_ddp.sh)
- Existing DDP support is present, but the round-two recommendation to avoid DDP until replay drift is understood is still correct.
- `X6` is implemented in the existing history reader path:
  - `history_read=True`
  - `history_read_mode="summary_bank"`
  - summary-bank logic is in `OuterHistoryReader._read_summary_bank()` in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
- `E3` remains the same memory-hub + detached-warmup configuration family, configured in [e3_memory_hubs_rsm.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm.yaml) and the round-two replay configs.
- Reports and aggregate plots are already centralized in:
  - [portfolio_balance.md](/home/catid/gnn3/reports/portfolio_balance.md)
  - [next_best_actions.md](/home/catid/gnn3/reports/next_best_actions.md)
  - [experiment_summary.csv](/home/catid/gnn3/reports/plots/experiment_summary.csv)

## Round-Three Change Points

- Dataset manifests and replay evidence:
  - [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py)
  - [config.py](/home/catid/gnn3/src/gnn3/train/config.py)
  - [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py)
- Robustness metrics and eval summaries:
  - [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py)
  - [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py)
  - [run_eval.py](/home/catid/gnn3/scripts/run_eval.py)
  - [run_eval_sweep.py](/home/catid/gnn3/scripts/run_eval_sweep.py)
- Matched E3 vs X6 contender configs:
  - `configs/experiments/`
- X6 ablation knobs:
  - [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
  - targeted new configs in `configs/experiments/`
- Round-three reporting:
  - `reports/e3_repro_audit_round3.md`
  - `reports/continuation_report_round3.md`
  - existing summary CSV and plot pipeline under `reports/plots/`

## Plausible Replay-Drift Sources

- The archived-best `E3` artifact predates manifest tracking. It records the same nominal config seed as the later replay, but its decision counts differ (`2254/596/596` vs `2149/541/541`), so it was not trained/evaluated on the same dataset instance.
- No deterministic training mode is enabled. Current training uses CUDA + bf16 autocast, dropout, and stochastic `penultimate_grad_prob` gating without deterministic-algorithm controls.
- The archived-best `E3` run also predates branch/stage/runtime provenance. Its `git_commit` is `unknown`, so there is no direct code-state reconstruction target.
- The checkpoint-selection policy changed after the archived-best run. That affects best-checkpoint evaluation and later OOD sweeps, although it does not by itself explain the final-summary replay gap.
- Before this round, val and test were generated from the same seed and episode count, so they were identical datasets. That leak is now fixed with split seed offsets and persisted manifests.
- The paired launcher does not enforce device isolation with `CUDA_VISIBLE_DEVICES`; it relies on per-config `train.device`. That is workable, but it is one more uncontrolled detail worth keeping explicit in matched runs.

## Planned Round-Three Split

- GPU0 = exploit
  - matched `E3` contender seeds
  - replay-gap audit helpers
  - exploit robustness tuning focused on regret/deadline behavior
- GPU1 = explore
  - matched `X6` contender seeds
  - X6 ablation ladder

Initial round-three assigned GPU-hours:

- exploit: `0.45`
- explore: `0.45`

Planned operating pattern:

- paired single-GPU jobs for matched E3/X6 seeds
- backfill idle GPU time with:
  - exploit OOD / robustness evals on GPU0
  - X6 ablation scouts on GPU1

Target cumulative window remains:

- exploit `45%` to `55%`
- explore `45%` to `55%`
