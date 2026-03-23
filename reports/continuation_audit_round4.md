# Continuation Audit Round 4

## What Exists Now

- Model registration is still config-driven rather than registry-driven.
  - `PacketMambaConfig` in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py) is the effective variant surface.
  - `ExperimentConfig` / `TrainConfig` / `BenchmarkConfig` in [config.py](/home/catid/gnn3/src/gnn3/train/config.py) remain the stable experiment interface.
- Corrected split-manifest discipline is implemented in code, not just reporting.
  - `hidden_corridor_config_for_split()` in [config.py](/home/catid/gnn3/src/gnn3/train/config.py) applies split-specific seed offsets.
  - `train_experiment()` in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py) persists `dataset_manifests.json`, split manifest hashes, and split seeds into `metadata.json` and `summary.json`.
- The current lead baseline is still `E3`.
  - Core architecture lives in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py).
  - The current matched baseline configs are [e3_memory_hubs_rsm_round3_seed211.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round3_seed211.yaml), [e3_memory_hubs_rsm_round3_seed212.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round3_seed212.yaml), and [e3_memory_hubs_rsm_round3_seed213.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round3_seed213.yaml).
- Benchmark and oracle logic are centralized in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py).
  - Deadlines are sampled in `sample_packets()`.
  - Cost-to-go supervision comes from `shortest_path()` and `oracle_rollout()`.
  - Decision batches are assembled by `collate_decisions()`.
- Rollout metrics and regret/deadline evaluation live in [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py).
- Training, checkpoint selection, manifests, and summary writing live in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py).
- Reporting still flows through:
  - experiment `summary.json` files under `artifacts/experiments/`
  - aggregate CSV/plot generation in [plot_experiment_summaries.py](/home/catid/gnn3/scripts/plot_experiment_summaries.py)
  - targeted eval sweeps in [run_eval_sweep.py](/home/catid/gnn3/scripts/run_eval_sweep.py)
  - portfolio tracking in [portfolio_balance.md](/home/catid/gnn3/reports/portfolio_balance.md)

## Exact Round-Four Change Map

- A0 oracle deadline-feasibility audit:
  - add audit helpers in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py) to compute per-episode feasible-on-time status, oracle slack, and suite breakdowns
  - add a small driver script under `scripts/` to emit CSV/JSON/plots from existing configs/manifests
  - write report output to `reports/oracle_deadline_audit_round4.md`
- A1 fresh matched `E3` rerun:
  - add round-four `E3` configs under `configs/experiments/`
  - reuse [run_train.py](/home/catid/gnn3/scripts/run_train.py), [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py), and existing summary aggregation
- A2 distributional slack / miss head:
  - extend `DecisionRecord` + `collate_decisions()` in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py) with derived deadline/slack labels
  - extend `PacketMambaConfig`, `PacketMambaModel`, and `compute_losses()` in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
  - extend validation/report metrics in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py) and possibly [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py) for calibration-aware outputs
- A3 candidate-path reranker / constrained selector:
  - path candidate generation should live next to the oracle in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py)
  - reranker model glue should live in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py) as an additive path scorer, not a separate model family
  - rollout-time path selection should be integrated in [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py)
- A4 verifier-backed outer refinement:
  - final-step auxiliary targets and losses belong in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
  - any verifier-derived labels should be computed in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py)
  - final-step-only training contract should remain centered in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py)
- B1 targeted hazard-memory scout:
  - if implemented architecturally, add a minimal structured side channel in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
  - derive compact hazard features from graph/packet state in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py)
  - keep config exposure additive through `PacketMambaConfig`
- Round-four plots/reports:
  - extend reporting helpers in `scripts/`
  - update [portfolio_balance.md](/home/catid/gnn3/reports/portfolio_balance.md)
  - update [next_best_actions.md](/home/catid/gnn3/reports/next_best_actions.md)
  - write [continuation_report_round4.md](/home/catid/gnn3/reports/continuation_report_round4.md)

## Likely Remaining Sources Of Variance

- small training budgets and only moderate per-seed episode counts
- per-graph deadline sampling in `sample_packets()` may create wide feasible/infeasible mix shifts across seeds
- rollout evaluation currently uses only `rollout_eval_episodes` slices, so tail metrics can move materially with small seed changes
- non-deterministic CUDA behavior is still possible even though split manifests are now fixed
- checkpoint selection still blends decision metrics with rollout metrics; this can prefer slightly different local minima than a pure deadline/regret target

## Metrics That Are Already Trustworthy

- split manifest hashes and split seed provenance
- next-hop accuracy on decision datasets
- average regret, p95 regret, worst regret, and deadline miss rate from [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py)
- cumulative GPU-hours and experiment bucket accounting from artifact summaries and [portfolio_balance.md](/home/catid/gnn3/reports/portfolio_balance.md)
- the causal importance of `detach_warmup`

## Metrics That Need Better Instrumentation

- oracle on-time feasibility by suite and by traffic/depth slice
- slack calibration quality rather than only value MAE / RMSE
- cost calibration quality in the high-regret tail
- verifier-confirmed on-time feasible fraction for model rollouts
- deadline miss severity decomposition:
  - impossible-under-oracle vs feasible-but-missed
  - first failing packet / first failing region
- risk-sensitive selection metrics for checkpoint ranking once deadline-aware heads exist

## Planned GPU-Hour Split

- Target split for this round: `70% exploit / 30% explore`
- Planned exploit usage:
  - A0 oracle audit
  - A1 matched `E3` rerun
  - A2 distributional slack / miss head
  - A3 candidate-path reranker
  - A4 verifier-backed outer refinement
- Planned explore usage:
  - B1 targeted hazard-memory scout only
- Operational policy:
  - GPU0 stays on exploit training/eval unless blocked
  - GPU1 runs the single exploration scout only after A0/A1 are stable; otherwise it backfills exploit reruns, OOD evals, and ablations
- Portfolio guardrail:
  - keep cumulative usage within `65–75% exploit` and `25–35% explore`
