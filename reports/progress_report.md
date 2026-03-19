# Progress Report

## What Was Built

- Reproducible Python 3.12 + `uv` environment with pinned PyTorch nightly CUDA 13.0 wheels in [requirements/torch-nightly-cu130.txt](/home/catid/gnn3/requirements/torch-nightly-cu130.txt).
- Research project scaffold with `src/`, `configs/`, `scripts/`, `tests/`, `artifacts/`, `reports/`, and `third_party/`.
- Reference fetch pipeline in [scripts/fetch_references.py](/home/catid/gnn3/scripts/fetch_references.py), cloned repos in `third_party/`, and hand-fixed cleaned markdown for the highest-priority papers in [reports/papers/md_clean](/home/catid/gnn3/reports/papers/md_clean).
- Hidden-Corridor Packet Routing benchmark, oracle, dataset flattening, and dense batch collation in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py).
- Packet-Mamba backbone with local edge mixing, bidirectional scan orderings, selective communication variants, RSM-style outer refinement hooks, and outer-round history reads in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py).
- Config-driven training, regret-aware checkpoint selection, rollout evaluation, DDP support, experiment logging, and plotting in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py), [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py), and [plot_experiment_summaries.py](/home/catid/gnn3/scripts/plot_experiment_summaries.py).

## Validation

- Unit and smoke coverage passes in [test_hidden_corridor.py](/home/catid/gnn3/tests/test_hidden_corridor.py): graph generation, oracle path choice, collation, model forward, and CPU smoke training.
- Both RTX 5090 GPUs were used in paired single-GPU sweep mode throughout the empirical loop.
- The `torchrun` path was validated with a 2-GPU DDP smoke config in [smoke_e3_ddp.yaml](/home/catid/gnn3/configs/experiments/smoke_e3_ddp.yaml) and [run_train_ddp.sh](/home/catid/gnn3/scripts/run_train_ddp.sh).
- Experiment summaries were written under [artifacts/experiments](/home/catid/gnn3/artifacts/experiments) and aggregated into [experiment_summary.csv](/home/catid/gnn3/reports/plots/experiment_summary.csv) and [experiment_summary.png](/home/catid/gnn3/reports/plots/experiment_summary.png).

## Key Results

Short-run benchmark results:

| direction | bucket | runs | next-hop accuracy | rollout solved rate | avg regret | take |
| --- | --- | ---: | --- | --- | --- | --- |
| E1 local-only baseline | exploit | 1 | 94.5% | 100% | 11.47 | Baseline works and trains cleanly. |
| E2 selective read | exploit | 3 | 94.2% to 96.7% | 100% in all runs | 1.51 to 2.57 | Safe improvement over baseline and reasonably stable across seeds. |
| E3 memory hubs + RSM | exploit | 3 | 96.7% to 97.4% | 100% in all runs | 0.037 to 1.18 | Strongest exploit path and now replicated across seeds. |
| X1 selective forward | explore | 6 | 93.2% to 96.5% | 100% in all runs | 0.49 to 5.89 | Promising risky direction; clearly better than baseline, but higher variance than E2. |
| X2 forward + read | explore | 1 | 93.1% | 100% | 8.63 | Negative result relative to X1 and E2 on the short budget. |
| X4 outer-round-history read | explore | 2 | 95.4% to 96.6% | 100% in both runs | 0.48 to 5.07 | Works, but is currently less reliable than E3 and less consistently strong than X1. |

Interpretation:

- The baseline is solid enough to trust the benchmark and trainer.
- Selective read is the safest improvement and consistently lowers regret.
- Memory hubs plus detached warm-up refinement are currently the strongest exploit-side path and now have multi-seed support.
- Forward-only communication is the most interesting exploration result: it works, but its variance suggests the mechanism is less stable than selective read.
- Forward + read does not justify more budget yet.
- Outer-round-history reads are viable but not yet convincing enough to outrank either `E3` or the best `X1` runs.

## GPU Portfolio

- Exploit GPU-hours: `0.3809`
- Explore GPU-hours: `0.3651`
- Split: `51.1% / 48.9%`

This remains inside the target operating band after `E3` replication, `X4` exploration, and the 2-GPU DDP smoke.

## Current Gaps

- `X1` is promising but variable; it needs either warm-starting or a better checkpoint selection criterion before longer budgets.
- `X3` and `X5` are still unimplemented.
- `X4` is implemented, but it needs at least one more short seed or a warm-started run before it should consume larger budgets.
- DDP is wired and smoke-tested, but not yet used for a real scaled shortlist run.
