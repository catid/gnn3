# Progress Report

## What Was Built

- Reproducible Python 3.12 + `uv` environment with pinned PyTorch nightly CUDA 13.0 wheels in [requirements/torch-nightly-cu130.txt](/home/catid/gnn3/requirements/torch-nightly-cu130.txt).
- Research project scaffold with `src/`, `configs/`, `scripts/`, `tests/`, `artifacts/`, `reports/`, and `third_party/`.
- Reference fetch pipeline in [scripts/fetch_references.py](/home/catid/gnn3/scripts/fetch_references.py), cloned repos in `third_party/`, and hand-fixed cleaned markdown for the highest-priority papers in [reports/papers/md_clean](/home/catid/gnn3/reports/papers/md_clean).
- Hidden-Corridor Packet Routing benchmark, oracle, dataset flattening, and dense batch collation in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py).
- Packet-Mamba backbone with local edge mixing, bidirectional scan orderings, selective communication variants, and RSM-style outer refinement hooks in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py).
- Config-driven training, rollout evaluation, experiment logging, and plotting in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py), [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py), and [plot_experiment_summaries.py](/home/catid/gnn3/scripts/plot_experiment_summaries.py).

## Validation

- Unit and smoke coverage passes in [test_hidden_corridor.py](/home/catid/gnn3/tests/test_hidden_corridor.py): graph generation, oracle path choice, collation, model forward, and CPU smoke training.
- Both RTX 5090 GPUs were used in paired single-GPU sweep mode throughout the empirical loop.
- Experiment summaries were written under [artifacts/experiments](/home/catid/gnn3/artifacts/experiments) and aggregated into [experiment_summary.csv](/home/catid/gnn3/reports/plots/experiment_summary.csv) and [experiment_summary.png](/home/catid/gnn3/reports/plots/experiment_summary.png).

## Key Results

Short-run benchmark results:

| direction | bucket | runs | next-hop accuracy | rollout solved rate | avg regret | take |
| --- | --- | ---: | --- | --- | --- | --- |
| E1 local-only baseline | exploit | 1 | 94.5% | 100% | 11.47 | Baseline works and trains cleanly. |
| E2 selective read | exploit | 3 | 94.2% to 96.7% | 100% in all runs | 1.51 to 2.57 | Safe improvement over baseline and reasonably stable across seeds. |
| E3 memory hubs + RSM | exploit | 1 | 96.7% | 100% | 0.037 | Best exploit result so far, but only one seed and much higher GPU cost. |
| X1 selective forward | explore | 6 | 93.2% to 96.5% | 100% in all runs | 0.49 to 5.89 | Promising risky direction; clearly better than baseline, but higher variance than E2. |
| X2 forward + read | explore | 1 | 93.1% | 100% | 8.63 | Negative result relative to X1 and E2 on the short budget. |

Interpretation:

- The baseline is solid enough to trust the benchmark and trainer.
- Selective read is the safest improvement and consistently lowers regret.
- Memory hubs plus detached warm-up refinement are currently the strongest exploit-side path.
- Forward-only communication is the most interesting exploration result: it works, but its variance suggests the mechanism is less stable than selective read.
- Forward + read does not justify more budget yet.

## GPU Portfolio

- Exploit GPU-hours: `0.1603`
- Explore GPU-hours: `0.1455`
- Split: `52.4% / 47.6%`

This is inside the target operating band after correcting for the longer `E3` run with additional `X1` replication seeds.

## Current Gaps

- `E3` needs at least two more short seeds before it should be treated as the default model to scale.
- `X1` is promising but variable; it needs either warm-starting or a better checkpoint selection criterion before longer budgets.
- `X3`/`X4`/`X5` are not implemented yet, and DDP mode has not been exercised because the paired single-GPU schedule gave better short-loop throughput.
