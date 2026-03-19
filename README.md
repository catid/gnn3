# gnn3

Packet-switched GNN routing research stack for Hidden-Corridor synthetic routing benchmarks,
Packet-Mamba3 backbones, selective communication variants, and RSM-inspired recursive
refinement training.

## Structure

- `src/gnn3/`: library code
- `configs/`: experiment configs
- `scripts/`: entry points and automation
- `tests/`: unit and smoke tests
- `reports/`: plans, notes, and experiment reports
- `artifacts/`: local outputs, checkpoints, plots, and logs
- `third_party/`: cloned reference repositories

## Environment

The project targets Python 3.12 with `uv` and PyTorch nightly CUDA 13.0 wheels.
Use `scripts/bootstrap_env.sh` to create and sync the environment.
