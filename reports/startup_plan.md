# Startup Plan

## Milestones

1. Bootstrap a reproducible Python 3.12 + `uv` research environment with test, training, and
   reporting entry points.
2. Implement Hidden-Corridor graph generation, oracle labels, and batch collation with unit tests.
3. Implement a modular Packet-Mamba3 baseline and the selective communication variants behind
   config switches.
4. Add RSM-inspired recursive refinement training, evaluation diagnostics, and two-GPU scheduling.
5. Run the first exploit/explore matrix and consolidate findings into progress and next-step reports.

## Risks

- PyTorch nightly and third-party graph packages may drift; keep the core stack pure PyTorch first.
- Mamba-3 reference code may lag behind the paper framing; keep the transition operator modular.
- Recursive refinement can hide silent training failures; rely on tiny overfit and 3-seed gates.
- Two-GPU throughput can regress if the dataloader or scheduler is underspecified.

## Gating Criteria

- Unit tests pass for graph generation, oracle correctness, collation, and model forwards.
- The baseline can overfit a tiny slice and finish a short smoke train without instability.
- Every new direction clears a 3-seed short comparison before it consumes larger GPU budgets.
- Exploit/explore GPU-hours stay within the target 45/55 to 55/45 band.
