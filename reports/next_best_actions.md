# Next Best Actions

1. Replicate `E3` for at least two additional short seeds and keep the same reporting stack so its strong first result is either confirmed or rejected quickly.
2. Warm-start `X1` from `E2` or `E3` checkpoints and compare against training from scratch; the current variance suggests initialization matters.
3. Do not spend more GPU time on `X2` until a concrete mechanism change exists; the current forward+read implementation is dominated by `X1`.
4. Implement `X4` outer-round-history reads next, because the codebase already has orderings, outer refinement, and settling metrics to support it with minimal architecture churn.
5. Change checkpoint selection to include regret and deadline-violation terms, not just accuracy plus solved rate, before longer runs.
6. If a single dominant configuration emerges after `E3` replication, add a DDP config and test `torchrun` on both GPUs for the next scale-up cycle.
