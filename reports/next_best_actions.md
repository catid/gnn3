# Next Best Actions

1. Run one real 2-GPU DDP comparison for the current shortlist winner, which is `E3`, to confirm that scale-up behavior matches the single-GPU sweeps.
2. Warm-start `X1` from `E2` and `E3` checkpoints and compare that against scratch training; the present variance still points to initialization sensitivity.
3. Add one more `X4` seed or a warm-started `X4` run before deciding whether cross-round history reads deserve more budget.
4. Do not spend more GPU time on `X2` until there is a concrete mechanism change; the current forward+read implementation is dominated by both `X1` and `E3`.
5. Implement `X3` shared recurrent stem and `X5` settling-based halting / abstention next, because the current codebase now has the shared transition hooks, outer history path, and settling metrics they need.
