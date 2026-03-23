# Next Best Actions

1. Keep `E3` as the lead baseline, but only on the rebalanced `oracle_calibrated` deadline suites. The original corrected suites are oracle-impossible and should be treated as benchmark diagnostics, not ranking evidence.
2. Preserve the split-manifest discipline from round three and the round-four deadline rebalance. Future runs should keep persisted manifests, split-specific seeds, and explicit suite identifiers in every report.
3. Keep `detach_warmup` in every exploit-side shortlist. That remains the strongest causal architectural result in the repo.
4. If one exploit-side branch deserves another contender batch, it is `A2` deadline/slack modeling rather than a new router. `A2` improved on-time and slack calibration materially, but its common-suite best checkpoint still matched `E3` on rollout metrics, so the next move should be checkpointing or training-contract changes that convert calibration gains into lower miss rate / p95 regret.
5. Treat candidate-path reranking as the next exploit prototype if `A2` still cannot move rollout quality. The unresolved failure mode is path-level deadline behavior under load, and the remaining open reranker issue is a better fit for that than more generic communication changes.
6. Keep verifier-backed refinement bounded. `A4` showed no fair common-suite gain over `E3`; only revisit it if a feasibility/slack auxiliary target can be shown to improve hard-case ranking rather than just local calibration.
7. Do not promote hazard memory from round four. The `B1` scout had positive standalone signal, but matched best-checkpoint rollout was identical to `E3` on all 3 seeds. If hazard summaries are revisited, do it only after an exploit-side variant produces a real miss-rate or p95-regret win.
