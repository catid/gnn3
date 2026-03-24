# Next Best Actions

1. Keep the rebalanced multiheavy curriculum as the robust exploit default. It still beats fresh `E3` in-distribution, and its hard-OOD behavior stays bounded while the combined reranker recipe does not.
2. Do not promote the combined multiheavy plus reranker recipe as the general lead. It wins on the matched baseline suites, but its hard-OOD regret still blows up on deeper-packets stress.
3. Do not promote the bounded traffic-gated reranker. It matches plain multiheavy exactly on all three matched baseline seeds and fixes seed311 OOD, but it still catastrophically fails `deeper_packets6` on seed312 (`1199.23` regret, `4797.20` p95).
4. Do not promote the verifier-filter reranker as the default exploit recipe. Its 3-seed matched mean got worse than plain multiheavy (`1.60` vs `1.32` regret, `6.31` vs `4.77` p95, `43.8%` vs `41.7%` miss) because seed313 regressed.
5. If reranking is revisited, keep only the verifier-backed path-feasibility filter branch. It is the only reranker variant with broad OOD value: 3-seed OOD mean regret `6.45` -> `3.15`, p95 `17.81` -> `11.88`, miss `95.8%` -> `50.0%`.
6. Do not promote either deadline-head add-on. The combined recipe stayed flat on the held-out rollout, and the plain-multiheavy deadline-head rerun remained flat even after fixing best-checkpoint evaluation in the trainer.
7. Preserve the rebalanced `oracle_calibrated` deadline suites and the split-manifest discipline. Old corrected deadline suites remain diagnostic-only because the oracle itself misses them.
8. Keep `detach_warmup` in every shortlist. That remains the strongest causal architectural requirement in the repo.
9. The next highest-value batch is still not another generic architecture branch. It is:
   first, keep pushing exploit work on plain multiheavy;
   second, if reranking continues, test conditional deployment or checkpoint-selection rules that preserve plain multiheavy in-distribution while turning on verifier-filter reranking for hard traffic/depth stress;
   third, revisit auxiliary heads only if they move held-out rollout on plain multiheavy rather than calibration metrics alone.
