# Next Best Actions

1. Keep the rebalanced multiheavy curriculum as the robust exploit default. It still beats fresh `E3` in-distribution, and its hard-OOD behavior stays bounded while the combined reranker recipe does not.
2. Do not promote the combined multiheavy plus reranker recipe as the general lead. It wins on the matched baseline suites, but its hard-OOD regret still blows up on deeper-packets stress.
3. Do not promote the new bounded traffic-gated reranker either. It matches plain multiheavy exactly on all three matched baseline seeds and fixes seed311 OOD, but it still catastrophically fails `deeper_packets6` on seed312 (`1199.23` regret, `4797.20` p95).
4. Do not promote either deadline-head add-on. The combined recipe stayed flat on the held-out rollout, and the plain-multiheavy deadline-head scout also matched multiheavy exactly on seed311 despite clean auxiliary calibration metrics.
5. Preserve the rebalanced `oracle_calibrated` deadline suites and the split-manifest discipline. Old corrected deadline suites remain diagnostic-only because the oracle itself misses them.
6. Keep `detach_warmup` in every shortlist. That remains the strongest causal architectural requirement in the repo.
7. The next highest-value batch is still not another generic architecture branch. It is:
   first, keep pushing exploit work on plain multiheavy;
   second, if reranking is revisited, add explicit path-feasibility or verifier-backed pruning for deeper/heavier traffic instead of another scalar gate;
   third, if auxiliary heads are revisited, require rollout movement on plain multiheavy rather than calibration-only gains.
