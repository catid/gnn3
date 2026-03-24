# Next Best Actions

1. Keep the rebalanced multiheavy curriculum as the robust exploit default. It still beats fresh `E3` in-distribution, and its hard-OOD behavior stays bounded while the combined reranker recipe does not.
2. Do not promote the combined multiheavy plus reranker recipe as the general lead. It wins on the matched baseline suites, but its hard-OOD regret still blows up on deeper-packets stress.
3. Do not promote the bounded traffic-gated reranker. It matches plain multiheavy exactly on all three matched baseline seeds and fixes seed311 OOD, but it still catastrophically fails `deeper_packets6` on seed312 (`1199.23` regret, `4797.20` p95).
4. If reranking is revisited, expand the new verifier-backed path-feasibility filter first. On seed312 it preserved in-distribution multiheavy rollout while cutting rebalanced OOD overall mean regret from `7.41` to `2.50` and deadline miss from `95.8%` to `54.2%`.
5. Do not promote either deadline-head add-on. The combined recipe stayed flat on the held-out rollout, and the plain-multiheavy deadline-head rerun remained flat even after fixing best-checkpoint evaluation in the trainer.
6. Preserve the rebalanced `oracle_calibrated` deadline suites and the split-manifest discipline. Old corrected deadline suites remain diagnostic-only because the oracle itself misses them.
7. Keep `detach_warmup` in every shortlist. That remains the strongest causal architectural requirement in the repo.
8. The next highest-value batch is still not another generic architecture branch. It is:
   first, keep pushing exploit work on plain multiheavy;
   second, expand verifier-backed reranker filtering to seed311/313 matched runs plus paired OOD confirmation;
   third, revisit auxiliary heads only if they move held-out rollout on plain multiheavy rather than calibration metrics alone.
