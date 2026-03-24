# Next Best Actions

1. Keep the rebalanced multiheavy curriculum as the robust exploit default. It still beats fresh `E3` in-distribution, and its hard-OOD behavior stays bounded while the combined reranker recipe does not.
2. Do not promote any reranker recipe. The combined reranker, traffic-gated reranker, verifier-filter reranker, and shared-config conditional deployment rule all failed the actual deployment bar.
3. Treat the earlier verifier-filter seed312 rescue as a targeted diagnostic only. It exposed a path-feasibility failure mode, but the gain did not survive a shared-config comparison.
4. Do not promote either deadline-head add-on. The combined recipe stayed flat on the held-out rollout, and the plain-multiheavy deadline-head rerun remained flat even after fixing best-checkpoint evaluation in the trainer.
5. Preserve the rebalanced `oracle_calibrated` deadline suites and the split-manifest discipline. Old corrected deadline suites remain diagnostic-only because the oracle itself misses them.
6. Keep `detach_warmup` in every shortlist. That remains the strongest causal architectural requirement in the repo.
7. The next highest-value batch is no longer another reranker follow-up. It is:
   first, keep pushing exploit work on plain multiheavy;
   second, focus on non-reranker ways to improve tail regret and deadline behavior on the shared suites;
   third, revisit auxiliary heads only if they move held-out rollout on plain multiheavy rather than calibration metrics alone.
