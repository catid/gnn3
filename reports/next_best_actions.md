# Next Best Actions

1. Keep the rebalanced multiheavy curriculum as the robust exploit default. It still beats fresh `E3` in-distribution, and its hard-OOD behavior stays bounded while the combined reranker recipe does not.
2. Do not promote the combined multiheavy plus reranker recipe as the general lead. It wins on the matched baseline suites, but its mean hard-OOD regret is `74.34` vs `6.45` for plain multiheavy because deeper-packets OOD blows up on seeds `311` and `312`.
3. Keep the combined reranker recipe as an in-distribution contender only. It is still the best bounded path-level extension on the base suite, but it now needs explicit OOD stabilization before further promotion.
4. Do not promote the standalone reranker as a separate recipe. Its third matched seed was a clean negative (`2.47` regret, `9.20` p95, `68.8%` miss), so the additive value only appears inside the combined recipe.
5. Do not promote the combined deadline-head add-on. On the actual seed311 test rollout it matched the combined baseline exactly, despite improving auxiliary calibration metrics.
6. Preserve the rebalanced `oracle_calibrated` deadline suites and the split-manifest discipline. Old corrected deadline suites remain diagnostic-only because the oracle itself misses them.
7. Keep `detach_warmup` in every shortlist. That remains the strongest causal architectural requirement in the repo.
8. The next highest-value batch is not another generic architecture branch. It is:
   first, stabilize or gate the reranker under deeper/heavier OOD pressure;
   second, if auxiliary heads are revisited, test them on plain multiheavy or as explicit reranker regularizers;
   third, only then, revisit broader architectural movement.
