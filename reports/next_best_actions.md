# Next Best Actions

1. Promote the rebalanced multiheavy curriculum from scout to the current lead exploit recipe. Across matched seeds `311/312/313`, it beat the fresh `E3` baseline on mean regret (`1.32` vs `2.25`), mean p95 regret (`4.77` vs `10.48`), and mean deadline miss (`41.7%` vs `54.2%`).
2. Promote the lightweight candidate-path reranker to the next contender issue. On matched seeds `311/312`, it improved regret (`1.79` vs `2.24`), p95 regret (`9.38` vs `10.41`), and deadline miss (`43.8%` vs `50.0%`) over the same-seed `E3` baseline with similar test next-hop accuracy.
3. Preserve the rebalanced `oracle_calibrated` deadline suites and the split-manifest discipline. Old corrected deadline suites remain diagnostic-only because the oracle itself misses them.
4. Keep `detach_warmup` in every shortlist. That remains the strongest causal architectural requirement in the repo.
5. Deprioritize `A2`, `A4`, and `B1` relative to the new exploit winners. `A2` still matters as a calibration mechanism, but multiheavy and reranking now have direct rollout wins; `A4` and `B1` do not.
6. The next highest-value batch is not another generic architecture branch. It is:
   first, a 3-seed contender for the path reranker on the rebalanced suites;
   second, a combined multiheavy plus path-reranker contender;
   third, only then, a revisit of deadline/slack heads if they improve those contenders’ tail behavior.
7. Do not reopen broad communication exploration until one of the new exploit leaders has been stress-tested on the hard rebalanced OOD suites and verified to keep its miss-rate / p95 gains.
