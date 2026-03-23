# Next Best Actions

1. Promote the rebalanced multiheavy curriculum from scout to the current lead exploit recipe. Across matched seeds `311/312/313`, it beat the fresh `E3` baseline on mean regret (`1.32` vs `2.25`), mean p95 regret (`4.77` vs `10.48`), and mean deadline miss (`41.7%` vs `54.2%`).
2. Promote the combined multiheavy plus lightweight candidate-path reranker recipe to the new lead exploit contender. Across matched seeds `311/312/313`, it improved on fresh `E3` (`1.23` vs `2.25` mean regret, `4.69` vs `10.48` mean p95, `39.6%` vs `54.2%` miss) and edged plain multiheavy (`1.23` vs `1.32` regret, `4.69` vs `4.77` p95, `39.6%` vs `41.7%` miss).
3. Do not promote the standalone reranker as a separate lead recipe. Its third matched seed was a clean negative (`2.47` regret, `9.20` p95, `68.8%` miss), so the additive value appears to come from pairing it with the multiheavy curriculum rather than from reranking alone.
4. Preserve the rebalanced `oracle_calibrated` deadline suites and the split-manifest discipline. Old corrected deadline suites remain diagnostic-only because the oracle itself misses them.
5. Keep `detach_warmup` in every shortlist. That remains the strongest causal architectural requirement in the repo.
6. Deprioritize `A2`, `A4`, and `B1` behind the exploit winners. `A2` still matters as a calibration mechanism, but multiheavy plus reranking now has the best direct rollout evidence; `A4` and `B1` do not.
7. The next highest-value batch is not another generic architecture branch. It is:
   first, OOD stress and deeper-traffic validation for multiheavy plus reranking;
   second, a calibration or verifier add-on only if it improves that lead recipe’s tail behavior;
   third, only then, revisit new architectural movement.
8. Do not reopen broad communication exploration until the combined exploit leader has been stress-tested on the hard rebalanced OOD suites and verified to keep its miss-rate / p95 gains.
