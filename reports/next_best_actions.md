# Next Best Actions

1. Keep plain `multiheavy` as the default exploit policy. The fresh round-eight guardrail batch on seeds `311 / 312 / 313` stayed in the established round-four / round-six band, so there is still no evidence that a new default constructor has beaten it.
2. Preserve the corrected `oracle_calibrated` suite discipline and the round-eight cached near-tie slice definitions. Future work should keep using:
   - score-based hard oracle-feasible slice
   - oracle near-tie slice
   - model near-tie slice
   - hard near-tie intersection
   - baseline-error subset inside the intersection
   - large-gap control slice
3. Treat the remaining opportunity as a **hard near-tie ambiguity** problem, not a large-gap problem. Across fresh round-eight audits, the hard near-tie intersection stays around a `9%` to `11%` baseline-error slice, while the large-gap control stays near solved (`0.0%` to `0.3%` error).
4. Keep reading the plateau as primarily a **decision-rule bottleneck**, not a missing-feature bottleneck. The frozen-feature audits show that the backbone already exposes useful local signals for:
   - oracle gap bucket
   - near-tie classification
   - deadline-risk bucket
   - pairwise top-2 ranking
5. Do not promote the round-eight direct critics as standalone policies. Their ordering is now clear:
   - `pairwise_rank` is the safest direct critic and the only one with low-harm nonzero recovery
   - `scalar_q` has real recovery signal but is too destructive directly
   - `risk_multi` is clearly worse than scalar
   - `late_unfreeze` and its tighter gate are still too harmful for direct deployment
6. Do not reopen bounded search in its current form. Round-eight full-suite and targeted two-suite search scouts were killed on runtime before they justified promotion.
7. Do not reopen the local path-cost tie-break backup in its current form. It was cheaper than critic-guided bounded search, but it still crossed the scout runtime line before it proved enough value.
8. Do not open distillation from round-eight results. No search-time correction branch first earned promotion, so distilling it would have violated the round-eight ladder.
9. If another round opens, bias it toward **cheaper offline or semi-offline ambiguity correction**, not heavier online search. The most plausible remaining directions are:
   - off-policy near-tie decision-set distillation from cached counterfactual comparisons
   - very small near-tie-only delta policies that are trained offline and evaluated directly, without runtime search
   - better calibration of when to abstain from changing the base policy, rather than more global constructor diversification
10. Keep the same hard gate for any future branch:
   - baseline-error correction rate on the hard near-tie slice
   - new-error rate on baseline-correct near-tie states
   - net corrected errors
   - slice regret / miss delta
   - runtime overhead
11. Keep `detach_warmup` mandatory in every future shortlist. That model contract is still unbroken by every round since it was established.
