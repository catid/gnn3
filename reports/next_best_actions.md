# Next Best Actions

1. Keep plain `multiheavy` as the default exploit policy. Round eleven did not find a precision-first defer/correct branch that beat it cleanly on the accepted hard near-tie frontier.
2. Keep using the round-nine frontier pack and guard as the only promotion surface:
   - hard near-tie intersection
   - stable near-tie
   - high-headroom near-tie
   - baseline-error subset inside the intersection
   - large-gap control slice
3. Treat the live opportunity as a **tiny stable-positive correction** problem, not a broad hard near-tie compute problem. Round eleven showed the canonical stable-positive pack has only `46` audited decisions total, with just `4` held-out positives across seeds `315` and `316`, and effectively zero source-signature overlap across seeds.
4. Do not promote fixed `compute5`. It still helps the narrow high-headroom and baseline-error source families, but aggregate hard near-tie helpfulness remains below harmfulness and the full-cost trade is still unjustified.
5. Do not promote broad learned defer gates. Linear and MLP gates stayed inert on held-out seeds. The only surviving gate was the simple `margin_regime` ranker, and even that only helped at very low coverage.
6. Keep `margin_regime` defer as the only round-eleven reference branch worth remembering. On held-out seeds, the `1%` budget recovered `50%` of the stable-positive pack and slightly improved the full hard near-tie slice (`90.53% -> 90.66%` target match, mean delta regret `-0.0071`) with no false positives. At `2%`, it recovered `75%` of the stable-positive pack and improved the full hard near-tie slice a bit more (`90.53% -> 90.73%`, mean delta regret `-0.0089`), but coverage remains too broad relative to the tiny source family for clean promotion.
7. Do not promote the top-2 comparator-with-abstain family. Frozen comparators were essentially inert; candidate-conditioned narrow comparators were actively harmful on held-out seed316 and regressed large-gap controls.
8. Do not promote subset-only distillation from the current stable-positive teacher bank. `pairwise` and `kl` recovered some positive states but still broke too many solved cases overall, while `residual` remained aggressively destructive. `gated_pairwise` was again the safest student, but it stayed too close to baseline on seed315 and still regressed aggregate hard near-tie on held-out seeds (`90.53% -> 90.46%`, mean delta regret `+0.0036`).
9. Keep the representation diagnosis unchanged. Round eleven still did not overturn the earlier conclusion that the backbone already carries most of the local signals; the open problem is still precision calibration and abstention on a tiny ambiguous subset.
10. If another round opens, bias it toward **ultra-low-coverage defer-to-teacher** rather than new student capacity:
   - richer or more diverse teacher bank first
   - larger and more stable stable-positive source-family construction
   - explicit operating-point selection under tiny coverage budgets
   - hard false-positive penalties and large-gap preservation
11. Keep the hard gate for every future branch:
   - baseline-error correction rate on the hard near-tie slice
   - new-error rate on baseline-correct near-tie states
   - net corrected errors
   - slice regret / miss delta
   - large-gap control preservation
   - runtime overhead
12. Keep `detach_warmup` mandatory in every future shortlist. That contract remains unbroken by every round since it was established.
