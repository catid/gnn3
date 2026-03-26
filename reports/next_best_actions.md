# Next Best Actions

1. Keep plain `multiheavy` as the default exploit policy. Round ten did not find a compute/distillation branch that beat it cleanly on the accepted hard near-tie frontier.
2. Keep using the round-nine frontier pack and guard as the only promotion surface:
   - hard near-tie intersection
   - stable near-tie
   - high-headroom near-tie
   - baseline-error subset inside the intersection
   - large-gap control slice
3. Treat the live opportunity as a **narrow high-headroom near-tie correction** problem, not a broad hard near-tie compute problem. Round ten showed that full hard near-tie compute is net-negative on average, while the high-headroom subset remains the only cleanly positive source family.
4. Do not promote fixed `compute5`. It still helps some source families, but aggregate hard near-tie helpfulness (`1.92%`) stayed below aggregate harmfulness (`2.42%`), and the full-cost trade remains unjustified.
5. Do not promote the current helpfulness gates. Frozen-feature probes found some ranking signal, but not enough stable calibration to produce a usable operating point on held-out seeds.
6. Do not promote the current selective-compute policies. The learned gate collapsed to effectively zero trigger on held-out seeds, keeping average outer steps at `3.0` and failing to improve the full hard near-tie frontier.
7. Do not promote pure offline distillation from the current teacher cache. `gated_pairwise` was the safest student and slightly improved overall and large-gap controls, but it still regressed the full hard near-tie slice. `kl` was nearly a no-op. More aggressive students recovered more baseline errors only by breaking too many solved cases.
8. Keep `gated_pairwise` as the only conservative student worth remembering from round ten, but only as a reference baseline for future conservative-override work. It is not a contender.
9. Keep the representation diagnosis unchanged. Round ten did not overturn the earlier conclusion that the backbone already carries most of the local signals; the open problem is still abstention and decision-rule calibration on ambiguous states.
10. If another round opens, bias it toward **precision-first conservative override** rather than broader compute:
   - explicit high-headroom-only correction
   - abstention-calibrated delta heads
   - candidate-conditioned override rules with strict new-error penalties
   - policy objectives that optimize net corrected errors, not raw recovery
11. Keep the hard gate for every future branch:
   - baseline-error correction rate on the hard near-tie slice
   - new-error rate on baseline-correct near-tie states
   - net corrected errors
   - slice regret / miss delta
   - large-gap control preservation
   - runtime overhead
12. Keep `detach_warmup` mandatory in every future shortlist. That contract remains unbroken by every round since it was established.
