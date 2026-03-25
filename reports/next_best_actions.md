# Next Best Actions

1. Keep plain `multiheavy` as the exploit default. The corrected feasible-suite guardrail remains the stable reference, and no round-seven constructor branch beat it.
2. Preserve the corrected `oracle_calibrated` feasible suites and split-manifest discipline. Round seven did not find any hidden replay issue; the remaining uncertainty is architectural, not dataset drift.
3. Do not reopen the round-seven “large-gap hard-feasible constructor” thesis in its current form. The new action-gap audit showed that the audited large-gap hard-feasible slice is already almost solved by `multiheavy`:
   - score-based hard-feasible error: `2.68%`
   - thresholded large-gap hard-feasible error: `0.30%`
   - quartile-defined large-gap hard-feasible error: `0.00%`
4. Treat the remaining opportunity as a hard near-tie problem, not a clear-gap problem. On the corrected score-based hard slice, `37 / 40` baseline mistakes sit in `near_tie`, not `large_gap`.
5. Do not reopen the poly-constructor branch in its current form. It failed before the disagreement gate mattered: epoch 1 collapsed to `5419.82` regret, `13497.61` p95 regret, and `93.8%` miss.
6. Do not reopen the self-improving constructor in its current form. It produced exact policy identity on the audited hard slice:
   - overall disagreement: `0.0%`
   - hard-feasible disagreement: `0.0%`
   - large-gap hard-feasible disagreement: `0.0%`
7. Do not reopen the tight-slack or depth-4 specialist teachers in their current form. They were also exactly policy-identical on the audited hard slice.
8. Do not promote the heavy specialist teacher. It was the only round-seven branch with real hard-slice movement, but the movement was anti-oracle:
   - `deeper_packets6` hard-slice disagreement: `8.0%`
   - `deeper_packets6` large-gap hard-slice disagreement: `7.69%`
   - `deeper_packets6` target-match fell from `0.98` to `0.90`
   - aggregate large-gap target-match fell from `1.00` to `0.949`
9. Use the frozen probe audit as the main diagnosis for the plateau. The current `multiheavy` encoder already linearly exposes most of the relevant local signals:
   - slack bucket: `0.87` to `0.90` OOD accuracy
   - critical packet proxy: `0.97+`
   - feasible continuation: essentially `1.0`
   - baseline strictly suboptimal: `0.95+`
   - oracle gap bucket: `0.65` to `0.77`
10. Read the probe result as “constructor bottleneck, not missing-signal bottleneck,” with one caveat: explicit depth/load regime generalization is weak under OOD label shift. If another architecture round opens, it should only do so with a materially stronger mechanism for hard near-tie ambiguity resolution or explicit structured credit assignment, not another side channel or small-head variant.
11. Keep the hard gate for any future architecture round:
   - report disagreement on the corrected score-based hard slice
   - report disagreement on the large-gap subset
   - report whether movement helps or hurts the hard near-tie slice
   - kill branches that stay policy-identical or move away from oracle targets on the audited hard slice
12. Keep `detach_warmup` mandatory in every future shortlist. That remains the strongest causal model contract in the repo.
