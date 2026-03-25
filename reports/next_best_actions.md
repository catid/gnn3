# Next Best Actions

1. Keep plain `multiheavy` as the default exploit policy. Round nine did not find a compute-state branch that beat it cleanly on the hard near-tie frontier without unacceptable new errors.
2. Preserve the corrected `oracle_calibrated` suite discipline and the round-nine frontier pack. Future work should keep using:
   - score-based hard oracle-feasible slice
   - hard near-tie intersection
   - stable near-tie
   - high-headroom near-tie
   - baseline-error subset inside the intersection
   - large-gap control slice
3. Treat the remaining opportunity as a **hard near-tie ambiguity** problem, not a large-gap or route-persistence problem. Round nine kept the large-gap controls effectively solved and found no evidence that short-lived route commitment is the missing mechanism.
4. Keep reading the plateau as primarily a **decision-rule bottleneck**, not a missing-feature bottleneck. The frozen-feature audits still support that interpretation, and round nine showed that extra compute can correct some right decisions without learning how to abstain from the harmful ones.
5. Do not promote fixed extra compute. `compute5` helped on seed314 but matched baseline exactly on seeds `315` and `316`; `compute7` was catastrophically bad.
6. Do not promote the tested adaptive-halting or triggered-continuation policies. The best direct policy (`fixed_final`) recovered `59.1%` of audited baseline near-tie errors on `deeper_packets6`, but still ended slightly below baseline target-match on the full hard near-tie slice and hurt large-gap controls. `risk_gate_tight` did not improve that trade, and the margin / learned gates were clearly worse.
7. Do not reopen the offline branch-teacher family in its current form. All `top_k {2,3}` by horizon `{1,2}` variants moved the frontier in the wrong direction: `0%` recovery with positive new-error in every cell.
8. Do not reopen the plain delay-mailbox family in its current form. Both seed314 mailbox scouts failed early and badly.
9. Do not open route-option persistence work from the current evidence. The route-persistence audit showed moderate oracle hub stability but `0%` unnecessary model flips on the audited frontier slice.
10. If another round opens, bias it toward **more conservative ambiguity correction**, not more raw compute. The most plausible remaining directions are:
   - tiny near-tie-only delta policies with explicit abstention penalties
   - offline correction rules that optimize net corrected errors, not just recovery
   - stricter conservative filters on when to override the base policy
11. Keep the same hard gate for any future branch:
   - baseline-error correction rate on the hard near-tie slice
   - new-error rate on baseline-correct near-tie states
   - net corrected errors
   - slice regret / miss delta
   - runtime overhead
12. Keep `detach_warmup` mandatory in every future shortlist. That model contract is still unbroken by every round since it was established.
