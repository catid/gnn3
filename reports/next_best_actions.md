# Next Best Actions

1. Keep plain `multiheavy` as the exploit default. The fresh round-six guardrail still holds at `1.32` mean regret, `4.77` mean p95 regret, and `41.7%` mean deadline miss on the feasible suites.
2. Preserve the corrected `oracle_calibrated` feasible suites and split-manifest discipline. The round-six regime audit confirmed that the current failure mode is deadline robustness under tight/high-load/deep regimes, not route existence.
3. Do not reopen round-six regime experts in their current form. The scout matched the baseline rollout exactly on the shared seed and showed `0.0` hard-feasible disagreement on every checked suite.
4. Do not reopen the current plannerized decoder in its present form. The scout moved the policy only cosmetically, stayed at `0.0` hard-feasible disagreement, and was catastrophically unstable on `heavy_dynamic`.
5. Do not reopen the current hazard-memory side channel in its present form. It collapsed at epoch 1 and then snapped back to the exact baseline selected rollout by epoch 2.
6. Do not open the repair branch unless a first-stage constructor branch first clears the hard policy-movement gate. Round six did not produce that prerequisite signal.
7. Keep the round-six hard gate for any future architecture pass:
   - report exact action agreement vs plain `multiheavy`
   - report disagreement on hard feasible cases
   - report how often the branch repairs `multiheavy` failures versus breaking `multiheavy` successes
8. Do not spend another cycle on rerankers, selector-only rules, train-only weighting/oversampling/DAgger tweaks, path-head promotion, or outer-step selection. Rounds four and five already exhausted those levers.
9. If a future round reopens architecture work, start from a constructor that is structurally capable of changing the hard slice identified in round six:
   - critical or very-tight slack
   - `5+` packets
   - high load
   - depth `4`
10. Keep `detach_warmup` mandatory in every shortlist. That remains the strongest causal model contract in the repo.
