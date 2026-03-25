# Continuation Audit Round 8

## What Exists Now

- Core benchmark and dataset generation still live in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py).
- Training config, split overrides, manifest hashing, and seed handling live in [config.py](/home/catid/gnn3/src/gnn3/train/config.py).
- Baseline training and checkpoint selection live in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py).
- Rollout evaluation lives in [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py).
- Decision-level and episode-level analysis lives in [policy_analysis.py](/home/catid/gnn3/src/gnn3/eval/policy_analysis.py).
- Hard-slice labeling from round seven lives in [hard_feasible.py](/home/catid/gnn3/src/gnn3/eval/hard_feasible.py).
- The current `multiheavy` baseline config family is already established in:
  - [e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml)
  - [a1_multiheavy_ood_deeper_packets6_round7_eval.yaml](/home/catid/gnn3/configs/experiments/a1_multiheavy_ood_deeper_packets6_round7_eval.yaml)
  - [a1_multiheavy_ood_heavy_dynamic_round7_eval.yaml](/home/catid/gnn3/configs/experiments/a1_multiheavy_ood_heavy_dynamic_round7_eval.yaml)
- Round-seven audit and compare scripts already provide the right scaffolding:
  - [run_round7_hard_feasible_audit.py](/home/catid/gnn3/scripts/run_round7_hard_feasible_audit.py)
  - [run_round7_probe_audit.py](/home/catid/gnn3/scripts/run_round7_probe_audit.py)
  - [run_round7_hard_slice_compare.py](/home/catid/gnn3/scripts/run_round7_hard_slice_compare.py)
  - [plot_round7_results.py](/home/catid/gnn3/scripts/plot_round7_results.py)

## Safe Extension Points

- Near-tie slice annotation and cached manifests should extend [hard_feasible.py](/home/catid/gnn3/src/gnn3/eval/hard_feasible.py).
- Any new decision-level audit features should extend [policy_analysis.py](/home/catid/gnn3/src/gnn3/eval/policy_analysis.py), not duplicate rollout logic.
- Counterfactual cache builders and round-eight reports should be new scripts under [scripts](/home/catid/gnn3/scripts), reusing the round-seven audit entrypoints.
- Critic/search logic should start as external round-eight scripts built on frozen `multiheavy` features and existing decision records before touching the core training loop.
- Only if a critic/search branch shows real near-tie improvement should model or training code be extended for distillation.
- Round-eight experiment configs should live under [configs/experiments](/home/catid/gnn3/configs/experiments) as explicit `round8_*` files so manifests and checkpoints remain traceable.

## Closed Families That Should Stay Closed

- Do not reopen round-four rerankers or verifier-filter deployment as default policies.
- Do not reopen round-five exploit-only weighting, sampling, DAgger, path-head, or outer-step families.
- Do not reopen round-six regime experts, plannerized decoder, or hazard-memory side channel in their prior form.
- Do not reopen round-seven poly constructor, self-improve, or specialist-teacher families in their prior form.

## Round-Eight Near-Tie Definition And Cache Plan

- Keep the round-seven score-based hard slice as the outer filter:
  - oracle-feasible
  - at least two of {critical-or-very-tight slack, `5+` packets, depth `4`, high load}
- Add and cache explicit round-eight slices:
  1. score-based hard oracle-feasible slice
  2. oracle-gap near-tie slice
  3. model-margin near-tie slice
  4. hard + oracle-feasible + near-tie intersection
  5. baseline-error subset inside the intersection
  6. large-gap control slice
- Persist the per-decision audit table and manifest CSVs in [reports/plots](/home/catid/gnn3/reports/plots) so critic/search experiments reuse the same exact slice definition.
- Persist the counterfactual all-action cache separately under [artifacts](/home/catid/gnn3/artifacts) because it is experiment input, not only reporting output.

## Hard Near-Tie Policy-Movement Gate

- Every serious branch must report, on the hard near-tie intersection:
  - baseline-error correction rate
  - new-error rate on baseline-correct cases
  - net corrected errors
  - regret delta
  - deadline-miss delta
  - action disagreement with plain `multiheavy`
- A branch that stays near-zero disagreement on the hard near-tie intersection is dead even if average metrics wiggle slightly.

## Exploit / Explore GPU-Hour Plan

- Planned split: `30%` exploit / audit / guardrail, `70%` explore / constructor-time correction.
- GPU0 lane:
  - fresh `multiheavy` guardrail
  - round-eight audits that need model evaluation
  - first seed of critic/search follow-up runs
- GPU1 lane:
  - parallel critic variants
  - search ablations
  - conditional distillation or backup tie-break scout
- CPU lane:
  - near-tie manifest generation
  - probe fitting
  - counterfactual cache assembly
  - plotting/report refresh

## 2x Work Execution Plan

1. Run audit + validation green first.
2. Build the round-eight near-tie cache and headroom audit before training any new branch.
3. Run the frozen-feature near-tie probe extension in parallel with the fresh guardrail.
4. Build one reusable counterfactual cache from the guardrail manifests and reuse it across critic variants.
5. Launch at least three critic variants quickly and kill zero-disagreement branches fast.
6. Only open bounded search after at least one critic shows real hard near-tie correction signal.
7. Only open distillation if search produces real slice gains.
8. Only open the backup path tie-break family if critic signal exists but search is too brittle or too slow.

## Exact Experiment Order

1. Validation green.
2. Fresh round-eight `multiheavy` guardrail on the round-eight manifests.
3. Near-tie headroom audit.
4. Frozen-feature near-tie probe audit.
5. Counterfactual cache build.
6. Critic family scouts:
   - frozen scalar-Q critic
   - frozen risk critic
   - late-unfreeze critic
   - cheap ranking variant only if the first three leave uncertainty
7. Promote only critics that show real hard near-tie correction.
8. Run bounded search variants on the promoted critic only.
9. Run distillation only if bounded search improves the hard near-tie slice.
10. Run the narrow path tie-break backup only if needed.
11. Refresh plots, portfolio, experiment summary, and final report.
12. Merge the completed branch back into `main`, rerun validation on `main`, and attempt push.

## Merge-Back-To-Main Plan

1. Finish or explicitly kill all intended round-eight branches with documented reasons.
2. Commit code, configs, reports, and plots on the round-eight branch.
3. Re-run the validation stack on the working branch.
4. Update [next_best_actions.md](/home/catid/gnn3/reports/next_best_actions.md) with the final handoff.
5. Merge back locally:
   - `git checkout main`
   - `git pull --ff-only origin main`
   - `git merge --no-ff round8/near-tie-critic-search -m "round8: near-tie critic and bounded search experiments"`
6. Re-run key validation on local `main`.
7. Push `main` if permitted; otherwise document the exact auth failure and leave local `main` clean and merged.
