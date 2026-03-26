# Round 10 Continuation Audit

## Accepted Baseline

The accepted default policy is still plain `multiheavy`.

The anchor remains the corrected round-four comparison against fresh `E3`:

- mean regret `1.32` vs `2.25`
- mean p95 regret `4.77` vs `10.48`
- mean deadline miss `41.7%` vs `54.2%`

Round six reproduced the same band. Rounds seven through nine changed the
diagnosis, not the default: large-gap hard-feasible mistakes are mostly gone,
and the real remaining frontier is the hard near-tie slice.

Round-nine also closed these families in their tested forms:

- large-gap targeting
- old reranker families
- direct critics as-is
- online search as-is
- plain delay mailbox
- route persistence
- broad adaptive halting from scratch
- generic history / state expansions
- train-only weighting / oversampling / DAgger tweak families

## Frontier Surface For Round 10

Round ten keeps the round-nine frontier pack as the only promotion surface.

Canonical slices:

- `hard_near_tie_intersection`
- `stable_near_tie`
- `high_headroom_near_tie`
- `baseline_error_intersection`
- `large_gap_control`

Canonical artifacts already in the repo:

- [round9_frontier_pack_seed314.json](/home/catid/gnn3/reports/plots/round9_frontier_pack_seed314.json)
- [round9_frontier_pack_seed314_decisions.csv](/home/catid/gnn3/reports/plots/round9_frontier_pack_seed314_decisions.csv)
- [frontier_pack_round9.md](/home/catid/gnn3/reports/frontier_pack_round9.md)
- [frontier_guard_round9.md](/home/catid/gnn3/reports/frontier_guard_round9.md)

Round ten reuses those thresholds for matched audits, then refreshes the
near-tie headroom and helpfulness labels on current baseline and `compute5`
runs.

## Current Code Paths

Model and baseline config:

- [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
- [config.py](/home/catid/gnn3/src/gnn3/train/config.py)
- [e3_memory_hubs_rsm_round9_multiheavy_seed314.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round9_multiheavy_seed314.yaml)
- [e3_memory_hubs_rsm_round9_compute5_seed314.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round9_compute5_seed314.yaml)

Frontier and decision analysis:

- [hard_feasible.py](/home/catid/gnn3/src/gnn3/eval/hard_feasible.py)
- [near_tie.py](/home/catid/gnn3/src/gnn3/eval/near_tie.py)
- [policy_analysis.py](/home/catid/gnn3/src/gnn3/eval/policy_analysis.py)
- [step_policy.py](/home/catid/gnn3/src/gnn3/eval/step_policy.py)

Round-nine compute / teacher infrastructure:

- [run_round9_frontier_pack.py](/home/catid/gnn3/scripts/run_round9_frontier_pack.py)
- [run_round9_compute_policy.py](/home/catid/gnn3/scripts/run_round9_compute_policy.py)
- [run_round9_branch_teacher.py](/home/catid/gnn3/scripts/run_round9_branch_teacher.py)

Round-ten changes landed in:

- [compute_helpfulness.py](/home/catid/gnn3/src/gnn3/eval/compute_helpfulness.py)
- [run_round10_helpfulness_audit.py](/home/catid/gnn3/scripts/run_round10_helpfulness_audit.py)
- [run_round10_feature_cache.py](/home/catid/gnn3/scripts/run_round10_feature_cache.py)
- [run_round10_helpfulness_probes.py](/home/catid/gnn3/scripts/run_round10_helpfulness_probes.py)
- [run_round10_offline_distill.py](/home/catid/gnn3/scripts/run_round10_offline_distill.py)
- [run_round10_selective_compute.py](/home/catid/gnn3/scripts/run_round10_selective_compute.py)

## Cached Artifacts To Reuse

Reusable model checkpoints already present:

- `artifacts/experiments/e3_memory_hubs_rsm_round9_multiheavy_seed{314,315,316}/checkpoints/best.pt`
- `artifacts/experiments/e3_memory_hubs_rsm_round9_compute5_seed{314,315,316}/checkpoints/best.pt`

Reusable frontier and compute artifacts:

- `reports/plots/round9_frontier_pack_seed314*`
- `reports/plots/round9_compute_policy_seed314_deeper_packets6_*`
- `reports/plots/round9_branch_teacher_seed314_compute5_*`

Round ten adds reusable per-seed helpfulness decisions and per-seed feature
caches rather than re-running raw counterfactual enumeration every time.

## Planned Round-10 Changes

Wave A:

- helpful / neutral / harmful labeling for `compute5` versus baseline on the
  frontier suites
- stability and headroom refresh across seeds `314 / 315 / 316`
- refreshed mixed easy/hard variable-compute readout

Wave B:

- frozen-state helpfulness probes
- candidate-conditioned helpfulness probes
- margin-only and margin-plus-regime baselines

Wave C:

- offline pairwise distill
- offline KL distill
- offline residual-logit distill
- gated versions of all three

Wave D:

- gate-triggered full `compute5` invocation
- gate-triggered top-2-localized compute invocation

## Promotion Gates

Round ten keeps the round-nine guard and tightens it to the helpful-compute
thesis.

Promote only if a branch does at least one of the following on the near-tie
frontier pack:

- disagreement >= `6%` and not strongly anti-oracle
- baseline-error near-tie recovery >= `30%`
- absolute hard near-tie regret improvement >= `0.10`
- a better regret-versus-average-compute tradeoff on the mixed compute suite

Also require:

- `large_gap_control` stays effectively solved
- no material broad feasible-suite regression
- compute overhead stays acceptable unless the branch is explicitly teacher-only

Kill immediately if:

- frontier disagreement is effectively zero
- movement is mostly anti-oracle
- gains come only from large runtime blowup
- the branch recreates a previously closed family in disguise

## Planned GPU-Hour Split

Round ten stays exploration-heavy while keeping a live exploit guardrail.

Planned split:

- exploit / validation / guardrail / audits: `1.10` GPU-hours
- explore / probes / students / selective compute: `1.90` GPU-hours
- planned split: `36.7% / 63.3%`

Operational lane plan:

- GPU0: baseline guardrails, helpfulness audits, cache generation, matched evals
- GPU1: probes, student variants, gated compute ablations
