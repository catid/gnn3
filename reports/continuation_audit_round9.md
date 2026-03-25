# Continuation Audit Round 9

## Current Accepted Baseline

- `multiheavy` remains the default exploit policy.
- The stable anchor is still the corrected feasible comparison from round four and the matched round-six rerun:
  - mean regret `1.32`
  - mean p95 regret `4.77`
  - mean deadline miss `41.7%`
- Round seven closed the large-gap hypothesis: remaining errors are concentrated in hard near-tie states, not large-gap hard-feasible mistakes.
- Round eight confirmed the near-tie opportunity is real and stable and that the bottleneck looks like decision rule / constructor behavior rather than missing backbone signal.

## Frontier Slice For Round 9

Round nine treats the frontier as the cached round-eight hard near-tie surface:

1. score-based hard oracle-feasible slice
2. oracle near-tie slice
3. model near-tie slice
4. hard near-tie intersection
5. baseline-error subset inside the hard near-tie intersection
6. large-gap control slice

The main promotion surface is:

- hard near-tie regret
- hard near-tie error rate
- hard near-tie miss rate
- disagreement vs `multiheavy` on the hard near-tie intersection
- recovery of audited baseline near-tie errors
- regret-at-fixed-average-compute on a mixed easy/hard suite

## Existing Code Paths

Safe extension points already in repo:

- model registration and recursive compute:
  - `src/gnn3/models/packet_mamba.py`
- experiment config loading:
  - `src/gnn3/train/config.py`
- training loop / summary / checkpoint bookkeeping:
  - `src/gnn3/train/trainer.py`
- rollout evaluation:
  - `src/gnn3/eval/rollout.py`
- step-policy selection:
  - `src/gnn3/eval/step_policy.py`
- decision and episode audits:
  - `src/gnn3/eval/policy_analysis.py`
- hard-slice and near-tie helpers:
  - `src/gnn3/eval/hard_feasible.py`
  - `src/gnn3/eval/near_tie.py`
- round-eight audit / critic / search scripts to extend rather than replace:
  - `scripts/run_round8_near_tie_audit.py`
  - `scripts/run_round8_probe_audit.py`
  - `scripts/run_round8_counterfactual_dataset.py`
  - `scripts/run_round8_train_critic.py`
  - `scripts/run_round8_near_tie_search.py`
  - `scripts/run_round8_path_tiebreak.py`
  - `scripts/plot_round8_results.py`

Round-nine additions should stay in the same pattern:

- reusable evaluation helpers under `src/gnn3/eval/`
- round-nine driver scripts under `scripts/`
- round-nine reports under `reports/`
- round-nine plots and machine-readable manifests under `reports/plots/`
- round-nine configs under `configs/experiments/`

## Old Branches That Stay Closed

Do not reopen in their previous form:

- rerankers and planner-reranker hybrids
- selector-only tuning
- train-only weighting / oversampling / DAgger tweaks
- current path-head promotion
- generic history banks / summary banks
- regime experts in the tested form
- poly-constructor / self-improve / specialist-teacher families in the tested forms
- broad large-gap targeting

Round nine is limited to compute-and-state mechanisms that can change the hard near-tie frontier without reopening those closed families in disguise.

## Round 9 Code Plan

1. Build a machine-readable frontier pack and guard on top of the cached near-tie audit path.
2. Add reusable outer-step / compute accounting utilities on top of existing per-step logits and rollout flow.
3. Add offline branch-teacher caching for ambiguous states.
4. Add adaptive halting / triggered continuation support using the existing recursive model outputs.
5. Add a minimal explicit delayed-state mailbox inside the existing outer refinement path.
6. Reuse the existing report / plot generation style to land matched summary artifacts.

## Planned Exploit / Explore Split

Target GPU-hour split:

- exploit / guardrail / audits: `30%`
- explore / compute-state branches: `70%`

Planned lanes:

- GPU0: fresh guardrails, frontier audits, teacher-cache generation, evaluation sweeps
- GPU1: extra-compute and delayed-state training scouts

## Promotion Gates

Promote a scout only if at least one is true on the corrected hard near-tie slice:

- disagreement vs `multiheavy` >= `6%` and not strongly anti-oracle
- baseline-error recovery >= `30%`
- absolute hard near-tie regret drop >= `0.10`
- meaningful improvement in regret-at-fixed-average-compute on the mixed compute suite

Also require:

- large-gap controls stay effectively solved
- no broad feasible-suite regression beyond tolerance
- runtime cost is acceptable unless the branch is explicitly teacher-only

Kill immediately if:

- hard near-tie disagreement is near zero
- disagreements are anti-oracle
- gains require large compute blowup without a clear frontier payoff
- the branch recreates a previously closed family under new naming

## 2x Work Execution Plan

Round nine should do roughly double the useful work of a typical recent round by keeping both GPUs busy with a live queue:

1. frontier pack + guard
2. source-family near-tie stratification
3. outer-step headroom audit
4. variable-compute benchmark
5. offline branch-teacher cache generation
6. branch-teacher grid: top-2 vs top-3
7. branch-teacher grid: short vs deeper branch horizon
8. branch-teacher trigger variants
9. adaptive halting scouts
10. triggered continuation scouts
11. teacher-for-compute distillation scouts
12. minimal delay-mailbox scouts
13. mailbox placement / delay-set ablations
14. mailbox + continuation combo scout
15. route-persistence audit
16. contingent compute/state scout only if earlier stages justify it

The round is complete only after all intended branches are either promoted or explicitly killed, reports/plots are updated, the branch is merged back into local `main`, and remote push is attempted with exact failure text documented if auth still blocks.
