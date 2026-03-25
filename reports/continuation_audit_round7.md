# Round 7 Continuation Audit

## What Exists Now

- The repo already has one unified `PacketMambaModel` implementation in `src/gnn3/models/packet_mamba.py` with feature flags for:
  - `memory_hubs` / `multiheavy` trunk behavior
  - deadline heads
  - planner decoder
  - repair decoder
  - hazard memory
  - regime experts
- The current exploit default is the round-four / round-six `multiheavy` line configured by:
  - `configs/experiments/e3_memory_hubs_rsm_round6_multiheavy_seed311.yaml`
  - `configs/experiments/e3_memory_hubs_rsm_round6_multiheavy_seed312.yaml`
  - `configs/experiments/e3_memory_hubs_rsm_round6_multiheavy_seed313.yaml`
- Benchmark generation, corrected split discipline, and manifest hashing already live in:
  - `src/gnn3/data/hidden_corridor.py`
  - `src/gnn3/train/config.py`
  - `src/gnn3/train/trainer.py`
- Existing policy comparison and regime audit scaffolding already exist:
  - `src/gnn3/eval/policy_analysis.py`
  - `scripts/run_round6_regime_audit.py`
  - `scripts/run_round6_policy_compare.py`
  - `scripts/plot_round6_results.py`
- Training / eval / reporting are already config-driven:
  - `scripts/run_train.py`
  - `scripts/run_eval.py`
  - `scripts/run_eval_sweep.py`
  - `artifacts/experiments/<experiment>/summary.json`

## Safe Extension Paths For Round 7

- Hard-feasible and action-gap audits:
  - extend `src/gnn3/eval/policy_analysis.py`
  - add round-seven analysis scripts under `scripts/`
  - reuse `candidate_cost_to_go`, `candidate_slack`, `candidate_on_time`, and manifest metadata already produced in `src/gnn3/data/hidden_corridor.py`
- Frozen representation probes:
  - add feature extraction hooks in `src/gnn3/models/packet_mamba.py`
  - keep probe training in a separate audit script so the training loop stays simple
- Constructor branches:
  - add small config flags and auxiliary losses in `PacketMambaConfig` / `TrainConfig`
  - keep the shared `multiheavy` trunk unchanged unless the branch explicitly needs a new constructor head
  - write new round-seven configs in `configs/experiments/`
- Reporting / plots:
  - add `scripts/plot_round7_results.py`
  - append round-seven tables into `reports/plots/experiment_summary.csv`

## Prior Branches That Are Closed And Should Not Be Reopened

- Rerankers and conditional reranker deployment
- Selector-only tuning and outer-step selection
- Train-only weighting / oversampling / DAgger refresh
- Current path-head promotion / path-first decoding
- Round-six regime experts in the old gating form
- Round-six planner side-head in the old form
- Generic hazard / history-memory variants as broad searches
- Repair before a constructor first earns promotion

Round seven should only reopen a family if the new branch changes the constructor itself and clears the hard-feasible disagreement gate.

## How Round 7 Will Measure Hard-Feasible Policy Movement

- Base slice:
  - oracle-feasible decisions only
  - critical / very-tight slack
  - high-load episodes
  - packet count `>= 5`
  - depth `4`
- Large-gap hard-feasible slice:
  - same as above, plus an oracle action-gap / continuation-gap threshold from the round-seven audit tables
- Per branch, report:
  - overall action agreement vs `multiheavy`
  - large-gap hard-feasible disagreement vs `multiheavy`
  - disagreement on baseline missed-deadline states
  - improvement rate on baseline failures
  - break rate on baseline successes
  - hard-feasible slice regret / p95 regret / miss

A constructor scout is only promotable if it both moves policy on the large-gap hard-feasible slice and improves slice-level regret or miss.

## Exact Code Paths For Round 7 Changes

- Baseline / audit / disagreement logic:
  - `src/gnn3/eval/policy_analysis.py`
  - `scripts/run_round7_hard_feasible_audit.py`
  - `scripts/run_round7_probe_audit.py`
  - `scripts/run_round7_policy_compare.py`
- Constructor branches:
  - `src/gnn3/models/packet_mamba.py`
  - `src/gnn3/train/config.py`
  - `src/gnn3/train/trainer.py`
  - `configs/experiments/*round7*.yaml`
- Plotting and reports:
  - `scripts/plot_round7_results.py`
  - `reports/continuation_report_round7.md`
  - `reports/hard_feasible_action_gap_round7.md`
  - `reports/probe_audit_round7.md`
  - `reports/portfolio_balance.md`
  - `reports/next_best_actions.md`

## Likely Sources Of Remaining Variance

- seed-to-seed graph and packet sampling variation even with corrected split discipline
- short-budget scout checkpoint selection differences across seeds
- small rollouts (`16` episodes in default training summaries) understating hard-slice variance
- hard-feasible slice size shrinking when both tight slack and large action-gap filters are applied
- branch instability in the first epoch causing apparent policy movement that does not survive selected-checkpoint evaluation

The existing manifest hashing and checkpoint metadata are trustworthy; the remaining uncertainty is mostly slice size and seed variance, not hidden replay drift.

## Metrics Already Trustworthy

- manifest hashes by split
- selected epoch / selection score
- test-rollout regret / p95 regret / miss from `summary.json`
- round-six regime buckets for slack / packets / load / depth
- action agreement and hard-feasible disagreement at the round-six threshold

## Metrics That Need Better Instrumentation This Round

- oracle action-gap and continuation-gap on each decision
- large-gap hard-feasible disagreement rather than only generic hard-feasible disagreement
- per-branch fix / break counts specifically on baseline hard-feasible failures
- frozen-feature probe quality for:
  - feasibility
  - gap bucket
  - strict suboptimality of the baseline choice

## Exact Experiment Ordering

1. validation green on current branch
2. fresh round-seven `multiheavy` guardrail reproduction
3. hard-feasible action-gap audit
4. frozen representation probe audit
5. launch poly-constructor scout and self-improving constructor scout in parallel
6. run specialist-teacher audit after the first two scouts are underway
7. promote only a branch that clears both disagreement and hard-slice improvement gates
8. open no contingent repair/planner follow-up unless a constructor first earns it
9. refresh plots, reports, portfolio ledger, and experiment summary

## Planned Exploit / Explore GPU-Hour Split

- exploit / audit / guardrail: `30%`
- explore / constructor search: `70%`

Target window:

- exploit: `25–35%`
- explore: `65–75%`

Practical lane plan:

- GPU0:
  - fresh `multiheavy` guardrail
  - hard-feasible audit evals
  - frozen feature extraction
  - specialist-teacher slice runs as needed
- GPU1:
  - poly-constructor scout
  - self-improving constructor scout
  - any promoted candidate reruns
- CPU:
  - action-gap tables
  - probe fitting
  - policy-compare summaries
  - plots / report refresh

## “2x Work” Execution Plan

- Complete both required audits before promoting any branch.
- Run three constructor scouts, not one-at-a-time serially:
  - poly-constructor
  - self-improving constructor
  - specialist-teacher audit
- Use a strict scout → candidate → contender ladder:
  - scout: `1` seed, small budget, mandatory large-gap hard-feasible compare
  - candidate: `2` seeds only if the scout shows real disagreement plus slice gain
  - contender: `3` seeds only if candidate evidence holds
- Kill weak branches immediately when:
  - large-gap hard-feasible disagreement is near zero
  - movement exists only on easy cases
  - the branch regresses OOD or hard-feasible miss without a compensating slice gain
- Keep both GPUs occupied by overlapping:
  - guardrail / audits on GPU0
  - constructor scouts on GPU1
  - CPU-side probe / plotting / compare jobs

The round is successful only if it closes more questions than round six: both audits completed, three constructor families tested, and either one promoted or all three explicitly closed with hard-feasible disagreement evidence.
