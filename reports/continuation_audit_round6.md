# Continuation Audit Round 6

## What Exists Now

- The current exploit default is the round-four `multiheavy` branch, configured by [e3_memory_hubs_rsm_round4_multiheavy_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round4_multiheavy_seed311.yaml) and its seed-matched siblings.
- Dataset generation, oracle supervision, packet-level candidate labels, and split-manifest hashing all live in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py). The corrected feasible suites use `deadline_mode: oracle_calibrated`.
- Experiment config loading and split-specific overrides live in [config.py](/home/catid/gnn3/src/gnn3/train/config.py). This is the safe place to add round-six config knobs without disturbing older experiments.
- Training, checkpoint selection, metadata/manifests, and rollout summaries live in [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py). The trainer already records git commit, branch, manifests, device placement, GPU-hours, and rollout metrics.
- The baseline model trunk, slow/fast state logic, hazard-memory hook, path-reranker scaffold, and final action readout all live in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py).
- Rollout evaluation is centralized in [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py). This is the safest place to add action-agreement and hard-case disagreement instrumentation without changing training semantics.
- Oracle feasibility analysis already exists in [oracle_analysis.py](/home/catid/gnn3/src/gnn3/eval/oracle_analysis.py) and [run_oracle_deadline_audit.py](/home/catid/gnn3/scripts/run_oracle_deadline_audit.py).
- Existing evaluation and plotting entrypoints are [run_train.py](/home/catid/gnn3/scripts/run_train.py), [run_eval.py](/home/catid/gnn3/scripts/run_eval.py), [run_eval_sweep.py](/home/catid/gnn3/scripts/run_eval_sweep.py), [plot_experiment_summaries.py](/home/catid/gnn3/scripts/plot_experiment_summaries.py), [plot_round4_results.py](/home/catid/gnn3/scripts/plot_round4_results.py), and [plot_round5_results.py](/home/catid/gnn3/scripts/plot_round5_results.py).
- Verifier-backed logic from the prior round is still reusable but must stay bounded:
  - verifier-filter logic is in `PacketMambaModel._readout` in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)
  - shared-config deployment logic is in [deployment.py](/home/catid/gnn3/src/gnn3/eval/deployment.py)
- There is already a path-level scaffold worth reusing for a plannerized branch: `candidate_path_nodes`, `candidate_path_mask`, `candidate_path_features`, `candidate_cost_to_go`, `candidate_slack`, and `candidate_on_time` are all produced by [make_decision_record](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py).

## Safe Code Paths To Extend

- Extend [PacketMambaConfig](/home/catid/gnn3/src/gnn3/models/packet_mamba.py) with narrowly scoped round-six options rather than introducing new model registries.
- Add regime descriptors in [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py) and thread them through `collate_decisions`, because the benchmark already exposes packet count, deadlines, candidate slack, and graph structure.
- Add regime-conditioned adapters or expert gates inside `PacketMambaModel._readout` or on top of the slow state in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py); this keeps the proven trunk intact.
- Add plannerized decode logic by reusing existing candidate-path tensors from [hidden_corridor.py](/home/catid/gnn3/src/gnn3/data/hidden_corridor.py) rather than bolting a new generic reranker onto the old path-head branch.
- Add repair-style evaluation/training helpers in [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py) plus a small bounded module in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py); this lets the second stage actually edit a failing rollout instead of only rescoring it.
- Add regime-audit scripts and round-six plotting scripts under `scripts/`, following the same artifact conventions as [run_oracle_deadline_audit.py](/home/catid/gnn3/scripts/run_oracle_deadline_audit.py) and [plot_round5_results.py](/home/catid/gnn3/scripts/plot_round5_results.py).

## Closed Branches That Should Stay Closed

- Do not reopen the old generic communication directions: `X1`, `X2`, `X4`, and `X6`.
- Do not reopen the current reranker family: additive reranker, gated reranker, verifier-filter reranker, or conditional verifier deployment.
- Do not reopen selector-only changes: outer-step selection, checkpoint-selection tuning, or path-first promotion of the current integrated path head.
- Do not reopen train-only exploit tweaks that already failed to move held-out rollout: tighter deadlines, packets6-train curriculum, critical oversampling, soft targets, pairwise loss, feasible-first targets, slack weighting, or bounded DAgger refresh.
- Do not reopen calibration-only auxiliary heads unless a new branch shows real rollout movement rather than only better head metrics.

## Current Guardrails

- The corrected split-manifest discipline is trustworthy and must be preserved. `train`, `val`, and `test` split seeds and manifest hashes are already written by [trainer.py](/home/catid/gnn3/src/gnn3/train/trainer.py).
- The most trustworthy metrics are held-out rollout regret, p95 regret, deadline miss rate, rollout next-hop accuracy, manifest hashes, and oracle feasibility computed from [oracle_analysis.py](/home/catid/gnn3/src/gnn3/eval/oracle_analysis.py).
- The least instrumented area is policy movement. There is no existing baseline-vs-branch disagreement report, so round six needs explicit action-agreement and hard-case disagreement logging in [rollout.py](/home/catid/gnn3/src/gnn3/eval/rollout.py) or a companion eval script.
- Remaining variance likely comes from seed-specific graph/topology draws, packet schedule draws, checkpoint selection on short runs, and the fact that many prior branches were close enough to baseline that small in-distribution differences washed out on the full held-out rollout.

## Exact Experiment Ordering

1. Run the green validation stack on the round-six branch.
2. Reproduce a fresh 3-seed `multiheavy` guardrail on the corrected feasible suites, using the existing round-four baseline contract.
3. Run a regime-stratified baseline failure audit and write [regime_audit_round6.md](/home/catid/gnn3/reports/regime_audit_round6.md).
4. Launch a 1-seed regime-expert/adaptor scout on GPU1 while GPU0 handles baseline evals, regime tables, and guardrail reruns.
5. Promote the regime-expert branch only if it shows both hard-case policy movement and stratum-specific gains.
6. If regime experts are weak, open the plannerized decoder scout next, reusing candidate-path tensors instead of reviving the old reranker branch.
7. Only after either the regime-expert or plannerized branch shows real policy movement should the repair branch open.
8. Keep the hazard-memory scout bounded and small; it is a side experiment, not the primary branch.

## Planned GPU-Hour Split

- Exploration target: `70%`
- Exploitation / baseline guardrail target: `30%`
- Allowed operating window: exploration `65–75%`, exploitation `25–35%`

Round-six budget intent:

- `A1` baseline reproduction + `A2` regime audit: about `30%`
- `B` regime-experts / adapters: about `35%`
- `C` plannerized decoder: about `25%`
- `D` bounded repair: about `5%` to `10%`, only if promoted by earlier evidence
- `E` hazard-memory scout: about `5%`

This round should prune aggressively. A branch that does not move policy on hard feasible cases should be killed after the scout rather than promoted on noise.
