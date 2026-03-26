# Round 11 Continuation Audit

## Accepted baseline

The accepted default policy remains plain `multiheavy`.

The anchor stays the corrected round-four comparison against fresh `E3`:

- mean regret `1.32` vs `2.25`
- mean p95 regret `4.77` vs `10.48`
- mean deadline miss `41.7%` vs `54.2%`

Round six reproduced the same baseline band, so the baseline is still stable.

## Frontier pack and guard

Round eleven keeps the round-nine frontier pack and guard as the only promotion
surface:

- `reports/frontier_pack_round9.md`
- `reports/frontier_guard_round9.md`
- `reports/plots/round9_frontier_pack_seed314_decisions.csv`
- `reports/plots/round9_frontier_pack_seed314.json`

Operational slices:

- `hard_near_tie_intersection`
- `stable_near_tie`
- `high_headroom_near_tie`
- `baseline_error_intersection`
- `large_gap_control`

Round ten narrowed the live opportunity further:

- broad hard near-tie compute is net negative
- high-headroom near-tie is the only clean positive source family
- baseline-error near-tie is also positive but noisier
- large-gap controls remain solved

Round eleven therefore adds a second gate on top of the existing frontier pack:

- Tier 1: stable-positive correction pack
- Tier 2: full near-tie frontier pack

## Cached audit path

The committed audit artifacts that can be reused directly are:

- `reports/plots/round10_helpfulness_seed314_decisions.csv`
- `reports/plots/round10_helpfulness_seed315_decisions.csv`
- `reports/plots/round10_helpfulness_seed316_decisions.csv`
- `reports/plots/round10_helpfulness_seed314_summary.csv`
- `reports/plots/round10_helpfulness_seed315_summary.csv`
- `reports/plots/round10_helpfulness_seed316_summary.csv`
- `reports/plots/round10_helpfulness_seed314_stability.csv`
- `reports/plots/round10_helpfulness_seed315_stability.csv`
- `reports/plots/round10_helpfulness_seed316_stability.csv`
- `reports/plots/round10_offline_distill_summary.csv`
- `reports/plots/round10_selective_compute_summary.csv`
- `reports/plots/round10_helpfulness_probe_summary.csv`

These already carry the core round-ten labels needed for round eleven:

- hard near-tie membership
- stable near-tie membership
- high-headroom near-tie membership
- baseline-error near-tie membership
- helpful / harmful / neutral compute labels
- per-state regret and miss deltas
- teacher next-hop derived from the compute comparison

The missing reusable artifact is the frozen feature cache. Round eleven should
rebuild per-seed caches from the current baseline checkpoint using the existing
feature-cache path instead of inventing a new cache format.

## Exact code paths

Evaluation and rollout:

- `src/gnn3/eval/policy_analysis.py`
- `src/gnn3/eval/compute_helpfulness.py`
- `src/gnn3/eval/near_tie.py`
- `src/gnn3/eval/step_policy.py`

Cached audit generation:

- `scripts/run_round9_frontier_pack.py`
- `scripts/run_round10_helpfulness_audit.py`
- `scripts/run_round10_feature_cache.py`

Offline distillation and gates:

- `scripts/run_round10_helpfulness_probes.py`
- `scripts/run_round10_offline_distill.py`
- `scripts/run_round10_selective_compute.py`

Frontier scoring:

- `scripts/run_round9_frontier_guard.py`

## Where round-eleven changes should live

Round eleven should stay additive and reuse the existing runner pattern:

- shared round-eleven helper logic:
  - `src/gnn3/eval/precision_correction.py`
- round-eleven drivers:
  - `scripts/run_round11_teacher_bank.py`
  - `scripts/run_round11_defer_gate.py`
  - `scripts/run_round11_top2_comparator.py`
  - `scripts/run_round11_subset_distill.py`
  - `scripts/run_round11_deployment_study.py` only if the earlier gates justify it
- targeted validation:
  - `tests/test_precision_correction.py`

This keeps round-eleven logic close to the existing round-nine/round-ten audit
stack and avoids changing the training or rollout abstractions used by the base
model.

## Closed branches that remain closed

Round eleven does not reopen:

- large-gap targeting
- broad extra-compute policies
- current helpfulness gates as deployment gates
- aggressive residual students as currently implemented
- old reranker / planner families
- online search as-is
- plain delay mailbox
- route persistence

## Planned round-eleven program

Wave A:

- teacher-bank audit from committed helpfulness artifacts
- stable-positive correction pack construction
- overlap and robustness audit across seeds, suites, and teacher variants

Wave B:

- budgeted defer-to-teacher gates at tiny coverage budgets
- linear, MLP, and margin/regime baselines
- explicit calibration for system-level regret / miss under budget

Wave C:

- top-2 comparator with abstain
- frozen-feature and candidate-conditioned variants
- broad-vs-narrow training comparison

Wave D:

- subset-only distillation on the stable-positive pack
- pairwise, KL, residual, and one justified gated variant

Wave E:

- deployment study only if a defer gate clears the stable-positive pack gate

## GPU-hour budget

Round eleven should bias toward precise exploration while keeping a live exploit
guardrail.

Target split:

- exploit / validation / guardrail: `35–45%`
- exploration: `55–65%`

Working plan:

- exploit bucket target: `0.45` GPU-hours
- explore bucket target: `0.70` GPU-hours
- total target: about `1.15` GPU-hours

GPU lane plan:

- GPU0: guardrails, teacher-bank rebuilds, matched validation, final contender checks
- GPU1: defer gates, comparator variants, subset-only students, calibration sweeps

## Validation status before launch

The branch was revalidated before round-eleven code changes:

- `uv run ruff check src tests scripts`
- `uv run pytest tests/test_compute_helpfulness.py tests/test_policy_analysis.py tests/test_step_policy.py -q`
- `uv run python scripts/run_train.py --config configs/experiments/smoke_local_cpu.yaml`
- cached frontier-pack smoke over:
  - `reports/plots/round9_frontier_pack_seed314_decisions.csv`
  - `reports/plots/round9_frontier_pack_seed314.json`

## Promotion rule entering round eleven

Every branch must clear both gates:

- Tier 1: stable-positive correction pack must improve meaningfully
- Tier 2: full near-tie frontier pack must stay neutral-to-positive with no material large-gap or global regression

If a branch wins only on the narrow subset but regresses the full frontier pack,
it is not deployable. If it stays safe everywhere but does not move the stable
positive pack, it is not interesting.
