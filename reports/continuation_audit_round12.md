# Continuation Audit Round 12

## Accepted baseline

- `multiheavy` remains the default exploit policy.
- The corrected round-4 anchor still stands: mean regret `1.32` vs fresh `E3` `2.25`, mean p95 regret `4.77` vs `10.48`, mean deadline miss `41.7%` vs `54.2%`.
- Round 6 reproduced the same baseline band, so baseline drift is not the main problem.
- Large-gap hard-feasible errors remain basically solved. The canonical negative control is still the large-gap hard-feasible slice from the round-9 frontier pack.

## Current frontier and round-11 reference

Round 9 established the canonical promotion surface:

- Tier 2: the round-9/10 hard near-tie frontier pack
- Key slices:
  - `hard_near_tie_intersection_case`
  - `stable_near_tie_case`
  - `high_headroom_near_tie_case`
  - `baseline_error_hard_near_tie_case`
  - `large_gap_hard_feasible_case`

Round 10 narrowed the live source families:

- broad hard near-tie extra compute is net negative
- high-headroom near-tie remains genuinely positive
- baseline-error near-tie remains genuinely positive

Round 11 narrowed the deployment reference:

- stable-positive correction pack was real but extremely sparse
- only the tiny `margin_regime` defer system survived
- best reference operating band was `1%` to `2%` nominal defer coverage
- that reference is useful, but still not promotable

Round 12 therefore targets two gates:

- Tier 1: a richer stable-positive-v2 correction pack
- Tier 2: the existing full near-tie frontier pack

Any claimed progress must improve Tier 1 and stay safe on Tier 2, while beating or matching the round-11
`margin_regime` defer reference at comparable coverage.

## Existing code paths

### Cached near-tie audits and frontier scoring

- Frontier pack generation: [scripts/run_round9_frontier_pack.py](/home/catid/gnn3/scripts/run_round9_frontier_pack.py)
- Frontier guard comparison: [scripts/run_round9_frontier_guard.py](/home/catid/gnn3/scripts/run_round9_frontier_guard.py)
- Shared frontier/helpfulness helpers: [src/gnn3/eval/compute_helpfulness.py](/home/catid/gnn3/src/gnn3/eval/compute_helpfulness.py)
- Stable-positive and defer helper utilities: [src/gnn3/eval/precision_correction.py](/home/catid/gnn3/src/gnn3/eval/precision_correction.py)

### Teacher-bank evaluation

- Round-11 teacher bank: [scripts/run_round11_teacher_bank.py](/home/catid/gnn3/scripts/run_round11_teacher_bank.py)
- Existing outputs:
  - `reports/plots/round11_teacher_bank_decisions.csv`
  - `reports/plots/round11_teacher_bank_summary.csv`
  - `reports/plots/round11_teacher_bank_seed_overlap.csv`
  - `reports/plots/round11_teacher_bank_stable_positive_manifest.csv`

### Defer-gate evaluation and deployment study

- Round-11 defer gate: [scripts/run_round11_defer_gate.py](/home/catid/gnn3/scripts/run_round11_defer_gate.py)
- Round-11 deployment aggregation: [scripts/run_round11_deployment_study.py](/home/catid/gnn3/scripts/run_round11_deployment_study.py)
- Existing reference outputs:
  - `reports/plots/round11_defer_gate_heldout_summary.csv`
  - `reports/plots/round11_deployment_study_summary.csv`

### Feature caches and student surfaces

- Feature cache builder: [scripts/run_round10_feature_cache.py](/home/catid/gnn3/scripts/run_round10_feature_cache.py)
- Existing local feature caches:
  - `reports/plots/round11_feature_cache_seed314.pt`
  - `reports/plots/round11_feature_cache_seed315.pt`
  - `reports/plots/round11_feature_cache_seed316.pt`
- Matching metadata:
  - `reports/plots/round11_feature_cache_seed314_metadata.csv`
  - `reports/plots/round11_feature_cache_seed315_metadata.csv`
  - `reports/plots/round11_feature_cache_seed316_metadata.csv`

### Existing student/comparator code paths

- Subset-only distillation: [scripts/run_round11_subset_distill.py](/home/catid/gnn3/scripts/run_round11_subset_distill.py)
- Top-2 comparator: [scripts/run_round11_top2_comparator.py](/home/catid/gnn3/scripts/run_round11_top2_comparator.py)

Round 12 should not reopen those families directly. The safest reuse is to borrow their feature loading and
evaluation structure for sparse defer/correct tooling, not to relaunch broad student policies.

## Reusable artifacts

Round 12 can reuse the following directly without regenerating expensive rollouts:

- Round-10 compute helpfulness decisions:
  - `reports/plots/round10_helpfulness_seed314_decisions.csv`
  - `reports/plots/round10_helpfulness_seed315_decisions.csv`
  - `reports/plots/round10_helpfulness_seed316_decisions.csv`
- Round-10 selective-compute held-out decisions:
  - `reports/plots/round10_selective_compute_decisions.csv`
- Round-11 teacher bank and stable-positive manifests:
  - `reports/plots/round11_teacher_bank_decisions.csv`
  - `reports/plots/round11_teacher_bank_stable_positive_manifest.csv`
- Round-11 defer-gate and deployment summaries:
  - `reports/plots/round11_defer_gate_heldout_summary.csv`
  - `reports/plots/round11_deployment_study_summary.csv`
- Round-11 subset-distill held-out decisions:
  - `reports/plots/round11_subset_distill_heldout_decisions.csv`
- Seed-314 auxiliary compute-policy decisions from round 9:
  - `reports/plots/round9_compute_policy_seed314_deeper_packets6_*_decisions.csv`

These artifacts are sufficient to build a richer offline teacher bank and to test ultra-low-coverage defer
families before considering any new student retry.

## Where round-12 changes should live

- Shared sparse-correction helpers should extend:
  - [src/gnn3/eval/precision_correction.py](/home/catid/gnn3/src/gnn3/eval/precision_correction.py)
- New round-12 experiment entrypoints should live in `scripts/`, parallel to round-10 and round-11 scripts:
  - `run_round12_teacher_bank.py`
  - `run_round12_ultralow_defer.py`
  - `run_round12_retrieval_defer.py`
  - `run_round12_committee_defer.py`
  - `run_round12_positive_mining.py`
  - `run_round12_deployment_study.py`
- Tests for new helper logic should live under `tests/`, alongside existing precision-correction and
  compute-helpfulness tests.

## Closed branches that remain closed

Round 12 should not reopen these as full families:

- large-gap targeting
- broad extra-compute policies
- broad helpfulness gates
- top-2 comparator as implemented in round 11
- subset-only distillation as implemented in round 11
- aggressive residual students
- old reranker / planner families
- online search
- delay mailbox
- route persistence

The only acceptable student work in round 12 is a very conservative retry, and only if teacher-bank expansion
materially improves the source family first.

## Planned round-12 experiment shape

Round 12 is a narrow teacher-bank and ultra-low-coverage defer campaign:

1. Expand the offline teacher bank from existing cross-seed artifacts plus seed-314 auxiliary compute-policy
   variants.
2. Build a stable-positive-v2 pack that is more conservative than round 11 and explicitly tracks committee and
   threshold sensitivity.
3. Re-evaluate defer at much smaller budgets than round 11.
4. Test retrieval/prototype and committee defer, because the positive family appears rare and locally structured.
5. Only then test positive-mining as a source-family expansion tool.
6. Only retry a conservative student if the source family materially improves.

## Exploit / explore GPU-hour budget

Planned split for round 12:

- exploit / validation / guardrail: `40%`
- exploration: `60%`

Expected use:

- GPU0:
  - baseline refreshes
  - teacher-bank audits
  - matched validations
  - deployment sweeps
- GPU1:
  - ultra-low-coverage defer sweeps
  - retrieval / prototype defer
  - committee defer
  - positive-mining and any conditional student retry

When either GPU is idle, fill it with:

- cached audit generation
- calibration sweeps
- matched pack evals
- report plots
- low-cost ablation reruns

## Validation status before launch

Round-12 branch validation was re-run before new code changes:

- `uv run ruff check src tests scripts`
- `uv run pytest tests/test_precision_correction.py tests/test_compute_helpfulness.py tests/test_policy_analysis.py tests/test_step_policy.py -q`
- `uv run python scripts/run_train.py --config configs/experiments/smoke_local_cpu.yaml`
- cached precision path smoke:
  - `uv run python scripts/run_round11_teacher_bank.py --decision-csvs ... --output-prefix reports/plots/round12_validation_teacher_bank`

That is enough to proceed with round-12 code and experiment work on the current repo state.
