# Continuation Audit Round 13

## Accepted Baseline

- Default deployment policy remains plain `multiheavy`.
- The corrected round-four anchor still defines the accepted baseline band:
  - mean regret `1.32` versus fresh E3 `2.25`
  - mean p95 regret `4.77` versus fresh E3 `10.48`
  - mean deadline miss `41.7%` versus fresh E3 `54.2%`
- Round six reproduced that band, so baseline stability is accepted.
- The open problem is not broad compute quality. The remaining opportunity is narrow correction on the hard near-tie slice, especially the `high_headroom_near_tie` and `baseline_error_hard_near_tie` families.

## Current Promotion Surfaces

### Tier 1

- Current narrow target pack is `stable_positive_v2`.
- Current source artifact:
  - `reports/plots/round12_teacher_bank_stable_positive_v2_manifest.csv`
- Current broader teacher-bank frame:
  - `reports/plots/round12_teacher_bank_decisions.csv`
- Round-thirteen work may promote a machine-readable `stable_positive_v3`, but only after overlap and stability checks.

### Tier 2

- The canonical broad promotion surface remains the round-nine/ten/eleven near-tie frontier pack:
  - `hard_near_tie_intersection_case`
  - `stable_near_tie_case`
  - `high_headroom_near_tie_case`
  - `baseline_error_hard_near_tie_case`
  - `large_gap_hard_feasible_case`
- These slices are already wired into the prototype runner `_slice_map(...)` implementations and into the round-eleven/twelve report stack.

## Accepted Live Shortlist And Operating Regions

### `prototype_memory_agree_blend_hybrid`

- Role: micro-budget Tier-1 reference.
- Accepted operating region:
  - `0.25%` overall coverage: `50%` held-out `stable_positive_v2` recovery
  - weaker but real hard near-tie band `90.53% -> 90.66%`
- Code path:
  - model: `src/gnn3/models/prototype_defer.py` `MemoryAgreementBlendPrototypeDeferHead`
  - runner: `scripts/run_prototype_memory_agreement_blend_defer.py`
  - artifacts: `reports/plots/prototype_memory_agreement_blend_defer_*`

### `prototype_sharp_negative_tail_support_agree_mix_hybrid`

- Role: best sub-`1%` full-band / coverage-efficient matched-band point.
- Accepted operating region:
  - `0.75%` overall coverage: `75%` held-out `stable_positive_v2`
  - hard near-tie `90.53% -> 90.73%`
  - overall mean delta regret about `-0.0144`
- Code path:
  - model: `src/gnn3/models/prototype_defer.py` `SharpNegativeTailSupportAgreementMixturePrototypeDeferHead`
  - runner: `scripts/run_prototype_sharp_negative_tail_support_agreement_mixture_defer.py`
  - artifacts: `reports/plots/prototype_sharp_negative_tail_support_agreement_mixture_defer_*`

### `prototype_negative_tail_support_agree_mix_hybrid`

- Role: maximum held-out recall around `1%` coverage.
- Accepted operating region:
  - `1.0%` overall coverage: `100%` held-out `stable_positive_v2`
  - hard near-tie `90.53% -> 90.80%`
- Code path:
  - model: `src/gnn3/models/prototype_defer.py` `NegativeTailSupportAgreementMixturePrototypeDeferHead`
  - runner: `scripts/run_prototype_negative_tail_support_agreement_mixture_defer.py`
  - artifacts: `reports/plots/prototype_negative_tail_support_agreement_mixture_defer_*`

### `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

- Role: higher-budget matched-band and higher-budget max-recall leader.
- Accepted operating region:
  - `1.01%` coverage: `75%` held-out `stable_positive_v2`, hard near-tie `90.53% -> 90.73%`, overall mean delta regret `-0.0145`
  - `1.52%` coverage: same `75% / 90.73%`, overall mean delta regret `-0.0159`
  - `2.00%` coverage: `100%` held-out `stable_positive_v2`, hard near-tie `90.53% -> 90.80%`, overall mean delta regret `-0.0167`
- Code path:
  - model: `src/gnn3/models/prototype_defer.py` `BranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
  - runner: `scripts/run_prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
  - artifacts: `reports/plots/prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer_*`

## Exact Reusable Code Paths

### Prototype Scoring / Support-Agree Mix

- Base shared/dual agreement mixture:
  - `src/gnn3/models/prototype_defer.py:372`
  - `AgreementMixturePrototypeDeferHead`
- Support-weighted bank readout:
  - `src/gnn3/models/prototype_defer.py:526`
  - `SupportWeightedAgreementMixturePrototypeDeferHead`

### Negative Cleanup

- Fixed negative-tail cleanup:
  - `src/gnn3/models/prototype_defer.py:860`
  - `NegativeTailSupportAgreementMixturePrototypeDeferHead`
- Sharp negative-tail cleanup:
  - `src/gnn3/models/prototype_defer.py:1096`
  - `SharpNegativeTailSupportAgreementMixturePrototypeDeferHead`
- Branch-specific sharp cleanup amplitude:
  - `src/gnn3/models/prototype_defer.py:1265`
  - `BranchStrengthSharpNegativeTailSupportAgreementMixturePrototypeDeferHead`

### Branchwise Fusion

- Branchwise hard max inside shared and dual branches:
  - `src/gnn3/models/prototype_defer.py:1577`
  - `BranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead`
- Closed but important comparison families:
  - branchwise lift: `src/gnn3/models/prototype_defer.py:1644`
  - branchwise margin-max: `src/gnn3/models/prototype_defer.py:1779`
  - joint-support branchwise fusion: `src/gnn3/models/prototype_defer.py:1892`

### Cached Near-Tie Audits / Stable-Positive Labeling

- shared helper surface:
  - `src/gnn3/eval/precision_correction.py`
- key functions:
  - `annotate_stable_positive_pack(...)` at `:40`
  - `build_source_signature(...)` at `:89`
  - `teacher_effect_labels(...)` at `:122`
  - `top_fraction_mask(...)` at `:198`
  - `decision_augmented_features(...)` at `:226`
  - `margin_regime_features(...)` at `:232`

### Defer / Deployment Evaluation

- round-eleven deployment comparison:
  - `scripts/run_round11_deployment_study.py`
- round-twelve deployment comparison:
  - `scripts/run_round12_deployment_study.py`
- round-twelve ultralow / retrieval defer comparison:
  - `scripts/run_round12_ultralow_defer.py`
  - `scripts/run_round12_retrieval_defer.py`

### Frontier-Pack Scoring

- Prototype family currently scores coverage cuts through per-runner `_evaluate_budget(...)` plus `_slice_map(...)`.
- The reusable scoring contract already exists in the live shortlist runners:
  - `scripts/run_prototype_memory_agreement_blend_defer.py`
  - `scripts/run_prototype_sharp_negative_tail_support_agreement_mixture_defer.py`
  - `scripts/run_prototype_negative_tail_support_agreement_mixture_defer.py`
  - `scripts/run_prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer.py`
- Round 13 should add shared frontier analysis on top of those outputs instead of replacing those runners.

## Exact Reusable Artifacts

- Training cache:
  - `reports/plots/round11_feature_cache_seed314.pt`
  - `reports/plots/round11_feature_cache_seed314_metadata.csv`
- Held-out caches:
  - `reports/plots/round11_feature_cache_seed315.pt`
  - `reports/plots/round11_feature_cache_seed315_metadata.csv`
  - `reports/plots/round11_feature_cache_seed316.pt`
  - `reports/plots/round11_feature_cache_seed316_metadata.csv`
- Teacher-bank decisions and manifests:
  - `reports/plots/round12_teacher_bank_decisions.csv`
  - `reports/plots/round12_teacher_bank_stable_positive_v2_manifest.csv`
  - `reports/plots/round12_teacher_bank_stable_positive_v2_committee_manifest.csv`
  - `reports/plots/round12_teacher_bank_harmful_teacher_bank_manifest.csv`
- Existing shortlist outputs ready for aggregation or comparison:
  - `reports/plots/prototype_memory_agreement_blend_defer_*`
  - `reports/plots/prototype_sharp_negative_tail_support_agreement_mixture_defer_*`
  - `reports/plots/prototype_negative_tail_support_agreement_mixture_defer_*`
  - `reports/plots/prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer_*`
- Existing report anchors:
  - `reports/continuation_report_round4.md`
  - `reports/continuation_report_round10.md`
  - `reports/continuation_report_round11.md`
  - `reports/teacher_bank_round11.md`
  - `reports/stable_positive_pack_round11.md`
  - `reports/defer_gate_round11.md`
  - `reports/deployment_study_round11.md`

## Where Round-13 Changes Should Live

- New campaign runners and aggregation logic should be additive under `scripts/`:
  - `scripts/run_round13_frontier_sweep.py`
  - `scripts/run_round13_branchwise_ablation.py`
  - `scripts/run_round13_stable_positive_v3.py`
  - `scripts/run_round13_hierarchical_defer.py`
  - `scripts/run_round13_retrieval_calibration.py`
  - conditional student / deployment scripts only if justified
- New model variants, if needed, should remain surgical additions inside:
  - `src/gnn3/models/prototype_defer.py`
- Shared evaluation helpers should stay additive inside:
  - `src/gnn3/eval/precision_correction.py`
- Reports and plot outputs should remain under:
  - `reports/`
  - `reports/plots/`

## Round-13 Exploit / Explore GPU Budget

- Planned total budget: `24.0` GPU-hours
- Exploit / validation / frontier mapping: `12.0` GPU-hours
- Explore / ablation / calibration / conditional student: `12.0` GPU-hours

### Planned Allocation

- GPU0 `12.0` GPU-hours
  - Wave A frontier sweep and robustness reruns: `4.0`
  - Wave D hierarchical defer / dispatcher: `3.0`
  - Wave G deployment panel if justified: `2.0`
  - slack for matched reruns and report plots: `3.0`
- GPU1 `12.0` GPU-hours
  - Wave B branchwise-max ablations: `4.0`
  - Wave C stable-positive-v3 mining: `3.0`
  - Wave E retrieval / calibration variants: `3.0`
  - Wave F conservative student retry if justified: `2.0`

## Round-13 Execution Notes

- Promotion remains two-gated:
  - Tier 1 `stable_positive_v2` initially, then `stable_positive_v3` only if it expands cleanly
  - Tier 2 full near-tie frontier pack
- Round 13 should not reopen:
  - broad extra-compute policies
  - broad helpfulness gates
  - old broad subset-distill families
  - top-2 comparator
  - old planner / reranker / mailbox / route-persistence families
- The first concrete deliverables after this audit are:
  - fast validation green
  - matched-budget frontier sweep across the four live shortlist systems
  - dominance table and early demotions if any shortlist member is fully dominated
