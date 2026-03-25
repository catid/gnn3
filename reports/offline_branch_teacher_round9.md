# Round 9 Offline Branch Teacher

## Goal

This audit tests the core round-nine extra-compute thesis in teacher form first:

- only on hard near-tie states
- only on the frontier-rich `deeper_packets6` slice
- using the seed314 `compute5` checkpoint as the constructor
- branching over top candidate next hops
- evaluating whether tiny extra branch-local refinement produces a better
  teacher choice than the base constructor action

The purpose is to measure upper bound and direction, not to make online search
production-ready.

## Setup

Target slice:

- `hard_near_tie_intersection_case`

Suite:

- `a1_multiheavy_ood_deeper_packets6_round9_eval_seed314`

Grid:

- `top_k = 2`, `decision_horizon = 1`
- `top_k = 2`, `decision_horizon = 2`
- `top_k = 3`, `decision_horizon = 1`
- `top_k = 3`, `decision_horizon = 2`

Artifacts:

- `reports/plots/round9_branch_teacher_seed314_compute5_k2_h1_deeper_packets6_summary.csv`
- `reports/plots/round9_branch_teacher_seed314_compute5_k2_h2_deeper_packets6_summary.csv`
- `reports/plots/round9_branch_teacher_seed314_compute5_k3_h1_deeper_packets6_summary.csv`
- `reports/plots/round9_branch_teacher_seed314_compute5_k3_h2_deeper_packets6_summary.csv`

## Results

Each run covered the same `16` target teacher decisions.

### `top_k=2`, horizon `1`

- teacher disagreement: `12.5%`
- teacher recovery: `0.0%`
- teacher new-error: `12.5%`
- teacher target-match: `81.25%`

### `top_k=2`, horizon `2`

- teacher disagreement: `6.25%`
- teacher recovery: `0.0%`
- teacher new-error: `6.25%`
- teacher target-match: `87.5%`

### `top_k=3`, horizon `1`

- teacher disagreement: `25.0%`
- teacher recovery: `0.0%`
- teacher new-error: `25.0%`
- teacher target-match: `68.75%`

### `top_k=3`, horizon `2`

- teacher disagreement: `18.75%`
- teacher recovery: `0.0%`
- teacher new-error: `18.75%`
- teacher target-match: `75.0%`

## Interpretation

This family is now decisively negative on the audited frontier slice.

What changed:

- branch-local extra compute did move decisions
- widening the branch set increased disagreement

What did not happen:

- none of the four variants recovered any baseline hard near-tie errors
- every variant introduced new errors instead

The only mild positive inside the table is relative, not absolute:

- `top_k=2`, horizon `2` was the least harmful variant

But even that cell remained net negative and stayed below the base target match.

## Verdict

Close the offline branch-teacher family in this form.

The round-nine compute thesis is not yet dead, but this specific mechanism is:

- simple branch-local refinement from the existing constructor
- over a small top-k candidate set
- scored with the current short-horizon branch objective

This does not produce a useful teacher on the hard near-tie frontier.

## Consequence For The Round

Do not open:

- teacher-for-search distillation from this teacher
- search-triggered continuation based on this branch objective

Continue only with cheaper or more direct compute policies that do not rely on
this teacher family as their supervising signal.
