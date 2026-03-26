# Prototype Suppressor Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` lead was mainly
missing a better way to suppress false positives rather than a better way to
retrieve positives.

The new head keeps the same projected prototype-space idea, but changes the
bank structure:

- one positive bank for stable-positive states
- one neutral negative bank for generic non-positive states
- one explicit harmful bank for harmful teacher-bank cases

The score then becomes:

- positive score
- minus neutral score
- minus a learned harmful suppressor term

Variants:

- `prototype_suppressor`: suppressor banks only
- `prototype_suppressor_hybrid`: suppressor banks plus the usual risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

The family is closed.

### `prototype_suppressor`

This variant is dead.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- never improved hard near-tie above baseline
- reduced fully to baseline behavior on the held-out panel

### `prototype_suppressor_hybrid`

This variant is also effectively dead.

Best overall point (`2.0%` nominal budget):

- overall coverage: `0.05%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall target match: `96.51% -> 96.52%`
- overall mean delta regret: `-0.0014`

So the hybrid only found a tiny conservative overall-risk niche and still did
not recover a single held-out stable-positive case.

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Best suppressor point (`prototype_suppressor_hybrid @ 2.0%`)

- overall coverage: `0.05%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `-0.0014`

So explicit harmful memories did not sharpen the true sparse-positive correction
problem at all.

## Interpretation

The live prototype family does not appear bottlenecked by lacking a separate
harmful suppressor bank.

Current read:

- the existing negative handling in `prototype_hybrid` is already enough for the
  limited harmful cases that matter
- carving out a dedicated harmful bank suppresses the whole signal too
  aggressively
- the missing gain is not simply “more explicit harmful memories”

This closes another plausible architecture path:

- the next gain is unlikely to come from splitting negatives into neutral and
  harmful suppressor banks
- the live `prototype_hybrid` remains the only architecture lead worth keeping
  open in this family

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `prototype_suppressor`
- `prototype_suppressor_hybrid`

Do not reopen suppressor-bank prototype defer in this form.

## Artifacts

- `scripts/run_prototype_suppressor_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_suppressor_defer_summary.csv`
- `reports/plots/prototype_suppressor_defer_decisions.csv`
- `reports/plots/prototype_suppressor_defer_summary.png`
