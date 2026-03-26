# Prototype Pair-Context Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` branch still needed a
small amount of **explicit top-2 pair context**, but only as an auxiliary
branch instead of a full feature replacement.

Architecture:

- prototype bank still runs on `decision_augmented_features(...)`
- auxiliary branch adds a lightweight top-2 pair summary

Pair summaries included:

- top-1 / top-2 score, cost, slack, and on-time values
- corresponding pairwise gaps
- optional candidate-feature delta norm and cosine similarity

Variants:

- `pair_scalar_hybrid`
- `pair_metric_hybrid`

Train/eval split stayed the same:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`

## Main result

The family is closed.

### `pair_scalar_hybrid`

- recovered only `25%` of held-out stable-positive-v2
- best hard near-tie target match only reached `90.53% -> 90.60%`
- hard near-tie mean delta regret only reached `-0.0023`

### `pair_metric_hybrid`

- best point was `2%` overall coverage
- recovered only `50%` of held-out stable-positive-v2
- hard near-tie target match only reached `90.53% -> 90.66%`
- hard near-tie mean delta regret only reached `-0.0042`

That is still materially weaker than the live `prototype_hybrid` lead.

## Comparison against the live lead

`prototype_hybrid @ 0.75%`

- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`

Best pair-context point (`pair_metric_hybrid @ 2%`)

- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0042`

So the pair-context branch improved overall caution, but it diluted the exact
Tier-1 signal we care about.

## Interpretation

The remaining live architecture signal does **not** seem to come from missing
explicit top-2 pair summaries.

The simpler explanation still fits best:

- the sparse positive family is recoverable through the prototype-memory
  geometry already
- auxiliary pair-context branches make the model more conservative
- but they do not improve the actual stable-positive recovery problem

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `pair_scalar_hybrid`
- `pair_metric_hybrid`

Do not reopen prototype pair-context defer in this form.

## Artifacts

- `scripts/run_prototype_pair_context_defer.py`
- `reports/plots/prototype_pair_context_defer_summary.csv`
- `reports/plots/prototype_pair_context_defer_decisions.csv`
- `reports/plots/prototype_pair_context_defer_summary.png`
