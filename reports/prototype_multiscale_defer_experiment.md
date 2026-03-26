# Prototype Multiscale Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` lead was missing a
coarser notion of the sparse positive family.

The new head keeps the same local positive/negative prototype bank geometry,
but adds a second **global centroid branch**:

- local branch: the same log-sum-exp positive vs negative prototype score
- coarse branch: positive-centroid vs negative-centroid similarity in the same
  encoded prototype space

Variants:

- `prototype_multiscale`: local prototypes plus coarse centroid branch
- `prototype_multiscale_hybrid`: the same multiscale score plus the existing
  additive risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

The family is closed.

### `prototype_multiscale`

This was the better Tier-1 member, but it still missed the live lead.

Best working point (`1.0%` nominal budget):

- overall coverage: `0.51%`
- stable-positive-v2 recovery: `50%`
- stable-positive-v2 precision: `75%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0029`
- overall target match: `96.51% -> 96.56%`
- overall mean delta regret: `-0.0014`

Larger budgets did not improve the real frontier:

- stable-positive-v2 stayed capped at `50%`
- hard near-tie stayed capped at `90.53% -> 90.66%`
- overall regret gains stayed very small

### `prototype_multiscale_hybrid`

This variant was better on aggregate overall regret, but worse on the actual
target subset.

Best overall point (`2.0%` nominal budget):

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `25%`
- stable-positive-v2 precision: `75%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0023`
- overall target match: `96.51% -> 96.80%`
- overall mean delta regret: `-0.0130`

So the hybrid acts more like a broad caution smoother than a sparse positive
retriever.

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Best Tier-1 multiscale point (`prototype_multiscale @ 1.0%`)

- overall coverage: `0.51%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0029`
- overall mean delta regret: `-0.0014`

Best aggregate multiscale point (`prototype_multiscale_hybrid @ 2.0%`)

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0023`
- overall mean delta regret: `-0.0130`

So the coarse centroid branch slightly improves broad overall caution, but it
does not improve the real sparse-positive correction problem.

## Interpretation

The live prototype family does not appear bottlenecked by lacking a global
positive centroid anchor.

Current read:

- local prototype retrieval already captures the useful rare structure
- a centroid/global branch mostly smooths the score toward broader conservative
  behavior
- that smoothing helps aggregate overall regret a bit, but it suppresses the
  rare held-out positive retrieval needed on Tier-1

This closes another plausible architecture path:

- the next gain is unlikely to come from layering a coarse prototype centroid
  over the existing bank
- the live `prototype_hybrid` remains the only architecture lead worth keeping
  open in this family

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `prototype_multiscale`
- `prototype_multiscale_hybrid`

Do not reopen multiscale centroid prototype defer in this form.

## Artifacts

- `scripts/run_prototype_multiscale_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_multiscale_defer_summary.csv`
- `reports/plots/prototype_multiscale_defer_decisions.csv`
- `reports/plots/prototype_multiscale_defer_summary.png`
