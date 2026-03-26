# Budget-Conditioned Memory Blend Follow-up

## Setup

This follow-up tested whether the memory-anchor family was failing because one
scalar lift gate had to serve two different operating regimes:

- ultra-low-coverage micro-budget correction
- broader matched-band correction

The new architecture kept the same geometry as
`prototype_memory_agree_blend_hybrid`:

- memory anchor score
- shared-plus-dual agreement score

But it replaced the single outer lift gate with two parallel calibration gates:

- `micro` gate with a stronger negative bias
- `matched` gate with a looser bias

The final logit took the stronger of the two lifted candidates.

Variants:

- `prototype_budget_memory`: budget-conditioned memory blend without the risk
  branch
- `prototype_budget_memory_hybrid`: same head plus the tiny margin/regime risk
  branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

Both variants are closed.

### `prototype_budget_memory`

This variant is inert.

- recovered `0%` of held-out `stable_positive_v2`
- left hard near-tie unchanged at `90.53%`
- only selected non-target states

Best point (`2.0%` nominal budget):

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `0.0000`

### `prototype_budget_memory_hybrid`

This variant is also dead on the real target.

At `0.10%` nominal budget:

- overall coverage: `0.07%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `-0.0013`

At `1.50%` nominal budget:

- overall coverage: `0.77%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `-0.0073`

At `2.00%` nominal budget:

- overall coverage: `1.01%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `-0.0079`

Large-gap controls stayed clean:

- large-gap target match: `99.79% -> 99.90%`
- large-gap mean delta regret: `-0.0065`

## Comparison against current leads

This head does not challenge any live branch.

Against the ultra-low-coverage lead:

`prototype_hybrid @ 0.10%`

- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`

`prototype_budget_memory_hybrid @ 0.10%`

- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`

Against the micro-budget memory follow-up:

`prototype_memory_agree_blend_hybrid @ 0.25%`

- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`

`prototype_budget_memory_hybrid @ 0.25%`

- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`

Against the matched-band family:

- `prototype_agree_mix_hybrid` and `prototype_evidence_agree_hybrid` both
  reach `75%` held-out stable-positive-v2 recovery and the full
  `90.53% -> 90.73%` hard near-tie band
- this head never recovers a single held-out stable-positive-v2 case

So it is not a near miss. It is a clear failure mode.

## Interpretation

This closes the “one score for two budgets” hypothesis.

Current read:

- the problem in the live memory-anchor branch is not just that one outer gate
  has to represent multiple coverage bands
- splitting that gate into micro and matched branches without changing the
  underlying geometry simply encourages the model to spend coverage on broad,
  safe, non-target fixes
- the useful Tier-1 sparse-positive signal comes from the geometry and anchor
  choice, not from this style of budget-conditioned calibration head

## Decision

Keep:

- `prototype_hybrid` as the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` as the best micro-budget Tier-1
  follow-up
- `prototype_agree_mix_hybrid` as the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` as the best aggregate-quality matched-band
  follow-up

Close:

- `prototype_budget_memory`
- `prototype_budget_memory_hybrid`

## Artifacts

- `scripts/run_budget_conditioned_memory_blend_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_budget_conditioned_memory_blend_defer_summary.csv`
- `reports/plots/prototype_budget_conditioned_memory_blend_defer_decisions.csv`
- `reports/plots/prototype_budget_conditioned_memory_blend_defer_summary.png`
