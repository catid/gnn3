# Memory-Agreement Blend Follow-up

## Setup

This follow-up tested whether the live prototype-memory geometry and the newer
agreement-mixture geometry could be combined without giving up the
ultra-low-coverage behavior of the memory branch.

The new head has two layers of control:

- a `prototype_memory`-style branch that produces the anchor score
- an inner agreement gate that mixes shared and dual prototype geometry
- an outer gate that sees both branch scores and can apply only a
  nonnegative lift from the agreement score above the memory anchor

So the architecture is intentionally asymmetric:

- memory score is the default
- agreement geometry can only widen the score when it clears the outer gate
- the tiny margin/regime risk branch is still optional

Variants:

- `prototype_memory_agree_blend`: anchor-plus-lift without the risk branch
- `prototype_memory_agree_blend_hybrid`: anchor-plus-lift with the tiny
  margin/regime risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

`prototype_memory_agree_blend` is closed. `prototype_memory_agree_blend_hybrid`
is alive, but only as a **micro-budget Tier-1 follow-up**.

### `prototype_memory_agree_blend`

This variant is effectively dead.

- recovered `0%` of held-out `stable_positive_v2`
- left hard near-tie unchanged at `90.53%`
- only selected inert control states and broad-safe non-target fixes

Best point (`0.50%` nominal budget):

- overall coverage: `0.51%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall target match: `96.51% -> 96.53%`
- overall mean delta regret: `-0.0009`

### `prototype_memory_agree_blend_hybrid`

This variant is a real positive follow-up, but only in the sub-`0.5%` regime.

At `0.25%` nominal budget:

- overall coverage: `0.25%`
- stable-positive-v2 recovery: `50%`
- stable-positive-v2 precision: `100%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall target match: `96.51% -> 96.59%`
- overall mean delta regret: `-0.0049`

At `0.50%` nominal budget:

- overall coverage: `0.51%`
- stable-positive-v2 recovery: `50%`
- stable-positive-v2 precision: `100%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall target match: `96.51% -> 96.67%`
- overall mean delta regret: `-0.0093`

At `1.50%` nominal budget:

- overall coverage: `1.09%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall target match: `96.51% -> 96.75%`
- overall mean delta regret: `-0.0124`

Large-gap controls stayed clean at those points:

- large-gap target match: `99.79% -> 99.90%`
- large-gap mean delta regret: `-0.0065`

## Comparison against current leads

`prototype_hybrid @ 0.25%`

- overall coverage: `0.25%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall mean delta regret: `-0.0064`

`prototype_memory_agree_blend_hybrid @ 0.25%`

- overall coverage: `0.25%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0049`

So the new hybrid clearly improves the **Tier-1 micro-budget point** at the
same tiny coverage, but it pays for that with weaker aggregate regret.

`prototype_hybrid @ 0.50%`

- overall coverage: `0.51%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall mean delta regret: `-0.0081`

`prototype_memory_agree_blend_hybrid @ 0.50%`

- overall coverage: `0.51%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0093`

So it also beats the `prototype_hybrid` point at the same mid-micro budget.

But it does **not** replace the live ultra-low-coverage lead:

`prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

The new memory-agreement blend never reaches that full `75%` recovery /
`90.73%` hard-slice band. It saturates early at the weaker `50%` recovery /
`90.66%` band.

## Interpretation

This is a real architecture improvement, but not the one that replaces the
current top leads.

Current read:

- memory geometry is a stronger ultra-low-coverage anchor than the
  score-only agreement family
- agreement geometry still adds useful Tier-1 lift when it is forced to act
  only as a positive override
- that combination improves the first real micro-budget correction band

What it does **not** show:

- a reason to replace `prototype_hybrid` as the best ultra-low-coverage lead
- a reason to replace `prototype_agree_mix_hybrid` as the most
  coverage-efficient matched-band follow-up
- a reason to replace `prototype_evidence_agree_hybrid` as the best
  aggregate-quality matched-band follow-up

So the correct role for this head is:

- a **micro-budget Tier-1 contender** below roughly `0.5%` overall coverage
- not a new primary prototype-family leader

## Decision

Keep:

- `prototype_hybrid` as the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` as the best current micro-budget
  Tier-1 follow-up
- `prototype_agree_mix_hybrid` as the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` as the best aggregate-quality
  matched-band follow-up

Close:

- `prototype_memory_agree_blend`

## Artifacts

- `scripts/run_prototype_memory_agreement_blend_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_memory_agreement_blend_defer_summary.csv`
- `reports/plots/prototype_memory_agreement_blend_defer_decisions.csv`
- `reports/plots/prototype_memory_agreement_blend_defer_summary.png`
