# Prototype Mixture Defer Follow-up

## Setup

This follow-up tested whether the live prototype family needed a **mixture of
geometries**, not just another residual or gate.

The new head combines two prototype scores:

- a shared-projection prototype score, like the live `prototype_hybrid` family
- a dual-projection prototype score with separate positive and negative query
  spaces

The final defer score is a learned mixture of those two geometry branches,
optionally followed by the same tiny margin/regime risk branch used by the live
prototype lead.

Variants:

- `prototype_mixture`: geometry mixture only
- `prototype_mixture_hybrid`: geometry mixture plus the tiny risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

`prototype_mixture` is dead. `prototype_mixture_hybrid` is the first real
positive architecture follow-up since `prototype_hybrid`.

### `prototype_mixture`

This variant is closed.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- never moved held-out hard near-tie above baseline
- only found a tiny overall-only niche

Best point (`2.0%` nominal budget):

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: `90.53% -> 90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall target match: `96.51% -> 96.54%`
- overall mean delta regret: `-0.0016`

### `prototype_mixture_hybrid`

This variant is alive.

At `0.75%` nominal budget:

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `50%`
- stable-positive-v2 precision: `100%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall target match: `96.51% -> 96.74%`
- overall mean delta regret: `-0.0124`

At `2.0%` nominal budget:

- overall coverage: `1.84%`
- stable-positive-v2 recovery: `75%`
- stable-positive-v2 precision: `100%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall target match: `96.51% -> 96.78%`
- overall mean delta regret: `-0.0138`

Large-gap controls stayed clean at the same `2.0%` point:

- large-gap target match: `99.79% -> 99.90%`
- large-gap mean delta regret: `-0.0065`

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

`prototype_mixture_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0124`

So the mixture branch is **not** the new ultra-low-coverage leader.

But at matched higher coverage it becomes the first follow-up to fully catch
the live prototype band:

Live `prototype_hybrid @ 2.0%`

- overall coverage: `1.92%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0134`

`prototype_mixture_hybrid @ 2.0%`

- overall coverage: `1.84%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0138`

So the mixture branch matches the hard frontier and slightly improves aggregate
regret at comparable coverage.

## Interpretation

This is the first clean sign that **changing the prototype geometry itself**
helps beyond the current live lead.

Current read:

- plain new geometry is still not enough
- but mixing shared and asymmetric geometry with the tiny risk branch adds real
  value
- the two geometry families appear complementary

What it does **not** show yet:

- a cleaner ultra-low-coverage leader than `prototype_hybrid`
- a reason to change the default exploit policy

## Decision

Keep:

- `prototype_hybrid` as the best ultra-low-coverage architecture lead
- `prototype_mixture_hybrid` as the new best matched-coverage follow-up inside
  the prototype family

Close:

- `prototype_mixture`

If another architecture round opens inside this family, compare those two heads
directly on a broader matched deployment panel before reopening any wider
family.

## Artifacts

- `scripts/run_prototype_mixture_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_mixture_defer_summary.csv`
- `reports/plots/prototype_mixture_defer_decisions.csv`
- `reports/plots/prototype_mixture_defer_summary.png`
