# Prototype-Memory Defer Follow-up

## Setup

This follow-up tested a small architecture change on the cached round-twelve
teacher-bank split:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`
- budgets: `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

The experiment was intended to reopen only the **architecture** side of the
precision-first defer problem. Round twelve showed that hand-built retrieval
rules were inert. The new question was whether a tiny learnable prototype bank
could recover the same sparse family more cleanly than kNN or static
prototypes.

Variants:

- `prototype_memory`: prototype bank only
- `prototype_hybrid`: prototype bank plus a tiny margin/regime risk branch
- `prototype_committee`: prototype bank trained on the stricter committee-only
  positives

## Main result

Only `prototype_hybrid` is alive.

### Dead variants

`prototype_memory`

- never recovered a held-out stable-positive-v2 case
- stayed functionally identical to baseline on hard near-tie

`prototype_committee`

- recovered only `25%` of held-out stable-positive-v2
- improved overall regret slightly, but never reached the round-eleven hard
  near-tie band

So raw prototype retrieval is still closed. The positive result comes from the
combination of learned local geometry and a tiny risk branch, not from
prototypes alone.

### Live variant: `prototype_hybrid`

At `0.75%` overall coverage:

- stable-positive-v2 recovery: `75%`
- stable-positive-v2 precision: `100%` inside the target slice
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall target match: `96.51% -> 96.68%`
- overall mean delta regret: `-0.0097`

At `1.0%` overall coverage:

- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0110`

At `2.0%` overall coverage:

- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0134`

## Comparison against existing references

Round-eleven `margin_regime` remains the strongest deployment-style baseline.

`round11 margin_regime @ 2%`

- overall coverage: `2.00%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0151`

`prototype_hybrid @ 0.75%`

- reaches the same held-out hard near-tie band
- does so at materially lower coverage
- but does **not** beat the round-eleven reference on overall regret

So the new branch is not a promoted policy yet, but it is stronger than the
round-twelve learned `margin_regime` rerun and much stronger than the dead
hand-built retrieval family.

## Interpretation

This is the first clean sign that a **learnable local-memory architecture**
helps where hand-built retrieval rules did not.

The result suggests:

- the sparse stable-positive family is locally structured in frozen-feature
  space
- fixed kNN / centroid rules were too rigid to exploit it
- a tiny learned prototype bank plus a cheap risk branch can recover most of
  the held-out stable-positive-v2 pack without reopening broad compute

What it does **not** show yet:

- a clean overall win over the round-eleven `margin_regime` reference
- a reason to change the default exploit policy

## Decision

Keep plain `multiheavy` as the default.

Promote only one new follow-up lead:

- `prototype_hybrid` as the next narrow architecture candidate for
  ultra-low-coverage defer/correct

Keep closed:

- plain prototype bank
- committee-only prototype bank
- hand-built retrieval / kNN / static prototype defer as previously tested

## Artifacts

- `scripts/run_prototype_memory_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `reports/plots/prototype_memory_defer_summary.csv`
- `reports/plots/prototype_memory_defer_decisions.csv`
- `reports/plots/prototype_memory_defer_summary.png`
