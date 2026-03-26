# Memory Dual-Lift Blend Follow-up

## Setup

This follow-up tested whether the current memory-anchor family needed a single
inner lift path at all.

The new architecture kept the same memory anchor as
`prototype_memory_agree_blend_hybrid`, but exposed two positive-lift paths in
parallel:

- a score-only agreement lift like `prototype_memory_agree_blend_hybrid`
- an evidence-aware agreement lift like `prototype_evidence_agree_hybrid`

The head then took the stronger lifted candidate per state:

- anchor candidate: memory score
- score-lift candidate
- evidence-lift candidate
- final score: max of the two lifted candidates, plus optional risk branch

Variants:

- `prototype_memory_duallift`: dual-lift memory blend without the risk branch
- `prototype_memory_duallift_hybrid`: same head plus the tiny margin/regime
  risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

Both variants are closed.

### `prototype_memory_duallift`

This variant is inert.

- recovered `0%` of held-out `stable_positive_v2`
- left hard near-tie unchanged at `90.53%`
- selected only non-target controls

Best point (`2.0%` nominal budget):

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `0.0000`

### `prototype_memory_duallift_hybrid`

This variant is also closed.

At `0.10%` nominal budget:

- overall coverage: `0.12%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall target match: `96.51% -> 96.57%`
- overall mean delta regret: `-0.0047`

At `0.50%` nominal budget:

- overall coverage: `0.51%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall mean delta regret: `-0.0061`

At `2.00%` nominal budget:

- overall coverage: `2.00%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall mean delta regret: `-0.0065`

Large-gap controls stayed clean:

- large-gap target match: unchanged at `99.79%`
- large-gap mean delta regret: `0.0000`

## Comparison against live references

`prototype_hybrid @ 0.10%`

- overall coverage: `0.12%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall mean delta regret: `-0.0050`

`prototype_memory_duallift_hybrid @ 0.10%`

- overall coverage: `0.12%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall mean delta regret: `-0.0047`

So the new head is basically policy-identical to the live ultra-low-coverage
lead, but very slightly worse on aggregate regret.

Against the memory-anchor micro-budget branch:

`prototype_memory_agree_blend_hybrid @ 0.25%`

- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`

`prototype_memory_duallift_hybrid @ 0.25%`

- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`

So it does not preserve the real micro-budget gain from the live
memory-agreement blend.

Against the matched-band family:

- it never exceeds `25%` held-out stable-positive-v2 recovery
- it never reaches the `90.53% -> 90.73%` hard near-tie band

So it also does not challenge `prototype_agree_mix_hybrid` or
`prototype_evidence_agree_hybrid`.

## Interpretation

This closes the “parallel lift path” question.

Current read:

- the score-only lift and evidence-aware lift do carry different signal
- but simply exposing both and taking the stronger candidate does not produce
  complementary behavior
- instead, the head collapses back to the weaker ultra-low-coverage pattern
  already captured by `prototype_hybrid`

So the useful lesson is:

- keep the current shortlist split by role
- do not try to combine all prototype-family lift paths in a simple max head

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

- `prototype_memory_duallift`
- `prototype_memory_duallift_hybrid`

## Artifacts

- `scripts/run_prototype_memory_duallift_blend_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_memory_duallift_blend_defer_summary.csv`
- `reports/plots/prototype_memory_duallift_blend_defer_decisions.csv`
- `reports/plots/prototype_memory_duallift_blend_defer_summary.png`
