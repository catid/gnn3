# Memory-Evidence Agreement Blend Follow-up

## Setup

This follow-up tested whether the new memory-agreement blend could be improved
by swapping its score-only inner agreement gate for the richer evidence-aware
gate from `prototype_evidence_agree_hybrid`.

The architecture kept the same outer shape as
`prototype_memory_agree_blend_hybrid`:

- memory branch stays the anchor
- broader shared-plus-dual agreement branch can only add a nonnegative lift
- tiny margin/regime risk branch stays optional

The change was inside the broader branch:

- score-only shared/dual agreement gate out
- evidence-aware shared/dual agreement gate in

Variants:

- `prototype_memory_evidence_blend`: memory anchor plus evidence-aware lift
  without the risk branch
- `prototype_memory_evidence_blend_hybrid`: same head plus the tiny
  margin/regime risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

Both variants are closed.

### `prototype_memory_evidence_blend`

This variant is dead.

- recovered `0%` of held-out `stable_positive_v2`
- left hard near-tie unchanged at `90.53%`
- only selected inert non-target states

Best point (`0.50%` nominal budget):

- overall coverage: `0.20%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `0.0000`

### `prototype_memory_evidence_blend_hybrid`

This variant is also closed.

At `0.50%` nominal budget:

- overall coverage: `0.51%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0023`
- overall target match: `96.51% -> 96.65%`
- overall mean delta regret: `-0.0081`

At `1.00%` nominal budget:

- overall coverage: `1.01%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0023`
- overall mean delta regret: `-0.0122`

At `2.00%` nominal budget:

- overall coverage: `2.00%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0042`
- overall target match: `96.51% -> 96.79%`
- overall mean delta regret: `-0.0127`

Large-gap controls stayed clean:

- large-gap target match: `99.79% -> 99.90%`
- large-gap mean delta regret: `-0.0065`

## Comparison against the live memory-anchor branch

Live `prototype_memory_agree_blend_hybrid @ 0.25%`

- overall coverage: `0.25%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0049`

`prototype_memory_evidence_blend_hybrid @ 0.25%`

- overall coverage: `0.25%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `-0.0050`

So the richer inner evidence gate **destroyed** the micro-budget Tier-1 gain
that made the memory-agreement blend interesting in the first place.

At `0.50%` nominal budget:

`prototype_memory_agree_blend_hybrid`

- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0093`

`prototype_memory_evidence_blend_hybrid`

- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0023`
- overall mean delta regret: `-0.0081`

So it also loses clearly at the next micro-budget point.

At `2.00%` nominal budget the new head still does not catch the live frontier:

- `prototype_memory_evidence_blend_hybrid`: `50%` recovery, `90.53% -> 90.66%`
- `prototype_hybrid`: `75%` recovery, `90.53% -> 90.73%`
- `prototype_agree_mix_hybrid`: `75%` recovery, `90.53% -> 90.73%`
- `prototype_evidence_agree_hybrid`: `75%` recovery, `90.53% -> 90.73%`

So the new head is neither the best micro-budget branch nor a matched-band
improvement.

## Interpretation

This closes a useful architecture question.

Current read:

- the memory anchor helps because it is sharp and locally selective
- the evidence-aware agreement gate helps in the matched-band family because it
  broadens the blend more intelligently
- putting those two together does **not** stack cleanly
- under the memory anchor, the richer inner gate becomes too conservative at
  the low-coverage frontier and still not strong enough to reach the full
  `90.73%` band later

So the right lesson is:

- keep `prototype_memory_agree_blend_hybrid` for the micro-budget niche
- keep `prototype_evidence_agree_hybrid` for higher-coverage aggregate quality
- do not try to fuse them in this direct way again

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

- `prototype_memory_evidence_blend`
- `prototype_memory_evidence_blend_hybrid`

## Artifacts

- `scripts/run_prototype_memory_evidence_agreement_blend_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_memory_evidence_agreement_blend_defer_summary.csv`
- `reports/plots/prototype_memory_evidence_agreement_blend_defer_decisions.csv`
- `reports/plots/prototype_memory_evidence_agreement_blend_defer_summary.png`
