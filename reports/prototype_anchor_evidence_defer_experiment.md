# Anchor-Biased Evidence Agreement Follow-up

## Setup

This follow-up tested whether the richer `prototype_evidence_agree_hybrid`
family was overusing the broader dual-geometry branch.

The new head keeps the same two branches as the evidence-calibrated
agreement-mixture:

- shared-projection prototype branch
- dual-projection prototype branch

But it adds an extra conservative score-anchor gate before the evidence gate.
That anchor gate only sees:

- shared score
- dual score
- absolute score disagreement
- shared/dual score product

So the blend has to clear two filters:

- a score-only conservative anchor
- the richer evidence-aware gate from the prior follow-up

Variants:

- `prototype_anchor_evidence`: anchored evidence mixture without the risk branch
- `prototype_anchor_evidence_hybrid`: anchored evidence mixture plus the tiny
  margin/regime risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

This family is closed.

- `prototype_anchor_evidence` found one weak positive niche, but only by paying
  a very high false-positive cost and only moving the hard near-tie slice into
  the weaker `90.53% -> 90.60%` band.
- `prototype_anchor_evidence_hybrid` is effectively dead on the real target. It
  selected some non-target control states, but recovered `0%` of held-out
  `stable_positive_v2` and never moved hard near-tie off baseline.

### `prototype_anchor_evidence`

This variant is closed.

Best Tier-1 point (`0.50%` nominal budget):

- overall coverage: `0.25%`
- held-out `stable_positive_v2` recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0010`
- overall target match: `96.51% -> 96.53%`
- overall mean delta regret: `-0.0005`

At higher budgets it did not find more held-out positives. It only spent more
coverage on false positives:

- at `2.0%` nominal budget, overall coverage rose to `1.00%`
- held-out `stable_positive_v2` recovery stayed at `25%`
- overall false-positive rate remained about `49%`
- hard near-tie stayed capped at the weaker `90.53% -> 90.60%` band

Large-gap controls stayed clean, but the useful Tier-1 movement was too small.

### `prototype_anchor_evidence_hybrid`

This variant is dead.

At every budget:

- held-out `stable_positive_v2` recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`

Even at the largest tested point (`2.0%` nominal budget):

- overall coverage: `1.00%`
- overall target match: unchanged at `96.51%`
- overall mean delta regret: `0.0000`

So the extra anchor made the evidence-aware hybrid too conservative on the only
slice that matters.

## Comparison against current leads

`prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- held-out `stable_positive_v2` recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

`prototype_agree_mix_hybrid @ 1.5%`

- overall coverage: `1.05%`
- held-out `stable_positive_v2` recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0137`

`prototype_evidence_agree_hybrid @ 2.0%`

- overall coverage: `2.00%`
- held-out `stable_positive_v2` recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0148`

`prototype_anchor_evidence @ 0.50%`

- overall coverage: `0.25%`
- held-out `stable_positive_v2` recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0010`
- overall mean delta regret: `-0.0005`

`prototype_anchor_evidence_hybrid @ 2.0%`

- overall coverage: `1.00%`
- held-out `stable_positive_v2` recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `0.0000`

So the anchor-biased blend does not improve either of the live fronts:

- it is much weaker than `prototype_hybrid` at ultra-low coverage
- it is much weaker than `prototype_agree_mix_hybrid` and
  `prototype_evidence_agree_hybrid` at matched-band coverage

## Interpretation

The evidence-aware agreement head was not suffering from being too willing to
blend. The extra conservative anchor mostly suppresses the rare positive states
that made that family useful.

Current read:

- score-only anchoring throws away too much of the fine-grained prototype
  evidence
- the richer evidence-aware gate already needed most of its freedom to recover
  the sparse stable-positive family
- adding another conservative filter makes the hybrid branch collapse toward
  inert behavior

This is not a useful architecture direction for the current source family.

## Decision

Keep:

- `prototype_hybrid` as the best ultra-low-coverage architecture lead
- `prototype_agree_mix_hybrid` as the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` as the best aggregate-quality matched-band
  follow-up

Close:

- `prototype_anchor_evidence`
- `prototype_anchor_evidence_hybrid`

## Artifacts

- `scripts/run_prototype_anchor_evidence_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_anchor_evidence_defer_summary.csv`
- `reports/plots/prototype_anchor_evidence_defer_decisions.csv`
- `reports/plots/prototype_anchor_evidence_defer_summary.png`
