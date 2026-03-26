# Sharpness-Aware Evidence Agreement Follow-up

## Setup

This follow-up tested whether the live evidence-agreement family was missing
prototype sharpness information.

The existing evidence-aware gate already sees top positive and negative matches,
but it does not know whether those matches are sharp or diffuse across the bank.
The new head adds branch-level sharpness features:

- shared positive top-1 minus top-2 gap
- shared negative top-1 minus top-2 gap
- dual positive top-1 minus top-2 gap
- dual negative top-1 minus top-2 gap

Those sharpness signals are added on top of the prior evidence features:

- shared and dual scores
- shared and dual positive/negative top matches
- shared and dual margins

Variants:

- `prototype_sharpness_evidence`: sharpness-aware evidence gate without the
  risk branch
- `prototype_sharpness_evidence_hybrid`: sharpness-aware evidence gate plus the
  tiny margin/regime risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

This family is closed.

- `prototype_sharpness_evidence` found one weak held-out positive niche, but
  only reached the weaker `90.53% -> 90.60%` hard near-tie band and aggregate
  improvement was tiny.
- `prototype_sharpness_evidence_hybrid` is dead on the target slice. It
  selected some non-target controls but recovered `0%` of held-out
  `stable_positive_v2`.

### `prototype_sharpness_evidence`

This variant is closed.

Best point (`0.50%` nominal budget):

- overall coverage: `0.25%`
- held-out `stable_positive_v2` recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0010`
- overall target match: `96.51% -> 96.52%`
- overall mean delta regret: `-0.0002`

At larger budgets it did not find any more held-out positives:

- held-out `stable_positive_v2` recovery stayed at `25%`
- hard near-tie stayed capped at the weaker `90.53% -> 90.60%` band
- overall mean delta regret stayed effectively flat at about `-0.0005`

Large-gap controls stayed clean, but the useful signal was too small.

### `prototype_sharpness_evidence_hybrid`

This variant is dead.

At every budget:

- held-out `stable_positive_v2` recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `0.0000`

So the risk-branch version fully collapsed to baseline behavior on the target
slice.

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

`prototype_sharpness_evidence @ 0.50%`

- overall coverage: `0.25%`
- held-out `stable_positive_v2` recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0010`
- overall mean delta regret: `-0.0002`

`prototype_sharpness_evidence_hybrid @ 2.0%`

- overall coverage: `1.00%`
- held-out `stable_positive_v2` recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `0.0000`

So the sharpness-aware gate does not improve either the ultra-low-coverage or
the matched-band frontier.

## Interpretation

This is another clean negative result.

Current read:

- branch sharpness is not the missing signal in the live evidence-agreement
  family
- top-match evidence already appears to capture almost all of the useful
  local structure that simple sharpness summaries can expose
- adding prototype sharpness does not improve sparse-positive selection, and
  the risk-branch version suppresses the signal completely

This is not a useful architecture direction for the current source family.

## Decision

Keep:

- `prototype_hybrid` as the best ultra-low-coverage architecture lead
- `prototype_agree_mix_hybrid` as the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` as the best aggregate-quality matched-band
  follow-up

Close:

- `prototype_sharpness_evidence`
- `prototype_sharpness_evidence_hybrid`

## Artifacts

- `scripts/run_prototype_sharpness_evidence_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_sharpness_evidence_defer.json`
- `reports/plots/prototype_sharpness_evidence_defer_summary.csv`
- `reports/plots/prototype_sharpness_evidence_defer_decisions.csv`
- `reports/plots/prototype_sharpness_evidence_defer_summary.png`
