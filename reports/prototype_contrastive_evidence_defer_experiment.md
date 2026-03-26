# Contrastive Evidence Agreement Follow-up

## Setup

This follow-up tested whether the evidence-aware agreement family was using the
wrong gate inputs.

Instead of feeding the gate raw shared and dual evidence values, the new head
uses explicit contrastive features:

- shared score
- dual score
- score delta and absolute score delta
- positive evidence delta
- negative-evidence cleanliness delta
- margin delta
- combined evidence-balance delta
- shared margin
- dual margin

So the gate sees how the two branches differ, not just their raw values.

Variants:

- `prototype_contrastive_evidence`: contrastive evidence gate without the risk
  branch
- `prototype_contrastive_evidence_hybrid`: contrastive evidence gate plus the
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

- `prototype_contrastive_evidence` is dead.
- `prototype_contrastive_evidence_hybrid` is a weak matched-band branch, but it
  still does not clear the current prototype-family bar.

### `prototype_contrastive_evidence`

This variant is dead.

At every budget:

- held-out `stable_positive_v2` recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`

At `2.0%` nominal budget it started to regress aggregate quality:

- overall coverage: `1.92%`
- overall target match: `96.51% -> 96.50%`
- overall mean delta regret: `+0.0010`

### `prototype_contrastive_evidence_hybrid`

This variant is still closed.

It found a small real signal, but only at larger budgets and still below the
live frontier.

At `1.5%` nominal budget:

- overall coverage: `1.52%`
- held-out `stable_positive_v2` recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0048`
- overall mean delta regret: `-0.0067`

At `2.0%` nominal budget:

- overall coverage: `1.91%`
- held-out `stable_positive_v2` recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall target match: `96.51% -> 96.64%`
- overall mean delta regret: `-0.0081`

Large-gap controls stayed clean at that best point:

- large-gap target match: `99.79% -> 99.84%`
- large-gap mean delta regret: `-0.0036`

So the contrastive gate does learn something useful, but it still only reaches
the weaker `90.53% -> 90.66%` hard-slice band, and it needs much higher
coverage than the live ultra-low-coverage lead to do it.

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

`prototype_contrastive_evidence_hybrid @ 2.0%`

- overall coverage: `1.91%`
- held-out `stable_positive_v2` recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0081`

So it improves on the dead positive-lift branch in Tier-1 recovery, but it
still loses clearly to every live architecture lead.

## Interpretation

This is another useful negative result.

The explicit delta features do surface some real signal:

- the hybrid recovered `50%` of held-out `stable_positive_v2`
- it moved hard near-tie into the weaker positive band
- it preserved large-gap controls

But they still do not isolate the true frontier well enough:

- coverage remains high
- false-positive selection is still broad
- aggregate quality is clearly below the live matched-band leaders

Current read:

- raw-vs-delta gating is not the core missing ingredient
- the current live agreement/evidence heads already capture most of what simple
  contrastive deltas can express
- the remaining gap still looks like precision calibration and subset
  selection, not a missing first-order evidence-difference feature

## Decision

Keep:

- `prototype_hybrid` as the best ultra-low-coverage architecture lead
- `prototype_agree_mix_hybrid` as the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` as the best aggregate-quality matched-band
  follow-up

Close:

- `prototype_contrastive_evidence`
- `prototype_contrastive_evidence_hybrid`

## Artifacts

- `scripts/run_prototype_contrastive_evidence_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_contrastive_evidence_defer.json`
- `reports/plots/prototype_contrastive_evidence_defer_summary.csv`
- `reports/plots/prototype_contrastive_evidence_defer_decisions.csv`
- `reports/plots/prototype_contrastive_evidence_defer_summary.png`
