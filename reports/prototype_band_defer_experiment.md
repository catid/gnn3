# Prototype Bandpass Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` lead was losing value
because the residual risk branch could act too broadly instead of only in the
ambiguous prototype-score band.

The new head keeps the same local positive/negative prototype score, but gates
the residual risk branch with a learned **bandpass gate** over the prototype
score itself:

- outside the band, the head mostly trusts the raw prototype score
- near the ambiguous band, the residual branch is allowed to adjust the score

Variants:

- `prototype_band_hybrid`: wider, softer ambiguity band
- `prototype_band_sharp_hybrid`: narrower, sharper ambiguity band

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

The family is closed.

### `prototype_band_hybrid`

This was the better Tier-1 member, but it still missed the live lead.

Best working point (`1.0%` nominal budget):

- overall coverage: `0.70%`
- stable-positive-v2 recovery: `50%`
- stable-positive-v2 precision: `75%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0029`
- overall target match: `96.51% -> 96.56%`
- overall mean delta regret: `-0.0014`

Larger budgets did not improve the actual frontier:

- stable-positive-v2 stayed capped at `50%`
- hard near-tie stayed capped at `90.53% -> 90.66%`
- overall regret gains stayed tiny

### `prototype_band_sharp_hybrid`

This variant was even more conservative and weaker.

Best working point (`1.0%` nominal budget):

- overall coverage: `0.51%`
- stable-positive-v2 recovery: `25%`
- stable-positive-v2 precision: `75%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0019`
- overall mean delta regret: `-0.0007`

At `2.0%` it also started to introduce mild harmful selection in the overall
panel, which is the wrong failure mode for such a narrow correction family.

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Best bandpass point (`prototype_band_hybrid @ 1.0%`)

- overall coverage: `0.70%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0029`
- overall mean delta regret: `-0.0014`

So restricting residual action to the ambiguous score band does not recover the
missing held-out positives. It mainly weakens the already-working hybrid.

## Interpretation

The live prototype family does not appear bottlenecked by residual spill outside
the ambiguous score region.

Current read:

- the additive `prototype_hybrid` residual is already acting in a good enough
  place
- explicitly band-limiting that residual throws away too much of the useful
  sparse-positive correction signal
- the remaining gap is not simply “residuals should only act near score zero”

This closes another plausible architecture path:

- the next gain is unlikely to come from a score-band gate on the existing
  residual branch
- the live `prototype_hybrid` remains the only architecture lead worth keeping
  open in this family

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `prototype_band_hybrid`
- `prototype_band_sharp_hybrid`

Do not reopen score-band prototype defer in this form.

## Artifacts

- `scripts/run_prototype_band_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_band_defer_summary.csv`
- `reports/plots/prototype_band_defer_decisions.csv`
- `reports/plots/prototype_band_defer_summary.png`
