# Prototype Interaction Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` lead was bottlenecked
because its residual branch could not see the current prototype evidence state.

The new head keeps the same positive-vs-negative prototype score, but augments
the residual branch with a tiny set of prototype-score summary scalars:

- pooled prototype score
- top positive prototype similarity
- top negative prototype similarity
- top-gap between the two

Variants:

- `prototype_interaction_hybrid`: additive residual conditioned on
  `risk_features + prototype_summary`
- `prototype_interaction_gated`: the same residual, but with an extra learned
  gate over the prototype summary itself

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

The family is closed.

### `prototype_interaction_hybrid`

This variant improved broad overall regret, but it still missed the actual
frontier.

Best overall point (`2.0%` nominal budget):

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `25%`
- stable-positive-v2 precision: `75%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0023`
- overall target match: `96.51% -> 96.76%`
- overall mean delta regret: `-0.0122`

That looks decent as a broad risk-correction policy, but it is still far too
weak on the real sparse-positive target:

- stable-positive-v2 stayed capped at `25%`
- hard near-tie stayed capped at `90.53% -> 90.60%`

### `prototype_interaction_gated`

This variant is effectively dead.

- recovered `0%` of held-out `stable_positive_v2`
- never improved hard near-tie above baseline
- only found a tiny conservative overall-risk niche

So explicitly gating the interaction residual suppressed the signal too much.

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Best interaction point (`prototype_interaction_hybrid @ 2.0%`)

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0023`
- overall mean delta regret: `-0.0122`

So the interaction residual helps broad overall caution, but it weakens the
rare positive retrieval that actually matters on Tier-1 and hard near-tie.

## Interpretation

The live prototype family does not appear bottlenecked by the residual branch
being unaware of the current prototype score state.

Current read:

- the extra prototype-summary context lets the residual become a better broad
  calibrator
- but it still does not recover the rare held-out positive family
- the real missing gain is not just a better interaction between the base score
  and the existing risk features

This closes another plausible architecture path:

- the next gain is unlikely to come from a richer score-aware residual layered
  on top of the same prototype bank
- the live `prototype_hybrid` remains the only architecture lead worth keeping
  open in this family

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `prototype_interaction_hybrid`
- `prototype_interaction_gated`

Do not reopen score-aware prototype interaction residuals in this form.

## Artifacts

- `scripts/run_prototype_interaction_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_interaction_defer_summary.csv`
- `reports/plots/prototype_interaction_defer_decisions.csv`
- `reports/plots/prototype_interaction_defer_summary.png`
