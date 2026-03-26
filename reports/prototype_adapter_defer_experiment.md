# Prototype Adapter-Memory Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` lead was bottlenecked
by using a fixed linear prototype score in encoded prototype space.

The new head keeps the same positive/negative prototype bank geometry as
`prototype_hybrid`, but adds a tiny learned adapter over the encoded
prototype-space features before the final defer logit.

Variants:

- `prototype_adapter`: prototype bank plus adapter only
- `prototype_adapter_hybrid`: prototype bank plus adapter plus the existing
  additive risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

The family is closed.

### `prototype_adapter`

This variant is dead.

- recovered `0%` of held-out `stable_positive_v2` through `1.00%` nominal budget
- only reached `25%` held-out `stable_positive_v2` recovery at `1.50%` and
  `2.00%`
- never improved hard near-tie beyond the weak `90.53% -> 90.60%` band
- overall mean delta regret stayed essentially flat at `0.0000` through `1.00%`

So the adapter alone does not unlock any missing correction signal.

### `prototype_adapter_hybrid`

This was the better adapter variant, but it still missed the live lead.

Best working point (`1.5%` nominal budget):

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `50%`
- stable-positive-v2 precision: `75%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0042`
- overall target match: `96.51% -> 96.68%`
- overall mean delta regret: `-0.0070`

`2.0%` nominal budget improved aggregate overall regret slightly more:

- overall mean delta regret: `-0.0077`

But Tier-1 behavior stayed capped:

- stable-positive-v2 recovery stayed at `50%`
- hard near-tie stayed capped at `90.53% -> 90.66%`

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Best adapter point (`prototype_adapter_hybrid @ 1.5%`)

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0042`
- overall mean delta regret: `-0.0070`

So the adapter keeps the general low-coverage shape of the live lead, but it
blunts the actual sparse-positive correction signal instead of sharpening it.

## Interpretation

The live prototype-memory lead does not appear bottlenecked by missing shallow
nonlinearity inside the encoded prototype space.

Current read:

- the useful signal is already captured by the simpler prototype-bank geometry
- adding a tiny learned adapter mostly smooths the score instead of improving
  the rare stable-positive retrieval
- the additive risk branch still matters, but the adapter does not improve how
  that branch uses the prototype evidence

This closes another plausible architecture path:

- the next gain is unlikely to come from a shallow adapter over the same
  prototype-space statistics
- the live `prototype_hybrid` remains the only architecture lead worth keeping
  open in this family

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `prototype_adapter`
- `prototype_adapter_hybrid`

Do not reopen prototype adapters in this form.

## Artifacts

- `scripts/run_prototype_adapter_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_adapter_defer_summary.csv`
- `reports/plots/prototype_adapter_defer_decisions.csv`
- `reports/plots/prototype_adapter_defer_summary.png`
