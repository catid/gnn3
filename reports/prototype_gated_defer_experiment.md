# Prototype Gated-Hybrid Defer Follow-up

## Setup

This follow-up tested a more structural variant of the live
`prototype_hybrid` architecture:

- keep the same prototype-memory geometry on `decision_augmented_features(...)`
- change only how risk interacts with the prototype score

Instead of adding a small risk residual on top of the prototype logit, the new
head lets risk **gate** the prototype evidence directly.

Variants:

- `prototype_gated_scale`: risk learns only a multiplicative gate on the
  prototype score
- `prototype_gated_offset`: risk learns a multiplicative gate plus a small
  residual bias

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

The family is closed.

### `prototype_gated_scale`

This variant is dead.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- never improved hard near-tie target match above baseline
- only deferred on false positives and large-gap controls

So pure multiplicative suppression is too aggressive for the sparse positive
family.

### `prototype_gated_offset`

This variant is alive only in a weak sense and does **not** beat the current
lead.

Best working point (`0.50%` nominal budget):

- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall target match: `96.51% -> 96.61%`
- overall mean delta regret: `-0.0072`

Larger budgets improved aggregate overall regret a bit, but they did not
improve the actual Tier-1 outcome:

- stable-positive-v2 stayed capped at `50%`
- hard near-tie stayed capped at `90.53% -> 90.66%`

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Best gated point (`prototype_gated_offset @ 0.50%`)

- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0072`

So the gate helped suppress some spill, but it also suppressed too much of the
actual positive correction signal.

## Interpretation

The live architecture seems to need a **soft additive risk correction**, not a
risk-conditioned rescaling of prototype evidence.

Current read:

- local prototype geometry still carries the sparse positive signal
- a learned multiplicative gate suppresses that signal too aggressively
- even adding a residual bias back does not recover the missing Tier-1 recall

This closes another plausible architecture path:

- the problem is not just “risk should turn prototype evidence up or down”
- the working interaction is still the simpler additive hybrid in
  `prototype_hybrid`

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `prototype_gated_scale`
- `prototype_gated_offset`

Do not reopen gated prototype defer in this form.

## Artifacts

- `scripts/run_prototype_gated_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_gated_defer_summary.csv`
- `reports/plots/prototype_gated_defer_decisions.csv`
- `reports/plots/prototype_gated_defer_summary.png`
