# Prototype-Triage Defer Follow-up

## Setup

This follow-up tested whether the remaining false-positive spill in
`prototype_hybrid` came from a missing **neutral memory** rather than a missing
positive-memory geometry.

Train/eval split:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` and `harmful_teacher_bank_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Variants:

- `triage_memory`: explicit positive / neutral / harmful prototype banks
- `triage_hybrid`: the same triage bank plus the small risk branch used by
  `prototype_hybrid`

The defer score was:

- positive logit minus `logsumexp(neutral, harmful)`

So this branch should only help if explicit neutral suppression matters more
than the broader positive-memory ranking used by `prototype_hybrid`.

## Main result

The family is closed.

### `triage_memory`

- never recovered a held-out stable-positive-v2 case
- stayed effectively identical to baseline on hard near-tie

### `triage_hybrid`

Best operating point was around `1–2%` overall coverage:

- stable-positive-v2 recovery: only `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: only `-0.0023`
- overall mean delta regret: about `-0.0108`

That is much weaker than `prototype_hybrid`.

## Comparison against the live prototype lead

`prototype_hybrid @ 0.75%`

- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

`triage_hybrid @ 1.0%`

- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0023`
- overall mean delta regret: `-0.0097`

So the triage bank preserved some overall caution, but it gave away too much of
the Tier-1 signal.

## Interpretation

The sparse positive family does **not** appear to be missing an explicit
neutral/harmful memory bank.

The stronger explanation is:

- the positive family is already locally recoverable by the simpler
  `prototype_hybrid`
- adding neutral/harmful memories makes the head overly conservative
- the open problem is still ranking the positive family, not modeling a richer
  abstain taxonomy

## Decision

Keep:

- `prototype_hybrid` as the only live prototype-memory architecture lead

Close:

- `triage_memory`
- `triage_hybrid`

Do not reopen explicit positive / neutral / harmful prototype triage in this
form.

## Artifacts

- `scripts/run_prototype_triage_defer.py`
- `reports/plots/prototype_triage_defer_summary.csv`
- `reports/plots/prototype_triage_defer_decisions.csv`
- `reports/plots/prototype_triage_defer_summary.png`
