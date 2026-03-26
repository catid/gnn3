# Candidate-Aware Prototype Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` branch was still
missing **top-2 action-pair structure** rather than teacher-bank structure.

Instead of feeding the prototype bank only the decision-level frozen features,
this branch replaced the input geometry with `candidate_pair_features(...)`,
which adds:

- top-1 candidate features
- top-2 candidate features
- top-1 minus top-2 deltas
- top-1 / top-2 score, cost-to-go, and on-time signals

Train/eval split stayed the same as the round-twelve and prototype-memory
follow-ups:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`

Variants:

- `candidate_memory`
- `candidate_hybrid`

## Main result

The family is closed.

Neither variant recovered the sparse positive family in a useful way.

### `candidate_memory`

- recovered `0%` of held-out stable-positive-v2 until `2%` coverage
- even at `2%`, recovered only `25%`
- hard near-tie target match only reached `90.53% -> 90.60%`
- hard near-tie mean delta regret only reached `-0.0023`

### `candidate_hybrid`

- recovered `0%` of held-out stable-positive-v2 at every budget
- improved overall regret slightly through conservative selection
- never moved the held-out hard near-tie frontier at all

So the added top-2 pair geometry did **not** help the prototype family find the
true stable-positive states.

## Comparison against the live lead

`prototype_hybrid @ 0.75%`

- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`

Best candidate-aware point (`candidate_memory @ 2%`)

- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0023`

So the candidate-aware branch is clearly weaker than the existing
`prototype_hybrid` lead.

## Interpretation

This negative result narrows the remaining architecture hypothesis further.

The live signal does **not** appear to come from missing explicit top-2
candidate-pair structure. The simpler prototype-memory geometry was already
enough to recover the sparse positive family.

So the current best hypothesis remains:

- the problem is primarily about calibrating a tiny positive family
- not about adding a richer candidate-pair encoding

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `candidate_memory`
- `candidate_hybrid`

Do not reopen candidate-aware prototype defer in this form.

## Artifacts

- `scripts/run_candidate_prototype_defer.py`
- `reports/plots/candidate_prototype_defer_summary.csv`
- `reports/plots/candidate_prototype_defer_decisions.csv`
- `reports/plots/candidate_prototype_defer_summary.png`
