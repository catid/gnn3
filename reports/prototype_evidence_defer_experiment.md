# Prototype Evidence-Readout Defer Follow-up

## Setup

This follow-up tested whether the live `prototype_hybrid` lead was losing the
rare positive signal by collapsing prototype evidence too early into a single
log-sum-exp score.

The new head keeps the same projected positive/negative prototype banks, but
changes the readout:

- compute top-k positive prototype matches
- compute top-k negative prototype matches
- expose those local evidence values directly to a tiny readout MLP

Variants:

- `prototype_evidence`: top-k prototype evidence readout only
- `prototype_evidence_hybrid`: the same evidence readout plus the existing
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

### `prototype_evidence`

This was the better member, but it still missed the live lead badly.

Best working point (`0.75%` nominal budget):

- overall coverage: `0.38%`
- stable-positive-v2 recovery: `25%`
- stable-positive-v2 precision: `75%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0010`
- overall target match: `96.51% -> 96.55%`
- overall mean delta regret: `-0.0011`

Larger budgets did not improve the real frontier:

- stable-positive-v2 stayed capped at `25%`
- hard near-tie stayed capped at `90.53% -> 90.60%`
- overall regret gains stayed negligible

### `prototype_evidence_hybrid`

This variant is dead.

- selected effectively no held-out decisions at every useful budget
- recovered `0%` of held-out `stable_positive_v2`
- never improved hard near-tie above baseline
- reduced to baseline behavior on the held-out panel

So adding the risk branch on top of the evidence readout collapsed the sparse
positive signal completely.

## Comparison against the live lead

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Best evidence point (`prototype_evidence @ 0.75%`)

- overall coverage: `0.38%`
- stable-positive-v2 recovery: `25%`
- hard near-tie target match: `90.53% -> 90.60%`
- hard near-tie mean delta regret: `-0.0010`
- overall mean delta regret: `-0.0011`

So the richer readout over local prototype evidence does not recover the sparse
held-out positives that the simpler `prototype_hybrid` already finds.

## Interpretation

The live prototype family does not appear bottlenecked by using a simple pooled
prototype score.

Current read:

- the useful signal is not in a more complex combination of the top prototype
  matches
- the small evidence MLP mostly smooths the readout and loses the real Tier-1
  positives
- adding the risk branch on top of that evidence readout suppresses the signal
  even further

This closes another plausible architecture path:

- the next gain is unlikely to come from a more expressive readout over the same
  local prototype match pattern
- the live `prototype_hybrid` remains the only architecture lead worth keeping
  open in this family

## Decision

Keep:

- `prototype_hybrid` as the only live narrow architecture lead

Close:

- `prototype_evidence`
- `prototype_evidence_hybrid`

Do not reopen prototype evidence-readout defer in this form.

## Artifacts

- `scripts/run_prototype_evidence_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_evidence_defer_summary.csv`
- `reports/plots/prototype_evidence_defer_decisions.csv`
- `reports/plots/prototype_evidence_defer_summary.png`
