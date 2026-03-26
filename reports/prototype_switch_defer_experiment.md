# Switch-Gated Prototype Ensemble Follow-up

## Setup

This follow-up tested whether the current prototype family could keep the
ultra-low-coverage strength of `prototype_hybrid` while selectively switching
into the broader agreement-mixture regime only when it helped.

The new switch head builds two candidate branches:

- a sharp shared-projection branch, like the working `prototype_hybrid`
- an agreement-mixture branch, like `prototype_agree_mix_hybrid`

It then learns a second tiny per-state gate over the two branch scores so the
model can choose which finished branch to trust.

Variants:

- `prototype_switch`: branch switch without the risk branches
- `prototype_switch_hybrid`: branch switch with separate tiny risk branches for
  the shared and agreement paths

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

Both variants are closed.

### `prototype_switch`

This variant is dead.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- never moved the hard near-tie slice off baseline
- selected small amounts of broad-safe controls with no measurable regret gain

Best point (`2.0%` nominal budget):

- overall coverage: `1.00%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `0.0000`

### `prototype_switch_hybrid`

This variant collapsed fully to baseline.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- selected `0%` coverage at every budget
- left all Tier-1 and Tier-2 metrics unchanged

Best point (`2.0%` nominal budget):

- overall coverage: `0.00%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `0.0000`

## Comparison against current leads

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

Live `prototype_agree_mix_hybrid @ 1.50%`

- overall coverage: `1.05%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0137`

`prototype_switch` and `prototype_switch_hybrid`

- never recovered any held-out stable-positive-v2 cases
- never moved the real hard near-tie frontier
- therefore never approached either live lead

## Interpretation

This makes the current architecture picture sharper:

- a learned switch between the two current good branches is not enough
- the sparse-positive family is not just a branch routing problem
- the useful signal seems to depend on how the agreement geometry itself is
  formed, not merely on selecting between two finished branch scores

The failure modes are different:

- plain switch selected inert, non-Tier-1 cases
- hybrid switch over-regularized into complete abstention

So the current prototype family still supports:

- `prototype_hybrid` for ultra-low coverage
- `prototype_agree_mix_hybrid` for the broader matched band

But not a simple per-state switch between them.

## Decision

Keep:

- `prototype_hybrid` as the best ultra-low-coverage architecture lead
- `prototype_agree_mix_hybrid` as the best coverage-efficient matched-band
  follow-up

Close:

- `prototype_switch`
- `prototype_switch_hybrid`

## Artifacts

- `scripts/run_prototype_switch_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_switch_defer_summary.csv`
- `reports/plots/prototype_switch_defer_decisions.csv`
- `reports/plots/prototype_switch_defer_summary.png`
