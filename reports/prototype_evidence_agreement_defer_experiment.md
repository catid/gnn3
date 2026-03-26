# Evidence-Calibrated Agreement Mixture Follow-up

## Setup

This follow-up tested whether the live agreement-mixture follow-up was still
limited by using only the final shared and dual branch scores in its gate.

The new head keeps the same two geometry branches as
`prototype_agree_mix_hybrid`:

- shared-projection prototype branch
- dual-projection prototype branch

But the blend gate now also sees branch-internal prototype evidence:

- shared positive top match
- shared negative top match
- dual positive top match
- dual negative top match
- shared and dual branch margins

So the gate can distinguish:

- score agreement with weak evidence
- score agreement with strong positive support
- score disagreement caused by asymmetric positive or negative evidence

Variants:

- `prototype_evidence_agree`: evidence-calibrated mixture without the risk branch
- `prototype_evidence_agree_hybrid`: evidence-calibrated mixture plus the tiny
  margin/regime risk branch

Train/eval split stayed fixed:

- train: `round11_feature_cache_seed314`
- held-out eval: `round11_feature_cache_seed315`, `round11_feature_cache_seed316`
- labels: `stable_positive_v2_case` from
  `reports/plots/round12_teacher_bank_decisions.csv`

Budgets:

- `0.10%`, `0.25%`, `0.50%`, `0.75%`, `1.00%`, `1.50%`, `2.00%`

## Main result

`prototype_evidence_agree` is closed. `prototype_evidence_agree_hybrid` is
alive.

### `prototype_evidence_agree`

This variant is closed.

- recovered `0%` of held-out `stable_positive_v2`
- never moved the hard near-tie slice off baseline
- only selected inert controls and broad-safe non-Tier-1 states

Best point (`2.0%` nominal budget):

- overall coverage: `1.01%`
- stable-positive-v2 recovery: `0%`
- hard near-tie target match: unchanged at `90.53%`
- hard near-tie mean delta regret: `0.0000`
- overall mean delta regret: `-0.0027`

### `prototype_evidence_agree_hybrid`

This variant is a real positive follow-up.

At `1.0%` nominal budget:

- overall coverage: `1.01%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0042`
- overall target match: `96.51% -> 96.74%`
- overall mean delta regret: `-0.0126`

At `1.5%` nominal budget:

- overall coverage: `1.52%`
- stable-positive-v2 recovery: `75%`
- stable-positive-v2 precision: `100%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall target match: `96.51% -> 96.78%`
- overall mean delta regret: `-0.0140`

At `2.0%` nominal budget:

- overall coverage: `2.00%`
- stable-positive-v2 recovery: `75%`
- stable-positive-v2 precision: `100%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall target match: `96.51% -> 96.80%`
- overall mean delta regret: `-0.0148`

Large-gap controls stayed clean at those matched-band points:

- large-gap target match: `99.79% -> 99.90%`
- large-gap mean delta regret: `-0.0065`

## Comparison against current leads

Live `prototype_hybrid @ 0.75%`

- overall coverage: `0.76%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0097`

`prototype_agree_mix_hybrid @ 1.5%`

- overall coverage: `1.05%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0137`

`prototype_evidence_agree_hybrid @ 1.5%`

- overall coverage: `1.52%`
- stable-positive-v2 recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0140`

So it does not replace `prototype_hybrid` as the ultra-low-coverage leader, and
it is not the most coverage-efficient matched-band follow-up.

But it does become the best **aggregate-quality** matched-band follow-up so far:

`prototype_mixture_hybrid @ 2.0%`

- overall coverage: `1.84%`
- overall mean delta regret: `-0.0138`

`prototype_agree_mix_hybrid @ 1.5%`

- overall coverage: `1.05%`
- overall mean delta regret: `-0.0137`

`prototype_evidence_agree_hybrid @ 2.0%`

- overall coverage: `2.00%`
- overall mean delta regret: `-0.0148`

So the richer evidence-calibrated gate improves aggregate regret a bit more than
the earlier matched-band heads, but it pays for that with higher coverage.

## Interpretation

This is another real positive result inside the prototype family.

Current read:

- the score-only agreement gate was not extracting all of the useful signal
- branch-internal positive and negative prototype evidence helps the gate make
  slightly better matched-band decisions
- that gain appears in aggregate regret, not in a cleaner ultra-low-coverage
  Tier-1 frontier

What it still does **not** show:

- a better ultra-low-coverage leader than `prototype_hybrid`
- a more coverage-efficient matched-band leader than `prototype_agree_mix_hybrid`

So the evidence-calibrated head is best understood as:

- a quality-oriented matched-band follow-up
- not a frontier simplification

## Decision

Keep:

- `prototype_hybrid` as the best ultra-low-coverage architecture lead
- `prototype_agree_mix_hybrid` as the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` as the best aggregate-quality matched-band
  follow-up

Close:

- `prototype_evidence_agree`

## Artifacts

- `scripts/run_prototype_evidence_agreement_defer.py`
- `src/gnn3/models/prototype_defer.py`
- `tests/test_prototype_defer.py`
- `reports/plots/prototype_evidence_agreement_defer_summary.csv`
- `reports/plots/prototype_evidence_agreement_defer_decisions.csv`
- `reports/plots/prototype_evidence_agreement_defer_summary.png`
