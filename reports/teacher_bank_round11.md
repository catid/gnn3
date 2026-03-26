# Round 11 Teacher Bank Audit

## Scope

Round eleven opened with an offline teacher-bank audit over the committed
round-ten helpfulness decision artifacts:

- `reports/plots/round10_helpfulness_seed314_decisions.csv`
- `reports/plots/round10_helpfulness_seed315_decisions.csv`
- `reports/plots/round10_helpfulness_seed316_decisions.csv`

The operative teacher in this first bank is the audited fixed-compute
`compute5` branch from rounds nine and ten. Baseline is plain `multiheavy`.

The question was not whether `compute5` helps on average. The question was:

> can we isolate a narrow subset of hard near-tie states where the teacher is
> stably positive enough to justify a precision-first defer/correct policy?

## Aggregate bank result

Across the three audited seeds:

- total audited decisions: `13121`
- hard near-tie decisions: `2173`
- stable-positive pack decisions: `46`
- unstable-positive decisions: `3`
- harmful teacher decisions on hard near-tie: `70`

So the correction opportunity exists, but it is tiny:

- stable-positive pack share of hard near-tie: `2.12%`
- harmful share of hard near-tie: `3.22%`

When the stable-positive pack does fire, it is strong:

- stable-positive target match: `0.0% -> 100.0%`
- mean teacher regret gain: `2.7310`
- p95 teacher regret gain floor: about `0.6505`
- mean miss gain: `0.0435`

This is a real teacher signal, not noise. But it is much narrower than the
full hard near-tie frontier.

## Where the bank is positive

The positive source families are exactly the ones expected from round ten.

High-headroom near-tie:

- decisions: `123`
- helpful rate: `28.46%`
- harmful rate: `0.00%`
- stable-positive rate: `28.46%`
- mean delta regret: `-0.9506`

Baseline-error near-tie:

- decisions: `244`
- helpful rate: `20.08%`
- harmful rate: `0.00%`
- stable-positive rate: `18.85%`
- mean delta regret: `-0.5185`

Broad hard near-tie remains net negative even after the stable-positive relabel:

- helpful rate: `2.25%`
- harmful rate: `3.22%`
- mean delta regret: `+0.1751`

So round eleven should not try to learn a broad near-tie correction policy.
It must target the narrow stable-positive corner only.

## Stability across seeds

The main new finding is how concentrated the stable-positive pack is:

- seed314 stable-positive decisions: `42`
- seed315 stable-positive decisions: `1`
- seed316 stable-positive decisions: `3`

Share of hard near-tie by seed:

- seed314: `6.14%`
- seed315: `0.12%`
- seed316: `0.46%`

That is far more brittle than the round-ten aggregate helpfulness summary made
it look.

The source-signature overlap is also effectively zero:

- seed314 vs seed315 signature Jaccard: `0.0`
- seed314 vs seed316 signature Jaccard: `0.0`
- seed315 vs seed316 signature Jaccard: `0.0`

This means the stable-positive source family is not transferring cleanly at the
current signature granularity. Any gate or student that succeeds will need to
be extremely conservative.

## Decision

Wave A stays barely positive, but only under a very narrow interpretation:

- there is a real stable-positive correction pack
- it is nontrivial enough to test precision-first correction work
- but it is sparse and unstable enough that broad correction is no longer a
  credible thesis

Operational consequence:

- continue with defer gates, comparator, and subset-only distillation
- treat the seed314-heavy stable-positive pack as a stress test, not as proof
  of easy deployability
- require unusually strong held-out precision before opening deployment study

## Artifacts

- `reports/plots/round11_teacher_bank_decisions.csv`
- `reports/plots/round11_teacher_bank_summary.csv`
- `reports/plots/round11_teacher_bank_seed_summary.csv`
- `reports/plots/round11_teacher_bank_seed_overlap.csv`
- `reports/plots/round11_teacher_bank_suite_overlap.csv`
- `reports/plots/round11_teacher_bank_stable_positive_manifest.csv`
