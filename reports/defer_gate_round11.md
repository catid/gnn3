# Round 11 Budgeted Defer Gate

## Setup

Round eleven replaced the broad round-ten helpfulness gate with a precision-first
defer-to-teacher gate trained on the seed314 stable-positive pack and evaluated
on held-out seed315 and seed316 feature caches.

Families:

- `linear`
- `mlp`
- `margin_regime`

Coverage budgets:

- `0.25%`
- `0.5%`
- `1%`
- `2%`
- `5%`

The question was not whether a classifier can rank stable-positive states. The
question was whether a tiny-coverage defer policy can improve the combined
system on held-out seeds.

## Main result

Only the simple `margin_regime` gate survived.

The learned gates were effectively dead:

- `linear` never recovered a held-out stable-positive case
- `mlp` stayed near-zero trigger and also never recovered a held-out
  stable-positive case

So the only usable gate family is the simple ambiguity/regime baseline.

## Held-out operating points

Aggregate held-out behavior for `margin_regime`:

At `1%` budget:

- stable-positive pack recovery: `50%`
- stable-positive precision: `100%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- hard near-tie false-positive defer rate: `0.0`
- overall target match: `96.51% -> 96.75%`
- overall mean delta regret: `-0.0134`

At `2%` budget:

- stable-positive pack recovery: `75%`
- stable-positive precision: `100%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- hard near-tie correction rate: `0.20%`
- overall mean delta regret: `-0.0151`

At `5%` budget the gate stopped being precision-first:

- stable-positive recovery stayed at `75%`
- hard near-tie false-positive defer rose to about `1.01%`
- hard near-tie mean delta regret flipped positive to `+0.0116`
- hard near-tie target match returned to baseline

So the live operating band is narrow: roughly `1–2%` coverage, not broader.

## Seed behavior

The held-out seeds were uneven in exactly the way the teacher-bank audit
predicted.

Seed315:

- only one stable-positive case existed
- `margin_regime` found it at every budget from `0.25%` onward
- hard near-tie gain was small but clean:
  - target match `91.59% -> 91.71%`
  - mean delta regret `-0.0085`

Seed316:

- three stable-positive cases existed
- `margin_regime` recovered:
  - `1 / 3` at `1%`
  - `2 / 3` at `2%`
- hard near-tie gain was again small:
  - target match `89.19% -> 89.50%` at `2%`
  - mean delta regret `-0.0095`

This is a real held-out signal, but it is still tiny and depends on a very
small source family.

## Decision

Direct defer-to-teacher is not promoted as a new default policy.

What survives:

- `margin_regime` is the only round-eleven branch with a real held-out
  operating point
- the useful band is tiny-coverage deferral only
- the safe region appears to be around `1–2%` coverage

Why it still does not promote:

- the stable-positive source family is only `4` held-out cases total
- the gate is carried almost entirely by simple margin/regime ranking, not a
  richer learned signal
- the broad hard near-tie gain is real but too small to justify policy
  promotion yet

So the defer gate stays as a reference deployment study branch, not a
contender.

## Artifacts

- `reports/plots/round11_defer_gate_heldout_summary.csv`
- `reports/plots/round11_defer_gate_heldout.json`
- `reports/plots/round11_defer_gate_heldout_summary.png`
