# Round 12 Committee Defer

## Setup

Round twelve’s most promising new branch used the richer teacher bank directly.

Families:

- `committee_only`
- `margin_committee`

The score uses:

- agreement among safe teachers
- positive-gain filtering
- optional margin/high-headroom tie-breaking

This is an offline bank-backed defer system, not a standalone deployable
learned gate.

## All-seed upper-bound result

On the full audited bank, committee defer looks strong.

`committee_only`:

- at `0.25%` nominal budget:
  - overall coverage: `0.2515%`
  - stable-positive-v2 recovery: `34.8%`
  - hard near-tie mean delta regret: `-0.0267`
  - overall mean delta regret: `-0.0114`
- at `0.50%` nominal budget:
  - overall coverage: `0.5030%`
  - stable-positive-v2 recovery: `65.2%`
  - hard near-tie mean delta regret: `-0.0347`
  - overall mean delta regret: `-0.0159`

That is the strongest raw round-twelve correction signal.

## Held-out deployment-style result

Once recomputed on held-out seeds only, the branch is still positive, but much
smaller.

Best held-out operating point is effectively:

- `committee_only @ 0.50%` or `margin_committee @ 0.75%`

Held-out metrics:

- overall coverage: about `0.24%` to `0.25%`
- stable-positive-v2 recovery: `50%`
- stable-positive-v2 precision: `100%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0029`
- overall mean delta regret: about `-0.0089`
- large-gap control stays safe

So the committee family wins on **precision**, not on total recovered value.

## Comparison to the round-eleven reference

Round-eleven `margin_regime @ 2%` still gives the stronger held-out aggregate
deployment result:

- stable-positive recovery: `75%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0151`

The committee family is cheaper and more precise:

- overall coverage only about `0.25%`
- perfect stable-positive precision on held-out positives

But it is weaker on the full hard near-tie frontier, and it depends on
per-state teacher-bank annotations that are not available to a real deployment
policy.

## Decision

Keep committee defer only as an **offline upper-bound reference**.

What it proves:

- the richer bank can support a cleaner high-precision correction rule than the
  broad learned gates
- the positive family is real enough to exploit if the bank is available at
  decision time

What it does not prove:

- that a deployable online policy now exists
- that round twelve beat the round-eleven reference operating point

## Artifacts

- `reports/plots/round12_committee_defer_summary.csv`
- `reports/plots/round12_committee_defer_decisions.csv`
- `reports/plots/round12_committee_defer_summary.png`
- `reports/plots/round12_deployment_study_summary.csv`
