# Round 12 Deployment Study

## Setup

The round-twelve deployment panel compared:

- plain `multiheavy`
- the round-eleven `margin_regime` reference
- round-twelve ultra-low `margin_regime`
- round-twelve committee defer variants

Held-out seeds:

- `315`
- `316`

Baseline outer steps were treated as `3`, deferred teacher path as `5`.

## Main result

No round-twelve branch beat the round-eleven reference once the comparison was
made on held-out seeds only.

### Round-eleven reference remains strongest

`round11 margin_regime @ 2%`

- overall coverage: `2.00%`
- stable-positive recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0151`
- average outer steps: `3.0401`

### Best round-twelve learned gate only matches the hard-slice band

`round12 margin_regime @ 0.50%`

- overall coverage: `0.51%`
- stable-positive recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0075`
- average outer steps: `3.0101`

So it is cheaper, but not stronger.

### Best round-twelve committee upper bound is cleaner but smaller

`round12 margin_committee @ 0.75%`

- overall coverage: `0.25%`
- stable-positive recovery: `50%`
- stable-positive precision: `100%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0029`
- overall mean delta regret: `-0.0089`
- average outer steps: `3.0051`

This is the cleanest low-coverage operating point, but not the strongest
frontier win, and it depends on offline per-state teacher-bank annotations.

## Interpretation

Round twelve improved the **upper-bound precision story**, not the deployment
story.

The committee branch shows:

- richer bank information can define a cleaner correction subset
- a tiny, very precise defer rule exists if that bank is already known

But it does not show:

- that the branch is deployable online
- or that it beats the round-eleven reference once system-level held-out
  metrics are compared fairly

## Decision

No contender is promoted.

Keep:

- plain `multiheavy` as the default
- round-eleven `margin_regime` as the only remembered deployable-like reference
- round-twelve committee defer only as an offline upper bound

## Artifacts

- `reports/plots/round12_deployment_study_summary.csv`
- `reports/plots/round12_deployment_study_panel.csv`
- `reports/plots/round12_deployment_study_summary.png`
