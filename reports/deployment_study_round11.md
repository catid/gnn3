# Round 11 Deployment Study

## Scope

Round eleven opened deployment study only because one branch survived the held-out
gating sweep:

- `margin_regime` defer-to-teacher

This study is deliberately narrow. It is not a new online search system. It is
an offline deployment estimate for:

- baseline `multiheavy` everywhere
- defer to the audited `compute5` teacher only on gate-positive states

The cost proxy assumes:

- baseline outer steps: `3`
- deferred teacher outer steps: `5`

So average outer steps are estimated as:

- `3 + 2 * coverage`

## Main result

The deployment study is positive enough to remember, but not positive enough to
promote.

Best budgets:

At `1%` nominal budget:

- stable-positive recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0134`
- average outer steps: `3.0203`
- compute multiplier: `1.0068x`

At `2%` nominal budget:

- stable-positive recovery: `75%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0151`
- average outer steps: `3.0401`
- compute multiplier: `1.0134x`

At `5%` nominal budget the trade breaks:

- stable-positive recovery does not improve beyond `75%`
- hard near-tie mean delta regret flips to `+0.0116`
- hard near-tie false-positive defer rises to about `1.01%`
- average outer steps rise to `3.1002`

So the only credible deployment band is tiny coverage, roughly `1–2%`.

## Why it still does not promote

Three things keep this from becoming the new default:

1. The source family is too small.
   - held-out stable-positive pack size is only `4` states total
2. The gain on the full Tier-2 frontier is real but tiny.
   - hard near-tie target-match gain is only `+0.13` to `+0.20` points
3. The policy is still just a calibrated defer wrapper around the old teacher.
   - it is not a robust learned correction policy

So this is a valid deployment reference point, but not a contender.

## Decision

Deployment study is closed without contender promotion.

What survives:

- a tiny low-coverage defer-to-teacher operating point exists
- it can improve the stable-positive pack and slightly help the full near-tie
  frontier at almost no compute cost

Why it still stays closed:

- the stable-positive pack is too sparse and unstable
- the quality gain is too small to justify changing the default deployment path

## Artifacts

- `reports/plots/round11_deployment_study_summary.csv`
- `reports/plots/round11_deployment_study.json`
- `reports/plots/round11_deployment_study_summary.png`
