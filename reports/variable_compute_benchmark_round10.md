# Round 10 Variable-Compute Benchmark Refresh

## Purpose

Round 9 established that fixed extra compute has some headroom but no robust promotion.
Round 10 refreshed the compute frontier with the new helpfulness audit and the held-out selective-compute tests.

## Baseline frontier

The relevant points remain:

- plain `multiheavy`: default `3` outer steps
- fixed `compute5`: fixed `5` outer steps, or about `1.67x` the base outer-step budget
- round-10 selective-compute policies: intended to spend extra compute only on predicted-helpful states

## Refresh result

The refreshed frontier still favors plain `multiheavy`.

Fixed `compute5` is not a deployable frontier win:

- aggregate hard near-tie helpful rate: `1.92%`
- aggregate hard near-tie harmful rate: `2.42%`
- aggregate hard near-tie mean delta regret: `+0.1143`

So paying the full `5`-step cost is not justified on the accepted round-10 frontier surface.

The selective-compute policies do not change the picture:

- held-out compute multiplier: `1.0`
- held-out average outer steps: `3.0`
- hard near-tie mean delta regret: `+0.0082`
- hard near-tie target match: `91.03%` vs baseline `91.16%`

That means the gate became so conservative that it effectively refused to spend extra compute.

## Interpretation

Round 10 did not find a better regret-at-fixed-compute frontier than plain `multiheavy`.

The current benchmark picture is:

- full extra compute spends too much for too little stable gain
- learned selective compute spends almost nothing and therefore buys almost nothing
- there is still a narrow high-headroom benefit slice, but no tested policy reaches it at useful scale

## Decision

Variable compute remains an open hypothesis only in a narrower form:

- high-precision, high-headroom-only compute triggers
- or a one-pass student that faithfully preserves only the conservative part of the compute correction

Neither was achieved in round 10.

## Artifact basis

- `reports/compute_helpfulness_round10.md`
- `reports/selective_compute_round10.md`
- `reports/continuation_report_round9.md`
