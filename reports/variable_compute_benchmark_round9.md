# Round 9 Variable-Compute Benchmark

## Goal

Round nine is testing whether extra conditional compute can improve the hard
near-tie frontier without paying a broad deployment penalty.

This benchmark separates three questions:

1. does extra fixed compute help at all?
2. is any help robust across matched seeds?
3. can adaptive compute target the hard near-tie slice more efficiently than a
   global increase in outer rounds?

## Benchmark Surfaces

The benchmark should be read on three surfaces:

- corrected feasible base suite
- hard near-tie frontier suites
  - `deeper_packets6`
  - `heavy_dynamic`
  - mixed-compute
- large-gap control slice

The primary decision surface remains the hard near-tie frontier, not broad mean
metrics on easy states.

## Fixed-Compute Baseline Results

Matched seed314 / 315 / 316 fixed-compute results settled the first question.

### Seed314

- `multiheavy`: regret `1.898`, p95 `8.140`, miss `43.75%`
- `compute5`: regret `1.302`, p95 `5.277`, miss `31.25%`

This was a real positive scout and justified continuing the compute family.

### Seed315

- `multiheavy`: regret `2.107`, p95 `6.164`, miss `56.25%`
- `compute5`: regret `2.107`, p95 `6.164`, miss `56.25%`

Exact baseline match.

### Seed316

- `multiheavy`: regret `1.508`, p95 `7.307`, miss `50.00%`
- `compute5`: regret `1.508`, p95 `7.307`, miss `50.00%`

Exact baseline match.

## Fixed-Compute Verdict

Fixed `compute5` is not a robust constructor upgrade.

The family verdict is now:

- extra fixed compute can help on individual seeds
- that help did not replicate across the matched three-seed comparison
- therefore plain fixed deeper thinking is not sufficient by itself

The next admissible question is narrower:

- can compute be applied selectively on the hard near-tie frontier?

## Deeper Fixed Compute

The first deeper fixed-compute scout (`compute7` on seed314) failed its early
gate immediately:

- epoch-1 regret `8893.68`
- epoch-1 p95 regret `24061.70`
- epoch-1 miss `100%`

So the round does not support simply increasing outer rounds globally.

## Current Adaptive-Compute Lane

The remaining benchmark question was whether compute could be used selectively
rather than globally. That lane is now closed on the audited frontier suite.

### `deeper_packets6` hard near-tie slice

Baseline context:

- `321` hard near-tie decisions
- `44` baseline hard near-tie errors
- baseline hard near-tie target-match `86.29%`

Best tested continuation policies:

- `fixed_final`
  - hard near-tie recovery `8.10%`
  - hard near-tie new-error `9.03%`
  - hard near-tie target-match `85.36%`
  - large-gap control target-match `97.32%`
- `risk_gate_tight`
  - same hard near-tie and control behavior as `fixed_final`
  - but no cleaner selective-compute win
- `margin_gate_050` / `margin_gate_100`
  - both degraded hard near-tie target-match to `81.31%`
  - both damaged large-gap controls badly
- `learned_gate`
  - hard near-tie target-match `79.13%`
  - large-gap control target-match `85.49%`
- `fixed_middle`
  - catastrophic control damage

So the compute-policy result is:

- extra compute does change some of the right decisions
- but none of the tested selectors preserve a net gain on the frontier slice

## Current Read

The variable-compute benchmark already supports two conclusions:

1. a real near-tie compute opportunity exists, because seed314 improved under
   extra fixed compute
2. that opportunity is not broad enough to justify global fixed compute

Round nine adds a third conclusion:

3. the tested cheap continuation rules were not selective enough to preserve the
   useful part of that extra-compute signal

So the current benchmark verdict is negative for both:

- unconditional fixed extra compute as a default policy
- the tested direct compute-compression rules as deployment replacements
