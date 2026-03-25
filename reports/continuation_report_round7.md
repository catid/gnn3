# Continuation Report Round 7

## Scope

Round seven opened a constructor-only architecture pass aimed at one question:

- can a new branch produce real disagreement with plain `multiheavy` on the hard-feasible slice and improve behavior there?

This pass completed:

1. a repo-grounded round-seven audit
2. a refreshed `multiheavy` guardrail reproduction
3. a hard-feasible action-gap audit
4. a frozen representation probe audit
5. a poly-constructor scout
6. a self-improving constructor scout
7. an independent specialist-teacher audit
8. a round-seven policy-movement and portfolio refresh

Key artifacts:

- [continuation_audit_round7.md](/home/catid/gnn3/reports/continuation_audit_round7.md)
- [hard_feasible_action_gap_round7.md](/home/catid/gnn3/reports/hard_feasible_action_gap_round7.md)
- [probe_audit_round7.md](/home/catid/gnn3/reports/probe_audit_round7.md)
- [round7_multiheavy_guardrail.csv](/home/catid/gnn3/reports/plots/round7_multiheavy_guardrail.csv)
- [round7_hard_feasible_action_gap_summary.csv](/home/catid/gnn3/reports/plots/round7_hard_feasible_action_gap_summary.csv)
- [round7_probe_audit_summary.csv](/home/catid/gnn3/reports/plots/round7_probe_audit_summary.csv)
- [round7_policy_movement_summary.csv](/home/catid/gnn3/reports/plots/round7_policy_movement_summary.csv)
- [round7_hard_slice_branch_summary.csv](/home/catid/gnn3/reports/plots/round7_hard_slice_branch_summary.csv)
- [round7_portfolio_usage.csv](/home/catid/gnn3/reports/plots/round7_portfolio_usage.csv)

## Fresh Guardrail

The fresh round-seven `multiheavy` guardrail stayed in the same band as round six.

Base corrected feasible suite on seed `312`:

- test next-hop accuracy: `95.47%`
- rollout next-hop accuracy: `94.53%`
- average regret: `1.92`
- p95 regret: `5.45`
- deadline miss rate: `43.8%`

The OOD guardrail stayed hard but feasible:

- `branching3`: regret `5.18`, p95 `15.13`, miss `75.0%`
- `deeper_packets6`: regret `4.30`, p95 `13.39`, miss `75.0%`
- `heavy_dynamic`: regret `3.64`, p95 `11.70`, miss `75.0%`

That is the round-seven baseline anchor.

## Audit Result: The Main Hypothesis Changed

The action-gap audit was the most important round-seven result.

The original large-gap hard-feasible target was too strict and, after correction, mostly wrong as a constructor hypothesis.

Using the corrected score-based hard slice:

- hard-feasible decisions: `1490`
- hard-feasible episodes: `85`
- thresholded large-gap hard-feasible decisions: `677`
- thresholded large-gap hard-feasible episodes: `35`

Baseline error pattern:

- score-based hard-feasible error: `2.68%` (`40 / 1490`)
- thresholded large-gap hard-feasible error: `0.30%` (`2 / 677`)
- quartile-defined hard large-gap error: `0.00%` (`0 / 360`)
- hard near-tie error: `5.49%` (`37 / 674`)

This means the audited opportunity is **not** “flip obvious large-gap hard-feasible mistakes.” That slice is already almost solved by `multiheavy`.

The remaining opportunity, if any, is in hard **near-tie** states, especially in:

- `deeper_packets6`
- `heavy_dynamic`

That changed how the constructor scouts were judged:

- branches with zero hard-slice disagreement were killed quickly
- branches with disagreement that moved away from oracle targets were also killed quickly

## Probe Result: Representation Is Not The Main Bottleneck

The frozen probe audit says the backbone already linearly exposes most of the signals a better constructor would need.

OOD probe accuracy ranges:

- slack bucket: `0.869` to `0.887`
- critical packet proxy: `0.968` to `0.978`
- feasible continuation: `0.999` to `1.000`
- baseline strictly suboptimal: `0.958` to `0.973`
- oracle gap bucket: `0.649` to `0.727`

Only explicit depth/load regime labels generalized poorly under OOD suite shift:

- `0.000` on `branching3`
- `0.000` on `deeper_packets6`
- `0.310` on `heavy_dynamic`

The main takeaway is still clear:

- the encoder is already carrying slack, feasibility, and baseline-error information
- round seven therefore looks like a constructor bottleneck, not a simple missing-signal bottleneck

## Constructor Scouts

### B1. Poly-Constructor

This branch failed before the disagreement gate mattered.

On seed `312`, epoch `1` collapsed hard:

- selection score: `0.383`
- rollout next-hop accuracy: `50.7%`
- average regret: `5419.82`
- p95 regret: `13497.61`
- deadline miss rate: `93.8%`

Verdict:

- killed immediately
- no promotion path

### B2. Self-Improving Constructor

This branch snapped to the exact baseline selected policy and then stayed there.

Base-suite selected rollout on seed `312` matched baseline exactly:

- rollout next-hop accuracy: `98.06%`
- average regret: `0.368`
- p95 regret: `1.754`
- miss rate: `56.3%`

More importantly, the focused hard-slice compare showed exact policy identity across the audited OOD hard slice:

- overall hard-slice disagreement: `0.0%`
- hard-feasible disagreement: `0.0%`
- large-gap hard-feasible disagreement: `0.0%`
- hard near-tie disagreement: `0.0%`

Verdict:

- killed
- exact policy reproduction, not a new constructor

### B3. Specialist Teachers

Three independent teachers were trained:

- tight-slack teacher
- heavy `5+` packet teacher
- depth-4 teacher

#### Tight Teacher

- base-suite selected rollout matched baseline exactly
- hard-slice disagreement: `0.0%`

Verdict:

- killed

#### Depth-4 Teacher

- base-suite selected rollout matched baseline exactly
- hard-slice disagreement: `0.0%`

Verdict:

- killed

#### Heavy Teacher

This was the only round-seven constructor with real hard-slice movement.

On the audited OOD hard slice:

- overall disagreement: `3.67%`
- hard-feasible disagreement: `4.71%`
- large-gap hard-feasible disagreement: `5.13%`

The movement concentrated in `deeper_packets6`:

- overall disagreement: `8.0%`
- hard-feasible disagreement: `8.0%`
- large-gap hard-feasible disagreement: `7.69%`
- hard near-tie disagreement: `6.67%`

That was enough to clear the “is it moving at all?” question, but it failed the quality gate:

- `deeper_packets6` hard-slice target-match fell from `0.98` to `0.90`
- aggregate hard-slice target-match fell from `0.9647` to `0.9176`
- aggregate large-gap target-match fell from `1.000` to `0.9487`

So the heavy teacher is the first constructor in this family to move materially, but it moved in the wrong direction.

Verdict:

- killed
- informative negative: movement without improvement

## Promotion Decision

No round-seven branch was promoted.

This satisfies the round-seven ladder cleanly:

- all three constructor families were tested
- one family failed catastrophically before compare (`poly`)
- two families produced exact policy identity (`self-improve`, `tight teacher`, `depth4 teacher`)
- one sub-branch produced real disagreement (`heavy teacher`) but degraded hard-slice target quality

There is no contingent promotion branch to open after that.

## Round-Seven Conclusion

Round seven closed more questions than round six.

The two decisive new answers are:

1. the remaining opportunity is **not** a large-gap hard-feasible mistake slice
2. the backbone already encodes most of the relevant local signals, so the plateau is not explained by a simple representation failure

That leaves a much tighter conclusion:

- plain `multiheavy` remains the default policy
- the current constructor families are closed
- another constructor round should only open if it targets hard near-tie ambiguity with a materially stronger policy-forming mechanism than:
  - poly heads on the same trunk
  - bounded self-imitation of the same policy
  - small independent specialists trained on the same supervision

## Portfolio And Resource Use

Round-seven actual GPU-hours stayed inside the requested exploration-heavy window.

Round-seven totals:

- exploit: `0.1330`
- explore: `0.3806`
- split: `25.9% exploit / 74.1% explore`

That satisfies the `25–35% / 65–75%` target while still keeping the baseline guardrail and both audits completed.

## Recommendation

Keep plain `multiheavy` as the exploit default.

Do not reopen the round-seven constructor families in their current form. If a future architecture round is opened, it should start from the corrected round-seven diagnosis:

- the real residual error is on hard near-tie states
- large-gap hard-feasible states are already mostly solved
- the backbone already knows much of what a better constructor would need
- the next useful mechanism has to change policy formation more deeply than the branches tested here
