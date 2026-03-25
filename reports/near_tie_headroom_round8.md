# Near-Tie Headroom Round 8

## Scope

Round eight redefined the target slice directly around the current bottleneck:

- score-based hard oracle-feasible decisions
- oracle near-tie decisions
- model near-tie decisions
- the hard near-tie intersection
- the baseline-error subset inside that intersection
- a large-gap hard-feasible control slice

Artifacts:

- [round8_near_tie_headroom_summary.csv](/home/catid/gnn3/reports/plots/round8_near_tie_headroom_summary.csv)
- [round8_near_tie_headroom_suite_summary.csv](/home/catid/gnn3/reports/plots/round8_near_tie_headroom_suite_summary.csv)
- [round8_near_tie_headroom_decisions.csv](/home/catid/gnn3/reports/plots/round8_near_tie_headroom_decisions.csv)
- [round8_near_tie_headroom_summary.png](/home/catid/gnn3/reports/plots/round8_near_tie_headroom_summary.png)
- [round8_near_tie_headroom.json](/home/catid/gnn3/reports/plots/round8_near_tie_headroom.json)

## Fresh Guardrail Context

The audit used the fresh round-eight `multiheavy` guardrail checkpoint:

- seed `312`
- selected epoch `2`
- base corrected feasible-suite rollout regret `1.92`
- p95 regret `5.45`
- deadline miss `43.8%`

That keeps the audit anchored to current code and current manifests rather than archived numbers.

## Main Result

The old round-seven conclusion survives the corrected round-eight audit:

- the large-gap hard-feasible slice is still almost solved
- the remaining opportunity is concentrated in the hard near-tie intersection

Slice summary:

- hard-feasible: `1491` decisions, baseline error rate `2.75%`
- oracle near-tie: `716` decisions, baseline error rate `5.45%`
- model near-tie: `555` decisions, baseline error rate `7.39%`
- hard near-tie intersection: `419` decisions, baseline error rate `9.31%`
- baseline-error intersection: `39` decisions
- large-gap control: `679` decisions, baseline error rate `0.29%`

This is enough headroom to justify constructor-time near-tie work. It is not enough headroom to justify reopening broad large-gap or generic constructor families.

## Headroom Quality

The hard near-tie intersection is not only where most remaining errors live. It is also where the average local opportunity is still meaningful:

- mean regret headroom on the hard near-tie intersection: `1.47`
- p95 regret headroom on that slice: `3.10`
- mean regret headroom on the baseline-error subset: `1.47`
- p95 regret headroom on the baseline-error subset: `3.10`

By contrast, the large-gap control has much larger local gap values but almost no errors left:

- large-gap error rate: `0.29%`
- effective tie rate: `0.0%`

That makes it the right control slice, not the right optimization target.

## Suite Breakdown

The near-tie opportunity is concentrated in the same hard OOD suites that were already problematic:

- `deeper_packets6`: `186` hard near-tie decisions, `13` errors
- `heavy_dynamic`: `224` hard near-tie decisions, `26` errors
- `branching3`: `8` hard near-tie decisions, `0` errors
- base corrected feasible suite: `1` hard near-tie decision, `0` errors

So the actionable round-eight target is effectively:

- ambiguous hard decisions
- mostly in `deeper_packets6` and `heavy_dynamic`
- with oracle feasibility still intact

## Second-Seed Stability Check

A second fresh exploit guardrail and headroom audit on seed `313` landed in the same regime.

Seed `313` summary:

- hard-feasible decisions: `1201`
- hard near-tie intersection decisions: `344`
- baseline-error intersection decisions: `38`
- hard near-tie error rate: `11.0%`
- large-gap control error rate: `0.0%`

The exact counts move by seed, but the qualitative picture does not:

- large-gap remains effectively solved
- the remaining opportunity still sits in the hard near-tie slice
- the baseline-error subset remains on the order of a few dozen decisions per audited seed

## Tie Sensitivity

The perturbation-based sanity check says many near-tie decisions are still stable under small cost perturbations, but the baseline-error subset is less stable than the slice as a whole:

- hard near-tie intersection effective-tie rate: `4.77%`
- baseline-error intersection effective-tie rate: `17.95%`

Interpretation:

- the remaining errors are not pure noise
- but a nontrivial minority of them are genuinely ambiguity-sensitive

That supports bounded search and counterfactual value supervision more than another monolithic constructor family.

## Third-Seed Stability Check

A third fresh exploit guardrail plus near-tie audit on seed `311` landed in the same regime again.

Seed `311` summary:

- hard-feasible decisions: `1637`
- hard near-tie intersection decisions: `470`
- baseline-error intersection decisions: `47`
- hard near-tie error rate: `10.0%`
- large-gap control error rate: `0.27%`

Headroom stayed in the same range:

- hard near-tie mean regret headroom: `1.39`
- hard near-tie p95 regret headroom: `3.78`
- hard near-tie effective-tie rate: `4.68%`
- baseline-error intersection effective-tie rate: `21.28%`

So across three fresh audited seeds:

- the large-gap control is still effectively solved
- the remaining opportunity remains concentrated in the hard near-tie intersection
- the baseline-error subset remains small but real, and consistently more ambiguity-sensitive than the slice as a whole

## Thresholds Used

Round eight used thresholds derived from the audited hard slice:

- near-tie oracle gap threshold: `4.293`
- model-margin threshold: `2.770`
- large-gap threshold: `4.573`
- large-gap ratio threshold: `0.264`

These values should be treated as round-eight cached slice definitions and reused for matched branch comparisons.

## Decision

The headroom is real enough to continue.

The round should proceed with:

1. cached counterfactual all-action supervision on the hard near-tie slice
2. critic variants that can change decisions on that slice
3. bounded search only if the critic shows real correction of baseline mistakes

The audit does **not** support reopening:

- large-gap constructor work
- generic policy diversification
- broad reranking
