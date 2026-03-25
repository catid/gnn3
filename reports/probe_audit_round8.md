# Probe Audit Round 8

## Scope

Round eight extended the frozen-feature audit specifically for the hard near-tie diagnosis.

Artifacts:

- [round8_probe_audit_summary.csv](/home/catid/gnn3/reports/plots/round8_probe_audit_summary.csv)
- [round8_probe_audit_summary.png](/home/catid/gnn3/reports/plots/round8_probe_audit_summary.png)
- [round8_probe_audit.json](/home/catid/gnn3/reports/plots/round8_probe_audit.json)

The probe suite used the fresh round-eight `multiheavy` seed `312` checkpoint on:

- train split of the same config
- corrected feasible base eval suite
- `branching3`
- `deeper_packets6`
- `heavy_dynamic`

## Main Result

The frozen backbone still exposes enough information to support a better ambiguous-state decision rule.

The strongest round-eight signals are:

- oracle gap bucket remains linearly predictable OOD at `0.657` to `0.762`
- oracle near-tie classification remains strong, though it degrades on `deeper_packets6`
- deadline-risk bucket remains strong OOD at `0.877` to `0.902`
- pairwise top-2 ranking stays very strong OOD at `0.942` to `0.963`

This is consistent with the round-seven conclusion that the plateau is not primarily a missing-feature problem.

## Key Probe Results

### Oracle Gap Bucket

OOD accuracy:

- base corrected feasible suite: `0.762`
- `branching3`: `0.735`
- `deeper_packets6`: `0.657`
- `heavy_dynamic`: `0.733`

This is not perfect, but it is good enough to justify trying counterfactual value supervision instead of assuming the backbone cannot separate ambiguous actions.

### Oracle Near-Tie

OOD accuracy:

- base corrected feasible suite: `0.975`
- `branching3`: `0.978`
- `deeper_packets6`: `0.750`
- `heavy_dynamic`: `0.882`

The main degradation is in `deeper_packets6`, which is exactly where round-eight expects the most useful work.

### Deadline-Miss Risk Bucket

OOD accuracy:

- base corrected feasible suite: `0.902`
- `branching3`: `0.897`
- `deeper_packets6`: `0.893`
- `heavy_dynamic`: `0.877`

So the frozen features still linearly encode deadline risk well even under the harder OOD regimes.

### Pairwise Top-2 Ranking

OOD accuracy:

- base corrected feasible suite: `0.951`
- `branching3`: `0.951`
- `deeper_packets6`: `0.963`
- `heavy_dynamic`: `0.942`

This is the strongest constructive signal in the probe audit. It suggests the feature space already contains enough relative ordering information to support:

- a counterfactual all-action critic
- bounded near-tie search scored by that critic

### Baseline-Error and Search-Helpful Probes

These binary tasks reported near-perfect accuracy, but they are too imbalanced to headline directly:

- base corrected feasible suite: `1.000`
- `deeper_packets6`: `0.989`
- `heavy_dynamic`: `0.994`

Interpretation:

- they are still consistent with “the signal exists”
- but the class imbalance is severe enough that these should be treated as weak supporting evidence, not the main conclusion

## Decision

The round-eight probe audit supports the same practical conclusion as the headroom audit:

- the backbone is not obviously missing the local signals needed for a better ambiguous-state decision rule
- the next bottleneck is likely in counterfactual decision shaping, ranking, or bounded search

That means the next justified experiment family is:

1. counterfactual all-action critics on cached near-tie decisions
2. bounded search only if those critics actually correct baseline mistakes on the target slice

It does **not** justify reopening:

- broad constructor diversification
- generic feature-side memory changes
- shallow training-loss tweaks with no decision-rule change

## Follow-Up Note

A seed `311` confirmatory rerun of the round-eight probe suite was started late in the round as exploit-side confirmation. It was stopped once the fresh three-seed guardrail batch and third-seed near-tie headroom audit had already fixed the round decision. Round eight therefore treats the completed seed `312` probe audit, plus the existing round-seven probe evidence, as sufficient for the representation-side conclusion.
