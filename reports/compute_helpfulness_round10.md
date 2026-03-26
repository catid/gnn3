# Round 10 Compute Helpfulness Audit

## Scope

This audit refreshes the round-9 extra-compute thesis on the accepted round-9 frontier surface.
The comparison is plain `multiheavy` vs `compute5` on the audited mixed-compute and OOD suites for seeds
`314`, `315`, and `316`.

The question is not whether extra compute can ever help. The question is whether the helpful-compute slice is
large enough and stable enough to support a cheap gate or distilled student.

## Aggregate result

Across all audited decisions, the helpful-compute slice exists but is narrow:

- overall decisions: `13121`
- overall action-change rate: `2.81%`
- overall helpful rate: `1.07%`
- overall harmful rate: `1.74%`
- overall mean delta regret: `+0.0838`
- overall baseline target match: `95.91%`
- overall compute target match: `95.22%`

On the full hard near-tie slice, extra compute is not robust enough to promote directly:

- hard near-tie decisions: `2173`
- disagreement / action-change rate: `4.34%`
- helpful rate: `1.92%`
- harmful rate: `2.42%`
- mean delta regret: `+0.1143`
- baseline target match: `89.52%`
- compute target match: `89.02%`

So the broad hard near-tie slice is still net-negative for fixed extra compute.

## Stable positive corner

The one consistently positive source family is the high-headroom near-tie subset:

- high-headroom near-tie decisions: `123`
- helpful rate: `18.90%`
- harmful rate: `0.00%`
- mean delta regret: `-0.6030`
- compute target match gain: `0.00 -> 18.90%`

The baseline-error near-tie subset is also genuinely positive:

- baseline-error near-tie decisions: `244`
- helpful rate / recovery rate: `13.36%`
- harmful rate: `0.00%`
- mean delta regret: `-0.3455`
- target match: `0.84% -> 14.20%`

This is the key round-10 go signal. Extra compute is not broadly useful on all hard near-tie states, but it
does recover a meaningful subset of audited baseline errors when those cases also have large enough remaining
headroom.

## Seed and suite stability

Helpfulness is not stable on the broad hard near-tie slice:

- seed314 mixed-compute hard near-tie: `6.19%` helpful vs `4.42%` harmful
- seed315 mixed-compute hard near-tie: `0.28%` helpful vs `1.13%` harmful
- seed316 mixed-compute hard near-tie: `1.51%` helpful vs `1.01%` harmful

The `deeper_packets6` OOD slice is especially hostile:

- seed314 hard near-tie: `8.22%` helpful vs `10.14%` harmful
- seed315 hard near-tie: `0.00%` helpful vs `4.05%` harmful
- seed316 hard near-tie: `0.00%` helpful vs `0.00%` harmful

`heavy_dynamic` is mostly neutral, with small positive pockets on seeds `314` and `316`, but not enough volume
to define the whole frontier policy.

By contrast, high-headroom near-tie remains the only cleanly positive family:

- seed314 high-headroom near-tie helpful rate: `48.18%`
- seed315 high-headroom near-tie helpful rate: `1.85%`
- seed316 high-headroom near-tie helpful rate: `20.00%`

This is still unstable in magnitude, but unlike the broader hard near-tie slice it is not anti-oracle.

## Decision

Round 10 should continue, but only under a narrower thesis:

- do not treat all hard near-tie states as compute-helpful
- do not promote fixed `compute5` directly
- treat high-headroom near-tie and audited baseline-error near-tie states as the candidate helpful-compute source
  families
- require any gate or student to improve those source families without regressing the broader hard near-tie slice

## Artifacts

- Per-seed audit summaries:
  - `reports/plots/round10_helpfulness_seed314_summary.csv`
  - `reports/plots/round10_helpfulness_seed315_summary.csv`
  - `reports/plots/round10_helpfulness_seed316_summary.csv`
- Per-seed source-family stability tables:
  - `reports/plots/round10_helpfulness_seed314_stability.csv`
  - `reports/plots/round10_helpfulness_seed315_stability.csv`
  - `reports/plots/round10_helpfulness_seed316_stability.csv`
