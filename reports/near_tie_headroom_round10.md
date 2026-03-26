# Round 10 Near-Tie Headroom Refresh

## Purpose

This refresh uses the round-10 helpfulness audit artifacts directly, rather than relying only on the older
round-8 headroom summary. The goal is to confirm that the accepted frontier pack still targets the right source
families before the student and selective-compute wave.

## Aggregate refreshed headroom

Across the audited seeds `314`, `315`, and `316`, the hard near-tie frontier still has real remaining headroom:

- hard near-tie decisions: `2173`
- mean baseline error rate: `11.33%`
- mean baseline continuation gap: `0.1938`
- mean p95 continuation gap: `1.5543`
- mean oracle action gap: `0.7251`
- effective tie under perturbation rate: `4.01%`

The stable near-tie slice remains almost the same story:

- stable near-tie decisions: `2088`
- mean baseline error rate: `9.94%`
- mean baseline continuation gap: `0.1981`
- mean p95 continuation gap: `1.6137`
- mean oracle action gap: `0.7470`

So the accepted frontier pack is still pointed at a real residual error band.

## Where the real headroom lives

The high-headroom near-tie subset is the true remaining opportunity:

- high-headroom near-tie decisions: `123`
- baseline error rate: `100.0%`
- mean baseline continuation gap: `2.7007`
- mean p95 continuation gap: `5.0302`
- mean oracle action gap: `2.4392`
- effective tie under perturbation rate: `0.0%`

This slice is small, but it is exactly the kind of state where extra compute has room to matter and where the
helpfulness audit found the only cleanly positive signal.

The baseline-error near-tie slice is also still substantial:

- baseline-error near-tie decisions: `244`
- baseline error rate: `98.62%`
- mean baseline continuation gap: `1.6772`
- mean p95 continuation gap: `3.6978`
- mean oracle action gap: `1.5445`
- effective tie under perturbation rate: `16.54%`

This makes it a valid teacher/student target, but not all of it is stable enough for a compute gate.

## Control slice remains solved

The large-gap control still behaves like a solved sanity-check slice:

- large-gap control decisions: `2936`
- baseline error rate: `0.20%`
- mean baseline continuation gap: `0.0169`
- mean oracle action gap: `6.7424`

This remains the correct negative-control slice for all round-10 promotion logic.

## Interpretation

Round 10 should not aim to “improve hard near-tie” as one monolithic regime.

The refreshed headroom says:

- broad hard near-tie still contains real residual error, but much of it is low-headroom or ambiguous enough that
  fixed extra compute is not consistently beneficial
- high-headroom near-tie is the most valuable remaining subset for selective compute or teacher-derived correction
- large-gap controls remain solved and must stay solved

This supports the round-10 thesis:

- first identify the subset where extra compute is genuinely helpful
- then test whether a cheap student or compute gate can recover that benefit without paying the full online cost

## Artifact basis

This report was derived from:

- `reports/plots/round10_helpfulness_seed314_decisions.csv`
- `reports/plots/round10_helpfulness_seed315_decisions.csv`
- `reports/plots/round10_helpfulness_seed316_decisions.csv`
