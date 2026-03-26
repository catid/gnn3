# Round 10 Selective Compute

## Setup

Round 10 tested two cheap gated-compute policies derived from the seed314 helpfulness cache:

- `triggered_full_compute`
- `triggered_top2_compute`

Both were evaluated on held-out seed315 and seed316 caches.
The gate threshold chosen from the training cache was `0.35`.

## Main result

Selective compute did not open a useful compute/quality frontier.

Both policies collapsed to effectively base-compute behavior:

- mean reported trigger rate rounds to `0.0` on every slice
- average outer steps stayed at `3.0`
- compute multiplier stayed at `1.0`

So the gate is too conservative or too poorly calibrated to spend extra compute in a meaningful way on held-out
seeds.

## Held-out effect

Because the trigger almost never fires, the held-out policies behave like the base model with tiny incidental
corrections:

- overall target match: `96.57%` vs baseline `96.43%`
- overall mean delta regret: `-0.0066`
- hard near-tie target match: `91.03%` vs baseline `91.16%`
- hard near-tie mean delta regret: `+0.0082`
- baseline-error near-tie correction rate: `1.45%`
- high-headroom near-tie correction rate: `2.22%`

Those tiny gains are not coming from a real compute policy. They are just residual low-volume differences in the
gate path.

## Decision

The current selective-compute family is closed.

Round 10 showed:

- a broad helpfulness gate is not calibrated well enough to safely spend extra compute
- once the gate is made conservative enough to avoid harm, it stops firing
- the result is a policy that is effectively indistinguishable from baseline in compute cost

So the current gate does not rescue the extra-compute thesis.

## Artifact basis

- `reports/plots/round10_selective_compute_summary.csv`
- `reports/plots/round10_selective_compute_decisions.csv`
- `reports/plots/round10_selective_compute.json`
- `reports/plots/round10_selective_compute_summary.png`
