# Round 10 Continuation Report

## Objective

Round ten tested the remaining live thesis from round nine:

> Can we identify the subset of hard near-tie states where extra compute truly helps, and then cheaply imitate or
> selectively invoke that correction without paying full runtime or causing global regressions?

This was a compute-and-distillation campaign, not a broad architecture sweep.

## Accepted baseline

`multiheavy` remains the default exploit policy.

The anchor result is unchanged from the corrected round-four comparison:

- mean regret: `1.32` vs fresh `E3` `2.25`
- mean p95 regret: `4.77` vs `10.48`
- mean deadline miss: `41.7%` vs `54.2%`

Round six reproduced the same baseline band, and rounds seven through ten did not dislodge it.

## Round 10 findings

### 1. The helpful-compute slice is real, but much narrower than the full hard near-tie frontier

The broad hard near-tie slice is still not a direct compute win:

- aggregate helpful rate: `1.92%`
- aggregate harmful rate: `2.42%`
- aggregate mean delta regret: `+0.1143`

The only clean positive source family is high-headroom near-tie:

- helpful rate: `18.90%`
- harmful rate: `0.00%`
- mean delta regret: `-0.6030`

The baseline-error near-tie subset is also genuinely positive:

- recovery rate: `13.36%`
- mean delta regret: `-0.3455`

So the extra-compute thesis survives only in a narrow high-headroom correction band.

### 2. Frozen-state helpfulness prediction is not good enough to gate compute safely

The probe suite found some ranking signal, mostly from simple ambiguity features:

- `margin_only` helpful AUROC: `0.8757` on seed315, `0.8535` on seed316
- `margin_plus_regime` helpful AUROC: `0.7763` on seed315, `0.9552` on seed316

But none of the helpfulness probes yielded a usable operating point:

- helpful-task precision/recall at `0.5` stayed at `0 / 0`
- several variants collapsed to zero trigger
- candidate-conditioned helpfulness overfit seed315 and failed on seed316

So the helpful slice is only weakly gateable from the current frozen state.

### 3. Offline distillation found one safe student, but no frontier-pack winner

`gated_pairwise` was the best conservative student:

- overall target match: `96.50%` vs baseline `96.43%`
- overall mean delta regret: `-0.0027`
- large-gap control target match: `99.96%` vs baseline `99.82%`

But it still missed the main gate:

- hard near-tie target match: `91.01%` vs baseline `91.16%`
- hard near-tie mean delta regret: `+0.0110`

Aggressive students (`residual`, `gated_residual`, `gated_kl`) recovered many more baseline-error cases, but only
by regressing the broad hard near-tie slice too much.

Decision: no offline student promoted.

### 4. Selective compute collapsed to the base policy

The held-out gate-triggered compute policies did not open a useful compute frontier:

- chosen threshold: `0.35`
- mean held-out trigger rate rounded to `0.0`
- average outer steps stayed at `3.0`
- compute multiplier stayed at `1.0`

Because the gate almost never fired, selective compute behaved like baseline with tiny residual differences:

- overall mean delta regret: `-0.0066`
- hard near-tie mean delta regret: `+0.0082`
- hard near-tie target match: `91.03%` vs baseline `91.16%`

Decision: selective compute closed in its current form.

## Promotion verdict

No round-ten branch earned contender status.

Closed or not promoted:

- broad fixed compute as a deployment policy
- frozen-feature helpfulness gates as-is
- pure offline distillation as a winner
- selective compute as a frontier policy

Best surviving reference:

- `gated_pairwise` is the safest student found this round, but it is still only a reference baseline for future
  conservative-override work, not a winner.

## Updated recommendation

Keep plain `multiheavy`.

If another round opens, it should target only the narrowest remaining hypothesis:

- precision-first correction on high-headroom baseline-error near-tie states
- explicit abstention penalties
- policy objectives based on net corrected errors, not raw recovery

Do not reopen:

- broad hard near-tie compute gating
- current frozen-feature helpfulness probes as deployment gates
- aggressive residual-style correction heads
- generic online selective compute from the current gate

## Round outputs

- `reports/continuation_audit_round10.md`
- `reports/compute_helpfulness_round10.md`
- `reports/near_tie_headroom_round10.md`
- `reports/variable_compute_benchmark_round10.md`
- `reports/helpfulness_probe_round10.md`
- `reports/offline_distill_round10.md`
- `reports/selective_compute_round10.md`

Key plot/data artifacts:

- `reports/plots/round10_helpfulness_seed314_summary.csv`
- `reports/plots/round10_helpfulness_seed315_summary.csv`
- `reports/plots/round10_helpfulness_seed316_summary.csv`
- `reports/plots/round10_helpfulness_probe_summary.csv`
- `reports/plots/round10_offline_distill_summary.csv`
- `reports/plots/round10_selective_compute_summary.csv`

## Merge / push status

Round ten has not yet been merged back to local `main` in this report draft.
The final merged commit hash and remote push outcome are added after the local merge and push attempt.
