# Round 10 Helpfulness Probes

## Question

Can frozen `multiheavy` state predict where extra compute is actually useful well enough to support a practical
 gate?

Training used the round-10 seed314 helpfulness cache. Evaluation used held-out seed315 and seed316 caches.

## Main result

Not well enough.

The helpful class is too sparse and too unstable for a practical frozen-feature gate in the current form.

The strongest ranking signal came from simple ambiguity features, not from a robust learned helpfulness model:

- `margin_only` helpful AUROC: `0.8757` on seed315, `0.8535` on seed316
- `margin_plus_regime` helpful AUROC: `0.7763` on seed315, `0.9552` on seed316

But those models do not give a usable operating point:

- all helpful-task variants had `0.0` precision and `0.0` recall at the default `0.5` threshold
- several variants collapsed to `0` trigger rate entirely
- the candidate-conditioned probe massively overfit seed315 (`0.9924` helpful AUROC) and collapsed on seed316
  (`0.2152`)

## What is predictable

Harmful flips are easier to rank than helpful ones on seed315:

- `margin_plus_regime` harmful AUROC: `0.8996`
- `margin_only` harmful AUROC: `0.8401`
- linear harmful AUROC: `0.7390`

But even that does not transfer cleanly to seed316:

- `margin_plus_regime` harmful AUROC falls to `0.7061`
- linear harmful AUROC falls to `0.2280`
- candidate-conditioned harmful AUROC falls to `0.3322`

So the signal is real, but calibration and transfer are weak.

## Interpretation

Round 10 does not support a broad “predict helpful compute from frozen state” story.

What the probes do support is a narrower claim:

- low-margin plus regime features contain some rank-order information
- that information is not calibrated enough yet to open a reliable high-recall helpfulness gate
- candidate-conditioned features can become extremely optimistic on one held-out seed without surviving the next

This is consistent with the audit result: the helpful-compute slice is narrow, high-headroom, and unstable in
volume across seeds.

## Decision

The current frozen-feature helpfulness gate is not good enough to justify promotion on its own.

Future work should only reopen this direction if it is explicitly conservative, for example:

- optimize precision-first abstention on the high-headroom subset
- treat margin/regime features as weak priors, not as a standalone helpfulness oracle
- require held-out transfer across at least two seeds before opening online compute again

## Artifacts

- `reports/plots/round10_helpfulness_probe_summary.csv`
- `reports/plots/round10_helpfulness_probe.json`
- `reports/plots/round10_helpfulness_probe_summary.png`
