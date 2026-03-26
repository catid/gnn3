# Round 12 Retrieval / Prototype Defer

## Setup

Round twelve tested whether the sparse positive family is local in frozen
feature space instead of broadly classifiable.

Families:

- `knn_v2`
- `prototype_committee`
- `margin_retrieval`

All three defer to the same audited `compute5` teacher once selected.

## Result

The family is closed.

Across held-out seeds:

- stable-positive-v2 recall stayed `0.0` at every budget for every variant
- hard near-tie mean delta regret stayed exactly `0.0`
- overall mean delta regret stayed exactly `0.0`
- selected states were almost entirely false positives

This was not a “small positive” result. It was effectively inert.

## Interpretation

The sparse positive family does **not** look like a clean local cluster in the
current frozen feature space at the granularity tested here.

That is consistent with the rest of the diagnosis:

- the representation carries many useful local signals
- but the remaining cross-seed positive family is too sparse and too
  source-fragile to be captured by a simple prototype or nearest-neighbor rule

## Decision

Close retrieval defer.

Do not reopen:

- kNN defer
- prototype defer
- margin plus retrieval hybrids in the current form

## Artifacts

- `reports/plots/round12_retrieval_defer_summary.csv`
- `reports/plots/round12_retrieval_defer_decisions.csv`
- `reports/plots/round12_retrieval_defer_summary.png`
