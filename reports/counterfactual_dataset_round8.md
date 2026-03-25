# Counterfactual Dataset Round 8

## Scope

Round eight built a cached all-action counterfactual dataset from the fresh `multiheavy` seed `312` guardrail checkpoint so critic variants can iterate without rebuilding rollout-derived supervision each time.

Artifacts:

- [round8_counterfactual_dataset_seed312.csv](/home/catid/gnn3/artifacts/round8_counterfactual_dataset_seed312.csv)
- [round8_counterfactual_dataset_seed312_summary.csv](/home/catid/gnn3/artifacts/round8_counterfactual_dataset_seed312_summary.csv)
- [round8_counterfactual_dataset_seed312.pt](/home/catid/gnn3/artifacts/round8_counterfactual_dataset_seed312.pt)
- [round8_counterfactual_dataset_seed312.json](/home/catid/gnn3/artifacts/round8_counterfactual_dataset_seed312.json)

The cache stores per-candidate rows with:

- suite / split identity
- episode and decision coordinates
- candidate node
- target and predicted action flags
- hard-feasible and near-tie slice flags
- oracle rank and model top-k rank
- candidate feature tensors
- normalized cost / miss / tail / regret-delta targets

## Main Result

The cached supervision is highly concentrated.

Split summary:

- train: `9004` candidate rows, `54` hard near-tie rows, `0` baseline-error rows
- base corrected feasible eval: `2758` candidate rows, `20` hard near-tie rows, `0` baseline-error rows
- `branching3`: `3842` candidate rows, `20` hard near-tie rows, `0` baseline-error rows
- `deeper_packets6`: `5134` candidate rows, `590` hard near-tie rows, `85` baseline-error rows
- `heavy_dynamic`: `4269` candidate rows, `194` hard near-tie rows, `35` baseline-error rows

Total cached rows:

- `25,007` candidate rows

## Interpretation

This reinforces the headroom audit:

- the critic problem is not spread evenly across the benchmark
- almost all of the actionable near-tie correction signal sits inside:
  - `deeper_packets6`
  - `heavy_dynamic`

That means round-eight critic success should be judged primarily by whether it improves decisions in those suites without creating too many new errors elsewhere.

## Practical Use

The cache is now the standard round-eight supervision source for:

1. frozen-trunk scalar Q critics
2. multi-risk critics
3. pairwise or listwise near-tie ranking variants
4. later gated bounded-search experiments

This should avoid repeated rollout recomputation for every critic variant and keep the scout ladder fast enough to satisfy the round-eight workload target.

