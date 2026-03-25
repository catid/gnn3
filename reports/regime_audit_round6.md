# Regime Audit Round 6

## Scope

- Baseline checkpoint: [e3_memory_hubs_rsm_round6_multiheavy_seed311](./plots/round6_regime_audit_seed311.json)
- Evaluation suites:
  - [e3_memory_hubs_rsm_round6_multiheavy_seed311.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round6_multiheavy_seed311.yaml)
  - [a2_multiheavy_ood_branching3_round6_eval.yaml](/home/catid/gnn3/configs/experiments/a2_multiheavy_ood_branching3_round6_eval.yaml)
  - [a2_multiheavy_ood_deeper_packets6_round6_eval.yaml](/home/catid/gnn3/configs/experiments/a2_multiheavy_ood_deeper_packets6_round6_eval.yaml)
  - [a2_multiheavy_ood_heavy_dynamic_round6_eval.yaml](/home/catid/gnn3/configs/experiments/a2_multiheavy_ood_heavy_dynamic_round6_eval.yaml)
- Summary artifact: [round6_regime_audit_seed311_summary.csv](/home/catid/gnn3/reports/plots/round6_regime_audit_seed311_summary.csv)

## Main Finding

`multiheavy` is not failing because it cannot find routes. It solves every audited episode, and every audited episode still has at least one oracle-feasible path. The failure mode is deadline robustness under tight, crowded, deeper regimes.

Across the combined baseline-plus-OOD audit:

- all strata stayed at `1.0` solved rate
- all strata stayed at `1.0` feasible-episode fraction
- deadline miss and tail regret still exploded in the hard regimes

That means the round-six architecture work should target policy quality under constraint pressure, not generic feasibility or generic communication capacity.

## Regimes That Actually Fail

By deadline tightness:

- `critical` slack band: `130` episodes, mean regret `6.59`, p95 regret `22.54`, miss rate `100%`
- `very_tight` slack band: `80` episodes, mean regret `4.60`, p95 regret `13.67`, miss rate `76.3%`
- `tight` slack band: `22` episodes, mean regret `2.70`, p95 regret `9.53`, miss rate `40.9%`
- `moderate` and `loose` bands were much safer, with miss rates `0%` and `25%`

By packet pressure:

- `5+` packets: `94` episodes, mean regret `8.83`, p95 regret `25.34`, miss rate `96.8%`
- `4` packets: miss rate `91.1%`
- `3` packets: miss rate `85.7%`
- `1` packet stayed much safer at `30.0%` miss and `0.84` regret

By load:

- `high_load`: `82` episodes, mean regret `7.35`, p95 regret `26.38`, miss rate `96.3%`
- `mid_load`: miss rate `84.8%`
- `low_load`: miss rate `69.6%`

By depth:

- depth `4`: `121` episodes, mean regret `7.12`, p95 regret `25.00`, miss rate `90.1%`
- depth `3`: mean regret `3.46`, p95 regret `11.81`, miss rate `77.5%`

By hub asymmetry:

- `low_gap`: mean regret `8.17`, miss rate `96.2%`
- `mid_gap`: mean regret `6.45`, miss rate `81.0%`
- `high_gap`: mean regret `1.63`, miss rate `74.4%`

The strongest recurring bad slice is:

- critical or very-tight slack
- `5+` packets
- high load
- depth `4`

That is the slice round-six branches need to move.

## Implication For Branch Design

This audit supports three constraints on the next architectural tests:

1. A branch must change behavior in the tight/high-load/deep strata, not just improve aggregate averages.
2. A branch that leaves hard-case policy agreement near `1.0` with `multiheavy` is not interesting enough to promote.
3. The most credible next inductive bias is one that conditions on regime structure or decodes path-level cost/constraint tradeoffs more explicitly.

## Immediate Recommendation

- Keep `multiheavy` as the guardrail baseline.
- Use this audit as the kill/promotion gate for round-six scouts.
- Focus the first real architectural scout on explicit path-level planning structure, then measure:
  - disagreement on hard feasible cases
  - miss-rate change in critical and very-tight strata
  - p95 regret change in `5+` packet and depth-`4` strata
