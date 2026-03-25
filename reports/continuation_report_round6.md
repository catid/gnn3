# Continuation Report Round 6

## Scope

Round six opened a new architectural pass only where the inductive bias looked strong enough to move the learned policy more materially than the exploit-side levers already exhausted in rounds four and five.

This pass completed:

1. a fresh `multiheavy` guardrail reproduction on the corrected feasible suites
2. a regime-stratified failure audit
3. a regime-expert scout
4. a plannerized decoder scout
5. a bounded hazard-memory scout
6. a gated decision to scope out repair rather than open another branch after the first-stage scouts failed the policy-movement bar

Key artifacts:

- [continuation_audit_round6.md](/home/catid/gnn3/reports/continuation_audit_round6.md)
- [regime_audit_round6.md](/home/catid/gnn3/reports/regime_audit_round6.md)
- [round6_multiheavy_guardrail.csv](/home/catid/gnn3/reports/plots/round6_multiheavy_guardrail.csv)
- [round6_regime_audit_seed311_summary.csv](/home/catid/gnn3/reports/plots/round6_regime_audit_seed311_summary.csv)
- [round6_regime_audit_seed311_summary.png](/home/catid/gnn3/reports/plots/round6_regime_audit_seed311_summary.png)
- [round6_scout_seed312_compare.csv](/home/catid/gnn3/reports/plots/round6_scout_seed312_compare.csv)
- [round6_scout_seed312_compare.png](/home/catid/gnn3/reports/plots/round6_scout_seed312_compare.png)
- [round6_policy_movement_summary.csv](/home/catid/gnn3/reports/plots/round6_policy_movement_summary.csv)
- [round6_policy_movement_summary.png](/home/catid/gnn3/reports/plots/round6_policy_movement_summary.png)
- [round6_portfolio_usage.csv](/home/catid/gnn3/reports/plots/round6_portfolio_usage.csv)
- [round6_portfolio_usage.png](/home/catid/gnn3/reports/plots/round6_portfolio_usage.png)

## Fresh Multiheavy Guardrail

The fresh round-six `multiheavy` baseline reproduced the current exploit default cleanly on the corrected feasible suites.

Across matched seeds `311 / 312 / 313`:

- mean test next-hop accuracy: `96.10%`
- mean rollout next-hop accuracy: `95.52%`
- mean regret: `1.32`
- mean p95 regret: `4.77`
- mean deadline miss rate: `41.7%`

Per-seed rollout remained in the same band as the established round-four default:

- seed `311`: regret `1.50`, p95 `5.96`, miss `43.8%`
- seed `312`: regret `1.92`, p95 `5.45`, miss `43.8%`
- seed `313`: regret `0.55`, p95 `2.90`, miss `37.5%`

This is the reference point for every round-six claim.

## Regime Audit

The regime audit was decisive. `multiheavy` is not failing because it cannot find routes:

- every audited episode stayed oracle-feasible
- every audited episode stayed solved

The actual failure mode is deadline robustness under tight, crowded, deeper regimes.

Worst strata:

- `critical` slack band: mean regret `6.59`, p95 `22.54`, miss `100%`
- `very_tight` slack band: mean regret `4.60`, p95 `13.67`, miss `76.3%`
- `5+` packets: mean regret `8.83`, p95 `25.34`, miss `96.8%`
- `high_load`: mean regret `7.35`, p95 `26.38`, miss `96.3%`
- depth `4`: mean regret `7.12`, p95 `25.00`, miss `90.1%`

That closes the loop on the benchmark contract: the next useful branch must change behavior in the tight / high-load / deep slice, not generic feasibility.

## Architectural Scouts

### B1. Regime Experts

This branch added small regime-conditioned expert heads on top of the shared `multiheavy` trunk.

Shared-seed result on seed `312`:

- exact test-rollout match to baseline on regret, p95 regret, and miss rate
- same selected epoch and same selection score as the fresh baseline

Policy movement gate:

- baseline suite agreement: `1.000`
- max overall disagreement across checked suites: `1.31%`
- max hard-feasible disagreement: `0.0%`

Verdict:

- killed early
- no evidence of meaningful hard-case policy movement

### C1. Plannerized Decoder

This branch reused the candidate-path scaffold and let a plannerized decoder influence final selection through learned path cost and on-time scores.

Shared-seed result on seed `312`:

- baseline regret `1.92` -> planner `1.78`
- p95 regret unchanged at `5.45`
- miss unchanged at `43.8%`
- next-hop accuracy `94.53%` -> `95.31%`

That small gain did not survive the hard gate:

- baseline suite disagreement: `0.26%`
- max overall disagreement across checked suites: `0.58%`
- max hard-feasible disagreement: `0.0%`

The OOD check also broke the case for promotion:

- `branching3`: improved regret `6.84 -> 4.13`
- `deeper_packets6`: worsened regret `3.42 -> 8.62`
- `heavy_dynamic`: catastrophic failure, regret `8.36 -> 838.29`, p95 `29.44 -> 3348.89`

Verdict:

- killed early
- slight seed-level improvement was metric noise, not meaningful policy movement

### E1. Hazard Memory

This branch added a narrow structured hazard-memory side channel into the slow state rather than another generic history bank.

Training behavior on seed `312`:

- epoch `1` collapsed badly: rollout regret `5089.89`, p95 `14859.06`, miss `87.5%`
- by epoch `2`, the selected validation behavior snapped back to the baseline selection score

Selected test result:

- exact match to the fresh seed312 baseline on regret, p95 regret, miss rate, and rollout next-hop accuracy

Verdict:

- killed early
- unstable first step, then baseline-equivalent selected policy

## Repair Decision

The repair branch was **not opened**.

That was an explicit round-six gate, not a skipped task. The experiment order required a first-stage constructor branch to show real hard-case policy movement before opening repair. Neither regime experts nor the plannerized decoder cleared that bar, so opening repair would have violated the round-six discipline.

## Policy-Movement Conclusion

Round six did not produce a promoted contender.

Across the branches that mattered:

- `multiheavy` stayed the robust default
- regime experts did not move hard feasible cases at all
- the plannerized decoder moved policy only cosmetically and failed badly on one hard OOD suite
- hazard memory did not survive beyond a transient unstable epoch

The strongest positive result of the round is negative knowledge:

- the current plateau is real
- small architectural side channels on top of `multiheavy` are not enough
- the repo should not reopen repair, reranking, selector-only tuning, or train-only exploit tweaks without a constructor branch that first proves it can actually change the hard slice

## Portfolio And Resource Use

Round-six actual GPU-hours landed inside the requested exploration-heavy window.

Round-six totals:

- exploit: `0.3604`
- explore: `0.9976`
- split: `26.5% exploit / 73.5% explore`

The exploration total includes the three architecture scouts plus the two long diagnostic policy-compare runs that were stopped after `21m52s` each once the branch-kill evidence was already sufficient.

## Recommendation

Keep plain `multiheavy` as the exploit default.

Do not promote any round-six branch. Do not open repair unless a future constructor branch first shows nontrivial hard-case disagreement and a credible fix rate on `multiheavy` failures. If a new architectural round is opened later, it needs to be materially stronger than round-six regime prompts, small experts, planner-cost side heads, or narrow hazard memory.
