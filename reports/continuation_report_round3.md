# Continuation Report Round 3

## Scope

This round is focused on a fair `X6` versus `E3` comparison under matched conditions, plus exploit-side work on regret and deadline robustness rather than solved-rate inflation.

Key artifact bundle:

- [round3_matched_x6_vs_e3.csv](/home/catid/gnn3/reports/plots/round3_matched_x6_vs_e3.csv)
- [round3_matched_x6_vs_e3.png](/home/catid/gnn3/reports/plots/round3_matched_x6_vs_e3.png)
- [round3_x6_ablation_summary.csv](/home/catid/gnn3/reports/plots/round3_x6_ablation_summary.csv)
- [round3_x6_ablation_summary.png](/home/catid/gnn3/reports/plots/round3_x6_ablation_summary.png)
- [round3_exploit_robustness.csv](/home/catid/gnn3/reports/plots/round3_exploit_robustness.csv)
- [round3_exploit_robustness.png](/home/catid/gnn3/reports/plots/round3_exploit_robustness.png)
- [round3_exploit_ood_comparison.csv](/home/catid/gnn3/reports/plots/round3_exploit_ood_comparison.csv)
- [round3_exploit_ood_comparison.png](/home/catid/gnn3/reports/plots/round3_exploit_ood_comparison.png)
- [portfolio_usage_round3.csv](/home/catid/gnn3/reports/plots/portfolio_usage_round3.csv)
- [portfolio_usage_round3.png](/home/catid/gnn3/reports/plots/portfolio_usage_round3.png)

## Methodology Corrections

- Round three fixed a validation/test leak in the prior stack.
  - Train, val, and test now use split-specific `HiddenCorridorConfig.seed` offsets.
  - Every run now persists `dataset_manifests.json` plus split manifest hashes in `metadata.json` and `summary.json`.
- The archived-best `E3` artifact is now treated as historical signal, not as a replayable baseline.
  - See [e3_repro_audit_round3.md](/home/catid/gnn3/reports/e3_repro_audit_round3.md).
- Seed `211` in the matched contender batch started before the split-fix commit was written.
  - `metadata.json` still reports the earlier commit hash.
  - `summary.json` reports the committed branch head after the fix.
  - The run itself used the corrected split manifests shown in its saved manifest hashes.

## Matched Contender Status

Artifacts:

- [round3_matched_x6_vs_e3.csv](/home/catid/gnn3/reports/plots/round3_matched_x6_vs_e3.csv)
- [round3_matched_x6_vs_e3.png](/home/catid/gnn3/reports/plots/round3_matched_x6_vs_e3.png)

Three-seed matched result on the corrected split:

- `E3` mean test next-hop accuracy: `96.39%`
- `X6` mean test next-hop accuracy: `96.65%`
- delta: `+0.25` percentage points for `X6`
- `E3` mean regret: `2.11`
- `X6` mean regret: `2.11`
- `E3` mean p95 regret: `8.35`
- `X6` mean p95 regret: `8.35`
- both models had `100%` deadline miss rate on this corrected test family

Per-seed summary:

- seed `211`: `X6` improved test next-hop accuracy from `96.31%` to `96.79%`, with identical rollout regret and deadline behavior
- seed `212`: `X6` improved test next-hop accuracy from `96.39%` to `96.67%`, with identical rollout regret and deadline behavior
- seed `213`: `X6` tied `E3` exactly on both test next-hop accuracy and rollout metrics

Decision:

- `X6` is **not promoted** over `E3` from this round-three matched contender batch
- it did not meet the promotion bar of:
  - `>= 10%` mean regret improvement at matched accuracy
  - or `>= 0.4` percentage point next-hop accuracy improvement with no worse regret
  - or a clear deadline / p95 / OOD advantage
- the corrected split strips out the large round-two story; what remains is a small decision-accuracy / value-calibration edge with no rollout-quality gain

## Exploit Robustness Status

First result:

- `e3_memory_hubs_rsm_round3_calibration_seed211`

Calibration-weight scout:

- test next-hop accuracy: `96.79%`
- value MAE: `6.57`
- rollout next-hop accuracy: `93.55%`
- average regret: `2.57`
- p95 regret: `8.20`
- deadline miss rate: `100%`

Interpretation:

- increasing `value_weight` and `route_weight` did **not** improve rollout quality on the seed-211 split
- it slightly improved decision accuracy versus matched `E3`, but made value error worse and left regret / deadlines unchanged

OOD follow-up on the calibration checkpoint:

- [e3_round3_calibration_ood.csv](/home/catid/gnn3/reports/plots/e3_round3_calibration_ood.csv)
- [e3_round3_calibration_ood.png](/home/catid/gnn3/reports/plots/e3_round3_calibration_ood.png)

OOD result summary:

- branching stress regret: `7.08`, deadline miss rate `100%`
- deeper-tree stress regret: `6.02`, deadline miss rate `100%`
- heavy-dynamic stress regret: `6.77`, deadline miss rate `100%`

Take:

- the exploit-side weakness is not fixed by simple loss reweighting
- the bottleneck still looks like deadline / traffic robustness under pressure rather than solved-rate
- the heavier multi-packet curriculum scout was the next exploit-side test

Multiheavy scout:

- test next-hop accuracy: `96.35%`
- value MAE: `6.04`
- rollout next-hop accuracy: `95.73%`
- average regret: `2.39`
- p95 regret: `10.08`
- deadline miss rate: `100%`

OOD follow-up on the multiheavy checkpoint:

- [e3_round3_multiheavy_ood.csv](/home/catid/gnn3/reports/plots/e3_round3_multiheavy_ood.csv)
- [e3_round3_multiheavy_ood.png](/home/catid/gnn3/reports/plots/e3_round3_multiheavy_ood.png)

OOD result summary:

- branching stress regret: `7.08`, deadline miss rate `100%`
- deeper-tree stress regret: `6.02`, deadline miss rate `100%`
- heavy-dynamic stress regret: `6.77`, deadline miss rate `100%`

Take:

- multi-packet curriculum pressure is the only exploit-side change that improved mean in-distribution regret at all
- it did **not** improve p95 regret in-distribution
- it did **not** improve the OOD rollout metrics relative to the calibration scout

## Explore Ablation Status

First result:

- `x6_e3_history_summary_bank_h1_round3_seed211`

`H1` summary-bank ablation:

- test next-hop accuracy: `96.79%`
- value MAE: `5.42`
- rollout next-hop accuracy: `93.55%`
- average regret: `2.57`
- p95 regret: `8.20`
- deadline miss rate: `100%`

Interpretation:

- limiting the summary bank to the latest outer round did **not** hurt rollout quality relative to full `X6`
- this is evidence that multi-round history depth is not carrying the residual `X6` signal on this benchmark
- dense-history read scout result:
  - test next-hop accuracy: `96.79%`
  - value MAE: `4.96`
  - rollout next-hop accuracy: `93.55%`
  - average regret: `2.57`
  - p95 regret: `8.20`
  - deadline miss rate: `100%`
- dense-history reads also match the full `X6` rollout regime on seed `211`

Take:

- the residual `X6` signal is not coming from multi-round history depth
- it is also not coming from summary-bank compression versus dense history reads
- what remains is a small decision-level accuracy / value-calibration effect with no measured rollout gain

## Recommendation

1. Keep `E3` as the lead baseline for the next pass. `X6` is not promoted from round three.
2. Stop spending meaningful GPU budget on full summary-bank history variants for this benchmark. If history reads are revisited, use the cheapest controls (`H1` or dense-history) rather than the full `X6` path.
3. Treat corrected-split deadline behavior as the top blocker. Every matched contender and every scout in this round still had `100%` deadline miss rate.
4. If there is one exploit-side branch worth a cautious follow-up, it is heavier traffic curriculum pressure rather than more loss reweighting; even there, the evidence is only mixed, with slightly better mean regret in-distribution but no p95 or OOD win.
5. The next real leverage point is likely benchmark/training-contract work: deadline scaling or slack audit, explicit deadline-feasibility targets or verifier-backed objectives, and traffic-heavy curricula evaluated across at least 2 more seeds before promotion.

## Portfolio

- round-three added exploit GPU-hours: `0.6544`
- round-three added explore GPU-hours: `0.5735`
- round-three split: `53.3% exploit / 46.7% explore`
- cumulative split remains within target at `51.8% exploit / 48.2% explore`
