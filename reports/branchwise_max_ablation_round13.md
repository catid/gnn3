# Round 13 Branchwise-Max Ablation

## Scope

Wave B isolated why `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` works. The analysis used the archived summary exports from the closed follow-up family and produced:

- `reports/plots/round13_branchwise_ablation_table.csv`
- `reports/plots/round13_branchwise_ablation_best.csv`
- `reports/plots/round13_branchwise_ablation_summary.png`

The ablation groups were:

- max timing
- branch source
- negative cleanup source
- support / agree / mix
- calibration

## B1. Max Timing

At every matched budget in the tested band, branchwise max before the outer mix beat the softer alternatives:

- `0.75%`: branchwise-max-before-mix `66.7%` recall, hard near-tie `90.53% -> 90.60%`, overall mean delta regret `-0.0137`
- `1.00%`: `83.3%` recall, `90.53% -> 90.60%`, overall mean delta regret `-0.0145`
- `1.50%`: `83.3%` recall, `90.53% -> 90.60%`, overall mean delta regret `-0.0159`
- `2.00%`: `100.0%` recall, `90.53% -> 90.68%`, overall mean delta regret `-0.0167`

Takeaway:

- hard max inside the shared and dual branches remains better than global max after the outer mix
- hard max remains better than branch-local learned lift

## B2. Branch Source

The gain requires both branches.

- shared-only sharp cleanup was dead or near-dead
- dual-only sharp cleanup found only a tiny `25%` recall niche and never moved beyond the weak `90.53% -> 90.60%` band
- shared-plus-dual branchwise max was the only configuration that preserved the round-13 frontier

Takeaway:

- the live gain is not a single-branch artifact
- both shared and dual branches contribute necessary correction signal

## B3. Negative Cleanup Source

The source-family split is still real:

- `0.75%`: sharp negative tail is best
  - `75.0%` recall
  - hard near-tie `90.53% -> 90.73%`
  - overall mean delta regret `-0.0144`
- `1.00%` to `2.00%`: fixed negative tail is best on recall and hard-slice target match
  - `100.0%` recall
  - hard near-tie `90.53% -> 90.80%`
  - but only `-0.0131` to `-0.0138` overall mean delta regret

Takeaway:

- sharp negative cleanup provides the sub-`1%` efficient lane
- fixed negative cleanup provides the archived high-recall lane
- branchwise max works because it fuses those source families inside the branches instead of replacing either one globally

## B4. Support / Agree / Mix

The progression across bank geometry stays consistent:

- `0.75%`: sharp-negative support-agree mix is best
- `1.00%`: the older support-weighted agreement mixture is still the best archived single-branch matched-band reference
- `1.50%` to `2.00%`: branchwise-max support-agree mix becomes best

Takeaway:

- support weighting and agreement mixing remain necessary
- branchwise max is the right refinement of the support-weighted agreement-mixture geometry, not a replacement for it

## B5. Calibration

Several calibration variants were real signal but none replaced hard branchwise max:

- `0.75%`: branch-strength sharp was best among calibration-only controls
- `1.00%`: branchwise margin-max was best among calibration-only controls
- `1.50%`: branch-calibrated sharp, branch-strength sharp, branchwise margin-max, and learned-gate sharp effectively tied
- `2.00%`: branchwise margin-max was best among calibration-only controls, but still below branchwise max

Takeaway:

- calibration freedom helps, but only as a secondary adjustment
- the dominant mechanism is still branch-local hard fusion, not better slope/threshold learning

## Simplification Decision

Wave B supports the simplest surviving mechanism:

- keep support-weighted agreement mixture
- keep branch-local negative cleanup
- keep both shared and dual branches
- fuse the sharp and fixed cleanup paths with a hard branchwise max before the outer mix

Do not promote:

- global max after mix
- soft branchwise lift
- margin-thresholded max as the default
- strict joint-support fusion
- calibration-only replacements for branchwise max

## Round-13 Mechanism Conclusion

The accepted mechanism is now:

1. support-weighted bank retrieval
2. branch-local negative cleanup
3. shared + dual branch scoring
4. hard branchwise max between the sharp-cleanup and fixed-cleanup paths
5. outer agreement mixture

That is the simplest explanation that survives both the archived frontier and the rerun robustness gate.
