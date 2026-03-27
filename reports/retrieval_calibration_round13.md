# Round 13 Retrieval / Calibration

## Scope

Wave E stayed inside the accepted prototype thesis:

- prototype retrieval quality matters
- negative cleanup matters
- branch-local fusion matters
- soft learned lifts should not be reopened as the main direction

Round 13 did not launch a new broad retrieval sweep. Instead it used:

- the archived frontier panel
- rerun robustness
- the branchwise-max ablation table
- the v3 mining manifest

to decide which retrieval / calibration components remain worth keeping.

Primary artifacts:

- `reports/plots/round13_branchwise_ablation_table.csv`
- `reports/plots/round13_branchwise_ablation_best.csv`
- `reports/plots/round13_prototype_frontier_*.csv`
- `reports/plots/round13_stable_positive_v3_*.csv`

## Retrieval Conclusions

### Support-weighted retrieval still matters

The support-weighted agreement-mixture family remains the correct base geometry. Wave B preserved the older result:

- support-weighted agreement mixture is the best archived matched-band base at `1.00%`
- branchwise-max improves that geometry at `1.50%+`

### Negative cleanup is the actual retrieval lever

Wave B re-confirmed that the live retrieval improvements come from negative-bank cleanup:

- sharp negative cleanup is the best sub-`1%` archived lane
- fixed negative cleanup is the best archived high-recall lane
- branchwise max is the right fusion between them once robustness is required

### Shared and dual banks are both necessary

Single-branch retrieval variants failed:

- shared-only sharp cleanup: dead
- dual-only sharp cleanup: tiny weak niche only

So the accepted retrieval structure is:

- support-weighted bank scoring
- both shared and dual banks
- branch-local negative cleanup

## Calibration Conclusions

Round 13 calibration-only controls were real signal, but none replaced hard branchwise max.

Best calibration-only controls:

- `0.75%`: branch-strength sharp
- `1.00%`: branchwise margin-max
- `1.50%`: branch-calibrated sharp / branch-strength sharp / branchwise margin-max / learned-gate sharp effectively tie
- `2.00%`: branchwise margin-max

Why they are not promoted:

- all of them trail hard branchwise max on the archived matched-band frontier
- none of them improve the rerun robustness picture enough to replace branchwise max
- several of them recover only the archived frontier, not the robust rerun frontier

## Stable-Positive-v3 Feedback

The v3 mining pass closes the door on more aggressive retrieval / calibration expansion this round:

- `stable_positive_v3_total` stayed at `4`
- `new_v3_total` stayed at `0`
- `31` unstable positives remain
- `71` useful hard negatives were found

Implication:

- future retrieval work should target false-positive cleanup and prototype pruning
- it should not assume a larger positive pack is already available

## Promotion Decision

Keep:

- support-weighted agreement-mixture retrieval
- branch-local negative cleanup
- hard branchwise max before the outer mix

Do not promote:

- calibration-only replacements for hard branchwise max
- score-band composite routing
- retrieval changes that depend on the demoted memory-agree lane
- retrieval changes that assume `stable_positive_v3` has expanded

## Round-13 Retrieval / Calibration Conclusion

The round closes with one accepted retrieval / calibration answer:

- keep the robust `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` family

and one accepted future constraint:

- any later retrieval / calibration change must beat branchwise max **and** survive the rerun robustness gate before it is treated as a real operating point.
