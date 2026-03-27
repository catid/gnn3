# Round 13 Stable-Positive-v3 Mining

## Scope

Wave C used the live prototype family as a positive-mining instrument. The mining pass aggregated selected states from:

- archived shortlist exports
- rerun-1 shortlist exports
- rerun-2 shortlist exports

across:

- `prototype_memory_agree_blend_hybrid`
- `prototype_sharp_negative_tail_support_agree_mix_hybrid`
- `prototype_negative_tail_support_agree_mix_hybrid`
- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

Artifacts:

- `reports/plots/round13_stable_positive_v3_manifest.csv`
- `reports/plots/round13_stable_positive_v3_summary.csv`
- `reports/plots/round13_stable_positive_v3_source_family_summary.csv`
- `reports/plots/round13_stable_positive_v3_overlap.csv`
- `reports/plots/round13_stable_positive_v3_summary.png`

## Result

`stable_positive_v3` did **not** expand beyond `stable_positive_v2`.

Summary:

- `stable_positive_v2_total`: `4`
- `stable_positive_v3_total`: `4`
- `new_v3_total`: `0`

Category counts:

- `stable_positive_v3`: `4`
- `unstable_positive`: `31`
- `useful_hard_negative`: `71`
- `dead_noisy_positive`: `837`

## What Expanded

The live prototype family did surface more structure, but not more stable positives.

### Unstable Positives

There are `31` unstable positives with strong average teacher gain:

- mean best safe teacher gain: `4.23`
- mean selection events: `40.5`
- mean model-family count: `3.32`
- mean source-run count: `2.94`

Interpretation:

- the family is repeatedly finding useful states beyond `stable_positive_v2`
- but those states still fail the stability filter across model families and reruns

### Useful Hard Negatives

There are `71` useful hard negatives:

- mean selection events: `24.8`
- mean model-family count: `2.59`
- mean source-run count: `2.75`

Interpretation:

- the live family is good at re-finding high-value false positives
- those negatives are strong candidates for any future cleanup-threshold tuning or prototype-pruning pass

## Overlap

Source-family overlap is moderate, not universal.

Model-family candidate Jaccard:

- branchwise-max vs negative-tail: `0.213`
- branchwise-max vs sharp-negative: `0.182`
- memory-agree vs negative-tail: `0.240`
- negative-tail vs sharp-negative: `0.261`

Source-run candidate Jaccard:

- archived vs rerun1: `0.435`
- archived vs rerun2: `0.261`
- rerun1 vs rerun2: `0.226`

Interpretation:

- the candidate pool is not stable enough to justify a broader student retry
- rerun drift is still too high for mined positives beyond the original four

## Source-Family Breakdown

Per-family stable-positive-v3 coverage:

- branchwise-max: `4`
- negative-tail: `4`
- sharp-negative: `3`
- memory-agree: `2`

Per-family unstable-positive counts:

- branchwise-max: `30`
- sharp-negative: `29`
- memory-agree: `22`
- negative-tail: `22`

Interpretation:

- branchwise-max is the best mining source
- sharp-negative remains useful as a source-family probe
- memory-agree contributes too little stable signal to justify keeping it on the active frontier

## Go / No-Go

Wave C is a **no-go** for a broader positive-pack promotion.

Accept:

- `stable_positive_v2` remains the only promotion-safe narrow target pack
- `stable_positive_v3_manifest.csv` is useful as an audit artifact
- the unstable-positive and useful-hard-negative subsets are useful for future mechanism analysis

Do not accept:

- a larger stable-positive pack
- a student retry driven by broader mined positives
- a composite dispatcher justified mainly by mined-source expansion

## Round-13 Mining Conclusion

The round-13 prototype family improved mechanism understanding, not the positive source set:

- no new stable positives
- many unstable positives
- many reusable hard negatives

That closes the round-13 student-expansion thesis unless a later round first stabilizes some of the `31` unstable positives into a genuinely larger pack.
