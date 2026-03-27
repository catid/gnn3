# Continuation Report Round 13

## Executive Summary

Round 13 was a prototype-frontier and stable-positive-v3 campaign, not a broad architecture sweep.

The main result is narrower than the archived shortlist suggested:

- only `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` survives the rerun robustness gate as a promoted branch
- archived frontier diversity was real but not robust
- `stable_positive_v3` did not expand beyond `stable_positive_v2`
- hierarchical dispatch failed
- conservative student retry remained closed

## What Was Validated

Round 13 started green:

- repo audit completed
- lint and smoke validation passed
- cached deployment, frontier, and prototype smoke paths all ran cleanly

See `reports/continuation_audit_round13.md`.

## Wave A: Matched Frontier

See `reports/prototype_frontier_round13.md`.

Archived frontier:

- `0.25%` to `0.50%`: memory-agree
- `0.75%`: sharp-negative
- `1.00%` to `1.50%`: negative-tail
- `2.00%+`: branchwise-max on aggregate regret

Fresh reruns changed the accepted state:

- rerun 1: branchwise-max leads every budget
- rerun 2: branchwise-max leads every budget
- memory-agree collapsed fully
- negative-tail collapsed fully in rerun 2 and nearly fully in rerun 1
- sharp-negative only recovered a partial middle lane in rerun 2 and was dead in rerun 1

Accepted round-13 frontier result:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` is the only robust live prototype branch

## Wave B: Mechanism Ablation

See `reports/branchwise_max_ablation_round13.md`.

Accepted mechanism:

1. support-weighted agreement-mixture banks
2. both shared and dual branches
3. branch-local negative cleanup
4. hard branchwise max between the sharp-cleanup and fixed-cleanup paths before the outer mix

Rejected as defaults:

- global max after mix
- soft learned lift
- strict joint support
- calibration-only replacements for branchwise max

## Wave C: Stable-Positive-v3

See `reports/stable_positive_v3_round13.md`.

Results:

- `stable_positive_v2_total = 4`
- `stable_positive_v3_total = 4`
- `new_v3_total = 0`
- `31` unstable positives
- `71` useful hard negatives

Interpretation:

- the family is still finding real signal beyond `stable_positive_v2`
- but that signal is not stable enough to support a new positive pack

## Wave D: Hierarchical Dispatcher

See `reports/hierarchical_defer_round13.md`.

Result:

- static budget ladder was only an archived convenience view
- score-band dispatcher did not beat the best single branch
- dispatcher performance was clearly worse on rerun 1 and rerun 2

Accepted outcome:

- keep separate operating points only as analysis tools
- do not promote a composite dispatcher

## Wave E: Retrieval / Calibration

See `reports/retrieval_calibration_round13.md`.

Result:

- support-weighted agreement-mixture retrieval still stands
- negative cleanup is still the real lever
- calibration variants were informative but not promotable
- branchwise max remains the accepted retrieval/calibration answer

## Waves F and G: Student / Deployment

Closed without launch:

- `reports/conservative_student_round13.md`
- `reports/deployment_study_round13.md`

Reason:

- no v3 pack expansion
- no promoted composite frontier
- no new robust branch beyond branchwise-max

## Final Accepted State

### Baseline

- `multiheavy` remains the default deployment policy

### Prototype Side

Promoted:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`
  - accepted as the robust prototype correction reference

Demoted from active promotion surface:

- `prototype_memory_agree_blend_hybrid`
- `prototype_sharp_negative_tail_support_agree_mix_hybrid`
- `prototype_negative_tail_support_agree_mix_hybrid`

These remain mechanistic probes, not accepted robust operating points.

## Round-13 Lessons

1. The live prototype family is real, but the robust frontier is much narrower than archived single-seed outputs suggested.
2. Fusion belongs inside the shared and dual branches before the outer mix.
3. Hard branchwise max remains better than soft lifts or thresholded replacements.
4. Negative cleanup remains essential.
5. The positive source family did not expand, so student compression and composite routing remain premature.

## Next Step

If another round opens, the first question should no longer be:

- which of four archived prototype lanes wins which budget

It should be:

- how to make the branchwise-max family itself more stable and more precise without assuming a broader positive pack already exists

That implies future work should bias toward:

- branchwise-max bank pruning
- branchwise-max false-positive cleanup against the `71` useful hard negatives
- branchwise-max rerun robustness first, before reopening any composite or student path
