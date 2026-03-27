# Round 13 Conservative Student Retry

## Decision

Wave F was **not run**.

## Why It Was Closed

Round 13 required both of the following before reopening conservative student compression:

1. a materially larger or more stable positive source pack
2. a promoted best prototype or composite frontier that justified imitation

Neither condition was met.

### Source-Pack Gate Failed

From `reports/stable_positive_v3_round13.md`:

- `stable_positive_v2_total`: `4`
- `stable_positive_v3_total`: `4`
- `new_v3_total`: `0`
- `31` unstable positives remain unstable

That means the positive source family did not expand.

### Composite Gate Failed

From `reports/hierarchical_defer_round13.md`:

- the score-band dispatcher lost to the best single branch on the archived frontier
- it lost clearly on rerun 1
- it lost clearly on rerun 2

So there is no new promoted composite teacher to distill.

### Robust Single-Branch Gate Narrowed

From `reports/prototype_frontier_round13.md`:

- only `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` survived the rerun robustness gate as a promoted operating point
- the previously broader shortlist narrowed rather than expanded

That is the wrong regime for a student retry. The bottleneck is still sparse decision precision on a tiny unstable subset, not missing teacher coverage.

## Round-13 Conclusion

Conservative student compression remains closed.

Reopen only if a later round first achieves at least one of:

- a larger stable-positive pack than `stable_positive_v2`
- a composite dispatcher that beats the best single branch at matched budgets
- a clearly stronger robust single-branch frontier than the current branchwise-max result
