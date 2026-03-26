# Round 12 Stable-Positive-v2 Pack

## Definition

Round twelve replaced the round-eleven single-teacher pack with a teacher-bank
version:

- `stable_positive_v2`
- `stable_positive_v2_committee`
- `stable_positive_v2_strict`
- `unstable_positive_v2`
- `harmful_teacher_bank`

The intended change was:

> preserve only states where the richer bank gives a positive correction with
> enough margin and enough stability to trust a precision-first defer policy.

## Result

The canonical pack did not expand.

Counts:

- `stable_positive_v2`: `46`
- `stable_positive_v2_committee`: `30`
- `stable_positive_v2_strict`: `30`

Held-out positives across seeds `315 / 316` remain just:

- `4` total for `stable_positive_v2`
- `2` total for the committee subset

So round twelve improved **confidence**, not **coverage**.

## Where the pack lives

The positive family still sits exactly where rounds ten and eleven said it
would:

- high-headroom near-tie
- baseline-error near-tie
- especially seed314 `deeper_packets6`

It does **not** generalize as a broad hard near-tie family.

## Overlap and robustness

Fine signature overlap stayed zero for all seed pairs.

Coarse signature overlap for `stable_positive_v2` stayed tiny:

- `0.0833`
- `0.1667`
- `0.0`

Committee and strict subsets dropped back to zero overlap even at coarse
signature level.

Threshold robustness was also narrow:

- raising the regret-gain floor to `0.50` removed only one case
- raising it to `0.75` removed five more
- the pack stays small no matter where the threshold is set

This means the pack is real, but still source-fragile.

## Operational use

Round twelve uses two related surfaces:

- Tier 1 diagnostic pack: `stable_positive_v2`
- Tier 1 high-confidence upper bound: `stable_positive_v2_committee`

For evaluation:

- Tier 1 still measures whether a branch can hit the narrow positive family
- Tier 2 remains the full round-nine/ten hard near-tie frontier pack

## Decision

Keep `stable_positive_v2` as the canonical round-twelve narrow correction pack,
but do **not** interpret it as evidence that the problem has become broadly
learnable.

The committee subset is useful for:

- very high-precision offline upper bounds
- bank-backed defer analysis

It is not large enough to justify reopening student compression by itself.

## Artifacts

- `reports/plots/round12_teacher_bank_stable_positive_v2_manifest.csv`
- `reports/plots/round12_teacher_bank_stable_positive_v2_committee_manifest.csv`
- `reports/plots/round12_teacher_bank_seed_summary.csv`
- `reports/plots/round12_teacher_bank_seed_overlap.csv`
- `reports/plots/round12_teacher_bank_sensitivity.csv`
