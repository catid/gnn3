# Round 12 Teacher Bank Expansion

## Scope

Round twelve reopened the teacher bank, but only offline and only on the
committed hard near-tie audited decision artifacts.

Teacher sources included:

- round-ten `compute5`
- round-ten triggered selective-compute policies
- round-eleven subset-distill students
- seed314-only round-nine compute-policy auxiliaries around the non-catastrophic
  fixed-compute region

The goal was not to find a new online policy. The goal was to see whether a
richer offline bank could enlarge or stabilize the correction source family.

## Main result

The richer bank did **not** materially enlarge the stable-positive family.

Across all audited seeds:

- total audited decisions: `13121`
- hard near-tie decisions: `2173`
- round-eleven stable-positive pack: `46`
- round-twelve `stable_positive_v2`: `46`
- round-twelve committee subset: `30`
- round-twelve strict subset: `30`

So the richer bank tightened the high-confidence subset, but it did not create
new cross-seed positive mass.

## Which teachers actually matter

The best-safe teacher table is decisive:

- `compute5` is the best safe teacher on `45 / 46` stable-positive-v2 cases
- `gated_pairwise` is the best safe teacher on `1 / 46`
- the other bank members mostly act as:
  - seed314-only support on already-positive cases
  - or negative / inert controls

This means the richer bank is mostly a **confidence filter** on top of
`compute5`, not a new source of independent corrections.

## Stability across seeds

Seed counts remained extremely concentrated:

- seed314: `42` stable-positive-v2 decisions
- seed315: `1`
- seed316: `3`

The overlap story stayed weak:

- fine-signature Jaccard is `0.0` for every seed pair
- coarse-signature Jaccard for v2 is only:
  - seed314 vs seed315: `0.0833`
  - seed314 vs seed316: `0.1667`
  - seed315 vs seed316: `0.0`
- committee / strict subsets fall back to `0.0` even at coarse signature level

So the richer bank did not solve the transfer problem from round eleven.

## Sensitivity

The v2 pack is robust to modest margin changes, but only in the sense that it
stays small:

- `min_regret_gain >= 0.10`: `46` v2 / `30` committee
- `min_regret_gain >= 0.25`: `46` v2 / `30` committee
- `min_regret_gain >= 0.50`: `45` v2 / `30` committee
- `min_regret_gain >= 0.75`: `41` v2 / `26` committee

So threshold tuning cannot rescue this round by itself.

## Decision

The richer teacher bank is useful, but only as a better offline labeling and
committee source.

What survives:

- `compute5` remains the only cross-seed teacher with real stable-positive mass
- committee support is useful as a precision filter
- the committee subset is a better **high-confidence upper bound** than the
  raw v2 pack

What does not survive:

- the thesis that richer bank membership would materially enlarge the canonical
  stable-positive family

## Artifacts

- `reports/plots/round12_teacher_bank_decisions.csv`
- `reports/plots/round12_teacher_bank_teachers.csv`
- `reports/plots/round12_teacher_bank_summary.csv`
- `reports/plots/round12_teacher_bank_teacher_summary.csv`
- `reports/plots/round12_teacher_bank_seed_summary.csv`
- `reports/plots/round12_teacher_bank_seed_overlap.csv`
- `reports/plots/round12_teacher_bank_sensitivity.csv`
- `reports/plots/round12_teacher_bank_stable_positive_v2_manifest.csv`
- `reports/plots/round12_teacher_bank_stable_positive_v2_committee_manifest.csv`
