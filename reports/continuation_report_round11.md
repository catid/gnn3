# Round 11 Continuation Report

## Objective

Round eleven tested the next narrowed thesis from round ten:

> The remaining headroom is concentrated in a narrow, high-value subset of hard
> near-tie states. Can a very high-precision defer/correct policy recover that
> headroom without broad regressions or large runtime cost?

This was a precision-first correction campaign, not a broad model-family sweep.

## Accepted baseline

`multiheavy` remains the default exploit policy.

The accepted anchor is still the corrected round-four comparison:

- mean regret: `1.32` vs fresh `E3` `2.25`
- mean p95 regret: `4.77` vs `10.48`
- mean deadline miss: `41.7%` vs `54.2%`

Rounds six through eleven did not dislodge that baseline.

## Round 11 findings

### 1. The stable-positive correction subset is real, but extremely sparse

Round eleven’s new Tier-1 surface was the stable-positive correction pack.

Teacher-bank audit:

- hard near-tie decisions: `2173`
- stable-positive pack decisions: `46`
- stable-positive share of hard near-tie: `2.12%`
- harmful teacher share of hard near-tie: `3.22%`

Per-case value inside the pack is strong:

- target match: `0.0% -> 100.0%`
- mean teacher regret gain: `2.7310`
- mean miss gain: `0.0435`

But transfer is weak:

- seed314 stable-positive decisions: `42`
- seed315 stable-positive decisions: `1`
- seed316 stable-positive decisions: `3`
- source-signature overlap Jaccard across seed pairs: `0.0`

So the correction opportunity is real but much more fragile than the broad
round-ten high-headroom story suggested.

### 2. Broad learned defer gates failed; only the simple margin/regime defer survived

The defer-gate sweep tested:

- `linear`
- `mlp`
- `margin_regime`

Held-out verdict:

- `linear`: dead
- `mlp`: dead
- `margin_regime`: only surviving branch

Best held-out operating points for `margin_regime`:

At `1%` nominal budget:

- stable-positive recovery: `50%`
- stable-positive precision: `100%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0134`

At `2%` nominal budget:

- stable-positive recovery: `75%`
- stable-positive precision: `100%`
- hard near-tie target match: `90.53% -> 90.73%`
- hard near-tie mean delta regret: `-0.0089`
- overall mean delta regret: `-0.0151`

At `5%`, the gate stopped being safe and the hard near-tie gain disappeared.

So the only live defer story is a tiny-coverage handoff to the teacher, not a
general learned helpfulness gate.

### 3. Top-2 comparator with abstain did not work

The comparator family was tested in both frozen and candidate-conditioned forms,
with broad and narrow training scopes.

Result:

- frozen variants were essentially policy-identical
- candidate-conditioned narrow variants were actively harmful

On held-out seed316, `candidate_conditioned:narrow` regressed badly:

- hard near-tie target match fell to `86.15%` at `1%` budget
- hard near-tie target match fell to `77.32%` at `5%`
- large-gap control also regressed

Decision: comparator family closed.

### 4. Subset-only distillation still did not produce a deployable student

Subset-only student families:

- `pairwise`
- `kl`
- `residual`
- `gated_pairwise`

The failure pattern remained familiar:

- `residual` was aggressively destructive
- `pairwise` and `kl` recovered some positive states but regressed solved slices
- `gated_pairwise` stayed safest, but still missed the real gate

Aggregate held-out `gated_pairwise`:

- overall target match: `96.51% -> 96.60%`
- large-gap control target match: `99.79% -> 99.90%`
- stable-positive pack target match: `50%`
- hard near-tie target match: `90.53% -> 90.46%`
- hard near-tie mean delta regret: `+0.0036`

So even the safest subset-only student was still not a winner.

### 5. A tiny deployment-time defer system exists, but it is not strong enough to promote

Because `margin_regime` survived held-out gating, round eleven opened a narrow
deployment study.

Offline deployment estimate:

- baseline `multiheavy` everywhere
- defer to `compute5` teacher only on gate-positive states
- baseline steps `3`, deferred teacher steps `5`

Best budget was around `2%`:

- average outer steps: `3.0401`
- compute multiplier: `1.0134x`
- stable-positive recovery: `75%`
- hard near-tie mean delta regret: `-0.0089`

This is a real precision-first operating point, but it is still too small and
too dependent on a tiny stable-positive pack to justify changing the default
deployment policy.

## Promotion verdict

No round-eleven branch earned contender status.

Closed or not promoted:

- broad learned defer gates
- top-2 comparator with abstain
- subset-only students
- deployment of the defer-to-teacher system as a new default

Best surviving reference:

- `margin_regime` defer at `1–2%` coverage is the only branch worth remembering
  from round eleven, but only as a tiny deployment reference, not a promoted
  policy.

## Updated recommendation

Keep plain `multiheavy`.

If another round opens, it should be even narrower than round eleven:

- richer teacher-bank construction before new student capacity
- explicit stable-positive source-family enlargement
- ultra-low-coverage defer policies only
- hard false-positive penalties and large-gap preservation

Do not reopen:

- broad defer gates
- top-2 comparator family
- subset-only distillation as currently implemented
- broad extra-compute or search-style correction

## Round outputs

- `reports/continuation_audit_round11.md`
- `reports/teacher_bank_round11.md`
- `reports/stable_positive_pack_round11.md`
- `reports/defer_gate_round11.md`
- `reports/top2_comparator_round11.md`
- `reports/subset_distill_round11.md`
- `reports/deployment_study_round11.md`

Key artifacts:

- `reports/plots/round11_teacher_bank_summary.csv`
- `reports/plots/round11_teacher_bank_stable_positive_manifest.csv`
- `reports/plots/round11_defer_gate_heldout_summary.csv`
- `reports/plots/round11_top2_comparator_heldout_summary.csv`
- `reports/plots/round11_subset_distill_heldout_summary.csv`
- `reports/plots/round11_deployment_study_summary.csv`

## Merge / push status

Round eleven is committed locally on `main`.

- local round-eleven commit: `f202b30`
- the oversized `reports/plots/round11_feature_cache_seed*.pt` files were removed
  from local git history after GitHub rejected the first push
- `.gitignore` now excludes:
  - `reports/plots/*feature_cache*.pt`
  - `artifacts/experiments/**/checkpoints/`

Validation on local `main` passed:

- `uv run ruff check src tests scripts`
- `uv run pytest tests/test_precision_correction.py tests/test_compute_helpfulness.py tests/test_policy_analysis.py tests/test_step_policy.py -q`
- `uv run python scripts/run_train.py --config configs/experiments/smoke_local_cpu.yaml`

Remote sync and push are still blocked in this shell:

- `git pull --ff-only origin main` -> `git@github.com: Permission denied (publickey)`
- `bd sync` -> failed on remote pull with the same SSH error
- `git push origin main` -> `git@github.com: Permission denied (publickey)`

So the round is complete locally on `main`, the large-file history problem is
fixed, and publishing now only requires a shell with GitHub write access.
