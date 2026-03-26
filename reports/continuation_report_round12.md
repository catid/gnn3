# Round 12 Continuation Report

## Objective

Round twelve tested the next narrowed thesis from round eleven:

> Can a richer offline teacher bank define a stronger stable-positive
> correction family, and can an ultra-low-coverage defer/correct policy exploit
> that family without broad regressions?

This was a teacher-bank and precision-first defer campaign, not a broad model
family search.

## Accepted baseline

`multiheavy` remains the default exploit policy.

The accepted anchor is still the corrected round-four comparison:

- mean regret: `1.32` vs fresh `E3` `2.25`
- mean p95 regret: `4.77` vs `10.48`
- mean deadline miss: `41.7%` vs `54.2%`

Rounds six through twelve did not dislodge that baseline.

## Round 12 findings

### 1. Richer teacher-bank construction improved confidence, not coverage

Round twelve expanded the offline teacher bank with:

- round-ten `compute5`
- triggered selective-compute policies
- round-eleven subset-only students
- seed314 non-catastrophic round-nine compute-policy auxiliaries

The result did **not** enlarge the canonical stable-positive family:

- hard near-tie decisions: `2173`
- round-eleven stable-positive pack: `46`
- round-twelve `stable_positive_v2`: `46`
- round-twelve committee subset: `30`
- held-out stable-positive-v2 decisions on seeds `315 / 316`: `4`

Teacher identity also stayed extremely concentrated:

- `compute5` is the best safe teacher on `45 / 46` v2 cases
- `gated_pairwise` accounts for the remaining `1 / 46`

So the richer bank mostly sharpened the confidence filter around the existing
teacher; it did not create a new broad correction family.

### 2. Stable-positive-v2 is still real, but still fragile

The pack remains narrow and poorly transferable:

- seed314 v2 count: `42`
- seed315 v2 count: `1`
- seed316 v2 count: `3`

Overlap stayed weak:

- fine-signature Jaccard is `0.0` for every seed pair
- coarse-signature Jaccard is only `0.0833`, `0.1667`, and `0.0`
- committee / strict subsets fall back to `0.0` overlap even at coarse level

So round twelve did not solve the source-family transfer problem from round
eleven.

### 3. Ultra-low learned defer still reduces to the simple margin/regime rule

Round-twelve defer families:

- `linear`
- `mlp`
- `margin_regime`

Held-out verdict:

- `linear`: dead
- `mlp`: dead
- `margin_regime`: only surviving learned gate

Best round-twelve learned operating point:

- `margin_regime @ 0.50%`
- stable-positive-v2 recovery: `50%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0071`
- overall mean delta regret: `-0.0075`

This is real, but it does not beat the round-eleven reference, which still
delivers stronger held-out overall and hard-slice deltas at `1–2%` coverage.

### 4. Retrieval / prototype defer is closed

Frozen-feature retrieval was tested via:

- `knn_v2`
- `prototype_committee`
- `margin_retrieval`

All variants were effectively dead:

- stable-positive-v2 recall stayed `0.0`
- hard near-tie mean delta regret stayed `0.0`
- overall mean delta regret stayed `0.0`

So the sparse positive family does not behave like a clean local cluster in the
current frozen feature space.

### 5. Committee defer is the strongest round-twelve branch, but only as an offline upper bound

On the full audited bank, committee defer looked strong:

- `committee_only @ 0.50%`:
  - stable-positive-v2 recovery: `65.2%`
  - hard near-tie mean delta regret: `-0.0347`
  - overall mean delta regret: `-0.0159`

But the honest held-out panel is smaller:

- best held-out operating point is effectively `committee_only @ 0.50%` or
  `margin_committee @ 0.75%`
- overall coverage: about `0.25%`
- stable-positive-v2 recovery: `50%`
- stable-positive precision: `100%`
- hard near-tie target match: `90.53% -> 90.66%`
- hard near-tie mean delta regret: `-0.0029`
- overall mean delta regret: about `-0.0089`

This is the cleanest low-coverage result of the round, but it still does not
beat the round-eleven reference on the full held-out hard near-tie frontier,
and it is not deployable because it requires per-state teacher-bank knowledge.

### 6. Positive mining does not justify reopening student compression

Round twelve also tested whether training-only mining could enlarge the source
family enough to justify a conservative student retry.

Result:

- fine signature mining: `0` held-out recall
- best coarse mining (`high_headroom_union`): `100%` held-out recall, but only
  `5.97%` precision at `0.77%` coverage
- broader coarse mining: `1.77%` to `2.94%` precision
- regime-signature mining: `0.26%` to `0.44%` precision and nonzero harmful
  selection

So mining can recover the positives only by selecting a much larger mostly
neutral set. That is not enough to justify reopening student compression.

## Promotion verdict

No round-twelve branch earned contender status.

Closed or not promoted:

- richer teacher-bank expansion as a source-family growth mechanism
- ultra-low learned defer beyond the existing `margin_regime` reference
- retrieval / prototype defer
- committee defer as a deployed policy
- conservative student retry

Best surviving round-twelve result:

- committee-backed defer is the best new **offline upper bound**
- but the round-eleven `margin_regime` reference remains the best remembered
  deployment-style operating point

## Updated recommendation

Keep plain `multiheavy`.

Keep the round-nine/ten near-tie frontier pack as the canonical Tier-2 surface.

Keep `stable_positive_v2` only as a narrow Tier-1 diagnostic surface.

Remember:

- round-eleven `margin_regime` at `1–2%` coverage as the current reference
- round-twelve committee defer only as a bank-backed upper bound

Do not reopen:

- broad learned defer gates
- retrieval / prototype defer
- subset/student compression for the current source family
- broad extra-compute families
- older reranker / planner families

If another round opens, it should be even narrower:

- richer teachers first, not bigger students
- teacher-bank agreement and correction margin before any new gate
- explicit held-out comparison against the round-eleven reference at matched or
  lower coverage

## Round outputs

- `reports/continuation_audit_round12.md`
- `reports/teacher_bank_round12.md`
- `reports/stable_positive_pack_round12.md`
- `reports/ultralow_defer_round12.md`
- `reports/retrieval_defer_round12.md`
- `reports/committee_defer_round12.md`
- `reports/positive_mining_round12.md`
- `reports/deployment_study_round12.md`

Key artifacts:

- `reports/plots/round12_teacher_bank_summary.csv`
- `reports/plots/round12_teacher_bank_stable_positive_v2_manifest.csv`
- `reports/plots/round12_ultralow_defer_summary.csv`
- `reports/plots/round12_retrieval_defer_summary.csv`
- `reports/plots/round12_committee_defer_summary.csv`
- `reports/plots/round12_positive_mining_summary.csv`
- `reports/plots/round12_deployment_study_summary.csv`

## Merge / push status

Round twelve is merged into local `main` at commit
`8b68fa47c97ba2e9d46c3192889e079f82bf8a54` (`round12: teacher-bank ultralow
defer experiments`).

Post-merge validation on `main` passed:

- `uv run ruff check src tests scripts`
- `uv run pytest tests/test_precision_correction.py tests/test_compute_helpfulness.py tests/test_policy_analysis.py tests/test_step_policy.py -q`
- `uv run python scripts/run_train.py --config configs/experiments/smoke_local_cpu.yaml`

Remote landing is still blocked by SSH auth in this shell:

- `git pull --ff-only origin main`
  - `git@github.com: Permission denied (publickey).`
- `bd sync`
  - `pulling: git pull failed: exit status 1`
  - `git@github.com: Permission denied (publickey).`
- `git push origin main`
  - `git@github.com: Permission denied (publickey).`

The repo is clean locally on `main` and ready to push from a shell with GitHub
write access.
