# Round 9 Outer-Step Headroom

## Question

Does extra recursive compute help on the current frontier, and if so, how much
before the branch turns into pure cost or outright instability?

## Current Evidence

The first fixed-compute sweep already gives a strong partial answer.

### `multiheavy` baseline

- seed314: regret `1.898`, p95 `8.140`, miss `43.8%`
- seed315: regret `2.107`, p95 `6.164`, miss `56.2%`

### `compute5`

- seed314: regret `1.302`, p95 `5.277`, miss `31.2%`
- seed315: regret `2.107`, p95 `6.164`, miss `56.2%`

Interpretation:

- five outer refinement steps can help on at least one matched seed
- the effect is not yet robust enough to promote on two seeds
- the family needs the third matched seed to decide whether the gain is real or
  just variance

### `compute7`

- seed314 epoch-1 rollout: regret `8893.68`, p95 `24061.70`, miss `100%`

Interpretation:

- simply pushing fixed outer compute deeper is not a free win
- the useful part of the compute thesis, if any, lives near the `compute5`
  scale, not in unbounded fixed-depth expansion

## Interim Conclusion

The round-nine extra-compute thesis is still alive, but only in a narrow form:

- `compute5` remains the active family
- `compute7` is closed as a hard negative
- the next decision depends on the matched seed316 pair

If seed316 confirms the seed314 gain, the family is worth promoting to the
frontier-guard and bounded conditional-compute comparisons.

If seed316 stays flat like seed315, fixed extra compute should be treated as
high-variance and not promoted as the main path.
