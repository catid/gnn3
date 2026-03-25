# Round 9 Outer-Step Headroom

## Question

Does extra recursive compute help on the current frontier, and if so, how much
before the branch turns into pure cost or outright instability?

## Current Evidence

The fixed-compute sweep is now complete on matched seeds `314 / 315 / 316`.

### `multiheavy` baseline

- seed314: regret `1.898`, p95 `8.140`, miss `43.8%`
- seed315: regret `2.107`, p95 `6.164`, miss `56.2%`
- seed316: regret `1.508`, p95 `7.307`, miss `50.0%`

### `compute5`

- seed314: regret `1.302`, p95 `5.277`, miss `31.2%`
- seed315: regret `2.107`, p95 `6.164`, miss `56.2%`
- seed316: regret `1.508`, p95 `7.307`, miss `50.0%`

Interpretation:

- five outer refinement steps can help on at least one matched seed
- the effect did not replicate on the other two matched seeds
- fixed extra compute therefore remains a real headroom signal but not a robust
  deployment policy

### `compute7`

- seed314 epoch-1 rollout: regret `8893.68`, p95 `24061.70`, miss `100%`

Interpretation:

- simply pushing fixed outer compute deeper is not a free win
- the useful part of the compute thesis, if any, lives near the `compute5`
  scale, not in unbounded fixed-depth expansion

## Frontier Policy Read

Round-nine slice-only compute-policy audits on `deeper_packets6` sharpened the
same conclusion.

`fixed_final` did recover many baseline near-tie errors:

- baseline-error near-tie recovery: `59.09%`
- hard near-tie disagreement: `17.13%`

But it was still not a win overall on the frontier:

- hard near-tie target-match: `85.36%` vs baseline `86.29%`
- large-gap control target-match: `97.32%`

`fixed_middle` was far worse and catastrophically damaged large-gap controls.

## Final Conclusion

The round-nine extra-compute thesis is real but narrow:

- extra outer compute can improve some hard near-tie decisions
- the current fixed policies do not localize those good changes cleanly enough
- `compute7` is closed as a hard negative
- the next question is not whether more steps can ever help, but whether they
  can be compressed into a much more selective continuation rule without
  sacrificing the control slices
