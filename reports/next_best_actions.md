# Next Best Actions

1. Keep `E3` as the lead baseline. Do not promote full `X6` from round three: across the corrected three-seed matched batch it gained only `+0.25` percentage points in mean test next-hop accuracy, with identical mean regret, identical p95 regret, identical rollout accuracy, and identical deadline misses.
2. Keep the split-manifest discipline from round three. Future runs should retain split-specific dataset seeds and persisted manifests; the earlier val/test aliasing problem materially changed the interpretation of the older results.
3. Keep `detach_warmup` in every exploit-side shortlist. That remains the strongest causal architectural result.
4. Treat deadline behavior on the corrected split as the primary blocker. Every matched contender and every scout in round three still had `100%` deadline miss rate, so benchmark/training-contract work now has higher leverage than more communication variants.
5. If history reads are revisited, use the cheapest controls only. `X6 H1` and dense-history reads matched the full `X6` rollout regime on seed `211`, so the full summary-bank mechanism is not justified for this benchmark.
6. If there is one exploit-side continuation bet worth another short batch, it is heavier multi-packet curriculum pressure. Run at least 2 more seeds, compare against matched `E3` on both in-distribution and OOD p95 regret, and stop immediately if the p95 / deadline metrics stay flat.
7. Defer broader exploration (`X2`, `X3`, `X5`, `H_test` scaling) until the benchmark’s deadline regime is better calibrated and one exploit path can beat the current `E3` baseline on rollout quality, not just decision accuracy.
