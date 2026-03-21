# Next Best Actions

1. Promote `X6` compressed outer-history summary-bank reads to the next contender batch and run a third seed plus a matched-seed `E3` compare. The first two seeds averaged 97.20% test next-hop accuracy and 1.60 regret, which is the best continuation signal on this branch.
2. Investigate same-config reproducibility drift in `E3` and `X1`. Both round-two replays were materially weaker than the archived best artifacts, so the next engineering pass should lock down data-loader RNG, deterministic settings, and checkpoint-selection sensitivity before any DDP scale-up claim.
3. Keep `detach_warmup` in the exploit path. The no-detach ablation collapsed immediately, with 42.9% val accuracy and zero solved rollouts after the first epoch.
4. Focus exploit-side work on regret and deadline robustness, not solved rate. `A1` showed that `E3` still solves the harder OOD suites, but cost calibration degrades sharply on deeper/heavier traffic.
5. Defer `X2`, `X3`, `X5`, and `H_test` scaling until either `E3` or `X6` has stable 3-seed evidence on the continuation branch.
