# Continuation Report Round 2

## Scope

This round continued the existing repo rather than rebuilding anything. Work stayed additive and repo-first:

- audited the actual code and wrote [continuation_audit.md](/home/catid/gnn3/reports/continuation_audit.md)
- reran the fast green stack on the continuation branch
- added provenance fields for branch and run stage to experiment outputs
- reproduced one exploit baseline and one explore baseline on the live branch
- ran one exploit batch and one explore batch with real artifacts
- kept the cumulative exploit/explore balance at `50.3% / 49.7%`

## Validation And Infra

Green checks on `cont/e3-history-read-x1-rescue`:

- `ruff check src tests scripts`
- `pytest tests/test_hidden_corridor.py -q`
- `python scripts/run_train.py --config configs/experiments/smoke_local_cpu.yaml`

Small infra additions:

- branch/stage/notes/device placement are now written into `metadata.json` and `summary.json`
- [run_eval_sweep.py](/home/catid/gnn3/scripts/run_eval_sweep.py) evaluates one checkpoint across multiple OOD configs and writes CSV/JSON/PNG artifacts
- the new exploration mechanism is a compressed outer-history `summary_bank` mode in [packet_mamba.py](/home/catid/gnn3/src/gnn3/models/packet_mamba.py)

## Reproductions

### Exploit Replay

- `e3_memory_hubs_rsm_round2_repro`
- test next-hop accuracy: `95.66%`
- rollout next-hop accuracy: `93.90%`
- average regret: `3.07`

Interpretation:

- the archived `E3` winner did not reproduce at the same quality on the continuation branch
- this is the main exploit-side risk now: the prior best artifact was real, but the direction is less repeatable than it looked

### Explore Replay

- `x1_selective_forward_round2_repro`
- test next-hop accuracy: `93.81%`
- rollout next-hop accuracy: `94.29%`
- average regret: `3.16`

Interpretation:

- `X1` still works, but the replay again showed the same variance problem seen in round one

## Exploit Batch

### A1: E3 OOD Stress

Artifacts:

- [e3_ood_stress_round2.csv](/home/catid/gnn3/reports/plots/e3_ood_stress_round2.csv)
- [e3_ood_stress_round2.png](/home/catid/gnn3/reports/plots/e3_ood_stress_round2.png)

Results:

- branching-factor-3 / packet-6 stress:
  - next-hop accuracy `95.04%`
  - solved rate `100%`
  - regret `4.83`
  - deadline violations `4.13`
- deeper-tree / packet-6 stress:
  - next-hop accuracy `96.48%`
  - solved rate `100%`
  - regret `8.94`
  - deadline violations `4.44`
- heavy-dynamic / packet-8 stress:
  - next-hop accuracy `95.86%`
  - solved rate `100%`
  - regret `9.01`
  - deadline violations `4.81`

Take:

- `E3` does not break first on solvability
- it breaks first on cost calibration and deadline behavior as graph depth, packet count, and congestion pressure increase

### A2: No-Detach Ablation

- `a2_e3_no_detach_round2_scout`
- killed early after epoch 1
- val next-hop accuracy: `42.9%`
- rollout solved rate: `0%`
- average regret: `6097.15`

Take:

- detached warm-up is not cosmetic in the current exploit path; removing it causes an immediate training collapse

## Explore Batch

### B1: Compressed Outer-History Summary Bank

Artifacts:

- [x6_e3_history_summary_bank_round2_scout/summary.json](/home/catid/gnn3/artifacts/experiments/x6_e3_history_summary_bank_round2_scout/summary.json)
- [x6_e3_history_summary_bank_round2_seed212/summary.json](/home/catid/gnn3/artifacts/experiments/x6_e3_history_summary_bank_round2_seed212/summary.json)

Results:

- seed 211:
  - test next-hop accuracy `96.97%`
  - rollout next-hop accuracy `95.45%`
  - average regret `2.68`
- seed 212:
  - test next-hop accuracy `97.44%`
  - rollout next-hop accuracy `96.97%`
  - average regret `0.53`

Two-seed scout average:

- test next-hop accuracy `97.20%`
- rollout next-hop accuracy `96.21%`
- average regret `1.60`

Take:

- this is the strongest new direction from the continuation session
- it outperformed the round-two `X1` replay and also beat the round-two `E3` replay on this branch
- it is still only a 2-seed exploration result, so it should be promoted to a 3-seed contender rather than declared the new winner

## Portfolio

Artifacts:

- [portfolio_balance.md](/home/catid/gnn3/reports/portfolio_balance.md)
- [portfolio_usage_round2.csv](/home/catid/gnn3/reports/plots/portfolio_usage_round2.csv)
- [portfolio_usage_round2.png](/home/catid/gnn3/reports/plots/portfolio_usage_round2.png)

Round-two cumulative totals:

- exploit GPU-hours: `0.6143`
- explore GPU-hours: `0.6063`
- split: `50.3% / 49.7%`

## Recommendation

1. Promote `X6` to a 3-seed contender against a matched-seed `E3` baseline.
2. Keep `detach_warmup` in every exploit-side shortlist variant.
3. Treat exploit-side OOD work as a regret/deadline problem, not a solved-rate problem.
4. Do not scale with DDP yet; same-config reproducibility needs to be tightened first.
5. Defer `H_test` scaling and broader exploration until the branch has one stable 3-seed winner on current code.
