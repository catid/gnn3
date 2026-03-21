# Portfolio Balance

| experiment_id | bucket | rationale | assigned_gpu_hours | actual_gpu_hours | status | key_result |
| --- | --- | --- | ---: | ---: | --- | --- |
| E1 | exploit | Local-only Packet-Mamba3 baseline is the minimum stable reference. | 0.50 | 0.0138 | completed | 94.5% next-hop accuracy, 100% rollout solved rate, regret 11.47. |
| E2 | exploit | Receiver-side selective reads are the safest communication upgrade to compare against local-only. | 0.75 | 0.0446 | completed | 3 short seeds; next-hop accuracy 94.2%-96.7%, regret 1.51-2.57. |
| E3 | exploit | Memory hubs plus detached warm-up test whether safer selective reads benefit from RSM-style refinement. | 1.25 | 0.3089 | completed | 3 short seeds; mean next-hop accuracy 97.1%, mean regret 0.72, best regret 0.037. |
| DDP-smoke | exploit | Verify the `torchrun`/DDP path before scaling a shortlisted model across both GPUs. | 0.10 | 0.0136 | completed | 2-GPU smoke run completed successfully after enabling unused-parameter detection for detached warm-up. |
| X1 | explore | Sender-side forwarding may help when monitor information must be pushed instead of pulled. | 1.25 | 0.1190 | completed | 6 short seeds; next-hop accuracy 93.2%-96.5%, regret 0.49-5.89, generally strong but more variable than E2. |
| X2 | explore | Forward plus read probes whether bidirectional routing improves over sender-only communication. | 0.75 | 0.0266 | completed | Negative result on first short run: 93.1% next-hop accuracy, regret 8.63, clearly worse than X1. |
| X4 | explore | Outer-round-history selective reads test JK-like retrieval across refinement depth. | 1.00 | 0.2196 | completed | 2 runs; mean next-hop accuracy 96.0%, mean regret 2.78, with one strong run at regret 0.48 and one weak run at 5.07. |
| E3-R2-repro | exploit | Reproduce the archived `E3` winner on the continuation branch before scaling or ablating it further. | 0.15 | 0.1111 | completed | Same config family reran weaker than the archived best: 95.66% test next-hop accuracy, 3.07 regret. |
| A1 | exploit | Stress the current `E3` checkpoint on larger and heavier OOD settings before spending more training budget. | 0.08 | 0.0547 | completed | All three OOD sweeps stayed at 100% solved rate, but regret rose to 4.83-9.01 and deadline violations to 4.13-4.81. |
| A2 | exploit | Check whether detached warm-up is actually helping by disabling it in the current `E3` path. | 0.15 | 0.0676 | killed-early | Early negative result: after epoch 1, val next-hop accuracy was 42.9%, rollout solved rate was 0%, and regret was 6097.15. |
| X1-R2-repro | explore | Reproduce the archived `X1` baseline on the continuation branch before promoting alternatives. | 0.05 | 0.0150 | completed | Weak replay: 93.81% test next-hop accuracy and 3.16 regret, consistent with prior variance concerns. |
| X6-scout | explore | Test compressed outer-history summary-bank reads on top of the `E3` memory-hub backbone. | 0.15 | 0.1026 | completed | First scout was promising: 96.97% test next-hop accuracy and 2.68 regret. |
| X6-seed212 | explore | Verify that the `X6` history-bank signal survives a second seed before promotion. | 0.15 | 0.1235 | completed | Strong replication: 97.44% test next-hop accuracy and 0.53 regret. |

Current cumulative GPU-hours:

- Exploit: `0.6143`
- Explore: `0.6063`
- Split: `50.3% / 49.7%`
