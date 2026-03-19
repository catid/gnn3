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

Current cumulative GPU-hours:

- Exploit: `0.3809`
- Explore: `0.3651`
- Split: `51.1% / 48.9%`
