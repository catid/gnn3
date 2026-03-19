# Portfolio Balance

| experiment_id | bucket | rationale | assigned_gpu_hours | actual_gpu_hours | status | key_result |
| --- | --- | --- | ---: | ---: | --- | --- |
| E1 | exploit | Local-only Packet-Mamba3 baseline is the minimum stable reference. | 0.50 | 0.0138 | completed | 94.5% next-hop accuracy, 100% rollout solved rate, regret 11.47. |
| E2 | exploit | Receiver-side selective reads are the safest communication upgrade to compare against local-only. | 0.75 | 0.0446 | completed | 3 short seeds; next-hop accuracy 94.2%-96.7%, regret 1.51-2.57. |
| E3 | exploit | Memory hubs plus detached warm-up test whether safer selective reads benefit from RSM-style refinement. | 0.75 | 0.1019 | completed | Best exploit result so far: 96.7% next-hop accuracy, 100% rollout solved rate, regret 0.037. |
| X1 | explore | Sender-side forwarding may help when monitor information must be pushed instead of pulled. | 1.25 | 0.1190 | completed | 6 short seeds; next-hop accuracy 93.2%-96.5%, regret 0.49-5.89, generally strong but more variable than E2. |
| X2 | explore | Forward plus read probes whether bidirectional routing improves over sender-only communication. | 0.75 | 0.0266 | completed | Negative result on first short run: 93.1% next-hop accuracy, regret 8.63, clearly worse than X1. |

Current cumulative GPU-hours:

- Exploit: `0.1603`
- Explore: `0.1455`
- Split: `52.4% / 47.6%`
