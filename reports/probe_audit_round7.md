# Probe Audit Round 7

## Scope

- Frozen backbone:
  - [e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml)
  - [best.pt](/home/catid/gnn3/artifacts/experiments/e3_memory_hubs_rsm_round7_multiheavy_seed312/checkpoints/best.pt)
- Evaluated suites:
  - [e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml](/home/catid/gnn3/configs/experiments/e3_memory_hubs_rsm_round7_multiheavy_seed312.yaml)
  - [a1_multiheavy_ood_branching3_round7_eval.yaml](/home/catid/gnn3/configs/experiments/a1_multiheavy_ood_branching3_round7_eval.yaml)
  - [a1_multiheavy_ood_deeper_packets6_round7_eval.yaml](/home/catid/gnn3/configs/experiments/a1_multiheavy_ood_deeper_packets6_round7_eval.yaml)
  - [a1_multiheavy_ood_heavy_dynamic_round7_eval.yaml](/home/catid/gnn3/configs/experiments/a1_multiheavy_ood_heavy_dynamic_round7_eval.yaml)
- Artifacts:
  - [round7_probe_audit_summary.csv](/home/catid/gnn3/reports/plots/round7_probe_audit_summary.csv)
  - [round7_probe_audit_summary.png](/home/catid/gnn3/reports/plots/round7_probe_audit_summary.png)
  - [round7_probe_audit.json](/home/catid/gnn3/reports/plots/round7_probe_audit.json)

## Main Finding

The frozen `multiheavy` backbone already linearly encodes most of the signals that round seven cared about.

Strong probe results:

- slack bucket:
  - base eval `0.897`
  - OOD `0.887 / 0.885 / 0.869`
- critical packet proxy:
  - base eval `0.966`
  - OOD `0.968 / 0.978 / 0.972`
- feasible continuation:
  - base eval `1.000`
  - OOD `1.000 / 0.999 / 1.000`
- baseline strictly suboptimal:
  - base eval `0.948`
  - OOD `0.958 / 0.973 / 0.967`

More moderate but still meaningful:

- oracle gap bucket:
  - base eval `0.771`
  - OOD `0.727 / 0.649 / 0.695`

Weakest result:

- depth/load regime:
  - base eval `0.681`
  - OOD `0.000 / 0.000 / 0.310`

That pattern matters. The backbone is not missing the basic local decision signals. It already exposes:

- deadline/slack pressure
- critical-packet identity
- continuation feasibility
- whether the baseline action is strictly suboptimal

So round seven does **not** support a simple “the encoder cannot see the right information” story.

## What The Probe Says About The Plateau

The strongest interpretation is:

- this is primarily a **constructor bottleneck**
- not a missing-signal bottleneck for local feasibility/slack/suboptimality

The probe can recover the baseline’s own failure indicator at `0.95+` OOD accuracy, which is especially important. That means the shared representation already contains substantial information about whether the current action is wrong, yet the round-seven constructor branches still failed to turn that into a better policy.

The oracle gap bucket probe is weaker but still useful:

- it drops to `0.649` on `deeper_packets6`
- it stays well above chance on every audited suite

That suggests the backbone has some margin information, but not perfectly. The remaining constructor opportunity, if any, is likely in how that uncertainty is turned into a decision on near-tie states, not in basic route-feasibility awareness.

## One Caveat: Regime Token Generalization

The depth/load regime probe did not generalize well:

- `0.0` on `branching3`
- `0.0` on `deeper_packets6`
- only `0.31` on `heavy_dynamic`

That does **not** overturn the main conclusion, because the other probes remain strong. But it does show one specific weakness:

- the backbone does not cleanly encode a reusable explicit regime code across OOD suite shifts

Round-six regime prompts and small regime experts already failed to turn that into a better policy, so this is not a license to reopen the old expert family. It is only a warning that explicit regime-ID conditioning is still weak.

## Recommendation

- Read round seven as evidence that the current plateau is mainly a constructor problem.
- Do not open another round built around “just expose slack/feasibility signals better.” The backbone already does that.
- If a future architectural round opens, it should target:
  - hard near-tie ambiguity resolution
  - structured credit assignment for close action alternatives
  - constructor mechanisms that can exploit already-present signals instead of merely re-encoding them
