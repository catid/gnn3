#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("artifacts/experiments")
PLOTS = Path("reports/plots")
ABORTED_COMPARE_GPU_HOURS = 1312.0 / 3600.0


def _load_summary(experiment: str) -> dict[str, float]:
    summary = json.loads((ROOT / experiment / "summary.json").read_text(encoding="utf-8"))
    return {
        "experiment": experiment,
        "gpu_hours": float(summary["gpu_hours"]),
        "test_next_hop_accuracy": float(summary["test"]["next_hop_accuracy"]),
        "rollout_next_hop_accuracy": float(summary["test_rollout"]["next_hop_accuracy"]),
        "average_regret": float(summary["test_rollout"]["average_regret"]),
        "p95_regret": float(summary["test_rollout"]["p95_regret"]),
        "deadline_miss_rate": float(summary["test_rollout"]["deadline_miss_rate"]),
    }


def _plot_scout_compare(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    labels = df["variant"].tolist()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    axes[0].bar(labels, df["test_next_hop_accuracy"], color=colors[: len(df)])
    axes[0].set_title("Test Next-Hop Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, df["rollout_next_hop_accuracy"], color=colors[: len(df)])
    axes[1].set_title("Rollout Next-Hop Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(labels, df["average_regret"], color=colors[: len(df)])
    axes[2].set_title("Average Regret")
    axes[2].tick_params(axis="x", rotation=20)

    axes[3].bar(labels, df["deadline_miss_rate"], color=colors[: len(df)])
    axes[3].set_title("Deadline Miss Rate")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(PLOTS / "round6_scout_seed312_compare.png", dpi=160)
    plt.close(fig)


def _plot_policy_gate(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = df["branch"].tolist()

    axes[0].bar(labels, df["max_overall_disagreement"], color="#ff7f0e")
    axes[0].set_title("Max Overall Disagreement")
    axes[0].set_ylim(0.0, max(0.02, float(df["max_overall_disagreement"].max()) * 1.15))
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, df["max_hard_feasible_disagreement"], color="#d62728")
    axes[1].set_title("Max Hard-Feasible Disagreement")
    axes[1].set_ylim(0.0, max(0.01, float(df["max_hard_feasible_disagreement"].max()) * 1.15 + 0.001))
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(PLOTS / "round6_policy_movement_summary.png", dpi=160)
    plt.close(fig)


def _plot_portfolio(df: pd.DataFrame, totals: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    colors = ["#1f77b4" if bucket == "exploit" else "#ff7f0e" for bucket in df["bucket"]]

    axes[0].bar(df["experiment_id"], df["actual_gpu_hours"], color=colors)
    axes[0].set_title("Round 6 GPU-Hours by Task")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(totals["bucket"], totals["actual_gpu_hours"], color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Round 6 Exploit vs Explore")
    axes[1].set_ylim(0.0, max(1.05, float(totals["actual_gpu_hours"].max()) * 1.15))

    fig.tight_layout()
    fig.savefig(PLOTS / "round6_portfolio_usage.png", dpi=160)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    scout_df = pd.DataFrame(
        [
            {"variant": "Multiheavy", **_load_summary("e3_memory_hubs_rsm_round6_multiheavy_seed312")},
            {"variant": "RegimeExperts", **_load_summary("b1_multiheavy_regime_experts_round6_seed312")},
            {"variant": "Planner", **_load_summary("c1_multiheavy_planner_round6_seed312")},
            {"variant": "HazardMemory", **_load_summary("e1_multiheavy_hazard_memory_round6_seed312")},
        ]
    )
    scout_df.to_csv(PLOTS / "round6_scout_seed312_compare.csv", index=False)
    _plot_scout_compare(scout_df)

    movement_rows: list[dict[str, float | str]] = []
    for branch, filename in (
        ("RegimeExperts", "round6_regime_experts_seed312_policy_movement.csv"),
        ("Planner", "round6_planner_seed312_policy_movement.csv"),
    ):
        frame = pd.read_csv(PLOTS / filename)
        movement_rows.append(
            {
                "branch": branch,
                "max_overall_disagreement": float(frame["overall_disagreement"].max()),
                "max_hard_feasible_disagreement": float(frame["hard_feasible_disagreement"].max()),
                "baseline_suite_agreement": float(
                    frame.loc[
                        frame["suite"] == "e3_memory_hubs_rsm_round6_multiheavy_seed312",
                        "action_agreement",
                    ].iloc[0]
                ),
            }
        )
    movement_df = pd.DataFrame(movement_rows)
    movement_df.to_csv(PLOTS / "round6_policy_movement_summary.csv", index=False)
    _plot_policy_gate(movement_df)

    portfolio_df = pd.DataFrame(
        [
            {
                "experiment_id": "A1-R6-baseline",
                "bucket": "exploit",
                "actual_gpu_hours": 0.3604230656226476,
                "status": "completed",
                "key_result": "Fresh 3-seed multiheavy guardrail reproduction.",
            },
            {
                "experiment_id": "A2-R6-regime-audit",
                "bucket": "exploit",
                "actual_gpu_hours": 0.0,
                "status": "completed",
                "key_result": "Regime audit completed from the guardrail checkpoint.",
            },
            {
                "experiment_id": "B1-R6-regime-experts",
                "bucket": "explore",
                "actual_gpu_hours": 0.060636690391434565,
                "status": "killed-early",
                "key_result": "Shared-seed rollout matched baseline exactly and hard-feasible disagreement stayed at 0.",
            },
            {
                "experiment_id": "B1-R6-regime-compare",
                "bucket": "explore",
                "actual_gpu_hours": ABORTED_COMPARE_GPU_HOURS,
                "status": "stopped",
                "key_result": "Full-suite compare was aborted after 21m52s once zero hard-case movement was already established.",
            },
            {
                "experiment_id": "C1-R6-planner",
                "bucket": "explore",
                "actual_gpu_hours": 0.10101607700188954,
                "status": "killed-early",
                "key_result": "Slight seed312 gains, but 0 hard-feasible disagreement and catastrophic heavy_dynamic OOD.",
            },
            {
                "experiment_id": "E1-R6-hazard-memory",
                "bucket": "explore",
                "actual_gpu_hours": 0.10703421148988936,
                "status": "killed-early",
                "key_result": "Epoch1 collapsed, then selected test rollout snapped back to the baseline exactly.",
            },
            {
                "experiment_id": "E1-R6-hazard-compare",
                "bucket": "explore",
                "actual_gpu_hours": ABORTED_COMPARE_GPU_HOURS,
                "status": "stopped",
                "key_result": "Full-suite compare was aborted after 21m52s because the scout was already non-promotable.",
            },
            {
                "experiment_id": "D1-R6-repair",
                "bucket": "explore",
                "actual_gpu_hours": 0.0,
                "status": "scoped-out",
                "key_result": "Not opened because regime experts and planner failed the policy-movement gate.",
            },
        ]
    )
    totals = portfolio_df.groupby("bucket", as_index=False)["actual_gpu_hours"].sum()
    total_gpu_hours = float(totals["actual_gpu_hours"].sum())
    split = {
        row["bucket"]: float(row["actual_gpu_hours"]) / total_gpu_hours if total_gpu_hours > 0.0 else 0.0
        for _, row in totals.iterrows()
    }
    summary_row = pd.DataFrame(
        [
            {
                "experiment_id": "round6_total",
                "bucket": "summary",
                "actual_gpu_hours": total_gpu_hours,
                "status": "completed",
                "key_result": (
                    f"exploit={split.get('exploit', 0.0):.3f}, "
                    f"explore={split.get('explore', 0.0):.3f}"
                ),
            }
        ]
    )
    pd.concat([portfolio_df, summary_row], ignore_index=True).to_csv(PLOTS / "round6_portfolio_usage.csv", index=False)
    _plot_portfolio(portfolio_df, totals)


if __name__ == "__main__":
    main()
