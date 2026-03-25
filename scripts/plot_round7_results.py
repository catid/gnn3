#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("artifacts/experiments")
PLOTS = Path("reports/plots")
LOGS = Path("artifacts/round7_logs")

TRAIN_ROWS = [
    ("A1-R7-baseline", "exploit", "Multiheavy", "e3_memory_hubs_rsm_round7_multiheavy_seed312"),
    ("B1-R7-poly-constructor", "explore", "PolyConstructor", "b1_multiheavy_poly_constructor_round7_seed312"),
    ("B2-R7-self-improve", "explore", "SelfImprove", "b2_multiheavy_self_improve_round7_seed312"),
    ("B3-R7-teacher-heavy", "explore", "TeacherHeavy", "b3_multiheavy_teacher_heavy_packets6_round7_seed312"),
    ("B3-R7-teacher-tight", "explore", "TeacherTight", "b3_multiheavy_teacher_tightslack_round7_seed312"),
    ("B3-R7-teacher-depth4", "explore", "TeacherDepth4", "b3_multiheavy_teacher_depth4_round7_seed312"),
]

COMPARE_ROWS = [
    ("SelfImprove", "round7_self_improve_hardslice_vs_multiheavy", "r7_hardslice_selfimprove.time", "explore"),
    ("TeacherHeavy", "round7_teacher_heavy_hardslice_vs_multiheavy", "r7_hardslice_teacher_heavy.time", "explore"),
    ("TeacherTight", "round7_teacher_tight_hardslice_vs_multiheavy", "r7_hardslice_teacher_tight.time", "explore"),
    ("TeacherDepth4", "round7_teacher_depth4_hardslice_vs_multiheavy", "r7_hardslice_teacher_depth4.time", "explore"),
]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_best_metrics(experiment: str) -> dict[str, float] | None:
    rows = _read_jsonl(ROOT / experiment / "metrics.jsonl")
    if not rows:
        return None
    best_row = max(rows, key=lambda row: (float(row["selection_score"]), -int(row["epoch"])))
    last_row = rows[-1]
    val = best_row["val"]
    rollout = best_row["rollout"]
    return {
        "experiment": experiment,
        "gpu_hours": float(last_row["elapsed_seconds"]) / 3600.0,
        "selection_score": float(best_row["selection_score"]),
        "next_hop_accuracy": float(val["next_hop_accuracy"]),
        "rollout_next_hop_accuracy": float(rollout["next_hop_accuracy"]),
        "average_regret": float(rollout["average_regret"]),
        "p95_regret": float(rollout["p95_regret"]),
        "deadline_miss_rate": float(rollout["deadline_miss_rate"]),
    }


def _time_file_hours(log_name: str | None) -> float:
    if not log_name:
        return 0.0
    log_path = LOGS / log_name
    if not log_path.exists():
        return 0.0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("real "):
            return float(line.split()[1]) / 3600.0
    return 0.0


def _load_compare(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    summary_path = PLOTS / f"{prefix}_summary.csv"
    decisions_path = PLOTS / f"{prefix}_decisions.csv"
    if not summary_path.exists() or not decisions_path.exists():
        return None
    return (
        pd.read_csv(summary_path),
        pd.read_csv(decisions_path),
    )


def _decision_metrics(summary_df: pd.DataFrame) -> dict[str, float]:
    if summary_df.empty:
        return {
            "candidate_target_match": 0.0,
            "base_target_match": 0.0,
            "overall_disagreement": 0.0,
            "hard_feasible_disagreement": 0.0,
            "large_gap_hard_feasible_disagreement": 0.0,
        }
    weights = summary_df["decisions"].clip(lower=1.0)
    return {
        "candidate_target_match": float((summary_df["candidate_target_match"] * weights).sum() / weights.sum()),
        "base_target_match": float((summary_df["base_target_match"] * weights).sum() / weights.sum()),
        "overall_disagreement": float((summary_df["overall_disagreement"] * weights).sum() / weights.sum()),
        "hard_feasible_disagreement": float((summary_df["hard_feasible_disagreement"] * weights).sum() / weights.sum()),
        "large_gap_hard_feasible_disagreement": float(
            (summary_df["large_gap_hard_feasible_disagreement"] * weights).sum() / weights.sum()
        ),
    }


def _hard_slice_metrics(decisions: pd.DataFrame) -> dict[str, float]:
    hard_df = decisions.loc[decisions["hard_feasible_case"]].copy()
    large_df = decisions.loc[decisions["large_gap_hard_feasible_case"]].copy()
    if hard_df.empty:
        return {
            "hard_slice_decisions": 0.0,
            "large_gap_hard_slice_decisions": 0.0,
            "hard_slice_candidate_target_match": 0.0,
            "hard_slice_base_target_match": 0.0,
            "large_gap_candidate_target_match": 0.0,
            "large_gap_base_target_match": 0.0,
        }
    return {
        "hard_slice_decisions": float(len(hard_df)),
        "large_gap_hard_slice_decisions": float(len(large_df)),
        "hard_slice_candidate_target_match": float(hard_df["candidate_target_match"].mean()),
        "hard_slice_base_target_match": float(hard_df["base_target_match"].mean()),
        "large_gap_candidate_target_match": float(large_df["candidate_target_match"].mean()) if len(large_df) else 0.0,
        "large_gap_base_target_match": float(large_df["base_target_match"].mean()) if len(large_df) else 0.0,
    }


def _plot_scout_compare(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    labels = df["variant"].tolist()
    colors = ["#1f77b4" if variant == "Multiheavy" else "#ff7f0e" for variant in labels]

    axes[0].bar(labels, df["overall_disagreement"], color=colors)
    axes[0].set_title("Hard-Slice Overall Disagreement")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(labels, df["hard_feasible_disagreement"], color=colors)
    axes[1].set_title("Hard-Feasible Disagreement")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(labels, df["large_gap_hard_feasible_disagreement"], color=colors)
    axes[2].set_title("Large-Gap Hard-Feasible Disagreement")
    axes[2].tick_params(axis="x", rotation=25)

    axes[3].bar(labels, df["target_match_delta"], color=colors)
    axes[3].axhline(0.0, color="black", linewidth=1.0)
    axes[3].set_title("Target-Match Delta")
    axes[3].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(PLOTS / "round7_scout_seed312_compare.png", dpi=160)
    plt.close(fig)


def _plot_policy_movement(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = df["branch"].tolist()

    axes[0].bar(labels, df["max_overall_disagreement"], color="#ff7f0e")
    axes[0].set_title("Max Overall Disagreement")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(labels, df["max_large_gap_hard_feasible_disagreement"], color="#d62728")
    axes[1].set_title("Max Large-Gap Hard-Feasible Disagreement")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(
        labels,
        df["hard_slice_candidate_target_match"] - df["hard_slice_base_target_match"],
        color="#2ca02c",
    )
    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[2].set_title("Hard-Slice Target-Match Delta")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(PLOTS / "round7_policy_movement_summary.png", dpi=160)
    plt.close(fig)


def _plot_hard_slice(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = df["branch"].tolist()

    axes[0].bar(
        labels,
        df["hard_slice_candidate_target_match"] - df["hard_slice_base_target_match"],
        color="#1f77b4",
    )
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_title("Hard-Slice Target-Match Delta")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(
        labels,
        df["large_gap_candidate_target_match"] - df["large_gap_base_target_match"],
        color="#9467bd",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Large-Gap Target-Match Delta")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(PLOTS / "round7_hard_slice_branch_summary.png", dpi=160)
    plt.close(fig)


def _plot_portfolio(df: pd.DataFrame, totals: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    colors = ["#1f77b4" if bucket == "exploit" else "#ff7f0e" for bucket in df["bucket"]]

    axes[0].bar(df["experiment_id"], df["actual_gpu_hours"], color=colors)
    axes[0].set_title("Round 7 GPU-Hours by Task")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(totals["bucket"], totals["actual_gpu_hours"], color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Round 7 Exploit vs Explore")
    axes[1].set_ylim(0.0, max(0.05, float(totals["actual_gpu_hours"].max()) * 1.15))

    fig.tight_layout()
    fig.savefig(PLOTS / "round7_portfolio_usage.png", dpi=160)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    scout_rows: list[dict[str, object]] = []
    baseline_added = False
    movement_rows: list[dict[str, object]] = []
    portfolio_rows: list[dict[str, object]] = [
        {
            "experiment_id": "A2-R7-hard-gap-audit",
            "bucket": "exploit",
            "actual_gpu_hours": 0.0,
            "status": "completed",
            "key_result": "Hard-feasible action-gap audit ran CPU-side on the fresh round-seven guardrail.",
        },
        {
            "experiment_id": "A3-R7-probe-audit",
            "bucket": "exploit",
            "actual_gpu_hours": 0.0,
            "status": "completed",
            "key_result": "Frozen-feature representation probe audit ran CPU-side on the fresh round-seven guardrail.",
        },
    ]
    baseline_eval_hours = _time_file_hours("r7_baseline_eval.time")
    if baseline_eval_hours > 0.0:
        portfolio_rows.append(
            {
                "experiment_id": "A1-R7-baseline-eval",
                "bucket": "exploit",
                "actual_gpu_hours": baseline_eval_hours,
                "status": "completed",
                "key_result": "Guardrail OOD evaluation sweep on corrected round-seven suites.",
            }
        )

    for experiment_id, bucket, label, experiment in TRAIN_ROWS:
        metrics = _load_best_metrics(experiment)
        if metrics is None:
            continue
        portfolio_rows.append(
            {
                "experiment_id": experiment_id,
                "bucket": bucket,
                "actual_gpu_hours": metrics["gpu_hours"],
                "status": "completed" if label == "Multiheavy" else "killed-early",
                "key_result": label,
            }
        )

    for label, prefix, log_name, compare_bucket in COMPARE_ROWS:
        loaded = _load_compare(prefix)
        if loaded is None:
            continue
        summary_df, decisions_df = loaded
        suite_metrics = _decision_metrics(summary_df)
        hard_metrics = _hard_slice_metrics(decisions_df)
        summary_row = {
            "branch": label,
            "base_suite_overall_disagreement": 0.0,
            "max_overall_disagreement": float(summary_df["overall_disagreement"].max()),
            "max_hard_feasible_disagreement": float(summary_df["hard_feasible_disagreement"].max()),
            "max_large_gap_hard_feasible_disagreement": float(
                summary_df["large_gap_hard_feasible_disagreement"].max()
            ),
            "max_improves_base_failures": 0.0,
            "max_breaks_base_successes": 0.0,
            **hard_metrics,
        }
        movement_rows.append(summary_row)

        scout_rows.append(
            {
                "variant": label,
                "overall_disagreement": suite_metrics["overall_disagreement"],
                "hard_feasible_disagreement": suite_metrics["hard_feasible_disagreement"],
                "large_gap_hard_feasible_disagreement": suite_metrics["large_gap_hard_feasible_disagreement"],
                "target_match_delta": suite_metrics["candidate_target_match"] - suite_metrics["base_target_match"],
            }
        )

        if not baseline_added:
            scout_rows.append(
                {
                    "variant": "Multiheavy",
                    "overall_disagreement": 0.0,
                    "hard_feasible_disagreement": 0.0,
                    "large_gap_hard_feasible_disagreement": 0.0,
                    "target_match_delta": 0.0,
                }
            )
            baseline_added = True

        runtime_hours = _time_file_hours(log_name)
        if compare_bucket in {"explore", "exploit"} and runtime_hours > 0.0:
            portfolio_rows.append(
                {
                    "experiment_id": f"{label}-compare",
                    "bucket": compare_bucket,
                    "actual_gpu_hours": runtime_hours,
                    "status": "completed",
                    "key_result": f"{label} disagreement compare.",
                }
            )

    if scout_rows:
        scout_df = pd.DataFrame(scout_rows)
        scout_df.to_csv(PLOTS / "round7_scout_seed312_compare.csv", index=False)
        _plot_scout_compare(scout_df)

    if movement_rows:
        movement_df = pd.DataFrame(movement_rows).sort_values("branch").reset_index(drop=True)
        movement_df.to_csv(PLOTS / "round7_policy_movement_summary.csv", index=False)
        hard_slice_df = movement_df[
            [
                "branch",
                "hard_slice_decisions",
                "large_gap_hard_slice_decisions",
                "hard_slice_base_target_match",
                "hard_slice_candidate_target_match",
                "large_gap_base_target_match",
                "large_gap_candidate_target_match",
                "max_large_gap_hard_feasible_disagreement",
            ]
        ].copy()
        hard_slice_df.to_csv(PLOTS / "round7_hard_slice_branch_summary.csv", index=False)
        _plot_policy_movement(movement_df)
        _plot_hard_slice(hard_slice_df)

    portfolio_df = pd.DataFrame(portfolio_rows)
    if not portfolio_df.empty:
        totals = portfolio_df.groupby("bucket", as_index=False)["actual_gpu_hours"].sum()
        total_gpu_hours = float(totals["actual_gpu_hours"].sum())
        split = {
            row["bucket"]: (
                float(row["actual_gpu_hours"]) / total_gpu_hours if total_gpu_hours > 0.0 else 0.0
            )
            for _, row in totals.iterrows()
        }
        summary_row = pd.DataFrame(
            [
                {
                    "experiment_id": "round7_total",
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
        pd.concat([portfolio_df, summary_row], ignore_index=True).to_csv(
            PLOTS / "round7_portfolio_usage.csv",
            index=False,
        )
        _plot_portfolio(portfolio_df[portfolio_df["bucket"] != "summary"], totals)


if __name__ == "__main__":
    main()
