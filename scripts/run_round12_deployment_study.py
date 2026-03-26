#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY_COLS = ["suite", "episode_index", "decision_index"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank-decisions-csv", required=True)
    parser.add_argument("--ultralow-decisions-csv", required=True)
    parser.add_argument("--committee-decisions-csv", required=True)
    parser.add_argument("--round11-summary-csv", required=True)
    parser.add_argument("--heldout-seeds", nargs="+", type=int, default=[315, 316])
    parser.add_argument("--base-steps", type=float, default=3.0)
    parser.add_argument("--teacher-steps", type=float, default=5.0)
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round12_deployment_study",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _slice_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "overall": pd.Series([True] * len(frame), index=frame.index),
        "stable_positive_v2": frame["stable_positive_v2_case"],
        "stable_positive_v2_committee": frame["stable_positive_v2_committee_case"],
        "hard_near_tie": frame["hard_near_tie_intersection_case"],
        "high_headroom_near_tie": frame["high_headroom_near_tie_case"],
        "baseline_error_near_tie": frame["baseline_error_hard_near_tie_case"],
        "large_gap_control": frame["large_gap_hard_feasible_case"],
    }


def _aggregate_decisions(
    base: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    family: str,
    variant: str,
    teacher_target_col: str,
    teacher_delta_regret_col: str,
    teacher_delta_miss_col: str,
    teacher_action_col: str,
    base_steps: float,
    teacher_steps: float,
) -> pd.DataFrame:
    merged = base.merge(
        decisions[[*KEY_COLS, "budget_pct", "selected"]],
        on=KEY_COLS,
        how="inner",
    )
    selected = merged["selected"].fillna(False).astype(bool).to_numpy(copy=True)
    rows: list[dict[str, object]] = []
    for slice_name, mask in _slice_map(merged).items():
        target = merged.loc[mask].copy()
        idx = target.index.to_numpy()
        selected_target = selected[idx]
        decisions_count = len(target)
        if decisions_count <= 0:
            continue
        selected_count = max(int(selected_target.sum()), 1)
        teacher_target = target[teacher_target_col].to_numpy(copy=True).astype(bool)
        base_target = target["base_target_match"].to_numpy(copy=True).astype(bool)
        combined = np.where(selected_target, teacher_target, base_target)
        delta_regret = np.where(selected_target, target[teacher_delta_regret_col].to_numpy(copy=True), 0.0)
        delta_miss = np.where(selected_target, target[teacher_delta_miss_col].to_numpy(copy=True), 0.0)
        teacher_action = target[teacher_action_col].astype(str).to_numpy(copy=True)
        base_action = target["base_predicted_next_hop"].astype(str).to_numpy(copy=True)
        coverage = float(selected_target.mean())
        average_steps = base_steps + coverage * max(teacher_steps - base_steps, 0.0)
        rows.append(
            {
                "family": family,
                "variant": variant,
                "budget_pct": float(target["budget_pct"].iloc[0]),
                "slice": slice_name,
                "decisions": decisions_count,
                "coverage": coverage,
                "defer_precision": float(
                    np.logical_and(selected_target, target["stable_positive_v2_case"].to_numpy(copy=True)).sum()
                    / selected_count
                ),
                "false_positive_rate": float(
                    np.logical_and(selected_target, ~target["stable_positive_v2_case"].to_numpy(copy=True)).sum()
                    / selected_count
                ),
                "harmful_selection_rate": float(
                    np.logical_and(selected_target, target["harmful_teacher_bank_case"].to_numpy(copy=True)).sum()
                    / selected_count
                ),
                "correction_rate": float(
                    np.logical_and(selected_target, target["baseline_error_hard_near_tie_case"].to_numpy(copy=True)).mean()
                ),
                "new_error_rate": float(
                    np.logical_and(np.logical_and(selected_target, base_target), ~teacher_target).mean()
                ),
                "base_target_match": float(base_target.mean()),
                "system_target_match": float(combined.mean()),
                "mean_delta_regret": float(delta_regret.mean()),
                "mean_delta_miss": float(delta_miss.mean()),
                "selected_disagreement": float(np.logical_and(selected_target, teacher_action != base_action).mean()),
                "average_outer_steps": average_steps,
                "compute_multiplier": average_steps / max(base_steps, 1e-6),
                "runtime_proxy_multiplier": average_steps / max(base_steps, 1e-6),
            }
        )
    return pd.DataFrame(rows)


def _baseline_rows(base: pd.DataFrame, *, base_steps: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for slice_name, mask in _slice_map(base).items():
        target = base.loc[mask].copy()
        if target.empty:
            continue
        rows.append(
            {
                "family": "baseline",
                "variant": "multiheavy",
                "budget_pct": 0.0,
                "slice": slice_name,
                "decisions": len(target),
                "coverage": 0.0,
                "defer_precision": 0.0,
                "false_positive_rate": 0.0,
                "harmful_selection_rate": 0.0,
                "correction_rate": 0.0,
                "new_error_rate": 0.0,
                "base_target_match": float(target["base_target_match"].mean()),
                "system_target_match": float(target["base_target_match"].mean()),
                "mean_delta_regret": 0.0,
                "mean_delta_miss": 0.0,
                "selected_disagreement": 0.0,
                "average_outer_steps": base_steps,
                "compute_multiplier": 1.0,
                "runtime_proxy_multiplier": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _choose_panel(summary_df: pd.DataFrame) -> pd.DataFrame:
    hard = summary_df.loc[summary_df["slice"] == "hard_near_tie"].copy()
    overall = summary_df.loc[summary_df["slice"] == "overall", ["family", "variant", "budget_pct", "mean_delta_regret"]].rename(
        columns={"mean_delta_regret": "overall_delta_regret"}
    )
    large = summary_df.loc[
        summary_df["slice"] == "large_gap_control",
        ["family", "variant", "budget_pct", "mean_delta_regret", "harmful_selection_rate"],
    ].rename(
        columns={
            "mean_delta_regret": "large_gap_delta_regret",
            "harmful_selection_rate": "large_gap_harmful_selection_rate",
        }
    )
    merged = hard.merge(overall, on=["family", "variant", "budget_pct"], how="left").merge(
        large, on=["family", "variant", "budget_pct"], how="left"
    )
    panel_rows = []
    for (family, _variant), group in merged.groupby(["family", "variant"], sort=False):
        if family == "baseline":
            panel_rows.append(group.iloc[[0]].copy())
            continue
        feasible = group.loc[
            (group["overall_delta_regret"] <= 1e-9)
            & (group["large_gap_delta_regret"] <= 1e-9)
            & (group["harmful_selection_rate"] <= 1e-9)
        ].copy()
        chosen = feasible if not feasible.empty else group
        chosen = chosen.sort_values(["mean_delta_regret", "coverage", "budget_pct"], ascending=[True, True, True]).iloc[[0]]
        panel_rows.append(chosen)
    return pd.concat(panel_rows, ignore_index=True)


def _plot(summary_df: pd.DataFrame, panel_df: pd.DataFrame, output_path: Path) -> None:
    hard = summary_df.loc[summary_df["slice"] == "hard_near_tie"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (family, variant), group in hard.groupby(["family", "variant"], sort=False):
        label = f"{family}:{variant}"
        axes[0].plot(group["coverage"], group["mean_delta_regret"], marker="o", label=label)
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_title("Held-out Hard Near-Tie Delta Regret")
    axes[0].set_xlabel("Coverage")
    axes[0].set_ylabel("Mean Delta Regret")
    axes[0].legend(fontsize=8)

    panel_hard = panel_df.copy()
    axes[1].scatter(panel_hard["average_outer_steps"], panel_hard["mean_delta_regret"])
    for _, row in panel_hard.iterrows():
        axes[1].annotate(
            f"{row['family']}:{row['variant']}@{row['budget_pct']}",
            (row["average_outer_steps"], row["mean_delta_regret"]),
            fontsize=8,
        )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Selected Deployment Panel")
    axes[1].set_xlabel("Average Outer Steps")
    axes[1].set_ylabel("Hard Near-Tie Mean Delta Regret")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.teacher_bank_decisions_csv)
    base = base.loc[base["seed"].isin(args.heldout_seeds)].copy()

    ultralow = pd.read_csv(args.ultralow_decisions_csv)
    ultralow = ultralow.loc[ultralow["split"].isin([f"round11_feature_cache_seed{seed}" for seed in args.heldout_seeds])].copy()

    committee = pd.read_csv(args.committee_decisions_csv)
    committee = committee.loc[committee["seed"].isin(args.heldout_seeds)].copy()

    summary_rows = [
        _baseline_rows(base, base_steps=args.base_steps),
    ]

    for variant in sorted(ultralow["variant"].unique()):
        for _budget_pct, group in ultralow.loc[ultralow["variant"] == variant].groupby("budget_pct", sort=False):
            summary_rows.append(
                _aggregate_decisions(
                    base,
                    group,
                    family="round12_ultralow",
                    variant=str(variant),
                    teacher_target_col="compute_target_match",
                    teacher_delta_regret_col="delta_regret",
                    teacher_delta_miss_col="delta_miss",
                    teacher_action_col="compute_predicted_next_hop",
                    base_steps=args.base_steps,
                    teacher_steps=args.teacher_steps,
                )
            )

    for variant in sorted(committee["variant"].unique()):
        for _budget_pct, group in committee.loc[committee["variant"] == variant].groupby("budget_pct", sort=False):
            summary_rows.append(
                _aggregate_decisions(
                    base,
                    group,
                    family="round12_committee",
                    variant=str(variant),
                    teacher_target_col="best_safe_teacher_target_match",
                    teacher_delta_regret_col="best_safe_teacher_delta_regret",
                    teacher_delta_miss_col="best_safe_teacher_delta_miss",
                    teacher_action_col="best_safe_teacher_action",
                    base_steps=args.base_steps,
                    teacher_steps=args.teacher_steps,
                )
            )

    summary_df = pd.concat(summary_rows, ignore_index=True)

    round11 = pd.read_csv(args.round11_summary_csv).copy()
    round11["family"] = "round11_reference"
    summary_df = pd.concat(
        [
            summary_df,
            round11[
                [
                    "family",
                    "variant",
                    "budget_pct",
                    "slice",
                    "decisions",
                    "coverage",
                    "defer_precision",
                    "false_positive_rate",
                    "correction_rate",
                    "new_error_rate",
                    "base_target_match",
                    "system_target_match",
                    "mean_delta_regret",
                    "mean_delta_miss",
                    "selected_disagreement",
                    "average_outer_steps",
                    "compute_multiplier",
                    "runtime_proxy_multiplier",
                ]
            ],
        ],
        ignore_index=True,
    )
    summary_df["harmful_selection_rate"] = summary_df["harmful_selection_rate"].fillna(0.0)

    panel_df = _choose_panel(summary_df)
    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    panel_df.to_csv(output_prefix.with_name(output_prefix.name + "_panel.csv"), index=False)
    _plot(summary_df, panel_df, output_prefix.with_name(output_prefix.name + "_summary.png"))
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "heldout_seeds": args.heldout_seeds,
                "panel": panel_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(panel_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
