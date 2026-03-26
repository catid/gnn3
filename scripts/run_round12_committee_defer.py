#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gnn3.eval.precision_correction import ULTRALOW_COVERAGE_BUDGETS, top_fraction_mask

VARIANTS = ("committee_only", "margin_committee")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank-decisions-csv", required=True)
    parser.add_argument(
        "--coverage-budgets",
        nargs="+",
        type=float,
        default=list(ULTRALOW_COVERAGE_BUDGETS),
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round12_committee_defer",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std <= 1e-6:
        return np.zeros_like(values, dtype=float)
    return (values - float(values.mean())) / std


def _score_map(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    committee_base = np.where(
        frame["best_safe_teacher_helpful"].to_numpy(copy=True).astype(bool)
        & (frame["committee_support"].to_numpy(copy=True) >= 2)
        & (~frame["harmful_teacher_bank_case"].to_numpy(copy=True).astype(bool)),
        frame["committee_support"].to_numpy(copy=True).astype(float)
        * frame["best_safe_teacher_gain"].to_numpy(copy=True).clip(min=0.0),
        -1e9,
    )
    margin = -frame["base_model_margin"].to_numpy(copy=True)
    high_value = (
        frame["high_headroom_near_tie_case"].to_numpy(copy=True).astype(float)
        + frame["baseline_error_hard_near_tie_case"].to_numpy(copy=True).astype(float)
    )
    margin_committee = committee_base.copy()
    active = committee_base > -1e8
    margin_committee[active] = (
        _zscore(committee_base[active])
        + 0.75 * _zscore(margin[active])
        + 0.25 * _zscore(high_value[active])
    )
    return {
        "committee_only": committee_base,
        "margin_committee": margin_committee,
    }


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


def _evaluate_budget(frame: pd.DataFrame, scores: np.ndarray, *, budget_pct: float, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = top_fraction_mask(scores, budget_pct) & (scores > -1e8)
    decision_frame = frame[["suite", "episode_index", "decision_index", "seed"]].copy()
    decision_frame["variant"] = variant
    decision_frame["budget_pct"] = budget_pct
    decision_frame["score"] = scores
    decision_frame["selected"] = selected

    rows: list[dict[str, object]] = []
    for slice_name, mask in _slice_map(frame).items():
        target = frame.loc[mask].copy()
        idx = target.index.to_numpy()
        selected_target = selected[idx]
        selected_count = max(int(selected_target.sum()), 1)
        combined_target_match = np.where(
            selected_target,
            target["best_safe_teacher_target_match"].to_numpy(copy=True),
            target["base_target_match"].to_numpy(copy=True),
        )
        delta_regret = np.where(selected_target, target["best_safe_teacher_delta_regret"].to_numpy(copy=True), 0.0)
        delta_miss = np.where(selected_target, target["best_safe_teacher_delta_miss"].to_numpy(copy=True), 0.0)
        rows.append(
            {
                "variant": variant,
                "budget_pct": budget_pct,
                "slice": slice_name,
                "decisions": len(target),
                "coverage": float(selected_target.mean()) if len(target) else 0.0,
                "defer_precision": float(
                    np.logical_and(selected_target, target["stable_positive_v2_case"].to_numpy(copy=True)).sum() / selected_count
                ),
                "false_positive_rate": float(
                    np.logical_and(selected_target, ~target["stable_positive_v2_case"].to_numpy(copy=True)).sum() / selected_count
                ),
                "harmful_selection_rate": float(
                    np.logical_and(selected_target, target["harmful_teacher_bank_case"].to_numpy(copy=True)).sum() / selected_count
                ),
                "correction_rate": float(
                    np.logical_and(selected_target, target["baseline_error_hard_near_tie_case"].to_numpy(copy=True)).mean()
                )
                if len(target)
                else 0.0,
                "new_error_rate": float(
                    np.logical_and(
                        np.logical_and(selected_target, target["base_target_match"].to_numpy(copy=True).astype(bool)),
                        ~target["best_safe_teacher_target_match"].to_numpy(copy=True).astype(bool),
                    ).mean()
                )
                if len(target)
                else 0.0,
                "base_target_match": float(target["base_target_match"].mean()) if len(target) else 0.0,
                "system_target_match": float(combined_target_match.mean()) if len(target) else 0.0,
                "mean_delta_regret": float(delta_regret.mean()) if len(target) else 0.0,
                "mean_delta_miss": float(delta_miss.mean()) if len(target) else 0.0,
            }
        )
    return pd.DataFrame(rows), decision_frame


def _plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    stable = summary_df.loc[summary_df["slice"] == "stable_positive_v2"].copy()
    hard = summary_df.loc[summary_df["slice"] == "hard_near_tie"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for variant, group in stable.groupby("variant", sort=False):
        axes[0].plot(group["budget_pct"], group["defer_precision"], marker="o", label=variant)
    for variant, group in hard.groupby("variant", sort=False):
        axes[1].plot(group["budget_pct"], group["mean_delta_regret"], marker="o", label=variant)
    axes[0].set_title("Committee Precision vs Coverage")
    axes[1].set_title("Committee Hard Near-Tie Delta Regret")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    for axis in axes:
        axis.set_xlabel("Budget %")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.teacher_bank_decisions_csv)
    score_map = _score_map(frame)

    summary_rows: list[pd.DataFrame] = []
    decision_rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        for budget_pct in args.coverage_budgets:
            summary_df, decision_df = _evaluate_budget(frame, score_map[variant], budget_pct=budget_pct, variant=variant)
            summary_rows.append(summary_df)
            decision_rows.append(decision_df)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    decisions_df = pd.concat(decision_rows, ignore_index=True)
    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    decisions_df.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    _plot(summary_df, output_prefix.with_name(output_prefix.name + "_summary.png"))
    output_prefix.with_suffix(".json").write_text(
        json.dumps({"coverage_budgets": args.coverage_budgets}, indent=2),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
