#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, help="Held-out defer-gate summary CSV.")
    parser.add_argument("--variant", default="margin_regime", help="Gate variant to aggregate.")
    parser.add_argument("--base-steps", type=float, default=3.0, help="Baseline outer-step count.")
    parser.add_argument("--teacher-steps", type=float, default=5.0, help="Deferred teacher outer-step count.")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round11_deployment_study",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _aggregate(summary_df: pd.DataFrame, *, variant: str, base_steps: float, teacher_steps: float) -> pd.DataFrame:
    df = summary_df.loc[summary_df["variant"] == variant].copy()
    rows: list[dict[str, float | str]] = []
    for (budget_pct, slice_name), group in df.groupby(["budget_pct", "slice"], sort=False):
        decisions = int(group["decisions"].sum())
        if decisions <= 0:
            continue
        row: dict[str, float | str] = {
            "variant": variant,
            "budget_pct": float(budget_pct),
            "slice": str(slice_name),
            "decisions": decisions,
        }
        for column in [
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
        ]:
            row[column] = float((group[column] * group["decisions"]).sum() / decisions)
        average_steps = base_steps + float(row["coverage"]) * max(teacher_steps - base_steps, 0.0)
        row["average_outer_steps"] = average_steps
        row["compute_multiplier"] = average_steps / max(base_steps, 1e-6)
        row["runtime_proxy_multiplier"] = row["compute_multiplier"]
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(agg_df: pd.DataFrame, output_path: Path) -> None:
    overall = agg_df.loc[agg_df["slice"] == "overall"].copy()
    stable = agg_df.loc[agg_df["slice"] == "stable_positive_pack"].copy()
    hard = agg_df.loc[agg_df["slice"] == "hard_near_tie"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(overall["average_outer_steps"], overall["mean_delta_regret"], marker="o")
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_title("Overall Delta Regret vs Avg Steps")
    axes[0].set_xlabel("Average Outer Steps")

    axes[1].plot(hard["average_outer_steps"], hard["mean_delta_regret"], marker="o")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Hard Near-Tie Delta Regret")
    axes[1].set_xlabel("Average Outer Steps")

    axes[2].plot(stable["average_outer_steps"], stable["system_target_match"], marker="o")
    axes[2].set_title("Stable-Positive Recovery")
    axes[2].set_xlabel("Average Outer Steps")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.summary_csv)
    agg_df = _aggregate(
        summary_df,
        variant=args.variant,
        base_steps=args.base_steps,
        teacher_steps=args.teacher_steps,
    )
    agg_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "variant": args.variant,
                "base_steps": args.base_steps,
                "teacher_steps": args.teacher_steps,
                "summary": agg_df.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    _plot(agg_df, output_prefix.with_name(output_prefix.name + "_summary.png"))


if __name__ == "__main__":
    main()
